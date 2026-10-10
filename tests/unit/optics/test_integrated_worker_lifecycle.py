"""Worker/arena/guard integration with CPU native leaves, not GPU acceptance.

The captured import/environment intake is doubled. Binding occurs before the
real setup/arena constructors, and each request drives the production dispatcher.
"""
# Imported fixtures are registered by pytest.
# ruff: noqa: F401, F811

import json
import secrets
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from optical_worker import protocol as p
from optical_worker.manifest import build_integrated_worker_runtime_manifest
from optical_worker.worker import ClosedWorkerSession, _WorkerSample
from tests.unit.optics.test_native_final_graph import final_session
from tests.unit.optics.test_native_render_preparation import graph_session
from tests.unit.optics.test_native_render_warmup import executable_graph
from tests.unit.optics.test_prepared_fixed_render_inputs import fixed_inputs, fixed_storage_session

pytestmark = [pytest.mark.parametrize("fail_at", [None]), pytest.mark.parametrize("diagnostic", [False])]


@pytest.fixture(autouse=True)
def bound_worker(fixed_storage_session, tmp_path, diagnostic):
    f = fixed_storage_session
    source = f.session
    previous_phase = source.phase
    source.phase, source.root, source.revision = "imported", tmp_path, "a" * 40
    source.catalog_sha256 = "a" * 64
    source.failed = False
    sources = (p.FrozenSourceDigest("optical_worker/worker.py", "b" * 64),)
    source.snapshot = NS(sources=sources)
    source.runtime_observation.environment = NS(sha256="c" * 64, to_payload=lambda: {"roots": []})
    source.runtime_observation.inputs = {"case_id": f.p.prepared.manifest.case_id}
    repository = Path(__file__).resolve().parents[3]
    for relative in ("benchmarks/results/q54_p12_4/gate.json",
                     "optical_worker/data/integrated_worker_runtime_contract_v3.json"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repository / relative).read_bytes().replace(b"\r\n", b"\n"))
    manifest = build_integrated_worker_runtime_manifest(
        source,
        pass_mode="diagnostic_profiled" if diagnostic else "primary_unprofiled", go2_model_dir=None,
    )
    worker = source.open_worker(manifest)
    source.phase = previous_phase
    return worker


@pytest.fixture
def worker_graph(bound_worker, final_session):
    x, worker = final_session, bound_worker
    # The setup fixtures prepared native storage and warmed it with CPU leaves.
    # Final plans, sealing, target barrier and sample execution are production code.
    x.p.runtime.cuda.synchronize = x.p.runtime.synchronize_device = lambda target: None
    # Warp's real scope objects can be re-entered for enqueue and finish. The
    # earlier setup-only fixture used single-use generator context managers.
    class Scope:
        def __init__(self, owner, field):
            self.owner, self.field = owner, field

        def __enter__(self):
            self.previous = getattr(x.f.state, self.field)
            setattr(x.f.state, self.field, self.owner)

        def __exit__(self, *_):
            setattr(x.f.state, self.field, self.previous)

    x.p.runtime.ScopedDevice = lambda owner: Scope(owner, "device")
    x.p.runtime.ScopedStream = lambda owner: Scope(owner, "stream")
    graph = x.session.prepare_fixed_final_graph(sample_count=2)
    worker._graph = graph
    worker._samples = tuple(_WorkerSample(i, secrets.token_hex(16), secrets.token_hex(16),
                                         secrets.token_hex(16), plan=plan)
                            for i, plan in enumerate(graph.samples))
    for record in worker._samples:
        for token, state in ((record.prepared_token, "PREPARED"),
                             (record.pending_token, "COMPLETION_RECORDED"),
                             (record.resolved_token, "RESOLVED_HELD")):
            worker._tokens[token] = record, state
    x.controller.record_worker_transition(worker, "worker_prepared", {
        "sample_count": 2, "graph_sha256": graph.sha256,
        "token_registry_sha256": worker._token_registry_sha256(),
    })
    worker._state = "PREPARED_IDLE"
    x.sample_calls = []

    def leaf(kind, value):
        record = worker._active_sample
        assert record is not None and record.submission_possible
        assert x.controller._measured_sample_submission_latched
        if kind == "wait":
            x.controller._guard_client.require_live_ticket(x.controller._guard_ticket, role="resolve")
        else:
            assert worker._native_operation == "execute"
            assert x.f.state.device is x.p.device and x.f.state.stream is x.f.stream
        x.sample_calls.append((kind, value))

    x.state.action = leaf
    x.worker, x.final_graph = worker, graph
    return x


def request(worker, operation, **arguments):
    return worker.handle({"protocol_version": p.INTEGRATED_WORKER_PROTOCOL_VERSION,
                          "sequence": worker._next_sequence, "operation": operation, "arguments": arguments})


def run_sample(worker, index):
    record = worker._samples[index]
    pending = request(worker, "EXECUTE_PREPARED_SAMPLE", prepared_token=record.prepared_token)["payload"]
    resolved = request(worker, "RESOLVE_PREPARED_SAMPLE", pending_token=pending["pending_token"])["payload"]
    request(worker, "RELEASE_RESOLVED_SAMPLE", resolved_token=resolved["resolved_token"])


def test_two_samples_reclaim_before_close_and_release_only_after_written_ack(worker_graph):
    x = worker_graph
    for index in range(2):
        run_sample(x.worker, index)
    c = x.controller
    assert len([kind for kind, _ in x.sample_calls if kind == "wait"]) == 2
    assert x.worker._state == "DRAINED_IDLE"
    response = request(x.worker, "CLOSE_SESSION")
    assert c.guardian.state == "SEALED" and not x.worker.terminal
    proof = response["payload"]["post_reclaim_idle"]
    assert proof["submitted_count"] == proof["resolved_count"] == proof["reclaimed_count"] == 2
    assert proof["historical_submission_latched"] is True
    before = list(x.sample_calls)
    x.worker.commit_written_response(response["sequence"])
    assert x.sample_calls == before
    assert x.worker.terminal and c.guardian.state == "RELEASED"
    assert c._measured_sample_submission_latched
    assert c.termination_evidence.lifecycle.to_payload()["summary"]["disposition"] == "clean_reclaimed_exit"


def test_cancel_remaining_while_pending_still_requires_resolve_and_release(worker_graph):
    x = worker_graph
    first, second = x.worker._samples
    pending = request(x.worker, "EXECUTE_PREPARED_SAMPLE", prepared_token=first.prepared_token)["payload"]
    request(x.worker, "CANCEL_PREPARED_SAMPLE", prepared_token=second.prepared_token)
    from optical_worker.lifecycle import WorkerSetupProtocolError
    with pytest.raises(WorkerSetupProtocolError, match="consumed"):
        request(x.worker, "CLOSE_SESSION")
    resolved = request(x.worker, "RESOLVE_PREPARED_SAMPLE", **pending)["payload"]
    request(x.worker, "RELEASE_RESOLVED_SAMPLE", **resolved)
    response = request(x.worker, "CLOSE_SESSION")
    proof = response["payload"]["post_reclaim_idle"]
    assert proof["submitted_count"] == proof["cancelled_count"] == 1
    x.worker.commit_written_response(response["sequence"])
    assert x.controller.guardian.state == "RELEASED"


def test_copied_idle_proof_cannot_authorize_cleanup(worker_graph):
    from optical_worker.guarded_evidence import PostReclaimIdleEvidence
    from optical_worker.lifecycle import WorkerSetupProtocolError

    x = worker_graph
    for record in x.worker._samples:
        request(x.worker, "CANCEL_PREPARED_SAMPLE", prepared_token=record.prepared_token)
    response = request(x.worker, "CLOSE_SESSION")
    copied = PostReclaimIdleEvidence.from_payload(response["payload"]["post_reclaim_idle"])
    with pytest.raises(WorkerSetupProtocolError, match="exact post-reclaim"):
        x.controller.commit_clean_close(_worker=x.worker, _proof=copied)
    assert x.controller.guardian.state == "SEALED"
    x.worker.commit_written_response(response["sequence"])
    assert not x.controller._measured_sample_submission_latched


def test_consumed_and_out_of_order_tokens_cannot_submit_again(worker_graph):
    x = worker_graph
    first, second = x.worker._samples
    with pytest.raises(p.ClosedWorkerProtocolError, match="FIFO"):
        request(x.worker, "EXECUTE_PREPARED_SAMPLE", prepared_token=second.prepared_token)
    assert not x.sample_calls
    run_sample(x.worker, 0)
    before = list(x.sample_calls)
    for operation, argument, token in (
        ("EXECUTE_PREPARED_SAMPLE", "prepared_token", first.prepared_token),
        ("RESOLVE_PREPARED_SAMPLE", "pending_token", first.pending_token),
        ("RELEASE_RESOLVED_SAMPLE", "resolved_token", first.resolved_token),
    ):
        with pytest.raises(p.ClosedWorkerProtocolError, match="consumed"):
            request(x.worker, operation, **{argument: token})
    assert x.sample_calls == before
    run_sample(x.worker, 1)


def test_old_completed_wait_cannot_resolve_next_sample_in_replayed_evidence(worker_graph):
    from copy import deepcopy

    from optical_worker.guarded_evidence import GuardedLifecycleEvidence
    from optical_worker.wait_protocol import GuardProtocolError

    x = worker_graph
    run_sample(x.worker, 0)
    run_sample(x.worker, 1)
    payload = x.controller.evidence().to_payload()
    original = deepcopy(payload)
    resolutions = [event for event in payload["transition_trace"] if event["operation"] == "sample_resolved"]
    resolutions[1]["details"]["resolve_sequence"] = resolutions[0]["details"]["resolve_sequence"]
    with pytest.raises(GuardProtocolError, match="terminal wait"):
        GuardedLifecycleEvidence.from_payload(payload)
    GuardedLifecycleEvidence.from_payload(original)


@pytest.mark.parametrize("failure", ["wait", "disarmed"])
def test_failed_terminal_wait_retains_unresolved_generation(worker_graph, monkeypatch, failure):
    x = worker_graph
    first = x.worker._samples[0]
    pending = request(x.worker, "EXECUTE_PREPARED_SAMPLE", prepared_token=first.prepared_token)["payload"]
    original = x.state.action

    def fail_wait(kind, value):
        original(kind, value)
        if kind == "wait":
            raise RuntimeError("injected native wait failure")

    if failure == "wait":
        x.state.action = fail_wait
    else:
        def fail_disarmed(ticket):
            assert ticket.role == "resolve"
            raise RuntimeError("injected lost DISARMED")
        monkeypatch.setattr(x.controller._guard_client, "complete", fail_disarmed)
    with pytest.raises(RuntimeError, match="injected"):
        request(x.worker, "RESOLVE_PREPARED_SAMPLE", **pending)
    summary = x.controller.termination_evidence.lifecycle.to_payload()["summary"]
    assert summary["disposition"] == "quarantined_process_exit"
    assert summary["emergency_sync_attempt_count"] == 0
    assert x.controller.guardian.state == "TRANSFERRED_TO_QUARANTINE"
    arena = x.final_graph.arena.lifecycle_evidence()
    assert arena.completion_resolution_count == arena.slot_reclaim_count == 0
    assert first.submission_possible and x.controller._measured_sample_submission_latched


def test_cleanup_failure_after_close_ack_requires_hard_exit(worker_graph, monkeypatch):
    x = worker_graph
    run_sample(x.worker, 0)
    run_sample(x.worker, 1)
    response = request(x.worker, "CLOSE_SESSION")

    def fail_cleanup(guardian, _authority):
        assert guardian is x.controller.guardian
        raise RuntimeError("injected release failure")

    monkeypatch.setattr(type(x.controller.guardian), "release_whole_graph", fail_cleanup)
    before = list(x.sample_calls)
    x.worker.commit_written_response(response["sequence"])
    assert x.sample_calls == before
    assert x.worker.hard_exit_required
    summary = x.worker.termination_evidence.lifecycle.to_payload()["summary"]
    assert summary["disposition"] == "cleanup_failed_exit"
    assert summary["custody"] == "remaining_quarantine"


def test_failure_before_first_enqueue_burns_generation_without_submission(worker_graph, monkeypatch):
    x = worker_graph
    control = x.final_graph.arena._control

    def fail_before_enqueue(lease):
        assert lease.state == "LEASED"
        raise RuntimeError("injected pre-leaf failure")

    monkeypatch.setattr(control, "_begin_enqueue", fail_before_enqueue)
    record = x.worker._samples[0]
    with pytest.raises(RuntimeError, match="pre-leaf"):
        request(x.worker, "EXECUTE_PREPARED_SAMPLE", prepared_token=record.prepared_token)
    assert not x.sample_calls and not record.submission_possible
    assert not x.controller._measured_sample_submission_latched
    evidence = control.lifecycle_evidence()
    assert evidence.slots[0].state == "FREE" and evidence.slots[0].generation == 1
    assert evidence.terminal_event_record_count == 0
    assert any(event.operation == "cancelled_generation_burned" for event in evidence.transitions)
    assert x.worker._state == "POISONED"


@pytest.mark.parametrize("failure", ["launch", "record"])
def test_submit_then_raise_never_exports_pending_or_claims_no_submission(worker_graph, failure):
    x = worker_graph
    original = x.state.action

    def submit_then_raise(kind, value):
        original(kind, value)
        if kind == failure:
            raise RuntimeError("injected submit-then-raise")

    x.state.action = submit_then_raise
    record = x.worker._samples[0]
    with pytest.raises(RuntimeError, match="submit-then-raise"):
        request(x.worker, "EXECUTE_PREPARED_SAMPLE", prepared_token=record.prepared_token)
    assert record.submission_possible and x.controller._measured_sample_submission_latched
    assert not x.controller._replay.final_quiescence_current
    assert x.worker._state == "POISONED" and record.state != "COMPLETION_RECORDED"
    assert x.final_graph.arena.lifecycle_evidence().terminal_event_record_count == 0
    synchronizations = []
    result = x.controller.terminate(fault="dispatcher_failure", detection_phase="injected_sample_failure",
                                  emergency_target_synchronize=lambda: synchronizations.append(1))
    summary = result.lifecycle.to_payload()["summary"]
    assert synchronizations == [1] and summary["disposition"] == "aborted_synced_exit"


def test_every_delivery_shipping_row_rejects_before_runtime_configuration(bound_worker):
    worker = bound_worker
    worker._state = "IDLE_UNPREPARED"
    gate = json.loads((Path(__file__).resolve().parents[3]
                       / "benchmarks/results/q54_p12_4/gate.json").read_text())
    before = worker._lifecycle.evidence().sha256
    rows = [row for row in gate["mandatory_rows"] if row["boundary"] not in {"R0", "R1"}]
    assert rows
    for row in rows:
        with pytest.raises(p.ClosedWorkerProtocolError, match="stage_scope_rejected"):
            request(worker, "PREPARE_FROZEN_ROW", row_id=row["row_id"], case_id=row["case_id"],
                    repeat_index=0, sample_count=1)
    assert not worker._samples and not worker._tokens
    assert worker._lifecycle.evidence().sha256 == before


def test_oversized_prepared_response_rejects_before_tokens_or_runtime_configuration(bound_worker):
    worker = bound_worker
    worker._state = "IDLE_UNPREPARED"
    before = worker._lifecycle.evidence().sha256
    with pytest.raises(p.ClosedWorkerProtocolError, match="protocol capacity"):
        request(worker, "PREPARE_FROZEN_ROW", row_id="r0_asymmetric_full_shadow",
                case_id=worker._manifest.case_id, repeat_index=0, sample_count=1_000_000)
    assert not worker._samples and not worker._tokens
    assert worker._lifecycle.evidence().sha256 == before


@pytest.mark.parametrize("field,value", [
    ("control_idle_timeout_ms", 1), ("response_write_timeout_ms", 1), ("abort_watchdog_ms", 1),
    ("python_version", "0.0.0"), ("gate_sha256", "f" * 64),
    ("worker_script_relative_path", "tools/optical_closed_worker.py"),
])
def test_native_manifest_rejects_changed_process_or_contract_metadata(
    bound_worker, monkeypatch, field, value,
):
    from dataclasses import replace

    from optical_worker.manifest import validate_integrated_worker_runtime_manifest

    worker = bound_worker
    # Descriptive validation only; the already-bound source cannot bind another worker.
    monkeypatch.setattr(worker._native_source, "phase", "imported")
    validate_integrated_worker_runtime_manifest(worker._manifest, worker._native_source)
    changed = replace(worker._manifest, **{field: value})
    with pytest.raises(p.ClosedWorkerProtocolError, match="actual process and frozen contract"):
        validate_integrated_worker_runtime_manifest(changed, worker._native_source)


def test_native_runtime_contract_cannot_be_changed_to_extend_deadlines(bound_worker, monkeypatch):
    worker = bound_worker
    source = worker._native_source
    monkeypatch.setattr(source, "phase", "imported")
    path = source.root / "optical_worker/data/integrated_worker_runtime_contract_v3.json"
    payload = json.loads(path.read_bytes())
    payload["control_idle_timeout_ms"] *= 10
    path.write_text(json.dumps(payload), encoding="utf8")
    with pytest.raises(p.ClosedWorkerProtocolError, match="frozen release"):
        build_integrated_worker_runtime_manifest(source, pass_mode="primary_unprofiled", go2_model_dir=None)


def test_native_fault_reply_fits_frozen_codec_and_binds_full_separate_evidence(worker_graph):
    x = worker_graph
    run_sample(x.worker, 0)
    terminal = x.controller.terminate(fault="channel_eof", detection_phase="request_loop",
                                      emergency_target_synchronize=lambda: None)
    # Reproduce the real CUDA intake failure's secondary envelope-depth error.
    with pytest.raises(p.ClosedWorkerProtocolError, match="nesting"):
        p.encode_framed_message(x.worker._error(3, {"termination_evidence": terminal.to_payload()}),
                                max_bytes=p.MAX_RESPONSE_BYTES)
    response = x.worker.terminal_fault_response(sequence=3, code="EOFError", message="control channel EOF",
                                                fault="channel_eof", detection_phase="request_loop")
    frame = p.encode_framed_message(response, max_bytes=p.MAX_RESPONSE_BYTES)
    decoded = p.decode_canonical_message(frame[4:], max_bytes=p.MAX_RESPONSE_BYTES)
    assert decoded["payload"]["termination_evidence_sha256"] == terminal.sha256
    assert decoded["payload"]["disposition"] == "aborted_synced_exit"
    assert "termination_evidence" not in decoded["payload"]
    assert x.worker.termination_evidence is terminal


@pytest.mark.parametrize("corruption", [None, "digest", "truncated"])
def test_supervisor_reads_actual_terminal_journal_over_stale_prefix(worker_graph, tmp_path, corruption):
    from optical_worker.worker import _journal
    from tests.support.guarded_custody_process import _read_worker_lifecycle

    x = worker_graph
    prefix = x.controller.evidence().to_payload()
    (tmp_path / "prefix.json").write_text(json.dumps(prefix), encoding="utf8")
    run_sample(x.worker, 0)
    terminal = x.controller.terminate(fault="channel_eof", detection_phase="request_loop",
                                      emergency_target_synchronize=lambda: None)
    journal = tmp_path / "worker_journal.bin"
    with journal.open("wb") as output:
        _journal(output.fileno(), {"kind": "worker_loop_fault", "message": "original failure"})
        _journal(output.fileno(), {"kind": "transport_termination", "evidence": terminal.to_payload(),
                                  "evidence_sha256": "0" * 64 if corruption == "digest" else terminal.sha256})
    if corruption == "truncated":
        journal.write_bytes(journal.read_bytes()[:-1])
    if corruption is not None:
        with pytest.raises(RuntimeError, match="digest mismatch|partial frame"):
            _read_worker_lifecycle(tmp_path, native_worker_mode=True)
    else:
        recovered = _read_worker_lifecycle(tmp_path, native_worker_mode=True)
        assert recovered == terminal.lifecycle.to_payload()
        assert recovered != prefix
        # The legacy driver continues to use its own persisted prefix.
        assert _read_worker_lifecycle(tmp_path, native_worker_mode=False) == prefix
