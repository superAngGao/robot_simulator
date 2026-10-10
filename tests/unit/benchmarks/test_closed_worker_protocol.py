from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, ModuleType

import pytest

from optical_worker.lifecycle import (
    DeviceScopeCompromisedError,
    DeviceScopeDescriptor,
    NativeWorkerSetupLifecycle,
    TransportTerminationEvidence,
    WorkerSetupProtocolError,
    validate_transport_termination_evidence,
)
from optical_worker.protocol import (
    CLOSED_WORKER_PROTOCOL_VERSION,
    MAX_REQUEST_BYTES,
    ClosedWorkerProtocolError,
    canonical_json_bytes,
    decode_canonical_message,
    validate_request,
)
from optical_worker.worker import ClosedWorkerSession


class _Owner:
    pass


def _noop_abort_notifier(device_ordinal: int) -> None:
    assert device_ordinal >= 0


def _descriptor(target: int, operation: str = "test_allocation") -> DeviceScopeDescriptor:
    return DeviceScopeDescriptor.exact_target(
        operation_id=operation,
        leaf_kind="allocation",
        device_ordinal=target,
        scope_source="unit_test_contract",
        runtime_identity="fake_runtime_v1",
    )


def _enter(
    lifecycle: NativeWorkerSetupLifecycle,
    descriptor: DeviceScopeDescriptor,
) -> None:
    permit = lifecycle.begin_device_leaf(descriptor)
    lifecycle.enter_device_leaf(permit)


def test_package_import_and_device_descriptor_are_cuda_runtime_free():
    root = Path(__file__).resolve().parents[3]
    script = (
        "import sys;"
        f"sys.path.insert(0,{str(root)!r});"
        "import optical_worker;"
        "optical_worker.DeviceScopeDescriptor.exact_target("
        "operation_id='probe',leaf_kind='allocation',device_ordinal=7,"
        "scope_source='unit_probe',runtime_identity='probe_v1');"
        "print(','.join(sorted(set(sys.modules)&{'numpy','torch','warp'})))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == ""


def test_canonical_codec_rejects_noncanonical_duplicate_float_and_objects():
    payload = {"a": 1, "b": [True, None, "x"]}
    encoded = canonical_json_bytes(payload)
    assert decode_canonical_message(encoded, max_bytes=MAX_REQUEST_BYTES) == payload

    for invalid in (b'{"b":1,"a":2}', b'{"a":1,"a":2}', b'{"a":1.0}'):
        with pytest.raises(ClosedWorkerProtocolError):
            decode_canonical_message(invalid, max_bytes=MAX_REQUEST_BYTES)
    with pytest.raises(ClosedWorkerProtocolError):
        canonical_json_bytes({"callable": lambda: None})


def test_request_schema_rejects_dynamic_operations_handles_and_extra_fields():
    base = {
        "protocol_version": CLOSED_WORKER_PROTOCOL_VERSION,
        "sequence": 0,
        "operation": "HELLO",
        "arguments": {"manifest_sha256": "a" * 64, "lane_id": "warp", "device_ordinal": 0},
    }
    assert validate_request(base)["operation"] == "HELLO"
    for changed in (
        {**base, "operation": "CALL_RAW_LAUNCH"},
        {**base, "arguments": {**base["arguments"], "stream_handle": 0xDEADBEEF}},
        {**base, "callback": "module.function"},
    ):
        with pytest.raises(ClosedWorkerProtocolError):
            validate_request(changed)


def test_repository_import_ledger_rejects_loaded_repo_module_outside_allowlist(
    runtime_manifest,
    monkeypatch,
):
    from optical_worker.manifest import validate_repository_import_ledger
    from optical_worker.protocol import FrozenSourceDigest

    root = Path(__file__).resolve().parents[3]
    loaded_paths = set()
    for module in tuple(sys.modules.values()):
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str):
            continue
        try:
            loaded_paths.add(Path(origin).resolve().relative_to(root).as_posix())
        except ValueError:
            pass
    fake_name = "unreviewed_repository_module_probe"
    fake = ModuleType(fake_name)
    fake.__file__ = str(root / "unreviewed_worker_probe.py")
    monkeypatch.setitem(sys.modules, fake_name, fake)
    manifest = replace(
        runtime_manifest,
        repository_root=str(root),
        source_digests=tuple(
            FrozenSourceDigest(relative_path=path, sha256="1" * 64) for path in sorted(loaded_paths)
        ),
    )
    with pytest.raises(ClosedWorkerProtocolError, match="outside source allowlist"):
        validate_repository_import_ledger(manifest)


def test_target_only_dirty_failure_syncs_once_before_whole_graph_release():
    calls: list[str] = []
    lifecycle = NativeWorkerSetupLifecycle(
        lane_id="cuda",
        target_device_ordinal=1,
        emergency_sync_notifier=lambda ordinal: calls.append(f"watchdog:{ordinal}"),
        session_identity="target-only",
    )
    _enter(lifecycle, _descriptor(1))
    owner = _Owner()
    lifecycle.adopt_returned_device_owner(
        owner,
        actual_device_ordinal=1,
        role="device_buffer",
        operation_id="test_allocation",
    )

    def synchronize() -> None:
        assert lifecycle.guardian.contains_exact_owner(owner)
        calls.append("sync")

    evidence = lifecycle.terminate(
        fault="setup_failure",
        detection_phase="first_upload",
        emergency_target_synchronize=synchronize,
    )
    assert calls == ["watchdog:1", "sync"]
    assert evidence.disposition == "aborted_synced_exit"
    assert evidence.emergency_sync_attempted_ordinal == 1
    assert evidence.resource_release_authorized is True
    assert lifecycle.guardian.state == "RELEASED"


def test_low_level_leaf_requires_the_exact_single_use_pre_leaf_permit():
    lifecycle = NativeWorkerSetupLifecycle(
        lane_id="cuda",
        target_device_ordinal=0,
        emergency_sync_notifier=_noop_abort_notifier,
        session_identity="leaf-permit",
    )
    permit = lifecycle.begin_device_leaf(_descriptor(0))
    with pytest.raises(WorkerSetupProtocolError, match="previous setup leaf"):
        lifecycle.begin_device_leaf(_descriptor(0, operation="second_allocation"))
    with pytest.raises(WorkerSetupProtocolError, match="permit"):
        lifecycle.enter_device_leaf(object())
    lifecycle.enter_device_leaf(permit)
    with pytest.raises(WorkerSetupProtocolError, match="permit"):
        lifecycle.enter_device_leaf(permit)


def test_watchdog_arm_failure_quarantines_without_entering_emergency_sync():
    def fail_notifier(device_ordinal: int) -> None:
        raise BrokenPipeError(f"supervisor pipe closed for cuda:{device_ordinal}")

    lifecycle = NativeWorkerSetupLifecycle(
        lane_id="cuda",
        target_device_ordinal=0,
        emergency_sync_notifier=fail_notifier,
        session_identity="watchdog-arm-failure",
    )
    _enter(lifecycle, _descriptor(0))
    owner = _Owner()
    lifecycle.adopt_returned_device_owner(
        owner,
        actual_device_ordinal=0,
        role="device_buffer",
        operation_id="test_allocation",
    )
    evidence = lifecycle.terminate(
        fault="setup_failure",
        detection_phase="upload",
        emergency_target_synchronize=lambda: pytest.fail("sync must not start without watchdog"),
    )
    assert evidence.disposition == "quarantined_process_exit"
    assert evidence.emergency_sync_attempt_count == 0
    assert evidence.emergency_sync_outcome == "watchdog_arm_failed"
    assert lifecycle.guardian.contains_exact_owner(owner)


def test_foreign_returned_owner_is_adopted_then_forces_no_sync_quarantine():
    lifecycle = NativeWorkerSetupLifecycle(
        lane_id="warp",
        target_device_ordinal=1,
        emergency_sync_notifier=_noop_abort_notifier,
        session_identity="foreign-owner",
    )
    _enter(lifecycle, _descriptor(1))
    owner = _Owner()
    with pytest.raises(DeviceScopeCompromisedError):
        lifecycle.adopt_returned_device_owner(
            owner,
            actual_device_ordinal=0,
            role="wrong_device_buffer",
            operation_id="test_allocation",
        )
    sync_calls: list[int] = []
    evidence = lifecycle.terminate(
        fault="device_scope_compromised",
        detection_phase="returned_owner_validation",
        emergency_target_synchronize=lambda: sync_calls.append(1),
    )
    assert sync_calls == []
    assert evidence.possible_dirty_device_ordinals == (0, 1)
    assert evidence.disposition == "multi_or_unknown_device_quarantine_exit"
    assert evidence.resource_release_authorized is False
    assert evidence.guardian_transfer_committed is True
    assert lifecycle.guardian.contains_exact_owner(owner)


def test_unknown_scope_is_irreversible_and_cannot_become_aborted_synced():
    lifecycle = NativeWorkerSetupLifecycle(
        lane_id="cuda",
        target_device_ordinal=2,
        emergency_sync_notifier=_noop_abort_notifier,
        session_identity="unknown-scope",
    )
    unknown = DeviceScopeDescriptor.unknown(
        operation_id="opaque_helper",
        leaf_kind="compile_or_jit",
        scope_source="unreviewed_helper",
        runtime_identity="opaque_v1",
    )
    with pytest.raises(DeviceScopeCompromisedError):
        _enter(lifecycle, unknown)
    evidence = lifecycle.terminate(
        fault="device_scope_compromised",
        detection_phase="compile",
        emergency_target_synchronize=lambda: pytest.fail("unknown scope must not synchronize"),
    )
    assert evidence.unknown_device_scope is True
    assert evidence.emergency_sync_attempt_count == 0
    assert evidence.disposition == "multi_or_unknown_device_quarantine_exit"


def test_quiescence_requires_seal_exact_target_and_supports_clean_cancel():
    lifecycle = NativeWorkerSetupLifecycle(
        lane_id="cuda",
        target_device_ordinal=0,
        emergency_sync_notifier=_noop_abort_notifier,
        session_identity="quiescent",
    )
    _enter(lifecycle, _descriptor(0))
    _enter(
        lifecycle,
        DeviceScopeDescriptor.exact_target(
            operation_id="final_barrier",
            leaf_kind="device_synchronize",
            device_ordinal=0,
            scope_source="unit_test_contract",
            runtime_identity="fake_sync_v1",
        ),
    )
    lifecycle.seal_guardian()
    quiescence = lifecycle.commit_target_barrier(
        a2_readiness_sha256="a" * 64,
        prepared_graph_sha256="b" * 64,
    )
    assert quiescence.possible_dirty_device_ordinals == (0,)
    assert quiescence.proven_quiescent_device_ordinals == (0,)
    evidence = lifecycle.terminate(
        fault="channel_eof",
        detection_phase="prepared_idle",
        emergency_target_synchronize=lambda: pytest.fail("quiescent cancel must not resync"),
    )
    assert evidence.disposition == "cancelled_before_submission_exit"
    assert evidence.setup_quiescence_sha256 == quiescence.sha256


def test_termination_validator_replays_summary_and_rejects_public_rehash_forgery():
    lifecycle = NativeWorkerSetupLifecycle(
        lane_id="cuda",
        target_device_ordinal=0,
        emergency_sync_notifier=_noop_abort_notifier,
        session_identity="replay",
    )
    _enter(lifecycle, _descriptor(0))
    evidence = lifecycle.terminate(
        fault="setup_failure",
        detection_phase="upload",
        emergency_target_synchronize=lambda: None,
    )
    validate_transport_termination_evidence(evidence)
    assert TransportTerminationEvidence.from_payload(evidence.to_payload()) == evidence
    for forged in (
        replace(evidence, setup_operation_count=99),
        replace(evidence, possible_dirty_device_ordinals=(0, 1)),
        replace(evidence, disposition="cancelled_before_submission_exit"),
        replace(evidence, setup_work_counts=MappingProxyType({"allocation": 99})),
        replace(
            evidence,
            guardian_coverage=MappingProxyType(
                {**evidence.guardian_coverage, "per_device_resource_counts": {"0": 99}}
            ),
        ),
    ):
        with pytest.raises(WorkerSetupProtocolError):
            validate_transport_termination_evidence(forged)


def test_closed_worker_session_keeps_gpu_prepare_and_result_exit_closed(runtime_manifest):
    session = ClosedWorkerSession(
        runtime_manifest,
        emergency_sync_notifier=_noop_abort_notifier,
    )
    hello = session.handle(
        {
            "protocol_version": CLOSED_WORKER_PROTOCOL_VERSION,
            "sequence": 0,
            "operation": "HELLO",
            "arguments": {
                "manifest_sha256": runtime_manifest.sha256,
                "lane_id": runtime_manifest.lane_id,
                "device_ordinal": runtime_manifest.device_ordinal,
            },
        }
    )
    assert hello["status"] == "ok"
    opened = session.handle(
        {
            "protocol_version": CLOSED_WORKER_PROTOCOL_VERSION,
            "sequence": 1,
            "operation": "OPEN_FROZEN_SESSION",
            "arguments": {
                "gate_sha256": runtime_manifest.gate_sha256,
                "workload_set_sha256": runtime_manifest.workload_set_sha256,
            },
        }
    )
    assert opened["payload"]["result_capable"] is False
    rejected = session.handle(
        {
            "protocol_version": CLOSED_WORKER_PROTOCOL_VERSION,
            "sequence": 2,
            "operation": "PREPARE_FROZEN_ROW",
            "arguments": {
                "row_id": "r0_asymmetric_full_shadow",
                "case_id": "asymmetric_synthetic_7x3_v1",
                "repeat_index": 0,
                "sample_count": 1,
            },
        }
    )
    assert rejected["status"] == "error"
    assert rejected["payload"]["code"] == "GPU_PREPARATION_DISABLED_PENDING_C4C0_REVIEW"
    assert rejected["payload"]["measured_sample_gpu_submission_delta"] == 0
    assert rejected["payload"]["termination_evidence"]["disposition"] == ("cancelled_before_submission_exit")


@pytest.fixture
def runtime_manifest(tmp_path):
    from optical_worker.manifest import AST_POLICY_SHA256, current_environment_identity
    from optical_worker.protocol import (
        CLOSED_WORKER_RUNTIME_MANIFEST_VERSION,
        CLOSED_WORKER_STATE_MACHINE_VERSION,
        CLOSED_WORKER_TRUST_MODEL_VERSION,
        ClosedWorkerRuntimeManifest,
        FrozenSourceDigest,
        canonical_sha256,
    )

    environment = current_environment_identity()
    sources = (FrozenSourceDigest(relative_path="worker.py", sha256="1" * 64),)
    payload = {
        "schema_version": CLOSED_WORKER_RUNTIME_MANIFEST_VERSION,
        "protocol_version": CLOSED_WORKER_PROTOCOL_VERSION,
        "trust_model_version": CLOSED_WORKER_TRUST_MODEL_VERSION,
        "state_machine_version": CLOSED_WORKER_STATE_MACHINE_VERSION,
        "interpreter_executable": sys.executable,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "environment": environment,
        "environment_sha256": canonical_sha256(environment),
        "site_package_paths": [],
        "repository_root": str(tmp_path),
        "repository_revision": "f" * 40,
        "repository_clean": True,
        "worker_script_relative_path": "tools/optical_closed_worker.py",
        "lane_id": "cuda",
        "device_ordinal": 0,
        "gate_relative_path": "benchmarks/results/q54_p12_4/gate.json",
        "gate_sha256": "2" * 64,
        "workload_set_sha256": "3" * 64,
        "source_digests": [source.to_payload() for source in sources],
        "source_manifest_sha256": canonical_sha256([source.to_payload() for source in sources]),
        "ast_policy_sha256": AST_POLICY_SHA256,
        "control_idle_timeout_ms": 120000,
        "response_write_timeout_ms": 30000,
        "abort_watchdog_ms": 30000,
        "gpu_preparation_enabled": False,
    }
    return ClosedWorkerRuntimeManifest.from_payload(json.loads(canonical_json_bytes(payload)))


@pytest.mark.parametrize("boundary", ["read_idle", "close_ack_write"])
def test_worker_loop_timeout_preserves_fault_and_terminates_session(runtime_manifest, monkeypatch, boundary):
    from optical_worker import worker
    from optical_worker.protocol import FRAME_HEADER_BYTES, encode_framed_message

    session = ClosedWorkerSession(runtime_manifest, emergency_sync_notifier=_noop_abort_notifier)
    requests = [
        ("HELLO", {"manifest_sha256": runtime_manifest.sha256,
                   "lane_id": runtime_manifest.lane_id, "device_ordinal": runtime_manifest.device_ordinal}),
        ("OPEN_FROZEN_SESSION", {"gate_sha256": runtime_manifest.gate_sha256,
                                 "workload_set_sha256": runtime_manifest.workload_set_sha256}),
    ]
    if boundary == "close_ack_write":
        requests.append(("CLOSE_SESSION", {}))
    chunks = []
    for sequence, (operation, arguments) in enumerate(requests):
        frame = encode_framed_message({"protocol_version": CLOSED_WORKER_PROTOCOL_VERSION,
                                      "sequence": sequence, "operation": operation, "arguments": arguments},
                                     max_bytes=MAX_REQUEST_BYTES)
        chunks.extend((frame[:FRAME_HEADER_BYTES], frame[FRAME_HEADER_BYTES:]))
    incoming = iter(chunks)
    responses, journal = [], []
    message = "control channel idle timeout" if boundary == "read_idle" else "response write timeout"
    failure = TimeoutError(message)
    write_attempts = 0

    def read(descriptor, size, **kwargs):
        try:
            chunk = next(incoming)
        except StopIteration:
            raise failure
        assert len(chunk) == size
        return chunk

    def write(descriptor, frame, **kwargs):
        nonlocal write_attempts
        write_attempts += 1
        if boundary == "close_ack_write" and write_attempts == 3:
            assert session._state == "CLOSE_ACK_PENDING"
            assert session.termination_evidence is None
            raise failure
        responses.append(decode_canonical_message(frame[FRAME_HEADER_BYTES:], max_bytes=len(frame)))

    monkeypatch.setattr(worker, "ClosedWorkerSession", lambda *args, **kwargs: session)
    monkeypatch.setattr(worker, "_read_exact", read)
    monkeypatch.setattr(worker, "_write_all", write)
    monkeypatch.setattr(worker, "_journal", lambda descriptor, record: journal.append(record))
    assert worker.run_worker_loop(
        command_descriptor=-1, liveness_descriptor=-1, journal_descriptor=-1,
        abort_signal_descriptor=-1, abort_ack_descriptor=-1, manifest=runtime_manifest,
    ) == 64
    fault = "control_idle_timeout" if boundary == "read_idle" else "response_write_timeout"
    assert journal[0] == {"kind": "worker_loop_fault", "fault": fault,
                          "exception_type": "TimeoutError", "message": message}
    assert len(responses) == 3 and responses[-1]["status"] == "error"
    assert responses[-1]["payload"]["code"] == "TimeoutError"
    assert responses[-1]["payload"]["message"] == message
    assert session.terminal and session._state == "TERMINATED"
    evidence = session.termination_evidence
    assert evidence is not None
    validate_transport_termination_evidence(evidence)
    assert evidence.disposition == "cancelled_before_submission_exit"
    assert evidence.emergency_sync_attempt_count == 0
    assert journal[-1]["kind"] == "transport_termination"
