"""Fixed driver over actual worker dispatch/reader, with CPU native endpoints."""
# ruff: noqa: F401, F811

import hashlib
import json

import numpy as np
import pytest

from tests.unit.optics.test_final_resolve_readback import (
    bound_worker,
    copying_worker,
    executable_graph,
    final_sample_count,
    final_session,
    fixed_inputs,
    fixed_storage_session,
    graph_session,
    readback_directory,
    request,
    resolving_reader,
    staged_readback,
    worker_graph,
)

pytestmark = [
    pytest.mark.parametrize("fail_at", [None]), pytest.mark.parametrize("diagnostic", [False, True]),
    pytest.mark.parametrize("readback_directory", [True], indirect=True),
]


@pytest.mark.parametrize("final_sample_count", [4], indirect=True)
def test_fixed_driver_persists_actual_generation_before_release(copying_worker):
    x = copying_worker
    prepared = x.readback_holder["readback"]
    for generation, record in enumerate(x.worker._samples, 1):
        pending = request(
            x.worker, "EXECUTE_PREPARED_SAMPLE", prepared_token=record.prepared_token,
        )["payload"]
        for _, _, source, _ in prepared.copies:
            source.data.fill(generation)
        resolved = request(x.worker, "RESOLVE_PREPARED_SAMPLE", **pending)["payload"]
        assert record.state == "RESOLVED_HELD"
        path = x.readback_directory / f"final_sample_{record.index:04d}.json"
        evidence = json.loads(path.read_bytes())
        assert evidence["coverage"] == "resolve_only" and evidence["complete"] is False
        assert evidence["sample_index"] == record.index
        assert evidence["resolve_request_sequence"] == x.worker._next_sequence - 1
        assert evidence["parameters"] == x.final_graph.frames[record.index][0].to_payload()
        rows = evidence["rows"]
        assert rows[0]["edge"] == "before" and rows[-1]["edge"] == "resolved_outer"
        assert all(row["failure"] is None and row["state"] is not None for row in rows)
        assert all(row["generation"] == generation for row in rows)
        assert len({row["resolve_sequence"] for row in rows}) == 1
        scope_edges = [row["edge"] for row in rows if row["scope_kind"] is not None]
        assert {"before_entry", "after_entry", "before_exit", "after_exit"} == set(scope_edges)
        assert any(row["operation"] == "diagnostic_wait" and row["edge"] == "after" for row in rows)
        leaves = [row for row in rows if row["callable_token"] is not None]
        copy_owners = {id(source) for _, _, source, _ in prepared.copies}
        assert {row["owner_token"] for row in leaves if row["operation"] == "diagnostic_copy"} == (
            copy_owners | {id(prepared.event)}
        )
        assert {row["owner_token"] for row in leaves if row["operation"] == "diagnostic_wait"} == {
            id(prepared.event)
        }
        for name, desc in evidence["channels"].items():
            raw = (x.readback_directory / desc["file"]).read_bytes()
            assert len(raw) == desc["bytes"] and hashlib.sha256(raw).hexdigest() == desc["sha256"]
            values = np.frombuffer(raw, dtype=desc["dtype"]).reshape(desc["shape"])
            assert np.all(values == generation), name
        request(x.worker, "RELEASE_RESOLVED_SAMPLE", **resolved)
    assert x.final_graph.arena.lifecycle_evidence().slot_reclaim_count == 4


@pytest.mark.parametrize("fault", ["copy", "wait", "persist"])
def test_fixed_driver_failure_does_not_release_or_erase_completed_state(copying_worker, monkeypatch, fault):
    from tests.support import native_runtime_process as driver

    x = copying_worker
    pending = request(x.worker, "EXECUTE_PREPARED_SAMPLE",
                      prepared_token=x.worker._samples[0].prepared_token)["payload"]
    if fault == "persist":
        def fail_write(*args):
            raise x.readback_state.failure

        monkeypatch.setattr(driver, "_write_bytes", fail_write)
    else:
        x.readback_state.fault = fault
    with pytest.raises(RuntimeError) as caught:
        request(x.worker, "RESOLVE_PREPARED_SAMPLE", **pending)
    assert caught.value is x.readback_state.failure
    if fault == "persist":
        assert x.controller._replay.failed and x.controller.guardian.state == "SEALED"
        with pytest.raises((RuntimeError, ValueError)):
            request(x.worker, "RELEASE_RESOLVED_SAMPLE", resolved_token=x.worker._samples[0].resolved_token)
        # Direct handle() has no enclosing request loop. Invoke its original
        # terminal handler, retaining the original persistence exception.
        x.worker.terminal_fault_response(
            sequence=x.worker._next_sequence - 1, code=type(caught.value).__name__, message=str(caught.value),
            fault="dispatcher_failure", detection_phase="request_loop", cause=caught.value,
        )
    assert x.controller.guardian.state == "TRANSFERRED_TO_QUARANTINE"
    assert x.readback_holder["observer"].failure is caught.value
    arena = x.final_graph.arena.lifecycle_evidence()
    assert arena.slot_reclaim_count == 0
    assert arena.completion_resolution_count == (1 if fault == "persist" else 0)
    assert x.worker._samples[0].state == ("RESOLVED_HELD" if fault == "persist" else "COMPLETION_RECORDED")
    assert not (x.readback_directory / "final_sample_0000.json").exists()


def test_late_getter_failure_keeps_already_observed_foreign_scope(copying_worker, monkeypatch):
    x = copying_worker
    primary = x.reader.primary

    def foreign_then_fail(ordinal, *args):
        if ordinal == 2:
            raise x.readback_state.failure
        if ordinal == 0:
            if x.p.lane == "cuda":
                return True
            args[0]._obj.value, args[1]._obj.value = 0, 1
            return 0
        return primary(ordinal, *args)

    monkeypatch.setattr(x.reader, "primary", foreign_then_fail)
    pending = request(x.worker, "EXECUTE_PREPARED_SAMPLE",
                      prepared_token=x.worker._samples[0].prepared_token)["payload"]
    with pytest.raises(RuntimeError) as caught:
        request(x.worker, "RESOLVE_PREPARED_SAMPLE", **pending)
    assert caught.value is x.readback_state.failure
    assert x.controller.possible_dirty_device_ordinals == (0, 1)
    assert x.controller.unknown_device_scope
    assert not any(kind == "wait" for kind, _ in x.sample_calls)
    artifact = json.loads((x.readback_directory / "final_sample_0000_partial.json").read_bytes())
    assert artifact["failure"]["message"] == str(caught.value)
    assert artifact["rows"][-1]["failure"] is not None and artifact["rows"][-1]["state"] is None
    assert artifact["sample_state"] == "COMPLETION_RECORDED" and artifact["raw"]


@pytest.mark.parametrize("fault", ["after_entry", "copy_and_exit"])
def test_original_scope_unwinds_once_without_losing_first_failure(copying_worker, monkeypatch, fault):
    x = copying_worker
    probe = x.readback_holder["observer"]
    prepared = x.readback_holder["readback"]
    facade = prepared.copies[0][3][1] if x.p.lane == "cuda" else prepared.device_scope
    scope = facade.original
    scope_type = type(scope)
    original_exit, observe = scope_type.__exit__, probe.observe
    exits = []
    secondary = RuntimeError("secondary scope exit failure")

    def exit_once(actual, *args):
        result = original_exit(actual, *args)
        if actual is scope:
            exits.append(args[1])
            if fault == "copy_and_exit":
                raise secondary
        return result

    def broken(*args):
        raise x.readback_state.failure

    def observed(operation, edge, owner, *args, **kwargs):
        if fault == "after_entry" and edge == "after_entry" and owner is scope:
            monkeypatch.setattr(x.reader, "primary", broken)
        return observe(operation, edge, owner, *args, **kwargs)

    monkeypatch.setattr(scope_type, "__exit__", exit_once)
    monkeypatch.setattr(probe, "observe", observed)
    pending = request(x.worker, "EXECUTE_PREPARED_SAMPLE",
                      prepared_token=x.worker._samples[0].prepared_token)["payload"]
    ambient = x.f.state.device, x.f.state.stream
    if fault == "copy_and_exit":
        x.readback_state.fault = "copy"
    with pytest.raises(RuntimeError) as caught:
        request(x.worker, "RESOLVE_PREPARED_SAMPLE", **pending)
    assert caught.value is x.readback_state.failure
    assert exits == [caught.value] and probe.depth == {"device": 0, "stream": 0}
    assert (x.f.state.device, x.f.state.stream) == ambient
    assert x.controller.guardian.state == "TRANSFERRED_TO_QUARANTINE"
    if fault == "copy_and_exit":
        assert any(error is secondary for error in x.reader.held)
    else:
        assert not x.readback_state.calls
    assert not any(row["edge"] == "after_exit" for row in probe.rows)


@pytest.mark.parametrize("fault", [None, "rgb", "identity", "bytes", "persist"])
def test_original_readback_checked_against_cpu_before_release(copying_worker, monkeypatch, fault):
    from tests.support import native_runtime_process as driver
    from tests.support.final_worker_numeric import FinalWorkerNumericRun
    from tests.unit.optics.test_final_worker_numeric import physical_channels

    x = copying_worker
    numeric = FinalWorkerNumericRun(x.readback_directory, {
        "root": str(x.session.root), "revision": x.session.revision,
        "catalog_sha256": x.session.catalog_sha256, "lane": x.session.lane,
        "case_id": x.final_graph.spec.case_id, "row_id": x.final_graph.spec.row_id,
        "sample_count": len(x.worker._samples), "pass_mode": x.final_graph.spec.diagnostic_mode,
        "go2_model_dir": None,
    })
    _, truth = numeric.reference.render(0)
    actual = physical_channels(truth)
    prepared = x.readback_holder["readback"]
    pending = request(x.worker, "EXECUTE_PREPARED_SAMPLE",
                      prepared_token=x.worker._samples[0].prepared_token)["payload"]
    for name, _, source, _ in prepared.copies:
        np.copyto(source.data, actual[name])
    if fault == "rgb":
        next(source for name, _, source, _ in prepared.copies if name == "rgb").data[0, 0] += 0.1
    resolved = request(x.worker, "RESOLVE_PREPARED_SAMPLE", **pending)["payload"]
    path = x.readback_directory / "final_sample_0000.json"
    if fault == "identity":
        payload = json.loads(path.read_bytes())
        payload["worker_start_nonce"] = "0" * 32
        path.write_text(json.dumps(payload), encoding="utf8")
    elif fault == "bytes":
        (x.readback_directory / "final_sample_0000_rgb.bin").write_bytes(b"corrupt")
    elif fault == "persist":
        original_write = driver._write

        def persist(directory, name, payload):
            if name.endswith("_numeric.json"):
                raise OSError("numeric report persistence failed")
            return original_write(directory, name, payload)

        monkeypatch.setattr(driver, "_write", persist)

    def check_then_release():
        numeric.check(0, x.worker._next_sequence - 1, x.worker._manifest, x.worker._worker_evidence)
        request(x.worker, "RELEASE_RESOLVED_SAMPLE", **resolved)

    if fault is None:
        check_then_release()
        assert (x.readback_directory / "final_sample_0000_numeric.json").exists()
        assert x.final_graph.arena.lifecycle_evidence().slot_reclaim_count == 1
    else:
        with pytest.raises((AssertionError, ValueError, OSError)):
            check_then_release()
        arena = x.final_graph.arena.lifecycle_evidence()
        assert arena.completion_resolution_count == 1 and arena.slot_reclaim_count == 0
        assert x.worker._samples[0].state == "RESOLVED_HELD"
        assert not (x.readback_directory / "final_sample_0000_numeric.json").exists()
