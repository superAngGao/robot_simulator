"""Production observation placement with modeled device leaves, not GPU proof."""

# ruff: noqa: F401, F811
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from optical_worker.lifecycle import WorkerSetupProtocolError
from optical_worker.native_bootstrap import _ExternalSourceLoader, _RuntimeObservation, _SourceSession
from optical_worker.native_readback import _leaf, _readback_native_arrays
from optical_worker.native_scope import NativeStateReader, require_reader
from tests.unit.optics.test_guarded_native_readback import _run, readback_case


def _bind(c, monkeypatch):
    session = NS(
        lane=c.prepared.backend,
        failed=False,
        phase="initializing_runtime",
        bound_controller=lambda: c.lifecycle,
        runtime_owners=[],
        _native_scope_record=None,
    )
    session.runtime_observation = _RuntimeObservation(session)
    session.runtime_observation.environment = NS(to_payload=lambda: {})
    c.lifecycle._native_source_session = session
    c.lifecycle.adopt_host_owner(session, role="captured_source_session", operation_id="source_custody_bind")
    reader = NativeStateReader(session, c.runtime, c.runtime, c.target)
    reader.stream = c.stream
    c.stream.cuda_stream = 123
    reader.binding = NS(
        session=session,
        controller=c.lifecycle,
        module=c.runtime,
        runtime=c.runtime,
        target=c.target,
        stream=c.stream,
    )
    c.lifecycle.adopt_host_owner(reader.binding, role="runtime_setup_binding", operation_id="binding")
    session._native_scope_record = (
        reader,
        c.lifecycle,
        c.runtime,
        c.runtime,
        c.target,
        c.stream,
        reader.binding,
    )
    reader.count, reader.current_stream_baseline, reader.target_context = 8, 123, 9000
    original = Path.resolve
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda p, **kw: Path(str(os.getpid()))
        if str(p).replace("\\", "/") == "/proc/thread-self"
        else original(p, **kw),
    )

    def state(pristine):
        assert not pristine
        common = {
            "primary_contexts": [{"ordinal": i, "flags": 0, "active": i == 1} for i in range(8)],
            "target_current_stream": 123,
            "held_stream": {"ordinal": 1, "handle": 123},
        }
        if session.lane == "cuda":
            return dict(
                common, initialized=True, logical_device=0, logical_basis="native_return", raw_device=1
            )
        interior = c.state.current_context is c.target
        return dict(
            common,
            default_device=1 if interior else 0,
            current_context=9000 if interior else None,
            current_context_device=1 if interior else None,
            target_context=9000,
        )

    reader._state = state
    session.phase = "validating_input"
    return session, reader


def test_input_readback_records_each_actual_call_including_empty_array(readback_case, monkeypatch):
    c = readback_case
    with c.lifecycle.segment():
        session, reader = _bind(c, monkeypatch)
        report = _run(c)
        assert report is not None
    rows = [r["details"] for r in session.runtime_observation.journal if r["kind"] == "native_scope"]
    assert [r["site"] for r in rows if r["edge"] == "before" and r["site"] != "readback_scope"] == [
        "copy",
        "copy",
        "copy",
        "event_create",
        "event_record",
        "event_wait",
    ]
    assert len(rows) == (14 if session.lane == "warp" else 12)
    assert c.state.event() in reader.held


@pytest.mark.parametrize("fault", ["getter", "append", "adopt_event"])
def test_returned_event_is_held_before_observation_or_adoption_failure(readback_case, monkeypatch, fault):
    c = readback_case
    error = RuntimeError("after-event observation failure")
    with pytest.raises(RuntimeError, match="after-event|event registration"):
        with c.lifecycle.segment():
            session, reader = _bind(c, monkeypatch)
            if fault == "getter":
                state = reader._state

                def failed_state(pristine):
                    if c.state.event is not None:
                        raise error
                    return state(pristine)

                reader._state = failed_state
            elif fault == "append":
                append = session.runtime_observation.append

                def failed_append(kind, details):
                    result = append(kind, details)
                    if (
                        kind == "native_scope"
                        and details["site"] == "event_create"
                        and details["edge"] == "after"
                    ):
                        raise error
                    return result

                session.runtime_observation.append = failed_append
            else:
                c.state.fault = "adopt_event"
            _run(c)
    event = c.state.event()
    assert event is not None and any(owner is event for owner in reader.held)
    assert "record" not in c.state.trace and "wait" not in c.state.trace
    summary = c.lifecycle.termination_evidence.lifecycle.to_payload()["summary"]
    assert summary["release_status"] == "not_authorized"
    assert summary["normal_complete_count"] == summary["emergency_sync_attempt_count"] == 0


def test_captured_input_cannot_omit_its_reader(readback_case, monkeypatch):
    c = readback_case
    with pytest.raises(ValueError, match="reader is missing"):
        with c.lifecycle.segment():
            session, reader = _bind(c, monkeypatch)
            session._native_scope_record = None
            _run(c)
    assert c.state.trace == []


def test_later_bvh_shared_readback_does_not_emit_input_phase_receipts(readback_case, monkeypatch):
    c = readback_case
    with c.lifecycle.segment():
        session, reader = _bind(c, monkeypatch)
        session.phase = "preparing_fixed_render_storage"
        arrays = tuple(
            (row.name, row.shape, row.dtype, c.prepared.arrays[row.name]) for row in c.source.manifest.arrays
        )
        owners = []
        c.lifecycle.adopt_host_owner(owners, role="later_bvh_host", operation_id="later_bvh_host")
        _leaf(c.lifecycle, "later_bvh_scope", "runtime_import_or_init", "warp_resource_scope_v1")
        with c.runtime.ScopedDevice(c.target) if session.lane == "warp" else nullcontext():
            values = _readback_native_arrays(
                arrays, c.lifecycle, c.runtime, c.target, c.stream, session.lane == "cuda", owners
            )
        assert len(values) == len(arrays)
    assert session.runtime_observation.journal == [] and reader.raw == []


def test_copied_reader_rejects_before_query(readback_case, monkeypatch):
    import copy

    c = readback_case
    with pytest.raises(ValueError, match="identity changed"):
        with c.lifecycle.segment():
            session, reader = _bind(c, monkeypatch)
            clone = copy.copy(reader)
            assert require_reader(session) is reader
            clone._ticket()
    assert c.state.trace == []


def test_caught_reader_identity_error_still_fails_original_guard(readback_case, monkeypatch):
    import copy

    c = readback_case
    with pytest.raises(WorkerSetupProtocolError, match="compromised|failed or terminal"):
        with c.lifecycle.segment():
            session, reader = _bind(c, monkeypatch)
            with pytest.raises(ValueError, match="identity changed"):
                copy.copy(reader)._ticket()
            assert c.lifecycle._replay.failed
    assert c.state.trace == []
    assert c.lifecycle.termination_evidence.lifecycle.to_payload()["summary"]["normal_complete_count"] == 0


@pytest.mark.parametrize(
    "attack",
    ["missing_source", "foreign_source", "phase", "missing_reader", "copied_reader", "copied_binding",
     "copied_binding_record"],
)
def test_input_provenance_rejection_is_sticky_before_native_work(readback_case, monkeypatch, attack):
    import copy

    c = readback_case
    with pytest.raises(WorkerSetupProtocolError, match="failed or terminal|compromised"):
        with c.lifecycle.segment():
            session, reader = _bind(c, monkeypatch)
            original = session._native_scope_record
            if attack == "missing_source":
                c.lifecycle._native_source_session = None
            elif attack == "foreign_source":
                c.lifecycle._native_source_session = copy.copy(session)
            elif attack == "phase":
                session.phase = "preparing_fixed_render_storage"
            elif attack == "missing_reader":
                session._native_scope_record = None
            elif attack == "copied_reader":
                session._native_scope_record = (copy.copy(reader), *original[1:])
            else:
                reader.binding = copy.copy(reader.binding)
                if attack == "copied_binding_record":
                    session._native_scope_record = (*original[:6], reader.binding)
            with pytest.raises((ValueError, WorkerSetupProtocolError)):
                _run(c)
            assert c.lifecycle._replay.failed
            assert c.state.trace == [] and c.state.scope_entries == 0 and reader.raw == []
            c.lifecycle._native_source_session = session
            session.phase = "validating_input"
            session._native_scope_record = original
            reader.binding = original[6]
            _run(c)  # Restoring the projection cannot revive the original guard.
    summary = c.lifecycle.termination_evidence.lifecycle.to_payload()["summary"]
    assert summary["normal_complete_count"] == summary["emergency_sync_attempt_count"] == 0
    assert summary["release_status"] == "not_authorized"


@pytest.mark.parametrize("attack", ["missing_source", "foreign_source", "phase"])
def test_registered_upload_source_cannot_be_lost_before_consumption(readback_case, monkeypatch, attack):
    import copy

    from optical_worker.native_setup import (
        _create_upload_context,
        _upload_cuda_native_input,
        _upload_warp_native_input,
    )

    c = readback_case
    with pytest.raises(WorkerSetupProtocolError, match="failed or terminal"):
        with c.lifecycle.segment():
            session, reader = _bind(c, monkeypatch)
            session.phase = "uploading"
            context = _create_upload_context(c.lifecycle, c.runtime, c.target, c.stream)
            assert context.source_session is session
            if attack == "phase":
                session.phase = "preparing_fixed_render_storage"
            else:
                c.lifecycle._native_source_session = (
                    None if attack == "missing_source" else copy.copy(session)
                )
            upload = _upload_cuda_native_input if session.lane == "cuda" else _upload_warp_native_input
            with pytest.raises((ValueError, WorkerSetupProtocolError)):
                upload(c.source, context=context)
            assert c.state.trace == [] and c.state.scope_entries == 0 and reader.raw == []
            assert c.lifecycle._replay.failed
            c.lifecycle._require_setup()


@pytest.mark.parametrize("readback_case", ["cuda"], indirect=True)
@pytest.mark.parametrize("fault", [None, "stream_exit", "after_query", "after_append"])
def test_artifact_pairs_surround_real_loader_and_retain_return(readback_case, monkeypatch, fault):
    c = readback_case
    module, spec, calls = object(), object(), []
    with pytest.raises(RuntimeError, match="artifact boundary") if fault else nullcontext():
        with c.lifecycle.segment():
            session, reader = _bind(c, monkeypatch)
            session.phase = "initializing_runtime"
            session._cuda_artifact_runtime = lambda name: (c.runtime, c.target, c.stream)
            session._cuda_artifact_leaf = lambda name, op: _leaf(
                c.lifecycle, name + "." + op, "runtime_import_or_init", "python_extension_" + op + "_v1"
            )
            admission = NS(records={}, session=session)
            session.runtime_owners.append(admission)

            def create(actual):
                assert actual is spec
                calls.append("create")
                return module

            def execute(actual):
                assert actual is module
                calls.append("exec")

            loader = _ExternalSourceLoader(
                admission, "extension", NS(create_module=create, exec_module=execute), "/unused", None
            )

            @contextmanager
            def stream_scope(actual):
                assert actual is c.stream
                yield
                assert admission.records["extension"]["module"] is module
                calls.append("stream_exit")
                if fault == "stream_exit":
                    raise RuntimeError("artifact boundary")

            c.runtime.cuda.stream = stream_scope
            original_state = reader._state

            def state(pristine):
                if fault == "after_query" and calls:
                    raise RuntimeError("artifact boundary")
                return original_state(pristine)

            reader._state = state
            append = session.runtime_observation.append

            def observed(kind, details):
                result = append(kind, details)
                if kind == "native_scope" and details["edge"] == "after" and fault == "after_append":
                    raise RuntimeError("artifact boundary")
                return result

            session.runtime_observation.append = observed
            assert _SourceSession._create_cuda_artifact(session, loader, spec, {"sequence": 7}) is module
            _SourceSession._exec_cuda_artifact(session, loader, module)
    assert admission.records["extension"]["module"] is module
    journal = session.runtime_observation.journal
    pairs = [(r["details"]["site"], r["details"]["edge"]) for r in journal if r["kind"] == "native_scope"]
    assert journal[1]["kind"] == "returned_owner"
    if fault:
        assert calls == ["create", "stream_exit"]
        assert pairs == [("cuda_artifact_create", "before")] + (
            [("cuda_artifact_create", "after")] if fault == "after_append" else []
        )
        assert (
            c.lifecycle.termination_evidence.lifecycle.to_payload()["summary"]["release_status"]
            == "not_authorized"
        )
    else:
        assert pairs == [
            ("cuda_artifact_" + op, edge) for op in ("create", "exec") for edge in ("before", "after")
        ]
