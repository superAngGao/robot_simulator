"""Readback ordering/custody units: real guard and a2 registry, fake GPU leaves.

CPU-backed test arrays are NOT real CUDA allocation or cold-admission evidence.
The shared production host validator, not a test mathematical copy, checks values.
"""

from __future__ import annotations

import gc
import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from optical_worker.guarded_lifecycle import NativeGuardedSetupLifecycle
from optical_worker.native_contract import (
    CudaPreparedNativeInput,
    WarpPreparedNativeInput,
    register_prepared_native_input,
)
from optical_worker.native_input import CanonicalNativeInput
from optical_worker.native_readback import _readback_prepared_native_input
from tests.support.guarded_peer import Peer


class _Storage:
    def __init__(self, data):
        self.data, self._cdata = data, id(self)

    def nbytes(self):
        return self.data.nbytes

    def data_ptr(self):
        return self.data.ctypes.data


class _CudaArray:
    def __init__(self, data, device):
        self.data, self.device = data.copy(), device
        self.storage = _Storage(self.data)
        self.dtype, self.shape = self.data.dtype, self.data.shape

    def untyped_storage(self):
        return self.storage

    def data_ptr(self):
        return self.storage.data_ptr()

    def numel(self):
        return self.data.size

    def element_size(self):
        return self.dtype.itemsize

    def stride(self):
        return tuple(stride // self.dtype.itemsize for stride in self.data.strides)

    def is_contiguous(self):
        return True


class _WarpArray(_CudaArray):
    is_contiguous = True

    @property
    def ptr(self):
        return self.data_ptr()

    @property
    def capacity(self):
        return self.data.nbytes

    @property
    def strides(self):
        return self.data.strides


@pytest.fixture(params=["cuda", "warp"])
def readback_case(request, monkeypatch):
    lane = request.param
    peer = Peer()
    lifecycle = NativeGuardedSetupLifecycle(lane_id=lane, channel=peer.channel, configuration=peer.config)
    target = SimpleNamespace(type="cuda", index=1, ordinal=1, is_cuda=True)
    stream = SimpleNamespace(device=target)
    state = SimpleNamespace(trace=[], pending=[], views=[], hosts=[], fault=None, event=None,
                            scope_entries=0, scope_exits=0, current_context=None)
    source = CanonicalNativeInput.from_arrays(
        case_id="asymmetric_synthetic_7x3_v1",
        materializer_id="readback_unit_not_result_capable",
        arrays={
            "empty": np.empty((0, 3), dtype=np.float32),
            "float_values": np.array([[0.0, 3.25, np.nan]], dtype=np.float32),
            "integer_values": np.array([7, -3, 11], dtype=np.int32),
        },
        array_semantics={"empty": "unit.empty", "float_values": "unit.float", "integer_values": "unit.int"},
        topology_arrays=("integer_values",),
        semantics={"purpose": "readback unit"},
    )
    array_type = _CudaArray if lane == "cuda" else _WarpArray
    prepared_type = CudaPreparedNativeInput if lane == "cuda" else WarpPreparedNativeInput
    prepared = prepared_type(
        source.manifest,
        1,
        {name: array_type(value, target) for name, value in source.arrays.items()},
        object(),
        source,
    )
    register_prepared_native_input(prepared, source)

    def mark(operation):
        lifecycle._require_setup()
        if lane == "warp":
            assert state.current_context is target
        trace = lifecycle.evidence().to_payload()["transition_trace"]
        assert trace[-1]["operation"] == "low_level_leaf_entered"
        state.trace.append(operation)

    class View:
        def __init__(self, host):
            self.host = host

        def copy_(self, device_array, *, non_blocking):
            assert non_blocking is True
            mark("copy")
            lifecycle.require_adopted(self, self.host, device_array, stream)
            state.pending.append((self.host, device_array.data))
            if state.fault == "copy":
                raise RuntimeError("copy may already be submitted")

    def host_view(host):
        lifecycle.require_adopted(host)
        view = View(host)
        state.views.append(weakref.ref(view))
        state.hosts.append(weakref.ref(host))
        return view

    class Event:
        def __init__(self):
            self.device = None if lane == "cuda" else target
            if state.fault == "foreign_create":
                self.device = SimpleNamespace(index=0, ordinal=0, type="cuda", is_cuda=True)

        def record(self, actual_stream):
            assert actual_stream is stream
            lifecycle.require_adopted(self, stream)
            mark("record")
            self.device = actual_stream.device
            if state.fault == "foreign_record":
                self.device = SimpleNamespace(index=0, ordinal=0, type="cuda", is_cuda=True)
            if state.fault == "unknown_record":
                self.device = None
            if state.fault == "record":
                raise RuntimeError("event record failed")

        def synchronize(self):
            lifecycle.require_adopted(self)
            mark("wait")
            if state.fault == "wait":
                raise RuntimeError("event wait failed")
            for host, values in state.pending:
                np.copyto(host, values)
            if state.fault == "content":
                state.pending[-1][0][0] += 1
            state.pending.clear()

    def event(**kwargs):
        mark("event")
        if state.fault == "event":
            raise RuntimeError("event creation failed after copies")
        if lane == "cuda":
            assert kwargs == {"enable_timing": False}
        else:
            assert kwargs == {"target": target} or kwargs == {"device": target}
        value = Event()
        state.event = weakref.ref(value)
        return value

    class Scope:
        def __init__(self, device):
            assert device is target

        def __enter__(self):
            lifecycle._require_setup()
            assert lifecycle._guard_events[-1]["operation"] == "low_level_leaf_entered"
            state.scope_entries += 1
            self.previous = state.current_context
            state.current_context = target

        def __exit__(self, exc_type, exc, traceback):
            state.scope_exits += 1
            state.current_context = self.previous
            if state.fault == "scope_exit":
                raise RuntimeError("scope restore failed")

    runtime = SimpleNamespace(
        ScopedDevice=Scope,
        from_numpy=host_view,
        cuda=SimpleNamespace(
            device=lambda value: nullcontext(), stream=lambda value: nullcontext(), Event=event
        ),
        array=lambda host, **kwargs: host_view(host),
        copy=lambda view, array, **kwargs: view.copy_(array, non_blocking=True),
        Event=event,
        synchronize_event=lambda value: value.synchronize(),
    )
    for dtype in (
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
    ):
        setattr(runtime, dtype, np.dtype(dtype))
    stream.record_event = lambda value: value.record(stream)
    for value in (source, prepared, runtime, target, stream, *prepared.arrays.values()):
        lifecycle.adopt_host_owner(value, role="unit_existing_owner", operation_id="unit_existing_owner")
    original_adopt = lifecycle.adopt_host_owner

    def adopt(value, **kwargs):
        if state.fault == "adopt_view" and kwargs["role"] == "validation_d2h_host_view":
            raise RuntimeError("view registration failed")
        if state.fault == "adopt_report" and kwargs["role"] == "native_input_validation_report":
            raise RuntimeError("report registration failed")
        return original_adopt(value, **kwargs)

    original_device_adopt = lifecycle.adopt_returned_device_owner

    def adopt_device(value, **kwargs):
        if state.fault == "adopt_event" and kwargs["role"] == "validation_readback_event":
            raise RuntimeError("event registration failed after copies")
        return original_device_adopt(value, **kwargs)

    monkeypatch.setattr(lifecycle, "adopt_host_owner", adopt)
    monkeypatch.setattr(lifecycle, "adopt_returned_device_owner", adopt_device)
    try:
        yield SimpleNamespace(
            source=source,
            prepared=prepared,
            lifecycle=lifecycle,
            runtime=runtime,
            target=target,
            stream=stream,
            state=state,
        )
    finally:
        if lifecycle.termination_evidence is None:
            lifecycle.terminate(
                fault="setup_failure", detection_phase="unit_end", emergency_target_synchronize=lambda: None
            )
        peer.close()


def _run(case):
    return _readback_prepared_native_input(
        case.source,
        case.prepared,
        lifecycle=case.lifecycle,
        runtime=case.runtime,
        target=case.target,
        stream=case.stream,
    )


def test_actual_shared_validator_sees_only_completed_explicit_host_buffers(readback_case, monkeypatch):
    from optical_worker import native_input

    case = readback_case
    original = native_input._validate_native_input_host_observations
    calls = []

    def check(source, prepared, observations):
        assert source is case.source and prepared is case.prepared
        assert case.state.trace[-1] == "wait" and not case.state.pending
        calls.append(True)
        return original(source, prepared, observations)

    monkeypatch.setattr(native_input, "_validate_native_input_host_observations", check)
    with case.lifecycle.segment():
        report = _run(case)
    assert calls == [True]
    assert case.state.trace == ["copy", "copy", "copy", "event", "record", "wait"]
    assert report.content_sha256 == case.source.manifest.content_sha256
    assert report.validated_arrays == tuple(case.prepared.arrays)
    for reference in case.state.hosts + case.state.views:
        assert reference() is not None
        case.lifecycle.require_adopted(reference())
    assert case.lifecycle.evidence().to_payload()["summary"]["normal_complete_count"] == 1
    assert not hasattr(report, "warmup_frames_completed")
    assert case.state.scope_entries == case.state.scope_exits == (1 if case.prepared.backend == "warp" else 0)
    assert case.state.current_context is None


@pytest.mark.parametrize(
    "fault", ["copy", "event", "record", "wait", "adopt_view", "adopt_event", "adopt_report", "content"]
)
def test_failed_readback_retains_operands_without_second_sync_or_complete(readback_case, fault):
    case = readback_case
    case.state.fault = fault
    with pytest.raises((RuntimeError, ValueError)):
        with case.lifecycle.segment():
            _run(case)
    gc.collect()
    assert all(reference() is not None for reference in case.state.hosts + case.state.views)
    if case.state.event is not None:
        assert case.state.event() is not None
    summary = case.lifecycle.termination_evidence.lifecycle.to_payload()["summary"]
    assert summary["normal_complete_count"] == 0
    assert summary["emergency_sync_attempt_count"] == 0
    assert case.lifecycle.guardian.state == "TRANSFERRED_TO_QUARANTINE"
    assert case.state.trace.count("wait") <= 1
    assert case.state.current_context is None
    assert case.state.scope_entries == case.state.scope_exits
    if fault in {"copy", "event", "record", "adopt_view", "adopt_event"}:
        assert "wait" not in case.state.trace


def test_scope_restore_error_prevents_returning_the_retained_report(readback_case):
    case = readback_case
    case.state.fault = "scope_exit"
    if case.prepared.backend == "cuda":
        with case.lifecycle.segment():
            assert _run(case).content_sha256 == case.source.manifest.content_sha256
        assert case.state.scope_entries == 0
    else:
        with pytest.raises(RuntimeError, match="scope restore failed"):
            with case.lifecycle.segment():
                _run(case)
        summary = case.lifecycle.termination_evidence.lifecycle.to_payload()["summary"]
        assert summary["normal_complete_count"] == summary["emergency_sync_attempt_count"] == 0
        assert summary["release_status"] == "not_authorized"
        assert any(row.role == "native_input_validation_report" for row in case.lifecycle.guardian._records)
        assert case.state.scope_entries == case.state.scope_exits == 1


def test_readback_cannot_start_outside_an_armed_segment(readback_case):
    case = readback_case
    with pytest.raises(RuntimeError, match="armed"):
        _run(case)
    assert not case.state.trace and not case.state.hosts


@pytest.mark.parametrize("fault", ["foreign_create", "foreign_record", "unknown_record"])
def test_event_actual_device_scope_is_checked_before_wait(readback_case, fault):
    case = readback_case
    case.state.fault = fault
    with pytest.raises((RuntimeError, ValueError, AttributeError)):
        with case.lifecycle.segment():
            _run(case)
    assert "wait" not in case.state.trace
    summary = case.lifecycle.termination_evidence.lifecycle.to_payload()["summary"]
    assert summary["disposition"] == "multi_or_unknown_device_quarantine_exit"
    assert summary["emergency_sync_attempt_count"] == 0
    if fault.startswith("foreign"):
        assert summary["possible_dirty_device_ordinals"] == [0, 1]
    else:
        assert summary["unknown_device_scope"] is True


def test_storage_rebinding_is_rejected_before_any_readback_leaf(readback_case):
    case = readback_case
    array = case.prepared.arrays["float_values"]
    array.data = array.data.copy()
    array.storage = _Storage(array.data)
    with pytest.raises(ValueError, match="physical allocation"):
        with case.lifecycle.segment():
            _run(case)
    assert not case.state.trace and not case.state.hosts


def _prepare_reusable(case):
    from optical_worker.native_readback import _prepare_native_readback

    retained = []
    case.lifecycle.adopt_host_owner(retained, role="unit_staging", operation_id="unit_staging")
    arrays = tuple((row.name, row.shape, row.dtype, case.prepared.arrays[row.name])
                   for row in case.source.manifest.arrays)
    return _prepare_native_readback(arrays, case.lifecycle, case.runtime, case.target,
                                    case.stream, case.prepared.backend == "cuda", retained)


def test_reusable_readback_initializes_event_and_reuses_original_host_storage(readback_case, monkeypatch):
    from optical_worker.native_readback import _copy_prepared_native_readback

    case = readback_case
    with case.lifecycle.segment():
        prepared = _prepare_reusable(case)
    assert case.state.trace == ["event", "record", "wait"]
    assert prepared.event.device is case.target
    hosts = tuple(host for _, host in prepared.observations)
    original_ids = tuple(id(host) for host in hosts)
    event = prepared.event

    def late_creation(*args, **kwargs):
        pytest.fail("readback created storage, host views, events or native scopes after preparation")

    monkeypatch.setattr(np, "empty", late_creation)
    for name in ("from_numpy", "array", "Event", "ScopedDevice"):
        monkeypatch.setattr(case.runtime, name, late_creation)
    for name in ("device", "stream", "Event"):
        monkeypatch.setattr(case.runtime.cuda, name, late_creation)
    for generation in range(1, 5):
        case.prepared.arrays["float_values"].data[:] = generation + 0.25
        case.prepared.arrays["integer_values"].data[:] = generation
        with case.lifecycle.segment():
            observed = _copy_prepared_native_readback(prepared, case.lifecycle)
        assert observed is prepared.observations and prepared.event is event
        assert tuple(id(host) for _, host in observed) == original_ids
        for name, host in observed:
            np.testing.assert_array_equal(host, case.prepared.arrays[name].data)
    assert case.state.trace == ["event", "record", "wait"] + ["copy", "copy", "copy", "record", "wait"] * 4
    assert not case.state.pending and case.state.current_context is None
    assert case.state.scope_entries == case.state.scope_exits


@pytest.mark.parametrize("fault", ["event", "record", "wait", "adopt_view", "adopt_event"])
def test_prepared_readback_failure_retains_partial_owners(readback_case, fault):
    case = readback_case
    case.state.fault = fault
    with pytest.raises(RuntimeError):
        with case.lifecycle.segment():
            _prepare_reusable(case)
    gc.collect()
    assert all(reference() is not None for reference in case.state.hosts + case.state.views)
    if case.state.event is not None:
        assert case.state.event() is not None
    assert case.lifecycle.guardian.state == "TRANSFERRED_TO_QUARANTINE"
    assert case.state.trace.count("wait") <= 1 and "copy" not in case.state.trace


@pytest.mark.parametrize("fault", ["copy", "record", "wait"])
def test_reusable_readback_failure_never_returns_or_waits_again(readback_case, fault):
    from optical_worker.native_readback import _copy_prepared_native_readback

    case = readback_case
    with case.lifecycle.segment():
        prepared = _prepare_reusable(case)
    case.state.trace.clear()
    case.state.fault = fault
    with pytest.raises(RuntimeError):
        with case.lifecycle.segment():
            _copy_prepared_native_readback(prepared, case.lifecycle)
    assert case.lifecycle.guardian.state == "TRANSFERRED_TO_QUARANTINE"
    assert case.state.trace.count("wait") == (1 if fault == "wait" else 0)
    assert prepared.event is case.state.event()
    before = tuple(case.state.trace)
    with pytest.raises(RuntimeError):
        _copy_prepared_native_readback(prepared, case.lifecycle)
    assert tuple(case.state.trace) == before
