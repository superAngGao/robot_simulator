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
    state = SimpleNamespace(trace=[], pending=[], views=[], hosts=[], fault=None, event=None)
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

    runtime = SimpleNamespace(
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
    if fault in {"copy", "event", "record", "adopt_view", "adopt_event"}:
        assert "wait" not in case.state.trace


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
