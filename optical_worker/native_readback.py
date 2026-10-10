"""Explicit validation D2H and reusable diagnostic staging under existing guards.

This is input verification, not B3 delivery, final readiness or a public worker
request. The input path remains single-use; the fixed correctness driver can
prepare reusable storage before seal. No runtime discovery or fallback sync.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

from .guarded_lifecycle import NativeGuardedSetupLifecycle
from .lifecycle import DeviceScopeDescriptor
from .native_contract import (
    CudaPreparedNativeInput,
    WarpPreparedNativeInput,
    assert_prepared_storage_registered,
    warp_dtype_from_host_dtype,
)
from .native_scope import reader_for_input
from .native_setup import (
    _array_operation_id,
    _descriptor,
    _enter_device_leaf,
    _event_ordinal,
    _torch_copy_exact,
    _torch_event_exact,
    _torch_record_event_exact,
    _torch_wait_event_host_exact,
    _warp_copy_exact,
    _warp_event_exact,
    _warp_record_event_exact,
    _warp_wait_event_host_exact,
)

_NO_READER_SCOPE = nullcontext()


def _leaf(lifecycle, operation, kind, runtime_identity):
    _enter_device_leaf(
        lifecycle,
        _descriptor(lifecycle, operation_id=operation, leaf_kind=kind, runtime_identity=runtime_identity),
    )


def _readback_prepared_native_input(source, prepared, *, lifecycle, runtime, target, stream):
    """One observed readback, using the same CPU content validator as a2.

    All pageable host arrays, native host views and the completion event stay
    held until the controller's terminal disposition, including registration
    failure after a returned owner. The caller owns the guard and abort policy.
    """
    if type(lifecycle) is not NativeGuardedSetupLifecycle:
        raise TypeError("validation readback requires the exact guarded lifecycle")
    lifecycle._require_setup()
    reader = reader_for_input(lifecycle, runtime, target, stream, phase="validating_input", input_entry=True)
    if type(prepared) not in (CudaPreparedNativeInput, WarpPreparedNativeInput):
        raise TypeError("validation readback requires exact registered prepared input")
    if prepared.backend != lifecycle._lane_id:
        raise ValueError("validation readback backend differs from its guarded lane")
    if prepared.source is not source or prepared.device_ordinal != lifecycle.target_device_ordinal:
        raise ValueError("validation readback source/target differs from prepared storage")
    assert_prepared_storage_registered(prepared)
    lifecycle.require_adopted(source, prepared, runtime, target, stream)
    cuda = type(prepared) is CudaPreparedNativeInput
    if cuda:
        if (
            target.type != "cuda"
            or target.index != lifecycle.target_device_ordinal
            or stream.device != target
        ):
            raise ValueError("CUDA validation stream/target drift")
    elif (
        not target.is_cuda
        or int(target.ordinal) != lifecycle.target_device_ordinal
        or stream.device is not target
    ):
        raise ValueError("Warp validation stream/target drift")

    # This list is adopted first, and every returned owner goes into it before
    # the more specific registration, which fault injection may reject.
    retained = []
    lifecycle.adopt_host_owner(
        retained, role="validation_readback_custody", operation_id="validation_readback_custody"
    )
    if cuda:
        return _readback_held_input(source, prepared, lifecycle, runtime, target, stream, cuda, retained)
    _leaf(lifecycle, "indexed_validation_scope", "runtime_import_or_init", "warp_scoped_device_v1")
    # The real Warp D2H helper's native ContextGuard does not restore a prior
    # null context. Keep its indexed work inside the runtime's restoring scope.
    # The report remains held on error and is returned only after scope exit.
    with reader.outer_scope("readback_scope") if reader is not None else nullcontext():
        with runtime.ScopedDevice(target):
            report = _readback_held_input(
                source, prepared, lifecycle, runtime, target, stream, cuda, retained
            )
    return report


def _readback_held_input(source, prepared, lifecycle, runtime, target, stream, cuda, retained):
    from .native_input import _validate_native_input_host_observations

    arrays = tuple((row.name, row.shape, row.dtype, prepared.arrays[row.name])
                   for row in source.manifest.arrays)
    observations = _readback_native_arrays(arrays, lifecycle, runtime, target, stream, cuda, retained)

    assert_prepared_storage_registered(prepared)
    report = _validate_native_input_host_observations(source, prepared, observations)
    retained.append(report)
    lifecycle.adopt_host_owner(
        report, role="native_input_validation_report", operation_id="native_input_validation_report"
    )
    return report


def _readback_native_arrays(arrays, lifecycle, runtime, target, stream, cuda, retained):
    """Shared setup D2H body; caller retains the list and owns the active guard."""
    reader = reader_for_input(lifecycle, runtime, target, stream, phase="validating_input")

    observations = []
    retained.append(observations)
    for name, shape, dtype, device_array in arrays:
        host, view = _prepare_readback_host(name, shape, dtype, lifecycle, runtime, cuda, retained)
        lifecycle.require_adopted(host, view, device_array, stream)
        _copy_readback_array(name, view, device_array, lifecycle, runtime, stream, cuda, reader)
        observations.append((name, host))

    event = _prepare_readback_event(lifecycle, runtime, target, stream, cuda, retained, reader)
    _record_wait_readback(event, lifecycle, runtime, stream, cuda, reader)
    return observations


def _prepare_readback_host(name, shape, dtype, lifecycle, runtime, cuda, retained):
    import numpy as np

    host = np.empty(shape, dtype=np.dtype(dtype), order="C")
    retained.append(host)
    lifecycle.adopt_host_owner(
        host, role="validation_d2h_host_storage", operation_id=_array_operation_id("validation_host", name),
    )
    view = (runtime.from_numpy(host) if cuda else runtime.array(
        host, dtype=warp_dtype_from_host_dtype(runtime, host.dtype), device="cpu", copy=False,
    ))
    retained.append(view)
    lifecycle.adopt_host_owner(
        view, role="validation_d2h_host_view", operation_id=_array_operation_id("validation_view", name),
    )
    return host, view


def _copy_readback_array(name, view, device_array, lifecycle, runtime, stream, cuda, reader, scopes=None):
    _leaf(
        lifecycle, _array_operation_id("validation_d2h", name), "device_to_host_copy",
        "torch_copy_explicit_stream_v1" if cuda else "warp_copy_explicit_stream_v1",
    )
    with reader.operation("copy") if reader is not None else _NO_READER_SCOPE:
        if cuda:
            _torch_copy_exact(runtime, view, device_array, stream, _scopes=scopes)
        else:
            _warp_copy_exact(runtime, view, device_array, stream)


def _prepare_readback_event(lifecycle, runtime, target, stream, cuda, retained, reader):
    _leaf(
        lifecycle,
        "validation_event_create",
        "event_create",
        "torch_cuda_event_v1" if cuda else "warp_event_explicit_device_v1",
    )
    with reader.operation("event_create") if reader is not None else _NO_READER_SCOPE:
        event = _torch_event_exact(runtime) if cuda else _warp_event_exact(runtime, target=target)
        if reader is not None:
            reader.hold_returned(event)
    retained.append(event)
    lifecycle.adopt_returned_device_owner(
        event,
        actual_device_ordinal=_event_ordinal(event, cuda=cuda, lifecycle=lifecycle, allow_unbound=True),
        role="validation_readback_event",
        operation_id="validation_event_create",
    )
    return event


def _record_wait_readback(event, lifecycle, runtime, stream, cuda, reader):
    lifecycle.require_adopted(event, stream)
    _leaf(
        lifecycle,
        "validation_event_record",
        "event_record",
        "torch_event_record_indexed_stream_v1" if cuda else "warp_stream_record_event_v1",
    )
    if cuda:
        with reader.operation("event_record") if reader is not None else _NO_READER_SCOPE:
            _torch_record_event_exact(event, stream)
    else:
        with reader.operation("event_record") if reader is not None else _NO_READER_SCOPE:
            _warp_record_event_exact(stream, event)
    actual_ordinal = _event_ordinal(event, cuda=cuda, lifecycle=lifecycle)
    if actual_ordinal != lifecycle.target_device_ordinal:
        # The next wait would touch this observed device. Commit that scope
        # before the controller rejects foreign-device work; never wait first.
        _enter_device_leaf(
            lifecycle,
            DeviceScopeDescriptor.exact_target(
                operation_id="validation_foreign_event_wait",
                leaf_kind="event_wait",
                device_ordinal=actual_ordinal,
                scope_source="returned_readback_event",
                runtime_identity="observed_event_device_v1",
            ),
        )
        raise RuntimeError("foreign readback event scope was unexpectedly accepted")
    _leaf(
        lifecycle,
        "validation_event_wait",
        "event_wait",
        "torch_event_synchronize_v1" if cuda else "warp_synchronize_event_v1",
    )
    if cuda:
        with reader.operation("event_wait") if reader is not None else _NO_READER_SCOPE:
            _torch_wait_event_host_exact(event)
    else:
        with reader.operation("event_wait") if reader is not None else _NO_READER_SCOPE:
            _warp_wait_event_host_exact(runtime, event)



@dataclass(frozen=True, eq=False, slots=True)
class _PreparedNativeReadback:
    lifecycle: object
    runtime: object
    target: object
    stream: object
    cuda: bool
    observations: tuple
    copies: tuple
    event: object
    device_scope: object


def _prepare_native_readback(arrays, lifecycle, runtime, target, stream, cuda, retained):
    """Prepare reusable diagnostic operands and initialize actual event storage.

    The caller owns the original setup ticket and an already-adopted custody
    list. This creates no final-operation authority and grants no result lease.
    """
    if type(lifecycle) is not NativeGuardedSetupLifecycle or type(cuda) is not bool:
        raise TypeError("prepared readback requires its exact guarded lifecycle and lane")
    lifecycle._require_setup()
    if (cuda != (lifecycle._lane_id == "cuda")
            or (target.type != "cuda" if cuda else not target.is_cuda)
            or (stream.device != target if cuda else stream.device is not target)
            or (target.index if cuda else target.ordinal) != lifecycle.target_device_ordinal):
        raise ValueError("prepared readback differs from its retained native target")
    lifecycle.require_adopted(retained, runtime, target, stream)
    observations, copies = [], []
    retained.extend((observations, copies))
    for name, shape, dtype, device_array in arrays:
        lifecycle.require_adopted(device_array)
        host, view = _prepare_readback_host(name, shape, dtype, lifecycle, runtime, cuda, retained)
        scopes = None
        if cuda:
            device_scope = runtime.cuda.device(target)
            retained.append(device_scope)
            stream_scope = runtime.cuda.stream(stream)
            retained.append(stream_scope)
            scopes = (device_scope, stream_scope)
        observations.append((name, host))
        copies.append((name, view, device_array, scopes))
    device_scope = nullcontext() if cuda else runtime.ScopedDevice(target)
    retained.append(device_scope)
    if not cuda:
        _leaf(lifecycle, "indexed_validation_scope", "runtime_import_or_init", "warp_scoped_device_v1")
    with device_scope:
        event = _prepare_readback_event(lifecycle, runtime, target, stream, cuda, retained, None)
        # Torch events are lazy. A wrapper alone is not prepared native storage.
        _record_wait_readback(event, lifecycle, runtime, stream, cuda, None)
    prepared = _PreparedNativeReadback(
        lifecycle, runtime, target, stream, cuda, tuple(observations), tuple(copies), event, device_scope,
    )
    retained.append(prepared)
    lifecycle.adopt_host_owner(
        prepared, role="prepared_validation_readback", operation_id="prepared_readback",
    )
    return prepared


def _copy_prepared_native_readback(prepared, lifecycle):
    """Reuse prepared operands under an original setup guard.

    Final RESOLVE use stays closed until its separately bound D1 observation
    aperture is implemented. This helper never creates or completes a ticket.
    """
    if type(prepared) is not _PreparedNativeReadback or prepared.lifecycle is not lifecycle:
        raise ValueError("prepared readback belongs to another lifecycle")
    lifecycle._require_setup()
    lifecycle.require_adopted(prepared, prepared.event, prepared.stream)
    if prepared.cuda:
        same_target = prepared.stream.device == prepared.target and prepared.event.device == prepared.target
    else:
        same_target = prepared.stream.device is prepared.target and prepared.event.device is prepared.target
    if not same_target:
        raise ValueError("prepared readback native target changed")
    if not prepared.cuda:
        _leaf(lifecycle, "indexed_validation_scope", "runtime_import_or_init", "warp_scoped_device_v1")
    with prepared.device_scope:
        for name, view, device_array, scopes in prepared.copies:
            _copy_readback_array(name, view, device_array, lifecycle, prepared.runtime,
                                 prepared.stream, prepared.cuda, None, scopes)
        _record_wait_readback(
            prepared.event, lifecycle, prepared.runtime, prepared.stream, prepared.cuda, None,
        )
    return prepared.observations
