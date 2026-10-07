"""Explicit validation D2H inside an already armed guarded setup segment.

This is input verification, not B3 delivery, final readiness or a public worker
request. The fixed captured-session driver validates its original runtime binding
and calls this once. No runtime discovery, callback or fallback synchronization.
"""

from __future__ import annotations

from .guarded_lifecycle import NativeGuardedSetupLifecycle
from .lifecycle import DeviceScopeDescriptor
from .native_contract import (
    CudaPreparedNativeInput,
    WarpPreparedNativeInput,
    assert_prepared_storage_registered,
    warp_dtype_from_host_dtype,
)
from .native_setup import (
    _array_operation_id,
    _descriptor,
    _enter_device_leaf,
    _torch_copy_exact,
    _torch_event_exact,
    _torch_record_event_exact,
    _torch_wait_event_host_exact,
    _warp_copy_exact,
    _warp_event_exact,
    _warp_record_event_exact,
    _warp_wait_event_host_exact,
)


def _leaf(lifecycle, operation, kind, runtime_identity):
    _enter_device_leaf(
        lifecycle,
        _descriptor(lifecycle, operation_id=operation, leaf_kind=kind, runtime_identity=runtime_identity),
    )


def _event_ordinal(event, *, cuda, lifecycle, allow_unbound=False):
    try:
        device = event.device
        if cuda and device is None and allow_unbound:
            # Torch Event has no native allocation until the first record.
            # Its prospective scope is the exact stream passed below; actual
            # bound metadata is mandatory after record, before any wait.
            return lifecycle.target_device_ordinal
        ordinal = device.index if cuda else device.ordinal
        if (
            type(ordinal) is not int
            or not 0 <= ordinal <= 1023
            or (cuda and device.type != "cuda")
            or (not cuda and not device.is_cuda)
        ):
            raise ValueError("readback event has no valid indexed CUDA device")
        return ordinal
    except BaseException:
        lifecycle.mark_unknown_device_scope(
            source="readback_event_metadata", reason="event_device_unavailable"
        )
        raise


def _readback_prepared_native_input(source, prepared, *, lifecycle, runtime, target, stream):
    """One observed readback, using the same CPU content validator as a2.

    All pageable host arrays, native host views and the completion event stay
    held until the controller's terminal disposition, including registration
    failure after a returned owner. The caller owns the guard and abort policy.
    """
    if type(lifecycle) is not NativeGuardedSetupLifecycle:
        raise TypeError("validation readback requires the exact guarded lifecycle")
    lifecycle._require_setup()
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

    import numpy as np

    from .native_input import _validate_native_input_host_observations

    # This list is adopted first, and every returned owner goes into it before
    # the more specific registration, which fault injection may reject.
    retained = []
    lifecycle.adopt_host_owner(
        retained, role="validation_readback_custody", operation_id="validation_readback_custody"
    )
    observations = []
    retained.append(observations)
    for manifest in source.manifest.arrays:
        device_array = prepared.arrays[manifest.name]
        host = np.empty(manifest.shape, dtype=np.dtype(manifest.dtype), order="C")
        retained.append(host)
        lifecycle.adopt_host_owner(
            host,
            role="validation_d2h_host_storage",
            operation_id=_array_operation_id("validation_host", manifest.name),
        )
        if cuda:
            view = runtime.from_numpy(host)
        else:
            view = runtime.array(
                host, dtype=warp_dtype_from_host_dtype(runtime, host.dtype), device="cpu", copy=False
            )
        retained.append(view)
        lifecycle.adopt_host_owner(
            view,
            role="validation_d2h_host_view",
            operation_id=_array_operation_id("validation_view", manifest.name),
        )
        lifecycle.require_adopted(host, view, device_array, stream)
        _leaf(
            lifecycle,
            _array_operation_id("validation_d2h", manifest.name),
            "device_to_host_copy",
            "torch_copy_explicit_stream_v1" if cuda else "warp_copy_explicit_stream_v1",
        )
        if cuda:
            _torch_copy_exact(runtime, view, device_array, stream)
        else:
            _warp_copy_exact(runtime, view, device_array, stream)
        observations.append((manifest.name, host))

    _leaf(
        lifecycle,
        "validation_event_create",
        "event_create",
        "torch_cuda_event_v1" if cuda else "warp_event_explicit_device_v1",
    )
    event = _torch_event_exact(runtime) if cuda else _warp_event_exact(runtime, target=target)
    retained.append(event)
    lifecycle.adopt_returned_device_owner(
        event,
        actual_device_ordinal=_event_ordinal(event, cuda=cuda, lifecycle=lifecycle, allow_unbound=True),
        role="validation_readback_event",
        operation_id="validation_event_create",
    )
    lifecycle.require_adopted(event, stream)
    _leaf(
        lifecycle,
        "validation_event_record",
        "event_record",
        "torch_event_record_indexed_stream_v1" if cuda else "warp_stream_record_event_v1",
    )
    if cuda:
        _torch_record_event_exact(event, stream)
    else:
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
        _torch_wait_event_host_exact(event)
    else:
        _warp_wait_event_host_exact(runtime, event)

    assert_prepared_storage_registered(prepared)
    report = _validate_native_input_host_observations(source, prepared, observations)
    retained.append(report)
    lifecycle.adopt_host_owner(
        report, role="native_input_validation_report", operation_id="native_input_validation_report"
    )
    return report
