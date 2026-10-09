"""Register-before-copy a2 preparation for the closed worker.

The module itself is stdlib-only at import time.  Backend imports happen only
inside the two exact-lane entry points, after their setup lifecycle and
guardian already exist.
"""

from __future__ import annotations


import weakref
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .lifecycle import (
    DeviceScopeDescriptor,
    NativeWorkerSetupLifecycle,
    SetupQuiescenceEvidence,
    TransportTerminationEvidence,
    _array_operation_id,
)
from .native_contract import (
    CudaPreparedNativeInput,
    WarpPreparedNativeInput,
    register_prepared_native_input,
    warp_dtype_from_host_dtype,
)


_CUDA_EXTENSION_SETS = {}


class CudaExtensionSet:
    """Only the captured source session registers a usable extension set."""

    __slots__ = ("session", "controller", "binding", "admission", "recipes")

    def __init__(self):
        raise TypeError("CUDA extension sets are issued by their source session")

    def __setattr__(self, name, value):
        raise TypeError("CUDA extension set is frozen")

    def __delattr__(self, name):
        raise TypeError("CUDA extension set is frozen")


def _require_cuda_extension(owner, recipe_name, context):
    if type(owner) is not CudaExtensionSet:
        raise TypeError("captured preparation needs its issued CUDA extension set")
    record = _CUDA_EXTENSION_SETS.get(owner.session)
    if record is None or record[0] is not owner:
        raise ValueError("CUDA extension set is copied, foreign or unregistered")
    session = record[1][0].session
    try:
        module = session.require_cuda_extension(owner, recipe_name)
        # The set is checked before inspecting caller context metadata. The
        # registered stream must be the actual stream used by preparation.
        from optics.cuda_timeline import CudaOpticalExecutionContext

        binding = record[1][0]
        if (type(context) is not CudaOpticalExecutionContext or context.device != binding.target
                or context.stream is not binding.stream or context.stream_handle != binding.stream_identity[1]):
            raise ValueError("CUDA factory context differs from its issued runtime binding")
        context._validate_identity()
        return module
    except BaseException as error:
        session.fail(error)
        raise


def _cuda_artifact_create_exact(torch, target, stream, loader, spec, frame):
    with torch.cuda.device(target):
        with torch.cuda.stream(stream):
            module = loader.delegate.create_module(spec)
            loader._retain_created_module(module, frame)
            return module


def _cuda_artifact_exec_exact(torch, target, stream, delegate, module):
    with torch.cuda.device(target):
        with torch.cuda.stream(stream):
            delegate.exec_module(module)


def _prepare_cuda_factory_owners(session, prepared, torch, target, stream, extensions):
    """Reuse ordinary factories and retain each return inside its device scope."""
    session._runtime_leaf("cuda_factory_definitions", "runtime_import_or_init")
    from optics.cuda_timeline import CudaOpticalExecutionContext
    from optics.cuda_direct_light import CudaDirectLightNativeRenderer
    from optical_worker.native_input import (
        build_prepared_cuda_scene_update_factory, build_prepared_cuda_lbvh_rebuild_factory,
    )
    from optical_worker.native_contract import assert_prepared_storage_registered

    session._runtime_leaf("cuda_factory_context", "runtime_import_or_init")
    with torch.cuda.device(target), torch.cuda.stream(stream):
        context = CudaOpticalExecutionContext(device=target, stream=stream)
        session._hold_runtime_owner(context, "cuda_factory_context")
        session._runtime_leaf("cuda_scene_factory", "runtime_import_or_init")
        assert_prepared_storage_registered(prepared)
        scene = build_prepared_cuda_scene_update_factory(prepared, context=context, extensions=extensions)
        session._hold_runtime_owner(scene, "cuda_scene_factory")
        session._runtime_leaf("cuda_lbvh_factory", "runtime_import_or_init")
        lbvh = build_prepared_cuda_lbvh_rebuild_factory(prepared, context=context, extensions=extensions)
        session._hold_runtime_owner(lbvh, "cuda_lbvh_factory")
        session._runtime_leaf("cuda_render_factory", "runtime_import_or_init")
        renderer = CudaDirectLightNativeRenderer.prepare(context, extensions=extensions)
        session._hold_runtime_owner(renderer, "cuda_render_factory")
        factories = (context, scene, lbvh, renderer)
        session._hold_runtime_owner(factories, "cuda_factory_owners")
        return factories


class NativeWorkerPreparationAborted(RuntimeError):
    """A setup exception reached its required terminal resource disposition."""

    def __init__(self, message: str, *, evidence: TransportTerminationEvidence) -> None:
        super().__init__(message)
        self.evidence = evidence


def _descriptor(
    lifecycle: NativeWorkerSetupLifecycle,
    *,
    operation_id: str,
    leaf_kind: str,
    runtime_identity: str,
) -> DeviceScopeDescriptor:
    return DeviceScopeDescriptor.exact_target(
        operation_id=operation_id,
        leaf_kind=leaf_kind,
        device_ordinal=lifecycle.target_device_ordinal,
        scope_source="closed_worker_allowlist",
        runtime_identity=runtime_identity,
    )


def _enter_device_leaf(
    lifecycle: NativeWorkerSetupLifecycle,
    descriptor: DeviceScopeDescriptor,
) -> None:
    permit = lifecycle.begin_device_leaf(descriptor)
    lifecycle.enter_device_leaf(permit)


def _torch_empty_exact(torch: Any, *, shape: tuple[int, ...], dtype: Any, target: Any) -> Any:
    return torch.empty(shape, dtype=dtype, device=target)


def _torch_copy_exact(torch: Any, destination: Any, source: Any, stream: Any) -> None:
    # StreamContext restores its source stream via set_stream(). Enter the
    # indexed device first so that source and destination streams belong to the
    # target, rather than restoring an ambient foreign-device stream. The outer
    # device guard uses Torch's lazy device restoration on CUDA 12.
    with torch.cuda.device(stream.device):
        with torch.cuda.stream(stream):
            destination.copy_(source, non_blocking=True)


def _torch_event_exact(torch: Any) -> Any:
    return torch.cuda.Event(enable_timing=False)


def _torch_record_event_exact(event: Any, stream: Any) -> None:
    event.record(stream)


def _torch_wait_event_host_exact(event: Any) -> None:
    event.synchronize()


def _warp_empty_exact(wp: Any, *, shape: tuple[int, ...], dtype: Any, target: Any) -> Any:
    return wp.empty(shape=shape, dtype=dtype, device=target)


def _warp_copy_exact(wp: Any, destination: Any, source: Any, stream: Any) -> None:
    wp.copy(destination, source, stream=stream)


def _warp_event_exact(wp: Any, *, target: Any) -> Any:
    return wp.Event(device=target)


def _warp_record_event_exact(stream: Any, event: Any) -> None:
    stream.record_event(event)


def _warp_wait_event_host_exact(wp: Any, event: Any) -> None:
    wp.synchronize_event(event)


def _warp_synchronize_target_exact(wp: Any, target: Any) -> None:
    with wp.ScopedDevice(target):
        wp.synchronize_device(target)


def _event_ordinal(event, *, cuda, lifecycle, allow_unbound=False):
    """Read held event metadata; only an unrecorded CUDA event may be unbound."""
    try:
        device = event.device
        if cuda and device is None and allow_unbound:
            return lifecycle.target_device_ordinal
        ordinal = device.index if cuda else device.ordinal
        if (type(ordinal) is not int or not 0 <= ordinal <= 1023
                or (cuda and device.type != "cuda") or (not cuda and not device.is_cuda)):
            raise ValueError("event has no valid indexed CUDA device")
        return ordinal
    except BaseException:
        lifecycle.mark_unknown_device_scope(source="event_device_metadata", reason="event_device_unavailable")
        raise


def _require_ready_event_target(event, *, cuda, lifecycle):
    actual = _event_ordinal(event, cuda=cuda, lifecycle=lifecycle)
    if actual != lifecycle.target_device_ordinal:
        # Register the observed foreign scope before any native wait. The
        # existing lifecycle rejects this descriptor without admitting work.
        _enter_device_leaf(lifecycle, DeviceScopeDescriptor.exact_target(
            operation_id="foreign_ready_event_wait", leaf_kind="event_wait",
            device_ordinal=actual, scope_source="returned_ready_event",
            runtime_identity="observed_event_device_v1",
        ))
        raise ValueError("foreign ready event scope was unexpectedly accepted")


def _abort(
    lifecycle: NativeWorkerSetupLifecycle,
    *,
    exception: BaseException,
    sync: Callable[[], None] | None,
) -> NativeWorkerPreparationAborted:
    if lifecycle.guardian.state == "CONSTRUCTING" and not lifecycle.guardian.contains_exact_owner(exception):
        lifecycle.adopt_host_owner(
            exception,
            role="setup_exception_graph",
            operation_id="setup_exception",
        )
    evidence = lifecycle.terminate(
        fault="setup_failure",
        detection_phase="a2_native_input_preparation",
        emergency_target_synchronize=sync,
    )
    return NativeWorkerPreparationAborted(
        "closed-worker native input preparation aborted",
        evidence=evidence,
    )


@dataclass(frozen=True, eq=False)
class _NativeUploadContext:
    """Internal mechanics shared by C0 and the guarded runtime binding.

    This is NOT readiness or a public execution capability. Only the fixed
    wrappers create it; one consumption burns its registration before upload.
    """

    lifecycle: NativeWorkerSetupLifecycle
    runtime: object
    target: object
    stream: object


_UPLOAD_CONTEXTS = {}


def _create_upload_context(lifecycle, runtime, target, stream):
    context = _NativeUploadContext(lifecycle, runtime, target, stream)
    key = id(context)

    def discard(reference):
        record = _UPLOAD_CONTEXTS.get(key)
        if record is not None and record[0] is reference:
            _UPLOAD_CONTEXTS.pop(key, None)

    _UPLOAD_CONTEXTS[key] = (
        weakref.ref(context, discard),
        lifecycle,
        runtime,
        target,
        stream,
    )
    try:
        lifecycle.adopt_host_owner(
            context,
            role="native_upload_context",
            operation_id="native_upload_context_create",
        )
    except BaseException:
        # The C0 lifecycle is slotted/non-weakrefable. Its exact strong identity
        # is retained only until immediate consumption, or failed construction.
        # Context construction enqueues no device work; existing operand custody
        # and the caller's failure policy remain responsible for runtime owners.
        _UPLOAD_CONTEXTS.pop(key, None)
        raise
    return context


def _consume_upload_context(context):
    if type(context) is not _NativeUploadContext:
        raise TypeError("upload core requires its exact internal context")
    record = _UPLOAD_CONTEXTS.pop(id(context), None)
    if (
        record is None
        or record[0]() is not context
        or record[1] is not context.lifecycle
        or record[2] is not context.runtime
        or record[3] is not context.target
        or record[4] is not context.stream
    ):
        raise ValueError("upload context is unregistered, consumed or has changed")
    context.lifecycle.require_adopted(context)
    return context.lifecycle, context.runtime, context.target, context.stream


@contextmanager
def _output_array_allocation(runtime, target, *, lane, stream=None, context=None):
    """One allocation implementation for ordinary and guarded output builders.

    The optional context is the existing single-use preparation context, not a
    callback. The guarded caller supplies it from its validated runtime binding.
    Every returned array is retained before metadata inspection or adoption.
    Native failures remain fatal even when caught within the original guard.
    """
    lifecycle = None
    if lane not in {"cuda", "warp"}:
        raise ValueError("unknown output allocation lane")
    if context is not None:
        from .guarded_lifecycle import NativeGuardedSetupLifecycle

        lifecycle, held_runtime, held_target, held_stream = _consume_upload_context(context)
        if type(lifecycle) is not NativeGuardedSetupLifecycle:
            raise TypeError("guarded output allocation requires its guarded lifecycle")
        lifecycle._require_setup()
        ticket = lifecycle._guard_ticket
        if (runtime is not held_runtime or lifecycle._lane_id != lane
                or (target != held_target if lane == "cuda" else target is not held_target)
                or (stream is not None and stream is not held_stream)):
            raise ValueError("output allocation differs from its runtime binding")
        target, stream = held_target, held_stream
        lifecycle.require_adopted(runtime, target, stream)
        owners = []
        lifecycle.adopt_host_owner(owners, role="slot_array_custody", operation_id="slot_array_custody")
        _enter_device_leaf(lifecycle, _descriptor(lifecycle,
            operation_id="slot_allocation_scope", leaf_kind="runtime_import_or_init",
            runtime_identity=lane + "_output_scope_v1"))

    active = False

    def empty(name, shape, dtype):
        if not active:
            raise ValueError("output allocator is outside its original scope")
        with lifecycle._observe_public_operation() if lifecycle is not None else nullcontext():
            if lifecycle is not None:
                lifecycle._require_setup()
                if lifecycle._guard_ticket is not ticket:
                    raise ValueError("output allocator is outside its original guard")
                _enter_device_leaf(lifecycle, _descriptor(lifecycle,
                    operation_id=_array_operation_id("slot_allocate", name), leaf_kind="allocation",
                    runtime_identity=lane + "_empty_explicit_device_v1"))
            result = runtime.empty(shape, dtype=dtype, device=target)
            if lifecycle is not None:
                owners.append(result)
                ordinal = int(result.device.index if lane == "cuda" else result.device.ordinal)
                lifecycle.adopt_returned_device_owner(result, actual_device_ordinal=ordinal,
                    role="slot_device_array", operation_id=_array_operation_id("slot_allocate", name))
            return result

    with lifecycle._observe_public_operation() if lifecycle is not None else nullcontext():
        device_scope = runtime.cuda.device(target) if lane == "cuda" else runtime.ScopedDevice(target)
        with device_scope:
            stream_scope = (runtime.cuda.stream(stream) if lane == "cuda" else runtime.ScopedStream(stream)) \
                if stream is not None else nullcontext()
            with stream_scope:
                active = True
                try:
                    yield empty
                finally:
                    active = False


def _upload_cuda_native_input(source, *, context):
    """Single allocation/upload implementation; the caller owns failure policy."""
    lifecycle, torch, target, stream = _consume_upload_context(context)
    import numpy as np

    prepared: dict[str, Any] = {}
    for manifest in source.manifest.arrays:
        value = source.arrays[manifest.name]
        host = np.array(value, copy=True, order="C")
        lifecycle.adopt_host_owner(
            host,
            role="cuda_h2d_host_staging",
            operation_id=_array_operation_id("cuda_stage", manifest.name),
        )
        host_tensor = torch.from_numpy(host)
        lifecycle.adopt_host_owner(
            host_tensor,
            role="cuda_h2d_torch_host_view",
            operation_id=_array_operation_id("cuda_host_view", manifest.name),
        )
        _enter_device_leaf(
            lifecycle,
            _descriptor(
                lifecycle,
                operation_id=_array_operation_id("cuda_allocate", manifest.name),
                leaf_kind="allocation",
                runtime_identity="torch_empty_explicit_device_v1",
            ),
        )
        destination = _torch_empty_exact(
            torch,
            shape=tuple(host.shape),
            dtype=host_tensor.dtype,
            target=target,
        )
        lifecycle.adopt_returned_device_owner(
            destination,
            actual_device_ordinal=(
                destination.device.index
                if destination.device.index is not None
                else lifecycle.target_device_ordinal
            ),
            role="cuda_a2_device_array",
            operation_id=_array_operation_id("cuda_allocate", manifest.name),
        )
        lifecycle.require_adopted(host, host_tensor, destination, stream)
        _enter_device_leaf(
            lifecycle,
            _descriptor(
                lifecycle,
                operation_id=_array_operation_id("cuda_upload", manifest.name),
                leaf_kind="host_to_device_copy",
                runtime_identity="torch_copy_explicit_stream_v1",
            ),
        )
        _torch_copy_exact(torch, destination, host_tensor, stream)
        prepared[manifest.name] = destination

    ready_owners = []
    lifecycle.adopt_host_owner(ready_owners, role="ready_event_custody", operation_id="ready_event_custody")
    _enter_device_leaf(
        lifecycle,
        _descriptor(
            lifecycle,
            operation_id="cuda_ready_event_create",
            leaf_kind="event_create",
            runtime_identity="torch_cuda_event_v1",
        ),
    )
    ready_event = _torch_event_exact(torch)
    ready_owners.append(ready_event)
    lifecycle.adopt_returned_device_owner(
        ready_event,
        actual_device_ordinal=_event_ordinal(ready_event, cuda=True, lifecycle=lifecycle, allow_unbound=True),
        role="cuda_a2_ready_event",
        operation_id="cuda_ready_event_create",
    )
    lifecycle.require_adopted(ready_event, stream)
    _enter_device_leaf(
        lifecycle,
        _descriptor(
            lifecycle,
            operation_id="cuda_ready_event_record",
            leaf_kind="event_record",
            runtime_identity="torch_event_record_indexed_stream_v1",
        ),
    )
    _torch_record_event_exact(ready_event, stream)
    _require_ready_event_target(ready_event, cuda=True, lifecycle=lifecycle)
    _enter_device_leaf(
        lifecycle,
        _descriptor(
            lifecycle,
            operation_id="cuda_ready_event_host_wait",
            leaf_kind="event_wait",
            runtime_identity="torch_event_synchronize_v1",
        ),
    )
    _torch_wait_event_host_exact(ready_event)
    result = CudaPreparedNativeInput(
        manifest=source.manifest,
        device_ordinal=lifecycle.target_device_ordinal,
        arrays=MappingProxyType(prepared),
        ready_event=ready_event,
        source=source,
    )
    lifecycle.adopt_returned_device_owner(
        result,
        actual_device_ordinal=lifecycle.target_device_ordinal,
        role="cuda_a2_prepared_input",
        operation_id="cuda_a2_result",
    )
    register_prepared_native_input(result, source)
    return result


def _upload_warp_native_input(source, *, context):
    """Single allocation/upload implementation; the caller owns failure policy."""
    lifecycle, wp, target, stream = _consume_upload_context(context)
    import numpy as np

    prepared: dict[str, Any] = {}
    with wp.ScopedDevice(target):
        for manifest in source.manifest.arrays:
            value = source.arrays[manifest.name]
            host = np.array(value, copy=True, order="C")
            lifecycle.adopt_host_owner(
                host,
                role="warp_h2d_host_staging",
                operation_id=_array_operation_id("warp_stage", manifest.name),
            )
            host_array = wp.array(
                host,
                dtype=warp_dtype_from_host_dtype(wp, value.dtype),
                device="cpu",
                copy=False,
            )
            lifecycle.adopt_host_owner(
                host_array,
                role="warp_h2d_cpu_array_view",
                operation_id=_array_operation_id("warp_host_view", manifest.name),
            )
            _enter_device_leaf(
                lifecycle,
                _descriptor(
                    lifecycle,
                    operation_id=_array_operation_id("warp_allocate", manifest.name),
                    leaf_kind="allocation",
                    runtime_identity="warp_empty_explicit_device_v1",
                ),
            )
            destination = _warp_empty_exact(
                wp,
                shape=tuple(host.shape),
                dtype=warp_dtype_from_host_dtype(wp, value.dtype),
                target=target,
            )
            lifecycle.adopt_returned_device_owner(
                destination,
                actual_device_ordinal=int(destination.device.ordinal),
                role="warp_a2_device_array",
                operation_id=_array_operation_id("warp_allocate", manifest.name),
            )
            lifecycle.require_adopted(host, host_array, destination, stream)
            _enter_device_leaf(
                lifecycle,
                _descriptor(
                    lifecycle,
                    operation_id=_array_operation_id("warp_upload", manifest.name),
                    leaf_kind="host_to_device_copy",
                    runtime_identity="warp_copy_explicit_stream_v1",
                ),
            )
            _warp_copy_exact(wp, destination, host_array, stream)
            prepared[manifest.name] = destination

        ready_owners = []
        lifecycle.adopt_host_owner(ready_owners, role="ready_event_custody", operation_id="ready_event_custody")
        _enter_device_leaf(
            lifecycle,
            _descriptor(
                lifecycle,
                operation_id="warp_ready_event_create",
                leaf_kind="event_create",
                runtime_identity="warp_event_explicit_device_v1",
            ),
        )
        ready_event = _warp_event_exact(wp, target=target)
        ready_owners.append(ready_event)
        lifecycle.adopt_returned_device_owner(
            ready_event,
            actual_device_ordinal=_event_ordinal(ready_event, cuda=False, lifecycle=lifecycle),
            role="warp_a2_ready_event",
            operation_id="warp_ready_event_create",
        )
        lifecycle.require_adopted(ready_event, stream)
        _enter_device_leaf(
            lifecycle,
            _descriptor(
                lifecycle,
                operation_id="warp_ready_event_record",
                leaf_kind="event_record",
                runtime_identity="warp_stream_record_event_v1",
            ),
        )
        _warp_record_event_exact(stream, ready_event)
        _require_ready_event_target(ready_event, cuda=False, lifecycle=lifecycle)
        _enter_device_leaf(
            lifecycle,
            _descriptor(
                lifecycle,
                operation_id="warp_ready_event_host_wait",
                leaf_kind="event_wait",
                runtime_identity="warp_synchronize_event_v1",
            ),
        )
        _warp_wait_event_host_exact(wp, ready_event)
    result = WarpPreparedNativeInput(
        manifest=source.manifest,
        device_ordinal=lifecycle.target_device_ordinal,
        arrays=MappingProxyType(prepared),
        ready_event=ready_event,
        source=source,
    )
    lifecycle.adopt_returned_device_owner(
        result,
        actual_device_ordinal=lifecycle.target_device_ordinal,
        role="warp_a2_prepared_input",
        operation_id="warp_a2_result",
    )
    register_prepared_native_input(result, source)
    return result


def prepare_cuda_native_input_for_closed_worker(
    source: object,
    *,
    lifecycle: NativeWorkerSetupLifecycle,
) -> object:
    """Prepare exact Torch-owned a2 allocations without allocation+copy calls."""

    if lifecycle.state != "NO_SETUP_GPU_WORK":
        raise RuntimeError("closed-worker CUDA a2 preparation must start a fresh setup epoch")
    if not lifecycle.guardian.contains_exact_owner(source):
        lifecycle.adopt_host_owner(
            source,
            role="canonical_native_input",
            operation_id="cuda_a2_source",
        )
    torch: Any | None = None
    target: Any | None = None
    try:
        _enter_device_leaf(
            lifecycle,
            _descriptor(
                lifecycle,
                operation_id="cuda_runtime_import",
                leaf_kind="runtime_import_or_init",
                runtime_identity="torch_cuda_indexed_v1",
            ),
        )
        import torch as torch_module

        torch = torch_module
        target = torch.device("cuda", lifecycle.target_device_ordinal)
        _enter_device_leaf(
            lifecycle,
            _descriptor(
                lifecycle,
                operation_id="cuda_setup_stream_create",
                leaf_kind="allocation",
                runtime_identity="torch_cuda_stream_explicit_device_v1",
            ),
        )
        stream = torch.cuda.Stream(device=target)
        lifecycle.adopt_returned_device_owner(
            stream,
            actual_device_ordinal=lifecycle.target_device_ordinal,
            role="cuda_setup_stream",
            operation_id="cuda_setup_stream_create",
        )

        context = _create_upload_context(lifecycle, torch, target, stream)
        return _upload_cuda_native_input(source, context=context)
    except BaseException as exc:
        if torch is None or target is None:
            lifecycle.mark_unknown_device_scope(
                source="cuda_runtime_import",
                reason="target_sync_unavailable_after_import_failure",
            )
            sync = None
        else:

            def sync() -> None:
                torch.cuda.synchronize(target)

        raise _abort(lifecycle, exception=exc, sync=sync) from exc


def prepare_warp_native_input_for_closed_worker(
    source: object,
    *,
    lifecycle: NativeWorkerSetupLifecycle,
) -> object:
    """Prepare exact Warp-owned a2 allocations with registered H2D operands."""

    if lifecycle.state != "NO_SETUP_GPU_WORK":
        raise RuntimeError("closed-worker Warp a2 preparation must start a fresh setup epoch")
    if not lifecycle.guardian.contains_exact_owner(source):
        lifecycle.adopt_host_owner(
            source,
            role="canonical_native_input",
            operation_id="warp_a2_source",
        )
    wp: Any | None = None
    target: Any | None = None
    try:
        _enter_device_leaf(
            lifecycle,
            _descriptor(
                lifecycle,
                operation_id="warp_runtime_import",
                leaf_kind="runtime_import_or_init",
                runtime_identity="warp_cuda_indexed_v1",
            ),
        )
        import warp as warp_module

        wp = warp_module
        target = wp.get_device(f"cuda:{lifecycle.target_device_ordinal}")
        if not target.is_cuda or int(target.ordinal) != lifecycle.target_device_ordinal:
            raise RuntimeError("Warp resolved a device different from the frozen target")
        lifecycle.adopt_returned_device_owner(
            target,
            actual_device_ordinal=int(target.ordinal),
            role="warp_target_device",
            operation_id="warp_runtime_import",
        )
        _enter_device_leaf(
            lifecycle,
            _descriptor(
                lifecycle,
                operation_id="warp_setup_stream_resolve",
                leaf_kind="runtime_import_or_init",
                runtime_identity="warp_get_stream_explicit_device_v1",
            ),
        )
        stream = wp.get_stream(target)
        lifecycle.adopt_returned_device_owner(
            stream,
            actual_device_ordinal=int(stream.device.ordinal),
            role="warp_setup_stream",
            operation_id="warp_setup_stream_resolve",
        )

        context = _create_upload_context(lifecycle, wp, target, stream)
        return _upload_warp_native_input(source, context=context)
    except BaseException as exc:
        if wp is None or target is None:
            lifecycle.mark_unknown_device_scope(
                source="warp_runtime_import",
                reason="target_sync_unavailable_after_import_failure",
            )
            sync = None
        else:

            def sync() -> None:
                _warp_synchronize_target_exact(wp, target)

        raise _abort(lifecycle, exception=exc, sync=sync) from exc


def seal_and_quiesce_cuda_setup(
    lifecycle: NativeWorkerSetupLifecycle,
    *,
    a2_readiness_sha256: str,
    prepared_graph_sha256: str,
) -> SetupQuiescenceEvidence:
    """Seal, then issue the final exact-target Torch setup barrier."""

    import torch

    target = torch.device("cuda", lifecycle.target_device_ordinal)
    _enter_device_leaf(
        lifecycle,
        _descriptor(
            lifecycle,
            operation_id="cuda_final_setup_barrier",
            leaf_kind="device_synchronize",
            runtime_identity="torch_synchronize_explicit_device_v1",
        ),
    )
    lifecycle.seal_guardian()
    try:
        torch.cuda.synchronize(target)
    except BaseException as exc:
        raise _abort(
            lifecycle,
            exception=exc,
            sync=lambda: torch.cuda.synchronize(target),
        ) from exc
    return lifecycle.commit_target_barrier(
        a2_readiness_sha256=a2_readiness_sha256,
        prepared_graph_sha256=prepared_graph_sha256,
    )


def seal_and_quiesce_warp_setup(
    lifecycle: NativeWorkerSetupLifecycle,
    *,
    a2_readiness_sha256: str,
    prepared_graph_sha256: str,
) -> SetupQuiescenceEvidence:
    """Seal, then issue the final exact-target Warp setup barrier."""

    import warp as wp

    target = wp.get_device(f"cuda:{lifecycle.target_device_ordinal}")
    _enter_device_leaf(
        lifecycle,
        _descriptor(
            lifecycle,
            operation_id="warp_final_setup_barrier",
            leaf_kind="device_synchronize",
            runtime_identity="warp_synchronize_explicit_device_v1",
        ),
    )
    lifecycle.seal_guardian()
    try:
        _warp_synchronize_target_exact(wp, target)
    except BaseException as exc:
        raise _abort(
            lifecycle,
            exception=exc,
            sync=lambda: _warp_synchronize_target_exact(wp, target),
        ) from exc
    return lifecycle.commit_target_barrier(
        a2_readiness_sha256=a2_readiness_sha256,
        prepared_graph_sha256=prepared_graph_sha256,
    )
