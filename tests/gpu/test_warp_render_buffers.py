"""Real allocations only; no recorded render bundle is executed by these tests."""

from __future__ import annotations

import gc
import weakref
from dataclasses import FrozenInstanceError, replace

import pytest

from optics.warp_native_render import WarpDirectLightBuffers, warp_direct_light_buffer_layout

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def wp():
    runtime = pytest.importorskip("warp")
    runtime.init()
    if runtime.get_cuda_device_count() == 0:
        pytest.skip("CUDA is required")
    return runtime


def _kwargs(wp, n=5, diagnostics=False, device=None):
    if device is None:
        device = wp.get_cuda_device(0)
    arrays = {
        name: wp.empty(shape, dtype=getattr(wp, dtype), device=device)
        for name, dtype, shape in warp_direct_light_buffer_layout(n, traversal_diagnostics=diagnostics)
    }
    from optics.output_contract import SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES

    overlay = (
        tuple(arrays.pop(name) for name in SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES) if diagnostics else ()
    )
    return dict(num_rays=n, traversal_diagnostics=diagnostics, traversal_counters=overlay, **arrays)


@pytest.mark.parametrize("n", [0, 1, 5])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_real_caller_owned_buffers_validate_without_runtime_work(wp, monkeypatch, n, diagnostics):
    values = _kwargs(wp, n, diagnostics)

    def forbidden(*args, **kwargs):
        raise AssertionError("buffer validation must not allocate, transfer, initialize, or submit")

    with monkeypatch.context() as patch:
        for name in ("init", "get_device", "empty", "zeros", "copy", "launch", "synchronize_device"):
            patch.setattr(wp, name, forbidden)
        buffers = WarpDirectLightBuffers(**values)
        buffers.validate_bound()
    assert len(buffers.arrays) == (23 if diagnostics else 14)
    assert buffers.origins is values["origins"]


@pytest.mark.parametrize("name", ["origins", "range_m", "material_index", "bvh_max_stack_depth", "rgb"])
def test_wrong_shape_or_dtype_rejected(wp, name):
    values = _kwargs(wp)
    original = values[name]
    wrong_dtype = wp.int32 if original.dtype is wp.float32 else wp.float32
    values[name] = wp.empty(original.shape, dtype=wrong_dtype, device=original.device)
    with pytest.raises(ValueError, match="dtype/shape"):
        WarpDirectLightBuffers(**values)
    values[name] = wp.empty((original.size + 1,), dtype=original.dtype, device=original.device)
    with pytest.raises(ValueError, match="dtype/shape"):
        WarpDirectLightBuffers(**values)


def test_no_diagnostic_alias_dummy_or_mutable_container(wp):
    values = _kwargs(wp, diagnostics=True)
    overlay = values["traversal_counters"]
    for wrong in (overlay[:8], (*overlay, overlay[0]), list(overlay), (overlay[0],) * 9):
        with pytest.raises(ValueError):
            WarpDirectLightBuffers(**{**values, "traversal_counters": wrong})
    buffers = WarpDirectLightBuffers(**values)
    with pytest.raises(ValueError):
        replace(buffers, traversal_diagnostics=False)


def test_output_overlap_and_input_overlap_rejected_but_disjoint_views_allowed(wp):
    values = _kwargs(wp)
    owner = wp.empty((30,), dtype=wp.float32, device=values["origins"].device)
    values["origins"] = owner[:15].reshape((5, 3))
    values["directions"] = owner[15:].reshape((5, 3))
    buffers = WarpDirectLightBuffers(**values)
    buffers.validate_bound()
    with pytest.raises(ValueError, match="overlap"):
        replace(buffers, directions=owner[12:27].reshape((5, 3)))
    with pytest.raises(ValueError, match="overlap"):
        buffers.validate_bound(read_only_inputs=(("scene", owner[14:16]),))
    with pytest.raises(ValueError, match="independent"):
        replace(buffers, directions=buffers.origins)
    reference = weakref.ref(owner)
    del values, owner
    gc.collect()
    assert reference() is not None
    buffers.validate_bound()


def test_raw_nonowning_output_is_not_a_storage_owner(wp):
    values = _kwargs(wp)
    original = values["origins"]
    raw = wp.array(ptr=original.ptr, shape=original.shape, dtype=original.dtype, device=original.device)
    assert raw.deleter is None
    values["origins"] = raw
    with pytest.raises(ValueError, match="owning root"):
        WarpDirectLightBuffers(**values)


def test_seal_baseline_cannot_be_reopened_and_output_order_drift_is_detected(wp):
    buffers = WarpDirectLightBuffers(**_kwargs(wp))
    for name, value in (("origins", buffers.directions), ("_signature", ()), ("_snapshots", ())):
        with pytest.raises(FrozenInstanceError):
            setattr(buffers, name, value)
        with pytest.raises(FrozenInstanceError):
            delattr(buffers, name)
    with pytest.raises(RuntimeError, match="cannot be rebuilt"):
        buffers.__post_init__()
    origins = buffers.origins
    object.__setattr__(buffers, "origins", buffers.directions)
    object.__setattr__(buffers, "directions", origins)
    with pytest.raises(ValueError, match="binding changed"):
        buffers.validate_bound()


@pytest.mark.parametrize("attribute", ["ptr", "capacity", "shape", "strides", "size", "_ref"])
def test_same_object_physical_storage_drift_rejected(wp, attribute):
    buffers = WarpDirectLightBuffers(**_kwargs(wp))
    array = buffers.origins
    old = getattr(array, attribute)
    replacement = {
        "ptr": buffers.directions.ptr,
        "capacity": array.capacity + 4,
        "shape": (1, 15),
        "strides": (4, 20),
        "size": array.size + 1,
        "_ref": array,
    }[attribute]
    try:
        setattr(array, attribute, replacement)
        with pytest.raises(ValueError):
            buffers.validate_bound()
        with pytest.raises(RuntimeError, match="cannot be rebuilt"):
            buffers.__post_init__()
    finally:
        setattr(array, attribute, old)
    buffers.validate_bound()


def test_root_owner_pointer_drift_is_detected(wp):
    values = _kwargs(wp)
    root = wp.empty((10, 3), dtype=wp.float32, device=values["origins"].device)
    values["origins"] = root[:5]
    buffers = WarpDirectLightBuffers(**values)
    pointer = root.ptr
    try:
        root.ptr += 4
        with pytest.raises(ValueError):
            buffers.validate_bound()
    finally:
        root.ptr = pointer
    buffers.validate_bound()


def test_cpu_wrong_device_and_noncontiguous_storage_rejected(wp):
    values = _kwargs(wp)
    with pytest.raises(ValueError, match="CUDA device"):
        WarpDirectLightBuffers(**{**values, "origins": wp.empty((5, 3), device="cpu")})
    storage = wp.empty((5, 6), dtype=wp.float32, device=values["origins"].device)
    with pytest.raises(ValueError, match="contiguous"):
        WarpDirectLightBuffers(**{**values, "origins": storage[:, :3]})
    if wp.get_cuda_device_count() < 2:
        pytest.skip("separate real second-device coverage requires two GPUs")
    other = wp.empty((5, 3), dtype=wp.float32, device=wp.get_cuda_device(1))
    with pytest.raises(ValueError, match="same indexed CUDA device"):
        WarpDirectLightBuffers(**{**values, "directions": other})
