from __future__ import annotations

import numpy as np
import pytest

from benchmarks.optical.eligibility import FIXED_NATIVE_SCENE_INPUT
from benchmarks.optical.native import (
    CanonicalNativeInput,
    assert_independent_native_inputs,
    build_prepared_cuda_local_scene,
    build_prepared_warp_local_scene,
    certify_native_readiness,
    prepare_cuda_native_input,
    prepare_warp_native_input,
    validate_prepared_native_input,
)
from benchmarks.optical.workloads import (
    build_canonical_fixed_fixture,
    build_canonical_packed_scene,
)
from optical_worker.native_contract import (
    assert_cuda_prepared_authority,
    assert_warp_prepared_authority,
)

pytestmark = pytest.mark.gpu


def test_cuda_readiness_rejects_same_tensor_rebound_to_another_storage():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    source = _source()
    prepared = prepare_cuda_native_input(source, device="cuda:0")
    report = validate_prepared_native_input(source, prepared)
    readiness = certify_native_readiness(
        prepared, report, compile_complete=True, warmup_frames_completed=1,
    )
    assert_cuda_prepared_authority(prepared, readiness)

    tensor = prepared.arrays["triangle_v0"]
    old_pointer = tensor.data_ptr()
    replacement = torch.empty_like(tensor)
    tensor.set_(replacement)
    assert tensor.data_ptr() != old_pointer
    with pytest.raises(ValueError, match="physical allocation"):
        assert_cuda_prepared_authority(prepared, readiness)


def test_warp_readiness_rejects_same_array_pointer_drift():
    wp = pytest.importorskip("warp")
    if wp.get_cuda_device_count() == 0:
        pytest.skip("CUDA is required")
    source = _source()
    prepared = prepare_warp_native_input(source, device="cuda:0")
    report = validate_prepared_native_input(source, prepared)
    readiness = certify_native_readiness(
        prepared, report, compile_complete=True, warmup_frames_completed=1,
    )
    assert_warp_prepared_authority(prepared, readiness)

    array = prepared.arrays["triangle_v0"]
    replacement = wp.empty(array.shape, dtype=array.dtype, device=array.device)
    old_pointer = int(array.ptr)
    try:
        array.ptr = replacement.ptr
        assert int(array.ptr) != old_pointer
        with pytest.raises(ValueError, match="physical allocation"):
            assert_warp_prepared_authority(prepared, readiness)
    finally:
        array.ptr = old_pointer
    assert_warp_prepared_authority(prepared, readiness)


def test_warp_and_cuda_native_inputs_are_direct_independent_and_value_equal(monkeypatch):
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    if not torch.cuda.is_available() or wp.get_cuda_device_count() == 0:
        pytest.skip("CUDA is required")

    def _bridge_forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Warp/Torch bridge must not participate in native preparation")

    monkeypatch.setattr(wp, "to_torch", _bridge_forbidden)
    monkeypatch.setattr(wp, "from_torch", _bridge_forbidden)
    monkeypatch.setattr(torch.utils.dlpack, "from_dlpack", _bridge_forbidden)
    monkeypatch.setattr(torch.utils.dlpack, "to_dlpack", _bridge_forbidden)

    target_ordinal = 1 if torch.cuda.device_count() > 1 and wp.get_cuda_device_count() > 1 else 0
    ambient_ordinal = 0
    torch.cuda.set_device(ambient_ordinal)
    source = _source()

    warp_input = prepare_warp_native_input(source, device=f"cuda:{target_ordinal}")
    assert torch.cuda.current_device() == ambient_ordinal
    cuda_input = prepare_cuda_native_input(source, device=f"cuda:{target_ordinal}")
    assert torch.cuda.current_device() == ambient_ordinal

    warp_report = validate_prepared_native_input(source, warp_input)
    cuda_report = validate_prepared_native_input(source, cuda_input)
    assert warp_report.manifest_sha256 == cuda_report.manifest_sha256 == source.manifest.sha256
    assert warp_report.content_sha256 == cuda_report.content_sha256
    assert warp_report.topology_sha256 == cuda_report.topology_sha256
    assert warp_report.validated_arrays == cuda_report.validated_arrays
    assert_independent_native_inputs(warp_input, cuda_input)
    warp_ready = certify_native_readiness(
        warp_input,
        warp_report,
        compile_complete=True,
        warmup_frames_completed=10,
    )
    cuda_ready = certify_native_readiness(
        cuda_input,
        cuda_report,
        compile_complete=True,
        warmup_frames_completed=10,
    )
    assert warp_ready.backend == "warp"
    assert cuda_ready.backend == "cuda"
    assert warp_ready.content_sha256 == cuda_ready.content_sha256

    for value in warp_input.arrays.values():
        assert value.device.is_cuda
        assert int(value.device.ordinal) == target_ordinal
    for value in cuda_input.arrays.values():
        assert value.is_cuda
        assert value.device.index == target_ordinal
        assert value.is_contiguous()


def test_native_preparation_rejects_cpu_targets():
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    del torch, wp
    source = _source()

    with pytest.raises(ValueError, match="CUDA device"):
        prepare_warp_native_input(source, device="cpu")
    with pytest.raises(ValueError, match="CUDA device"):
        prepare_cuda_native_input(source, device="cpu")


def test_result_capable_scene_wrappers_reuse_exact_independent_native_allocations(monkeypatch):
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    if not torch.cuda.is_available() or wp.get_cuda_device_count() == 0:
        pytest.skip("CUDA is required")

    def _bridge_forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("scene wrapper must not create a Warp/Torch bridge")

    monkeypatch.setattr(wp, "to_torch", _bridge_forbidden)
    monkeypatch.setattr(wp, "from_torch", _bridge_forbidden)
    monkeypatch.setattr(torch.utils.dlpack, "from_dlpack", _bridge_forbidden)
    monkeypatch.setattr(torch.utils.dlpack, "to_dlpack", _bridge_forbidden)

    target = 1 if torch.cuda.device_count() > 1 and wp.get_cuda_device_count() > 1 else 0
    ambient = 0
    torch.cuda.set_device(ambient)
    source = _result_capable_source()
    warp_input = prepare_warp_native_input(source, device=f"cuda:{target}")
    cuda_input = prepare_cuda_native_input(source, device=f"cuda:{target}")
    assert_independent_native_inputs(warp_input, cuda_input)

    warp_scene = build_prepared_warp_local_scene(warp_input)
    cuda_scene = build_prepared_cuda_local_scene(cuda_input)
    assert torch.cuda.current_device() == ambient
    assert warp_scene.scene_contract_sha256 == source.manifest.scene_contract_sha256
    assert cuda_scene.scene_contract_sha256 == source.manifest.scene_contract_sha256
    for name in source.manifest.scene_array_names:
        assert getattr(warp_scene, name) is warp_input.arrays[name]
        assert cuda_scene.arrays[name] is cuda_input.arrays[name]


def _source() -> CanonicalNativeInput:
    return CanonicalNativeInput.from_arrays(
        case_id="asymmetric_synthetic_7x3_v1",
        materializer_id="gpu_asymmetric_native_v1",
        arrays={
            "bvh_child_l": np.array([1, -1, -1], dtype=np.int32),
            "bvh_lower": np.array(
                [[-2.0, -1.0, 1.5], [-2.0, -1.0, 1.5], [0.25, -1.0, 2.5]],
                dtype=np.float32,
            ),
            "triangle_v0": np.array(
                [[-1.75, -0.5, 2.0], [0.25, -1.0, 3.0]],
                dtype=np.float32,
            ),
            "visibility": np.array([True, False], dtype=np.bool_),
        },
        array_semantics={
            "bvh_child_l": "fixed BVH left-child topology",
            "bvh_lower": "fixed BVH lower bounds",
            "triangle_v0": "world-space triangle vertex zero",
            "visibility": "canonical per-primitive visibility mask",
        },
        topology_arrays=("bvh_child_l", "bvh_lower"),
        semantics={
            "coordinate_system": "world",
            "primitive_order": "canonical_source_order",
            "flat_pixel_index": "py * width + px",
        },
    )


def _result_capable_source() -> CanonicalNativeInput:
    packed = build_canonical_packed_scene("asymmetric_synthetic_7x3_v1")
    fixture = build_canonical_fixed_fixture(packed)
    return CanonicalNativeInput.from_scene_contract(
        packed,
        input_contract=FIXED_NATIVE_SCENE_INPUT,
        fixed_fixture=fixture,
    )
