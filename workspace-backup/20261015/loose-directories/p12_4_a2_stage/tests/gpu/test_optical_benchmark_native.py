from __future__ import annotations

import numpy as np
import pytest

from benchmarks.optical.native import (
    CanonicalNativeInput,
    assert_independent_native_inputs,
    certify_native_readiness,
    prepare_cuda_native_input,
    prepare_warp_native_input,
    validate_prepared_native_input,
)

pytestmark = pytest.mark.gpu


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
