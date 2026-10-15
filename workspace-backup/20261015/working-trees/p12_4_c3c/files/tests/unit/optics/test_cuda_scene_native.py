from __future__ import annotations

import inspect
from dataclasses import fields

from optics.cuda_scene import (
    _CPP_SOURCE,
    _CUDA_SOURCE,
    CUDA_SCENE_UPDATE_OPERATION_ID,
    CudaSceneUpdateStage,
    CudaSceneUpdateStageEvidence,
)


def test_cuda_scene_extension_uses_fixed_out_style_abi_without_allocation():
    for entry in ("update_triangles_world_aabb_out", "update_planes_world_out"):
        assert entry in _CPP_SOURCE
        assert entry in _CUDA_SOURCE
    for forbidden in (
        "std::vector",
        "torch::empty",
        "torch::zeros",
        "torch::full",
        "TensorOptions",
        "cudaMalloc",
        "cudaFree",
        "cudaEvent",
        "cudaDeviceSynchronize",
        "cudaStreamSynchronize",
    ):
        assert forbidden not in _CPP_SOURCE
        assert forbidden not in _CUDA_SOURCE
    assert "c10::cuda::CUDAGuard" in _CUDA_SOURCE
    assert "at::cuda::getCurrentCUDAStream(triangle_v0_world.get_device())" in _CUDA_SOURCE
    assert "at::cuda::getCurrentCUDAStream(plane_normal_world.get_device())" in _CUDA_SOURCE
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK" in _CUDA_SOURCE


def test_cuda_scene_timed_stage_only_invokes_prebound_entries():
    source = inspect.getsource(CudaSceneUpdateStage.enqueue)
    assert "transaction.enqueue_stage(self)" in source
    assert "self._triangle_entry(*self._triangle_args)" in source
    assert "self._plane_entry(*self._plane_args)" in source
    for forbidden in (
        "torch.empty",
        "torch.zeros",
        "load_inline",
        "record_event",
        "synchronize",
        "to_torch",
        "from_torch",
        "contiguous",
    ):
        assert forbidden not in source


def test_cuda_scene_evidence_has_operation_but_no_c4_lifecycle_claim():
    assert CUDA_SCENE_UPDATE_OPERATION_ID == "torch_cuda_local_scene_transform_and_aabb_out_v1"
    names = {field.name for field in fields(CudaSceneUpdateStageEvidence)}
    assert "operation_id" in names
    assert "extension_source_sha256" in names
    assert {
        "slot_id",
        "generation",
        "lease",
        "ring",
        "ring_depth",
        "result_capable",
    }.isdisjoint(names)
