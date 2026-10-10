from __future__ import annotations

import inspect
from dataclasses import fields
from types import SimpleNamespace

import pytest

from optics.cuda_scene import (
    _CPP_SOURCE,
    _CUDA_SOURCE,
    CUDA_SCENE_UPDATE_OPERATION_ID,
    CudaSceneUpdateStage,
    CudaSceneUpdateStageEvidence,
    _triangle_abi_arguments,
    _validate_triangle_abi_binding,
)


def _triangle_stage():
    scene = SimpleNamespace(num_triangles=1, arrays={name: object() for name in (
        "geometry_vertices_local", "triangle_vertex_indices", "triangle_instance_index",
        "instance_body_index", "instance_X_body_geometry_R", "instance_X_body_geometry_r",
    )})
    transforms = SimpleNamespace(x_world_R=object(), x_world_r=object(), env_idx=0)
    outputs = SimpleNamespace(**{name: object() for name in (
        "triangle_v0_world", "triangle_e1_world", "triangle_e2_world",
        "triangle_normal_world", "triangle_aabb_min", "triangle_aabb_max",
    )})
    return SimpleNamespace(factory=SimpleNamespace(local_scene=scene), transforms=transforms,
                           outputs=outputs, _triangle_args=_triangle_abi_arguments(scene, transforms, outputs))


def test_triangle_stage_accepts_its_current_indexed_abi():
    _validate_triangle_abi_binding(_triangle_stage())


@pytest.mark.parametrize("argument", range(15))
def test_triangle_stage_rejects_replaced_tensor_or_environment(argument):
    stage = _triangle_stage()
    args = list(stage._triangle_args)
    args[argument] = 1 if argument == 8 else object()
    stage._triangle_args = tuple(args)
    with pytest.raises(ValueError, match="ABI binding changed"):
        _validate_triangle_abi_binding(stage)


@pytest.mark.parametrize("environment", [False, 0.0])
def test_triangle_stage_rejects_equal_noninteger_environment(environment):
    stage = _triangle_stage()
    args = list(stage._triangle_args)
    args[8] = environment
    stage._triangle_args = tuple(args)
    with pytest.raises(ValueError, match="ABI binding changed"):
        _validate_triangle_abi_binding(stage)


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
