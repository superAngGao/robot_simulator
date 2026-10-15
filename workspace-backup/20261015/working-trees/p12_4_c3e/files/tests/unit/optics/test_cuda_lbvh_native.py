from __future__ import annotations

import inspect
from dataclasses import fields
from pathlib import Path

import pytest

from optics._cuda_lbvh_source import CPP_SOURCE, CUDA_SOURCE
from optics.cuda_lbvh import (
    CUDA_LBVH_ACCELERATION_ID,
    CUDA_LBVH_MAX_PRIMITIVES,
    CUDA_LBVH_OPERATION_ID,
    CUDA_LBVH_TOPOLOGY_LIFETIME,
    CUDA_LBVH_TOPOLOGY_ORIGIN_ID,
    CudaLbvhRebuildStage,
    CudaLbvhRebuildStageEvidence,
    _validate_primitive_count,
)


def test_cuda_lbvh_extension_uses_fixed_caller_owned_abi_without_allocation():
    assert "cuda_lbvh_workspace_requirements" in CPP_SOURCE
    assert "cuda_lbvh_rebuild_out" in CPP_SOURCE
    assert "cuda_lbvh_rebuild_out" in CUDA_SOURCE
    assert CPP_SOURCE.count("torch::Tensor") == 20
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
        assert forbidden not in CPP_SOURCE
        assert forbidden not in CUDA_SOURCE
    assert "c10::cuda::CUDAGuard" in CUDA_SOURCE
    assert "at::cuda::getCurrentCUDAStream(node_bounds_min.get_device())" in CUDA_SOURCE
    assert "cub::DeviceReduce::Min" in CUDA_SOURCE
    assert "cub::DeviceReduce::Max" in CUDA_SOURCE
    assert "cub::DeviceRadixSort::SortPairs" in CUDA_SOURCE
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK" in CUDA_SOURCE


def test_cuda_lbvh_timed_stage_only_invokes_prebound_entry():
    source = inspect.getsource(CudaLbvhRebuildStage.enqueue)
    assert "transaction.enqueue_stage(self)" in source
    assert "self._launch_bound_entry()" in source
    launch_source = inspect.getsource(CudaLbvhRebuildStage._launch_bound_entry)
    assert "self._extension_entry(*self._extension_args)" in launch_source
    for forbidden in (
        "torch.empty",
        "torch.zeros",
        "load_inline",
        "record_event",
        "synchronize",
        "to_torch",
        "from_torch",
        "contiguous",
        "workspace_requirements",
    ):
        assert forbidden not in source


def test_cuda_lbvh_evidence_freezes_acceleration_identity_without_c4_claims():
    assert CUDA_LBVH_ACCELERATION_ID == "cuda_morton_karras_lbvh_rebuild_v1"
    assert CUDA_LBVH_OPERATION_ID == "cuda_lbvh_scene_bounds_morton_sort_topology_bounds_out_v1"
    assert CUDA_LBVH_TOPOLOGY_ORIGIN_ID == "cuda_morton30_karras_source_index_tiebreak_v1"
    assert CUDA_LBVH_TOPOLOGY_LIFETIME == "transaction_output_arena_lease_v1"
    names = {field.name for field in fields(CudaLbvhRebuildStageEvidence)}
    assert {
        "workspace_requirements_sha256",
        "workspace_coverage_sha256",
        "input_storage_sha256",
        "output_storage_sha256",
        "dispatch_plan_sha256",
    }.issubset(names)
    assert {
        "slot_id",
        "generation",
        "lease",
        "ring",
        "ring_depth",
        "result_capable",
    }.isdisjoint(names)


def test_cuda_lbvh_example_is_a_thin_production_caller():
    root = Path(__file__).resolve().parents[3]
    source = (root / "examples" / "cuda_lbvh_extension_spike.py").read_text(encoding="utf-8")
    assert "from optics.cuda_lbvh import" in source
    assert "CudaLbvhRebuildStageFactory.prepare" in source
    assert "CUDA_SOURCE" not in source
    assert "CPP_SOURCE" not in source
    assert "load_inline" not in source
    assert "PYBIND11_MODULE" not in source


def test_cuda_lbvh_python_limit_matches_int32_flattened_node_boundary():
    assert CUDA_LBVH_MAX_PRIMITIVES == 357_913_941
    assert (CUDA_LBVH_MAX_PRIMITIVES * 2 - 1) * 3 <= (1 << 31) - 1
    assert ((CUDA_LBVH_MAX_PRIMITIVES + 1) * 2 - 1) * 3 > (1 << 31) - 1
    _validate_primitive_count(CUDA_LBVH_MAX_PRIMITIVES)
    with pytest.raises(ValueError, match="int32 flattened-node range"):
        _validate_primitive_count(CUDA_LBVH_MAX_PRIMITIVES + 1)
    assert "constexpr int64_t kMaxPrimitiveCount = 357913941;" in CUDA_SOURCE
    assert "primitive_count <= kMaxPrimitiveCount" in CUDA_SOURCE
    assert "primitive_count_64 <= kMaxPrimitiveCount" in CUDA_SOURCE


def test_cuda_lbvh_maximum_grid_ceil_division_cannot_overflow_int32():
    bound_value_count = (CUDA_LBVH_MAX_PRIMITIVES * 2 - 1) * 3
    block_count = bound_value_count // 256 + (bound_value_count % 256 != 0)
    assert bound_value_count == 2_147_483_643
    assert block_count == 8_388_608
    assert "ceil_div_positive_int(bound_value_count, kThreads)" in CUDA_SOURCE
    assert "ceil_div_positive_int(kMaxBoundValueCount, kThreads) == 8388608" in CUDA_SOURCE
    assert "+ kThreads - 1" not in CUDA_SOURCE
