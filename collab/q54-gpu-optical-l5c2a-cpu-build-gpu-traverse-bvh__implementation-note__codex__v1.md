# Q54 GPU Optical L5C.2a CPU-Build/GPU-Traverse BVH Implementation Note

Date: 2026-05-05

## Scope

Implemented the first L5C.2 correctness bridge:

- CPU median-split BVH build from per-frame triangle AABBs.
- GPU ray-major BVH traversal with one thread per ray and fixed stack size 32.
- Plane primitives remain outside the BVH and run through the analytical pass before mesh traversal.
- Plane/triangle and triangle/triangle tie-break semantics continue to use `_is_better_hit`.
- Primitive metadata is now uploaded with the device scene so later BLAS/TLAS work does not have to reinterpret global triangle ids.

## Files

- `optics/device_scene.py`
  - Added triangle primitive metadata buffers:
    - `triangle_primitive_global_id`
    - `triangle_primitive_index_within_instance`
    - `triangle_geometry_index`
    - `triangle_geometry_primitive_index`
- `optics/device_bvh.py`
  - Added `DeviceOpticalBvh`, `DeviceBvhBuildStats`, and `build_device_bvh_from_snapshot`.
  - Builder records `num_nodes`, `num_leaves`, `max_depth`, `leaf_size`, and `sah_quality_cost`.
  - Nodes are uploaded in BFS order and include `node_depth` plus `level_ranges` for the later GPU refit path.
- `optics/warp_execution.py`
  - Added `GpuDeviceBvhOpticalExecutor`.
  - Added `_device_scene_bvh_first_hit_kernel`.
  - Added status channels:
    - `bvh_stack_overflow_count`
    - `bvh_max_stack_depth`
- `benchmarks/bench_optical_device_scene.py`
  - Added `--use-bvh`.
  - Split benchmark columns into `update_ms`, `cpu_build_ms`, `gpu_traverse_ms`, and `stage_ms`.
- `tests/gpu/test_optical_gpu_runtime.py`
  - Added BVH parity coverage for world-static triangle meshes.
  - Added plane/triangle equal-distance tie-break coverage.

## Current Semantics

The BVH contains triangle primitives only. Planes intentionally remain outside the tree:

1. Run plane analytical pass.
2. Traverse triangle BVH.
3. Merge candidates with `_is_better_hit`.

This preserves the existing source-order behavior, including equal-distance plane/triangle cases.

## Known Limits

- CPU build is intentionally median split, not SAH.
- The stack overflow path reports through a status channel; it does not yet raise synchronously from `execute`.
- Traversal does not yet order near/far children by entry distance.
- GPU refit/LBVH/OptiX are not implemented in this step.

## Verification

Commands run:

```bash
python -m py_compile optics/device_scene.py optics/device_bvh.py optics/warp_execution.py tests/gpu/test_optical_gpu_runtime.py
conda run -n env_tilelang_20260119 python -m pytest tests/gpu/test_optical_gpu_runtime.py -q
conda run -n env_tilelang_20260119 python -m pytest tests/unit/optics -q
conda run -n env_tilelang_20260119 python benchmarks/bench_optical_device_scene.py --case smoke --repeat 1 --warmup 0 --use-bvh
```

Results:

- `tests/gpu/test_optical_gpu_runtime.py`: 15 passed
- `tests/unit/optics`: 57 passed
- BVH smoke benchmark completed and reported split update/build/traverse metrics.
