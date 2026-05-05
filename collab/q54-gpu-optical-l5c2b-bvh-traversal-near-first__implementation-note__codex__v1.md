# Q54 GPU Optical L5C.2b BVH Near-First Traversal Implementation Note

Date: 2026-05-05

## Scope

Optimized the GPU BVH traversal kernel by ordering internal-node children by ray/AABB entry distance.

Implementation details:

- Added `_intersect_aabb_for_ray_with_interval`, which returns hit plus AABB entry/exit interval.
- BVH traversal stack now stores both `node_id` and `t_near`.
- Internal nodes push the farther child first and the nearer child second, so the LIFO stack visits the nearer child first.
- Popped nodes skip traversal if stored `t_near > best_t`, avoiding a repeated node AABB test.

## Benchmark Snapshot

Command:

```bash
conda run -n env_tilelang_20260119 python benchmarks/bench_optical_device_scene.py \
  --case robot_dense_single --repeat 5 --warmup 2 --refit-bvh
```

Observed result after this change:

- `robot_dense_single`: 161,280 tris x 65,536 rays
- GPU refit p50: 0.295 ms
- GPU traversal p50: 1.020 ms

For comparison, the previous refit traversal p50 was about 1.37 ms. A first near/far attempt that still retested node AABBs on pop regressed to about 1.93 ms, so the `t_near` stack is important.

## Verification

Commands run:

```bash
python -m py_compile optics/warp_execution.py
conda run -n env_tilelang_20260119 python -m pytest tests/gpu/test_optical_gpu_runtime.py -q
conda run -n env_tilelang_20260119 python -m pytest tests/unit/optics -q
conda run -n env_tilelang_20260119 ruff check optics/warp_execution.py tests/gpu/test_optical_gpu_runtime.py
```

Results:

- `tests/gpu/test_optical_gpu_runtime.py`: 16 passed
- `tests/unit/optics`: 57 passed
- `ruff check`: passed
