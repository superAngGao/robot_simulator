# Q54 GPU Optical L5C.2b GPU BVH Refit Implementation Note

Date: 2026-05-05

## Scope

Implemented the first GPU BVH refit path:

- BVH topology is still built once by the existing CPU median-split builder.
- Per-frame node bounds are updated on GPU from `triangle_aabb_min/max`.
- Refit runs as:
  1. leaf bounds kernel over all nodes, skipping internal nodes;
  2. internal bounds kernels from deepest level back to root using stored `level_ranges`.
- `GpuDeviceBvhOpticalExecutor` now waits on `DeviceOpticalBvh.ready_event`.
- Benchmark harness supports `--refit-bvh` and reports `gpu_refit_ms` separately.

## Benchmark Snapshot

Commands:

```bash
conda run -n env_tilelang_20260119 python benchmarks/bench_optical_device_scene.py \
  --case robot_proxy_pose --repeat 3 --warmup 1 --refit-bvh
conda run -n env_tilelang_20260119 python benchmarks/bench_optical_device_scene.py \
  --case robot_dense_single --repeat 3 --warmup 1 --refit-bvh
```

Observed p50 results:

- `robot_proxy_pose`: update 0.185 ms, GPU refit 0.179 ms, traversal 0.610 ms.
- `robot_dense_single`: update 0.197 ms, GPU refit 0.295 ms, traversal 1.368 ms.

Before refit, `robot_dense_single` CPU BVH rebuild p50 was about 4464 ms. The refit path removes that per-frame rebuild bottleneck for fixed-topology scenes.

## Verification

Commands run:

```bash
python -m py_compile optics/device_bvh.py optics/warp_execution.py optics/__init__.py \
  benchmarks/bench_optical_device_scene.py tests/gpu/test_optical_gpu_runtime.py
conda run -n env_tilelang_20260119 python -m pytest tests/gpu/test_optical_gpu_runtime.py -q
conda run -n env_tilelang_20260119 python -m pytest tests/unit/optics -q
```

Results:

- `tests/gpu/test_optical_gpu_runtime.py`: 16 passed
- `tests/unit/optics`: 57 passed

## Remaining Work

- Refit currently uses one kernel launch per BVH level; dense robot depth is 16, so this is acceptable but not final.
- Traversal does not yet order children by near hit distance.
- This is refit for fixed primitive topology; topology rebuild/LBVH is still needed for topology-changing scenes.
