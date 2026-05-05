# Q54 GPU Optical L5C.3 Shadow Direct Light Implementation Note

Date: 2026-05-05

## Scope

Extended `GpuDeviceBvhDirectLightOpticalExecutor` from no-shadow shading to CPU-parity direct lighting with shadow rays.

Implemented:

- `shadows` and `shadow_bias` options on `GpuDeviceBvhDirectLightOpticalExecutor`.
- GPU any-hit occlusion helper for shadow rays:
  - triangle BVH traversal with early exit;
  - analytical plane occlusion pass;
  - role-mask filtering consistent with first-hit traversal.
- Directional and point light shadow distance semantics:
  - directional: large finite max distance;
  - point: `distance_to_light - shadow_bias`.

The shade kernel now follows the CPU direct-light structure:

1. start from `ambient_rgb * albedo`;
2. for each light, compute Lambert term and attenuation;
3. if shadows are enabled, cast a biased shadow ray;
4. add light contribution only when unoccluded;
5. miss rays keep background RGB and zero intensity.

## Verification

Commands run:

```bash
python -m py_compile benchmarks/bench_optical_device_scene.py \
  optics/device_scene.py optics/warp_execution.py optics/__init__.py \
  tests/gpu/test_optical_gpu_runtime.py
conda run -n env_tilelang_20260119 python -m pytest tests/gpu/test_optical_gpu_runtime.py -q
conda run -n env_tilelang_20260119 python -m pytest tests/unit/optics -q
conda run -n env_tilelang_20260119 ruff check benchmarks/bench_optical_device_scene.py \
  optics/warp_execution.py tests/gpu/test_optical_gpu_runtime.py \
  optics/device_scene.py optics/__init__.py
```

Results:

- `tests/gpu/test_optical_gpu_runtime.py`: 20 passed
- `tests/unit/optics`: 57 passed
- `ruff check`: passed

## Benchmark Comparison And Decision

Compared three GPU paths with `--refit-bvh`:

```bash
conda run -n env_tilelang_20260119 python benchmarks/bench_optical_device_scene.py \
  --case robot_dense_single --repeat 5 --warmup 2 --refit-bvh
conda run -n env_tilelang_20260119 python benchmarks/bench_optical_device_scene.py \
  --case robot_dense_single --repeat 5 --warmup 2 --refit-bvh --direct-light
conda run -n env_tilelang_20260119 python benchmarks/bench_optical_device_scene.py \
  --case robot_dense_single --repeat 5 --warmup 2 --refit-bvh --direct-light --shadows
```

Observed p50 results:

- `robot_dense_single` first-hit: refit 0.295 ms, traversal 1.055 ms.
- `robot_dense_single` direct no-shadow: refit 0.290 ms, execute 1.327 ms.
- `robot_dense_single` direct shadow: refit 0.351 ms, execute 2.930 ms.

Also checked a heavier multi-robot scene:

```bash
conda run -n env_tilelang_20260119 python benchmarks/bench_optical_device_scene.py \
  --case robot_dense_pack --repeat 3 --warmup 1 --refit-bvh
conda run -n env_tilelang_20260119 python benchmarks/bench_optical_device_scene.py \
  --case robot_dense_pack --repeat 3 --warmup 1 --refit-bvh --direct-light
conda run -n env_tilelang_20260119 python benchmarks/bench_optical_device_scene.py \
  --case robot_dense_pack --repeat 3 --warmup 1 --refit-bvh --direct-light --shadows
```

Observed p50 results:

- `robot_dense_pack` first-hit: refit 0.400 ms, traversal 2.263 ms.
- `robot_dense_pack` direct no-shadow: refit 0.325 ms, execute 1.370 ms.
- `robot_dense_pack` direct shadow: refit 0.323 ms, execute 1.808 ms.

The pack numbers show some run-to-run noise, so the exact no-shadow vs first-hit ordering should not be overinterpreted. The important observation is that inline shadow any-hit does not explode into a different performance class on current robot scenes.

Decision for L5C.3:

- Keep inline shadow any-hit inside the direct-light shade kernel for now.
- Treat it as the canonical implementation until benchmarks show shadow execution exceeds first-hit by roughly 3x on target scenes, or until multiple lights/high-resolution renders make the shade kernel occupancy visibly poor.
- Do not introduce split shadow-ray buffers yet; that would add memory traffic, launch scheduling, and another result-lifetime surface before the current path has shown a real bottleneck.

## Notes

- Shadow traversal is implemented inside the shade kernel for simplicity and parity.
- This is not necessarily the final performance shape; if shadow cost dominates, split/batch shadow rays or fuse with a more specialized any-hit kernel.
- Shadow stack overflow is currently not reported separately from first-hit traversal.
