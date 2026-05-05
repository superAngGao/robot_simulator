# Q54 GPU Optical L5C Progress And Next Plan Review Request

Date: 2026-05-05

## Owner Summary

L5C now has a device-resident optical scene path, derived triangle layout,
CPU-built/GPU-traversed BVH, GPU BVH refit, near-first BVH traversal, and a
GPU BVH direct-light executor with optional inline shadow any-hit. The next
decision is how to harden this direct-light path without prematurely turning it
into a path tracer or adding split shadow kernels before benchmark data justifies
the extra pipeline surface.

## Current Implemented State

L5C.1:

- `DeviceOpticalSceneCache` / `DeviceOpticalSceneSnapshot` own device-resident
  primitive buffers and respect Q52 resource lifetime constraints.
- Role masks are int64 with a 63-role limit.
- Triangle layout is now derived SoA:
  - `triangle_v0_world`
  - `triangle_e1_world`
  - `triangle_e2_world`
  - `triangle_normal_world`
- The old `triangles_world[N, 9]` buffer was removed after parity stabilized.
- Per-triangle AABB traversal remains a benchmarkable variant, not the main path.

L5C.2:

- `DeviceOpticalBvh` provides flat SoA BVH buffers for GPU traversal.
- L5C.2a landed a CPU-built/GPU-traversed BVH bridge.
- L5C.2b added GPU level-by-level BVH refit for fixed topology.
- Traversal now visits children near-first and stores `(node_id, t_near)` on the
  fixed stack to avoid repeated AABB tests at pop time.
- Stack size is currently `_MAX_BVH_STACK = 32`, with overflow count and observed
  max stack depth returned as diagnostic channels.
- Planes stay outside the BVH and are handled as an analytical side pass merged
  through the same `_is_better_hit` source-order tie-break semantics.

L5C.3:

- Device scene material/light buffers were added:
  - `material_albedo_rgb`
  - `triangle_material_index`
  - `plane_material_index`
  - `light_kind`
  - `light_position_or_direction_world`
  - `light_intensity`
  - `light_color_rgb`
- `GpuDeviceBvhOpticalExecutor` now emits `material_index` for hit rays.
- `GpuDeviceBvhDirectLightOpticalExecutor` emits `rgb` and `intensity`.
- GPU shading matches the CPU direct-light executor for:
  - ambient term;
  - directional lights;
  - point-light inverse-square attenuation;
  - background color for miss rays;
  - optional shadows.
- Shadow rays are currently implemented inline in the shade kernel as any-hit
  traversal over the triangle BVH plus analytical plane occlusion.

## Verification

Latest local verification:

```bash
python -m py_compile benchmarks/bench_optical_device_scene.py \
  optics/device_scene.py optics/warp_execution.py optics/__init__.py \
  tests/gpu/test_optical_gpu_runtime.py

conda run -n env_tilelang_20260119 python -m pytest \
  tests/gpu/test_optical_gpu_runtime.py -q

conda run -n env_tilelang_20260119 python -m pytest tests/unit/optics -q

conda run -n env_tilelang_20260119 ruff check \
  benchmarks/bench_optical_device_scene.py optics/warp_execution.py \
  tests/gpu/test_optical_gpu_runtime.py optics/device_scene.py optics/__init__.py
```

Results:

- GPU optical runtime tests: 20 passed.
- Unit optics tests: 57 passed.
- Ruff: passed.
- `git diff --check`: passed.

## Benchmark Snapshot

Benchmark CLI now supports:

- `--use-bvh`
- `--refit-bvh`
- `--direct-light`
- `--shadows`

Recent p50 measurements with `--refit-bvh`:

```text
robot_dense_single
  first-hit BVH:       update 0.200 ms, refit 0.295 ms, execute 1.055 ms
  direct no-shadow:    update 0.204 ms, refit 0.290 ms, execute 1.327 ms
  direct with-shadow:  update 0.219 ms, refit 0.351 ms, execute 2.930 ms

robot_dense_pack
  first-hit BVH:       update 0.270 ms, refit 0.400 ms, execute 2.263 ms
  direct no-shadow:    update 0.210 ms, refit 0.325 ms, execute 1.370 ms
  direct with-shadow:  update 0.214 ms, refit 0.323 ms, execute 1.808 ms
```

The `robot_dense_pack` measurements still show process-to-process noise, so the
exact no-shadow vs first-hit ordering should not be overinterpreted. The useful
signal is that inline shadow any-hit has not exploded into a different
performance class on the current robot-like scenes.

Current decision:

- Keep inline shadow any-hit inside the direct-light shade kernel for L5C.3.
- Do not introduce split shadow-ray buffers/kernels yet.
- Revisit split shadow kernels if target scenes show shadow execution above
  roughly `3x` primary first-hit, or if multi-light/high-resolution renders make
  the shade kernel occupancy clearly poor.

## Near-Term Plan

L5C.3a hardening:

- Add explicit shadow traversal diagnostics:
  - shadow stack overflow count;
  - optional max shadow stack depth;
  - per-light or aggregate shadow-ray count when useful.
- Add focused parity tests for:
  - plane occluders;
  - triangle occluders;
  - equal-distance source-order tie-break with plane plus triangle;
  - multi-light additive shading;
  - zero-light and disabled-light scenes.
- Keep shadow bias small and explicit; avoid hiding self-shadow artifacts by
  increasing the default unless tests demonstrate the need.

L5C.3b GPU preview/example:

- Add a GPU direct-light preview path for the Menagerie Go2 scene.
- Save README-ready images only after the GPU path is used, rather than spending
  more time pushing the CPU reference renderer to higher resolution.
- Keep the current CPU preview images as temporary visual checkpoints.

L5C.3c benchmark cleanup:

- Run first-hit/no-shadow/shadow comparisons inside one benchmark process where
  practical, to reduce cross-process noise.
- Report component timings separately:
  - update;
  - BVH build or refit;
  - first-hit traversal;
  - shade/shadow;
  - staging/end-to-end when requested.
- Keep benchmark scenes robot-like:
  - single Menagerie Go2;
  - multi-Go2 pack;
  - close ego camera;
  - dense repeated robot mesh;
  - later, one license-clear large static mesh.

## Deferred Path-Tracing / Stray-Light Plan

We should not mix path tracing into the current L5C.3 direct-light work. The
current path is deterministic first-hit plus direct illumination. That is the
right surface for depth, segmentation, basic RGB, direct shadows, and benchmark
alignment.

When stray light / indirect transport becomes important, plan it as a separate
executor family rather than mutating the direct-light executor:

1. Define the optical scope:
   - diffuse interreflection;
   - specular/glossy reflection;
   - refraction;
   - sensor glare or lens artifacts;
   - volumetric scattering, if any.
2. Extend payloads deliberately:
   - material BSDF parameters beyond albedo;
   - emissive surfaces or area lights;
   - RNG state;
   - sample accumulation buffers;
   - per-pixel sample count and variance diagnostics.
3. Start with a small CPU reference or GPU single-bounce prototype that shares
   the BVH query helpers but has separate result channels and tests.
4. Move to a GPU Monte Carlo path tracer only after direct-light BVH traversal,
   shadow any-hit, and high-resolution camera examples are stable.
5. Compare against external renderers only on aligned transport modes; do not
   compare a multi-bounce path-traced frame against the current first-hit or
   direct-light timing.

This keeps direct-light as the reliable robotics/sensor workhorse while leaving
a clean path to physically richer rendering later.

## Open Questions For Claude Review

1. Is keeping inline shadow any-hit as the L5C.3 canonical path the right
   decision given the current benchmark signal?
2. Should shadow stack overflow diagnostics be required before high-resolution
   GPU examples, or can that land immediately after the preview path?
3. Is the proposed separation between deterministic direct-light and future
   path-tracing executors clean enough for long-term maintenance?
4. Are there additional CPU-vs-GPU parity tests that should block the next
   commit after L5C.3?
5. Should direct-light support also be added to the non-BVH
   `GpuDeviceSceneOpticalExecutor`, or should GPU RGB require BVH from now on?
