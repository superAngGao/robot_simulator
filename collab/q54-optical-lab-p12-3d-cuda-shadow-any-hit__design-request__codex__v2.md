# Q54 P12.3d Design Request V2: CUDA Shadow Any-Hit Functionality

Author: Codex
Status: ready for review
Related plan:
`collab/q54-optical-lab-p12-3-cuda-direct-light-backend__implementation-plan__codex__v1.md`
Supersedes:
`collab/q54-optical-lab-p12-3d-cuda-shadow-any-hit__design-request__codex__v1.md`

## Owner Summary

P12.3c implemented the CUDA no-shadow direct-light path. P12.3f-pre then proved
that `cuda_direct_light + cuda_lbvh + synthetic_body_triangle + shadows=False`
can run through the normal Lab smoke path.

The next functional slice is P12.3d: implement CUDA hard-shadow any-hit so the
same production executor can support `shadows=True` for host-ray rendering.

Benchmarking and optimization should not be mixed into this slice. First make
the CUDA shadow path correct and integrated; then run a separate benchmark
slice using the same production executor/workflow boundaries.

## Current State

Current executor shape:

```text
CudaDeviceBvhDirectLightOpticalExecutor.execute(...)
  -> validate output profile
  -> if shadows: NotImplementedError("pending P12.3d")
  -> CudaDeviceBvhOpticalExecutor first-hit
  -> shade_direct_light_no_shadow(...)
  -> OpticalComputeResult(location="device")
```

Already available:

- CUDA LBVH first-hit traversal;
- CUDA no-shadow shading;
- Torch CUDA tensor result channels;
- output profile filtering for `DIRECT_LIGHT_FULL`, `RGB_PREVIEW`,
  `RENDER_ONLY`;
- `shadow_stack_overflow_count` and `shadow_max_stack_depth` placeholder
  counters in the no-shadow path;
- Lab smoke for `shadows=False`.

Still missing:

- hard-shadow visibility tests;
- shadow traversal counters from real traversal;
- `shadows=True` parity tests;
- validator relaxation for Lab `shadows=True`.

## Scope

### Do

- Implement CUDA shadow any-hit for host-ray direct-light execution.
- Use the existing CUDA LBVH and device scene snapshot contracts.
- Preserve the current production executor boundary:

```text
first-hit geometry
  -> direct-light shading with optional shadow visibility
  -> OpticalComputeResult
```

- Support hard shadows for triangles.
- Apply `shadow_bias` when launching shadow rays.
- Preserve output profile behavior.
- Emit shadow diagnostics:
  - `shadow_stack_overflow_count`;
  - `shadow_max_stack_depth`.
- Add tiny occluder parity tests against CPU/Warp direct-light behavior.

### Do Not

- Do not add benchmark suites in this slice.
- Do not add benchmark-only execution paths.
- Do not add `cuda_shadow_*` render profile fields yet; timing field design
  belongs to P12.3d-benchmark.
- Do not fuse camera raygen, first-hit, and shading into a preview-only kernel.
- Do not implement CUDA camera raygen.
- Do not enable Go2 CUDA shadows.
- Do not enable `rgb8`, async delivery, public preset backend overrides, or
  matrix expansion.
- Do not claim final performance while device-wide sync remains.

## No Mixed Benchmark Path Rule

P12.3d must not introduce a separate "benchmark path" that calls private kernels
or bypasses normal result/channel/profile contracts.

Allowed later:

```text
executor-level benchmark
  = uses CudaDeviceBvhDirectLightOpticalExecutor / GpuDeviceBvhDirectLightOpticalExecutor
  = uses normal OpticalComputeResult channels
  = may use small synthetic scenes and ray counts

Lab-level benchmark
  = uses run_scenario(...) / OpticalLabRenderSession
  = reads normal frame_timing.csv and render profile fields
```

Not allowed:

```text
benchmark-only shadow_any_hit(...) caller
benchmark-only mixed CUDA/Warp shading path
temporary script that assembles kernels differently from the executor
```

Sub-phase timing can be emitted by the production executor through
`render_profile`, but benchmark collection belongs to a later
`P12.3d-benchmark` slice.

## Proposed Implementation

### 1. CUDA Extension Function

Add one shadow-capable shading function rather than a standalone public
benchmark function:

```cpp
std::vector<torch::Tensor> shade_direct_light_with_shadows(
    torch::Tensor hit_mask,
    torch::Tensor position_world,
    torch::Tensor normal_world,
    torch::Tensor material_index,
    torch::Tensor material_albedo_rgb,
    torch::Tensor light_kind,
    torch::Tensor light_position_or_direction_world,
    torch::Tensor light_intensity,
    torch::Tensor light_color_rgb,
    int64_t num_lights,
    ... scene triangle buffers ...,
    ... CUDA LBVH buffers ...,
    double shadow_bias,
    double ambient_r,
    double ambient_g,
    double ambient_b,
    double background_r,
    double background_g,
    double background_b);
```

The function returns the same high-level output shape as the no-shadow shader:

- `rgb`;
- `intensity`;
- `shadow_stack_overflow_count`;
- `shadow_max_stack_depth`.

The scene and BVH buffers passed to this function should reuse the existing
CUDA first-hit buffer layout conventions. P12.3d should not introduce a second
triangle/BVH layout just for shadows.

### 2. Kernel Shape

The first implementation should be a single shade kernel that performs any-hit
traversal per light:

```text
for each primary ray hit:
  start with ambient * albedo
  for each light:
    compute light direction and max shadow distance
    offset origin by normal * shadow_bias
    traverse CUDA LBVH
    if any blocker before light:
      skip diffuse contribution
    else:
      add Lambert contribution
```

This keeps output ownership simple and avoids introducing an intermediate
shadow mask channel before we know it is needed. A separate shadow-mask kernel
can be reconsidered only after P12.3d-benchmark shows a concrete need, or if a
future consumer such as debug visualization requires shadow masks as a stable
artifact.

### 3. Shadow Geometry Support

Minimum accepted P12.3d support:

- triangle mesh occluders;
- directional lights;
- point lights if the existing no-shadow shader path can keep them without
  extra complexity;
- multiple lights if the current loop remains simple.

Planes may remain non-shadow-casting in the first implementation only if that
is documented and tests do not depend on plane occlusion. The tiny parity scene
should use triangle occluders.

Implementation note: if planes are non-occluders in P12.3d, add a code comment
near the shadow traversal/shader path. The behavior should not live only in
collab documentation.

### 4. Python Executor Flow

Update `CudaDeviceBvhDirectLightOpticalExecutor.execute(...)`:

```text
geometry = first_hit.execute(...)
if self.shadows:
    return _shade_geometry_with_shadows(...)
return _shade_geometry_no_shadow(...)
```

The shadow path should use the same output profile filtering and resource
lifetime pattern as the no-shadow path.

### 5. Validator Policy

Keep Lab-level `shadows=True` rejected for `cuda_direct_light` until executor
parity is implemented and reviewed.

After parity tests pass, a follow-up in the same P12.3d implementation may
relax the pre-smoke validator for only:

```text
scene_preset="synthetic_body_triangle"
render_backend="cuda_direct_light"
accel_backend="cuda_lbvh"
shadows=True
video_raygen="host"
video_readback_delivery="sync"
readback_payload="rgb"
```

Do not relax Go2.

Review decision: executor parity and validator relaxation should land in the
same implementation commit once GPU parity passes, so executor, test, and Lab
availability stay aligned.

## Tests

### Unit

- `CudaDeviceBvhDirectLightOpticalExecutor(shadows=True)` no longer raises the
  P12.3d `NotImplementedError` once CUDA dependencies are present.
- CPU-only import safety remains unchanged.
- output profile filtering keeps required diagnostic counters for shadow path.

### GPU Parity

Add a tiny static occluder scene:

```text
visible receiver triangle
occluder triangle between receiver and light
one directional light
two or more host rays:
  - one ray hits a shadowed point
  - one ray hits an unshadowed point
```

Compare:

- `CpuDirectLightOpticalExecutor(shadows=True)`;
- `GpuDeviceBvhDirectLightOpticalExecutor(shadows=True)`;
- `CudaDeviceBvhDirectLightOpticalExecutor(shadows=True)`.

Use a non-zero `shadow_bias` consistent with the CPU/Warp direct-light
executors. Do not set bias to zero merely to simplify parity; incorrect bias
handling should be able to surface as an RGB mismatch.

Assertions:

- hit mask and range still match first-hit expectations;
- CUDA RGB/intensity match CPU/Warp within the existing direct-light tolerance;
- shadow counters exist;
- overflow counter is zero;
- max stack depth is non-negative;
- no-shadow CUDA result differs from shadow CUDA result on the occluded ray.

### Lab Smoke

After executor parity passes, extend the existing synthetic CUDA Lab smoke or
add a sibling test for `shadows=True`:

```text
frames=1
warmup_renders=1
scene_preset="synthetic_body_triangle"
render_backend="cuda_direct_light"
accel_backend="cuda_lbvh"
shadows=True
video_raygen="host"
video_readback_delivery="sync"
readback_payload="rgb"
```

Assertions should mirror P12.3f-pre:

- `scenario_config.json` records `shadows == true`;
- `frame_timing.csv` records `render_backend == "cuda_direct_light"`;
- output path comes from metadata, not hard-coded writer filenames.

## Validation Commands

Expected implementation validation:

```bash
PYTHONPATH=. ruff check optics/cuda_direct_light.py \
  tools/optical_pipeline_lab/runner.py \
  tests/gpu/test_optical_gpu_runtime.py \
  tests/unit/optics/test_device_optical.py \
  tests/unit/optics/test_optical_pipeline_lab.py

PYTHONPATH=. pytest -q tests/unit/optics/test_device_optical.py \
  -k "cuda_direct_light"

PYTHONPATH=. pytest -q tests/unit/optics/test_optical_pipeline_lab.py \
  -k "cuda_direct_light"

conda run -n env_tilelang_20260119 env PYTHONPATH=. pytest -q \
  tests/gpu/test_optical_gpu_runtime.py \
  -k "cuda_direct_light_shadow"
```

## Follow-Up Slices

### P12.3d-benchmark

After functional parity lands, add a benchmark design that:

- uses only production executor/workflow boundaries;
- compares CPU, Warp, CUDA shadow paths;
- records render profile sub-phases where already exposed by the executor;
- avoids private kernel callers and temporary scripts.

### P12.3d-opt

After benchmark results are reviewed, run an optimization slice targeted by the
measured bottlenecks.

### P12.3e

CUDA camera raygen remains separate.

### Full P12.3f

Go2 CUDA visual checks remain separate and should wait until feature coverage is
high enough that failures are diagnosable.

## Resolved Review Decisions

1. Use one shadow-capable shade kernel for P12.3d. Do not add an intermediate
   shadow-mask kernel yet.
2. Triangle-only shadow occlusion is acceptable for this slice. Plane
   non-occluder behavior must be documented in code if deferred.
3. Relax validator support for synthetic `cuda_direct_light + shadows=True` in
   the same commit as executor parity, after GPU parity passes.
4. Do not add `cuda_shadow_*` render profile fields in P12.3d. Add timing fields
   in P12.3d-benchmark when the measurement design is explicit.
5. Parity tests must use a non-zero `shadow_bias` matching CPU/Warp behavior.
