# Q54 P12.3 Implementation Plan: CUDA Direct-Light Backend

## Context

P12.1 approved the direct-light backend family:

```text
cpu_direct_light          CPU reference / correctness path
warp_bvh_direct_light     existing Warp GPU path
cuda_direct_light         CUDA direct-light performance path
```

P12.2 implemented `cpu_direct_light` as a narrow static, host-raygen, sync
reference backend. P12.3 is the main performance-oriented step: implement a
real `cuda_direct_light` backend using CUDA LBVH plus hand-written CUDA render
kernels.

This plan is intentionally implementation-driven. It does not extract a backend
registry yet; P12.5 owns that after CPU/Warp/CUDA direct-light paths all exist.

## Goals

1. Add a real `cuda_direct_light` render backend.
2. Use hand-written CUDA kernels, not Warp render kernels.
3. Reuse the existing `DeviceOpticalSceneSnapshot` and `DeviceOpticalBvh`
   buffer contracts.
4. Preserve the direct-light output/profile/readback contract:
   `DIRECT_LIGHT_FULL`, `RGB_PREVIEW`, and `RENDER_ONLY`.
5. Support Optical Lab static video product workflow enough to produce
   `scenario_config.json`, `frame_timing.csv`, and RGB artifacts through the
   P10/P11 product path.
6. Establish a clean dispatch point for later benchmark/matrix work.

## Non-Goals

- No path tracing, accumulation buffers, denoising, OptiX, or multisample
  semantics.
- No `cuda_fused_rgb` preview-only shortcut.
- No public `run_optical_lab_preset(..., render_backend=...)` override yet.
- No backend registry extraction yet.
- No dynamic/physics published-frame support in the first accepted CUDA backend.
- No final performance claims until stream/event synchronization avoids
  per-frame global CUDA synchronization.

## User-Facing Shape

P12.3 should keep backend selection at the config/workflow level:

```python
from dataclasses import replace
from pathlib import Path

from tools.optical_pipeline_lab import AccelBackend, ArtifactOutput, RenderBackend
from tools.optical_pipeline_lab.presets import get_preset

config = replace(
    get_preset("go2_video_ordered_static"),
    accel_backend=AccelBackend.CUDA_LBVH,
    render_backend=RenderBackend.CUDA_DIRECT_LIGHT,
)

output = ArtifactOutput(
    root=Path("runs/p12/cuda_direct"),
    frames=120,
    video_raygen="gpu",
)
```

P12.3 does not need to expose:

```python
run_optical_lab_preset(..., render_backend="cuda_direct_light")
```

That belongs to P12.6 after backend behavior and benchmark reporting are stable.

## Internal Assembly Interface

P12.2 exposed a gap: reviewed preset runtime creation is easy, but explicit
config runtime creation still requires manual assembly in tests. P12.3 should
consider adding an internal helper:

```python
def create_static_asset_lab_runtime_for_config(
    config: OpticalLabScenarioConfig,
    *,
    output: ArtifactOutput,
    runtime_kwargs: Mapping[str, object] | None = None,
) -> StaticAssetLabRuntime:
    ...
```

Then the existing preset helper can delegate:

```python
def create_static_asset_lab_runtime(preset: str, *, output: ArtifactOutput, ...):
    config = get_preset(preset)
    ...
    return create_static_asset_lab_runtime_for_config(config, output=output, ...)
```

This keeps public P11/P12 APIs stable while making explicit backend configs
first-class inside the lab.

## Executor API

Add a new module:

```text
optics/cuda_direct_light.py
```

Primary class:

```python
class CudaDeviceBvhDirectLightOpticalExecutor:
    """CUDA direct-light executor over DeviceOpticalSceneSnapshot + DeviceOpticalBvh."""

    capabilities = ...
    supported_profiles = {
        OpticalOutputProfile.DIRECT_LIGHT_FULL,
        OpticalOutputProfile.RGB_PREVIEW,
        OpticalOutputProfile.RENDER_ONLY,
    }

    def __init__(
        self,
        *,
        device=None,
        stream=None,
        shadows: bool = True,
        ambient_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
        background_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
        shadow_bias: float = 1.0e-6,
    ) -> None: ...

    def execute(
        self,
        snapshot: DeviceOpticalSceneSnapshot,
        bvh: DeviceOpticalBvh,
        spec: OpticalRaySensorSpec,
        *,
        output_profile: OpticalOutputProfile | str = OpticalOutputProfile.DIRECT_LIGHT_FULL,
        render_profile: list[tuple[str, float]] | None = None,
    ) -> OpticalComputeResult: ...

    def execute_camera(
        self,
        snapshot: DeviceOpticalSceneSnapshot,
        bvh: DeviceOpticalBvh,
        camera: OpticalPinholeCameraSpec,
        *,
        output_profile: OpticalOutputProfile | str = OpticalOutputProfile.DIRECT_LIGHT_FULL,
        render_profile: list[tuple[str, float]] | None = None,
    ) -> OpticalComputeResult: ...
```

The method shape intentionally mirrors `GpuDeviceBvhDirectLightOpticalExecutor`
so `OpticalLabRenderSession` only switches executor construction and keeps
`execute_request(...)` stable.

## CUDA Extension API

Use the existing `torch.utils.cpp_extension.load_inline` approach already
validated by `optics/cuda_lbvh.py`.

Proposed Python wrapper:

```python
@lru_cache(maxsize=1)
def _load_cuda_direct_light_extension():
    return load_inline(
        name="robot_sim_cuda_direct_light_v1",
        cpp_sources=[CPP_SOURCE],
        cuda_sources=[CUDA_SOURCE],
        with_cuda=True,
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        verbose=False,
    )
```

Proposed extension functions:

```cpp
std::vector<torch::Tensor> first_hit_rays(
    torch::Tensor origins,            // float32 [N,3]
    torch::Tensor directions,         // float32 [N,3]
    double max_distance,
    torch::Tensor tri_v0,             // float32 [T,3]
    torch::Tensor tri_e1,             // float32 [T,3]
    torch::Tensor tri_e2,             // float32 [T,3]
    torch::Tensor tri_normal,         // float32 [T,3]
    torch::Tensor bvh_bounds_min,     // float32 [M,3]
    torch::Tensor bvh_bounds_max,     // float32 [M,3]
    torch::Tensor bvh_left,           // int32 [M]
    torch::Tensor bvh_right,          // int32 [M]
    torch::Tensor bvh_start,          // int32 [M]
    torch::Tensor bvh_count,          // int32 [M]
    torch::Tensor bvh_prim_ids,       // int32 [T]
    torch::Tensor primitive_source_order_key,
    torch::Tensor primitive_instance_index,
    torch::Tensor primitive_geometry_index,
    torch::Tensor material_for_instance
);

std::vector<torch::Tensor> first_hit_camera(...);

std::vector<torch::Tensor> shade_direct_light(
    torch::Tensor hit_mask,
    torch::Tensor position_world,
    torch::Tensor normal_world,
    torch::Tensor material_index,
    torch::Tensor material_albedo_rgb,
    torch::Tensor light_kind,
    torch::Tensor light_position_or_direction_world,
    torch::Tensor light_intensity,
    torch::Tensor light_color_rgb,
    ... bvh + triangle buffers for shadow any-hit ...
);
```

The exact function list can evolve during implementation, but the interface
should preserve three logical kernels:

1. primary first-hit traversal;
2. shadow any-hit traversal;
3. direct-light shading.

Combining 2 and 3 inside one kernel is acceptable if the Python result contract
stays the same. Combining primary traversal and shading is not the first target;
that would drift toward future `cuda_fused_rgb`.

## Channel Ownership

The executor should return an `OpticalComputeResult(location="device")` whose
channels may be native CUDA torch tensors.

P12.3 should not force the CUDA backend to wrap every tensor as a Warp array.
That would keep delivery/readback coupled to the older Warp executor shape and
make CUDA look like an implementation detail of Warp. Torch CUDA tensors are a
first-class device buffer representation for this backend and should be
accepted by staging/readback directly.

Proposed output shape:

```python
rgb = torch.empty((num_rays, 3), dtype=torch.float32, device=device)

return OpticalComputeResult(
    location="device",
    channels={"rgb": rgb, ...},
    resources=(rgb, ...),
)
```

This requires a small but important delivery/readback refactor before or during
the CUDA executor work:

```python
def device_channel_to_torch(value: object) -> torch.Tensor:
    if is_torch_cuda_tensor(value):
        return value.contiguous()
    if is_warp_array(value):
        return wp.to_torch(value).contiguous()
    raise TypeError(...)

def device_channel_to_numpy(value: object) -> np.ndarray:
    tensor = device_channel_to_torch(value)
    return tensor.detach().cpu().numpy()
```

Affected helpers:

- `optics.device.stage_optical_compute_result_to_host(...)`
- `optics.device.stage_optical_channels(...)`
- `tools.optical_pipeline_lab.async_readback.TorchAsyncReadbackRing`
- any RGB8 pack path that assumes Warp arrays only

Rationale:

- `cuda_direct_light` should own CUDA tensors natively.
- Async readback is already Torch-based; accepting torch tensors removes an
  unnecessary Warp round-trip.
- Warp executors remain supported through the same adapter by converting Warp
  arrays with `wp.to_torch(...)`.
- This creates the right boundary for future CUDA backends and eventually
  `cuda_fused_rgb`.

## Render Session Dispatch

Extend the existing P12.2 dispatch point:

```python
if options.render_backend == "cpu_direct_light":
    ...
elif options.render_backend == "cuda_direct_light":
    ...
else:
    ... # warp_bvh_direct_light
```

`cuda_direct_light` should reuse:

- `DeviceOpticalSceneCache`;
- `snapshot_from_gpu_frame(..., include_aabb=True)`;
- `build_cuda_lbvh_from_snapshot(...)`;
- `OpticalLabRenderFrameContext`;
- existing video product and delivery logic.

The only new render-session resource should be the CUDA executor.

## Validation Rules

Scenario compatibility:

```text
cuda_direct_light requires accel_backend=cuda_lbvh
cuda_direct_light supports static geometry first
cuda_direct_light rejects CPU_BVH
cuda_direct_light rejects OPTIX
```

Run-option compatibility:

```text
cuda_direct_light initially supports sync delivery
cuda_direct_light should support video_raygen="host" first
cuda_direct_light should support video_raygen="gpu" before P12.3 is considered complete
cuda_direct_light may reject rgb8 until output staging is verified
```

Recommended initial validation sequence:

1. P12.3a: allow explicit config only for host raygen + sync + rgb/full.
2. P12.3b: add CUDA camera raygen and allow `video_raygen="gpu"`.
3. P12.3c: enable rgb8 only if `pack_rgb8` works unchanged with CUDA channels.

## Implementation Slices

### P12.3a: CUDA Direct-Light Plan and Skeleton

- Add this plan.
- Add `optics/cuda_direct_light.py` skeleton with dependency checks.
- Add import/export wiring only if it does not import CUDA dependencies eagerly.
- Add validation fail-fast for `RenderBackend.CUDA_DIRECT_LIGHT`.
- Add or design the device-channel adapter that accepts both Warp arrays and
  torch CUDA tensors.

Acceptance:

- CPU-only imports remain safe.
- Reserved/invalid CUDA backend configs fail clearly.
- No lab behavior changes for CPU/Warp backends.

### P12.3b: First-Hit Host-Ray Kernel

- Add `first_hit_rays(...)` CUDA extension function.
- Convert snapshot/BVH Warp arrays to torch views using `wp.to_torch(...)`.
- Upload host rays or create torch views for ray inputs.
- Output geometry channels:
  - `hit_mask`
  - `range_m`
  - `position_world`
  - `normal_world`
  - `numeric_instance_id`
  - `material_index`
  - `bvh_stack_overflow_count`
  - `bvh_max_stack_depth`
- Compare against `GpuDeviceBvhOpticalExecutor` and CPU BVH on tiny scenes.
- Return CUDA torch tensor channels and stage them through the new adapter.

Acceptance:

- No Warp render kernels are called.
- CUDA LBVH traversal matches existing Warp BVH first-hit within tolerance.

### P12.3c: Minimal Direct-Light Shading

- Add direct-light shade output for at least one directional light.
- Support `shadows=False` first.
- Match CPU/Warp no-shadow RGB within tolerance.
- Preserve output profile filtering.

Acceptance:

- `RGB_PREVIEW` and `DIRECT_LIGHT_FULL` produce expected channels.
- Tiny static scene renders through `OpticalLabRenderSession`.

### P12.3d: Shadow Any-Hit

- Add shadow any-hit traversal against CUDA LBVH triangles.
- Add hard shadow behavior with `shadow_bias`.
- Emit shadow diagnostics:
  - `shadow_stack_overflow_count`
  - `shadow_max_stack_depth`
  - optional traversal counters when profiling is enabled.

Acceptance:

- Tiny scene with occluder matches CPU/Warp hard-shadow result.
- Overflow counters are present and zero in non-overflow tests.

### P12.3e: CUDA Camera Raygen

- Add `first_hit_camera(...)` or a separate CUDA raygen kernel.
- Allow `video_raygen="gpu"` for `cuda_direct_light`.
- Keep host-ray path for parity tests.

Acceptance:

- Host-ray and GPU-camera raygen agree for a tiny camera.
- Optical Lab static video product can run with `video_raygen="gpu"`.

### P12.3f: Lab Workflow Smoke

- Add explicit config workflow smoke using:
  - `render_backend=cuda_direct_light`
  - `accel_backend=cuda_lbvh`
  - tiny synthetic static scene
  - video + debug products
- Verify:
  - `scenario_config.json` records `cuda_direct_light`;
  - `frame_timing.csv` records `cuda_direct_light` and `cuda_lbvh`;
  - host RGB shape and basic numeric sanity.

Acceptance:

- GPU test passes in `env_tilelang_20260119`.
- Existing Go2 `warp_bvh_direct_light/cuda_lbvh` smoke remains unchanged.

## Numerical Tolerances

Initial parity tolerances:

```text
RGB:    abs <= 1/255 and rel <= 1%
Depth:  abs <= 1e-4 for small scenes
Normal: cosine similarity > 0.999
Mask/id channels: exact equality
```

For P12.3 first-hit tests:

```text
range_m:        abs <= 1e-6 on tiny scenes
position_world: abs <= 1e-6 on tiny scenes
normal_world:   abs <= 1e-6 on tiny scenes
```

The looser RGB tolerance accounts for float32 CUDA arithmetic and direct-light
operation ordering.

## Main Risk Areas

### 1. Stream and Event Interop

Existing CUDA LBVH JIT code uses torch CUDA extension launches and currently
synchronizes for simplicity. A render backend cannot make global synchronization
the final performance story.

Plan:

- Spike stream interop early.
- Prefer launching extension kernels on the current torch stream that is aligned
  with the lab render stream.
- If Warp stream to torch `ExternalStream` bridging is unavailable, allow an
  initial correctness-only sync but mark it as incomplete for performance.
- P12.3 is not complete until result readiness is stream/event ordered enough
  for existing delivery to consume safely.

### 2. Channel Type and Lifetime

Plan:

- Treat torch CUDA tensors as first-class device channel values.
- Add a shared device-channel adapter that supports both torch CUDA tensors and
  Warp arrays.
- Keep output tensors in `OpticalComputeResult.resources` when needed to make
  ownership explicit, but do not require Warp views for CUDA outputs.
- Add tests that stage CUDA executor results to host after the executor returns.

### 3. BVH Traversal Correctness

The CUDA LBVH builder has leaf-size-1 topology and no refit. Traversal must
respect:

- closest hit distance;
- source-order tie-breaks;
- stack overflow reporting;
- primitive id remapping through `bvh.prim_ids`.

Plan:

- Start with tiny triangle grids where expected hits are obvious.
- Compare against existing Warp BVH executor before adding shading.

### 4. Scene Feature Scope

The existing Warp direct-light path supports triangles and planes, multiple
lights, roles, and shadow diagnostics. CUDA should not silently ignore features.

Plan:

- First CUDA implementation may support a narrower subset, but unsupported
  features must fail fast.
- Minimum accepted P12.3 subset:
  - triangle meshes;
  - at least one directional light;
  - material albedo;
  - hard shadows against triangles;
  - camera RGB preview.
- Plane support, point lights, and multiple lights can be staged after the first
  CUDA RGB parity smoke if they are not needed by reviewed P12 scenarios.

### 5. Output Profile Contract

`RGB_PREVIEW` still guarantees diagnostic counters. `RENDER_ONLY` still
guarantees overflow counters. Missing counters will break delivery rows.

Plan:

- Allocate counters in all profiles.
- Filter channels only after all required internal work is complete.

### 6. CPU-Only Import Safety

`optics` currently imports optional CUDA modules safely. `cuda_direct_light.py`
must not make CPU-only imports fail.

Plan:

- Guard `torch`, `warp`, and `load_inline` imports like `cuda_lbvh.py`.
- Lazy-load the extension with `lru_cache`.
- Keep top-level module import dependency-light.

### 7. Boundary With `cuda_fused_rgb`

It will be tempting to fuse camera raygen, first-hit, and RGB shading early.
That is future `cuda_fused_rgb`, not P12.3.

Plan:

- Keep normal direct-light channels and diagnostics.
- Avoid a preview-only result contract.
- Optimize after parity and benchmark contract are stable.

## Tests

Unit-level:

- CPU-only import of `optics` still succeeds without CUDA extension tooling.
- Validation:
  - `cuda_direct_light + cuda_lbvh` accepted for static config;
  - `cuda_direct_light + cpu_bvh` rejected;
  - unsupported run options rejected with precise messages.

GPU-level:

- CUDA first-hit matches Warp BVH first-hit on tiny static triangle mesh.
- CUDA no-shadow direct-light matches CPU/Warp no-shadow RGB.
- CUDA shadow direct-light matches CPU/Warp hard-shadow RGB on tiny occluder.
- CUDA executor result can be staged to host through existing staging helpers.
- Lab workflow smoke writes correct `scenario_config.json` and
  `frame_timing.csv`.

Real smoke:

```bash
conda run -n env_tilelang_20260119 env PYTHONPATH=. \
  python examples/optical_lab/go2_video_ordered_static.py \
  --frames 1 \
  --out /tmp/robot_simulator_p12_cuda_direct_go2_smoke \
  --warmup-renders 0
```

This command should remain Warp by default. A separate explicit-config smoke
should exercise `cuda_direct_light` until P12.6 public override lands.

## Review Questions

1. Should P12.3 require GPU camera raygen before being called complete, or is
   host-ray CUDA direct-light enough for the first accepted backend?
2. Is the proposed `CudaDeviceBvhDirectLightOpticalExecutor` class shape the
   right interface, mirroring the Warp direct-light executor?
3. Is the proposed device-channel adapter the right way to support CUDA torch
   tensors directly while preserving Warp executor compatibility?
4. Is triangle-only support acceptable for the first CUDA RGB parity smoke if
   unsupported planes/point lights fail fast?
5. Should we add `create_static_asset_lab_runtime_for_config(...)` in P12.3 to
   avoid manual runtime assembly in explicit backend workflow tests?
6. Is it acceptable for the first kernel slice to use a correctness-only
   synchronization fallback while stream interop is being worked out, or should
   stream-ordered readiness be a blocker from the first commit?
