Initiative: q54-optical-lab-p12-3e-cuda-camera-raygen
Stage: proposal
Author: codex
Version: v1
Date: 2026-08-06
Status: in_review
Related Files: collab/q54-optical-lab-p12-3-cuda-direct-light-backend__implementation-plan__codex__v1.md; collab/q54-optical-lab-p12-3d-cuda-shadow-any-hit__implementation-note__codex__v1.md; optics/cuda_direct_light.py; optics/warp_execution.py; tools/optical_pipeline_lab/runner.py; tests/gpu/test_optical_gpu_runtime.py
Owner Summary: Implement P12.3e as a separate CUDA pinhole-ray generation stage that feeds the existing production first-hit and direct-light paths. Reuse the established camera convention and profile contract, preserve host-ray parity, and enable only `cuda_direct_light + cuda_lbvh + video_raygen=gpu`; do not fuse raygen/traversal/shading or expand async/rgb8/Go2 scope.

## Problem

The CUDA direct-light backend is functionally complete for host-provided rays:

```text
host rays -> CUDA LBVH first-hit -> CUDA direct-light shade
```

Both CUDA camera entry points are still explicit P12.3e stubs:

```text
CudaDeviceBvhOpticalExecutor.execute_camera(...)
CudaDeviceBvhDirectLightOpticalExecutor.execute_camera(...)
```

The Lab therefore rejects:

```text
render_backend="cuda_direct_light"
accel_backend="cuda_lbvh"
video_raygen="gpu"
```

The repository already has a production Warp camera-ray path and an established
camera convention. P12.3e should add the equivalent CUDA stage without creating
a second render pipeline or weakening host/GPU parity coverage.

## Goal

Add device-side pinhole camera ray generation for the existing CUDA first-hit
and direct-light executors, then enable the corresponding Optical Lab option.

Acceptance conditions:

1. CUDA camera raygen agrees with materialized host rays on a deterministic tiny
   camera and scene.
2. Both no-shadow and hard-shadow direct-light camera paths reuse the already
   landed CUDA shading implementation.
3. A normal Lab static synthetic video run succeeds with
   `video_raygen="gpu"` and records `raygen_mode=gpu`.
4. Host-ray execution remains available and unchanged for parity tests.
5. CPU-only import safety and existing output-profile guarantees remain intact.

## Open Questions Addressed

None in `OPEN_QUESTIONS.md`. P12.3e is already an accepted sub-stage of Q54.

## References Consulted

### In-repo primary references

- `sensing.optical.build_pinhole_camera_rays(...)` defines the canonical host
  lowering and OpenCV-style camera coordinates: +X right, +Y down, +Z optical
  axis.
- `optics.warp_execution._pinhole_camera_raygen_kernel` defines the existing
  production GPU formula, flat row-major pixel order, device buffer layout, and
  `raygen_kernel` timing phase.
- `GpuDeviceBvhOpticalExecutor.execute_camera(...)` proves that camera raygen
  should remain a stage feeding the shared traversal path.
- The P12.3 implementation plan explicitly permits a separate CUDA raygen
  kernel and rejects premature raygen/traversal/shading fusion.

### External design alignment

The existing repository references favor a fixed-phase GPU pipeline over
per-component dispatch. This proposal follows that rule: one flat raygen kernel
produces SoA-compatible device tensors consumed by the current first-hit stage.
No new scene graph, sensor backend hierarchy, or public rendering contract is
introduced.

## Scope

### In scope

- CUDA pinhole camera ray generation into contiguous Torch CUDA tensors.
- CUDA geometry executor `execute_camera(...)`.
- CUDA direct-light executor `execute_camera(...)` for all currently supported
  direct-light output profiles.
- Lab validation for `cuda_direct_light + cuda_lbvh + video_raygen=gpu`.
- Unit validation and targeted GPU parity/workflow coverage.
- Canonical `raygen_kernel` timing emission.

### Out of scope

- Fused raygen + traversal + shading kernels.
- Async delivery, RGB8 packing, or readback-policy expansion.
- Go2 CUDA visual acceptance; that remains P12.3f.
- New camera models, distortion, rolling shutter, depth of field, or batching.
- Removal of the host-ray CUDA path.
- Performance claims or benchmark-matrix expansion; those remain P12.4.

## Affected Files / Layers

### `optics/cuda_direct_light.py`

- Add the CUDA raygen extension entry point and kernel.
- Factor host upload from the shared first-hit execution over device tensors.
- Implement both `execute_camera(...)` methods.
- Preserve device-channel ownership in `OpticalComputeResult.resources`.

### `tools/optical_pipeline_lab/runner.py`

- Remove only the temporary P12.3e host-ray restriction for
  `RenderBackend.CUDA_DIRECT_LIGHT`.
- Retain existing restrictions for sync delivery and non-RGB8 payloads.
- Retain the global `gpu raygen -> ray cache off` validation.

### `tests/unit/optics/test_optical_pipeline_lab.py`

- Replace the old expected rejection with an accepted GPU-raygen configuration.
- Assert unrelated CUDA restrictions remain active.

### `tests/gpu/test_optical_gpu_runtime.py`

- Add low-level host-ray versus CUDA-camera parity.
- Add direct-light camera parity for no-shadow and shadow paths.
- Add one production Lab workflow smoke with `video_raygen="gpu"`.

## Proposed Design

### 1. Keep raygen as an explicit extension stage

Add a Torch CUDA extension function:

```text
pinhole_camera_rays(
    width, height,
    fx, fy, cx, cy,
    origin_world[3],
    R_world_camera[9],
) -> (origins[num_rays, 3], directions[num_rays, 3])
```

The wrapper allocates contiguous float32 tensors on the current input/device
stream. One CUDA thread handles one flat row-major pixel:

```text
px = ray % width
py = ray / width
d_camera = normalize([(px - cx) / fx, (py - cy) / fy, 1])
d_world = R_world_camera @ d_camera
origin_world[ray] = camera translation
```

This is intentionally equivalent to both the host implementation's
`directions_camera @ R.T` and the Warp kernel's component-wise multiplication.

The extension name should be version-bumped because its exported function set
changes. CPU-only imports remain guarded by the existing optional dependency
boundary.

### 2. Factor first-hit execution over device ray tensors

Refactor `CudaDeviceBvhOpticalExecutor.execute(...)` into three conceptual
steps without changing its public contract:

```text
host request validation
-> host ray upload
-> shared _execute_device_rays(...)
```

`execute_camera(...)` becomes:

```text
camera request validation
-> CUDA pinhole raygen
-> shared _execute_device_rays(...)
```

The shared helper owns:

- snapshot/BVH ready-event synchronization;
- scene/BVH Torch channel views;
- the existing `first_hit_rays(...)` call;
- CUDA synchronization used by the current correctness-first backend;
- result construction and resource lifetime.

It receives request metadata explicitly (`frame_id`, `sim_time`, `env_idx`,
`sensor_id`, `max_distance`, `sensor_role`) so it does not fabricate an
`OpticalRaySensorSpec` or materialize host rays.

### 3. Share request-context validation

Extract common validation over:

```text
snapshot frame/env
BVH frame/env/device
executor device
finite positive max_distance
```

Host `OpticalRaySensorSpec` and `OpticalPinholeCameraSpec` entry points both use
that helper. Existing constructor-level camera checks remain owned by
`OpticalPinholeCameraSpec`.

### 4. Reuse direct-light shading unchanged

`CudaDeviceBvhDirectLightOpticalExecutor.execute_camera(...)` should mirror the
host-ray method:

```text
geometry = self._first_hit.execute_camera(...)
if shadows:
    return self._shade_geometry_with_shadows(..., sensor_role=camera.sensor_role)
return self._shade_geometry_no_shadow(...)
```

No duplicate shade kernel or camera-specific output contract is added. This
also preserves all current profile filtering and diagnostic counters.

### 5. Preserve resource lifetime

The generated Torch `origins` and `directions` tensors must remain in the
geometry result's `resources`. Direct-light results already extend geometry
resources, so the tensors remain alive through staging/delivery.

The result remains flat at the executor layer. Existing camera/image wrappers
own `(height, width)` interpretation; no image-shaped device tensor contract is
introduced here.

### 6. Emit the established timing phase

Record the raygen call plus the current correctness synchronization as:

```text
raygen_kernel
```

This is the canonical phase already consumed by
`RENDER_PROFILE_PHASES` and `frame_timing.csv`. CUDA-specific diagnostic phases
may be added only if useful, but they do not replace the canonical phase.

P12.3e does not change the current global synchronization strategy. Stream/event
optimization belongs to later performance work and must not be hidden behind a
camera feature patch.

### 7. Narrow Lab validation change

For `RenderBackend.CUDA_DIRECT_LIGHT`, allow:

```text
video_raygen in {"host", "gpu"}
```

Continue to require:

```text
accel_backend = cuda_lbvh
video_readback_delivery = sync
readback_payload != rgb8
video_ray_cache = off when video_raygen = gpu
```

Do not expand `can_run_scenario(...)` from synthetic static scenes to Go2 in
this slice.

## Validation Plan

All GPU commands use the verified project environment:

```bash
conda run -n robot_sim env \
  PYTHONPATH=. \
  CUDA_VISIBLE_DEVICES=0 \
  CUDA_HOME=/home/ang.gao/anaconda3/envs/robot_sim \
  TORCH_EXTENSIONS_DIR=/data3/ang.gao/.cache/torch_extensions_robot_sim_cuda128 \
  python -m pytest ...
```

### Static/unit validation

```bash
conda run -n robot_sim env PYTHONPATH=. \
  python -m pytest tests/unit/optics/test_optical_pipeline_lab.py \
  -k "cuda_direct_light" -q

conda run -n robot_sim env PYTHONPATH=. \
  python -m pytest tests/unit/optics -q

conda run -n robot_sim python -m ruff check \
  optics/cuda_direct_light.py \
  tools/optical_pipeline_lab/runner.py \
  tests/unit/optics/test_optical_pipeline_lab.py \
  tests/gpu/test_optical_gpu_runtime.py
```

### GPU parity

Add tests with a small non-square camera and non-identity camera transform so
width/height indexing and rotation conventions are both exercised.

Required assertions:

- host-ray and CUDA-camera `hit_mask` exact;
- `range_m`, `position_world`, and `normal_world` within `1e-6`;
- instance/material ids exact;
- RGB within `1e-5` for no-shadow and shadow paths;
- overflow counters present and zero;
- camera result metadata matches the camera request;
- generated result stays valid through host staging.

### Lab workflow

Run one tiny synthetic static scenario with:

```text
render_backend=cuda_direct_light
accel_backend=cuda_lbvh
video_raygen=gpu
video_readback_delivery=sync
readback_payload=rgb
```

Assert:

- the run completes through the normal Lab runner;
- scenario metadata records the CUDA backend;
- timing rows record `raygen_mode=gpu`;
- `render_raygen_kernel_ms` is finite when timing diagnostics are enabled;
- produced RGB shape and finite-value sanity match the existing smoke contract.

### Regression gates

```bash
conda run -n robot_sim env PYTHONPATH=. \
  python -m pytest tests/ -m "not (slow or gpu)" -q

conda run -n robot_sim env PYTHONPATH=. \
  python -m pytest tests/ -m "not slow" -q
```

The full suite remains required before push.

## Risks

### Numerical convention drift

The highest correctness risk is transposition or pixel-order drift. A
non-identity rotation and non-square camera are mandatory in parity coverage;
an identity/downward-only fixture is insufficient.

### Accidental synchronization claims

The current CUDA backend synchronizes for correctness. P12.3e must not describe
the result as asynchronous or event-ordered merely because raygen runs on a CUDA
stream.

### Resource lifetime regression

If generated rays are local temporaries only, later staging may observe freed
storage. The tensors must be retained in `OpticalComputeResult.resources` and
validated after executor return.

### Premature fusion

Fusing raygen, traversal, and shade would make geometry/direct-light parity and
diagnostic channels harder to test. Fusion remains a separate backend or later
optimization after P12.4 measurements.

### Validation overreach

Enabling Go2 or async/rgb8 while landing camera raygen would mix independent
acceptance gates. Runner changes must remove only the P12.3e host-ray guard.

## Tradeoffs

### Separate ray tensors cost memory

The design allocates two `(num_rays, 3)` float32 tensors. This is less memory
efficient than a fused kernel, but it preserves one traversal implementation,
observable stage boundaries, host/GPU parity, and reusable geometry outputs.

### Extension-level allocation is less reusable than an output-buffer API

Returning new tensors is consistent with the existing Torch extension style and
keeps P12.3e narrow. Buffer reuse can be introduced later only with benchmark
evidence and an explicit lifetime contract.

### Correctness synchronization remains

Keeping the existing synchronization makes timing include completion and avoids
new stream-interop semantics in a feature slice. It is not the final performance
design, but it is explicit and comparable to the current CUDA stages.

## Recommendation

Approve the staged, non-fused implementation above. Claude review should focus
on:

1. whether factoring `_execute_device_rays(...)` preserves every host-ray
   validation and resource guarantee;
2. whether the proposed parity fixture is strong enough to catch matrix and
   pixel-order mistakes;
3. whether any Lab capability gate beyond the P12.3e host-ray restriction would
   be relaxed accidentally;
4. whether canonical timing and current synchronization are represented
   honestly.

