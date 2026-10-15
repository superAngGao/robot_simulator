Initiative: q54-optical-lab-p12-3e-cuda-camera-raygen
Stage: proposal
Author: codex
Version: v2
Date: 2026-08-07
Status: in_review
Related Files: collab/q54-optical-lab-p12-3e-cuda-camera-raygen__proposal__codex__v1.md; collab/q54-optical-lab-p12-3-cuda-direct-light-backend__implementation-plan__codex__v1.md; collab/q54-optical-lab-p12-3d-cuda-shadow-any-hit__implementation-note__codex__v1.md; optics/cuda_direct_light.py; tools/optical_pipeline_lab/render_session.py; tools/optical_pipeline_lab/runner.py; tests/gpu/test_optical_gpu_runtime.py
Owner Summary: Revise P12.3e as a CUDA-first replacement step, not a Warp/Torch stream-interoperability step. The CUDA executor has an explicit target device, launches every participating extension entry point under a CUDA device guard on that device's current Torch stream, and remains blocking. Direct ray tensors are compared with the canonical host raygen before integration assertions. CUDA-event timing is resolved at the existing terminal synchronization. Warp remains only as a temporary optical reference and input boundary until P12.4 proves parity and the owner-defined frame budget; no bridging work is planned.

## Revision Decision

This file supersedes v1 for review. It preserves v1 as history and incorporates
Claude's request-changes review plus the owner's architectural decision:

> Do not bridge Warp and Torch streams. If the CUDA optical path is fast enough,
> retire the Warp optical path.

The changes from v1 are:

1. define the target CUDA device and current-stream contract end to end;
2. remove the misleading Warp `stream` argument from CUDA executor constructors;
3. require direct origins/directions parity, not only hit or RGB parity;
4. use CUDA events for `raygen_kernel` without adding a raygen-only global sync;
5. cover all three supported direct-light output profiles;
6. make environment evidence a tracked, reproducible prerequisite;
7. define a measured path toward retiring Warp optical rendering without
   expanding this stage into physics-backend replacement.

## Problem

The CUDA direct-light backend accepts host-provided rays, but both CUDA camera
entry points are P12.3e stubs:

```text
host rays -> CUDA LBVH first-hit -> CUDA direct-light shade
camera    -> NotImplementedError
```

The v1 proposal left two important contracts implicit:

- a raygen call containing only scalar camera parameters could not determine the
  correct CUDA device when the process current device differed from the
  executor's device;
- the CUDA executor accepted a Warp stream even though its extension kernels
  launch on a Torch current stream.

That ambiguity is more dangerous than a deliberate blocking boundary. It can
silently allocate on the wrong GPU or suggest stream ordering that does not
exist. P12.3e therefore defines a narrow CUDA-owned execution contract and does
not attempt Warp/Torch stream interoperability.

## Goal

Implement CUDA pinhole camera ray generation as the first camera stage of the
existing CUDA first-hit/direct-light pipeline, with explicit device ownership,
direct numerical parity, and honest timing semantics.

Acceptance conditions:

1. generated CUDA origins and directions agree element-by-element with
   `build_pinhole_camera_rays(...)` for a fixture that exposes indexing,
   intrinsics, translation, and rotation errors;
2. every CUDA extension entry point used by the camera render executes under a
   guard for the executor's target device and uses that device's current Torch
   stream;
3. setting the process current device to a different GPU does not move outputs
   away from the executor target or break the camera render;
4. the camera path works with `direct_light_full`, `rgb_preview`, and
   `render_only`, with and without shadows where applicable;
5. Lab accepts only the existing blocking
   `cuda_direct_light + cuda_lbvh + video_raygen=gpu` combination;
6. `raygen_kernel` measures kernel execution with CUDA events and does not add a
   new raygen-only global synchronization;
7. host-ray and Warp optical paths remain available as parity oracles until the
   later performance gate is passed.

## Architectural Direction

### CUDA-first, no bridging

The transitional execution boundary is deliberately:

```text
Warp-owned scene/snapshot ready event
    -> host-block until ready
    -> Torch tensors/views on explicit target CUDA device
    -> target device's current Torch stream
       [raygen -> first-hit -> shade]
    -> existing target-device synchronization
    -> blocking OpticalComputeResult(ready_event=None)
```

There is no attempt to make a Warp stream wait on a Torch event, make a Torch
stream wait on a Warp event, wrap an external stream, or claim asynchronous
completion. The existing Warp ready-event wait is a temporary producer/consumer
boundary, not stream bridging.

### What "retire Warp" means here

The target is to retire Warp from the optical render path after CUDA proves
correctness and performance on real workloads. It does not mean removing Warp
from physics in P12.3e. Physics backend ownership is a separate architectural
decision with a much larger scope.

P12.3e also does not yet make scene snapshots, LBVH storage, or delivery fully
CUDA-native. Those remaining Warp-owned optical boundaries are recorded as
follow-up migration work rather than hidden behind interoperability code.

## Scope

### In scope

- CUDA pinhole raygen into contiguous float32 Torch CUDA tensors.
- An explicit target-device contract for CUDA raygen, first-hit, and shade.
- Removal of `stream` from both CUDA optical executor constructors and their
  call sites.
- Shared first-hit execution over device ray tensors.
- CUDA camera execution for all currently supported direct-light profiles.
- Narrow Lab enablement for CUDA LBVH plus GPU raygen.
- Direct ray parity, multi-GPU device-guard coverage, integration parity, and
  Lab workflow tests.
- CUDA-event timing for the canonical raygen profile phase.
- Tracked environment reproduction metadata before implementation is accepted.

### Out of scope

- Warp/Torch stream bridging or custom external CUDA streams.
- An asynchronous CUDA optical result contract.
- Fused raygen/traversal/shading kernels.
- RGB8 packing, async delivery, or readback-policy expansion.
- Go2 CUDA visual acceptance; that remains P12.3f.
- Deleting the host-ray path or Warp optical oracle in this commit.
- Removing Warp from physics.
- Final CUDA-owned scene snapshot, LBVH, or delivery migration.
- Performance claims; P12.4 owns the comparison and default-backend decision.

## Device and Stream Contract

### Python device ownership

`CudaDeviceBvhOpticalExecutor` and
`CudaDeviceBvhDirectLightOpticalExecutor` continue to accept `device`, resolve it
once, and store it as `self.device`. They no longer accept or store `stream`.

The camera helper has an explicit device argument rather than consulting the
ambient current device:

```python
_generate_pinhole_camera_ray_tensors(
    camera: OpticalPinholeCameraSpec,
    *,
    device: torch.device,
    render_profile: list[tuple[str, float]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, _CudaEventTiming | None]
```

The wrapper converts `self.device` with `_torch_device_for_warp_device(...)`,
requires `device.type == "cuda"`, and allocates:

```text
origins:    [height * width, 3], float32, contiguous, target device
directions: [height * width, 3], float32, contiguous, target device
```

Allocation happens inside `with torch.cuda.device(target_device)` but allocation
placement is still explicit (`device=target_device`). Correctness must not
depend on the process-global current CUDA device.

Before launching first-hit or shade, all scene, BVH, ray, hit, material, and
light tensors used by that entry point must be validated as CUDA tensors on the
same target device. A mismatch is rejected with a deterministic error before
kernel launch.

### C++/CUDA device guard

Use an output-buffer extension entry point so the device anchor is unambiguous:

```text
pinhole_camera_rays_out(
    origins,
    directions,
    width, height,
    fx, fy, cx, cy,
    origin_world[3],
    R_world_camera[9],
) -> void
```

The C++ wrapper validates output dtype, shape, contiguity, CUDA placement, and
same-device ownership. It derives the device index from `origins`, installs a
CUDA device guard, and launches with the current CUDA stream for that explicit
device index.

The same guard/indexed-current-stream rule must be applied to the existing
`first_hit_rays`, no-shadow shade, and shadow shade entry points. Otherwise a
correct raygen could still be followed by a wrong-device allocation or launch
when the ambient current device differs from the executor target.

Conceptually, every participating entry point follows:

```cpp
target = anchor_tensor.device();
CUDADeviceGuard guard(target);
validate_all_tensors_on(target);
stream = getCurrentCUDAStream(target.index());
allocate_outputs_with(anchor_tensor.options());
launch_on(stream);
```

The exact guard type/header should match the installed Torch extension API;
the behavioral contract above is the review requirement.

### Torch current stream only

P12.3e supports only the target device's current Torch stream. It does not
accept a Warp stream, a raw `cudaStream_t`, or an external-stream wrapper.

Raygen, first-hit, and shade are enqueued in order on that same current Torch
stream. The current CUDA backend already synchronizes the target device before
return and emits `ready_event=None`; P12.3e preserves that blocking behavior.

`tools/optical_pipeline_lab/render_session.py` must branch executor construction:

```text
CUDA executor: device only, no stream
Warp executor: device plus workspace.stream
```

Unit fakes and direct constructor call sites are updated accordingly. Passing a
`stream` argument to a CUDA executor becomes an immediate `TypeError`, rather
than a silently ignored ordering promise.

## Proposed Implementation

### 1. Add the raygen kernel and testable Python helper

Add one CUDA thread per flat pixel:

```text
flat_index = py * width + px
px = flat_index % width
py = flat_index / width
d_camera = normalize([(px - cx) / fx, (py - cy) / fy, 1])
d_world = R_world_camera @ d_camera
origins[flat_index] = camera_translation_world
directions[flat_index] = d_world
```

The helper remains module-private but is directly callable by GPU tests. This
keeps the low-level ray contract observable without creating a new public API.
The extension name is version-bumped because its exported symbols and device
guard behavior change.

### 2. Factor first-hit over device ray tensors

Refactor geometry execution into:

```text
execute(host ray spec)
    -> validate request context
    -> upload host rays to explicit target device
    -> _execute_device_rays(...)

execute_camera(camera spec)
    -> validate request context
    -> generate CUDA rays on explicit target device
    -> _execute_device_rays(...)
```

The shared helper owns input ready-event waits, channel views, device checks,
the existing first-hit extension call, terminal synchronization, profile-event
resolution, result construction, and resource lifetime. Request metadata is
passed explicitly; it does not fabricate a host ray spec.

Generated origins and directions remain in `OpticalComputeResult.resources` so
their storage survives through shading and delivery.

### 3. Reuse direct-light shading

The direct-light camera entry point only selects the requested output profile,
calls the geometry camera path, and passes the geometry result into the same
shadow or no-shadow shading helpers as the host-ray path. It introduces no
camera-specific shader, channel layout, or material behavior.

### 4. Narrow Lab capability change

For `render_backend=cuda_direct_light`, allow `video_raygen` in `{host, gpu}`
while keeping all current restrictions:

```text
accel_backend = cuda_lbvh
video_readback_delivery = sync
readback_payload != rgb8
video_ray_cache = off when video_raygen = gpu
scenario remains within the existing synthetic/static CUDA support boundary
```

No other capability guard is relaxed.

## Timing Contract

The canonical logical phase is `raygen_kernel`. The timing row and CSV column
derived by the existing Lab code are `render_raygen_kernel_ms`. There is no new
top-level field named only `raygen_kernel_ms`.

When profiling is enabled:

1. allocate ray buffers before the start event;
2. record a Torch CUDA start event on the target device's current stream;
3. launch only `pinhole_camera_rays_out(...)`;
4. record the end event on the same stream;
5. enqueue first-hit and any later work without an intermediate synchronization;
6. at the existing terminal target-device synchronization, resolve
   `start.elapsed_time(end)` and append `("raygen_kernel", elapsed_ms)`.

Thus `raygen_kernel` measures device execution rather than CPU launch time and
does not add a raygen-only global barrier. Buffer allocation and camera-parameter
preparation remain separately observable as `raygen_buffer_alloc` and
`raygen_camera_params` when useful.

If execution fails before the terminal sync, an unresolved timing token must
not append a misleading value. Profiling disabled must avoid event allocation.

## Affected Files / Layers

### `optics/cuda_direct_light.py`

- add raygen C++/CUDA bindings and the private Python helper;
- add device guards and same-device validation to all camera-path extension
  entry points;
- remove CUDA executor `stream` parameters;
- factor shared device-ray first-hit execution;
- implement both camera entry points;
- retain ray tensors and timing events for the required lifetime.

### `tools/optical_pipeline_lab/render_session.py`

- construct Warp and CUDA executors through explicit backend branches;
- pass `workspace.stream` only to the Warp executor.

### `tools/optical_pipeline_lab/runner.py`

- remove only the temporary CUDA host-ray restriction;
- preserve CUDA LBVH, sync-delivery, non-RGB8, ray-cache, and scenario guards.

### `tests/unit/optics/test_optical_pipeline_lab.py`

- update executor-construction fakes for the no-stream CUDA contract;
- accept the supported CUDA GPU-raygen combination;
- continue asserting rejection of unrelated combinations;
- check profile-key mapping from `raygen_kernel` to
  `render_raygen_kernel_ms`.

### `tests/gpu/test_optical_gpu_runtime.py`

- direct ray tensor parity;
- device guard and wrong-current-device coverage;
- camera first-hit and direct-light parity;
- all supported output profiles;
- no-shadow and hard-shadow integration;
- normal Lab GPU-raygen smoke.

### Tracked environment evidence

Before the implementation is submitted for review, add a small tracked
reproduction bundle under `deploy/robot_sim_cuda128/`:

```text
README.md             relay/offline install and validation commands
conda-explicit.txt    exact Conda package URLs/builds from `conda list --explicit`
python-packages.txt   exact Python package inventory from `python -m pip freeze`
wheelhouse.sha256     SHA-256 manifest for the 62 relayed wheels; no wheel binaries
```

The README records the environment path only as an example, not a portable
requirement, and explains that network failure is handled by transferring the
wheelhouse from the local machine rather than switching package families or
installing unreviewed alternatives. The hash claim is evidence only after both
ends are checked against the tracked manifest. Chat output alone is not treated
as reproducible evidence.

## Validation Plan

### 1. Direct raygen parity (required low-level test)

Use a fixture with all of the following:

- non-square image, for example width 7 and height 3;
- `fx != fy`;
- non-centered `cx` and `cy`;
- non-zero camera translation;
- compound, non-identity world rotation;
- enough pixels to sample corners, interior pixels, and row boundaries.

Materialize canonical host rays with `build_pinhole_camera_rays(...)` and assert:

- CUDA origins equal host origins element-by-element;
- CUDA directions equal host directions element-by-element within the agreed
  float32 tolerance;
- output shape, dtype, contiguity, and target device are exact;
- explicit indices `(py, px)` map to `py * width + px`, including the last pixel
  of one row and first pixel of the next;
- every direction has unit length within tolerance.

These assertions are independent of BVH hits and RGB.

### 2. Device/stream contract tests

On a machine with at least two CUDA devices:

1. intentionally set Torch's current device to `cuda:0`;
2. create the scene/BVH and executor for `cuda:1`;
3. invoke direct raygen and the full camera path without changing the ambient
   current device to `cuda:1`;
4. assert every output/resource tensor is on `cuda:1` and results match the host
   reference.

Skip this test only when fewer than two devices are visible. A one-GPU test also
asserts that CUDA executor constructors reject the removed `stream` keyword and
that all participating tensors must share the executor target device.

No test may interpret a Warp stream as ordering the CUDA executor.

### 3. First-hit and direct-light integration parity

Use an asymmetric scene whose geometry or materials distinguish rows, columns,
and neighboring pixels. Do not rely on a single wide plane with uniform
material. Compare host-ray CUDA execution with CUDA-camera execution:

- `hit_mask`, instance IDs, and material IDs exact;
- range, position, and normal within explicit float32 tolerances;
- RGB/intensity within explicit tolerances;
- BVH and shadow overflow counters present and zero;
- request metadata exact;
- result remains valid through host staging.

Run both no-shadow and hard-shadow variants.

### 4. Supported profile matrix

Parameterize the camera path over:

```text
direct_light_full
rgb_preview
render_only
```

For each profile, assert the exact guaranteed channel set and absence of
channels excluded by that profile. At least one shadow and one no-shadow case
must pass through this matrix; additional combinations may be kept minimal if
the shared shade implementation is directly demonstrated.

### 5. Lab workflow

Run the normal tiny static/synthetic workflow with:

```text
render_backend=cuda_direct_light
accel_backend=cuda_lbvh
video_raygen=gpu
video_readback_delivery=sync
readback_payload=rgb
video_ray_cache=off
```

Assert successful completion, CUDA backend metadata, `raygen_mode=gpu`, valid
RGB output, finite `render_raygen_kernel_ms` when profiling is enabled, and no
relaxation of the existing unsupported combinations.

### 6. Commands and gates

Use the tracked environment and target GPU, for example:

```bash
conda run -n robot_sim env PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
  python -m pytest tests/gpu/test_optical_gpu_runtime.py \
  -k "cuda and (camera or raygen)" -q

conda run -n robot_sim env PYTHONPATH=. \
  python -m pytest tests/unit/optics -q

conda run -n robot_sim env PYTHONPATH=. \
  python -m pytest tests/ -m "not (slow or gpu)" -q

conda run -n robot_sim python -m ruff check \
  optics/cuda_direct_light.py \
  tools/optical_pipeline_lab/render_session.py \
  tools/optical_pipeline_lab/runner.py \
  tests/unit/optics/test_optical_pipeline_lab.py \
  tests/gpu/test_optical_gpu_runtime.py
```

Run the full `not slow` suite before push. The two-device test is recorded as
skipped rather than silently treated as passed when the hardware is unavailable.

## Implementation Sequence

### Phase 0: evidence and baseline

- add the tracked environment reproduction bundle;
- rerun the existing unit, targeted GPU, and commit-gate baselines;
- record GPU count so the multi-GPU test result is interpretable.

### Phase 1: contract cleanup

- remove CUDA executor `stream` parameters and branch Lab construction;
- add explicit target-device validation and guards to existing first-hit/shade
  bindings;
- run host-ray CUDA regressions before adding raygen.

### Phase 2: raygen and geometry camera path

- add output-buffer raygen binding and private helper;
- add direct ray parity and wrong-current-device tests;
- factor `_execute_device_rays(...)` and implement geometry `execute_camera`;
- verify resource lifetime and timing-event resolution.

### Phase 3: direct-light and Lab path

- implement direct-light `execute_camera` through shared shade helpers;
- cover all three profiles and both shadow modes;
- remove only the Lab P12.3e host-ray guard;
- run the normal GPU-raygen workflow and regression gates.

### Phase 4: review handoff

- write a Codex implementation note with changed files, exact commands, results,
  environment evidence, limitations, and commit hash;
- ask Claude to review device/stream behavior, direct ray parity, timing, profile
  coverage, and scope adherence;
- do not enable CUDA as the default backend or delete Warp in this stage.

## Warp Optical Retirement Gate (P12.4 and later)

P12.3e makes CUDA camera rendering runnable; it does not prove that CUDA should
replace Warp. P12.4 must compare warmed, end-to-end optical workloads rather
than isolated kernel launch times:

- raygen, LBVH build/refit, first-hit, shade, and delivery;
- representative resolutions, scene sizes, shadow modes, and output profiles;
- static and dynamic scene updates;
- stable post-warmup latency, throughput, memory use, and correctness;
- compile/warmup costs reported separately.

Before benchmarking, the owner records the absolute frame-time or throughput
budget for the intended workload. The decision is not based merely on "faster
than Warp"; CUDA must meet that product budget while passing parity and
stability gates.

If it passes:

1. make the CUDA optical backend the default for the validated workload;
2. freeze the Warp optical path except for temporary reference tests;
3. migrate remaining optical scene, BVH, and delivery ownership to CUDA-native
   contracts;
4. remove the Warp optical executor after the CUDA path covers all retained
   capabilities and rollback evidence is captured.

If it fails, optimize the CUDA implementation directly. Do not spend the next
stage building a Warp/Torch stream bridge unless the owner explicitly reverses
this decision.

## Risks and Tradeoffs

### Blocking boundary limits peak overlap

Host-blocking Warp ready events and terminal Torch synchronization prevent
producer/render overlap. This is intentional for correctness and contract
clarity. The cost is measured in P12.4 and removed by migrating ownership to
CUDA, not by introducing a bridge that the target architecture does not want.

### Device-guard work expands the patch slightly

Guarding first-hit and shade as well as raygen is broader than adding one
kernel, but it is required for the executor-level device promise and the
requested multi-GPU test. Leaving downstream kernels ambient-device-dependent
would make the new raygen contract misleading.

### Separate ray buffers cost memory

Two `[height * width, 3]` float32 buffers cost more than a fused kernel. They
keep raygen directly testable, preserve the shared first-hit implementation,
and provide the evidence needed before later fusion or buffer reuse.

### Constructor change touches tests and Lab setup

Removing `stream` is a small internal API break. It is preferable to retaining
an argument whose semantics are false. Warp executors keep their stream API.

### Warp remains temporarily present in optics

The scene snapshot and LBVH boundary still involve Warp data types/events after
P12.3e. This is explicitly transitional. Deleting them before parity and
performance evidence would combine camera raygen with a much larger ownership
migration.

## Review Checklist for Claude

Request approval only if the revised proposal is sufficiently explicit about:

1. no Warp/Torch stream bridging and no asynchronous claim;
2. target-device allocation plus guards for raygen, first-hit, and shade;
3. removal of the CUDA executor `stream` parameter;
4. direct origins/directions and row-major index assertions;
5. two-device wrong-current-device coverage with a recorded hardware skip;
6. CUDA-event timing resolved at the existing terminal synchronization;
7. all three supported output profiles;
8. tracked environment reproduction evidence;
9. Warp optical retirement only after an owner-defined P12.4 performance gate.

## Recommendation

Approve P12.3e under this revised contract. It closes the review gaps without
investing in stream bridging, makes CUDA device ownership testable, and moves
the project toward a CUDA-owned optical pipeline while retaining Warp only long
enough to serve as a correctness/performance oracle.
