# Q54 Optical Pipeline Lab A8 Complex Dynamic Scene Plan

Date: 2026-05-11
Author: Codex
Status: draft-plan

## Context

A5-A7 make the static Go2 lab path flow through an internal pipeline/frame
context shape. That is enough for API cleanup, but it does not yet test the
future physics integration shape:

```text
physics step
  -> per-frame GpuPublishedFrame
  -> pipeline.begin_frame(frame_inputs)
  -> snapshot/refit or rebuild
  -> render
```

The current implemented lab preset is still static:

```text
go2_video_ordered_static
geometry_mode = static
accel_policy = build_once
scene_preset = go2_menagerie_static
```

`dynamic_rigid` and `refit_each_frame` are reserved and fail loudly today.

## Goal

Add one deliberately small but more realistic test scenario before making the
pipeline/session boundary more generic.

The goal is not a full physics benchmark. The goal is to exercise:

- `begin_frame(frame_inputs)` with a different frame object per render frame;
- per-frame `DeviceOpticalSceneSnapshot`;
- BVH refit or explicit rebuild policy;
- stable render/delivery CSV fields under a non-static scene state.

## Recommended Scenario

Name:

```text
go2_video_ordered_dynamic_refit_smoke
```

Shape:

```text
scene: Go2 Menagerie visual scene
geometry_mode: dynamic_rigid
accel_backend: cuda_lbvh or CPU BVH depending refit support
accel_policy: refit_each_frame
frames: 2-5 for smoke
resolution: start at 160x120, then 1080p once stable
readback: rgb8
delivery: torch_async ring_depth=2
render_profile: on
```

Frame motion:

```text
frame 0: original static pose
frame 1+: apply a small deterministic rigid transform to one or more visual bodies
```

The motion should be large enough to change AABBs and rendered pixels, but small
enough to avoid invalid model states.

## Implementation Options

### Option A: Synthetic Published Frame Perturbation

Build the Go2 scene as today, then generate a sequence of `GpuPublishedFrame`
objects with a deterministic pose perturbation.

Pros:

- closest to the future physics integration shape;
- directly tests `begin_frame(frame_inputs)`;
- avoids adding a full simulator dependency to the lab runner.

Cons:

- requires a clear helper for mutating/copying published body poses;
- must be careful not to mutate the baseline static frame in place.

Implementation pre-step:

```text
A8.0: verify GpuPublishedFrame clone/perturb path
  - inspect GpuPublishedFrame fields and ownership;
  - confirm whether wp.clone or an equivalent array copy is available;
  - add a CPU/import-safe test for helper behavior where possible;
  - only then wire begin_frame(frame_inputs=other_frame).
```

### Option B: Tiny Synthetic Scene

Create a minimal registry with a plane plus one moving mesh/box, then publish
two frames.

Pros:

- small and fast;
- easier to reason about image differences and refit correctness.

Cons:

- less representative than Go2;
- may require new scene construction helpers outside current Menagerie import.

### Option C: Full Physics Step

Use the simulator/physics stack to step Go2 and publish frames into the optical
pipeline.

Pros:

- highest fidelity integration test.

Cons:

- bigger dependency surface;
- slower and less deterministic;
- likely too much for the first dynamic optical lab scenario.

## Recommendation

Start with Option A if `GpuPublishedFrame` can be copied/perturbed cleanly.
Fallback to Option B if published-frame mutation is awkward.

Do not start with full physics stepping. That should come after the pipeline can
consume per-frame published state in a controlled lab smoke.

Before implementing the dynamic smoke, run a CPU-only or import-safe probe for
the published-frame cloning path. The probe should answer:

```text
Can we construct a new GpuPublishedFrame with copied/independent Warp arrays?
Can we perturb the copy without mutating the static baseline session.gpu_frame?
```

Direct in-place mutation of `session.gpu_frame` is not acceptable, because it
would invalidate the static baseline and make per-frame behavior alias-prone.

## Required API Work

The current `Go2RenderPipeline.begin_frame(...)` only accepts:

```text
None
or the exact session.gpu_frame object
```

A8 should extend it to:

```python
def begin_frame(
    self,
    frame_inputs: GpuPublishedFrame | None = None,
    *,
    env_idx: int = 0,
) -> Go2RenderFrameContext:
    ...
```

Behavior:

```text
frame_inputs is None:
  use the existing static session.gpu_frame / snapshot / bvh

frame_inputs is session.gpu_frame:
  same as static path

frame_inputs is a different GpuPublishedFrame:
  snapshot_from_gpu_frame(frame_inputs)
  refit_device_bvh_from_snapshot(...) if supported
  otherwise rebuild acceleration according to policy
  return a frame context carrying frame-specific snapshot/bvh
```

This likely requires `Go2RenderFrameContext` to carry `snapshot` and `bvh`
instead of always reading `session.snapshot` / `session.bvh`.

Concrete frame-context shape:

```python
@dataclass
class Go2RenderFrameContext:
    session: Go2RenderSession
    env_idx: int = 0
    snapshot: object | None = None
    bvh: object | None = None

    def render(self, request: RenderRequest) -> RenderResult:
        snapshot = self.snapshot if self.snapshot is not None else self.session.snapshot
        bvh = self.bvh if self.bvh is not None else self.session.bvh
        ...
```

The static path can keep returning a context with `snapshot=None` and `bvh=None`.
The dynamic path should populate both fields with frame-specific resources.

Acceleration policy sequencing:

```text
first dynamic smoke:
  CPU BVH or another backend known to support refit
  expectation: accel_refit_ms populated

second dynamic smoke:
  CUDA LBVH rebuild path if supports_refit=False
  expectation: accel_rebuild_ms populated
```

This avoids confusing "dynamic frame support" with the current CUDA LBVH refit
limitation. If CUDA LBVH reports `supports_refit=False`, a `refit_each_frame`
policy should either fail loudly or intentionally fall back to rebuild with
clear CSV metadata.

## Acceptance Criteria

CPU tests:

- published-frame clone/perturb helper can create an independent frame object
  or the plan explicitly switches to the tiny synthetic scene fallback;
- `begin_frame(frame_inputs=session.gpu_frame)` preserves current static path.
- `begin_frame(frame_inputs=other_frame)` creates a frame context with
  frame-specific snapshot/BVH objects using mocked cache/refit helpers.
- Dynamic/refit reserved scenario no longer fails once the implementation lands.

GPU smoke:

```text
frames: 2-5
resolution: 160x120 first
readback: rgb8
delivery: torch_async
render_profile: on
```

CSV expectations:

- `snapshot_ms` populated for dynamic frames;
- `accel_refit_ms` populated for the first refit-capable smoke;
- `accel_rebuild_ms` populated for the CUDA LBVH rebuild smoke, if CUDA LBVH
  still reports `supports_refit=False`;
- `render_execute_ms`, `pack_rgb8_ms`, `readback_host_ms` still populated;
- no stack overflow;
- optional: frame RGB differs between frame 0 and frame 1.

## Non-Goals

- Do not optimize refit/rebuild performance in the first dynamic smoke.
- Do not add path tracing.
- Do not design the public `OpticalCameraStream` API yet.
- Do not run full RL/simulator integration as the first step.

## Open Questions

1. Can we safely clone/perturb `GpuPublishedFrame` without mutating the static
   baseline frame?
2. Does the current CUDA LBVH object report `supports_refit=False` for Go2,
   requiring rebuild rather than refit?
3. Should the first dynamic smoke use CPU BVH for refit support, or CUDA LBVH
   rebuild for closer parity with current baseline?
4. Where should frame motion helpers live: lab-only tools or physics test
   helpers?

## Claude Review Notes

Claude review of the A7/A8 plan agreed that static Go2 is enough for A7 but not
for physics-style `begin_frame(frame_inputs)` semantics.

Accepted A8 additions:

```text
1. Verify the GpuPublishedFrame clone/perturb path before implementation.
2. Use a refit-capable backend for the first dynamic smoke; treat CUDA LBVH
   rebuild as a second smoke if supports_refit=False.
3. Extend Go2RenderFrameContext with optional frame-specific snapshot/bvh fields
   that fall back to session.snapshot/session.bvh for the static path.
```
