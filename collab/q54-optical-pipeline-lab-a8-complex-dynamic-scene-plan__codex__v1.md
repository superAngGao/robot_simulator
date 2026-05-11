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

A8.0 update: `tools.optical_pipeline_lab.dynamic_frames` now has a lab-only
pose-frame clone/perturb helper. It can build an independent pose-only
`GpuPublishedFrame` by copying `x_world_R_wp` / `x_world_r_wp`, and can apply
small deterministic translation offsets through host memory for synthetic
smokes. This is not the future real-time path; real physics integration should
borrow publisher-owned frames.

The current Go2 Menagerie importer bakes body transforms into
`X_body_geometry` and does not assign `body_index` to visual instances.
Consequently, the current static Go2 registry has no body-bound visual geometry
for `GpuPublishedFrame` perturbation to move. Option A for Go2 would require an
importer change that preserves body indices and frame transforms. Until that
exists, the first dynamic/refit smoke should prefer Option B: a tiny synthetic
body-bound scene.

A8.1 update: the tiny synthetic body-bound scene probe is now in place. It uses
`make_body_bound_triangle_registry(...)`, `make_gpu_pose_frame(...)`, and
`clone_and_perturb_gpu_published_pose_frame(...)` to verify that a perturbed
pose-only `GpuPublishedFrame` changes the device snapshot's world triangle and
AABB. This validates the physics-published-frame shape without entering the
full benchmark loop yet.

A8.2 update: `Go2RenderPipeline.begin_frame(frame_inputs=other_frame)` now
supports dynamic frame inputs. For non-static frames it builds a frame-specific
snapshot, then refits the current BVH when supported or rebuilds with the
session's configured BVH backend otherwise. The returned `Go2RenderFrameContext`
owns the frame-specific snapshot/BVH references and non-NaN `prepare_timing`
fields. Static `begin_frame()` behavior remains unchanged.

A8.3 update: the video loop can now consume an internal `args.video_frame_inputs`
sequence. Each rendered frame passes its corresponding `GpuPublishedFrame` into
`pipeline.begin_frame(...)`, propagates the frame-specific `geometry_mode`, and
keeps camera/ray metadata aligned with the dynamic frame's `frame_id` and
`sim_time`. This remains a lab/test hook rather than a public CLI preset. The
synthetic body-bound GPU smoke verifies that the sync `readback=none` video loop
writes `frame_timing.csv` rows with `geometry_mode=dynamic_rigid`,
non-NaN `snapshot_ms`, non-NaN `accel_refit_ms`, and NaN `accel_rebuild_ms`.

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
    prepare_timing: dict[str, float] = field(default_factory=dict)

    def render(self, request: RenderRequest) -> RenderResult:
        snapshot = self.snapshot if self.snapshot is not None else self.session.snapshot
        bvh = self.bvh if self.bvh is not None else self.session.bvh
        ...
```

The static path can keep returning a context with `snapshot=None` and `bvh=None`.
The dynamic path should populate both fields with frame-specific resources.
`prepare_timing` should carry `snapshot_ms`, `accel_refit_ms`, and/or
`accel_rebuild_ms`, matching the A6 pattern where render timing is owned by
`RenderResult.timing`.

Frame-specific snapshot/BVH resources are render-preparation resources, not
`FrameResult` resources. For async delivery, the device render result may need
to stay alive until readback completion, but the dynamic snapshot/BVH should be
releasable or reusable after render completion unless the concrete executor or
device event contract requires a longer lifetime. A8 should make this lifetime
explicit in the implementation note.

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
