# Q54 A8.0 Dynamic Frame Probe Implementation Note

Date: 2026-05-11
Author: Codex
Status: implemented

## Scope

A8.0 verifies the published-frame clone/perturb boundary before wiring dynamic
rendering into `Go2RenderPipeline.begin_frame(frame_inputs=...)`.

This slice does not enable the dynamic/refit scenario yet.

## Implementation

Added `tools.optical_pipeline_lab.dynamic_frames` with lab-only helpers:

```text
gpu_pose_shape(frame)
clone_gpu_published_pose_frame(frame, wp_module=...)
clone_and_perturb_gpu_published_pose_frame(frame, wp_module=..., translation_offsets=...)
```

The helpers clone only the pose arrays consumed by `DeviceOpticalSceneCache`:

```text
x_world_R_wp
x_world_r_wp
```

The returned `GpuPublishedFrame` is intentionally pose-only:

```text
q/qdot/v/contact/telemetry refs -> None
slot_meta -> None
ready_event -> None
```

That makes the clone independent from a physics publish-ring slot. This is
appropriate for controlled lab smokes, but it is not the future production path.
Production physics integration should borrow publisher-owned frames.

`clone_and_perturb_gpu_published_pose_frame(...)` applies deterministic
translation offsets keyed by `(env_idx, body_idx)`. It stages translations
through host memory, which is acceptable for a small synthetic smoke but should
not enter a hot path.

## Important Discovery

The current Go2 Menagerie importer bakes body transforms into each visual
instance's `X_body_geometry` and does not set `body_index`. The static Go2 lab
frame also has zero body transforms:

```text
x_world_R_wp shape = (1, 0, 3, 3)
x_world_r_wp shape = (1, 0, 3)
```

Therefore, perturbing a `GpuPublishedFrame` will not move the current Go2 visual
scene. Option A from the A8 plan requires an importer change that preserves
body-bound instances.

Recommended next dynamic smoke:

```text
Option B: tiny synthetic body-bound scene
  one or two body-bound primitives
  pose frame with >= 1 body
  begin_frame(frame_inputs=perturbed_frame)
  snapshot/refit or rebuild timing populated
```

## Validation

CPU tests cover:

- pose shape validation;
- clone produces independent pose arrays;
- clone is pose-only and not tied to source `slot_meta`;
- perturb applies translation offsets without mutating the source frame;
- invalid body index raises.

```text
conda run -n env_tilelang_20260119 python -m pytest \
  tests/unit/optics/test_optical_pipeline_lab.py -q
```

## Next Step

Implement the first dynamic smoke against a small synthetic body-bound scene
rather than the current static Go2 Menagerie import. Go2 can join the dynamic
path later if the importer is changed to preserve `body_index` and per-frame
body transforms.
