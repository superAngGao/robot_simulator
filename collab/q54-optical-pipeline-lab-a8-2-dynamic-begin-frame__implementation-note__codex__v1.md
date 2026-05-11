# Q54 A8.2 Dynamic Begin Frame Implementation Note

Date: 2026-05-11
Author: Codex
Status: implemented

## Scope

A8.2 wires dynamic frame inputs into the internal pipeline frame boundary:

```text
Go2RenderPipeline.begin_frame(frame_inputs=other_gpu_frame)
  -> frame-specific snapshot
  -> refit or rebuild acceleration
  -> Go2RenderFrameContext(snapshot=..., bvh=..., prepare_timing=...)
```

This still does not add a public `OpticalCameraStream` API or a full dynamic
video benchmark preset. It only makes the frame-context boundary dynamic-ready.

## Implementation

`Go2RenderSession` now remembers:

```text
bvh_backend
bvh_split_strategy
```

so dynamic rebuilds use the same acceleration backend configuration as setup.

`Go2RenderPipeline.begin_frame(...)` keeps the static path unchanged:

```text
frame_inputs is None or session.gpu_frame:
  return static context with NaN prepare timing
```

For a different `GpuPublishedFrame`, it now:

1. calls `session.cache.snapshot_from_gpu_frame(..., include_aabb=True)`;
2. synchronizes the snapshot ready event and records `snapshot_ms`;
3. if `session.bvh.stats.supports_refit`, calls
   `refit_device_bvh_from_snapshot(...)` and records `accel_refit_ms`;
4. otherwise rebuilds with `cuda_lbvh` or CPU BVH according to
   `session.bvh_backend` and records `accel_rebuild_ms`;
5. returns a `Go2RenderFrameContext` carrying the frame-specific snapshot/BVH
   and `FramePrepareTiming`.

`Go2RenderFrameContext.render(...)` already accepts frame-specific snapshot/BVH
from the A8 timing-ownership foundation, so no render signature change was
needed.

## Validation

CPU tests:

- static `begin_frame(frame_inputs=session.gpu_frame)` still returns the static
  context with NaN prepare timing;
- dynamic begin-frame refit path uses the frame-specific snapshot and returns
  non-NaN `snapshot_ms` / `accel_refit_ms`;
- dynamic begin-frame rebuild path uses the session BVH backend/split strategy
  and returns non-NaN `accel_rebuild_ms`.

GPU smoke:

```text
tests/gpu/test_optical_gpu_runtime.py::
  test_optical_lab_dynamic_begin_frame_populates_prepare_timing_with_synthetic_frame
```

The smoke uses the synthetic body-bound scene from A8.1 and verifies:

- dynamic `begin_frame(frame_inputs=moved_frame)` creates a new snapshot;
- the refit-capable CPU BVH path populates `snapshot_ms` and `accel_refit_ms`;
- `accel_rebuild_ms` remains NaN in the refit path;
- the frame-specific snapshot AABB reflects the moved body pose.

Commands:

```text
conda run -n env_tilelang_20260119 python -m pytest \
  tests/unit/optics/test_optical_pipeline_lab.py -q

45 passed

conda run -n env_tilelang_20260119 python -m pytest \
  tests/gpu/test_optical_gpu_runtime.py::test_optical_lab_synthetic_body_bound_frame_moves_snapshot_geometry \
  tests/gpu/test_optical_gpu_runtime.py::test_optical_lab_dynamic_begin_frame_populates_prepare_timing_with_synthetic_frame -q

2 passed

conda run -n env_tilelang_20260119 python -m ruff check \
  tools/optical_pipeline_lab/go2_backend.py \
  tests/unit/optics/test_optical_pipeline_lab.py \
  tests/gpu/test_optical_gpu_runtime.py

All checks passed.
```

## Next Step

Add a small synthetic dynamic lab preset/backend or runner path that calls
`begin_frame(frame_inputs=frame_n)` inside the video loop and writes non-NaN
`snapshot_ms` / `accel_refit_ms` rows to `frame_timing.csv`.
