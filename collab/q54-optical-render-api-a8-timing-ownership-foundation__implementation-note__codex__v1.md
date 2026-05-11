# Q54 A8 Timing Ownership Foundation Implementation Note

Date: 2026-05-11
Author: Codex
Status: implemented

## Scope

This slice implements the accepted FrameResult/timing ownership foundation
without enabling dynamic/refit scenes yet.

It is intentionally smaller than full A8:

- no `GpuPublishedFrame` clone/perturb helper yet;
- no dynamic `begin_frame(frame_inputs=other_frame)` implementation yet;
- no delivery facade yet;
- static Go2 lab behavior and CSV schema remain compatible.

## Code Changes

### Runtime Timing Blocks

Added dependency-light dataclasses in `optics.render_api`:

```text
FramePrepareTiming:
  snapshot_ms
  accel_refit_ms
  accel_rebuild_ms

RenderTimingSummary:
  execute_ms
  profile_sum_ms
  overhead_ms
  phases

DeliveryTimingSummary:
  pack_rgb8_ms
  readback_submit_ms
  readback_wait_ms
  readback_host_ms
  image_build_ms
  encode_write_ms

FrameTimingSummary:
  work_sum_ms
  observed_frame_ms
  critical_path_ms
  instant_fps

FrameResult:
  frame_id
  sim_time
  env_idx
  prepare/render/delivery/summary blocks
  completed_frame_index
```

`FrameResult` is a lightweight completed-frame observation summary. It does not
own `GpuPublishedFrame`, device snapshots, or BVH resources.

Each timing block has a flat mapping helper so the current CSV vocabulary can
stay stable while internal ownership becomes typed.

`FrameTimingSummary.to_flat_mapping()` intentionally exports only the existing
CSV vocabulary (`frame_total_ms` and `instant_fps`) for this slice.
`work_sum_ms` and `critical_path_ms` remain internal summary fields until the
lab CSV schema explicitly grows those columns.

### RenderResult Summary

`RenderResult` now optionally carries:

```text
render_timing: RenderTimingSummary | None
```

`Go2RenderFrameContext.render(...)` fills this summary from the same flat timing
mapping currently used by the lab CSV. Existing `RenderResult.timing` remains in
place for compatibility.

### Go2 Frame Context Preparation Timing

`Go2RenderFrameContext` now carries:

```text
snapshot: object | None
bvh: object | None
prepare_timing: Mapping[str, float]
```

Static `begin_frame(...)` returns NaN prepare timing via `FramePrepareTiming`.
The sync and async video row builders now read `snapshot_ms`,
`accel_refit_ms`, and `accel_rebuild_ms` from the rendered frame's
`prepare_timing` rather than hardcoding NaN at the row construction site.

`Go2RenderSession.execute_request(...)` accepts optional `snapshot` and `bvh`
arguments so a later dynamic frame context can render against frame-specific
resources without changing the render call shape.

## Compatibility

- `frame_timing.csv` schema is unchanged.
- Static Go2 behavior is unchanged: prepare timing columns remain NaN.
- `RenderResult.timing` remains available for existing call sites.
- `FrameResult` is introduced as an internal target shape but is not yet wired
  into the hot loop.

## Validation

```text
python -m pytest tests/unit/optics/test_render_api.py \
  tests/unit/optics/test_optical_pipeline_lab.py -q

48 passed

conda run -n env_tilelang_20260119 python -m pytest \
  tests/unit/optics/test_render_api.py \
  tests/unit/optics/test_optical_pipeline_lab.py -q

48 passed

conda run -n env_tilelang_20260119 python -m ruff check \
  optics/render_api.py \
  tools/optical_pipeline_lab/go2_backend.py \
  tests/unit/optics/test_render_api.py \
  tests/unit/optics/test_optical_pipeline_lab.py

All checks passed.
```

## Next Step

A8.0 should probe whether `GpuPublishedFrame` can be cloned and perturbed
without mutating the static baseline frame. Only after that should
`begin_frame(frame_inputs=other_frame)` build a frame-specific snapshot/BVH and
populate non-NaN `FramePrepareTiming`.
