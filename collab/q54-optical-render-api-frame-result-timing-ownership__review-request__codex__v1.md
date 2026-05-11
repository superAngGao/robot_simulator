# Q54 FrameResult + Timing Ownership Decision Review Request

Date: 2026-05-11
Author: Codex
Status: review-request

## Context

After A5-A7, the Go2 lab path has an internal pipeline/frame-context shape:

```text
pipeline.begin_frame(...)
  -> RenderFrameContext.render(...)
  -> RenderResult
```

A6 moved render timing into `RenderResult.timing`, but the full per-frame CSV
still mixes prepare, render, delivery, async, and diagnostics fields in one flat
schema. That is useful for analysis, but the internal ownership model is now
too blurry for A8 dynamic/refit work.

## Proposed Decision

Keep the CSV flat, but make internal ownership explicit:

```text
RenderFrameContext:
  frame-scoped lifecycle/input handle
  borrows frame inputs
  may hold frame-specific snapshot/BVH while rendering

RenderResult:
  render-owned result
  owns OpticalComputeResult plus render_execute/profile/overhead timing
  does not own RGB8 pack/readback/write timing

DeliveryResult:
  delivery-owned result
  owns pack/readback/write timing plus lag/drop/backpressure metadata

FrameResult:
  frame-bound observation summary
  identifies frame_id, sim_time, env_idx
  aggregates prepare + render + delivery summaries
  does not own or mutate GpuPublishedFrame, snapshot, or BVH
```

Timing ownership:

```text
prepare timing:
  snapshot_ms, accel_refit_ms, accel_rebuild_ms
  owned by begin_frame / frame-context preparation

render timing:
  render_execute_ms, render profile phases, render_overhead_ms
  owned by RenderResult

delivery timing:
  pack_rgb8_ms, readback_submit_ms, readback_wait_ms, readback_host_ms,
  image_build_ms, encode_write_ms
  owned by DeliveryResult

frame summary:
  work_sum_ms, observed_frame_ms, critical_path_ms, instant_fps
  owned by FrameResult
```

The important semantic point is that `FrameResult` should bind to a frame
identity, but should not become a resource owner. Physics publishes or lends a
frame; the optical pipeline borrows it during `begin_frame`; the output summary
records what happened for that frame.

## Review Questions

1. Do you agree that `FrameResult` should be frame-bound but lightweight, without
   owning `GpuPublishedFrame`, snapshot, or BVH resources?
2. Is it correct to keep `RenderResult` scoped to render-owned timing only,
   leaving RGB8 pack/readback/write to `DeliveryResult`?
3. Should `snapshot_ms` and `accel_refit_ms`/`accel_rebuild_ms` be owned by
   `RenderFrameContext`/`begin_frame`, rather than `RenderResult`?
4. Do you agree that `frame_timing.csv` should remain a flat export format while
   the internal API moves toward typed timing blocks?
5. For async delivery, do the proposed `work_sum_ms`, `observed_frame_ms`, and
   `critical_path_ms` summary names separate total work from overlapped wall-time
   clearly enough?

## Proposed Next Step

Do not implement the full `FrameResult` yet. Use this as the ownership target
for A8 dynamic/refit planning, then introduce the typed timing/result blocks only
where they remove ambiguity from the implementation.
