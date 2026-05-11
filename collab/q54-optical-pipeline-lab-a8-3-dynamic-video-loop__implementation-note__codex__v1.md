# Q54 Optical Pipeline Lab A8.3 Dynamic Video Loop Implementation Note

Date: 2026-05-11
Author: Codex
Status: implemented

## Scope

A8.3 wires dynamic frame inputs into the existing Go2 lab video loop without
adding a public CLI preset yet.

The new path is intentionally small:

```text
args.video_frame_inputs[frame_index]
  -> _render_video_frame(...)
  -> pipeline.begin_frame(frame_inputs=...)
  -> frame-specific snapshot/refit timing
  -> normal render/delivery row construction
```

This proves the hot loop can carry per-frame `GpuPublishedFrame` inputs after
A8.2, while keeping the production-facing command surface unchanged.

## Implementation

- `_render_video_frame(...)` now reads an optional internal
  `args.video_frame_inputs` sequence.
- If a dynamic frame input is present, camera and CPU-ray metadata are rebound
  to the frame input's `frame_id` and `sim_time` before constructing
  `RenderRequest`.
- `_RenderedVideoFrame` and `_AsyncVideoReadbackJob` now carry
  `geometry_mode`, so sync and async CSV rows no longer hardcode `static`.
- `geometry_mode` defaults to `dynamic_rigid` whenever a frame input is present,
  unless an internal `args.video_geometry_mode` override is supplied.
- `prepare_timing` continues to flow from `Go2RenderFrameContext` into sync and
  async row construction.

## Validation

CPU/unit:

```text
test_render_video_frame_passes_dynamic_frame_inputs
test_sync_video_readback_none_row_does_not_stage
```

GPU smoke:

```text
test_optical_lab_dynamic_video_loop_writes_prepare_timing_csv
```

The GPU smoke uses the synthetic body-bound triangle scene from A8.1, two
pose-only `GpuPublishedFrame` objects, and the CPU BVH refit path. It verifies:

```text
geometry_mode == dynamic_rigid
snapshot_ms >= 0
accel_refit_ms >= 0
accel_rebuild_ms is NaN
readback_host_ms is NaN for readback=none
frame_timing.csv is written
```

Commands run:

```text
conda run -n env_tilelang_20260119 python -m pytest tests/unit/optics/test_optical_pipeline_lab.py -q
conda run -n env_tilelang_20260119 python -m pytest tests/gpu/test_optical_gpu_runtime.py::test_optical_lab_synthetic_body_bound_frame_moves_snapshot_geometry tests/gpu/test_optical_gpu_runtime.py::test_optical_lab_dynamic_begin_frame_populates_prepare_timing_with_synthetic_frame tests/gpu/test_optical_gpu_runtime.py::test_optical_lab_dynamic_video_loop_writes_prepare_timing_csv -q
conda run -n env_tilelang_20260119 python -m ruff check tools/optical_pipeline_lab/go2_backend.py tests/unit/optics/test_optical_pipeline_lab.py tests/gpu/test_optical_gpu_runtime.py
```

## Notes

This is still not the final dynamic benchmark scenario. It is a loop-integration
proof that keeps dynamic frames behind internal test hooks. The next production
step is to decide how a real lab scenario or future physics bridge supplies the
per-frame published frames and whether the first public dynamic preset should
use RGB8 async delivery or a smaller `readback=none` functional smoke.
