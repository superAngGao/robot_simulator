# Q54 Optical Render API A6 Render Timing Internalization Review Request

Date: 2026-05-11
Author: Codex
Status: ready-for-review

## Context

A5 introduced the internal `OpticalRenderPipeline` / `RenderFrameContext`
protocol shape and a concrete Go2 pipeline wrapper. Claude's A5 review passed
but identified the main leak:

```text
Go2RenderFrameContext.render(request, render_profile=...)
```

The protocol shape is `render(request) -> RenderResult`, but the concrete Go2
implementation still accepted a mutable list side-channel for timing. A6 removes
that side-channel.

## Files Changed

```text
tools/optical_pipeline_lab/go2_backend.py
tests/unit/optics/test_optical_pipeline_lab.py
collab/q54-optical-render-api-a6-render-timing-internalization__implementation-note__codex__v1.md
```

No commit has been made yet.

## Implementation Summary

### 1. Frame Context Owns Render Timing

`Go2RenderFrameContext.render(...)` now matches the protocol method shape:

```python
def render(self, request: RenderRequest) -> RenderResult:
    ...
```

It now owns:

- creating the executor `render_profile` buffer from
  `request.diagnostics.profile_timing` / `traversal_counters`;
- calling `Go2RenderSession.execute_request(...)`;
- waiting on `compute_result.ready_event`;
- measuring `render_execute_ms`;
- converting executor profile tuples into render timing fields.

### 2. RenderResult.timing Is Populated

The returned `RenderResult` now has:

```text
render_execute_ms
render_raygen_*_ms
render_first_hit_*_ms
render_shade_*_ms
render_overhead_ms
```

`RenderResult.compute` still carries the original `OpticalComputeResult`.

### 3. Video Frame Path Consumes RenderResult.timing

`_render_video_frame(...)` now calls:

```python
render_result = frame_context.render(render_request)
result = render_result.compute
render_execute_ms = render_result.timing["render_execute_ms"]
render_profile_row = _render_profile_row_from_timing(render_result.timing)
```

It no longer creates or passes a mutable `render_profile` list.

### 4. CSV Semantics Preserved

Frame CSV columns and meanings are unchanged:

- `render_execute_ms` still includes executor dispatch + ready-event wait;
- render profile phase columns still come from executor profile tuples;
- `render_overhead_ms` still uses the existing estimate;
- `pack_rgb8_ms`, readback timing, async overlap, and traversal counters remain
  in their prior paths.

## Validation

CPU:

```text
ruff check optics/render_api.py tools/optical_pipeline_lab/go2_backend.py tests/unit/optics/test_render_api.py tests/unit/optics/test_optical_pipeline_lab.py
  All checks passed

ruff format --check optics/render_api.py tools/optical_pipeline_lab/go2_backend.py tests/unit/optics/test_render_api.py tests/unit/optics/test_optical_pipeline_lab.py
  4 files already formatted

conda run -n env_tilelang_20260119 python -m pytest tests/unit/optics/test_render_api.py tests/unit/optics/test_optical_pipeline_lab.py -q
  46 passed
```

GPU smoke:

```bash
conda run -n env_tilelang_20260119 python -m tools.optical_pipeline_lab run \
  --preset go2_video_ordered_static \
  --out out/optical_pipeline_lab/a6_render_timing_internalization_smoke_gpu1 \
  --device cuda:1 \
  --width 1920 \
  --height 1080 \
  --frames 2 \
  --warmup-renders 5 \
  --progress-every 1 \
  --readback rgb8 \
  --video-readback-delivery torch_async \
  --video-readback-ring-depth 2 \
  --render-profile
```

CSV sanity:

```text
rows: 2
render_execute_ms:          13.114 / 17.131
render_raygen_kernel_ms:     3.339 / 0.084
render_first_hit_kernel_ms:  2.183 / 2.117
render_shade_kernel_ms:      2.567 / 3.292
render_overhead_ms:         -4.719 / -8.408
pack_rgb8_ms:                2.383 / 2.225
readback_host_ms:            0.168 / 0.169
shadow_traversal_ray_count:  3,873,731 / 3,773,378
```

This smoke is functional, not a performance baseline.

## Review Questions

1. Is it correct for `Go2RenderFrameContext.render(request)` to synchronize
   `compute_result.ready_event` internally so `render_execute_ms` remains owned
   by the render call?

2. Is `RenderResult.timing` using existing CSV-style keys
   (`render_execute_ms`, `render_shade_kernel_ms`, etc.) acceptable, or should
   runtime timing use unprefixed/internal names and let the lab translate?

3. Should traversal counters remain outside `RenderResult.diagnostics` for now?
   Current behavior keeps them as device channels staged during readback.

4. Does `_render_profile_row_from_timing(...)` correctly preserve the existing
   lab CSV shape, including `render_overhead_ms`?

5. Is A6 enough to close the A5 protocol mismatch, or should
   `Go2RenderSession.execute_request(...)` also stop accepting `render_profile`
   in this slice?

## Proposed Verdict Criteria

PASS if:

- `frame.render(request)` is now considered a clean protocol-conforming call;
- `RenderResult.timing` is an acceptable home for render profile phases;
- CSV compatibility and GPU smoke are acceptable.

NEEDS WORK if:

- render timing should not synchronize inside frame context;
- timing key names should be changed before committing;
- traversal counters should be moved into `RenderResult.diagnostics` now.
