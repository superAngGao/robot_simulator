# Q54 Optical Render API A6 Render Timing Internalization Implementation Note

Date: 2026-05-11
Author: Codex
Status: implementation-note

## Scope

Internalized Go2 render-profile timing into `RenderResult.timing`.

This removes the A5-side `render_profile` mutable-list side channel from the
`RenderFrameContext.render(...)` call shape. Delivery/readback loops and CSV
schema remain unchanged.

## Changes

`tools/optical_pipeline_lab/go2_backend.py`

- `Go2RenderFrameContext.render(...)` now matches the protocol shape:

```python
render(request: RenderRequest) -> RenderResult
```

- The frame context now owns:
  - creating the render-profile buffer from `request.diagnostics`;
  - calling `Go2RenderSession.execute_request(...)`;
  - synchronizing `compute_result.ready_event`;
  - measuring `render_execute_ms`;
  - converting executor profile tuples into the existing render timing row.
- `Go2RenderFrameContext.render(...)` returns only after
  `compute_result.ready_event` has been synchronized. Callers should treat the
  returned `RenderResult.compute` as render-complete and should not synchronize
  it again for render timing purposes.

- Returned `RenderResult.timing` now includes:

```text
render_execute_ms
render_raygen_camera_params_ms
render_raygen_buffer_alloc_ms
render_raygen_kernel_ms
render_first_hit_..._ms
render_shade_..._ms
render_overhead_ms
```

- `_render_video_frame(...)` now consumes:

```python
render_result = frame_context.render(render_request)
result = render_result.compute
render_execute_ms = render_result.timing["render_execute_ms"]
render_profile_row = _render_profile_row_from_timing(render_result.timing)
```

The lab frame CSV still writes the same columns and values as before.

## Why This Matters

A5 introduced `RenderFrameContext`, whose protocol method is
`render(request) -> RenderResult`. The concrete Go2 implementation temporarily
accepted `render_profile=...`, which leaked lab timing mechanics through the
protocol boundary.

A6 closes that gap:

```text
before:
  frame.render(request, render_profile=external_mutable_list)

after:
  frame.render(request) -> RenderResult(compute=..., timing=...)
```

This makes render-side timing ownership match the API shape and makes future
delivery facade work less tangled.

## Compatibility

- `render_execute_ms` still measures executor dispatch plus waiting for
  `ready_event`.
- `render_profile` phases still come from the existing executor profile tuple
  list.
- `render_overhead_ms` still uses the existing
  `render_execute_ms - matched_profile_phase_total` estimate.
- `render_overhead_ms` is a derived estimate, not a directly measured phase.
  It may be negative because `render_execute_ms` is a CPU wall-clock interval
  while some profile phases are GPU/event-backed timings. Negative values should
  be preserved for diagnostics rather than clamped.
- Traversal counters are still carried through device result channels and
  readback; they are not moved into `RenderResult.diagnostics` in A6.
- Sync and async delivery loops remain separate.
- `_prepare_torch_async_readback_ring(...)` still uses the lower-level
  `Go2RenderSession.execute_request(..., render_profile=...)` warmup path. This
  is the remaining render-profile side-channel and should be an A7 cleanup
  target; route warmup through `pipeline.begin_frame(...).render(...)` before
  removing `render_profile` from `execute_request(...)`.

## Validation

CPU:

```text
ruff check tools/optical_pipeline_lab/go2_backend.py tests/unit/optics/test_optical_pipeline_lab.py
  All checks passed

ruff format --check tools/optical_pipeline_lab/go2_backend.py tests/unit/optics/test_optical_pipeline_lab.py
  2 files already formatted

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

This was a functional smoke, not a performance baseline. It confirms
`render_overhead_ms` and the render profile phases still materialize in
`frame_timing.csv` after timing ownership moved into `RenderResult.timing`.
