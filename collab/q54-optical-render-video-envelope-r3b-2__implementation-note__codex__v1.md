# Q54 Render Video Envelope R3b.2 Implementation Note

Date: 2026-05-13
Author: Codex
Status: implemented

## Scope

Implemented the accepted R3b.2 helper from:

```text
q54-optical-render-video-envelope-r3b-2-design__review-request__codex__v1.md
```

This is a narrow render-timing envelope cleanup. It does not change
`RenderedVideoFrame.result`, `render_profile_row`, video delivery behavior,
CSV schema, or public API.

## Implementation

`RenderedVideoFrame` now exposes:

```python
render_execute_ms_value()
```

Fallback order:

```text
1. render.render_timing.execute_ms
2. render.timing["render_execute_ms"]
3. stored render_execute_ms
```

The first available source is authoritative. If `render.render_timing` exists
and `execute_ms` is NaN, the helper returns NaN rather than falling through.

Updated read points:

```text
VideoDeliveryFacade._complete_job:
  overlap-ratio render input

VideoFrameTimingRowBuilder.build_row:
  frame_timing.csv render_execute_ms

VideoFrameTimingRowBuilder.progress_line:
  stdout render=...
```

## Deferred

`result` remains the existing required dataclass field.

`render_profile_row` remains a stored flattened dict.

Future property migration should still be done separately:

```text
result -> render.compute
render_execute_ms field -> _render_execute_ms + property
render_profile_row derivation
```

Those should not move in the same patch.

## Validation

CPU/unit:

```text
python -m pytest -q \
  tests/unit/optics/test_render_api.py \
  tests/unit/optics/test_optical_pipeline_lab.py
71 passed
```

Lint:

```text
ruff check tools/optical_pipeline_lab/delivery.py \
  tests/unit/optics/test_optical_pipeline_lab.py \
  optics/render_api.py \
  tests/unit/optics/test_render_api.py
All checks passed!
```
