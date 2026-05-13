# Q54 Render Video Envelope R3b.1 Implementation Note

Date: 2026-05-13
Author: Codex
Status: implemented

## Scope

Implemented the conservative R3b.1 path from:

```text
q54-optical-render-video-envelope-r3b-plan__review-request__codex__v1.md
```

This is an envelope-shape cleanup only. It does not change video delivery,
`create_delivery_runtime(...)`, CSV output, or public API.

## Implementation

`RenderedVideoFrame` now has an optional runtime render reference:

```python
render: RenderResult | None = None
```

The existing required `result` field remains unchanged. This avoids churn in
unit tests and keeps current lab delivery code stable.

`_render_video_frame(...)` now fills:

```python
render=render_result
```

at the single production construction point.

## Boundary

`RenderedVideoFrame` is still the lab video envelope:

```text
RenderResult-derived compute/timing
+ video-loop metadata
+ CSV/progress metadata
```

It is not a replacement for `RenderResult`, and `RenderResult` still does not
carry camera, loop frame index, or lab CSV state.

## Deferred

Do not make `result` a compatibility property yet. A later slice can do that
after all call sites are comfortable with `render`.

When `render_execute_ms` eventually becomes derived, use explicit fallback:

```python
if self.render is not None and self.render.render_timing is not None:
    return self.render.render_timing.execute_ms
if self.render is not None:
    return float(self.render.timing.get("render_execute_ms", float("nan")))
return self._render_execute_ms
```

Do not rely on `or`, because `None` and `NaN` carry different meanings.

## Validation

CPU/unit:

```text
python -m pytest -q \
  tests/unit/optics/test_render_api.py \
  tests/unit/optics/test_optical_pipeline_lab.py
68 passed
```

Lint:

```text
ruff check tools/optical_pipeline_lab/delivery.py \
  tools/optical_pipeline_lab/go2_backend.py \
  tests/unit/optics/test_optical_pipeline_lab.py \
  optics/render_api.py \
  tests/unit/optics/test_render_api.py
All checks passed!
```
