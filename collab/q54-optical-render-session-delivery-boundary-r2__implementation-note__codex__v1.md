# Q54 Optical RenderSession Delivery Boundary R2 Implementation Note

Date: 2026-05-13
Author: Codex
Status: implemented

## Scope

Implemented the small R2 bridge from the delivery boundary plan:

```text
q54-optical-render-session-delivery-boundary-plan__review-request__codex__v1.md
```

This does not make `VideoDeliveryFacade` implement `OpticalDeliveryRuntime`.
That is intentional: the runtime protocol submits `RenderResult`, while the lab
facade submits `RenderedVideoFrame`, which carries lab-only render metadata for
CSV/progress output.

## Implementation

`tools/optical_pipeline_lab/delivery.py` now lets a completed lab frame export
the CPU-safe runtime vocabulary:

```python
DeliveredVideoFrame.to_runtime_delivery_result()
```

The bridge maps:

- `completed_frame_index`
- `host_channels`
- `delivery_timing -> DeliveryResult.delivery`
- `readback_lag_frames -> lag_frames`
- `readback_ring_depth -> ring_depth`
- `readback_ring_block_count -> ring_block_count`

The bridge intentionally does not carry lab-only fields:

- `observed_frame_ms`
- `frame_path`
- `overlap_ratio`

## Boundary Note

`RenderedVideoFrame` is still a lab-local render-to-delivery interface. It is
not a replacement for `optics.render_api.RenderResult`.

That means R2 is a conversion bridge, not a protocol conformance step:

```text
VideoDeliveryFacade.submit(RenderedVideoFrame) -> DeliveredVideoFrame
DeliveredVideoFrame.to_runtime_delivery_result() -> DeliveryResult
```

A future R3 may add a real `create_delivery_runtime(...)` hook, but it should
first decide where the extra lab metadata required by row/progress builders
lives.

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
ruff check optics/render_api.py \
  tools/optical_pipeline_lab/delivery.py \
  tools/optical_pipeline_lab/go2_backend.py \
  tests/unit/optics/test_render_api.py \
  tests/unit/optics/test_optical_pipeline_lab.py
All checks passed!
```
