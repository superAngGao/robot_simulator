# Q54 Optical RenderSession Delivery Boundary R1 Implementation Note

Date: 2026-05-13
Author: Codex
Status: implemented

## Scope

Implemented R1 from:

```text
q54-optical-render-session-delivery-boundary-plan__review-request__codex__v1.md
```

This is a CPU-safe contract/type alignment step only. It does not move the lab
delivery implementation out of `tools/optical_pipeline_lab`, and it does not add
a public camera stream API.

## Implementation

`optics/render_api.py` now includes:

- `OpticalDeliveryRuntime` protocol with explicit
  `submit(...)`, `complete_available(...)`, and `flush()` methods;
- `complete_available(latest_rendered_frame_index=...)`, matching the lab
  facade's ring-depth-1 completion path;
- `OpticalRenderPipeline.create_delivery_runtime(...)`;
- `OpticalRenderPipeline.deliver(...)` retained as sync-only convenience;
- `DeliveryResult.completed_frame_index`;
- `DeliveryResult.delivery: DeliveryTimingSummary`;
- `DeliveryResult.frame_index` as a transition alias for
  `completed_frame_index`.

`tools/optical_pipeline_lab/go2_backend.py` adds a transitional
`Go2RenderPipeline.create_delivery_runtime(...)` method that raises
`NotImplementedError`, preserving the existing structural protocol test while
delivery remains owned by the lab benchmark loops.

## Validation

CPU/unit:

```text
python -m pytest -q \
  tests/unit/optics/test_render_api.py \
  tests/unit/optics/test_optical_pipeline_lab.py
67 passed
```

Lint:

```text
ruff check optics/render_api.py \
  tools/optical_pipeline_lab/go2_backend.py \
  tests/unit/optics/test_render_api.py \
  tests/unit/optics/test_optical_pipeline_lab.py
All checks passed!
```

## Notes For R2

When the lab facade bridges `DeliveredVideoFrame` to runtime
`DeliveryResult`, do not carry lab-only fields into `DeliveryResult`:

- `observed_frame_ms` remains frame-summary or CSV-row data;
- `frame_path` remains writer/consumer-adapter data;
- `overlap_ratio` remains lab analysis data.
