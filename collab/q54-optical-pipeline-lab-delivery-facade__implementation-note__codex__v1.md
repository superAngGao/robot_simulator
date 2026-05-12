# Q54 Optical Pipeline Lab Delivery Facade Implementation Note

Date: 2026-05-12
Author: Codex
Status: implemented

## Scope

Implemented the lab-local delivery facade from the accepted follow-up plan:

```text
q54-optical-pipeline-lab-delivery-facade-plan__review-followup__codex__v1.md
```

This is still `tools/optical_pipeline_lab` infrastructure. It does not add a
production `OpticalRenderSession` or public camera API.

## Implementation

Added:

```text
tools/optical_pipeline_lab/delivery.py
```

The new module owns:

- `RenderedVideoFrame`
- `DeliveredVideoFrame`
- `VideoDeliveryFacade`
- `VideoFrameTimingRowBuilder`
- `video_delivery_request(...)`
- selected readback/channel/image/write helpers

`go2_backend.py` now renders a frame, submits it to the facade, records any
completed delivery rows, and drains at the end of the loop.

## Review Follow-up Decisions Applied

### RGB8 Pack Is Delivery-Owned

`_render_video_frame(...)` no longer packs RGB8 and `RenderedVideoFrame` no
longer has `pack_rgb8_ms`.

The facade packs RGB8 when `DeliveryRequest.payload == RGB8` and records timing
in `DeliveryTimingSummary.pack_rgb8_ms`.

### Explicit Async Methods

The facade uses:

```python
submit(...)
complete_available()
flush()
```

rather than returning an iterable from `submit(...)`.

The current torch async path preserves ordered/no-drop semantics. Ring depth 1
and ring depth 2 metadata paths are unit-covered.

### Completed Frame Identity

`DeliveredVideoFrame.completed_frame_index` is explicit. Async row construction
uses the completed frame identity, while lag/ring metadata remains delivery
state.

### CSV Compatibility

CSV field names are unchanged.

CSV `delivery_policy` values remain:

```text
sync
torch_async
```

The internal runtime enum still uses `TORCH_ASYNC_ORDERED`.

## Validation

CPU/unit:

```text
python -m pytest -q tests/unit/optics/test_optical_pipeline_lab.py
52 passed
```

Lint:

```text
ruff check tools/optical_pipeline_lab \
  tests/unit/optics/test_optical_pipeline_lab.py \
  tests/gpu/test_optical_gpu_runtime.py
All checks passed!
```

GPU-selected smoke in this environment:

```text
python -m pytest -q tests/gpu/test_optical_gpu_runtime.py \
  -k "dynamic_smoke_preset or dynamic_video_loop"
2 skipped, 29 deselected
```

The GPU smokes were skipped by the existing `Warp or CUDA not available` guard.

## Notes

The facade is intentionally still lab-local. Moving this into `optics/` should
wait for another RenderSession/workspace review.
