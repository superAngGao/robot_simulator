# Q54 Optical Render API A3 Delivery Adapter Implementation Note

Date: 2026-05-11
Author: Codex
Status: implementation-note

## Scope

Implemented the minimal A3 delivery boundary adapter for the Go2 optical lab
backend.

This is intentionally not a full delivery-loop rewrite. The existing sync and
`torch_async` benchmark paths remain in place, but both now construct an
internal `DeliveryRequest` before staging/readback/writing.

## Changes

`tools/optical_pipeline_lab/go2_backend.py`

- Added `_video_delivery_request(...)` to translate lab CLI strings into runtime
  delivery API types:
  - `readback=none` -> `ReadbackPayload.NONE` + `DeliveryPolicy.DEVICE_ONLY`;
  - `readback=rgb/rgb8/full`, sync -> `DeliveryPolicy.SYNC_HOST`;
  - `readback=rgb/rgb8`, `torch_async` -> `DeliveryPolicy.TORCH_ASYNC_ORDERED`;
  - `write_frames=True` -> `WritePolicy.PNG_SEQUENCE`.
- Sync video path now uses `delivery_request.payload` and
  `delivery_request.write_policy` for host staging and optional PNG writes.
- Async video path now passes `DeliveryRequest` through warmup ring creation and
  frame completion:
  - ring depth comes from `delivery_request.ring_depth`;
  - RGB8 warmup packing checks `delivery_request.payload`;
  - async CSV rows still preserve the established lab-facing values:
    `delivery_policy=torch_async`, `readback_mode=torch_async_rgb8`.

`tests/unit/optics/test_optical_pipeline_lab.py`

- Added coverage for lab-to-runtime delivery mapping:
  - sync `none` becomes device-only;
  - async `rgb8` becomes ordered Torch async with requested ring depth;
  - sync `full` remains valid;
  - async `full` is rejected by `DeliveryRequest` compatibility checks.

## Behavior Preserved

The benchmark CSV schema and lab-facing values are intentionally unchanged.

For async RGB8:

```text
readback_payload: rgb8
delivery_policy: torch_async
readback_mode: torch_async_rgb8
readback_ring_depth: 2
```

`render_execute_ms`, `pack_rgb8_ms`, render-profile columns, traversal counters,
readback timing, and overlap columns keep their existing meanings.

## Validation

CPU:

```text
ruff check tools/optical_pipeline_lab/go2_backend.py tests/unit/optics/test_optical_pipeline_lab.py optics/render_api.py tests/unit/optics/test_render_api.py
  All checks passed

ruff format --check tools/optical_pipeline_lab/go2_backend.py tests/unit/optics/test_optical_pipeline_lab.py optics/render_api.py tests/unit/optics/test_render_api.py
  4 files already formatted

conda run -n env_tilelang_20260119 python -m pytest tests/unit/optics/test_render_api.py tests/unit/optics/test_optical_pipeline_lab.py -q
  40 passed
```

GPU smoke:

```bash
conda run -n env_tilelang_20260119 python -m tools.optical_pipeline_lab run \
  --preset go2_video_ordered_static \
  --out out/optical_pipeline_lab/a3_delivery_adapter_smoke_gpu1 \
  --device cuda:1 \
  --width 1920 \
  --height 1080 \
  --frames 3 \
  --warmup-renders 5 \
  --progress-every 1 \
  --readback rgb8 \
  --video-readback-delivery torch_async \
  --video-readback-ring-depth 2 \
  --render-profile
```

Result:

```text
rows: 3
frame_total_mean: 7.956 ms
render_execute_mean: 6.552 ms
pack_rgb8_mean: 0.139 ms
readback_host_mean: 2.399 ms
overflow: 0 primary / 0 shadow
frame_timing_csv: out/optical_pipeline_lab/a3_delivery_adapter_smoke_gpu1/frame_timing.csv
```

## Follow-Up

The delivery boundary is now present, but delivery execution is still split
between `_run_video_benchmark` and `_run_video_benchmark_torch_async`.

Suggested next slices:

1. A3.1: introduce a small delivered-frame helper/result object only if it
   removes duplication without changing timing semantics.
2. A3.2: move `render_profile` allocation into the session/request execution
   path, after another GPU smoke confirms A3 remains stable.
3. A4: make traversal counter intent explicit instead of coupling it to
   `profile_timing`, when diagnostics become part of the generic pipeline.
4. If `TorchAsyncReadbackRing` later gains stronger `ring_depth` constraints
   such as an upper bound, keep those constraints synchronized with
   `DeliveryRequest`. A3 currently relies on both layers sharing the same
   `ring_depth > 0` rule.
5. Revisit CSV `delivery_policy` vocabulary in A4. A3 intentionally preserves
   the historical lab value `torch_async`; switching rows to runtime enum values
   such as `torch_async_ordered` should be treated as a separate compatibility
   decision.
