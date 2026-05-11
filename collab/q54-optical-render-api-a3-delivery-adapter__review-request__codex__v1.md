# Q54 Optical Render API A3 Delivery Adapter Review Request

Date: 2026-05-11
Author: Codex
Status: ready-for-review

## Context

A1/A2 introduced CPU-safe render/delivery contracts in `optics/render_api.py`
and a thin `Go2RenderSession.execute_request(RenderRequest, ...)` adapter.
Claude's last review green-lit A1/A2 and called out A3 as the next cleanup:
bring `DeliveryRequest` into the lab path without changing validated
benchmark/CSV behavior.

This patch implements that minimal A3 slice. It is intentionally not a full
delivery-loop refactor.

## Files Changed

```text
tools/optical_pipeline_lab/go2_backend.py
tests/unit/optics/test_optical_pipeline_lab.py
collab/q54-optical-render-api-a3-delivery-adapter__implementation-note__codex__v1.md
```

No commit has been made yet.

## Implementation Summary

### 1. Lab CLI -> Runtime DeliveryRequest Adapter

Added:

```python
_video_delivery_request(
    *,
    readback_mode: str,
    delivery_mode: str,
    ring_depth: int,
    write_frames: bool,
) -> DeliveryRequest
```

Mapping:

```text
readback=none
  -> ReadbackPayload.NONE + DeliveryPolicy.DEVICE_ONLY

readback=rgb/rgb8/full, delivery=sync
  -> payload rgb/rgb8/full + DeliveryPolicy.SYNC_HOST

readback=rgb/rgb8, delivery=torch_async
  -> payload rgb/rgb8 + DeliveryPolicy.TORCH_ASYNC_ORDERED

write_frames=True
  -> WritePolicy.PNG_SEQUENCE
```

Invalid combinations are left to `DeliveryRequest.__post_init__`, e.g.
`full + torch_async` raises `ValueError("TORCH_ASYNC_ORDERED delivery requires payload=RGB or RGB8")`.

### 2. Sync Video Path Uses DeliveryRequest

`_run_video_benchmark(...)` now builds a `DeliveryRequest` and uses:

```text
delivery_request.payload.value
delivery_request.write_policy
```

for:

- host staging via `_readback_video_result(...)`;
- optional RGB PNG write;
- `readback_host_ms` NaN behavior for device-only payload.

The row still writes the established lab-facing values:

```text
readback_mode: rgb/rgb8/full/none
write_mode: rgb_png/none
```

### 3. Torch Async Video Path Uses DeliveryRequest

`_run_video_benchmark_torch_async(...)` now builds a `DeliveryRequest` and
passes it through:

```text
_prepare_torch_async_readback_ring(...)
_complete_torch_async_video_readback(...)
```

The request now owns:

- payload used for warmup output profile and RGB8 warmup pack;
- async ring depth;
- write policy.

The row intentionally preserves existing CSV semantics:

```text
delivery_policy: torch_async
readback_mode: torch_async_rgb / torch_async_rgb8
readback_ring_depth: <requested depth>
```

This means the runtime enum value `torch_async_ordered` is internal only for
now; the lab CSV keeps its previous vocabulary.

### 4. Tests

Added `test_video_delivery_request_maps_lab_options_to_runtime_api`.

It covers:

- sync `none` -> `DEVICE_ONLY`;
- async `rgb8` -> `TORCH_ASYNC_ORDERED`, requested ring depth, PNG write policy;
- sync `full` -> `SYNC_HOST`;
- async `full` rejected by runtime delivery validation.

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
readback_payload: rgb8
delivery_policy: torch_async
readback_mode: torch_async_rgb8
readback_ring_depth: 2

frame_total_mean: 7.956 ms
render_execute_mean: 6.552 ms
pack_rgb8_mean: 0.139 ms
readback_host_mean: 2.399 ms
overflow: 0 primary / 0 shadow
```

CSV:

```text
out/optical_pipeline_lab/a3_delivery_adapter_smoke_gpu1/frame_timing.csv
```

## Review Questions

1. Is `_video_delivery_request(...)` the right A3 boundary, or should it live in
   `runner.py` beside `build_menagerie_example_args(...)`?

2. Is it correct that `readback=none` forces internal
   `DeliveryPolicy.DEVICE_ONLY` even when the CLI delivery mode is `"sync"`?
   The existing CSV defaults may still show lab scenario `delivery_policy=sync`
   unless a row overrides it.

3. Should async CSV continue using the historical lab value
   `delivery_policy=torch_async`, or should A3 switch rows to the runtime enum
   value `torch_async_ordered`?

4. Does passing `DeliveryRequest` into `_prepare_torch_async_readback_ring(...)`
   and `_complete_torch_async_video_readback(...)` strike the right balance, or
   should A3 introduce a small delivery context object to avoid threading
   request fields through multiple helpers?

5. Is preserving `_run_video_benchmark` and `_run_video_benchmark_torch_async`
   as separate loops still the right scope for A3? My recommendation is yes:
   merge/factor loops only after another stable GPU baseline, because async
   timing semantics are already validated.

6. Any concern with keeping `DeliveryRequest` internal to `go2_backend.py` for
   now instead of exposing it through `runner.py` / matrix code?

7. Is the current test coverage enough for this slice, or should we add a
   no-GPU smoke around sync row construction with a fake staged result?

## Known Non-Goals

- No public API exported from `optics.__init__`.
- No generic `OpticalRenderPipeline` implementation yet.
- No path-tracing delivery/accumulation handling.
- No change to `render_execute_ms`, `pack_rgb8_ms`, traversal counters, or
  overlap timing semantics.
- No merge of sync and async benchmark loops in this patch.

## Proposed Verdict Criteria

PASS if:

- runtime delivery mapping is considered a reasonable internal A3 boundary;
- CSV compatibility is acceptable;
- no hidden semantic regression in async ring warmup/completion is found.

NEEDS WORK if:

- CSV should switch immediately to runtime enum vocabulary;
- `DeliveryRequest` should be constructed earlier in `runner.py`;
- the async ring helper now has an unclear ownership contract.
