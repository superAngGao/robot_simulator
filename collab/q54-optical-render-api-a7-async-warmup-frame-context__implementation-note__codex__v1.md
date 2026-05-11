# Q54 Optical Render API A7 Async Warmup Frame Context Implementation Note

Date: 2026-05-11
Author: Codex
Status: implementation-note

## Scope

Routed the ordered async readback warmup render through the same
`Go2RenderPipeline.begin_frame(...).render(...)` path used by normal video
frames.

This removes the last lab-level call site that directly invoked:

```python
Go2RenderSession.execute_request(..., render_profile=...)
```

## Changes

`tools/optical_pipeline_lab/go2_backend.py`

- `_prepare_torch_async_readback_ring(...)` now accepts `Go2RenderPipeline`
  instead of `Go2RenderSession`.
- Warmup now does:

```python
warmup_frame = pipeline.begin_frame(env_idx=warmup_camera.env_idx)
warmup_result = warmup_frame.render(warmup_request).compute
```

- RGB8 pack and async ring construction remain unchanged after the warmup
  render result is produced.
- The normal async video loop still owns submit/complete timing and CSV rows.

`tests/unit/optics/test_optical_pipeline_lab.py`

- Added a CPU-only test confirming async warmup:
  - calls `pipeline.begin_frame(...)`;
  - sends the warmup `RenderRequest` through frame context `render(...)`;
  - preserves render-profile/traversal-counter intent;
  - passes expected channels and ring depth into
    `TorchAsyncReadbackRing.from_warmup_result(...)`.

## Validation

CPU:

```text
ruff check tools/optical_pipeline_lab/go2_backend.py tests/unit/optics/test_optical_pipeline_lab.py
  All checks passed

ruff format --check tools/optical_pipeline_lab/go2_backend.py tests/unit/optics/test_optical_pipeline_lab.py
  2 files already formatted

conda run -n env_tilelang_20260119 python -m pytest tests/unit/optics/test_render_api.py tests/unit/optics/test_optical_pipeline_lab.py -q
  47 passed
```

GPU smoke:

```bash
conda run -n env_tilelang_20260119 python -m tools.optical_pipeline_lab run \
  --preset go2_video_ordered_static \
  --out out/optical_pipeline_lab/a7_async_warmup_frame_context_smoke_gpu1 \
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
readback_payload: rgb8 / rgb8
delivery_policy: torch_async / torch_async
readback_mode: torch_async_rgb8 / torch_async_rgb8
readback_ring_depth: 2 / 2
render_execute_ms: 5.491 / 6.058
render_overhead_ms: -0.208 / -0.263
pack_rgb8_ms: 0.179 / 0.174
readback_host_ms: 0.168 / 0.165
shadow_traversal_ray_count: 3,873,731 / 3,773,378
```

## Remaining Boundary

`Go2RenderSession.execute_request(...)` still accepts `render_profile`, but it
is now only used by `Go2RenderFrameContext.render(...)`. That makes it an
internal lower-level helper rather than a lab hot-loop side channel.

Do not remove the parameter until the executor/session boundary is redesigned;
the frame context still needs a way to pass the profile buffer down to the
executor.
