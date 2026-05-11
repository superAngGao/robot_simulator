# Q54 Optical Render API A7 Async Warmup Frame Context Review Request

Date: 2026-05-11
Author: Codex
Status: ready-for-review

## Context

A6 internalized normal-frame render timing into `RenderResult.timing`.
The remaining lab-level bypass was async readback ring warmup:

```python
session.execute_request(warmup_request, render_profile=...)
```

A7 routes that warmup render through the same pipeline/frame context path as
normal video frames.

## Files Changed

```text
tools/optical_pipeline_lab/go2_backend.py
tests/unit/optics/test_optical_pipeline_lab.py
collab/q54-optical-render-api-a7-async-warmup-frame-context__implementation-note__codex__v1.md
collab/q54-optical-pipeline-lab-a8-complex-dynamic-scene-plan__codex__v1.md
```

No commit has been made yet.

## Implementation Summary

`_prepare_torch_async_readback_ring(...)` now accepts `Go2RenderPipeline`:

```python
warmup_frame = pipeline.begin_frame(env_idx=warmup_camera.env_idx)
warmup_result = warmup_frame.render(warmup_request).compute
```

RGB8 pack and ring construction remain unchanged after the warmup render result
is produced.

Added a CPU-only test that verifies:

- warmup calls `pipeline.begin_frame(...)`;
- warmup request is rendered by the frame context;
- traversal-counter intent still follows render diagnostics;
- expected channels/ring depth are passed to
  `TorchAsyncReadbackRing.from_warmup_result(...)`.

Also added an A8 draft plan for a future complex dynamic/refit smoke scene,
because static Go2 is enough for A7 but not enough for physics-style
`begin_frame(frame_inputs)` semantics.

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

## Review Questions

1. Is it correct for async ring warmup to route through
   `pipeline.begin_frame(...).render(...)` rather than directly calling
   `session.execute_request(...)`?
2. Is it acceptable that `session.pack_rgb8(...)` remains outside frame context
   in `_prepare_torch_async_readback_ring(...)`?
3. Is the CPU-only monkeypatch test enough for this narrow A7 cleanup?
4. Does the A8 dynamic/refit plan identify the right missing test coverage
   before physics integration?

## Proposed Verdict Criteria

PASS if:

- warmup now exercises the pipeline/frame context boundary;
- RGB8 async smoke still works;
- complex dynamic/refit coverage can be deferred to A8.

NEEDS WORK if:

- warmup should also move RGB8 pack into a future delivery context now;
- A8 dynamic/refit plan should be implemented before committing A7.
