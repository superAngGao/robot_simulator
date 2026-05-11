# Q54 Optical Render API A5 Pipeline Entrypoint Implementation Note

Date: 2026-05-11
Author: Codex
Status: implementation-note

## Scope

Implemented the minimal A5 internal pipeline entrypoint slice.

This introduces a structural pipeline/frame shape and routes the Go2 video
render hot path through a frame context, without changing CSV timing semantics
or delivery loop ownership.

## Changes

`optics/render_api.py`

- Added `@runtime_checkable` `RenderFrameContext` protocol:
  - `frame_id`
  - `sim_time`
  - `env_idx`
  - `render(request: RenderRequest) -> RenderResult`
- Added `@runtime_checkable` `OpticalRenderPipeline` protocol:
  - `begin_frame(frame_inputs: object | None = None, *, env_idx: int = 0)`
  - `deliver(rendered: RenderResult, request: DeliveryRequest | None = None)`
- Kept these names explicit-import only; they are not re-exported from
  `optics.__init__`.

`tools/optical_pipeline_lab/go2_backend.py`

- Added `Go2RenderPipeline`:
  - wraps the existing `Go2RenderSession`;
  - exposes `begin_frame(...)`;
  - keeps `session` as a public transitional field for setup/warmup paths;
  - leaves `deliver(...)` unimplemented because delivery remains owned by the
    existing lab benchmark loops.
- Added `Go2RenderFrameContext`:
  - exposes frame metadata from `session.scene.frame`;
  - calls `Go2RenderSession.execute_request(...)`;
  - returns `RenderResult(compute=compute_result)`;
  - leaves `RenderResult.timing` empty in A5.
- `render_many_views(...)` now creates a `Go2RenderPipeline` first and keeps
  using `pipeline.session` for setup/warmup/refit.
- `_render_video_frame(...)` now calls:

```python
frame_context = pipeline.begin_frame(env_idx=camera.env_idx)
render_result = frame_context.render(render_request, render_profile=render_profile)
result = render_result.compute
```

Sync and async benchmark loops remain separate.

## Transitional Boundaries

A5 intentionally leaves these as future cleanup:

- setup/warmup/refit still access `pipeline.session` directly;
- `RenderResult.timing` is empty; lab CSV timing remains authoritative;
- `Go2RenderPipeline.deliver(...)` raises `NotImplementedError`;
- `DeliveryResult` remains vocabulary only, not a hot-loop execution object;
- the future context-manager shape for path tracing accumulation is documented
  but not implemented.

## Validation

CPU:

```text
ruff check optics/render_api.py tools/optical_pipeline_lab/go2_backend.py tests/unit/optics/test_render_api.py tests/unit/optics/test_optical_pipeline_lab.py
  All checks passed

ruff format --check optics/render_api.py tools/optical_pipeline_lab/go2_backend.py tests/unit/optics/test_render_api.py tests/unit/optics/test_optical_pipeline_lab.py
  4 files already formatted

conda run -n env_tilelang_20260119 python -m pytest tests/unit/optics/test_render_api.py tests/unit/optics/test_optical_pipeline_lab.py -q
  45 passed
```

GPU smoke:

```bash
conda run -n env_tilelang_20260119 python -m tools.optical_pipeline_lab run \
  --preset go2_video_ordered_static \
  --out out/optical_pipeline_lab/a5_pipeline_entrypoint_smoke_gpu1 \
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
shadow_traversal_ray_count: 3,873,731 / 3,773,378
pack_rgb8_ms: 0.177 / 0.180
render_execute_ms: 5.502 / 7.819
```

Frame timing CSV:

```text
out/optical_pipeline_lab/a5_pipeline_entrypoint_smoke_gpu1/frame_timing.csv
```

## Review Notes

Claude review: PASS / green light to submit.

Accepted review notes:

1. `OpticalRenderPipeline` / `RenderFrameContext` remain sufficiently internal
   because they are explicit imports from `optics.render_api` and are not
   re-exported from `optics.__init__`.
2. The concrete `Go2RenderPipeline.session` field is an acceptable transitional
   escape hatch for setup/warmup/refit paths.
3. `Go2RenderFrameContext.render(..., render_profile=...)` is acceptable in A5
   but is the main A6 cleanup target. The mutable `render_profile` side-channel
   prevents the concrete method from matching the protocol's clean
   `render(request)` shape and should be internalized into
   `RenderResult.timing`.
4. `env_idx` is currently a dataclass field while the protocol declares it as a
   read-only property. This is fine for runtime structural checks and current
   tests, but if static type checking with mypy/pyright is introduced, consider
   switching the concrete class to an explicit `@property`.
5. A6 should prioritize render timing internalization over a delivery facade;
   delivery loop factoring can wait until render-side timing ownership is clean.
