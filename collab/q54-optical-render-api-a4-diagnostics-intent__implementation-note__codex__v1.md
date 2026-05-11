# Q54 Optical Render API A4 Diagnostics Intent Implementation Note

Date: 2026-05-11
Author: Codex
Status: implementation-note

## Scope

Small follow-up after A3: make the lab path read diagnostic intent from
`RenderRequest.diagnostics` instead of directly coupling readback channels to
`args.render_profile`.

This is not a full generic `OpticalRenderSession` promotion.

## Changes

`tools/optical_pipeline_lab/go2_backend.py`

- `_video_render_request(...)` now accepts an optional `traversal_counters`
  override. Existing callers keep the old default:
  `traversal_counters = profile_timing`.
- Added `_render_profile_buffer_for_request(request)`.
  - returns a mutable profile list when either timing or traversal counters are
    requested;
  - returns `None` otherwise.
- Added `_include_shadow_traversal_stats(request)`.
  - readback channel selection now follows
    `request.diagnostics.traversal_counters`.
- `_RenderedVideoFrame` carries `include_shadow_traversal_stats`, so the sync
  readback path does not need to consult `args.render_profile`.
- Async ring warmup channel selection now follows the warmup
  `RenderRequest.diagnostics.traversal_counters`.

## Important Compatibility Note

The underlying Warp executor still uses `render_profile is not None` as the
implementation switch for both detailed timing and shadow traversal counter
allocation. A4 only centralizes that compatibility rule in
`_render_profile_buffer_for_request(...)`.

Future executor work should split those two concerns explicitly.

## Validation

CPU:

```text
ruff check tools/optical_pipeline_lab/go2_backend.py tests/unit/optics/test_optical_pipeline_lab.py
  All checks passed

ruff format --check tools/optical_pipeline_lab/go2_backend.py tests/unit/optics/test_optical_pipeline_lab.py
  2 files already formatted

conda run -n env_tilelang_20260119 python -m pytest tests/unit/optics/test_render_api.py tests/unit/optics/test_optical_pipeline_lab.py -q
  42 passed
```

GPU smoke:

```bash
conda run -n env_tilelang_20260119 python -m tools.optical_pipeline_lab run \
  --preset go2_video_ordered_static \
  --out out/optical_pipeline_lab/a4_diagnostics_intent_smoke_gpu1 \
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
shadow_traversal_ray_count:        3,873,731 / 3,773,378
shadow_traversal_node_visit_count: 8,925,865 / 15,017,112
shadow_traversal_plane_test_count: 3,809,384 / 3,508,150
```

This confirms traversal diagnostics still materialize through RGB8 async
readback after the intent plumbing change.
