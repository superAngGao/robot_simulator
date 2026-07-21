# Q54 P11.6b-3 Implementation Note: Tick Providers And Generic Video Product

Owner: Codex
Date: 2026-07-21

## Summary

Implemented the provider/product slice from the approved P11.6b V3 design:
video product execution is now runtime-agnostic at the product boundary.

New product-facing provider adapters:

- `TickFrameContextProvider`
- `PhysicsTickFrameContextProvider`
- `StaticTickFrameContextProvider`
- `physics_tick_frame_context_provider(...)`
- `static_tick_frame_context_provider(...)`

Video product changes:

- introduced `VideoFrameProduct`;
- kept `PhysicsVideoFrameProduct = VideoFrameProduct` as a compatibility alias;
- changed product consumption to call
  `frame_provider.begin_frame_for_tick(tick)`;
- split generic `build_video_frame_product(...)` from
  `build_physics_video_frame_product(...)`.

The physics builder now performs physics render-runtime construction, wraps the
low-level physics frame provider with a tick provider, then delegates generic
video delivery/CSV/product assembly to `build_video_frame_product(...)`.

## Design Alignment

This resolves the second V3 blocker:

```text
build_physics_video_frame_product(...)
  -> physics render runtime construction
  -> physics tick provider
  -> build_video_frame_product(...)
```

`VideoFrameProduct` no longer knows whether a frame context comes from physics,
static assets, synthetic frames, or a future runtime owner. That distinction is
owned by the injected tick provider.

## Changed Files

- `tools/optical_pipeline_lab/frame_providers.py`
  - new product-facing tick provider adapters.
- `tools/optical_pipeline_lab/runner.py`
  - generalized `PhysicsVideoFrameProduct` to `VideoFrameProduct`;
  - added compatibility alias;
  - added `build_video_frame_product(...)`;
  - kept `build_physics_video_frame_product(...)` as a physics-specific wrapper.
- `tools/optical_pipeline_lab/__init__.py`
  - lazy-exported the new provider/product boundary names.
- `tests/unit/optics/test_optical_pipeline_lab.py`
  - added provider adapter coverage;
  - added generic `VideoFrameProduct` coverage;
  - extended lazy export assertions.
- `MANIFEST.md`
  - registered `frame_providers.py`;
  - updated runner description.

## Tests

Focused coverage should include:

```bash
PYTHONPATH=. pytest -q tests/unit/optics/test_optical_pipeline_lab.py \
  -k "tick_frame_context_providers or video_frame_product_uses_tick_frame_provider or video_product_spec or physics_video_product_runner"
```

Full optical lab unit coverage should still pass.

## Non-Goals

- No static runtime factory yet.
- No Go2 video product registration yet.
- No `run_optical_lab_preset(...)` runtime factory dispatch change yet.
- No `go2_backend.py` deletion yet.
