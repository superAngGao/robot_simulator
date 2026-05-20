# Q54 Provider-Backed Video Benchmark P3 Implementation Note

Date: 2026-05-20
Author: Codex
Status: implementation-note

## Scope

Added the P3 provider-backed video benchmark entrypoint. This keeps the current
Go2/static `run_video_benchmark(...)` path intact while adding a new path whose
per-frame render input is an already acquired `OpticalLabRenderFrameContext`.

No CLI behavior changed, and `runner.py` is still untouched.

## Changes

`tools/optical_pipeline_lab/video_loop.py`

- Added `run_video_benchmark_with_frame_contexts(...)`.
  - Takes `scene`, `args`, `frame_provider`, `build_video_camera`, delivery
    helpers, and optional frame identity / geometry mode callbacks.
  - Builds `VideoRenderPlan` outside provider ownership.
  - Acquires the frame context through `frame_provider.begin_frame(...)`.
  - Calls `render_video_frame_from_context(...)`.
  - Reuses the same delivery facade, timing row builder, and delivered-frame
    recording helpers as the existing video benchmark.
- Added `build_provider_backed_torch_async_warmup_result(...)`.
  - Uses frame index `0` as a real warmup acquisition.
  - Calls provider `begin_frame(...)` instead of `pipeline.begin_frame(...)`.
  - Returns the warmup compute result and shadow-traversal inclusion flag for
    async readback ring setup.
- Refactored shared benchmark mechanics into small private helpers:
  - `_validate_video_benchmark_args(...)`
  - `_video_delivery_request_for_args(...)`
  - `_video_frame_timing_recorder(...)`
  - `_video_ray_cache_for_scene(...)`
  - `_video_frame_timing_row_builder(...)`
  - `_run_video_delivery_loop(...)`

`tests/unit/optics/test_optical_pipeline_lab.py`

- Added a provider-backed benchmark test that verifies:
  - frame identity is consumed into request/camera identity;
  - provider receives `env_idx` from the planned camera;
  - provider enter/exit wraps render;
  - delivery rows preserve geometry mode and prepare timing.
- Added a provider-backed warmup test that verifies:
  - warmup uses provider lifecycle;
  - provider receives the planned camera `env_idx`;
  - render diagnostics/traversal-counter intent reaches the warmup request.

`GPU_OPTICAL_PIPELINE_DESIGN.md`

- Marked P3 complete in the active render foundation plan.

`MANIFEST.md`

- Recorded the provider-backed benchmark and warmup helper.
- Updated Q54 sensing/optics collected test count to 249.

## Key Boundary

The new benchmark deliberately does not accept the old
`render_frame(pipeline, args, frame_index, ray_cache)` callback. That callback
is coupled to `pipeline.begin_frame(...)`, which is the exact coupling P1/P2/P3
are removing.

Customization for the provider-backed path should happen through:

- frame-context providers,
- frame identity callbacks,
- geometry mode callbacks,
- future frame consumers.

## Validation

Focused tests:

```text
conda run -n env_tilelang_20260119 python -m pytest \
  tests/unit/optics/test_optical_pipeline_lab.py -q \
  -k "provider_backed or run_video_benchmark_with_frame_contexts"

2 passed, 96 deselected
```

Static checks:

```text
conda run -n env_tilelang_20260119 python -m ruff check \
  tools/optical_pipeline_lab/video_loop.py \
  tests/unit/optics/test_optical_pipeline_lab.py

All checks passed
```

## 关键思考

1. Why add a new benchmark function instead of extending the old one in place?

   The existing `run_video_benchmark(...)` is the stable Go2/static path and
   still supports the legacy `render_frame(...)` callback. The provider-backed
   path has a different ownership model, so a separate function makes the
   boundary explicit and avoids breaking current callers.

2. Why factor `_run_video_delivery_loop(...)`?

   Delivery submit/complete/flush ordering is subtle, especially for ordered
   async readback. Sharing that loop prevents the new provider-backed path from
   becoming a fork of delivery semantics.

3. Why use frame index `0` for provider-backed warmup?

   The design rejected a magic `frame_index=-1`. A real frame acquisition keeps
   provider lifecycle and frame identity semantics identical to normal frames.
   Physics providers still reject `torch_async` until their warmup source is
   explicitly resolved.

## Residual Risks

- The new provider-backed benchmark is CPU-unit covered but not yet wired into a
  GPU smoke. That belongs to the later physics video smoke slice.
- Provider-backed warmup exists, but `PhysicsFrameContextProvider` still rejects
  `torch_async` until the physics warmup frame source is specified.
- The old `build_torch_async_warmup_result(...)` remains for current Go2/static
  callers; it should be retired only after those callers migrate to providers.
