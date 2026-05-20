# Q54 Frame Context Providers P2 Implementation Note

Date: 2026-05-20
Author: Codex
Status: implementation-note

## Scope

Added the first narrow frame-context provider layer for the Optical Pipeline
Lab. This is P2 of the physics/video boundary plan:

```text
provider acquires frame context
video consumes OpticalLabRenderFrameContext
delivery remains outside provider ownership
```

This commit does not add a physics video loop, does not alter `runner.py`, and
does not change CLI behavior.

## Changes

`tools/optical_pipeline_lab/frame_contexts.py`

- Added `StaticFrameContextProvider`.
  - Wraps `pipeline.begin_frame(frame_inputs=None, env_idx=...)`.
  - Ignores `frame_index` because the scene is static.
- Added `SyntheticFrameSequenceContextProvider`.
  - Selects `frame_inputs[frame_index]`.
  - Passes selected frame inputs into `pipeline.begin_frame(...)`.
  - Out-of-range indexes use the sequence's normal `IndexError`.
- Added `PhysicsFrameContextProvider`.
  - Wraps `PhysicsLabRenderRuntime.begin_frame(...)`.
  - Yields `OpticalLabRenderFrameContext` to callers, not the physics lease.
  - Lets provider exit complete the underlying `PhysicsLabFrameLease`.
  - Rejects `delivery_mode="torch_async"` until provider-backed warmup exists.

`tests/unit/optics/test_optical_pipeline_lab.py`

- Added coverage for static provider `frame_inputs=None` and `env_idx` pass-through.
- Added coverage for synthetic provider frame selection and `env_idx` pass-through.
- Added coverage for physics provider borrow -> begin_frame -> complete ordering.
- Added coverage for physics provider cleanup when the consumer body raises.
- Added coverage for construction-time `torch_async` rejection.

`GPU_OPTICAL_PIPELINE_DESIGN.md`

- Marked P1 and P2 complete in the active render foundation plan.

`MANIFEST.md`

- Added `frame_contexts.py` to the Q54 file table and flow summary.
- Updated Q54 sensing/optics collected test count to 247.

## Key Boundary

The provider layer intentionally wraps `PhysicsLabFrameLease` instead of
changing `PhysicsLabFrameLease.__enter__`.

Existing lower-level physics tests and helpers still see:

```python
with runtime.begin_frame(...) as lease:
    lease.frame_context
```

The new video-facing provider exposes the cleaner boundary:

```python
with physics_provider.begin_frame(...) as frame_context:
    render_video_frame_from_context(frame_context, plan)
```

That keeps the older lease API stable while giving P3/P4 a frame-context-first
surface.

## Validation

Focused provider/lifecycle tests:

```text
conda run -n env_tilelang_20260119 python -m pytest \
  tests/unit/optics/test_optical_pipeline_lab.py -q \
  -k "frame_context_provider or begin_physics_render_frame or physics_render_runtime_begin_frame"

9 passed, 87 deselected
```

Full lab unit file:

```text
conda run -n env_tilelang_20260119 python -m pytest \
  tests/unit/optics/test_optical_pipeline_lab.py -q

96 passed
```

Collection count:

```text
conda run -n env_tilelang_20260119 python -m pytest --collect-only -q \
  tests/unit/optics tests/unit/sensing \
  tests/gpu/test_optical_warp_executor.py \
  tests/gpu/test_optical_gpu_runtime.py

247 tests collected
```

Static checks:

```text
conda run -n env_tilelang_20260119 python -m ruff check \
  tools/optical_pipeline_lab/frame_contexts.py \
  tests/unit/optics/test_optical_pipeline_lab.py

All checks passed

git diff --check

clean
```

## 关键思考

1. Why not modify `PhysicsLabFrameLease.__enter__` to return frame context?

   That would match the future provider contract directly, but it would also
   change the existing lower-level lease API. The safer P2 slice is a wrapper:
   physics_source remains lease-oriented, while frame_contexts becomes
   video-facing and context-oriented.

2. Why reject `torch_async` in the provider constructor?

   The current async warmup still calls `pipeline.begin_frame(...)` directly.
   Allowing physics providers to use that path would bypass the physics borrow
   lifecycle during warmup. Failing at construction time makes the unsupported
   combination obvious before any frame loop starts.

3. Why keep providers outside `runner.py`?

   `runner.py` should validate config and dispatch implemented scenarios. The
   provider layer is reusable lower-level infrastructure for future video
   benchmark and workflow/runtime slices; putting it in `runner.py` would make
   the runner an orchestration dump too early.

## Residual Risks

- Provider-backed async warmup is still not implemented. This is intentionally
  deferred to P3.
- No GPU test directly uses `frame_contexts.py` yet. Existing GPU smokes already
  cover physics dynamic begin-frame behavior; provider-level GPU coverage should
  land with the P3/P5 video path that consumes providers.
- The provider interface is lab-internal. Do not export it as a public simulator
  API until it survives the provider-backed benchmark path.
