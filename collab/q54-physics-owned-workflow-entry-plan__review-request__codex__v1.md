# Q54 Physics-Owned Workflow Entry Plan

Author: Codex
Date: 2026-05-22
Status: review request

## Summary

This plan extends the current Q54 physics video runner without replacing its
architecture.

P6 proved the bridge from an already selected physics-published frame into the
render/video/delivery path:

```text
published_frame_for_index(i)
  -> PhysicsFrameContextProvider
  -> OpticalLabRenderFrameContext
  -> FrameWorkflowRunner
  -> video delivery
```

The next slice should prove the adjacent boundary:

```text
physics step/publish for frame i
  -> published frame
  -> existing P6 render/video/delivery bridge
```

This is not a new render architecture. It is the natural next step after P6:
physics owns time, render consumes a frame, and delivery consumes render-owned
products.

## Current Boundary

The current explicit runner entry is:

```python
run_physics_video_scenario(
    config,
    options,
    *,
    engine,
    registry,
    base_frame,
    published_frame_for_index,
    build_video_camera,
    synchronize_event,
    pack_rgb8,
    ...
)
```

This is an assembly/bridge entry. It intentionally does not step physics. The
caller provides `published_frame_for_index(frame_index)`, so the optical lab
does not own physics time or engine lifecycle.

That boundary was correct for P6. It let the render side prove:

- source/runtime assembly from physics-owned objects;
- provider lifecycle and physics borrow completion;
- current-frame identity from `OpticalLabRenderFrameContext`;
- delivery and timing metadata after render;
- `run_scenario(...)` remains guarded because it cannot construct a physics
  engine.

## Design Philosophy

The ownership rule should stay simple:

```text
Physics owns time and dynamic frame publication.
Render owns frame-scoped optical preparation and RenderResult creation.
Video owns camera/request planning and RenderedVideoFrame packaging.
Delivery owns readback/write/backpressure/completion.
Workflow owns ordering only; it does not own another layer's data.
```

This means the render pipeline should not call `engine.step(...)` from inside
`OpticalLabRenderPipeline`, `OpticalLabRenderSession`, or
`OpticalLabRenderFrameContext`.

It also means a video benchmark loop should not be the top-level owner of
physics time. Video is one product path. RL observations, debug probes, or
future sensor products should be able to consume the same physics-owned frame
workflow without becoming part of video delivery.

## Proposed Next Entry

Add a narrow physics-owned workflow entry that introduces a frame stepper
callback, while still reusing the P6 bridge.

P6 should keep its existing `PhysicsPublishedFrameForIndex` vocabulary because
that helper only asks for "the frame for index i". The caller may replay a
recorded frame sequence, select from a cache, or step physics; P6 does not
care.

P7 should introduce a separate `PhysicsPublishedFrameStepper` vocabulary
because the new helper's semantic contract is stronger: the callback is
expected to advance or select physics time for the requested frame before
returning the published frame.

Candidate type:

```python
PhysicsPublishedFrameStepper = Callable[[int], object]
```

or, if we want to leave room for future control input without committing to RL
now:

```python
PhysicsPublishedFrameStepper = Callable[[int, object | None], object]
```

Recommended first slice: keep the callable minimal and keyword-extensible:

```python
PhysicsPublishedFrameStepper = Callable[..., object]
```

but document the intended call shape:

```python
published_frame = step_physics_frame(frame_index)
```

The new workflow helper can be narrow and video-focused at first:

```python
run_physics_stepped_video_scenario(
    config,
    options,
    *,
    engine,
    registry,
    base_frame,
    step_physics_frame,
    build_video_camera,
    synchronize_event,
    pack_rgb8,
    ...
)
```

`step_physics_frame(frame_index)` owns the physics details:

```text
optional action/control
engine.step(...)
publish/update latest frame
return GpuPublishedFrame
```

The optical lab should not look inside that callback. After the callback returns
a published frame, the workflow should delegate to the same provider/render/
delivery path already proven by P6.

## Why Not Replace `run_physics_video_scenario(...)`?

Keep `run_physics_video_scenario(...)` for now.

It is the lower-level assembly entry and remains valuable for tests, benchmarks,
and callers that already have a published-frame schedule. The new stepped
workflow should be a thin layer above it, not a replacement.

The relationship should be:

```text
run_physics_stepped_video_scenario(...)
  calls step_physics_frame(i)
  then uses the P6 bridge semantics
```

not:

```text
run_physics_video_scenario(...)
  secretly steps physics
```

This preserves a useful debugging boundary: if rendering fails, we can still
reproduce with a known `published_frame_for_index(...)` callback.

## Entry Placement

Do not put the physics step loop directly inside `go2_backend.py`,
`static_asset_source.py`, or `render_session.py`.

The likely first home is `runner.py`, but only as an explicit lab helper, not as
plain `run_scenario(...)` behavior. The helper should keep the same style as
`run_physics_video_scenario(...)`: all physics-owned objects are passed in by
the caller.

Do not enable CLI physics engine construction yet. That is a separate decision:
the CLI would need to know how to construct a concrete physics scene, action
source, engine lifecycle, and cleanup policy.

## Ordering Contract

For each frame:

```text
step_physics_frame(i)                  # physics-owned time advancement
published_frame returned
workflow.step(... published_frame ...) # provider borrow + render
provider exits                         # complete physics borrow
delivery submit/complete
timing row write
```

The P6 borrow/delivery ordering remains unchanged:

```text
with physics_provider.begin_frame(..., published_frame=published_frame):
    render from OpticalLabRenderFrameContext
provider exit completes physics borrow
delivery consumes render-owned buffers
```

If render later becomes fully async, this contract must be revisited so the
physics borrow is released only after render GPU work no longer needs the
borrowed physics frame.

`workflow.flush()` must run only after all per-frame provider
`begin_frame(...)` contexts have exited. Physics borrows are always released
before delivery flush runs.

## RL Boundary

Do not design RL into this slice, but do not block it.

The future RL-compatible shape is:

```text
policy/action source
  -> physics step/publish
  -> one frame context
  -> video consumer, observation consumer, debug consumer, ...
```

The first stepped video helper can use only the video consumer. It should avoid
choices that would make video the only possible product:

- keep `FrameWorkflowRunner` result typed and narrow;
- do not use `Mapping[str, object]` products yet;
- keep consumer registration discussion open until real observation tensors
  appear;
- keep action/control input inside `step_physics_frame(...)` for now.

## Static Assets

Static asset rendering remains separate and simpler.

Static asset sources have no physics-owned time. They should keep using
`static_asset_source.py`, `StaticFrameContextProvider`, and existing static
video paths. The new stepped workflow should not make static preview paths look
like physics paths.

The division remains:

```text
static_asset_source.py:
  build non-simulated optical source data

physics_source.py:
  wrap physics-published frames and borrow lifecycle

runner.py / frame_runtime.py:
  coordinate explicit lab workflows
```

## Proposed Slices

### P7.1: Add A Stepped Physics Video Helper

Add a helper that loops over `options.frames`, calls
`step_physics_frame(frame_index)`, and then runs the same render/video/delivery
path as P6.

The alias introduction ships with this helper rather than as a separate code
slice. A standalone alias-only commit has low value; the naming decision should
be reviewed in the context of the helper that uses it.

Scope:

- introduce `PhysicsPublishedFrameStepper`;
- reuse `create_physics_render_runtime_for_config(...)`;
- reuse `PhysicsFrameContextProvider`;
- reuse `FrameWorkflowRunner`;
- reuse `build_video_render_plan(...)` and
  `render_video_frame_from_context(...)`;
- keep `torch_async` rejected.

Tests:

- unit test verifies callback order:

```text
step_physics_frame(0)
provider.begin_frame(... published_frame=frame0)
render consumer
delivery
```

- unit test verifies `step_physics_frame` exceptions stop the workflow before
  provider borrow;
- unit test verifies render exceptions inside the provider still complete the
  physics borrow and propagate through `FrameWorkflowRunner.step(...)` without
  suppression.

### P7.2: GPU Smoke With Real Engine Step

Add one focused GPU smoke that uses the existing tiny synthetic physics scene.

The test should prove:

- the stepper calls real `GpuEngine.step(...)` or the smallest existing
  equivalent;
- the returned published frame is the one borrowed by the provider;
- rendered range changes with physics state, not with a stale base frame;
- `frame_timing.csv` still records `frame_source == "physics_runtime"`;
- physics borrow completion happens before delivery submission.

This should be the only GPU test required for the new entry.

### P7.3: Decide Whether `run_scenario(...)` Should Remain Guarded

Default recommendation: keep it guarded.

Plain `run_scenario(...)` should not construct physics engines until we have a
real CLI-level physics scene/action configuration. The explicit helper is enough
for lab tests and integration callers.

If we later enable CLI physics runtime, that should be a separate P8 decision.

## Non-Goals

- no generic `SimulationFrameRuntime` export;
- no RL observation API;
- no automatic physics engine construction;
- no Go2/static asset involvement;
- no torch async physics delivery;
- no change to render session ownership;
- no change to physics publish ring semantics.

## Open Review Questions

1. Is the next entry correctly framed as a thin layer above P6, rather than a
   replacement for `run_physics_video_scenario(...)`?

2. Is `PhysicsPublishedFrameStepper` the right companion to the existing
   `PhysicsPublishedFrameForIndex`, with `ForIndex` kept for replay/selection
   semantics and `Stepper` used only when the callback is expected to advance
   physics time?

3. Should the first stepper signature be strictly
   `Callable[[int], GpuPublishedFrame]`, or stay duck-typed as
   `Callable[..., object]` to avoid importing physics types into runner-facing
   annotations?

4. Should action/control remain hidden inside `step_physics_frame(...)` for now,
   or should the workflow already accept an optional `action_for_index(...)`
   callback?

5. Should the stepped helper live in `runner.py` next to
   `run_physics_video_scenario(...)`, or should we introduce a small
   `physics_workflow.py` module before this grows?

6. Is it still correct to keep plain `run_scenario(...)` guarded after P7?

7. What is the minimum GPU smoke needed to prove "physics owns time" without
   adding a broad physics test matrix?
