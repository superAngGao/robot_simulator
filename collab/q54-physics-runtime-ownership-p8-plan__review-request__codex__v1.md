# Q54 Physics Runtime Ownership P8 Plan

Author: Codex
Date: 2026-05-25
Status: accepted for P8.1 implementation

## Summary

P7 closed the frame handoff boundary:

```text
physics step/publish
  -> published frame
  -> render/video/delivery
```

The remaining question is not how render consumes physics frames. That is now
working. The P8 question is who owns the wider physics runtime:

```text
engine construction
reset/lifecycle
action/control source
step cadence
cleanup
optional video/observation products
```

This plan recommends keeping `run_scenario(...)` guarded while adding one small
runtime-owner abstraction for explicit lab callers. CLI exposure should come
later, after the owner boundary is real in code and tests.

## Current State

P6 provides the replay/selection entry:

```python
run_physics_video_scenario(
    ...,
    published_frame_for_index=lambda i: frame,
)
```

P7 provides the stepped entry:

```python
run_physics_stepped_video_scenario(
    ...,
    step_physics_frame=lambda i: published_frame,
)
```

Both require the caller to provide:

- `engine`
- `registry`
- `base_frame`
- camera builder
- readback helpers
- bounds/metadata

The optical lab does not construct a physics engine. Plain `run_scenario(...)`
still rejects `FrameSourceKind.PHYSICS_RUNTIME` before any output directory is
created.

## Design Principle

The P8 owner should be responsible for lifecycle, not rendering internals.

```text
Runtime owner:
  construct/reset/close engine
  produce actions or scripted controls
  step physics and publish frames
  decide which products to request

Optical lab:
  assemble render runtime from caller-owned physics objects
  consume published frames
  run render/video/delivery products
  write timing/reporting
```

Do not move engine construction into:

- `render_session.py`
- `physics_source.py`
- `video_loop.py`
- `static_asset_source.py`
- `go2_backend.py`

Those modules are not lifecycle owners.

## Candidate Owners

### Option A: CLI Owns The Physics Engine

Shape:

```text
run_scenario(config, options)
  -> construct physics scene/engine from scenario config
  -> step scripted frames
  -> render/video/delivery
```

Pros:

- easiest user-facing command;
- scenario config becomes self-contained;
- useful for demos and benchmark commands.

Cons:

- forces CLI config to encode physics scene construction too early;
- needs reset policy, action source, cleanup, and error handling immediately;
- risks making video benchmark semantics look like simulation runtime
  semantics;
- poor fit for RL, where policy/training loop usually owns actions and reset.

Recommendation:

Do not choose this as the first P8 implementation. Keep it as a later P8/P9
CLI exposure after ownership is tested through an explicit runtime owner.

### Option B: Sensor Loop Owns The Physics Engine

Shape:

```text
sensor/runtime loop
  -> engine.step(...)
  -> publish frame
  -> optical/video/sensor products
```

Pros:

- closer to "sensor products consume the current simulation frame";
- good fit for non-RL simulation playback;
- can keep video as one optional product.

Cons:

- the repo does not yet have a dedicated sensor-loop runtime that owns physics;
- still needs a generic action/reset story;
- could become a dumping ground if introduced before a concrete minimal use.

Recommendation:

Good medium-term home, but not the first implementation unless we already have
a concrete sensor-loop integration target.

### Option C: RL / Environment Runtime Owns The Physics Engine

Shape:

```text
env.step(action)
  -> pre-physics action processing
  -> engine.step(...)
  -> publish frame
  -> observation products
  -> optional video/debug products
```

Pros:

- matches the strongest long-term ownership model: policy/action drives time;
- rendering and video naturally become optional consumers;
- avoids making CLI/video own action semantics.

Cons:

- RL observation product shape is not designed yet;
- introducing a real environment runtime now would be premature;
- would likely need typed observation outputs, reset semantics, and batched env
  policy.

Recommendation:

Keep this as the long-term target. P8 should avoid decisions that block it, but
should not introduce RL-specific APIs yet.

### Option D: Explicit Lab Runtime Owner

Shape:

```python
with PhysicsLabScenarioRuntime(...) as runtime:
    runtime.reset()
    rows = run_physics_stepped_video_scenario(
        ...,
        engine=runtime.engine,
        registry=runtime.registry,
        base_frame=runtime.base_frame,
        step_physics_frame=runtime.step_frame,
    )
```

Pros:

- keeps lifecycle explicit and testable;
- does not overload `run_scenario(...)`;
- can start with the existing synthetic body triangle scene;
- can later be reused by CLI, sensor loop, or RL wrappers;
- makes cleanup and reset policy visible.

Cons:

- one more lab helper object;
- not a final user-facing API;
- still narrow until observation products exist.

Recommendation:

This is the best first P8 slice.

## Proposed P8 Direction

Add a narrow lab-internal explicit runtime owner for the one implemented physics
smoke scene.

Candidate name:

```python
PhysicsLabScenarioRuntime
```

or, if we want to avoid sounding too general:

```python
PhysicsBodyTriangleRuntime
```

Recommended first name:

```python
PhysicsLabScenarioRuntime
```

because the object owns runtime lifecycle, not just asset construction. Keep it
inside `tools/optical_pipeline_lab/` and do not export it as a production API.

## Proposed Interface

Initial scope:

```python
@dataclass
class PhysicsLabScenarioRuntime:
    engine: object
    registry: object
    base_frame: object
    bounds_min: object | None
    bounds_max: object | None
    metadata: Mapping[str, object]

    def step_frame(self, frame_index: int) -> object:
        ...

    def close(self) -> None:
        ...

    def __enter__(self) -> PhysicsLabScenarioRuntime:
        ...

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
```

Factory:

```python
create_physics_body_triangle_lab_runtime(
    *,
    device: str,
    initial_height: float,
    height_for_frame: Callable[[int], float],
) -> PhysicsLabScenarioRuntime
```

`step_frame(i)` should:

```text
compute scripted body height for i
call engine.step(...)
return engine.latest_published_frame()
```

The runtime owner may synchronize the published frame in tests if needed, but
the ownership contract should remain: the returned frame is the current physics
published frame for that step.

## Proposed Slices

### P8.1: Add Explicit Lab Runtime Owner

Add the runtime owner for the existing synthetic body triangle physics smoke.

Scope:

- helper lives in a new narrow module, likely `physics_runtime.py`;
- no CLI;
- no `run_scenario(...)` physics enablement;
- no RL/action API;
- no generic `SimulationFrameRuntime`;
- no torch async.

Tests:

- unit test with a fake engine verifies:
  - reset/base frame happens at construction or factory time;
  - `step_frame(i)` calls engine step before returning latest frame;
  - `close()` is idempotent;
  - context manager calls close on normal and exceptional exit.

### P8.2: Wire Runtime Owner Into Existing Stepped Video Helper

Add a focused GPU smoke using:

```python
with create_physics_body_triangle_lab_runtime(...) as runtime:
    rows = run_physics_stepped_video_scenario(
        ...,
        engine=runtime.engine,
        registry=runtime.registry,
        base_frame=runtime.base_frame,
        step_physics_frame=runtime.step_frame,
        bounds_min=runtime.bounds_min,
        bounds_max=runtime.bounds_max,
        metadata=runtime.metadata,
    )
```

This should prove lifecycle-owner integration without exposing CLI.

Tests:

- rendered range changes with scripted runtime heights;
- runtime closes after workflow;
- `run_scenario(...)` remains guarded.

### P8.3: Decide Whether To Expose CLI

After P8.1/P8.2, decide whether to add a CLI path.

Default recommendation:

Keep CLI guarded unless a concrete command is useful immediately. If enabled,
the CLI should call the explicit runtime owner instead of constructing engine
objects inline.

## What P8 Should Not Do Yet

- no `SimulationFrameRuntime` export;
- no RL observation product API;
- no policy/action callback in runner signatures;
- no generic scenario-to-physics-scene compiler;
- no Go2/Menagerie physics path;
- no static asset runtime changes;
- no `torch_async` physics delivery.

## Review Questions

1. Is Option D, an explicit lab runtime owner, the right first P8 slice?

2. Should the first runtime-owner module be named `physics_runtime.py`,
   `physics_lab_runtime.py`, or something narrower?

3. Is `PhysicsLabScenarioRuntime` too broad for a first helper that only supports
   the synthetic body triangle scene?

4. Should `step_frame(i)` own frame synchronization, or should callers/tests
   synchronize the returned published frame when they need host access?

5. Should `height_for_frame(i)` be the first scripted control hook, or should the
   runtime hard-code the two smoke heights and remain even narrower?

6. Is it correct to keep `run_scenario(...)` guarded through P8.1/P8.2?

7. What is the minimum cleanup contract for `GpuEngine` today? Does it need an
   explicit close/destroy call, or is idempotent placeholder cleanup enough until
   physics exposes one?
