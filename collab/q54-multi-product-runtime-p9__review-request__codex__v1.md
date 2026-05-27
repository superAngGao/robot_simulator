# Q54 Multi-Product Runtime / P9 Review Request

Author: Codex
Date: 2026-05-27
Status: P9.1 tick contract implemented; P9.2+ pending review

## Summary

P8 proved that an explicit physics runtime owner can drive the optical video
path:

```text
PhysicsLabScenarioRuntime
  -> step_frame(i)
  -> published frame
  -> PhysicsFrameContextProvider
  -> video render
  -> delivery/timing
```

P8.3 then closed the `run_scenario(...)` question:

```text
run_scenario(...) is a value-object lab runner.
It must not construct or own live physics runtime dependencies.
```

P9 should therefore focus on the next ownership boundary:

```text
one physics-owned frame tick
  -> video product
  -> future RL observation product
  -> debug/readback product
```

The key recommendation is:

```text
Do not introduce a broad user-facing SimulationRuntime yet.
First define a small multi-product frame runtime contract.
```

## Current Shape

The current P8 path has three useful layers.

### Physics owner

`tools.optical_pipeline_lab.physics_runtime.PhysicsLabScenarioRuntime` owns live
physics resources:

```text
engine
registry
base_frame
step_frame(frame_index)
bounds
metadata
close()
```

It is deliberately explicit. Callers pass its live objects to the physics video
runner; no value-object config constructs the engine implicitly.

### Frame-context provider

`PhysicsFrameContextProvider` borrows a published physics frame and creates an
`OpticalLabRenderFrameContext`:

```text
published_frame
  -> runtime.begin_frame(...)
  -> render frame context lease
```

This is already a useful boundary: it converts a physics frame into a render
context without deciding which products should be generated from that context.

### Video-focused workflow

`FrameWorkflowRunner` currently coordinates a narrow product:

```text
provider.begin_frame(...)
  -> video_consumer(frame_context, frame_index)
  -> optional video delivery
  -> delivered video recorder
```

Its own docstring is correct: it is intentionally narrower than a future
simulation frame runtime and is a lab-internal video-focused workflow.

## Problem To Solve

Today the first full physics-owned path is video-shaped. That is good for P8,
but P9 should avoid making video the central abstraction.

Future products have different shapes:

- video: render request, RGB/full readback, encoder/delivery/timing rows;
- RL observation: named tensor fields, normalization, policy-facing batch shape;
- debug readback: CPU materialization/logging, maybe sparse or optional fields;
- sensing products: typed readings, backend-neutral published-contract fields.

Those should be products of a physics frame tick, not hidden inside the physics
runtime owner and not squeezed into `run_scenario(...)`.

## Proposed P9 Vocabulary

### `SimulationFrameTick`

Introduce a typed frame-tick object before introducing a full simulation runtime:

```python
@dataclass(frozen=True)
class SimulationFrameTick:
    frame_index: int
    env_idx: int
    frame_id: int
    sim_time: float
    published_frame: object
    metadata: Mapping[str, object]
```

This represents the shared per-frame fact that all products consume.

For P9, the tick can be produced by `PhysicsLabScenarioRuntime.step_frame(i)`.
Later, a real environment/runtime can produce the same tick from action/reset
loops.

### Product Interfaces

Define products as consumers of a tick plus whatever adapter context they need:

```python
class FrameProduct(Protocol):
    product_name: str

    def begin_run(self, run_context) -> None: ...
    def consume(self, tick: SimulationFrameTick) -> object | None: ...
    def end_run(self) -> object | None: ...
```

This is intentionally schematic. The important boundary is that products own
their product-specific state:

```text
video product
  -> render pipeline/session/delivery/timing rows

observation product
  -> observation schema/tensor staging/policy input buffer

debug product
  -> selected CPU readback/logging materialization
```

The physics runtime owner should not know these details.

### Product Results

Use typed result envelopes instead of returning loosely related tuples:

```python
@dataclass(frozen=True)
class FrameProductResult:
    product_name: str
    frame_index: int
    payload: object | None
    timing: Mapping[str, float]
    metadata: Mapping[str, object]
```

This gives later code a place to record per-product timings without forcing
video timing fields onto observation or debug products.

## What Not To Do In P9

Do not make `SimulationRuntime` a top-level owner yet if it must answer all of
these questions at once:

- action source;
- reset policy;
- episode lifecycle;
- batching / env count;
- video delivery;
- observation schema;
- debug readback;
- CLI semantics.

That would recreate the god-object risk identified in P8.3.

Instead, P9 should prove a smaller contract:

```text
explicit physics tick owner
  -> product registry/list
  -> per-product consume/end lifecycle
  -> typed product results
```

## Relationship To `FrameSourceKind`

`FrameSourceKind` currently combines two dimensions:

```text
where frame data comes from
who owns the frame clock
```

P9 should not immediately split the enum in code. The value appears in presets,
validation gates, timing CSV defaults, GPU tests, and serialized
`scenario_config.json`.

The design direction should be:

```text
frame_source: static_asset | synthetic_sequence | physics_published_frame
clock_owner: runner | external_physics_runtime
```

But that should be introduced as a schema/metadata migration after the product
contract is clearer.

## Relationship To RL Observations

The existing RL observation schema work already defines stable observation field
names, ordering, and normalization:

```text
ObsSchema
ObsFieldSpec
locomotion_obs_schema(...)
```

P9 should not rebuild that inside optical lab code. A future observation product
should consume `SimulationFrameTick.published_frame` and produce the schema's
named observation tensor/product.

Important boundary:

```text
published physics/sensing contract
  -> observation product
  -> policy input
```

The product should not infer values from private physics scratch. If an
observation needs a missing field, the published contract or sensing layer
should grow first.

## Relationship To Sensing

The sensing phase-1 decision kept readings conservative:

- no inferred sensor values;
- no duplicate orientation representation;
- no synthetic contact masks;
- backend asymmetry remains visible through optional fields.

P9 should preserve that discipline. A product may adapt published readings into
video/observation/debug payloads, but it should not create hidden sensor truth
from private engine internals.

## Suggested Implementation Sequence

### P9.1: Name The Tick Contract

Add a tiny local dataclass/protocol around physics-owned frame ticks.

Goal:

```text
step physics
  -> SimulationFrameTick
```

No user-facing CLI. No generic simulation runtime yet.

Implementation note: P9.1 is now represented by
`tools.optical_pipeline_lab.frame_tick.SimulationFrameTick` and
`PhysicsLabScenarioRuntime.step_tick(...)`. The method wraps the existing
`step_frame(...)` path, preserving the explicit physics owner while exposing a
shared per-frame fact for later products.

### P9.2: Factor Video Into A Product

Keep existing video behavior, but move the video-specific consume/delivery logic
behind a product-shaped adapter.

Goal:

```text
tick
  -> VideoFrameProduct.consume(tick)
  -> same rendered/delivered video rows as today
```

This should be a behavior-preserving refactor with focused tests.

### P9.3: Add A Minimal Debug Product

Before RL, add a simple debug product that records frame identity and selected
metadata from the tick. This tests multi-product orchestration without pulling
in observation tensor requirements too early.

Goal:

```text
tick
  -> video product
  -> debug product
```

### P9.4: Design Observation Product Separately

Only after P9.1-P9.3 should we wire an RL observation product. It should use the
existing `ObsSchema` contract and published/sensing data, not optical-render
internals.

## Open Questions For Review

1. Is `SimulationFrameTick` the right small concept, or should it be named more
   narrowly, such as `PhysicsFrameTick` or `PublishedFrameTick`?

2. Should product orchestration live near `FrameWorkflowRunner`, or should P9
   introduce a sibling runner to avoid overloading the video-focused class?

3. Should video be converted into a `FrameProduct` before any debug/observation
   product exists, or should we first add a tiny no-op/debug product to prove
   the orchestration shape?

4. What is the minimum typed product result needed for timing/reporting without
   leaking video-specific fields into every product?

5. Should `frame_source`/`clock_owner` schema migration wait until after P9
   product contracts are tested?
