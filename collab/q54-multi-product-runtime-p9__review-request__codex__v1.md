# Q54 Multi-Product Runtime / P9 Review Request

Author: Codex
Date: 2026-05-27
Status: P9.1/P9.2a implemented; P9.2b pending

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

Recommended split: do not migrate video in the first P9.2 commit. First prove
the product contract with a tiny sibling runner and a debug/identity product,
then migrate video behind that contract.

Reason:

```text
FrameWorkflowRunner is intentionally video-focused.
P9 needs a tick/product runner, not a renamed video runner.
```

Trying to generalize `FrameWorkflowRunner` in place would mix two jobs:

- current video delivery lifecycle and ordering;
- future multi-product tick orchestration.

The cleaner path is:

```text
FrameWorkflowRunner
  -> keep as the current video-focused workflow while P9 is introduced

MultiProductFrameRunner
  -> new sibling runner for tick -> products
```

This keeps P9.2 small and gives the video migration a stable target.

#### P9.2a: Product Contract And Sibling Runner

Add a new local module, tentatively:

```text
tools.optical_pipeline_lab.frame_products
```

with:

```python
@dataclass(frozen=True)
class FrameProductResult:
    product_name: str
    frame_index: int
    frame_id: int
    sim_time: float
    env_idx: int
    payload: object | None = None
    timing: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)


class FrameProduct(Protocol):
    product_name: str

    def begin_run(self) -> object | None: ...
    def consume(self, tick: SimulationFrameTick) -> FrameProductResult | None: ...
    def end_run(self) -> object | None: ...
```

and a tiny sibling runner:

```python
@dataclass
class MultiProductFrameRunner:
    products: tuple[FrameProduct, ...]

    def begin_run(self) -> Mapping[str, object | None]: ...
    def step(self, tick: SimulationFrameTick) -> tuple[FrameProductResult | None, ...]: ...
    def end_run(self) -> Mapping[str, object | None]: ...
```

`begin_run()` and `end_run()` intentionally return mappings keyed by
`product_name`, not positional tuples of untyped objects. This keeps lifecycle
outputs routable without requiring every product to force run-level state into a
per-frame `FrameProductResult`.

`step(...)` intentionally preserves product order and keeps `None` entries. A
`None` result means "this product consumed or observed the tick but produced no
per-frame result". The returned tuple length must equal `len(products)`.

P9.2a does not introduce provider borrowing. `MultiProductFrameRunner.step()`
only receives a `SimulationFrameTick` and calls products. Any product that needs
a render/physics frame context, such as the later video product, owns its own
borrow context inside `consume(tick)`. This keeps provider lifecycle decisions
out of the generic product runner.

Initial product:

```text
DebugFrameProduct
  -> records tick identity and selected metadata
  -> no render pipeline
  -> no delivery
  -> no GPU dependency
```

P9.2a tests:

- result envelope copies `frame_index`, `frame_id`, `sim_time`, and `env_idx`
  from the tick;
- product ordering is stable;
- `None` consume results are represented positionally, not filtered;
- `step(...)` returns one slot per product even when some products return
  `None`;
- product exceptions stop the step and preserve the original exception;
- begin/end lifecycle is called in product order;
- begin/end return mappings keyed by `product_name`.

Non-goals for P9.2a:

- no video migration;
- no provider borrow lifecycle in `MultiProductFrameRunner`;
- no tick-carried frame context;
- no RL observation product;
- no CLI or preset schema changes;
- no `FrameSourceKind` split.

Implementation note: P9.2a is now represented by
`tools.optical_pipeline_lab.frame_products`. It adds
`FrameProductResult`, `FrameProduct`, `MultiProductFrameRunner`, and
`DebugFrameProduct`. The implementation preserves product result positions,
keeps provider borrowing out of the generic runner, and aggregates begin/end
outputs by `product_name`.

#### P9.2b: Video Product Adapter

After P9.2a is reviewed, wrap the existing physics video behavior behind a
product-shaped adapter.

P9.2b should introduce a parallel product-runner entry path instead of reusing
`run_physics_stepped_video_scenario(...)` internally. The old helper remains
available for the existing video-only lab path; the new path proves:

```text
PhysicsLabScenarioRuntime.step_tick(...)
  -> MultiProductFrameRunner
  -> PhysicsVideoFrameProduct
```

The two entries should have separate validation functions so callers cannot
accidentally route a product-runner scenario through the older video-only
helper just because both use `FrameSourceKind.PHYSICS_RUNTIME`.

Tentative shape:

```text
PhysicsVideoFrameProduct
  owns:
    PhysicsFrameContextProvider
    video render consumer
    VideoDeliveryFacade
    delivered video recorder

  consume(tick):
    provider.begin_frame(... published_frame=tick.published_frame ...)
    render video
    submit/complete delivery
    return FrameProductResult(payload=rendered/delivered summary)
```

The important behavior contract is unchanged:

```text
physics step
  -> published frame tick
  -> provider borrow
  -> render
  -> provider release
  -> delivery submit/complete
```

P9.2b should preserve the existing ordering guarantee from P7/P8:

```text
physics borrow is released before video delivery submit/flush
```

P9.2b tests:

- new product-runner entry does not call or depend on
  `run_physics_stepped_video_scenario(...)`;
- product-runner validation is distinct from
  `validate_physics_video_run(...)`;
- existing `run_physics_stepped_video_scenario(...)` behavior remains
  equivalent;
- provider borrow receives `tick.published_frame`;
- provider context exits before delivery submit;
- render failure exits provider and propagates;
- stepper failure stops before provider borrow;
- timing rows and `scenario_config.json` stay unchanged;
- `torch_async` remains rejected until provider-backed warmup exists.

Non-goals for P9.2b:

- no multi-product video+debug production in the same run yet unless P9.2a's
  contract makes it trivial;
- no observation tensors;
- no action/reset/episode lifecycle;
- no user-facing runtime command.

#### Why Debug Product Before Video Migration?

The debug product is intentionally boring. That is the point: it proves the
product runner without importing render/video complexity. Once that is stable,
video migration becomes a behavior-preserving adapter exercise rather than a
contract-design exercise.

This answers the current open question with:

```text
First add a tiny debug/identity product to prove orchestration,
then convert video into a product.
```

### P9.3: Prove Video And Debug Products Together

After P9.2a/P9.2b, run video and debug products from the same tick stream. This
tests actual multi-product orchestration without pulling in observation tensor
requirements too early.

Goal:

```text
tick
  -> video product
  -> debug product
```

The debug product already exists from P9.2a; P9.3 proves it can run next to the
video product while preserving the video borrow-before-delivery guarantee.

### P9.4: Design Observation Product Separately

Only after P9.1-P9.3 should we wire an RL observation product. It should use the
existing `ObsSchema` contract and published/sensing data, not optical-render
internals.

## Open Questions For Review

1. Is `SimulationFrameTick` the right small concept, or should it be named more
   narrowly, such as `PhysicsFrameTick` or `PublishedFrameTick`?

2. Should product orchestration live near `FrameWorkflowRunner`, or should P9
   introduce a sibling runner to avoid overloading the video-focused class?

   Proposed answer: introduce a sibling runner. Keep `FrameWorkflowRunner` as
   the current video-focused workflow while P9 proves the tick/product contract.

3. Should video be converted into a `FrameProduct` before any debug/observation
   product exists, or should we first add a tiny no-op/debug product to prove
   the orchestration shape?

   Proposed answer: add the debug/identity product first as P9.2a, then migrate
   video as P9.2b.

4. What is the minimum typed product result needed for timing/reporting without
   leaking video-specific fields into every product?

   Proposed answer: `FrameProductResult` should carry only tick identity,
   `product_name`, optional payload, timing map, and metadata map. Video-specific
   delivered-frame details stay inside the video payload or video recorder.
   `MultiProductFrameRunner.step(...)` should preserve product positions and
   return `None` in a product's slot when that product has no per-frame result.
   Run lifecycle outputs should be keyed by `product_name`.

5. Should `frame_source`/`clock_owner` schema migration wait until after P9
   product contracts are tested?

   Proposed answer: yes. The enum/schema split should wait until P9.2a/P9.2b
   prove which metadata needs to be serialized. P9.2b should still introduce a
   distinct product-runner validation path so the old video-only physics helper
   and new product-runner entry cannot be confused while both still use
   `FrameSourceKind.PHYSICS_RUNTIME`.

## Claude Review Follow-up

Claude reviewed the P9.2 plan on 2026-05-28 and accepted the overall direction:

- debug/identity product before video migration;
- sibling `MultiProductFrameRunner` instead of generalizing
  `FrameWorkflowRunner`;
- minimal `FrameProductResult` with video details kept in payload/recorders.

The plan now incorporates the requested clarifications:

1. `MultiProductFrameRunner.step(...)` does not own provider borrow/release.
   Products that need frame contexts, including the future video product, own
   their borrow scope inside `consume(tick)`.
2. `None` consume results are represented positionally. The result tuple keeps
   one slot per product, preserving product order.
3. P9.2b introduces a new product-runner entry path parallel to
   `run_physics_stepped_video_scenario(...)`, with distinct validation, rather
   than routing the new product workflow through the existing video-only helper.
4. `begin_run()` / `end_run()` aggregate lifecycle outputs by `product_name`
   instead of returning unlabelled positional objects.
