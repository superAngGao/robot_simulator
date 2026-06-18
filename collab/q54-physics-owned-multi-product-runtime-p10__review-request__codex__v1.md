Initiative: q54-physics-owned-multi-product-runtime-p10
Stage: review-request
Author: Codex
Version: v1
Date: 2026-06-16
Status: superseded-by-v2
Related Files: collab/q54-multi-product-runtime-p9__review-request__codex__v1.md, tools/optical_pipeline_lab/frame_tick.py, tools/optical_pipeline_lab/frame_products.py, tools/optical_pipeline_lab/observation_products.py, tools/optical_pipeline_lab/physics_runtime.py, tools/optical_pipeline_lab/runner.py
Owner Summary: P10 should turn P9's tick/product pieces into a lab-internal physics-owned multi-product runtime proof. The first slice should prove video + debug + observation on the same tick stream without adding a public SimulationRuntime or touching render backends.

# Q54 Physics-Owned Multi-Product Runtime / P10 Review Request

## Owner Summary

P9 proved the core boundary: a physics-owned `SimulationFrameTick` can feed
video, debug, and observation products without making physics or render own each
other's internals.

P10 should not jump to a broad public `SimulationRuntime`. The next slice should
turn the P9 pieces into a slightly more explicit lab-internal runtime contract:

```text
PhysicsLabScenarioRuntime owns clock/resources
  -> SimulationFrameTick stream
  -> MultiProductFrameRunner
       -> video product
       -> debug product
       -> observation product
  -> typed run outputs
```

Rendering backend work is intentionally out of scope for P10. Backends can grow
incrementally after the runtime/product boundary is stable.

Recommended next step: implement P10.1 as a small code slice that proves
video + debug + observation can consume the same physics-owned tick stream in
one run, without changing the plain `run_scenario(...)` value-object contract.

## Current State From P9

Implemented pieces:

- `tools.optical_pipeline_lab.frame_tick.SimulationFrameTick`
- `simulation_frame_tick_from_published_frame(...)`
- `FrameProductResult`
- `FrameProduct`
- `MultiProductFrameRunner`
- `DebugFrameProduct`
- `PhysicsVideoFrameProduct`
- `PublishedStateObservationProduct`
- package-level lazy exports from `tools.optical_pipeline_lab`
- `frame_source=physics_published_frame`
- `clock_owner=external_physics_runtime`

The current P9 product video entry is:

```text
run_physics_stepped_video_product_scenario(...)
  -> create PhysicsVideoFrameProduct
  -> append extra_products
  -> for frame_index:
       tick = scenario_runtime.step_tick(frame_index)
       product_runner.step(tick)
  -> product_runner.end_run()
  -> return video FrameTimingRecorder
```

This is good enough to prove P9, but it still exposes video as the primary run
shape. P10 should make the multi-product runtime shape more explicit while
keeping video as one product.

## Boundary Decision

P10 should keep these ownership rules:

1. Physics runtime owns the clock and published frame lifecycle.
2. `MultiProductFrameRunner` owns ordered product orchestration only.
3. Products own their own consumption details.
4. Products needing render contexts own frame-provider borrow/release inside
   `consume(tick)`.
5. Plain `run_scenario(...)` remains value-object based and must not construct
   live physics runtime dependencies.

This prevents a single god object:

```text
bad:
  physics runtime owns physics + render + delivery + obs + policy

good:
  physics runtime owns clock/resources
  runner owns product order/lifecycle
  products own product-specific work
```

## P10 Goals

P10 should answer:

1. Can video, debug, and observation products consume the same physics-owned tick
   stream in one orchestrated run?
2. Can run-level outputs be returned by product name rather than by a
   video-specific object?
3. Can the lab expose a multi-product runtime entry without pretending to be the
   final public `SimulationRuntime`?
4. Can observation stay on the published/sensing contract and avoid private
   physics/render internals while participating in the same run?

## Non-Goals

P10 should not:

- add a public simulator-wide `SimulationRuntime`;
- change the physics solver/core pipeline;
- add a new render backend;
- enable CLI construction of arbitrary live physics runtimes from value-object
  scenario configs;
- wire a policy network or action loop;
- make observation products depend on render frame contexts;
- make `MultiProductFrameRunner` own frame-provider borrow/release.

## Proposed Vocabulary

### `PhysicsMultiProductRunResult`

Add a small run result envelope for the explicit product runtime path:

```python
@dataclass(frozen=True)
class PhysicsMultiProductRunResult:
    rows: FrameTimingRecorder | None
    begin_outputs: Mapping[str, object | None]
    end_outputs: Mapping[str, object | None]
    product_results: tuple[tuple[FrameProductResult | None, ...], ...]
```

Open point for review: whether `rows` should stay named `video_rows` instead of
generic `rows`. The conservative first slice can use `video_rows` because only
the video product writes `FrameTimingRecorder` today.

### `run_physics_multi_product_scenario(...)`

Introduce a lab-internal entry that makes products first-class:

```python
def run_physics_multi_product_scenario(
    config: OpticalLabScenarioConfig,
    options: LabRunOptions,
    *,
    scenario_runtime: PhysicsLabScenarioRuntime,
    products: tuple[FrameProduct, ...],
) -> PhysicsMultiProductRunResult:
    ...
```

This function should not create video internals. It should:

- validate the physics-owned source/clock contract;
- write `scenario_config.json`;
- call `begin_run()`;
- produce ticks through `scenario_runtime.step_tick(...)`;
- call `MultiProductFrameRunner.step(tick)`;
- collect positional per-frame results;
- call `end_run()`;
- return outputs keyed by `product_name`.

Video setup can remain in a helper that creates `PhysicsVideoFrameProduct`.

## Suggested Implementation Slices

### P10.1: Prove Three Products On One Tick Stream

Add a focused test using fakes:

```text
PhysicsLabScenarioRuntime.step_tick(...)
  -> PhysicsVideoFrameProduct
  -> DebugFrameProduct
  -> PublishedStateObservationProduct
```

The test should verify:

- exactly one `step_tick` per frame;
- video consumes the tick before debug/observation if product order says so;
- debug and observation receive the same `frame_id` / `sim_time`;
- observation consumes `tick.published_frame`;
- render borrow/release stays inside video product;
- no product reads private physics scratch or render internals.

This may be implemented first by extending the existing
`extra_products=(...)` path if that keeps the diff small.

### P10.2: Return A Run Result Envelope

Today `run_physics_stepped_video_product_scenario(...)` returns only
`FrameTimingRecorder`. That keeps P9 behavior simple but hides product lifecycle
outputs.

P10.2 should either:

- introduce `run_physics_multi_product_scenario(...)`; or
- add a parallel result-returning helper while preserving the old return type.

Recommended conservative path:

```text
keep:
  run_physics_stepped_video_product_scenario(...) -> FrameTimingRecorder

add:
  run_physics_multi_product_scenario(...) -> PhysicsMultiProductRunResult
```

Then the video-specific helper can become a thin adapter later, not in the first
P10 commit.

### P10.3: Observation Product Integration Contract

Add tests that run `PublishedStateObservationProduct` inside the product runner
with a published frame carrying:

- `q`
- `qdot`
- `v_bodies`
- `contact_mask`

The expected observation vector should be asserted numerically, reusing the P9.4
coverage style. The important P10 addition is not the vector math; it is that
the observation result is returned as a product result in the same run lifecycle
as video/debug.

### P10.4: Documentation And Matrix Metadata

Only after P10.1-P10.3:

- update `GPU_OPTICAL_PIPELINE_DESIGN.md` with the final runtime/product
  boundary;
- update `MANIFEST.md` if a new module is added;
- decide whether matrix summaries need product-set metadata such as
  `products=video,debug,observation`.

## Validation Plan

CPU/unit:

```bash
PYTHONPATH=. pytest -q tests/unit/optics/test_optical_pipeline_lab.py \
  -k "multi_product or physics_video_product or published_state_observation_product"
```

Warp-enabled smoke:

```bash
conda run -n robot_sim env PYTHONPATH=. python -m pytest -q \
  tests/unit/optics/test_optical_pipeline_lab.py
```

GPU smoke should remain targeted unless P10 changes GPU runtime behavior. P10 is
mostly orchestration, not a render backend change.

## Review Questions

1. Should P10 first extend the existing `extra_products` path, or introduce
   `run_physics_multi_product_scenario(...)` immediately?

   Codex recommendation: extend tests around the existing path for P10.1, then
   add the result-returning entry in P10.2.

2. Should the first run result envelope store all per-frame product results?

   Codex recommendation: yes for lab/runtime tests. It gives reviewable evidence
   that products consumed the same ticks. If memory becomes a concern later, add
   an opt-out or streaming sink.

3. Should observation run after video or before video?

   Codex recommendation: product order should be explicit. Tests should prove the
   configured order. The first production-like helper can keep video first to
   preserve the P9 ordering guarantee already tested.

4. Should P10 create a public `SimulationRuntime` name?

   Codex recommendation: no. Use lab-internal names until action/reset/policy
   ownership exists. P10 is still an ownership proof, not the final public runtime.

5. Should render backend work block P10?

   Codex recommendation: no. Backend expansion is orthogonal and incremental.
   P10 should keep using existing fake/unit coverage plus the current Warp lab
   smoke.

## Recommended Next Commit

Implement P10.1:

- add a test proving video + debug + observation consume one tick stream;
- keep using the existing P9 product path if possible;
- do not add a new runtime abstraction yet;
- do not touch render backend code.
