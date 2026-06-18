Initiative: q54-physics-owned-multi-product-runtime-p10
Stage: review-request
Author: Codex
Version: v2
Date: 2026-06-18
Status: in_review
Related Files: collab/q54-physics-owned-multi-product-runtime-p10__review-request__codex__v1.md, collab/q54-multi-product-runtime-p9__review-request__codex__v1.md, tools/optical_pipeline_lab/frame_tick.py, tools/optical_pipeline_lab/frame_products.py, tools/optical_pipeline_lab/observation_products.py, tools/optical_pipeline_lab/physics_runtime.py, tools/optical_pipeline_lab/runner.py
Owner Summary: P10 should start from the user-facing workflow shape, not from the internal runner. The API should make scenario intent, live runtime ownership, product configuration, schedule, and artifacts explicit. Internal boundaries such as MultiProductFrameRunner should serve that workflow instead of becoming the public interface.

# Q54 P10 User-Centered Physics-Owned Multi-Product Workflow

## Owner Summary

P9 proved that physics-owned published frames can be consumed as a shared
`SimulationFrameTick` by video, debug, and observation products. P10 should now
design the workflow a user, a generated script, or a future training loop should
call.

The goal is not merely "less code." In an AI-assisted development workflow,
users at every level can generate boilerplate. The system is easy to use only if
the API has clear concepts, explicit ownership, composable objects, and
fail-fast errors that make wrong calls obvious.

P10 should therefore start from this northbound shape:

```text
user run intent
  -> explicit live physics runtime owner
  -> product specs or product instances
  -> schedule
  -> artifact/output policy
  -> typed workflow result
```

`MultiProductFrameRunner` remains an internal executor. It should not become the
thing users must understand first.

Recommended next step: review and approve the user workflow contract before
implementing P10.1. The first code slice should then prove video + debug +
observation on one physics-owned tick stream while preserving this outward API
direction.

## Why V2

V1 correctly identified the internal P10 runtime boundary, but it started too
close to `MultiProductFrameRunner`. That risks letting internal mechanics define
the external API.

V2 changes the design order:

```text
first:  user-facing workflow semantics
then:   workflow-owned context and product specs
then:   internal runner/result shape
last:   implementation slices
```

Rendering backend expansion remains out of scope. It is an incremental backend
capability problem and should not block P10.

## P9 Baseline

P9 implemented:

- `SimulationFrameTick`
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

The current P9 entry is still video-shaped:

```text
run_physics_stepped_video_product_scenario(...)
  -> create PhysicsVideoFrameProduct
  -> append extra_products
  -> tick = scenario_runtime.step_tick(frame_index)
  -> MultiProductFrameRunner.step(tick)
  -> return video FrameTimingRecorder
```

That was acceptable for P9. P10 should make products and workflow intent the
primary concept.

## API Design Principles

P10 APIs should optimize for:

1. Intent-first calls.
   Users should express scenario, runtime, products, schedule, and output policy,
   not runner lifecycle steps.

2. Explicit live ownership.
   A value-object config must not secretly create a GPU engine or live physics
   runtime. If a workflow owns a runtime, that ownership must be visible.

3. Typed product configuration.
   Observation products need schema, joint indices, and contact body names.
   Video products need camera/readback/delivery/write intent. These cannot be
   hidden behind bare strings except as narrow convenience sugar.

4. Composability across call levels.
   A product should work in scenario-level runs, runtime-level runs, and future
   loop-level training code.

5. AI-friendly names and errors.
   The API should be easy for generated code to call correctly. Parameter names
   should encode concepts like `runtime`, `products`, `schedule`, `output`, and
   `owns_runtime`.

6. No hidden magic.
   The workflow may provide defaults from presets, but it must not guess
   actuated joints, contact body ordering, runtime ownership, or live resource
   construction.

## Caller Categories

The user categories differ by semantic control level, not by how much code they
want to write.

### 1. Scenario-Level User

Controls scenario intent, product set, schedule, and artifacts.

Target shape:

```python
result = run_optical_lab_workflow(
    scenario="physics_body_triangle_video_smoke",
    runtime=runtime,
    products=[
        VideoProductSpec(),
        DebugProductSpec(metadata_keys=None),
        ObservationProductSpec(
            schema=schema,
            actuated_q_indices=actuated_q_indices,
            actuated_v_indices=actuated_v_indices,
            contact_body_names=("left_foot", "right_foot"),
        ),
    ],
    schedule=FixedFrameSchedule(frames=120),
    output=ArtifactOutput(root=Path("runs/p10")),
)
```

Key properties:

- `scenario` can resolve to `OpticalLabScenarioConfig`;
- `runtime` is live and explicit;
- `products` are typed specs, not ambiguous strings;
- `schedule` is explicit rather than a naked integer in the long-term API;
- `output` owns artifact policy and paths;
- result is product-oriented, not video-only.

### 2. Runtime-Level User

Already owns a tick-producing runtime and wants to attach products.

Target shape:

```python
with create_physics_body_triangle_lab_runtime(...) as runtime:
    result = run_physics_product_workflow(
        runtime=runtime,
        products=(video_product, observation_product, debug_product),
        schedule=FixedFrameSchedule(frames=120),
        output=ArtifactOutput(root=Path("runs/p10")),
    )
```

Key properties:

- product instances are accepted;
- no preset is required;
- artifacts are optional unless products need them;
- workflow still returns typed outputs;
- internal runner lifecycle remains hidden.

### 3. Loop-Level User

Future RL/training or interactive code controls stepping and actions.

Target shape:

```python
with runtime.session() as session:
    tick = session.step(action)
    obs_result = observation_product.consume(tick)
    action = policy(obs_result.payload["observation"])
```

P10 does not need to implement this full loop. It must avoid blocking it:

- `SimulationFrameTick` should remain product-neutral;
- observation should depend on published/sensing data, not render context;
- schedules should not assume every future run is fixed-frame benchmark video;
- product instances should remain reusable outside scenario-level helpers.

## Inputs That Must Be Explicit

| Concept | Required For | User Provides | Workflow May Default | Notes |
|---|---|---:|---:|---|
| live runtime / tick source | all physics-owned runs | yes | no | Must expose ownership and close semantics |
| scenario/config | lab scenario runs | yes | from preset | Declarative metadata only, not live construction |
| schedule | fixed benchmark runs | yes | maybe | Start with fixed frames; leave room for until/done later |
| output/artifact policy | artifact-producing runs | yes | maybe | Should centralize paths and write intent |
| product specs | scenario-level workflow | yes | maybe video default only | Specs should be typed |
| product instances | runtime-level workflow | yes | no | Advanced but still clean |
| observation schema | observation product | yes | no | Must not guess |
| actuated q/v indices | joint observations | yes | no | Existing fail-fast guards should stay |
| contact body names | contact mask observations | yes | no | Defines mask ordering |
| camera/readback/delivery options | video product | yes or preset | yes from config/options | Do not leak delivery facade to user |
| runtime ownership flag | workflows with close | yes | default false | Factories can set true |

## Proposed P10 Vocabulary

### `FixedFrameSchedule`

Start narrow:

```python
@dataclass(frozen=True)
class FixedFrameSchedule:
    frames: int
    env_idx: int = 0
```

Why not pass only `frames: int` forever?

- fixed-frame benchmark runs are only one schedule type;
- future training loops need action/reset/done semantics;
- naming schedule explicitly keeps API extensible without pretending P10 solves
  RL loop ownership.

### `ArtifactOutput`

Centralize run outputs:

```python
@dataclass(frozen=True)
class ArtifactOutput:
    root: Path
    write_scenario_config: bool = True
    write_video_frames: bool | None = None
    write_timing_csv: bool = True
```

P10 may initially map this to existing `LabRunOptions`. The user-facing concept
should still be output/artifact policy, not scattered paths.

### Product Specs

P10 should prefer typed specs for scenario-level APIs:

```python
@dataclass(frozen=True)
class VideoProductSpec:
    product_name: str = "video"
    camera: object | None = None
    readback: str | None = None
    delivery: str | None = None

@dataclass(frozen=True)
class DebugProductSpec:
    product_name: str = "debug"
    metadata_keys: tuple[str, ...] | None = None

@dataclass(frozen=True)
class ObservationProductSpec:
    schema: ObsSchema
    actuated_q_indices: object
    actuated_v_indices: object
    contact_body_names: tuple[str, ...] = ()
    product_name: str = "observation"
```

Open point: whether P10.1 should implement these specs immediately. Codex
recommendation: document them now, but allow P10.1 to use product instances if
that keeps the proof small.

### `OpticalLabPhysicsProductWorkflow`

The workflow is the user-facing wrapper around live runtime, declarative config,
product configuration, schedule, and artifacts.

Sketch:

```python
@dataclass
class OpticalLabPhysicsProductWorkflow:
    runtime: PhysicsLabScenarioRuntime
    config: OpticalLabScenarioConfig
    options: LabRunOptions
    products: tuple[FrameProduct, ...] = ()
    product_specs: tuple[object, ...] = ()
    owns_runtime: bool = False

    def build_products(self) -> tuple[FrameProduct, ...]:
        ...

    def run(self, schedule: FixedFrameSchedule) -> PhysicsProductRunResult:
        ...

    def close(self) -> None:
        ...
```

What it should hold:

- live runtime reference;
- explicit runtime ownership flag;
- declarative scenario config;
- run/output options;
- product specs or instances;
- artifact paths derived from output/options;
- run result assembly state.

What it should hide:

- `VideoDeliveryFacade`;
- `FrameTimingRecorder` construction;
- `PhysicsFrameContextProvider`;
- `MultiProductFrameRunner.begin_run/end_run`;
- video row builder internals;
- provider borrow/release implementation details.

### `PhysicsProductRunResult`

Use a user-facing result that is named by product, not only by position:

```python
@dataclass(frozen=True)
class PhysicsProductRunResult:
    out: Path | None
    scenario_config_path: Path | None
    video_rows: FrameTimingRecorder | None
    begin_outputs: Mapping[str, object | None]
    end_outputs: Mapping[str, object | None]
    product_results: Mapping[str, tuple[FrameProductResult, ...]]
    frame_results: tuple[tuple[FrameProductResult | None, ...], ...]
    artifacts: Mapping[str, object]
```

Rationale:

- `frame_results` preserves internal positional proof of product ordering;
- `product_results` gives users a natural named lookup;
- `video_rows` is explicit because only video writes timing rows today;
- artifacts can grow without turning the result into another video-specific
  object.

## Internal Boundary

Internal execution can remain simple:

```text
workflow.run(...)
  -> build products
  -> MultiProductFrameRunner(products)
  -> begin_outputs = runner.begin_run()
  -> for schedule frame:
       tick = runtime.step_tick(frame_index, env_idx=...)
       frame_results.append(runner.step(tick))
  -> end_outputs = runner.end_run()
  -> assemble PhysicsProductRunResult
```

`MultiProductFrameRunner` still owns:

- product order;
- fail-fast lifecycle;
- positional per-frame result preservation.

It should not own:

- live runtime resources;
- scenario config validation;
- artifact path policy;
- video provider borrow/release;
- observation schema validation.

## Non-Goals

P10 should not:

- create the final public platform-wide `SimulationRuntime`;
- change physics core/solver pipeline internals;
- add or complete render backends;
- make `run_scenario(...)` construct live physics runtimes;
- make observation depend on render frame contexts;
- require users to understand `MultiProductFrameRunner` for normal workflows;
- support full RL action/reset/done ownership yet.

## Suggested Implementation Slices

### P10.0: Approve User Workflow Contract

Review this document before code. Specifically decide:

- workflow name;
- whether first implementation exposes product specs or product instances only;
- result shape;
- runtime ownership behavior;
- whether `FixedFrameSchedule` and `ArtifactOutput` should be first-class now.

### P10.1: Three Products On One Tick Stream

Smallest proof:

- use existing products;
- run video + debug + observation in the same physics-owned tick stream;
- assert shared `frame_id` / `sim_time`;
- assert observation consumes `tick.published_frame`;
- keep render borrow/release inside video product;
- keep `run_scenario(...)` untouched.

P10.1 may use the existing P9 `extra_products` path if needed, but the test name
and assertions should describe the future workflow boundary.

### P10.2: Result Envelope

Add `PhysicsProductRunResult` or an equivalent internal result type:

- collect `begin_outputs`;
- collect `end_outputs`;
- collect positional `frame_results`;
- expose named `product_results`;
- keep `video_rows` explicit.

### P10.3: Workflow Wrapper

Add `OpticalLabPhysicsProductWorkflow` or the reviewed name:

- holds runtime/config/options/products;
- owns close semantics if requested;
- hides runner lifecycle;
- returns `PhysicsProductRunResult`.

### P10.4: Product Specs

Add typed specs if not already done:

- `VideoProductSpec`;
- `DebugProductSpec`;
- `ObservationProductSpec`;
- conversion from specs to product instances.

### P10.5: Convenience Entry

Only after the workflow object is clear:

```python
run_optical_lab_workflow(...)
```

This should be a thin wrapper around the workflow object, not the primary owner
of all semantics.

## Validation Plan

Focused unit tests:

```bash
PYTHONPATH=. pytest -q tests/unit/optics/test_optical_pipeline_lab.py \
  -k "multi_product or physics_product_workflow or published_state_observation_product"
```

Warp-enabled lab regression:

```bash
conda run -n robot_sim env PYTHONPATH=. python -m pytest -q \
  tests/unit/optics/test_optical_pipeline_lab.py
```

Targeted GPU smoke only if P10 changes live GPU runtime behavior. P10 should
mostly remain orchestration and API shape.

## Review Questions For Claude

1. Is `OpticalLabPhysicsProductWorkflow` the right level/name, or should this be
   `PhysicsProductWorkflow`, `PhysicsOwnedProductWorkflow`, or something else?

2. Should product specs be introduced in the first P10 code slice, or should P10.1
   prove the workflow using product instances first?

3. Should `FixedFrameSchedule` be first-class now, or is `frames: int` acceptable
   for one more slice?

4. Should `ArtifactOutput` be first-class now, or should P10 initially keep
   `LabRunOptions` and document the desired output abstraction?

5. Should the workflow default `owns_runtime=False` and require factories to opt
   into ownership?

6. Should the result expose both `frame_results` and named `product_results`, or
   is one enough?

7. Does this design leave enough room for future loop-level RL/action ownership?

## Codex Recommendation

Approve the v2 direction, then implement P10.1 conservatively:

- first prove video + debug + observation on the same tick stream;
- keep product instances acceptable;
- do not force product specs in the first code diff;
- keep `run_scenario(...)` unchanged;
- do not touch render backend code;
- use the P10.1 test to drive the exact result/workflow API for P10.2/P10.3.

