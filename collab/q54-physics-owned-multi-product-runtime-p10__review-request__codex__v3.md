Initiative: q54-physics-owned-multi-product-runtime-p10
Stage: review-request
Author: Codex
Version: v3
Date: 2026-06-18
Status: in_review
Related Files: collab/q54-physics-owned-multi-product-runtime-p10__review-request__codex__v2.md, collab/q54-multi-product-runtime-p9__review-request__codex__v1.md, tools/optical_pipeline_lab/frame_tick.py, tools/optical_pipeline_lab/frame_products.py, tools/optical_pipeline_lab/observation_products.py, tools/optical_pipeline_lab/physics_runtime.py, tools/optical_pipeline_lab/runner.py
Owner Summary: P10 should introduce a physics-owned batch workflow centered on clear user-facing intent, explicit runtime ownership, and multi-product orchestration. V3 accepts Claude's review: use `PhysicsOwnedProductWorkflow`, avoid early schedule/spec over-abstraction, rename `LabRunOptions` toward `ArtifactOutput`, and keep loop-level RL outside the workflow abstraction.

# Q54 P10 Physics-Owned Product Workflow

## Owner Summary

P9 proved the core tick/product boundary:

```text
physics-owned published frame
  -> SimulationFrameTick
  -> video/debug/observation products
```

P10 should turn that proof into a clearer user-facing batch workflow while
keeping internals well contained:

```text
user run intent
  -> explicit live physics runtime
  -> product instances first
  -> fixed frame count for P10
  -> artifact/output policy
  -> PhysicsProductRunResult
```

The important correction from v2 is to reduce early abstraction. P10 does not
need `FixedFrameSchedule`, a full product-spec hierarchy, or a public
`SimulationRuntime` yet. It does need an explicit workflow boundary, typed run
results, and a path where video, debug, and observation consume the same
physics-owned tick stream.

## Changes From V2 Review

Accepted changes:

1. Rename the workflow to `PhysicsOwnedProductWorkflow`.
   The core concept is that physics owns the tick stream. `OpticalLab` is the
   application context, not the ownership boundary.

2. Use `frames: int` for P10.1-P10.3.
   `FixedFrameSchedule` is deferred until P11 or the first action-driven/done
   schedule appears.

3. Rename `LabRunOptions` toward `ArtifactOutput`.
   The existing type mostly describes output/artifact behavior. P10.1 should
   either rename it directly or introduce the rename with a compatibility alias.

4. Avoid dual workflow state.
   A workflow should not hold both `products` and `product_specs` at the same
   time. P10.1 should accept product instances. Product specs can be added later
   through factory helpers.

5. Keep `owns_runtime=False` by default.
   Factories that create a runtime may opt into ownership.

6. Return both positional `frame_results` and named `product_results`.
   `product_results` can be a cached derived view over `frame_results`.

7. Make loop-level RL explicit as out-of-workflow.
   The workflow is a batch/run abstraction. Future RL/action loops should reuse
   products and ticks directly rather than being forced through the batch
   workflow.

## P9 Baseline

Already implemented:

- `SimulationFrameTick`
- `simulation_frame_tick_from_published_frame(...)`
- `FrameProductResult`
- `FrameProduct`
- `MultiProductFrameRunner`
- `DebugFrameProduct`
- `PhysicsVideoFrameProduct`
- `PublishedStateObservationProduct`
- `frame_source=physics_published_frame`
- `clock_owner=external_physics_runtime`

The remaining issue is API shape. The P9 entry still reads as video-primary:

```text
run_physics_stepped_video_product_scenario(...)
  -> video product
  -> extra_products
  -> return video FrameTimingRecorder
```

P10 should make the workflow/product run primary, while preserving the existing
video helper as a compatibility path until it can become a thin adapter.

## API Principles

P10 should optimize for clarity and AI-era composability, not just fewer lines
of user code.

1. Intent first.
   The API should expose runtime, products, frame count, and output policy.

2. Explicit ownership.
   Live runtime construction and closing must be visible. Value-object configs
   must not secretly construct engines.

3. Instances before specs.
   P10.1 should accept already-built `FrameProduct` instances. Typed specs are
   convenience/factory surface and should not block the core proof.

4. Fail-fast product configuration.
   Observation still requires schema, q/v indices, and contact body ordering
   unless a reviewed scenario factory can derive them.

5. Internal runner stays internal.
   `MultiProductFrameRunner` proves ordering and lifecycle. Users should not
   have to orchestrate begin/step/end manually for normal workflows.

6. Batch workflow is not an RL loop.
   P10 workflow runs a fixed number of frames. Future action/reset/done loops
   will reuse ticks/products directly.

## User-Facing Levels

### Scenario-Level Batch User

Controls scenario intent, product set, frame count, and artifacts.

Target shape after specs/factories exist:

```python
result = run_optical_lab_workflow(
    scenario="physics_body_triangle_video_smoke",
    runtime=runtime,
    products=[
        VideoProductSpec(),
        DebugProductSpec(metadata_keys=None),
        ObservationProductSpec.from_scenario(
            scenario="physics_body_triangle_video_smoke",
            schema=schema,
        ),
    ],
    frames=120,
    output=ArtifactOutput(root=Path("runs/p10")),
)
```

P10.1 does not need to implement all of this. The important design point is that
typed specs/factories are the user-friendly layer, while the core workflow can
start with product instances.

### Runtime-Level Batch User

Already has a live runtime and products.

P10.1 target:

```python
with create_physics_body_triangle_lab_runtime(...) as runtime:
    result = PhysicsOwnedProductWorkflow(
        runtime=runtime,
        products=(video_product, observation_product, debug_product),
        output=ArtifactOutput(root=Path("runs/p10")),
        owns_runtime=False,
    ).run(frames=120)
```

or a thin convenience function:

```python
result = run_physics_product_workflow(
    runtime=runtime,
    products=(video_product, observation_product, debug_product),
    frames=120,
    output=ArtifactOutput(root=Path("runs/p10")),
)
```

The function may construct a temporary workflow internally. The workflow object
is still useful if it owns artifact setup, product construction, and result
assembly.

### Loop-Level RL / Interactive User

Not a P10 workflow user.

Future target:

```python
tick = runtime.step(action)
obs_result = observation_product.consume(tick)
action = policy(obs_result.payload["observation"])
```

This path should bypass `PhysicsOwnedProductWorkflow` because it is single-step
and action-driven. P10 must preserve reusable product instances so this later
loop does not need a different observation concept.

## Explicit Inputs

| Concept | P10.1 Required | Notes |
|---|---:|---|
| live runtime / tick source | yes | Must expose ownership and close semantics |
| products | yes | P10.1 accepts product instances |
| frames | yes | Use `frames: int`; defer schedule abstraction |
| output/artifact policy | yes for artifact runs | Rename `LabRunOptions` toward `ArtifactOutput` |
| observation schema | observation only | Must be explicit unless a scenario factory derives it |
| actuated q/v indices | joint observations | Keep fail-fast guards |
| contact body names | contact mask observations | Defines mask order |
| video config/camera/delivery | video only | Workflow should hide delivery facade construction |
| owns_runtime | optional | Default false |

## Proposed Types

### `ArtifactOutput`

P10.1 should stop growing the name `LabRunOptions` if the type is really
artifact/output policy.

Recommended path:

```python
@dataclass(frozen=True)
class ArtifactOutput:
    out: Path
    frames: int = 1
    progress_every: int = 0
    video_readback: str = "none"
    video_readback_delivery: str = "sync"
    video_readback_ring_depth: int = 2
    write_frames: bool = False
    ...
```

Compatibility option:

```python
LabRunOptions = ArtifactOutput
```

Open point: whether to do the rename in P10.1 or keep it as P10.2 if the diff is
too large. Claude recommends doing it in P10.1 to avoid dual configuration.

### `PhysicsOwnedProductWorkflow`

The workflow should hold one normalized product collection.

```python
@dataclass
class PhysicsOwnedProductWorkflow:
    runtime: PhysicsLabScenarioRuntime
    products: tuple[FrameProduct, ...]
    output: ArtifactOutput | None = None
    owns_runtime: bool = False

    def run(self, *, frames: int, env_idx: int = 0) -> PhysicsProductRunResult:
        ...

    def close(self) -> None:
        ...
```

Rules:

- no separate `product_specs` field;
- no `FixedFrameSchedule` in P10.1-P10.3;
- `run(...)` may be one-shot at first, but reusable semantics must be decided;
- if reusable, repeated `run(...)` must define whether product state is reset by
  `begin_run()`;
- if one-shot, enforce that with a clear error.

Codex recommendation: implement as reusable if product lifecycle reset is
already reliable; otherwise start one-shot and document it explicitly.

### `PhysicsProductRunResult`

```python
@dataclass(frozen=True)
class PhysicsProductRunResult:
    frame_results: tuple[tuple[FrameProductResult | None, ...], ...]
    begin_outputs: Mapping[str, object | None]
    end_outputs: Mapping[str, object | None]
    video_rows: FrameTimingRecorder | None = None
    artifacts: Mapping[str, object] = field(default_factory=dict)

    @cached_property
    def product_results(self) -> Mapping[str, tuple[FrameProductResult, ...]]:
        ...
```

Rationale:

- `frame_results` preserves ordering and missing-result slots;
- `product_results` is the user-friendly named lookup;
- deriving `product_results` avoids duplicated state;
- `video_rows` remains explicit because only video owns timing rows today.

### Product Specs

Defer broad specs until P10.4.

Only introduce `ObservationProductSpec.from_scenario(...)` earlier if it can be
implemented cleanly and genuinely reduces scenario-level friction without
guessing hidden state.

Target later:

```python
ObservationProductSpec.from_scenario(
    scenario="physics_body_triangle_video_smoke",
    schema=schema,
)
```

Rules for `.from_scenario(...)`:

- may derive actuated q/v indices only from reviewed robot/scenario metadata;
- may derive contact body names only from explicit scenario metadata;
- must fail fast if metadata is missing;
- must not infer contact order from private physics scratch.

## Internal Execution

P10 execution can stay close to P9:

```text
workflow.run(frames=N)
  -> runner = MultiProductFrameRunner(products)
  -> begin_outputs = runner.begin_run()
  -> for frame_index in range(N):
       tick = runtime.step_tick(frame_index)
       frame_results.append(runner.step(tick))
  -> end_outputs = runner.end_run()
  -> return PhysicsProductRunResult(...)
```

`MultiProductFrameRunner` keeps:

- product order;
- fail-fast begin/step/end;
- positional result slots.

Workflow owns:

- runtime ownership/closing;
- artifact/output policy;
- conversion from product lifecycle outputs to run result;
- optional compatibility with video rows.

Products own:

- video borrow/render/release/delivery;
- observation schema adaptation;
- debug capture.

## Non-Goals

P10 should not:

- create a public platform-wide `SimulationRuntime`;
- add a render backend;
- change physics core or solver pipeline;
- make `run_scenario(...)` construct live physics runtimes;
- require users to understand `MultiProductFrameRunner`;
- implement action/reset/done RL loop ownership;
- introduce a schedule protocol before a second schedule type exists.

## Implementation Slices

### P10.0: Approve V3 API Direction

Decide before code:

- workflow name: `PhysicsOwnedProductWorkflow`;
- workflow one-shot vs reusable semantics;
- whether `ArtifactOutput` rename lands in P10.1;
- whether P10.1 should be product instances only;
- result shape with cached `product_results`.

### P10.1: Workflow + Result With Product Instances

Implement:

- `PhysicsOwnedProductWorkflow`;
- `PhysicsProductRunResult`;
- `frames: int` run loop;
- video + debug + observation on the same physics-owned tick stream;
- `owns_runtime=False` default;
- no product specs except maybe a narrow observation scenario factory if reviewed.

Tests:

- one `step_tick` per frame;
- all products see same `frame_id` / `sim_time`;
- observation consumes `tick.published_frame`;
- video borrow/release remains inside video product;
- fail-fast behavior still comes from `MultiProductFrameRunner`;
- `product_results` groups named results from `frame_results`.

### P10.2: ArtifactOutput Rename / Compatibility

If not done in P10.1:

- rename `LabRunOptions` to `ArtifactOutput`;
- keep `LabRunOptions` as compatibility alias if needed;
- update tests and docs.

### P10.3: Function Convenience

Add:

```python
run_physics_product_workflow(...)
```

It should construct a temporary `PhysicsOwnedProductWorkflow` and return
`PhysicsProductRunResult`.

### P10.4: Product Specs

Add typed product specs and factories:

- `ObservationProductSpec`;
- `ObservationProductSpec.from_scenario(...)`;
- optional `VideoProductSpec`;
- optional `DebugProductSpec`.

Specs should build product instances and then use the same workflow path.

### P10.5: Scenario-Level Convenience

Add only after specs are clear:

```python
run_optical_lab_workflow(...)
```

This is a convenience layer, not the owner of core semantics.

## Validation Plan

Focused unit tests:

```bash
PYTHONPATH=. pytest -q tests/unit/optics/test_optical_pipeline_lab.py \
  -k "physics_product_workflow or multi_product or published_state_observation_product"
```

Warp-enabled lab regression:

```bash
conda run -n robot_sim env PYTHONPATH=. python -m pytest -q \
  tests/unit/optics/test_optical_pipeline_lab.py
```

GPU smoke only if P10 changes live GPU runtime behavior.

## Remaining Review Questions

1. Should `PhysicsOwnedProductWorkflow.run(...)` be one-shot or reusable?

2. Should `ArtifactOutput` be a direct P10.1 rename of `LabRunOptions`, or should
   the rename wait until after workflow/result lands?

3. Should P10.1 implement only product instances, or include
   `ObservationProductSpec.from_scenario(...)` if metadata is already available?

4. Should `run_physics_stepped_video_product_scenario(...)` remain as-is until
   P10.3, or become a thin adapter immediately?

## Codex Recommendation

Adopt v3 and implement P10.1 conservatively:

- workflow name: `PhysicsOwnedProductWorkflow`;
- product instances first;
- `frames: int`, no schedule abstraction;
- `PhysicsProductRunResult` with `frame_results` plus cached
  `product_results`;
- `owns_runtime=False` by default;
- keep loop-level RL outside the workflow;
- do not touch render backend code.

If the `LabRunOptions -> ArtifactOutput` rename is small after inspection, do it
in P10.1. If it fans out too much, defer to P10.2 but do not introduce a second
parallel output config type.

