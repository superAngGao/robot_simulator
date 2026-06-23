Initiative: q54-physics-owned-multi-product-runtime-p10
Stage: review-request
Author: Codex
Version: v4
Date: 2026-06-23
Status: in_review
Related Files: collab/q54-physics-owned-multi-product-runtime-p10__review-request__codex__v3.md, collab/q54-multi-product-runtime-p9__review-request__codex__v1.md, tools/optical_pipeline_lab/frame_products.py, tools/optical_pipeline_lab/physics_runtime.py, tools/optical_pipeline_lab/runner.py
Owner Summary: V4 resolves the P10 v3 critical review issues. `ArtifactOutput` consistently uses `root`, product result names come from the existing `FrameProduct.product_name` contract, P10.1 workflow is one-shot, generic run results avoid top-level `video_rows`, and `env_idx` is deferred out of the workflow API for now.

# Q54 P10 Physics-Owned Product Workflow / V4

## Owner Summary

P10 should implement a narrow, one-shot, physics-owned batch workflow:

```text
explicit live physics runtime
  -> product instances with unique product_name
  -> fixed frames: int
  -> ArtifactOutput(root=...)
  -> PhysicsProductRunResult
```

This keeps the API clear for human and AI-generated callers while avoiding early
abstractions that do not yet have a second use case.

V4 makes five binding decisions before code:

1. `ArtifactOutput` uses `root`, not `out`.
2. Product identity comes from `FrameProduct.product_name`.
3. `PhysicsOwnedProductWorkflow` is one-shot in P10.1.
4. Video-specific outputs live in product outputs/artifacts, not a top-level
   `video_rows` field.
5. `env_idx` stays out of the P10.1 workflow API.

## Review Issues Resolved

### 1. ArtifactOutput Field Name

Use `root` consistently:

```python
@dataclass(frozen=True)
class ArtifactOutput:
    root: Path
    progress_every: int = 0
    video_readback: str = "none"
    video_readback_delivery: str = "sync"
    video_readback_ring_depth: int = 2
    write_frames: bool = False
```

Examples must also use:

```python
ArtifactOutput(root=Path("runs/p10"))
```

Compatibility rule:

- Do not use `LabRunOptions = ArtifactOutput` if field names differ.
- If P10.1 renames `LabRunOptions`, either do a hard cutover where usage is
  limited or provide an explicit compatibility constructor that maps `out` to
  `root`.
- Do not keep two parallel output configuration types.

### 2. Product Identification

P9 already defines the key identity field:

```python
class FrameProduct(Protocol):
    product_name: str
```

P10 should use this contract. `PhysicsOwnedProductWorkflow` accepts:

```python
products: tuple[FrameProduct, ...]
```

Rules:

- each product must expose `product_name`;
- product names must be unique;
- duplicate validation can reuse `MultiProductFrameRunner`;
- `PhysicsProductRunResult.product_results` groups by `result.product_name`;
- no class-name derivation;
- no `Mapping[str, FrameProduct]` as the primary API in P10.1.

This preserves ordered product execution while giving the user named lookups.

### 3. Workflow Reusability

P10.1 workflow is one-shot.

Reason:

- reusable workflows require artifact overwrite/subdirectory policy;
- reusable workflows require product reset semantics across runs;
- reusable workflows need additional tests before they are trustworthy.

P10.1 behavior:

```python
workflow = PhysicsOwnedProductWorkflow(...)
result = workflow.run(frames=120)
workflow.run(frames=120)  # raises RuntimeError
```

Required error:

```text
PhysicsOwnedProductWorkflow has already run
```

Reusability may become P10.2+ only after defining:

- whether artifacts overwrite or allocate a new run directory;
- whether product state is reset by `begin_run()`;
- whether repeated runs share or replace output manifests.

### 4. Generic Run Result

Do not put `video_rows` at the top level of the generic run result.

Use:

```python
@dataclass(frozen=True)
class PhysicsProductRunResult:
    frame_results: tuple[tuple[FrameProductResult | None, ...], ...]
    begin_outputs: Mapping[str, object | None]
    end_outputs: Mapping[str, object | None]
    artifacts: Mapping[str, object] = field(default_factory=dict)

    @cached_property
    def product_results(self) -> Mapping[str, tuple[FrameProductResult, ...]]:
        ...
```

Video-specific outputs belong in:

```python
result.end_outputs["video"]
result.artifacts["video_timing_csv"]
result.artifacts["video_frames_dir"]
```

If the existing video product returns a `FrameTimingRecorder` from `end_run()`,
the workflow should leave it under `end_outputs["video"]` or copy only the path
into `artifacts`.

### 5. env_idx

Remove `env_idx` from the P10.1 workflow API:

```python
def run(self, *, frames: int) -> PhysicsProductRunResult:
    ...
```

Reason:

- current P10 proof is single-env;
- vectorized env selection has not yet been designed;
- exposing `env_idx` now creates API weight without a user-facing story.

Internal calls may continue to rely on `PhysicsLabScenarioRuntime.step_tick(...)`
defaults. Add `env_idx` only when vectorized runtime/product semantics are
designed.

## P10.1 API Shape

### Runtime-Level Batch User

```python
with create_physics_body_triangle_lab_runtime(...) as runtime:
    observation_product = PublishedStateObservationProduct(
        engine=runtime.engine,
        schema=schema,
        actuated_q_indices=actuated_q_indices,
        actuated_v_indices=actuated_v_indices,
        contact_body_names=("left_foot", "right_foot"),
    )
    workflow = PhysicsOwnedProductWorkflow(
        runtime=runtime,
        products=(video_product, observation_product, debug_product),
        output=ArtifactOutput(root=Path("runs/p10")),
        owns_runtime=False,
    )
    result = workflow.run(frames=120)
```

This example intentionally shows observation construction. P10.1 should not hide
schema or joint/contact metadata.

### Scenario-Level Batch User

Scenario-level specs/factories are deferred until P10.4:

```python
ObservationProductSpec.from_scenario(...)
VideoProductSpec(...)
DebugProductSpec(...)
```

Those specs should build product instances and then call the same workflow path.

### Loop-Level User

Loop-level RL/action code does not use `PhysicsOwnedProductWorkflow`:

```python
tick = runtime.step(action)
obs_result = observation_product.consume(tick)
action = policy(obs_result.payload["observation"])
```

P10 must preserve reusable products and `SimulationFrameTick`, but workflow is a
batch abstraction, not the final action loop API.

## Proposed Types

### `ArtifactOutput`

```python
@dataclass(frozen=True)
class ArtifactOutput:
    root: Path
    frames: int = 1
    progress_every: int = 0
    video_readback: str = "none"
    video_readback_delivery: str = "sync"
    video_readback_ring_depth: int = 2
    write_frames: bool = False
```

Open implementation detail:

- `frames` may remain on `ArtifactOutput` only if this is a direct
  `LabRunOptions` rename.
- The workflow `run(frames=...)` parameter is authoritative in P10.1.
- If both exist during transition, tests must assert they cannot silently
  conflict.

### `PhysicsOwnedProductWorkflow`

```python
@dataclass
class PhysicsOwnedProductWorkflow:
    runtime: PhysicsLabScenarioRuntime
    products: tuple[FrameProduct, ...]
    output: ArtifactOutput | None = None
    owns_runtime: bool = False
    _has_run: bool = field(default=False, init=False)

    def run(self, *, frames: int) -> PhysicsProductRunResult:
        if self._has_run:
            raise RuntimeError("PhysicsOwnedProductWorkflow has already run")
        self._has_run = True
        ...

    def close(self) -> None:
        if self.owns_runtime:
            self.runtime.close()
```

Rules:

- one-shot in P10.1;
- product instances only;
- no `product_specs` field;
- no `env_idx` parameter;
- no schedule abstraction;
- `owns_runtime=False` by default.

### `PhysicsProductRunResult`

```python
@dataclass(frozen=True)
class PhysicsProductRunResult:
    frame_results: tuple[tuple[FrameProductResult | None, ...], ...]
    begin_outputs: Mapping[str, object | None]
    end_outputs: Mapping[str, object | None]
    artifacts: Mapping[str, object] = field(default_factory=dict)

    @cached_property
    def product_results(self) -> Mapping[str, tuple[FrameProductResult, ...]]:
        grouped: dict[str, list[FrameProductResult]] = {}
        for frame in self.frame_results:
            for result in frame:
                if result is None:
                    continue
                grouped.setdefault(result.product_name, []).append(result)
        return {name: tuple(values) for name, values in grouped.items()}
```

## Internal Execution

```text
workflow.run(frames=N)
  -> validate one-shot state
  -> runner = MultiProductFrameRunner(products)
  -> begin_outputs = runner.begin_run()
  -> for frame_index in range(N):
       tick = runtime.step_tick(frame_index)
       frame_results.append(runner.step(tick))
  -> end_outputs = runner.end_run()
  -> artifacts = collect paths/manifests only
  -> return PhysicsProductRunResult(...)
```

`MultiProductFrameRunner` remains the ordered internal executor. It already
validates duplicate `product_name` values and preserves positional result slots.

## P10.1 Implementation Scope

Implement:

- `ArtifactOutput` rename or compatibility plan with `root`;
- `PhysicsOwnedProductWorkflow`;
- `PhysicsProductRunResult`;
- one-shot `run(frames=...)`;
- product instances only;
- video + debug + observation same tick stream test;
- no render backend changes.

Do not implement yet:

- `FixedFrameSchedule`;
- product spec hierarchy;
- `ObservationProductSpec.from_scenario(...)`;
- reusable workflow;
- loop-level RL/action API;
- top-level scenario convenience API.

## Required Tests

Focused P10.1 tests:

- workflow runs exactly one `step_tick` per frame;
- video/debug/observation receive the same `frame_id` and `sim_time`;
- observation product construction explicitly passes schema/q indices/v indices;
- `product_results` groups from `frame_results` by `product_name`;
- duplicate product names still fail fast;
- second `workflow.run(...)` raises `RuntimeError`;
- `owns_runtime=True` closes the runtime when workflow closes;
- `owns_runtime=False` does not close it;
- video timing outputs stay in `end_outputs["video"]` or artifact paths, not a
  top-level generic field.

Validation commands:

```bash
PYTHONPATH=. pytest -q tests/unit/optics/test_optical_pipeline_lab.py \
  -k "physics_product_workflow or multi_product or published_state_observation_product"
```

Warp-enabled regression:

```bash
conda run -n robot_sim env PYTHONPATH=. python -m pytest -q \
  tests/unit/optics/test_optical_pipeline_lab.py
```

## Remaining Open Questions

1. Should `ArtifactOutput` include `frames`, or should frame count live only on
   `workflow.run(frames=...)` after the `LabRunOptions` rename?

2. Should P10.1 hard-cut from `LabRunOptions(out=...)` to
   `ArtifactOutput(root=...)`, or provide a short compatibility constructor?

3. Should `run_physics_stepped_video_product_scenario(...)` remain unchanged
   until after workflow/result lands, or become an adapter in P10.1?

## Codex Recommendation

Proceed with P10.1 after this v4 review:

- product instances only;
- `PhysicsOwnedProductWorkflow` one-shot;
- `ArtifactOutput(root=...)`;
- no `env_idx`;
- generic `PhysicsProductRunResult` without top-level `video_rows`;
- derive named `product_results` from `FrameProductResult.product_name`;
- defer specs, schedules, reusable runs, and scenario-level convenience.

