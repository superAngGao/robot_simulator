Initiative: q54-physics-owned-multi-product-runtime-p11
Stage: design-request
Author: Codex
Version: v1
Date: 2026-07-13
Status: in_review
Related Files: collab/q54-physics-owned-multi-product-runtime-p10__review-request__codex__v4.md, tools/optical_pipeline_lab/product_workflow.py, tools/optical_pipeline_lab/product_specs.py, tools/optical_pipeline_lab/physics_runtime.py, tools/optical_pipeline_lab/presets.py
Owner Summary: P11 should add the user-facing preset workflow layer above P10. It should make the official Optical Pipeline Lab path easy to call, while keeping physics as the stepping owner and keeping P10 as the reusable workflow/product composition layer.

# Q54 P11 Preset-Level Optical Lab Workflow

## Owner Summary

P10 established the correct internal boundary:

```text
explicit live physics runtime
  -> product specs / product instances
  -> run_optical_lab_products(...)
  -> PhysicsOwnedProductWorkflow
  -> PhysicsProductRunResult
```

P11 should add the next layer up:

```text
user preset request
  -> registered runtime factory
  -> registered product selection
  -> run_optical_lab_products(...)
  -> PhysicsProductRunResult
```

P11 is not a new runner, not a new physics runtime abstraction, and not a render
backend project. It is the official preset-level user workflow for lab and
benchmark callers.

## Review Decisions

Claude review accepted the direction and made the following binding decisions:

1. Keep both entry points:
   - `run_optical_lab_preset(...)` for P11 preset-level users.
   - `run_optical_lab_products(...)` for P10 advanced composition.
2. Support string products immediately for `"video"` and `"debug"`.
3. Initially support only reviewed presets, starting with
   `physics_body_triangle_video_smoke`.
4. Examples should be executable, but include `--dry-run` so tests can validate
   imports and argument parsing without GPU work.
5. Keep `device` as a flat common argument and group advanced runtime options in
   `runtime_kwargs`.

These decisions are reflected in the API and slice plan below.

The primary user should be able to write:

```python
from pathlib import Path

from tools.optical_pipeline_lab.preset_workflows import run_optical_lab_preset

result = run_optical_lab_preset(
    preset="physics_body_triangle_video_smoke",
    frames=120,
    products=("video", "debug"),
    out=Path("runs/p11/body_triangle"),
    device="cuda:0",
)
```

Advanced users should still be able to bypass P11 and use the P10 primitives
directly:

```python
with create_runtime_for_lab_preset("physics_body_triangle_video_smoke") as runtime:
    result = run_optical_lab_products(
        preset="physics_body_triangle_video_smoke",
        runtime=runtime,
        frames=120,
        products=(video_spec, debug_spec, observation_spec),
        out=Path("runs/manual"),
    )
```

## Layering

P11 should keep a clear file/module boundary.

Recommended structure:

```text
tools/optical_pipeline_lab/
  product_workflow.py    # P10: workflow core and low-level helpers
  product_specs.py       # P10: Video/Debug/Observation specs
  preset_runtime.py      # P11: preset -> runtime factory
  preset_products.py     # P11: product string/spec resolution
  preset_workflows.py    # P11: main user-facing preset workflow API
  presets.py             # existing scenario config presets
```

Responsibilities:

- `product_workflow.py`: keep P10 one-shot workflow semantics and result shape.
- `product_specs.py`: keep reusable product specs that build concrete products.
- `preset_runtime.py`: create live physics runtimes for explicitly registered
  lab presets.
- `preset_products.py`: translate user product selections into specs/instances.
- `preset_workflows.py`: compose preset config, runtime, products, artifacts,
  and P10 execution.

Do not keep growing `product_workflow.py` with all future user-facing entry
points. P10 modules should remain reusable building blocks; P11 modules should
be the public lab orchestration layer.

## Design References

P11 follows established patterns from the project reference matrix:

- **Gymnasium env registry**: `gym.make("HalfCheetah-v4")` proves that string
  identifiers are a good public entry point when they resolve through an
  explicit registry. P11 uses the same idea, but keeps product/runtime choices
  stricter: unsupported presets and products fail fast instead of being guessed.
- **Isaac Lab config workflow**: Isaac Lab's manager/config style separates
  user-facing experiment configuration from lower-level simulation execution.
  P11 mirrors that separation: preset/product selection lives above P10's
  product workflow and above the physics runtime owner.
- **MuJoCo benchmark/preset style**: MuJoCo examples and benchmarks are
  commonly driven from named models/scenes plus output configuration. P11 uses a
  similar preset-level workflow, while keeping product outputs more flexible via
  P10 product specs.

The key design choice is explicit registration over inference: P11 should feel
easy to call, but every default it supplies must be reviewed and named.

## Public API

### `run_optical_lab_preset(...)`

Main P11 entry point:

```python
def run_optical_lab_preset(
    *,
    preset: str,
    frames: int,
    products: Iterable[str | ProductSpec | FrameProduct],
    out: Path | None = None,
    output: ArtifactOutput | None = None,
    device: str | None = None,
    owns_runtime: bool = True,
    runtime_kwargs: Mapping[str, object] | None = None,
) -> PhysicsProductRunResult:
    ...
```

Expected behavior:

```text
run_optical_lab_preset(...)
  -> config = get_preset(preset)
  -> runtime = create_runtime_for_lab_preset(preset, device=..., **runtime_kwargs)
  -> product_specs = resolve_lab_product_specs(preset=preset, products=products)
  -> run_optical_lab_products(config=config, runtime=runtime, products=product_specs, ...)
  -> close runtime when owned
  -> return PhysicsProductRunResult
```

Rules:

- `preset` is required.
- `frames` is required at this level. P11 is a batch artifact workflow, not an
  open-ended loop.
- `out` and `output` are mutually compatible only when they point to the same
  root, following the existing P10 `ArtifactOutput` behavior.
- Default `owns_runtime=True`, because this entry point creates the runtime.
- Return the existing `PhysicsProductRunResult`; do not introduce a new result
  type.

### `create_runtime_for_lab_preset(...)`

Runtime factory:

```python
def create_runtime_for_lab_preset(
    preset: str,
    *,
    device: str | None = None,
    **kwargs: object,
) -> PhysicsLabScenarioRuntime:
    ...
```

Initial support:

```text
physics_body_triangle_video_smoke
```

This should call the reviewed runtime factory:

```python
create_physics_body_triangle_lab_runtime(...)
```

Rules:

- Only create runtimes for explicitly registered presets.
- Unsupported presets fail fast with a clear `NotImplementedError`.
- Do not infer runtime factories from arbitrary scenario config fields.
- Do not make lab the low-level physics owner. The created runtime still owns
  stepping; P11 only owns the preset-level lifecycle.

### `resolve_lab_product_specs(...)`

Product selection helper:

```python
def resolve_lab_product_specs(
    *,
    preset: str,
    products: Iterable[str | ProductSpec | FrameProduct],
) -> tuple[ProductSpec | FrameProduct, ...]:
    ...
```

Supported string products in P11.1:

```text
debug
video
```

Rules:

- `"debug"` resolves to `DebugProductSpec()`.
- `"video"` resolves to a preset-specific `VideoProductSpec(...)` only when the
  preset has reviewed default video dependencies.
- Existing `ProductSpec` values pass through.
- Existing `FrameProduct` instances pass through.
- Unknown strings raise `ValueError`.
- Duplicate product names should still fail through the existing P10
  `MultiProductFrameRunner` path after materialization.

Observation rule:

- Do not auto-create observation specs in P11.1.
- Users must pass `ObservationProductSpec(...)` or
  `ObservationProductSpec.from_scenario(...)`.
- P11 may add observation defaults only after the preset declares reviewed robot
  metadata such as actuated q/v indices and contact body names.

Example:

```python
products = resolve_lab_product_specs(
    preset="physics_body_triangle_video_smoke",
    products=("video", "debug"),
)
```

Mixed advanced example:

```python
products = resolve_lab_product_specs(
    preset="physics_body_triangle_video_smoke",
    products=(
        "video",
        "debug",
        ObservationProductSpec.from_scenario(
            get_preset("physics_body_triangle_video_smoke"),
            schema=schema,
            actuated_q_indices=actuated_q_indices,
            actuated_v_indices=actuated_v_indices,
            contact_body_names=("left_foot", "right_foot"),
        ),
    ),
)
```

## Product Selection Policy

P11 should permit string product selection because this is the layer most users
will call, including AI-generated callers. However, string selection must be
limited and explicit.

Allowed:

```python
products=("video", "debug")
```

Allowed:

```python
products=("video", DebugProductSpec(metadata_keys=None))
```

Allowed:

```python
products=("debug", ObservationProductSpec(...))
```

Not allowed in P11.1:

```python
products=("observation",)
```

Reason: observation requires explicit schema and robot metadata. Automatically
guessing those values would create a misleading API.

Required error message shape:

```python
raise ValueError(
    'Product string "observation" is not supported. '
    "Observation requires explicit robot metadata "
    "(schema, actuated indices, contact bodies). "
    "Use ObservationProductSpec.from_scenario(...) instead. "
    "See examples/optical_lab/physics_body_triangle_observation.py."
)
```

## Runtime Factory Policy

P11 can create and close runtimes, but only at the preset workflow level.

The relationship remains:

```text
physics runtime owns stepping
P11 owns preset-level setup/teardown
P10 owns product workflow execution
products consume SimulationFrameTick values
```

Initial registry:

```python
_RUNTIME_FACTORIES = {
    "physics_body_triangle_video_smoke": create_physics_body_triangle_lab_runtime,
}
```

The factory should accept a narrow set of keyword arguments first:

- `device`
- `initial_height`
- `height_for_frame`
- `dt`
- `bounds_min`
- `bounds_max`
- `metadata`
- `synchronize_event`

Do not add a generic `**config_to_runtime` inference layer.

## Examples

P11 should include examples because this is now the main user-facing layer.

Recommended files:

```text
examples/optical_lab/
  README.md
  physics_body_triangle_video_debug.py
  physics_body_triangle_observation.py
```

### `physics_body_triangle_video_debug.py`

Purpose: minimal official artifact workflow.

Sketch:

```python
import argparse
from pathlib import Path

from tools.optical_pipeline_lab.preset_workflows import run_optical_lab_preset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    preset = "physics_body_triangle_video_smoke"
    frames = 120
    products = ("video", "debug")
    out = Path("runs/examples/physics_body_triangle_video_debug")

    if args.dry_run:
        print(f"Would run: preset={preset} frames={frames} products={products} out={out}")
        return

    result = run_optical_lab_preset(
        preset=preset,
        frames=frames,
        products=products,
        out=out,
    )
    print(result.artifacts)
    print(sorted(result.product_results))


if __name__ == "__main__":
    main()
```

### `physics_body_triangle_observation.py`

Purpose: show why observation is explicit.

Sketch:

```python
import argparse
from pathlib import Path

import numpy as np

from rl_env.obs import locomotion_obs_schema
from tools.optical_pipeline_lab.preset_workflows import run_optical_lab_preset
from tools.optical_pipeline_lab.presets import get_preset
from tools.optical_pipeline_lab.product_specs import ObservationProductSpec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = get_preset("physics_body_triangle_video_smoke")
    schema = locomotion_obs_schema(
        num_actuated_joints=2,
        num_contact_bodies=2,
        include_contact_mask=True,
    )
    observation = ObservationProductSpec.from_scenario(
        config,
        schema=schema,
        actuated_q_indices=np.array([7, 8]),
        actuated_v_indices=np.array([6, 7]),
        contact_body_names=("left_foot", "right_foot"),
    )
    out = Path("runs/examples/physics_body_triangle_observation")

    if args.dry_run:
        print(
            "Would run: "
            f"preset={config.scenario_name} frames=120 products=('debug', observation) out={out}"
        )
        return

    result = run_optical_lab_preset(
        preset=config.scenario_name,
        frames=120,
        products=("debug", observation),
        out=out,
    )
    print(result.product_results["observation"][-1].payload["observation"])


if __name__ == "__main__":
    main()
```

The exact observation indices above are placeholders unless already validated
for the preset. If they are not reviewed metadata, the example should either use
the currently reviewed test values or be marked as illustrative only.

## Non-Goals

P11 should not:

- implement an RL action/reset/done loop;
- make `PhysicsOwnedProductWorkflow` reusable;
- add schedule abstractions;
- add render backends;
- implement render backend selection;
- add multi-preset batching or comparison workflows;
- infer physics runtimes from arbitrary scenario config fields;
- infer observation schema, joint indices, velocity indices, or contact body
  names;
- create a new result type;
- move Optical Pipeline Lab into a top-level production package;
- replace P10 advanced APIs.

## Error Handling

Required fail-fast behavior:

- unsupported preset in `create_runtime_for_lab_preset(...)`;
- unsupported product string in `resolve_lab_product_specs(...)`;
- `"observation"` string without explicit observation spec;
- conflicting `out` and `output.root`;
- negative `frames`;
- `products=()` is allowed only if P10 already allows empty product workflows;
  otherwise reject consistently at the P10 layer.

Runtime ownership:

- `run_optical_lab_preset(...)` creates the runtime and should close it on
  success or failure.
- If runtime creation fails, no close is required.
- If product resolution or workflow execution fails after runtime creation, the
  runtime must be closed.

Artifact behavior:

- Write `scenario_config.json` through the existing P10 path.
- Do not create a separate P11 manifest unless there is a concrete consumer.
- Do not put video-specific timing rows at top level; keep them in product
  outputs as P10 decided.

## Implementation Slices

### P11.1: Design Review

This document.

Claude review should focus on:

- module boundaries;
- public API names;
- product string policy;
- runtime factory policy;
- examples and non-goals.

### P11.2: Runtime Factory

Add:

```text
tools/optical_pipeline_lab/preset_runtime.py
```

Implement:

- `create_runtime_for_lab_preset(...)`;
- tests for supported preset;
- tests for unsupported preset;
- runtime close via context manager remains from `PhysicsLabScenarioRuntime`.

### P11.3: Product Selection

Add:

```text
tools/optical_pipeline_lab/preset_products.py
```

Implement:

- `resolve_lab_product_specs(...)`;
- string `"debug"`;
- string `"video"` for the reviewed smoke preset;
- explicit observation spec pass-through;
- unknown string fail-fast;
- `"observation"` string fail-fast with message requiring explicit spec and the
  observation example path.

### P11.4: Preset Runner

Add:

```text
tools/optical_pipeline_lab/preset_workflows.py
```

Implement:

- `run_optical_lab_preset(...)`;
- runtime ownership and cleanup;
- output/out handling by reusing `ArtifactOutput`;
- calls into P10 `run_optical_lab_products(...)`;
- basic tests for video/debug minimal path with fakes, not real GPU work;
- setup/teardown failure tests proving runtime cleanup.

### P11.5: Examples and Documentation

Add:

```text
examples/optical_lab/README.md
examples/optical_lab/physics_body_triangle_video_debug.py
examples/optical_lab/physics_body_triangle_observation.py
```

Tests should at least import examples or run their `--help` / dry path if they
support one. Full GPU execution can stay out of unit tests.

Example tests should live with the current optical lab tests unless a dedicated
examples test module becomes cleaner:

```python
def test_optical_lab_examples_dry_run():
    subprocess.run(
        [
            sys.executable,
            "examples/optical_lab/physics_body_triangle_video_debug.py",
            "--dry-run",
        ],
        check=True,
    )
```

## Review Questions

1. Should the main entry be named `run_optical_lab_preset(...)`, or should it
   be `run_optical_lab_products(...)` with runtime creation optional?

   Decision: use `run_optical_lab_preset(...)` for P11 and keep
   `run_optical_lab_products(...)` as the existing P10 advanced composition
   entry.

2. Should `products=("video", "debug")` be accepted immediately, or should P11
   require explicit specs until examples settle?

   Decision: accept `"video"` and `"debug"` immediately. This is the central
   user-facing value of P11, and both mappings are reviewed and narrow.

3. Should P11.1 support only `physics_body_triangle_video_smoke`, or should the
   runtime factory accept other physics-published presets if they appear in
   `presets.py`?

   Decision: support only reviewed presets, initially
   `physics_body_triangle_video_smoke`.

4. Should examples be executable by default, or should they include a dry-run
   mode so unit tests can exercise the user API without GPU work?

   Decision: examples should be executable by default and include `--dry-run`.
   Unit tests should exercise dry-run/import paths, not full GPU execution.

5. Should `run_optical_lab_preset(...)` expose runtime kwargs flatly, or group
   them under `runtime_options` / `runtime_kwargs`?

   Decision: expose common `device` flatly and group advanced runtime options
   under `runtime_kwargs`.

## Codex Recommendation

Proceed with P11 only after this design is reviewed.

Recommended binding decisions:

- add new P11 modules instead of expanding `product_workflow.py`;
- expose `run_optical_lab_preset(...)` as the main user-facing entry;
- keep `run_optical_lab_products(...)` as the P10 advanced composition entry;
- allow explicit string products for `"video"` and `"debug"`;
- require explicit observation specs;
- support only `physics_body_triangle_video_smoke` initially;
- add examples in the same P11 sequence, not after the fact.
