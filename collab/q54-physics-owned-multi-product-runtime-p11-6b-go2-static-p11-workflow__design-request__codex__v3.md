Initiative: q54-physics-owned-multi-product-runtime-p11
Stage: design-request
Author: Codex
Version: v3
Date: 2026-07-21
Status: in_review
Related Files: collab/q54-physics-owned-multi-product-runtime-p11-6b-go2-static-p11-workflow__design-request__codex__v2.md, collab/q54-physics-owned-multi-product-runtime-p11-6b-go2-static-p11-workflow__design-review__claude__v1.md, tools/optical_pipeline_lab/product_workflow.py, tools/optical_pipeline_lab/product_specs.py, tools/optical_pipeline_lab/preset_workflows.py, tools/optical_pipeline_lab/preset_runtime.py, tools/optical_pipeline_lab/runner.py, tools/optical_pipeline_lab/frame_contexts.py
Owner Summary: V3 resolves the hidden blockers in the V2 design: the product workflow entry must stop routing all preset work through the physics-only scenario validator, and the video product builder must split generic video assembly from physics-specific render-runtime construction.

# Q54 P11.6b Design V3: Generic Product Workflow Entry For Static Go2

## Review Outcome

Claude's follow-up review on V2 is accepted. The V2 direction is still correct,
but it missed two implementation blockers already present in the code:

1. `run_optical_lab_preset(...)` currently reaches
   `run_physics_product_scenario(...)`, whose validator rejects static asset
   presets before any provider abstraction can run.
2. `VideoProductSpec.build(...)` currently reaches
   `build_physics_video_frame_product(...)`, which assumes
   `runtime.engine/runtime.registry/runtime.base_frame` and constructs a
   physics render runtime.

This V3 makes the missing workflow and builder split explicit.

## Final Public API Target

Default Go2 static preset usage should require only normal workflow arguments:

```python
from pathlib import Path
from tools.optical_pipeline_lab import run_optical_lab_preset

result = run_optical_lab_preset(
    preset="go2_video_ordered_static",
    frames=120,
    products=("video", "debug"),
    out=Path("runs/p11/go2"),
)
```

Advanced scene overrides are explicit runtime overrides:

```python
result = run_optical_lab_preset(
    preset="go2_video_ordered_static",
    frames=120,
    products=("video",),
    out=Path("runs/p11/go2_custom"),
    runtime_kwargs={
        "model_dir": "out/external/mujoco_menagerie/unitree_go2",
        "model_xml": "go2.xml",
    },
)
```

`ArtifactOutput.model_dir/model_xml` are not part of the new P11 static preset
contract.

## Decision 1: Add A Generic Product Scenario Runner

Current call chain:

```text
run_optical_lab_preset(...)
  -> run_optical_lab_products(...)
  -> run_optical_lab_workflow(...)
  -> run_physics_product_scenario(...)
  -> validate_physics_product_scenario(...)
```

That cannot support `go2_video_ordered_static` because static presets have:

```text
frame_source = static_asset_builder
clock_owner = runner
```

The fix is not to bypass P10 from P11. Instead, make P10's convenience workflow
generic and keep the physics-specific API as a specialization.

Add:

```python
def run_lab_product_scenario(
    config: OpticalLabScenarioConfig,
    output: ArtifactOutput,
    *,
    runtime: object,
    products: Iterable[object],
    frames: int | None = None,
    owns_runtime: bool = False,
) -> ProductRunResult:
    ...
```

Validation:

```python
def validate_lab_product_scenario(config, output) -> None:
    validate_run(config, output)
```

Behavior:

- resolves frame count;
- validates product inputs;
- writes `scenario_config.json`;
- builds concrete products from `ProductBuildContext`;
- runs the shared `ProductWorkflowRunner`;
- closes owned runtime on setup/run failures.

Then change:

```python
run_optical_lab_workflow(...)
  -> run_lab_product_scenario(...)
```

Keep:

```python
run_physics_product_scenario(...)
  -> validate_physics_product_scenario(...)
  -> run_lab_product_scenario(...)
```

This preserves existing physics-specific fail-fast semantics while allowing
P11 public workflows to use static runtime owners.

Naming compatibility:

```python
ProductRunResult = PhysicsProductRunResult
ProductWorkflowRunner = PhysicsOwnedProductWorkflow
```

Do not remove the old names in this slice.

## Decision 2: Split Generic Video Product Assembly From Physics Runtime Construction

Current helper:

```python
build_physics_video_frame_product(...)
```

does two things:

1. Builds a physics render runtime from `scenario_runtime.engine`,
   `scenario_runtime.registry`, and `scenario_runtime.base_frame`.
2. Assembles the video frame product, delivery facade, row recorder, and camera
   plan inputs.

For static Go2, step 1 is different and step 2 is the same.

Add a generic helper:

```python
def build_video_frame_product(
    config: OpticalLabScenarioConfig,
    options: ArtifactOutput,
    *,
    runtime: object,
    scene: object,
    frame_provider: TickFrameContextProvider,
    build_video_camera: Callable[[object, object, int], object],
    synchronize_event: Callable[[object], None],
    pack_rgb8: Callable[[object], object],
    consumer_id: str = "optical_lab_video_product",
    product_name: str = "video",
) -> VideoFrameProduct:
    ...
```

This helper owns only generic assembly:

- build video args;
- create `VideoDeliveryFacade`;
- create `FrameTimingRecorder`;
- create `VideoFrameProduct`.

Keep a physics wrapper:

```python
def build_physics_video_frame_product(...):
    validate_physics_video_product_run(...)
    physics_render_runtime = create_physics_render_runtime_for_config(...)
    frame_provider = physics_tick_frame_context_provider(...)
    return build_video_frame_product(
        ...,
        runtime=physics_render_runtime,
        scene=physics_render_runtime.pipeline.session.scene,
        frame_provider=frame_provider,
        ...
    )
```

Add a static path:

```python
def build_static_video_frame_product(...):
    frame_provider = static_tick_frame_context_provider(runtime.pipeline)
    return build_video_frame_product(
        ...,
        runtime=runtime,
        scene=runtime.pipeline.session.scene,
        frame_provider=frame_provider,
        ...
    )
```

Review note: the static helper may be avoided if `VideoProductSpec` supplies
`scene_factory` and `frame_provider_factory` directly to the generic helper.
The important boundary is that `VideoFrameProduct` is generic and
`create_physics_render_runtime_for_config(...)` is not called for static runs.

## Decision 3: Use Tick-Based Frame Providers

Create product-facing wrappers in:

```text
tools/optical_pipeline_lab/frame_providers.py
```

Protocol:

```python
class TickFrameContextProvider(Protocol):
    def begin_frame_for_tick(self, tick: SimulationFrameTick):
        """Return a context manager yielding an OpticalLabRenderFrameContext."""
```

Wrappers:

```python
@dataclass(frozen=True)
class PhysicsTickFrameContextProvider:
    provider: PhysicsFrameContextProvider

    def begin_frame_for_tick(self, tick: SimulationFrameTick):
        return self.provider.begin_frame(
            tick.frame_index,
            env_idx=tick.env_idx,
            published_frame=tick.published_frame,
        )


@dataclass(frozen=True)
class StaticTickFrameContextProvider:
    provider: StaticFrameContextProvider

    def begin_frame_for_tick(self, tick: SimulationFrameTick):
        return self.provider.begin_frame(
            tick.frame_index,
            env_idx=tick.env_idx,
        )
```

`VideoFrameProduct.consume(tick)` calls only:

```python
with self.frame_provider.begin_frame_for_tick(tick) as frame_context:
    ...
```

## Decision 4: Runtime Factory Signature

V2 proposed a workflow-aware runtime factory that accepted `preset`, `config`,
and `output`. Claude correctly pointed out the redundancy around `config`.

Use this signature instead:

```python
def create_runtime_for_lab_workflow(
    preset: str,
    *,
    output: ArtifactOutput,
    device: str | None = None,
    runtime_kwargs: Mapping[str, object] | None = None,
) -> object:
    ...
```

The factory resolves the preset config internally when needed.

Why still pass `output`:

- `output.fps` is a run-level fact needed by `StaticAssetLabRuntime.step_tick`
  to assign `sim_time`.
- `output.video_*` values are not scene defaults, but they may influence
  runtime construction choices later.
- `output` is already resolved before runtime ownership begins.

What must not happen:

- static runtime construction must not read `output.model_dir/model_xml` as the
  default source of asset paths.

Static scene defaults come from preset runtime config:

```python
_STATIC_ASSET_PRESET_DEFAULTS = {
    "go2_video_ordered_static": {
        "scene_preset": "go2_menagerie_static",
        "model_dir": "out/external/mujoco_menagerie/unitree_go2",
        "model_xml": "go2.xml",
    },
}
```

Apply `runtime_kwargs` as explicit overrides.

## Decision 5: Product Spec Provider Injection

Do not put `if frame_source == ...` in `VideoProductSpec.build(...)`.

Extend `VideoProductSpec` with build hooks:

```python
@dataclass(frozen=True)
class VideoProductSpec:
    build_video_camera: Callable[[object, object, int], object]
    synchronize_event: Callable[[object], None]
    pack_rgb8: Callable[[object], object]
    frame_provider_factory: Callable[[ProductBuildContext], TickFrameContextProvider]
    scene_factory: Callable[[ProductBuildContext], object]
    consumer_id: str = "optical_lab_video_product"
    product_name: str = "video"

    def build(self, context: ProductBuildContext) -> FrameProduct:
        return build_video_frame_product(
            context.config,
            context.output,
            runtime=context.runtime,
            scene=self.scene_factory(context),
            frame_provider=self.frame_provider_factory(context),
            build_video_camera=self.build_video_camera,
            synchronize_event=self.synchronize_event,
            pack_rgb8=self.pack_rgb8,
            consumer_id=self.consumer_id,
            product_name=self.product_name,
        )
```

Preset factories decide what provider is used:

```python
create_physics_body_triangle_video_product_spec()
  -> physics tick provider factory

create_go2_video_ordered_static_product_spec()
  -> static tick provider factory
```

## Decision 6: ArtifactOutput Field Fate

`ArtifactOutput` currently includes:

```text
model_dir
model_xml
```

V3 decision:

- keep these fields during P11.6b for legacy compatibility with
  `build_menagerie_example_args(...)`, `run_scenario(...)`, and
  `menagerie_static_runner.py`;
- do not use them as the default source of Go2 P11 static preset asset paths;
- mark them in docs/comments as legacy Menagerie CLI compatibility fields;
- prefer `runtime_kwargs={"model_dir": ..., "model_xml": ...}` for P11 preset
  overrides;
- remove or relocate the fields in a later cleanup after legacy static CLI
  options are separated from `ArtifactOutput`.

This avoids a broad API break while preventing new P11 code from depending on
the mixed responsibility.

## Revised Implementation Slices

### P11.6b-2: Generic Workflow Entry

- add `ProductRunResult` and `ProductWorkflowRunner` aliases;
- add `run_lab_product_scenario(...)`;
- add `validate_lab_product_scenario(...)`;
- change `run_optical_lab_workflow(...)` to call the generic runner;
- keep `run_physics_product_scenario(...)` as a physics-validating wrapper.

### P11.6b-3: Provider + Generic Video Product

- add `frame_providers.py`;
- generalize `PhysicsVideoFrameProduct` to `VideoFrameProduct`;
- keep `PhysicsVideoFrameProduct = VideoFrameProduct`;
- split `build_video_frame_product(...)` from the physics render-runtime wrapper.

### P11.6b-4: Static Runtime + Go2 Product Registration

- add `StaticAssetLabRuntime`;
- add `create_runtime_for_lab_workflow(...)`;
- add Go2 static scene defaults;
- register `"video"` for `go2_video_ordered_static`;
- update `run_optical_lab_preset(...)` to resolve output, create workflow
  runtime, then delegate to `run_optical_lab_products(...)` with explicit
  config/runtime.

### P11.6b-5: Delete Shim

- migrate generic tests away from `go2_backend`;
- delete `tools/optical_pipeline_lab/go2_backend.py`;
- update MANIFEST and design docs.

### P11.6b-6: Real Go2 Render Smoke

- run `run_optical_lab_preset("go2_video_ordered_static", ...)` in the Warp
  conda environment;
- keep RGB readback for end-to-end delivery verification.

## Tests To Add Or Update

1. `test_run_optical_lab_workflow_accepts_static_asset_runtime`
   - verifies static config no longer trips physics-only validation.

2. `test_run_physics_product_scenario_still_rejects_static_asset_config`
   - preserves existing specialized fail-fast behavior.

3. `test_video_frame_product_uses_tick_frame_provider`
   - proves render/delivery path is provider-driven.

4. `test_video_product_spec_uses_provider_factory_without_frame_source_dispatch`
   - uses fake factories and confirms no physics runtime attributes are
     required by the generic build path.

5. `test_create_runtime_for_lab_workflow_uses_static_preset_defaults`
   - default Go2 asset paths come from preset config.

6. `test_create_runtime_for_lab_workflow_allows_runtime_kwargs_overrides`
   - advanced users can override `model_dir/model_xml`.

7. `test_artifact_output_model_paths_are_not_used_for_p11_static_defaults`
   - prevents accidental regression to mixed responsibilities.

8. `test_go2_backend_module_is_deleted`
   - after the shim deletion slice.

## Real Smoke Command

```bash
conda run -n robot_sim env PYTHONPATH=. python - <<'PY'
from pathlib import Path
from tools.optical_pipeline_lab import run_optical_lab_preset

run_optical_lab_preset(
    preset="go2_video_ordered_static",
    frames=1,
    products=("video", "debug"),
    out=Path("out/p11_go2_static_smoke"),
)
PY
```

Expected artifacts:

```text
out/p11_go2_static_smoke/scenario_config.json
out/p11_go2_static_smoke/frame_timing.csv
```

Use runtime kwargs only if the Menagerie asset is not at the default preset
path.

## Open Questions For Review

1. Is `run_lab_product_scenario(...)` the right generic name, or should it be
   `run_product_scenario(...)` inside the lab package?
2. Should `create_runtime_for_lab_workflow(...)` live in `preset_runtime.py`
   initially, or should static runtime construction get a new
   `static_runtime.py` from the start?
3. Should `ArtifactOutput.model_dir/model_xml` emit deprecation warnings now,
   or should they remain silent compatibility fields until the legacy CLI path
   is separated?
