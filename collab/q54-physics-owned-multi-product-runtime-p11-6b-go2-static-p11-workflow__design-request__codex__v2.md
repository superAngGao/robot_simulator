Initiative: q54-physics-owned-multi-product-runtime-p11
Stage: design-request
Author: Codex
Version: v2
Date: 2026-07-20
Status: in_review
Related Files: collab/q54-physics-owned-multi-product-runtime-p11-6b-go2-static-p11-workflow__design-request__codex__v1.md, collab/q54-physics-owned-multi-product-runtime-p11-6b-go2-static-p11-workflow__design-review__claude__v1.md, tools/optical_pipeline_lab/frame_contexts.py, tools/optical_pipeline_lab/frame_products.py, tools/optical_pipeline_lab/product_specs.py, tools/optical_pipeline_lab/preset_runtime.py, tools/optical_pipeline_lab/preset_products.py, tools/optical_pipeline_lab/preset_workflows.py, tools/optical_pipeline_lab/runner.py
Owner Summary: Revises the P11.6b Go2 static workflow plan after Claude review. The key change is to avoid a duplicated static video product: make video product runtime-agnostic through a tick-based frame-context provider protocol, move Go2 asset defaults into preset/static scene config, and delete `go2_backend.py` only after Go2 runs through `run_optical_lab_preset(...)`.

# Q54 P11.6b Design V2: Go2 Static Workflow Without Video Product Duplication

## Review Outcome

Claude's v1 design review is accepted on the main architecture points:

1. Do not implement a separate `StaticVideoFrameProduct`.
2. Do not make `VideoProductSpec.build(...)` dispatch on `frame_source`.
3. Do not require normal Go2 preset users to pass `model_dir/model_xml` through
   `ArtifactOutput`.
4. Keep the deletion order: static P11 workflow first, delete
   `go2_backend.py` after tests and real smoke pass.

This v2 replaces the implementation plan from v1. It keeps the user-level goal
unchanged:

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

Advanced static asset overrides remain possible, but they are explicit
overrides, not required preset boilerplate:

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

## Corrected Architecture

### Layering

```text
run_optical_lab_preset(...)
  -> preset config
  -> preset runtime factory
  -> product spec resolver
  -> ProductWorkflowRunner
  -> VideoFrameProduct + Debug/Observation
```

Runtime differences are handled below the public API:

```text
physics preset:
  PhysicsLabScenarioRuntime.step_tick(...)
  PhysicsTickFrameContextProvider.begin_frame_for_tick(tick)

static Go2 preset:
  StaticAssetLabRuntime.step_tick(...)
  StaticTickFrameContextProvider.begin_frame_for_tick(tick)
```

The video product does not know which runtime mode it is using.

## 1. Tick-Based Frame Provider Protocol

Claude suggested a general `begin_frame(..., **kwargs)` provider. Codex agrees
with the abstraction but recommends a stricter tick-based method:

```python
class TickFrameContextProvider(Protocol):
    def begin_frame_for_tick(self, tick: SimulationFrameTick):
        """Return a context manager yielding an OpticalLabRenderFrameContext."""
```

Why not `**kwargs`:

- `SimulationFrameTick` is already the shared per-frame contract for products.
- The provider can decide whether `tick.published_frame` matters.
- A named `begin_frame_for_tick(...)` method makes product code simpler and
  avoids implicit keyword compatibility.

Provider behavior:

```text
PhysicsTickFrameContextProvider:
  uses tick.published_frame
  calls PhysicsFrameContextProvider.begin_frame(..., published_frame=...)

StaticTickFrameContextProvider:
  ignores tick.published_frame
  calls StaticFrameContextProvider.begin_frame(...)
```

This keeps the existing low-level provider methods intact while adding a stable
product-facing provider protocol.

Suggested module:

```text
tools/optical_pipeline_lab/frame_providers.py
```

Possible implementation:

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

## 2. Generalize `PhysicsVideoFrameProduct`

Rename or alias the current physics-specific product:

```text
PhysicsVideoFrameProduct -> VideoFrameProduct
```

`VideoFrameProduct` owns the generic render/delivery/CSV logic:

```text
begin_run()
consume(tick)
  -> frame_provider.begin_frame_for_tick(tick)
  -> build_video_render_plan(...)
  -> render_video_frame_from_context(...)
  -> delivery submit/complete
  -> FrameProductResult
end_run()
  -> delivery.flush()
  -> rows.write_csv()
```

Compatibility:

```python
PhysicsVideoFrameProduct = VideoFrameProduct
```

Keep the old name temporarily for existing tests/imports if needed, but new
code and docs should use `VideoFrameProduct`.

The only product-facing runtime dependency is:

```python
frame_provider: TickFrameContextProvider
scene: object
```

That means static, physics, synthetic, and future deformable/fluid runtimes can
share one video product.

## 3. Preset Carries Static Scene Defaults

Go2 asset paths are preset/static-scene configuration, not artifact output
configuration.

Add static scene defaults in a registry, likely in `preset_runtime.py` or a
small `static_runtime.py` helper:

```python
_STATIC_ASSET_PRESET_DEFAULTS = {
    "go2_video_ordered_static": {
        "scene_preset": "go2_menagerie_static",
        "model_dir": "out/external/mujoco_menagerie/unitree_go2",
        "model_xml": "go2.xml",
    },
}
```

`ArtifactOutput` remains about artifact and delivery options:

```text
root
frames/fps
readback delivery
frame output policy
```

It should no longer be the required source of `model_dir/model_xml` for normal
Go2 preset runs.

Advanced overrides:

```python
runtime_kwargs={"model_dir": "...", "model_xml": "..."}
```

This is explicit and keeps the default API consistent across physics and static
presets:

```python
run_optical_lab_preset(..., out=Path(...))
```

## 4. Static Runtime Owner

Add a static runtime that looks enough like the product workflow runtime:

```python
@dataclass
class StaticAssetLabRuntime:
    pipeline: OpticalLabRenderPipeline
    scene: object
    base_frame: object
    metadata: Mapping[str, object]
    fps: float

    def step_tick(self, frame_index: int, *, env_idx: int = 0) -> SimulationFrameTick:
        return SimulationFrameTick(
            frame_index=frame_index,
            env_idx=env_idx,
            frame_id=int(self.base_frame.frame_id) + int(frame_index),
            sim_time=float(self.base_frame.sim_time) + float(frame_index) / self.fps,
            published_frame=self.base_frame,
            metadata={
                **self.metadata,
                "frame_source": "static_asset_builder",
            },
        )

    def close(self) -> None:
        return None
```

The static runtime may reuse the same `base_frame` because the scene is static.
Frame identity still advances through the tick, and the static provider ignores
`tick.published_frame`.

Construction should use:

```text
static_asset_source.build_static_asset_render_source(...)
render_session.OpticalLabRenderPipeline.create_from_source_factory(...)
```

It must not call:

```text
menagerie_static_runner.render_many_views(...)
go2_backend.py
```

## 5. Runtime Factory Shape

The existing `create_runtime_for_lab_preset(...)` is physics-oriented and does
not receive `ArtifactOutput`. Rather than force all logic through it, add a
workflow-aware factory used by `run_optical_lab_preset(...)`:

```python
def create_runtime_for_lab_workflow(
    preset: str,
    *,
    config: OpticalLabScenarioConfig,
    output: ArtifactOutput,
    device: str | None = None,
    runtime_kwargs: Mapping[str, object] | None = None,
) -> object:
    ...
```

Behavior:

```text
physics-owned preset:
  delegates to create_runtime_for_lab_preset(...)

static asset preset:
  reads preset static defaults
  applies explicit runtime_kwargs overrides
  builds StaticAssetLabRuntime
```

Review point: this function can live in `preset_runtime.py` initially. If it
starts growing, split static construction into `static_runtime.py`.

## 6. `VideoProductSpec` Uses Provider Injection

Do not add central dispatch in `VideoProductSpec.build(...)`.

Instead, extend the spec with product assembly hooks:

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

Preset factories decide the runtime path:

```python
def create_physics_body_triangle_video_product_spec() -> VideoProductSpec:
    return VideoProductSpec(
        build_video_camera=build_lab_video_camera,
        synchronize_event=synchronize_ready_event,
        pack_rgb8=pack_video_rgb8,
        frame_provider_factory=physics_tick_frame_context_provider_for_context,
        scene_factory=lambda context: context.runtime.pipeline.session.scene,
    )


def create_go2_video_ordered_static_product_spec() -> VideoProductSpec:
    return VideoProductSpec(
        build_video_camera=build_lab_video_camera,
        synchronize_event=synchronize_ready_event,
        pack_rgb8=pack_video_rgb8,
        frame_provider_factory=static_tick_frame_context_provider_for_context,
        scene_factory=lambda context: context.runtime.pipeline.session.scene,
    )
```

No `if frame_source == ...` is needed inside `VideoProductSpec.build(...)`.

## 7. Product Workflow Naming

Introduce a generic alias:

```python
ProductRunResult = PhysicsProductRunResult
```

Optionally also:

```python
ProductWorkflowRunner = PhysicsOwnedProductWorkflow
```

Keep old names for compatibility in this slice. Do not do a broad rename unless
the implementation diff stays small.

## 8. Delete `go2_backend.py`

Deletion happens only after the static Go2 P11 path is tested.

Required changes:

- migrate generic tests away from `go2_backend`;
- delete `tools/optical_pipeline_lab/go2_backend.py`;
- remove the MANIFEST shim row;
- update `GPU_OPTICAL_PIPELINE_DESIGN.md` from "deprecated shim" to "deleted";
- update examples docs to say Go2 static has a P11 preset workflow path;
- keep `menagerie_static_runner.py` as legacy CLI/benchmark only.

## Tests

### Unit Tests

Required additions/changes:

1. `test_tick_frame_context_provider_uses_physics_published_frame`
   - verifies physics provider passes `tick.published_frame`.

2. `test_tick_frame_context_provider_ignores_static_published_frame`
   - verifies static provider does not require/use `published_frame`.

3. `test_video_frame_product_uses_tick_frame_provider`
   - one generic product test; no separate static/physics product duplicate.

4. `test_resolve_lab_product_specs_supports_go2_video`
   - `"video"` is registered for `go2_video_ordered_static`.

5. `test_run_optical_lab_preset_runs_go2_static_products_through_p11`
   - monkeypatch static runtime construction to avoid real GPU;
   - assert `scenario_config.json` is written;
   - assert runtime closes when owned;
   - assert no `menagerie_static_runner.render_many_views(...)` call.

6. `test_go2_backend_module_is_deleted`
   - path does not exist;
   - subprocess import fails.

7. Existing generic tests should import:
   - `video_loop` for video helper functions;
   - `render_session` for `OpticalLabRender*`;
   - `menagerie_static_runner` only for legacy CLI/benchmark behavior.

### Real Smoke

Use RGB readback for end-to-end verification:

```bash
conda run -n robot_sim env PYTHONPATH=. python - <<'PY'
from pathlib import Path
from tools.optical_pipeline_lab import run_optical_lab_preset

run_optical_lab_preset(
    preset="go2_video_ordered_static",
    frames=1,
    products=("video", "debug"),
    out=Path("out/p11_go2_static_smoke"),
    runtime_kwargs={
        "model_dir": "out/external/mujoco_menagerie/unitree_go2",
        "model_xml": "go2.xml",
    },
)
PY
```

The default path should also work without `runtime_kwargs` if the Menagerie
assets are present at the preset default location.

Expected artifacts:

```text
out/p11_go2_static_smoke/scenario_config.json
out/p11_go2_static_smoke/frame_timing.csv
```

## Non-Goals

- Do not delete `menagerie_static_runner.py`.
- Do not make Go2 physics-owned.
- Do not add observation string support for Go2.
- Do not add `output_options={...}` to the public API in this slice.
- Do not split the monolithic optical lab test file unless necessary.
- Do not redesign render backends.

## Revised Implementation Slices

### P11.6b-1: Design V2

This document.

### P11.6b-2: Provider + Generic Video Product

- add tick frame provider protocol/wrappers;
- generalize `PhysicsVideoFrameProduct` to `VideoFrameProduct`;
- keep compatibility alias;
- add focused provider/product tests.

### P11.6b-3: Static Runtime + Go2 Product Registration

- add static runtime construction with preset defaults;
- add workflow-aware runtime factory;
- register Go2 video product spec;
- add P11 static workflow tests;
- keep `go2_backend.py` temporarily.

### P11.6b-4: Delete Shim

- migrate tests away from `go2_backend`;
- delete the file;
- update MANIFEST/design docs;
- run full optical lab tests.

### P11.6b-5: Real Go2 Render Smoke

- run `run_optical_lab_preset("go2_video_ordered_static", ...)` in the Warp
  conda environment;
- record command and result in implementation note.

## Open Review Questions

1. Should the tick provider protocol live in a new `frame_providers.py`, or
   beside the existing low-level providers in `frame_contexts.py`?
2. Should `build_video_frame_product(...)` live in `runner.py` initially, or
   move with `VideoFrameProduct` to a new `video_frame_product.py` module?
3. Should static preset defaults live in `preset_runtime.py`, or in a dedicated
   `static_runtime.py` registry?
4. Should `ProductWorkflowRunner = PhysicsOwnedProductWorkflow` be introduced
   now, or only `ProductRunResult = PhysicsProductRunResult`?
