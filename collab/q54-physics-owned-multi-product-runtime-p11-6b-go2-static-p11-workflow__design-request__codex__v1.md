Initiative: q54-physics-owned-multi-product-runtime-p11
Stage: design-request
Author: Codex
Version: v1
Date: 2026-07-20
Status: in_review
Related Files: tools/optical_pipeline_lab/go2_backend.py, tools/optical_pipeline_lab/menagerie_static_runner.py, tools/optical_pipeline_lab/preset_workflows.py, tools/optical_pipeline_lab/preset_runtime.py, tools/optical_pipeline_lab/preset_products.py, tools/optical_pipeline_lab/product_workflow.py, examples/optical_lab/README.md, tests/unit/optics/test_optical_pipeline_lab.py
Owner Summary: Delete the deprecated `go2_backend.py` shim only after the original Go2 static rendering preset can run through the P11 public workflow surface. This plan splits the work so the API path is reviewed before code churn.

# Q54 P11.6b Design: Delete `go2_backend.py` After Go2 Static P11 Workflow

## User Request

The desired end state is stronger than P11.6a:

```text
- Delete tools/optical_pipeline_lab/go2_backend.py.
- Verify tests still pass without the shim.
- Re-run the original Go2 rendering through the P11 public interface.
```

The important nuance is the last line. Deleting the shim is only meaningful if
the old Go2 render path is not simply moved to another hidden legacy bypass.
`go2_video_ordered_static` should be runnable through:

```python
from pathlib import Path

from tools.optical_pipeline_lab import ArtifactOutput, run_optical_lab_preset

result = run_optical_lab_preset(
    preset="go2_video_ordered_static",
    frames=2,
    products=("video", "debug"),
    output=ArtifactOutput(
        root=Path("runs/p11/go2_video_ordered_static"),
        model_dir="out/external/mujoco_menagerie/unitree_go2",
        model_xml="go2.xml",
        frames=2,
        video_readback_delivery="sync",
        video_raygen="gpu",
        fail_on_overflow=False,
    ),
)
```

The exact option shape is open to review. The core requirement is that a user
should not import `menagerie_static_runner.py` or call
`render_many_views(...)` for the reviewed preset workflow.

## Current State

P11.6a completed the first half of the naming cleanup:

```text
menagerie_static_runner.py  # owns legacy static CLI/benchmark implementation
go2_backend.py              # deprecated compatibility shim only
```

However, P11 public workflow support is still physics-owned only:

```text
run_optical_lab_preset(...)
  -> create_runtime_for_lab_preset(...)
  -> run_optical_lab_products(...)
  -> run_physics_product_scenario(...)
```

`create_runtime_for_lab_preset("go2_video_ordered_static")` currently raises
`NotImplementedError`. `resolve_lab_product_specs(..., products=("video",))`
also intentionally rejects video for that preset.

So deleting `go2_backend.py` by itself is possible only after migrating tests to
`menagerie_static_runner.py`, but that would not satisfy the user-level goal.
The missing piece is a static-asset preset workflow through the same P11 API.

## Design Goal

Make P11 support two reviewed runtime ownership modes behind one public call:

```text
physics-owned preset:
  run_optical_lab_preset(...)
    -> PhysicsLabScenarioRuntime.step_tick(...)
    -> PhysicsVideoFrameProduct / Debug / Observation

static-asset preset:
  run_optical_lab_preset(...)
    -> StaticAssetLabRuntime.step_tick(...)
    -> StaticVideoFrameProduct / Debug
```

Both paths should produce `PhysicsProductRunResult` for now, or the result type
should be renamed before static support lands. The current type is structurally
generic, but its name is physics-specific.

## Proposed API Shape

The existing P11 call remains valid:

```python
run_optical_lab_preset(
    preset="physics_body_triangle_video_smoke",
    frames=120,
    products=("video", "debug"),
    out=Path("runs/p11/physics"),
)
```

Add static Go2 support through the same function:

```python
from tools.optical_pipeline_lab import ArtifactOutput

run_optical_lab_preset(
    preset="go2_video_ordered_static",
    frames=120,
    products=("video", "debug"),
    output=ArtifactOutput(
        root=Path("runs/p11/go2"),
        model_dir="out/external/mujoco_menagerie/unitree_go2",
        model_xml="go2.xml",
        frames=120,
        video_readback_delivery="sync",
        fail_on_overflow=False,
    ),
)
```

Review point: this keeps the public API unchanged, but it does require users
who customize static asset paths to know `ArtifactOutput`. Alternatives:

1. Keep `output: ArtifactOutput | None` as the only advanced option and require:

   ```python
   output=ArtifactOutput(
       root=Path("runs/p11/go2"),
       model_dir="...",
       model_xml="...",
       frames=120,
       fail_on_overflow=False,
   )
   ```

2. Add explicit keyword passthroughs only for static assets:

   ```python
   model_dir="...",
   model_xml="...",
   ```

3. Add a broad convenience bag:

   ```python
   output_options={
       "model_dir": "...",
       "model_xml": "...",
   }
   ```

4. Keep current API unchanged and document that static Go2 advanced options use
   `output=ArtifactOutput(...)`.

Codex recommendation: choose option 4 for the first implementation. It avoids a
new public options bag and keeps `run_optical_lab_preset(...)` thin.

## Internal Design

### 1. Rename Generic Workflow Result

`PhysicsProductRunResult` is already structurally generic:

```python
frame_results
begin_outputs
end_outputs
artifacts
product_results
```

But the name becomes wrong once static workflows use it.

Proposed low-risk migration:

```python
ProductRunResult = PhysicsProductRunResult
```

Then update new P11/static-facing functions to annotate `ProductRunResult`.
Keep `PhysicsProductRunResult` exported as a compatibility alias until a later
cleanup.

Do not rename every internal test in the same slice unless review asks for it.

### 2. Add Static Runtime Owner

Add a lab-local runtime object, probably in a new module:

```text
tools/optical_pipeline_lab/static_runtime.py
```

Target shape:

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
            frame_id=int(self.base_frame.frame_id) + frame_index,
            sim_time=float(self.base_frame.sim_time) + frame_index / self.fps,
            published_frame=self.base_frame,
            metadata={
                "frame_source": "static_asset_builder",
                "scene_preset": ...,
                ...
            },
        )

    def close(self) -> None:
        return None
```

Static ticks reuse the same published frame for rendering because the geometry
is static. Tick identity advances so debug artifacts and CSV rows still reflect
the requested frame count.

### 3. Construct Static Runtime From Scenario + Output

Static runtime construction needs `ArtifactOutput.model_dir/model_xml`, so it
cannot be a pure `preset -> runtime` factory unless `ArtifactOutput` is passed
in.

Prefer adding a second factory layer used by `run_optical_lab_preset(...)`:

```python
create_runtime_for_lab_workflow(
    preset: str,
    *,
    config: OpticalLabScenarioConfig,
    output: ArtifactOutput,
    device: str | None = None,
    runtime_kwargs: Mapping[str, object] | None = None,
) -> object
```

Behavior:

- physics-owned presets delegate to existing `create_runtime_for_lab_preset`;
- static asset presets call
  `OpticalLabRenderPipeline.create_from_source_factory(...)` using
  `static_asset_source.build_static_asset_render_source(...)`;
- the static args are derived from the same `build_menagerie_example_args(...)`
  or a renamed generic `build_static_asset_video_args(...)`;
- no call to `menagerie_static_runner.render_many_views(...)`.

Review point: this factory may live in `preset_runtime.py` or a new
`preset_runtime_workflow.py`. Codex recommendation: keep it in
`preset_runtime.py` while the registry is small.

### 4. Add Static Video Product

Do not reuse `PhysicsVideoFrameProduct`; it assumes `published_frame` is passed
to a physics frame provider.

Add either:

```text
tools/optical_pipeline_lab/static_products.py
```

or place beside the existing product code:

```python
@dataclass
class StaticVideoFrameProduct:
    runtime: StaticAssetLabRuntime
    config: OpticalLabScenarioConfig
    args: argparse.Namespace
    frame_provider: StaticFrameContextProvider
    delivery: VideoDeliveryFacade
    ...
```

Implementation mirrors `PhysicsVideoFrameProduct` but:

- uses `static_frame_context_provider(runtime.pipeline)`;
- does not pass `published_frame`;
- uses `runtime.pipeline.session.scene` for camera planning;
- writes the same `frame_timing.csv` schema.

This is small duplication but gives a clear lifecycle boundary. A later cleanup
can extract a shared `VideoFrameProductBase` only if the duplication becomes
real maintenance pain.

### 5. Make `VideoProductSpec.build(...)` Dispatch By Frame Source

Current `VideoProductSpec.build(...)` always builds physics video. Change it to:

```python
if is_physics_published_frame_source(context.config.frame_source):
    return build_physics_video_frame_product(...)
if context.config.frame_source is FrameSourceKind.STATIC_ASSET_BUILDER:
    return build_static_video_frame_product(...)
raise ValueError(...)
```

This keeps user-facing product selection stable:

```python
products=("video", "debug")
```

### 6. Register Go2 Video Product

In `preset_products.py`:

```python
_VIDEO_PRODUCT_FACTORIES = {
    "physics_body_triangle_video_smoke": create_physics_body_triangle_video_product_spec,
    "go2_video_ordered_static": create_go2_video_ordered_static_product_spec,
}
```

The Go2 product spec should use the same generic camera builder and RGB packer:

```python
create_go2_video_ordered_static_product_spec()
  -> VideoProductSpec(
       build_video_camera=build_lab_video_camera,
       synchronize_event=synchronize_ready_event,
       pack_rgb8=pack_video_rgb8,
     )
```

No import of `menagerie_static_runner.py` or `go2_backend.py`.

### 7. Delete `go2_backend.py`

After tests are migrated:

- delete `tools/optical_pipeline_lab/go2_backend.py`;
- remove the MANIFEST shim row;
- update `GPU_OPTICAL_PIPELINE_DESIGN.md` from "deprecated shim" to "deleted";
- update `examples/optical_lab/README.md` to remove references to legacy
  `go2_backend.py` paths;
- keep `menagerie_static_runner.py` only as a legacy CLI/benchmark module.

## Tests

Required unit tests:

1. `test_create_runtime_for_lab_workflow_builds_static_go2_runtime`
   - monkeypatch static source/render pipeline builders;
   - verify `model_dir/model_xml` come from `ArtifactOutput`;
   - verify no `menagerie_static_runner.render_many_views(...)` call.

2. `test_resolve_lab_product_specs_supports_go2_video`
   - `preset="go2_video_ordered_static", products=("video", "debug")`;
   - returns `VideoProductSpec`, `DebugProductSpec`.

3. `test_run_optical_lab_preset_runs_go2_static_products_through_p11`
   - monkeypatch static runtime factory to avoid real GPU;
   - use concrete fake video/debug products or product specs;
   - assert `scenario_config.json` is written;
   - assert runtime closes when owned.

4. `test_static_video_frame_product_uses_static_frame_provider`
   - use fake pipeline/provider/delivery;
   - assert render -> delivery -> row write behavior.

5. `test_go2_backend_module_is_deleted`
   - assert the file path does not exist;
   - subprocess import should fail with `ModuleNotFoundError`.

6. Existing generic render/video tests should import:
   - `video_loop` for generic helpers;
   - `render_session` for `OpticalLabRender*`;
   - `menagerie_static_runner` only for legacy CLI-specific tests.

Required integration/smoke verification:

```bash
conda run -n robot_sim env PYTHONPATH=. python - <<'PY'
from pathlib import Path
from tools.optical_pipeline_lab import ArtifactOutput, run_optical_lab_preset

run_optical_lab_preset(
    preset="go2_video_ordered_static",
    frames=1,
    products=("video", "debug"),
    output=ArtifactOutput(
        root=Path("out/p11_go2_static_smoke"),
        model_dir="out/external/mujoco_menagerie/unitree_go2",
        model_xml="go2.xml",
        frames=1,
        video_readback_delivery="sync",
        video_raygen="gpu",
        fail_on_overflow=False,
    ),
)
PY
```

This is the replacement for "rerun the original Go2 rendering." It must write
`scenario_config.json` and `frame_timing.csv` through the P11 product workflow,
not through `menagerie_static_runner.render_many_views(...)`.

## Non-Goals

- Do not delete `menagerie_static_runner.py` in this slice. It remains a legacy
  CLI/benchmark path until a separate example cleanup.
- Do not make Go2 physics-owned.
- Do not add observation product string support for Go2; observation still
  requires explicit robot metadata and is physics-oriented.
- Do not redesign render backends.
- Do not split the monolithic optical lab test file in this slice.

## Implementation Slices

### P11.6b-1: Static Runtime + Product Plan Review

This document.

### P11.6b-2: Static Go2 P11 Workflow

- add static runtime factory;
- add static video product;
- register Go2 video product;
- add tests for the P11 path;
- keep `go2_backend.py` shim temporarily.

### P11.6b-3: Delete Shim

- migrate tests away from `go2_backend`;
- delete `go2_backend.py`;
- update MANIFEST/design docs;
- run full optical lab tests.

### P11.6b-4: Real Go2 Render Smoke

- run the reviewed `run_optical_lab_preset(...)` Go2 smoke in the Warp conda
  environment;
- record command and artifacts in the implementation note.

## Review Questions

1. Should the public API add `output_options={...}`, or require
   `output=ArtifactOutput(...)` for advanced static options?
2. Should `ProductRunResult` be introduced now as the generic alias, or should
   the result rename wait until after static support works?
3. Should static runtime construction live in `preset_runtime.py`, or in a new
   `static_runtime.py` plus a small registry hook?
4. Is duplicating a small `StaticVideoFrameProduct` acceptable, or should we
   extract a shared video product base immediately?
5. Should the real Go2 smoke use `readback_payload='none'` for speed, or keep
   the original RGB readback behavior to prove rendering/delivery end-to-end?
