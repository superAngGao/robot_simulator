# Q54 P11.6b-4 Implementation Note: Register Go2 Static P11 Workflow

Owner: Codex
Date: 2026-07-21
Related Design: `collab/q54-physics-owned-multi-product-runtime-p11-6b-go2-static-p11-workflow__design-request__codex__v3.md`
Related Files:
- `tools/optical_pipeline_lab/preset_runtime.py`
- `tools/optical_pipeline_lab/preset_products.py`
- `tools/optical_pipeline_lab/preset_workflows.py`
- `tools/optical_pipeline_lab/product_specs.py`
- `tools/optical_pipeline_lab/video_products.py`
- `tools/optical_pipeline_lab/__init__.py`
- `tests/unit/optics/test_optical_pipeline_lab.py`
- `MANIFEST.md`

## Summary

P11.6b-4 makes the original Go2 static preset enter the P11 public workflow path
without deleting the deprecated `go2_backend.py` shim yet.

The new path is:

```text
run_optical_lab_preset("go2_video_ordered_static", products=(...))
  -> resolve_lab_product_specs(...)
  -> create_runtime_for_lab_workflow(...)
  -> StaticAssetLabRuntime.step_tick(...)
  -> run_optical_lab_products(...)
  -> ProductWorkflowRunner / MultiProductFrameRunner
```

This keeps P11 as a thin public workflow layer and leaves product materialization,
scenario config writing, execution, and runtime cleanup in the generic P10
workflow implementation.

## Implementation

### 1. Workflow Runtime Factory

`preset_runtime.py` now has two layers:

- `create_runtime_for_lab_preset(...)`
  - remains physics-only;
  - still supports `physics_body_triangle_video_smoke`;
  - still rejects `go2_video_ordered_static`.
- `create_runtime_for_lab_workflow(...)`
  - dispatches reviewed workflow presets;
  - delegates physics presets to `create_runtime_for_lab_preset(...)`;
  - creates `StaticAssetLabRuntime` for `go2_video_ordered_static`.

`StaticAssetLabRuntime` owns:

- the `OpticalLabRenderPipeline`;
- the static scene and base GPU frame;
- `step_tick(frame_index)` producing `SimulationFrameTick` values with
  `frame_source="static_asset_builder"`;
- an idempotent `close()` marker for workflow ownership tests.

The Go2 static model defaults live with the static workflow runtime defaults:

```text
scene_preset = go2_menagerie_static
model_dir    = out/external/mujoco_menagerie/unitree_go2
model_xml    = go2.xml
```

`runtime_kwargs` can override these defaults. `ArtifactOutput.model_dir` and
`ArtifactOutput.model_xml` are intentionally not used as the P11 static asset
source of truth.

### 2. Runtime-Aware Video Product Specs

`VideoProductSpec` now carries a `product_builder` callback:

- physics specs use `build_physics_video_product_from_spec(...)`;
- Go2 static specs use `build_static_video_product_from_spec(...)`.

The static builder reuses the generic `build_video_frame_product(...)` path with:

- `frame_contexts.static_frame_context_provider(runtime.pipeline)`;
- `frame_providers.static_tick_frame_context_provider(...)`.

This avoids duplicating a separate static video product.

### 3. Go2 Static Product Registration

`preset_products.py` now registers:

```text
go2_video_ordered_static -> create_go2_video_ordered_static_product_spec()
```

So:

```python
resolve_lab_product_specs(
    preset="go2_video_ordered_static",
    products=("video", "debug"),
)
```

returns a static video spec and a debug spec.

### 4. Public Preset Workflow

`run_optical_lab_preset(...)` now:

1. resolves product selections before creating a runtime;
2. resolves `ArtifactOutput` before creating a runtime;
3. calls `create_runtime_for_lab_workflow(...)`;
4. delegates to `run_optical_lab_products(..., owns_runtime=True)`.

This keeps the P11 facade thin and lets P10 own config serialization, product
building, execution, and cleanup.

## Tests Added / Updated

Focused tests cover:

- top-level lazy export for `create_runtime_for_lab_workflow`;
- physics dispatch through `create_runtime_for_lab_workflow(...)`;
- static dispatch through `create_runtime_for_lab_workflow(...)`;
- `create_static_asset_lab_runtime(...)` default/override wiring without real
  Warp rendering;
- Go2 static `"video"` product registration;
- public `run_optical_lab_preset("go2_video_ordered_static", products=("debug",))`
  through a fake static runtime;
- output conflict fail-fast before runtime creation;
- runtime cleanup when product build fails after runtime creation.

Existing no-`go2_backend` subprocess guards remain in place for product resolver
and public preset workflow imports.

## Verification

```bash
ruff check tools/optical_pipeline_lab/__init__.py \
  tools/optical_pipeline_lab/preset_runtime.py \
  tools/optical_pipeline_lab/preset_workflows.py \
  tools/optical_pipeline_lab/preset_products.py \
  tools/optical_pipeline_lab/product_specs.py \
  tools/optical_pipeline_lab/video_products.py \
  tests/unit/optics/test_optical_pipeline_lab.py
```

Result:

```text
All checks passed!
```

```bash
conda run -n robot_sim env PYTHONPATH=. python -m pytest -q \
  tests/unit/optics/test_optical_pipeline_lab.py \
  -k "create_runtime_for_lab_workflow or create_static_asset_lab_runtime or resolve_lab_product_specs or run_optical_lab_preset"
```

Result:

```text
17 passed, 169 deselected
```

```bash
conda run -n robot_sim env PYTHONPATH=. python -m pytest -q \
  tests/unit/optics/test_optical_pipeline_lab.py
```

Result:

```text
186 passed
```

Real Go2 static P11 smoke:

```bash
conda run -n env_tilelang_20260119 env PYTHONPATH=. python -c "..."
```

The smoke used:

```python
run_optical_lab_preset(
    "go2_video_ordered_static",
    frames=1,
    products=("video", "debug"),
    output=ArtifactOutput(
        root=Path("out/optical_pipeline_lab/p11_go2_static_preset_smoke"),
        frames=1,
        fps=30.0,
        warmup_renders=0,
        video_readback_delivery="sync",
        video_readback_ring_depth=1,
    ),
)
```

Result:

```text
video_frame 1/1: total=48.494ms, render=20.587ms, readback=27.611ms, fps=20.62
artifacts {'root': PosixPath('out/optical_pipeline_lab/p11_go2_static_preset_smoke')}
products ['debug', 'video']
video 1 [1]
debug 1 [1]
```

Artifacts:

- `out/optical_pipeline_lab/p11_go2_static_preset_smoke/scenario_config.json`
- `out/optical_pipeline_lab/p11_go2_static_preset_smoke/frame_timing.csv`

The serialized scenario has:

```text
scenario_name = go2_video_ordered_static
frame_source  = static_asset_builder
clock_owner   = runner
```

## Not Done

- Did not delete `tools/optical_pipeline_lab/go2_backend.py`.
- Did not migrate legacy `go2_backend` unit tests.
- Did not add a public Go2 example yet.

These remain for the next slice after review:

1. run the original Go2 static rendering through `run_optical_lab_preset(...)`;
2. migrate remaining generic tests away from `go2_backend.py`;
3. delete the deprecated shim once the P11 path is proven.
