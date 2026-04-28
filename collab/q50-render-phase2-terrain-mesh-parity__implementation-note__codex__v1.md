Initiative: q50-render-backend
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-04-28
Status: implemented
Related Files: OPEN_QUESTIONS.md#Q50, physics/geometry.py, rendering/scene_builder.py, rendering/shape_artists.py, rendering/backends/rerun_backend.py, tests/unit/collision/test_geometry.py, tests/unit/rendering/test_render_scene.py, tests/unit/rendering/test_shape_artists.py, tests/rendering/test_matplotlib_backend.py, tests/rendering/test_rerun_backend.py
Owner Summary: Render phase 2 closes the first backend feature-parity gap after Q50 Step 4: `RenderScene.terrain` is now consumed by Rerun, and raw triangle `mesh` rendering has an explicit vertices+faces contract shared by scene_builder, MatplotlibBackend, and RerunBackend.

## Review Follow-Up (2026-04-28)

Claude review accepted the main contract and the following small follow-ups were applied:

- `draw_mesh()` distinguishes expected missing topology from malformed arrays:
  missing `vertices`/`faces` silently skips, malformed shapes log a warning.
- `MeshShape(faces=..., vertices=None)` is covered by an explicit test.
- Rerun terrain logging test now inspects the `env_0/terrain` log payload instead of the last global `Mesh3D` call.
- `draw_convex_hull()` now consumes precomputed `faces` when available, avoiding per-frame SciPy hull recomputation in the Matplotlib backend path.
- `MeshShape.faces` docstring now states that faces are rendering/export
  topology and are not consumed directly by collision code.
- `RerunBackend` now treats missing mesh topology as an expected debug-level
  skip, while malformed mesh arrays warn and skip before reaching `rr.Mesh3D`.
- Rerun malformed mesh coverage was added.

## Summary

This pass implements the small but important render phase-2 contract cleanup that Q50 left open:

- `RerunBackend` now consumes `scene.terrain`.
- `MeshShape` can carry optional triangle faces in addition to vertices.
- `scene_builder` forwards `mesh` vertices and faces into `PositionedShape.params`.
- Matplotlib and Rerun both render `mesh` when `vertices + faces` are available.
- Meshes without triangle faces remain non-fatal and are skipped.

This does not turn `RenderScene` into a camera/LiDAR sensor contract. It only makes the existing debug/export scene contract more complete.

## Implemented Details

### Terrain

`RerunBackend.render_frame()` now logs supported terrain as `rr.Mesh3D`:

- `flat`: a small square plane at `z`.
- `halfspace`: a square plane through `point` with tangent basis derived from `normal`.

The terrain entity path is:

```text
env_{env_index}/terrain
```

This brings Rerun in line with Matplotlib for the terrain field already present on `RenderScene`.

The debug plane half-size defaults to `1.0` and is configurable:

```python
RerunBackend(terrain_half_size=1.0)
```

### Mesh

`physics.geometry.MeshShape` now accepts:

```python
MeshShape(filename, vertices=..., faces=..., scale=...)
```

The rendering contract is intentionally narrow:

- `vertices`: `(N, 3)` float local-frame positions.
- `faces`: `(F, 3)` integer triangle indices.
- faces are validated against vertices when vertices are available.

`scene_builder._shape_to_type_params()` now emits:

```python
{
    "vertices": verts_or_none,
    "faces": faces_or_none,
    "filename": shape.filename,
}
```

Backends render only when both arrays are present. This avoids pretending that filename-only mesh rendering is solved.

## Tests

Verified with:

```bash
PYTHONPATH=. pytest tests/unit/collision/test_geometry.py -q
PYTHONPATH=. pytest tests/rendering tests/unit/rendering tests/integration/test_render_scene_integration.py tests/integration/test_published_frame_render_backend_integration.py -q
python -m compileall physics rendering tests/unit/collision/test_geometry.py tests/unit/rendering/test_render_scene.py tests/rendering/test_matplotlib_backend.py tests/rendering/test_rerun_backend.py
```

Results:

- `30 passed`
- `57 passed, 6 skipped`
- `compileall` passed

The skipped tests are the existing optional `rerun-sdk` dependency skips in this environment.

After review follow-up:

- default env subset: `48 passed, 6 skipped`
- Rerun-enabled `env_tilelang_20260119` rendering target: `65 passed`
- `ruff` passed
- `compileall` passed

## Remaining Render Gaps

Still deferred:

- filename-only mesh loading directly inside rendering
- textured/material mesh rendering
- heightmap terrain rendering
- retained/realtime render view
- Vulkan/realtime backend
- render-backed camera/RGB/segmentation integration, which remains future `sensor_rendering/` work per Q53
