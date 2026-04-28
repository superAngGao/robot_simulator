Initiative: q50-render-backend
Stage: review-request
Author: codex
Version: v1
Date: 2026-04-28
Status: review-followup-applied
Related Files: OPEN_QUESTIONS.md#Q50, collab/q50-render-phase2-terrain-mesh-parity__implementation-note__codex__v1.md, physics/geometry.py, rendering/scene_builder.py, rendering/shape_artists.py, rendering/backends/rerun_backend.py, tests/unit/collision/test_geometry.py, tests/unit/rendering/test_render_scene.py, tests/unit/rendering/test_shape_artists.py, tests/rendering/test_matplotlib_backend.py, tests/rendering/test_rerun_backend.py
Owner Summary: Review request for render phase 2 parity: Rerun now consumes terrain, raw triangle meshes have an explicit `vertices + faces` rendering contract, and Matplotlib/Rerun share the same non-fatal mesh fallback behavior.

## Review Follow-Up Applied

Claude review accepted the overall architecture and raised three non-blocking observations. This patch has absorbed them:

1. `draw_mesh()` now silently skips expected `None` topology but logs a warning for malformed array shapes.
2. `MeshShape(faces=..., vertices=None)` has an explicit test documenting that faces without vertices are allowed but index ranges cannot be validated.
3. `test_flat_terrain_logs_mesh3d` now extracts the terrain mesh payload from the `env_0/terrain` log call instead of relying on global `Mesh3D.call_args`.

In addition, the pre-existing Matplotlib inconsistency was fixed:

- `draw_convex_hull()` now consumes precomputed `faces` when provided and only falls back to `scipy.spatial.ConvexHull` when faces are absent or malformed.

## Review Goal

Please review this as a render-contract / backend-parity change, not as a physics collision change.

The intended outcome is:

1. `RenderScene.terrain` is no longer ignored by `RerunBackend`.
2. Raw triangle mesh rendering has a minimal explicit contract.
3. Missing mesh topology remains safe and non-fatal.
4. This does not broaden `RenderScene` into a camera/LiDAR/sensor execution packet.

## Change Summary

### 1. `MeshShape` can carry triangle faces

`physics.geometry.MeshShape` now accepts optional:

```python
faces: NDArray[np.integer] | None
```

Validation:

- faces must be `(F, 3)`.
- faces are stored as `int32`.
- if vertices are present, face indices must be in range.

This is only a rendering contract extension. Collision paths still use existing convex hull / convex decomposition semantics.

### 2. `scene_builder` forwards mesh topology

`rendering.scene_builder._shape_to_type_params(MeshShape)` now returns:

```python
{
    "vertices": verts_or_none,
    "faces": faces_or_none,
    "filename": shape.filename,
}
```

Backends can render a raw mesh only when both `vertices` and `faces` are available.

### 3. Matplotlib renders triangle meshes

`rendering.shape_artists.draw_mesh()` draws `mesh` shapes using `Poly3DCollection`.

Fallback:

- missing vertices or faces returns `[]`.
- malformed local arrays return `[]`.

### 4. Rerun renders triangle meshes and terrain

`RerunBackend` now:

- logs `mesh` shapes as `rr.Mesh3D` when `vertices + faces` are available.
- logs `scene.terrain` as `env_{env_index}/terrain`.
- supports terrain types:
  - `flat`: square plane at `z`.
  - `halfspace`: square plane through `point` using a tangent basis from `normal`.

Fallback:

- missing mesh vertices/faces logs a warning and skips.
- unknown terrain returns `None`.
- near-zero halfspace normal logs a warning and skips.

## Files Changed

- `physics/geometry.py`
- `rendering/scene_builder.py`
- `rendering/shape_artists.py`
- `rendering/backends/rerun_backend.py`
- `tests/unit/collision/test_geometry.py`
- `tests/unit/rendering/test_render_scene.py`
- `tests/unit/rendering/test_shape_artists.py`
- `tests/rendering/test_matplotlib_backend.py`
- `tests/rendering/test_rerun_backend.py`
- `OPEN_QUESTIONS.md`
- `collab/q50-render-phase2-terrain-mesh-parity__implementation-note__codex__v1.md`

## Verification

Base/default environment, where `rerun-sdk` is not installed:

```bash
PYTHONPATH=. pytest tests/unit/collision/test_geometry.py tests/rendering tests/unit/rendering tests/integration/test_render_scene_integration.py tests/integration/test_published_frame_render_backend_integration.py -q
python -m compileall physics rendering tests/unit/collision/test_geometry.py tests/unit/rendering/test_render_scene.py tests/unit/rendering/test_shape_artists.py tests/rendering/test_matplotlib_backend.py tests/rendering/test_rerun_backend.py
ruff check physics/geometry.py rendering/scene_builder.py rendering/shape_artists.py rendering/backends/rerun_backend.py tests/unit/collision/test_geometry.py tests/unit/rendering/test_render_scene.py tests/unit/rendering/test_shape_artists.py tests/rendering/test_matplotlib_backend.py tests/rendering/test_rerun_backend.py
git diff --check
```

Results:

- `87 passed, 6 skipped`
- `compileall` passed
- `ruff` passed
- `git diff --check` clean

Follow-up local subset after review absorption:

```bash
PYTHONPATH=. pytest tests/unit/collision/test_geometry.py tests/unit/rendering/test_shape_artists.py tests/rendering/test_rerun_backend.py -q
ruff check physics/geometry.py rendering/shape_artists.py tests/unit/collision/test_geometry.py tests/unit/rendering/test_shape_artists.py tests/rendering/test_rerun_backend.py
```

Results:

- `48 passed, 6 skipped`
- `ruff` passed

Rerun-enabled environment:

```bash
conda run -n env_tilelang_20260119 python -m pytest tests/rendering tests/unit/rendering tests/integration/test_render_scene_integration.py tests/integration/test_published_frame_render_backend_integration.py -q
```

Result:

- before follow-up: `63 passed`
- after follow-up: `65 passed`

Note: using `conda run -n env_tilelang_20260119 bash -lc 'pytest ...'` accidentally resolves to `/home/ga/.local/bin/pytest` and the base interpreter, so use `python -m pytest`.

## Specific Review Questions

1. Is adding optional `faces` directly to `MeshShape` acceptable, or should render-only mesh topology live in a separate render metadata object?
2. Should filename-only mesh rendering remain unsupported at this layer, or should rendering own a lazy mesh loader?
3. Is the Rerun terrain plane size default (`half_size=1.0`) acceptable for debug parity, or should it become configurable on `RerunBackend`?
4. Should `draw_mesh()` silently skip malformed arrays, or should it log/raise to surface bad render scene construction earlier?
5. Does the `RenderScene` contract remain narrow enough after this change, especially with Q53's sensor/rendering boundary?

## Known Non-Goals

- No material/texture support.
- No filename-only mesh loading.
- No heightmap terrain rendering.
- No retained/realtime render view.
- No Vulkan/realtime backend.
- No camera/RGB/segmentation/depth sensor integration.
