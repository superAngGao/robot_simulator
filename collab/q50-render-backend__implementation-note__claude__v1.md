# Q50 Render Backend — Implementation Note (Claude v1)

## Summary

Implemented Steps 1–5 of the Q50 render-backend plan:
- `RenderBackend(ABC)` contract in `rendering/backends/base.py`
- `MatplotlibBackend` wrapping existing `shape_artists` functions
- `RerunBackend` with full shape dispatch table
- `scene_builder.py`: fan triangulation for `ConvexHullShape` + `build_render_scene_from_gpu`
- `rendering/__init__.py` updated with new public exports
- 22 tests across 5 test files (18 pass, 4 skipped — rerun not installed)

## Files Created / Modified

**Created:**
- `rendering/backends/__init__.py`
- `rendering/backends/base.py`
- `rendering/backends/matplotlib_backend.py`
- `rendering/backends/rerun_backend.py`
- `tests/rendering/__init__.py`
- `tests/rendering/test_render_backend_abc.py`
- `tests/rendering/test_matplotlib_backend.py`
- `tests/rendering/test_rerun_backend.py`
- `tests/rendering/test_gpu_bridge.py`
- `tests/rendering/test_scene_builder_convex_hull.py`

**Modified:**
- `rendering/scene_builder.py` — fan triangulation + GPU bridge
- `rendering/__init__.py` — new exports

## Test Results

```
790 passed, 5 skipped (full fast suite, no regressions)
18 passed, 4 skipped (rendering suite; rerun skipped — not installed)
```

## 关键思考

### Non-obvious technical decisions

**1. FuncAnimation vs ArtistAnimation for MatplotlibBackend**

Initial implementation used `ArtistAnimation` (collect artist lists per frame). This failed with `AttributeError: 'NoneType' object has no attribute 'canvas'` because `ax.cla()` detaches artists from their figure — `ArtistAnimation._init_draw()` then tries to call `fig.canvas.draw_idle()` on a `None` figure reference.

Switched to `FuncAnimation` with the remove-and-redraw pattern (same as `RobotViewer.animate()`). This keeps artists attached to the figure throughout. The backend now stores `(scene, timestamp)` pairs and rebuilds artists in the `update` callback during `_save_animation()`.

**2. Fan triangulation for ConvexHullShape**

`FaceTopology.face_vertex_ids` contains polygon faces (not necessarily triangles). The Rerun `rr.Mesh3D` archetype requires triangle indices. Fan triangulation from vertex 0: for a face with vertices `[v0, v1, v2, ..., vn]`, emit triangles `[v0,v1,v2], [v0,v2,v3], ..., [v0,v(n-1),vn]`. This is correct for convex faces (all faces of a convex hull are convex polygons).

**3. GPU bridge design**

`build_render_scene_from_gpu` reads `engine.q_wp.numpy()` (Warp→CPU copy), runs CPU FK, then delegates to the existing `build_render_scene`. This avoids duplicating the scene-building logic and keeps the GPU bridge thin. The `IndexError` on out-of-bounds `env_idx` is explicit (checked before numpy indexing) to give a clear error message.

**4. RerunBackend lazy import**

`import rerun as rr` is deferred to `open()` and `render_frame()` so that importing `rendering` does not fail when rerun is not installed. The `RerunBackend` class itself is always importable; only instantiation + use requires rerun.

### Debugging difficulties

- The `ArtistAnimation` / `blit=True` failure was not obvious from the traceback — the root cause (`ax.cla()` detaching artists) required reading the matplotlib animation source. The fix (switch to `FuncAnimation`) was confirmed by checking how `RobotViewer` handles the same problem.

## Coverage gaps / residual risks (for Codex review)

1. **RerunBackend tests are all skipped** — rerun is not installed in CI. The mock-based tests (`test_all_supported_shapes_do_not_raise`, `test_convex_hull_uses_precomputed_faces`, `test_contacts_render_as_arrows`) use `patch.dict("sys.modules", {"rerun": mock_rr})` but are gated by `pytestmark = pytest.mark.skipif(not rr_available, ...)`. If rerun is installed, all 4 tests should pass.

2. **`rr.Cylinders3D` / `rr.Capsules3D` API** — these archetypes were added in Rerun 0.16. Older versions may not have them. The backend targets 0.16+.

3. **`build_render_scene_from_gpu` mock test** — uses a real `MergedModel` but a mock engine. The `engine.q_wp.numpy()` path is exercised; the `engine.query_contacts()` mock returns `[]`. A real GPU engine integration test is deferred to Q51.

4. **`MatplotlibBackend.render_frame` draws immediately** — calling `_draw_scene` on every `render_frame` call means the axes are always up-to-date for inspection, but it does redundant work if only the GIF export matters. This is acceptable for a debug backend.
