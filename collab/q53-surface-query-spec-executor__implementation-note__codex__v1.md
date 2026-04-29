Initiative: q53-sensing-rendering-boundary
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-04-29
Status: implemented
Related Files: OPEN_QUESTIONS.md#Q53, sensing/surface_query.py, sensing/__init__.py, tests/unit/sensing/test_surface_query.py
Owner Summary: Adds the first `SurfaceQuerySpec` / executor slice under `sensing/`, preserving the Q53 boundary: sensing owns backend-neutral query specs/results, execution is explicit, and rendering remains out of the dependency path.

# Q53 Surface Query Spec / Executor

## Summary

This pass implements the first non-numeric sensing boundary from Q53:

- `SurfaceQuerySpec`
- `SurfaceQueryResult`
- `SurfaceQueryExecutor` protocol
- `CpuPlaneSurfaceQueryExecutor`

The implementation intentionally does not import `rendering`, does not use
`RenderScene` as a query scene, and does not produce camera/image payloads.

## Contract

`SurfaceQuerySpec` describes a world-frame batch of ray-like surface queries for
one published frame / env:

```python
SurfaceQuerySpec(
    frame_id=frame_id,
    sim_time=sim_time,
    env_idx=env_idx,
    origins_world=origins,      # (num_rays, 3)
    directions_world=directions # (num_rays, 3)
)
```

Directions are normalized inside the spec. This keeps `SurfaceQueryResult.distance`
as a metric distance in world units and avoids mixing sensor distance with raw ray
parameters.

`SurfaceQueryResult` carries:

- `hit_mask`
- `distance`
- `position_world`
- `normal_world`

Misses use:

- `hit_mask=False`
- `distance=np.inf`
- NaN position / normal rows

## Executor

`SurfaceQueryExecutor` is a protocol:

```python
class SurfaceQueryExecutor(Protocol):
    def execute(self, spec: SurfaceQuerySpec) -> SurfaceQueryResult:
        ...
```

`CpuPlaneSurfaceQueryExecutor` is the first concrete executor. It supports:

- `FlatTerrain`
- `HalfSpaceTerrain`

It deliberately rejects:

- `HeightmapTerrain`
- mesh terrain
- body geometry

Those need future CPU/GPU acceleration structures and should not be hidden inside
the spec builder.

## Boundary

This pass follows Q53:

- `sensing/` owns query specs/results.
- Query execution is explicit and separable.
- `rendering/` is not imported.
- `RenderScene` is not used as the canonical sensor query world.
- Future debug overlays may visualize query results, but the results stay owned
  by the sensing/query runtime.

## Verification

```bash
PYTHONPATH=. pytest tests/unit/sensing -q
ruff check sensing tests/unit/sensing
python -m compileall sensing tests/unit/sensing
```

Results:

- surface-query unit tests: `14 passed`
- sensing unit tests: `26 passed`
- sensing/publish/render bridge subset: `73 passed`
- GPU API: `40 passed`
- `ruff` passed
- `compileall` passed

## Deferred

1. Query builders from sensor attachments / scan patterns.
2. Body-geometry and mesh terrain queries.
3. GPU/Warp surface-query executor.
4. LiDAR/range-finder readings built on query results.
5. Render-backed RGB / segmentation in the future `sensor_rendering/` layer.
