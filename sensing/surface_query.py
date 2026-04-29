"""Surface-query specs and host-side executors for ray-like sensors.

This module owns backend-neutral query descriptions and results. It
intentionally does not import `rendering`: render scenes and debug overlays may
visualize query results later, but they are not the canonical query execution
contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from physics.terrain import FlatTerrain, HalfSpaceTerrain, Terrain


@dataclass
class SurfaceQuerySpec:
    """Batch of world-frame ray queries for one published frame / env.

    `origins_world` and `directions_world` are shaped `(num_rays, 3)`. Directions
    are normalized by default so result distances are metric distances in world
    units. A scalar `max_distance` applies to every ray.
    """

    frame_id: int
    sim_time: float
    env_idx: int
    origins_world: object
    directions_world: object
    max_distance: float = np.inf

    def __post_init__(self) -> None:
        origins = np.asarray(self.origins_world, dtype=np.float64)
        directions = np.asarray(self.directions_world, dtype=np.float64)
        if origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError("origins_world must have shape (num_rays, 3)")
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError("directions_world must have shape (num_rays, 3)")
        if origins.shape[0] != directions.shape[0]:
            raise ValueError("origins_world and directions_world must have the same ray count")
        max_distance = float(self.max_distance)
        if max_distance <= 0.0:
            raise ValueError("max_distance must be > 0")

        norms = np.linalg.norm(directions, axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError("directions_world must not contain zero-length directions")
        directions = directions / norms[:, None]

        self.origins_world = origins.copy()
        self.directions_world = directions.copy()
        self.max_distance = max_distance

    @property
    def num_rays(self) -> int:
        return int(self.origins_world.shape[0])


@dataclass
class SurfaceQueryResult:
    """Result for a `SurfaceQuerySpec`.

    Misses are represented by `hit_mask=False`, `distance=np.inf`, and NaN
    position/normal rows. CPU executors return NumPy arrays. Future GPU
    executors may return device arrays; callers are responsible for staging to
    host when needed.
    """

    frame_id: int
    sim_time: float
    env_idx: int
    hit_mask: object
    distance: object
    position_world: object
    normal_world: object


class SurfaceQueryExecutor(Protocol):
    """Executor interface for CPU/GPU surface-query runtimes."""

    def execute(self, spec: SurfaceQuerySpec) -> SurfaceQueryResult:
        """Execute a query spec and return owned result arrays."""


class CpuPlaneSurfaceQueryExecutor:
    """CPU executor for ray queries against an infinite plane.

    This is the first Q53 execution backend. It supports `FlatTerrain` and
    `HalfSpaceTerrain` without depending on rendering or on engine-private
    buffers. Heightfields, mesh terrain, and body geometry are deliberately left
    to future executors.
    """

    def __init__(self, *, normal_world: object, point_world: object) -> None:
        normal = np.asarray(normal_world, dtype=np.float64)
        point = np.asarray(point_world, dtype=np.float64)
        if normal.shape != (3,):
            raise ValueError("normal_world must have shape (3,)")
        if point.shape != (3,):
            raise ValueError("point_world must have shape (3,)")
        norm = np.linalg.norm(normal)
        if norm <= 1e-12:
            raise ValueError("normal_world must be non-zero")
        self.normal_world = normal / norm
        self.point_world = point.copy()

    @classmethod
    def from_terrain(cls, terrain: Terrain) -> "CpuPlaneSurfaceQueryExecutor":
        """Build a plane executor for terrain that has an infinite-plane shape."""

        if isinstance(terrain, FlatTerrain):
            return cls(
                normal_world=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                point_world=np.array([0.0, 0.0, terrain.height_at(0.0, 0.0)], dtype=np.float64),
            )
        if isinstance(terrain, HalfSpaceTerrain):
            return cls(normal_world=terrain.normal_world, point_world=terrain.point_on_plane)
        raise NotImplementedError(f"Surface query terrain executor does not support {type(terrain).__name__}")

    def execute(self, spec: SurfaceQuerySpec) -> SurfaceQueryResult:
        origins = np.asarray(spec.origins_world, dtype=np.float64)
        directions = np.asarray(spec.directions_world, dtype=np.float64)
        n = self.normal_world
        denom = directions @ n
        numer = (self.point_world - origins) @ n

        distance = np.full(spec.num_rays, np.inf, dtype=np.float64)
        hit_mask = np.zeros(spec.num_rays, dtype=bool)
        position = np.full((spec.num_rays, 3), np.nan, dtype=np.float64)
        normal = np.full((spec.num_rays, 3), np.nan, dtype=np.float64)

        valid = np.abs(denom) > 1e-12
        t = np.full(spec.num_rays, np.inf, dtype=np.float64)
        t[valid] = numer[valid] / denom[valid]
        hit_mask[:] = valid & (t >= 0.0) & (t <= spec.max_distance)

        if np.any(hit_mask):
            distance[hit_mask] = t[hit_mask]
            position[hit_mask] = origins[hit_mask] + directions[hit_mask] * t[hit_mask, None]
            normal[hit_mask] = n

        return SurfaceQueryResult(
            frame_id=spec.frame_id,
            sim_time=spec.sim_time,
            env_idx=spec.env_idx,
            hit_mask=hit_mask,
            distance=distance,
            position_world=position,
            normal_world=normal,
        )
