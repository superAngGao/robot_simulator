"""
Terrain models for contact height queries.

Provides an abstract Terrain base and concrete implementations:
  - FlatTerrain        : constant-height ground plane
  - HeightmapTerrain   : 2D grid-based heightmap (Phase 2+ stub)

References:
  Mirtich & Canny (1995) — terrain contact conventions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Terrain(ABC):
    """Abstract terrain interface for height and normal queries."""

    @abstractmethod
    def height_at(self, x: float, y: float) -> float:
        """Return terrain height at world-frame position (x, y) [m]."""

    @abstractmethod
    def normal_at(self, x: float, y: float) -> NDArray[np.float64]:
        """Return unit surface normal at position (x, y) in world frame."""


class FlatTerrain(Terrain):
    """Infinite flat ground plane at a fixed height."""

    def __init__(self, z: float = 0.0) -> None:
        self._z = float(z)

    def height_at(self, x: float, y: float) -> float:
        return self._z

    def normal_at(self, x: float, y: float) -> NDArray[np.float64]:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)


class HalfSpaceTerrain(Terrain):
    """Inclined infinite-plane terrain.

    Defined by a world-frame outward normal and a point on the surface.
    Generalises ``FlatTerrain`` to arbitrary orientations.

    Example — 30-degree incline around Y axis::

        normal = [-sin(pi/6), 0, cos(pi/6)]
        terrain = HalfSpaceTerrain(normal=normal, point=np.zeros(3), mu=0.3)

    Args:
        normal : Outward surface normal (will be unit-normalised).
        point  : A point on the plane in world frame (default: origin).
        mu     : Coulomb friction coefficient (read by CollisionPipeline).
    """

    def __init__(
        self,
        normal: NDArray[np.float64],
        point: NDArray[np.float64] | None = None,
        mu: float = 0.5,
    ) -> None:
        n = np.asarray(normal, dtype=np.float64)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            raise ValueError("HalfSpaceTerrain normal must be non-zero")
        self._normal = n / norm
        self._point = np.asarray(point, dtype=np.float64) if point is not None else np.zeros(3)
        self.mu = float(mu)

    @property
    def normal_world(self) -> NDArray[np.float64]:
        """Outward surface normal in world frame (unit vector)."""
        return self._normal

    @property
    def point_on_plane(self) -> NDArray[np.float64]:
        """A point on the surface in world frame."""
        return self._point

    def height_at(self, x: float, y: float) -> float:
        """Return z-coordinate of the plane surface at world (x, y).

        Solves ``dot(normal, [x, y, z] - point) = 0`` for z.
        Returns ``-1e6`` for (nearly) vertical planes where nz ≈ 0.
        """
        nx, ny, nz = self._normal
        if abs(nz) < 1e-12:
            return -1e6
        px, py, pz = self._point
        return pz + (nx * (px - x) + ny * (py - y)) / nz

    def normal_at(self, x: float, y: float) -> NDArray[np.float64]:
        return self._normal.copy()


class HeightmapTerrain(Terrain):
    """Grid-based heightmap terrain (Phase 2+ stub).

    Args:
        heightmap  : 2-D array of heights, shape (rows, cols) [m].
        resolution : Grid cell size [m] (same in x and y).
        origin     : World-frame (x, y) coordinates of the heightmap corner [m].
    """

    def __init__(
        self,
        heightmap: NDArray[np.float64],
        resolution: float,
        origin: NDArray[np.float64],
    ) -> None:
        self._heightmap = heightmap
        self._resolution = float(resolution)
        self._origin = np.asarray(origin, dtype=np.float64)

    def height_at(self, x: float, y: float) -> float:
        raise NotImplementedError("HeightmapTerrain not yet implemented (Phase 2+)")

    def normal_at(self, x: float, y: float) -> NDArray[np.float64]:
        raise NotImplementedError("HeightmapTerrain not yet implemented (Phase 2+)")
