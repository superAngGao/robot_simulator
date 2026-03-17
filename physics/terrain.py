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
