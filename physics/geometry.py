"""
Collision geometry description for articulated body links.

Provides abstract CollisionShape and concrete shape types (Box, Sphere,
Cylinder, Mesh) plus the BodyCollisionGeometry container that associates
shapes with a body index.

References:
  ROS URDF spec §collision element.
  Featherstone (2008) §2.1 — link geometry conventions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Abstract shape base
# ---------------------------------------------------------------------------


class CollisionShape(ABC):
    """Abstract base for all collision geometry shapes."""

    @abstractmethod
    def half_extents_approx(self) -> NDArray[np.float64]:
        """Return a conservative AABB half-extent (3,) for broad-phase checks."""


# ---------------------------------------------------------------------------
# Concrete shapes
# ---------------------------------------------------------------------------


class BoxShape(CollisionShape):
    """Axis-aligned box."""

    def __init__(self, size: tuple[float, float, float]) -> None:
        self._size = np.asarray(size, dtype=np.float64)

    def half_extents_approx(self) -> NDArray[np.float64]:
        return self._size / 2.0


class SphereShape(CollisionShape):
    """Sphere."""

    def __init__(self, radius: float) -> None:
        self._radius = float(radius)

    def half_extents_approx(self) -> NDArray[np.float64]:
        r = self._radius
        return np.array([r, r, r], dtype=np.float64)


class CylinderShape(CollisionShape):
    """Cylinder aligned with the local Z axis."""

    def __init__(self, radius: float, length: float) -> None:
        self._radius = float(radius)
        self._length = float(length)

    def half_extents_approx(self) -> NDArray[np.float64]:
        r = self._radius
        return np.array([r, r, self._length / 2.0], dtype=np.float64)


class MeshShape(CollisionShape):
    """Mesh geometry (Phase 3 stub)."""

    def __init__(self, filename: str) -> None:
        self.filename = filename

    def half_extents_approx(self) -> NDArray[np.float64]:
        raise NotImplementedError("MeshShape AABB not yet implemented (Phase 3)")


# ---------------------------------------------------------------------------
# Shape instance and body geometry container
# ---------------------------------------------------------------------------


@dataclass
class ShapeInstance:
    """A shape placed at a fixed offset within a body frame."""

    shape: CollisionShape
    origin_xyz: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    origin_rpy: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3, dtype=np.float64))


@dataclass
class BodyCollisionGeometry:
    """All collision shapes associated with one body.

    Attributes:
        body_index : Index of the body in the RobotTree body list.
        shapes     : List of ShapeInstance objects.
    """

    body_index: int
    shapes: list[ShapeInstance]

    def aabb_half_extents(self) -> NDArray[np.float64]:
        """Return element-wise maximum half-extents across all shapes."""
        if not self.shapes:
            return np.zeros(3, dtype=np.float64)
        extents = np.stack([s.shape.half_extents_approx() for s in self.shapes])
        return np.max(extents, axis=0)
