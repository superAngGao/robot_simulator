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
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .spatial import SpatialTransform

# ---------------------------------------------------------------------------
# Abstract shape base
# ---------------------------------------------------------------------------


class CollisionShape(ABC):
    """Abstract base for all collision geometry shapes."""

    @abstractmethod
    def half_extents_approx(self) -> NDArray[np.float64]:
        """Return a conservative AABB half-extent (3,) for broad-phase checks."""

    def support_point(self, direction: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the point on the shape farthest in the given direction.

        Used by GJK/EPA for convex collision detection.
        Direction is in the shape's local frame.

        Reference: van den Bergen (2003), §4.3.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support GJK queries.")


# ---------------------------------------------------------------------------
# Concrete shapes
# ---------------------------------------------------------------------------


class BoxShape(CollisionShape):
    """Axis-aligned box."""

    def __init__(self, size: tuple[float, float, float]) -> None:
        self._size = np.asarray(size, dtype=np.float64)

    def half_extents_approx(self) -> NDArray[np.float64]:
        return self._size / 2.0

    def support_point(self, direction: NDArray[np.float64]) -> NDArray[np.float64]:
        h = self._size / 2.0
        return np.sign(direction) * h


class SphereShape(CollisionShape):
    """Sphere."""

    def __init__(self, radius: float) -> None:
        self._radius = float(radius)

    @property
    def radius(self) -> float:
        return self._radius

    def half_extents_approx(self) -> NDArray[np.float64]:
        r = self._radius
        return np.array([r, r, r], dtype=np.float64)

    def support_point(self, direction: NDArray[np.float64]) -> NDArray[np.float64]:
        n = np.linalg.norm(direction)
        if n < 1e-12:
            return np.array([self._radius, 0.0, 0.0])
        return (direction / n) * self._radius


class CylinderShape(CollisionShape):
    """Cylinder aligned with the local Z axis."""

    def __init__(self, radius: float, length: float) -> None:
        self._radius = float(radius)
        self._length = float(length)

    def half_extents_approx(self) -> NDArray[np.float64]:
        r = self._radius
        return np.array([r, r, self._length / 2.0], dtype=np.float64)

    def support_point(self, direction: NDArray[np.float64]) -> NDArray[np.float64]:
        # Cylinder = disk + line segment along Z
        # Disk support: project direction onto XY, normalize, scale by radius
        dxy = direction[:2]
        nxy = np.linalg.norm(dxy)
        if nxy < 1e-12:
            sx, sy = self._radius, 0.0
        else:
            sx = dxy[0] / nxy * self._radius
            sy = dxy[1] / nxy * self._radius
        # Z support: sign(dz) * half_length
        sz = np.sign(direction[2]) * (self._length / 2.0) if abs(direction[2]) > 1e-12 else 0.0
        return np.array([sx, sy, sz], dtype=np.float64)


class CapsuleShape(CollisionShape):
    """Capsule (sphere-swept line segment) aligned with local Z axis.

    Geometry: two hemispheres of radius r at z = ±length/2, connected
    by a cylinder of radius r. Total height = length + 2*radius.
    """

    def __init__(self, radius: float, length: float) -> None:
        self._radius = float(radius)
        self._length = float(length)

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def length(self) -> float:
        return self._length

    def half_extents_approx(self) -> NDArray[np.float64]:
        r = self._radius
        return np.array([r, r, self._length / 2.0 + r], dtype=np.float64)

    def support_point(self, direction: NDArray[np.float64]) -> NDArray[np.float64]:
        # Capsule = Minkowski sum of line segment + sphere
        # support = segment_support + sphere_support
        # Segment along Z: support = sign(dz) * half_length * [0,0,1]
        half_len = self._length / 2.0
        if abs(direction[2]) < 1e-12:
            seg_z = 0.0
        else:
            seg_z = half_len if direction[2] > 0 else -half_len
        seg = np.array([0.0, 0.0, seg_z])
        # Sphere: radius * normalize(direction)
        n = np.linalg.norm(direction)
        if n < 1e-12:
            return seg + np.array([self._radius, 0.0, 0.0])
        return seg + (direction / n) * self._radius


class ConvexHullShape(CollisionShape):
    """Convex hull defined by a vertex cloud.

    Used as the output format for convex decomposition (V-HACD / CoACD).
    Reference: van den Bergen (2003) §4.3 — GJK support mapping.
    """

    def __init__(self, vertices: NDArray[np.float64]) -> None:
        v = np.asarray(vertices, dtype=np.float64)
        if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] < 4:
            raise ValueError(f"ConvexHullShape requires (N>=4, 3) vertices, got {v.shape}")
        self._vertices = v

    @property
    def vertices(self) -> NDArray[np.float64]:
        return self._vertices

    def half_extents_approx(self) -> NDArray[np.float64]:
        return np.max(np.abs(self._vertices), axis=0)

    def support_point(self, direction: NDArray[np.float64]) -> NDArray[np.float64]:
        dots = self._vertices @ direction
        return self._vertices[np.argmax(dots)].copy()


class MeshShape(CollisionShape):
    """Mesh geometry with optional vertex data.

    When vertices are provided, supports GJK queries (support_point) and
    AABB computation. When only a filename is given (no vertices loaded),
    these operations raise NotImplementedError.
    """

    def __init__(self, filename: str, vertices: NDArray[np.float64] | None = None) -> None:
        self.filename = filename
        self._vertices: NDArray[np.float64] | None = None
        if vertices is not None:
            v = np.asarray(vertices, dtype=np.float64)
            if v.ndim != 2 or v.shape[1] != 3:
                raise ValueError(f"MeshShape vertices must be (N, 3), got {v.shape}")
            self._vertices = v

    @property
    def vertices(self) -> NDArray[np.float64] | None:
        return self._vertices

    def half_extents_approx(self) -> NDArray[np.float64]:
        if self._vertices is None:
            raise NotImplementedError("MeshShape AABB requires loaded vertices")
        return np.max(np.abs(self._vertices), axis=0)

    def support_point(self, direction: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._vertices is None:
            raise NotImplementedError("MeshShape support_point requires loaded vertices")
        dots = self._vertices @ direction
        return self._vertices[np.argmax(dots)].copy()


# ---------------------------------------------------------------------------
# Shape instance and body geometry container
# ---------------------------------------------------------------------------


@dataclass
class ShapeInstance:
    """A shape placed at a fixed offset within a body frame."""

    shape: CollisionShape
    origin_xyz: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    origin_rpy: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def world_pose(self, X_body: "SpatialTransform") -> "SpatialTransform":
        """Compose body world-frame transform with this shape's local offset.

        Returns X_body directly if offset is zero (fast path).
        """
        from .spatial import SpatialTransform as _ST

        if np.all(self.origin_xyz == 0) and np.all(self.origin_rpy == 0):
            return X_body
        X_offset = _ST.from_rpy(
            self.origin_rpy[0],
            self.origin_rpy[1],
            self.origin_rpy[2],
            r=self.origin_xyz,
        )
        return X_body.compose(X_offset)


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
        """Return element-wise maximum half-extents across all shapes.

        Accounts for shape offsets (origin_xyz) by expanding the AABB.
        """
        if not self.shapes:
            return np.zeros(3, dtype=np.float64)
        extents = []
        for s in self.shapes:
            he = s.shape.half_extents_approx()
            he_with_offset = he + np.abs(s.origin_xyz)
            extents.append(he_with_offset)
        return np.max(np.stack(extents), axis=0)
