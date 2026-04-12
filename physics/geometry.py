"""
Collision geometry description for articulated body links.

Provides abstract CollisionShape and concrete shape types (Box, Sphere,
Cylinder, Mesh) plus the BodyCollisionGeometry container that associates
shapes with a body index.

References:
  ROS URDF spec §collision element.
  Featherstone (2008) §2.1 — link geometry conventions.
  Dirk Gregorius (GDC 2015) — face topology for contact manifold generation.
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
# Face topology for polyhedral shapes (contact manifold generation)
# ---------------------------------------------------------------------------


class FaceTopology:
    """Pre-computed face/edge topology for convex polyhedra.

    Used by Sutherland-Hodgman contact clipping (方案 B) after GJK/EPA
    produces a penetration normal. Smooth shapes (Sphere, Capsule,
    Cylinder) do not have face topology — their contact manifold is
    generated analytically.

    Attributes:
        normals         : (F, 3) outward unit normals per face.
        vertices        : (V, 3) unique vertex positions (local frame).
        face_vertex_ids : list of F arrays, each containing ordered vertex
                          indices for one face polygon.
        edges           : (E, 2, 3) unique undirected edges as (start, end)
                          point pairs in local frame.

    Reference: Dirk Gregorius, "Robust Contact Creation" (GDC 2015).
    """

    __slots__ = ("normals", "vertices", "face_vertex_ids", "edges")

    def __init__(
        self,
        normals: NDArray[np.float64],
        vertices: NDArray[np.float64],
        face_vertex_ids: list[NDArray[np.intp]],
        edges: NDArray[np.float64],
    ) -> None:
        self.normals = normals
        self.vertices = vertices
        self.face_vertex_ids = face_vertex_ids
        self.edges = edges

    def find_support_face(self, direction: NDArray[np.float64]) -> int:
        """Return the face index whose normal is most aligned with *direction*.

        O(F) brute-force scan.  Sufficient for F < 200; for larger
        polyhedra, upgrade to adjacency-graph hill climbing (Q43).
        """
        return int(np.argmax(self.normals @ direction))

    def face_polygon(self, face_idx: int) -> NDArray[np.float64]:
        """Return ordered (N, 3) vertices of the face polygon."""
        return self.vertices[self.face_vertex_ids[face_idx]]

    def side_planes(self, face_idx: int) -> list[tuple[NDArray[np.float64], float]]:
        """Return side clipping planes for *face_idx*.

        Each plane is (outward_normal, offset) such that a point p is
        *inside* when ``dot(outward_normal, p) <= offset``.

        The planes are perpendicular to the face and extend inward from
        each edge of the face polygon.  Used by Sutherland-Hodgman
        clipping to restrict the incident face polygon to the reference
        face's lateral extent.
        """
        verts = self.face_polygon(face_idx)
        face_n = self.normals[face_idx]
        n_verts = len(verts)
        planes = []
        for i in range(n_verts):
            v0 = verts[i]
            v1 = verts[(i + 1) % n_verts]
            edge = v1 - v0
            # Outward side plane normal: edge × face_normal (points outward)
            side_n = np.cross(edge, face_n)
            nrm = np.linalg.norm(side_n)
            if nrm < 1e-12:
                continue
            side_n = side_n / nrm
            planes.append((side_n, float(np.dot(side_n, v0))))
        return planes

    @property
    def num_faces(self) -> int:
        return len(self.face_vertex_ids)

    @property
    def num_edges(self) -> int:
        return self.edges.shape[0]


def _build_box_face_topology(half: NDArray[np.float64]) -> FaceTopology:
    """Build FaceTopology for an axis-aligned box with given half-extents."""
    hx, hy, hz = half
    # 8 vertices — same order as BoxShape.contact_vertices
    verts = np.array(
        [
            [-hx, -hy, -hz],  # 0
            [-hx, -hy, hz],  # 1
            [-hx, hy, -hz],  # 2
            [-hx, hy, hz],  # 3
            [hx, -hy, -hz],  # 4
            [hx, -hy, hz],  # 5
            [hx, hy, -hz],  # 6
            [hx, hy, hz],  # 7
        ],
        dtype=np.float64,
    )
    # 6 faces: outward normal + CCW vertex ring (viewed from outside)
    normals = np.array(
        [
            [1, 0, 0],  # +X face
            [-1, 0, 0],  # -X face
            [0, 1, 0],  # +Y face
            [0, -1, 0],  # -Y face
            [0, 0, 1],  # +Z face
            [0, 0, -1],  # -Z face
        ],
        dtype=np.float64,
    )
    face_ids = [
        np.array([4, 6, 7, 5]),  # +X: vertices with x = +hx
        np.array([0, 1, 3, 2]),  # -X: vertices with x = -hx
        np.array([2, 3, 7, 6]),  # +Y: vertices with y = +hy
        np.array([0, 4, 5, 1]),  # -Y: vertices with y = -hy
        np.array([1, 5, 7, 3]),  # +Z: vertices with z = +hz
        np.array([0, 2, 6, 4]),  # -Z: vertices with z = -hz
    ]
    # 12 unique edges
    edge_set: set[tuple[int, int]] = set()
    for fv in face_ids:
        for i in range(len(fv)):
            a, b = int(fv[i]), int(fv[(i + 1) % len(fv)])
            key = (min(a, b), max(a, b))
            edge_set.add(key)
    edges = np.array(
        [[verts[a], verts[b]] for a, b in sorted(edge_set)],
        dtype=np.float64,
    )
    return FaceTopology(normals, verts, face_ids, edges)


def _build_convexhull_face_topology(vertices: NDArray[np.float64]) -> FaceTopology:
    """Build FaceTopology for a ConvexHullShape from its vertex cloud.

    Uses scipy.spatial.ConvexHull to compute faces and edges.
    """
    from scipy.spatial import ConvexHull

    hull = ConvexHull(vertices)
    hull_verts = vertices[hull.vertices]

    # Re-index: original indices → compact indices
    reindex = {orig: i for i, orig in enumerate(hull.vertices)}

    normals_list = []
    face_ids_list = []
    for simplex, eq in zip(hull.simplices, hull.equations):
        n = eq[:3]
        nrm = np.linalg.norm(n)
        if nrm < 1e-12:
            continue
        n = n / nrm
        normals_list.append(n)

        # Simplex gives 3 vertex indices (triangulated face)
        fids = np.array([reindex[s] for s in simplex])

        # Ensure CCW winding: if cross(v1-v0, v2-v0) · n < 0, flip
        v0, v1, v2 = hull_verts[fids[0]], hull_verts[fids[1]], hull_verts[fids[2]]
        if np.dot(np.cross(v1 - v0, v2 - v0), n) < 0:
            fids = fids[::-1]
        face_ids_list.append(fids)

    normals = np.array(normals_list, dtype=np.float64)

    # Unique edges
    edge_set: set[tuple[int, int]] = set()
    for fids in face_ids_list:
        for i in range(len(fids)):
            a, b = int(fids[i]), int(fids[(i + 1) % len(fids)])
            key = (min(a, b), max(a, b))
            edge_set.add(key)
    edges = np.array(
        [[hull_verts[a], hull_verts[b]] for a, b in sorted(edge_set)],
        dtype=np.float64,
    )

    return FaceTopology(normals, hull_verts, face_ids_list, edges)


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

    def contact_vertices(self) -> NDArray[np.float64] | None:
        """Return all vertices of the shape in local frame, or None.

        Shapes with a finite vertex set (Box, ConvexHull) return (N, 3).
        Smooth shapes (Sphere, Capsule, Cylinder) return None.
        Used by halfspace_convex_query for multi-point contact.
        """
        return None

    def face_topology(self) -> FaceTopology | None:
        """Return pre-computed face topology, or None for smooth shapes.

        Polyhedral shapes (Box, ConvexHull) return a FaceTopology with
        face normals, vertex rings, and edges.  Smooth shapes (Sphere,
        Capsule, Cylinder) return None — their contact manifold is
        generated by shape-specific analytical methods.
        """
        return None


# ---------------------------------------------------------------------------
# Concrete shapes
# ---------------------------------------------------------------------------


class BoxShape(CollisionShape):
    """Axis-aligned box."""

    def __init__(self, size: tuple[float, float, float]) -> None:
        self._size = np.asarray(size, dtype=np.float64)
        self._face_topo = _build_box_face_topology(self._size / 2.0)

    @property
    def size(self) -> NDArray[np.float64]:
        return self._size

    def half_extents_approx(self) -> NDArray[np.float64]:
        return self._size / 2.0

    def support_point(self, direction: NDArray[np.float64]) -> NDArray[np.float64]:
        h = self._size / 2.0
        return np.sign(direction) * h

    def contact_vertices(self) -> NDArray[np.float64]:
        """Return all 8 box corner vertices in local frame."""
        return self._face_topo.vertices.copy()

    def face_topology(self) -> FaceTopology:
        return self._face_topo


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

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def length(self) -> float:
        return self._length

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
        self._face_topo = _build_convexhull_face_topology(v)

    @property
    def vertices(self) -> NDArray[np.float64]:
        return self._face_topo.vertices

    def half_extents_approx(self) -> NDArray[np.float64]:
        return np.max(np.abs(self._face_topo.vertices), axis=0)

    def support_point(self, direction: NDArray[np.float64]) -> NDArray[np.float64]:
        dots = self._face_topo.vertices @ direction
        return self._face_topo.vertices[np.argmax(dots)].copy()

    def contact_vertices(self) -> NDArray[np.float64]:
        return self._face_topo.vertices

    def face_topology(self) -> FaceTopology:
        return self._face_topo


class MeshShape(CollisionShape):
    """Mesh geometry with optional vertex data.

    When vertices are provided, supports GJK queries (support_point) and
    AABB computation. When only a filename is given (no vertices loaded),
    these operations raise NotImplementedError.
    """

    def __init__(
        self,
        filename: str,
        vertices: NDArray[np.float64] | None = None,
        scale: tuple[float, float, float] | NDArray[np.float64] | None = None,
    ) -> None:
        self.filename = filename
        if scale is not None:
            self._scale = np.asarray(scale, dtype=np.float64)
        else:
            self._scale = np.ones(3, dtype=np.float64)
        self._vertices: NDArray[np.float64] | None = None
        if vertices is not None:
            v = np.asarray(vertices, dtype=np.float64)
            if v.ndim != 2 or v.shape[1] != 3:
                raise ValueError(f"MeshShape vertices must be (N, 3), got {v.shape}")
            if not np.allclose(self._scale, 1.0):
                v = v * self._scale
            self._vertices = v

    @property
    def scale(self) -> NDArray[np.float64]:
        return self._scale

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

    def contact_vertices(self) -> NDArray[np.float64] | None:
        return self._vertices


class HalfSpaceShape(CollisionShape):
    """Infinite half-space collision shape.

    In its local frame the surface is at z = 0 with outward normal +z.
    The solid occupies z <= 0.  Orientation is controlled by the
    SpatialTransform (pose) applied at the usage site, exactly like
    Drake ``HalfSpace``, Bullet ``btStaticPlaneShape``, and MuJoCo
    ``<geom type="plane">``.

    NOT compatible with GJK/EPA — use ``halfspace_convex_query()``
    from ``gjk_epa.py`` for collision detection.

    Reference: Drake geometry::HalfSpace, Bullet btStaticPlaneShape.
    """

    def half_extents_approx(self) -> NDArray[np.float64]:
        return np.array([1e6, 1e6, 1e6], dtype=np.float64)

    # support_point() intentionally NOT overridden — inherited
    # NotImplementedError from CollisionShape base class.


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
