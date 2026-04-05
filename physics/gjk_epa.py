"""
GJK/EPA collision detection for convex shapes.

GJK (Gilbert-Johnson-Keerthi): determines if two convex shapes intersect
by building a simplex in the Minkowski difference.

EPA (Expanding Polytope Algorithm): when GJK finds intersection, computes
penetration depth and contact normal by expanding the simplex.

References:
  van den Bergen (2003) — Collision Detection in Interactive 3D Environments.
  Casey Muratori — GJK explained simply.
  Bullet Physics — btGjkEpa2.cpp
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .geometry import CollisionShape
from .spatial import SpatialTransform, Vec3


@dataclass
class ContactManifold:
    """Result of a collision query between two shapes."""

    body_i: int  # body index (or -1 for ground)
    body_j: int  # body index (or -1 for ground)
    normal: Vec3  # contact normal (world frame, from j to i)
    depth: float  # penetration depth (positive = penetrating)
    points: list[Vec3] = field(default_factory=list)  # contact points (world)
    point_depths: list[float] | None = None  # per-point depths; None = use depth

    def depth_at(self, i: int) -> float:
        """Return penetration depth for the i-th contact point."""
        if self.point_depths is not None:
            return self.point_depths[i]
        return self.depth


# ---------------------------------------------------------------------------
# Support function
# ---------------------------------------------------------------------------


def _support(
    shape_a: CollisionShape,
    pose_a: SpatialTransform,
    shape_b: CollisionShape,
    pose_b: SpatialTransform,
    direction: Vec3,
) -> Vec3:
    """Minkowski difference support point: s_A(d) - s_B(-d)."""
    # Transform direction to local frames
    d_local_a = pose_a.R.T @ direction
    d_local_b = pose_b.R.T @ (-direction)

    # Support points in local frames
    s_a_local = shape_a.support_point(d_local_a)
    s_b_local = shape_b.support_point(d_local_b)

    # Transform back to world
    s_a_world = pose_a.R @ s_a_local + pose_a.r
    s_b_world = pose_b.R @ s_b_local + pose_b.r

    return s_a_world - s_b_world


def _support_ground(
    shape: CollisionShape,
    pose: SpatialTransform,
    direction: Vec3,
    ground_z: float = 0.0,
) -> Vec3:
    """Minkowski difference support: shape vs ground half-space (z <= ground_z)."""
    # Shape support in direction
    d_local = pose.R.T @ direction
    s_local = shape.support_point(d_local)
    s_world = pose.R @ s_local + pose.r

    # Ground half-space support in -direction
    # Ground is the set {(x,y,z) : z <= ground_z}
    # support(-d) = argmax_{p in ground} dot(p, -d)
    # For x,y: ±inf in direction of -dx, -dy → we only care about z
    # For GJK we need finite support → use shape's XY footprint
    g = np.array([0.0, 0.0, ground_z])
    if direction[2] < 0:
        g[2] = ground_z
    else:
        g[2] = -1e6  # effectively -inf but finite

    return s_world - g


# ---------------------------------------------------------------------------
# GJK Algorithm
# ---------------------------------------------------------------------------


def _triple_product(a: Vec3, b: Vec3, c: Vec3) -> Vec3:
    """(a × b) × c = b(a·c) - a(b·c)"""
    return b * np.dot(a, c) - a * np.dot(b, c)


def _do_simplex_2(simplex: list, direction: np.ndarray) -> bool:
    """Handle line segment simplex. Returns True if origin is inside."""
    B, A = simplex[0], simplex[1]  # A is newest
    AB = B - A
    AO = -A
    if np.dot(AB, AO) > 0:
        direction[:] = _triple_product(AB, AO, AB)
    else:
        simplex[:] = [A]
        direction[:] = AO
    return False


def _do_simplex_3(simplex: list, direction: np.ndarray) -> bool:
    """Handle triangle simplex."""
    C, B, A = simplex[0], simplex[1], simplex[2]
    AB = B - A
    AC = C - A
    AO = -A
    ABC_normal = np.cross(AB, AC)

    if np.dot(np.cross(ABC_normal, AC), AO) > 0:
        if np.dot(AC, AO) > 0:
            simplex[:] = [C, A]
            direction[:] = _triple_product(AC, AO, AC)
        else:
            simplex[:] = [B, A]
            return _do_simplex_2(simplex, direction)
    elif np.dot(np.cross(AB, ABC_normal), AO) > 0:
        simplex[:] = [B, A]
        return _do_simplex_2(simplex, direction)
    else:
        if np.dot(ABC_normal, AO) > 0:
            direction[:] = ABC_normal
        else:
            simplex[:] = [B, C, A]
            direction[:] = -ABC_normal
    return False


def _do_simplex_4(simplex: list, direction: np.ndarray) -> bool:
    """Handle tetrahedron simplex. Returns True if origin is inside."""
    D, C, B, A = simplex[0], simplex[1], simplex[2], simplex[3]
    AB = B - A
    AC = C - A
    AD = D - A
    AO = -A

    ABC = np.cross(AB, AC)
    ACD = np.cross(AC, AD)
    ADB = np.cross(AD, AB)

    if np.dot(ABC, AO) > 0:
        simplex[:] = [C, B, A]
        direction[:] = ABC
        return False
    if np.dot(ACD, AO) > 0:
        simplex[:] = [D, C, A]
        direction[:] = ACD
        return False
    if np.dot(ADB, AO) > 0:
        simplex[:] = [B, D, A]
        direction[:] = ADB
        return False

    return True  # Origin is inside tetrahedron


def gjk(
    shape_a: CollisionShape,
    pose_a: SpatialTransform,
    shape_b: CollisionShape,
    pose_b: SpatialTransform,
    max_iter: int = 64,
) -> tuple[bool, list]:
    """GJK intersection test.

    Returns:
        (intersecting, simplex) — if intersecting, simplex contains the
        tetrahedron enclosing the origin in Minkowski difference.
    """
    direction = pose_a.r - pose_b.r
    if np.linalg.norm(direction) < 1e-10:
        direction = np.array([1.0, 0.0, 0.0])

    A = _support(shape_a, pose_a, shape_b, pose_b, direction)
    simplex = [A]
    direction = -A.copy()

    for _ in range(max_iter):
        if np.linalg.norm(direction) < 1e-12:
            return True, simplex

        A = _support(shape_a, pose_a, shape_b, pose_b, direction)

        if np.dot(A, direction) < 0:
            return False, simplex  # No intersection

        simplex.append(A)

        n = len(simplex)
        if n == 2:
            if _do_simplex_2(simplex, direction):
                return True, simplex
        elif n == 3:
            if _do_simplex_3(simplex, direction):
                return True, simplex
        elif n == 4:
            if _do_simplex_4(simplex, direction):
                return True, simplex

    return False, simplex


# ---------------------------------------------------------------------------
# EPA Algorithm
# ---------------------------------------------------------------------------


def epa(
    shape_a: CollisionShape,
    pose_a: SpatialTransform,
    shape_b: CollisionShape,
    pose_b: SpatialTransform,
    simplex: list,
    max_iter: int = 64,
    tolerance: float = 1e-6,
) -> tuple[Vec3, float]:
    """EPA: find penetration depth and contact normal.

    Args:
        simplex: tetrahedron from GJK (4 points enclosing origin).

    Returns:
        (normal, depth) — normal points from B to A.
    """
    # Ensure we have a tetrahedron
    while len(simplex) < 4:
        # Degenerate: add points to form tetrahedron
        if len(simplex) == 1:
            d = np.array([1.0, 0.0, 0.0])
            simplex.append(_support(shape_a, pose_a, shape_b, pose_b, d))
        elif len(simplex) == 2:
            e = simplex[1] - simplex[0]
            d = np.cross(e, np.array([1, 0, 0]))
            if np.linalg.norm(d) < 1e-10:
                d = np.cross(e, np.array([0, 1, 0]))
            simplex.append(_support(shape_a, pose_a, shape_b, pose_b, d))
        elif len(simplex) == 3:
            AB = simplex[1] - simplex[0]
            AC = simplex[2] - simplex[0]
            d = np.cross(AB, AC)
            simplex.append(_support(shape_a, pose_a, shape_b, pose_b, d))

    # Build initial polytope (tetrahedron faces)
    # Each face is (i, j, k) with outward-pointing normal
    vertices = list(simplex)
    faces = [
        (0, 1, 2),
        (0, 3, 1),
        (0, 2, 3),
        (1, 3, 2),
    ]

    # Ensure normals point outward (away from origin)
    corrected_faces = []
    for f in faces:
        a, b, c = vertices[f[0]], vertices[f[1]], vertices[f[2]]
        n = np.cross(b - a, c - a)
        if np.dot(n, a) < 0:
            corrected_faces.append((f[0], f[2], f[1]))
        else:
            corrected_faces.append(f)
    faces = corrected_faces

    for _ in range(max_iter):
        # Find closest face to origin
        min_dist = float("inf")
        min_normal = np.zeros(3)

        for fi, (i, j, k) in enumerate(faces):
            a, b, c = vertices[i], vertices[j], vertices[k]
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n = n / nn
            dist = abs(np.dot(n, a))
            if dist < min_dist:
                min_dist = dist
                min_normal = n

        # New support point in direction of closest face normal
        new_point = _support(shape_a, pose_a, shape_b, pose_b, min_normal)
        new_dist = np.dot(new_point, min_normal)

        if new_dist - min_dist < tolerance:
            # Converged
            depth = min_dist
            # Ensure normal points from B to A
            if np.dot(min_normal, pose_a.r - pose_b.r) < 0:
                min_normal = -min_normal
            return min_normal, depth

        # Expand polytope: remove faces visible from new point, add new faces
        new_idx = len(vertices)
        vertices.append(new_point)

        # Find visible faces
        visible = []
        for fi, (i, j, k) in enumerate(faces):
            a = vertices[i]
            n = np.cross(vertices[j] - a, vertices[k] - a)
            if np.dot(n, new_point - a) > 0:
                visible.append(fi)

        # Find horizon edges
        edges = []
        for fi in visible:
            i, j, k = faces[fi]
            for e in [(i, j), (j, k), (k, i)]:
                # Edge is on horizon if adjacent face is not visible
                shared = False
                for fj in visible:
                    if fj == fi:
                        continue
                    face_verts = set(faces[fj])
                    if e[0] in face_verts and e[1] in face_verts:
                        shared = True
                        break
                if not shared:
                    edges.append(e)

        # Remove visible faces, add new faces from horizon edges to new point
        new_faces = [f for fi, f in enumerate(faces) if fi not in visible]
        for e0, e1 in edges:
            new_faces.append((e0, e1, new_idx))
        faces = new_faces

    # Did not converge — return best estimate
    return min_normal, min_dist


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def gjk_epa_query(
    shape_a: CollisionShape,
    pose_a: SpatialTransform,
    shape_b: CollisionShape,
    pose_b: SpatialTransform,
) -> Optional[ContactManifold]:
    """Test two convex shapes for intersection and compute contact info.

    Returns ContactManifold if penetrating, None if separated.
    """
    intersecting, simplex = gjk(shape_a, pose_a, shape_b, pose_b)
    if not intersecting:
        return None

    normal, depth = epa(shape_a, pose_a, shape_b, pose_b, simplex)
    if depth < 1e-10:
        return None

    # Contact point: midpoint of deepest penetration
    # Approximate by support points on each shape along normal
    d_local_a = pose_a.R.T @ normal
    s_a = pose_a.R @ shape_a.support_point(d_local_a) + pose_a.r
    d_local_b = pose_b.R.T @ (-normal)
    s_b = pose_b.R @ shape_b.support_point(d_local_b) + pose_b.r
    contact_point = (s_a + s_b) / 2.0

    return ContactManifold(
        body_i=-1,
        body_j=-1,
        normal=normal,
        depth=depth,
        points=[contact_point],
    )


def ground_contact_query(
    shape: CollisionShape,
    pose: SpatialTransform,
    ground_z: float = 0.0,
    margin: float = 0.0,
) -> Optional[ContactManifold]:
    """Test a convex shape against a ground plane at z=ground_z.

    Simplified: projects shape vertices (support points in -Z direction)
    below ground plane to find contact.

    Args:
        shape    : Convex collision shape.
        pose     : World-frame pose of the shape.
        ground_z : Ground plane height.
        margin   : Detection margin [m]. When > 0, contacts are detected
                   before penetration (distance < margin). Returned depth
                   can be negative (gap, no penetration yet).

    Returns:
        ContactManifold if within margin, None if separated beyond margin.
    """
    # Find lowest point of shape
    d_local = pose.R.T @ np.array([0.0, 0.0, -1.0])
    s_local = shape.support_point(d_local)
    s_world = pose.R @ s_local + pose.r

    depth = ground_z - s_world[2]  # positive = penetrating
    if depth <= -margin:
        return None

    normal = np.array([0.0, 0.0, 1.0])
    contact_point = s_world.copy()
    contact_point[2] = ground_z

    return ContactManifold(
        body_i=-1,
        body_j=-1,
        normal=normal,
        depth=depth,
        points=[contact_point],
    )


def halfspace_convex_query(
    convex_shape: CollisionShape,
    convex_pose: SpatialTransform,
    hs_normal_world: Vec3,
    hs_point_world: Vec3,
    margin: float = 0.0,
) -> Optional[ContactManifold]:
    """Test a convex shape against an infinite half-space.

    The half-space solid occupies ``dot(n, p - p0) <= 0``.  The surface
    normal ``n`` points *outward* from the solid (into free space).

    Algorithm (O(1) — one support query + one dot product):
      1.  Find the support point of the convex shape in direction ``-n``
          (the deepest point into the half-space).
      2.  ``signed_dist = dot(n, support - p0)``  (positive = above plane).
      3.  ``depth = -signed_dist``  (positive = penetrating).
      4.  Contact point = projection of the support point onto the plane.

    This generalises ``ground_contact_query`` to arbitrary plane
    orientations.  For a z-up ground plane at height *h* it is
    equivalent to ``ground_contact_query(shape, pose, ground_z=h)``.

    Reference: Drake point-pair penetration for HalfSpace vs convex.

    Args:
        convex_shape    : Convex collision shape (must support ``support_point``).
        convex_pose     : World-frame pose of the shape.
        hs_normal_world : Outward unit normal of the half-space in world frame.
        hs_point_world  : A point on the half-space surface in world frame.
        margin          : Detection margin [m] (same semantics as
                          ``ground_contact_query``).

    Returns:
        ContactManifold if penetrating or within margin, else None.
    """
    # Check for multi-point contact (Box, ConvexHull)
    verts_local = convex_shape.contact_vertices()

    if verts_local is not None:
        # Enumerate all vertices and collect those below the plane
        verts_world = (convex_pose.R @ verts_local.T).T + convex_pose.r  # (N, 3)
        signed_dists = verts_world @ hs_normal_world - np.dot(hs_normal_world, hs_point_world)
        penetrating = signed_dists < margin  # below or within margin
        if not np.any(penetrating):
            return None

        points = []
        point_depths = []
        max_depth = 0.0
        for i in np.where(penetrating)[0]:
            sd = float(signed_dists[i])
            d = -sd
            cp = verts_world[i] - sd * hs_normal_world  # project onto plane
            points.append(cp)
            point_depths.append(d)
            if d > max_depth:
                max_depth = d

        return ContactManifold(
            body_i=-1,
            body_j=-1,
            normal=hs_normal_world.copy(),
            depth=max_depth,
            points=points,
            point_depths=point_depths,
        )

    # Single-point fallback (Sphere, Capsule, Cylinder)
    neg_normal = -hs_normal_world
    d_local = convex_pose.R.T @ neg_normal
    s_local = convex_shape.support_point(d_local)
    s_world = convex_pose.R @ s_local + convex_pose.r

    signed_dist = float(np.dot(hs_normal_world, s_world - hs_point_world))
    depth = -signed_dist
    if depth <= -margin:
        return None

    contact_point = s_world - signed_dist * hs_normal_world

    return ContactManifold(
        body_i=-1,
        body_j=-1,
        normal=hs_normal_world.copy(),
        depth=depth,
        points=[contact_point],
    )
