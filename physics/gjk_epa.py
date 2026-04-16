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

from .contact_tolerances import CONTACT_CONVEX_MARGIN, CONTACT_FACE_ALIGN_THRESHOLD
from .geometry import CollisionShape, FaceTopology
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


def _perpendicular_to(v: Vec3) -> Vec3:
    """Return a unit vector perpendicular to *v*."""
    a = np.abs(v)
    if a[0] <= a[1] and a[0] <= a[2]:
        axis = np.array([1.0, 0.0, 0.0])
    elif a[1] <= a[2]:
        axis = np.array([0.0, 1.0, 0.0])
    else:
        axis = np.array([0.0, 0.0, 1.0])
    p = np.cross(v, axis)
    n = np.linalg.norm(p)
    if n < 1e-15:
        return axis
    return p / n


def _do_simplex_2(simplex: list, direction: np.ndarray) -> bool:
    """Handle line segment simplex. Returns True if origin is inside."""
    B, A = simplex[0], simplex[1]  # A is newest
    AB = B - A
    AO = -A
    if np.dot(AB, AO) > 0:
        new_dir = _triple_product(AB, AO, AB)
        # When AO is parallel to AB (origin lies ON the segment),
        # the triple product is zero.  Pick a perpendicular direction
        # to continue building the simplex instead of terminating
        # with a degenerate 2-point simplex.
        if np.linalg.norm(new_dir) < 1e-12:
            new_dir = _perpendicular_to(AB)
        direction[:] = new_dir
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
    margin: float = CONTACT_CONVEX_MARGIN,
) -> tuple[Vec3, float]:
    """EPA: find penetration depth and contact normal.

    Runs on *inflated* shapes (original + margin), following the PhysX
    design.  Inflating both shapes by ``margin`` guarantees that all faces
    of the initial polytope have ``dist_to_origin ≥ margin > 0``, making
    the \"face through origin\" degenerate case impossible.

    The returned depth is corrected: ``depth = EPA_depth - 2*margin``.
    Normal direction is unchanged (isotropic inflation preserves it).

    Args:
        simplex: tetrahedron from GJK (4 points enclosing origin).
        margin:  Per-shape inflation radius [m]. Same value as the margin
                 used in the two-phase gjk_epa_query pipeline.

    Returns:
        (normal, depth) — normal points from B to A.

    Reference:
        PhysX GuEPA.cpp — core-shape EPA + margin readback.
        CONTACT_CONVEX_MARGIN default defined in contact_tolerances.py.
    """
    # --- Build initial polytope that strictly contains the origin ---
    #
    # EPA runs on INFLATED shapes (original + margin per shape).
    # Inflating the support function by margin guarantees that all faces
    # of the initial polytope have dist_to_origin ≥ margin > 0, making
    # the "face through origin" degenerate case impossible (PhysX design).
    #
    # For < 4-point simplexes, we additionally build a hexahedron to
    # ensure geometric enclosure.

    def _sup(d):
        # Inflated Minkowski difference support:
        # s_A(d) + margin*d̂  -  (s_B(-d) + margin*(-d̂))
        # = _support(A,B,d) + 2*margin*d̂
        base = _support(shape_a, pose_a, shape_b, pose_b, d)
        d_norm = np.linalg.norm(d)
        if d_norm > 1e-15 and margin > 0:
            base = base + (2.0 * margin / d_norm) * d
        return base

    if len(simplex) <= 2:
        # --- Hexahedron from line (or point) ---
        if len(simplex) == 1:
            simplex.append(_sup(np.array([1.0, 0.0, 0.0])))
        line_dir = simplex[1] - simplex[0]
        ln = np.linalg.norm(line_dir)
        if ln < 1e-15:
            line_dir = np.array([1.0, 0.0, 0.0])
        else:
            line_dir = line_dir / ln
        p1 = _perpendicular_to(line_dir)
        p2 = np.cross(line_dir, p1)

        v0 = simplex[0]
        v1 = simplex[1]
        v2 = _sup(p1)
        v3 = _sup(-p1)
        v4 = _sup(p2)
        v5 = _sup(-p2)
        vertices = [v0, v1, v2, v3, v4, v5]
        # 8 triangular faces of the hexahedron (two pyramids sharing a
        # quadrilateral equator: v2-v4-v3-v5)
        faces = [
            (0, 2, 4),
            (0, 4, 3),
            (0, 3, 5),
            (0, 5, 2),
            (1, 4, 2),
            (1, 3, 4),
            (1, 5, 3),
            (1, 2, 5),
        ]
    elif len(simplex) == 3:
        # Try both normal directions for the 4th point
        AB = simplex[1] - simplex[0]
        AC = simplex[2] - simplex[0]
        n = np.cross(AB, AC)
        d_plus = _sup(n)
        d_minus = _sup(-n)
        # Pick the one that puts origin more inside
        # (larger minimum signed distance from origin to all faces)
        for candidate in [d_plus, d_minus]:
            simplex_try = list(simplex) + [candidate]
            # Quick check: is origin inside?
            inside = True
            for i, (a, b, c) in enumerate([(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]):
                va, vb, vc = simplex_try[a], simplex_try[b], simplex_try[c]
                fn = np.cross(vb - va, vc - va)
                if np.dot(fn, -va) < -1e-10:
                    inside = False
                    break
            if inside:
                simplex = simplex_try
                break
        if len(simplex) < 4:
            simplex.append(d_plus)  # fallback
        vertices = list(simplex)
        faces = [
            (0, 1, 2),
            (0, 3, 1),
            (0, 2, 3),
            (1, 3, 2),
        ]
    else:
        # Already 4 points — check for degenerate tetrahedron.
        #
        # A degenerate tetrahedron has one or more faces whose plane passes
        # through (or very near) the origin.  EPA will select that face as
        # "closest" on iteration 0 and expand in the wrong direction,
        # converging to an incorrect depth/normal.
        #
        # Detection: for each face compute |dot(face_normal, vertex)| which
        # equals the distance from the origin to the face plane.  If the
        # minimum across all 4 faces is below DEGENERATE_FACE_EPS, the
        # tetrahedron is degenerate and we rebuild as a hexahedron.
        #
        # Reference: Bullet btGjkEpa2.cpp hexahedron fallback.
        _DEGENERATE_FACE_EPS = 1e-8
        tet_faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
        min_face_dist = float("inf")
        for fa, fb, fc in tet_faces:
            va, vb, vc = simplex[fa], simplex[fb], simplex[fc]
            fn = np.cross(vb - va, vc - va)
            fn_norm = np.linalg.norm(fn)
            if fn_norm < 1e-15:
                min_face_dist = 0.0
                break
            fn = fn / fn_norm
            min_face_dist = min(min_face_dist, abs(float(np.dot(fn, va))))

        if min_face_dist < _DEGENERATE_FACE_EPS:
            # Degenerate — rebuild as hexahedron using the line between
            # the two most-separated simplex vertices as the spine.
            dists = [np.linalg.norm(simplex[i] - simplex[j]) for i in range(4) for j in range(i + 1, 4)]
            pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
            best = int(np.argmax(dists))
            i0, i1 = pairs[best]
            line_dir = simplex[i1] - simplex[i0]
            ln = np.linalg.norm(line_dir)
            if ln < 1e-15:
                line_dir = np.array([1.0, 0.0, 0.0])
            else:
                line_dir = line_dir / ln
            p1 = _perpendicular_to(line_dir)
            p2 = np.cross(line_dir, p1)
            v0 = simplex[i0]
            v1 = simplex[i1]
            v2 = _sup(p1)
            v3 = _sup(-p1)
            v4 = _sup(p2)
            v5 = _sup(-p2)
            vertices = [v0, v1, v2, v3, v4, v5]
            faces = [
                (0, 2, 4),
                (0, 4, 3),
                (0, 3, 5),
                (0, 5, 2),
                (1, 4, 2),
                (1, 3, 4),
                (1, 5, 3),
                (1, 2, 5),
            ]
        else:
            vertices = list(simplex)
            faces = tet_faces

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
        # Find closest face to origin, skipping degenerate faces.
        # A face is degenerate when dist_to_origin ≈ 0 (plane through origin);
        # selecting it as "closest" would expand in an arbitrary direction.
        _DEGENERATE_SKIP_EPS = 1e-8
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
            if dist < _DEGENERATE_SKIP_EPS:
                continue  # skip degenerate face
            if dist < min_dist:
                min_dist = dist
                min_normal = n

        if min_dist == float("inf"):
            # All faces degenerate (shouldn't happen with hexahedron init);
            # fall back to centroid-based direction.
            centroid = np.mean(vertices, axis=0)
            cn = np.linalg.norm(centroid)
            min_normal = centroid / cn if cn > 1e-15 else np.array([0.0, 0.0, 1.0])
            min_dist = float(np.dot(min_normal, vertices[0]))

        # New support point in direction of closest face normal (inflated)
        new_point = _sup(min_normal)
        new_dist = np.dot(new_point, min_normal)

        if new_dist - min_dist < tolerance:
            # Converged — subtract 2*margin to recover true penetration depth
            depth = max(min_dist - 2.0 * margin, 0.0)
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

    # Did not converge — return best estimate (subtract margin correction)
    return min_normal, max(min_dist - 2.0 * margin, 0.0)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Contact manifold generation (方案 B: face clipping + edge-edge)
# ---------------------------------------------------------------------------

# Alignment threshold: if both shapes' best face dot < this, switch to
# edge-edge path.  For a box, edge-edge normals give dot ≈ 0.707 against
# face normals.  Threshold ~0.9 (~25°) catches edge-edge while letting
# face-edge (one side well-aligned) through the face clipping path.
_FACE_ALIGN_THRESHOLD = CONTACT_FACE_ALIGN_THRESHOLD


def _clip_polygon_by_plane(
    polygon: list[Vec3],
    plane_normal: Vec3,
    plane_offset: float,
) -> list[Vec3]:
    """Sutherland-Hodgman: clip polygon against half-plane dot(n, p) <= d.

    Reference: Erin Catto, "Contact Manifolds" (GDC 2007).
    """
    if not polygon:
        return []
    output: list[Vec3] = []
    n = len(polygon)
    for i in range(n):
        current = polygon[i]
        nxt = polygon[(i + 1) % n]
        d_curr = np.dot(plane_normal, current) - plane_offset
        d_nxt = np.dot(plane_normal, nxt) - plane_offset
        if d_curr <= 0:  # current inside
            output.append(current)
            if d_nxt > 0:  # next outside → add intersection
                t = d_curr / (d_curr - d_nxt)
                output.append(current + t * (nxt - current))
        elif d_nxt <= 0:  # current outside, next inside → add intersection
            t = d_curr / (d_curr - d_nxt)
            output.append(current + t * (nxt - current))
    return output


def _segment_closest_points(
    p0: Vec3,
    p1: Vec3,
    q0: Vec3,
    q1: Vec3,
) -> tuple[Vec3, Vec3]:
    """Closest points between two line segments P0-P1 and Q0-Q1.

    Returns (point_on_P, point_on_Q).
    Reference: Ericson (2004) §5.1.9.
    """
    d1 = p1 - p0
    d2 = q1 - q0
    r = p0 - q0
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))

    EPS = 1e-12
    if a < EPS and e < EPS:
        return p0.copy(), q0.copy()
    if a < EPS:
        t = np.clip(f / e, 0.0, 1.0)
        return p0.copy(), q0 + t * d2
    c = float(np.dot(d1, r))
    if e < EPS:
        s = np.clip(-c / a, 0.0, 1.0)
        return p0 + s * d1, q0.copy()

    b = float(np.dot(d1, d2))
    denom = a * e - b * b

    if denom > EPS:
        s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
    else:
        s = 0.0

    t = (b * s + f) / e
    if t < 0.0:
        t = 0.0
        s = np.clip(-c / a, 0.0, 1.0)
    elif t > 1.0:
        t = 1.0
        s = np.clip((b - c) / a, 0.0, 1.0)

    return p0 + s * d1, q0 + t * d2


def _find_best_edge_on_face(
    topo: FaceTopology,
    face_idx: int,
    pose: SpatialTransform,
    epa_normal: Vec3,
) -> tuple[Vec3, Vec3]:
    """Return the (world-frame) edge on *face_idx* most perpendicular to *epa_normal*.

    For edge-edge contacts, the contact edge is the one whose direction
    is most perpendicular to the EPA normal (smallest |dot(edge_dir, n)|).
    """
    poly = topo.face_polygon(face_idx)
    nv = len(poly)
    best_dot = float("inf")
    best_p0 = pose.R @ poly[0] + pose.r
    best_p1 = pose.R @ poly[1] + pose.r
    for i in range(nv):
        v0_w = pose.R @ poly[i] + pose.r
        v1_w = pose.R @ poly[(i + 1) % nv] + pose.r
        edge_dir = v1_w - v0_w
        edge_len = np.linalg.norm(edge_dir)
        if edge_len < 1e-12:
            continue
        edge_dir = edge_dir / edge_len
        d = abs(float(np.dot(edge_dir, epa_normal)))
        if d < best_dot:
            best_dot = d
            best_p0 = v0_w
            best_p1 = v1_w
    return best_p0, best_p1


def build_contact_manifold(
    shape_a: CollisionShape,
    pose_a: SpatialTransform,
    shape_b: CollisionShape,
    pose_b: SpatialTransform,
    normal: Vec3,
    depth: float,
) -> ContactManifold:
    """Generate a multi-point contact manifold from GJK/EPA output.

    Given the penetration *normal* (from B to A) and *depth*, produces
    contact points via face identification + Sutherland-Hodgman clipping,
    or edge-edge closest-point for edge contacts.

    Falls back to single support-midpoint for smooth shape pairs.

    Reference:
      Dirk Gregorius, "Robust Contact Creation" (GDC 2015).
      ODE dBoxBox2 — SAT + clipping reference implementation.
    """
    topo_a = shape_a.face_topology()
    topo_b = shape_b.face_topology()

    # If either shape lacks face topology (smooth shape), fall back to
    # single-point support midpoint.
    if topo_a is None or topo_b is None:
        return _single_point_fallback(shape_a, pose_a, shape_b, pose_b, normal, depth)

    # --- Face identification (in local frames) ---
    # EPA normal points from B to A.  The contact face on each shape
    # points TOWARD the other body:
    #   A's contact face → anti-parallel to EPA normal → search -normal
    #   B's contact face → parallel to EPA normal     → search +normal
    n_local_a = pose_a.R.T @ (-normal)  # A's face toward B
    n_local_b = pose_b.R.T @ normal  # B's face toward A
    face_a = topo_a.find_support_face(n_local_a)
    face_b = topo_b.find_support_face(n_local_b)
    dot_a = float(np.dot(topo_a.normals[face_a], n_local_a))
    dot_b = float(np.dot(topo_b.normals[face_b], n_local_b))

    # --- Edge-edge detection ---
    if dot_a < _FACE_ALIGN_THRESHOLD and dot_b < _FACE_ALIGN_THRESHOLD:
        return _edge_edge_manifold(
            topo_a,
            face_a,
            pose_a,
            topo_b,
            face_b,
            pose_b,
            normal,
            depth,
        )

    # --- Face clipping path (handles face-face, face-edge, vertex-face) ---
    # Choose reference shape: the one with better face alignment.
    # The reference face normal (world) points outward from the reference
    # shape's contact face — i.e., TOWARD the incident shape.
    if dot_a >= dot_b:
        ref_topo, ref_face, ref_pose = topo_a, face_a, pose_a
        inc_topo, inc_face, inc_pose = topo_b, face_b, pose_b
        # A's contact face normal points toward B → anti-parallel to EPA normal
        ref_normal_world = pose_a.R @ topo_a.normals[face_a]
    else:
        ref_topo, ref_face, ref_pose = topo_b, face_b, pose_b
        inc_topo, inc_face, inc_pose = topo_a, face_a, pose_a
        # B's contact face normal points toward A → parallel to EPA normal
        ref_normal_world = pose_b.R @ topo_b.normals[face_b]

    return _face_clip_manifold(
        ref_topo,
        ref_face,
        ref_pose,
        inc_topo,
        inc_face,
        inc_pose,
        ref_normal_world,
        normal,
        depth,
    )


def _single_point_fallback(
    shape_a: CollisionShape,
    pose_a: SpatialTransform,
    shape_b: CollisionShape,
    pose_b: SpatialTransform,
    normal: Vec3,
    depth: float,
) -> ContactManifold:
    """Original single-point contact via support-midpoint."""
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


def _face_clip_manifold(
    ref_topo: FaceTopology,
    ref_face: int,
    ref_pose: SpatialTransform,
    inc_topo: FaceTopology,
    inc_face: int,
    inc_pose: SpatialTransform,
    ref_normal_world: Vec3,
    epa_normal: Vec3,
    epa_depth: float,
) -> ContactManifold:
    """Generate contact points by clipping the incident face against the
    reference face's side planes (Sutherland-Hodgman).

    Handles face-face (4 pts), face-edge (2 pts), vertex-face (1 pt)
    uniformly — the clipping naturally degrades.
    """
    # Reference face: world-frame polygon, normal, and a point on the plane
    ref_poly_local = ref_topo.face_polygon(ref_face)
    ref_poly_world = [(ref_pose.R @ v + ref_pose.r) for v in ref_poly_local]
    ref_n_world = ref_normal_world
    ref_point_on_plane = ref_poly_world[0]

    # Side planes in world frame
    # Each side plane passes through edge i (vertex i → vertex i+1).
    # Transform normal to world; recompute offset using vertex i in world.
    side_planes_local = ref_topo.side_planes(ref_face)
    side_planes_world = []
    for i, (sn_local, _) in enumerate(side_planes_local):
        sn_world = ref_pose.R @ sn_local
        v_world = ref_poly_world[i]  # vertex i lies on this side plane
        sd_world = float(np.dot(sn_world, v_world))
        side_planes_world.append((sn_world, sd_world))

    # Incident face: world-frame polygon
    inc_poly_local = inc_topo.face_polygon(inc_face)
    polygon = [inc_pose.R @ v + inc_pose.r for v in inc_poly_local]

    # Clip incident polygon against each side plane
    for plane_n, plane_d in side_planes_world:
        polygon = _clip_polygon_by_plane(polygon, plane_n, plane_d)
        if not polygon:
            break

    if not polygon:
        # Degenerate: clipping eliminated everything — fall back
        return ContactManifold(
            body_i=-1,
            body_j=-1,
            normal=epa_normal,
            depth=epa_depth,
            points=[],
            point_depths=[],
        )

    # Filter: keep only points below (or on) the reference face plane
    ref_d = float(np.dot(ref_n_world, ref_point_on_plane))
    points = []
    point_depths = []
    for p in polygon:
        signed_dist = float(np.dot(ref_n_world, p)) - ref_d
        pt_depth = -signed_dist  # positive = below reference plane
        if pt_depth > -1e-6:  # small tolerance for on-plane points
            # Project point onto contact plane (midpoint correction)
            contact_pt = p - signed_dist * ref_n_world
            points.append(contact_pt)
            point_depths.append(max(pt_depth, 0.0))

    if not points:
        return ContactManifold(
            body_i=-1,
            body_j=-1,
            normal=epa_normal,
            depth=epa_depth,
            points=[],
            point_depths=[],
        )

    return ContactManifold(
        body_i=-1,
        body_j=-1,
        normal=epa_normal,
        depth=max(point_depths),
        points=points,
        point_depths=point_depths,
    )


def _edge_edge_manifold(
    topo_a: FaceTopology,
    face_a: int,
    pose_a: SpatialTransform,
    topo_b: FaceTopology,
    face_b: int,
    pose_b: SpatialTransform,
    normal: Vec3,
    depth: float,
) -> ContactManifold:
    """Generate a single contact point for edge-edge contact.

    Finds the most perpendicular edge on each shape's support face,
    then computes the closest point pair between the two line segments.
    """
    p0_a, p1_a = _find_best_edge_on_face(topo_a, face_a, pose_a, normal)
    p0_b, p1_b = _find_best_edge_on_face(topo_b, face_b, pose_b, normal)

    pt_a, pt_b = _segment_closest_points(p0_a, p1_a, p0_b, p1_b)
    contact_point = (pt_a + pt_b) / 2.0

    return ContactManifold(
        body_i=-1,
        body_j=-1,
        normal=normal,
        depth=depth,
        points=[contact_point],
    )


# ---------------------------------------------------------------------------
# GJK closest-distance mode (for convex margin)
# ---------------------------------------------------------------------------


def _support_shrunk(
    shape: CollisionShape,
    pose: SpatialTransform,
    direction: Vec3,
    margin: float,
) -> Vec3:
    """Support point on shape shrunk by *margin* (Minkowski erosion).

    Moves the support point inward by *margin* along the query direction.
    This is the standard Bullet/Jolt formula for convex radius.
    """
    d_local = pose.R.T @ direction
    s_local = shape.support_point(d_local)
    s_world = pose.R @ s_local + pose.r
    d_norm = np.linalg.norm(direction)
    if d_norm > 1e-15:
        s_world = s_world - margin * (direction / d_norm)
    return s_world


def _support_shrunk_diff(
    shape_a: CollisionShape,
    pose_a: SpatialTransform,
    shape_b: CollisionShape,
    pose_b: SpatialTransform,
    direction: Vec3,
    margin: float,
) -> Vec3:
    """Minkowski difference support on margin-shrunk shapes."""
    return _support_shrunk(shape_a, pose_a, direction, margin) - _support_shrunk(
        shape_b, pose_b, -direction, margin
    )


def gjk_distance(
    shape_a: CollisionShape,
    pose_a: SpatialTransform,
    shape_b: CollisionShape,
    pose_b: SpatialTransform,
    margin: float = 0.0,
    max_iter: int = 64,
) -> Optional[tuple[float, Vec3, Vec3]]:
    """GJK closest-distance between two (optionally margin-shrunk) shapes.

    Runs GJK in closest-point mode on shapes shrunk by *margin*.  Returns
    the distance and closest points in world frame, or ``None`` if the
    shrunk shapes intersect (deep penetration — caller should use EPA).

    Args:
        shape_a, pose_a : First shape and its world-frame pose.
        shape_b, pose_b : Second shape and its world-frame pose.
        margin          : Minkowski erosion applied to both shapes [m].
        max_iter        : Maximum GJK iterations.

    Returns:
        ``(distance, closest_on_A, closest_on_B)`` if separated, else ``None``.

    Reference:
        Gino van den Bergen (2003) §4.3 — GJK distance algorithm.
        Bullet btGjkPairDetector — closest-point mode.
    """
    direction = pose_a.r - pose_b.r
    if np.linalg.norm(direction) < 1e-10:
        return None  # centers coincide → intersecting

    simplex_pts_a: list[Vec3] = []  # support points on A (world)
    simplex_pts_b: list[Vec3] = []  # support points on B (world)
    simplex: list[Vec3] = []  # Minkowski difference points

    def _sup(d: Vec3) -> tuple[Vec3, Vec3, Vec3]:
        sa = _support_shrunk(shape_a, pose_a, d, margin)
        sb = _support_shrunk(shape_b, pose_b, -d, margin)
        return sa, sb, sa - sb

    sa, sb, w = _sup(direction)
    simplex_pts_a.append(sa)
    simplex_pts_b.append(sb)
    simplex.append(w)
    direction = -w.copy()

    for _ in range(max_iter):
        if np.linalg.norm(direction) < 1e-12:
            return None  # origin on simplex → intersecting

        sa, sb, w = _sup(direction)

        # If new point doesn't advance toward origin, we've found closest
        if np.dot(w, direction) < np.dot(simplex[0], direction) - 1e-10:
            # Compute closest points from simplex barycentric coords
            # For simplicity: use the last simplex point pair
            # (full barycentric interpolation is complex; this is accurate
            # enough for the margin use case where we only need distance)
            break

        simplex_pts_a.append(sa)
        simplex_pts_b.append(sb)
        simplex.append(w)

        # Reduce simplex to closest feature
        direction_new = np.zeros(3)
        if (
            _do_simplex_2(simplex, direction_new)
            if len(simplex) == 2
            else (
                _do_simplex_3(simplex, direction_new)
                if len(simplex) == 3
                else _do_simplex_4(simplex, direction_new)
            )
        ):
            return None  # origin inside simplex → intersecting

        direction = direction_new
        # Trim support point lists to match reduced simplex
        # (approximate: keep last len(simplex) entries)
        simplex_pts_a = simplex_pts_a[-len(simplex) :]
        simplex_pts_b = simplex_pts_b[-len(simplex) :]

    # Closest point on Minkowski difference = last simplex point
    # Closest points on A and B = corresponding support points
    closest_diff = simplex[-1]
    dist = float(np.linalg.norm(closest_diff))
    if dist < 1e-10:
        return None  # intersecting

    closest_a = simplex_pts_a[-1]
    closest_b = simplex_pts_b[-1]
    return dist, closest_a, closest_b


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Sphere analytical dispatch (bypasses GJK/EPA for smooth shapes)
# ---------------------------------------------------------------------------


def _sphere_any_manifold(
    sph: CollisionShape,  # SphereShape at runtime
    pose_sph: SpatialTransform,
    other: CollisionShape,
    pose_other: SpatialTransform,
) -> Optional[ContactManifold]:
    """Analytical contact between a sphere and any convex shape.

    Dispatch table:
      sphere-sphere   : closed-form (centre distance).
      sphere-capsule  : closed-form (point-segment distance).
      sphere-polytope : GJK intersection test + EPA.
                        EPA is numerically stable for sphere-vs-polyhedral
                        (only one smooth shape); degeneracy only arises when
                        *both* shapes are smooth (sphere-sphere, handled above).

    Normal convention: points from *other* toward *sph* (i.e. from B to A
    when sph=A).

    Reference:
      Ericson (2004) §5.1 Closest Point on Convex Shape to Point.
      PhysX narrow-phase: dedicated sphere-sphere / sphere-capsule paths.
    """
    from .geometry import CapsuleShape, SphereShape

    r_sph = sph.radius
    c_sph = pose_sph.r  # sphere centre in world frame

    # --- sphere vs sphere ---
    if isinstance(other, SphereShape):
        r_other = other.radius
        delta = c_sph - pose_other.r
        dist = float(np.linalg.norm(delta))
        depth = r_sph + r_other - dist
        if depth <= 0.0:
            return None
        if dist < 1e-12:
            # Coincident centres — pick +Z as fallback normal
            normal = np.array([0.0, 0.0, 1.0])
        else:
            normal = delta / dist
        cp = pose_other.r + normal * r_other
        return ContactManifold(body_i=-1, body_j=-1, normal=normal, depth=depth, points=[cp])

    # --- sphere vs capsule ---
    if isinstance(other, CapsuleShape):
        from .capsule_collision import _capsule_axis_endpoints, _segment_closest_points

        r_other = other.radius
        e0, e1, _ = _capsule_axis_endpoints(other, pose_other)
        # Closest point on capsule core segment to sphere centre
        p_seg, _ = _segment_closest_points(e0, e1, c_sph, c_sph)
        delta = c_sph - p_seg
        dist = float(np.linalg.norm(delta))
        depth = r_sph + r_other - dist
        if depth <= 0.0:
            return None
        if dist < 1e-12:
            # Centre on segment axis — pick any perpendicular
            axis = e1 - e0
            ax_n = float(np.linalg.norm(axis))
            axis = axis / ax_n if ax_n > 1e-12 else np.array([0.0, 0.0, 1.0])
            ref = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(axis, ref))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            normal = np.cross(axis, ref)
            normal /= float(np.linalg.norm(normal))
        else:
            normal = delta / dist
        cp = p_seg + normal * r_other
        return ContactManifold(body_i=-1, body_j=-1, normal=normal, depth=depth, points=[cp])

    # --- sphere vs any other convex shape (Box, Cylinder, ConvexHull, …) ---
    #
    # The zero-radius-point trick (gjk_distance with SphereShape(0.0)) is
    # unreliable for tessellated shapes: GJK finds the closest *vertex* of the
    # prism/polytope, not the closest *face point*.  For a sphere directly
    # above a cylinder's flat cap, the rim vertex is farther than the face
    # centre, so the computed distance exceeds the sphere radius and the
    # contact is missed entirely.
    #
    # Instead, run GJK on the full sphere:
    #   • intersecting  → EPA gives the correct normal + depth for pen ≳ 2 mm.
    #   • not intersecting → sphere surface is separated from *other*; no contact.
    #
    # EPA accuracy at very shallow penetrations (< 2 mm) degrades for
    # tessellated shapes due to simplex degeneracy, but (a) the convex-margin
    # pipeline (test_convex_margin) requires pen = 3*MARGIN = 3 mm, which is
    # reliably handled, and (b) the simulation slop filter (default 5 mm)
    # removes sub-millimetre contacts before they reach the solver.
    #
    # Reference: Ericson (2004) §9.4 — EPA; see Q47 (OPEN_QUESTIONS.md) for
    # the principled long-term fix (Jolt InnerShape + ConvexRadius).
    intersecting, simplex = gjk(sph, pose_sph, other, pose_other)
    if not intersecting:
        # Sphere surface does not penetrate *other* → no contact.
        return None
    normal, depth = epa(sph, pose_sph, other, pose_other, simplex, margin=0.0)
    if depth < 1e-10:
        return None
    # Contact point: on *other*'s surface, along -normal from sphere centre.
    cp = c_sph - normal * r_sph
    return ContactManifold(body_i=-1, body_j=-1, normal=normal, depth=depth, points=[cp])


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def gjk_epa_query(
    shape_a: CollisionShape,
    pose_a: SpatialTransform,
    shape_b: CollisionShape,
    pose_b: SpatialTransform,
    margin: float = CONTACT_CONVEX_MARGIN,
) -> Optional[ContactManifold]:
    """Test two convex shapes for intersection and compute contact manifold.

    Two-phase pipeline (Jolt / Bullet convex-margin approach):

    Phase 1 — GJK closest-distance on margin-shrunk shapes.
      If distance < 2*margin (shallow contact), resolve without EPA.
      Normal = separation direction (numerically stable, no polytope).

    Phase 2 — Full GJK + EPA on original shapes (deep penetration).
      Only reached when shapes penetrate beyond the margin.

    Then builds a multi-point contact manifold via face clipping
    (Sutherland-Hodgman) for polyhedral shapes, or single-point fallback
    for smooth shapes.

    Capsule pairs dispatch to analytical multi-point handlers and skip
    GJK/EPA entirely.

    Args:
        margin : Convex margin [m]. Default = CONTACT_CONVEX_MARGIN (1e-3).
                 Set to 0 to disable and use pure GJK+EPA.
                 When InterfaceMaterial is implemented (Q18.9), this will
                 be read from ShapeInstance.interface.margin instead.

    Returns ContactManifold if penetrating (or within margin), None if separated.
    """
    from .capsule_collision import (
        capsule_box_manifold,
        capsule_capsule_manifold,
        capsule_convexhull_manifold,
        capsule_cylinder_manifold,
    )
    from .geometry import BoxShape, CapsuleShape, ConvexHullShape, CylinderShape, SphereShape

    # -----------------------------------------------------------------------
    # SphereShape: analytical dispatch — bypass GJK/EPA entirely.
    #
    # Smooth shapes (sphere, capsule hemispheres) have continuous support
    # functions that produce degenerate GJK simplices near tangent contact.
    # EPA then picks an arbitrary face → wrong normal/depth.  The fix is to
    # compute contact analytically from the sphere centre and the closest
    # point on the other shape, which is always numerically exact.
    #
    # See Q47 (OPEN_QUESTIONS.md) for the principled long-term solution
    # (Jolt-style inner-shape + ConvexRadius architecture).
    # -----------------------------------------------------------------------
    a_is_sph = isinstance(shape_a, SphereShape)
    b_is_sph = isinstance(shape_b, SphereShape)
    if a_is_sph or b_is_sph:
        # Canonicalize: sphere as first argument
        if b_is_sph and not a_is_sph:
            sph, pose_sph = shape_b, pose_b
            other_s, pose_other_s = shape_a, pose_a
            flip_s = True
        else:
            sph, pose_sph = shape_a, pose_a
            other_s, pose_other_s = shape_b, pose_b
            flip_s = False

        manifold = _sphere_any_manifold(sph, pose_sph, other_s, pose_other_s)
        if manifold is not None and flip_s:
            manifold.normal = -manifold.normal
        return manifold

    a_is_cap = isinstance(shape_a, CapsuleShape)
    b_is_cap = isinstance(shape_b, CapsuleShape)
    if a_is_cap or b_is_cap:
        # Canonicalize: capsule as first argument
        if b_is_cap and not a_is_cap:
            cap, pose_cap = shape_b, pose_b
            other, pose_other = shape_a, pose_a
            flip = True
        else:
            cap, pose_cap = shape_a, pose_a
            other, pose_other = shape_b, pose_b
            flip = False

        manifold: Optional[ContactManifold] = None
        dispatched = False
        if isinstance(other, CapsuleShape):
            manifold = capsule_capsule_manifold(cap, pose_cap, other, pose_other)
            dispatched = True
        elif isinstance(other, BoxShape):
            manifold = capsule_box_manifold(cap, pose_cap, other, pose_other)
            dispatched = True
        elif isinstance(other, CylinderShape):
            manifold = capsule_cylinder_manifold(cap, pose_cap, other, pose_other)
            dispatched = True
        elif isinstance(other, ConvexHullShape):
            manifold = capsule_convexhull_manifold(cap, pose_cap, other, pose_other)
            dispatched = True

        if dispatched:
            if manifold is not None and flip:
                manifold.normal = -manifold.normal
            return manifold
        # capsule vs sphere: already handled above (sphere dispatch runs first).

    # --- Phase 1: GJK closest-distance on margin-shrunk shapes ---
    if margin > 0:
        result = gjk_distance(shape_a, pose_a, shape_b, pose_b, margin=margin)
        if result is not None:
            dist, cp_a, cp_b = result
            # dist is the distance between margin-shrunk shapes.
            # Contact occurs when the margin "halos" overlap: dist < 2*margin.
            # depth = 2*margin - dist  (positive = penetrating into margin zone)
            if dist < 2.0 * margin:
                # Shallow contact — resolved without EPA
                sep = cp_a - cp_b
                sep_len = float(np.linalg.norm(sep))
                if sep_len > 1e-10:
                    normal = sep / sep_len
                else:
                    normal = pose_a.r - pose_b.r
                    n_len = float(np.linalg.norm(normal))
                    normal = normal / n_len if n_len > 1e-10 else np.array([0.0, 0.0, 1.0])
                depth = 2.0 * margin - dist
                contact_point = (cp_a + cp_b) / 2.0
                return ContactManifold(
                    body_i=-1,
                    body_j=-1,
                    normal=normal,
                    depth=depth,
                    points=[contact_point],
                )
            else:
                return None  # separated beyond margin
        # result is None → shrunk shapes intersect → fall through to EPA

    # --- Phase 2: Full GJK + EPA on original shapes ---
    intersecting, simplex = gjk(shape_a, pose_a, shape_b, pose_b)
    if not intersecting:
        return None

    normal, depth = epa(shape_a, pose_a, shape_b, pose_b, simplex, margin=margin)
    if depth < 1e-10:
        return None

    return build_contact_manifold(shape_a, pose_a, shape_b, pose_b, normal, depth)


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
    # Capsule / Cylinder: dispatch to analytical multi-point handlers.
    from .capsule_collision import capsule_halfspace_manifold
    from .cylinder_collision import cylinder_halfspace_manifold
    from .geometry import CapsuleShape, CylinderShape

    n_up = np.array([0.0, 0.0, 1.0])
    p_ground = np.array([0.0, 0.0, ground_z])
    if isinstance(shape, CapsuleShape):
        return capsule_halfspace_manifold(shape, pose, n_up, p_ground, margin=margin)
    if isinstance(shape, CylinderShape):
        return cylinder_halfspace_manifold(shape, pose, n_up, p_ground, margin=margin)

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
    # Capsule / Cylinder: dispatch to analytical multi-point handlers.
    from .capsule_collision import capsule_halfspace_manifold
    from .cylinder_collision import cylinder_halfspace_manifold
    from .geometry import CapsuleShape, CylinderShape

    if isinstance(convex_shape, CapsuleShape):
        return capsule_halfspace_manifold(
            convex_shape, convex_pose, hs_normal_world, hs_point_world, margin=margin
        )
    if isinstance(convex_shape, CylinderShape):
        return cylinder_halfspace_manifold(
            convex_shape, convex_pose, hs_normal_world, hs_point_world, margin=margin
        )

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
