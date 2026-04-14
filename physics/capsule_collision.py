"""
Analytical multi-point contact manifold for capsule shapes.

Capsule is a sphere-swept line segment (Minkowski sum of segment + sphere),
so contact with a plane / box / another capsule can be treated by
collapsing the capsule to its core segment, finding the relevant
closest-point pair (or 2-point overlap), and offsetting by the radius.

The handlers here bypass GJK/EPA entirely because segment geometry has
closed-form closest-point formulas that are both faster and more robust
(EPA can degenerate on near-parallel configurations).

Scope covered:
  - `capsule_halfspace_manifold`   — capsule vs infinite plane (1 or 2 pts)
  - `capsule_capsule_manifold`     — capsule vs capsule (1 or 2 pts)
  - `capsule_box_manifold`         — capsule vs OBB (1 or 2 pts)
  - `capsule_cylinder_manifold`    — capsule vs cylinder (1 or 2 pts)
  - `capsule_convexhull_manifold`  — capsule vs convex hull (1 or 2 pts)

Capsule-sphere still falls through to the generic GJK/EPA single-point
path, which is exact (smooth shape, single contact point).

References:
  ODE `dCollideCCTL` (capsule-capsule), `dCollideCCB` (capsule-box) —
    segment-based analytical formulation, epsilon-guarded parallel path.
  MuJoCo `mjc_CapsulePlane`, `mjc_CapsuleCapsule`, `mjc_CapsuleBox`.
  Ericson (2004) §5.1.9 Closest Point Between Two Segments.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .contact_tolerances import CONTACT_NEAR_PARALLEL_COS
from .geometry import BoxShape, CapsuleShape, ConvexHullShape, CylinderShape, SphereShape
from .gjk_epa import ContactManifold, _segment_closest_points
from .spatial import SpatialTransform, Vec3

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _capsule_axis_endpoints(caps: CapsuleShape, pose: SpatialTransform) -> tuple[Vec3, Vec3, Vec3]:
    """Return (endpoint_0, endpoint_1, axis_world) for the capsule core segment.

    The core segment is the line between the two hemisphere centers.
    Axis convention: local +Z, so world axis = R @ [0,0,1].
    """
    hl = caps.length / 2.0
    axis = pose.R @ np.array([0.0, 0.0, 1.0])
    e0 = pose.r - hl * axis
    e1 = pose.r + hl * axis
    return e0, e1, axis


def _sphere_vs_box(
    center_w: Vec3,
    radius: float,
    box_pos: Vec3,
    box_R: np.ndarray,
    half_extents: Vec3,
) -> Optional[tuple[float, Vec3, Vec3]]:
    """Sphere vs OBB closest-point contact.

    Returns ``(depth, normal_world, contact_point_world)`` where normal
    points from the box toward the sphere, or ``None`` if separated.
    When the sphere center is inside the box, we push out along the
    shallowest axis (common OBB convention).
    """
    p_local = box_R.T @ (center_w - box_pos)
    cp_local = np.clip(p_local, -half_extents, half_extents)
    diff = p_local - cp_local
    dist = float(np.linalg.norm(diff))

    if dist < _EPS:
        # Center inside box — push along shallowest axis
        face_gap = half_extents - np.abs(p_local)  # positive = distance to each face
        axis_idx = int(np.argmin(face_gap))
        sign = 1.0 if p_local[axis_idx] >= 0 else -1.0
        normal_local = np.zeros(3)
        normal_local[axis_idx] = sign
        depth = radius + face_gap[axis_idx]
        # Surface point on the exit face
        cp_local = p_local.copy()
        cp_local[axis_idx] = sign * half_extents[axis_idx]
    else:
        normal_local = diff / dist
        depth = radius - dist
        if depth <= 0.0:
            return None

    normal_w = box_R @ normal_local
    cp_w = box_R @ cp_local + box_pos
    return depth, normal_w, cp_w


# ---------------------------------------------------------------------------
# capsule vs half-space
# ---------------------------------------------------------------------------


def capsule_halfspace_manifold(
    caps: CapsuleShape,
    pose: SpatialTransform,
    hs_normal_world: Vec3,
    hs_point_world: Vec3,
    margin: float = 0.0,
) -> Optional[ContactManifold]:
    """Capsule vs infinite half-space; up to 2 contact points.

    Half-space: ``dot(n, p - p0) <= 0`` is solid; ``n`` points outward.

    Algorithm:
      For each core-segment endpoint (two hemisphere centers), compute the
      signed distance to the plane. The endpoint's hemisphere penetrates
      iff ``radius - signed_dist > -margin``. Every penetrating endpoint
      contributes one contact at the projected foot on the plane.

      With the axis nearly perpendicular to the plane, only one endpoint
      penetrates → single point. With axis parallel, both endpoints
      penetrate equally → two points. Intermediate orientations self-select.
    """
    r = caps.radius
    e0, e1, _ = _capsule_axis_endpoints(caps, pose)
    n = np.asarray(hs_normal_world, dtype=np.float64)
    p0 = np.asarray(hs_point_world, dtype=np.float64)

    sd0 = float(np.dot(n, e0 - p0))
    sd1 = float(np.dot(n, e1 - p0))
    depth0 = r - sd0
    depth1 = r - sd1

    hit0 = depth0 > -margin
    hit1 = depth1 > -margin
    if not (hit0 or hit1):
        return None

    points: list[Vec3] = []
    point_depths: list[float] = []
    max_depth = 0.0
    if hit0:
        cp0 = e0 - sd0 * n  # foot of perpendicular from e0 onto plane
        points.append(cp0)
        point_depths.append(depth0)
        if depth0 > max_depth:
            max_depth = depth0
    if hit1:
        cp1 = e1 - sd1 * n
        points.append(cp1)
        point_depths.append(depth1)
        if depth1 > max_depth:
            max_depth = depth1

    return ContactManifold(
        body_i=-1,
        body_j=-1,
        normal=n.copy(),
        depth=max_depth,
        points=points,
        point_depths=point_depths if len(points) > 1 else None,
    )


# ---------------------------------------------------------------------------
# capsule vs capsule
# ---------------------------------------------------------------------------


def capsule_capsule_manifold(
    caps_a: CapsuleShape,
    pose_a: SpatialTransform,
    caps_b: CapsuleShape,
    pose_b: SpatialTransform,
    near_parallel_cos: float = CONTACT_NEAR_PARALLEL_COS,
) -> Optional[ContactManifold]:
    """Capsule vs capsule; up to 2 contact points when axes are near-parallel.

    Near-parallel test: ``|cross(axis_a, axis_b)| < near_parallel_cos``
    (axes are unit, so this is ``sin(angle) < threshold``).
    """
    r_a, r_b = caps_a.radius, caps_b.radius
    a0, a1, axis_a = _capsule_axis_endpoints(caps_a, pose_a)
    b0, b1, axis_b = _capsule_axis_endpoints(caps_b, pose_b)

    p_a, p_b = _segment_closest_points(a0, a1, b0, b1)
    diff = p_a - p_b
    dist = float(np.linalg.norm(diff))
    depth = r_a + r_b - dist
    if depth <= 0.0:
        return None

    if dist < _EPS:
        # Collinear centers — pick any perpendicular direction
        ref = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(axis_a, ref))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        normal = np.cross(axis_a, ref)
        nrm = float(np.linalg.norm(normal))
        normal = normal / nrm if nrm > _EPS else ref
    else:
        normal = diff / dist  # from B toward A (b → a)

    sin_angle = float(np.linalg.norm(np.cross(axis_a, axis_b)))
    near_parallel = sin_angle < near_parallel_cos

    if not near_parallel:
        cp = p_b + normal * r_b
        return ContactManifold(body_i=-1, body_j=-1, normal=normal, depth=depth, points=[cp])

    # Near-parallel path: project segment B onto A's parameterization,
    # find the overlap interval [t_lo, t_hi] ⊂ [0, 1], place contacts at
    # both ends. For truly parallel segments the perpendicular offset is
    # constant along the overlap, so depth is (near-)constant too.
    seg_a_vec = a1 - a0
    seg_a_len_sq = float(np.dot(seg_a_vec, seg_a_vec))
    if seg_a_len_sq < _EPS:
        cp = p_b + normal * r_b
        return ContactManifold(body_i=-1, body_j=-1, normal=normal, depth=depth, points=[cp])

    def _proj(p):
        return float(np.dot(p - a0, seg_a_vec)) / seg_a_len_sq

    tb0 = _proj(b0)
    tb1 = _proj(b1)
    t_lo = max(0.0, min(tb0, tb1))
    t_hi = min(1.0, max(tb0, tb1))
    if t_lo >= t_hi - 1e-9:
        # Overlap collapses to a point — single contact is sufficient
        cp = p_b + normal * r_b
        return ContactManifold(body_i=-1, body_j=-1, normal=normal, depth=depth, points=[cp])

    pa_lo = a0 + t_lo * seg_a_vec
    pa_hi = a0 + t_hi * seg_a_vec
    # For parallel axes, the perpendicular separation is the same along
    # the overlap, so we re-use `dist` (depth identical for both points).
    cp_lo = pa_lo + (r_b - dist) * normal
    cp_hi = pa_hi + (r_b - dist) * normal

    return ContactManifold(
        body_i=-1,
        body_j=-1,
        normal=normal,
        depth=depth,
        points=[cp_lo, cp_hi],
        point_depths=[depth, depth],
    )


# ---------------------------------------------------------------------------
# capsule vs box
# ---------------------------------------------------------------------------


def capsule_box_manifold(
    caps: CapsuleShape,
    pose_c: SpatialTransform,
    box: BoxShape,
    pose_b: SpatialTransform,
    near_parallel_cos: float = CONTACT_NEAR_PARALLEL_COS,
) -> Optional[ContactManifold]:
    """Capsule vs OBB; up to 2 contact points.

    Strategy:
      1. Query sphere-vs-OBB at both capsule endpoints. If both contact with
         similar normals AND the capsule axis is near-parallel to the
         contact face, return those two endpoint contacts.
      2. Otherwise, fall back to a single-point contact at the segment's
         closest point to the box. We approximate the closest segment
         point via endpoint sampling plus the segment-midpoint, returning
         the deepest. Good enough for penalty contact on convex interiors;
         exact only if the true closest is at an endpoint or midpoint.

    The midpoint fallback matters for perpendicular capsule touching a
    box face at its centre — endpoints alone would miss it.
    """
    r = caps.radius
    e0, e1, axis_w = _capsule_axis_endpoints(caps, pose_c)
    h = np.asarray(box.size, dtype=np.float64) / 2.0

    c0 = _sphere_vs_box(e0, r, pose_b.r, pose_b.R, h)
    c1 = _sphere_vs_box(e1, r, pose_b.r, pose_b.R, h)

    # 2-point path: both endpoints hit with matching normals, axis parallel
    # to the contact face (axis ⊥ normal).
    if c0 is not None and c1 is not None:
        d0, n0, cp0 = c0
        d1, n1, cp1 = c1
        if float(np.dot(n0, n1)) > 0.99:
            n_avg = n0 + n1
            nrm = float(np.linalg.norm(n_avg))
            n_mean = n_avg / nrm if nrm > _EPS else n0
            if abs(float(np.dot(axis_w, n_mean))) < near_parallel_cos:
                return ContactManifold(
                    body_i=-1,
                    body_j=-1,
                    normal=n_mean,
                    depth=max(d0, d1),
                    points=[cp0, cp1],
                    point_depths=[d0, d1],
                )

    # Single-point: pick the deepest among {endpoint_0, endpoint_1, midpoint}.
    midpoint = 0.5 * (e0 + e1)
    cmid = _sphere_vs_box(midpoint, r, pose_b.r, pose_b.R, h)
    candidates = [c for c in (c0, c1, cmid) if c is not None]
    if not candidates:
        return None
    candidates.sort(key=lambda c: -c[0])
    depth, normal, cp = candidates[0]
    return ContactManifold(body_i=-1, body_j=-1, normal=normal, depth=depth, points=[cp])


# ---------------------------------------------------------------------------
# capsule vs cylinder
# ---------------------------------------------------------------------------


def _cylinder_axis_endpoints(cyl: CylinderShape, pose: SpatialTransform) -> tuple[Vec3, Vec3, Vec3]:
    """Return (endpoint_0, endpoint_1, axis_world) for the cylinder core segment."""
    hl = cyl.length / 2.0
    axis = pose.R @ np.array([0.0, 0.0, 1.0])
    c0 = pose.r - hl * axis
    c1 = pose.r + hl * axis
    return c0, c1, axis


def capsule_cylinder_manifold(
    caps: CapsuleShape,
    pose_c: SpatialTransform,
    cyl: CylinderShape,
    pose_y: SpatialTransform,
    near_parallel_cos: float = CONTACT_NEAR_PARALLEL_COS,
) -> Optional[ContactManifold]:
    """Capsule vs cylinder; up to 2 contact points when axes are near-parallel.

    The capsule core segment and the cylinder core segment are both line
    segments.  We find their closest points via
    ``_segment_closest_points()``, then check penetration at combined
    radius ``r_cap + r_cyl``.

    When the axes are near-parallel (``sin(angle) < threshold``), we
    project one segment onto the other to find the axial overlap interval
    and return 2 contact points at the overlap endpoints (line contact).
    Otherwise a single contact at the closest-point pair.

    Note: this treats the cylinder as a sphere-swept segment of radius
    ``r_cyl`` — the same simplification Bullet/ODE use for capsule-like
    narrowphase on cylinders.  The exact end-cap flat face is handled by
    the prism face-topology path (``build_contact_manifold`` via GJK/EPA)
    for face-on contacts; this handler is strictly for side-contact
    scenarios where axis proximity dominates.

    References:
      ODE ``dCollideCCyC`` (capsule-cylinder, segment-segment core).
      Ericson (2004) §5.1.9 Closest Point Between Two Segments.
    """
    r_cap = caps.radius
    r_cyl = cyl.radius
    r_sum = r_cap + r_cyl

    e0, e1, axis_cap = _capsule_axis_endpoints(caps, pose_c)
    c0, c1, axis_cyl = _cylinder_axis_endpoints(cyl, pose_y)

    p_cap, p_cyl = _segment_closest_points(e0, e1, c0, c1)
    diff = p_cap - p_cyl
    dist = float(np.linalg.norm(diff))
    depth = r_sum - dist
    if depth <= 0.0:
        return None

    # Normal: from cylinder toward capsule
    if dist < _EPS:
        # Collinear — pick perpendicular direction
        ref = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(axis_cap, ref))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        normal = np.cross(axis_cap, ref)
        nrm = float(np.linalg.norm(normal))
        normal = normal / nrm if nrm > _EPS else ref
    else:
        normal = diff / dist

    sin_angle = float(np.linalg.norm(np.cross(axis_cap, axis_cyl)))
    near_parallel = sin_angle < near_parallel_cos

    if not near_parallel:
        # Single contact on the cylinder surface along the normal
        cp = p_cyl + normal * r_cyl
        return ContactManifold(body_i=-1, body_j=-1, normal=normal, depth=depth, points=[cp])

    # Near-parallel path: find axial overlap between the two segments,
    # place contacts at both ends of the overlap interval.
    # Project cylinder segment onto capsule segment parameterization.
    seg_cap = e1 - e0
    seg_cap_len_sq = float(np.dot(seg_cap, seg_cap))
    if seg_cap_len_sq < _EPS:
        cp = p_cyl + normal * r_cyl
        return ContactManifold(body_i=-1, body_j=-1, normal=normal, depth=depth, points=[cp])

    def _proj(p: Vec3) -> float:
        return float(np.dot(p - e0, seg_cap)) / seg_cap_len_sq

    tc0 = _proj(c0)
    tc1 = _proj(c1)
    t_lo = max(0.0, min(tc0, tc1))
    t_hi = min(1.0, max(tc0, tc1))
    if t_lo >= t_hi - 1e-9:
        cp = p_cyl + normal * r_cyl
        return ContactManifold(body_i=-1, body_j=-1, normal=normal, depth=depth, points=[cp])

    # Two contact points at the overlap endpoints, on the cylinder surface
    pa_lo = e0 + t_lo * seg_cap
    pa_hi = e0 + t_hi * seg_cap
    cp_lo = pa_lo + (r_cyl - dist) * normal
    cp_hi = pa_hi + (r_cyl - dist) * normal

    return ContactManifold(
        body_i=-1,
        body_j=-1,
        normal=normal,
        depth=depth,
        points=[cp_lo, cp_hi],
        point_depths=[depth, depth],
    )


# ---------------------------------------------------------------------------
# capsule vs convex hull
# ---------------------------------------------------------------------------


def _sphere_vs_convexhull(
    center_w: Vec3,
    radius: float,
    hull: ConvexHullShape,
    pose_h: SpatialTransform,
) -> Optional[tuple[float, Vec3, Vec3]]:
    """Sphere vs ConvexHull closest-point contact via GJK/EPA.

    Returns ``(depth, normal_world, contact_point_world)`` where normal
    points from the hull toward the sphere, or ``None`` if separated.
    """
    from .gjk_epa import build_contact_manifold, epa, gjk

    sph = SphereShape(radius)
    pose_s = SpatialTransform.from_translation(center_w)

    intersecting, simplex = gjk(sph, pose_s, hull, pose_h)
    if not intersecting:
        return None

    normal, depth = epa(sph, pose_s, hull, pose_h, simplex)
    if depth < 1e-10:
        return None

    # Build a single-point manifold to extract the contact point
    m = build_contact_manifold(sph, pose_s, hull, pose_h, normal, depth)
    if m is None:
        return None

    return depth, m.normal.copy(), m.points[0].copy()


def capsule_convexhull_manifold(
    caps: CapsuleShape,
    pose_c: SpatialTransform,
    hull: ConvexHullShape,
    pose_h: SpatialTransform,
    near_parallel_cos: float = CONTACT_NEAR_PARALLEL_COS,
) -> Optional[ContactManifold]:
    """Capsule vs ConvexHull; up to 2 contact points.

    Strategy (mirrors ``capsule_box_manifold``):
      1. Query sphere-vs-hull at both capsule endpoint hemispheres via
         GJK/EPA.  If both endpoints contact with similar normals AND the
         capsule axis is near-perpendicular to the contact normal (axis
         near-parallel to the face), return 2 contact points.
      2. Otherwise, also query at the capsule midpoint, and return the
         deepest single contact.

    This naturally handles:
      - Face contact (capsule lying on a flat hull face) → 2 pts
      - Edge/vertex contact (capsule end poking a hull edge) → 1 pt
      - Perpendicular contact (capsule tip into a face) → 1 pt

    References:
      Same endpoint-sphere strategy as ``capsule_box_manifold``.
    """
    r = caps.radius
    e0, e1, axis_w = _capsule_axis_endpoints(caps, pose_c)

    c0 = _sphere_vs_convexhull(e0, r, hull, pose_h)
    c1 = _sphere_vs_convexhull(e1, r, hull, pose_h)

    # 2-point path: both endpoints hit with matching normals, axis ⊥ normal
    if c0 is not None and c1 is not None:
        d0, n0, cp0 = c0
        d1, n1, cp1 = c1
        if float(np.dot(n0, n1)) > 0.99:
            n_avg = n0 + n1
            nrm = float(np.linalg.norm(n_avg))
            n_mean = n_avg / nrm if nrm > _EPS else n0
            if abs(float(np.dot(axis_w, n_mean))) < near_parallel_cos:
                return ContactManifold(
                    body_i=-1,
                    body_j=-1,
                    normal=n_mean,
                    depth=max(d0, d1),
                    points=[cp0, cp1],
                    point_depths=[d0, d1],
                )

    # Single-point: pick the deepest among {endpoint_0, endpoint_1, midpoint}.
    midpoint = 0.5 * (e0 + e1)
    cmid = _sphere_vs_convexhull(midpoint, r, hull, pose_h)
    candidates = [c for c in (c0, c1, cmid) if c is not None]
    if not candidates:
        return None
    candidates.sort(key=lambda c: -c[0])
    depth, normal, cp = candidates[0]
    return ContactManifold(body_i=-1, body_j=-1, normal=normal, depth=depth, points=[cp])
