"""
Analytical cylinder vs half-space contact manifold.

While `CylinderShape` carries an N-gon prism representation for generic
GJK/EPA + Sutherland-Hodgman body-body contact, its contact with an
*infinite* plane (ground / half-space) is worth treating analytically:

  * Axis near-parallel to plane (cylinder lying on its side)
      → 2 contacts at the projected axis endpoints
  * Axis near-perpendicular to plane (end-cap down)
      → up to 4 contacts sampled at the rim (quadrant sample)
  * Tilted axis
      → single deepest point on the lower rim

The analytical path gives exact contact points on the true circle (not the
prism approximation) while still capping at 4 contacts — the industry
convention (PhysX / Bullet) to keep the Delassus matrix well-conditioned.

References:
  MuJoCo `mjc_CylinderPlane` (analytical, 4-point rim sampling)
  Geometric Tools, "Intersection of Cylinder and Plane" (2001).
  ODE notes cylinder-cylinder / cylinder-box as "too complicated" — we
    reuse the prism S-H path for those, not analytical.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .contact_tolerances import CONTACT_NEAR_PARALLEL_COS
from .geometry import CylinderShape
from .gjk_epa import ContactManifold
from .spatial import SpatialTransform, Vec3

_EPS = 1e-12


def _axis_world(pose: SpatialTransform) -> Vec3:
    return pose.R @ np.array([0.0, 0.0, 1.0])


def cylinder_halfspace_manifold(
    cyl: CylinderShape,
    pose: SpatialTransform,
    hs_normal_world: Vec3,
    hs_point_world: Vec3,
    margin: float = 0.0,
    near_parallel_cos: float = CONTACT_NEAR_PARALLEL_COS,
) -> Optional[ContactManifold]:
    """Cylinder vs infinite half-space; 1, 2, or up to 4 contact points.

    The half-space is solid where ``dot(n, p - p0) <= 0`` with outward
    normal ``n``.
    """
    r = cyl.radius
    hl = cyl.length / 2.0
    axis = _axis_world(pose)
    n = np.asarray(hs_normal_world, dtype=np.float64)
    p0 = np.asarray(hs_point_world, dtype=np.float64)

    # Build the in-plane tangent frame for the cylinder cross-section.
    # e_radial = component of -n perpendicular to the cylinder axis.
    axis_dot_n = float(np.dot(axis, n))
    abs_adn = abs(axis_dot_n)

    # End-cap centres in world
    top_centre = pose.r + hl * axis
    bot_centre = pose.r - hl * axis

    # Case A: axis near-parallel to plane (cylinder on its side) → 2 pts
    if abs_adn < near_parallel_cos:
        # For each end, the lowest rim point is offset from the axis centre
        # by -r * n_tangential where n_tangential = n - (n·axis)*axis.
        n_tan = n - axis_dot_n * axis
        nt_len = float(np.linalg.norm(n_tan))
        if nt_len < _EPS:
            return None  # degenerate: axis ⊥ plane AND ∥ plane — impossible
        n_tan = n_tan / nt_len
        # Lowest rim point at each end = centre - r * n_tan
        tip_bot_end = bot_centre - r * n_tan
        tip_top_end = top_centre - r * n_tan
        sd0 = float(np.dot(n, tip_bot_end - p0))  # signed distance to plane
        sd1 = float(np.dot(n, tip_top_end - p0))
        depth0 = -sd0  # positive = penetrating
        depth1 = -sd1
        hit0 = depth0 > -margin
        hit1 = depth1 > -margin
        if not (hit0 or hit1):
            return None
        pts: list[Vec3] = []
        pds: list[float] = []
        max_d = 0.0
        if hit0:
            cp0 = tip_bot_end - sd0 * n  # foot on plane
            pts.append(cp0)
            pds.append(depth0)
            max_d = max(max_d, depth0)
        if hit1:
            cp1 = tip_top_end - sd1 * n
            pts.append(cp1)
            pds.append(depth1)
            max_d = max(max_d, depth1)
        return ContactManifold(
            body_i=-1,
            body_j=-1,
            normal=n.copy(),
            depth=max_d,
            points=pts,
            point_depths=pds if len(pts) > 1 else None,
        )

    # Case B: axis near-perpendicular to plane (end-cap facing plane) → up
    # to 4 rim samples on the lower end face.
    if abs_adn > 1.0 - near_parallel_cos:
        # Lower end: the cap centre whose outward normal has the LARGEST
        # component along -n (i.e. the cap facing into the half-space).
        if axis_dot_n > 0.0:
            # +Z cap normal is +axis; -axis cap faces toward -n side → lower
            lower_centre = bot_centre
        else:
            lower_centre = top_centre
        # In-plane basis orthogonal to axis
        ref = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(axis, ref))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(axis, ref)
        e1 = e1 / float(np.linalg.norm(e1))
        e2 = np.cross(axis, e1)
        e2 = e2 / float(np.linalg.norm(e2))
        # 4 quadrant samples on the rim
        sample_dirs = [e1, e2, -e1, -e2]
        pts = []
        pds = []
        max_d = 0.0
        for d in sample_dirs:
            p_rim = lower_centre + r * d
            sd = float(np.dot(n, p_rim - p0))
            depth = -sd
            if depth > -margin:
                cp = p_rim - sd * n  # project to plane
                pts.append(cp)
                pds.append(depth)
                max_d = max(max_d, depth)
        if not pts:
            return None
        return ContactManifold(
            body_i=-1,
            body_j=-1,
            normal=n.copy(),
            depth=max_d,
            points=pts,
            point_depths=pds if len(pts) > 1 else None,
        )

    # Case C: tilted — single deepest point. The deepest point of a tilted
    # cylinder vs a plane is always on the lower rim (circle edge of the
    # end cap that faces into the half-space).
    if axis_dot_n > 0.0:
        lower_centre = bot_centre
    else:
        lower_centre = top_centre
    # Direction along the rim that minimises dot(n, rim_point): project -n
    # onto the plane perpendicular to axis.
    n_tan = n - axis_dot_n * axis
    nt_len = float(np.linalg.norm(n_tan))
    if nt_len < _EPS:
        # Shouldn't happen in Case C but guard anyway — use arbitrary.
        ref = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(axis, ref))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        d = np.cross(axis, ref)
        d = d / float(np.linalg.norm(d))
    else:
        d = -n_tan / nt_len  # points toward deepest rim point
    p_rim = lower_centre + r * d
    sd = float(np.dot(n, p_rim - p0))
    depth = -sd
    if depth <= -margin:
        return None
    cp = p_rim - sd * n
    return ContactManifold(
        body_i=-1,
        body_j=-1,
        normal=n.copy(),
        depth=depth,
        points=[cp],
    )
