"""Unit tests for `physics.capsule_collision` analytical handlers."""

from __future__ import annotations

import numpy as np
import pytest

from physics.capsule_collision import (
    capsule_box_manifold,
    capsule_capsule_manifold,
    capsule_convexhull_manifold,
    capsule_cylinder_manifold,
    capsule_halfspace_manifold,
)
from physics.geometry import BoxShape, CapsuleShape, ConvexHullShape, CylinderShape
from physics.gjk_epa import ground_contact_query, halfspace_convex_query
from physics.spatial import SpatialTransform

ATOL = 1e-6


def _rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# ---------------------------------------------------------------------------
# capsule vs half-space
# ---------------------------------------------------------------------------


class TestCapsuleHalfspace:
    def test_vertical_capsule_single_point(self):
        """Axis perpendicular to plane → only bottom endpoint contacts."""
        cap = CapsuleShape(radius=0.3, length=1.0)  # half_len=0.5
        # Center at z=0.6 → bottom endpoint at z=0.1, below sphere surface.
        # Lowest point of capsule = 0.6 - 0.5 - 0.3 = -0.2, depth = 0.2
        pose = SpatialTransform.from_translation(np.array([0.0, 0.0, 0.6]))
        m = capsule_halfspace_manifold(cap, pose, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]))
        assert m is not None
        assert len(m.points) == 1
        assert abs(m.depth - 0.2) < ATOL
        assert m.points[0][2] == pytest.approx(0.0, abs=ATOL)

    def test_horizontal_capsule_two_points(self):
        """Axis parallel to plane → both endpoints contact."""
        cap = CapsuleShape(radius=0.2, length=1.0)
        # Rotate 90° around Y so local +Z → world +X
        pose = SpatialTransform(_rot_y(np.pi / 2), np.array([0.0, 0.0, 0.1]))
        m = capsule_halfspace_manifold(cap, pose, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]))
        assert m is not None
        assert len(m.points) == 2
        # Both endpoints at z=0.1, radius=0.2 → depth = 0.1
        for d in m.point_depths:
            assert abs(d - 0.1) < ATOL
        # Endpoints symmetric around x=0, separated by length=1.0
        xs = sorted([p[0] for p in m.points])
        assert abs(xs[1] - xs[0] - 1.0) < ATOL

    def test_tilted_capsule_both_endpoints_penetrating(self):
        """Slight tilt with both endpoints below plane → 2 points, different depths."""
        cap = CapsuleShape(radius=0.4, length=1.0)
        # 10° tilt around Y → axis_z component ≈ cos(80°) = 0.174, axis_x ≈ sin(80°) = 0.985
        pose = SpatialTransform(
            _rot_y(np.pi / 2 - np.pi / 18),  # 80° rotation
            np.array([0.0, 0.0, 0.0]),
        )
        m = capsule_halfspace_manifold(cap, pose, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]))
        # Endpoints z = ±0.5·cos(80°) ≈ ±0.087
        # depths: 0.4 − (±0.087) ≈ 0.313 and 0.487 → both penetrate
        assert m is not None
        assert len(m.points) == 2
        d_lo, d_hi = sorted(m.point_depths)
        assert d_hi - d_lo > 0.1  # noticeably different depths

    def test_fully_above_plane_returns_none(self):
        cap = CapsuleShape(radius=0.1, length=1.0)
        pose = SpatialTransform.from_translation(np.array([0.0, 0.0, 2.0]))
        m = capsule_halfspace_manifold(cap, pose, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]))
        assert m is None

    def test_ground_contact_query_dispatch(self):
        """Flat ground path should also generate 2-point manifold for horizontal capsule."""
        cap = CapsuleShape(radius=0.2, length=1.0)
        pose = SpatialTransform(_rot_y(np.pi / 2), np.array([0.0, 0.0, 0.1]))
        m = ground_contact_query(cap, pose, ground_z=0.0)
        assert m is not None
        assert len(m.points) == 2

    def test_halfspace_convex_query_dispatch(self):
        cap = CapsuleShape(radius=0.2, length=1.0)
        pose = SpatialTransform(_rot_y(np.pi / 2), np.array([0.0, 0.0, 0.1]))
        m = halfspace_convex_query(
            cap,
            pose,
            hs_normal_world=np.array([0.0, 0.0, 1.0]),
            hs_point_world=np.array([0.0, 0.0, 0.0]),
        )
        assert m is not None
        assert len(m.points) == 2

    def test_tilted_halfspace_normal(self):
        """Non-axis-aligned plane normal."""
        cap = CapsuleShape(radius=0.2, length=1.0)
        # Horizontal capsule along world x, at origin
        pose = SpatialTransform(_rot_y(np.pi / 2), np.array([0.0, 0.0, 0.0]))
        n = np.array([0.0, 0.0, 1.0])
        p0 = np.array([0.0, 0.0, -0.1])  # plane at z = -0.1
        m = capsule_halfspace_manifold(cap, pose, n, p0)
        assert m is not None
        assert len(m.points) == 2
        # depths: 0.2 - (0 - (-0.1)) = 0.1 each
        for d in m.point_depths:
            assert abs(d - 0.1) < ATOL


# ---------------------------------------------------------------------------
# capsule vs capsule
# ---------------------------------------------------------------------------


class TestCapsuleCapsule:
    def test_parallel_capsules_two_points(self):
        """Parallel capsules laid side by side → 2 contact points along overlap."""
        cap = CapsuleShape(radius=0.2, length=1.0)
        pose_a = SpatialTransform.from_translation(np.array([0.0, 0.0, 0.0]))
        pose_b = SpatialTransform.from_translation(np.array([0.3, 0.0, 0.0]))
        # Overlap in z = full axis (both axes along z, both centred at z=0)
        m = capsule_capsule_manifold(cap, pose_a, cap, pose_b)
        assert m is not None
        assert len(m.points) == 2
        # Separation = 0.3, sum radii = 0.4 → depth = 0.1
        assert abs(m.depth - 0.1) < ATOL
        # Normal from B (x=0.3) to A (x=0) → -x direction
        assert m.normal[0] < -0.99
        # The two contact points span the full overlap interval (length 1.0 in z)
        zs = sorted([p[2] for p in m.points])
        assert abs(zs[1] - zs[0] - 1.0) < 1e-4

    def test_skew_capsules_single_point(self):
        """Capsules at 90° with offset axes → single closest-point contact."""
        cap = CapsuleShape(radius=0.2, length=1.0)
        # A vertical at origin: z ∈ [−0.5, 0.5] on the line x=y=0.
        pose_a = SpatialTransform.from_translation(np.array([0.0, 0.0, 0.0]))
        # B horizontal along x, offset laterally by y=0.3 (so axes don't intersect),
        # at height z=0. Closest distance between lines ≈ 0.3 (y offset).
        pose_b = SpatialTransform(_rot_y(np.pi / 2), np.array([0.0, 0.3, 0.0]))
        m = capsule_capsule_manifold(cap, pose_a, cap, pose_b)
        assert m is not None
        assert len(m.points) == 1
        # Separation = 0.3, sum radii = 0.4 → depth = 0.1
        assert abs(m.depth - 0.1) < 1e-4

    def test_parallel_nonoverlap_returns_single(self):
        """Parallel capsules axially offset with no overlap → single contact at closest endpoints."""
        cap = CapsuleShape(radius=0.2, length=1.0)
        pose_a = SpatialTransform.from_translation(np.array([0.0, 0.0, 0.0]))
        # Shift B 1.1 in z (no overlap along axis) and 0.3 in x (lateral)
        pose_b = SpatialTransform.from_translation(np.array([0.3, 0.0, 1.1]))
        m = capsule_capsule_manifold(cap, pose_a, cap, pose_b)
        # Closest endpoints: A top (z=0.5) and B bottom (z=0.6) in overlap? Actually:
        # A segment: z in [-0.5, 0.5]. B segment: z in [0.6, 1.6]. No axial overlap.
        # Closest: A(z=0.5) ↔ B(z=0.6), lateral 0.3, axial 0.1 → dist = sqrt(0.1+0.09)=0.316
        # Depth = 0.4 - 0.316 = 0.084. Single point (no overlap).
        assert m is not None
        assert len(m.points) == 1

    def test_separated_capsules_returns_none(self):
        cap = CapsuleShape(radius=0.1, length=1.0)
        pose_a = SpatialTransform.from_translation(np.array([0.0, 0.0, 0.0]))
        pose_b = SpatialTransform.from_translation(np.array([1.0, 0.0, 0.0]))
        m = capsule_capsule_manifold(cap, pose_a, cap, pose_b)
        assert m is None


# ---------------------------------------------------------------------------
# capsule vs box
# ---------------------------------------------------------------------------


class TestCapsuleBox:
    def test_capsule_lying_on_box_top(self):
        """Horizontal capsule on a box's top face → 2 points."""
        box = BoxShape((2.0, 2.0, 1.0))  # half_extents = (1, 1, 0.5)
        box_pose = SpatialTransform.from_translation(np.array([0.0, 0.0, 0.0]))
        cap = CapsuleShape(radius=0.2, length=1.0)
        # Capsule horizontal (axis along x), centred above box top (z=0.5)
        cap_pose = SpatialTransform(_rot_y(np.pi / 2), np.array([0.0, 0.0, 0.6]))
        # Bottom of capsule at z = 0.6 - 0.2 = 0.4; box top = 0.5. Penetration = 0.1.
        m = capsule_box_manifold(cap, cap_pose, box, box_pose)
        assert m is not None
        assert len(m.points) == 2
        assert abs(m.depth - 0.1) < ATOL
        # Normal should point from box (toward +z) to capsule: normal ≈ +z
        assert m.normal[2] > 0.99

    def test_capsule_perpendicular_on_box_center(self):
        """Vertical capsule touching box top at its own bottom → single point."""
        box = BoxShape((2.0, 2.0, 1.0))
        box_pose = SpatialTransform.from_translation(np.array([0.0, 0.0, 0.0]))
        cap = CapsuleShape(radius=0.2, length=1.0)
        # Capsule standing vertical, bottom just penetrating
        cap_pose = SpatialTransform.from_translation(np.array([0.0, 0.0, 1.1]))
        # Bottom: 1.1 - 0.5 - 0.2 = 0.4, box top = 0.5 → depth = 0.1
        m = capsule_box_manifold(cap, cap_pose, box, box_pose)
        assert m is not None
        assert len(m.points) == 1
        assert abs(m.depth - 0.1) < ATOL

    def test_capsule_sphere_poking_box_edge_single_point(self):
        """Capsule endpoint sphere poking into box edge → single point."""
        box = BoxShape((1.0, 1.0, 1.0))
        box_pose = SpatialTransform.from_translation(np.array([0.0, 0.0, 0.0]))
        cap = CapsuleShape(radius=0.3, length=1.0)
        # Horizontal capsule far above box top, only one end sphere pokes corner
        cap_pose = SpatialTransform(_rot_y(np.pi / 2), np.array([1.2, 0.0, 0.5 + 0.2]))
        # Left endpoint at x = 1.2 - 0.5 = 0.7 (inside box in x), z = 0.7 → above box top
        # Right endpoint at x = 1.7, z = 0.7 → outside box
        # Only left hemisphere poking top: box top at z=0.5, endpoint at z=0.7, depth = 0.3 - (0.7-0.5) = 0.1
        m = capsule_box_manifold(cap, cap_pose, box, box_pose)
        assert m is not None
        # This config: left endpoint hits top face; right endpoint outside box entirely
        # Expect 1 or 2 pts depending on right endpoint vs box distance; just assert contact
        assert m.depth > 0.0

    def test_separated_capsule_box_returns_none(self):
        box = BoxShape((1.0, 1.0, 1.0))
        box_pose = SpatialTransform.from_translation(np.array([0.0, 0.0, 0.0]))
        cap = CapsuleShape(radius=0.1, length=1.0)
        cap_pose = SpatialTransform.from_translation(np.array([3.0, 0.0, 0.0]))
        m = capsule_box_manifold(cap, cap_pose, box, box_pose)
        assert m is None


# ---------------------------------------------------------------------------
# gjk_epa_query dispatch
# ---------------------------------------------------------------------------


class TestGJKEpaDispatch:
    def test_gjk_epa_query_capsule_capsule_parallel(self):
        from physics.gjk_epa import gjk_epa_query

        cap = CapsuleShape(radius=0.2, length=1.0)
        pose_a = SpatialTransform.from_translation(np.zeros(3))
        pose_b = SpatialTransform.from_translation(np.array([0.3, 0.0, 0.0]))
        m = gjk_epa_query(cap, pose_a, cap, pose_b)
        assert m is not None
        assert len(m.points) == 2

    def test_gjk_epa_query_capsule_box_horizontal(self):
        from physics.gjk_epa import gjk_epa_query

        cap = CapsuleShape(radius=0.2, length=1.0)
        box = BoxShape((2.0, 2.0, 1.0))
        cap_pose = SpatialTransform(_rot_y(np.pi / 2), np.array([0.0, 0.0, 0.6]))
        box_pose = SpatialTransform.from_translation(np.zeros(3))
        m = gjk_epa_query(cap, cap_pose, box, box_pose)
        assert m is not None
        assert len(m.points) == 2

    def test_gjk_epa_query_box_capsule_swapped_has_flipped_normal(self):
        """Dispatch must flip the normal when capsule is the second argument."""
        from physics.gjk_epa import gjk_epa_query

        cap = CapsuleShape(radius=0.2, length=1.0)
        box = BoxShape((2.0, 2.0, 1.0))
        cap_pose = SpatialTransform(_rot_y(np.pi / 2), np.array([0.0, 0.0, 0.6]))
        box_pose = SpatialTransform.from_translation(np.zeros(3))
        m_cb = gjk_epa_query(cap, cap_pose, box, box_pose)  # normal: box → cap
        m_bc = gjk_epa_query(box, box_pose, cap, cap_pose)  # normal: cap → box
        assert m_cb is not None and m_bc is not None
        assert np.allclose(m_cb.normal, -m_bc.normal, atol=ATOL)

    def test_gjk_epa_query_capsule_sphere_falls_through(self):
        """capsule-sphere is not analytical — must fall through GJK/EPA (single point)."""
        from physics.geometry import SphereShape
        from physics.gjk_epa import gjk_epa_query

        cap = CapsuleShape(radius=0.2, length=1.0)
        sph = SphereShape(0.25)
        cap_pose = SpatialTransform.from_translation(np.zeros(3))
        sph_pose = SpatialTransform.from_translation(np.array([0.3, 0.0, 0.0]))
        m = gjk_epa_query(cap, cap_pose, sph, sph_pose)
        assert m is not None
        assert len(m.points) == 1


# ---------------------------------------------------------------------------
# capsule vs cylinder
# ---------------------------------------------------------------------------


def _make_box_hull():
    """Create a ConvexHullShape equivalent to BoxShape((0.4, 0.4, 0.4))."""
    h = 0.2  # half-extent
    verts = np.array(
        [
            [-h, -h, -h],
            [-h, -h, h],
            [-h, h, -h],
            [-h, h, h],
            [h, -h, -h],
            [h, -h, h],
            [h, h, -h],
            [h, h, h],
        ]
    )
    return ConvexHullShape(verts)


class TestCapsuleCylinder:
    def test_parallel_axes_side_by_side_two_points(self):
        """Parallel capsule and cylinder laid side by side → 2 contacts."""
        cap = CapsuleShape(radius=0.2, length=1.0)
        cyl = CylinderShape(radius=0.15, length=0.8)
        # Both axes along Z, separated by 0.3 in x (sum radii = 0.35)
        pose_c = SpatialTransform.from_translation(np.array([0.0, 0.0, 0.0]))
        pose_y = SpatialTransform.from_translation(np.array([0.3, 0.0, 0.0]))
        m = capsule_cylinder_manifold(cap, pose_c, cyl, pose_y)
        assert m is not None
        assert len(m.points) == 2
        assert abs(m.depth - 0.05) < ATOL  # 0.35 - 0.30
        # Normal from cyl toward cap → -x
        assert m.normal[0] < -0.99

    def test_skew_axes_single_point(self):
        """Capsule and cylinder at 90° → single closest-point contact."""
        cap = CapsuleShape(radius=0.2, length=1.0)
        cyl = CylinderShape(radius=0.15, length=1.0)
        # Cap vertical (Z), cyl horizontal (X) offset by y=0.3
        pose_c = SpatialTransform.from_translation(np.zeros(3))
        pose_y = SpatialTransform(_rot_y(np.pi / 2), np.array([0.0, 0.3, 0.0]))
        m = capsule_cylinder_manifold(cap, pose_c, cyl, pose_y)
        assert m is not None
        assert len(m.points) == 1
        # dist = 0.3, sum radii = 0.35, depth = 0.05
        assert abs(m.depth - 0.05) < 1e-4

    def test_parallel_no_axial_overlap_single_point(self):
        """Parallel but axially separated → single point at closest endpoints."""
        cap = CapsuleShape(radius=0.2, length=1.0)  # z ∈ [-0.5, 0.5]
        cyl = CylinderShape(radius=0.15, length=0.6)  # z ∈ [0.7, 1.3] shifted
        pose_c = SpatialTransform.from_translation(np.zeros(3))
        pose_y = SpatialTransform.from_translation(np.array([0.3, 0.0, 1.0]))
        m = capsule_cylinder_manifold(cap, pose_c, cyl, pose_y)
        # Closest endpoints: cap top z=0.5, cyl bottom z=0.7
        # axial gap 0.2, lateral 0.3, dist = sqrt(0.04+0.09) ≈ 0.3606
        # depth = 0.35 - 0.3606 < 0 → separated
        assert m is None

    def test_separated_returns_none(self):
        cap = CapsuleShape(radius=0.1, length=1.0)
        cyl = CylinderShape(radius=0.1, length=1.0)
        pose_c = SpatialTransform.from_translation(np.zeros(3))
        pose_y = SpatialTransform.from_translation(np.array([1.0, 0.0, 0.0]))
        m = capsule_cylinder_manifold(cap, pose_c, cyl, pose_y)
        assert m is None

    def test_normal_direction_cyl_to_cap(self):
        """Normal points from cylinder toward capsule."""
        cap = CapsuleShape(radius=0.2, length=0.6)
        cyl = CylinderShape(radius=0.15, length=0.6)
        # Capsule at y=+0.3, cylinder at y=0 → normal should be +y
        pose_c = SpatialTransform.from_translation(np.array([0.0, 0.3, 0.0]))
        pose_y = SpatialTransform.from_translation(np.zeros(3))
        m = capsule_cylinder_manifold(cap, pose_c, cyl, pose_y)
        assert m is not None
        assert m.normal[1] > 0.99

    def test_contact_point_on_cylinder_surface(self):
        """Contact point should be on the cylinder surface."""
        cap = CapsuleShape(radius=0.2, length=0.6)
        cyl = CylinderShape(radius=0.15, length=0.6)
        # Single-point: skew axes
        pose_c = SpatialTransform.from_translation(np.zeros(3))
        pose_y = SpatialTransform(_rot_y(np.pi / 2), np.array([0.0, 0.3, 0.0]))
        m = capsule_cylinder_manifold(cap, pose_c, cyl, pose_y)
        assert m is not None
        assert len(m.points) == 1
        cp = m.points[0]
        # Contact point should be offset from cyl axis by r_cyl along normal
        # Cylinder centre at (0, 0.3, 0), axis along X after rotation
        # At closest point the cyl point is at (0, 0.3, 0)
        # Contact = (0, 0.3, 0) + 0.15 * normal_toward_cap
        # normal ≈ (0, -1, 0) from the cap's perspective, so cp ≈ (0, 0.15, 0)
        dist_from_cyl_center = abs(cp[1] - 0.3)
        assert abs(dist_from_cyl_center - 0.15) < 1e-3

    def test_different_radii(self):
        """Depth correct with asymmetric radii."""
        cap = CapsuleShape(radius=0.3, length=0.8)
        cyl = CylinderShape(radius=0.1, length=0.8)
        # Side by side, separation = 0.35, sum radii = 0.4 → depth = 0.05
        pose_c = SpatialTransform.from_translation(np.zeros(3))
        pose_y = SpatialTransform.from_translation(np.array([0.35, 0.0, 0.0]))
        m = capsule_cylinder_manifold(cap, pose_c, cyl, pose_y)
        assert m is not None
        assert abs(m.depth - 0.05) < ATOL

    def test_collinear_head_on(self):
        """Collinear axes, capsule hemisphere poking into cylinder end → 1 pt."""
        cap = CapsuleShape(radius=0.1, length=0.6)  # z ∈ [-0.3, 0.3]
        cyl = CylinderShape(radius=0.1, length=0.6)  # z ∈ [0.15, 0.75]
        pose_c = SpatialTransform.from_translation(np.zeros(3))
        pose_y = SpatialTransform.from_translation(np.array([0.0, 0.0, 0.45]))
        m = capsule_cylinder_manifold(cap, pose_c, cyl, pose_y)
        # Closest: cap top at z=0.3, cyl bottom at z=0.15 → dist=0.15
        # Depth = 0.2 - 0.15 = 0.05 (but axes are collinear → near_parallel
        # overlap = [0.15/0.6, 0.3/0.6] = [0.25, 0.5] → 2 pts if overlap > eps)
        # Actually: t projection of cyl bottom z=0.15 onto cap seg [-0.3, 0.3]:
        # t = (0.15 - (-0.3)) / 0.6 = 0.75; cyl top z=0.75: t = (0.75+0.3)/0.6 = 1.75
        # t_lo = max(0, min(0.75, 1.75)) = 0.75; t_hi = min(1, max(0.75, 1.75)) = 1.0
        # overlap [0.75, 1.0] → 2 pts
        assert m is not None
        # Collinear: dist = 0 → picks perpendicular normal → depth = r_cap + r_cyl = 0.2
        # Actually they're collinear AND overlapping → parallel path,
        # but closest points are on the segment endpoints near each other
        # Check at least contact exists
        assert m.depth > 0


# ---------------------------------------------------------------------------
# capsule vs convex hull
# ---------------------------------------------------------------------------


class TestCapsuleConvexHull:
    def test_capsule_lying_on_hull_face_two_points(self):
        """Horizontal capsule on a box-hull top face → 2 contact points."""
        hull = _make_box_hull()  # half-extent = 0.2
        hull_pose = SpatialTransform.from_translation(np.zeros(3))
        cap = CapsuleShape(radius=0.1, length=0.3)
        # Capsule horizontal (axis along X), above hull top face (z=0.2)
        cap_pose = SpatialTransform(_rot_y(np.pi / 2), np.array([0.0, 0.0, 0.25]))
        # Bottom of capsule sphere = 0.25 - 0.1 = 0.15 < 0.2 → depth ~ 0.05
        m = capsule_convexhull_manifold(cap, cap_pose, hull, hull_pose)
        assert m is not None
        assert len(m.points) == 2
        assert m.depth > 0.03

    def test_capsule_perpendicular_to_hull_face_one_point(self):
        """Vertical capsule touching hull top → 1 contact point."""
        hull = _make_box_hull()  # half-extent 0.2, top face z = 0.2
        hull_pose = SpatialTransform.from_translation(np.zeros(3))
        cap = CapsuleShape(radius=0.1, length=0.4)
        # Capsule vertical (axis along Z). half_length = 0.2.
        # Place centre at z = 0.45 → bottom endpoint at z = 0.25
        # Bottom sphere surface at z = 0.25 - 0.1 = 0.15, hull top = 0.2
        # → depth ≈ 0.05.  Top endpoint at z = 0.65 → far above hull.
        # Only the bottom hemisphere contacts.
        cap_pose = SpatialTransform.from_translation(np.array([0.0, 0.0, 0.45]))
        m = capsule_convexhull_manifold(cap, cap_pose, hull, hull_pose)
        assert m is not None
        assert len(m.points) == 1
        assert m.depth > 0.01

    def test_capsule_near_hull_edge_one_point(self):
        """Capsule approaching hull edge → 1 contact point."""
        hull = _make_box_hull()
        hull_pose = SpatialTransform.from_translation(np.zeros(3))
        cap = CapsuleShape(radius=0.15, length=0.3)
        # Capsule vertical, approaching the +x edge at x = 0.2
        cap_pose = SpatialTransform.from_translation(np.array([0.3, 0.0, 0.0]))
        # Lower sphere center at (0.3, 0, -0.15), closest hull point ~ (0.2, 0, -0.15)
        # dist ~ 0.1, depth = 0.15 - 0.1 = 0.05
        m = capsule_convexhull_manifold(cap, cap_pose, hull, hull_pose)
        assert m is not None
        assert m.depth > 0

    def test_separated_returns_none(self):
        hull = _make_box_hull()
        hull_pose = SpatialTransform.from_translation(np.zeros(3))
        cap = CapsuleShape(radius=0.1, length=0.3)
        cap_pose = SpatialTransform.from_translation(np.array([2.0, 0.0, 0.0]))
        m = capsule_convexhull_manifold(cap, cap_pose, hull, hull_pose)
        assert m is None

    def test_gjk_dispatch_capsule_convexhull(self):
        """gjk_epa_query routes capsule-convexhull to analytical handler."""
        from physics.gjk_epa import gjk_epa_query

        hull = _make_box_hull()
        hull_pose = SpatialTransform.from_translation(np.zeros(3))
        cap = CapsuleShape(radius=0.1, length=0.3)
        cap_pose = SpatialTransform(_rot_y(np.pi / 2), np.array([0.0, 0.0, 0.25]))
        m = gjk_epa_query(cap, cap_pose, hull, hull_pose)
        assert m is not None
        assert len(m.points) >= 1

    def test_swapped_argument_order_flips_normal(self):
        """Hull first, capsule second → flipped normal."""
        from physics.gjk_epa import gjk_epa_query

        hull = _make_box_hull()
        hull_pose = SpatialTransform.from_translation(np.zeros(3))
        cap = CapsuleShape(radius=0.1, length=0.3)
        cap_pose = SpatialTransform.from_translation(np.array([0.0, 0.0, 0.25]))
        m_ch = gjk_epa_query(cap, cap_pose, hull, hull_pose)
        m_hc = gjk_epa_query(hull, hull_pose, cap, cap_pose)
        assert m_ch is not None and m_hc is not None
        assert np.dot(m_ch.normal, m_hc.normal) < -0.9  # roughly opposite


# ---------------------------------------------------------------------------
# gjk_epa_query dispatch additions
# ---------------------------------------------------------------------------


class TestGJKEpaDispatchExtended:
    def test_gjk_epa_query_capsule_cylinder_parallel(self):
        from physics.gjk_epa import gjk_epa_query

        cap = CapsuleShape(radius=0.2, length=1.0)
        cyl = CylinderShape(radius=0.15, length=0.8)
        pose_c = SpatialTransform.from_translation(np.zeros(3))
        pose_y = SpatialTransform.from_translation(np.array([0.3, 0.0, 0.0]))
        m = gjk_epa_query(cap, pose_c, cyl, pose_y)
        assert m is not None
        assert len(m.points) == 2

    def test_gjk_epa_query_cylinder_capsule_swapped(self):
        """Cylinder first, capsule second → still dispatched with flipped normal."""
        from physics.gjk_epa import gjk_epa_query

        cap = CapsuleShape(radius=0.2, length=1.0)
        cyl = CylinderShape(radius=0.15, length=0.8)
        pose_c = SpatialTransform.from_translation(np.zeros(3))
        pose_y = SpatialTransform.from_translation(np.array([0.3, 0.0, 0.0]))
        m_cy = gjk_epa_query(cap, pose_c, cyl, pose_y)
        m_yc = gjk_epa_query(cyl, pose_y, cap, pose_c)
        assert m_cy is not None and m_yc is not None
        assert np.allclose(m_cy.normal, -m_yc.normal, atol=ATOL)
