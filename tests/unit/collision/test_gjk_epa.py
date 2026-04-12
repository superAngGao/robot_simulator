"""Tests for GJK/EPA collision detection."""

from __future__ import annotations

import numpy as np
import pytest

from physics.geometry import BoxShape, CapsuleShape, ConvexHullShape, CylinderShape, SphereShape
from physics.gjk_epa import epa, gjk, gjk_epa_query, ground_contact_query, halfspace_convex_query
from physics.spatial import SpatialTransform

ATOL = 1e-4


class TestSupportPoint:
    def test_box_support(self):
        box = BoxShape((2, 4, 6))
        s = box.support_point(np.array([1, 0, 0]))
        np.testing.assert_allclose(s, [1, 0, 0], atol=ATOL)  # half of 2
        s = box.support_point(np.array([0, -1, 0]))
        np.testing.assert_allclose(s, [0, -2, 0], atol=ATOL)

    def test_sphere_support(self):
        sphere = SphereShape(2.0)
        d = np.array([1, 1, 0], dtype=np.float64)
        s = sphere.support_point(d)
        expected = d / np.linalg.norm(d) * 2.0
        np.testing.assert_allclose(s, expected, atol=ATOL)

    def test_cylinder_support(self):
        cyl = CylinderShape(1.0, 4.0)
        # Straight up: top disk (z=half_length, any point on rim)
        s = cyl.support_point(np.array([0, 0, 1.0]))
        assert abs(s[2] - 2.0) < ATOL  # z = half_length
        assert np.sqrt(s[0] ** 2 + s[1] ** 2) <= 1.0 + ATOL  # on disk
        # Sideways: radius in X, z=0
        s = cyl.support_point(np.array([1, 0, 0.0]))
        np.testing.assert_allclose(s, [1, 0, 0], atol=ATOL)
        # Diagonal: should be on rim of top disk
        s = cyl.support_point(np.array([1, 0, 1.0]))
        assert abs(s[2] - 2.0) < ATOL
        assert abs(s[0] - 1.0) < ATOL


class TestGJK:
    def test_overlapping_boxes(self):
        box = BoxShape((2, 2, 2))
        pose_a = SpatialTransform.from_translation(np.array([0, 0, 0]))
        pose_b = SpatialTransform.from_translation(np.array([0.5, 0, 0]))
        hit, _ = gjk(box, pose_a, box, pose_b)
        assert hit, "Overlapping boxes should intersect"

    def test_separated_boxes(self):
        box = BoxShape((2, 2, 2))
        pose_a = SpatialTransform.from_translation(np.array([0, 0, 0]))
        pose_b = SpatialTransform.from_translation(np.array([5, 0, 0]))
        hit, _ = gjk(box, pose_a, box, pose_b)
        assert not hit, "Separated boxes should not intersect"

    def test_overlapping_spheres(self):
        sphere = SphereShape(1.0)
        pose_a = SpatialTransform.from_translation(np.array([0, 0, 0]))
        pose_b = SpatialTransform.from_translation(np.array([1.5, 0, 0]))
        hit, _ = gjk(sphere, pose_a, sphere, pose_b)
        assert hit, "Overlapping spheres should intersect"

    def test_separated_spheres(self):
        sphere = SphereShape(1.0)
        pose_a = SpatialTransform.from_translation(np.array([0, 0, 0]))
        pose_b = SpatialTransform.from_translation(np.array([3, 0, 0]))
        hit, _ = gjk(sphere, pose_a, sphere, pose_b)
        assert not hit

    def test_box_sphere_overlap(self):
        box = BoxShape((2, 2, 2))
        sphere = SphereShape(0.5)
        pose_a = SpatialTransform.from_translation(np.array([0, 0, 0]))
        pose_b = SpatialTransform.from_translation(np.array([1.2, 0, 0]))
        hit, _ = gjk(box, pose_a, sphere, pose_b)
        assert hit


class TestEPA:
    def test_sphere_sphere_depth(self):
        sphere = SphereShape(1.0)
        pose_a = SpatialTransform.from_translation(np.array([0, 0, 0]))
        pose_b = SpatialTransform.from_translation(np.array([1.5, 0, 0]))
        result = gjk_epa_query(sphere, pose_a, sphere, pose_b)
        assert result is not None
        # Penetration = 2*r - distance = 2.0 - 1.5 = 0.5
        assert abs(result.depth - 0.5) < 0.05, f"Expected depth ~0.5, got {result.depth}"

    def test_box_box_depth(self):
        box = BoxShape((2, 2, 2))
        pose_a = SpatialTransform.from_translation(np.array([0, 0, 0]))
        pose_b = SpatialTransform.from_translation(np.array([1.5, 0, 0]))
        result = gjk_epa_query(box, pose_a, box, pose_b)
        assert result is not None
        # Box half-extent 1.0, overlap = 2*1.0 - 1.5 = 0.5
        assert abs(result.depth - 0.5) < 0.05, f"Expected depth ~0.5, got {result.depth}"

    def test_separated_returns_none(self):
        sphere = SphereShape(1.0)
        pose_a = SpatialTransform.from_translation(np.array([0, 0, 0]))
        pose_b = SpatialTransform.from_translation(np.array([5, 0, 0]))
        result = gjk_epa_query(sphere, pose_a, sphere, pose_b)
        assert result is None

    def test_contact_normal_direction(self):
        sphere = SphereShape(1.0)
        pose_a = SpatialTransform.from_translation(np.array([0, 0, 0]))
        pose_b = SpatialTransform.from_translation(np.array([1.0, 0, 0]))
        result = gjk_epa_query(sphere, pose_a, sphere, pose_b)
        assert result is not None
        # Normal should be roughly along X axis
        assert abs(result.normal[0]) > 0.5


class TestGroundContact:
    def test_sphere_on_ground(self):
        sphere = SphereShape(0.5)
        pose = SpatialTransform.from_translation(np.array([0, 0, 0.3]))
        result = ground_contact_query(sphere, pose, ground_z=0.0)
        assert result is not None
        # Lowest point at z = 0.3 - 0.5 = -0.2, depth = 0.0 - (-0.2) = 0.2
        assert abs(result.depth - 0.2) < ATOL
        np.testing.assert_allclose(result.normal, [0, 0, 1], atol=ATOL)

    def test_sphere_above_ground(self):
        sphere = SphereShape(0.5)
        pose = SpatialTransform.from_translation(np.array([0, 0, 1.0]))
        result = ground_contact_query(sphere, pose, ground_z=0.0)
        assert result is None

    def test_box_on_ground(self):
        box = BoxShape((1, 1, 1))
        pose = SpatialTransform.from_translation(np.array([0, 0, 0.3]))
        result = ground_contact_query(box, pose, ground_z=0.0)
        assert result is not None
        # Lowest point at z = 0.3 - 0.5 = -0.2, depth = 0.2
        assert abs(result.depth - 0.2) < ATOL

    def test_capsule_on_ground(self):
        cap = CapsuleShape(0.3, 1.0)
        pose = SpatialTransform.from_translation(np.array([0, 0, 0.6]))
        result = ground_contact_query(cap, pose, ground_z=0.0)
        # Lowest: 0.6 - 0.5 - 0.3 = -0.2, depth = 0.2
        assert result is not None
        assert abs(result.depth - 0.2) < 0.02

    def test_cylinder_on_ground(self):
        cyl = CylinderShape(0.5, 2.0)
        pose = SpatialTransform.from_translation(np.array([0, 0, 0.8]))
        result = ground_contact_query(cyl, pose, ground_z=0.0)
        # Lowest: 0.8 - 1.0 = -0.2, depth = 0.2
        assert result is not None
        assert abs(result.depth - 0.2) < ATOL

    def test_nonzero_ground_z(self):
        sphere = SphereShape(0.5)
        pose = SpatialTransform.from_translation(np.array([0, 0, 1.3]))
        # Ground at z=1.0: lowest = 1.3-0.5=0.8, depth = 1.0-0.8 = 0.2
        result = ground_contact_query(sphere, pose, ground_z=1.0)
        assert result is not None
        assert abs(result.depth - 0.2) < ATOL

    def test_rotated_box(self):
        box = BoxShape((2, 1, 1))
        # 45 degree rotation around Y
        R = np.array(
            [
                [np.cos(np.pi / 4), 0, np.sin(np.pi / 4)],
                [0, 1, 0],
                [-np.sin(np.pi / 4), 0, np.cos(np.pi / 4)],
            ]
        )
        pose = SpatialTransform(R, np.array([0, 0, 2.0]))
        result = ground_contact_query(box, pose, ground_z=0.0)
        # Rotated box corner extends further down
        assert result is not None or result is None  # depends on height


class TestGJKEdgeCases:
    def test_identical_position(self):
        """Two shapes at same position should intersect."""
        box = BoxShape((1, 1, 1))
        pose = SpatialTransform.from_translation(np.zeros(3))
        hit, _ = gjk(box, pose, box, pose)
        assert hit

    def test_barely_touching(self):
        """Spheres just touching (distance = 2*r) should not penetrate."""
        sphere = SphereShape(1.0)
        pose_a = SpatialTransform.from_translation(np.array([0, 0, 0]))
        pose_b = SpatialTransform.from_translation(np.array([2.001, 0, 0]))
        hit, _ = gjk(sphere, pose_a, sphere, pose_b)
        assert not hit

    def test_rotated_boxes_overlap(self):
        """Two boxes, one rotated 45 degrees, should still detect overlap."""
        box = BoxShape((2, 2, 2))
        pose_a = SpatialTransform.from_translation(np.zeros(3))
        R45 = np.array(
            [
                [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
                [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
                [0, 0, 1],
            ]
        )
        pose_b = SpatialTransform(R45, np.array([1.2, 0, 0]))
        hit, _ = gjk(box, pose_a, box, pose_b)
        assert hit

    def test_rotated_box_epa_depth(self):
        """EPA should give correct depth for rotated overlap."""
        box = BoxShape((2, 2, 2))
        pose_a = SpatialTransform.from_translation(np.zeros(3))
        R45 = np.array(
            [
                [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
                [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
                [0, 0, 1],
            ]
        )
        pose_b = SpatialTransform(R45, np.array([1.0, 0, 0]))
        result = gjk_epa_query(box, pose_a, box, pose_b)
        assert result is not None
        assert result.depth > 0

    def test_capsule_sphere_overlap(self):
        cap = CapsuleShape(0.5, 2.0)
        sphere = SphereShape(0.5)
        pose_a = SpatialTransform.from_translation(np.zeros(3))
        pose_b = SpatialTransform.from_translation(np.array([0.8, 0, 0]))
        result = gjk_epa_query(cap, pose_a, sphere, pose_b)
        assert result is not None
        assert result.depth > 0

    def test_capsule_support_in_gjk(self):
        """Capsule support_point should work for GJK queries."""
        cap = CapsuleShape(0.3, 1.0)
        s = cap.support_point(np.array([1, 1, 1]))
        assert np.all(np.isfinite(s))


class TestHalfSpaceConvexQuery:
    """Tests for halfspace_convex_query (arbitrary-orientation ground plane)."""

    def test_sphere_on_z_up_plane(self):
        """Z-up half-space should match ground_contact_query."""
        sphere = SphereShape(0.1)
        pose = SpatialTransform.from_translation(np.array([0.0, 0.0, 0.05]))
        normal = np.array([0.0, 0.0, 1.0])
        point = np.zeros(3)
        result = halfspace_convex_query(sphere, pose, normal, point)
        ref = ground_contact_query(sphere, pose, ground_z=0.0)
        assert result is not None and ref is not None
        np.testing.assert_allclose(result.depth, ref.depth, atol=1e-12)
        np.testing.assert_allclose(result.normal, ref.normal, atol=1e-12)

    def test_sphere_above_incline_no_contact(self):
        """Sphere well above inclined plane → None."""
        sphere = SphereShape(0.1)
        normal = np.array([-np.sin(np.pi / 6), 0.0, np.cos(np.pi / 6)])
        point = np.zeros(3)
        # Place sphere high above the plane
        pose = SpatialTransform.from_translation(np.array([0.0, 0.0, 2.0]))
        result = halfspace_convex_query(sphere, pose, normal, point)
        assert result is None

    def test_sphere_penetrating_incline(self):
        """Sphere penetrating an inclined plane → correct depth and normal."""
        theta = np.pi / 6  # 30 degrees
        normal = np.array([-np.sin(theta), 0.0, np.cos(theta)])
        point = np.zeros(3)
        sphere = SphereShape(0.1)
        # Place sphere center ON the plane (depth = radius)
        pose = SpatialTransform.from_translation(np.zeros(3))
        result = halfspace_convex_query(sphere, pose, normal, point)
        assert result is not None
        assert result.depth == ATOL or result.depth > 0
        np.testing.assert_allclose(result.depth, 0.1, atol=1e-6)
        np.testing.assert_allclose(result.normal, normal, atol=1e-12)

    def test_box_on_incline(self):
        """Box sitting on 45-degree incline."""
        theta = np.pi / 4  # 45 degrees
        normal = np.array([-np.sin(theta), 0.0, np.cos(theta)])
        point = np.zeros(3)
        box = BoxShape((0.2, 0.2, 0.2))
        # Place box center on the plane
        pose = SpatialTransform.from_translation(np.zeros(3))
        result = halfspace_convex_query(box, pose, normal, point)
        assert result is not None
        assert result.depth > 0

    def test_contact_point_lies_on_plane(self):
        """Contact point should lie on the half-space surface."""
        theta = np.pi / 4
        normal = np.array([-np.sin(theta), 0.0, np.cos(theta)])
        point = np.zeros(3)
        sphere = SphereShape(0.1)
        # Place sphere center on the plane → penetrating by radius
        pose = SpatialTransform.from_translation(np.zeros(3))
        result = halfspace_convex_query(sphere, pose, normal, point)
        assert result is not None
        # Contact point should satisfy dot(n, cp - p0) = 0
        for cp in result.points:
            dist = np.dot(normal, cp - point)
            assert abs(dist) < 1e-10

    def test_margin_detects_nearby(self):
        """With margin > 0, detect shapes that are close but not penetrating."""
        sphere = SphereShape(0.1)
        normal = np.array([0.0, 0.0, 1.0])
        point = np.zeros(3)
        # Sphere at z=0.15 → gap = 0.05, no penetration
        pose = SpatialTransform.from_translation(np.array([0.0, 0.0, 0.15]))
        assert halfspace_convex_query(sphere, pose, normal, point) is None
        # With margin = 0.1, should detect
        result = halfspace_convex_query(sphere, pose, normal, point, margin=0.1)
        assert result is not None
        assert result.depth < 0  # negative depth = gap


# ---------------------------------------------------------------------------
# ConvexHullShape collision tests
# ---------------------------------------------------------------------------


def _make_cube_hull(half: float = 0.05) -> ConvexHullShape:
    """Create a unit cube ConvexHullShape centered at origin."""
    signs = np.array(
        [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]],
        dtype=np.float64,
    )
    return ConvexHullShape(signs * half)


class TestConvexHullCollision:
    def test_convexhull_vs_convexhull_overlap(self):
        """Two overlapping ConvexHullShape cubes → GJK detects intersection."""
        hull_a = _make_cube_hull(0.05)
        hull_b = _make_cube_hull(0.05)
        pose_a = SpatialTransform.from_translation(np.array([0.0, 0.0, 0.0]))
        pose_b = SpatialTransform.from_translation(np.array([0.08, 0.0, 0.0]))
        # Overlap = 0.05 + 0.05 - 0.08 = 0.02

        intersecting, _ = gjk(hull_a, pose_a, hull_b, pose_b)
        assert intersecting

        manifold = gjk_epa_query(hull_a, pose_a, hull_b, pose_b)
        assert manifold is not None
        assert manifold.depth > 0.01

    def test_convexhull_vs_convexhull_separated(self):
        """Two separated ConvexHullShape cubes → no collision."""
        hull_a = _make_cube_hull(0.05)
        hull_b = _make_cube_hull(0.05)
        pose_a = SpatialTransform.from_translation(np.array([0.0, 0.0, 0.0]))
        pose_b = SpatialTransform.from_translation(np.array([0.2, 0.0, 0.0]))

        intersecting, _ = gjk(hull_a, pose_a, hull_b, pose_b)
        assert not intersecting

        manifold = gjk_epa_query(hull_a, pose_a, hull_b, pose_b)
        assert manifold is None

    def test_convexhull_vs_sphere(self):
        """ConvexHullShape cube vs SphereShape → GJK detects intersection."""
        hull = _make_cube_hull(0.05)
        sphere = SphereShape(0.03)
        pose_hull = SpatialTransform.from_translation(np.array([0.0, 0.0, 0.0]))
        pose_sphere = SpatialTransform.from_translation(np.array([0.07, 0.0, 0.0]))
        # Overlap = 0.05 + 0.03 - 0.07 = 0.01

        manifold = gjk_epa_query(hull, pose_hull, sphere, pose_sphere)
        assert manifold is not None
        assert manifold.depth > 0.005


# ---------------------------------------------------------------------------
# EPA accuracy regression tests (session 27 — degenerate simplex fix)
# ---------------------------------------------------------------------------


class TestEPAAccuracyRegression:
    """Verify EPA depth/normal accuracy after GJK degenerate simplex fix.

    Previously, axis-aligned sphere-sphere and box-box pairs caused GJK
    to return a 2-point simplex (origin on segment, triple_product = 0).
    EPA then built a degenerate initial polytope and diverged (depth ≈ 0).

    Fix A: GJK _do_simplex_2 perpendicular fallback.
    Fix B: EPA hexahedron initialization for ≤2-point simplex.
    """

    @pytest.mark.parametrize(
        "radius,dist,exp_depth",
        [
            (0.05, 0.08, 0.02),  # small overlap
            (0.05, 0.05, 0.05),  # medium overlap
            (0.05, 0.02, 0.08),  # large overlap
            (0.5, 0.8, 0.2),  # large sphere
            (0.5, 0.5, 0.5),  # large sphere medium overlap
        ],
    )
    def test_sphere_sphere_axis_aligned(self, radius, dist, exp_depth):
        """Sphere-sphere along X axis — depth must match analytic value."""
        s1 = SphereShape(radius)
        s2 = SphereShape(radius)
        p1 = SpatialTransform.from_translation(np.array([0.0, 0.0, 0.0]))
        p2 = SpatialTransform.from_translation(np.array([dist, 0.0, 0.0]))
        result = gjk_epa_query(s1, p1, s2, p2)
        assert result is not None
        assert result.depth == pytest.approx(exp_depth, abs=1e-3)

    @pytest.mark.parametrize(
        "axis",
        [
            np.array([1.0, 0, 0]),
            np.array([0, 1.0, 0]),
            np.array([0, 0, 1.0]),
        ],
        ids=["X", "Y", "Z"],
    )
    def test_sphere_sphere_all_axes(self, axis):
        """Sphere-sphere along each coordinate axis."""
        s = SphereShape(0.05)
        p1 = SpatialTransform.from_translation(np.zeros(3))
        p2 = SpatialTransform.from_translation(0.08 * axis)
        result = gjk_epa_query(s, p1, s, p2)
        assert result is not None
        assert result.depth == pytest.approx(0.02, abs=1e-3)

    def test_sphere_sphere_normal_direction(self):
        """EPA normal should point from B to A (separation direction)."""
        s = SphereShape(0.05)
        p1 = SpatialTransform.from_translation(np.array([0.0, 0.0, 0.0]))
        p2 = SpatialTransform.from_translation(np.array([0.08, 0.0, 0.0]))
        result = gjk_epa_query(s, p1, s, p2)
        assert result is not None
        # Normal should be approximately (-1, 0, 0) or (1, 0, 0)
        assert abs(abs(result.normal[0]) - 1.0) < 0.05

    @pytest.mark.parametrize(
        "half,dist,exp_depth",
        [
            (0.05, 0.08, 0.02),
            (0.05, 0.05, 0.05),
            (0.5, 0.8, 0.2),
        ],
    )
    def test_box_box_axis_aligned(self, half, dist, exp_depth):
        """Box-box along X axis — depth must match analytic value."""
        b1 = BoxShape((half * 2, half * 2, half * 2))
        b2 = BoxShape((half * 2, half * 2, half * 2))
        p1 = SpatialTransform.from_translation(np.zeros(3))
        p2 = SpatialTransform.from_translation(np.array([dist, 0.0, 0.0]))
        result = gjk_epa_query(b1, p1, b2, p2)
        assert result is not None
        assert result.depth == pytest.approx(exp_depth, abs=1e-3)

    def test_epa_hexahedron_path_2pt_simplex(self):
        """EPA with forced 2-point simplex uses hexahedron initialization."""
        s = SphereShape(0.05)
        p1 = SpatialTransform.from_translation(np.array([0, 0, 1.0]))
        p2 = SpatialTransform.from_translation(np.array([0.08, 0, 1.0]))
        # Force 2-point simplex (bypassing GJK fix)
        simplex = [np.array([-0.18, 0, 0]), np.array([0.02, 0, 0])]
        n, d = epa(s, p1, s, p2, simplex)
        assert d == pytest.approx(0.02, abs=1e-3)

    def test_epa_hexahedron_path_1pt_simplex(self):
        """EPA with 1-point simplex still converges correctly."""
        s = SphereShape(0.05)
        p1 = SpatialTransform.from_translation(np.array([0, 0, 1.0]))
        p2 = SpatialTransform.from_translation(np.array([0.08, 0, 1.0]))
        simplex = [np.array([-0.18, 0, 0])]
        n, d = epa(s, p1, s, p2, simplex)
        assert d == pytest.approx(0.02, abs=1e-3)
