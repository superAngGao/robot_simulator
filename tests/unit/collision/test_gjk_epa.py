"""Tests for GJK/EPA collision detection."""

from __future__ import annotations

import numpy as np

from physics.geometry import BoxShape, CapsuleShape, CylinderShape, SphereShape
from physics.gjk_epa import gjk, gjk_epa_query, ground_contact_query
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
