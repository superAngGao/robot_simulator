"""Unit tests for `physics.cylinder_collision` analytical handlers and
N-gon prism tessellation on `CylinderShape`."""

from __future__ import annotations

import numpy as np

from physics.cylinder_collision import cylinder_halfspace_manifold
from physics.geometry import BoxShape, CylinderShape
from physics.gjk_epa import gjk_epa_query, ground_contact_query, halfspace_convex_query
from physics.spatial import SpatialTransform

ATOL = 1e-6


def _rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# ---------------------------------------------------------------------------
# Cylinder vs half-space analytical
# ---------------------------------------------------------------------------


class TestCylinderHalfspace:
    def test_standing_cylinder_four_rim_points(self):
        """Axis perpendicular to plane → 4 rim-quadrant contacts."""
        cyl = CylinderShape(radius=0.5, length=1.0)
        pose = SpatialTransform.from_translation(np.array([0.0, 0.0, 0.4]))
        # Bottom cap at z=-0.1 below plane → penetrating by 0.1
        m = cylinder_halfspace_manifold(cyl, pose, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]))
        assert m is not None
        assert len(m.points) == 4
        # All contacts at the same depth (rim is coplanar)
        for d in m.point_depths:
            assert abs(d - 0.1) < ATOL
        # Contacts projected onto the plane (z=0) at radius r around origin
        for p in m.points:
            assert abs(p[2] - 0.0) < ATOL
            assert abs(np.sqrt(p[0] ** 2 + p[1] ** 2) - 0.5) < ATOL

    def test_lying_cylinder_two_line_points(self):
        """Axis parallel to plane → 2 contacts at segment endpoints."""
        cyl = CylinderShape(radius=0.3, length=2.0)
        # Rotate so local +Z → world +X (axis horizontal)
        pose = SpatialTransform(_rot_y(np.pi / 2), np.array([0.0, 0.0, 0.2]))
        m = cylinder_halfspace_manifold(cyl, pose, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]))
        assert m is not None
        assert len(m.points) == 2
        # Lowest point at z = 0.2 - 0.3 = -0.1, depth = 0.1
        for d in m.point_depths:
            assert abs(d - 0.1) < ATOL
        # Contacts separated by full length along x axis
        xs = sorted([p[0] for p in m.points])
        assert abs(xs[1] - xs[0] - 2.0) < ATOL

    def test_tilted_cylinder_single_rim_point(self):
        """Axis tilted 30° → single deepest rim contact."""
        cyl = CylinderShape(radius=0.5, length=1.0)
        # 30° rotation around Y → axis tilted 30° from +Z
        pose = SpatialTransform(_rot_y(np.pi / 6), np.array([0.0, 0.0, 0.0]))
        m = cylinder_halfspace_manifold(cyl, pose, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]))
        assert m is not None
        assert len(m.points) == 1
        # The deepest rim point lies on the lower end cap, at the rim
        # direction that maximises -z. Just check non-trivial depth.
        assert m.depth > 0.4

    def test_fully_above_plane_returns_none(self):
        cyl = CylinderShape(radius=0.1, length=1.0)
        pose = SpatialTransform.from_translation(np.array([0.0, 0.0, 2.0]))
        m = cylinder_halfspace_manifold(cyl, pose, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]))
        assert m is None

    def test_ground_contact_query_dispatch_lying(self):
        cyl = CylinderShape(radius=0.3, length=2.0)
        pose = SpatialTransform(_rot_y(np.pi / 2), np.array([0.0, 0.0, 0.2]))
        m = ground_contact_query(cyl, pose, ground_z=0.0)
        assert m is not None
        assert len(m.points) == 2

    def test_halfspace_convex_query_dispatch_standing(self):
        cyl = CylinderShape(radius=0.5, length=1.0)
        pose = SpatialTransform.from_translation(np.array([0.0, 0.0, 0.4]))
        m = halfspace_convex_query(
            cyl,
            pose,
            hs_normal_world=np.array([0.0, 0.0, 1.0]),
            hs_point_world=np.array([0.0, 0.0, 0.0]),
        )
        assert m is not None
        assert len(m.points) == 4


# ---------------------------------------------------------------------------
# Prism tessellation behaviour
# ---------------------------------------------------------------------------


class TestCylinderPrismTessellation:
    def test_default_n_tessellation_is_12(self):
        cyl = CylinderShape(radius=1.0, length=2.0)
        assert cyl.n_tessellation == 12

    def test_custom_n_tessellation_changes_faces(self):
        cyl_8 = CylinderShape(radius=1.0, length=2.0, n_tessellation=8)
        cyl_16 = CylinderShape(radius=1.0, length=2.0, n_tessellation=16)
        topo_8 = cyl_8.face_topology()
        topo_16 = cyl_16.face_topology()
        # 2 caps + N sides
        assert topo_8.num_faces == 10
        assert topo_16.num_faces == 18

    def test_support_point_is_prism_vertex(self):
        cyl = CylinderShape(radius=1.0, length=4.0, n_tessellation=12)
        # Support in direction (cos 0, sin 0, 1) should hit the top-cap vertex at angle 0
        s = cyl.support_point(np.array([1.0, 0.0, 1.0]))
        # Top ring vertex at angle 0: [1, 0, 2]
        assert abs(s[0] - 1.0) < ATOL
        assert abs(s[1]) < ATOL
        assert abs(s[2] - 2.0) < ATOL

    def test_face_topology_covers_cap_and_side_faces(self):
        cyl = CylinderShape(radius=1.0, length=2.0, n_tessellation=6)
        topo = cyl.face_topology()
        # Face 0: top cap, normal +z
        assert np.allclose(topo.normals[0], [0, 0, 1], atol=ATOL)
        # Face 1: bottom cap, normal -z
        assert np.allclose(topo.normals[1], [0, 0, -1], atol=ATOL)
        # Side face normals lie in xy plane (z=0)
        for i in range(2, 2 + 6):
            assert abs(topo.normals[i][2]) < ATOL
            assert abs(np.linalg.norm(topo.normals[i]) - 1.0) < ATOL

    def test_invalid_n_tessellation_raises(self):
        import pytest

        with pytest.raises(ValueError, match="n_tessellation must be >= 3"):
            CylinderShape(radius=1.0, length=2.0, n_tessellation=2)


# ---------------------------------------------------------------------------
# Cylinder-Box body-body contact via prism S-H path
# ---------------------------------------------------------------------------


class TestCylinderBoxViaGjkEpa:
    def test_cylinder_on_box_top_face_multipoint(self):
        """Cylinder lying flat on a wide box → should generate multi-point contact.

        With the prism tessellation, body-body cylinder-box goes through
        GJK/EPA + Sutherland-Hodgman face clipping. The contact manifold
        should include more than 1 point for a full-face contact.
        """
        box = BoxShape((4.0, 4.0, 2.0))
        box_pose = SpatialTransform.from_translation(np.array([0.0, 0.0, 0.0]))
        # Horizontal cylinder, axis along +Y, resting on box top (z=1.0)
        cyl = CylinderShape(radius=0.3, length=1.5, n_tessellation=12)
        cyl_pose = SpatialTransform(_rot_x(np.pi / 2), np.array([0.0, 0.0, 1.25]))
        # Cylinder bottom (on side) at z = 1.25 - 0.3 = 0.95, box top = 1.0, depth=0.05
        m = gjk_epa_query(cyl, cyl_pose, box, box_pose)
        assert m is not None
        assert m.depth > 0.0
        # At least 2 contacts for the line-of-contact along the cylinder axis.
        assert len(m.points) >= 2

    def test_cylinder_end_against_box_face(self):
        """Cylinder standing on a flat box top → N-gon rim multi-point."""
        box = BoxShape((4.0, 4.0, 2.0))
        box_pose = SpatialTransform.from_translation(np.array([0.0, 0.0, 0.0]))
        cyl = CylinderShape(radius=0.5, length=1.0, n_tessellation=12)
        # Vertical cylinder, bottom just penetrating top face
        cyl_pose = SpatialTransform.from_translation(np.array([0.0, 0.0, 1.45]))
        # Bottom cap at z=0.95, box top z=1.0 → depth 0.05
        m = gjk_epa_query(cyl, cyl_pose, box, box_pose)
        assert m is not None
        assert m.depth > 0.0
        # Face-face contact should give multiple points (up to 4 after S-H cap)
        assert len(m.points) >= 2
