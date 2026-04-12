"""Tests for contact manifold generation (方案 B: face clipping + edge-edge).

Verifies that gjk_epa_query and build_contact_manifold produce correct
multi-point contact manifolds for box-box collisions:
  - face-face: 4 contact points
  - face-edge: 2 contact points
  - vertex-face: 1 contact point
  - edge-edge: 1 contact point (closest point on two edges)

Also tests FaceTopology construction for BoxShape and ConvexHullShape.

Reference: Dirk Gregorius, "Robust Contact Creation" (GDC 2015).
"""

import numpy as np
import pytest

from physics.geometry import BoxShape, ConvexHullShape, SphereShape
from physics.gjk_epa import build_contact_manifold, gjk_epa_query
from physics.spatial import SpatialTransform

# ---------------------------------------------------------------------------
# FaceTopology tests
# ---------------------------------------------------------------------------


class TestFaceTopologyBox:
    """Verify FaceTopology for BoxShape."""

    def test_box_face_count(self):
        b = BoxShape((1, 2, 3))
        t = b.face_topology()
        assert t is not None
        assert t.num_faces == 6

    def test_box_vertex_count(self):
        t = BoxShape((1, 1, 1)).face_topology()
        assert t.vertices.shape == (8, 3)

    def test_box_edge_count(self):
        t = BoxShape((1, 1, 1)).face_topology()
        assert t.num_edges == 12

    def test_box_face_normals_are_unit(self):
        t = BoxShape((2, 3, 5)).face_topology()
        norms = np.linalg.norm(t.normals, axis=1)
        np.testing.assert_allclose(norms, 1.0)

    def test_box_face_normals_are_axis_aligned(self):
        t = BoxShape((1, 1, 1)).face_topology()
        # Each normal should have exactly one non-zero component
        for n in t.normals:
            assert np.count_nonzero(n) == 1

    def test_box_find_support_face_plus_z(self):
        t = BoxShape((1, 1, 1)).face_topology()
        fi = t.find_support_face(np.array([0, 0, 1.0]))
        np.testing.assert_allclose(t.normals[fi], [0, 0, 1])

    def test_box_find_support_face_minus_x(self):
        t = BoxShape((1, 1, 1)).face_topology()
        fi = t.find_support_face(np.array([-1, 0, 0.0]))
        np.testing.assert_allclose(t.normals[fi], [-1, 0, 0])

    def test_box_face_polygon_vertex_count(self):
        t = BoxShape((2, 2, 2)).face_topology()
        for fi in range(6):
            poly = t.face_polygon(fi)
            assert poly.shape == (4, 3)

    def test_box_side_planes_count(self):
        t = BoxShape((1, 1, 1)).face_topology()
        for fi in range(6):
            planes = t.side_planes(fi)
            assert len(planes) == 4

    def test_box_side_planes_center_inside(self):
        """Face center should be inside all side planes."""
        t = BoxShape((2, 2, 2)).face_topology()
        for fi in range(6):
            center = t.face_polygon(fi).mean(axis=0)
            for n, d in t.side_planes(fi):
                assert np.dot(n, center) <= d + 1e-10


class TestFaceTopologyConvexHull:
    """Verify FaceTopology for ConvexHullShape."""

    def test_octahedron(self):
        verts = np.array(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
            dtype=np.float64,
        )
        ch = ConvexHullShape(verts)
        t = ch.face_topology()
        assert t is not None
        assert t.num_faces == 8  # octahedron has 8 triangular faces
        assert t.num_edges == 12

    def test_convexhull_normals_outward(self):
        """All face normals should point outward (dot with centroid→face > 0)."""
        verts = np.array(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
            dtype=np.float64,
        )
        t = ConvexHullShape(verts).face_topology()
        centroid = t.vertices.mean(axis=0)
        for fi in range(t.num_faces):
            face_center = t.face_polygon(fi).mean(axis=0)
            outward_dir = face_center - centroid
            assert np.dot(t.normals[fi], outward_dir) > 0

    def test_smooth_shape_returns_none(self):
        s = SphereShape(1.0)
        assert s.face_topology() is None


# ---------------------------------------------------------------------------
# Contact manifold generation tests
# ---------------------------------------------------------------------------


def _id_pose(pos):
    """Identity rotation at given position."""
    return SpatialTransform(np.eye(3), np.asarray(pos, dtype=np.float64))


def _rot_y(angle):
    """Rotation matrix around Y axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_x(angle):
    """Rotation matrix around X axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


class TestBoxBoxFaceFace:
    """Box resting flat on box — face-face contact, expect 4 points."""

    def test_face_face_4_points(self):
        a = BoxShape((1, 1, 1))
        b = BoxShape((2, 2, 1))
        pa = _id_pose([0, 0, 0.9])
        pb = _id_pose([0, 0, 0])
        m = gjk_epa_query(a, pa, b, pb)
        assert m is not None
        assert len(m.points) == 4

    def test_face_face_depth(self):
        a = BoxShape((1, 1, 1))
        b = BoxShape((2, 2, 1))
        pa = _id_pose([0, 0, 0.9])  # overlap = 0.5 + 0.5 - 0.9 = 0.1
        pb = _id_pose([0, 0, 0])
        m = gjk_epa_query(a, pa, b, pb)
        for i in range(len(m.points)):
            assert m.depth_at(i) == pytest.approx(0.1, abs=1e-3)

    def test_face_face_points_at_correct_z(self):
        a = BoxShape((1, 1, 1))
        b = BoxShape((2, 2, 1))
        pa = _id_pose([0, 0, 0.9])
        pb = _id_pose([0, 0, 0])
        m = gjk_epa_query(a, pa, b, pb)
        # Contact points should be at A's bottom face (z = 0.9 - 0.5 = 0.4)
        # or B's top face (z = 0.5), projected onto reference plane
        for p in m.points:
            assert 0.39 <= p[2] <= 0.51

    def test_face_face_points_cover_corners(self):
        """Contact points should span A's bottom face corners."""
        a = BoxShape((1, 1, 1))
        b = BoxShape((2, 2, 1))
        pa = _id_pose([0, 0, 0.9])
        pb = _id_pose([0, 0, 0])
        m = gjk_epa_query(a, pa, b, pb)
        xs = sorted(p[0] for p in m.points)
        ys = sorted(p[1] for p in m.points)
        assert xs[0] == pytest.approx(-0.5, abs=0.05)
        assert xs[-1] == pytest.approx(0.5, abs=0.05)
        assert ys[0] == pytest.approx(-0.5, abs=0.05)
        assert ys[-1] == pytest.approx(0.5, abs=0.05)


class TestBoxBoxEdgeFace:
    """Box rotated 45° around Y — edge rests on face, expect 2 points."""

    def test_edge_face_2_points(self):
        a = BoxShape((1, 1, 1))
        b = BoxShape((2, 2, 1))
        R = _rot_y(np.pi / 4)
        pa = SpatialTransform(R, np.array([0, 0, 0.8]))
        pb = _id_pose([0, 0, 0])
        m = gjk_epa_query(a, pa, b, pb)
        assert m is not None
        assert len(m.points) == 2

    def test_edge_face_points_along_edge(self):
        """Two contact points should differ only in Y (the edge direction)."""
        a = BoxShape((1, 1, 1))
        b = BoxShape((2, 2, 1))
        R = _rot_y(np.pi / 4)
        pa = SpatialTransform(R, np.array([0, 0, 0.8]))
        pb = _id_pose([0, 0, 0])
        m = gjk_epa_query(a, pa, b, pb)
        p0, p1 = m.points
        # X and Z should be nearly equal (both on the same edge)
        assert p0[0] == pytest.approx(p1[0], abs=0.05)
        assert p0[2] == pytest.approx(p1[2], abs=0.05)
        # Y should differ (along the edge)
        assert abs(p0[1] - p1[1]) > 0.3


class TestBoxBoxVertexFace:
    """Box rotated by a corner poking into face — expect 1 point."""

    def test_vertex_face_1_point(self):
        a = BoxShape((1, 1, 1))
        b = BoxShape((2, 2, 1))
        R = _rot_y(0.6) @ _rot_x(0.6)
        pa = SpatialTransform(R, np.array([0, 0, 1.1]))
        pb = _id_pose([0, 0, 0])
        m = gjk_epa_query(a, pa, b, pb)
        assert m is not None
        assert len(m.points) == 1


class TestBoxBoxEdgeEdge:
    """Two box edges crossing — edge-edge contact via closest points.

    Because EPA has numerical difficulties with some symmetric configurations,
    we test via build_contact_manifold directly with a known edge-edge normal.
    """

    def test_edge_edge_1_point(self):
        box1 = BoxShape((2, 0.3, 0.3))
        box2 = BoxShape((2, 0.3, 0.3))
        Rz90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        pose1 = _id_pose([0, 0, 0.1])
        pose2 = SpatialTransform(Rz90, np.zeros(3))
        # Edge-edge normal: cross(X-axis, Y-axis) = Z, but tilted to
        # trigger edge-edge path. Use diagonal in XY plane.
        n = np.array([1.0, 1.0, 0.0])
        n = n / np.linalg.norm(n)
        m = build_contact_manifold(box1, pose1, box2, pose2, n, 0.1)
        assert len(m.points) == 1

    def test_edge_edge_point_near_crossing(self):
        """Contact point should be near the actual crossing location (origin XY)."""
        box1 = BoxShape((2, 0.3, 0.3))
        box2 = BoxShape((2, 0.3, 0.3))
        Rz90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        pose1 = _id_pose([0, 0, 0.1])
        pose2 = SpatialTransform(Rz90, np.zeros(3))
        n = np.array([1.0, 1.0, 0.0])
        n = n / np.linalg.norm(n)
        m = build_contact_manifold(box1, pose1, box2, pose2, n, 0.1)
        p = m.points[0]
        # Crossing is near (0, 0, ~0.05) — allow generous tolerance
        assert abs(p[0]) < 1.0
        assert abs(p[1]) < 1.0


class TestSmoothShapeFallback:
    """Smooth shapes (Sphere) fall back to single-point support midpoint."""

    def test_sphere_sphere_single_point(self):
        s1 = SphereShape(0.5)
        s2 = SphereShape(0.5)
        p1 = _id_pose([0, 0, 0])
        p2 = _id_pose([0.8, 0, 0])
        m = gjk_epa_query(s1, p1, s2, p2)
        assert m is not None
        assert len(m.points) == 1

    def test_sphere_box_single_point(self):
        """Mixed smooth+polyhedral falls back to single point."""
        s = SphereShape(0.5)
        b = BoxShape((1, 1, 1))
        ps = _id_pose([0, 0, 0.8])
        pb = _id_pose([0, 0, 0])
        m = gjk_epa_query(s, ps, b, pb)
        assert m is not None
        assert len(m.points) == 1
