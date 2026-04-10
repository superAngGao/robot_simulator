"""
Unit tests for physics/geometry.py — collision shape primitives.
"""

import numpy as np
import pytest

from physics.geometry import (
    BodyCollisionGeometry,
    BoxShape,
    ConvexHullShape,
    CylinderShape,
    MeshShape,
    ShapeInstance,
    SphereShape,
)
from physics.spatial import SpatialTransform

ATOL = 1e-12


class TestBoxShape:
    def test_half_extents(self):
        s = BoxShape((0.2, 0.4, 0.6))
        np.testing.assert_allclose(s.half_extents_approx(), [0.1, 0.2, 0.3], atol=ATOL)

    def test_unit_cube(self):
        s = BoxShape((1.0, 1.0, 1.0))
        np.testing.assert_allclose(s.half_extents_approx(), [0.5, 0.5, 0.5], atol=ATOL)


class TestSphereShape:
    def test_half_extents(self):
        s = SphereShape(0.5)
        np.testing.assert_allclose(s.half_extents_approx(), [0.5, 0.5, 0.5], atol=ATOL)

    def test_small_radius(self):
        s = SphereShape(0.01)
        np.testing.assert_allclose(s.half_extents_approx(), [0.01, 0.01, 0.01], atol=ATOL)


class TestCylinderShape:
    def test_half_extents(self):
        s = CylinderShape(radius=0.1, length=0.4)
        np.testing.assert_allclose(s.half_extents_approx(), [0.1, 0.1, 0.2], atol=ATOL)


class TestConvexHullShape:
    def test_support_point_cube(self):
        """8 cube vertices: support in +x should return the +x face vertex."""
        verts = np.array(
            [
                [1, 1, 1],
                [1, 1, -1],
                [1, -1, 1],
                [1, -1, -1],
                [-1, 1, 1],
                [-1, 1, -1],
                [-1, -1, 1],
                [-1, -1, -1],
            ],
            dtype=float,
        )
        ch = ConvexHullShape(verts)
        sp = ch.support_point(np.array([1, 0, 0]))
        assert sp[0] == 1.0

    def test_support_point_tetrahedron(self):
        verts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]], dtype=float)
        ch = ConvexHullShape(verts)
        sp = ch.support_point(np.array([1, 0, 0]))
        np.testing.assert_allclose(sp, [1, 0, 0], atol=ATOL)
        sp_neg = ch.support_point(np.array([-1, -1, -1]))
        np.testing.assert_allclose(sp_neg, [-1, -1, -1], atol=ATOL)

    def test_half_extents(self):
        verts = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 1], [-1, -1, -1]], dtype=float)
        ch = ConvexHullShape(verts)
        np.testing.assert_allclose(ch.half_extents_approx(), [2, 3, 1], atol=ATOL)

    def test_rejects_too_few_vertices(self):
        with pytest.raises(ValueError):
            ConvexHullShape(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            ConvexHullShape(np.array([[1, 0], [0, 1], [0, 0], [1, 1]]))


class TestMeshShape:
    def test_not_implemented_without_vertices(self):
        s = MeshShape("robot.stl")
        with pytest.raises(NotImplementedError):
            s.half_extents_approx()
        with pytest.raises(NotImplementedError):
            s.support_point(np.array([1, 0, 0]))

    def test_stores_filename(self):
        s = MeshShape("path/to/mesh.obj")
        assert s.filename == "path/to/mesh.obj"

    def test_with_vertices(self):
        verts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        s = MeshShape("test.stl", vertices=verts)
        np.testing.assert_allclose(s.half_extents_approx(), [1, 1, 1], atol=ATOL)
        sp = s.support_point(np.array([0, 1, 0]))
        np.testing.assert_allclose(sp, [0, 1, 0], atol=ATOL)

    def test_rejects_bad_vertices(self):
        with pytest.raises(ValueError):
            MeshShape("test.stl", vertices=np.array([1, 2, 3]))

    def test_contact_vertices_with_data(self):
        verts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        s = MeshShape("test.stl", vertices=verts)
        cv = s.contact_vertices()
        assert cv is not None
        np.testing.assert_allclose(cv, verts, atol=ATOL)

    def test_contact_vertices_none_without_vertices(self):
        s = MeshShape("test.stl")
        assert s.contact_vertices() is None

    def test_scale_applied_to_vertices(self):
        verts = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        s = MeshShape("test.stl", vertices=verts, scale=(2.0, 1.0, 0.5))
        np.testing.assert_allclose(s.vertices, [[2, 2, 1.5], [8, 5, 3]], atol=ATOL)
        np.testing.assert_allclose(s.scale, [2.0, 1.0, 0.5], atol=ATOL)

    def test_scale_default_is_identity(self):
        s = MeshShape("test.stl")
        np.testing.assert_allclose(s.scale, [1.0, 1.0, 1.0], atol=ATOL)


class TestShapeInstanceWorldPose:
    def test_zero_offset_returns_same(self):
        si = ShapeInstance(SphereShape(0.1))
        X = SpatialTransform.from_translation(np.array([1, 2, 3]))
        result = si.world_pose(X)
        assert result is X

    def test_translation_offset(self):
        si = ShapeInstance(SphereShape(0.1), origin_xyz=np.array([0.5, 0, 0]))
        X = SpatialTransform.identity()
        result = si.world_pose(X)
        np.testing.assert_allclose(result.r, [0.5, 0, 0], atol=ATOL)

    def test_combined_body_and_offset(self):
        si = ShapeInstance(SphereShape(0.1), origin_xyz=np.array([0.1, 0, 0]))
        X = SpatialTransform.from_translation(np.array([1, 0, 0]))
        result = si.world_pose(X)
        np.testing.assert_allclose(result.r, [1.1, 0, 0], atol=ATOL)

    def test_rotation_offset(self):
        si = ShapeInstance(SphereShape(0.1), origin_rpy=np.array([0, 0, np.pi / 2]))
        X = SpatialTransform.identity()
        result = si.world_pose(X)
        # After 90-deg yaw, x-axis of shape points along y
        expected_R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(result.R, expected_R, atol=1e-10)


class TestAabbWithOffset:
    def test_offset_increases_half_extents(self):
        s = SphereShape(0.1)  # half = [0.1, 0.1, 0.1]
        geom_no_offset = BodyCollisionGeometry(0, [ShapeInstance(s)])
        geom_with_offset = BodyCollisionGeometry(0, [ShapeInstance(s, origin_xyz=np.array([0.5, 0, 0]))])
        he0 = geom_no_offset.aabb_half_extents()
        he1 = geom_with_offset.aabb_half_extents()
        assert he1[0] > he0[0]
        np.testing.assert_allclose(he1, [0.6, 0.1, 0.1], atol=ATOL)

    def test_multi_shape_with_offsets(self):
        s1 = SphereShape(0.1)
        s2 = BoxShape((0.2, 0.2, 0.2))  # half = [0.1, 0.1, 0.1]
        geom = BodyCollisionGeometry(
            0,
            [
                ShapeInstance(s1, origin_xyz=np.array([0.5, 0, 0])),
                ShapeInstance(s2, origin_xyz=np.array([0, 0.3, 0])),
            ],
        )
        he = geom.aabb_half_extents()
        # s1: [0.1+0.5, 0.1, 0.1] = [0.6, 0.1, 0.1]
        # s2: [0.1, 0.1+0.3, 0.1] = [0.1, 0.4, 0.1]
        # max: [0.6, 0.4, 0.1]
        np.testing.assert_allclose(he, [0.6, 0.4, 0.1], atol=ATOL)


class TestBodyCollisionGeometry:
    def test_single_shape(self):
        shape = BoxShape((0.2, 0.4, 0.6))
        geom = BodyCollisionGeometry(body_index=0, shapes=[ShapeInstance(shape=shape)])
        np.testing.assert_allclose(geom.aabb_half_extents(), [0.1, 0.2, 0.3], atol=ATOL)

    def test_multiple_shapes_takes_max(self):
        s1 = BoxShape((0.2, 0.1, 0.1))  # half = [0.1, 0.05, 0.05]
        s2 = SphereShape(0.08)  # half = [0.08, 0.08, 0.08]
        geom = BodyCollisionGeometry(
            body_index=0,
            shapes=[ShapeInstance(shape=s1), ShapeInstance(shape=s2)],
        )
        # Element-wise max: [0.1, 0.08, 0.08]
        np.testing.assert_allclose(geom.aabb_half_extents(), [0.1, 0.08, 0.08], atol=ATOL)

    def test_empty_shapes(self):
        geom = BodyCollisionGeometry(body_index=0, shapes=[])
        np.testing.assert_allclose(geom.aabb_half_extents(), [0, 0, 0], atol=ATOL)
