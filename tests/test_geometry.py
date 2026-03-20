"""
Unit tests for physics/geometry.py — collision shape primitives.
"""

import numpy as np
import pytest

from physics.geometry import (
    BodyCollisionGeometry,
    BoxShape,
    CylinderShape,
    MeshShape,
    ShapeInstance,
    SphereShape,
)

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


class TestMeshShape:
    def test_not_implemented(self):
        s = MeshShape("robot.stl")
        with pytest.raises(NotImplementedError):
            s.half_extents_approx()

    def test_stores_filename(self):
        s = MeshShape("path/to/mesh.obj")
        assert s.filename == "path/to/mesh.obj"


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
