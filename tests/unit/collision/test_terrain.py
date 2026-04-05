"""
Unit tests for physics/terrain.py.
"""

import numpy as np
import pytest

from physics.terrain import FlatTerrain, HalfSpaceTerrain, HeightmapTerrain


class TestFlatTerrain:
    def test_height_at_default(self):
        t = FlatTerrain()
        assert t.height_at(0.0, 0.0) == 0.0
        assert t.height_at(100.0, -50.0) == 0.0

    def test_height_at_nonzero_z(self):
        t = FlatTerrain(z=0.5)
        assert t.height_at(0.0, 0.0) == 0.5
        assert t.height_at(99.0, -99.0) == 0.5

    def test_normal_always_up(self):
        t = FlatTerrain()
        np.testing.assert_allclose(t.normal_at(0.0, 0.0), [0, 0, 1])
        np.testing.assert_allclose(t.normal_at(50.0, -30.0), [0, 0, 1])

    def test_normal_nonzero_z(self):
        t = FlatTerrain(z=10.0)
        np.testing.assert_allclose(t.normal_at(0.0, 0.0), [0, 0, 1])


class TestHalfSpaceTerrain:
    def test_flat_equivalent(self):
        """z-up half-space at z=0.5 matches FlatTerrain(0.5)."""
        t = HalfSpaceTerrain(normal=np.array([0, 0, 1.0]), point=np.array([0, 0, 0.5]))
        assert t.height_at(0.0, 0.0) == pytest.approx(0.5)
        assert t.height_at(99.0, -42.0) == pytest.approx(0.5)
        np.testing.assert_allclose(t.normal_at(0.0, 0.0), [0, 0, 1])

    def test_incline_height(self):
        """30-degree incline around Y: height decreases with x."""
        theta = np.pi / 6
        normal = np.array([-np.sin(theta), 0.0, np.cos(theta)])
        t = HalfSpaceTerrain(normal=normal, point=np.zeros(3))
        # At x=0: z=0.  At x=1: plane slopes down.
        assert t.height_at(0.0, 0.0) == pytest.approx(0.0, abs=1e-12)
        # dot(n, [x, 0, z]) = 0 => z = sin(theta)/cos(theta) * x = tan(theta)*x
        # BUT normal points into the "uphill" direction, so positive x is downhill.
        # height_at(x=1) = (nx*(0-1))/nz = -(-sin30)/cos30 = tan30 ≈ 0.577
        expected_z = np.tan(theta) * 1.0
        assert t.height_at(1.0, 0.0) == pytest.approx(expected_z, rel=1e-10)

    def test_normal_at_returns_unit(self):
        normal = np.array([1.0, 1.0, 1.0])
        t = HalfSpaceTerrain(normal=normal, point=np.zeros(3))
        n = t.normal_at(5.0, 3.0)
        np.testing.assert_allclose(np.linalg.norm(n), 1.0)
        np.testing.assert_allclose(n, normal / np.linalg.norm(normal))

    def test_vertical_plane_height(self):
        """Vertical plane (nz=0) returns -1e6."""
        t = HalfSpaceTerrain(normal=np.array([1, 0, 0.0]), point=np.zeros(3))
        assert t.height_at(0.0, 0.0) == -1e6

    def test_mu_attribute(self):
        t = HalfSpaceTerrain(normal=np.array([0, 0, 1.0]), mu=0.3)
        assert t.mu == pytest.approx(0.3)

    def test_zero_normal_raises(self):
        with pytest.raises(ValueError):
            HalfSpaceTerrain(normal=np.array([0, 0, 0.0]))

    def test_properties(self):
        normal = np.array([0, 0, 1.0])
        point = np.array([1, 2, 3.0])
        t = HalfSpaceTerrain(normal=normal, point=point)
        np.testing.assert_allclose(t.normal_world, [0, 0, 1])
        np.testing.assert_allclose(t.point_on_plane, [1, 2, 3])


class TestHeightmapTerrain:
    def test_height_at_not_implemented(self):
        hm = np.zeros((10, 10))
        t = HeightmapTerrain(hm, resolution=0.1, origin=np.zeros(2))
        with pytest.raises(NotImplementedError):
            t.height_at(0.0, 0.0)

    def test_normal_at_not_implemented(self):
        hm = np.zeros((10, 10))
        t = HeightmapTerrain(hm, resolution=0.1, origin=np.zeros(2))
        with pytest.raises(NotImplementedError):
            t.normal_at(0.0, 0.0)
