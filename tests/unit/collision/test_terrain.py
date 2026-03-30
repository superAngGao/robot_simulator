"""
Unit tests for physics/terrain.py.
"""

import numpy as np
import pytest

from physics.terrain import FlatTerrain, HeightmapTerrain


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
