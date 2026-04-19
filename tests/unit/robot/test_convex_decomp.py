"""Unit tests for robot.convex_decomp."""

from __future__ import annotations

import numpy as np
import pytest

from physics.geometry import ConvexHullShape

trimesh = pytest.importorskip("trimesh", reason="trimesh required")


from robot.convex_decomp import DecompBackend, decompose_mesh  # noqa: E402


def _box_mesh(half: float = 0.05):
    return trimesh.creation.box(extents=[2 * half] * 3)


def _sphere_mesh(r: float = 0.05):
    return trimesh.creation.icosphere(subdivisions=2, radius=r)


def _cylinder_mesh(r: float = 0.04, h: float = 0.20):
    return trimesh.creation.cylinder(radius=r, height=h, sections=16)


# ---------------------------------------------------------------------------
# single-hull fallback (always available)
# ---------------------------------------------------------------------------


class TestSingleHull:
    def test_box_returns_one_piece(self):
        hulls = decompose_mesh(_box_mesh(), backend=DecompBackend.SINGLE)
        assert len(hulls) == 1
        assert isinstance(hulls[0], ConvexHullShape)

    def test_box_vertices_count(self):
        hulls = decompose_mesh(_box_mesh(0.05), backend=DecompBackend.SINGLE)
        # Box convex hull has 8 vertices
        assert hulls[0].vertices.shape == (8, 3)

    def test_sphere_vertices_positive(self):
        hulls = decompose_mesh(_sphere_mesh(), backend=DecompBackend.SINGLE)
        assert hulls[0].vertices.shape[0] >= 4

    def test_cylinder_vertices_positive(self):
        hulls = decompose_mesh(_cylinder_mesh(), backend=DecompBackend.SINGLE)
        assert hulls[0].vertices.shape[0] >= 4

    def test_hull_vertices_are_float64(self):
        hulls = decompose_mesh(_box_mesh(), backend=DecompBackend.SINGLE)
        assert hulls[0].vertices.dtype == np.float64

    def test_hull_vertices_within_mesh_bounds(self):
        half = 0.05
        hulls = decompose_mesh(_box_mesh(half), backend=DecompBackend.SINGLE)
        verts = hulls[0].vertices
        assert np.all(np.abs(verts) <= half + 1e-9)

    def test_irregular_convex_shape(self):
        """Random convex point cloud → single hull."""
        rng = np.random.default_rng(42)
        pts = rng.standard_normal((30, 3))
        from scipy.spatial import ConvexHull

        ch = ConvexHull(pts)
        mesh = trimesh.Trimesh(vertices=pts, faces=ch.simplices)
        hulls = decompose_mesh(mesh, backend=DecompBackend.SINGLE)
        assert len(hulls) == 1
        assert hulls[0].vertices.shape[0] >= 4


# ---------------------------------------------------------------------------
# auto backend (falls through to single when coacd/vhacd absent)
# ---------------------------------------------------------------------------


class TestAutoBackend:
    def test_auto_returns_at_least_one_piece(self):
        hulls = decompose_mesh(_box_mesh())
        assert len(hulls) >= 1
        assert all(isinstance(h, ConvexHullShape) for h in hulls)

    def test_auto_all_pieces_have_enough_vertices(self):
        hulls = decompose_mesh(_cylinder_mesh())
        for h in hulls:
            assert h.vertices.shape[0] >= 4

    def test_auto_support_point_works(self):
        """ConvexHullShape.support_point must work after decomposition."""
        hulls = decompose_mesh(_box_mesh())
        d = np.array([0.0, 0.0, 1.0])
        for h in hulls:
            sp = h.support_point(d)
            assert sp.shape == (3,)


# ---------------------------------------------------------------------------
# missing optional backend raises ImportError
# ---------------------------------------------------------------------------


class TestMissingBackend:
    def test_coacd_missing_raises(self):
        import sys

        if "coacd" in sys.modules:
            pytest.skip("coacd is installed")
        with pytest.raises(ImportError, match="coacd"):
            decompose_mesh(_box_mesh(), backend=DecompBackend.COACD)

    def test_vhacd_missing_raises(self):
        import sys

        if "vhacdx" in sys.modules:
            pytest.skip("vhacdx is installed")
        with pytest.raises(ImportError, match="vhacdx"):
            decompose_mesh(_box_mesh(), backend=DecompBackend.VHACD)
