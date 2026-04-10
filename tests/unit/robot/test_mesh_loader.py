"""Tests for robot.mesh_loader — mesh file loading and convex hull conversion."""

from __future__ import annotations

import os

import numpy as np
import pytest

from physics.geometry import ConvexHullShape
from robot.mesh_loader import load_mesh, resolve_mesh_path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_has_trimesh = True
try:
    import trimesh
except ImportError:
    _has_trimesh = False

needs_trimesh = pytest.mark.skipif(not _has_trimesh, reason="trimesh not installed")


def _write_cube_stl(path: str, extents: tuple[float, float, float] = (0.1, 0.1, 0.1)):
    """Write a cube STL to *path* using trimesh."""
    mesh = trimesh.creation.box(extents=extents)
    mesh.export(path)


# ---------------------------------------------------------------------------
# resolve_mesh_path
# ---------------------------------------------------------------------------


class TestResolveMeshPath:
    def test_relative_path(self):
        result = resolve_mesh_path("meshes/cube.stl", "/home/user/robot")
        assert result == os.path.normpath("/home/user/robot/meshes/cube.stl")

    def test_absolute_path(self):
        result = resolve_mesh_path("/opt/meshes/cube.stl", "/home/user/robot")
        assert result == "/opt/meshes/cube.stl"

    def test_package_uri_raises(self):
        with pytest.raises(NotImplementedError, match="package://"):
            resolve_mesh_path("package://my_robot/meshes/cube.stl", "/any/dir")


# ---------------------------------------------------------------------------
# load_mesh
# ---------------------------------------------------------------------------


@needs_trimesh
class TestLoadMesh:
    def test_load_cube_stl(self, tmp_path):
        stl_path = str(tmp_path / "cube.stl")
        _write_cube_stl(stl_path, extents=(0.1, 0.1, 0.1))

        shape = load_mesh(stl_path)

        assert isinstance(shape, ConvexHullShape)
        assert shape.vertices.shape[1] == 3
        assert shape.vertices.shape[0] >= 4
        # Cube hull should have 8 vertices
        assert shape.vertices.shape[0] == 8
        # Half-extents should be ~0.05
        he = shape.half_extents_approx()
        np.testing.assert_allclose(he, [0.05, 0.05, 0.05], atol=1e-6)

    def test_load_with_scale(self, tmp_path):
        stl_path = str(tmp_path / "cube.stl")
        _write_cube_stl(stl_path, extents=(0.1, 0.1, 0.1))

        shape = load_mesh(stl_path, scale=(2.0, 1.0, 0.5))

        he = shape.half_extents_approx()
        np.testing.assert_allclose(he, [0.10, 0.05, 0.025], atol=1e-6)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_mesh("/nonexistent/path/mesh.stl")

    def test_support_point_works(self, tmp_path):
        """Loaded ConvexHullShape should support GJK queries."""
        stl_path = str(tmp_path / "cube.stl")
        _write_cube_stl(stl_path, extents=(0.2, 0.2, 0.2))

        shape = load_mesh(stl_path)
        sp = shape.support_point(np.array([1.0, 0.0, 0.0]))
        assert sp[0] == pytest.approx(0.1, abs=1e-6)

    def test_contact_vertices_works(self, tmp_path):
        """Loaded ConvexHullShape should return contact vertices."""
        stl_path = str(tmp_path / "cube.stl")
        _write_cube_stl(stl_path, extents=(0.1, 0.1, 0.1))

        shape = load_mesh(stl_path)
        cv = shape.contact_vertices()
        assert cv is not None
        assert cv.shape[0] == 8
