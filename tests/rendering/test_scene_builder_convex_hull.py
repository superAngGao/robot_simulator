"""Tests for ConvexHullShape fan triangulation in scene_builder."""

from __future__ import annotations

import numpy as np

from physics.geometry import ConvexHullShape
from rendering.scene_builder import _shape_to_type_params

# Octahedron vertices — 6 points, guaranteed convex hull with multiple faces
_OCTAHEDRON = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=np.float64,
)


class TestSceneBuilderConvexHull:
    def test_returns_faces_key(self):
        shape = ConvexHullShape(_OCTAHEDRON)
        shape_type, params = _shape_to_type_params(shape)
        assert shape_type == "convex_hull"
        assert "faces" in params

    def test_faces_shape_is_f_by_3_int(self):
        shape = ConvexHullShape(_OCTAHEDRON)
        _, params = _shape_to_type_params(shape)
        faces = params["faces"]
        vertices = params["vertices"]
        assert faces.ndim == 2
        assert faces.shape[1] == 3
        assert faces.dtype in (np.int32, np.int64)
        assert vertices.ndim == 2
        assert vertices.shape[1] == 3
        assert vertices.dtype == np.float64
