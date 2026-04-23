"""Tests for RerunBackend.

Rerun is an optional dependency. All tests are skipped if it is not installed.
The .rrd file-output test requires a real rr.save() call; shape-dispatch tests
mock the rr module to avoid network connections.
"""

from __future__ import annotations

import importlib
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rendering.backends.rerun_backend import RerunBackend
from rendering.render_scene import ContactPoint, PositionedShape, RenderScene, TerrainInfo

rr_available = importlib.util.find_spec("rerun") is not None
pytestmark = pytest.mark.skipif(not rr_available, reason="rerun not installed")


def _make_scene(shape_types=None) -> RenderScene:
    shapes = []
    _params = {
        "box": {"size": (0.2, 0.2, 0.2)},
        "sphere": {"radius": 0.1},
        "capsule": {"radius": 0.05, "length": 0.2},
        "cylinder": {"radius": 0.05, "length": 0.2},
        "convex_hull": {
            "vertices": np.array(
                [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
                dtype=np.float64,
            ),
            "faces": np.array([[0, 2, 4], [0, 4, 3], [1, 2, 4]], dtype=np.int32),
        },
    }
    for i, st in enumerate(shape_types or []):
        shapes.append(
            PositionedShape(
                shape_type=st,
                params=_params.get(st, {}),
                position=np.zeros(3),
                rotation=np.eye(3),
                body_index=i,
                body_name=f"body_{i}",
            )
        )
    contacts = [
        ContactPoint(
            position=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            depth=0.01,
            body_i=0,
            body_j=-1,
        )
    ]
    return RenderScene(
        shapes=shapes,
        contacts=contacts,
        terrain=TerrainInfo(terrain_type="flat", params={"z": 0.0}),
        skeleton_links=[(np.zeros(3), np.array([0.1, 0.0, 0.0]))],
        body_positions=[np.zeros(3)],
        body_names=["body_0"],
    )


class TestRerunBackend:
    def test_set_output_saves_rrd(self, tmp_path):
        """open() with save_path -> .rrd file created."""
        out = str(tmp_path / "debug.rrd")
        b = RerunBackend(app_id="test_app", save_path=out)
        b.open()
        b.render_frame(_make_scene(["box"]), timestamp=0.0)
        b.close()
        assert os.path.exists(out)

    def test_all_supported_shapes_do_not_raise(self):
        """All supported shape types render without error (mocked rr)."""
        mock_rr = MagicMock()
        mock_rr.Quaternion = MagicMock(return_value=MagicMock())
        with patch.dict("sys.modules", {"rerun": mock_rr}):
            b = RerunBackend()
            b.open()
            scene = _make_scene(["box", "sphere", "capsule", "cylinder", "convex_hull"])
            b.render_frame(scene, timestamp=0.0)
            b.close()

    def test_convex_hull_uses_precomputed_faces(self):
        """convex_hull branch reads params['faces'] — does NOT recompute hull."""
        mock_rr = MagicMock()
        mock_rr.Quaternion = MagicMock(return_value=MagicMock())
        faces = np.array([[0, 2, 4], [0, 4, 3]], dtype=np.int32)
        verts = np.array(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
            dtype=np.float64,
        )
        shape = PositionedShape(
            shape_type="convex_hull",
            params={"vertices": verts, "faces": faces},
            position=np.zeros(3),
            rotation=np.eye(3),
            body_index=0,
            body_name="hull",
        )
        scene = RenderScene(
            shapes=[shape],
            contacts=[],
            terrain=TerrainInfo(terrain_type="flat", params={"z": 0.0}),
            skeleton_links=[],
            body_positions=[],
            body_names=[],
        )
        with patch.dict("sys.modules", {"rerun": mock_rr}):
            b = RerunBackend()
            b.open()
            b.render_frame(scene, timestamp=0.0)
        # Mesh3D must have been called with the pre-computed faces
        call_kwargs = mock_rr.Mesh3D.call_args
        assert call_kwargs is not None
        passed_faces = call_kwargs.kwargs.get("triangle_indices")
        assert passed_faces is not None
        assert passed_faces.shape == (2, 3)

    def test_contacts_render_as_arrows(self):
        """Contacts are logged as rr.Arrows3D."""
        mock_rr = MagicMock()
        mock_rr.Quaternion = MagicMock(return_value=MagicMock())
        with patch.dict("sys.modules", {"rerun": mock_rr}):
            b = RerunBackend()
            b.open()
            b.render_frame(_make_scene([]), timestamp=0.0)
        mock_rr.Arrows3D.assert_called_once()
