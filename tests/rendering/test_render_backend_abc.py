"""Tests for RenderBackend ABC contract."""

from __future__ import annotations

import os

import numpy as np

from rendering.backends.base import RenderBackend
from rendering.backends.matplotlib_backend import MatplotlibBackend
from rendering.backends.rerun_backend import RerunBackend
from rendering.render_scene import RenderScene, TerrainInfo


def _empty_scene() -> RenderScene:
    return RenderScene(
        shapes=[],
        contacts=[],
        terrain=TerrainInfo(terrain_type="flat", params={"z": 0.0}),
        skeleton_links=[],
        body_positions=[],
        body_names=[],
    )


def _scene_with_mesh() -> RenderScene:
    from rendering.render_scene import PositionedShape

    mesh_shape = PositionedShape(
        shape_type="mesh",
        params={"vertices": None, "filename": "dummy.obj"},
        position=np.zeros(3),
        rotation=np.eye(3),
        body_index=0,
        body_name="body",
    )
    return RenderScene(
        shapes=[mesh_shape],
        contacts=[],
        terrain=TerrainInfo(terrain_type="flat", params={"z": 0.0}),
        skeleton_links=[],
        body_positions=[],
        body_names=[],
    )


class TestRenderBackendABC:
    def test_matplotlib_is_subclass(self):
        assert issubclass(MatplotlibBackend, RenderBackend)

    def test_rerun_is_subclass(self):
        assert issubclass(RerunBackend, RenderBackend)

    def test_matplotlib_supports_offscreen(self):
        assert MatplotlibBackend().supports_offscreen is True

    def test_rerun_supports_offscreen(self):
        assert RerunBackend().supports_offscreen is True

    def test_set_output_does_not_raise(self):
        b = MatplotlibBackend()
        b.set_output("/tmp/test_out.gif")  # no-op path, just must not raise

    def test_empty_scene_open_render_close(self, tmp_path):
        """open/render_frame/close on empty scene must not raise."""
        os.environ.setdefault("MPLBACKEND", "Agg")
        b = MatplotlibBackend()
        b.open()
        b.render_frame(_empty_scene(), timestamp=0.0)
        b.close()

    def test_mesh_shape_does_not_raise_matplotlib(self):
        """Unsupported 'mesh' shape type must be silently skipped."""
        os.environ.setdefault("MPLBACKEND", "Agg")
        b = MatplotlibBackend()
        b.open()
        b.render_frame(_scene_with_mesh(), timestamp=0.0)
        b.close()
