"""Tests for MatplotlibBackend."""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

from rendering.backends.matplotlib_backend import MatplotlibBackend
from rendering.render_scene import ContactPoint, PositionedShape, RenderScene, TerrainInfo


def _make_scene(shape_types=None) -> RenderScene:
    shapes = []
    if shape_types:
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
        for i, st in enumerate(shape_types):
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
    skeleton_links = [
        (np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.0, 0.0])),
    ]
    return RenderScene(
        shapes=shapes,
        contacts=contacts,
        terrain=TerrainInfo(terrain_type="flat", params={"z": 0.0}),
        skeleton_links=skeleton_links,
        body_positions=[np.zeros(3)],
        body_names=["body_0"],
    )


class TestMatplotlibBackend:
    def test_offscreen_render_does_not_raise(self):
        b = MatplotlibBackend()
        b.open()
        b.render_frame(_make_scene(), timestamp=0.0)
        b.close()

    def test_set_output_saves_gif(self, tmp_path):
        out = str(tmp_path / "out.gif")
        b = MatplotlibBackend()
        b.set_output(out)
        b.open()
        b.render_frame(_make_scene(), timestamp=0.0)
        b.render_frame(_make_scene(), timestamp=0.1)
        b.close()
        assert os.path.exists(out)

    def test_all_shape_types_render(self):
        b = MatplotlibBackend()
        b.open()
        scene = _make_scene(["box", "sphere", "capsule", "cylinder", "convex_hull"])
        b.render_frame(scene, timestamp=0.0)
        b.close()

    def test_skeleton_links_produce_line3d(self):
        from mpl_toolkits.mplot3d.art3d import Line3D

        b = MatplotlibBackend()
        b.open()
        b.render_frame(_make_scene(), timestamp=0.0)
        # After render_frame, the axes should contain at least one Line3D
        lines = [c for c in b._ax.get_children() if isinstance(c, Line3D)]
        assert len(lines) > 0
        b.close()

    def test_env_index_accepted(self):
        b = MatplotlibBackend()
        b.open()
        b.render_frame(_make_scene(), timestamp=0.0, env_index=3)
        b.close()
