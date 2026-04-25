"""Integration test: published frame -> RenderScene -> backend."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import numpy as np

from physics.cpu_engine import CpuEngine
from physics.geometry import BodyCollisionGeometry, BoxShape, ShapeInstance
from physics.joint import FreeJoint
from physics.merged_model import merge_models
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from physics.terrain import FlatTerrain
from rendering import MatplotlibBackend, render_latest_published_frame
from robot.model import RobotModel


def _box_model():
    tree = RobotTreeNumpy(gravity=9.81)
    tree.add_body(
        Body(
            "box",
            0,
            FreeJoint("root"),
            SpatialInertia(1, np.eye(3) * 0.001, np.zeros(3)),
            SpatialTransform.identity(),
            -1,
        )
    )
    tree.finalize()
    return RobotModel(
        tree=tree,
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(shape=BoxShape((0.2, 0.15, 0.1)))])],
        contact_body_names=["box"],
    )


class TestPublishedFrameRenderBackendIntegration:
    def test_cpu_published_frame_renders_through_matplotlib_backend(self, tmp_path):
        merged = merge_models({"r": _box_model()}, terrain=FlatTerrain())
        engine = CpuEngine(merged, dt=1e-3)
        q = np.zeros(merged.nq)
        q[0] = 1.0
        q[6] = 0.15
        tau = np.zeros(merged.nv)
        engine.step(q=q, qdot=np.zeros(merged.nv), tau=tau, dt=1e-3)

        out = str(tmp_path / "published.gif")
        backend = MatplotlibBackend(save_path=out)
        backend.open()
        scene = render_latest_published_frame(engine, backend)
        backend.close()

        assert len(scene.shapes) == 1
        assert scene.shapes[0].shape_type == "box"
        assert len(scene.body_names) == 1
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
