"""Integration test: physics simulation → RenderScene → matplotlib PNG."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np

from physics.cpu_engine import CpuEngine
from physics.geometry import BodyCollisionGeometry, BoxShape, ShapeInstance, SphereShape
from physics.joint import FreeJoint
from physics.merged_model import merge_models
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from physics.terrain import FlatTerrain
from rendering import RobotViewer, build_render_scene
from robot.model import RobotModel


def _box_model():
    """Single floating box body."""
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


def _sphere_model():
    """Single floating sphere body."""
    tree = RobotTreeNumpy(gravity=9.81)
    tree.add_body(
        Body(
            "ball",
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
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(shape=SphereShape(0.08))])],
        contact_body_names=["ball"],
    )


class TestRenderSceneIntegration:
    def test_box_render_to_png(self, tmp_path):
        """Box near ground → build RenderScene → render → PNG exists."""
        model = _box_model()
        terrain = FlatTerrain()
        merged = merge_models({"r": model}, terrain=terrain)
        engine = CpuEngine(merged, dt=1e-4)

        q = np.zeros(merged.nq)
        q[0] = 1.0
        q[6] = 0.04  # box half-z = 0.05, so penetrating ground slightly
        engine.step(q, np.zeros(merged.nv), np.zeros(merged.nv))

        X = merged.tree.forward_kinematics(q)
        contacts = engine.query_contacts()
        scene = build_render_scene(merged, X, contacts=contacts, terrain=terrain)

        viewer = RobotViewer(merged.tree)
        out_path = str(tmp_path / "box_render.png")
        viewer.render_pose(render_scene=scene, show=False, save_path=out_path)

        import os

        assert os.path.isfile(out_path)
        assert os.path.getsize(out_path) > 1000  # non-trivial PNG

    def test_mixed_shapes_render(self, tmp_path):
        """Box + sphere models merged → render both shapes."""
        box_model = _box_model()
        sphere_model = _sphere_model()
        terrain = FlatTerrain()
        merged = merge_models({"box": box_model, "ball": sphere_model}, terrain=terrain)

        q = np.zeros(merged.nq)
        q[0] = 1.0  # box qw
        q[6] = 0.3  # box z
        q[7] = 1.0  # sphere qw
        q[13] = 0.3  # sphere z
        q[11] = 0.3  # sphere x offset

        X = merged.tree.forward_kinematics(q)
        scene = build_render_scene(merged, X, terrain=terrain)

        assert len(scene.shapes) == 2
        types = {s.shape_type for s in scene.shapes}
        assert types == {"box", "sphere"}

        viewer = RobotViewer(merged.tree)
        out_path = str(tmp_path / "mixed_shapes.png")
        viewer.render_pose(render_scene=scene, show=False, save_path=out_path)

        import os

        assert os.path.isfile(out_path)

    def test_two_leg_collision_render(self, tmp_path):
        """Two-leg collision scenario → render with contacts visible."""
        from tests.integration.test_mesh_cpu_pipeline import _build_go2_leg

        leg_a = _build_go2_leg("L")
        leg_b = _build_go2_leg("R")
        terrain = FlatTerrain()
        merged = merge_models({"a": leg_a, "b": leg_b}, terrain=terrain)
        engine = CpuEngine(merged, dt=2e-4)

        q = np.zeros(merged.nq)
        qdot = np.zeros(merged.nv)
        nq_a = leg_a.tree.nq

        q[0] = 1.0
        q[4] = -0.05
        q[6] = 0.5
        q[nq_a] = 1.0
        q[nq_a + 4] = 0.05
        q[nq_a + 6] = 0.5

        # Place them close enough to collide immediately
        engine.step(q, qdot, np.zeros(merged.nv))
        X = merged.tree.forward_kinematics(q)
        contacts = engine.query_contacts()
        scene = build_render_scene(merged, X, contacts=contacts, terrain=terrain)

        assert len(scene.shapes) == 6  # 3 links × 2 legs

        viewer = RobotViewer(merged.tree, floor_size=0.5)
        out_path = str(tmp_path / "two_legs.png")
        viewer.render_pose(render_scene=scene, show=False, save_path=out_path)

        import os

        assert os.path.isfile(out_path)
