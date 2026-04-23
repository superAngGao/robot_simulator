"""Tests for build_render_scene_from_gpu bridge."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from rendering.render_scene import RenderScene
from rendering.scene_builder import build_render_scene_from_gpu


def _make_mock_engine(n_envs: int = 2, nq: int = 7):
    """Build a minimal mock GpuPhysicsEngine."""
    from physics.geometry import BodyCollisionGeometry, BoxShape, ShapeInstance
    from physics.joint import FreeJoint
    from physics.merged_model import merge_models
    from physics.robot_tree import Body, RobotTreeNumpy
    from physics.spatial import SpatialInertia, SpatialTransform
    from physics.terrain import FlatTerrain
    from robot.model import RobotModel

    tree = RobotTreeNumpy(gravity=9.81)
    tree.add_body(
        Body(
            name="base",
            index=0,
            joint=FreeJoint("root"),
            inertia=SpatialInertia(1.0, np.eye(3) * 0.001, np.zeros(3)),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()
    model = RobotModel(
        tree=tree,
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(shape=BoxShape((0.1, 0.1, 0.1)))])],
        contact_body_names=["base"],
    )
    merged = merge_models({"r": model}, terrain=FlatTerrain())

    # Build q array: identity quaternion for each env
    q_data = np.zeros((n_envs, merged.nq), dtype=np.float32)
    q_data[:, 0] = 1.0  # qw = 1

    q_wp_mock = MagicMock()
    q_wp_mock.numpy.return_value = q_data

    engine = MagicMock()
    engine.merged = merged
    engine.q_wp = q_wp_mock
    engine.query_contacts.return_value = []
    return engine


class TestBuildRenderSceneFromGpu:
    def test_returns_render_scene(self):
        engine = _make_mock_engine()
        scene = build_render_scene_from_gpu(engine, env_idx=0)
        assert isinstance(scene, RenderScene)

    def test_shape_count_matches_merged(self):
        engine = _make_mock_engine()
        scene = build_render_scene_from_gpu(engine, env_idx=0)
        # One box shape on the single body
        assert len(scene.shapes) == 1
        assert scene.shapes[0].shape_type == "box"

    def test_env_idx_out_of_bounds_raises(self):
        engine = _make_mock_engine(n_envs=2)
        with pytest.raises(IndexError):
            build_render_scene_from_gpu(engine, env_idx=5)

    def test_include_contacts_false_gives_empty_contacts(self):
        engine = _make_mock_engine()
        scene = build_render_scene_from_gpu(engine, env_idx=0, include_contacts=False)
        assert scene.contacts == []
        engine.query_contacts.assert_not_called()
