from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from physics.cpu_engine import CpuEngine
from physics.geometry import BodyCollisionGeometry, ShapeInstance, SphereShape
from physics.joint import FreeJoint
from physics.merged_model import merge_models
from physics.publish import GpuPublishedFrame
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from physics.terrain import FlatTerrain
from robot.model import RobotModel
from sensing import build_state_sample_view


def _single_body_model():
    tree = RobotTreeNumpy(gravity=9.81)
    tree.add_body(
        Body(
            name="body",
            index=0,
            joint=FreeJoint("root"),
            inertia=SpatialInertia(1.0, np.eye(3) * 0.001, np.zeros(3)),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()
    return RobotModel(
        tree=tree,
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(shape=SphereShape(0.05))])],
        contact_body_names=["body"],
    )


class _ArrayWrapper:
    def __init__(self, array):
        self._array = np.asarray(array)

    def numpy(self):
        return self._array


class TestStateSampleView:
    def test_cpu_frame_builds_state_sample_view(self):
        merged = merge_models({"r": _single_body_model()}, terrain=FlatTerrain())
        engine = CpuEngine(merged, dt=1e-3)
        q, qdot = merged.tree.default_state()
        tau = np.zeros(merged.nv)
        engine.step(q=q, qdot=qdot, tau=tau, dt=1e-3)

        view = build_state_sample_view(engine)

        assert view.frame_id == 0
        assert view.q is not None and view.q.shape == q.shape
        assert view.qdot is not None and view.qdot.shape == qdot.shape
        assert view.X_world is not None
        assert view.v_bodies is not None
        assert view.telemetry is not None
        assert view.contact_count is not None

    def test_gpu_frame_builds_state_sample_view(self):
        engine = MagicMock()
        engine.nc_sensor = 1
        frame = GpuPublishedFrame(
            slot_id=0,
            frame_id=4,
            sim_time=0.004,
            step_index=4,
            env_mask_wp=None,
            q_wp=_ArrayWrapper(np.array([[1.0, 0.0, 0.0]], dtype=np.float32)),
            qdot_wp=_ArrayWrapper(np.array([[0.1, 0.2]], dtype=np.float32)),
            x_world_R_wp=_ArrayWrapper(np.eye(3, dtype=np.float32).reshape(1, 1, 3, 3)),
            x_world_r_wp=_ArrayWrapper(np.array([[[0.0, 0.0, 0.3]]], dtype=np.float32)),
            v_bodies_wp=_ArrayWrapper(np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]], dtype=np.float32)),
            contact_count_wp=_ArrayWrapper(np.array([2], dtype=np.int32)),
            contact_cache_ref=None,
            telemetry_ref={
                "qacc_smooth_wp": _ArrayWrapper(np.array([[7.0, 8.0]], dtype=np.float32)),
                "qacc_total_wp": _ArrayWrapper(np.array([[9.0, 10.0]], dtype=np.float32)),
                "force_sensor_wp": _ArrayWrapper(np.array([[1.0, 2.0, 3.0]], dtype=np.float32)),
            },
        )

        view = build_state_sample_view(engine, frame=frame, env_idx=0)

        assert view.frame_id == 4
        np.testing.assert_allclose(view.q, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(view.qdot, [0.1, 0.2])
        np.testing.assert_allclose(view.v_bodies, [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        assert len(view.X_world) == 1
        assert view.contact_count == 2
        assert view.telemetry is not None
        np.testing.assert_allclose(view.telemetry.qacc_total, [9.0, 10.0])
