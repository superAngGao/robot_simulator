from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from physics.cpu_engine import CpuEngine
from physics.engine import ContactInfo
from physics.geometry import BodyCollisionGeometry, BoxShape, ShapeInstance, SphereShape
from physics.joint import FreeJoint
from physics.merged_model import merge_models
from physics.publish import GpuPublishedFrame
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from physics.terrain import FlatTerrain
from rendering import build_render_scene_from_published_frame
from robot.model import RobotModel


def _single_body_model(shape=SphereShape(0.05), body_name="body"):
    tree = RobotTreeNumpy(gravity=9.81)
    tree.add_body(
        Body(
            name=body_name,
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
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(shape=shape)])],
        contact_body_names=[body_name],
    )


class _ArrayWrapper:
    def __init__(self, array):
        self._array = np.asarray(array)

    def numpy(self):
        return self._array


class TestPublishedFrameBridge:
    def test_cpu_published_frame_builds_render_scene_without_recomputing_fk(self):
        merged = merge_models(
            {"r": _single_body_model(BoxShape((0.2, 0.2, 0.2)), body_name="box")}, terrain=FlatTerrain()
        )
        engine = CpuEngine(merged, dt=1e-3)
        q, qdot = merged.tree.default_state()
        q[0] = 1.0
        q[6] = 0.3
        tau = np.zeros(merged.nv)
        engine.step(q=q, qdot=qdot, tau=tau, dt=1e-3)

        scene = build_render_scene_from_published_frame(engine)

        assert len(scene.shapes) == 1
        assert scene.shapes[0].shape_type == "box"
        assert len(scene.body_names) == 1
        assert scene.body_names[0].endswith("box")
        assert scene.sensor_data is not None
        assert scene.sensor_data.frame_id == 0
        assert len(scene.sensor_data.imu_readings) == 1
        assert scene.sensor_data.joint_state is not None
        assert scene.sensor_data.force is not None
        assert scene.sensor_data.force.qfrc_applied is not None
        assert scene.sensor_data.contact is not None
        assert scene.sensor_data.contact.contact_count is not None

    def test_published_frame_sensor_data_can_be_disabled(self):
        merged = merge_models({"r": _single_body_model(body_name="ball")}, terrain=FlatTerrain())
        engine = CpuEngine(merged, dt=1e-3)
        q, qdot = merged.tree.default_state()
        tau = np.zeros(merged.nv)
        engine.step(q=q, qdot=qdot, tau=tau, dt=1e-3)

        scene = build_render_scene_from_published_frame(engine, include_sensor_data=False)

        assert scene.sensor_data is None

    def test_gpu_published_frame_builds_render_scene_from_slot_buffers(self):
        merged = merge_models(
            {"r": _single_body_model(BoxShape((0.1, 0.1, 0.1)), body_name="base")}, terrain=FlatTerrain()
        )
        engine = MagicMock()
        engine.merged = merged
        engine.nc_sensor = 1
        engine.query_contacts.return_value = []

        frame = GpuPublishedFrame(
            slot_id=0,
            frame_id=3,
            sim_time=0.003,
            step_index=3,
            env_mask_wp=None,
            q_wp=_ArrayWrapper(np.array([[1, 0, 0, 0, 0, 0, 0]], dtype=np.float32)),
            qdot_wp=_ArrayWrapper(np.zeros((1, merged.nv), dtype=np.float32)),
            x_world_R_wp=_ArrayWrapper(np.eye(3, dtype=np.float32).reshape(1, 1, 3, 3)),
            x_world_r_wp=_ArrayWrapper(np.array([[[0.0, 0.0, 0.4]]], dtype=np.float32)),
            v_bodies_wp=_ArrayWrapper(np.zeros((1, 1, 6), dtype=np.float32)),
            contact_count_wp=_ArrayWrapper(np.array([1], dtype=np.int32)),
            contact_cache_ref={
                "contact_bi_wp": _ArrayWrapper(np.array([[0]], dtype=np.int32)),
                "contact_bj_wp": _ArrayWrapper(np.array([[-1]], dtype=np.int32)),
                "contact_depth_wp": _ArrayWrapper(np.array([[0.01]], dtype=np.float32)),
                "contact_normal_wp": _ArrayWrapper(np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)),
                "contact_point_wp": _ArrayWrapper(np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)),
                "contact_active_wp": _ArrayWrapper(np.array([[1]], dtype=np.int32)),
            },
            telemetry_ref={
                "qacc_smooth_wp": _ArrayWrapper(np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=np.float32)),
                "qacc_total_wp": _ArrayWrapper(np.array([[1.1, 1.2, 1.3, 1.4, 1.5, 1.6]], dtype=np.float32)),
                "force_sensor_wp": _ArrayWrapper(np.array([[7.0, 8.0, 9.0]], dtype=np.float32)),
            },
        )
        engine.latest_published_frame.return_value = frame

        scene = build_render_scene_from_published_frame(engine, env_idx=0)

        assert len(scene.shapes) == 1
        assert scene.shapes[0].shape_type == "box"
        assert len(scene.contacts) == 1
        assert scene.contacts[0].body_j == -1
        assert scene.sensor_data is not None
        assert scene.sensor_data.frame_id == 3
        assert len(scene.sensor_data.imu_readings) == 1
        np.testing.assert_allclose(scene.sensor_data.force.contact_force, [[7.0, 8.0, 9.0]])
        assert scene.sensor_data.force.qfrc_applied is None
        assert scene.sensor_data.contact.contact_count == 1
        engine.query_contacts.assert_not_called()

    def test_gpu_published_frame_falls_back_to_engine_query_contacts_when_dense_block_missing(self):
        merged = merge_models({"r": _single_body_model(body_name="ball")}, terrain=FlatTerrain())
        engine = MagicMock()
        engine.merged = merged
        engine.nc_sensor = 0
        engine.query_contacts.return_value = [
            ContactInfo(
                body_i=0,
                body_j=-1,
                depth=0.02,
                normal=np.array([0.0, 0.0, 1.0]),
                point=np.array([0.0, 0.0, 0.0]),
            )
        ]
        frame = GpuPublishedFrame(
            slot_id=0,
            frame_id=4,
            sim_time=0.004,
            step_index=4,
            env_mask_wp=None,
            q_wp=_ArrayWrapper(np.array([[1, 0, 0, 0, 0, 0, 0]], dtype=np.float32)),
            qdot_wp=_ArrayWrapper(np.zeros((1, merged.nv), dtype=np.float32)),
            x_world_R_wp=_ArrayWrapper(np.eye(3, dtype=np.float32).reshape(1, 1, 3, 3)),
            x_world_r_wp=_ArrayWrapper(np.array([[[0.0, 0.0, 0.2]]], dtype=np.float32)),
            v_bodies_wp=_ArrayWrapper(np.zeros((1, 1, 6), dtype=np.float32)),
            contact_count_wp=_ArrayWrapper(np.array([1], dtype=np.int32)),
            contact_cache_ref=None,
            telemetry_ref=None,
        )

        scene = build_render_scene_from_published_frame(engine, frame=frame, env_idx=0)

        assert len(scene.contacts) == 1
        engine.query_contacts.assert_called_once_with(0)
