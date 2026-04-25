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
from physics.telemetry import build_telemetry_snapshot_from_published_frame
from physics.terrain import FlatTerrain
from robot.model import RobotModel


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


class TestTelemetrySnapshot:
    def test_cpu_frame_maps_force_state_to_snapshot(self):
        merged = merge_models({"r": _single_body_model()}, terrain=FlatTerrain())
        engine = CpuEngine(merged, dt=1e-3)
        q, qdot = merged.tree.default_state()
        tau = np.zeros(merged.nv)
        engine.step(q=q, qdot=qdot, tau=tau, dt=1e-3)

        snapshot = build_telemetry_snapshot_from_published_frame(engine)

        assert snapshot.frame_id == 0
        assert snapshot.qfrc_passive is not None
        assert snapshot.qfrc_actuator is not None
        assert snapshot.qfrc_applied is not None
        assert snapshot.qacc_smooth is not None
        assert snapshot.qacc_total is not None
        assert snapshot.qfrc_actuator.shape == (merged.nv,)
        assert snapshot.force_sensor is None

    def test_gpu_frame_maps_published_telemetry_buffers(self):
        engine = MagicMock()
        engine.nc_sensor = 2
        frame = GpuPublishedFrame(
            slot_id=0,
            frame_id=5,
            sim_time=0.005,
            step_index=5,
            env_mask_wp=None,
            q_wp=None,
            qdot_wp=None,
            x_world_R_wp=None,
            x_world_r_wp=None,
            v_bodies_wp=None,
            contact_count_wp=None,
            contact_cache_ref=None,
            telemetry_ref={
                "qacc_smooth_wp": _ArrayWrapper(np.array([[1.0, 2.0]], dtype=np.float32)),
                "qacc_total_wp": _ArrayWrapper(np.array([[3.0, 4.0]], dtype=np.float32)),
                "force_sensor_wp": _ArrayWrapper(
                    np.array([[10.0, 11.0, 12.0, 20.0, 21.0, 22.0]], dtype=np.float32)
                ),
            },
        )

        snapshot = build_telemetry_snapshot_from_published_frame(engine, frame=frame, env_idx=0)

        assert snapshot.frame_id == 5
        np.testing.assert_allclose(snapshot.qacc_smooth, [1.0, 2.0])
        np.testing.assert_allclose(snapshot.qacc_total, [3.0, 4.0])
        np.testing.assert_allclose(snapshot.force_sensor, [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])

    def test_gpu_force_sensor_stays_flat_when_sensor_count_is_unknown(self):
        engine = MagicMock()
        frame = GpuPublishedFrame(
            slot_id=0,
            frame_id=6,
            sim_time=0.006,
            step_index=6,
            env_mask_wp=None,
            q_wp=None,
            qdot_wp=None,
            x_world_R_wp=None,
            x_world_r_wp=None,
            v_bodies_wp=None,
            contact_count_wp=None,
            contact_cache_ref=None,
            telemetry_ref={
                "qacc_smooth_wp": _ArrayWrapper(np.array([[1.0]], dtype=np.float32)),
                "qacc_total_wp": _ArrayWrapper(np.array([[2.0]], dtype=np.float32)),
                "force_sensor_wp": _ArrayWrapper(
                    np.array([[10.0, 11.0, 12.0, 20.0, 21.0, 22.0]], dtype=np.float32)
                ),
            },
        )

        snapshot = build_telemetry_snapshot_from_published_frame(engine, frame=frame, env_idx=0)

        assert snapshot.frame_id == 6
        np.testing.assert_allclose(snapshot.force_sensor, [10.0, 11.0, 12.0, 20.0, 21.0, 22.0])
