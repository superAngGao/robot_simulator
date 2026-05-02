from __future__ import annotations

import numpy as np

from physics.cpu_engine import CpuEngine
from physics.geometry import BodyCollisionGeometry, ShapeInstance, SphereShape
from physics.joint import FreeJoint
from physics.merged_model import merge_models
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from physics.telemetry import TelemetrySnapshot
from physics.terrain import FlatTerrain
from robot.model import RobotModel
from sensing import (
    ContactStateReading,
    ForceSensorReading,
    IMUReading,
    JointStateReading,
    OpticalCameraReading,
    RangeSensorReading,
    StateSampleView,
    build_contact_state_reading,
    build_force_sensor_reading,
    build_imu_reading,
    build_joint_state_reading,
    build_range_sensor_reading,
    build_state_sample_view,
)
from sensing.surface_query import SurfaceQueryResult


def _make_view() -> StateSampleView:
    return StateSampleView(
        frame_id=3,
        step_index=3,
        sim_time=0.003,
        env_idx=0,
        q=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        qdot=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        X_world=[
            SpatialTransform.from_rpy(0.0, 0.0, 0.25, r=np.array([1.0, 2.0, 3.0])),
        ],
        v_bodies=np.array([[1.0, 2.0, 3.0, 0.4, 0.5, 0.6]], dtype=np.float64),
        contact_count=2,
        contact_mask=np.array([1, 0], dtype=np.int32),
        telemetry=TelemetrySnapshot(
            frame_id=3,
            step_index=3,
            sim_time=0.003,
            env_idx=0,
            qfrc_applied=np.array([4.0, 5.0, 6.0], dtype=np.float64),
            tau_smooth=np.array([14.0, 15.0, 16.0], dtype=np.float64),
            force_sensor=np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float64),
        ),
    )


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


class TestJointStateReading:
    def test_reading_classes_are_exported_from_package(self):
        assert JointStateReading is not None
        assert IMUReading is not None
        assert ForceSensorReading is not None
        assert ContactStateReading is not None
        assert OpticalCameraReading is not None
        assert RangeSensorReading is not None

    def test_builds_full_joint_state_from_view(self):
        view = _make_view()

        reading = build_joint_state_reading(view)

        assert reading.frame_id == 3
        np.testing.assert_allclose(reading.joint_pos, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(reading.joint_vel, [0.1, 0.2, 0.3])

    def test_supports_joint_subset(self):
        view = _make_view()

        reading = build_joint_state_reading(view, joint_indices=[0, 2])

        np.testing.assert_allclose(reading.joint_pos, [1.0, 3.0])
        np.testing.assert_allclose(reading.joint_vel, [0.1, 0.3])


class TestIMUReading:
    def test_builds_orientation_and_angular_velocity_from_view(self):
        view = _make_view()

        reading = build_imu_reading(view, body_index=0)

        assert reading.frame_id == 3
        assert reading.body_index == 0
        np.testing.assert_allclose(reading.orientation_world_R, view.X_world[0].R)
        np.testing.assert_allclose(reading.angular_velocity_body, [0.4, 0.5, 0.6])
        assert reading.linear_acceleration_body is None

    def test_missing_state_fields_remain_none(self):
        view = _make_view()
        view.X_world = None
        view.v_bodies = None

        reading = build_imu_reading(view, body_index=0)

        assert reading.orientation_world_R is None
        assert reading.angular_velocity_body is None
        assert reading.linear_acceleration_body is None


class TestForceSensorReading:
    def test_builds_force_reading_from_telemetry(self):
        view = _make_view()

        reading = build_force_sensor_reading(view)

        assert reading.frame_id == 3
        np.testing.assert_allclose(reading.qfrc_applied, [4.0, 5.0, 6.0])
        np.testing.assert_allclose(reading.tau_smooth, [14.0, 15.0, 16.0])
        np.testing.assert_allclose(reading.contact_force, [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        assert reading.body_force is None

    def test_supports_force_sensor_subset(self):
        view = _make_view()

        reading = build_force_sensor_reading(view, sensor_indices=[1])

        np.testing.assert_allclose(reading.contact_force, [[10.0, 11.0, 12.0]])

    def test_missing_telemetry_fields_remain_none(self):
        view = _make_view()
        view.telemetry = None

        reading = build_force_sensor_reading(view)

        assert reading.qfrc_applied is None
        assert reading.tau_smooth is None
        assert reading.body_force is None
        assert reading.contact_force is None

    def test_cpu_engine_force_sensor_contact_force_is_none(self):
        merged = merge_models({"r": _single_body_model()}, terrain=FlatTerrain())
        engine = CpuEngine(merged, dt=1e-3)
        q, qdot = merged.tree.default_state()
        tau = np.zeros(merged.nv)
        engine.step(q=q, qdot=qdot, tau=tau, dt=1e-3)

        view = build_state_sample_view(engine)
        reading = build_force_sensor_reading(view)

        assert reading.qfrc_applied is not None
        assert reading.tau_smooth is not None
        assert reading.contact_force is None


class TestContactStateReading:
    def test_phase1_exposes_contact_count(self):
        view = _make_view()

        reading = build_contact_state_reading(view)

        assert reading.frame_id == 3
        assert reading.contact_count == 2
        np.testing.assert_allclose(reading.contact_mask, [1, 0])


class TestRangeSensorReading:
    def test_builds_range_reading_from_surface_query_result(self):
        result = SurfaceQueryResult(
            frame_id=8,
            sim_time=0.008,
            env_idx=2,
            hit_mask=np.array([True, False], dtype=bool),
            distance=np.array([1.5, np.inf], dtype=np.float32),
            position_world=np.array([[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan]], dtype=np.float64),
            normal_world=np.array([[0.0, 0.0, 1.0], [np.nan, np.nan, np.nan]], dtype=np.float64),
        )

        reading = build_range_sensor_reading(result)

        assert reading.frame_id == 8
        assert reading.sim_time == 0.008
        assert reading.env_idx == 2
        np.testing.assert_array_equal(reading.hit_mask, [True, False])
        np.testing.assert_allclose(reading.range_m, [1.5, np.inf])
        assert reading.range_m.dtype == np.float64
        np.testing.assert_allclose(reading.hit_position_world[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(reading.hit_normal_world[0], [0.0, 0.0, 1.0])

    def test_can_omit_hit_payloads(self):
        result = SurfaceQueryResult(
            frame_id=8,
            sim_time=0.008,
            env_idx=2,
            hit_mask=np.array([True], dtype=bool),
            distance=np.array([1.5], dtype=np.float64),
            position_world=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            normal_world=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        )

        reading = build_range_sensor_reading(result, include_hits=False)

        np.testing.assert_allclose(reading.range_m, [1.5])
        np.testing.assert_array_equal(reading.hit_mask, [True])
        assert reading.hit_position_world is None
        assert reading.hit_normal_world is None

    def test_preserves_nan_hit_payloads_for_all_miss_result(self):
        result = SurfaceQueryResult(
            frame_id=9,
            sim_time=0.009,
            env_idx=0,
            hit_mask=np.array([False, False], dtype=bool),
            distance=np.array([np.inf, np.inf], dtype=np.float64),
            position_world=np.full((2, 3), np.nan, dtype=np.float64),
            normal_world=np.full((2, 3), np.nan, dtype=np.float64),
        )

        reading = build_range_sensor_reading(result)

        np.testing.assert_array_equal(reading.hit_mask, [False, False])
        assert np.all(np.isinf(reading.range_m))
        assert np.all(np.isnan(reading.hit_position_world))
        assert np.all(np.isnan(reading.hit_normal_world))
