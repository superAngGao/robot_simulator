from __future__ import annotations

import numpy as np

from optics import (
    CpuReferenceOpticalExecutor,
    OpticalInstanceSpec,
    OpticalMaterialSpec,
    OpticalSceneCache,
    OpticalWorldRegistry,
)
from physics.publish import CpuPublishedFrame
from sensing import OpticalRaySensorSpec


def _frame(*, frame_id: int = 11, sim_time: float = 0.11) -> CpuPublishedFrame:
    return CpuPublishedFrame(
        frame_id=frame_id,
        sim_time=sim_time,
        step_index=frame_id,
        env_mask=None,
        q=np.zeros(0, dtype=np.float64),
        qdot=np.zeros(0, dtype=np.float64),
        X_world=[],
        v_bodies=[],
        contact_count=0,
        contacts=[],
        telemetry=None,
    )


def _snapshot_with_floor():
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat_floor"))
    registry.add_plane_geometry("floor", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 0.0])
    registry.add_instance(OpticalInstanceSpec("floor", "floor", "mat_floor"))
    return OpticalSceneCache(registry).snapshot_from_published_frame(_frame())


class TestCpuReferenceOpticalExecutorSchema:
    def test_capabilities_match_returned_channels(self):
        snapshot = _snapshot_with_floor()
        spec = OpticalRaySensorSpec(
            frame_id=snapshot.frame_id,
            sim_time=snapshot.sim_time,
            env_idx=snapshot.env_idx,
            sensor_id="schema_probe",
            origins_world=[[0.0, 0.0, 1.0]],
            directions_world=[[0.0, 0.0, -1.0]],
        )

        executor = CpuReferenceOpticalExecutor()
        result = executor.execute(snapshot, spec)

        assert set(result.channels) == executor.capabilities

    def test_channel_schema_and_miss_values_are_stable(self):
        snapshot = _snapshot_with_floor()
        spec = OpticalRaySensorSpec(
            frame_id=snapshot.frame_id,
            sim_time=snapshot.sim_time,
            env_idx=snapshot.env_idx,
            sensor_id="schema_probe",
            origins_world=[
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            directions_world=[
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
            ],
        )

        result = CpuReferenceOpticalExecutor().execute(snapshot, spec)

        assert result.location == "host"
        assert result.ready_event is None
        assert result.channel("range_m").shape == (2,)
        assert result.channel("range_m").dtype == np.float64
        assert result.channel("hit_mask").shape == (2,)
        assert result.channel("hit_mask").dtype == bool
        assert result.channel("position_world").shape == (2, 3)
        assert result.channel("position_world").dtype == np.float64
        assert result.channel("normal_world").shape == (2, 3)
        assert result.channel("normal_world").dtype == np.float64
        assert result.channel("material_id").shape == (2,)
        assert result.channel("material_id").dtype == object
        assert result.channel("instance_id").shape == (2,)
        assert result.channel("instance_id").dtype == object
        assert result.channel("numeric_instance_id").shape == (2,)
        assert result.channel("numeric_instance_id").dtype == np.int64

        np.testing.assert_array_equal(result.channel("hit_mask"), [True, False])
        assert np.isinf(result.channel("range_m")[1])
        assert np.all(np.isnan(result.channel("position_world")[1]))
        assert np.all(np.isnan(result.channel("normal_world")[1]))
        assert result.channel("material_id").tolist() == ["mat_floor", None]
        assert result.channel("instance_id").tolist() == ["floor", None]
        np.testing.assert_array_equal(result.channel("numeric_instance_id"), [1, 0])
