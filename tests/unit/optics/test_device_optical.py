from __future__ import annotations

import numpy as np
import pytest

from optics import (
    MAX_PRIMITIVES_PER_INSTANCE,
    OpticalComputeResult,
    OpticalFrameInputs,
    OpticalInstanceSpec,
    OpticalMaterialSpec,
    OpticalSceneCache,
    OpticalWorldRegistry,
    build_host_optical_primitive_workload,
    pack_source_order_key,
    stage_optical_channels,
    stage_optical_compute_result_to_host,
)
from physics.publish import CpuPublishedFrame


class _ArrayLike:
    def __init__(self, value):
        self._value = np.asarray(value)

    def numpy(self):
        return self._value


def _frame(*, frame_id: int = 51, sim_time: float = 0.51) -> CpuPublishedFrame:
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


def _snapshot(registry: OpticalWorldRegistry):
    return OpticalSceneCache(registry).snapshot_from_frame_inputs(
        OpticalFrameInputs.from_published_frame(_frame())
    )


def test_pack_source_order_key_preserves_lexicographic_order():
    assert pack_source_order_key(0, 9) < pack_source_order_key(1, 0)
    assert pack_source_order_key(2, 3) < pack_source_order_key(2, 4)


def test_pack_source_order_key_rejects_out_of_range_values():
    with pytest.raises(ValueError, match="instance_index"):
        pack_source_order_key(-1, 0)
    with pytest.raises(ValueError, match="primitive_index"):
        pack_source_order_key(0, -1)
    with pytest.raises(ValueError, match="MAX_PRIMITIVES"):
        pack_source_order_key(0, MAX_PRIMITIVES_PER_INSTANCE)


def test_build_host_optical_primitive_workload_filters_roles_and_packs_keys():
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat"))
    registry.add_plane_geometry("plane", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 0.0])
    registry.add_triangle_mesh_geometry(
        "tri",
        vertices_local=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        triangles=[[0, 1, 2], [3, 4, 5]],
    )
    registry.add_instance(OpticalInstanceSpec("plane_depth", "plane", "mat", roles=frozenset({"depth"})))
    registry.add_instance(OpticalInstanceSpec("tri_rgb", "tri", "mat", roles=frozenset({"rgb"})))
    registry.add_instance(OpticalInstanceSpec("tri_depth", "tri", "mat", roles=frozenset({"depth"})))
    snapshot = _snapshot(registry)

    workload = build_host_optical_primitive_workload(snapshot, sensor_role="depth")

    assert workload.plane_normal_world.shape == (1, 3)
    assert workload.triangles_world.shape == (1, 3, 3)
    np.testing.assert_array_equal(workload.plane_numeric_instance_id, [1])
    np.testing.assert_array_equal(workload.triangle_numeric_instance_id, [3])
    np.testing.assert_array_equal(workload.plane_source_order_key, [pack_source_order_key(0, 0)])
    np.testing.assert_array_equal(workload.triangle_source_order_key, [pack_source_order_key(2, 0)])
    assert workload.triangles_world.dtype == np.float32
    assert workload.triangle_source_order_key.dtype == np.int64


def test_stage_optical_compute_result_to_host_normalizes_device_channels():
    device_result = OpticalComputeResult(
        frame_id=7,
        sim_time=0.07,
        env_idx=2,
        sensor_id="gpu_probe",
        location="device",
        channels={
            "hit_mask": _ArrayLike(np.array([1, 0], dtype=np.int32)),
            "range_m": _ArrayLike(np.array([1.5, np.inf], dtype=np.float32)),
            "position_world": _ArrayLike(
                np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]], dtype=np.float32)
            ),
            "normal_world": _ArrayLike(
                np.array([[0.0, 0.0, 1.0], [np.nan, np.nan, np.nan]], dtype=np.float32)
            ),
            "numeric_instance_id": _ArrayLike(np.array([4, 0], dtype=np.int32)),
        },
    )

    host = stage_optical_compute_result_to_host(device_result)

    assert host.location == "host"
    assert host.ready_event is None
    assert host.frame_id == 7
    np.testing.assert_array_equal(host.channel("hit_mask"), [True, False])
    assert host.channel("hit_mask").dtype == bool
    assert host.channel("range_m").dtype == np.float64
    assert host.channel("position_world").dtype == np.float64
    assert host.channel("normal_world").dtype == np.float64
    assert host.channel("numeric_instance_id").dtype == np.int64
    np.testing.assert_allclose(host.channel("range_m"), [1.5, np.inf])


def test_stage_optical_channels_stages_selected_channels_only():
    device_result = OpticalComputeResult(
        frame_id=7,
        sim_time=0.07,
        env_idx=2,
        sensor_id="gpu_probe",
        location="device",
        channels={
            "hit_mask": _ArrayLike(np.array([1, 0], dtype=np.int32)),
            "rgb": _ArrayLike(np.array([[0.1, 0.2, 0.3]], dtype=np.float32)),
            "numeric_instance_id": _ArrayLike(np.array([4, 0], dtype=np.int32)),
        },
    )

    staged = stage_optical_channels(device_result, ("rgb", "numeric_instance_id"))

    assert set(staged) == {"rgb", "numeric_instance_id"}
    assert staged["rgb"].dtype == np.float64
    assert staged["numeric_instance_id"].dtype == np.int64
    np.testing.assert_allclose(staged["rgb"], [[0.1, 0.2, 0.3]], rtol=1e-6)
    np.testing.assert_array_equal(staged["numeric_instance_id"], [4, 0])


def test_stage_optical_compute_result_to_host_rejects_host_result():
    result = OpticalComputeResult(frame_id=1, sim_time=0.0, env_idx=0, sensor_id="probe")

    with pytest.raises(ValueError, match="device result"):
        stage_optical_compute_result_to_host(result)


def test_stage_optical_channels_rejects_host_result():
    result = OpticalComputeResult(frame_id=1, sim_time=0.0, env_idx=0, sensor_id="probe")

    with pytest.raises(ValueError, match="device result"):
        stage_optical_channels(result, ("rgb",))
