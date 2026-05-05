from __future__ import annotations

import numpy as np
import pytest

from optics import (
    CpuReferenceOpticalExecutor,
    GpuBruteForceOpticalExecutor,
    OpticalFrameInputs,
    OpticalInstanceSpec,
    OpticalMaterialSpec,
    OpticalSceneCache,
    OpticalWorldRegistry,
    stage_optical_compute_result_to_host,
)
from physics.publish import CpuPublishedFrame
from sensing import (
    OpticalPinholeCameraSpec,
    OpticalRaySensorSpec,
    build_pinhole_camera_image_result,
    build_pinhole_camera_rays,
)

try:
    import warp as wp  # noqa: F401

    HAS_WARP = True
except Exception:
    HAS_WARP = False

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not HAS_WARP, reason="Warp or CUDA not available"),
]


def _frame(*, frame_id: int = 61, sim_time: float = 0.61) -> CpuPublishedFrame:
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


def _ray_spec(snapshot, *, origins=None, directions=None, sensor_role: str = "depth") -> OpticalRaySensorSpec:
    return OpticalRaySensorSpec(
        frame_id=snapshot.frame_id,
        sim_time=snapshot.sim_time,
        env_idx=snapshot.env_idx,
        sensor_id="gpu_probe",
        origins_world=[[0.0, 0.0, 2.0]] if origins is None else origins,
        directions_world=[[0.0, 0.0, -1.0]] if directions is None else directions,
        max_distance=10.0,
        sensor_role=sensor_role,
    )


def _large_triangle(z: float) -> list[list[float]]:
    return [
        [-10.0, -10.0, z],
        [10.0, -10.0, z],
        [0.0, 10.0, z],
    ]


def _assert_gpu_matches_cpu(snapshot, spec) -> None:
    cpu = CpuReferenceOpticalExecutor().execute(snapshot, spec)
    device = GpuBruteForceOpticalExecutor(device="cuda:0").execute(snapshot, spec)
    assert device.location == "device"
    staged = stage_optical_compute_result_to_host(device)

    assert set(staged.channels) == GpuBruteForceOpticalExecutor.capabilities
    np.testing.assert_array_equal(staged.channel("hit_mask"), cpu.channel("hit_mask"))
    np.testing.assert_allclose(staged.channel("range_m"), cpu.channel("range_m"), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        staged.channel("position_world"),
        cpu.channel("position_world"),
        rtol=1e-5,
        atol=1e-5,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        staged.channel("normal_world"),
        cpu.channel("normal_world"),
        rtol=1e-5,
        atol=1e-5,
        equal_nan=True,
    )
    np.testing.assert_array_equal(staged.channel("numeric_instance_id"), cpu.channel("numeric_instance_id"))


def test_gpu_bruteforce_matches_cpu_for_plane_and_miss():
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat_floor"))
    registry.add_plane_geometry("floor", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 0.0])
    registry.add_instance(OpticalInstanceSpec("floor", "floor", "mat_floor"))
    snapshot = _snapshot(registry)
    spec = _ray_spec(
        snapshot,
        origins=[[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]],
        directions=[[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
    )

    _assert_gpu_matches_cpu(snapshot, spec)


def test_gpu_bruteforce_matches_cpu_for_triangle_mesh():
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat_tri"))
    registry.add_triangle_mesh_geometry("tri", vertices_local=_large_triangle(0.0), triangles=[[0, 1, 2]])
    registry.add_instance(OpticalInstanceSpec("tri_instance", "tri", "mat_tri"))
    snapshot = _snapshot(registry)
    spec = _ray_spec(
        snapshot,
        origins=[[0.0, 0.0, 2.0], [20.0, 20.0, 2.0]],
        directions=[[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
    )

    _assert_gpu_matches_cpu(snapshot, spec)


def test_gpu_bruteforce_source_order_tie_break_matches_cpu():
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat_plane"))
    registry.add_material(OpticalMaterialSpec("mat_tri"))
    registry.add_plane_geometry("plane", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 0.0])
    registry.add_triangle_mesh_geometry("tri", vertices_local=_large_triangle(0.0), triangles=[[0, 1, 2]])
    registry.add_instance(OpticalInstanceSpec("plane_instance", "plane", "mat_plane"))
    registry.add_instance(OpticalInstanceSpec("tri_instance", "tri", "mat_tri"))
    snapshot = _snapshot(registry)
    spec = _ray_spec(snapshot)

    _assert_gpu_matches_cpu(snapshot, spec)


def test_gpu_bruteforce_role_filtering_matches_cpu():
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat_debug"))
    registry.add_plane_geometry("debug_plane", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 0.0])
    registry.add_instance(
        OpticalInstanceSpec("debug_only", "debug_plane", "mat_debug", roles=frozenset({"debug"}))
    )
    snapshot = _snapshot(registry)
    spec = _ray_spec(snapshot, sensor_role="depth")

    _assert_gpu_matches_cpu(snapshot, spec)


def test_gpu_bruteforce_staged_result_supports_camera_postprocess():
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat_wall"))
    registry.add_triangle_mesh_geometry("wall", vertices_local=_large_triangle(2.0), triangles=[[0, 1, 2]])
    registry.add_instance(OpticalInstanceSpec("wall_instance", "wall", "mat_wall"))
    snapshot = _snapshot(registry)
    camera = OpticalPinholeCameraSpec(
        frame_id=snapshot.frame_id,
        sim_time=snapshot.sim_time,
        env_idx=snapshot.env_idx,
        sensor_id="cam",
        width=1,
        height=1,
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
        max_distance=10.0,
    )
    rays = build_pinhole_camera_rays(camera)
    device = GpuBruteForceOpticalExecutor(device="cuda:0").execute(snapshot, rays)
    staged = stage_optical_compute_result_to_host(device)
    image = build_pinhole_camera_image_result(staged, camera, rays=rays)

    np.testing.assert_array_equal(image.channel("hit_mask"), [[True]])
    np.testing.assert_allclose(image.channel("range_m"), [[2.0]], atol=1e-5)
    np.testing.assert_allclose(image.channel("depth_m"), [[2.0]], atol=1e-5)
