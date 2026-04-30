from __future__ import annotations

import numpy as np
import pytest

from optics import (
    CpuReferenceOpticalExecutor,
    OpticalFrameInputs,
    OpticalInstanceSpec,
    OpticalMaterialSpec,
    OpticalSceneCache,
    OpticalWorldRegistry,
)
from physics.publish import CpuPublishedFrame
from physics.spatial import SpatialTransform, rot_y
from sensing import (
    OpticalPinholeCameraSpec,
    build_pinhole_camera_image_result,
    build_pinhole_camera_rays,
)


def _frame(*, frame_id: int = 9, sim_time: float = 0.09) -> CpuPublishedFrame:
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


def _flat_plane_snapshot():
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat_wall"))
    registry.add_plane_geometry(
        "wall",
        normal_local=[0.0, 0.0, -1.0],
        point_local=[0.0, 0.0, 2.0],
    )
    registry.add_instance(
        OpticalInstanceSpec(
            instance_id="wall_instance",
            geometry_id="wall",
            material_id="mat_wall",
            roles=frozenset({"depth", "segmentation"}),
        )
    )
    frame = _frame()
    return OpticalSceneCache(registry).snapshot_from_frame_inputs(
        OpticalFrameInputs.from_published_frame(frame)
    )


class TestOpticalPinholeCameraSpec:
    def test_builds_image_shaped_ray_batch(self):
        spec = OpticalPinholeCameraSpec(
            frame_id=9,
            sim_time=0.09,
            env_idx=0,
            sensor_id="cam",
            width=3,
            height=3,
            fx=1.0,
            fy=1.0,
            cx=1.0,
            cy=1.0,
        )

        rays = build_pinhole_camera_rays(spec)

        assert rays.num_rays == 9
        assert rays.ray_shape == (3, 3)
        np.testing.assert_allclose(rays.origins_world, np.zeros((9, 3)))
        np.testing.assert_allclose(rays.directions_world[4], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(
            rays.directions_world[0],
            np.array([-1.0, -1.0, 1.0]) / np.sqrt(3.0),
        )

    def test_applies_camera_world_rotation(self):
        spec = OpticalPinholeCameraSpec(
            frame_id=9,
            sim_time=0.09,
            env_idx=0,
            sensor_id="cam",
            width=1,
            height=1,
            fx=1.0,
            fy=1.0,
            cx=0.0,
            cy=0.0,
            X_world_camera=SpatialTransform(rot_y(np.pi / 2.0), np.array([1.0, 2.0, 3.0])),
        )

        rays = build_pinhole_camera_rays(spec)

        np.testing.assert_allclose(rays.origins_world, [[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(rays.directions_world, [[1.0, 0.0, 0.0]], atol=1e-12)
        np.testing.assert_allclose(spec.optical_axis_world, [1.0, 0.0, 0.0], atol=1e-12)

    def test_rejects_invalid_intrinsics(self):
        with pytest.raises(ValueError, match="fx"):
            OpticalPinholeCameraSpec(
                frame_id=9,
                sim_time=0.09,
                env_idx=0,
                sensor_id="cam",
                width=3,
                height=3,
                fx=0.0,
                fy=1.0,
                cx=1.0,
                cy=1.0,
            )


class TestOpticalPinholeCameraImageResult:
    def test_postprocesses_flat_executor_result_to_image_channels(self):
        snapshot = _flat_plane_snapshot()
        spec = OpticalPinholeCameraSpec(
            frame_id=9,
            sim_time=0.09,
            env_idx=0,
            sensor_id="cam",
            width=3,
            height=3,
            fx=1.0,
            fy=1.0,
            cx=1.0,
            cy=1.0,
            sensor_role="depth",
        )

        rays = build_pinhole_camera_rays(spec)
        flat = CpuReferenceOpticalExecutor().execute(snapshot, rays)
        image = build_pinhole_camera_image_result(flat, spec, rays=rays)

        assert image.image_shape == (3, 3)
        assert image.channel("range_m").shape == (3, 3)
        assert image.channel("position_world").shape == (3, 3, 3)
        assert image.channel("normal_world").shape == (3, 3, 3)
        np.testing.assert_array_equal(image.channel("hit_mask"), np.ones((3, 3), dtype=bool))
        np.testing.assert_allclose(image.channel("depth_m"), np.full((3, 3), 2.0))
        np.testing.assert_allclose(image.channel("range_m")[1, 1], 2.0)
        np.testing.assert_allclose(image.channel("range_m")[0, 0], 2.0 * np.sqrt(3.0))
        assert image.channel("material_id").shape == (3, 3)
        assert image.channel("instance_id")[1, 1] == "wall_instance"
        np.testing.assert_array_equal(image.channel("numeric_instance_id"), np.ones((3, 3), dtype=np.int64))

    def test_postprocessor_rejects_mismatched_sensor_id(self):
        snapshot = _flat_plane_snapshot()
        spec = OpticalPinholeCameraSpec(
            frame_id=9,
            sim_time=0.09,
            env_idx=0,
            sensor_id="cam",
            width=1,
            height=1,
            fx=1.0,
            fy=1.0,
            cx=0.0,
            cy=0.0,
        )
        rays = build_pinhole_camera_rays(spec)
        flat = CpuReferenceOpticalExecutor().execute(snapshot, rays)
        other = OpticalPinholeCameraSpec(
            frame_id=9,
            sim_time=0.09,
            env_idx=0,
            sensor_id="other_cam",
            width=1,
            height=1,
            fx=1.0,
            fy=1.0,
            cx=0.0,
            cy=0.0,
        )

        with pytest.raises(ValueError, match="sensor_id"):
            build_pinhole_camera_image_result(flat, other)

    def test_postprocessor_rejects_mismatched_rays(self):
        snapshot = _flat_plane_snapshot()
        spec = OpticalPinholeCameraSpec(
            frame_id=9,
            sim_time=0.09,
            env_idx=0,
            sensor_id="cam",
            width=1,
            height=1,
            fx=1.0,
            fy=1.0,
            cx=0.0,
            cy=0.0,
        )
        rays = build_pinhole_camera_rays(spec)
        flat = CpuReferenceOpticalExecutor().execute(snapshot, rays)
        mismatched_rays = build_pinhole_camera_rays(
            OpticalPinholeCameraSpec(
                frame_id=9,
                sim_time=0.09,
                env_idx=0,
                sensor_id="other_cam",
                width=1,
                height=1,
                fx=1.0,
                fy=1.0,
                cx=0.0,
                cy=0.0,
            )
        )

        with pytest.raises(ValueError, match="rays.sensor_id"):
            build_pinhole_camera_image_result(flat, spec, rays=mismatched_rays)
