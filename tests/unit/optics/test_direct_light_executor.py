from __future__ import annotations

import numpy as np
import pytest

from optics import (
    CpuDirectLightOpticalExecutor,
    CpuReferenceOpticalExecutor,
    MissingAccelerationError,
    OpticalFrameInputs,
    OpticalInstanceSpec,
    OpticalLightSpec,
    OpticalMaterialSpec,
    OpticalOutputProfile,
    OpticalSceneCache,
    OpticalWorldRegistry,
)
from physics.publish import CpuPublishedFrame
from sensing import (
    OpticalPinholeCameraSpec,
    OpticalRaySensorSpec,
    build_pinhole_camera_image_result,
    build_pinhole_camera_rays,
)


def _frame(*, frame_id: int = 31, sim_time: float = 0.31) -> CpuPublishedFrame:
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


def _snapshot(registry: OpticalWorldRegistry, *, acceleration: str = "cpu_bvh"):
    return OpticalSceneCache(registry).snapshot_from_frame_inputs(
        OpticalFrameInputs.from_published_frame(_frame()),
        acceleration=acceleration,
    )


def _rgb_ray(snapshot, *, origins=None, directions=None) -> OpticalRaySensorSpec:
    return OpticalRaySensorSpec(
        frame_id=snapshot.frame_id,
        sim_time=snapshot.sim_time,
        env_idx=snapshot.env_idx,
        sensor_id="rgb_probe",
        origins_world=[[-1.0, 0.0, 0.0]] if origins is None else origins,
        directions_world=[[1.0, 0.0, 0.0]] if directions is None else directions,
        sensor_role="rgb",
    )


def _registry_with_vertical_plane(*, albedo=(0.5, 0.25, 0.125)) -> OpticalWorldRegistry:
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat_wall", albedo_rgb=albedo))
    registry.add_plane_geometry("wall", normal_local=[1.0, 0.0, 0.0], point_local=[0.0, 0.0, 0.0])
    registry.add_instance(OpticalInstanceSpec("wall", "wall", "mat_wall", roles=frozenset({"rgb"})))
    return registry


def _add_blocker(registry: OpticalWorldRegistry, *, x: float, roles=frozenset({"rgb"})) -> None:
    registry.add_material(OpticalMaterialSpec(f"mat_blocker_{x}"))
    registry.add_triangle_mesh_geometry(
        f"blocker_{x}",
        vertices_local=[
            [x, -1.0, -1.0],
            [x, 1.0, -1.0],
            [x, 0.0, 1.0],
        ],
        triangles=[[0, 1, 2]],
    )
    registry.add_instance(
        OpticalInstanceSpec(
            f"blocker_{x}",
            f"blocker_{x}",
            f"mat_blocker_{x}",
            roles=roles,
        )
    )


class TestCpuDirectLightOpticalExecutor:
    def test_directional_light_front_lit_lambertian_rgb_and_intensity(self):
        registry = _registry_with_vertical_plane()
        registry.add_light(
            OpticalLightSpec(
                "sun",
                "directional",
                position_or_direction_world=[1.0, 0.0, 0.0],
                intensity=2.0,
                color_rgb=(1.0, 0.5, 0.25),
            )
        )
        snapshot = _snapshot(registry)
        spec = _rgb_ray(snapshot)

        result = CpuDirectLightOpticalExecutor(shadows=False).execute(snapshot, spec)

        expected_rgb = np.array([[1.0, 0.25, 0.0625]], dtype=np.float64)
        assert result.channel("rgb").shape == (1, 3)
        assert result.channel("rgb").dtype == np.float64
        assert result.channel("intensity").shape == (1,)
        assert result.channel("intensity").dtype == np.float64
        np.testing.assert_allclose(result.channel("rgb"), expected_rgb)
        np.testing.assert_allclose(result.channel("intensity"), expected_rgb @ [0.2126, 0.7152, 0.0722])
        assert set(result.channels) == CpuDirectLightOpticalExecutor.capabilities
        assert result.output_profile is OpticalOutputProfile.DIRECT_LIGHT_FULL

    def test_rgb_preview_profile_filters_to_rgb_hit_mask_and_diagnostics(self):
        registry = _registry_with_vertical_plane()
        snapshot = _snapshot(registry)
        spec = _rgb_ray(snapshot)

        result = CpuDirectLightOpticalExecutor(shadows=False).execute(
            snapshot,
            spec,
            output_profile=OpticalOutputProfile.RGB_PREVIEW,
        )

        assert result.output_profile is OpticalOutputProfile.RGB_PREVIEW
        assert set(result.channels) == OpticalOutputProfile.RGB_PREVIEW.guaranteed_channels
        assert result.has_channel("rgb")
        assert result.has_channel("hit_mask")
        assert not result.has_channel("range_m")
        assert not result.has_channel("intensity")

    def test_render_only_profile_keeps_diagnostics_only(self):
        registry = _registry_with_vertical_plane()
        snapshot = _snapshot(registry)

        result = CpuDirectLightOpticalExecutor(shadows=False).execute(
            snapshot,
            _rgb_ray(snapshot),
            output_profile="render_only",
        )

        assert result.output_profile is OpticalOutputProfile.RENDER_ONLY
        assert set(result.channels) == OpticalOutputProfile.RENDER_ONLY.guaranteed_channels

    def test_rejects_unsupported_profile(self):
        registry = _registry_with_vertical_plane()
        snapshot = _snapshot(registry)

        with pytest.raises(ValueError, match="output_profile"):
            CpuDirectLightOpticalExecutor(shadows=False).execute(
                snapshot,
                _rgb_ray(snapshot),
                output_profile=OpticalOutputProfile.GEOMETRY_FULL,
            )

    def test_directional_light_back_facing_surface_contributes_zero(self):
        registry = _registry_with_vertical_plane()
        registry.add_light(
            OpticalLightSpec("sun", "directional", position_or_direction_world=[-1.0, 0.0, 0.0])
        )
        snapshot = _snapshot(registry)

        result = CpuDirectLightOpticalExecutor(shadows=False).execute(snapshot, _rgb_ray(snapshot))

        np.testing.assert_allclose(result.channel("rgb"), [[0.0, 0.0, 0.0]])
        np.testing.assert_allclose(result.channel("intensity"), [0.0])

    def test_point_light_uses_inverse_square_attenuation(self):
        registry = _registry_with_vertical_plane(albedo=(1.0, 0.5, 0.25))
        registry.add_light(
            OpticalLightSpec(
                "lamp",
                "point",
                position_or_direction_world=[2.0, 0.0, 0.0],
                intensity=8.0,
                color_rgb=(1.0, 1.0, 1.0),
            )
        )
        snapshot = _snapshot(registry)

        result = CpuDirectLightOpticalExecutor(shadows=False).execute(snapshot, _rgb_ray(snapshot))

        np.testing.assert_allclose(result.channel("rgb"), [[2.0, 1.0, 0.5]])

    def test_ambient_only_hit_and_miss_background(self):
        registry = _registry_with_vertical_plane(albedo=(0.5, 0.25, 0.125))
        snapshot = _snapshot(registry)
        spec = _rgb_ray(
            snapshot,
            origins=[[-1.0, 0.0, 0.0], [-1.0, 5.0, 0.0]],
            directions=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )

        result = CpuDirectLightOpticalExecutor(
            shadows=False,
            ambient_rgb=(0.2, 0.2, 0.2),
            background_rgb=(0.1, 0.2, 0.3),
        ).execute(snapshot, spec)

        np.testing.assert_allclose(result.channel("rgb")[0], [0.1, 0.05, 0.025])
        np.testing.assert_allclose(result.channel("rgb")[1], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(result.channel("intensity")[1], 0.0)

    def test_disabled_light_is_ignored(self):
        registry = _registry_with_vertical_plane()
        registry.add_light(
            OpticalLightSpec(
                "disabled",
                "directional",
                position_or_direction_world=[1.0, 0.0, 0.0],
                intensity=100.0,
                enabled=False,
            )
        )
        snapshot = _snapshot(registry)

        result = CpuDirectLightOpticalExecutor(shadows=False).execute(snapshot, _rgb_ray(snapshot))

        np.testing.assert_allclose(result.channel("rgb"), [[0.0, 0.0, 0.0]])

    def test_shadow_ray_blocks_directional_light(self):
        registry = _registry_with_vertical_plane(albedo=(1.0, 1.0, 1.0))
        _add_blocker(registry, x=0.5)
        registry.add_light(
            OpticalLightSpec("sun", "directional", position_or_direction_world=[1.0, 0.0, 0.0])
        )
        snapshot = _snapshot(registry)

        result = CpuDirectLightOpticalExecutor(shadows=True).execute(snapshot, _rgb_ray(snapshot))

        np.testing.assert_allclose(result.channel("rgb"), [[0.0, 0.0, 0.0]])

    def test_shadow_ray_blocks_point_light_before_light_distance(self):
        registry = _registry_with_vertical_plane(albedo=(1.0, 1.0, 1.0))
        _add_blocker(registry, x=0.5)
        registry.add_light(OpticalLightSpec("lamp", "point", position_or_direction_world=[1.0, 0.0, 0.0]))
        snapshot = _snapshot(registry)

        result = CpuDirectLightOpticalExecutor(shadows=True).execute(snapshot, _rgb_ray(snapshot))

        np.testing.assert_allclose(result.channel("rgb"), [[0.0, 0.0, 0.0]])

    def test_shadow_ray_does_not_block_occluder_behind_point_light(self):
        registry = _registry_with_vertical_plane(albedo=(1.0, 1.0, 1.0))
        _add_blocker(registry, x=2.0)
        registry.add_light(OpticalLightSpec("lamp", "point", position_or_direction_world=[1.0, 0.0, 0.0]))
        snapshot = _snapshot(registry)

        result = CpuDirectLightOpticalExecutor(shadows=True).execute(snapshot, _rgb_ray(snapshot))

        np.testing.assert_allclose(result.channel("rgb"), [[1.0, 1.0, 1.0]])

    def test_shadow_rays_use_primary_sensor_role(self):
        registry = _registry_with_vertical_plane(albedo=(1.0, 1.0, 1.0))
        _add_blocker(registry, x=0.5, roles=frozenset({"debug"}))
        registry.add_light(
            OpticalLightSpec("sun", "directional", position_or_direction_world=[1.0, 0.0, 0.0])
        )
        snapshot = _snapshot(registry)

        result = CpuDirectLightOpticalExecutor(shadows=True).execute(snapshot, _rgb_ray(snapshot))

        np.testing.assert_allclose(result.channel("rgb"), [[1.0, 1.0, 1.0]])

    def test_camera_postprocess_reshapes_rgb_and_intensity(self):
        registry = OpticalWorldRegistry()
        registry.add_material(OpticalMaterialSpec("mat_wall", albedo_rgb=(0.4, 0.5, 0.6)))
        registry.add_plane_geometry("wall", normal_local=[0.0, 0.0, -1.0], point_local=[0.0, 0.0, 2.0])
        registry.add_instance(OpticalInstanceSpec("wall", "wall", "mat_wall", roles=frozenset({"rgb"})))
        registry.add_light(
            OpticalLightSpec("sun", "directional", position_or_direction_world=[0.0, 0.0, -1.0])
        )
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
            sensor_role="rgb",
        )
        rays = build_pinhole_camera_rays(camera)
        flat = CpuDirectLightOpticalExecutor(shadows=False).execute(snapshot, rays)
        image = build_pinhole_camera_image_result(flat, camera, rays=rays)

        assert image.channel("rgb").shape == (1, 1, 3)
        assert image.channel("intensity").shape == (1, 1)
        np.testing.assert_allclose(image.channel("rgb"), [[[0.4, 0.5, 0.6]]])

    def test_default_executor_requires_bvh_acceleration(self):
        registry = _registry_with_vertical_plane()
        snapshot = _snapshot(registry, acceleration="none")

        with pytest.raises(MissingAccelerationError):
            CpuDirectLightOpticalExecutor().execute(snapshot, _rgb_ray(snapshot))

    def test_tests_can_inject_reference_executor_without_shadows(self):
        registry = _registry_with_vertical_plane()
        registry.add_light(
            OpticalLightSpec("sun", "directional", position_or_direction_world=[1.0, 0.0, 0.0])
        )
        snapshot = _snapshot(registry, acceleration="none")

        result = CpuDirectLightOpticalExecutor(
            geometric_executor=CpuReferenceOpticalExecutor(),
            shadows=False,
        ).execute(snapshot, _rgb_ray(snapshot))

        np.testing.assert_allclose(result.channel("rgb"), [[0.5, 0.25, 0.125]])

    def test_zero_directional_light_vector_raises(self):
        registry = _registry_with_vertical_plane()
        registry.add_light(
            OpticalLightSpec("bad", "directional", position_or_direction_world=[0.0, 0.0, 0.0])
        )
        snapshot = _snapshot(registry)

        with pytest.raises(ValueError, match="direction"):
            CpuDirectLightOpticalExecutor().execute(snapshot, _rgb_ray(snapshot))

    def test_point_light_at_hit_position_returns_zero_singular_contribution(self):
        registry = _registry_with_vertical_plane(albedo=(1.0, 1.0, 1.0))
        registry.add_light(OpticalLightSpec("singular", "point", position_or_direction_world=[0.0, 0.0, 0.0]))
        snapshot = _snapshot(registry)

        result = CpuDirectLightOpticalExecutor(shadows=False).execute(snapshot, _rgb_ray(snapshot))

        np.testing.assert_allclose(result.channel("rgb"), [[0.0, 0.0, 0.0]])
        assert np.all(np.isfinite(result.channel("rgb")))
