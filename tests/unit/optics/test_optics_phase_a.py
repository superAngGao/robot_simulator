from __future__ import annotations

import numpy as np
import pytest

from optics import (
    CpuReferenceOpticalExecutor,
    OpticalComputeResult,
    OpticalFrameInputs,
    OpticalInstanceSpec,
    OpticalMaterialSpec,
    OpticalSceneCache,
    OpticalWorldRegistry,
)
from physics.publish import CpuPublishedFrame
from physics.spatial import SpatialTransform
from sensing import OpticalRaySensorSpec


def _frame(*, frame_id: int = 3, sim_time: float = 0.03, X_world=None) -> CpuPublishedFrame:
    return CpuPublishedFrame(
        frame_id=frame_id,
        sim_time=sim_time,
        step_index=frame_id,
        env_mask=None,
        q=np.zeros(0, dtype=np.float64),
        qdot=np.zeros(0, dtype=np.float64),
        X_world=[] if X_world is None else X_world,
        v_bodies=[],
        contact_count=0,
        contacts=[],
        telemetry=None,
    )


class TestOpticalRaySensorSpec:
    def test_lives_in_sensing_and_normalizes_directions(self):
        spec = OpticalRaySensorSpec(
            frame_id=3,
            sim_time=0.03,
            env_idx=0,
            sensor_id="depth_probe",
            origins_world=[[0.0, 0.0, 1.0]],
            directions_world=[[0.0, 0.0, -2.0]],
        )

        assert spec.num_rays == 1
        np.testing.assert_allclose(spec.directions_world, [[0.0, 0.0, -1.0]])

    def test_rejects_malformed_direction_batch(self):
        with pytest.raises(ValueError, match="same ray count"):
            OpticalRaySensorSpec(
                frame_id=0,
                sim_time=0.0,
                env_idx=0,
                sensor_id="bad",
                origins_world=[[0.0, 0.0, 1.0]],
                directions_world=[[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
            )


class TestOpticalWorldRegistry:
    def test_registers_material_geometry_and_instance(self):
        registry = OpticalWorldRegistry()
        registry.add_material(OpticalMaterialSpec("mat_floor", albedo_rgb=(0.8, 0.7, 0.6)))
        registry.add_plane_geometry("floor_plane", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 0.0])
        instance = registry.add_instance(
            OpticalInstanceSpec(
                instance_id="floor",
                geometry_id="floor_plane",
                material_id="mat_floor",
            )
        )

        assert registry.materials["mat_floor"].albedo_rgb == (0.8, 0.7, 0.6)
        assert len(registry.instances) == 1
        assert instance.numeric_instance_id == 1
        assert registry.instances[0].numeric_instance_id == 1
        assert registry.instances[0].roles == frozenset({"rgb", "depth", "lidar", "segmentation"})

    def test_registry_owned_numeric_instance_ids_are_stable_and_monotonic(self):
        registry = OpticalWorldRegistry()
        registry.add_material(OpticalMaterialSpec("mat_floor"))
        registry.add_plane_geometry("floor_plane", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 0.0])

        first = registry.add_instance(OpticalInstanceSpec("floor_a", "floor_plane", "mat_floor"))
        explicit = registry.add_instance(
            OpticalInstanceSpec("floor_b", "floor_plane", "mat_floor", numeric_instance_id=7)
        )
        after_explicit = registry.add_instance(OpticalInstanceSpec("floor_c", "floor_plane", "mat_floor"))

        assert first.numeric_instance_id == 1
        assert explicit.numeric_instance_id == 7
        assert after_explicit.numeric_instance_id == 8

    def test_rejects_background_numeric_instance_id(self):
        with pytest.raises(ValueError, match="numeric_instance_id"):
            OpticalInstanceSpec("background", "floor_plane", "mat_floor", numeric_instance_id=0)

    def test_rejects_instance_with_unknown_material(self):
        registry = OpticalWorldRegistry()
        registry.add_plane_geometry("floor_plane", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 0.0])

        with pytest.raises(KeyError, match="Unknown material_id"):
            registry.add_instance(
                OpticalInstanceSpec(
                    instance_id="floor",
                    geometry_id="floor_plane",
                    material_id="missing",
                )
            )


class TestOpticalSceneCache:
    def test_snapshot_composes_body_bound_geometry_transform(self):
        registry = OpticalWorldRegistry()
        registry.add_material(OpticalMaterialSpec("mat_body"))
        registry.add_plane_geometry("local_plane", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 0.0])
        registry.add_instance(
            OpticalInstanceSpec(
                instance_id="body_plane",
                geometry_id="local_plane",
                material_id="mat_body",
                body_index=0,
                X_body_geometry=SpatialTransform.from_translation(np.array([0.0, 0.0, 0.25])),
            )
        )
        frame = _frame(X_world=[SpatialTransform.from_translation(np.array([1.0, 2.0, 0.5]))])

        inputs = OpticalFrameInputs.from_published_frame(frame)
        snapshot = OpticalSceneCache(registry).snapshot_from_frame_inputs(inputs)

        assert snapshot.frame_id == 3
        assert snapshot.env_idx == 0
        assert snapshot.instances[0].body_index == 0
        assert snapshot.instances[0].numeric_instance_id == 1
        np.testing.assert_allclose(snapshot.instances[0].X_world_geometry.r, [1.0, 2.0, 0.75])

    def test_phase_a_rejects_cpu_multi_env_selection(self):
        registry = OpticalWorldRegistry()

        with pytest.raises(NotImplementedError, match="one CPU env"):
            OpticalSceneCache(registry).snapshot_from_published_frame(_frame(), env_idx=1)

    def test_published_frame_wrapper_matches_frame_inputs_entrypoint(self):
        registry = OpticalWorldRegistry()
        registry.add_material(OpticalMaterialSpec("mat_body"))
        registry.add_plane_geometry("local_plane", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 0.0])
        registry.add_instance(
            OpticalInstanceSpec(
                instance_id="body_plane",
                geometry_id="local_plane",
                material_id="mat_body",
                body_index=0,
            )
        )
        frame = _frame(X_world=[SpatialTransform.from_translation(np.array([0.0, 0.0, 0.5]))])
        cache = OpticalSceneCache(registry)

        wrapped = cache.snapshot_from_published_frame(frame)
        direct = cache.snapshot_from_frame_inputs(OpticalFrameInputs.from_published_frame(frame))

        assert wrapped.frame_id == direct.frame_id
        assert wrapped.sim_time == direct.sim_time
        np.testing.assert_allclose(
            wrapped.instances[0].X_world_geometry.r,
            direct.instances[0].X_world_geometry.r,
        )

    def test_frame_inputs_rejects_mismatched_rigid_frame_metadata(self):
        frame = _frame(frame_id=3, sim_time=0.03)

        with pytest.raises(ValueError, match="frame_id"):
            OpticalFrameInputs(frame_id=4, sim_time=0.03, env_idx=0, rigid=frame)

        with pytest.raises(ValueError, match="sim_time"):
            OpticalFrameInputs(frame_id=3, sim_time=0.04, env_idx=0, rigid=frame)


class TestCpuReferenceOpticalExecutor:
    def test_returns_first_hit_range_and_material_id_for_planes(self):
        registry = OpticalWorldRegistry()
        registry.add_material(OpticalMaterialSpec("mat_far"))
        registry.add_material(OpticalMaterialSpec("mat_near"))
        registry.add_plane_geometry("far_plane", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 0.0])
        registry.add_plane_geometry("near_plane", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 1.0])
        registry.add_instance(OpticalInstanceSpec("far", "far_plane", "mat_far"))
        registry.add_instance(OpticalInstanceSpec("near", "near_plane", "mat_near"))
        snapshot = OpticalSceneCache(registry).snapshot_from_published_frame(_frame())
        spec = OpticalRaySensorSpec(
            frame_id=snapshot.frame_id,
            sim_time=snapshot.sim_time,
            env_idx=snapshot.env_idx,
            sensor_id="depth_probe",
            origins_world=[[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]],
            directions_world=[[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
        )

        result = CpuReferenceOpticalExecutor().execute(snapshot, spec)

        assert isinstance(result, OpticalComputeResult)
        np.testing.assert_array_equal(result.channel("hit_mask"), [True, False])
        np.testing.assert_allclose(result.channel("range_m")[0], 1.0)
        assert np.isinf(result.channel("range_m")[1])
        assert result.channel("material_id").tolist() == ["mat_near", None]
        assert result.channel("instance_id").tolist() == ["near", None]
        np.testing.assert_array_equal(result.channel("numeric_instance_id"), [2, 0])
        np.testing.assert_allclose(result.channel("position_world")[0], [0.0, 0.0, 1.0])

    def test_intersects_body_bound_triangle_mesh(self):
        registry = OpticalWorldRegistry()
        registry.add_material(OpticalMaterialSpec("mat_triangle"))
        registry.add_triangle_mesh_geometry(
            "tri",
            vertices_local=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            triangles=[[0, 1, 2]],
        )
        registry.add_instance(OpticalInstanceSpec("body_tri", "tri", "mat_triangle", body_index=0))
        frame = _frame(X_world=[SpatialTransform.from_translation(np.array([1.0, 0.0, 0.0]))])
        snapshot = OpticalSceneCache(registry).snapshot_from_published_frame(frame)
        spec = OpticalRaySensorSpec(
            frame_id=snapshot.frame_id,
            sim_time=snapshot.sim_time,
            env_idx=snapshot.env_idx,
            sensor_id="seg_probe",
            origins_world=[[1.25, 0.25, 1.0]],
            directions_world=[[0.0, 0.0, -1.0]],
        )

        result = CpuReferenceOpticalExecutor().execute(snapshot, spec)

        np.testing.assert_array_equal(result.channel("hit_mask"), [True])
        np.testing.assert_allclose(result.channel("range_m"), [1.0])
        assert result.channel("material_id").tolist() == ["mat_triangle"]
        np.testing.assert_allclose(result.channel("position_world"), [[1.25, 0.25, 0.0]])

    def test_filters_instances_by_sensor_role(self):
        registry = OpticalWorldRegistry()
        registry.add_material(OpticalMaterialSpec("mat_debug"))
        registry.add_plane_geometry("debug_plane", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 1.0])
        registry.add_instance(
            OpticalInstanceSpec(
                "debug_only",
                "debug_plane",
                "mat_debug",
                roles=frozenset({"debug"}),
            )
        )
        snapshot = OpticalSceneCache(registry).snapshot_from_published_frame(_frame())
        spec = OpticalRaySensorSpec(
            frame_id=snapshot.frame_id,
            sim_time=snapshot.sim_time,
            env_idx=snapshot.env_idx,
            sensor_id="depth_probe",
            origins_world=[[0.0, 0.0, 2.0]],
            directions_world=[[0.0, 0.0, -1.0]],
            sensor_role="depth",
        )

        result = CpuReferenceOpticalExecutor().execute(snapshot, spec)

        np.testing.assert_array_equal(result.channel("hit_mask"), [False])
        assert np.isinf(result.channel("range_m")[0])
        np.testing.assert_array_equal(result.channel("numeric_instance_id"), [0])

    def test_rejects_frame_mismatch(self):
        registry = OpticalWorldRegistry()
        snapshot = OpticalSceneCache(registry).snapshot_from_published_frame(_frame(frame_id=3))
        spec = OpticalRaySensorSpec(
            frame_id=4,
            sim_time=snapshot.sim_time,
            env_idx=snapshot.env_idx,
            sensor_id="stale",
            origins_world=[[0.0, 0.0, 1.0]],
            directions_world=[[0.0, 0.0, -1.0]],
        )

        with pytest.raises(ValueError, match="frame_id"):
            CpuReferenceOpticalExecutor().execute(snapshot, spec)
