from __future__ import annotations

import numpy as np
import pytest

from optics import (
    CpuBvhOpticalExecutor,
    CpuReferenceOpticalExecutor,
    MissingAccelerationError,
    OpticalFrameInputs,
    OpticalInstanceSpec,
    OpticalMaterialSpec,
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


def _frame(*, frame_id: int = 21, sim_time: float = 0.21) -> CpuPublishedFrame:
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


def _large_triangle(z: float) -> list[list[float]]:
    return [
        [-10.0, -10.0, z],
        [10.0, -10.0, z],
        [0.0, 10.0, z],
    ]


def _small_triangle(center_x: float, z: float = 0.0) -> list[list[float]]:
    return [
        [center_x - 0.25, -0.25, z],
        [center_x + 0.25, -0.25, z],
        [center_x, 0.25, z],
    ]


def _strip_triangles(count: int) -> tuple[list[list[float]], list[list[int]]]:
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    for i in range(count):
        start = len(vertices)
        vertices.extend(_small_triangle(float(i)))
        triangles.append([start, start + 1, start + 2])
    return vertices, triangles


def _snapshot(registry: OpticalWorldRegistry, *, acceleration: str = "none"):
    return OpticalSceneCache(registry).snapshot_from_frame_inputs(
        OpticalFrameInputs.from_published_frame(_frame()),
        acceleration=acceleration,
    )


def _ray_spec(snapshot, *, origins=None, directions=None, sensor_role: str = "depth") -> OpticalRaySensorSpec:
    return OpticalRaySensorSpec(
        frame_id=snapshot.frame_id,
        sim_time=snapshot.sim_time,
        env_idx=snapshot.env_idx,
        sensor_id="probe",
        origins_world=[[0.0, 0.0, 1.0]] if origins is None else origins,
        directions_world=[[0.0, 0.0, -1.0]] if directions is None else directions,
        sensor_role=sensor_role,
    )


def _assert_matching_channels(reference, accelerated) -> None:
    assert set(accelerated.channels) == CpuBvhOpticalExecutor.capabilities
    for name in ("range_m", "hit_mask", "position_world", "normal_world", "numeric_instance_id"):
        np.testing.assert_allclose(accelerated.channel(name), reference.channel(name), equal_nan=True)
    assert accelerated.channel("material_id").tolist() == reference.channel("material_id").tolist()
    assert accelerated.channel("instance_id").tolist() == reference.channel("instance_id").tolist()


class TestCpuBvhAccelerationBuild:
    def test_scene_cache_packs_world_space_triangle_bvh(self):
        registry = OpticalWorldRegistry()
        registry.add_material(OpticalMaterialSpec("mat_tri"))
        registry.add_triangle_mesh_geometry(
            "tri",
            vertices_local=_large_triangle(0.0),
            triangles=[[0, 1, 2]],
        )
        registry.add_instance(OpticalInstanceSpec("tri_instance", "tri", "mat_tri"))

        snapshot = _snapshot(registry, acceleration="cpu_bvh")

        assert snapshot.acceleration is not None
        assert snapshot.acceleration.kind == "cpu_bvh"
        assert len(snapshot.acceleration.nodes) == 1
        np.testing.assert_allclose(snapshot.acceleration.triangles_world[0], _large_triangle(0.0))
        np.testing.assert_array_equal(snapshot.acceleration.primitive_instance_indices, [0])
        np.testing.assert_array_equal(snapshot.acceleration.primitive_source_order_keys, [[0, 0]])

    def test_plane_only_snapshot_has_empty_bvh_payload(self):
        registry = OpticalWorldRegistry()
        registry.add_material(OpticalMaterialSpec("mat_floor"))
        registry.add_plane_geometry("floor", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 0.0])
        registry.add_instance(OpticalInstanceSpec("floor", "floor", "mat_floor"))

        snapshot = _snapshot(registry, acceleration="cpu_bvh")

        assert snapshot.acceleration is not None
        assert snapshot.acceleration.triangles_world.shape == (0, 3, 3)
        assert snapshot.acceleration.nodes == []

    def test_skips_degenerate_triangles_at_pack_time(self):
        registry = OpticalWorldRegistry()
        registry.add_material(OpticalMaterialSpec("mat_tri"))
        registry.add_triangle_mesh_geometry(
            "tri",
            vertices_local=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                *_large_triangle(0.0),
            ],
            triangles=[[0, 1, 2], [3, 4, 5]],
        )
        registry.add_instance(OpticalInstanceSpec("tri_instance", "tri", "mat_tri"))

        snapshot = _snapshot(registry, acceleration="cpu_bvh")

        assert snapshot.acceleration is not None
        assert snapshot.acceleration.triangles_world.shape == (1, 3, 3)
        np.testing.assert_array_equal(snapshot.acceleration.primitive_source_order_keys, [[0, 1]])


class TestCpuBvhOpticalExecutor:
    def test_matches_reference_for_triangle_mesh(self):
        registry = OpticalWorldRegistry()
        registry.add_material(OpticalMaterialSpec("mat_tri"))
        registry.add_triangle_mesh_geometry("tri", vertices_local=_large_triangle(0.0), triangles=[[0, 1, 2]])
        registry.add_instance(OpticalInstanceSpec("tri_instance", "tri", "mat_tri"))
        reference_snapshot = _snapshot(registry)
        bvh_snapshot = _snapshot(registry, acceleration="cpu_bvh")
        spec = _ray_spec(
            bvh_snapshot,
            origins=[[0.0, 0.0, 1.0], [20.0, 20.0, 1.0]],
            directions=[[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
        )
        reference_spec = _ray_spec(
            reference_snapshot,
            origins=spec.origins_world,
            directions=spec.directions_world,
        )

        reference = CpuReferenceOpticalExecutor().execute(reference_snapshot, reference_spec)
        accelerated = CpuBvhOpticalExecutor().execute(bvh_snapshot, spec)

        _assert_matching_channels(reference, accelerated)

    def test_traverses_internal_bvh_nodes(self):
        vertices, triangles = _strip_triangles(6)
        registry = OpticalWorldRegistry()
        registry.add_material(OpticalMaterialSpec("mat_strip"))
        registry.add_triangle_mesh_geometry("strip", vertices_local=vertices, triangles=triangles)
        registry.add_instance(OpticalInstanceSpec("strip_instance", "strip", "mat_strip"))
        snapshot = _snapshot(registry, acceleration="cpu_bvh")
        spec = _ray_spec(snapshot, origins=[[5.0, 0.0, 1.0]], directions=[[0.0, 0.0, -1.0]])

        result = CpuBvhOpticalExecutor().execute(snapshot, spec)

        assert snapshot.acceleration is not None
        assert len(snapshot.acceleration.nodes) > 1
        np.testing.assert_array_equal(result.channel("hit_mask"), [True])
        np.testing.assert_allclose(result.channel("range_m"), [1.0])
        assert result.channel("instance_id").tolist() == ["strip_instance"]

    def test_parallel_ray_on_aabb_boundary_is_inclusive(self):
        registry = OpticalWorldRegistry()
        registry.add_material(OpticalMaterialSpec("mat_tri"))
        registry.add_triangle_mesh_geometry(
            "tri",
            vertices_local=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            triangles=[[0, 1, 2]],
        )
        registry.add_instance(OpticalInstanceSpec("tri_instance", "tri", "mat_tri"))
        snapshot = _snapshot(registry, acceleration="cpu_bvh")
        spec = _ray_spec(snapshot, origins=[[0.0, 0.0, 1.0]], directions=[[0.0, 0.0, -1.0]])

        result = CpuBvhOpticalExecutor().execute(snapshot, spec)

        np.testing.assert_array_equal(result.channel("hit_mask"), [True])
        np.testing.assert_allclose(result.channel("range_m"), [1.0])

    def test_plane_side_pass_works_with_empty_bvh(self):
        registry = OpticalWorldRegistry()
        registry.add_material(OpticalMaterialSpec("mat_floor"))
        registry.add_plane_geometry("floor", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 0.0])
        registry.add_instance(OpticalInstanceSpec("floor", "floor", "mat_floor"))
        snapshot = _snapshot(registry, acceleration="cpu_bvh")
        spec = _ray_spec(snapshot)

        result = CpuBvhOpticalExecutor().execute(snapshot, spec)

        np.testing.assert_array_equal(result.channel("hit_mask"), [True])
        np.testing.assert_allclose(result.channel("range_m"), [1.0])
        assert result.channel("instance_id").tolist() == ["floor"]

    def test_global_source_order_tie_breaks_plane_before_mesh(self):
        registry = OpticalWorldRegistry()
        registry.add_material(OpticalMaterialSpec("mat_plane"))
        registry.add_material(OpticalMaterialSpec("mat_tri"))
        registry.add_plane_geometry("plane", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 0.0])
        registry.add_triangle_mesh_geometry("tri", vertices_local=_large_triangle(0.0), triangles=[[0, 1, 2]])
        registry.add_instance(OpticalInstanceSpec("plane_instance", "plane", "mat_plane"))
        registry.add_instance(OpticalInstanceSpec("tri_instance", "tri", "mat_tri"))
        snapshot = _snapshot(registry, acceleration="cpu_bvh")
        spec = _ray_spec(snapshot)

        result = CpuBvhOpticalExecutor().execute(snapshot, spec)

        np.testing.assert_allclose(result.channel("range_m"), [1.0])
        assert result.channel("instance_id").tolist() == ["plane_instance"]

    def test_global_source_order_tie_breaks_mesh_before_plane(self):
        registry = OpticalWorldRegistry()
        registry.add_material(OpticalMaterialSpec("mat_plane"))
        registry.add_material(OpticalMaterialSpec("mat_tri"))
        registry.add_plane_geometry("plane", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 0.0])
        registry.add_triangle_mesh_geometry("tri", vertices_local=_large_triangle(0.0), triangles=[[0, 1, 2]])
        registry.add_instance(OpticalInstanceSpec("tri_instance", "tri", "mat_tri"))
        registry.add_instance(OpticalInstanceSpec("plane_instance", "plane", "mat_plane"))
        snapshot = _snapshot(registry, acceleration="cpu_bvh")
        spec = _ray_spec(snapshot)

        result = CpuBvhOpticalExecutor().execute(snapshot, spec)

        np.testing.assert_allclose(result.channel("range_m"), [1.0])
        assert result.channel("instance_id").tolist() == ["tri_instance"]

    def test_near_equal_non_coplanar_hit_keeps_true_nearest_when_outside_t_eps(self):
        registry = OpticalWorldRegistry()
        registry.add_material(OpticalMaterialSpec("mat_far"))
        registry.add_material(OpticalMaterialSpec("mat_near"))
        registry.add_triangle_mesh_geometry(
            "far", vertices_local=_large_triangle(-1e-8), triangles=[[0, 1, 2]]
        )
        registry.add_triangle_mesh_geometry(
            "near", vertices_local=_large_triangle(0.0), triangles=[[0, 1, 2]]
        )
        registry.add_instance(OpticalInstanceSpec("far_first", "far", "mat_far"))
        registry.add_instance(OpticalInstanceSpec("near_second", "near", "mat_near"))
        snapshot = _snapshot(registry, acceleration="cpu_bvh")
        spec = _ray_spec(snapshot)

        result = CpuBvhOpticalExecutor().execute(snapshot, spec)

        np.testing.assert_allclose(result.channel("range_m"), [1.0])
        assert result.channel("instance_id").tolist() == ["near_second"]

    def test_roles_filtering_matches_reference(self):
        registry = OpticalWorldRegistry()
        registry.add_material(OpticalMaterialSpec("mat_debug"))
        registry.add_triangle_mesh_geometry("tri", vertices_local=_large_triangle(0.0), triangles=[[0, 1, 2]])
        registry.add_instance(
            OpticalInstanceSpec("debug_tri", "tri", "mat_debug", roles=frozenset({"debug"}))
        )
        snapshot = _snapshot(registry, acceleration="cpu_bvh")
        spec = _ray_spec(snapshot, sensor_role="depth")

        result = CpuBvhOpticalExecutor().execute(snapshot, spec)

        np.testing.assert_array_equal(result.channel("hit_mask"), [False])
        assert np.isinf(result.channel("range_m")[0])
        np.testing.assert_array_equal(result.channel("numeric_instance_id"), [0])

    def test_missing_acceleration_raises(self):
        registry = OpticalWorldRegistry()
        registry.add_material(OpticalMaterialSpec("mat_tri"))
        registry.add_triangle_mesh_geometry("tri", vertices_local=_large_triangle(0.0), triangles=[[0, 1, 2]])
        registry.add_instance(OpticalInstanceSpec("tri_instance", "tri", "mat_tri"))
        snapshot = _snapshot(registry)
        spec = _ray_spec(snapshot)

        with pytest.raises(MissingAccelerationError, match="CPU BVH"):
            CpuBvhOpticalExecutor().execute(snapshot, spec)

    def test_camera_postprocess_accepts_bvh_result(self):
        registry = OpticalWorldRegistry()
        registry.add_material(OpticalMaterialSpec("mat_wall"))
        registry.add_triangle_mesh_geometry(
            "wall",
            vertices_local=[
                [-10.0, -10.0, 2.0],
                [10.0, -10.0, 2.0],
                [0.0, 10.0, 2.0],
            ],
            triangles=[[0, 1, 2]],
        )
        registry.add_instance(OpticalInstanceSpec("wall_instance", "wall", "mat_wall"))
        snapshot = _snapshot(registry, acceleration="cpu_bvh")
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
        )
        rays = build_pinhole_camera_rays(camera)
        flat = CpuBvhOpticalExecutor().execute(snapshot, rays)
        image = build_pinhole_camera_image_result(flat, camera, rays=rays)

        np.testing.assert_array_equal(image.channel("hit_mask"), [[True]])
        np.testing.assert_allclose(image.channel("range_m"), [[2.0]])
        np.testing.assert_allclose(image.channel("depth_m"), [[2.0]])
