from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
import pytest

import tools.optical_pipeline_lab.go2_backend as go2_backend
from optics import (
    CpuDirectLightOpticalExecutor,
    DeviceOpticalSceneCache,
    GpuBruteForceOpticalExecutor,
    GpuDeviceBvhDirectLightOpticalExecutor,
    GpuDeviceBvhOpticalExecutor,
    GpuDeviceSceneOpticalExecutor,
    OpticalFrameInputs,
    OpticalInstanceSpec,
    OpticalLightSpec,
    OpticalMaterialSpec,
    OpticalOutputProfile,
    OpticalSceneCache,
    OpticalWorldRegistry,
    build_cuda_lbvh_from_snapshot,
    build_device_bvh_from_snapshot,
    execute_optical_on_gpu_published_frame,
    refit_device_bvh_from_snapshot,
    stage_optical_compute_result_to_host,
)
from physics.geometry import BodyCollisionGeometry, ShapeInstance, SphereShape
from physics.joint import FreeJoint
from physics.merged_model import merge_models
from physics.publish import ConsumerState, CpuPublishedFrame, GpuPublishedFrame, PublishPolicy
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from robot.model import RobotModel
from sensing import OpticalPinholeCameraSpec, OpticalRaySensorSpec, build_pinhole_camera_rays
from tools.optical_pipeline_lab import dynamic_frames
from tools.optical_pipeline_lab.presets import get_preset
from tools.optical_pipeline_lab.runner import LabRunOptions, apply_run_overrides, run_scenario

try:
    import warp as wp

    from physics.gpu_engine import GpuEngine

    HAS_WARP = True
except Exception:
    HAS_WARP = False

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not HAS_WARP, reason="Warp or CUDA not available"),
]


def _ball_model() -> RobotModel:
    tree = RobotTreeNumpy(gravity=9.81)
    tree.add_body(
        Body(
            name="ball",
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
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(SphereShape(0.05))])],
        contact_body_names=["ball"],
    )


def _make_engine():
    merged = merge_models({"ball": _ball_model()})
    engine = GpuEngine(merged, num_envs=1, device="cuda:0")
    q0, _ = merged.tree.default_state()
    q0[6] = 0.5
    engine.reset(q0=q0)
    return engine


def _world_static_registry() -> OpticalWorldRegistry:
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat_floor"))
    registry.add_plane_geometry("floor", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 0.0])
    registry.add_instance(OpticalInstanceSpec("floor", "floor", "mat_floor"))
    return registry


def _body_bound_registry() -> OpticalWorldRegistry:
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat_body"))
    registry.add_plane_geometry("body_plane", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 0.0])
    registry.add_instance(OpticalInstanceSpec("body_plane", "body_plane", "mat_body", body_index=0))
    return registry


def _body_bound_triangle_registry() -> OpticalWorldRegistry:
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat_body"))
    registry.add_triangle_mesh_geometry(
        "body_tri",
        vertices_local=[
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
        ],
        triangles=[[0, 1, 2]],
    )
    registry.add_instance(
        OpticalInstanceSpec(
            "body_tri",
            "body_tri",
            "mat_body",
            body_index=0,
            X_body_geometry=SpatialTransform.from_translation(np.array([0.0, 0.0, 0.25])),
        )
    )
    return registry


def _world_static_triangle_registry() -> OpticalWorldRegistry:
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat_tri"))
    registry.add_triangle_mesh_geometry(
        "tri",
        vertices_local=[
            [-10.0, -10.0, 0.0],
            [10.0, -10.0, 0.0],
            [0.0, 10.0, 0.0],
        ],
        triangles=[[0, 1, 2]],
    )
    registry.add_instance(OpticalInstanceSpec("tri_instance", "tri", "mat_tri"))
    return registry


def _world_static_triangle_grid_registry() -> OpticalWorldRegistry:
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat_grid"))
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    for row in range(4):
        for col in range(4):
            x = col * 0.4 - 0.8
            y = row * 0.4 - 0.8
            base = len(vertices)
            vertices.extend(
                [
                    [x, y, 0.0],
                    [x + 0.25, y, 0.0],
                    [x, y + 0.25, 0.0],
                ]
            )
            triangles.append([base, base + 1, base + 2])
    registry.add_triangle_mesh_geometry("grid", vertices_local=vertices, triangles=triangles)
    registry.add_instance(OpticalInstanceSpec("grid_instance", "grid", "mat_grid"))
    return registry


def _plane_then_triangle_registry() -> OpticalWorldRegistry:
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat_surface"))
    registry.add_plane_geometry("floor", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 0.0])
    registry.add_triangle_mesh_geometry(
        "tri",
        vertices_local=[
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        triangles=[[0, 1, 2]],
    )
    registry.add_instance(OpticalInstanceSpec("floor_instance", "floor", "mat_surface"))
    registry.add_instance(OpticalInstanceSpec("tri_instance", "tri", "mat_surface"))
    return registry


def _material_light_registry() -> OpticalWorldRegistry:
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat_plane", albedo_rgb=(0.2, 0.4, 0.6)))
    registry.add_material(OpticalMaterialSpec("mat_tri", albedo_rgb=(0.8, 0.1, 0.3)))
    registry.add_light(
        OpticalLightSpec(
            "sun",
            kind="directional",
            position_or_direction_world=[0.0, 0.0, 1.0],
            intensity=1.5,
            color_rgb=(1.0, 0.9, 0.8),
        )
    )
    registry.add_light(
        OpticalLightSpec(
            "disabled_fill",
            kind="point",
            position_or_direction_world=[1.0, 2.0, 3.0],
            intensity=10.0,
            color_rgb=(0.1, 0.2, 0.3),
            enabled=False,
        )
    )
    registry.add_plane_geometry("floor", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 0.0])
    registry.add_triangle_mesh_geometry(
        "tri",
        vertices_local=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        triangles=[[0, 1, 2]],
    )
    registry.add_instance(OpticalInstanceSpec("floor_instance", "floor", "mat_plane"))
    registry.add_instance(OpticalInstanceSpec("tri_instance", "tri", "mat_tri"))
    return registry


def _lit_triangle_registry() -> OpticalWorldRegistry:
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat_lit", albedo_rgb=(0.8, 0.1, 0.3)))
    registry.add_light(
        OpticalLightSpec(
            "sun",
            kind="directional",
            position_or_direction_world=[0.0, 0.0, 1.0],
            intensity=2.0,
            color_rgb=(1.0, 0.5, 0.25),
        )
    )
    registry.add_triangle_mesh_geometry(
        "tri",
        vertices_local=[
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        triangles=[[0, 1, 2]],
    )
    registry.add_instance(OpticalInstanceSpec("tri_instance", "tri", "mat_lit"))
    return registry


def _point_lit_triangle_registry() -> OpticalWorldRegistry:
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat_lit", albedo_rgb=(0.25, 0.5, 0.75)))
    registry.add_light(
        OpticalLightSpec(
            "lamp",
            kind="point",
            position_or_direction_world=[0.0, 0.0, 4.0],
            intensity=16.0,
            color_rgb=(0.5, 1.0, 0.25),
        )
    )
    registry.add_triangle_mesh_geometry(
        "tri",
        vertices_local=[
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        triangles=[[0, 1, 2]],
    )
    registry.add_instance(OpticalInstanceSpec("tri_instance", "tri", "mat_lit"))
    return registry


def _zero_light_triangle_registry() -> OpticalWorldRegistry:
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat_unlit", albedo_rgb=(0.3, 0.5, 0.7)))
    registry.add_triangle_mesh_geometry(
        "tri",
        vertices_local=[
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        triangles=[[0, 1, 2]],
    )
    registry.add_instance(OpticalInstanceSpec("tri_instance", "tri", "mat_unlit"))
    return registry


def _multi_light_triangle_registry() -> OpticalWorldRegistry:
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat_lit", albedo_rgb=(0.5, 0.25, 0.75)))
    registry.add_light(
        OpticalLightSpec(
            "key",
            kind="directional",
            position_or_direction_world=[0.0, 0.0, 1.0],
            intensity=1.0,
            color_rgb=(1.0, 0.5, 0.25),
        )
    )
    registry.add_light(
        OpticalLightSpec(
            "fill",
            kind="directional",
            position_or_direction_world=[0.0, 1.0, 1.0],
            intensity=2.0,
            color_rgb=(0.25, 1.0, 0.5),
        )
    )
    registry.add_triangle_mesh_geometry(
        "tri",
        vertices_local=[
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        triangles=[[0, 1, 2]],
    )
    registry.add_instance(OpticalInstanceSpec("tri_instance", "tri", "mat_lit"))
    return registry


def _point_shadowed_floor_registry() -> OpticalWorldRegistry:
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat_floor", albedo_rgb=(0.6, 0.4, 0.2)))
    registry.add_material(OpticalMaterialSpec("mat_blocker", albedo_rgb=(0.1, 0.1, 0.1)))
    registry.add_light(
        OpticalLightSpec(
            "lamp",
            kind="point",
            position_or_direction_world=[2.0, 0.0, 2.0],
            intensity=16.0,
            color_rgb=(1.0, 1.0, 1.0),
        )
    )
    registry.add_plane_geometry("floor", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 0.0])
    registry.add_triangle_mesh_geometry(
        "point_shadow_blocker",
        vertices_local=[
            [1.0, -1.0, 0.5],
            [1.0, 1.0, 0.5],
            [1.0, 0.0, 1.5],
        ],
        triangles=[[0, 1, 2]],
    )
    registry.add_instance(OpticalInstanceSpec("floor_instance", "floor", "mat_floor"))
    registry.add_instance(OpticalInstanceSpec("blocker_instance", "point_shadow_blocker", "mat_blocker"))
    return registry


def _plane_shadowed_floor_registry() -> OpticalWorldRegistry:
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat_floor", albedo_rgb=(0.6, 0.4, 0.2)))
    registry.add_material(OpticalMaterialSpec("mat_occluder", albedo_rgb=(0.1, 0.1, 0.1)))
    registry.add_light(
        OpticalLightSpec(
            "lamp",
            kind="point",
            position_or_direction_world=[2.0, 0.0, 2.0],
            intensity=16.0,
            color_rgb=(1.0, 1.0, 1.0),
        )
    )
    registry.add_triangle_mesh_geometry(
        "floor_patch",
        vertices_local=[
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        triangles=[[0, 1, 2]],
    )
    registry.add_plane_geometry(
        "shadow_plane",
        normal_local=[1.0, 0.0, 0.0],
        point_local=[1.0, 0.0, 0.0],
    )
    registry.add_instance(OpticalInstanceSpec("floor_patch_instance", "floor_patch", "mat_floor"))
    registry.add_instance(OpticalInstanceSpec("shadow_plane_instance", "shadow_plane", "mat_occluder"))
    return registry


def _shadowed_floor_registry() -> OpticalWorldRegistry:
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat_floor", albedo_rgb=(0.7, 0.6, 0.5)))
    registry.add_material(OpticalMaterialSpec("mat_wall", albedo_rgb=(0.2, 0.2, 0.2)))
    registry.add_light(
        OpticalLightSpec(
            "sun",
            kind="directional",
            position_or_direction_world=[1.0, 0.0, 1.0],
            intensity=2.0,
            color_rgb=(1.0, 1.0, 1.0),
        )
    )
    registry.add_plane_geometry("floor", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 0.0])
    registry.add_triangle_mesh_geometry(
        "shadow_wall",
        vertices_local=[
            [0.5, -1.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.0, 2.0],
        ],
        triangles=[[0, 1, 2]],
    )
    registry.add_instance(OpticalInstanceSpec("floor_instance", "floor", "mat_floor"))
    registry.add_instance(OpticalInstanceSpec("shadow_wall_instance", "shadow_wall", "mat_wall"))
    return registry


def _consumer(consumer_id: str = "optical_camera") -> ConsumerState:
    return ConsumerState(
        consumer_id=consumer_id,
        consumer_kind="render_backed_sensing",
        qos_mode="lossless",
        access_mode="borrow",
        consumer_location="device",
    )


def _ray_spec(frame) -> OpticalRaySensorSpec:
    return OpticalRaySensorSpec(
        frame_id=frame.frame_id,
        sim_time=frame.sim_time,
        env_idx=0,
        sensor_id="gpu_camera",
        origins_world=[[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]],
        directions_world=[[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
        max_distance=10.0,
    )


def _downward_camera_spec(frame) -> OpticalPinholeCameraSpec:
    return OpticalPinholeCameraSpec(
        frame_id=frame.frame_id,
        sim_time=frame.sim_time,
        env_idx=0,
        sensor_id="gpu_camera",
        width=2,
        height=1,
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
        X_world_camera=SpatialTransform(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ],
                dtype=np.float64,
            ),
            np.array([0.0, 0.0, 2.0], dtype=np.float64),
        ),
        max_distance=10.0,
        sensor_role="rgb",
    )


def _cpu_frame_like(frame) -> CpuPublishedFrame:
    return CpuPublishedFrame(
        frame_id=frame.frame_id,
        sim_time=frame.sim_time,
        step_index=frame.step_index,
        env_mask=None,
        q=np.zeros(0, dtype=np.float64),
        qdot=np.zeros(0, dtype=np.float64),
        X_world=[],
        v_bodies=[],
        contact_count=0,
        contacts=[],
        telemetry=None,
    )


def _many_role_registry() -> OpticalWorldRegistry:
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat_roles"))
    registry.add_plane_geometry("role_plane", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 0.0])
    for index in range(41):
        registry.add_instance(
            OpticalInstanceSpec(
                f"role_{index:02d}_plane",
                "role_plane",
                "mat_roles",
                roles=frozenset({f"role_{index:02d}"}),
            )
        )
    return registry


def _execute_device_scene(
    engine,
    *,
    consumer_id: str,
    registry: OpticalWorldRegistry,
    spec: OpticalRaySensorSpec,
    frame,
    use_aabb: bool = False,
):
    stream = wp.Stream(device=engine._device)
    borrowed = engine.borrow_device_frame(consumer_id, frame.frame_id, stream=stream)
    cache = DeviceOpticalSceneCache(registry, device=engine._device, stream=stream)
    snapshot = cache.snapshot_from_gpu_frame(
        borrowed,
        env_idx=spec.env_idx,
        stream=stream,
        include_aabb=use_aabb,
    )
    result = GpuDeviceSceneOpticalExecutor(
        device=engine._device,
        stream=stream,
        use_aabb=use_aabb,
    ).execute(snapshot, spec)
    done_event = engine.complete_device_consumer(consumer_id, frame.frame_id, stream=stream)
    return result, snapshot, borrowed, done_event


def _execute_device_bvh_scene(
    engine,
    *,
    consumer_id: str,
    registry: OpticalWorldRegistry,
    spec: OpticalRaySensorSpec,
    frame,
):
    stream = wp.Stream(device=engine._device)
    borrowed = engine.borrow_device_frame(consumer_id, frame.frame_id, stream=stream)
    cache = DeviceOpticalSceneCache(registry, device=engine._device, stream=stream)
    snapshot = cache.snapshot_from_gpu_frame(
        borrowed,
        env_idx=spec.env_idx,
        stream=stream,
        include_aabb=True,
    )
    bvh = build_device_bvh_from_snapshot(snapshot, device=engine._device, stream=stream)
    result = GpuDeviceBvhOpticalExecutor(
        device=engine._device,
        stream=stream,
    ).execute(snapshot, bvh, spec)
    done_event = engine.complete_device_consumer(consumer_id, frame.frame_id, stream=stream)
    return result, snapshot, bvh, borrowed, done_event


def test_execute_optical_on_gpu_published_frame_completes_q52_consumer():
    engine = _make_engine()
    consumer = _consumer()
    engine.register_consumer(consumer)
    engine.step()
    frame = engine.latest_published_frame()
    stream = wp.Stream(device=engine._device)

    result = execute_optical_on_gpu_published_frame(
        engine,
        consumer.consumer_id,
        _world_static_registry(),
        _ray_spec(frame),
        frame=frame,
        stream=stream,
    )

    assert result.location == "device"
    assert result.ready_event is consumer.device_done_event
    assert result.resources
    assert consumer.latest_seen_frame_id == frame.frame_id
    assert consumer.device_completed_frame_id == frame.frame_id
    assert isinstance(result.ready_event, wp.Event)

    staged = stage_optical_compute_result_to_host(result)
    np.testing.assert_array_equal(staged.channel("hit_mask"), [True, False])
    np.testing.assert_allclose(staged.channel("range_m"), [2.0, np.inf])
    np.testing.assert_array_equal(staged.channel("numeric_instance_id"), [1, 0])


def test_optical_result_buffers_survive_published_slot_reuse_after_completion():
    engine = _make_engine()
    engine.set_publish_policy(PublishPolicy(on_ring_full="block"))
    consumer = _consumer()
    engine.register_consumer(consumer)
    engine.step()
    frame = engine.latest_published_frame()
    stream = wp.Stream(device=engine._device)

    result = execute_optical_on_gpu_published_frame(
        engine,
        consumer.consumer_id,
        _world_static_registry(),
        _ray_spec(frame),
        frame=frame,
        stream=stream,
    )

    engine.step()
    engine.step()
    engine.step()
    wp.synchronize()
    staged = stage_optical_compute_result_to_host(result)

    np.testing.assert_array_equal(staged.channel("hit_mask"), [True, False])
    np.testing.assert_allclose(staged.channel("range_m"), [2.0, np.inf])
    np.testing.assert_array_equal(staged.channel("numeric_instance_id"), [1, 0])


def test_execute_optical_on_gpu_published_frame_supports_body_bound_registry():
    engine = _make_engine()
    consumer = _consumer()
    engine.register_consumer(consumer)
    engine.step()
    frame = engine.latest_published_frame()
    wp.synchronize_event(frame.ready_event)
    body_z = float(frame.x_world_r_wp.numpy()[0, 0, 2])

    result = execute_optical_on_gpu_published_frame(
        engine,
        consumer.consumer_id,
        _body_bound_registry(),
        _ray_spec(frame),
        frame=frame,
        stream=wp.Stream(device=engine._device),
    )

    staged = stage_optical_compute_result_to_host(result)
    np.testing.assert_array_equal(staged.channel("hit_mask"), [True, False])
    np.testing.assert_allclose(staged.channel("range_m"), [2.0 - body_z, np.inf], atol=1e-6)
    np.testing.assert_allclose(staged.channel("position_world")[0], [0.0, 0.0, body_z], atol=1e-6)
    np.testing.assert_array_equal(staged.channel("numeric_instance_id"), [1, 0])


def test_body_bound_optical_result_survives_published_slot_reuse_after_completion():
    engine = _make_engine()
    engine.set_publish_policy(PublishPolicy(on_ring_full="block"))
    consumer = _consumer()
    engine.register_consumer(consumer)
    engine.step()
    frame = engine.latest_published_frame()
    wp.synchronize_event(frame.ready_event)
    body_z = float(frame.x_world_r_wp.numpy()[0, 0, 2])

    result = execute_optical_on_gpu_published_frame(
        engine,
        consumer.consumer_id,
        _body_bound_registry(),
        _ray_spec(frame),
        frame=frame,
        stream=wp.Stream(device=engine._device),
    )

    engine.step()
    engine.step()
    engine.step()
    staged = stage_optical_compute_result_to_host(result)

    np.testing.assert_array_equal(staged.channel("hit_mask"), [True, False])
    np.testing.assert_allclose(staged.channel("range_m"), [2.0 - body_z, np.inf], atol=1e-6)


def test_device_scene_cache_updates_world_static_plane_from_gpu_frame():
    engine = _make_engine()
    consumer = _consumer()
    engine.register_consumer(consumer)
    engine.step()
    frame = engine.latest_published_frame()
    stream = wp.Stream(device=engine._device)
    borrowed = engine.borrow_device_frame(consumer.consumer_id, frame.frame_id, stream=stream)
    cache = DeviceOpticalSceneCache(_world_static_registry(), device=engine._device, stream=stream)

    snapshot = cache.snapshot_from_gpu_frame(borrowed, env_idx=0, stream=stream)
    done_event = engine.complete_device_consumer(consumer.consumer_id, frame.frame_id, stream=stream)
    wp.synchronize_event(done_event)
    wp.synchronize_event(snapshot.ready_event)

    np.testing.assert_allclose(snapshot.plane_point_world.numpy(), [[0.0, 0.0, 0.0]], atol=1e-6)
    np.testing.assert_allclose(snapshot.plane_normal_world.numpy(), [[0.0, 0.0, 1.0]], atol=1e-6)
    assert snapshot.resources[0] is borrowed


def test_device_scene_cache_updates_body_bound_triangle_from_gpu_frame():
    engine = _make_engine()
    consumer = _consumer()
    engine.register_consumer(consumer)
    engine.step()
    frame = engine.latest_published_frame()
    wp.synchronize_event(frame.ready_event)
    body_z = float(frame.x_world_r_wp.numpy()[0, 0, 2])
    stream = wp.Stream(device=engine._device)
    borrowed = engine.borrow_device_frame(consumer.consumer_id, frame.frame_id, stream=stream)
    cache = DeviceOpticalSceneCache(_body_bound_triangle_registry(), device=engine._device, stream=stream)

    snapshot = cache.snapshot_from_gpu_frame(borrowed, env_idx=0, stream=stream, include_aabb=True)
    done_event = engine.complete_device_consumer(consumer.consumer_id, frame.frame_id, stream=stream)
    wp.synchronize_event(done_event)
    wp.synchronize_event(snapshot.ready_event)

    v0 = snapshot.triangle_v0_world.numpy()
    e1 = snapshot.triangle_e1_world.numpy()
    e2 = snapshot.triangle_e2_world.numpy()
    normals = snapshot.triangle_normal_world.numpy()
    aabb_min = snapshot.triangle_aabb_min.numpy()
    aabb_max = snapshot.triangle_aabb_max.numpy()
    triangles = np.stack([v0, v0 + e1, v0 + e2], axis=1)
    expected_z = body_z + 0.25
    np.testing.assert_allclose(triangles[0, :, 2], [expected_z, expected_z, expected_z], atol=1e-6)
    np.testing.assert_allclose(triangles[0, 0, :2], [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(triangles[0, 1, :2], [0.2, 0.0], atol=1e-6)
    np.testing.assert_allclose(triangles[0, 2, :2], [0.0, 0.2], atol=1e-6)
    np.testing.assert_allclose(normals, [[0.0, 0.0, 1.0]], atol=1e-6)
    np.testing.assert_allclose(aabb_min, [[0.0, 0.0, expected_z]], atol=1e-6)
    np.testing.assert_allclose(aabb_max, [[0.2, 0.2, expected_z]], atol=1e-6)


def test_optical_lab_synthetic_body_bound_frame_moves_snapshot_geometry():
    registry = dynamic_frames.make_body_bound_triangle_registry(geometry_z_offset=0.25)
    base_frame = dynamic_frames.make_gpu_pose_frame(
        wp_module=wp,
        translations=np.zeros((1, 1, 3), dtype=np.float32),
        frame_id=21,
        sim_time=0.21,
        device="cuda:0",
    )
    moved_frame = dynamic_frames.clone_and_perturb_gpu_published_pose_frame(
        base_frame,
        wp_module=wp,
        translation_offsets={(0, 0): [0.0, 0.0, 0.4]},
        frame_id=22,
        sim_time=0.22,
    )
    cache = DeviceOpticalSceneCache(registry, device="cuda:0")

    base_snapshot = cache.snapshot_from_gpu_frame(base_frame, env_idx=0, include_aabb=True)
    moved_snapshot = cache.snapshot_from_gpu_frame(moved_frame, env_idx=0, include_aabb=True)
    wp.synchronize_event(base_snapshot.ready_event)
    wp.synchronize_event(moved_snapshot.ready_event)

    base_triangles = np.stack(
        [
            base_snapshot.triangle_v0_world.numpy(),
            base_snapshot.triangle_v0_world.numpy() + base_snapshot.triangle_e1_world.numpy(),
            base_snapshot.triangle_v0_world.numpy() + base_snapshot.triangle_e2_world.numpy(),
        ],
        axis=1,
    )
    moved_triangles = np.stack(
        [
            moved_snapshot.triangle_v0_world.numpy(),
            moved_snapshot.triangle_v0_world.numpy() + moved_snapshot.triangle_e1_world.numpy(),
            moved_snapshot.triangle_v0_world.numpy() + moved_snapshot.triangle_e2_world.numpy(),
        ],
        axis=1,
    )

    np.testing.assert_allclose(base_triangles[0, :, 2], [0.25, 0.25, 0.25], atol=1e-6)
    np.testing.assert_allclose(moved_triangles - base_triangles, [[[0.0, 0.0, 0.4]] * 3], atol=1e-6)
    np.testing.assert_allclose(moved_snapshot.triangle_aabb_min.numpy(), [[0.0, 0.0, 0.65]], atol=1e-6)
    np.testing.assert_allclose(moved_snapshot.triangle_aabb_max.numpy(), [[0.2, 0.2, 0.65]], atol=1e-6)
    np.testing.assert_allclose(base_frame.x_world_r_wp.numpy(), np.zeros((1, 1, 3), dtype=np.float32))


def test_optical_lab_dynamic_begin_frame_populates_prepare_timing_with_synthetic_frame():
    registry = dynamic_frames.make_body_bound_triangle_registry(geometry_z_offset=0.25)
    wp.init()
    stream = wp.Stream(device="cuda:0")
    base_frame = dynamic_frames.make_gpu_pose_frame(
        wp_module=wp,
        translations=np.zeros((1, 1, 3), dtype=np.float32),
        frame_id=31,
        sim_time=0.31,
        device="cuda:0",
    )
    moved_frame = dynamic_frames.clone_and_perturb_gpu_published_pose_frame(
        base_frame,
        wp_module=wp,
        translation_offsets={(0, 0): [0.0, 0.0, 0.4]},
        frame_id=32,
        sim_time=0.32,
    )
    cache = DeviceOpticalSceneCache(registry, device="cuda:0", stream=stream)
    base_snapshot = cache.snapshot_from_gpu_frame(base_frame, env_idx=0, stream=stream, include_aabb=True)
    base_bvh = build_device_bvh_from_snapshot(base_snapshot, device="cuda:0", stream=stream)
    wp.synchronize_event(base_bvh.ready_event)
    scene = SimpleNamespace(frame=SimpleNamespace(frame_id=base_frame.frame_id, sim_time=base_frame.sim_time))
    session = go2_backend.OpticalLabRenderSession(
        scene=scene,
        device=wp.get_device("cuda:0"),
        stream=stream,
        gpu_frame=base_frame,
        cache=cache,
        snapshot=base_snapshot,
        bvh=base_bvh,
        executor=None,
        bvh_backend="cpu",
        bvh_split_strategy="sort",
    )

    frame_context = go2_backend.OpticalLabRenderPipeline(session=session).begin_frame(
        frame_inputs=moved_frame,
        env_idx=0,
    )
    wp.synchronize_event(frame_context.snapshot.ready_event)
    wp.synchronize_event(frame_context.bvh.ready_event)

    assert frame_context.snapshot is not base_snapshot
    assert frame_context.bvh.frame_id == moved_frame.frame_id
    assert frame_context.prepare_timing["snapshot_ms"] >= 0.0
    assert frame_context.prepare_timing["accel_refit_ms"] >= 0.0
    assert np.isnan(float(frame_context.prepare_timing["accel_rebuild_ms"]))
    np.testing.assert_allclose(
        frame_context.snapshot.triangle_aabb_min.numpy(),
        [[0.0, 0.0, 0.65]],
        atol=1e-6,
    )


def test_optical_lab_dynamic_video_loop_writes_prepare_timing_csv(tmp_path):
    registry = dynamic_frames.make_body_bound_triangle_registry(geometry_z_offset=0.25)
    wp.init()
    stream = wp.Stream(device="cuda:0")
    base_frame = dynamic_frames.make_gpu_pose_frame(
        wp_module=wp,
        translations=np.zeros((1, 1, 3), dtype=np.float32),
        frame_id=41,
        sim_time=0.41,
        device="cuda:0",
    )
    frame0 = dynamic_frames.clone_gpu_published_pose_frame(
        base_frame,
        wp_module=wp,
        frame_id=42,
        sim_time=0.42,
    )
    frame1 = dynamic_frames.clone_and_perturb_gpu_published_pose_frame(
        base_frame,
        wp_module=wp,
        translation_offsets={(0, 0): [0.0, 0.0, 0.4]},
        frame_id=43,
        sim_time=0.43,
    )
    cache = DeviceOpticalSceneCache(registry, device="cuda:0", stream=stream)
    base_snapshot = cache.snapshot_from_gpu_frame(base_frame, env_idx=0, stream=stream, include_aabb=True)
    base_bvh = build_device_bvh_from_snapshot(base_snapshot, device="cuda:0", stream=stream)
    wp.synchronize_event(base_bvh.ready_event)
    scene = SimpleNamespace(
        frame=SimpleNamespace(frame_id=base_frame.frame_id, sim_time=base_frame.sim_time),
        bounds_min=np.array([-0.2, -0.2, 0.0], dtype=np.float64),
        bounds_max=np.array([0.4, 0.4, 0.9], dtype=np.float64),
    )
    session = go2_backend.OpticalLabRenderSession(
        scene=scene,
        device=wp.get_device("cuda:0"),
        stream=stream,
        gpu_frame=base_frame,
        cache=cache,
        snapshot=base_snapshot,
        bvh=base_bvh,
        executor=GpuDeviceBvhDirectLightOpticalExecutor(device="cuda:0", stream=stream, shadows=False),
        bvh_backend="cpu",
        bvh_split_strategy="sort",
    )
    frame_timing_csv = tmp_path / "frame_timing.csv"

    rows = go2_backend._run_video_benchmark(
        go2_backend.OpticalLabRenderPipeline(session=session),
        SimpleNamespace(
            width=64,
            height=48,
            view="front",
            video_mode="fixed_view",
            video_frames=2,
            video_fps=30.0,
            video_raygen="gpu",
            video_ray_cache="off",
            video_readback="none",
            video_readback_delivery="sync",
            video_readback_ring_depth=2,
            video_frame_inputs=[frame0, frame1],
            video_geometry_mode="dynamic_rigid",
            write_frames=False,
            render_profile=False,
            fail_on_overflow=False,
            progress_every=0,
            frame_timing_csv=str(frame_timing_csv),
            lab_frame_defaults={"geometry_mode": "dynamic_rigid"},
        ),
        tmp_path,
    )

    assert len(rows._rows) == 2
    for row in rows._rows:
        assert row["geometry_mode"] == "dynamic_rigid"
        assert float(row["snapshot_ms"]) >= 0.0
        assert float(row["accel_refit_ms"]) >= 0.0
        assert np.isnan(float(row["accel_rebuild_ms"]))
        assert np.isnan(float(row["readback_host_ms"]))
    assert frame_timing_csv.exists()


def test_optical_lab_dynamic_smoke_preset_writes_prepare_timing_csv(tmp_path):
    config = apply_run_overrides(
        get_preset("synthetic_body_triangle_dynamic_smoke"),
        width=64,
        height=48,
        readback="none",
    )
    run_scenario(
        config,
        LabRunOptions(
            out=tmp_path / "dynamic_preset",
            frames=2,
            warmup_renders=0,
            progress_every=0,
            fail_on_overflow=False,
        ),
    )

    frame_timing_csv = tmp_path / "dynamic_preset" / "frame_timing.csv"
    with frame_timing_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    for row in rows:
        assert row["scenario_name"] == "synthetic_body_triangle_dynamic_smoke"
        assert row["scene_preset"] == "synthetic_body_triangle"
        assert row["geometry_mode"] == "dynamic_rigid"
        assert float(row["snapshot_ms"]) >= 0.0
        assert float(row["accel_refit_ms"]) >= 0.0
        assert np.isnan(float(row["accel_rebuild_ms"]))


def test_device_scene_packs_materials_and_lights_for_gpu_shading():
    cache = DeviceOpticalSceneCache(_material_light_registry(), device="cuda:0")
    scene = cache.scene

    assert scene.num_materials == 2
    assert scene.num_lights == 1
    np.testing.assert_allclose(
        scene.material_albedo_rgb.numpy(),
        [[0.2, 0.4, 0.6], [0.8, 0.1, 0.3]],
        atol=1e-6,
    )
    np.testing.assert_array_equal(scene.plane_material_index.numpy(), [0])
    np.testing.assert_array_equal(scene.triangle_material_index.numpy(), [1])
    np.testing.assert_array_equal(scene.light_kind.numpy(), [0])
    np.testing.assert_allclose(scene.light_position_or_direction_world.numpy(), [[0.0, 0.0, 1.0]])
    np.testing.assert_allclose(scene.light_intensity.numpy(), [1.5])
    np.testing.assert_allclose(scene.light_color_rgb.numpy(), [[1.0, 0.9, 0.8]])


def test_device_scene_cache_rotates_body_bound_triangle_normal():
    registry = _body_bound_triangle_registry()
    R_y_90 = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    frame = GpuPublishedFrame(
        slot_id=0,
        frame_id=17,
        sim_time=0.17,
        step_index=17,
        env_mask_wp=None,
        q_wp=None,
        qdot_wp=None,
        x_world_R_wp=wp.array(R_y_90.reshape(1, 1, 3, 3), dtype=wp.float32, device="cuda:0"),
        x_world_r_wp=wp.array(np.zeros((1, 1, 3), dtype=np.float32), dtype=wp.float32, device="cuda:0"),
        v_bodies_wp=None,
        contact_count_wp=None,
        contact_cache_ref=None,
        telemetry_ref=None,
        ready_event=None,
        slot_meta=None,
    )
    cache = DeviceOpticalSceneCache(registry, device="cuda:0")

    snapshot = cache.snapshot_from_gpu_frame(frame, include_aabb=True)
    wp.synchronize_event(snapshot.ready_event)

    v0 = snapshot.triangle_v0_world.numpy()
    e1 = snapshot.triangle_e1_world.numpy()
    e2 = snapshot.triangle_e2_world.numpy()
    normals = snapshot.triangle_normal_world.numpy()
    aabb_min = snapshot.triangle_aabb_min.numpy()
    aabb_max = snapshot.triangle_aabb_max.numpy()
    triangles = np.stack([v0, v0 + e1, v0 + e2], axis=1)

    expected = np.array(
        [
            [0.25, 0.0, 0.0],
            [0.25, 0.0, -0.2],
            [0.25, 0.2, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(triangles[0], expected, atol=1e-6)
    np.testing.assert_allclose(normals, [[1.0, 0.0, 0.0]], atol=1e-6)
    np.testing.assert_allclose(aabb_min, [[0.25, 0.0, -0.2]], atol=1e-6)
    np.testing.assert_allclose(aabb_max, [[0.25, 0.2, 0.0]], atol=1e-6)


def test_device_scene_executor_matches_l5b1_for_body_bound_plane():
    engine = _make_engine()
    legacy_consumer = _consumer("legacy_optical")
    device_consumer = _consumer("device_optical")
    engine.register_consumer(legacy_consumer)
    engine.register_consumer(device_consumer)
    engine.step()
    frame = engine.latest_published_frame()
    registry = _body_bound_registry()
    spec = _ray_spec(frame)

    legacy = execute_optical_on_gpu_published_frame(
        engine,
        legacy_consumer.consumer_id,
        registry,
        spec,
        frame=frame,
        stream=wp.Stream(device=engine._device),
    )
    device, snapshot, borrowed, done_event = _execute_device_scene(
        engine,
        consumer_id=device_consumer.consumer_id,
        registry=registry,
        spec=spec,
        frame=frame,
    )

    assert device.location == "device"
    assert device.ready_event is not None
    assert all(resource is not snapshot for resource in device.resources)
    assert all(resource is not borrowed for resource in device.resources)
    wp.synchronize_event(done_event)
    legacy_host = stage_optical_compute_result_to_host(legacy)
    device_host = stage_optical_compute_result_to_host(device)

    np.testing.assert_array_equal(device_host.channel("hit_mask"), legacy_host.channel("hit_mask"))
    np.testing.assert_allclose(device_host.channel("range_m"), legacy_host.channel("range_m"), atol=1e-6)
    np.testing.assert_allclose(
        device_host.channel("position_world"),
        legacy_host.channel("position_world"),
        atol=1e-6,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        device_host.channel("normal_world"),
        legacy_host.channel("normal_world"),
        atol=1e-6,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        device_host.channel("numeric_instance_id"),
        legacy_host.channel("numeric_instance_id"),
    )


def test_device_scene_derived_executor_matches_l5a_for_world_static_triangle_mesh():
    engine = _make_engine()
    consumer = _consumer()
    engine.register_consumer(consumer)
    engine.step()
    frame = engine.latest_published_frame()
    registry = _world_static_triangle_registry()
    spec = OpticalRaySensorSpec(
        frame_id=frame.frame_id,
        sim_time=frame.sim_time,
        env_idx=0,
        sensor_id="gpu_camera",
        origins_world=[[0.0, 0.0, 2.0], [20.0, 20.0, 2.0]],
        directions_world=[[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
        max_distance=10.0,
    )
    host_snapshot = OpticalSceneCache(registry).snapshot_from_frame_inputs(
        OpticalFrameInputs.from_published_frame(_cpu_frame_like(frame)),
        acceleration="cpu_bvh",
    )

    l5a = GpuBruteForceOpticalExecutor(device=engine._device).execute(host_snapshot, spec)
    device, _, _, done_event = _execute_device_scene(
        engine,
        consumer_id=consumer.consumer_id,
        registry=registry,
        spec=spec,
        frame=frame,
    )

    wp.synchronize_event(done_event)
    l5a_host = stage_optical_compute_result_to_host(l5a)
    device_host = stage_optical_compute_result_to_host(device)
    np.testing.assert_array_equal(device_host.channel("hit_mask"), l5a_host.channel("hit_mask"))
    np.testing.assert_allclose(device_host.channel("range_m"), l5a_host.channel("range_m"), atol=1e-6)
    np.testing.assert_allclose(
        device_host.channel("position_world"),
        l5a_host.channel("position_world"),
        atol=1e-6,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        device_host.channel("normal_world"),
        l5a_host.channel("normal_world"),
        atol=1e-6,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        device_host.channel("numeric_instance_id"),
        l5a_host.channel("numeric_instance_id"),
    )


def test_device_scene_aabb_executor_matches_non_aabb_for_world_static_triangle_mesh():
    engine = _make_engine()
    no_aabb_consumer = _consumer("device_no_aabb")
    aabb_consumer = _consumer("device_aabb")
    engine.register_consumer(no_aabb_consumer)
    engine.register_consumer(aabb_consumer)
    engine.step()
    frame = engine.latest_published_frame()
    registry = _world_static_triangle_registry()
    spec = OpticalRaySensorSpec(
        frame_id=frame.frame_id,
        sim_time=frame.sim_time,
        env_idx=0,
        sensor_id="gpu_camera",
        origins_world=[
            [0.0, 0.0, 2.0],
            [20.0, 20.0, 2.0],
            [9.5, -9.5, 2.0],
        ],
        directions_world=[
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
        ],
        max_distance=10.0,
    )

    no_aabb, _, _, no_aabb_done = _execute_device_scene(
        engine,
        consumer_id=no_aabb_consumer.consumer_id,
        registry=registry,
        spec=spec,
        frame=frame,
    )
    aabb, _, _, aabb_done = _execute_device_scene(
        engine,
        consumer_id=aabb_consumer.consumer_id,
        registry=registry,
        spec=spec,
        frame=frame,
        use_aabb=True,
    )

    wp.synchronize_event(no_aabb_done)
    wp.synchronize_event(aabb_done)
    no_aabb_host = stage_optical_compute_result_to_host(no_aabb)
    aabb_host = stage_optical_compute_result_to_host(aabb)
    np.testing.assert_array_equal(aabb_host.channel("hit_mask"), no_aabb_host.channel("hit_mask"))
    np.testing.assert_allclose(aabb_host.channel("range_m"), no_aabb_host.channel("range_m"), atol=1e-6)
    np.testing.assert_allclose(
        aabb_host.channel("position_world"),
        no_aabb_host.channel("position_world"),
        atol=1e-6,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        aabb_host.channel("normal_world"),
        no_aabb_host.channel("normal_world"),
        atol=1e-6,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        aabb_host.channel("numeric_instance_id"),
        no_aabb_host.channel("numeric_instance_id"),
    )


def test_device_bvh_executor_matches_device_scene_for_world_static_triangle_mesh():
    engine = _make_engine()
    scene_consumer = _consumer("device_scene")
    bvh_consumer = _consumer("device_bvh")
    engine.register_consumer(scene_consumer)
    engine.register_consumer(bvh_consumer)
    engine.step()
    frame = engine.latest_published_frame()
    registry = _world_static_triangle_grid_registry()
    spec = OpticalRaySensorSpec(
        frame_id=frame.frame_id,
        sim_time=frame.sim_time,
        env_idx=0,
        sensor_id="gpu_camera",
        origins_world=[
            [-0.75, -0.75, 2.0],
            [-0.35, 0.05, 2.0],
            [0.45, 0.45, 2.0],
            [2.0, 2.0, 2.0],
        ],
        directions_world=[
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
        ],
        max_distance=10.0,
    )

    scene, _, _, scene_done = _execute_device_scene(
        engine,
        consumer_id=scene_consumer.consumer_id,
        registry=registry,
        spec=spec,
        frame=frame,
    )
    bvh, snapshot, built_bvh, borrowed, bvh_done = _execute_device_bvh_scene(
        engine,
        consumer_id=bvh_consumer.consumer_id,
        registry=registry,
        spec=spec,
        frame=frame,
    )

    assert built_bvh.stats.num_nodes > 1
    assert built_bvh.stats.max_depth > 0
    assert all(resource is not snapshot for resource in bvh.resources)
    assert all(resource is not borrowed for resource in bvh.resources)
    wp.synchronize_event(scene_done)
    wp.synchronize_event(bvh_done)
    scene_host = stage_optical_compute_result_to_host(scene)
    bvh_host = stage_optical_compute_result_to_host(bvh)
    np.testing.assert_array_equal(bvh_host.channel("hit_mask"), scene_host.channel("hit_mask"))
    np.testing.assert_allclose(bvh_host.channel("range_m"), scene_host.channel("range_m"), atol=1e-6)
    np.testing.assert_allclose(
        bvh_host.channel("position_world"),
        scene_host.channel("position_world"),
        atol=1e-6,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        bvh_host.channel("normal_world"),
        scene_host.channel("normal_world"),
        atol=1e-6,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        bvh_host.channel("numeric_instance_id"),
        scene_host.channel("numeric_instance_id"),
    )


@pytest.mark.cuda_ext
def test_cuda_lbvh_executor_matches_cpu_bvh_for_world_static_triangle_mesh():
    pytest.importorskip("torch")
    engine = _make_engine()
    cpu_bvh_consumer = _consumer("cpu_bvh")
    cuda_bvh_consumer = _consumer("cuda_bvh")
    engine.register_consumer(cpu_bvh_consumer)
    engine.register_consumer(cuda_bvh_consumer)
    engine.step()
    frame = engine.latest_published_frame()
    registry = _world_static_triangle_grid_registry()
    spec = OpticalRaySensorSpec(
        frame_id=frame.frame_id,
        sim_time=frame.sim_time,
        env_idx=0,
        sensor_id="gpu_camera",
        origins_world=[
            [-0.75, -0.75, 2.0],
            [-0.35, 0.05, 2.0],
            [0.45, 0.45, 2.0],
            [2.0, 2.0, 2.0],
        ],
        directions_world=[
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
        ],
        max_distance=10.0,
    )

    cpu_bvh, _, _, _, cpu_done = _execute_device_bvh_scene(
        engine,
        consumer_id=cpu_bvh_consumer.consumer_id,
        registry=registry,
        spec=spec,
        frame=frame,
    )

    stream = wp.Stream(device=engine._device)
    borrowed = engine.borrow_device_frame(cuda_bvh_consumer.consumer_id, frame.frame_id, stream=stream)
    cache = DeviceOpticalSceneCache(registry, device=engine._device, stream=stream)
    snapshot = cache.snapshot_from_gpu_frame(
        borrowed,
        env_idx=spec.env_idx,
        stream=stream,
        include_aabb=True,
    )
    cuda_bvh = build_cuda_lbvh_from_snapshot(snapshot, device=engine._device, stream=stream)
    assert cuda_bvh.stats.split_strategy == "cuda_lbvh"
    assert not cuda_bvh.stats.supports_refit
    assert cuda_bvh.resources
    cuda_result = GpuDeviceBvhOpticalExecutor(
        device=engine._device,
        stream=stream,
    ).execute(snapshot, cuda_bvh, spec)
    cuda_done = engine.complete_device_consumer(cuda_bvh_consumer.consumer_id, frame.frame_id, stream=stream)

    wp.synchronize_event(cpu_done)
    wp.synchronize_event(cuda_done)
    cpu_host = stage_optical_compute_result_to_host(cpu_bvh)
    cuda_host = stage_optical_compute_result_to_host(cuda_result)
    np.testing.assert_array_equal(cuda_host.channel("hit_mask"), cpu_host.channel("hit_mask"))
    np.testing.assert_allclose(cuda_host.channel("range_m"), cpu_host.channel("range_m"), atol=1e-6)
    np.testing.assert_allclose(
        cuda_host.channel("position_world"),
        cpu_host.channel("position_world"),
        atol=1e-6,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        cuda_host.channel("normal_world"),
        cpu_host.channel("normal_world"),
        atol=1e-6,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        cuda_host.channel("numeric_instance_id"),
        cpu_host.channel("numeric_instance_id"),
    )
    np.testing.assert_array_equal(cuda_host.channel("bvh_stack_overflow_count"), [0])
    assert 1 <= int(cuda_host.channel("bvh_max_stack_depth")[0]) <= 32


def test_device_bvh_executor_preserves_plane_triangle_tie_break():
    engine = _make_engine()
    consumer = _consumer()
    engine.register_consumer(consumer)
    engine.step()
    frame = engine.latest_published_frame()
    spec = OpticalRaySensorSpec(
        frame_id=frame.frame_id,
        sim_time=frame.sim_time,
        env_idx=0,
        sensor_id="gpu_camera",
        origins_world=[[0.0, 0.0, 2.0]],
        directions_world=[[0.0, 0.0, -1.0]],
        max_distance=10.0,
    )

    result, _, bvh, _, done_event = _execute_device_bvh_scene(
        engine,
        consumer_id=consumer.consumer_id,
        registry=_plane_then_triangle_registry(),
        spec=spec,
        frame=frame,
    )

    assert bvh.num_primitives == 1
    wp.synchronize_event(done_event)
    staged = stage_optical_compute_result_to_host(result)
    np.testing.assert_array_equal(staged.channel("hit_mask"), [True])
    np.testing.assert_allclose(staged.channel("range_m"), [2.0], atol=1e-6)
    np.testing.assert_array_equal(staged.channel("numeric_instance_id"), [1])
    np.testing.assert_array_equal(staged.channel("bvh_stack_overflow_count"), [0])


def test_device_bvh_camera_raygen_matches_materialized_rays():
    engine = _make_engine()
    consumer = _consumer()
    engine.register_consumer(consumer)
    engine.step()
    frame = engine.latest_published_frame()
    registry = _lit_triangle_registry()
    camera = _downward_camera_spec(frame)
    rays = build_pinhole_camera_rays(camera)

    stream = wp.Stream(device=engine._device)
    borrowed = engine.borrow_device_frame(consumer.consumer_id, frame.frame_id, stream=stream)
    cache = DeviceOpticalSceneCache(registry, device=engine._device, stream=stream)
    snapshot = cache.snapshot_from_gpu_frame(borrowed, env_idx=0, stream=stream, include_aabb=True)
    bvh = build_device_bvh_from_snapshot(snapshot, device=engine._device, stream=stream)
    executor = GpuDeviceBvhOpticalExecutor(device=engine._device, stream=stream)
    materialized = executor.execute(snapshot, bvh, rays)
    camera_result = executor.execute_camera(snapshot, bvh, camera)
    done_event = engine.complete_device_consumer(consumer.consumer_id, frame.frame_id, stream=stream)

    wp.synchronize_event(done_event)
    materialized_host = stage_optical_compute_result_to_host(materialized)
    camera_host = stage_optical_compute_result_to_host(camera_result)
    np.testing.assert_array_equal(camera_host.channel("hit_mask"), materialized_host.channel("hit_mask"))
    np.testing.assert_allclose(
        camera_host.channel("range_m"),
        materialized_host.channel("range_m"),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        camera_host.channel("position_world"),
        materialized_host.channel("position_world"),
        atol=1e-6,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        camera_host.channel("normal_world"),
        materialized_host.channel("normal_world"),
        atol=1e-6,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        camera_host.channel("numeric_instance_id"),
        materialized_host.channel("numeric_instance_id"),
    )
    np.testing.assert_array_equal(
        camera_host.channel("material_index"),
        materialized_host.channel("material_index"),
    )
    np.testing.assert_array_equal(camera_host.channel("bvh_stack_overflow_count"), [0])


def test_device_bvh_direct_light_camera_raygen_rgb_preview_matches_materialized_rays():
    engine = _make_engine()
    consumer = _consumer()
    engine.register_consumer(consumer)
    engine.step()
    frame = engine.latest_published_frame()
    registry = _lit_triangle_registry()
    camera = _downward_camera_spec(frame)
    rays = build_pinhole_camera_rays(camera)

    stream = wp.Stream(device=engine._device)
    borrowed = engine.borrow_device_frame(consumer.consumer_id, frame.frame_id, stream=stream)
    cache = DeviceOpticalSceneCache(registry, device=engine._device, stream=stream)
    snapshot = cache.snapshot_from_gpu_frame(borrowed, env_idx=0, stream=stream, include_aabb=True)
    bvh = build_device_bvh_from_snapshot(snapshot, device=engine._device, stream=stream)
    executor = GpuDeviceBvhDirectLightOpticalExecutor(
        device=engine._device,
        stream=stream,
        shadows=False,
        ambient_rgb=(0.1, 0.2, 0.3),
        background_rgb=(0.01, 0.02, 0.03),
    )
    materialized = executor.execute(snapshot, bvh, rays, output_profile=OpticalOutputProfile.RGB_PREVIEW)
    camera_result = executor.execute_camera(
        snapshot,
        bvh,
        camera,
        output_profile=OpticalOutputProfile.RGB_PREVIEW,
    )
    done_event = engine.complete_device_consumer(consumer.consumer_id, frame.frame_id, stream=stream)

    wp.synchronize_event(done_event)
    materialized_host = stage_optical_compute_result_to_host(materialized)
    camera_host = stage_optical_compute_result_to_host(camera_result)
    assert camera_result.output_profile is OpticalOutputProfile.RGB_PREVIEW
    assert set(camera_result.channels) == OpticalOutputProfile.RGB_PREVIEW.guaranteed_channels
    np.testing.assert_array_equal(camera_host.channel("hit_mask"), materialized_host.channel("hit_mask"))
    np.testing.assert_allclose(camera_host.channel("rgb"), materialized_host.channel("rgb"), atol=1e-5)
    np.testing.assert_array_equal(camera_host.channel("bvh_stack_overflow_count"), [0])
    np.testing.assert_array_equal(camera_host.channel("shadow_stack_overflow_count"), [0])


def test_device_bvh_refit_matches_fresh_build_for_triangle_grid():
    engine = _make_engine()
    consumer = _consumer()
    engine.register_consumer(consumer)
    engine.step()
    frame = engine.latest_published_frame()
    registry = _world_static_triangle_grid_registry()
    spec = OpticalRaySensorSpec(
        frame_id=frame.frame_id,
        sim_time=frame.sim_time,
        env_idx=0,
        sensor_id="gpu_camera",
        origins_world=[
            [-0.75, -0.75, 2.0],
            [-0.35, 0.05, 2.0],
            [0.45, 0.45, 2.0],
            [2.0, 2.0, 2.0],
        ],
        directions_world=[
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
        ],
        max_distance=10.0,
    )
    stream = wp.Stream(device=engine._device)
    borrowed = engine.borrow_device_frame(consumer.consumer_id, frame.frame_id, stream=stream)
    cache = DeviceOpticalSceneCache(registry, device=engine._device, stream=stream)
    snapshot = cache.snapshot_from_gpu_frame(borrowed, env_idx=0, stream=stream, include_aabb=True)
    topology_bvh = build_device_bvh_from_snapshot(snapshot, device=engine._device, stream=stream)
    fresh_bvh = build_device_bvh_from_snapshot(snapshot, device=engine._device, stream=stream)
    refit_bvh = refit_device_bvh_from_snapshot(snapshot, topology_bvh, stream=stream)
    executor = GpuDeviceBvhOpticalExecutor(device=engine._device, stream=stream)
    fresh = executor.execute(snapshot, fresh_bvh, spec)
    refit = executor.execute(snapshot, refit_bvh, spec)
    done_event = engine.complete_device_consumer(consumer.consumer_id, frame.frame_id, stream=stream)

    wp.synchronize_event(done_event)
    fresh_host = stage_optical_compute_result_to_host(fresh)
    refit_host = stage_optical_compute_result_to_host(refit)
    np.testing.assert_array_equal(refit_host.channel("hit_mask"), fresh_host.channel("hit_mask"))
    np.testing.assert_allclose(refit_host.channel("range_m"), fresh_host.channel("range_m"), atol=1e-6)
    np.testing.assert_allclose(
        refit_host.channel("position_world"),
        fresh_host.channel("position_world"),
        atol=1e-6,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        refit_host.channel("normal_world"),
        fresh_host.channel("normal_world"),
        atol=1e-6,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        refit_host.channel("numeric_instance_id"),
        fresh_host.channel("numeric_instance_id"),
    )
    np.testing.assert_array_equal(refit_host.channel("bvh_stack_overflow_count"), [0])


def test_device_bvh_no_shadow_direct_light_matches_cpu():
    engine = _make_engine()
    consumer = _consumer()
    engine.register_consumer(consumer)
    engine.step()
    frame = engine.latest_published_frame()
    registry = _lit_triangle_registry()
    spec = OpticalRaySensorSpec(
        frame_id=frame.frame_id,
        sim_time=frame.sim_time,
        env_idx=0,
        sensor_id="gpu_camera",
        origins_world=[[0.0, 0.0, 2.0], [2.0, 2.0, 2.0]],
        directions_world=[[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
        max_distance=10.0,
        sensor_role="rgb",
    )
    host_snapshot = OpticalSceneCache(registry).snapshot_from_frame_inputs(
        OpticalFrameInputs.from_published_frame(_cpu_frame_like(frame)),
        acceleration="cpu_bvh",
    )
    cpu = CpuDirectLightOpticalExecutor(
        shadows=False,
        ambient_rgb=(0.1, 0.2, 0.3),
        background_rgb=(0.01, 0.02, 0.03),
    ).execute(host_snapshot, spec)

    stream = wp.Stream(device=engine._device)
    borrowed = engine.borrow_device_frame(consumer.consumer_id, frame.frame_id, stream=stream)
    cache = DeviceOpticalSceneCache(registry, device=engine._device, stream=stream)
    snapshot = cache.snapshot_from_gpu_frame(borrowed, env_idx=0, stream=stream, include_aabb=True)
    bvh = build_device_bvh_from_snapshot(snapshot, device=engine._device, stream=stream)
    gpu = GpuDeviceBvhDirectLightOpticalExecutor(
        device=engine._device,
        stream=stream,
        shadows=False,
        ambient_rgb=(0.1, 0.2, 0.3),
        background_rgb=(0.01, 0.02, 0.03),
    ).execute(snapshot, bvh, spec)
    done_event = engine.complete_device_consumer(consumer.consumer_id, frame.frame_id, stream=stream)

    wp.synchronize_event(done_event)
    gpu_host = stage_optical_compute_result_to_host(gpu)
    np.testing.assert_array_equal(gpu_host.channel("hit_mask"), cpu.channel("hit_mask"))
    np.testing.assert_allclose(gpu_host.channel("range_m"), cpu.channel("range_m"), atol=1e-6)
    np.testing.assert_allclose(gpu_host.channel("rgb"), cpu.channel("rgb"), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        gpu_host.channel("intensity"),
        cpu.channel("intensity"),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_array_equal(gpu_host.channel("shadow_stack_overflow_count"), [0])
    np.testing.assert_array_equal(gpu_host.channel("shadow_max_stack_depth"), [0])


def test_device_bvh_no_shadow_point_light_matches_cpu():
    engine = _make_engine()
    consumer = _consumer()
    engine.register_consumer(consumer)
    engine.step()
    frame = engine.latest_published_frame()
    registry = _point_lit_triangle_registry()
    spec = OpticalRaySensorSpec(
        frame_id=frame.frame_id,
        sim_time=frame.sim_time,
        env_idx=0,
        sensor_id="gpu_camera",
        origins_world=[[0.0, 0.0, 2.0], [2.0, 2.0, 2.0]],
        directions_world=[[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
        max_distance=10.0,
        sensor_role="rgb",
    )
    host_snapshot = OpticalSceneCache(registry).snapshot_from_frame_inputs(
        OpticalFrameInputs.from_published_frame(_cpu_frame_like(frame)),
        acceleration="cpu_bvh",
    )
    cpu = CpuDirectLightOpticalExecutor(
        shadows=False,
        ambient_rgb=(0.05, 0.05, 0.05),
        background_rgb=(0.0, 0.0, 0.0),
    ).execute(host_snapshot, spec)

    stream = wp.Stream(device=engine._device)
    borrowed = engine.borrow_device_frame(consumer.consumer_id, frame.frame_id, stream=stream)
    cache = DeviceOpticalSceneCache(registry, device=engine._device, stream=stream)
    snapshot = cache.snapshot_from_gpu_frame(borrowed, env_idx=0, stream=stream, include_aabb=True)
    bvh = build_device_bvh_from_snapshot(snapshot, device=engine._device, stream=stream)
    gpu = GpuDeviceBvhDirectLightOpticalExecutor(
        device=engine._device,
        stream=stream,
        shadows=False,
        ambient_rgb=(0.05, 0.05, 0.05),
        background_rgb=(0.0, 0.0, 0.0),
    ).execute(snapshot, bvh, spec)
    done_event = engine.complete_device_consumer(consumer.consumer_id, frame.frame_id, stream=stream)

    wp.synchronize_event(done_event)
    gpu_host = stage_optical_compute_result_to_host(gpu)
    np.testing.assert_array_equal(gpu_host.channel("hit_mask"), cpu.channel("hit_mask"))
    np.testing.assert_allclose(gpu_host.channel("rgb"), cpu.channel("rgb"), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        gpu_host.channel("intensity"),
        cpu.channel("intensity"),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_array_equal(gpu_host.channel("shadow_stack_overflow_count"), [0])
    np.testing.assert_array_equal(gpu_host.channel("shadow_max_stack_depth"), [0])


def test_device_bvh_zero_light_direct_light_matches_cpu():
    engine = _make_engine()
    consumer = _consumer()
    engine.register_consumer(consumer)
    engine.step()
    frame = engine.latest_published_frame()
    registry = _zero_light_triangle_registry()
    spec = OpticalRaySensorSpec(
        frame_id=frame.frame_id,
        sim_time=frame.sim_time,
        env_idx=0,
        sensor_id="gpu_camera",
        origins_world=[[0.0, 0.0, 2.0], [2.0, 2.0, 2.0]],
        directions_world=[[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
        max_distance=10.0,
        sensor_role="rgb",
    )
    host_snapshot = OpticalSceneCache(registry).snapshot_from_frame_inputs(
        OpticalFrameInputs.from_published_frame(_cpu_frame_like(frame)),
        acceleration="cpu_bvh",
    )
    cpu = CpuDirectLightOpticalExecutor(
        shadows=True,
        ambient_rgb=(0.2, 0.1, 0.05),
        background_rgb=(0.01, 0.02, 0.03),
    ).execute(host_snapshot, spec)

    stream = wp.Stream(device=engine._device)
    borrowed = engine.borrow_device_frame(consumer.consumer_id, frame.frame_id, stream=stream)
    cache = DeviceOpticalSceneCache(registry, device=engine._device, stream=stream)
    snapshot = cache.snapshot_from_gpu_frame(borrowed, env_idx=0, stream=stream, include_aabb=True)
    bvh = build_device_bvh_from_snapshot(snapshot, device=engine._device, stream=stream)
    gpu = GpuDeviceBvhDirectLightOpticalExecutor(
        device=engine._device,
        stream=stream,
        shadows=True,
        ambient_rgb=(0.2, 0.1, 0.05),
        background_rgb=(0.01, 0.02, 0.03),
    ).execute(snapshot, bvh, spec)
    done_event = engine.complete_device_consumer(consumer.consumer_id, frame.frame_id, stream=stream)

    wp.synchronize_event(done_event)
    gpu_host = stage_optical_compute_result_to_host(gpu)
    np.testing.assert_array_equal(gpu_host.channel("hit_mask"), cpu.channel("hit_mask"))
    np.testing.assert_allclose(gpu_host.channel("rgb"), cpu.channel("rgb"), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        gpu_host.channel("intensity"),
        cpu.channel("intensity"),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(gpu_host.channel("rgb")[0], [0.06, 0.05, 0.035], atol=1e-6)
    np.testing.assert_allclose(gpu_host.channel("rgb")[1], [0.01, 0.02, 0.03], atol=1e-6)
    np.testing.assert_array_equal(gpu_host.channel("shadow_stack_overflow_count"), [0])
    np.testing.assert_array_equal(gpu_host.channel("shadow_max_stack_depth"), [0])


def test_device_bvh_multi_light_direct_light_matches_cpu():
    engine = _make_engine()
    consumer = _consumer()
    engine.register_consumer(consumer)
    engine.step()
    frame = engine.latest_published_frame()
    registry = _multi_light_triangle_registry()
    spec = OpticalRaySensorSpec(
        frame_id=frame.frame_id,
        sim_time=frame.sim_time,
        env_idx=0,
        sensor_id="gpu_camera",
        origins_world=[[0.0, 0.0, 2.0]],
        directions_world=[[0.0, 0.0, -1.0]],
        max_distance=10.0,
        sensor_role="rgb",
    )
    host_snapshot = OpticalSceneCache(registry).snapshot_from_frame_inputs(
        OpticalFrameInputs.from_published_frame(_cpu_frame_like(frame)),
        acceleration="cpu_bvh",
    )
    cpu = CpuDirectLightOpticalExecutor(
        shadows=False,
        ambient_rgb=(0.1, 0.0, 0.2),
        background_rgb=(0.0, 0.0, 0.0),
    ).execute(host_snapshot, spec)

    stream = wp.Stream(device=engine._device)
    borrowed = engine.borrow_device_frame(consumer.consumer_id, frame.frame_id, stream=stream)
    cache = DeviceOpticalSceneCache(registry, device=engine._device, stream=stream)
    snapshot = cache.snapshot_from_gpu_frame(borrowed, env_idx=0, stream=stream, include_aabb=True)
    bvh = build_device_bvh_from_snapshot(snapshot, device=engine._device, stream=stream)
    gpu = GpuDeviceBvhDirectLightOpticalExecutor(
        device=engine._device,
        stream=stream,
        shadows=False,
        ambient_rgb=(0.1, 0.0, 0.2),
        background_rgb=(0.0, 0.0, 0.0),
    ).execute(snapshot, bvh, spec)
    done_event = engine.complete_device_consumer(consumer.consumer_id, frame.frame_id, stream=stream)

    wp.synchronize_event(done_event)
    gpu_host = stage_optical_compute_result_to_host(gpu)
    np.testing.assert_array_equal(gpu_host.channel("hit_mask"), cpu.channel("hit_mask"))
    np.testing.assert_allclose(gpu_host.channel("rgb"), cpu.channel("rgb"), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        gpu_host.channel("intensity"),
        cpu.channel("intensity"),
        rtol=1e-5,
        atol=1e-5,
    )
    assert float(gpu_host.channel("intensity")[0]) > 0.0
    np.testing.assert_array_equal(gpu_host.channel("shadow_stack_overflow_count"), [0])
    np.testing.assert_array_equal(gpu_host.channel("shadow_max_stack_depth"), [0])


def test_device_bvh_shadowed_direct_light_matches_cpu():
    engine = _make_engine()
    consumer = _consumer()
    engine.register_consumer(consumer)
    engine.step()
    frame = engine.latest_published_frame()
    registry = _shadowed_floor_registry()
    spec = OpticalRaySensorSpec(
        frame_id=frame.frame_id,
        sim_time=frame.sim_time,
        env_idx=0,
        sensor_id="gpu_camera",
        origins_world=[[0.0, 0.0, 2.0]],
        directions_world=[[0.0, 0.0, -1.0]],
        max_distance=10.0,
        sensor_role="rgb",
    )
    host_snapshot = OpticalSceneCache(registry).snapshot_from_frame_inputs(
        OpticalFrameInputs.from_published_frame(_cpu_frame_like(frame)),
        acceleration="cpu_bvh",
    )
    cpu = CpuDirectLightOpticalExecutor(
        shadows=True,
        ambient_rgb=(0.1, 0.1, 0.1),
        background_rgb=(0.0, 0.0, 0.0),
    ).execute(host_snapshot, spec)

    stream = wp.Stream(device=engine._device)
    borrowed = engine.borrow_device_frame(consumer.consumer_id, frame.frame_id, stream=stream)
    cache = DeviceOpticalSceneCache(registry, device=engine._device, stream=stream)
    snapshot = cache.snapshot_from_gpu_frame(borrowed, env_idx=0, stream=stream, include_aabb=True)
    bvh = build_device_bvh_from_snapshot(snapshot, device=engine._device, stream=stream)
    gpu = GpuDeviceBvhDirectLightOpticalExecutor(
        device=engine._device,
        stream=stream,
        shadows=True,
        ambient_rgb=(0.1, 0.1, 0.1),
        background_rgb=(0.0, 0.0, 0.0),
    ).execute(snapshot, bvh, spec)
    done_event = engine.complete_device_consumer(consumer.consumer_id, frame.frame_id, stream=stream)

    wp.synchronize_event(done_event)
    gpu_host = stage_optical_compute_result_to_host(gpu)
    np.testing.assert_array_equal(gpu_host.channel("hit_mask"), cpu.channel("hit_mask"))
    np.testing.assert_allclose(gpu_host.channel("rgb"), cpu.channel("rgb"), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        gpu_host.channel("intensity"),
        cpu.channel("intensity"),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(gpu_host.channel("rgb")[0], [[0.07, 0.06, 0.05]][0], atol=1e-5)
    np.testing.assert_array_equal(gpu_host.channel("shadow_stack_overflow_count"), [0])
    assert 1 <= int(gpu_host.channel("shadow_max_stack_depth")[0]) <= 32


def test_device_bvh_point_light_shadow_matches_cpu():
    engine = _make_engine()
    consumer = _consumer()
    engine.register_consumer(consumer)
    engine.step()
    frame = engine.latest_published_frame()
    registry = _point_shadowed_floor_registry()
    spec = OpticalRaySensorSpec(
        frame_id=frame.frame_id,
        sim_time=frame.sim_time,
        env_idx=0,
        sensor_id="gpu_camera",
        origins_world=[[0.0, 0.0, 2.0]],
        directions_world=[[0.0, 0.0, -1.0]],
        max_distance=10.0,
        sensor_role="rgb",
    )
    host_snapshot = OpticalSceneCache(registry).snapshot_from_frame_inputs(
        OpticalFrameInputs.from_published_frame(_cpu_frame_like(frame)),
        acceleration="cpu_bvh",
    )
    cpu = CpuDirectLightOpticalExecutor(
        shadows=True,
        ambient_rgb=(0.05, 0.05, 0.05),
        background_rgb=(0.0, 0.0, 0.0),
    ).execute(host_snapshot, spec)

    stream = wp.Stream(device=engine._device)
    borrowed = engine.borrow_device_frame(consumer.consumer_id, frame.frame_id, stream=stream)
    cache = DeviceOpticalSceneCache(registry, device=engine._device, stream=stream)
    snapshot = cache.snapshot_from_gpu_frame(borrowed, env_idx=0, stream=stream, include_aabb=True)
    bvh = build_device_bvh_from_snapshot(snapshot, device=engine._device, stream=stream)
    gpu = GpuDeviceBvhDirectLightOpticalExecutor(
        device=engine._device,
        stream=stream,
        shadows=True,
        ambient_rgb=(0.05, 0.05, 0.05),
        background_rgb=(0.0, 0.0, 0.0),
    ).execute(snapshot, bvh, spec)
    done_event = engine.complete_device_consumer(consumer.consumer_id, frame.frame_id, stream=stream)

    wp.synchronize_event(done_event)
    gpu_host = stage_optical_compute_result_to_host(gpu)
    np.testing.assert_array_equal(gpu_host.channel("hit_mask"), cpu.channel("hit_mask"))
    np.testing.assert_allclose(gpu_host.channel("rgb"), cpu.channel("rgb"), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        gpu_host.channel("intensity"),
        cpu.channel("intensity"),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(gpu_host.channel("rgb")[0], [0.03, 0.02, 0.01], atol=1e-5)
    np.testing.assert_array_equal(gpu_host.channel("shadow_stack_overflow_count"), [0])
    assert 1 <= int(gpu_host.channel("shadow_max_stack_depth")[0]) <= 32


def test_device_bvh_plane_shadow_matches_cpu():
    engine = _make_engine()
    consumer = _consumer()
    engine.register_consumer(consumer)
    engine.step()
    frame = engine.latest_published_frame()
    registry = _plane_shadowed_floor_registry()
    spec = OpticalRaySensorSpec(
        frame_id=frame.frame_id,
        sim_time=frame.sim_time,
        env_idx=0,
        sensor_id="gpu_camera",
        origins_world=[[0.0, 0.0, 2.0]],
        directions_world=[[0.0, 0.0, -1.0]],
        max_distance=10.0,
        sensor_role="rgb",
    )
    host_snapshot = OpticalSceneCache(registry).snapshot_from_frame_inputs(
        OpticalFrameInputs.from_published_frame(_cpu_frame_like(frame)),
        acceleration="cpu_bvh",
    )
    cpu = CpuDirectLightOpticalExecutor(
        shadows=True,
        ambient_rgb=(0.05, 0.05, 0.05),
        background_rgb=(0.0, 0.0, 0.0),
    ).execute(host_snapshot, spec)

    stream = wp.Stream(device=engine._device)
    borrowed = engine.borrow_device_frame(consumer.consumer_id, frame.frame_id, stream=stream)
    cache = DeviceOpticalSceneCache(registry, device=engine._device, stream=stream)
    snapshot = cache.snapshot_from_gpu_frame(borrowed, env_idx=0, stream=stream, include_aabb=True)
    bvh = build_device_bvh_from_snapshot(snapshot, device=engine._device, stream=stream)
    gpu = GpuDeviceBvhDirectLightOpticalExecutor(
        device=engine._device,
        stream=stream,
        shadows=True,
        ambient_rgb=(0.05, 0.05, 0.05),
        background_rgb=(0.0, 0.0, 0.0),
    ).execute(snapshot, bvh, spec)
    done_event = engine.complete_device_consumer(consumer.consumer_id, frame.frame_id, stream=stream)

    wp.synchronize_event(done_event)
    gpu_host = stage_optical_compute_result_to_host(gpu)
    np.testing.assert_array_equal(gpu_host.channel("hit_mask"), cpu.channel("hit_mask"))
    np.testing.assert_allclose(gpu_host.channel("rgb"), cpu.channel("rgb"), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        gpu_host.channel("intensity"),
        cpu.channel("intensity"),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(gpu_host.channel("rgb")[0], [0.03, 0.02, 0.01], atol=1e-5)
    np.testing.assert_array_equal(gpu_host.channel("shadow_stack_overflow_count"), [0])
    np.testing.assert_array_equal(gpu_host.channel("shadow_max_stack_depth"), [0])


def test_device_scene_executor_unknown_role_returns_all_misses():
    engine = _make_engine()
    consumer = _consumer()
    engine.register_consumer(consumer)
    engine.step()
    frame = engine.latest_published_frame()
    spec = OpticalRaySensorSpec(
        frame_id=frame.frame_id,
        sim_time=frame.sim_time,
        env_idx=0,
        sensor_id="gpu_camera",
        origins_world=[[0.0, 0.0, 2.0]],
        directions_world=[[0.0, 0.0, -1.0]],
        max_distance=10.0,
        sensor_role="missing_role",
    )

    result, _, _, done_event = _execute_device_scene(
        engine,
        consumer_id=consumer.consumer_id,
        registry=_world_static_registry(),
        spec=spec,
        frame=frame,
    )

    wp.synchronize_event(done_event)
    staged = stage_optical_compute_result_to_host(result)
    np.testing.assert_array_equal(staged.channel("hit_mask"), [False])
    np.testing.assert_allclose(staged.channel("range_m"), [np.inf])
    np.testing.assert_array_equal(staged.channel("numeric_instance_id"), [0])


def test_device_scene_executor_supports_int64_role_mask_above_31_bits():
    engine = _make_engine()
    consumer = _consumer()
    engine.register_consumer(consumer)
    engine.step()
    frame = engine.latest_published_frame()
    spec = OpticalRaySensorSpec(
        frame_id=frame.frame_id,
        sim_time=frame.sim_time,
        env_idx=0,
        sensor_id="gpu_camera",
        origins_world=[[0.0, 0.0, 2.0]],
        directions_world=[[0.0, 0.0, -1.0]],
        max_distance=10.0,
        sensor_role="role_40",
    )

    result, _, _, done_event = _execute_device_scene(
        engine,
        consumer_id=consumer.consumer_id,
        registry=_many_role_registry(),
        spec=spec,
        frame=frame,
    )

    wp.synchronize_event(done_event)
    staged = stage_optical_compute_result_to_host(result)
    np.testing.assert_array_equal(staged.channel("hit_mask"), [True])
    np.testing.assert_allclose(staged.channel("range_m"), [2.0], atol=1e-6)
    np.testing.assert_array_equal(staged.channel("numeric_instance_id"), [41])


def test_execute_optical_on_gpu_published_frame_rejects_timeline_mismatch():
    engine = _make_engine()
    consumer = _consumer()
    engine.register_consumer(consumer)
    engine.step()
    frame = engine.latest_published_frame()
    spec = OpticalRaySensorSpec(
        frame_id=frame.frame_id + 1,
        sim_time=frame.sim_time,
        env_idx=0,
        sensor_id="gpu_camera",
        origins_world=[[0.0, 0.0, 2.0]],
        directions_world=[[0.0, 0.0, -1.0]],
        max_distance=10.0,
    )

    with pytest.raises(ValueError, match="frame_id"):
        execute_optical_on_gpu_published_frame(
            engine,
            consumer.consumer_id,
            _world_static_registry(),
            spec,
            frame=frame,
            stream=wp.Stream(device=engine._device),
        )
