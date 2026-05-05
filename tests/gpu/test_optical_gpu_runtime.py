from __future__ import annotations

import numpy as np
import pytest

from optics import (
    DeviceOpticalSceneCache,
    GpuBruteForceOpticalExecutor,
    GpuDeviceBvhOpticalExecutor,
    GpuDeviceSceneOpticalExecutor,
    OpticalInstanceSpec,
    OpticalMaterialSpec,
    OpticalSceneCache,
    OpticalWorldRegistry,
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
from sensing import OpticalRaySensorSpec

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
    host_snapshot = OpticalSceneCache(registry).snapshot_from_published_frame(_cpu_frame_like(frame))

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
    np.testing.assert_array_equal(bvh_host.channel("bvh_stack_overflow_count"), [0])
    assert 1 <= int(bvh_host.channel("bvh_max_stack_depth")[0]) <= 32


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
