from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.optical.eligibility import (
    CANONICAL_NATIVE_REPLAY_INPUT,
    verified_runtime_entry_point_audit,
)
from benchmarks.optical.instrumentation import (
    instrument_runtime_entry_points,
    invoke_instrumented_entry_point,
)
from benchmarks.optical.native import (
    CanonicalNativeInput,
    build_prepared_warp_bvh_refit_factory,
    build_prepared_warp_scene_update_factory,
    build_prepared_warp_transform_inputs,
    prepare_warp_native_input,
)
from benchmarks.optical.timing import RuntimeOperationTrace, native_timed_submission
from benchmarks.optical.workloads import (
    build_canonical_packed_scene,
    build_canonical_transform_replay,
)
from optics.device import stage_optical_compute_result_to_host
from optics.device_bvh import (
    WARP_BVH_REFIT_ACCELERATION_ID,
    WARP_BVH_REFIT_OPERATION_ID,
    WARP_BVH_TOPOLOGY_ORIGIN_ID,
    WarpBvhRefitCommandFactory,
    build_device_bvh_from_snapshot,
    refit_device_bvh_from_snapshot,
)
from optics.device_scene import (
    WARP_SCENE_UPDATE_OPERATION_ID,
    WarpSceneTransformInputs,
    WarpSceneUpdateCommandFactory,
    allocate_warp_world_scene_buffers,
    build_device_optical_scene,
    update_device_optical_scene_from_gpu_frame,
    warp_replay_transform_inputs,
)
from optics.registry import OpticalInstanceSpec, OpticalMaterialSpec, OpticalWorldRegistry
from optics.render_api import OpticalRaySensorSpec
from optics.warp_execution import GpuDeviceBvhOpticalExecutor

pytestmark = pytest.mark.gpu


def test_prebound_warp_scene_and_refit_use_exact_a2_inputs_without_timed_contamination(
    monkeypatch,
):
    wp = pytest.importorskip("warp")
    if wp.get_cuda_device_count() == 0:
        pytest.skip("CUDA is required")

    setup = _canonical_dynamic_setup(wp)
    scene_bundle = setup.scene_factory.seal(
        transforms=setup.frame_one,
        outputs=setup.frame_one_world,
    )
    refit_bundle = setup.refit_factory.seal(world=setup.frame_one_world)
    assert scene_bundle.evidence.operation_id == WARP_SCENE_UPDATE_OPERATION_ID
    assert refit_bundle.evidence.operation_id == WARP_BVH_REFIT_OPERATION_ID
    assert refit_bundle.evidence.acceleration_id == WARP_BVH_REFIT_ACCELERATION_ID
    assert refit_bundle.evidence.topology_origin_id == WARP_BVH_TOPOLOGY_ORIGIN_ID
    assert scene_bundle.evidence.native_input_manifest_sha256 == setup.prepared.manifest.sha256
    assert refit_bundle.evidence.native_input_manifest_sha256 == setup.prepared.manifest.sha256
    assert scene_bundle.evidence.scene_contract_sha256 == setup.scene.scene_contract_sha256
    assert refit_bundle.evidence.scene_contract_sha256 == setup.scene.scene_contract_sha256
    assert refit_bundle.evidence.native_input_topology_sha256 == (
        setup.prepared.manifest.topology_sha256
    )
    assert setup.refit_factory.topology.topology_sha256 is not None
    assert refit_bundle.evidence.topology_sha256 == setup.refit_factory.topology.topology_sha256
    assert scene_bundle.evidence.transform_replay_sha256 == (
        setup.prepared.manifest.transform_replay_sha256
    )
    assert scene_bundle.evidence.transform_frame_index == 1
    assert scene_bundle.evidence.stream_handle == refit_bundle.evidence.stream_handle
    topology_before = {
        name: getattr(setup.refit_factory.topology, name).numpy().copy()
        for name in ("left", "right", "start", "count", "node_depth", "prim_ids")
    }

    audits = (
        verified_runtime_entry_point_audit(
            operation="scene_update",
            entry_point="optics.device_scene.WarpSceneUpdateCommandBundle.enqueue",
        ),
        verified_runtime_entry_point_audit(
            operation="acceleration",
            entry_point="optics.device_bvh.WarpBvhRefitCommandBundle.enqueue",
        ),
    )

    def _recording_forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("timed enqueue must not call wp.launch")

    trace = RuntimeOperationTrace(row_id="unit_warp_r1", lane="warp", boundary="R1")
    with monkeypatch.context() as scoped:
        scoped.setattr(wp, "launch", _recording_forbidden)
        with instrument_runtime_entry_points(audits) as observation:
            with native_timed_submission(
                row_id="unit_warp_r1",
                lane="warp",
                boundary="R1",
                operation_trace=trace,
            ):
                invoke_instrumented_entry_point(
                    "optics.device_scene.WarpSceneUpdateCommandBundle.enqueue",
                    scene_bundle,
                )
                invoke_instrumented_entry_point(
                    "optics.device_bvh.WarpBvhRefitCommandBundle.enqueue",
                    refit_bundle,
                )

    assert observation.violation_hook_id is None
    evidence = trace.evidence()
    assert [(record.operation, record.entry_point) for record in evidence.records] == [
        (
            "scene_update",
            "optics.device_scene.WarpSceneUpdateCommandBundle.enqueue",
        ),
        (
            "acceleration",
            "optics.device_bvh.WarpBvhRefitCommandBundle.enqueue",
        ),
    ]
    done = setup.stream.record_event()
    wp.synchronize_event(done)

    legacy = update_device_optical_scene_from_gpu_frame(
        setup.scene,
        SimpleNamespace(
            frame_id=1,
            sim_time=1.0,
            x_world_R_wp=setup.frame_one.x_world_R,
            x_world_r_wp=setup.frame_one.x_world_r,
            ready_event=None,
        ),
        env_idx=0,
        stream=setup.stream,
        include_aabb=True,
    )
    wp.synchronize_event(legacy.ready_event)
    for name in (
        "triangle_v0_world",
        "triangle_e1_world",
        "triangle_e2_world",
        "triangle_normal_world",
        "triangle_aabb_min",
        "triangle_aabb_max",
        "plane_normal_world",
        "plane_point_world",
    ):
        np.testing.assert_allclose(
            getattr(setup.frame_one_world, name).numpy(),
            getattr(legacy, name).numpy(),
            rtol=0.0,
            atol=1.0e-6,
        )

    fresh_bvh = build_device_bvh_from_snapshot(
        legacy,
        leaf_size=4,
        split_strategy="sort",
        device=setup.scene.device,
        stream=setup.stream,
    )
    wp.synchronize_event(fresh_bvh.ready_event)
    np.testing.assert_allclose(
        setup.refit_factory.topology.bounds_min.numpy(),
        fresh_bvh.bounds_min.numpy(),
        rtol=0.0,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        setup.refit_factory.topology.bounds_max.numpy(),
        fresh_bvh.bounds_max.numpy(),
        rtol=0.0,
        atol=1.0e-6,
    )
    for name, before in topology_before.items():
        np.testing.assert_array_equal(getattr(setup.refit_factory.topology, name).numpy(), before)

    refit_view = replace(
        setup.refit_factory.topology,
        frame_id=legacy.frame_id,
        env_idx=legacy.env_idx,
        ready_event=None,
    )
    rays = OpticalRaySensorSpec(
        frame_id=legacy.frame_id,
        sim_time=legacy.sim_time,
        env_idx=legacy.env_idx,
        sensor_id="warp_c3b_parity",
        origins_world=[[0.05, 0.05, 2.0], [2.0, 2.0, 2.0]],
        directions_world=[[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
        max_distance=10.0,
    )
    executor = GpuDeviceBvhOpticalExecutor(device=setup.scene.device, stream=setup.stream)
    refit_result = executor.execute(legacy, refit_view, rays)
    fresh_result = executor.execute(legacy, fresh_bvh, rays)
    final_done = setup.stream.record_event()
    wp.synchronize_event(final_done)
    refit_host = stage_optical_compute_result_to_host(refit_result)
    fresh_host = stage_optical_compute_result_to_host(fresh_result)
    for name in (
        "hit_mask",
        "range_m",
        "position_world",
        "normal_world",
        "numeric_instance_id",
    ):
        np.testing.assert_allclose(
            refit_host.channel(name),
            fresh_host.channel(name),
            rtol=0.0,
            atol=1.0e-6,
            equal_nan=True,
        )


def test_warp_scene_factory_rejects_alias_and_stream_drift_before_recording():
    wp = pytest.importorskip("warp")
    if wp.get_cuda_device_count() == 0:
        pytest.skip("CUDA is required")

    setup = _canonical_dynamic_setup(wp)
    with pytest.raises(AttributeError, match="immutable"):
        setup.scene_factory.native_input_manifest_sha256 = "0" * 64
    with pytest.raises(AttributeError, match="immutable"):
        setup.refit_factory.topology = replace(
            setup.refit_factory.topology,
            scene_contract_sha256="0" * 64,
        )

    aliased = replace(
        setup.frame_one_world,
        triangle_e1_world=setup.frame_one_world.triangle_v0_world,
    )
    with pytest.raises(ValueError, match="overlap"):
        setup.scene_factory.seal(transforms=setup.frame_one, outputs=aliased)

    wrong_aabb_dtype = replace(
        setup.frame_one_world,
        triangle_aabb_min=wp.empty(
            (setup.scene.num_triangles, 3),
            dtype=wp.int32,
            device=setup.scene.device,
        ),
    )
    with pytest.raises(ValueError, match="dtype/rank"):
        setup.refit_factory.seal(world=wrong_aabb_dtype)

    slot_bounds_min = wp.empty_like(setup.refit_factory.topology.bounds_min)
    slot_bounds_max = wp.empty_like(setup.refit_factory.topology.bounds_max)
    slot_bvh = replace(
        setup.refit_factory.topology,
        bounds_min=slot_bounds_min,
        bounds_max=slot_bounds_max,
        ready_event=None,
        resources=(setup.refit_factory.topology, slot_bounds_min, slot_bounds_max),
    )
    slot_bundle = setup.refit_factory.seal(world=setup.frame_one_world, bvh=slot_bvh)
    assert slot_bundle.bvh is slot_bvh
    assert setup.refit_factory.topology.bounds_min is not slot_bounds_min
    setup.scene_factory.seal(
        transforms=setup.frame_one,
        outputs=setup.frame_one_world,
    ).enqueue()
    slot_bundle.enqueue()
    slot_done = setup.stream.record_event()
    wp.synchronize_event(slot_done)
    assert np.isfinite(slot_bounds_min.numpy()).all()
    assert np.isfinite(slot_bounds_max.numpy()).all()

    replaced_topology = replace(
        slot_bvh,
        left=wp.empty_like(slot_bvh.left),
    )
    with pytest.raises(ValueError, match="replaced immutable topology storage"):
        setup.refit_factory.seal(
            world=setup.frame_one_world,
            bvh=replaced_topology,
        )

    wrong_scene_topology = replace(
        setup.refit_factory.topology,
        scene_contract_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="different scene contract"):
        build_prepared_warp_bvh_refit_factory(
            setup.prepared,
            wrong_scene_topology,
            stream=setup.stream,
        )

    drifted_scene_factory = WarpSceneUpdateCommandFactory(
        setup.scene,
        stream=setup.stream,
        native_input_manifest_sha256=setup.prepared.manifest.sha256,
        input_contract=setup.prepared.manifest.input_contract,
    )
    object.__setattr__(
        drifted_scene_factory,
        "native_input_manifest_sha256",
        "0" * 64,
    )
    with pytest.raises(ValueError, match="provenance drifted"):
        drifted_scene_factory.seal(
            transforms=setup.frame_one,
            outputs=setup.frame_one_world,
        )

    drifted_refit_factory = build_prepared_warp_bvh_refit_factory(
        setup.prepared,
        setup.refit_factory.topology,
        stream=setup.stream,
    )
    object.__setattr__(drifted_refit_factory, "topology", wrong_scene_topology)
    with pytest.raises(ValueError, match="scene contract drifted"):
        drifted_refit_factory.seal(world=setup.frame_one_world)

    wrong_stream = wp.Stream(device=setup.scene.device)
    wrong_factory = WarpSceneUpdateCommandFactory(
        setup.scene,
        stream=wrong_stream,
        native_input_manifest_sha256=setup.prepared.manifest.sha256,
        input_contract=setup.prepared.manifest.input_contract,
    )
    wrong_bundle = wrong_factory.seal(
        transforms=setup.frame_one,
        outputs=allocate_warp_world_scene_buffers(setup.scene, stream=wrong_stream),
    )
    assert wrong_bundle.evidence.stream_handle != setup.scene_factory.stream.cuda_stream
    drifted_stream_factory = WarpSceneUpdateCommandFactory(
        setup.scene,
        stream=setup.stream,
        native_input_manifest_sha256=setup.prepared.manifest.sha256,
        input_contract=setup.prepared.manifest.input_contract,
    )
    object.__setattr__(drifted_stream_factory, "stream", wrong_stream)
    with pytest.raises(ValueError, match="stream drifted"):
        drifted_stream_factory.seal(
            transforms=setup.frame_one,
            outputs=setup.frame_one_world,
        )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        wrong_bundle.enqueue(stream=setup.stream)
    assert not hasattr(wrong_bundle, "rebind")
    assert not hasattr(setup.refit_factory.seal(world=setup.frame_one_world), "rebind")


def test_prebound_warp_internal_refit_matches_fresh_grid_bounds_and_traversal():
    wp = pytest.importorskip("warp")
    if wp.get_cuda_device_count() == 0:
        pytest.skip("CUDA is required")

    scene = build_device_optical_scene(_body_grid_registry(), device="cuda:0")
    stream = wp.Stream(device=scene.device)
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (2, 1, 3, 3)).copy()
    translations = np.zeros((2, 1, 3), dtype=np.float32)
    translations[1, 0] = [0.15, -0.1, 0.35]
    rotation_wp = wp.array(rotations, dtype=wp.float32, device=scene.device)
    translation_wp = wp.array(translations, dtype=wp.float32, device=scene.device)
    frame_zero = warp_replay_transform_inputs(
        rotation_wp,
        translation_wp,
        frame_index=0,
        input_contract="unit_grid_replay_v1",
        transform_replay_sha256="a" * 64,
    )
    frame_one = warp_replay_transform_inputs(
        rotation_wp,
        translation_wp,
        frame_index=1,
        input_contract="unit_grid_replay_v1",
        transform_replay_sha256="a" * 64,
    )
    assert isinstance(frame_zero, WarpSceneTransformInputs)
    scene_factory = WarpSceneUpdateCommandFactory(
        scene,
        stream=stream,
        native_input_manifest_sha256="b" * 64,
        input_contract="unit_grid_replay_v1",
    )
    world_zero = allocate_warp_world_scene_buffers(scene, stream=stream)
    scene_factory.seal(transforms=frame_zero, outputs=world_zero).enqueue()
    zero_done = stream.record_event()
    wp.synchronize_event(zero_done)
    snapshot_zero = world_zero.as_snapshot(scene, frame_id=0, sim_time=0.0, env_idx=0)
    topology = build_device_bvh_from_snapshot(
        snapshot_zero,
        leaf_size=4,
        split_strategy="sort",
        stream=stream,
    )
    reference_topology = build_device_bvh_from_snapshot(
        snapshot_zero,
        leaf_size=4,
        split_strategy="sort",
        stream=stream,
    )
    assert topology.topology_sha256 is not None
    assert topology.topology_sha256 == reference_topology.topology_sha256
    topology_before = {
        name: getattr(topology, name).numpy().copy()
        for name in ("left", "right", "start", "count", "node_depth", "prim_ids")
    }

    world_one = allocate_warp_world_scene_buffers(scene, stream=stream)
    scene_bundle = scene_factory.seal(transforms=frame_one, outputs=world_one)
    refit_factory = WarpBvhRefitCommandFactory(
        topology,
        stream=stream,
        native_input_manifest_sha256="b" * 64,
        native_input_topology_sha256="c" * 64,
        scene_contract_sha256=scene.scene_contract_sha256,
        input_contract="unit_grid_replay_v1",
    )
    refit_bundle = refit_factory.seal(world=world_one)
    assert topology.stats.max_depth >= 1
    assert refit_bundle.dispatch_count == topology.stats.max_depth + 1
    scene_bundle.enqueue()
    refit_bundle.enqueue()
    refit_done = stream.record_event()
    wp.synchronize_event(refit_done)

    snapshot_one = world_one.as_snapshot(scene, frame_id=1, sim_time=1.0, env_idx=0)
    fresh = build_device_bvh_from_snapshot(
        snapshot_one,
        leaf_size=4,
        split_strategy="sort",
        stream=stream,
    )
    reference_refit = refit_device_bvh_from_snapshot(
        snapshot_one,
        reference_topology,
        stream=stream,
    )
    wp.synchronize_event(reference_refit.ready_event)
    np.testing.assert_allclose(
        topology.bounds_min.numpy(), reference_refit.bounds_min.numpy(), atol=1.0e-6
    )
    np.testing.assert_allclose(
        topology.bounds_max.numpy(), reference_refit.bounds_max.numpy(), atol=1.0e-6
    )
    for name, before in topology_before.items():
        np.testing.assert_array_equal(getattr(topology, name).numpy(), before)

    refit_view = replace(topology, frame_id=1, env_idx=0, ready_event=None)
    rays = OpticalRaySensorSpec(
        frame_id=1,
        sim_time=1.0,
        env_idx=0,
        sensor_id="warp_c3b_grid",
        origins_world=[[-0.55, -0.65, 2.0], [0.65, 0.55, 2.0], [2.0, 2.0, 2.0]],
        directions_world=[[0.0, 0.0, -1.0]] * 3,
        max_distance=10.0,
    )
    executor = GpuDeviceBvhOpticalExecutor(device=scene.device, stream=stream)
    refit_result = executor.execute(snapshot_one, refit_view, rays)
    fresh_result = executor.execute(snapshot_one, fresh, rays)
    traversal_done = stream.record_event()
    wp.synchronize_event(traversal_done)
    refit_host = stage_optical_compute_result_to_host(refit_result)
    fresh_host = stage_optical_compute_result_to_host(fresh_result)
    np.testing.assert_array_equal(
        refit_host.channel("hit_mask"), fresh_host.channel("hit_mask")
    )
    np.testing.assert_allclose(
        refit_host.channel("range_m"),
        fresh_host.channel("range_m"),
        atol=1.0e-6,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        refit_host.channel("numeric_instance_id"),
        fresh_host.channel("numeric_instance_id"),
    )


def test_prebound_warp_empty_triangle_path_records_no_dummy_refit_command():
    wp = pytest.importorskip("warp")
    if wp.get_cuda_device_count() == 0:
        pytest.skip("CUDA is required")

    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat_plane"))
    registry.add_plane_geometry(
        "floor",
        normal_local=[0.0, 0.0, 1.0],
        point_local=[0.0, 0.0, 0.0],
    )
    registry.add_instance(OpticalInstanceSpec("floor_instance", "floor", "mat_plane"))
    scene = build_device_optical_scene(registry, device="cuda:0")
    stream = wp.Stream(device=scene.device)
    rotation_wp = wp.array(
        np.empty((1, 0, 3, 3), dtype=np.float32),
        dtype=wp.float32,
        device=scene.device,
    )
    translation_wp = wp.array(
        np.empty((1, 0, 3), dtype=np.float32),
        dtype=wp.float32,
        device=scene.device,
    )
    transforms = warp_replay_transform_inputs(
        rotation_wp,
        translation_wp,
        frame_index=0,
        input_contract="unit_plane_replay_v1",
        transform_replay_sha256="d" * 64,
    )
    world = allocate_warp_world_scene_buffers(scene, stream=stream)
    scene_bundle = WarpSceneUpdateCommandFactory(
        scene,
        stream=stream,
        native_input_manifest_sha256="e" * 64,
        input_contract="unit_plane_replay_v1",
    ).seal(transforms=transforms, outputs=world)
    assert scene_bundle.dispatch_count == 1
    scene_bundle.enqueue()
    scene_done = stream.record_event()
    wp.synchronize_event(scene_done)
    snapshot = world.as_snapshot(scene, frame_id=0, sim_time=0.0, env_idx=0)
    bvh = build_device_bvh_from_snapshot(snapshot, stream=stream)
    refit_bundle = WarpBvhRefitCommandFactory(
        bvh,
        stream=stream,
        native_input_manifest_sha256="e" * 64,
        native_input_topology_sha256="f" * 64,
        scene_contract_sha256=scene.scene_contract_sha256,
        input_contract="unit_plane_replay_v1",
    ).seal(world=world)
    assert bvh.num_nodes == 0
    assert bvh.num_primitives == 0
    assert refit_bundle.dispatch_count == 0
    assert refit_bundle.evidence.dispatch_count == 0
    refit_bundle.enqueue()


@pytest.mark.parametrize(
    ("leaf_size", "split_strategy"),
    ((1, "sort"), (2, "sort"), (8, "sort"), (4, "partition")),
)
def test_warp_refit_factory_rejects_noncanonical_topology_contract(
    leaf_size,
    split_strategy,
):
    wp = pytest.importorskip("warp")
    if wp.get_cuda_device_count() == 0:
        pytest.skip("CUDA is required")

    with pytest.raises(ValueError, match="frozen leaf-4 topology origin"):
        _canonical_dynamic_setup(
            wp,
            leaf_size=leaf_size,
            split_strategy=split_strategy,
        )


def _canonical_dynamic_setup(
    wp,
    *,
    leaf_size: int = 4,
    split_strategy: str = "sort",
):
    packed = build_canonical_packed_scene("synthetic_dynamic_rigid_960x640_v1")
    replay = build_canonical_transform_replay("synthetic_dynamic_rigid_960x640_v1")
    source = CanonicalNativeInput.from_scene_contract(
        packed,
        input_contract=CANONICAL_NATIVE_REPLAY_INPUT,
        replay=replay,
    )
    prepared = prepare_warp_native_input(source, device="cuda:0")
    stream = wp.Stream(device=f"cuda:{prepared.device_ordinal}")
    scene_factory = build_prepared_warp_scene_update_factory(prepared, stream=stream)
    scene = scene_factory.local_scene
    frame_zero = build_prepared_warp_transform_inputs(prepared, 0)
    frame_one = build_prepared_warp_transform_inputs(prepared, 1)

    frame_zero_world = allocate_warp_world_scene_buffers(scene, stream=stream)
    scene_factory.seal(transforms=frame_zero, outputs=frame_zero_world).enqueue()
    zero_done = stream.record_event()
    wp.synchronize_event(zero_done)
    frame_zero_snapshot = frame_zero_world.as_snapshot(
        scene,
        frame_id=0,
        sim_time=0.0,
        env_idx=0,
    )
    bvh = build_device_bvh_from_snapshot(
        frame_zero_snapshot,
        leaf_size=leaf_size,
        split_strategy=split_strategy,
        device=scene.device,
        stream=stream,
    )
    refit_factory = build_prepared_warp_bvh_refit_factory(
        prepared,
        bvh,
        stream=stream,
    )
    return SimpleNamespace(
        prepared=prepared,
        scene=scene,
        stream=stream,
        frame_one=frame_one,
        frame_one_world=allocate_warp_world_scene_buffers(scene, stream=stream),
        scene_factory=scene_factory,
        refit_factory=refit_factory,
    )


def _body_grid_registry() -> OpticalWorldRegistry:
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
    registry.add_instance(
        OpticalInstanceSpec(
            "grid_instance",
            "grid",
            "mat_grid",
            body_index=0,
        )
    )
    return registry
