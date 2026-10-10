"""Production dynamic recording with real custody and CPU native-return doubles."""

from types import SimpleNamespace as NS

import pytest

from optical_worker import native_setup
from optics import device_bvh, device_scene
from tests.unit.optics.test_native_output_allocation import allocation as allocation


@pytest.mark.parametrize("allocation", ["warp"], indirect=True)
@pytest.mark.parametrize("fault", [None, "wait", "foreign", "unadopted"])
def test_refit_ready_dependency_precedes_recording(allocation, fault):  # noqa: F811
    p = allocation
    event = NS(device=object() if fault == "foreign" else p.target)
    if fault != "unadopted":
        p.controller.adopt_host_owner(event, role="fixture_event", operation_id="fixture_event")
    actions = []

    def wait(actual):
        assert actual is event and p.state.device is p.target and p.state.stream is p.stream
        actions.append("wait")
        if fault == "wait":
            raise p.failure

    def launch(*args, **kwargs):
        assert actions == ["wait"]
        actions.append("record")
        return p.runtime.empty((1,), dtype="command", device=p.target)

    p.stream.wait_event, p.runtime.launch = wait, launch

    def run():
        with p.controller.segment():
            context = native_setup._create_upload_context(p.controller, p.runtime, p.target, p.stream)
            with native_setup._warp_command_recording(
                p.runtime, p.target, p.stream, context=context, ready_event=event
            ) as record:
                return record(object(), dim=1, inputs=())

    if fault is None:
        p.controller.require_adopted(run())
        assert actions == ["wait", "record"]
    else:
        with pytest.raises((ValueError, RuntimeError)) as caught:
            run()
        if fault == "wait":
            assert caught.value is p.failure
        assert actions == (["wait"] if fault == "wait" else [])
        assert not p.state.arrays
        assert p.controller.guardian.state == "TRANSFERRED_TO_QUARANTINE"
    assert p.state.device == 0 and p.state.stream is None


@pytest.mark.parametrize("allocation", ["warp"], indirect=True)
@pytest.mark.parametrize("fault", [None, "allocation", "metadata", "adoption"])
def test_scene_recording_retains_partial_commands_and_original_failure(allocation, monkeypatch, fault):  # noqa: F811
    p = allocation
    p.state.fault = fault
    p.stream.cuda_stream = 7
    monkeypatch.setattr(device_scene, "wp", p.runtime)
    # Only geometric/storage validation is doubled here. Recording, returned
    # owner adoption, guard failure and scope restoration are production code.
    monkeypatch.setattr(
        device_scene.WarpSceneUpdateCommandFactory, "_validate_frozen_state", lambda self: None
    )
    for name in (
        "_validate_warp_transform_inputs",
        "_validate_warp_world_scene_buffers",
        "_validate_disjoint_warp_storage",
    ):
        monkeypatch.setattr(device_scene, name, lambda *args: None)
    for name in ("_warp_scene_command_inputs", "_warp_world_scene_arrays", "_warp_local_scene_arrays"):
        monkeypatch.setattr(device_scene, name, lambda *args: ())
    kernels = (object(), object())
    for name, kernel in zip(
        ("_update_triangles_derived_aabb_world_kernel", "_update_planes_world_kernel"), kernels
    ):
        monkeypatch.setattr(device_scene, name, kernel, raising=False)
    entered = []

    def launch(kernel, *, dim, inputs, device, stream, record_cmd, record_tape):
        assert device is p.target and stream is p.stream
        assert record_cmd is True and record_tape is False
        if entered:
            p.controller.require_adopted(*p.state.arrays)
        entered.append(kernel)
        return p.runtime.empty((dim,), dtype="command", device=device)

    p.runtime.launch = launch
    scene = NS(device=p.target, num_triangles=1, num_planes=1, scene_contract_sha256="a" * 64)
    for name in (
        "geometry_vertices_local",
        "triangle_vertex_indices",
        "triangle_instance_index",
        "instance_body_index",
        "instance_X_body_geometry_R",
        "instance_X_body_geometry_r",
        "plane_normal_local",
        "plane_point_local",
        "plane_instance_index",
    ):
        setattr(scene, name, object())
    outputs = NS(
        **{
            name: object()
            for name in (
                "triangle_v0_world",
                "triangle_e1_world",
                "triangle_e2_world",
                "triangle_normal_world",
                "triangle_aabb_min",
                "triangle_aabb_max",
                "plane_normal_world",
                "plane_point_world",
            )
        }
    )
    transforms = NS(
        input_contract="canonical_native_replay_v1",
        x_world_R=object(),
        x_world_r=object(),
        env_idx=0,
        transform_frame_index=0,
        transform_replay_sha256="b" * 64,
    )
    factory = object.__new__(device_scene.WarpSceneUpdateCommandFactory)
    factory.local_scene, factory.stream = scene, p.stream
    factory.input_contract, factory.native_input_manifest_sha256 = transforms.input_contract, "c" * 64

    def run():
        with p.controller.segment():
            context = native_setup._create_upload_context(p.controller, p.runtime, p.target, p.stream)
            return factory.seal(transforms=transforms, outputs=outputs, _context=context)

    if fault is None:
        bundle = run()
        assert bundle._commands == tuple(p.state.arrays)
        assert bundle.evidence.dispatch_count == 2
        p.controller.require_adopted(*bundle._commands)
    else:
        with pytest.raises(RuntimeError) as caught:
            run()
        assert caught.value is p.failure
        assert p.controller.guardian.state == "TRANSFERRED_TO_QUARANTINE"
        summary = p.controller.termination_evidence.lifecycle.to_payload()["summary"]
        assert summary["normal_complete_count"] == summary["emergency_sync_attempt_count"] == 0
    assert entered == list(kernels)
    assert p.state.device == 0 and p.state.stream is None
    assert all(
        any(
            type(root) is list and any(value is command for value in root)
            for root in p.controller.guardian._owners
        )
        for command in p.state.arrays
    )


@pytest.mark.parametrize("allocation", ["warp"], indirect=True)
@pytest.mark.parametrize("fault", [None, "allocation", "metadata", "adoption"])
def test_refit_recording_keeps_leaf_before_internal_failure(allocation, monkeypatch, fault):  # noqa: F811
    p = allocation
    p.state.fault = fault
    p.stream.cuda_stream = 7
    p.runtime.get_device = lambda value: value
    monkeypatch.setattr(device_bvh, "wp", p.runtime)
    monkeypatch.setattr(device_bvh.WarpBvhRefitCommandFactory, "_validate_frozen_state", lambda self: None)
    for name in (
        "_validate_matching_warp_bvh_topology",
        "_validate_warp_array",
        "_validate_disjoint_warp_storage",
    ):
        monkeypatch.setattr(device_bvh, name, lambda *args, **kwargs: None)
    for name in ("_warp_refit_input_arrays", "_warp_refit_output_arrays", "_warp_world_scene_arrays"):
        monkeypatch.setattr(device_bvh, name, lambda *args: ())
    kernels = (object(), object())
    for name, kernel in zip(("_refit_bvh_leaf_bounds_kernel", "_refit_bvh_internal_bounds_kernel"), kernels):
        monkeypatch.setattr(device_bvh, name, kernel, raising=False)
    entered = []

    def launch(kernel, *, dim, inputs, device, stream, record_cmd, record_tape):
        assert device is p.target and stream is p.stream
        assert record_cmd is True and record_tape is False
        if entered:
            p.controller.require_adopted(*p.state.arrays)
        entered.append(kernel)
        return p.runtime.empty((dim,), dtype="command", device=device)

    p.runtime.launch = launch
    world = device_scene.WarpWorldSceneBuffers(
        device=p.target,
        **{
            name: object()
            for name in (
                "triangle_v0_world",
                "triangle_e1_world",
                "triangle_e2_world",
                "triangle_normal_world",
                "triangle_aabb_min",
                "triangle_aabb_max",
                "plane_normal_world",
                "plane_point_world",
            )
        },
    )
    bvh = NS(
        device=p.target,
        num_primitives=1,
        num_nodes=2,
        ready_event=None,
        stats=NS(max_depth=1, level_ranges=((0, 1),), num_leaves=1),
        topology_origin_id=device_bvh.WARP_BVH_TOPOLOGY_ORIGIN_ID,
        topology_sha256="a" * 64,
    )
    for name in ("start", "count", "prim_ids", "bounds_min", "bounds_max", "left", "right"):
        setattr(bvh, name, object())
    factory = object.__new__(device_bvh.WarpBvhRefitCommandFactory)
    factory.topology, factory.stream = bvh, p.stream
    factory.native_input_manifest_sha256 = factory.native_input_topology_sha256 = "b" * 64
    factory.scene_contract_sha256, factory.input_contract = "c" * 64, "canonical_native_replay_v1"

    def run():
        with p.controller.segment():
            context = native_setup._create_upload_context(p.controller, p.runtime, p.target, p.stream)
            return factory.seal(world=world, _context=context)

    if fault is None:
        bundle = run()
        assert bundle._commands == tuple(p.state.arrays)
        assert bundle.evidence.dispatch_count == 2
        p.controller.require_adopted(*bundle._commands)
    else:
        with pytest.raises(RuntimeError) as caught:
            run()
        assert caught.value is p.failure
        assert p.controller.guardian.state == "TRANSFERRED_TO_QUARANTINE"
        summary = p.controller.termination_evidence.lifecycle.to_payload()["summary"]
        assert summary["normal_complete_count"] == summary["emergency_sync_attempt_count"] == 0
    assert entered == list(kernels)
    assert p.state.device == 0 and p.state.stream is None
    assert all(
        any(
            type(root) is list and any(value is command for value in root)
            for root in p.controller.guardian._owners
        )
        for command in p.state.arrays
    )
