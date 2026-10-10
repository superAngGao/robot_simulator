"""Recording/preflight only: no recorded render command executes in pytest."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from benchmarks.optical.native import (
    CanonicalNativeInput,
    build_prepared_warp_local_scene,
    certify_native_readiness,
    prepare_warp_native_input,
    validate_prepared_native_input,
)
from benchmarks.optical.workloads import build_canonical_fixed_fixture, build_canonical_packed_scene
from optics.device_bvh import DeviceBvhBuildStats, DeviceOpticalBvh
from optics.device_scene import WarpWorldSceneBuffers
from optics.execution import OpticalOutputProfile
from optics.output_contract import SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES
from optics.warp_native_render import (
    WarpCameraParameters,
    WarpDirectLightBuffers,
    WarpDirectLightRenderCommandFactory,
    warp_direct_light_buffer_layout,
)
from sensing.optical import OpticalPinholeCameraSpec

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def wp():
    runtime = pytest.importorskip("warp")
    runtime.init()
    if not runtime.get_cuda_device_count():
        pytest.skip("CUDA required")
    return runtime


def _setup(wp, device="cuda:0", *, replay=False):
    if replay:
        from benchmarks.optical.native import (
            build_prepared_warp_scene_update_factory,
            build_prepared_warp_transform_inputs,
        )
        from benchmarks.optical.workloads import build_canonical_transform_replay
        from optics.device_bvh import build_device_bvh_from_snapshot
        from optics.device_scene import allocate_warp_world_scene_buffers

        packed = build_canonical_packed_scene("synthetic_dynamic_rigid_960x640_v1")
        sequence = build_canonical_transform_replay(packed.source_contract_id)
        source = CanonicalNativeInput.from_scene_contract(
            packed, input_contract="canonical_native_replay_v1", replay=sequence
        )
        prepared = prepare_warp_native_input(source, device=device)
        stream = wp.Stream(device=device)
        c3_factory = build_prepared_warp_scene_update_factory(prepared, stream=stream)
        scene = c3_factory.local_scene
        world = allocate_warp_world_scene_buffers(scene, stream=stream)
        c3_factory.seal(transforms=build_prepared_warp_transform_inputs(prepared, 0), outputs=world).enqueue()
        wp.synchronize_stream(stream)
        bvh = build_device_bvh_from_snapshot(
            world.as_snapshot(scene, frame_id=0, sim_time=0.0, env_idx=0), device=scene.device, stream=stream
        )
        validation = validate_prepared_native_input(source, prepared)
        wp.load_module(module="optics.warp_native_render", device=device)
        readiness = certify_native_readiness(
            prepared, validation, compile_complete=True, warmup_frames_completed=0
        )
        camera = WarpCameraParameters.from_spec(
            OpticalPinholeCameraSpec(
                frame_id=0,
                sim_time=0.0,
                env_idx=0,
                sensor_id="test",
                width=7,
                height=3,
                fx=3.0,
                fy=4.0,
                cx=1.1,
                cy=0.7,
                max_distance=100.0,
                sensor_role="depth",
            ),
            role_table=scene.role_table,
        )
        factory = WarpDirectLightRenderCommandFactory.prepare(
            prepared_input=prepared,
            readiness=readiness,
            local_scene=scene,
            device=scene.device,
            stream=stream,
        )
        return SimpleNamespace(
            prepared=prepared,
            readiness=readiness,
            scene=scene,
            stream=stream,
            world=world,
            bvh=bvh,
            camera=camera,
            factory=factory,
        )
    packed = build_canonical_packed_scene("asymmetric_synthetic_7x3_v1")
    fixture = build_canonical_fixed_fixture(packed)
    source = CanonicalNativeInput.from_scene_contract(
        packed, input_contract="fixed_native_scene_v1", fixed_fixture=fixture
    )
    prepared = prepare_warp_native_input(source, device=device)
    validation = validate_prepared_native_input(source, prepared)
    # This test records only; zero execution warmup frames is represented honestly.
    wp.load_module(module="optics.warp_native_render", device=device)
    readiness = certify_native_readiness(
        prepared, validation, compile_complete=True, warmup_frames_completed=0
    )
    scene = build_prepared_warp_local_scene(prepared)
    stream = wp.Stream(device=scene.device)
    arrays = prepared.arrays
    world_names = (
        "triangle_v0_world",
        "triangle_e1_world",
        "triangle_e2_world",
        "triangle_normal_world",
        "plane_normal_world",
        "plane_point_world",
        "triangle_aabb_min",
        "triangle_aabb_max",
    )
    world = WarpWorldSceneBuffers(
        device=scene.device, **{n: arrays[n] for n in world_names}, resources=(prepared,)
    )
    host = fixture.arrays
    levels = tuple(tuple(int(v) for v in row) for row in host["bvh_level_ranges"])
    stats = DeviceBvhBuildStats(
        num_nodes=len(host["bvh_left"]),
        num_leaves=int((host["bvh_count"] > 0).sum()),
        max_depth=len(levels) - 1 if levels else 0,
        leaf_size=4,
        sah_quality_cost=0.0,
        level_ranges=levels,
    )
    bvh = DeviceOpticalBvh(
        device=scene.device,
        frame_id=0,
        env_idx=0,
        stats=stats,
        **{
            n: arrays[f"bvh_{n}"]
            for n in (
                "bounds_min",
                "bounds_max",
                "left",
                "right",
                "start",
                "count",
                "node_depth",
                "level_ranges",
                "prim_ids",
            )
        },
        primitive_global_id=scene.triangle_primitive_global_id,
        primitive_instance_index=scene.triangle_instance_index,
        primitive_index_within_instance=scene.triangle_primitive_index_within_instance,
        primitive_geometry_index=scene.triangle_geometry_index,
        primitive_geometry_primitive_index=scene.triangle_geometry_primitive_index,
        primitive_source_order_key=scene.triangle_source_order_key,
        scene_contract_sha256=scene.scene_contract_sha256,
        resources=(prepared,),
    )
    camera = WarpCameraParameters.from_spec(
        OpticalPinholeCameraSpec(
            frame_id=0,
            sim_time=0.0,
            env_idx=0,
            sensor_id="test",
            width=7,
            height=3,
            fx=3.0,
            fy=4.0,
            cx=1.1,
            cy=0.7,
            max_distance=100.0,
            sensor_role="depth",
        ),
        role_table=scene.role_table,
    )
    factory = WarpDirectLightRenderCommandFactory.prepare(
        prepared_input=prepared, readiness=readiness, local_scene=scene, device=scene.device, stream=stream
    )
    return SimpleNamespace(
        prepared=prepared,
        readiness=readiness,
        scene=scene,
        stream=stream,
        world=world,
        bvh=bvh,
        camera=camera,
        factory=factory,
    )


def _buffers(wp, setup, diagnostics=False, n=21):
    arrays = {
        name: wp.empty(shape, dtype=getattr(wp, dtype), device=setup.scene.device)
        for name, dtype, shape in warp_direct_light_buffer_layout(n, traversal_diagnostics=diagnostics)
    }
    counters = tuple(arrays.pop(n) for n in SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES) if diagnostics else ()
    return WarpDirectLightBuffers(
        num_rays=n, traversal_diagnostics=diagnostics, traversal_counters=counters, **arrays
    )


def _seal(setup, buffers, **kwargs):
    options = dict(
        world=setup.world,
        bvh=setup.bvh,
        camera=setup.camera,
        buffers=buffers,
        output_profile=OpticalOutputProfile.DIRECT_LIGHT_FULL,
        shadows=True,
        traversal_diagnostics=buffers.traversal_diagnostics,
    )
    options.update(kwargs)
    return setup.factory.seal_camera(**options)


def _no_execution(*args, **kwargs):
    raise AssertionError("recording tests cannot execute recorded GPU commands")


@pytest.mark.parametrize("shadows", [False, True])
@pytest.mark.parametrize("diagnostics", [False, True])
@pytest.mark.parametrize(
    "profile",
    list(
        (
            OpticalOutputProfile.DIRECT_LIGHT_FULL,
            OpticalOutputProfile.RGB_PREVIEW,
            OpticalOutputProfile.RENDER_ONLY,
        )
    ),
)
def test_frozen_sequence_compares_actual_warp_packed_scalars(wp, monkeypatch, shadows, diagnostics, profile):
    import struct

    from optical_worker.render_parameters import bind_render_parameters, load_frozen_render_sequence
    from optics.warp_native_render import WarpDirectLightRenderCommandBundle, WarpRenderQueryParameters

    setup = _setup(wp)
    record = bind_render_parameters(
        load_frozen_render_sequence(setup.prepared.manifest.case_id),
        sample_index=0,
        profile=profile.value,
        shadows=shadows,
        pass_mode="diagnostic_profiled" if diagnostics else "primary_unprofiled",
        role_to_mask=setup.prepared.source.packed_scene_owner.role_to_mask,
    )
    payload = record.to_payload()
    from optical_worker.render_parameters import decode_native_scalars

    values = decode_native_scalars(payload)
    query = WarpRenderQueryParameters(
        num_rays=21,
        frame_id=1,
        sim_time=0.0,
        env_idx=0,
        sensor_id=payload["sensor_id"],
        max_distance=values[16],
        sensor_role="rgb",
        sensor_role_mask=payload["sensor_role_mask"],
    )
    setup.camera = WarpCameraParameters(
        query=query,
        width=7,
        height=3,
        fx=values[0],
        fy=values[1],
        cx=values[2],
        cy=values[3],
        origin=values[4:7],
        rotation=values[7:16],
    )
    setup.bvh = replace(setup.bvh, frame_id=1)
    buffers = _buffers(wp, setup, diagnostics)
    monkeypatch.setattr(wp.Launch, "launch", _no_execution)
    bundle = _seal(
        setup,
        buffers,
        shadows=shadows,
        output_profile=profile,
        ambient_rgb=values[17:20],
        background_rgb=values[20:23],
        shadow_bias=values[23],
    )
    bundle.validate_resolved_parameters(record)
    scalar_checks = 0
    for command, descriptor in zip(bundle.commands, bundle.descriptors, strict=True):
        for index, argument in enumerate(descriptor.arguments):
            if type(argument) is float:
                parameter = command.params[index + 1]
                previous = parameter.value
                parameter.value = previous + 1.0
                try:
                    with pytest.raises(ValueError, match="scalar|changed"):
                        bundle.validate_resolved_parameters(record)
                finally:
                    parameter.value = previous
                scalar_checks += 1
    assert scalar_checks == (24 if shadows else 23)
    bundle.validate_resolved_parameters(record)
    signed_zero_checks = 0
    for command, descriptor in zip(bundle.commands, bundle.descriptors, strict=True):
        for index, argument in enumerate(descriptor.arguments):
            if type(argument) is float and argument == 0.0:
                parameter = command.params[index + 1]
                previous = parameter.value
                previous_bits = bytes(parameter)
                parameter.value = -0.0 if previous_bits == bytes(4) else 0.0
                assert bytes(parameter) != previous_bits
                try:
                    with pytest.raises(ValueError, match="scalar|changed"):
                        bundle.validate_resolved_parameters(record)
                    # Wrong native bits cannot become a newly accepted baseline.
                    with pytest.raises(ValueError, match="scalar"):
                        WarpDirectLightRenderCommandBundle(
                            binding=bundle.binding, descriptors=bundle.descriptors, commands=bundle.commands
                        )
                finally:
                    parameter.value = previous
                signed_zero_checks += 1
    assert signed_zero_checks > 0
    bundle.validate_resolved_parameters(record)


@pytest.mark.parametrize("shadows,diagnostics", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize(
    "profile",
    [
        OpticalOutputProfile.DIRECT_LIGHT_FULL,
        OpticalOutputProfile.RGB_PREVIEW,
        OpticalOutputProfile.RENDER_ONLY,
    ],
)
def test_recorded_camera_exact_plan_but_no_execution_api(wp, monkeypatch, shadows, diagnostics, profile):
    setup = _setup(wp)
    buffers = _buffers(wp, setup, diagnostics)
    monkeypatch.setattr(wp.Launch, "launch", _no_execution)
    bundle = _seal(setup, buffers, shadows=shadows, output_profile=profile)
    bundle.validate_bound()
    assert len(bundle.commands) == 5
    assert bundle.evidence.shadows is shadows
    assert bundle.evidence.traversal_diagnostics is diagnostics
    assert bundle.evidence.native_input_manifest_sha256 == setup.prepared.manifest.sha256
    for name in ("enqueue", "execute", "finish", "terminal_event", "completion_ticket"):
        assert not hasattr(bundle, name)


@pytest.mark.parametrize("n", [0, 1, 21])
def test_recorded_resident_rays_omit_raygen_and_zero_case_keeps_both_resets(wp, monkeypatch, n):
    setup = _setup(wp)
    buffers = _buffers(wp, setup, n=n)
    monkeypatch.setattr(wp.Launch, "launch", _no_execution)
    bundle = setup.factory.seal_device_rays(
        world=setup.world,
        bvh=setup.bvh,
        query=replace(setup.camera.query, num_rays=n),
        buffers=buffers,
        output_profile=OpticalOutputProfile.DIRECT_LIGHT_FULL,
        shadows=False,
        traversal_diagnostics=False,
    )
    bundle.validate_bound()
    assert bundle.evidence.dispatch_count == (4 if n else 2)
    assert "_pinhole_camera_raygen_kernel" not in bundle.evidence.command_entry_ids


def test_exact_a2_storage_and_readiness_are_required_before_recording(wp, monkeypatch):
    setup = _setup(wp)
    replacement = wp.clone(setup.scene.material_albedo_rgb)
    monkeypatch.setattr(wp, "launch", _no_execution)
    for scene, prepared, readiness in (
        (replace(setup.scene, material_albedo_rgb=replacement), setup.prepared, setup.readiness),
        (setup.scene, setup.prepared, replace(setup.readiness)),
        (setup.scene, replace(setup.prepared), setup.readiness),
    ):
        with pytest.raises((ValueError, TypeError)):
            WarpDirectLightRenderCommandFactory.prepare(
                prepared_input=prepared,
                readiness=readiness,
                local_scene=scene,
                device=scene.device,
                stream=setup.stream,
            )
    buffers = _buffers(wp, setup)
    with pytest.raises(ValueError, match="exact prepared allocation"):
        _seal(
            setup,
            buffers,
            world=replace(setup.world, triangle_v0_world=wp.clone(setup.world.triangle_v0_world)),
        )


@pytest.mark.parametrize(
    "kind", ["empty", "reorder", "replace", "scalar", "array", "params_addr", "bounds", "evidence"]
)
def test_recorded_native_leaf_drift_rejected_without_launch(wp, monkeypatch, kind):
    setup = _setup(wp)
    buffers = _buffers(wp, setup)
    monkeypatch.setattr(wp.Launch, "launch", _no_execution)
    bundle = _seal(setup, buffers)
    if kind == "empty":
        object.__setattr__(bundle, "commands", ())
    elif kind == "reorder":
        object.__setattr__(bundle, "commands", tuple(reversed(bundle.commands)))
    elif kind == "replace":
        other = _seal(setup, buffers)
        object.__setattr__(bundle, "commands", other.commands)
    elif kind == "scalar":
        bundle.commands[0].set_param_by_name("fx", 7.0)
    elif kind == "array":
        bundle.commands[2].set_param_by_name("position_world", buffers.normal_world)
    elif kind == "params_addr":
        bundle.commands[2].params_addr[-1] = bundle.commands[2].params_addr[-2]
    elif kind == "bounds":
        bundle.commands[0].set_dim(1)
    else:
        object.__setattr__(bundle, "evidence", replace(bundle.evidence, output_storage_sha256="0" * 64))
    with pytest.raises(ValueError):
        bundle.validate_bound()


def test_factory_bundle_and_binding_baselines_cannot_reopen(wp, monkeypatch):
    setup = _setup(wp)
    buffers = _buffers(wp, setup)
    monkeypatch.setattr(wp.Launch, "launch", _no_execution)
    bundle = _seal(setup, buffers)
    for owner in (setup.factory, bundle, bundle.binding):
        for name, value in (("_signature", ()), ("_sealed", False)):
            with pytest.raises((FrozenInstanceError, TypeError, AttributeError)):
                setattr(owner, name, value)
            with pytest.raises((FrozenInstanceError, TypeError, AttributeError)):
                delattr(owner, name)
        with pytest.raises(RuntimeError, match="cannot be rebuilt"):
            owner.__post_init__()
    object.__setattr__(setup.factory, "stream", wp.Stream(device=setup.scene.device))
    with pytest.raises(ValueError, match="changed"):
        bundle.validate_bound()


def test_dynamic_replay_frame_and_scene_provenance_are_separate(wp, monkeypatch):
    setup = _setup(wp, replay=True)
    buffers = _buffers(wp, setup)
    monkeypatch.setattr(wp.Launch, "launch", _no_execution)
    bundle = _seal(setup, buffers, transform_frame_index=0)
    bundle.validate_bound()
    assert bundle.evidence.fixed_render_fixture_sha256 is None
    assert bundle.evidence.transform_replay_sha256 == setup.prepared.manifest.transform_replay_sha256
    assert bundle.evidence.transform_frame_index == 0
    for index in (None, True, -1, setup.prepared.manifest.transform_frame_count):
        with pytest.raises(ValueError, match="replay frame"):
            _seal(setup, buffers, transform_frame_index=index)
    with pytest.raises(ValueError, match="prepared scene"):
        _seal(setup, buffers, transform_frame_index=0, bvh=replace(setup.bvh, scene_contract_sha256="0" * 64))


def test_recording_on_target_one_restores_ambient_zero(wp, monkeypatch):
    if wp.get_cuda_device_count() < 2:
        pytest.skip("two real devices required")
    setup = _setup(wp, device="cuda:1")
    buffers = _buffers(wp, setup)
    monkeypatch.setattr(wp.Launch, "launch", _no_execution)
    with wp.ScopedDevice("cuda:0"):
        before = wp.get_device()
        assert before.ordinal == 0
        bundle = _seal(setup, buffers)
        bundle.validate_bound()
        assert wp.get_device() is before
        assert all(command.device.ordinal == 1 for command in bundle.commands)


def test_runtime_kernel_replacement_is_not_hidden_by_unchanged_source_file(wp, monkeypatch):
    from optics import warp_native_render as native

    setup = _setup(wp)
    buffers = _buffers(wp, setup)
    monkeypatch.setattr(
        native, "_reset_bvh_primary_counters_kernel", native._reset_shadow_primary_counters_kernel
    )
    monkeypatch.setattr(wp, "launch", _no_execution)
    with pytest.raises(ValueError, match="changed"):
        _seal(setup, buffers)


def test_borrowed_stream_rejected_before_load_or_recording(wp, monkeypatch):
    setup = _setup(wp)
    borrowed = wp.Stream(device=setup.scene.device, cuda_stream=setup.stream.cuda_stream)
    assert borrowed.owner is False
    monkeypatch.setattr(wp, "load_module", _no_execution)
    monkeypatch.setattr(wp, "launch", _no_execution)
    monkeypatch.setattr(wp.Launch, "launch", _no_execution)
    with pytest.raises(ValueError, match="Warp-owned stream"):
        WarpDirectLightRenderCommandFactory.prepare(
            prepared_input=setup.prepared,
            readiness=setup.readiness,
            local_scene=setup.scene,
            device=setup.scene.device,
            stream=borrowed,
        )


def test_same_stream_ownership_drift_rejected_without_launch(wp, monkeypatch):
    setup = _setup(wp)
    buffers = _buffers(wp, setup)
    monkeypatch.setattr(wp.Launch, "launch", _no_execution)
    bundle = _seal(setup, buffers)
    bundle.validate_bound()
    assert setup.stream.owner is True
    try:
        setup.stream.owner = False
        with pytest.raises(ValueError, match="Warp-owned stream"):
            bundle.validate_bound()
        monkeypatch.setattr(wp, "launch", _no_execution)
        with pytest.raises(ValueError, match="Warp-owned stream"):
            _seal(setup, buffers)
    finally:
        # Restore actual ownership so the real CUDA stream remains normally managed.
        setup.stream.owner = True


def test_raising_recording_restores_nondefault_ambient_stream_and_device(wp, monkeypatch):
    if wp.get_cuda_device_count() < 2:
        pytest.skip("two real devices required")
    setup = _setup(wp, device="cuda:1")
    buffers = _buffers(wp, setup)
    ambient = wp.Stream(device="cuda:0")
    target_ambient = wp.Stream(device="cuda:1")
    monkeypatch.setattr(wp, "launch", _no_execution)
    with wp.ScopedStream(target_ambient), wp.ScopedStream(ambient):
        before = wp.get_device()
        with pytest.raises(AssertionError, match="cannot execute"):
            _seal(setup, buffers)
        assert wp.get_device() is before
        assert wp.get_stream("cuda:0") is ambient
        assert wp.get_stream("cuda:1") is target_ambient
