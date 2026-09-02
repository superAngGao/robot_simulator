from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.optical.eligibility import (
    CANONICAL_NATIVE_REPLAY_INPUT,
    FIXED_NATIVE_SCENE_INPUT,
    verified_runtime_entry_point_audit,
)
from benchmarks.optical.instrumentation import (
    instrument_runtime_entry_points,
    invoke_instrumented_entry_point,
)
from benchmarks.optical.native import (
    CanonicalNativeInput,
    build_prepared_cuda_scene_update_factory,
    build_prepared_cuda_transform_inputs,
    prepare_cuda_native_input,
)
from benchmarks.optical.timing import RuntimeOperationTrace, native_timed_submission
from benchmarks.optical.workloads import (
    build_canonical_fixed_fixture,
    build_canonical_packed_scene,
    build_canonical_transform_replay,
)
from optics.cuda_direct_light import (
    CudaDirectLightNativeRenderer,
    CudaFirstHitInputs,
    CudaShadeInputs,
    allocate_cuda_direct_light_buffers,
)
from optics.cuda_scene import (
    CUDA_SCENE_UPDATE_OPERATION_ID,
    CudaSceneTransformInputs,
    CudaSceneUpdateStageFactory,
    allocate_cuda_world_scene_buffers,
    build_cuda_optical_scene,
)
from optics.cuda_timeline import (
    CudaOpticalExecutionContext,
    CudaOpticalTimelineError,
    CudaOpticalTransactionPoisonedError,
)
from optics.execution import OpticalOutputProfile
from optics.registry import OpticalInstanceSpec, OpticalMaterialSpec, OpticalWorldRegistry
from optics.scene_contract import PackedOpticalSceneHost

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.cuda_ext,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Torch CUDA is unavailable"),
]


def test_cuda_scene_stage_matches_dynamic_host_reference_without_timed_contamination():
    setup = _dynamic_setup()
    stage = setup.factory.bind(transforms=setup.transforms, outputs=setup.outputs)
    assert stage.evidence.operation_id == CUDA_SCENE_UPDATE_OPERATION_ID
    assert stage.evidence.native_input_manifest_sha256 == setup.prepared.manifest.sha256
    assert stage.evidence.scene_contract_sha256 == setup.packed.scene_contract_sha256
    assert stage.evidence.transform_replay_sha256 == setup.replay.transform_replay_sha256
    assert stage.evidence.transform_frame_index == 1
    assert stage.evidence.dispatch_count == 1
    assert stage.evidence.triangle_extension_entry == "update_triangles_world_aabb_out"
    assert stage.evidence.plane_extension_entry is None

    completion_event = _initialized_event(setup.context.stream)
    transaction = setup.context.begin_transaction(
        stages=(stage,),
        completion_event=completion_event,
    )
    audits = (
        verified_runtime_entry_point_audit(
            operation="scene_update",
            entry_point="optics.cuda_scene.CudaSceneUpdateStage.enqueue",
        ),
        verified_runtime_entry_point_audit(
            operation="render",
            entry_point="optics.cuda_timeline.CudaOpticalTransaction.finish",
        ),
    )
    trace = RuntimeOperationTrace(row_id="unit_cuda_c3c", lane="cuda", boundary="R1")
    with instrument_runtime_entry_points(audits) as observation:
        with native_timed_submission(
            row_id=trace.row_id,
            lane=trace.lane,
            boundary=trace.boundary,
            operation_trace=trace,
        ):
            invoke_instrumented_entry_point(
                "optics.cuda_scene.CudaSceneUpdateStage.enqueue",
                stage,
                transaction,
            )
            completion = invoke_instrumented_entry_point(
                "optics.cuda_timeline.CudaOpticalTransaction.finish",
                transaction,
            )
    assert observation.violation_hook_id is None
    completion.event.synchronize()
    assert [value.entry_point for value in trace.evidence().records] == [
        "optics.cuda_scene.CudaSceneUpdateStage.enqueue",
        "optics.cuda_timeline.CudaOpticalTransaction.finish",
    ]

    expected = _host_world_reference(
        setup.packed,
        setup.replay.arrays["replay_x_world_R"][1:2],
        setup.replay.arrays["replay_x_world_r"][1:2],
    )
    for name, tensor in setup.outputs.tensor_items():
        np.testing.assert_allclose(
            tensor.cpu().numpy(),
            expected[name],
            rtol=0.0,
            atol=1.0e-6,
        )


def test_cuda_scene_stage_matches_asymmetric_fixed_fixture():
    packed = build_canonical_packed_scene("asymmetric_synthetic_7x3_v1")
    fixture = build_canonical_fixed_fixture(packed)
    source = CanonicalNativeInput.from_scene_contract(
        packed,
        input_contract=FIXED_NATIVE_SCENE_INPUT,
        fixed_fixture=fixture,
    )
    prepared = prepare_cuda_native_input(source, device="cuda:0")
    context = _context("cuda:0")
    factory = build_prepared_cuda_scene_update_factory(prepared, context=context)
    transforms = CudaSceneTransformInputs(
        x_world_R=torch.empty((1, 0, 3, 3), dtype=torch.float32, device=context.device),
        x_world_r=torch.empty((1, 0, 3), dtype=torch.float32, device=context.device),
        env_idx=0,
        input_contract=FIXED_NATIVE_SCENE_INPUT,
        transform_replay_sha256=None,
        transform_frame_index=None,
        resources=(prepared,),
    )
    outputs = allocate_cuda_world_scene_buffers(factory.local_scene)
    stage = factory.bind(transforms=transforms, outputs=outputs)
    assert stage.evidence.dispatch_count == 1
    transaction = context.begin_transaction(
        stages=(stage,),
        completion_event=_initialized_event(context.stream),
    )
    stage.enqueue(transaction)
    completion = transaction.finish()
    completion.event.synchronize()
    for name, tensor in outputs.tensor_items():
        np.testing.assert_allclose(
            tensor.cpu().numpy(),
            fixture.arrays[name],
            rtol=0.0,
            atol=1.0e-6,
        )


@pytest.mark.parametrize("primitive_kind", ("plane", "empty"))
def test_cuda_scene_empty_primitive_classes_do_not_bind_dummy_launches(primitive_kind):
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat"))
    if primitive_kind == "plane":
        registry.add_plane_geometry(
            "plane",
            normal_local=[0.0, 1.0, 0.0],
            point_local=[1.0, 2.0, 3.0],
        )
        registry.add_instance(OpticalInstanceSpec("plane", "plane", "mat"))
    packed = PackedOpticalSceneHost.from_registry(
        registry,
        scene_id=f"unit_{primitive_kind}",
    )
    local_scene = build_cuda_optical_scene(packed, device="cuda:0")
    local_scene.ready_event.synchronize()
    context = _context("cuda:0")
    factory = CudaSceneUpdateStageFactory.prepare(
        context=context,
        local_scene=local_scene,
        native_input_manifest_sha256="a" * 64,
        input_contract="unit_static_scene_v1",
    )
    transforms = CudaSceneTransformInputs(
        x_world_R=torch.empty((1, 0, 3, 3), dtype=torch.float32, device=context.device),
        x_world_r=torch.empty((1, 0, 3), dtype=torch.float32, device=context.device),
        env_idx=0,
        input_contract="unit_static_scene_v1",
        transform_replay_sha256=None,
        transform_frame_index=None,
        resources=(local_scene,),
    )
    outputs = allocate_cuda_world_scene_buffers(local_scene)
    stage = factory.bind(transforms=transforms, outputs=outputs)
    assert stage.evidence.triangle_extension_entry is None
    assert stage.evidence.plane_extension_entry == (
        "update_planes_world_out" if primitive_kind == "plane" else None
    )
    assert stage.evidence.dispatch_count == (1 if primitive_kind == "plane" else 0)
    transaction = context.begin_transaction(
        stages=(stage,),
        completion_event=_initialized_event(context.stream),
    )
    stage.enqueue(transaction)
    completion = transaction.finish()
    completion.event.synchronize()
    if primitive_kind == "plane":
        np.testing.assert_allclose(
            outputs.plane_normal_world.cpu().numpy(),
            [[0.0, 1.0, 0.0]],
            rtol=0.0,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            outputs.plane_point_world.cpu().numpy(),
            [[1.0, 2.0, 3.0]],
            rtol=0.0,
            atol=1.0e-6,
        )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_cuda_scene_restores_wrong_ambient_device_and_target_current_stream():
    setup = _dynamic_setup("cuda:1")
    stage = setup.factory.bind(transforms=setup.transforms, outputs=setup.outputs)
    completion_event = _initialized_event(setup.context.stream)
    target_previous_stream = torch.cuda.current_stream(torch.device("cuda:1"))
    torch.cuda.set_device(0)
    assert torch.cuda.current_device() == 0
    transaction = setup.context.begin_transaction(
        stages=(stage,),
        completion_event=completion_event,
    )
    stage.enqueue(transaction)
    assert torch.cuda.current_device() == 0
    assert torch.cuda.current_stream(torch.device("cuda:1")) == target_previous_stream
    completion = transaction.finish()
    assert torch.cuda.current_device() == 0
    assert torch.cuda.current_stream(torch.device("cuda:1")) == target_previous_stream
    completion.event.synchronize()
    assert torch.cuda.current_device() == 0

    failing_args = list(stage._triangle_args)
    failing_args[-1] = failing_args[-2]
    with pytest.raises(RuntimeError, match="overlap"):
        setup.factory.module.update_triangles_world_aabb_out(*failing_args)
    assert torch.cuda.current_device() == 0
    assert torch.cuda.current_stream(torch.device("cuda:1")) == target_previous_stream


@pytest.mark.parametrize("traversal_diagnostics", (False, True))
def test_cuda_scene_composes_with_c2_render_on_one_transaction(
    traversal_diagnostics,
):
    setup = _dynamic_setup()
    scene_stage = setup.factory.bind(transforms=setup.transforms, outputs=setup.outputs)
    scene = setup.factory.local_scene
    device = setup.context.device
    first_hit = CudaFirstHitInputs(
        plane_normal_world=setup.outputs.plane_normal_world,
        plane_point_world=setup.outputs.plane_point_world,
        plane_numeric_instance_id=scene.arrays["plane_numeric_instance_id"],
        plane_source_order_key=scene.arrays["plane_source_order_key"],
        plane_role_mask=scene.arrays["plane_role_mask"],
        plane_material_index=scene.arrays["plane_material_index"],
        num_planes=scene.num_planes,
        triangle_v0_world=setup.outputs.triangle_v0_world,
        triangle_e1_world=setup.outputs.triangle_e1_world,
        triangle_e2_world=setup.outputs.triangle_e2_world,
        triangle_normal_world=setup.outputs.triangle_normal_world,
        triangle_numeric_instance_id=scene.arrays["triangle_numeric_instance_id"],
        triangle_source_order_key=scene.arrays["triangle_source_order_key"],
        triangle_role_mask=scene.arrays["triangle_role_mask"],
        triangle_material_index=scene.arrays["triangle_material_index"],
        bvh_bounds_min=setup.outputs.triangle_aabb_min,
        bvh_bounds_max=setup.outputs.triangle_aabb_max,
        bvh_left=torch.tensor([-1], dtype=torch.int32, device=device),
        bvh_right=torch.tensor([-1], dtype=torch.int32, device=device),
        bvh_start=torch.tensor([0], dtype=torch.int32, device=device),
        bvh_count=torch.tensor([1], dtype=torch.int32, device=device),
        bvh_prim_ids=torch.tensor([0], dtype=torch.int32, device=device),
        num_bvh_nodes=1,
        sensor_role_mask=_all_role_mask(scene),
        max_distance=10.0,
    )
    shade = CudaShadeInputs(
        material_albedo_rgb=scene.arrays["material_albedo_rgb"],
        light_kind=scene.arrays["light_kind"],
        light_position_or_direction_world=scene.arrays["light_position_or_direction_world"],
        light_intensity=scene.arrays["light_intensity"],
        light_color_rgb=scene.arrays["light_color_rgb"],
        num_lights=scene.num_lights,
    )
    buffers = allocate_cuda_direct_light_buffers(
        num_rays=2,
        device=device,
        origins=torch.tensor(
            [[0.05, 0.05, 2.0], [2.0, 2.0, 2.0]],
            dtype=torch.float32,
            device=device,
        ),
        directions=torch.tensor(
            [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
            dtype=torch.float32,
            device=device,
        ),
        direct_light=True,
        traversal_diagnostics=traversal_diagnostics,
    )
    renderer = CudaDirectLightNativeRenderer.prepare(setup.context)
    render_stage = renderer.bind_direct_light(
        first_hit_inputs=first_hit,
        shade_inputs=shade,
        buffers=buffers,
        output_profile=OpticalOutputProfile.DIRECT_LIGHT_FULL,
        shadows=False,
        ambient_rgb=(0.1, 0.1, 0.1),
        background_rgb=(0.0, 0.0, 0.0),
        shadow_bias=1.0e-6,
    )
    transaction = setup.context.begin_transaction(
        stages=(scene_stage, render_stage),
        completion_event=_initialized_event(setup.context.stream),
    )
    scene_stage.enqueue(transaction)
    render_stage.enqueue(transaction)
    completion = transaction.finish()
    completion.event.synchronize()
    assert completion.stage_names == ("cuda_scene_update", "cuda_direct_light_render")
    assert buffers.hit_mask.cpu().tolist() == [1, 0]
    assert torch.isfinite(buffers.rgb).all().item()


def test_cuda_scene_rejects_storage_and_binding_drift_before_enqueue():
    setup = _dynamic_setup()
    with pytest.raises(AttributeError, match="immutable"):
        setup.factory._sealed = False
    with pytest.raises(AttributeError, match="immutable"):
        del setup.factory._sealed
    wrong_dtype = replace(
        setup.outputs,
        triangle_aabb_min=torch.empty(
            (setup.factory.local_scene.num_triangles, 3),
            dtype=torch.float64,
            device=setup.context.device,
        ),
    )
    with pytest.raises(ValueError, match="wrong dtype"):
        setup.factory.bind(transforms=setup.transforms, outputs=wrong_dtype)

    noncontiguous = replace(
        setup.outputs,
        triangle_aabb_min=torch.empty(
            (setup.factory.local_scene.num_triangles, 6),
            dtype=torch.float32,
            device=setup.context.device,
        )[:, ::2],
    )
    with pytest.raises(ValueError, match="contiguous"):
        setup.factory.bind(transforms=setup.transforms, outputs=noncontiguous)

    aliased = replace(
        setup.outputs,
        triangle_e1_world=setup.outputs.triangle_v0_world,
    )
    with pytest.raises(ValueError, match="overlap"):
        setup.factory.bind(transforms=setup.transforms, outputs=aliased)

    with pytest.raises(AttributeError, match="immutable"):
        setup.factory.native_input_manifest_sha256 = "0" * 64
    drifted_factory = build_prepared_cuda_scene_update_factory(
        setup.prepared,
        context=setup.context,
    )
    object.__setattr__(drifted_factory, "native_input_manifest_sha256", "0" * 64)
    with pytest.raises(AttributeError, match="immutable"):
        drifted_factory._frozen_signature = drifted_factory._factory_signature()
    with pytest.raises(ValueError, match="changed after construction"):
        drifted_factory.bind(transforms=setup.transforms, outputs=setup.outputs)

    stage = setup.factory.bind(transforms=setup.transforms, outputs=setup.outputs)
    with pytest.raises(AttributeError, match="immutable"):
        stage._sealed = False
    with pytest.raises(AttributeError, match="immutable"):
        del stage._sealed
    with pytest.raises(RuntimeError, match="one-way"):
        stage._freeze()

    permuted_stage = setup.factory.bind(
        transforms=setup.transforms,
        outputs=setup.outputs,
    )
    permuted_args = list(permuted_stage._triangle_args)
    permuted_args[8], permuted_args[9] = permuted_args[9], permuted_args[8]
    object.__setattr__(permuted_stage, "_triangle_args", tuple(permuted_args))
    with pytest.raises(AttributeError, match="immutable"):
        permuted_stage._binding_signature = permuted_stage._stage_signature()
    with pytest.raises(ValueError, match="ABI binding changed"):
        setup.context.begin_transaction(
            stages=(permuted_stage,),
            completion_event=_initialized_event(setup.context.stream),
        )

    object.__setattr__(setup.transforms, "transform_frame_index", 2)
    with pytest.raises(ValueError, match="changed"):
        setup.context.begin_transaction(
            stages=(stage,),
            completion_event=_initialized_event(setup.context.stream),
        )

    object.__setattr__(setup.transforms, "transform_frame_index", 1)
    transaction = setup.context.begin_transaction(
        stages=(stage,),
        completion_event=_initialized_event(setup.context.stream),
    )
    object.__setattr__(stage, "_triangle_args", stage._triangle_args[:-1])
    with pytest.raises(ValueError, match="binding changed"):
        stage.enqueue(transaction)
    assert transaction.state == "poisoned"
    with pytest.raises(CudaOpticalTimelineError, match="before successful finish"):
        _ = transaction.completion
    with pytest.raises(CudaOpticalTransactionPoisonedError):
        transaction.finish()


def _dynamic_setup(device: str = "cuda:0"):
    packed = build_canonical_packed_scene("synthetic_dynamic_rigid_960x640_v1")
    replay = build_canonical_transform_replay("synthetic_dynamic_rigid_960x640_v1")
    source = CanonicalNativeInput.from_scene_contract(
        packed,
        input_contract=CANONICAL_NATIVE_REPLAY_INPUT,
        replay=replay,
    )
    prepared = prepare_cuda_native_input(source, device=device)
    context = _context(device)
    factory = build_prepared_cuda_scene_update_factory(prepared, context=context)
    transforms = build_prepared_cuda_transform_inputs(prepared, 1)
    outputs = allocate_cuda_world_scene_buffers(factory.local_scene)
    return SimpleNamespace(
        packed=packed,
        replay=replay,
        prepared=prepared,
        context=context,
        factory=factory,
        transforms=transforms,
        outputs=outputs,
    )


def _context(device: str):
    resolved = torch.device(device)
    with torch.cuda.device(resolved):
        stream = torch.cuda.Stream(device=resolved)
    return CudaOpticalExecutionContext(device=resolved, stream=stream)


def _initialized_event(stream):
    with torch.cuda.device(stream.device), torch.cuda.stream(stream):
        event = torch.cuda.Event(enable_timing=False)
        event.record(stream)
    event.synchronize()
    return event


def _all_role_mask(scene) -> int:
    mask = 0
    for value in scene.role_to_mask.values():
        mask |= int(value)
    return mask


def _host_world_reference(packed, x_world_R, x_world_r):
    arrays = packed.arrays
    outputs = {
        "triangle_v0_world": np.empty((packed.num_triangles, 3), dtype=np.float32),
        "triangle_e1_world": np.empty((packed.num_triangles, 3), dtype=np.float32),
        "triangle_e2_world": np.empty((packed.num_triangles, 3), dtype=np.float32),
        "triangle_normal_world": np.empty((packed.num_triangles, 3), dtype=np.float32),
        "triangle_aabb_min": np.empty((packed.num_triangles, 3), dtype=np.float32),
        "triangle_aabb_max": np.empty((packed.num_triangles, 3), dtype=np.float32),
        "plane_normal_world": np.empty((packed.num_planes, 3), dtype=np.float32),
        "plane_point_world": np.empty((packed.num_planes, 3), dtype=np.float32),
    }
    transforms = []
    for instance in range(packed.num_instances):
        body = int(arrays["instance_body_index"][instance])
        body_R = np.eye(3) if body < 0 else np.asarray(x_world_R[0, body], dtype=np.float64)
        body_r = np.zeros(3) if body < 0 else np.asarray(x_world_r[0, body], dtype=np.float64)
        geometry_R = np.asarray(arrays["instance_X_body_geometry_R"][instance], dtype=np.float64).reshape(
            3, 3
        )
        geometry_r = np.asarray(arrays["instance_X_body_geometry_r"][instance], dtype=np.float64)
        transforms.append((body_R @ geometry_R, body_r + body_R @ geometry_r))
    for primitive in range(packed.num_triangles):
        instance = int(arrays["triangle_instance_index"][primitive])
        rotation, translation = transforms[instance]
        local = np.asarray(arrays["triangle_vertices_local"][primitive], dtype=np.float64).reshape(3, 3)
        world = local @ rotation.T + translation
        e1 = world[1] - world[0]
        e2 = world[2] - world[0]
        normal = np.cross(e1, e2)
        length = np.linalg.norm(normal)
        normal = normal / length if length > 1.0e-8 else np.zeros(3)
        outputs["triangle_v0_world"][primitive] = world[0]
        outputs["triangle_e1_world"][primitive] = e1
        outputs["triangle_e2_world"][primitive] = e2
        outputs["triangle_normal_world"][primitive] = normal
        outputs["triangle_aabb_min"][primitive] = np.min(world, axis=0)
        outputs["triangle_aabb_max"][primitive] = np.max(world, axis=0)
    for primitive in range(packed.num_planes):
        instance = int(arrays["plane_instance_index"][primitive])
        rotation, translation = transforms[instance]
        normal = rotation @ np.asarray(arrays["plane_normal_local"][primitive], dtype=np.float64)
        length = np.linalg.norm(normal)
        normal = normal / length if length > 0.0 else np.zeros(3)
        point = rotation @ np.asarray(arrays["plane_point_local"][primitive], dtype=np.float64) + translation
        outputs["plane_normal_world"][primitive] = normal
        outputs["plane_point_world"][primitive] = point
    return outputs
