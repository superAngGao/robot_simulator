from __future__ import annotations

import struct
from dataclasses import replace

import pytest
import torch

from benchmarks.optical.cases import get_case
from benchmarks.optical.cuda_arena import build_cuda_c3_binding_from_arena_target
from benchmarks.optical.eligibility import (
    CANONICAL_NATIVE_REPLAY_INPUT,
    FIXED_NATIVE_SCENE_INPUT,
)
from benchmarks.optical.native import (
    CanonicalNativeInput,
    CudaPreparedNativeInput,
    build_prepared_cuda_lbvh_rebuild_factory,
    build_prepared_cuda_scene_update_factory,
    build_prepared_cuda_transform_inputs,
    certify_native_readiness,
    prepare_cuda_native_input,
    validate_prepared_native_input,
    wrap_prepared_fixed_render_inputs,
)
from benchmarks.optical.receipts import C3OneShotNativeExecutionBindingEvidence
from benchmarks.optical.workloads import (
    build_canonical_fixed_fixture,
    build_canonical_packed_scene,
    build_canonical_transform_replay,
)
from optical_worker.render_parameters import bind_render_parameters, load_frozen_render_sequence
from optics.cuda_arena import (
    CudaActiveArenaSample,
    CudaArenaC3Provenance,
    CudaNativeOpticalArena,
    CudaPreparedArenaSample,
    build_cuda_dynamic_optical_arena,
    build_cuda_fixed_optical_arena,
    cuda_execution_stream_token,
)
from optics.cuda_direct_light import (
    CudaCameraParameters,
    CudaDirectLightNativeRenderer,
    CudaFirstHitInputs,
    CudaShadeInputs,
)
from optics.cuda_timeline import (
    CudaOpticalExecutionContext,
    CudaOpticalTimelineError,
    CudaOpticalTransaction,
    CudaOpticalTransactionTemplate,
    _CudaArenaLiveExecutionAuthority,
)
from optics.native_arena import (
    NATIVE_OPTICAL_ARENA_LAYOUT_SCHEMA_VERSION,
    NATIVE_OPTICAL_ARENA_SPEC_SCHEMA_VERSION,
    NativeOpticalArenaBackpressure,
    NativeOpticalArenaProtocolError,
    NativeOpticalArenaSpec,
)

pytestmark = pytest.mark.gpu


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA is required")


def _context(device: str = "cuda:0") -> CudaOpticalExecutionContext:
    target = torch.device(device)
    with torch.cuda.device(target):
        stream = torch.cuda.Stream(device=target)
    return CudaOpticalExecutionContext(device=target, stream=stream)


def _render_values(case_id):
    # Use the reviewed camera/shading ABI, never a tiny surrogate camera.
    payload = bind_render_parameters(
        load_frozen_render_sequence(case_id),
        sample_index=0,
        profile="direct_light_full",
        shadows=False,
        pass_mode="primary_unprofiled",
        role_to_mask={"rgb": 2},
    ).to_payload()
    from optical_worker.render_parameters import decode_native_scalars

    values = decode_native_scalars(payload)
    return payload, values


def _camera(case_id) -> CudaCameraParameters:
    payload, values = _render_values(case_id)
    return CudaCameraParameters(
        width=payload["width"],
        height=payload["height"],
        fx=values[0],
        fy=values[1],
        cx=values[2],
        cy=values[3],
        origin=values[4:7],
        rotation=values[7:16],
    )


_A2_AUTHORITIES: dict[int, tuple[CudaPreparedNativeInput, object]] = {}


def _spec(
    context: CudaOpticalExecutionContext,
    *,
    boundary: str,
    prepared: CudaPreparedNativeInput,
    readiness: object,
    primitive_count: int,
    node_capacity: int,
    diagnostic: bool = False,
) -> NativeOpticalArenaSpec:
    dynamic = boundary != "R0"
    manifest = prepared.manifest
    case = get_case(manifest.case_id)
    return NativeOpticalArenaSpec(
        schema_version=NATIVE_OPTICAL_ARENA_SPEC_SCHEMA_VERSION,
        row_id=f"unit_{boundary.lower()}",
        case_id=case.case_id,
        case_sha256=case.sha256,
        boundary=boundary,
        lane="cuda",
        device_ordinal=int(context.device.index),
        execution_stream_token=cuda_execution_stream_token(context),
        input_contract=manifest.input_contract,
        native_input_manifest_sha256=manifest.sha256,
        native_readiness_sha256=readiness.sha256,
        scene_contract_sha256=manifest.scene_contract_sha256,
        transform_replay_sha256=manifest.transform_replay_sha256 if dynamic else None,
        transform_frame_count=manifest.transform_frame_count if dynamic else 0,
        fixed_render_fixture_sha256=manifest.fixed_render_fixture_sha256,
        renderer_id="cuda_direct_light",
        acceleration_class=(
            "production_static"
            if boundary == "R2"
            else ("production_dynamic" if dynamic else "fixed_bvh_fixture")
        ),
        acceleration_id=("cuda_morton_karras_lbvh_rebuild_v1" if dynamic else "fixed_bvh_fixture"),
        width=case.width,
        height=case.height,
        num_rays=case.width * case.height,
        primitive_count=primitive_count,
        node_capacity=node_capacity,
        output_profile="direct_light_full",
        shadows=False,
        diagnostic_mode="diagnostic_profiled" if diagnostic else "primary_unprofiled",
        payload="none" if boundary == "R2" else "native_only",
        delivery_policy="device_only" if boundary == "R2" else "native_only",
        payload_contract_id="none_device_only_v1" if boundary == "R2" else None,
        ring_depth=1,
        ordered_delivery=True,
        arena_layout_schema_version=NATIVE_OPTICAL_ARENA_LAYOUT_SCHEMA_VERSION,
    )


def _provenance(arena, target):
    prepared, readiness = _A2_AUTHORITIES[id(arena)]
    binding = build_cuda_c3_binding_from_arena_target(
        target,
        prepared_input=prepared,
        native_readiness=readiness,
    )
    return arena.bind_c3_provenance(
        target=target,
        binding=binding,
        prepared_input=prepared,
        native_readiness=readiness,
    )


def _dynamic_arena(
    *,
    boundary: str = "R1",
    diagnostic: bool = False,
    device: str = "cuda:0",
):
    _require_cuda()
    context = _context(device)
    case_id = "synthetic_dynamic_rigid_960x640_v1"
    packed = build_canonical_packed_scene(case_id)
    replay = build_canonical_transform_replay(case_id)
    source = CanonicalNativeInput.from_scene_contract(
        packed,
        input_contract=CANONICAL_NATIVE_REPLAY_INPUT,
        replay=replay,
    )
    prepared = prepare_cuda_native_input(source, device=context.device)
    validation = validate_prepared_native_input(source, prepared)
    scene_factory = build_prepared_cuda_scene_update_factory(prepared, context=context)
    lbvh_factory = build_prepared_cuda_lbvh_rebuild_factory(prepared, context=context)
    transforms = tuple(
        build_prepared_cuda_transform_inputs(prepared, index)
        for index in range(prepared.manifest.transform_frame_count)
    )
    readiness = certify_native_readiness(
        prepared,
        validation,
        compile_complete=True,
        warmup_frames_completed=0,
    )
    renderer = CudaDirectLightNativeRenderer.prepare(context)
    spec = _spec(
        context,
        boundary=boundary,
        prepared=prepared,
        readiness=readiness,
        primitive_count=scene_factory.local_scene.num_triangles,
        node_capacity=lbvh_factory.requirements.node_capacity,
        diagnostic=diagnostic,
    )
    arena = build_cuda_dynamic_optical_arena(
        spec=spec,
        context=context,
        scene_factory=scene_factory,
        lbvh_factory=lbvh_factory,
        renderer=renderer,
        transforms=transforms,
        camera=_camera(case_id),
        sensor_role_mask=scene_factory.local_scene.role_to_mask["rgb"],
        max_distance=_render_values(case_id)[1][16],
        ambient_rgb=_render_values(case_id)[1][17:20],
        background_rgb=_render_values(case_id)[1][20:23],
        shadow_bias=_render_values(case_id)[1][23],
    )
    _A2_AUTHORITIES[id(arena)] = (prepared, readiness)
    return arena


def _fixed_inputs(prepared: CudaPreparedNativeInput, sensor_role_mask: int):
    arrays = prepared.arrays
    first = CudaFirstHitInputs(
        plane_normal_world=arrays["plane_normal_world"],
        plane_point_world=arrays["plane_point_world"],
        plane_numeric_instance_id=arrays["plane_numeric_instance_id"],
        plane_source_order_key=arrays["plane_source_order_key"],
        plane_role_mask=arrays["plane_role_mask"],
        plane_material_index=arrays["plane_material_index"],
        num_planes=int(arrays["plane_normal_world"].shape[0]),
        triangle_v0_world=arrays["triangle_v0_world"],
        triangle_e1_world=arrays["triangle_e1_world"],
        triangle_e2_world=arrays["triangle_e2_world"],
        triangle_normal_world=arrays["triangle_normal_world"],
        triangle_numeric_instance_id=arrays["triangle_numeric_instance_id"],
        triangle_source_order_key=arrays["triangle_source_order_key"],
        triangle_role_mask=arrays["triangle_role_mask"],
        triangle_material_index=arrays["triangle_material_index"],
        bvh_bounds_min=arrays["bvh_bounds_min"],
        bvh_bounds_max=arrays["bvh_bounds_max"],
        bvh_left=arrays["bvh_left"],
        bvh_right=arrays["bvh_right"],
        bvh_start=arrays["bvh_start"],
        bvh_count=arrays["bvh_count"],
        bvh_prim_ids=arrays["bvh_prim_ids"],
        num_bvh_nodes=int(arrays["bvh_left"].shape[0]),
        sensor_role_mask=sensor_role_mask,
        max_distance=_render_values(prepared.manifest.case_id)[1][16],
    )
    shade = CudaShadeInputs(
        material_albedo_rgb=arrays["material_albedo_rgb"],
        light_kind=arrays["light_kind"],
        light_position_or_direction_world=arrays["light_position_or_direction_world"],
        light_intensity=arrays["light_intensity"],
        light_color_rgb=arrays["light_color_rgb"],
        num_lights=int(arrays["light_kind"].shape[0]),
    )
    return first, shade


def _fixed_arena(*, device: str = "cuda:0", camera_override=None, ambient_override=None):
    _require_cuda()
    context = _context(device)
    packed = build_canonical_packed_scene("asymmetric_synthetic_7x3_v1")
    fixture = build_canonical_fixed_fixture(packed)
    source = CanonicalNativeInput.from_scene_contract(
        packed,
        input_contract=FIXED_NATIVE_SCENE_INPUT,
        fixed_fixture=fixture,
    )
    prepared = prepare_cuda_native_input(source, device=context.device)
    validation = validate_prepared_native_input(source, prepared)
    readiness = certify_native_readiness(
        prepared,
        validation,
        compile_complete=True,
        warmup_frames_completed=0,
    )
    fixed = wrap_prepared_fixed_render_inputs(prepared)
    first, shade = _fixed_inputs(prepared, packed.role_to_mask["rgb"])
    spec = _spec(
        context,
        boundary="R0",
        prepared=prepared,
        readiness=readiness,
        primitive_count=int(first.triangle_v0_world.shape[0]),
        node_capacity=first.num_bvh_nodes,
    )
    arena = build_cuda_fixed_optical_arena(
        spec=spec,
        context=context,
        renderer=CudaDirectLightNativeRenderer.prepare(context),
        first_hit_inputs=first,
        shade_inputs=shade,
        camera=camera_override or _camera("asymmetric_synthetic_7x3_v1"),
        ambient_rgb=ambient_override or _render_values(prepared.manifest.case_id)[1][17:20],
        background_rgb=_render_values(prepared.manifest.case_id)[1][20:23],
        shadow_bias=_render_values(prepared.manifest.case_id)[1][23],
        runtime_scene_arrays=fixed.scene_arrays,
        derived_fixture_arrays=fixed.derived_fixture_arrays,
        resource_owners=(prepared,),
    )
    _A2_AUTHORITIES[id(arena)] = (prepared, readiness)
    return arena


@pytest.mark.parametrize("wrong", ["focal", "rotation", "ambient"])
def test_canonical_c3_rejects_correct_scene_with_wrong_actual_render_scalars(wrong):
    camera = _camera("asymmetric_synthetic_7x3_v1")
    options = {}
    if wrong == "focal":
        options["camera_override"] = replace(camera, fx=camera.fx + 1.0)
    elif wrong == "rotation":
        rotation = list(camera.rotation)
        rotation[0], rotation[1] = rotation[1], rotation[0]
        options["camera_override"] = replace(camera, rotation=tuple(rotation))
    else:
        options["ambient_override"] = (1.0, 0.0, 0.0)
    arena = _fixed_arena(**options)
    target = arena.plan_target(transform_frame_index=None)
    before = arena.lifecycle_evidence()
    with pytest.raises(ValueError, match="actual CUDA scalar ABI"):
        _provenance(arena, target)
    assert arena.lifecycle_evidence() == before
    assert before.terminal_event_record_count == 0
    assert before.slots[0].state == "FREE"


def test_cuda_actual_bound_argument_permutation_fails_before_sample_leaf():
    arena = _fixed_arena()
    target = arena.plan_target(transform_frame_index=None)
    provenance = _provenance(arena, target)
    stage = arena._execution_for_target(target).transaction_template.stages[0]
    with pytest.raises(AttributeError):
        stage._sealed = False
    with pytest.raises(ValueError, match="cannot be rebuilt"):
        stage._freeze_storage_signature()
    original = stage._raygen_args
    arguments = list(original)
    arguments[4], arguments[5] = arguments[5], arguments[4]
    object.__setattr__(stage, "_raygen_args", tuple(arguments))
    before = arena.lifecycle_evidence()
    try:
        with pytest.raises(ValueError, match="ABI|changed"):
            arena.prepare_sample(target=target, provenance=provenance)
        assert arena.lifecycle_evidence() == before
        assert before.terminal_event_record_count == 0
    finally:
        object.__setattr__(stage, "_raygen_args", original)


def test_cuda_r1_arena_reuses_one_event_only_after_exact_reclaim(monkeypatch):
    arena = _dynamic_arena()
    target = arena.plan_target(transform_frame_index=0)
    terminal_token = target.evidence.terminal_event_token
    prepared = arena.prepare_sample(target=target, provenance=_provenance(arena, target))

    original_empty = torch.empty
    original_event = torch.cuda.Event

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("measured acquire/enqueue must not allocate")

    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(torch.cuda, "Event", forbidden)
    active = arena.acquire(prepared)
    assert not isinstance(active, NativeOpticalArenaBackpressure)
    active.enqueue()
    active.finish()
    with pytest.raises(NativeOpticalArenaProtocolError, match="payload-ready"):
        active.output_channels()
    blocked_target = arena.plan_target(transform_frame_index=1)
    blocked_plan = arena.prepare_sample(
        target=blocked_target,
        provenance=_provenance(arena, blocked_target),
    )
    assert isinstance(arena.acquire(blocked_plan), NativeOpticalArenaBackpressure)
    monkeypatch.setattr(torch, "empty", original_empty)
    monkeypatch.setattr(torch.cuda, "Event", original_event)

    active.resolve()
    output_view = active.output_channels()
    assert "hit_mask" in output_view
    active.release_device_result()
    active.reclaim()
    with pytest.raises(NativeOpticalArenaProtocolError, match="payload-ready"):
        _ = output_view["hit_mask"]
    next_active = arena.acquire(blocked_plan)
    assert not isinstance(next_active, NativeOpticalArenaBackpressure)
    assert next_active.c4_evidence.generation == 2
    assert next_active.plan_evidence.terminal_event_token == terminal_token
    next_active.enqueue()
    next_active.finish()
    next_active.resolve()
    next_active.release_device_result()
    next_active.reclaim()
    lifecycle = arena.lifecycle_evidence()
    assert lifecycle.terminal_event_record_count == 2
    assert lifecycle.slot_reclaim_count == 2
    arena.begin_drain()
    arena.close()


def test_cuda_r2_device_only_registers_explicit_endpoint_before_finish():
    arena = _dynamic_arena(boundary="R2")
    target = arena.plan_target(transform_frame_index=0)
    assert target.evidence.stage_names == (
        "cuda_scene_update",
        "cuda_lbvh_rebuild",
        "cuda_direct_light_render",
        "cuda_device_only_delivery",
    )
    active = arena.acquire(arena.prepare_sample(target=target, provenance=_provenance(arena, target)))
    assert not isinstance(active, NativeOpticalArenaBackpressure)
    active.enqueue()
    completion = active.finish()
    assert completion.stage_names[-1] == "cuda_device_only_delivery"
    active.resolve()
    active.release_device_result()
    active.reclaim()


def test_cuda_slot_manifest_exposes_exact_semantics_dtype_shape_and_events():
    arena = _dynamic_arena(diagnostic=True)
    target = arena.plan_target(transform_frame_index=0)
    by_name = {
        name: (record, record.logical_view(name))
        for record in target.slot_resources
        for name in record.logical_names
    }
    assert by_name["slot0.world.triangle_v0_world"][1].dtype == "torch.float32"
    assert by_name["slot0.world.triangle_v0_world"][1].shape == (1, 3)
    assert by_name["slot0.lbvh.left"][1].dtype == "torch.int32"
    assert by_name["slot0.render.rgb"][1].shape == (arena.spec.num_rays, 3)
    assert by_name["terminal_event"][0].resource_kind == "cuda_event"
    assert by_name["diagnostic.acceleration_kernel.start"][0].resource_kind == "cuda_event"
    session_names = {name for record in arena.session_resource_records for name in record.logical_names}
    assert "session.local_scene.triangle_vertices_local" in session_names
    assert "session.lbvh.workspace" in session_names
    rotation_record = next(
        record
        for record in arena.session_resource_records
        if "session.transform[0].x_world_R" in record.logical_names
    )
    assert "session.transform[1].x_world_R" in rotation_record.logical_names
    assert (
        rotation_record.logical_view("session.transform[0].x_world_R").active_offset_bytes
        < rotation_record.logical_view("session.transform[1].x_world_R").active_offset_bytes
    )
    assert (
        target.evidence.slot_resource_contract_sha256
        == arena.resource_manifest.slots[0].slot_storage_manifest_sha256
    )


def test_cuda_r0_arena_uses_render_only_template_and_rejects_foreign_target():
    arena = _fixed_arena()
    target = arena.plan_target(transform_frame_index=None)
    assert target.evidence.stage_names == ("cuda_direct_light_render",)
    provenance = _provenance(arena, target)
    active = arena.acquire(arena.prepare_sample(target=target, provenance=provenance))
    assert not isinstance(active, NativeOpticalArenaBackpressure)
    active.enqueue()
    active.finish()
    active.resolve()
    active.release_device_result()
    active.reclaim()
    object.__setattr__(provenance, "plan_binding_sha256", "d" * 64)
    with pytest.raises(NativeOpticalArenaProtocolError, match="not bound"):
        arena.prepare_sample(target=target, provenance=provenance)


def test_cuda_diagnostic_arena_owns_overlay_and_phase_events_only_when_enabled():
    arena = _dynamic_arena(diagnostic=True)
    target = arena.plan_target(transform_frame_index=0)
    assert target.evidence.diagnostic_phase_names == (
        "scene_update_kernel",
        "acceleration_kernel",
        "raygen_kernel",
        "first_hit_kernel",
        "shade_kernel",
    )
    assert len(target._render_buffers.shadow_traversal_counters) == 9
    active = arena.acquire(arena.prepare_sample(target=target, provenance=_provenance(arena, target)))
    assert not isinstance(active, NativeOpticalArenaBackpressure)
    active.enqueue()
    completion = active.finish()
    active.resolve()
    assert completion.deferred_profile is not None
    assert len(completion.deferred_profile.resolve_ms()) == 5
    active.release_device_result()
    active.reclaim()


def test_submitted_terminal_event_drift_poisons_instead_of_false_completion():
    arena = _dynamic_arena()
    target = arena.plan_target(transform_frame_index=0)
    active = arena.acquire(arena.prepare_sample(target=target, provenance=_provenance(arena, target)))
    assert not isinstance(active, NativeOpticalArenaBackpressure)
    context = active._transaction._context
    with context.activate():
        wrong_event = torch.cuda.Event()
        wrong_event.record(context.stream)
    wrong_event.synchronize()
    active.enqueue()
    object.__setattr__(active._transaction, "_completion_event", wrong_event)
    with pytest.raises(NativeOpticalArenaProtocolError, match="changed"):
        active.finish()
    assert arena.state == "POISONED"
    assert active.state == "POISONED"
    assert arena.lifecycle_evidence().terminal_event_record_count == 0


def test_active_wrapper_state_cannot_be_forged_by_normal_assignment():
    arena = _dynamic_arena()
    target = arena.plan_target(transform_frame_index=0)
    active = arena.acquire(arena.prepare_sample(target=target, provenance=_provenance(arena, target)))
    assert not isinstance(active, NativeOpticalArenaBackpressure)
    with pytest.raises(AttributeError, match="immutable"):
        active._state = "RECLAIMABLE"
    active.cancel_before_enqueue()


def test_c3_provenance_requires_exact_arena_storage_and_cannot_be_forged():
    arena = _dynamic_arena()
    target = arena.plan_target(transform_frame_index=0)
    with pytest.raises(TypeError, match="minted only"):
        CudaArenaC3Provenance(
            plan_binding_sha256="1" * 64,
            ordered_runtime_stage_evidence_sha256="2" * 64,
            c3_one_shot_binding_sha256="3" * 64,
            ordered_stage_evidence_sha256="4" * 64,
            dataflow_binding_sha256="5" * 64,
            _authority=object(),
        )
    assert not hasattr(target, "transaction_template")

    foreign_arena = _dynamic_arena()
    foreign_target = foreign_arena.plan_target(transform_frame_index=0)
    foreign_prepared, foreign_readiness = _A2_AUTHORITIES[id(foreign_arena)]
    foreign_binding = build_cuda_c3_binding_from_arena_target(
        foreign_target,
        prepared_input=foreign_prepared,
        native_readiness=foreign_readiness,
    )
    with pytest.raises(NativeOpticalArenaProtocolError, match="exact a2 allocation"):
        arena.bind_c3_provenance(
            target=target,
            binding=foreign_binding,
            prepared_input=foreign_prepared,
            native_readiness=foreign_readiness,
        )


def test_arena_controlled_template_and_transaction_cannot_bypass_c4_lease():
    arena = _dynamic_arena()
    target = arena.plan_target(transform_frame_index=0)
    executable = arena._execution_for_target(target)
    assert not hasattr(target, "_transaction_template")
    assert not hasattr(target, "stages")
    assert not hasattr(target, "completion_event")

    with pytest.raises(CudaOpticalTimelineError, match="arena-issued capability"):
        executable.transaction_template.activate()
    with pytest.raises(CudaOpticalTimelineError, match="arena-issued capability"):
        CudaOpticalTransaction(_template=executable.transaction_template)
    with pytest.raises(CudaOpticalTimelineError, match="cannot be rebound"):
        CudaOpticalTransactionTemplate(
            context=executable.transaction_template.context,
            stages=executable.transaction_template.stages,
            completion_event=executable.transaction_template.completion_event,
            diagnostic_events=None,
        )
    assert not hasattr(executable, "authorize")
    dormant = executable.activate()
    assert not hasattr(dormant, "_bind_arena_execution_lease")
    with pytest.raises(CudaOpticalTimelineError, match="exact timeline-minted"):
        with dormant._authorize_arena_execution(
            executable.activation_capability,
            operation="enqueue",
        ):
            raise AssertionError("static construction capability authorized execution")
    prepared = arena.prepare_sample(target=target, provenance=_provenance(arena, target))
    assert not hasattr(dormant, "_bind_arena_execution_authority")
    with pytest.raises(CudaOpticalTimelineError, match="active arena lease"):
        executable.transaction_template.stages[0].enqueue(prepared._active._transaction)
    assert prepared._active._execution_authority is None
    with pytest.raises(CudaOpticalTimelineError, match="exact timeline-minted"):
        with prepared._active._transaction._authorize_arena_execution(
            prepared._active._execution_authority,
            operation="enqueue",
        ):
            raise AssertionError("dormant authority authorized execution")
    lifecycle = arena.lifecycle_evidence()
    assert lifecycle.slots[0].state == "FREE"
    assert lifecycle.terminal_event_record_count == 0

    active = arena.acquire(prepared)
    assert not isinstance(active, NativeOpticalArenaBackpressure)
    with pytest.raises(CudaOpticalTimelineError, match="active arena lease"):
        executable.transaction_template.stages[0].enqueue(active._transaction)
    assert arena.lifecycle_evidence().terminal_event_record_count == 0
    execution_authority = active._execution_authority
    transaction = active._transaction
    active.cancel_before_enqueue()
    with pytest.raises(NativeOpticalArenaProtocolError):
        with transaction._authorize_arena_execution(
            execution_authority,
            operation="enqueue",
        ):
            raise AssertionError("cancelled authority authorized execution")


def test_one_real_c4_lease_cannot_mint_authority_for_a_second_transaction():
    arena = _dynamic_arena()
    target0 = arena.plan_target(transform_frame_index=0)
    active0 = arena.acquire(arena.prepare_sample(target=target0, provenance=_provenance(arena, target0)))
    assert not isinstance(active0, NativeOpticalArenaBackpressure)

    target1 = arena.plan_target(transform_frame_index=1)
    prepared1 = arena.prepare_sample(target=target1, provenance=_provenance(arena, target1))
    transaction1 = prepared1._active._transaction
    with pytest.raises(NativeOpticalArenaProtocolError, match="already bound"):
        transaction1._mint_arena_execution_authority(
            prepared1._execution_capability,
            active0._control_lease,
            active_owner=prepared1._active,
        )

    assert active0.state == "ACQUIRED"
    assert prepared1._active.state == "DORMANT"
    lifecycle = arena.lifecycle_evidence()
    assert lifecycle.slots[0].state == "LEASED"
    assert lifecycle.terminal_event_record_count == 0
    active0.cancel_before_enqueue()


def test_prepared_execution_capability_rejects_duck_owner_and_empty_binding():
    arena = _dynamic_arena()
    target = arena.plan_target(transform_frame_index=0)
    provenance = _provenance(arena, target)
    transaction = arena._execution_for_target(target).activate()

    class DuckOwner:
        _transaction = transaction
        _control_lease = None
        _state = "DORMANT"

    with pytest.raises(NativeOpticalArenaProtocolError, match="owner is invalid"):
        transaction._prepare_arena_execution_capability(
            active_owner=DuckOwner(),
            plan_binding_sha256=provenance.plan_binding_sha256,
            ordered_runtime_stage_evidence_sha256=(provenance.ordered_runtime_stage_evidence_sha256),
            c3_one_shot_binding_sha256="",
            ordered_stage_evidence_sha256=provenance.ordered_stage_evidence_sha256,
            dataflow_binding_sha256=provenance.dataflow_binding_sha256,
        )

    exact_owner = CudaActiveArenaSample(
        target=target,
        provenance=provenance,
        transaction=transaction,
        executable=arena._execution_for_target(target),
        arena_acquire_authority=arena._acquire_authority,
    )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        transaction._prepare_arena_execution_capability(
            active_owner=exact_owner,
            plan_binding_sha256=provenance.plan_binding_sha256,
            ordered_runtime_stage_evidence_sha256=(provenance.ordered_runtime_stage_evidence_sha256),
            c3_one_shot_binding_sha256="",
            ordered_stage_evidence_sha256=provenance.ordered_stage_evidence_sha256,
            dataflow_binding_sha256=provenance.dataflow_binding_sha256,
        )
    assert transaction.state == "validated_open"
    assert arena.lifecycle_evidence().slots[0].state == "FREE"


def test_duck_typed_and_unregistered_live_authorities_cannot_execute():
    arena = _dynamic_arena()
    target = arena.plan_target(transform_frame_index=0)
    executable = arena._execution_for_target(target)
    transaction = executable.activate()

    class ForgedAuthority:
        def _begin(self, *args, **kwargs):
            del args, kwargs

        def _assert_execution(self, *args, **kwargs):
            del args, kwargs

        def _end(self, *args, **kwargs):
            del args, kwargs

    ForgedAuthority.__module__ = "optics.cuda_timeline"
    ForgedAuthority.__name__ = "_CudaArenaLiveExecutionAuthority"
    forged = ForgedAuthority()
    with pytest.raises(CudaOpticalTimelineError, match="exact timeline-minted"):
        with transaction._authorize_arena_execution(forged, operation="enqueue"):
            for stage in executable.transaction_template.stages:
                stage.enqueue(transaction)

    unregistered_exact = object.__new__(_CudaArenaLiveExecutionAuthority)
    with pytest.raises(CudaOpticalTimelineError, match="live C4 lease"):
        with transaction._authorize_arena_execution(
            unregistered_exact,
            operation="enqueue",
        ):
            for stage in executable.transaction_template.stages:
                stage.enqueue(transaction)
    lifecycle = arena.lifecycle_evidence()
    assert transaction.state == "validated_open"
    assert lifecycle.slots[0].state == "FREE"
    assert lifecycle.terminal_event_record_count == 0


def test_replacing_registered_live_authority_fails_closed():
    arena = _dynamic_arena()
    target = arena.plan_target(transform_frame_index=0)
    active = arena.acquire(arena.prepare_sample(target=target, provenance=_provenance(arena, target)))
    assert not isinstance(active, NativeOpticalArenaBackpressure)
    unregistered_exact = object.__new__(_CudaArenaLiveExecutionAuthority)
    object.__setattr__(active, "_execution_authority", unregistered_exact)

    with pytest.raises(CudaOpticalTimelineError, match="live C4 lease"):
        active.enqueue()
    assert active.state == "POISONED"
    assert arena.state == "OPEN"
    assert arena.lifecycle_evidence().slots[0].state == "FREE"
    assert arena.lifecycle_evidence().terminal_event_record_count == 0


def test_live_execution_authority_cannot_finish_without_c4_lifecycle_commit():
    arena = _dynamic_arena()
    target = arena.plan_target(transform_frame_index=0)
    active = arena.acquire(arena.prepare_sample(target=target, provenance=_provenance(arena, target)))
    assert not isinstance(active, NativeOpticalArenaBackpressure)
    transaction = active._transaction
    authority = active._execution_authority
    executable = arena._execution_for_target(target)

    with transaction._authorize_arena_execution(authority, operation="enqueue"):
        for stage in executable.transaction_template.stages:
            stage.enqueue(transaction)
    assert active.state == "ENQUEUED"
    assert active._control_lease.state == "ENQUEUING"

    with transaction._authorize_arena_execution(authority, operation="finish"):
        completion = transaction.finish()
    assert active.state == "COMPLETION_RECORDED"
    assert active.completion is completion
    assert active._control_lease.state == "COMPLETION_RECORDED"
    assert arena.lifecycle_evidence().terminal_event_record_count == 1

    active.resolve()
    active.release_device_result()
    active.reclaim()
    with pytest.raises(NativeOpticalArenaProtocolError):
        with transaction._authorize_arena_execution(authority, operation="finish"):
            raise AssertionError("reclaimed authority authorized execution")


def test_poisoned_live_execution_authority_cannot_be_reused():
    arena = _dynamic_arena()
    target = arena.plan_target(transform_frame_index=0)
    active = arena.acquire(arena.prepare_sample(target=target, provenance=_provenance(arena, target)))
    assert not isinstance(active, NativeOpticalArenaBackpressure)
    active.enqueue()
    transaction = active._transaction
    authority = active._execution_authority
    active._control_lease.mark_submitted_failure("injected authority poison")

    with pytest.raises(NativeOpticalArenaProtocolError):
        with transaction._authorize_arena_execution(authority, operation="finish"):
            raise AssertionError("poisoned authority authorized execution")


@pytest.mark.parametrize("arena_factory", (_dynamic_arena, _fixed_arena))
def test_c3_adapter_rejects_same_content_reupload_instead_of_a2_allocation(arena_factory):
    arena = arena_factory()
    target = arena.plan_target(transform_frame_index=0 if arena.spec.boundary != "R0" else None)
    prepared, _readiness = _A2_AUTHORITIES[id(arena)]
    duplicate = prepare_cuda_native_input(prepared.source, device=f"cuda:{prepared.device_ordinal}")
    duplicate_validation = validate_prepared_native_input(prepared.source, duplicate)
    duplicate_readiness = certify_native_readiness(
        duplicate,
        duplicate_validation,
        compile_complete=True,
        warmup_frames_completed=0,
    )

    with pytest.raises(ValueError, match="exact a2 allocation"):
        build_cuda_c3_binding_from_arena_target(
            target,
            prepared_input=duplicate,
            native_readiness=duplicate_readiness,
        )
    valid_binding = build_cuda_c3_binding_from_arena_target(
        target,
        prepared_input=prepared,
        native_readiness=_readiness,
    )
    with pytest.raises(NativeOpticalArenaProtocolError, match="exact a2 allocation"):
        arena.bind_c3_provenance(
            target=target,
            binding=valid_binding,
            prepared_input=duplicate,
            native_readiness=duplicate_readiness,
        )


def test_c3_adapter_rejects_wrong_replay_view_and_uncertified_readiness():
    arena = _dynamic_arena()
    target = arena.plan_target(transform_frame_index=1)
    prepared, readiness = _A2_AUTHORITIES[id(arena)]
    duplicate = prepare_cuda_native_input(prepared.source, device=f"cuda:{prepared.device_ordinal}")
    wrong_target = replace(
        target,
        _transform_arrays={
            name: duplicate.arrays[name][1:2] for name in ("replay_x_world_R", "replay_x_world_r")
        },
    )
    with pytest.raises(ValueError, match="approved a2 replay view"):
        build_cuda_c3_binding_from_arena_target(
            wrong_target,
            prepared_input=prepared,
            native_readiness=readiness,
        )

    with pytest.raises(ValueError, match="not issued by the canonical a2"):
        build_cuda_c3_binding_from_arena_target(
            target,
            prepared_input=prepared,
            native_readiness=replace(readiness),
        )


def test_c3_adapter_revalidates_a2_content_instead_of_trusting_correct_hashes():
    arena = _dynamic_arena()
    target = arena.plan_target(transform_frame_index=0)
    prepared, readiness = _A2_AUTHORITIES[id(arena)]
    tensor = prepared.arrays["material_albedo_rgb"]
    original = tensor.clone()
    tensor.add_(1.0)
    try:
        with pytest.raises(ValueError, match="values do not match logical source"):
            build_cuda_c3_binding_from_arena_target(
                target,
                prepared_input=prepared,
                native_readiness=readiness,
            )
    finally:
        tensor.copy_(original)


def test_c3_adapter_rejects_same_a2_tensor_rebound_before_first_enqueue():
    arena = _dynamic_arena()
    target = arena.plan_target(transform_frame_index=0)
    prepared, readiness = _A2_AUTHORITIES[id(arena)]
    tensor = prepared.arrays["material_albedo_rgb"]
    original_storage = tensor.untyped_storage()
    original_pointer = tensor.data_ptr()
    replacement = torch.empty_like(tensor)
    try:
        tensor.set_(replacement)
        assert tensor.data_ptr() != original_pointer
        with pytest.raises(ValueError, match="physical allocation"):
            build_cuda_c3_binding_from_arena_target(
                target,
                prepared_input=prepared,
                native_readiness=readiness,
            )
        assert arena.lifecycle_evidence().terminal_event_record_count == 0
    finally:
        tensor.set_(original_storage, 0, tensor.shape, tensor.stride())


def test_c3_binding_serialization_cannot_forge_adapter_issuance():
    arena = _dynamic_arena()
    target = arena.plan_target(transform_frame_index=0)
    prepared, readiness = _A2_AUTHORITIES[id(arena)]
    issued = build_cuda_c3_binding_from_arena_target(
        target,
        prepared_input=prepared,
        native_readiness=readiness,
    )
    copied = C3OneShotNativeExecutionBindingEvidence.from_payload(issued.to_payload())

    with pytest.raises(NativeOpticalArenaProtocolError, match="not issued"):
        arena.bind_c3_provenance(
            target=target,
            binding=copied,
            prepared_input=prepared,
            native_readiness=readiness,
        )


def test_c3_moved_registry_ignores_injected_authority_method():
    from optical_worker.provenance import C3OneShotNativeExecutionBindingEvidence as ProductionBinding

    arena = _dynamic_arena()
    target = arena.plan_target(transform_frame_index=0)
    prepared, readiness = _A2_AUTHORITIES[id(arena)]
    binding = build_cuda_c3_binding_from_arena_target(
        target, prepared_input=prepared, native_readiness=readiness
    )
    assert type(binding) is ProductionBinding is C3OneShotNativeExecutionBindingEvidence
    calls = []
    forged = replace(binding)
    object.__setattr__(forged, "_assert_cuda_arena_adapter_authority", lambda **kw: calls.append(kw))
    with pytest.raises(NativeOpticalArenaProtocolError, match="canonical arena adapter"):
        arena.bind_c3_provenance(
            target=target, binding=forged, prepared_input=prepared, native_readiness=readiness
        )
    assert not calls
    object.__setattr__(binding, "_assert_cuda_arena_adapter_authority", lambda **kw: calls.append(kw))
    arena.bind_c3_provenance(
        target=target, binding=binding, prepared_input=prepared, native_readiness=readiness
    )
    assert not calls
    assert arena.lifecycle_evidence().terminal_event_record_count == 0
    arena.begin_drain()
    arena.close()


def test_measured_acquire_uses_only_constant_time_authority_checks(monkeypatch):
    arena = _dynamic_arena()
    target = arena.plan_target(transform_frame_index=0)
    prepared = arena.prepare_sample(target=target, provenance=_provenance(arena, target))

    def forbidden_graph_walk(*args, **kwargs):
        del args, kwargs
        raise AssertionError("measured acquire traversed the sealed resource graph")

    monkeypatch.setattr(CudaNativeOpticalArena, "_validate_configuration", forbidden_graph_walk)
    monkeypatch.setattr(CudaNativeOpticalArena, "_configuration_identity", forbidden_graph_walk)
    monkeypatch.setattr(CudaPreparedArenaSample, "_validate_bound", forbidden_graph_walk)
    active = arena.acquire(prepared)
    assert not isinstance(active, NativeOpticalArenaBackpressure)
    active.cancel_before_enqueue()


def test_actual_timed_enqueue_does_not_revalidate_or_materialize_render_evidence(monkeypatch):
    from optical_worker.timing import native_timed_submission
    from optics.cuda_direct_light import CudaDirectLightRenderStage

    arena = _fixed_arena()
    target = arena.plan_target(transform_frame_index=None)
    prepared = arena.prepare_sample(target=target, provenance=_provenance(arena, target))
    stage = arena._execution_for_target(target).transaction_template.stages[0]
    evidence = stage._cuda_transaction_stage_evidence()
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("full render preflight/evidence materialization inside submission")

    with monkeypatch.context() as patch:
        patch.setattr(CudaDirectLightRenderStage, "_validate_cuda_transaction_stage", forbidden)
        patch.setattr(CudaDirectLightRenderStage, "_build_stage_evidence", forbidden)
        with native_timed_submission(row_id="c2c4-native-regression", lane="cuda", boundary="R0"):
            active = arena.acquire(prepared)
            active.enqueue()
            active.finish()
    assert not calls
    assert stage._cuda_transaction_stage_evidence() is evidence
    assert arena.lifecycle_evidence().terminal_event_record_count == 1
    active.resolve()
    active.release_device_result()
    active.reclaim()
    arena.begin_drain()
    arena.close()


@pytest.mark.parametrize("field", ["max_distance", "sensor_role_mask", "num_planes", "num_bvh_nodes"])
def test_cached_render_evidence_does_not_hide_inline_first_hit_scalar_drift(field):
    arena = _fixed_arena()
    target = arena.plan_target(transform_frame_index=None)
    provenance = _provenance(arena, target)
    stage = arena._execution_for_target(target).transaction_template.stages[0]
    inputs = stage.first_hit_inputs
    original = getattr(inputs, field)
    before = arena.lifecycle_evidence()
    object.__setattr__(inputs, field, original + 1)
    try:
        with pytest.raises(ValueError, match="changed"):
            arena.prepare_sample(target=target, provenance=provenance)
        assert arena.lifecycle_evidence() == before
        assert before.terminal_event_record_count == 0
    finally:
        object.__setattr__(inputs, field, original)
    arena.begin_drain()
    arena.close()


def test_cuda_arena_restores_ambient_device_across_every_generation_transition():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible CUDA devices")
    arena = _dynamic_arena(device="cuda:1")
    target = arena.plan_target(transform_frame_index=0)
    prepared = arena.prepare_sample(target=target, provenance=_provenance(arena, target))

    torch.cuda.set_device(0)
    assert torch.cuda.current_device() == 0
    active = arena.acquire(prepared)
    assert not isinstance(active, NativeOpticalArenaBackpressure)
    assert torch.cuda.current_device() == 0
    active.enqueue()
    assert torch.cuda.current_device() == 0
    active.finish()
    assert torch.cuda.current_device() == 0
    active.resolve()
    assert torch.cuda.current_device() == 0
    active.release_device_result()
    assert torch.cuda.current_device() == 0
    active.reclaim()
    assert torch.cuda.current_device() == 0
