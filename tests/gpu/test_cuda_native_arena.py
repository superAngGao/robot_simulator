from __future__ import annotations

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
from optics.cuda_arena import (
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


def _camera() -> CudaCameraParameters:
    return CudaCameraParameters(
        width=2,
        height=1,
        fx=2.0,
        fy=2.0,
        cx=0.5,
        cy=0.5,
        origin=(0.0, 0.0, 0.0),
        rotation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
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
        width=2,
        height=1,
        num_rays=2,
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
        camera=_camera(),
        sensor_role_mask=scene_factory.local_scene.role_to_mask["rgb"],
        max_distance=10.0,
        ambient_rgb=(0.1, 0.1, 0.1),
        background_rgb=(0.0, 0.0, 0.0),
        shadow_bias=1.0e-6,
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
        max_distance=10.0,
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


def _fixed_arena(*, device: str = "cuda:0"):
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
        camera=_camera(),
        ambient_rgb=(0.1, 0.1, 0.1),
        background_rgb=(0.0, 0.0, 0.0),
        shadow_bias=1.0e-6,
        runtime_scene_arrays=fixed.scene_arrays,
        derived_fixture_arrays=fixed.derived_fixture_arrays,
        resource_owners=(prepared,),
    )
    _A2_AUTHORITIES[id(arena)] = (prepared, readiness)
    return arena


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
    assert by_name["slot0.render.rgb"][1].shape == (2, 3)
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
    with pytest.raises(CudaOpticalTimelineError, match="live C4 lease"):
        with dormant._authorize_arena_execution(
            executable.activation_capability,
            operation="enqueue",
        ):
            raise AssertionError("static construction capability authorized execution")
    prepared = arena.prepare_sample(target=target, provenance=_provenance(arena, target))
    with pytest.raises(NativeOpticalArenaProtocolError, match="transaction binding"):
        dormant._bind_arena_execution_authority(
            executable.activation_capability,
            prepared._active._execution_authority,
        )
    with pytest.raises(CudaOpticalTimelineError, match="active arena lease"):
        executable.transaction_template.stages[0].enqueue(prepared._active._transaction)
    with pytest.raises(NativeOpticalArenaProtocolError, match="lease binding"):
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
