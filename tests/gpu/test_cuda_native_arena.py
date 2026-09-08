from __future__ import annotations

import pytest
import torch

from benchmarks.optical.cuda_arena import build_cuda_c3_binding_from_arena_target
from optics.cuda_arena import (
    CudaArenaC3Provenance,
    CudaDeviceOnlyDeliveryStage,
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
from optics.cuda_lbvh import CudaLbvhRebuildStageFactory
from optics.cuda_scene import (
    CudaSceneUpdateStageFactory,
    build_cuda_optical_scene,
    cuda_replay_transform_inputs,
)
from optics.cuda_timeline import CudaOpticalExecutionContext
from optics.native_arena import (
    NATIVE_OPTICAL_ARENA_LAYOUT_SCHEMA_VERSION,
    NATIVE_OPTICAL_ARENA_SPEC_SCHEMA_VERSION,
    NativeOpticalArenaBackpressure,
    NativeOpticalArenaProtocolError,
    NativeOpticalArenaSpec,
)
from optics.registry import (
    OpticalInstanceSpec,
    OpticalLightSpec,
    OpticalMaterialSpec,
    OpticalWorldRegistry,
)
from optics.scene_contract import PackedOpticalSceneHost

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


def _registry() -> OpticalWorldRegistry:
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat", albedo_rgb=(0.4, 0.6, 0.8)))
    registry.add_light(
        OpticalLightSpec(
            "sun",
            "directional",
            position_or_direction_world=(0.0, 0.0, -1.0),
            intensity=1.0,
        )
    )
    registry.add_triangle_mesh_geometry(
        "triangle",
        vertices_local=((-2.0, -2.0, 2.0), (2.0, -2.0, 2.0), (0.0, 2.0, 2.0)),
        triangles=((0, 1, 2),),
    )
    registry.add_instance(
        OpticalInstanceSpec(
            "instance",
            "triangle",
            "mat",
            body_index=0,
            roles=frozenset({"rgb", "depth", "segmentation"}),
        )
    )
    return registry


def _spec(
    context: CudaOpticalExecutionContext,
    *,
    boundary: str,
    scene_sha256: str,
    replay_sha256: str | None,
    frame_count: int,
    diagnostic: bool = False,
) -> NativeOpticalArenaSpec:
    dynamic = boundary != "R0"
    return NativeOpticalArenaSpec(
        schema_version=NATIVE_OPTICAL_ARENA_SPEC_SCHEMA_VERSION,
        row_id=f"unit_{boundary.lower()}",
        case_id="unit_cuda_arena",
        case_sha256="1" * 64,
        boundary=boundary,
        lane="cuda",
        device_ordinal=int(context.device.index),
        execution_stream_token=cuda_execution_stream_token(context),
        input_contract="canonical_native_replay_v1" if dynamic else "fixed_native_scene_v1",
        native_input_manifest_sha256="2" * 64,
        native_readiness_sha256="3" * 64,
        scene_contract_sha256=scene_sha256,
        transform_replay_sha256=replay_sha256 if dynamic else None,
        transform_frame_count=frame_count if dynamic else 0,
        fixed_render_fixture_sha256=None if dynamic else "4" * 64,
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
        primitive_count=1,
        node_capacity=1,
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
    binding = build_cuda_c3_binding_from_arena_target(target)
    return arena.bind_c3_provenance(target=target, binding=binding)


def _dynamic_arena(
    *,
    boundary: str = "R1",
    diagnostic: bool = False,
    device: str = "cuda:0",
):
    _require_cuda()
    context = _context(device)
    packed = PackedOpticalSceneHost.from_registry(_registry(), scene_id="unit_cuda_arena")
    local_scene = build_cuda_optical_scene(packed, device=context.device)
    local_scene.ready_event.synchronize()
    replay_sha256 = "5" * 64
    rotations = torch.eye(3, dtype=torch.float32, device=context.device).reshape(1, 1, 3, 3)
    rotations = rotations.repeat(2, 1, 1, 1).contiguous()
    translations = torch.zeros((2, 1, 3), dtype=torch.float32, device=context.device)
    translations[1, 0, 0] = 0.1
    transforms = tuple(
        cuda_replay_transform_inputs(
            rotations,
            translations,
            frame_index=index,
            input_contract="canonical_native_replay_v1",
            transform_replay_sha256=replay_sha256,
            resources=(rotations, translations),
        )
        for index in range(2)
    )
    scene_factory = CudaSceneUpdateStageFactory.prepare(
        context=context,
        local_scene=local_scene,
        native_input_manifest_sha256="2" * 64,
        input_contract="canonical_native_replay_v1",
    )
    lbvh_factory = CudaLbvhRebuildStageFactory.prepare(
        context=context,
        primitive_count=1,
        native_input_manifest_sha256="2" * 64,
        scene_contract_sha256=packed.scene_contract_sha256,
        input_contract="canonical_native_replay_v1",
    )
    renderer = CudaDirectLightNativeRenderer.prepare(context)
    spec = _spec(
        context,
        boundary=boundary,
        scene_sha256=packed.scene_contract_sha256,
        replay_sha256=replay_sha256,
        frame_count=2,
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
        sensor_role_mask=local_scene.role_to_mask["rgb"],
        max_distance=10.0,
        ambient_rgb=(0.1, 0.1, 0.1),
        background_rgb=(0.0, 0.0, 0.0),
        shadow_bias=1.0e-6,
    )
    return arena


def _fixed_inputs(device):
    empty_f3 = torch.empty((0, 3), dtype=torch.float32, device=device)
    empty_i32 = torch.empty((0,), dtype=torch.int32, device=device)
    empty_i64 = torch.empty((0,), dtype=torch.int64, device=device)
    triangle_v0 = torch.tensor([[-2.0, -2.0, 2.0]], dtype=torch.float32, device=device)
    triangle_e1 = torch.tensor([[4.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    triangle_e2 = torch.tensor([[2.0, 4.0, 0.0]], dtype=torch.float32, device=device)
    first = CudaFirstHitInputs(
        plane_normal_world=empty_f3,
        plane_point_world=torch.empty_like(empty_f3),
        plane_numeric_instance_id=empty_i32,
        plane_source_order_key=empty_i64,
        plane_role_mask=torch.empty_like(empty_i64),
        plane_material_index=torch.empty_like(empty_i32),
        num_planes=0,
        triangle_v0_world=triangle_v0,
        triangle_e1_world=triangle_e1,
        triangle_e2_world=triangle_e2,
        triangle_normal_world=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=device),
        triangle_numeric_instance_id=torch.tensor([1], dtype=torch.int32, device=device),
        triangle_source_order_key=torch.tensor([0], dtype=torch.int64, device=device),
        triangle_role_mask=torch.tensor([1], dtype=torch.int64, device=device),
        triangle_material_index=torch.tensor([0], dtype=torch.int32, device=device),
        bvh_bounds_min=torch.tensor([[-2.0, -2.0, 2.0]], dtype=torch.float32, device=device),
        bvh_bounds_max=torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32, device=device),
        bvh_left=torch.tensor([-1], dtype=torch.int32, device=device),
        bvh_right=torch.tensor([-1], dtype=torch.int32, device=device),
        bvh_start=torch.tensor([0], dtype=torch.int32, device=device),
        bvh_count=torch.tensor([1], dtype=torch.int32, device=device),
        bvh_prim_ids=torch.tensor([0], dtype=torch.int32, device=device),
        num_bvh_nodes=1,
        sensor_role_mask=1,
        max_distance=10.0,
    )
    shade = CudaShadeInputs(
        material_albedo_rgb=torch.tensor([[0.4, 0.6, 0.8]], dtype=torch.float32, device=device),
        light_kind=torch.tensor([0], dtype=torch.int32, device=device),
        light_position_or_direction_world=torch.tensor(
            [[0.0, 0.0, -1.0]], dtype=torch.float32, device=device
        ),
        light_intensity=torch.tensor([1.0], dtype=torch.float32, device=device),
        light_color_rgb=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32, device=device),
        num_lights=1,
    )
    return first, shade


def _fixed_contract_arrays(first, shade, device):
    packed = PackedOpticalSceneHost.from_registry(_registry(), scene_id="unit_cuda_arena")
    runtime_scene = {name: torch.tensor(value.copy(), device=device) for name, value in packed.arrays.items()}
    runtime_scene.update(
        {
            name: value
            for name, value in first.tensor_items()
            if name
            in {
                "plane_numeric_instance_id",
                "plane_source_order_key",
                "plane_role_mask",
                "plane_material_index",
                "triangle_numeric_instance_id",
                "triangle_source_order_key",
                "triangle_role_mask",
                "triangle_material_index",
            }
        }
    )
    runtime_scene.update(dict(shade.tensor_items()))
    derived = {
        name: value
        for name, value in first.tensor_items()
        if name
        in {
            "plane_normal_world",
            "plane_point_world",
            "triangle_v0_world",
            "triangle_e1_world",
            "triangle_e2_world",
            "triangle_normal_world",
            "bvh_bounds_min",
            "bvh_bounds_max",
            "bvh_left",
            "bvh_right",
            "bvh_start",
            "bvh_count",
            "bvh_prim_ids",
        }
    }
    derived.update(
        {
            "triangle_aabb_min": first.bvh_bounds_min.clone(),
            "triangle_aabb_max": first.bvh_bounds_max.clone(),
            "bvh_node_depth": torch.tensor([0], dtype=torch.int32, device=device),
            "bvh_level_ranges": torch.tensor([[0, 1]], dtype=torch.int32, device=device),
        }
    )
    return runtime_scene, derived


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
    assert isinstance(target._transaction_template.stages[-1], CudaDeviceOnlyDeliveryStage)
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
    _require_cuda()
    context = _context()
    first, shade = _fixed_inputs(context.device)
    runtime_scene, derived_fixture = _fixed_contract_arrays(first, shade, context.device)
    spec = _spec(
        context,
        boundary="R0",
        scene_sha256="6" * 64,
        replay_sha256=None,
        frame_count=0,
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
        runtime_scene_arrays=runtime_scene,
        derived_fixture_arrays=derived_fixture,
    )
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
    context = target._transaction_template.context
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
    foreign_binding = build_cuda_c3_binding_from_arena_target(foreign_target)
    with pytest.raises(NativeOpticalArenaProtocolError, match="exact allocation"):
        arena.bind_c3_provenance(target=target, binding=foreign_binding)


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
