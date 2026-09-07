from __future__ import annotations

import gc
import hashlib
import json
import weakref
from dataclasses import replace

import pytest

from optics.native_arena import (
    C4_ACTIVE_LEASE_BINDING_SCHEMA_VERSION,
    C4_ARENA_LIFECYCLE_SCHEMA_VERSION,
    C4_PREPARED_SAMPLE_PLAN_SCHEMA_VERSION,
    NATIVE_OPTICAL_ARENA_LAYOUT_SCHEMA_VERSION,
    NATIVE_OPTICAL_ARENA_SPEC_SCHEMA_VERSION,
    NATIVE_OPTICAL_RESOURCE_MANIFEST_SCHEMA_VERSION,
    NATIVE_OPTICAL_SLOT_RESOURCE_SCHEMA_VERSION,
    C4ActiveLeaseBindingEvidence,
    C4ArenaLifecycleEvidence,
    C4PreparedSamplePlanEvidence,
    NativeOpticalArena,
    NativeOpticalArenaBackpressure,
    NativeOpticalArenaPoisonedError,
    NativeOpticalArenaProtocolError,
    NativeOpticalArenaResourceManifest,
    NativeOpticalArenaSpec,
    NativeOpticalSlotResourceManifest,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_SHA_E = "e" * 64


class _Owner:
    pass


def _spec(
    *,
    boundary: str = "R1",
    payload: str = "native_only",
    delivery_policy: str = "native_only",
    payload_contract_id: str | None = None,
    ring_depth: int = 1,
) -> NativeOpticalArenaSpec:
    dynamic = boundary != "R0"
    return NativeOpticalArenaSpec(
        schema_version=NATIVE_OPTICAL_ARENA_SPEC_SCHEMA_VERSION,
        row_id=f"row_{boundary.lower()}",
        case_id="case_v1",
        case_sha256=_SHA_A,
        boundary=boundary,
        lane="cuda",
        device_ordinal=1,
        execution_stream_token="torch-stream:1:17",
        input_contract="canonical_native_replay_v1" if dynamic else "fixed_native_scene_v1",
        native_input_manifest_sha256=_SHA_B,
        native_readiness_sha256=_SHA_C,
        scene_contract_sha256=_SHA_D,
        transform_replay_sha256=_SHA_E if dynamic else None,
        transform_frame_count=4 if dynamic else 0,
        fixed_render_fixture_sha256=None if dynamic else _SHA_E,
        renderer_id="cuda_direct_light",
        acceleration_class="production_dynamic" if dynamic else "fixed_bvh_fixture",
        acceleration_id=("cuda_morton_karras_lbvh_rebuild_v1" if dynamic else "fixed_bvh_fixture"),
        width=7,
        height=3,
        num_rays=21,
        primitive_count=5,
        node_capacity=9,
        output_profile="direct_light_full",
        shadows=True,
        diagnostic_mode="primary_unprofiled",
        payload=payload,
        delivery_policy=delivery_policy,
        payload_contract_id=payload_contract_id,
        ring_depth=ring_depth,
        ordered_delivery=True,
        arena_layout_schema_version=NATIVE_OPTICAL_ARENA_LAYOUT_SCHEMA_VERSION,
    )


def _manifest(
    spec: NativeOpticalArenaSpec,
) -> tuple[NativeOpticalArenaResourceManifest, dict[str, object]]:
    slots = tuple(
        NativeOpticalSlotResourceManifest(
            schema_version=NATIVE_OPTICAL_SLOT_RESOURCE_SCHEMA_VERSION,
            slot_id=slot_id,
            slot_storage_manifest_sha256=chr(ord("1") + slot_id) * 64,
            terminal_event_token=f"event:{slot_id}",
            resource_tokens=(f"slot-storage:{slot_id}", f"event:{slot_id}"),
            resource_owner_tokens=(f"slot-owner:{slot_id}", f"slot-owner:{slot_id}"),
            resource_nbytes=(1024 + slot_id, 0),
        )
        for slot_id in range(spec.ring_depth)
    )
    manifest = NativeOpticalArenaResourceManifest(
        schema_version=NATIVE_OPTICAL_RESOURCE_MANIFEST_SCHEMA_VERSION,
        arena_spec_sha256=spec.sha256,
        session_resource_tokens=("session-workspace",),
        session_resource_owner_tokens=("session-owner",),
        session_resource_nbytes=(256,),
        slots=slots,
        total_resource_nbytes=256 + sum(slot.total_resource_nbytes for slot in slots),
    )
    owners = {
        "session-owner": _Owner(),
        **{f"slot-owner:{slot.slot_id}": _Owner() for slot in slots},
    }
    return manifest, owners


def _arena(spec: NativeOpticalArenaSpec | None = None) -> NativeOpticalArena:
    resolved = _spec() if spec is None else spec
    manifest, owners = _manifest(resolved)
    return NativeOpticalArena(
        spec=resolved,
        resource_manifest=manifest,
        resource_owners=owners,
    )


def _plan(
    arena: NativeOpticalArena,
    *,
    frame: int | None = 0,
    planned_acquisition_offset: int = 0,
):
    return arena.prepare_plan(
        transform_frame_index=frame,
        c3_one_shot_binding_sha256=_SHA_A,
        ordered_stage_evidence_sha256=_SHA_B,
        dataflow_binding_sha256=_SHA_C,
        planned_acquisition_offset=planned_acquisition_offset,
    )


def _complete_and_reclaim(lease) -> None:
    lease.begin_enqueue()
    lease.record_completion()
    lease.resolve_terminal_completion()
    lease.release_device_result()
    assert lease.state == "RECLAIMABLE"
    lease.reclaim()


def _rehash_transition_trace(payload: dict[str, object]) -> None:
    encoded = json.dumps(
        payload["transitions"],
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    payload["transition_trace_sha256"] = hashlib.sha256(encoded).hexdigest()


def test_spec_resource_plan_and_lifecycle_evidence_round_trip_canonically():
    spec = _spec()
    manifest, owners = _manifest(spec)
    assert NativeOpticalArenaSpec.from_payload(spec.to_payload()) == spec
    assert NativeOpticalArenaResourceManifest.from_payload(manifest.to_payload()) == manifest

    arena = NativeOpticalArena(
        spec=spec,
        resource_manifest=manifest,
        resource_owners=owners,
    )
    plan = _plan(arena)
    assert plan.result_capable is False
    assert plan.evidence.schema_version == C4_PREPARED_SAMPLE_PLAN_SCHEMA_VERSION
    assert C4PreparedSamplePlanEvidence.from_payload(plan.evidence.to_payload()) == plan.evidence

    lease = arena.acquire(plan)
    assert not isinstance(lease, NativeOpticalArenaBackpressure)
    assert lease.result_capable is False
    assert lease.evidence.schema_version == C4_ACTIVE_LEASE_BINDING_SCHEMA_VERSION
    assert C4ActiveLeaseBindingEvidence.from_payload(lease.evidence.to_payload()) == lease.evidence
    _complete_and_reclaim(lease)

    lifecycle = arena.lifecycle_evidence()
    assert lifecycle.schema_version == C4_ARENA_LIFECYCLE_SCHEMA_VERSION
    assert lifecycle.slot_reclaim_count == 1
    assert lifecycle.terminal_event_record_count == 1
    assert lifecycle.completion_resolution_count == 1
    assert lifecycle.resource_release_authorized is False
    assert C4ArenaLifecycleEvidence.from_payload(lifecycle.to_payload()) == lifecycle


def test_exact_depth_one_slot_backpressures_without_mutating_generation():
    arena = _arena()
    first = arena.acquire(_plan(arena, frame=0))
    assert not isinstance(first, NativeOpticalArenaBackpressure)
    blocked_plan = _plan(arena, frame=1)
    blocked = arena.acquire(blocked_plan)
    assert isinstance(blocked, NativeOpticalArenaBackpressure)
    assert blocked.slot_id == 0
    assert blocked.active_generation == 1
    assert blocked.backpressure_count == 1

    _complete_and_reclaim(first)
    second = arena.acquire(blocked_plan)
    assert not isinstance(second, NativeOpticalArenaBackpressure)
    assert second.evidence.generation == 2


def test_discarded_speculative_plan_does_not_block_later_preparation():
    arena = _arena()
    discarded = _plan(arena)
    discarded_ref = weakref.ref(discarded)
    del discarded
    gc.collect()
    assert discarded_ref() is None

    replacement = _plan(arena)
    lease = arena.acquire(replacement)
    assert not isinstance(lease, NativeOpticalArenaBackpressure)
    assert lease.evidence.generation == 1


def test_competing_plans_are_non_reserving_and_only_first_acquire_wins():
    arena = _arena()
    winner = _plan(arena, frame=0)
    stale_competitor = _plan(arena, frame=1)
    lease = arena.acquire(winner)
    assert not isinstance(lease, NativeOpticalArenaBackpressure)
    before = arena.lifecycle_evidence()

    with pytest.raises(NativeOpticalArenaProtocolError, match="stale"):
        arena.acquire(stale_competitor)
    after = arena.lifecycle_evidence()
    assert after.transition_trace_sha256 == before.transition_trace_sha256
    assert lease.state == "LEASED"


def test_depth_two_batch_plans_every_slot_before_any_measured_acquire():
    spec = _spec(
        boundary="R2",
        payload="rgb",
        delivery_policy="torch_async_ordered",
        payload_contract_id="rgb_native_host_v1",
        ring_depth=2,
    )
    arena = _arena(spec)
    first_plan = _plan(arena, frame=0, planned_acquisition_offset=0)
    second_plan = _plan(arena, frame=1, planned_acquisition_offset=1)
    assert first_plan.evidence.slot_id == 0
    assert second_plan.evidence.slot_id == 1
    assert first_plan.evidence.expected_generation == 1
    assert second_plan.evidence.expected_generation == 1

    first = arena.acquire(first_plan)
    second = arena.acquire(second_plan)
    assert not isinstance(first, NativeOpticalArenaBackpressure)
    assert not isinstance(second, NativeOpticalArenaBackpressure)
    assert (first.evidence.slot_id, second.evidence.slot_id) == (0, 1)
    second.cancel_before_enqueue()
    first.cancel_before_enqueue()


def test_pre_enqueue_cancel_burns_generation_and_keeps_fifo_progress_safe():
    arena = _arena()
    first = arena.acquire(_plan(arena, frame=0))
    assert not isinstance(first, NativeOpticalArenaBackpressure)
    first.cancel_before_enqueue()
    with pytest.raises(NativeOpticalArenaProtocolError, match="stale"):
        _ = first.state

    second = arena.acquire(_plan(arena, frame=1))
    assert not isinstance(second, NativeOpticalArenaBackpressure)
    assert second.evidence.generation == 2
    operations = [item.operation for item in arena.lifecycle_evidence().transitions]
    assert "cancelled_generation_burned" in operations


def test_serialized_evidence_and_cross_arena_plan_cannot_act_as_live_authority():
    arena_a = _arena()
    arena_b = _arena()
    plan = _plan(arena_a)
    serialized = C4PreparedSamplePlanEvidence.from_payload(plan.evidence.to_payload())
    assert serialized == plan.evidence
    with pytest.raises(NativeOpticalArenaProtocolError, match="not owned"):
        arena_b.acquire(plan)
    with pytest.raises(NativeOpticalArenaProtocolError, match="not owned"):
        arena_a.acquire(serialized)  # type: ignore[arg-type]


def test_sealed_plan_drift_is_rejected_before_slot_mutation():
    arena = _arena()
    plan = _plan(arena)
    changed = replace(plan.evidence, dataflow_binding_sha256=_SHA_D)
    object.__setattr__(plan, "_evidence", changed)
    with pytest.raises(NativeOpticalArenaProtocolError, match="evidence changed"):
        arena.acquire(plan)
    lifecycle = arena.lifecycle_evidence()
    assert [item.operation for item in lifecycle.transitions] == ["arena_opened"]


def test_lightweight_acquire_rejects_arena_spec_drift_before_slot_mutation():
    arena = _arena()
    plan = _plan(arena)
    object.__setattr__(arena, "_spec", replace(arena.spec, row_id="wrong_row"))
    with pytest.raises(NativeOpticalArenaProtocolError, match="specification/resources changed"):
        arena.acquire(plan)
    assert [item.operation for item in arena.lifecycle_evidence().transitions] == ["arena_opened"]


def test_resource_manifest_requires_exact_resource_owner_byte_alignment():
    with pytest.raises(ValueError, match="must align"):
        NativeOpticalSlotResourceManifest(
            schema_version=NATIVE_OPTICAL_SLOT_RESOURCE_SCHEMA_VERSION,
            slot_id=0,
            slot_storage_manifest_sha256="1" * 64,
            terminal_event_token="event:0",
            resource_tokens=("slot-storage:0", "event:0"),
            resource_owner_tokens=("slot-owner:0",),
            resource_nbytes=(1024, 0),
        )


def test_plan_lease_and_arena_configuration_cannot_be_reopened_or_deleted():
    arena = _arena()
    plan = _plan(arena)
    with pytest.raises(AttributeError, match="immutable"):
        plan._sealed = False
    with pytest.raises(AttributeError, match="immutable"):
        del plan._evidence
    with pytest.raises(AttributeError, match="immutable"):
        arena._sealed = False
    with pytest.raises(AttributeError, match="immutable"):
        arena._frozen_signature = "0" * 64

    lease = arena.acquire(plan)
    assert not isinstance(lease, NativeOpticalArenaBackpressure)
    with pytest.raises(AttributeError, match="immutable"):
        lease._sealed = False
    with pytest.raises(AttributeError, match="immutable"):
        del lease._evidence


def test_live_lease_evidence_drift_is_rejected_before_first_enqueue():
    arena = _arena()
    lease = arena.acquire(_plan(arena))
    assert not isinstance(lease, NativeOpticalArenaBackpressure)
    object.__setattr__(
        lease,
        "_evidence",
        replace(lease.evidence, ordered_stage_evidence_sha256=_SHA_D),
    )
    with pytest.raises(NativeOpticalArenaProtocolError, match="evidence changed"):
        lease.begin_enqueue()


def test_illegal_post_enqueue_transition_poisons_all_active_storage():
    arena = _arena()
    first_plan = _plan(arena)
    lease = arena.acquire(first_plan)
    assert not isinstance(lease, NativeOpticalArenaBackpressure)
    lease.begin_enqueue()
    with pytest.raises(NativeOpticalArenaPoisonedError, match="out of order"):
        lease.release_device_result()
    assert arena.state == "POISONED"
    assert lease.state == "RETAINED_POISONED"
    lifecycle = arena.lifecycle_evidence()
    assert lifecycle.transitions[-1].operation == "arena_poisoned"
    assert lifecycle.external_hold_count_at_poison == 0
    assert lifecycle.external_hold_count_after_device_sync is None
    assert lifecycle.external_hold_discharge_count == 0
    assert lifecycle.device_emergency_sync_attempt_count == 0
    assert lifecycle.poisoned_slot_count == 1
    assert lifecycle.retained_pinned_storage_count == 0
    assert lifecycle.shutdown_transition_trace_sha256 is None
    assert lifecycle.resource_release_authorized is False
    with pytest.raises(NativeOpticalArenaPoisonedError):
        arena.acquire(first_plan)


def test_normal_drain_requires_explicit_reclaim_and_close_is_idempotent():
    arena = _arena()
    lease = arena.acquire(_plan(arena))
    assert not isinstance(lease, NativeOpticalArenaBackpressure)
    arena.begin_drain()
    with pytest.raises(NativeOpticalArenaProtocolError, match="free"):
        arena.close()
    _complete_and_reclaim(lease)
    lifecycle = arena.lifecycle_evidence()
    assert lifecycle.final_drain_complete is True
    assert lifecycle.final_reclaim_complete is True
    arena.close()
    arena.close()
    assert arena.state == "CLOSED"


def test_arena_strongly_retains_every_manifest_owner():
    spec = _spec()
    manifest, owners = _manifest(spec)
    refs = tuple(weakref.ref(owner) for owner in owners.values())
    arena = NativeOpticalArena(
        spec=spec,
        resource_manifest=manifest,
        resource_owners=owners,
    )
    del owners
    gc.collect()
    assert all(ref() is not None for ref in refs)
    assert arena.resource_manifest == manifest


def test_host_payload_control_arena_remains_non_result_capable_before_b3_activation():
    spec = _spec(
        boundary="R2",
        payload="rgb",
        delivery_policy="torch_async_ordered",
        payload_contract_id="rgb_native_host_v1",
        ring_depth=2,
    )
    arena = _arena(spec)
    lease = arena.acquire(_plan(arena, frame=0))
    assert not isinstance(lease, NativeOpticalArenaBackpressure)
    assert lease.result_capable is False
    assert lease.evidence.result_capable is False


def test_release_authorization_requires_both_device_and_external_hold_proofs():
    arena = _arena()
    payload = arena.lifecycle_evidence().to_payload()
    payload.update(
        {
            "device_emergency_sync_attempt_count": 1,
            "device_emergency_sync_succeeded": True,
            "device_quiesced": True,
            "resource_release_authorized": True,
            "shutdown_disposition": "aborted_synced_release_authorized",
            "external_hold_count": 1,
            "external_holds_discharged": False,
        }
    )
    with pytest.raises(ValueError, match="replayed transition trace|both safety proofs"):
        C4ArenaLifecycleEvidence.from_payload(payload)


def test_lifecycle_rejects_invented_operation_even_with_recomputed_public_hash():
    payload = _arena().lifecycle_evidence().to_payload()
    payload["transitions"][0]["operation"] = "invented_transition"
    _rehash_transition_trace(payload)
    with pytest.raises(ValueError, match="operation is invalid"):
        C4ArenaLifecycleEvidence.from_payload(payload)


def test_lifecycle_rejects_unknown_or_one_way_shutdown_claims():
    payload = _arena().lifecycle_evidence().to_payload()
    payload["shutdown_disposition"] = "made_up_disposition"
    with pytest.raises(ValueError, match="shutdown disposition is invalid"):
        C4ArenaLifecycleEvidence.from_payload(payload)

    payload = _arena().lifecycle_evidence().to_payload()
    payload["arena_state"] = "SYNCED_WAITING_EXTERNAL_HOLDS"
    payload["shutdown_disposition"] = "synced_waiting_external_holds"
    payload["shutdown_transition_trace_sha256"] = _SHA_A
    with pytest.raises(ValueError, match="replayed transition trace"):
        C4ArenaLifecycleEvidence.from_payload(payload)

    payload["arena_state"] = "ABORTED_SYNCED"
    payload["shutdown_disposition"] = "aborted_synced_release_authorized"
    payload["resource_release_authorized"] = False
    with pytest.raises(ValueError, match="replayed transition trace|release authorization"):
        C4ArenaLifecycleEvidence.from_payload(payload)


def test_lifecycle_replay_rejects_fabricated_summary_counters():
    arena = _arena()
    _complete_and_reclaim(arena.acquire(_plan(arena)))
    payload = arena.lifecycle_evidence().to_payload()
    payload["terminal_event_record_count"] = 77
    payload["slot_reclaim_count"] = 99
    with pytest.raises(ValueError, match="replayed transition trace"):
        C4ArenaLifecycleEvidence.from_payload(payload)

    payload = arena.lifecycle_evidence().to_payload()
    payload["transitions"] = payload["transitions"][:1]
    _rehash_transition_trace(payload)
    with pytest.raises(ValueError, match="replayed transition trace|slot summaries"):
        C4ArenaLifecycleEvidence.from_payload(payload)
