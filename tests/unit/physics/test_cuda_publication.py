from __future__ import annotations

import gc
import weakref
from contextlib import nullcontext

import pytest

from physics.cuda_publication import (
    CudaConsumerResourceGuardian,
    CudaDeviceConsumerBinding,
    CudaPublicationCompletionRequiredError,
    CudaPublicationPoisonedError,
    CudaPublicationQuarantinedError,
    CudaPublicationQuarantineRegistry,
    CudaPublicationStateError,
)
from physics.publish import ConsumerState, GpuPublishedFrame, PublishedRing


class _Owner:
    def __init__(self, *, handle: int = 123, device: int = 0) -> None:
        self.cuda_stream = handle
        self.device = device


class _Resource:
    pass


class _Event:
    _next_handle = 1

    def __init__(self) -> None:
        self.cuda_event = type(self)._next_handle
        type(self)._next_handle += 1


class _ExternalStream:
    def __init__(self, handle: int) -> None:
        self.handle = handle


class _Runtime:
    def __init__(self) -> None:
        self.waited: list[tuple[object, object]] = []
        self.recorded: list[tuple[object, object]] = []
        self.producer_waited: list[object] = []
        self.scope_entries = 0
        self.emergency_sync_count = 0
        self.fail_create_stream = False
        self.fail_wait = False
        self.fail_allocate = False
        self.fail_record = False
        self.fail_producer_wait = False
        self.fail_emergency_sync = False

    def device_scope(self):
        runtime = self

        class _Scope:
            def __enter__(self):
                runtime.scope_entries += 1

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Scope()

    def create_external_stream(self, cuda_stream_handle: int):
        if self.fail_create_stream:
            raise RuntimeError("create stream failed")
        return _ExternalStream(cuda_stream_handle)

    def wait_event(self, stream: object, event: object) -> None:
        if self.fail_wait:
            raise RuntimeError("wait failed")
        self.waited.append((stream, event))

    def allocate_event(self):
        if self.fail_allocate:
            raise RuntimeError("allocate failed")
        return _Event()

    def record_event(self, stream: object, event: object) -> None:
        if self.fail_record:
            raise RuntimeError("record failed")
        self.recorded.append((stream, event))

    def producer_wait_event(self, event: object) -> None:
        if self.fail_producer_wait:
            raise RuntimeError("producer wait failed")
        self.producer_waited.append(event)

    def emergency_synchronize(self) -> None:
        self.emergency_sync_count += 1
        if self.fail_emergency_sync:
            raise RuntimeError("emergency sync failed")

    @staticmethod
    def event_native_handle(event: object) -> int:
        return event.cuda_event


def _consumer() -> ConsumerState:
    return ConsumerState(
        consumer_id="camera",
        consumer_kind="render_backed_sensing",
        qos_mode="lossless",
        access_mode="borrow",
        consumer_location="device",
    )


def _frame(ring: PublishedRing, frame_id: int, event: object) -> GpuPublishedFrame:
    slot_id = frame_id % ring.ring_size
    meta = ring.mark_ready(
        slot_id=slot_id,
        frame_id=frame_id,
        step_index=frame_id,
        sim_time=float(frame_id),
        publish_event=event,
    )
    slot = ring.slot_buffers[slot_id]
    return GpuPublishedFrame(
        slot_id=slot_id,
        frame_id=frame_id,
        sim_time=float(frame_id),
        step_index=frame_id,
        env_mask_wp=None,
        q_wp=slot,
        qdot_wp=slot,
        x_world_R_wp=slot,
        x_world_r_wp=slot,
        v_bodies_wp=slot,
        contact_count_wp=None,
        contact_cache_ref=None,
        telemetry_ref=None,
        ready_event=event,
        slot_meta=meta,
    )


def _binding_fixture(*, frame_count: int = 3):
    consumer = _consumer()
    ring = PublishedRing(slot_buffers=[_Resource() for _ in range(max(3, frame_count + 1))])
    ring.register_consumer(consumer)
    frames = {frame_id: _frame(ring, frame_id, _Event()) for frame_id in range(frame_count)}
    owner = _Owner()
    resource = _Resource()
    guardian = CudaConsumerResourceGuardian.seal(
        device_ordinal=0,
        cuda_stream_owner=owner,
        resources=(resource,),
    )
    runtime = _Runtime()
    registry = CudaPublicationQuarantineRegistry()

    def _resolve(frame_id):
        resolved = max(frames) if frame_id is None else frame_id
        return frames[resolved]

    def _binding_state(consumer_state, binding_id, state):
        if consumer_state.cuda_binding_id is None:
            ring.bind_cuda_device_consumer(consumer_state, binding_id, state)
        else:
            ring.update_cuda_device_binding_state(consumer_state, binding_id, state)

    engine_owner = _Resource()
    binding = CudaDeviceConsumerBinding(
        consumer=consumer,
        device_ordinal=0,
        cuda_stream_handle=owner.cuda_stream,
        guardian=guardian,
        runtime=runtime,
        frame_resolver=_resolve,
        borrow_commit=ring.mark_cuda_device_consumer_borrowed,
        completion_commit=ring.mark_cuda_device_consumer_complete_exact,
        poison_commit=ring.poison_cuda_device_consumer,
        clear_done_event=ring.clear_cuda_device_done_event,
        binding_state_commit=_binding_state,
        engine_owner=engine_owner,
        physics_run_id="unit-run",
        quarantine_registry=registry,
    )
    return {
        "binding": binding,
        "consumer": consumer,
        "engine_owner": engine_owner,
        "frames": frames,
        "guardian": guardian,
        "owner": owner,
        "registry": registry,
        "resource": resource,
        "ring": ring,
        "runtime": runtime,
    }


def test_guardian_seal_validates_identity_uniqueness_and_single_use():
    owner = _Owner()
    resource = _Resource()

    with pytest.raises(ValueError, match="at least one"):
        CudaConsumerResourceGuardian.seal(
            device_ordinal=0,
            cuda_stream_owner=owner,
            resources=(),
        )
    with pytest.raises(ValueError, match="identity-unique"):
        CudaConsumerResourceGuardian.seal(
            device_ordinal=0,
            cuda_stream_owner=owner,
            resources=(resource, resource),
        )
    with pytest.raises(ValueError, match="does not match"):
        CudaConsumerResourceGuardian.seal(
            device_ordinal=1,
            cuda_stream_owner=owner,
            resources=(resource,),
        )

    fixture = _binding_fixture()
    guardian = fixture["guardian"]
    covered_resource = fixture["resource"]
    assert guardian.state == "bound"
    with pytest.raises(CudaPublicationStateError, match="cannot dispose"):
        guardian.assert_resource_disposal_allowed(covered_resource)
    with pytest.raises(CudaPublicationStateError, match="bound"):
        guardian.close()
    fixture["binding"].close()
    assert guardian.state == "released_to_session"
    guardian.assert_resource_disposal_allowed(covered_resource)
    guardian.close()
    assert guardian.state == "closed"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("latest_seen_frame_id", 0),
        ("device_completed_frame_id", 0),
        ("device_done_event", _Event()),
    ),
)
def test_ring_rejects_binding_any_legacy_device_state(field, value):
    consumer = ConsumerState(
        consumer_id="legacy",
        consumer_kind="render_backed_sensing",
        qos_mode="lossless",
        access_mode="borrow",
        consumer_location="device",
    )
    setattr(consumer, field, value)
    ring = PublishedRing(slot_buffers=[object()], consumers=[consumer])

    with pytest.raises(RuntimeError, match="fresh consumer"):
        ring.bind_cuda_device_consumer(consumer, "binding", "open")

    assert consumer.cuda_binding_id is None
    assert consumer.cuda_binding_state is None


def test_host_consumer_history_is_not_legacy_device_state():
    consumer = ConsumerState(
        consumer_id="host",
        consumer_kind="render_backed_sensing",
        qos_mode="lossless",
        access_mode="borrow",
        consumer_location="host",
        latest_seen_frame_id=3,
        acked_frame_id=3,
    )
    assert consumer.has_legacy_device_api_state is False


def test_binding_borrow_complete_is_fifo_exact_and_evidence_is_observed():
    fixture = _binding_fixture()
    binding = fixture["binding"]
    consumer = fixture["consumer"]
    runtime = fixture["runtime"]

    lease0 = binding.borrow(0)
    lease1 = binding.borrow(1)
    with pytest.raises(CudaPublicationStateError, match="out-of-order"):
        lease1.complete(last_reader_operation="scene_update")
    assert consumer.device_completed_frame_id == -1

    evidence0 = lease0.complete(last_reader_operation="scene_update")
    evidence1 = binding.complete(1, last_reader_operation="scene_update")
    assert consumer.device_completed_frame_id == 1
    assert len(runtime.waited) == 2
    assert len(runtime.recorded) == 2
    assert evidence0.publish_wait_enqueued is True
    assert evidence0.borrow_ordinal == 1
    assert evidence0.total_binding_borrow_count == 2
    assert evidence0.guardian_state == "bound"
    assert evidence0.physics_run_id == "unit-run"
    assert evidence0.physics_step_count == 1
    assert evidence0.publish_event_recorded is True
    assert evidence0.producer_publish_completion_recorded is True
    assert evidence0.binding_state == "open"
    assert evidence0.consumer_completion_recorded is True
    assert evidence0.slot_logically_reclaimable is True
    assert evidence0.done_event_retained is True
    assert evidence1.last_reader_operation == "scene_update"

    with pytest.raises(CudaPublicationStateError, match="duplicate"):
        lease0.complete(last_reader_operation="scene_update")
    with pytest.raises(CudaPublicationStateError, match="no active"):
        binding.complete(2, last_reader_operation="scene_update")


def test_context_exit_without_explicit_complete_is_safe_but_rejected():
    fixture = _binding_fixture()
    binding = fixture["binding"]

    with pytest.raises(CudaPublicationCompletionRequiredError, match="explicit complete"):
        with binding.borrow(0) as lease:
            lease.mark_last_reader_enqueued("scene_update")

    assert fixture["consumer"].device_completed_frame_id == 0
    assert binding.evidence(0).last_reader_operation == "scene_update"


def test_ordinary_exception_records_real_done_event_then_preserves_exception():
    fixture = _binding_fixture()
    binding = fixture["binding"]

    with pytest.raises(ValueError, match="body failed"):
        with binding.borrow(0) as lease:
            lease.mark_last_reader_enqueued("scene_update")
            raise ValueError("body failed")

    assert fixture["consumer"].device_completed_frame_id == 0
    assert len(fixture["runtime"].recorded) == 1
    assert binding.state == "open"


def test_exceptional_out_of_order_completion_poisons_all_active_slots():
    fixture = _binding_fixture()
    binding = fixture["binding"]
    lease0 = binding.borrow(0)

    with pytest.raises(CudaPublicationPoisonedError, match="non-head"):
        with binding.borrow(1):
            raise ValueError("later lease failed")

    assert binding.state == "poisoned"
    assert fixture["ring"].slot_meta[0].poisoned is True
    assert fixture["ring"].slot_meta[1].poisoned is True
    assert lease0.active is True


@pytest.mark.parametrize("failure", ["fail_wait", "fail_allocate", "fail_record"])
def test_cuda_wait_or_event_failure_poisons_without_logical_completion(failure):
    fixture = _binding_fixture()
    binding = fixture["binding"]
    runtime = fixture["runtime"]
    setattr(runtime, failure, True)

    if failure == "fail_wait":

        def action():
            binding.borrow(0)
    else:
        lease = binding.borrow(0)

        def action():
            lease.complete(last_reader_operation="scene_update")

    with pytest.raises(CudaPublicationPoisonedError):
        action()
    assert binding.state == "poisoned"
    assert fixture["consumer"].device_completed_frame_id == -1
    assert fixture["consumer"].device_poisoned is True


def test_record_failure_poisons_every_active_and_retained_slot():
    fixture = _binding_fixture()
    binding = fixture["binding"]
    runtime = fixture["runtime"]
    binding.borrow(0).complete(last_reader_operation="scene_update")
    lease1 = binding.borrow(1)
    binding.borrow(2)
    runtime.fail_record = True

    with pytest.raises(CudaPublicationPoisonedError):
        lease1.complete(last_reader_operation="scene_update")

    assert all(fixture["ring"].slot_meta[index].poisoned for index in (0, 1, 2))


def test_poison_abort_sync_success_precedes_guardian_release():
    fixture = _binding_fixture()
    binding = fixture["binding"]
    guardian = fixture["guardian"]
    runtime = fixture["runtime"]
    resource = fixture["resource"]
    resource_ref = weakref.ref(resource)
    del fixture["resource"]
    del resource

    runtime.fail_record = True
    lease = binding.borrow(0)
    with pytest.raises(CudaPublicationPoisonedError):
        lease.complete(last_reader_operation="scene_update")
    gc.collect()
    assert resource_ref() is not None
    assert guardian.state == "bound"

    disposition = binding.abort_poisoned()
    assert disposition.disposition == "emergency_sync_succeeded"
    assert runtime.emergency_sync_count == 1
    assert guardian.state == "released_after_emergency_sync"
    assert fixture["consumer"].device_done_event is None
    evidence = binding.evidence(0)
    assert evidence.emergency_sync_attempt_count == 1
    assert evidence.emergency_sync_success_count == 1
    assert evidence.emergency_sync_failure_category is None
    assert evidence.teardown_disposition == "emergency_sync_succeeded"
    assert binding.abort_poisoned() is disposition
    guardian.close()
    gc.collect()
    assert resource_ref() is None


def test_failed_emergency_sync_atomically_quarantines_cross_owner_graph():
    fixture = _binding_fixture()
    binding = fixture["binding"]
    guardian = fixture["guardian"]
    registry = fixture["registry"]
    runtime = fixture["runtime"]
    resource_ref = weakref.ref(fixture["resource"])
    owner_ref = weakref.ref(fixture["owner"])
    engine_ref = weakref.ref(fixture["engine_owner"])

    runtime.fail_record = True
    lease = binding.borrow(0)
    with pytest.raises(CudaPublicationPoisonedError):
        lease.complete(last_reader_operation="scene_update")
    runtime.fail_emergency_sync = True
    disposition = binding.abort_poisoned()
    assert disposition.disposition == "quarantined_until_process_exit"
    assert disposition.quarantine_accepted is True
    evidence = binding.evidence(0)
    assert evidence.emergency_sync_attempt_count == 1
    assert evidence.emergency_sync_success_count == 0
    assert evidence.emergency_sync_failure_category == "RuntimeError"
    assert disposition.process_terminal is True
    assert registry.contains_binding(binding.binding_id)
    assert guardian.state == "quarantined"
    evidence = binding.evidence(0)
    assert evidence.guardian_state == "quarantined"
    assert evidence.binding_state == "quarantined"
    assert evidence.quarantine_accepted is True
    assert evidence.process_terminal is True
    with pytest.raises(CudaPublicationQuarantinedError):
        guardian.close()

    del fixture["resource"]
    del fixture["owner"]
    del fixture["engine_owner"]
    gc.collect()
    assert resource_ref() is not None
    assert owner_ref() is not None
    assert engine_ref() is not None
    with pytest.raises(CudaPublicationQuarantinedError, match="terminate"):
        registry.require_device_available(0)


def test_binding_creation_failure_rolls_guardian_back_before_any_wait():
    owner = _Owner()
    guardian = CudaConsumerResourceGuardian.seal(
        device_ordinal=0,
        cuda_stream_owner=owner,
        resources=(_Resource(),),
    )
    runtime = _Runtime()
    runtime.fail_create_stream = True

    with pytest.raises(RuntimeError, match="create stream"):
        CudaDeviceConsumerBinding(
            consumer=_consumer(),
            device_ordinal=0,
            cuda_stream_handle=owner.cuda_stream,
            guardian=guardian,
            runtime=runtime,
            frame_resolver=lambda _: pytest.fail("frame resolver must not run"),
            borrow_commit=lambda *_: None,
            completion_commit=lambda *_: None,
            poison_commit=lambda *_: None,
            clear_done_event=lambda *_: None,
            binding_state_commit=lambda *_: None,
            engine_owner=_Resource(),
            physics_run_id="unit-partial-bind",
            quarantine_registry=CudaPublicationQuarantineRegistry(),
        )

    assert guardian.state == "sealed_unbound"
    guardian.close()


def test_active_close_rejected_then_final_drain_releases_guardian_hold():
    fixture = _binding_fixture()
    binding = fixture["binding"]
    guardian = fixture["guardian"]
    lease = binding.borrow(0)

    with pytest.raises(CudaPublicationStateError, match="active"):
        binding.close()
    lease.complete(last_reader_operation="scene_update")
    binding.close()

    assert binding.state == "closed"
    assert guardian.state == "released_to_session"
    assert len(fixture["runtime"].producer_waited) == 1
    assert fixture["consumer"].cuda_binding_state == "closed"
    evidence = binding.evidence(0)
    assert evidence.final_drain_wait_enqueued is True
    assert evidence.guardian_state == "released_to_session"


def test_done_event_retained_until_covered_slot_leaves_ring():
    fixture = _binding_fixture()
    binding = fixture["binding"]
    lease = binding.borrow(0)
    lease.complete(last_reader_operation="scene_update")
    assert binding.retained_done_event_count == 1

    binding.prepare_slot_reuse(0)
    assert binding.evidence(0).producer_reuse_wait_enqueued is True
    binding.release_reuse_safe_records((1, 2))
    assert binding.retained_done_event_count == 0
    assert binding.evidence(0).done_event_retained is False


def test_guardian_identity_mutation_poisons_on_next_completion():
    fixture = _binding_fixture()
    binding = fixture["binding"]
    owner = fixture["owner"]
    lease = binding.borrow(0)
    owner.cuda_stream += 1

    with pytest.raises(CudaPublicationPoisonedError, match="identity"):
        lease.complete(last_reader_operation="scene_update")
    assert binding.state == "poisoned"


def test_guardian_identity_mutation_poisons_on_next_borrow():
    fixture = _binding_fixture()
    binding = fixture["binding"]
    fixture["owner"].cuda_stream += 1

    with pytest.raises(CudaPublicationPoisonedError, match="identity"):
        binding.borrow(0)
    assert binding.state == "poisoned"
    assert fixture["consumer"].device_completed_frame_id == -1


def test_reentrant_abort_is_rejected_and_outer_abort_finishes_once():
    fixture = _binding_fixture()
    binding = fixture["binding"]
    runtime = fixture["runtime"]
    runtime.fail_record = True
    lease = binding.borrow(0)
    with pytest.raises(CudaPublicationPoisonedError):
        lease.complete(last_reader_operation="scene_update")

    def _reentrant_sync():
        with pytest.raises(CudaPublicationStateError, match="re-entrant"):
            binding.abort_poisoned()

    runtime.emergency_synchronize = _reentrant_sync
    disposition = binding.abort_poisoned()
    assert disposition.disposition == "emergency_sync_succeeded"


def test_finalizer_transfer_does_not_synchronize_and_is_idempotent():
    fixture = _binding_fixture()
    binding = fixture["binding"]
    runtime = fixture["runtime"]
    registry = fixture["registry"]

    with pytest.warns(ResourceWarning):
        binding.quarantine_from_finalizer("test_finalizer")
    binding.quarantine_from_finalizer("test_finalizer")
    assert runtime.emergency_sync_count == 0
    assert registry.capsule_count == 1
    assert binding.state == "quarantined"


def test_device_scope_is_used_for_every_runtime_cuda_operation():
    fixture = _binding_fixture()
    binding = fixture["binding"]
    runtime = fixture["runtime"]
    initial = runtime.scope_entries
    lease = binding.borrow(0)
    lease.complete(last_reader_operation="scene_update")
    binding.prepare_slot_reuse(0)
    binding.close()
    assert runtime.scope_entries - initial == 4


def test_fake_scope_does_not_hide_host_synchronization_calls():
    # The state machine has no host-query/synchronize callback.  Emergency sync
    # is separately named and remains zero throughout every successful hot path.
    fixture = _binding_fixture()
    binding = fixture["binding"]
    with nullcontext():
        binding.borrow(0).complete(last_reader_operation="scene_update")
    assert fixture["runtime"].emergency_sync_count == 0
