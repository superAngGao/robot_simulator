"""Real arena/guard coordination with a CPU stand-in for the device event."""

from types import SimpleNamespace as NS
from contextlib import nullcontext
from dataclasses import dataclass

import pytest

from optical_worker.guarded_lifecycle import NativeGuardedSetupLifecycle
from optical_worker.lifecycle import DeviceScopeDescriptor
from optics.cuda_arena import CudaPreparedArenaSample, _CudaExecutableTarget
from optics import cuda_timeline as timeline
from optics.native_arena import NativeOpticalArenaProtocolError
from tests.support.guarded_peer import Peer
from tests.unit.optics.test_native_optical_arena import _arena, _plan


@pytest.fixture
def completed_sample(monkeypatch):
    @dataclass(frozen=True)
    class Device:
        index: int
        type: str = "cuda"

    device = Device(1)
    stream = NS(device=device, cuda_stream=17)
    calls = []

    class Event:
        def __init__(self):
            self.device = device

        def record(self, selected):
            assert selected is stream
            calls.append("record")

        def synchronize(self):
            calls.append("wait")

    monkeypatch.setattr(timeline, "torch", NS(device=lambda value: value,
        cuda=NS(is_available=lambda: True, Event=Event,
                device=lambda value: nullcontext(), stream=lambda value: nullcontext())))
    context = timeline.CudaOpticalExecutionContext(device=device, stream=stream)

    class Stage:
        cuda_transaction_stage_name = "cpu_resolution_stage"

        def _validate_cuda_transaction_stage(self, actual):
            assert actual is context

        def _assert_cuda_transaction_stage_stable(self, actual):
            assert actual is context

        def _cuda_transaction_stage_evidence(self):
            return "cpu_resolution_evidence"

        def enqueue(self, transaction):
            with transaction.enqueue_stage(self):
                transaction.before_native_launch()

    stage, event = Stage(), Event()
    capability = context._new_controlled_transaction_capability()
    template = context.prepare_transaction_template(stages=(stage,), completion_event=event,
                                                    _activation_capability=capability)
    template._bind_arena_plan_identity(capability, slot_id=0, terminal_event_token="event:0",
        transform_frame_index=0, plan_binding_sha256="d" * 64, ordered_runtime_stage_evidence_sha256="e" * 64)
    transaction = template._activate_with_capability(capability)
    arena = _arena()
    plan = _plan(arena)
    target = NS(evidence=NS(sha256="f" * 64, slot_id=0))
    provenance = NS(plan_binding_sha256="d" * 64, ordered_runtime_stage_evidence_sha256="e" * 64,
                   c3_one_shot_binding_sha256="a" * 64, ordered_stage_evidence_sha256="b" * 64,
                   dataflow_binding_sha256="c" * 64)
    executable = _CudaExecutableTarget(target=target, transaction_template=template,
                                       activation_capability=capability, render_stage=stage)
    acquire_authority = object()
    prepared = CudaPreparedArenaSample(control_plan=plan, target=target, provenance=provenance,
        transaction=transaction, executable=executable, arena_acquire_authority=acquire_authority)
    sample = prepared._active
    lease = arena.acquire(plan)
    authority = transaction._mint_arena_execution_authority(prepared._execution_capability, lease,
                                                           active_owner=sample)
    sample._activate(lease, execution_authority=authority, arena_acquire_authority=acquire_authority)
    sample.enqueue()
    sample.finish()
    assert calls == ["record"]
    calls.clear()
    # Native operations and C3 inputs are CPU fixtures. Context, controlled
    # template, minted authority, transaction, lease and guard are production.
    return NS(arena=arena, lease=lease, sample=sample, calls=calls, event=event, transaction=transaction)


@pytest.fixture
def guarded_sample(completed_sample):
    x = completed_sample
    peer = Peer()
    controller = NativeGuardedSetupLifecycle(lane_id="cuda", channel=peer.channel, configuration=peer.config)
    try:
        with controller.segment("setup"):
            descriptor = DeviceScopeDescriptor.exact_target(
                operation_id="cpu_event_fixture", leaf_kind="event_record", device_ordinal=1,
                scope_source="cpu_test", runtime_identity="cpu_test_event")
            controller.enter_device_leaf(controller.begin_device_leaf(descriptor))
            for owner in (x.sample, x.transaction, x.transaction._context.stream, x.event):
                controller.adopt_host_owner(owner, role="resolution_owner", operation_id="cpu_event_fixture")
        controller.seal_guardian()
        yield x, controller
    finally:
        if controller.termination_evidence is None:
            controller.terminate(fault="setup_failure", detection_phase="cpu_test_exit",
                                 emergency_target_synchronize=lambda: None)
        peer.close()


def test_standalone_resolution_preserves_existing_lease_behavior(completed_sample):
    x = completed_sample
    x.sample.resolve()
    assert x.calls == ["wait"]
    assert x.sample.state == x.lease.state == "PAYLOAD_READY"
    assert x.arena.lifecycle_evidence().completion_resolution_count == 1
    x.sample.release_device_result()
    assert x.lease.state == "RECLAIMABLE"
    with pytest.raises(NativeOpticalArenaProtocolError):
        x.sample.resolve()
    assert x.calls == ["wait"]


def test_guarded_resolution_commits_only_after_disarmed(guarded_sample, monkeypatch):
    x, controller = guarded_sample
    original = controller._guard_client.complete

    def complete(ticket):
        assert ticket.role == "resolve"
        assert x.calls == ["wait"]
        assert x.sample.state == x.lease.state == "COMPLETION_RECORDED"
        evidence = x.arena.lifecycle_evidence()
        assert evidence.completion_resolution_count == 0
        assert evidence.slots[0].transaction_hold_count == 1
        original(ticket)

    monkeypatch.setattr(controller._guard_client, "complete", complete)
    x.sample.resolve(_guarded_controller=controller)
    assert x.sample.state == x.lease.state == "PAYLOAD_READY"
    assert x.arena.lifecycle_evidence().completion_resolution_count == 1
    assert controller._guard_events[-1]["operation"] == "normal_disarmed"
    assert controller._guard_events[-1]["guard"]["role"] == "resolve"


@pytest.mark.parametrize("failure_at", ["wait", "disarmed", "binding_drift"])
def test_failed_resolution_retains_unresolved_generation(guarded_sample, monkeypatch, failure_at):
    x, controller = guarded_sample
    failure = RuntimeError("injected resolution failure")

    def wait():
        x.calls.append("wait")
        if failure_at == "wait":
            raise failure
        if failure_at == "binding_drift":
            x.transaction._completion_event = object()

    def missing_disarmed(ticket):
        assert x.calls == ["wait"]
        raise failure

    x.event.synchronize = wait
    if failure_at == "disarmed":
        monkeypatch.setattr(controller._guard_client, "complete", missing_disarmed)
    with pytest.raises((RuntimeError, NativeOpticalArenaProtocolError)):
        x.sample.resolve(_guarded_controller=controller)
    evidence = x.arena.lifecycle_evidence()
    assert x.calls == ["wait"]
    assert x.sample.state == "POISONED"
    assert x.lease.state == "RETAINED_POISONED"
    assert evidence.completion_resolution_count == evidence.slot_reclaim_count == 0
    assert evidence.slots[0].transaction_hold_count == 1
    with pytest.raises(NativeOpticalArenaProtocolError):
        x.sample.release_device_result()
    with pytest.raises(NativeOpticalArenaProtocolError):
        x.sample.reclaim()
    with pytest.raises(NativeOpticalArenaProtocolError):
        x.sample.resolve()
    assert x.calls == ["wait"]
    if failure_at != "binding_drift":
        assert controller.guardian.state == "TRANSFERRED_TO_QUARANTINE"
        assert controller.termination_evidence.lifecycle.to_payload()["summary"]["emergency_sync_attempt_count"] == 0


def test_guarded_resolution_rejects_wrong_target_before_wait(guarded_sample):
    x, controller = guarded_sample
    x.transaction._context.device = type(x.transaction._context.device)(0)
    with pytest.raises(timeline.CudaOpticalTimelineError, match="device identity"):
        x.sample.resolve(_guarded_controller=controller)
    assert not x.calls
    assert x.arena.lifecycle_evidence().completion_resolution_count == 0
    assert x.lease.state == "RETAINED_POISONED"


@pytest.mark.parametrize("boundary", ["before_wait", "during_wait", "during_ack"])
@pytest.mark.parametrize("mutation", ["stream", "coherent_stream", "device", "handle", "event_device"])
def test_resolution_binding_drift_never_releases_holds(guarded_sample, monkeypatch, boundary, mutation):
    x, controller = guarded_sample
    context = x.transaction._context
    replacement = NS(device=context.device, cuda_stream=context.stream_handle + 1)

    def mutate():
        if mutation in {"stream", "coherent_stream"}:
            context.stream = replacement
            if mutation == "coherent_stream":
                context.stream_handle = replacement.cuda_stream
        elif mutation == "device":
            context.device = type(context.device)(0)
        elif mutation == "handle":
            context.stream.cuda_stream += 1
        else:
            x.event.device = type(context.device)(0)

    if boundary == "before_wait":
        mutate()
    elif boundary == "during_wait":
        def wait():
            x.calls.append("wait")
            mutate()
        x.event.synchronize = wait
    else:
        original = controller._guard_client.complete
        def complete(ticket):
            original(ticket)
            mutate()
        monkeypatch.setattr(controller._guard_client, "complete", complete)

    with pytest.raises((timeline.CudaOpticalTimelineError, NativeOpticalArenaProtocolError, ValueError)):
        x.sample.resolve(_guarded_controller=controller)
    evidence = x.arena.lifecycle_evidence()
    assert x.calls == ([] if boundary == "before_wait" else ["wait"])
    assert x.sample.state == "POISONED" and x.lease.state == "RETAINED_POISONED"
    assert evidence.completion_resolution_count == evidence.slot_reclaim_count == 0
    assert evidence.slots[0].transaction_hold_count == evidence.slots[0].device_result_hold_count == 1
