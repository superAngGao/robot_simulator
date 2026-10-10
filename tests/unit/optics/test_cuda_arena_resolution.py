"""Real arena/guard coordination with a CPU stand-in for the device event."""

from types import SimpleNamespace as NS

import pytest

from optical_worker.guarded_lifecycle import NativeGuardedSetupLifecycle
from optical_worker.lifecycle import DeviceScopeDescriptor
from optics.cuda_arena import CudaActiveArenaSample
from optics.native_arena import NativeOpticalArenaProtocolError
from tests.support.guarded_peer import Peer
from tests.unit.optics.test_native_optical_arena import _arena, _plan


@pytest.fixture
def completed_sample():
    arena = _arena()
    plan = _plan(arena)
    lease = arena.acquire(plan)
    lease.begin_enqueue()
    lease.record_completion()
    calls = []
    event = NS(synchronize=lambda: calls.append("wait"))
    context = NS(device=NS(index=1), stream=object())
    transaction = NS(_context=context, _stages=(), _completion_event=event, _diagnostic_events=())
    executable = NS(transaction_template=NS(stages=(), completion_event=event), activation_capability=object())
    sample = CudaActiveArenaSample(
        target=NS(evidence=NS(sha256="a" * 64)), provenance=object(), transaction=transaction,
        executable=executable, arena_acquire_authority=object())
    # Only the device binding is synthetic. Exercise the production lease,
    # immutable active-sample checks, guard client and supervisor state machine.
    object.__setattr__(sample, "_control_lease", lease)
    object.__setattr__(sample, "_completion", NS(event=event))
    object.__setattr__(sample, "_state", "COMPLETION_RECORDED")
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
    x.transaction._context.device.index = 0
    with pytest.raises(ValueError, match="target scope"):
        x.sample.resolve(_guarded_controller=controller)
    assert not x.calls
    assert x.arena.lifecycle_evidence().completion_resolution_count == 0
    assert x.lease.state == "RETAINED_POISONED"
