"""C2d1 CPU-only real client/core integration; GPU adoption remains closed."""

import copy
import socket
import threading
import uuid
from dataclasses import replace

import pytest

from optical_worker.guarded_evidence import (
    LIFECYCLE_VERSION,
    GuardedLifecycleEvidence,
    GuardedTerminationEvidence,
    replay_lifecycle,
)
from optical_worker.guarded_lifecycle import _TERMINAL_CUSTODY, NativeGuardedSetupLifecycle
from optical_worker.lifecycle import DeviceScopeDescriptor, WorkerSetupProtocolError
from optical_worker.native_setup import _abort
from optical_worker.wait_protocol import GuardConfiguration, GuardProtocolError, GuardWaitClient
from optical_worker.wait_supervision import GuardSupervisorCore, GuardSupervisorTransport


class Peer:
    def __init__(self):
        self.config = GuardConfiguration(uuid.uuid4().hex, 1, 1000, 5000)
        self.channel, self.remote = socket.socketpair()
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.errors = []
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        assert self.ready.wait(2)

    def run(self):
        self.core = GuardSupervisorCore(self.config)
        transport = GuardSupervisorTransport(self.remote, self.core)
        self.ready.set()
        try:
            while not self.stop.wait(0.001):
                import select

                self.before_service()
                read, write, _ = select.select(
                    [self.remote], [self.remote] if transport.wants_write else [], [], 0.01
                )
                transport.service(readable=bool(read), writable=bool(write))
                self.core.tick()
        except BaseException as error:
            self.errors.append(error)

    def before_service(self):
        pass

    def close(self):
        self.stop.set()
        self.thread.join(2)
        assert not self.thread.is_alive()
        self.channel.close()
        self.remote.close()
        assert not self.errors, self.errors


@pytest.fixture
def peer():
    peer = Peer()
    try:
        yield peer
    finally:
        peer.close()


def lifecycle(peer):
    return NativeGuardedSetupLifecycle(lane_id="cuda", channel=peer.channel, configuration=peer.config)


def descriptor(ordinal=1):
    return DeviceScopeDescriptor.exact_target(
        operation_id="fixed_cpu_leaf",
        leaf_kind="allocation",
        device_ordinal=ordinal,
        scope_source="cpu_test",
        runtime_identity="cpu_test_runtime",
    )


def terminate(controller, sync=None):
    return controller.terminate(
        fault="setup_failure", detection_phase="fixed_test", emergency_target_synchronize=sync
    )


def test_cpu_cancel_is_guarded_without_sync(peer):
    controller = lifecycle(peer)
    observed = []

    class Owner:
        def __del__(self):
            observed.append(peer.core._state)

    controller.adopt_host_owner(Owner(), role="cpu_owner", operation_id="cpu_setup")
    evidence = terminate(controller, lambda: pytest.fail("CPU cancellation synchronized"))
    payload = GuardedTerminationEvidence.from_payload(evidence.to_payload()).to_payload()["lifecycle"]
    assert len(observed) == 1
    # send() can wake the client before the peer commits ack_written; the
    # operation deadline and exact terminal identity must already exist.
    assert observed[0].state in {"ARM_ACK_PENDING", "ARMED_TERMINAL"}
    assert observed[0].active == (0, "emergency")
    assert observed[0].operation_deadline_ns is not None
    assert payload["summary"]["disposition"] == "cancelled_before_submission_exit"
    assert payload["summary"]["release_status"] == "completed"
    assert payload["summary"]["emergency_sync_attempt_count"] == 0
    assert payload["summary"]["normal_complete_count"] == 0
    assert controller in _TERMINAL_CUSTODY
    with pytest.raises(WorkerSetupProtocolError, match="single-use"):
        terminate(controller)
    assert controller.evidence().to_payload() == payload


def test_normal_setup_and_terminal_sync_are_separate(peer):
    controller = lifecycle(peer)
    owner = object()
    with controller.segment():
        permit = controller.begin_device_leaf(descriptor())
        controller.enter_device_leaf(permit)
        controller.adopt_returned_device_owner(
            owner, actual_device_ordinal=1, role="array", operation_id="fixed_cpu_leaf"
        )
    calls = []
    evidence = terminate(controller, lambda: calls.append(peer.core._state))
    summary = evidence.lifecycle.to_payload()["summary"]
    assert len(calls) == 1
    assert calls[0].state in {"ARM_ACK_PENDING", "ARMED_TERMINAL"}
    assert calls[0].active == (1, "emergency")
    assert calls[0].operation_deadline_ns is not None
    assert summary["setup_operation_count"] == summary["entered_leaf_count"] == 1
    assert summary["normal_complete_count"] == summary["emergency_sync_attempt_count"] == 1
    assert summary["disposition"] == "aborted_synced_exit"


@pytest.mark.parametrize("inner_abort", [False, True])
def test_failed_leaf_never_retries_and_holds_exception(peer, inner_abort):
    controller = lifecycle(peer)
    owner = object()
    original = RuntimeError("submit then raise")
    with pytest.raises(RuntimeError):
        with controller.segment():
            controller.adopt_host_owner(owner, role="source", operation_id="source_setup")
            controller.enter_device_leaf(controller.begin_device_leaf(descriptor()))
            try:
                raise original
            except RuntimeError as error:
                if inner_abort:
                    raise _abort(
                        controller, exception=error, sync=lambda: pytest.fail("retry sync")
                    ) from error
                raise
    evidence = controller.termination_evidence.lifecycle.to_payload()
    assert evidence["summary"]["entered_leaf_count"] == 1
    assert evidence["summary"]["normal_complete_count"] == 0
    assert evidence["summary"]["terminal_arm_attempt_count"] == 0
    assert evidence["summary"]["emergency_sync_attempt_count"] == 0
    assert controller.guardian.contains_exact_owner(owner)
    assert original in controller._failure_holds
    assert evidence["summary"]["custody"] == "full_quarantine"


@pytest.mark.parametrize("scope", ["foreign", "unknown"])
def test_compromised_scope_quarantines_before_leaf(peer, scope):
    controller = lifecycle(peer)
    value = (
        descriptor(0)
        if scope == "foreign"
        else DeviceScopeDescriptor.unknown(
            operation_id="unknown_call",
            leaf_kind="allocation",
            scope_source="cpu_test",
            runtime_identity="cpu_test_runtime",
        )
    )
    with pytest.raises(WorkerSetupProtocolError):
        with controller.segment():
            controller.begin_device_leaf(value)
    summary = controller.termination_evidence.lifecycle.to_payload()["summary"]
    assert summary["entered_leaf_count"] == 0
    assert summary["disposition"] == "multi_or_unknown_device_quarantine_exit"
    assert summary["emergency_sync_attempt_count"] == 0


@pytest.mark.parametrize(
    "method", ["commit_target_barrier", "commit_clean_close", "latch_measured_sample_submission"]
)
def test_admission_stays_closed(peer, method):
    controller = lifecycle(peer)
    with pytest.raises(WorkerSetupProtocolError):
        getattr(controller, method)()
    assert controller.evidence().to_payload()["summary"]["setup_operation_count"] == 0
    terminate(controller)


def test_ticket_history_requires_exact_and_just_completed_identity(peer):
    client = GuardWaitClient(peer.channel, peer.config)
    ticket = client.arm("setup")
    client.require_live_ticket(ticket, role="setup")
    with pytest.raises(GuardProtocolError):
        client.require_live_ticket(replace(ticket), role="setup")
    with pytest.raises(GuardProtocolError):
        client.require_completed_ticket(ticket)
    client.complete(ticket)
    client.require_completed_ticket(ticket)
    with pytest.raises(GuardProtocolError):
        client.require_live_ticket(ticket, role="setup")
    next_ticket = client.arm("resolve")
    with pytest.raises(GuardProtocolError):
        client.require_completed_ticket(ticket)
    client.complete(next_ticket)


def test_nested_segment_invalidates_outer_success(peer):
    controller = lifecycle(peer)
    with pytest.raises(WorkerSetupProtocolError):
        with controller.segment():
            with pytest.raises(WorkerSetupProtocolError):
                with controller.segment():
                    pytest.fail("nested body")
    summary = controller.termination_evidence.lifecycle.to_payload()["summary"]
    assert summary["normal_complete_count"] == 0
    assert summary["custody"] == "full_quarantine"


@pytest.mark.parametrize("attack", ["exit_before_enter", "double_enter", "double_exit", "reenter"])
def test_context_protocol_misuse_is_terminal_and_single_use(peer, attack):
    controller = lifecycle(peer)
    context = controller.segment()
    if attack != "exit_before_enter":
        context.__enter__()
    if attack in {"double_exit", "reenter"}:
        context.__exit__(None, None, None)
    with pytest.raises(WorkerSetupProtocolError):
        if attack in {"double_enter", "reenter"}:
            context.__enter__()
        else:
            context.__exit__(None, None, None)
    assert controller in _TERMINAL_CUSTODY
    assert controller.termination_evidence.lifecycle.to_payload()["summary"]["custody"] == "full_quarantine"
    if attack == "double_enter":
        # Drain only the Python generator; it cannot COMPLETE the failed guard.
        with pytest.raises(WorkerSetupProtocolError):
            context.__exit__(None, None, None)


def test_replay_rejects_changed_summaries_and_transitions(peer):
    payload = terminate(lifecycle(peer)).lifecycle.to_payload()
    for key, value in [
        ("normal_complete_count", 99),
        ("emergency_sync_attempt_count", 1),
        ("guardian_state", "SEALED"),
        ("custody", "full_quarantine"),
        ("guarded_segment_failed", 0),
        ("disposition", "aborted_synced_exit"),
    ]:
        forged = copy.deepcopy(payload)
        forged["summary"][key] = value
        with pytest.raises(GuardProtocolError):
            GuardedLifecycleEvidence.from_payload(forged)
    for index in range(len(payload["transition_trace"])):
        forged = copy.deepcopy(payload)
        forged["transition_trace"].insert(index, copy.deepcopy(forged["transition_trace"][index]))
        for position, event in enumerate(forged["transition_trace"]):
            event["index"] = position
        with pytest.raises(GuardProtocolError):
            replay_lifecycle(forged["configuration"], forged["lane_id"], forged["transition_trace"])


def test_handwritten_initial_prefix_has_no_completion_authority():
    config = GuardConfiguration("a" * 32, 1, 1000, 5000)
    payload = {
        "schema_version": LIFECYCLE_VERSION,
        "configuration": config.to_payload(),
        "lane_id": "warp",
        "record_kind": "prefix",
        "transition_trace": [
            {
                "index": 0,
                "epoch": 0,
                "operation": "guardian_created",
                "guard": None,
                "details": {"lane_id": "warp", "target_device_ordinal": 1},
            }
        ],
        "guardian_coverage": {
            "schema_version": "q54.p12.4.worker-guardian-coverage.v1",
            "session_identity": "a" * 32,
            "state": "CONSTRUCTING",
            "records": [],
            "per_device_resource_counts": {},
        },
        "summary": {
            "controller_state": "RUNNING",
            "guardian_state": "CONSTRUCTING",
            "guard_state": "IDLE",
            "guard_ref": None,
            "next_sequence": 0,
            "terminal_purpose": None,
            "guarded_segment_failed": False,
            "terminal_operation_failed": False,
            "possible_dirty_device_ordinals": [],
            "unknown_device_scope": False,
            "setup_operation_count": 0,
            "entered_leaf_count": 0,
            "setup_work_counts": {},
            "normal_complete_count": 0,
            "terminal_arm_attempt_count": 0,
            "emergency_sync_attempt_count": 0,
            "emergency_sync_outcome": "not_attempted",
            "custody": "none",
            "release_status": "not_authorized",
            "disposition": None,
        },
    }
    assert GuardedLifecycleEvidence.from_payload(payload).to_payload() == payload
    payload["record_kind"] = "terminal"
    with pytest.raises(GuardProtocolError):
        GuardedLifecycleEvidence.from_payload(payload)


@pytest.mark.parametrize(
    "field,value", [("role", "resolve"), ("sequence", 7), ("session", "b" * 32), ("device_ordinal", 0)]
)
def test_ticket_original_ack_fields_cannot_drift(peer, field, value):
    client = GuardWaitClient(peer.channel, peer.config)
    ticket = client.arm("setup")
    object.__setattr__(ticket, field, value)
    with pytest.raises(GuardProtocolError):
        client.require_live_ticket(ticket, role=ticket.role)


def test_foreign_thread_invalidates_outer_success(peer):
    controller = lifecycle(peer)
    errors = []

    def wrong_thread():
        try:
            controller.begin_device_leaf(descriptor())
        except WorkerSetupProtocolError as error:
            errors.append(error)

    with pytest.raises(WorkerSetupProtocolError):
        with controller.segment():
            thread = threading.Thread(target=wrong_thread)
            thread.start()
            thread.join(2)
            assert not thread.is_alive()
    assert len(errors) == 1
    payload = controller.termination_evidence.lifecycle.to_payload()
    assert payload["summary"]["entered_leaf_count"] == 0
    assert payload["summary"]["normal_complete_count"] == 0
    assert payload["summary"]["custody"] == "full_quarantine"
    controller._state = "NO_SETUP_GPU_WORK"
    with pytest.raises(WorkerSetupProtocolError):
        with controller.segment():
            pytest.fail("C0 field restoration revived failed controller")


def test_reentrant_sync_cannot_authorize_cleanup(peer):
    controller = lifecycle(peer)
    with controller.segment():
        controller.enter_device_leaf(controller.begin_device_leaf(descriptor()))

    def sync():
        with pytest.raises(WorkerSetupProtocolError):
            terminate(controller)

    summary = terminate(controller, sync).lifecycle.to_payload()["summary"]
    assert summary["emergency_sync_attempt_count"] == 1
    assert summary["emergency_sync_outcome"] == "failed"
    assert summary["release_status"] == "not_authorized"
    assert summary["custody"] == "full_quarantine"


def test_reentrant_destructor_cannot_report_success(peer):
    controller = lifecycle(peer)
    calls = []

    class ReentrantOwner:
        def __del__(self):
            try:
                terminate(controller)
            except WorkerSetupProtocolError:
                calls.append("rejected")

    controller.adopt_host_owner(ReentrantOwner(), role="reentrant_owner", operation_id="cpu_setup")
    summary = terminate(controller).lifecycle.to_payload()["summary"]
    assert calls == ["rejected"]
    assert summary["disposition"] == "cleanup_failed_exit"
    assert summary["guardian_state"] == "RELEASED"
    assert summary["custody"] == "remaining_quarantine"


@pytest.mark.parametrize("scope", [None, 0])
def test_returned_owner_is_retained_before_scope_rejection(peer, scope):
    controller = lifecycle(peer)
    owner = object()
    with pytest.raises(WorkerSetupProtocolError):
        with controller.segment():
            controller.enter_device_leaf(controller.begin_device_leaf(descriptor()))
            controller.adopt_returned_device_owner(
                owner, actual_device_ordinal=scope, role="returned", operation_id="cpu_leaf"
            )
    assert controller.guardian.contains_exact_owner(owner)
    payload = controller.termination_evidence.lifecycle.to_payload()
    assert payload["summary"]["disposition"] == "multi_or_unknown_device_quarantine_exit"
    assert payload["summary"]["emergency_sync_attempt_count"] == 0


@pytest.mark.parametrize("attack", ["delete", "duplicate", "guard", "epoch", "extra", "invent"])
def test_full_transaction_trace_rejects_local_edit_and_rehash(peer, attack):
    controller = lifecycle(peer)
    with controller.segment():
        controller.enter_device_leaf(controller.begin_device_leaf(descriptor()))
        controller.adopt_returned_device_owner(
            object(), actual_device_ordinal=1, role="returned", operation_id="cpu_leaf"
        )
    payload = terminate(controller, lambda: None).lifecycle.to_payload()
    trace = payload["transition_trace"]
    for index, original in enumerate(trace):
        forged = copy.deepcopy(trace)
        if attack == "delete":
            # Terminal intent omission is legitimately an unfinished prefix,
            # not something a consistency parser can identify as forgery.
            if index == len(trace) - 1:
                continue
            forged.pop(index)
        elif attack == "duplicate":
            forged.insert(index, copy.deepcopy(original))
        elif attack == "guard":
            forged[index]["guard"] = {"sequence": 9, "role": "setup"}
        elif attack == "epoch":
            forged[index]["epoch"] = 2
        elif attack == "extra":
            forged[index]["details"]["invented"] = True
        else:
            forged[index]["operation"] = "invented_transition"
        for position, event in enumerate(forged):
            event["index"] = position
        with pytest.raises(GuardProtocolError):
            replay_lifecycle(payload["configuration"], payload["lane_id"], forged)


class TerminalFaultPeer(Peer):
    """Inject before the real supervisor writes its committed terminal ACK."""

    def before_service(self):
        pending = self.core.pending_response
        if pending and pending["role"] == "emergency" and not getattr(self, "injected", False):
            self.injected = True
            with pytest.raises(WorkerSetupProtocolError):
                self.attack()


@pytest.mark.parametrize("dirty", [False, True])
@pytest.mark.parametrize("misuse", ["thread", "context"])
def test_terminal_ack_fault_prevents_all_subsequent_side_effects(dirty, misuse):
    peer = TerminalFaultPeer()
    try:
        controller = lifecycle(peer)
        owner = object()
        controller.adopt_host_owner(owner, role="sentinel", operation_id="setup")
        if dirty:
            with controller.segment():
                controller.enter_device_leaf(controller.begin_device_leaf(descriptor()))
        context = controller.segment()
        peer.attack = (
            lambda: controller.begin_device_leaf(descriptor())
            if misuse == "thread"
            else context.__exit__(None, None, None)
        )
        calls = []
        payload = terminate(controller, lambda: calls.append("sync")).lifecycle.to_payload()
        assert peer.injected and calls == []
        assert controller.guardian.contains_exact_owner(owner)
        operations = [event["operation"] for event in payload["transition_trace"]]
        assert operations.index("terminal_operation_failed") < operations.index(
            "guard_armed", operations.index("terminal_arm_started")
        )
        assert operations.count("terminal_operation_failed") == 1
        assert operations.count("terminal_arm_started") == 1
        assert "emergency_sync_started" not in operations
        assert "resource_release_authorized" not in operations
        assert "ordinary_cleanup_started" not in operations
        assert "cleanup_failed" not in operations
        assert payload["summary"]["custody"] == "full_quarantine"
        assert payload["summary"]["normal_complete_count"] == int(dirty)
        assert payload["summary"]["terminal_operation_failed"] is True
        assert GuardedLifecycleEvidence.from_payload(payload).to_payload() == payload
        with pytest.raises(WorkerSetupProtocolError, match="single-use"):
            terminate(controller)
        assert controller.evidence().to_payload() == payload
    finally:
        peer.close()


@pytest.mark.parametrize("dirty", [False, True])
@pytest.mark.parametrize("boundary", ["termination_started", "terminal_custody_retained"])
def test_terminal_entry_is_atomic_and_fault_before_arm_cannot_start_it(peer, monkeypatch, dirty, boundary):
    controller = lifecycle(peer)
    owner = object()
    controller.adopt_host_owner(owner, role="sentinel", operation_id="setup")
    if dirty:
        with controller.segment():
            controller.enter_device_leaf(controller.begin_device_leaf(descriptor()))
    checked = threading.Event()
    observations, errors, threads = [], [], []
    original_record = controller._record
    original_begin = controller._begin_termination

    def fault():
        acquired = controller._event_lock.acquire(blocking=False)
        observations.append(acquired)
        if acquired:
            controller._event_lock.release()
        checked.set()
        try:
            controller.begin_device_leaf(descriptor())
        except WorkerSetupProtocolError as error:
            errors.append(error)

    def record(operation, details):
        original_record(operation, details)
        if operation == boundary:
            thread = threading.Thread(target=fault)
            threads.append(thread)
            thread.start()
            assert checked.wait(2)
            assert observations == [False]
            assert errors == []  # Fault cannot commit in the entry critical section.

    def begin(*args):
        original_begin(*args)
        # Schedule the already waiting error before the next admission point.
        for thread in threads:
            thread.join(2)
            assert not thread.is_alive()
        assert len(errors) == 1

    monkeypatch.setattr(controller, "_record", record)
    monkeypatch.setattr(controller, "_begin_termination", begin)
    payload = terminate(controller, lambda: pytest.fail("pre-ARM fault synchronized")).lifecycle.to_payload()
    operations = [event["operation"] for event in payload["transition_trace"]]
    assert operations.index("terminal_custody_retained") < operations.index("terminal_operation_failed")
    assert "terminal_arm_started" not in operations
    assert "resource_release_authorized" not in operations
    assert payload["summary"]["custody"] == "full_quarantine"
    assert controller.guardian.contains_exact_owner(owner)
    assert GuardedLifecycleEvidence.from_payload(payload).to_payload() == payload


@pytest.mark.parametrize("boundary", ["before_sync", "before_release", "after_release"])
def test_terminal_success_boundaries_preserve_true_custody(peer, monkeypatch, boundary):
    controller = lifecycle(peer)
    owner = object()
    controller.adopt_host_owner(owner, role="sentinel", operation_id="setup")
    with controller.segment():
        controller.enter_device_leaf(controller.begin_device_leaf(descriptor()))
    method = "_synchronize" if boundary == "before_sync" else "_release"
    original = getattr(controller, method)

    def misuse():
        with pytest.raises(WorkerSetupProtocolError):
            controller.begin_device_leaf(descriptor())

    def boundary_call(*args):
        if boundary != "after_release":
            misuse()
        result = original(*args)
        if boundary == "after_release":
            misuse()
        return result

    monkeypatch.setattr(controller, method, boundary_call)
    sync_calls = []
    payload = terminate(controller, lambda: sync_calls.append(True)).lifecycle.to_payload()
    operations = [event["operation"] for event in payload["transition_trace"]]
    assert sync_calls == ([] if boundary == "before_sync" else [True])
    assert payload["summary"]["terminal_operation_failed"] is True
    if boundary == "after_release":
        assert payload["summary"]["disposition"] == "cleanup_failed_exit"
        assert payload["summary"]["custody"] == "remaining_quarantine"
        assert operations.index("guardian_released") < operations.index("terminal_operation_failed")
        assert not controller.guardian.contains_exact_owner(owner)
    else:
        assert payload["summary"]["custody"] == "full_quarantine"
        assert "resource_release_authorized" not in operations
        assert controller.guardian.contains_exact_owner(owner)


@pytest.mark.parametrize("during", ["sync", "destructor"])
def test_blocking_native_or_destructor_call_does_not_hold_controller_lock(peer, during):
    controller = lifecycle(peer)
    rejected, threads = [], []

    def wrong_thread():
        try:
            controller.begin_device_leaf(descriptor())
        except WorkerSetupProtocolError:
            rejected.append(True)

    def inject():
        thread = threading.Thread(target=wrong_thread)
        threads.append(thread)
        thread.start()
        thread.join(2)
        # Record rather than throw inside __del__; assertions run below.
        rejected.append(not thread.is_alive())

    class Owner:
        def __del__(self):
            inject()

    if during == "sync":
        with controller.segment():
            controller.enter_device_leaf(controller.begin_device_leaf(descriptor()))
    else:
        controller.adopt_host_owner(Owner(), role="sentinel", operation_id="setup")
    payload = terminate(controller, inject).lifecycle.to_payload()
    for thread in threads:
        thread.join(2)
        assert not thread.is_alive()
    assert rejected == [True, True]
    assert payload["summary"]["terminal_operation_failed"] is True
    assert payload["summary"]["custody"] == (
        "full_quarantine" if during == "sync" else "remaining_quarantine"
    )
