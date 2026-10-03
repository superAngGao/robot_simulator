"""C2d1 CPU-only real client/core integration; GPU adoption remains closed."""

import copy
import json
import os
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from optical_worker.guarded_evidence import (
    LIFECYCLE_VERSION,
    GuardedLifecycleEvidence,
    GuardedTerminationEvidence,
    replay_lifecycle,
)
from optical_worker.guarded_lifecycle import _TERMINAL_CUSTODY, NativeGuardedSetupLifecycle
from optical_worker.lifecycle import DeviceScopeDescriptor, WorkerSetupProtocolError
from optical_worker.native_setup import NativeWorkerPreparationAborted, _abort
from optical_worker.wait_protocol import GuardConfiguration, GuardProtocolError, GuardWaitClient
from tests.support.guarded_peer import Peer


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
@pytest.mark.parametrize(
    "misuse",
    [
        "thread",
        "context",
        "commit_target_barrier",
        "commit_clean_close",
        "latch_measured_sample_submission",
        "require_adopted",
    ],
)
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
        if misuse == "thread":
            peer.attack = lambda: controller.begin_device_leaf(descriptor())
        elif misuse == "context":
            peer.attack = lambda: context.__exit__(None, None, None)
        else:
            peer.attack = lambda: getattr(controller, misuse)()
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


@pytest.mark.parametrize(
    "scope",
    ["explicit_unknown", "returned_foreign", "returned_unknown", "descriptor_foreign", "descriptor_unknown"],
)
def test_compromised_scope_cannot_complete_even_when_scope_error_is_caught(peer, scope):
    controller = lifecycle(peer)
    owner = object()
    returned_owner = object()
    controller.adopt_host_owner(owner, role="sentinel", operation_id="setup")
    with pytest.raises(WorkerSetupProtocolError):
        with controller.segment():
            controller.enter_device_leaf(controller.begin_device_leaf(descriptor()))
            if scope == "explicit_unknown":
                controller.mark_unknown_device_scope(source="fixed_test", reason="unknown_return")
            else:
                with pytest.raises(WorkerSetupProtocolError):
                    if scope.startswith("returned"):
                        controller.adopt_returned_device_owner(
                            returned_owner,
                            actual_device_ordinal=0 if scope.endswith("foreign") else None,
                            role="returned",
                            operation_id="fixed_test",
                        )
                    else:
                        value = (
                            descriptor(0)
                            if scope.endswith("foreign")
                            else replace(descriptor(), device_ordinals=(), unreturned_resource_possible=True)
                        )
                        controller.begin_device_leaf(value)
            # Even swallowing the scope rejection cannot admit another leaf.
            with pytest.raises(WorkerSetupProtocolError):
                controller.begin_device_leaf(descriptor())
    payload = controller.termination_evidence.lifecycle.to_payload()
    with pytest.raises(WorkerSetupProtocolError):
        with controller.segment("resolve"):
            pytest.fail("new body after scope compromise")
    operations = [event["operation"] for event in payload["transition_trace"]]
    assert operations.count("normal_arm_started") == 1
    assert "normal_body_returned" not in operations
    assert "normal_complete_started" not in operations
    assert "terminal_arm_started" not in operations
    assert payload["summary"]["normal_complete_count"] == 0
    assert payload["summary"]["entered_leaf_count"] == 1
    assert payload["summary"]["disposition"] == "multi_or_unknown_device_quarantine_exit"
    assert controller.guardian.contains_exact_owner(owner)
    if scope.startswith("returned"):
        assert controller.guardian.contains_exact_owner(returned_owner)
    assert controller.evidence().to_payload() == payload


def test_unknown_scope_before_first_guard_rejects_arm(peer):
    controller = lifecycle(peer)
    controller.mark_unknown_device_scope(source="fixed_test", reason="unknown_import")
    with pytest.raises(WorkerSetupProtocolError):
        with controller.segment():
            pytest.fail("compromised scope entered first guard")
    payload = terminate(controller).lifecycle.to_payload()
    assert all("arm" not in event["operation"] for event in payload["transition_trace"])
    assert payload["summary"]["custody"] == "full_quarantine"


@pytest.mark.parametrize(
    "failure",
    [
        "foreign_descriptor",
        "unknown_descriptor",
        "foreign_owner",
        "unknown_owner",
        "explicit_unknown",
        "invalid_permit",
        "sealed_invalid_permit",
        "sealed_invalid_permit_unknown",
    ],
)
@pytest.mark.parametrize("propagate", [False, True])
def test_native_abort_commits_custody_before_outer_context_unwinds(peer, failure, propagate):
    controller = lifecycle(peer)
    sentinel, returned_owner = object(), object()
    controller.adopt_host_owner(sentinel, role="sentinel", operation_id="setup")
    before_exit = None
    original = None
    try:
        with controller.segment():
            if failure.startswith("sealed_"):
                controller.seal_guardian()
            try:
                if failure == "foreign_descriptor":
                    controller.begin_device_leaf(descriptor(0))
                elif failure == "unknown_descriptor":
                    controller.begin_device_leaf(
                        replace(descriptor(), device_ordinals=(), unreturned_resource_possible=True)
                    )
                elif failure in {"foreign_owner", "unknown_owner"}:
                    controller.adopt_returned_device_owner(
                        returned_owner,
                        actual_device_ordinal=0 if failure == "foreign_owner" else None,
                        role="returned",
                        operation_id="fixed_test",
                    )
                elif failure == "explicit_unknown":
                    controller.mark_unknown_device_scope(source="fixed_test", reason="unknown_return")
                    raise RuntimeError("original setup failure")
                else:
                    controller.enter_device_leaf(object())
            except BaseException as error:
                original = error
                if failure == "sealed_invalid_permit_unknown":
                    controller.mark_unknown_device_scope(source="test", reason="sealed_failure")
                aborted = _abort(controller, exception=error, sync=lambda: pytest.fail("abort retried sync"))
                assert type(aborted) is NativeWorkerPreparationAborted
                assert aborted.evidence is controller.termination_evidence
                before_exit = aborted.evidence.lifecycle.to_payload()
                assert controller.guardian.state == "TRANSFERRED_TO_QUARANTINE"
                assert controller.guardian.contains_exact_owner(sentinel)
                assert any(item is error for item in controller._failure_holds)
                if not failure.startswith("sealed_"):
                    assert controller.guardian.contains_exact_owner(error)
                if failure in {"foreign_owner", "unknown_owner"}:
                    assert controller.guardian.contains_exact_owner(returned_owner)
                summary = before_exit["summary"]
                assert summary["controller_state"] == "TERMINAL_INTENT"
                assert summary["custody"] == "full_quarantine"
                assert summary["normal_complete_count"] == summary["terminal_arm_attempt_count"] == 0
                assert summary["emergency_sync_attempt_count"] == 0
                assert summary["release_status"] == "not_authorized"
                operations = [event["operation"] for event in before_exit["transition_trace"]]
                assert operations.count("termination_started") == 1
                assert operations.count("terminal_custody_retained") == 1
                if propagate:
                    raise aborted from error
            else:
                pytest.fail("fault did not occur")
    except NativeWorkerPreparationAborted as error:
        assert propagate and error.__cause__ is original
    except WorkerSetupProtocolError:
        # Catching the aborted result cannot make the outer segment complete.
        assert not propagate
    else:
        pytest.fail("terminated segment exited normally")
    assert before_exit is not None
    assert controller.evidence().to_payload() == before_exit


@pytest.mark.parametrize("mutation", ["not_exception", "wrong_role", "wrong_operation"])
def test_failure_retention_does_not_reopen_normal_host_adoption(peer, mutation):
    controller = lifecycle(peer)
    value = object() if mutation == "not_exception" else RuntimeError("custody only")
    with pytest.raises(WorkerSetupProtocolError):
        with controller.segment():
            with pytest.raises(WorkerSetupProtocolError):
                controller.enter_device_leaf(object())
            with pytest.raises(WorkerSetupProtocolError):
                controller.adopt_host_owner(
                    value,
                    role="normal" if mutation == "wrong_role" else "setup_exception_graph",
                    operation_id="normal" if mutation == "wrong_operation" else "setup_exception",
                )
            assert not controller.guardian.contains_exact_owner(value)
    assert controller.termination_evidence.lifecycle.to_payload()["summary"]["custody"] == "full_quarantine"


def test_explicit_unknown_scope_rejects_normal_exit_without_another_public_call(peer):
    controller = lifecycle(peer)
    with pytest.raises(WorkerSetupProtocolError):
        with controller.segment():
            controller.mark_unknown_device_scope(source="fixed_test", reason="unknown_return")
    summary = controller.termination_evidence.lifecycle.to_payload()["summary"]
    assert summary["custody"] == "full_quarantine"
    assert summary["normal_complete_count"] == summary["terminal_arm_attempt_count"] == 0


@pytest.mark.parametrize("lane", ["cuda", "warp"])
@pytest.mark.parametrize("prior_fault", [False, True])
@pytest.mark.parametrize("propagate", [False, True])
def test_actual_prepare_import_failure_records_scope_before_unwind(tmp_path, lane, prior_fault, propagate):
    # A combined pytest collection may already import Torch/Warp. Observe a
    # genuine cold import in an isolated CPU child; never unload runtime modules.
    harness = Path(__file__).resolve().parents[2] / "support" / "guarded_custody_process.py"
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-B",
            str(harness),
            "import_probe_propagated" if propagate else "import_probe_caught",
            lane + ("_import_thread_failure" if prior_fault else "_import_failure"),
            "0",
            str(tmp_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        output, error = process.communicate(timeout=25)
        assert process.returncode == 0, (output, error)
        identity = json.loads(output)
        assert identity["pid"] == process.pid
        assert identity["ppid"] == os.getpid()
        print(json.dumps({"import_probe": identity, "returncode": process.returncode}))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)


def test_unknown_scope_reporting_still_rejects_wrong_thread_and_post_intent(peer):
    controller = lifecycle(peer)
    rejected = []

    def misuse():
        try:
            controller.mark_unknown_device_scope(source="test", reason="test")
        except WorkerSetupProtocolError as error:
            rejected.append(error)

    thread = threading.Thread(target=misuse)
    thread.start()
    thread.join(2)
    assert not thread.is_alive() and len(rejected) == 1
    assert not controller.evidence().to_payload()["summary"]["unknown_device_scope"]
    # Owner-thread failure reporting can tighten, but not clear, the failed state.
    controller.mark_unknown_device_scope(source="test", reason="observed_after_failure")
    before = terminate(controller).lifecycle.to_payload()
    assert before["summary"]["guarded_segment_failed"] and before["summary"]["unknown_device_scope"]
    misuse()
    assert len(rejected) == 2
    assert controller.evidence().to_payload() == before


def test_unknown_scope_reporting_during_terminal_sync_is_reentry_not_new_scope(peer):
    controller = lifecycle(peer)
    with controller.segment():
        controller.enter_device_leaf(controller.begin_device_leaf(descriptor()))

    def misuse():
        with pytest.raises(WorkerSetupProtocolError):
            controller.mark_unknown_device_scope(source="test", reason="late_scope")

    summary = terminate(controller, misuse).lifecycle.to_payload()["summary"]
    assert summary["terminal_operation_failed"] and not summary["unknown_device_scope"]
    assert summary["emergency_sync_outcome"] == "failed"
    assert summary["custody"] == "full_quarantine"


@pytest.mark.parametrize("checkpoint", ["descriptor_observed", "owner_adopted"])
def test_foreign_thread_failure_cannot_split_scope_notifications(peer, monkeypatch, checkpoint):
    controller = lifecycle(peer)
    attempted, finished = threading.Event(), threading.Event()
    threads, rejected = [], []
    record = controller._record

    def misuse():
        attempted.set()
        try:
            controller.require_adopted()
        except WorkerSetupProtocolError as error:
            rejected.append(error)
        finally:
            finished.set()

    def record_and_schedule(operation, details):
        record(operation, details)
        if operation == checkpoint and not threads:
            thread = threading.Thread(target=misuse)
            threads.append(thread)
            thread.start()
            assert attempted.wait(2)
            # The real metadata operation owns the same CPU state lock here.
            assert not finished.wait(0.02)

    monkeypatch.setattr(controller, "_record", record_and_schedule)
    with pytest.raises(NativeWorkerPreparationAborted):
        with controller.segment():
            if checkpoint == "descriptor_observed":
                controller.begin_device_leaf(descriptor())
            else:
                with pytest.raises(WorkerSetupProtocolError):
                    controller.adopt_returned_device_owner(
                        object(), actual_device_ordinal=0, role="returned", operation_id="test"
                    )
            threads[0].join(2)
            assert finished.is_set() and not threads[0].is_alive() and len(rejected) == 1
            assert not controller._replay.notifications
            controller.mark_unknown_device_scope(source="test", reason="partial_failure")
            aborted = _abort(controller, exception=RuntimeError("original"), sync=lambda: pytest.fail("sync"))
            assert aborted.evidence.lifecycle.to_payload()["summary"]["unknown_device_scope"]
            raise aborted


@pytest.mark.parametrize(
    "method",
    ["commit_target_barrier", "commit_clean_close", "latch_measured_sample_submission", "require_adopted"],
)
@pytest.mark.parametrize("during", ["sync", "cleanup"])
def test_caught_public_api_rejection_invalidates_terminal_success(peer, method, during):
    controller = lifecycle(peer)
    rejected = []

    def misuse():
        try:
            getattr(controller, method)()
        except WorkerSetupProtocolError:
            rejected.append(True)

    class Owner:
        def __del__(self):
            misuse()

    if during == "sync":
        with controller.segment():
            controller.enter_device_leaf(controller.begin_device_leaf(descriptor()))
    else:
        controller.adopt_host_owner(Owner(), role="sentinel", operation_id="setup")
    payload = terminate(controller, misuse).lifecycle.to_payload()
    assert rejected == [True]
    assert payload["summary"]["terminal_operation_failed"] is True
    assert payload["summary"]["custody"] == (
        "full_quarantine" if during == "sync" else "remaining_quarantine"
    )
    assert payload["summary"]["emergency_sync_outcome"] == ("failed" if during == "sync" else "not_attempted")
    with pytest.raises(WorkerSetupProtocolError):
        getattr(controller, method)()
    assert controller.evidence().to_payload() == payload


@pytest.mark.parametrize(
    "method",
    [
        "descriptor",
        "permit",
        "ownership",
        "host_adoption",
        "device_adoption",
        "unknown_scope",
        "seal",
        "commit_clean_close",
        "invalid_termination",
        "invalid_segment_role",
    ],
)
def test_swallowed_public_operation_error_cannot_complete_guard(peer, method):
    controller = lifecycle(peer)
    with pytest.raises(WorkerSetupProtocolError):
        with controller.segment():
            with pytest.raises((WorkerSetupProtocolError, GuardProtocolError, ValueError)):
                if method == "descriptor":
                    controller.begin_device_leaf(object())
                elif method == "permit":
                    controller.enter_device_leaf(object())
                elif method == "ownership":
                    controller.require_adopted(object())
                elif method == "host_adoption":
                    controller.adopt_host_owner(object(), role="!invalid", operation_id="test")
                elif method == "device_adoption":
                    controller.adopt_returned_device_owner(
                        object(), actual_device_ordinal=1, role="!invalid", operation_id="test"
                    )
                elif method == "unknown_scope":
                    controller.mark_unknown_device_scope(source="!invalid", reason="test")
                elif method == "seal":
                    controller.begin_device_leaf(descriptor())
                    controller.seal_guardian()
                elif method == "invalid_termination":
                    controller.terminate(
                        fault="not_a_fault", detection_phase="test", emergency_target_synchronize=None
                    )
                elif method == "invalid_segment_role":
                    with controller.segment([]):
                        pytest.fail("invalid role entered body")
                else:
                    controller.commit_clean_close()
    payload = controller.termination_evidence.lifecycle.to_payload()
    assert payload["summary"]["guarded_segment_failed"] is True
    assert payload["summary"]["normal_complete_count"] == 0
    assert payload["summary"]["terminal_arm_attempt_count"] == 0
    assert payload["summary"]["custody"] == "full_quarantine"
