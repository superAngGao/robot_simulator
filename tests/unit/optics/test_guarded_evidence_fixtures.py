"""Hand-authored v3 stories: expected summaries do not come from the reducer."""

import copy

import pytest

from optical_worker.guarded_evidence import (
    LIFECYCLE_VERSION,
    VALIDATION_VERSION,
    GuardedLifecycleEvidence,
    replay_lifecycle,
    validate_custody_artifact,
)
from optical_worker.lifecycle import UNKNOWN_DEVICE_OWNER
from optical_worker.protocol import canonical_sha256
from optical_worker.wait_protocol import GuardConfiguration, GuardProtocolError
from optical_worker.wait_supervision import GuardSupervisorCore


def story(kind):
    configuration = GuardConfiguration("a" * 32, 1, 1000, 5000).to_payload()
    coverage = {
        "schema_version": "q54.p12.4.worker-guardian-coverage.v1",
        "session_identity": "a" * 32,
        "state": "CONSTRUCTING",
        "records": [],
        "per_device_resource_counts": {},
    }
    summary = {
        "controller_state": "TERMINAL_INTENT",
        "guardian_state": "TRANSFERRED_TO_QUARANTINE",
        "guard_state": "FAILED",
        "guard_ref": {"sequence": 0, "role": "setup"},
        "next_sequence": 1,
        "terminal_purpose": None,
        "guarded_segment_failed": True,
        "terminal_operation_failed": False,
        "possible_dirty_device_ordinals": [1],
        "unknown_device_scope": False,
        "setup_operation_count": 1,
        "entered_leaf_count": 1,
        "setup_work_counts": {"allocation": 1},
        "normal_complete_count": 0,
        "terminal_arm_attempt_count": 0,
        "emergency_sync_attempt_count": 0,
        "emergency_sync_outcome": "not_attempted",
        "custody": "full_quarantine",
        "release_status": "not_authorized",
        "disposition": "quarantined_process_exit",
    }
    events = []
    epoch = 0
    guard = None

    def emit(operation, **details):
        events.append(
            {
                "index": len(events),
                "epoch": epoch,
                "operation": operation,
                "guard": copy.deepcopy(guard),
                "details": details,
            }
        )

    def guardian(operation, state):
        coverage["state"] = state
        emit(operation, coverage_sha256=canonical_sha256(coverage))

    emit("guardian_created", lane_id="warp", target_device_ordinal=1)
    guard = {"sequence": 0, "role": "setup"}
    emit("normal_arm_started")
    if kind == "arm_failure":
        emit("guard_failed", failure_kind="arm_ack")
        summary.update(
            next_sequence=0,
            possible_dirty_device_ordinals=[],
            setup_operation_count=0,
            entered_leaf_count=0,
            setup_work_counts={},
        )
    else:
        emit("guard_armed")
        epoch = 1
        descriptor = {
            "schema_version": "q54.p12.4.closed-worker-device-scope.v1",
            "operation_id": "fixed_leaf",
            "leaf_kind": "allocation",
            "device_ordinals": [0] if kind == "foreign" else ([] if kind == "unknown" else [1]),
            "scope_source": "fixed_source",
            "runtime_identity": "fixed_runtime",
            "unreturned_resource_possible": kind == "unknown",
        }
        emit("descriptor_observed", descriptor=descriptor)
        emit(
            "setup_operation_started",
            descriptor_sha256=canonical_sha256(descriptor),
            operation_id="fixed_leaf",
            leaf_kind="allocation",
        )
        if kind == "unknown":
            emit("unknown_scope_set", source="fixed_source", reason="unreturned_resource_possible")
            summary.update(
                possible_dirty_device_ordinals=[],
                unknown_device_scope=True,
                entered_leaf_count=0,
                disposition="multi_or_unknown_device_quarantine_exit",
            )
        else:
            emit(
                "dirty_ordinal_added",
                device_ordinal=0 if kind == "foreign" else 1,
                source="fixed_source",
                reason="pre_leaf_descriptor",
            )
            if kind == "foreign":
                summary.update(
                    possible_dirty_device_ordinals=[0],
                    entered_leaf_count=0,
                    disposition="multi_or_unknown_device_quarantine_exit",
                )
            elif kind != "before_entry":
                emit(
                    "low_level_leaf_entered",
                    descriptor_sha256=canonical_sha256(descriptor),
                    operation_sequence=1,
                )
            else:
                summary["entered_leaf_count"] = 0
        if kind in {"inner_abort", "before_entry", "foreign", "unknown"}:
            emit("guard_failed", failure_kind="operation")
        else:
            emit("normal_body_returned")
            emit("normal_complete_started")
            if kind == "complete_failure":
                emit("guard_failed", failure_kind="complete_ack")
            else:
                emit("normal_disarmed")
                guard = None
                summary.update(
                    guarded_segment_failed=False,
                    normal_complete_count=1,
                    next_sequence=2,
                    terminal_arm_attempt_count=1,
                    terminal_purpose="target_sync",
                    guard_ref={"sequence": 1, "role": "emergency"},
                    guard_state="ARMED",
                    emergency_sync_attempt_count=1,
                    emergency_sync_outcome="succeeded",
                )
    emit("termination_started", fault="setup_failure", detection_phase="fixed_fixture")
    emit("terminal_custody_retained")
    guardian("guardian_partial_frozen", "PARTIAL_FROZEN")
    release = kind in {"sync_success", "cleanup_before", "cleanup_after"}
    if kind in {"sync_success", "sync_failure", "cleanup_before", "cleanup_after"}:
        guard = {"sequence": 1, "role": "emergency"}
        emit("terminal_arm_started", purpose="target_sync")
        emit("guard_armed")
        emit("emergency_sync_started", device_ordinal=1)
        emit("emergency_sync_succeeded" if release else "emergency_sync_failed", device_ordinal=1)
        if not release:
            summary["emergency_sync_outcome"] = "failed"
    if release:
        emit("resource_release_authorized", reason="emergency_sync")
        coverage["state"] = "RELEASE_AUTHORIZED"
        emit("ordinary_cleanup_started")
        if kind == "sync_success":
            emit("guardian_released")
            coverage["state"] = "RELEASED"
            summary.update(
                guardian_state="RELEASED",
                custody="retained",
                release_status="completed",
                disposition="aborted_synced_exit",
            )
        else:
            state = "RELEASED" if kind == "cleanup_after" else "RELEASE_AUTHORIZED"
            emit("cleanup_failed", guardian_state=state)
            emit("quarantine_custody_committed", extent="remaining")
            coverage["state"] = state
            summary.update(
                guardian_state=state,
                custody="remaining_quarantine",
                release_status="failed",
                disposition="cleanup_failed_exit",
            )
    else:
        guardian("guardian_quarantine_transferred", "TRANSFERRED_TO_QUARANTINE")
        emit("quarantine_custody_committed", extent="full")
    emit("terminal_intent", disposition=summary["disposition"])
    return {
        "schema_version": LIFECYCLE_VERSION,
        "configuration": configuration,
        "lane_id": "warp",
        "record_kind": "terminal",
        "transition_trace": events,
        "guardian_coverage": coverage,
        "summary": summary,
    }


KINDS = (
    "inner_abort",
    "before_entry",
    "foreign",
    "unknown",
    "arm_failure",
    "complete_failure",
    "sync_success",
    "sync_failure",
    "cleanup_before",
    "cleanup_after",
)


@pytest.mark.parametrize("kind", KINDS)
def test_independently_written_stories(kind):
    payload = story(kind)
    assert GuardedLifecycleEvidence.from_payload(payload).to_payload() == payload


@pytest.mark.parametrize("kind", KINDS)
def test_duplicate_transition_matrix(kind):
    payload = story(kind)
    for index, event in enumerate(payload["transition_trace"]):
        trace = copy.deepcopy(payload["transition_trace"])
        trace.insert(index, copy.deepcopy(event))
        for position, item in enumerate(trace):
            item["index"] = position
        with pytest.raises(GuardProtocolError):
            replay_lifecycle(payload["configuration"], "warp", trace)


@pytest.mark.parametrize("kind", ["cleanup_before", "cleanup_after"])
def test_cleanup_cannot_claim_complete_quarantine_or_omit_start(kind):
    payload = story(kind)
    forged = copy.deepcopy(payload)
    for event in forged["transition_trace"]:
        if event["operation"] == "quarantine_custody_committed":
            event["details"]["extent"] = "full"
    with pytest.raises(GuardProtocolError):
        replay_lifecycle(forged["configuration"], "warp", forged["transition_trace"])
    if kind == "cleanup_after":
        trace = [
            event for event in payload["transition_trace"] if event["operation"] != "ordinary_cleanup_started"
        ]
        for index, event in enumerate(trace):
            event["index"] = index
        with pytest.raises(GuardProtocolError):
            replay_lifecycle(payload["configuration"], "warp", trace)


def test_bool_is_not_a_device_ordinal():
    payload = story("sync_success")
    payload["transition_trace"][0]["details"]["target_device_ordinal"] = True
    with pytest.raises(GuardProtocolError):
        GuardedLifecycleEvidence.from_payload(payload)


@pytest.mark.parametrize("kind", ["foreign", "unknown"])
def test_missing_scope_notification_cannot_manufacture_clean_release(kind):
    payload = story(kind)
    trace = [
        event
        for event in payload["transition_trace"]
        if event["operation"] not in {"dirty_ordinal_added", "unknown_scope_set"}
    ]
    for index, event in enumerate(trace):
        event["index"] = index
    replay = replay_lifecycle(payload["configuration"], "warp", trace)
    assert replay.summary()["disposition"] == "multi_or_unknown_device_quarantine_exit"
    assert replay.summary()["possible_dirty_device_ordinals"] == ([0] if kind == "foreign" else [])
    assert replay.summary()["unknown_device_scope"] == (kind == "unknown")


def prefix(payload, last_operation):
    """Cut actual grammar at an event; this helper is not an expected-summary oracle."""
    payload = copy.deepcopy(payload)
    index = next(
        i for i, event in enumerate(payload["transition_trace"]) if event["operation"] == last_operation
    )
    payload["transition_trace"] = payload["transition_trace"][: index + 1]
    return rebuilt_prefix(payload)


def rebuilt_prefix(payload):
    for index, event in enumerate(payload["transition_trace"]):
        event["index"] = index
    replay = replay_lifecycle(payload["configuration"], payload["lane_id"], payload["transition_trace"])
    payload.update(record_kind="prefix", summary=replay.summary(), guardian_coverage=replay.coverage())
    return payload


def artifact(worker, steps):
    config = GuardConfiguration.from_payload(
        worker["configuration"] if worker is not None else story("inner_abort")["configuration"]
    )
    core = GuardSupervisorCore(config)
    for operation, sequence, role, ack in steps:
        response = core.accept(config.message(operation, sequence, role))
        if ack:
            core.acknowledge_written(response)
    core.observe_exit(74)
    return {
        "schema_version": VALIDATION_VERSION,
        "worker": worker,
        "supervisor": core.evidence(require_final=True).to_payload(),
    }


SETUP_ARM = ("ARM", 0, "setup", True)
SETUP_COMPLETE = ("COMPLETE", 0, "setup", True)


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("received_ack", [False, True])
def test_conflicting_pending_arm_role_rejected(terminal, received_ack):
    worker = story("arm_failure")
    if not terminal:
        worker = prefix(worker, "normal_arm_started")
    with pytest.raises(GuardProtocolError, match="conflicts"):
        validate_custody_artifact(artifact(worker, [("ARM", 0, "resolve", received_ack)]))


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("extra", ["complete", "normal_arm", "emergency_arm"])
def test_failure_cannot_be_followed_by_complete_or_new_arm(terminal, extra):
    worker = story("inner_abort")
    if not terminal:
        worker = prefix(worker, "guard_failed")
    steps = [SETUP_ARM, SETUP_COMPLETE]
    if extra != "complete":
        steps.append(("ARM", 1, "resolve" if extra == "normal_arm" else "emergency", True))
    with pytest.raises(GuardProtocolError):
        validate_custody_artifact(artifact(worker, steps))


@pytest.mark.parametrize("state", ["no_arm", "arm_pending", "ack_written"])
def test_failed_arm_allows_unknown_matching_transport_outcome(state):
    steps = [] if state == "no_arm" else [("ARM", 0, "setup", state == "ack_written")]
    validate_custody_artifact(artifact(story("arm_failure"), steps))


@pytest.mark.parametrize("completion", ["none", "pending", "ack_written"])
def test_failed_complete_ack_can_follow_real_completion(completion):
    steps = [SETUP_ARM]
    if completion != "none":
        steps.append(("COMPLETE", 0, "setup", completion == "ack_written"))
    validate_custody_artifact(artifact(story("complete_failure"), steps))


def test_absent_worker_and_healthy_prefix_may_lag_supervisor():
    steps = [SETUP_ARM, SETUP_COMPLETE, ("ARM", 1, "resolve", True)]
    validate_custody_artifact(artifact(None, steps))
    for operation in ("guardian_created", "normal_arm_started", "guard_armed", "normal_complete_started"):
        validate_custody_artifact(artifact(prefix(story("sync_success"), operation), steps))


@pytest.mark.parametrize("extra_role", [None, "emergency", "setup", "resolve"])
def test_healthy_terminal_prefix_permits_only_next_emergency(extra_role):
    worker = prefix(story("sync_success"), "termination_started")
    steps = [SETUP_ARM, SETUP_COMPLETE]
    if extra_role:
        steps.append(("ARM", 1, extra_role, True))
    if extra_role in {None, "emergency"}:
        validate_custody_artifact(artifact(worker, steps))
    else:
        with pytest.raises(GuardProtocolError):
            validate_custody_artifact(artifact(worker, steps))


def compromised_prefix(source):
    if source.startswith("descriptor"):
        return prefix(story("unknown" if source.endswith("unknown") else "foreign"), "descriptor_observed")
    worker = prefix(story("inner_abort"), "low_level_leaf_entered")
    event = {
        "index": len(worker["transition_trace"]),
        "epoch": 1,
        "guard": {"sequence": 0, "role": "setup"},
    }
    if source == "explicit_unknown":
        event.update(
            operation="unknown_scope_set", details={"source": "fixed_source", "reason": "failed_owner"}
        )
    else:
        event.update(
            operation="owner_adopted",
            details={
                "sequence": 0,
                "role": "destination",
                "owner_type": "FixedOwner",
                "device_scope": UNKNOWN_DEVICE_OWNER if source.endswith("unknown") else 0,
                "operation_id": "fixed_leaf",
            },
        )
    worker["transition_trace"].append(event)
    return rebuilt_prefix(worker)


@pytest.mark.parametrize(
    "source",
    ["descriptor_foreign", "descriptor_unknown", "owner_foreign", "owner_unknown", "explicit_unknown"],
)
@pytest.mark.parametrize("extra", ["matching_only", "complete", "normal_arm", "emergency_arm"])
def test_compromised_prefix_before_failure_or_notification_is_irreversible(source, extra):
    worker = compromised_prefix(source)
    assert worker["summary"]["controller_state"] == "RUNNING"
    assert worker["summary"]["guarded_segment_failed"] is False
    steps = [SETUP_ARM]
    if extra != "matching_only":
        steps.append(SETUP_COMPLETE)
        if extra != "complete":
            steps.append(("ARM", 1, "resolve" if extra == "normal_arm" else "emergency", True))
        with pytest.raises(GuardProtocolError):
            validate_custody_artifact(artifact(worker, steps))
    else:
        validate_custody_artifact(artifact(worker, steps))


@pytest.mark.parametrize(
    "boundary",
    [
        "guardian_created",
        "terminal_custody_retained",
        "terminal_arm_started",
        "guard_armed",
        "emergency_sync_started",
        "emergency_sync_succeeded",
        "ordinary_cleanup_started",
    ],
)
def test_terminal_failure_event_replay_and_no_later_success(boundary):
    worker = story("sync_success")
    # Select the terminal ACK, not the earlier normal ACK.
    index = max(i for i, event in enumerate(worker["transition_trace"]) if event["operation"] == boundary)
    event = copy.deepcopy(worker["transition_trace"][index])
    event.update(operation="terminal_operation_failed", details={"failure_kind": "thread"})
    worker["transition_trace"].insert(index + 1, event)
    for position, item in enumerate(worker["transition_trace"]):
        item["index"] = position
    with pytest.raises(GuardProtocolError):
        replay_lifecycle(worker["configuration"], "warp", worker["transition_trace"])
    # After custody, the prefix itself is valid and its failure is irreversible.
    if boundary != "guardian_created":
        worker["transition_trace"] = worker["transition_trace"][: index + 2]
        worker = rebuilt_prefix(worker)
        assert worker["summary"]["terminal_operation_failed"] is True
        duplicate = copy.deepcopy(worker["transition_trace"][-1])
        duplicate["index"] += 1
        worker["transition_trace"].append(duplicate)
        with pytest.raises(GuardProtocolError):
            rebuilt_prefix(worker)
