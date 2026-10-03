"""Hand-authored v3 stories: expected summaries do not come from the reducer."""

import copy

import pytest

from optical_worker.guarded_evidence import LIFECYCLE_VERSION, GuardedLifecycleEvidence, replay_lifecycle
from optical_worker.protocol import canonical_sha256
from optical_worker.wait_protocol import GuardConfiguration, GuardProtocolError


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
