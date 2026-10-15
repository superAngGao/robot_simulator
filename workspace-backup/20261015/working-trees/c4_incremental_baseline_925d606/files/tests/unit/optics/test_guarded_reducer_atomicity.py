"""Rejecting a descriptive transition must not change the shared reducer."""

from copy import deepcopy

import pytest

from optical_worker.guarded_evidence import _Replay
from optical_worker.wait_protocol import GuardConfiguration, GuardProtocolError
from tests.unit.optics.test_guarded_evidence_fixtures import KINDS, story


def initialized():
    state = _Replay(GuardConfiguration("a" * 32, 1, 100, 1000), "cuda")
    state.apply({"index": 0, "epoch": 0, "operation": "guardian_created", "guard": None,
                 "details": {"lane_id": "cuda", "target_device_ordinal": 1}})
    return state


@pytest.mark.parametrize("fault", ["bad_descriptor", "bad_coverage"])
def test_late_invalid_transition_preserves_every_reducer_field(fault):
    state = initialized()
    if fault == "bad_descriptor":
        guard = {"sequence": 0, "role": "setup"}
        for operation in ("normal_arm_started", "guard_armed"):
            state.apply({"index": state.index, "epoch": 0, "operation": operation,
                         "guard": guard, "details": {}})
        event = {"index": state.index, "epoch": 1, "operation": "descriptor_observed",
                 "guard": guard, "details": {"descriptor": {"invalid": True}}}
    else:
        event = {"index": state.index, "epoch": 0, "operation": "guardian_sealed",
                 "guard": None, "details": {"coverage_sha256": "0" * 64}}
    before = deepcopy(vars(state))
    with pytest.raises(GuardProtocolError):
        state.apply(event)
    assert vars(state) == before


@pytest.mark.parametrize("kind", KINDS)
def test_invalid_fields_at_each_real_story_step_leave_state_unchanged(kind):
    payload = story(kind)
    state = _Replay(GuardConfiguration.from_payload(payload["configuration"]), payload["lane_id"])
    for event in payload["transition_trace"]:
        for field in event["details"]:
            corrupted = deepcopy(event)
            corrupted["details"][field] = None
            before = deepcopy(vars(state))
            with pytest.raises(GuardProtocolError):
                state.apply(corrupted)
            assert vars(state) == before, (kind, event["operation"], field)
        state.apply(event)
    assert state.summary() == payload["summary"]
    assert state.coverage() == payload["guardian_coverage"]


@pytest.mark.parametrize("allocation", ["scope_notification", "scope_ordinal", "work_counter"])
def test_setup_container_allocation_failure_precedes_all_state_changes(allocation):
    payload = story("foreign")
    state = _Replay(GuardConfiguration.from_payload(payload["configuration"]), payload["lane_id"])
    target = "setup_operation_started" if allocation == "work_counter" else "descriptor_observed"
    for event in payload["transition_trace"]:
        if event["operation"] == target:
            break
        state.apply(event)
    before = deepcopy(vars(state))

    class FullNotifications(list):
        def copy(self):
            return self

        def append(self, value):
            raise MemoryError("scope notification allocation")

    class FullOrdinals(set):
        def copy(self):
            return self

        def add(self, value):
            raise MemoryError("scope ordinal allocation")

    class FullCounters(dict):
        def copy(self):
            return self

        def __setitem__(self, key, value):
            raise MemoryError("work counter allocation")

    if allocation == "scope_notification":
        state.notifications = FullNotifications(state.notifications)
    elif allocation == "scope_ordinal":
        state.dirty = FullOrdinals(state.dirty)
    else:
        state.work_counts = FullCounters(state.work_counts)
    with pytest.raises(MemoryError):
        state.apply(event)
    assert vars(state) == before
