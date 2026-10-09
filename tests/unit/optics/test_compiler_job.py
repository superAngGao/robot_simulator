"""CPU production-core/job checks; these do not claim OS reaping proof."""

import copy

import pytest

from optical_worker.compilation_contract import CUDA_FOUNDATION_PHASES, cuda_foundation_phase
from optical_worker.compiler_job import CompilerJobRecord, _CompilerObligation
from optical_worker.protocol import canonical_json_bytes
from optical_worker.wait_protocol import GuardConfiguration, GuardProtocolError, I64_MAX
from optical_worker.wait_supervision import GuardSupervisorCore


SUPERVISOR = {"pid": 100, "starttime_ticks": 1, "sid": 50}
WORKER = {"pid": 200, "starttime_ticks": 2, "sid": 200}
CHILD = {"pid": 201, "starttime_ticks": 3, "sid": 200}


def make_job():
    clock = [100]
    configuration = GuardConfiguration("a" * 32, 1, 1, 120_000)
    core = GuardSupervisorCore(configuration, clock=lambda: clock[0])
    job = _CompilerObligation(core, SUPERVISOR)
    job.subreaper_ready()
    job.spawned(WORKER)
    return clock, core, job


def arm(core, job, sequence=0, *, ack=True):
    role = "emergency" if sequence == 9 else "setup"
    response = core.accept(core.configuration.message("ARM", sequence, role))
    job.sync_core()
    if ack:
        core.acknowledge_written(response)
        job.sync_core()


def complete(core, job, sequence):
    response = core.accept(core.configuration.message("COMPLETE", sequence, "setup"))
    job.sync_core()
    core.acknowledge_written(response)
    job.sync_core()


@pytest.mark.parametrize("pending_ack", [False, True])
def test_leader_first_exit_preserves_earliest_deadline_and_core_bytes(pending_ack):
    clock, core, job = make_job()
    arm(core, job, ack=not pending_ack)
    deadline = core.effective_deadline_ns
    source = "ack" if pending_ack else "operation"
    job.adopted(CHILD)
    job.reaped(WORKER, 0)
    exited = canonical_json_bytes(core.evidence().to_payload())
    assert core.state == "EXITED" and core.effective_deadline_ns is None
    clock[0] = deadline - 1
    job.tick()
    assert not job.replay.expired
    clock[0] += 1
    job.tick()
    assert job.replay.deadline == deadline and job.replay.deadline_source == source
    job.signal_attempted(CHILD)
    with pytest.raises(GuardProtocolError):
        job.signal_attempted(CHILD)
    with pytest.raises(GuardProtocolError):
        job.signal_attempted(WORKER)
    job.reaped(CHILD, 9)
    job.no_children()
    job.pipes_eof()
    job.close()
    assert canonical_json_bytes(core.evidence().to_payload()) == exited
    record = job.record(require_closed=True)
    assert CompilerJobRecord.from_bytes(record.to_bytes(), guard_evidence=core.evidence(), require_closed=True) == record
    assert record.to_payload()["summary"] == {
        "state": "CLOSED", "failed": True, "deadline_ns": deadline,
        "deadline_expired": True, "worker_reaped": True, "no_children": True, "pipes_eof": True,
    }


def test_live_ack_expiry_preserves_ack_source_before_core_clears_pending():
    clock, core, job = make_job()
    arm(core, job, ack=False)
    deadline = core.effective_deadline_ns
    clock[0] = deadline
    job.tick()
    assert job.replay.deadline_source == "ack" and job.replay.deadline == deadline
    assert core.kill_required
    core.record_kill_request()
    job.sync_core()
    job.signal_attempted(WORKER)
    job.reaped(WORKER, 9)
    job.no_children()
    job.pipes_eof()
    job.close()
    assert job.record(require_closed=True).to_payload()["summary"]["failed"]


def test_healthy_wire_zero_through_nine_requires_compile_quiescence():
    _, core, job = make_job()
    for sequence, phase in enumerate(CUDA_FOUNDATION_PHASES):
        assert cuda_foundation_phase(sequence, "emergency" if sequence == 9 else "setup") == phase
        arm(core, job, sequence)
        if sequence == 9:
            break
        if sequence >= 6:
            job.quiescent(sequence)
        complete(core, job, sequence)
    job.reaped(WORKER, 0)
    job.no_children()
    with pytest.raises(GuardProtocolError):
        job.close()
    job.pipes_eof()
    job.close()
    record = job.record(require_closed=True)
    assert record.to_payload()["summary"]["failed"] is False
    payload = record.to_payload()
    payload["events"] = [e for e in payload["events"] if e["operation"] != "compile_quiescent"]
    for index, event in enumerate(payload["events"]):
        event["index"] = index
    with pytest.raises(GuardProtocolError, match="quiescence"):
        CompilerJobRecord.from_payload(payload, guard_evidence=core.evidence())


def test_residual_child_prevents_compile_quiescence():
    _, core, job = make_job()
    for sequence in range(6):
        arm(core, job, sequence)
        complete(core, job, sequence)
    arm(core, job, 6)
    job.adopted(CHILD)
    with pytest.raises(GuardProtocolError, match="quiescence"):
        job.quiescent(6)
    assert core.state == "ARMED_NORMAL"
    assert not job.replay.quiescent


@pytest.mark.parametrize("bad_clock", [50, True, I64_MAX + 1])
def test_post_exit_clock_failure_never_renews_deadline(bad_clock):
    clock, core, job = make_job()
    arm(core, job)
    job.adopted(CHILD)
    job.reaped(WORKER, 0)
    before = canonical_json_bytes(core.evidence().to_payload())
    clock[0] = bad_clock
    job.tick()
    assert job.replay.expired and job.replay.deadline <= 100
    job.signal_attempted(CHILD)
    assert canonical_json_bytes(core.evidence().to_payload()) == before
    assert job.record().to_payload()["summary"]["failed"]


def test_newly_adopted_child_after_expiry_uses_same_obligation():
    clock, core, job = make_job()
    arm(core, job)
    deadline = core.effective_deadline_ns
    job.reaped(WORKER, 74 << 8)
    clock[0] = deadline
    job.tick()
    job.adopted(CHILD)
    job.signal_attempted(CHILD)
    job.signal_failed(CHILD, "permission")
    job.tick()
    assert job.replay.deadline == deadline and job.replay.state == "DRAINING"
    with pytest.raises(GuardProtocolError):
        job.no_children()
    with pytest.raises(GuardProtocolError):
        job.close()
    record = job.record()
    assert len([e for e in record.to_payload()["events"] if e["operation"] == "deadline_expired"]) == 1


def test_idle_exit_freezes_single_terminal_grace():
    clock, core, job = make_job()
    clock[0] = 200
    job.reaped(WORKER, 0)
    expected = 200 + 120_000_000_000
    assert job.replay.deadline == expected
    clock[0] = 300
    job.drain("worker_exit", now=300)
    assert job.replay.deadline == expected
    job.record()


@pytest.mark.parametrize("change", ["deadline", "sequence_bool", "identity", "status", "session", "summary"])
def test_canonical_replay_rejects_mutated_observations(change):
    _, core, job = make_job()
    arm(core, job)
    job.reaped(WORKER, 0)
    p = copy.deepcopy(job.record().to_payload())
    if change == "deadline":
        next(e for e in p["events"] if e["operation"] == "draining")["details"]["deadline_ns"] += 1
    elif change == "sequence_bool":
        next(e for e in p["events"] if e["operation"] == "deadline_snapshot"
             and e["details"]["sequence"] == 0)["details"]["sequence"] = False
    elif change == "identity":
        next(e for e in p["events"] if e["operation"] == "child_reaped")["details"]["identity"]["pid"] += 1
    elif change == "status":
        next(e for e in p["events"] if e["operation"] == "child_reaped")["details"]["wait_status"] = 0x137f
    elif change == "session":
        p["session"] = "b" * 32
    else:
        p["summary"]["failed"] = 0
    with pytest.raises(GuardProtocolError):
        CompilerJobRecord.from_payload(p, guard_evidence=core.evidence())
