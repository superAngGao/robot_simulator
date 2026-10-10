"""Post-ACK parent policy uses the existing supervisor, without native imports."""

import json
import socket
import subprocess
import sys
import threading
import time

import pytest

from optical_worker.wait_protocol import GuardConfiguration
from optical_worker.wait_supervision import GuardSupervisorCore
from tests.support.guarded_custody_process import _observe_worker_client_exit, _WorkerRequestObservation


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_delayed_supervisor_cannot_retire_a_late_response(offset):
    observation = _WorkerRequestObservation(123, GuardConfiguration("c" * 32, 1, 10, 100))
    observation.publish(0, "EXECUTE_PREPARED_SAMPLE", 1_000_000_000)
    deadline = observation.deadline_ns
    observation.response(0, deadline + offset)
    failed = observation.supervise(deadline + 50_000_000, core_failed=False)
    assert failed is (offset >= 0)
    assert observation.state == ("TERMINAL" if offset >= 0 else "RETIRED")
    if offset < 0:
        record = observation.wait_retired(0, threading.Event(), time.monotonic() + 1)
        assert record["observed_ns"] < record["deadline_ns"] < record["retired_ns"]
        assert record["worker_pid"] == 123 and record["session"] == "c" * 32


def test_expiry_wins_before_late_publication_and_never_reopens_requests():
    observation = _WorkerRequestObservation(123, GuardConfiguration("c" * 32, 1, 10, 100))
    observation.publish(0, "EXECUTE_PREPARED_SAMPLE", 1_000_000_000)
    deadline = observation.deadline_ns
    assert observation.supervise(deadline, core_failed=False)
    observation.response(0, deadline - 1)
    assert observation.supervise(deadline + 1, core_failed=False)
    assert observation.state == "TERMINAL" and observation.deadline_ns == deadline
    with pytest.raises(RuntimeError, match="not retired"):
        observation.publish(1, "EXECUTE_PREPARED_SAMPLE", deadline + 2)


def test_no_next_request_before_retirement_and_stale_response_cannot_clear_next():
    observation = _WorkerRequestObservation(123, GuardConfiguration("c" * 32, 1, 10, 100))
    observation.publish(0, "EXECUTE_PREPARED_SAMPLE", 1_000_000_000)
    observation.response(0, 1_000_000_001)
    with pytest.raises(RuntimeError, match="not retired"):
        observation.publish(1, "EXECUTE_PREPARED_SAMPLE", 1_000_000_002)
    assert not observation.supervise(1_000_000_003, core_failed=False)
    observation.publish(1, "EXECUTE_PREPARED_SAMPLE", 1_000_000_004)
    deadline = observation.deadline_ns
    observation.response(0, 1_000_000_005)
    assert observation.supervise(1_000_000_006, core_failed=False)
    assert observation.sequence == 1 and observation.deadline_ns == deadline


@pytest.mark.parametrize("failure", ["partial", "eof", "send", "status", "core"])
def test_request_failure_is_permanent_and_retains_first_error(failure):
    observation = _WorkerRequestObservation(123, GuardConfiguration("c" * 32, 1, 10, 100))
    observation.publish(0, "EXECUTE_PREPARED_SAMPLE", 1_000_000_000)
    if failure == "core":
        observation.response(0, 1_000_000_001)
    else:
        observation.failed(failure)
        observation.failed("secondary")
    assert observation.supervise(1_000_000_002, core_failed=failure == "core")
    assert observation.failure == ("core_failed" if failure == "core" else failure)
    with pytest.raises(RuntimeError, match="did not retire"):
        observation.wait_retired(0, threading.Event(), time.monotonic() + 1)


def test_client_thread_cannot_retire_itself_and_wait_releases_handoff_lock():
    observation = _WorkerRequestObservation(123, GuardConfiguration("c" * 32, 1, 10, 100))
    outcome, ready = [], threading.Event()

    def client():
        observation.publish(0, "EXECUTE_PREPARED_SAMPLE", time.monotonic_ns())
        observation.response(0, time.monotonic_ns())
        try:
            observation.supervise(time.monotonic_ns(), core_failed=False)
        except RuntimeError as error:
            outcome.append(str(error))
        ready.set()
        outcome.append(observation.wait_retired(0, threading.Event(), time.monotonic() + 2))

    thread = threading.Thread(target=client)
    thread.start()
    try:
        assert ready.wait(timeout=1)
        assert not observation.supervise(time.monotonic_ns(), core_failed=False)
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert "original supervisor" in outcome[0]
        assert outcome[1]["sequence"] == 0
    finally:
        observation.failed("test teardown")
        observation.supervise(time.monotonic_ns(), core_failed=True)
        thread.join(timeout=2)


def test_close_exit_deadline_is_not_renewed_and_transport_is_closed():
    now = [1_000_000_000]
    core = GuardSupervisorCore(GuardConfiguration("c" * 32, 1, 10, 100), clock=lambda: now[0])
    local, remote = socket.socketpair()
    stop, outcome = threading.Event(), {"close_ack_observed_ns": now[0], "completed": True}
    try:
        now[0] += 99_999_999
        _observe_worker_client_exit(core, local, stop, outcome, now[0])
        assert core.state == "IDLE" and not stop.is_set()
        now[0] += 1
        _observe_worker_client_exit(core, local, stop, outcome, now[0])
        assert core.state == "TERMINATING" and stop.is_set()
        assert remote.recv(1) == b""
        deadline = core.effective_deadline_ns
        assert deadline == now[0] + 100_000_000
        now[0] += 50_000_000
        _observe_worker_client_exit(core, local, stop, outcome, now[0])
        assert core.effective_deadline_ns == deadline
        assert outcome["completed"] is True  # ACK is not an observed exit.
        assert "worker_exit_observed_ns" not in outcome
        assert core.evidence().to_payload()["summary"]["arm_count"] == 0
    finally:
        local.close()
        remote.close()


@pytest.mark.parametrize("role", ["setup", "resolve", "emergency"])
def test_client_failure_preserves_original_armed_deadline_and_error(role):
    now = [1_000_000_000]
    config = GuardConfiguration("c" * 32, 1, 10, 100)
    core = GuardSupervisorCore(config, clock=lambda: now[0])
    response = core.accept(config.message("ARM", 0, role))
    core.acknowledge_written(response)
    original_deadline = core.effective_deadline_ns
    now[0] += 50_000_000
    outcome = {"completed": False, "error_type": "EOFError", "error": "original response EOF"}
    local, remote = socket.socketpair()
    try:
        _observe_worker_client_exit(core, local, threading.Event(), outcome, now[0])
        assert core.effective_deadline_ns == original_deadline
        assert outcome["error"] == "original response EOF"
        assert remote.recv(1) == b""
        summary = core.evidence().to_payload()["summary"]
        assert summary["arm_count"] == 1 and summary["normal_complete_count"] == 0
    finally:
        local.close()
        remote.close()


@pytest.mark.parametrize("close_fails", [False, True])
def test_transport_failure_cannot_prevent_supervisor_terminal_entry(close_fails):
    core = GuardSupervisorCore(GuardConfiguration("c" * 32, 1, 10, 100))
    calls = []

    class BrokenTransport:
        def shutdown(self, how):
            calls.append(("shutdown", how))
            raise OSError("original shutdown failure")

        def close(self):
            calls.append(("close",))
            if close_fails:
                raise OSError("secondary close failure")

    outcome = {"completed": False, "error": "original request failure"}
    _observe_worker_client_exit(core, BrokenTransport(), threading.Event(), outcome, time.monotonic_ns())
    assert core.state == "TERMINATING"
    assert outcome["error"] == "original request failure"
    assert calls == [("shutdown", socket.SHUT_RDWR), ("close",)]
    assert ("terminal_close_error" in outcome) is close_fails
    assert outcome["terminal_shutdown_error"] == "OSError"


@pytest.mark.parametrize("disposition", ["exit", "hang"])
def test_actual_child_exit_or_hang_after_ack_is_observed_and_reaped(disposition, tmp_path):
    # This proves OS disposal for the actual parent policy/core. The child's
    # ACK is modeled; the native integrated CLOSE/fault matrix is still required.
    config = GuardConfiguration("c" * 32, 1, 30, 100)
    core = GuardSupervisorCore(config)
    program = "import time; print('ACK', flush=True); "
    program += "time.sleep(30)" if disposition == "hang" else "pass"
    child = subprocess.Popen([sys.executable, "-I", "-B", "-c", program],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    local, remote = socket.socketpair()
    outcome, stop = {}, threading.Event()
    try:
        assert child.stdout.readline() == "ACK\n"
        outcome.update(close_ack_observed_ns=time.monotonic_ns(), completed=True)
        test_deadline = time.monotonic() + 5
        while child.poll() is None:
            assert time.monotonic() < test_deadline, "original supervisor did not dispose of child"
            _observe_worker_client_exit(core, local, stop, outcome, time.monotonic_ns())
            core.tick()
            if core.kill_required:
                core.record_kill_request()
                child.kill()
            time.sleep(0.005)
        core.observe_exit(child.returncode)
        outcome["worker_exit_observed_ns"] = time.monotonic_ns()
        child.wait(timeout=1)
        outcome["worker_reaped_ns"] = time.monotonic_ns()
        assert child.stderr.read() == ""
        evidence = core.evidence(require_final=True).to_payload()
        (tmp_path / "closure.json").write_text(json.dumps({"worker_pid": child.pid,
            "returncode": child.returncode, "supervisor": evidence, "parent": outcome}))
        summary = evidence["summary"]
        assert summary["complete"] and summary["arm_count"] == 0
        assert summary["kill_requested"] is (disposition == "hang")
        assert ("close_exit_wait_expired_ns" in outcome) is (disposition == "hang")
        assert (outcome["worker_reaped_ns"] >= outcome["worker_exit_observed_ns"]
                >= outcome["close_ack_observed_ns"])
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=2)
        child.stdout.close()
        child.stderr.close()
        local.close()
        remote.close()
