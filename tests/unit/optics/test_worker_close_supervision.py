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
from tests.support.guarded_custody_process import _observe_worker_client_exit


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
