from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tests.support.sacrificial_watchdog import (
    GuardedOperationChannel,
    SupervisorGuardState,
    WatchdogProtocolError,
    receive_message,
    send_message,
)

SESSION = "1234567890abcdef1234567890abcdef"


def request(operation="ARM", role="S", sequence=0):
    return dict(operation=operation, session=SESSION, role=role, sequence=sequence)


def test_deadline_precedes_ack_and_only_exact_setup_disarms():
    state = SupervisorGuardState(session=SESSION, operation_timeout=1.0)
    response = state.accept(request())
    assert state.deadline is not None
    assert response == request("ARMED")
    assert state.accept(request("COMPLETE")) == request("DISARMED")
    assert state.deadline is None
    state.accept(request(role="V", sequence=1))
    with pytest.raises(WatchdogProtocolError):
        state.accept(request("COMPLETE", role="V", sequence=1))
    assert state.deadline is not None
    assert state.failed


@pytest.mark.parametrize("mutation", ["session", "role", "sequence", "bool_sequence", "operation", "extra"])
def test_invalid_guard_message_cannot_arm(mutation):
    state = SupervisorGuardState(session=SESSION, operation_timeout=1.0)
    message = request()
    if mutation == "bool_sequence":
        message["sequence"] = False
    else:
        message[mutation] = "invalid"
    with pytest.raises(WatchdogProtocolError):
        state.accept(message)
    assert state.deadline is None
    assert state.failed


def test_duplicate_arm_and_expired_disarm_fail_closed():
    state = SupervisorGuardState(session=SESSION, operation_timeout=1.0)
    state.accept(request())
    with pytest.raises(WatchdogProtocolError):
        state.accept(request())
    assert state.failed and state.deadline is not None
    expired = SupervisorGuardState(session=SESSION, operation_timeout=0.001)
    expired.accept(request())
    time.sleep(0.005)
    with pytest.raises(WatchdogProtocolError, match="expired"):
        expired.accept(request("COMPLETE"))
    assert expired.failed and expired.deadline is not None


def test_boolean_sequence_ack_is_not_integer_authority():
    worker, server = socket.socketpair()

    def reply():
        message = receive_message(server, timeout=1.0)
        send_message(server, dict(message, operation="ARMED", sequence=False), timeout=1.0)

    thread = threading.Thread(target=reply)
    thread.start()
    try:
        channel = GuardedOperationChannel(worker, session=SESSION)
        with pytest.raises(WatchdogProtocolError, match="wrong, stale"):
            channel.arm("S")
    finally:
        thread.join(timeout=2)
        worker.close()
        server.close()
    assert not thread.is_alive()


@pytest.mark.skipif(os.name != "posix", reason="real inherited-FD supervisor is POSIX")
@pytest.mark.parametrize("role", ["S", "V", "E"])
@pytest.mark.parametrize(
    "fault", ["normal", "hang", "raise", "no_ack", "wrong_role", "wrong_sequence", "ack_eof"]
)
def test_real_supervisor_ack_drain_and_hard_exit(tmp_path, role, fault):
    root = Path(__file__).resolve().parents[3]
    read_fd, write_fd = os.pipe()
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                str(root / "tools/optical_sacrificial_test_supervisor.py"),
                str(tmp_path),
                role,
                fault,
                SESSION,
                str(read_fd),
            ],
            pass_fds=(read_fd,),
            capture_output=True,
            text=True,
            timeout=15,
        )
    finally:
        os.close(read_fd)
        os.close(write_fd)
    assert result.returncode == 0, result.stderr
    supervisor = json.loads((tmp_path / "supervisor.json").read_text())
    entries = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines()]
    operations = [item["operation"] for item in entries]
    assert operations[0] == "guardian_created"
    assert supervisor["no_restart"]
    if fault == "normal":
        assert supervisor["worker_return_code"] == 0, supervisor
        assert "owner_finalized" in operations
        assert entries[-1]["evidence"]["state"] == "RELEASED"
        assert supervisor["terminal_deadline_armed"] is (role != "S")
    else:
        assert "owner_finalized" not in operations
        assert "released" not in operations
        assert "inspection_completed" not in operations
        if fault == "hang":
            assert supervisor["killed"]
            assert supervisor["reason"] == "guard_deadline"
            assert "blocking_callable_entered" in operations
        else:
            assert supervisor["worker_return_code"] == 74, supervisor
            assert entries[-1]["evidence"]["state"] == "QUARANTINED"
        if fault in ("no_ack", "wrong_role", "wrong_sequence", "ack_eof"):
            assert "blocking_callable_entered" not in operations
        if role == "V":
            assert operations.count("launch_latched") == 2
            assert not any(item["role"] == "E" for item in supervisor["guard_history"])
