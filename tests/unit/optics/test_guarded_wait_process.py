"""Real process/watchdog tests; Linux execution is mandatory for C4c2b."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from optical_worker.wait_supervision import GuardSupervisorEvidence

pytestmark = pytest.mark.skipif(os.name != "posix", reason="private inherited POSIX descriptors")
HARNESS = Path(__file__).resolve().parents[2] / "support" / "guarded_wait_process.py"


@pytest.mark.parametrize(
    "fault",
    [
        "success",
        "normal_hang",
        "emergency_hang",
        "no_ack",
        "wrong_ack",
        "ack_eof",
        "send_failure",
        "stalled_disarm",
        "parent_loss",
        "trickle",
    ],
)
def test_real_supervisor_child_fault_matrix(fault, tmp_path):
    read_fd, write_fd = os.pipe()
    process = subprocess.Popen(
        [sys.executable, "-I", str(HARNESS), "supervisor", fault, str(read_fd), str(tmp_path)],
        pass_fds=(read_fd,),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    os.close(read_fd)
    try:
        if fault in {"parent_loss", "trickle"}:
            stop = time.monotonic() + 5
            while not (tmp_path / "leaf").exists():
                assert process.poll() is None
                assert time.monotonic() < stop
                time.sleep(0.01)
            os.close(write_fd)
            write_fd = None
        output, error = process.communicate(timeout=12)
        assert process.returncode == 0, (output, error)
        report = json.loads(output)
        evidence = GuardSupervisorEvidence.from_payload(report["evidence"], require_final=True).to_payload()
        summary = evidence["summary"]
        operations = [event["operation"] for event in evidence["trace"]]
        assert report["stderr"] == ""
        assert summary["complete"] and summary["state"] == "EXITED"
        assert operations[-1] == "process_exited"
        assert not (tmp_path / "destructor").exists()
        assert not (tmp_path / "atexit").exists()
        if fault == "success":
            assert summary["normal_complete_count"] == 2
            assert summary["observed_returncode"] == 0
            assert not summary["kill_requested"]
            assert (tmp_path / "completed").exists()
        elif fault in {"normal_hang", "emergency_hang", "stalled_disarm", "parent_loss", "trickle"}:
            assert summary["kill_requested"] and summary["deadline_expired"]
            assert summary["observed_returncode"] < 0
            assert operations.index("kill_requested") < operations.index("process_exited")
            assert summary["normal_complete_count"] == 0
            assert (tmp_path / "leaf").exists()
            if fault == "stalled_disarm":
                assert report["fault_injected"]
                assert "normal_complete_received" in operations
                assert [
                    event["details"]["operation"]
                    for event in evidence["trace"]
                    if event["operation"] == "ack_written"
                ] == ["ARMED"]
            if fault in {"parent_loss", "trickle"}:
                assert summary["terminal_reason"] == "parent_lost"
        else:
            assert not (tmp_path / "leaf").exists()
            assert summary["normal_complete_count"] == 0
            # no-ACK can race the supervisor's ACK deadline; both outcomes
            # remain process-terminal without entering the blocking leaf.
            assert summary["observed_returncode"] == 74 or (
                fault == "no_ack" and summary["observed_returncode"] < 0 and summary["kill_requested"]
            )
    finally:
        if write_fd is not None:
            os.close(write_fd)
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
