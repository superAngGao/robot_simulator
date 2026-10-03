"""Real child/supervisor C2d1 acceptance, mandatory on POSIX (CPU fixtures)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from optical_worker.guarded_evidence import validate_custody_artifact

pytestmark = pytest.mark.skipif(os.name != "posix", reason="private inherited POSIX descriptors")
HARNESS = Path(__file__).resolve().parents[2] / "support" / "guarded_custody_process.py"
FAULTS = (
    "cpu_success",
    "cpu_cleanup_hang",
    "cpu_exit_hang",
    "normal_success",
    "normal_leaf_raise",
    "inner_abort",
    "pre_leaf_raise",
    "foreign",
    "unknown",
    "sync_raise",
    "sync_hang",
    "cleanup_raise_before",
    "cleanup_raise_after",
    "normal_no_ack",
    "normal_wrong_ack",
    "normal_ack_eof",
    "normal_stalled_disarm",
    "terminal_no_ack",
    "terminal_wrong_ack",
    "terminal_ack_eof",
    "target_cleanup_hang",
    "target_exit_hang",
)


@pytest.mark.parametrize("fault", FAULTS)
def test_real_guarded_custody_process(fault, tmp_path):
    read_fd, write_fd = os.pipe()
    process = subprocess.Popen(
        [sys.executable, "-I", "-B", str(HARNESS), "supervisor", fault, str(read_fd), str(tmp_path)],
        pass_fds=(read_fd,),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    os.close(read_fd)
    try:
        output, error = process.communicate(timeout=16)
        assert process.returncode == 0, (output, error)
        report = json.loads(output)
        print(
            json.dumps(
                {
                    "fault": fault,
                    "supervisor_pid": process.pid,
                    "supervisor_returncode": process.returncode,
                    "worker_pid": report["worker_pid"],
                    "worker_returncode": report["worker_returncode"],
                },
                sort_keys=True,
            )
        )
        validate_custody_artifact(report["artifact"])
        summary = report["artifact"]["supervisor"]["summary"]
        trace = report["artifact"]["supervisor"]["trace"]
        operations = [event["operation"] for event in trace]
        assert summary["complete"] and operations[-1] == "process_exited"
        assert report["parent_alive"] and report["stderr"] == ""
        assert not (tmp_path / "atexit").exists()
        identity = json.loads((tmp_path / "identity.json").read_text(encoding="utf-8"))
        assert identity == {"pid": report["worker_pid"], "ppid": process.pid}
        if fault.endswith("hang"):
            assert summary["deadline_expired"] and summary["kill_requested"]
            assert report["worker_returncode"] < 0
            assert operations.index("kill_requested") < operations.index("process_exited")
        if fault.startswith(("cpu_", "terminal_")) or fault.startswith("cleanup_"):
            assert not (tmp_path / "sync").exists()
            assert not (tmp_path / "leaf").exists()
        if fault in {"normal_leaf_raise", "inner_abort", "pre_leaf_raise", "foreign", "unknown"}:
            assert not (tmp_path / "sync").exists()
            assert not (tmp_path / "destructor").exists()
            assert summary["normal_complete_count"] == 0
            assert summary["arm_count"] == 1
        if fault.startswith("terminal_") or fault in {
            "normal_no_ack",
            "normal_wrong_ack",
            "normal_ack_eof",
            "normal_stalled_disarm",
            "sync_raise",
        }:
            assert not (tmp_path / "destructor").exists()
        worker = report["artifact"]["worker"]
        if worker is not None and worker["record_kind"] == "terminal":
            expected = {
                "cpu_success": "cancelled_before_submission_exit",
                "cpu_exit_hang": "cancelled_before_submission_exit",
                "normal_success": "aborted_synced_exit",
                "target_exit_hang": "aborted_synced_exit",
                "foreign": "multi_or_unknown_device_quarantine_exit",
                "unknown": "multi_or_unknown_device_quarantine_exit",
                "cleanup_raise_before": "cleanup_failed_exit",
                "cleanup_raise_after": "cleanup_failed_exit",
            }.get(fault, "quarantined_process_exit")
            assert worker["summary"]["disposition"] == expected
        if fault.startswith("cleanup_raise"):
            assert worker["summary"]["custody"] == "remaining_quarantine"
            assert (tmp_path / "destructor").exists() == (fault == "cleanup_raise_after")
    finally:
        os.close(write_fd)
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
