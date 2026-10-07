"""Mandatory POSIX actual supervisor/child source custody; no GPU/runtime."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from optical_worker.guarded_evidence import validate_custody_artifact
from optical_worker.native_sources import CATALOG_PATH
from tests.support.guarded_custody_process import SOURCE_FAULTS
from tests.unit.optics.test_native_sources import release as _release_fixture

release = _release_fixture
pytestmark = pytest.mark.skipif(os.name != "posix", reason="private inherited POSIX descriptors")
HARNESS = Path(__file__).resolve().parents[2] / "support/guarded_custody_process.py"


@pytest.mark.parametrize("fault", SOURCE_FAULTS)
@pytest.mark.parametrize("lane", ["warp", "cuda"])
def test_captured_source_real_supervision(release, tmp_path, fault, lane):
    root, revision = release
    (tmp_path / "request.json").write_text(
        json.dumps(
            {
                "root": str(root),
                "revision": revision,
                "lane": lane,
                "catalog_sha256": hashlib.sha256((root / CATALOG_PATH).read_bytes()).hexdigest(),
            }
        )
    )
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
        if fault == "source_parent_loss":
            deadline = time.monotonic() + 15
            while not (tmp_path / "body_entry").exists():
                assert process.poll() is None, "child failed before parent-loss injection"
                assert time.monotonic() < deadline, "missing real armed import entry"
                time.sleep(0.01)
            os.close(write_fd)
            write_fd = None
        output, error = process.communicate(timeout=25)
        assert process.returncode == 0, (output, error)
        report = json.loads(output)
        print(
            json.dumps(
                {
                    "fault": fault,
                    "lane": lane,
                    "supervisor_pid": process.pid,
                    "supervisor_returncode": process.returncode,
                    "worker_pid": report["worker_pid"],
                    "worker_returncode": report["worker_returncode"],
                }
            )
        )
        validate_custody_artifact(report["artifact"])
        supervisor = report["artifact"]["supervisor"]
        summary = supervisor["summary"]
        assert summary["complete"] and supervisor["trace"][-1]["operation"] == "process_exited"
        assert report["stderr"] == ""
        assert json.loads((tmp_path / "identity.json").read_text()) == {
            "pid": report["worker_pid"],
            "ppid": process.pid,
        }
        assert not (tmp_path / "atexit").exists()
        assert not (tmp_path / "destructor").exists()
        if fault.endswith("hang") or fault == "source_parent_loss":
            assert summary["deadline_expired"] and summary["kill_requested"]
            assert report["worker_returncode"] < 0
        else:
            assert report["worker_returncode"] == (0 if fault == "source_success" else 74)
            assert (tmp_path / "no_runtime").exists()
        assert report["parent_alive"] == (fault != "source_parent_loss")
        if fault == "source_parent_loss":
            assert summary["terminal_reason"] == "parent_lost"
        worker = report["artifact"]["worker"]
        if worker is not None:
            observed = worker["summary"]
            assert observed["entered_leaf_count"] == 0
            assert observed["emergency_sync_attempt_count"] == 0
            if worker["record_kind"] == "terminal":
                success = fault in {"source_success", "source_exit_hang"}
                assert observed["disposition"] == (
                    "cancelled_before_submission_exit" if success else "quarantined_process_exit"
                )
                assert observed["normal_complete_count"] == int(success)
                assert observed["terminal_arm_attempt_count"] == int(success)
    finally:
        if write_fd is not None:
            os.close(write_fd)
        if process.poll() is None:
            # Parent loss remains observable to the real supervisor first.
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=3)
