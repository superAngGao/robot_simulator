"""Real captured dependencies and a2 helpers under fixed fault injection.

This is one required C2d3 matrix. Ambient/dual-device/ACK/EOF and Go2 proofs
remain separate. No renderer, final readiness, result or performance claim.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from optical_worker.native_sources import CATALOG_PATH
from optical_worker.runtime_custody import (
    RuntimeCustodyRecord, validate_runtime_custody_attachment, validate_runtime_custody_selection,
)
from optical_worker.runtime_environment import RuntimeEnvironmentContract
from tests.support.native_runtime_process import RUNTIME_FAULTS
from tests.support.runtime_custody_launch import capture_runtime_custody_inputs

ROOT = Path(__file__).resolve().parents[2]
PROFILES = {
    "cuda": ("183_cuda_stream_author_v1.json", "b4e47505a751377d48336beba5508cdbe2a2adcccdd10ae0e463e0767a537a47", 120000),
    "warp": ("183_warp_synthetic_author_v5.json", "0783cc820dcb738b8ec82586e2d14a12a7b9b81989e67d1ba9f86ce1ea3ed1b7", 30000),
}
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.name != "posix" or os.environ.get("ROBOT_SIM_TEST_C2D3_FAULTS") != "1",
    reason="explicit matching 183 deployment required; skipped faults are not validation")]


@pytest.mark.parametrize("lane", ["warp", "cuda"])
@pytest.mark.parametrize("fault", RUNTIME_FAULTS[3:])
def test_actual_runtime_fault_retains_original_guard_and_reaps(tmp_path, lane, fault):
    filename, digest, deadline = PROFILES[lane]
    environment_path = ROOT / "tests/data/runtime_environments" / filename
    environment = RuntimeEnvironmentContract.from_payload(json.loads(environment_path.read_bytes()))
    assert environment.sha256 == digest
    environment.verify_files(expected_sha256=digest)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    request = {
        "root": str(ROOT), "revision": revision, "lane": lane,
        "catalog_sha256": hashlib.sha256((ROOT / CATALOG_PATH).read_bytes()).hexdigest(),
        "environment_path": str(environment_path), "environment_sha256": digest,
        "case_id": "asymmetric_synthetic_7x3_v1", "go2_model_dir": None,
        "guard_configuration": {"protocol_version": "q54.closed-worker.guarded-wait.v1",
                                "session": "9" * 32, "device_ordinal": 1,
                                "ack_timeout_ms": 1000, "operation_timeout_ms": deadline},
    }
    inputs = capture_runtime_custody_inputs(request)
    (tmp_path / "request.json").write_text(json.dumps(request), encoding="utf-8")
    (tmp_path / "runtime_custody_inputs.json").write_text(json.dumps(inputs), encoding="utf-8")
    read_fd, write_fd = os.pipe()
    supervisor = subprocess.Popen([
        sys.executable, "-I", "-B", str(ROOT / "tests/support/guarded_custody_process.py"),
        "supervisor", fault, str(read_fd), str(tmp_path),
    ], pass_fds=(read_fd,), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    os.close(read_fd)
    try:
        output, error = supervisor.communicate(timeout=10 * deadline / 1000 + 45)
        (tmp_path / "supervisor_stdout.json").write_text(output, encoding="utf-8")
        (tmp_path / "supervisor_stderr.txt").write_text(error, encoding="utf-8")
        assert supervisor.returncode == 0, (output, error)
        result = json.loads(output)
        artifact = result["artifact"]
        (tmp_path / "custody_artifact.json").write_text(json.dumps(artifact), encoding="utf-8")
        assert result["stderr"] == error == ""
        assert result["parent_alive"] is True
        assert not (tmp_path / "ordinary_atexit.json").exists()
        marker = json.loads((tmp_path / "fault_entered.json").read_bytes())
        assert marker["fault"] == fault
        raw = (tmp_path / "runtime_custody.json").read_bytes()
        selection = json.loads((tmp_path / "runtime_custody_selection.json").read_bytes())
        validate_runtime_custody_selection(selection, raw)
        record = RuntimeCustodyRecord.from_payload(json.loads(raw), environment=environment, trusted_launch_inputs=inputs)
        assert validate_runtime_custody_attachment(record, environment, inputs, artifact) == "worker_prefix_matched"
        assert record.to_payload()["summary"]["final_execution_readiness_issued"] is False
        control, worker = artifact["supervisor"]["summary"], artifact["worker"]["summary"]
        assert control["complete"] and control["state"] == "EXITED"
        if fault.endswith("hang"):
            assert result["worker_returncode"] == -9
            assert control["deadline_expired"] and control["kill_requested"]
            assert control["terminal_reason"] == "deadline_expired"
        else:
            assert result["worker_returncode"] == 74
            assert not control["kill_requested"] and not control["deadline_expired"]
            assert worker["custody"] == "full_quarantine"
        if not fault.startswith("runtime_terminal_"):
            assert worker["emergency_sync_attempt_count"] == worker["terminal_arm_attempt_count"] == 0
            assert worker["release_status"] == "not_authorized"
            after = artifact["worker"]["transition_trace"][marker["lifecycle_event_count"]:]
            assert not {"normal_arm_started", "normal_complete_started", "terminal_arm_started",
                        "emergency_sync_started", "ordinary_cleanup_started"} & {row["operation"] for row in after}
            assert not (tmp_path / "input_validation_report.json").exists()
        print(json.dumps({"fault": fault, "lane": lane, "worker_pid": result["worker_pid"],
                          "supervisor_pid": supervisor.pid, "worker_returncode": result["worker_returncode"],
                          "selection": selection["selection"]}))
    finally:
        os.close(write_fd)
        if supervisor.poll() is None:
            # Real parent EOF goes through the original supervisor deadline.
            supervisor.communicate(timeout=deadline / 1000 + 15)
