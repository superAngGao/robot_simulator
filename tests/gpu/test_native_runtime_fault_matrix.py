"""Real captured dependencies and a2 helpers under fixed fault injection.

This is one required C2d3 matrix. Ambient/dual-device/ACK/EOF and Go2 proofs
remain separate. No renderer, final readiness, result or performance claim.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from optical_worker.native_sources import CATALOG_PATH
from optical_worker.runtime_custody import (
    RuntimeCustodyRecord, validate_runtime_custody_attachment, validate_runtime_custody_selection,
)
from optical_worker.runtime_environment import RuntimeEnvironmentContract
from tests.support.native_runtime_process import (
    RUNTIME_ACK_FAULTS, RUNTIME_NATIVE_FAULTS, RUNTIME_PARENT_FAULTS, RUNTIME_SCOPE_FAULTS,
    RUNTIME_GLOBAL_FAULTS, RUNTIME_CUDA_ONLY_FAULTS,
)
from tests.support.runtime_custody_launch import capture_runtime_custody_inputs

ROOT = Path(__file__).resolve().parents[2]
PROFILES = {
    "cuda": ("183_cuda_stream_author_v1.json", "b4e47505a751377d48336beba5508cdbe2a2adcccdd10ae0e463e0767a537a47", 120000),
    "warp": ("183_warp_synthetic_author_v5.json", "0783cc820dcb738b8ec82586e2d14a12a7b9b81989e67d1ba9f86ce1ea3ed1b7", 30000),
}
GO2_PROFILES = {
    "cuda": ("183_cuda_go2_author_v1.json", "79c3580baff1ef3a019c97a07177558808918b7e133caa430239e25b82853919", 120000),
    "warp": ("183_warp_go2_author_v3.json", "88d295b90bc4047b2f2cd4c3d0e48c33ead1a1050919b47d526b4576159146e8", 120000),
}
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.name != "posix" or os.environ.get("ROBOT_SIM_TEST_C2D3_FAULTS") != "1",
    reason="explicit matching 183 deployment required; skipped faults are not validation")]


@pytest.mark.parametrize("fault,lane", [
    (fault, lane) for fault in (RUNTIME_NATIVE_FAULTS[3:] + RUNTIME_ACK_FAULTS + RUNTIME_PARENT_FAULTS
                               + RUNTIME_SCOPE_FAULTS + RUNTIME_GLOBAL_FAULTS) for lane in ("warp", "cuda")
] + [(fault, "cuda") for fault in RUNTIME_CUDA_ONLY_FAULTS])
def test_actual_runtime_fault_retains_original_guard_and_reaps(tmp_path, lane, fault):
    go2 = fault in RUNTIME_GLOBAL_FAULTS
    filename, digest, deadline = (GO2_PROFILES if go2 else PROFILES)[lane]
    environment_path = ROOT / "tests/data/runtime_environments" / filename
    environment = RuntimeEnvironmentContract.from_payload(json.loads(environment_path.read_bytes()))
    assert environment.sha256 == digest
    environment.verify_files(expected_sha256=digest)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    request = {
        "root": str(ROOT), "revision": revision, "lane": lane,
        "catalog_sha256": hashlib.sha256((ROOT / CATALOG_PATH).read_bytes()).hexdigest(),
        "environment_path": str(environment_path), "environment_sha256": digest,
        "case_id": "go2_static_960x640_v1" if go2 else "asymmetric_synthetic_7x3_v1",
        "go2_model_dir": "/data2/shared/ang.gao/c2d2_frozen_assets/mujoco_menagerie/unitree_go2" if go2 else None,
        "guard_configuration": {"protocol_version": "q54.closed-worker.guarded-wait.v1",
                                "session": "9" * 32, "device_ordinal": 1,
                                "ack_timeout_ms": 1000, "operation_timeout_ms": deadline},
    }
    inputs = capture_runtime_custody_inputs(request)
    if fault in RUNTIME_SCOPE_FAULTS or fault == "runtime_c1_stream_only_copy":
        mapping = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,driver_version",
                                           "--format=csv,noheader"], text=True, timeout=15)
        assert "GPU-a4b256c9-c56a-0ef5-19fa-482196cc3274" in mapping
        assert "GPU-3a3112ff-837c-3b87-38d1-cb203d656589" in mapping
        (tmp_path / "gpu_mapping.txt").write_text(mapping, encoding="utf-8")
    (tmp_path / "request.json").write_text(json.dumps(request), encoding="utf-8")
    (tmp_path / "runtime_custody_inputs.json").write_text(json.dumps(inputs), encoding="utf-8")
    read_fd, write_fd = os.pipe()
    supervisor = subprocess.Popen([
        sys.executable, "-I", "-B", str(ROOT / "tests/support/guarded_custody_process.py"),
        "supervisor", fault, str(read_fd), str(tmp_path),
    ], pass_fds=(read_fd,), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    os.close(read_fd)
    try:
        if fault in RUNTIME_PARENT_FAULTS:
            # Close the actual inherited liveness pipe only after the original
            # controller's real ARMED response admits the selected phase.
            until = time.monotonic() + 10 * deadline / 1000 + 30
            while not (tmp_path / "fault_entered.json").exists():
                assert supervisor.poll() is None, "worker exited before the selected armed phase"
                assert time.monotonic() < until, "selected phase marker did not arrive"
                time.sleep(0.01)
            os.close(write_fd)
            write_fd = None
        output, error = supervisor.communicate(timeout=10 * deadline / 1000 + 45)
        (tmp_path / "supervisor_stdout.json").write_text(output, encoding="utf-8")
        (tmp_path / "supervisor_stderr.txt").write_text(error, encoding="utf-8")
        assert supervisor.returncode == 0, (output, error)
        result = json.loads(output)
        artifact = result["artifact"]
        (tmp_path / "custody_artifact.json").write_text(json.dumps(artifact), encoding="utf-8")
        assert result["stderr"] == error == ""
        assert result["parent_alive"] is (fault not in RUNTIME_PARENT_FAULTS)
        assert not (tmp_path / "ordinary_atexit.json").exists()
        marker = json.loads((tmp_path / ("protocol_fault_entered.json" if fault in RUNTIME_ACK_FAULTS
                                        else "fault_entered.json")).read_bytes())
        assert marker["fault"] == fault
        raw = (tmp_path / "runtime_custody.json").read_bytes()
        selection = json.loads((tmp_path / "runtime_custody_selection.json").read_bytes())
        validate_runtime_custody_selection(selection, raw)
        record = RuntimeCustodyRecord.from_payload(json.loads(raw), environment=environment, trusted_launch_inputs=inputs)
        assert validate_runtime_custody_attachment(record, environment, inputs, artifact) == "worker_prefix_matched"
        assert record.to_payload()["summary"]["final_execution_readiness_issued"] is False
        control, worker = artifact["supervisor"]["summary"], artifact["worker"]["summary"]
        assert control["complete"] and control["state"] == "EXITED"
        if fault in RUNTIME_SCOPE_FAULTS:
            scope = json.loads((tmp_path / "foreign_work.json").read_bytes())
            assert scope["configured_target"] == 1 and scope["actual_stream_device"] == 0
            assert scope["foreign_context_present"] is True
            assert scope["guard"] == {"sequence": 3, "role": "setup"}
            assert worker["unknown_device_scope"] is True
            if fault == "runtime_foreign_stream_return":
                assert 0 in worker["possible_dirty_device_ordinals"]
        if fault == "runtime_c1_stream_only_copy":
            scope = json.loads((tmp_path / "c1_scope.json").read_bytes())
            assert scope["before"]["logical"] == scope["after"]["logical"] == 0
            assert scope["before"]["contexts"][0] is False and scope["after"]["contexts"][0] is True
            assert worker["unknown_device_scope"] is True
        if fault in RUNTIME_ACK_FAULTS:
            assert result["injected"] is True
            target_sequence = 6 if fault.startswith("runtime_terminal_") else 3
            target_operation = "DISARMED" if fault.endswith("stalled_disarm") else "ARMED"
            assert marker["pending"]["sequence"] == target_sequence
            assert marker["pending"]["operation"] == target_operation
            assert not [e for e in artifact["supervisor"]["trace"] if e["operation"] == "ack_written"
                        and e["details"]["sequence"] == target_sequence
                        and e["details"]["operation"] == target_operation]
            assert result["worker_returncode"] in (74, -9)
            if result["worker_returncode"] == -9:
                assert control["kill_requested"] and control["deadline_expired"]
            assert worker["emergency_sync_attempt_count"] == 0
            assert worker["release_status"] == "not_authorized"
            assert not (tmp_path / "terminal_sync_entered.json").exists()
        elif fault in RUNTIME_PARENT_FAULTS:
            assert result["worker_returncode"] == -9
            assert control["deadline_expired"] and control["kill_requested"]
            assert control["terminal_reason"] == "parent_lost"
            assert marker["guard"]["sequence"] == RUNTIME_PARENT_FAULTS.index(fault)
        elif fault.endswith("hang"):
            assert result["worker_returncode"] == -9
            assert control["deadline_expired"] and control["kill_requested"]
            assert control["terminal_reason"] == "deadline_expired"
        else:
            assert result["worker_returncode"] == 74
            assert not control["kill_requested"] and not control["deadline_expired"]
            assert worker["custody"] == "full_quarantine"
        if not fault.startswith("runtime_terminal_") and fault not in RUNTIME_ACK_FAULTS:
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
        if write_fd is not None:
            os.close(write_fd)
        if supervisor.poll() is None:
            # Real parent EOF goes through the original supervisor deadline.
            supervisor.communicate(timeout=deadline / 1000 + 15)
