"""Opt-in real C2d3 observations, only after independent native preflight.

Run with the external live OS identity recorder used by the custody matrices.
Neither collection nor CPU codec tests authorize or prove these native cases.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from optical_worker.native_sources import CATALOG_PATH
from optical_worker.runtime_custody import (RuntimeCustodyRecord, validate_runtime_custody_attachment,
                                           validate_runtime_custody_selection)
from optical_worker.runtime_environment import RuntimeEnvironmentContract
from tests.support.native_scope_acceptance import assert_native_scope_success
from tests.support.runtime_custody_launch import capture_runtime_custody_inputs
from tests.gpu.test_native_runtime_fault_matrix import PROFILES, GO2_PROFILES

ROOT = Path(__file__).resolve().parents[2]
CASES = ("asymmetric_synthetic_7x3_v1", "synthetic_dynamic_rigid_960x640_v1",
         "go2_static_960x640_v1", "go2_static_1920x1080_v1")
NEGATIVE = ("runtime_scope_getter_error", "runtime_scope_after_return_raise", "runtime_scope_after_append_raise")
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.name != "posix" or os.environ.get("ROBOT_SIM_TEST_NATIVE_SCOPE") != "1",
    reason="explicit reviewed native observer on matching183 required; skipped is not proof")]


@pytest.mark.parametrize("lane,case,fault", [
    (lane, case, "runtime_scope_success") for lane in ("warp", "cuda") for case in CASES
] + [(lane, CASES[0], fault) for lane in ("warp", "cuda") for fault in NEGATIVE]
  + [("warp", CASES[0], "runtime_scope_missing_warp_scope"), ("cuda", CASES[0], "runtime_scope_c1_copy"),
     ("warp", CASES[0], "runtime_scope_warp_exit_not_restored")])
def test_actual_native_scope_under_original_guard(tmp_path, lane, case, fault):
    go2 = case.startswith("go2_")
    filename, digest, deadline = (GO2_PROFILES if go2 else PROFILES)[lane]
    environment_path = ROOT / "tests/data/runtime_environments" / filename
    environment = RuntimeEnvironmentContract.from_payload(json.loads(environment_path.read_bytes()))
    assert environment.sha256 == digest
    environment.verify_files(expected_sha256=digest)
    mapping = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,driver_version",
                                       "--format=csv,noheader"], text=True, timeout=15)
    rows = [tuple(part.strip() for part in line.split(",")) for line in mapping.splitlines()]
    assert len(rows) == 8 and {row[0] for row in rows} == {str(i) for i in range(8)}
    assert ("0", "GPU-a4b256c9-c56a-0ef5-19fa-482196cc3274", "595.71.05") in rows
    assert ("1", "GPU-3a3112ff-837c-3b87-38d1-cb203d656589", "595.71.05") in rows
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    (tmp_path / "gpu_mapping.txt").write_text(mapping)
    request = {"root": str(ROOT), "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
               "lane": lane, "catalog_sha256": hashlib.sha256((ROOT / CATALOG_PATH).read_bytes()).hexdigest(),
               "environment_path": str(environment_path), "environment_sha256": digest, "case_id": case,
               "go2_model_dir": "/data2/shared/ang.gao/c2d2_frozen_assets/mujoco_menagerie/unitree_go2" if go2 else None,
               "guard_configuration": {"protocol_version": "q54.closed-worker.guarded-wait.v1", "session": "b" * 32,
                                       "device_ordinal": 1, "ack_timeout_ms": 1000, "operation_timeout_ms": deadline}}
    inputs = capture_runtime_custody_inputs(request)
    (tmp_path / "request.json").write_text(json.dumps(request))
    (tmp_path / "runtime_custody_inputs.json").write_text(json.dumps(inputs))
    read_fd, write_fd = os.pipe()
    supervisor = subprocess.Popen([sys.executable, "-I", "-B", str(ROOT / "tests/support/guarded_custody_process.py"),
                                  "supervisor", fault, str(read_fd), str(tmp_path)],
                                 pass_fds=(read_fd,), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
    os.close(read_fd)
    try:
        output, error = supervisor.communicate(timeout=10 * deadline / 1000 + 45)
        (tmp_path / "supervisor_stdout.json").write_text(output)
        (tmp_path / "supervisor_stderr.txt").write_text(error)
        assert supervisor.returncode == 0, (output, error)
        result = json.loads(output)
        assert result["stderr"] == error == "" and result["parent_alive"] is True
        artifact = result["artifact"]
        (tmp_path / "custody_artifact.json").write_text(json.dumps(artifact))
        raw = (tmp_path / "runtime_custody.json").read_bytes()
        selection = json.loads((tmp_path / "runtime_custody_selection.json").read_bytes())
        assert artifact["worker"]["record_kind"] == "terminal"
        validate_runtime_custody_selection(selection, raw,
            observed_lifecycle_event_count=len(artifact["worker"]["transition_trace"]))
        record = RuntimeCustodyRecord.from_payload(json.loads(raw), environment=environment, trusted_launch_inputs=inputs)
        assert validate_runtime_custody_attachment(record, environment, inputs, artifact) == "worker_prefix_matched"
        identity = json.loads((tmp_path / "external_process_identity.json").read_bytes())
        assert identity["worker"]["pid"] == result["worker_pid"]
        assert identity["supervisor"]["pid"] == supervisor.pid and identity["parent"]["pid"] == os.getpid()
        assert not (tmp_path / "ordinary_atexit.json").exists()
        if fault == "runtime_scope_success":
            assert result["worker_returncode"] == 0 and selection["selection"] == "current"
            assert_native_scope_success(record, artifact, identity)
        else:
            marker = json.loads((tmp_path / "fault_entered.json").read_bytes())
            assert marker["fault"] == fault
            worker = artifact["worker"]["summary"]
            assert result["worker_returncode"] == 74 and worker["unknown_device_scope"] is True
            assert worker["emergency_sync_attempt_count"] == worker["terminal_arm_attempt_count"] == 0
            assert worker["release_status"] == "not_authorized"
            assert not (tmp_path / "terminal_sync_entered.json").exists()
            assert not (tmp_path / "input_validation_report.json").exists()
            assert (tmp_path / "native_scope_diagnostic.json").exists()
            if fault == "runtime_scope_warp_exit_not_restored":
                assert marker["boundary"] == "readback_exit_omitted_context_pop"
                assert marker["guard"] == {"sequence": 5, "role": "setup"}
                assert worker["custody"] == "full_quarantine" and worker["guarded_segment_failed"] is True
                diagnostic = json.loads((tmp_path / "native_scope_diagnostic.json").read_bytes())
                observations = [row for row in diagnostic["raw"]
                                if row.get("site") == "readback_scope" and row.get("edge") == "after"]
                assert len(observations) == 1
                state = observations[0]["state"]
                assert state["default_device"] == 0
                assert state["current_context"] == state["target_context"] is not None
                assert state["current_context_device"] == 1
                assert [row["ordinal"] for row in state["primary_contexts"] if row["active"]] == [1]
                after = artifact["worker"]["transition_trace"][marker["lifecycle_event_count"]:]
                assert not {"normal_complete_started", "normal_arm_started", "terminal_arm_started",
                            "emergency_sync_started", "ordinary_cleanup_started"} & {
                                row["operation"] for row in after}
        print(json.dumps({"fault": fault, "lane": lane, "case": case, "worker_pid": result["worker_pid"],
                          "supervisor_pid": supervisor.pid, "worker_returncode": result["worker_returncode"],
                          "selection": selection["selection"]}))
    finally:
        os.close(write_fd)
        if supervisor.poll() is None:
            supervisor.communicate(timeout=deadline / 1000 + 15)
