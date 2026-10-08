"""Actual cold Warp upload/readback plus terminal-disposal supervision.

Opt-in host-specific author acceptance, not a product deployment profile or
complete C2d3 acceptance. No parent Torch/Warp import, fixture uploads or warmup.
CUDA/Go2/foreign-device and setup fault matrices remain separate required work.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from optical_worker.guarded_evidence import validate_custody_artifact
from optical_worker.lane_contract import CANONICAL_NATIVE_REPLAY_INPUT, FIXED_NATIVE_SCENE_INPUT
from optical_worker.native_input import CanonicalNativeInput
from optical_worker.native_sources import CATALOG_PATH
from optical_worker.runtime_environment import RuntimeEnvironmentContract
from optical_worker.workloads import (
    build_canonical_fixed_fixture,
    build_canonical_packed_scene,
    build_canonical_transform_replay,
)

ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT = ROOT / "tests/data/runtime_environments/183_warp_synthetic_author_v5.json"
ENVIRONMENT_SHA256 = "0783cc820dcb738b8ec82586e2d14a12a7b9b81989e67d1ba9f86ce1ea3ed1b7"
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.name != "posix" or os.environ.get("ROBOT_SIM_TEST_C2D3_COLD_WARP") != "1",
        reason="explicit matching 183 deployment required; a skip is NOT C2d3 acceptance",
    ),
]


def _expected_source(case_id):
    # Independent host-only materialization through the sole canonical builders.
    # No manually copied dtype table, replay values or hash from child output.
    packed = build_canonical_packed_scene(case_id)
    if case_id == "synthetic_dynamic_rigid_960x640_v1":
        return CanonicalNativeInput.from_scene_contract(
            packed,
            input_contract=CANONICAL_NATIVE_REPLAY_INPUT,
            replay=build_canonical_transform_replay(case_id),
        )
    return CanonicalNativeInput.from_scene_contract(
        packed,
        input_contract=FIXED_NATIVE_SCENE_INPUT,
        fixed_fixture=build_canonical_fixed_fixture(packed),
    )


@pytest.mark.parametrize("case_id", ["asymmetric_synthetic_7x3_v1", "synthetic_dynamic_rigid_960x640_v1"])
@pytest.mark.parametrize(
    "fault", ["runtime_success", "runtime_terminal_sync_raise", "runtime_terminal_sync_hang"]
)
def test_actual_cold_canonical_warp_setup_and_terminal_disposition(tmp_path, fault, case_id):
    environment = RuntimeEnvironmentContract.from_payload(json.loads(ENVIRONMENT.read_bytes()))
    # The trusted test configuration pins bytes, not a caller self-reported hash.
    assert environment.sha256 == ENVIRONMENT_SHA256
    environment.verify_files(expected_sha256=ENVIRONMENT_SHA256)
    expected = _expected_source(case_id)
    expected_names = [array.name for array in expected.manifest.arrays]
    array_count = len(expected_names)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, timeout=15).decode().strip()
    request = {
        "root": str(ROOT),
        "revision": revision,
        "lane": "warp",
        "catalog_sha256": hashlib.sha256((ROOT / CATALOG_PATH).read_bytes()).hexdigest(),
        "guard_configuration": {
            "protocol_version": "q54.closed-worker.guarded-wait.v1",
            "session": "f" * 32,
            "device_ordinal": 1,
            "ack_timeout_ms": 1000,
            "operation_timeout_ms": 30000,
        },
        "environment_path": str(ENVIRONMENT),
        "environment_sha256": ENVIRONMENT_SHA256,
        "case_id": case_id,
        "go2_model_dir": None,
    }
    from tests.support.runtime_custody_launch import capture_runtime_custody_inputs

    inputs = capture_runtime_custody_inputs(request)
    (tmp_path / "runtime_custody_inputs.json").write_text(json.dumps(inputs), encoding="utf-8")
    (tmp_path / "request.json").write_text(json.dumps(request), encoding="utf-8")
    read_fd, write_fd = os.pipe()
    supervisor = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-B",
            str(ROOT / "tests/support/guarded_custody_process.py"),
            "supervisor",
            fault,
            str(read_fd),
            str(tmp_path),
        ],
        pass_fds=(read_fd,),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    os.close(read_fd)
    try:
        output, error = supervisor.communicate(timeout=300)
        assert supervisor.returncode == 0, (output, error)
        result = json.loads(output)
        artifact = result["artifact"]
        validate_custody_artifact(artifact)
        from optical_worker.runtime_custody import (
            RuntimeCustodyRecord, validate_runtime_custody_attachment, validate_runtime_custody_selection,
        )

        raw_record = (tmp_path / "runtime_custody.json").read_bytes()
        selection = json.loads((tmp_path / "runtime_custody_selection.json").read_bytes())
        validate_runtime_custody_selection(
            selection, raw_record, observed_lifecycle_event_count=(
                len(artifact["worker"]["transition_trace"])
                if artifact["worker"]["record_kind"] == "terminal" else None))
        assert selection["selection"] == "current"
        runtime_record = RuntimeCustodyRecord.from_payload(
            json.loads(raw_record), environment=environment, trusted_launch_inputs=inputs)
        assert validate_runtime_custody_attachment(runtime_record, environment, inputs, artifact) == "worker_prefix_matched"
        assert runtime_record.to_payload()["summary"]["input_validation_observed"] is True
        assert runtime_record.to_payload()["summary"]["final_execution_readiness_issued"] is False
        assert result["stderr"] == error == ""
        assert result["parent_alive"] is True
        assert json.loads((tmp_path / "identity.json").read_text()) == {
            "pid": result["worker_pid"],
            "ppid": supervisor.pid,
        }
        control = artifact["supervisor"]["summary"]
        worker = artifact["worker"]["summary"]
        assert control["complete"] and control["state"] == "EXITED"
        assert control["arm_count"] == 7 and control["normal_complete_count"] == 6
        assert worker["setup_work_counts"] == {
            "allocation": array_count,
            "host_to_device_copy": array_count,
            "device_to_host_copy": array_count,
            "event_create": 2,
            "event_record": 2,
            "event_wait": 2,
            "runtime_import_or_init": 6,
        }
        report = json.loads((tmp_path / "input_validation_report.json").read_text())
        assert report["final_execution_readiness_issued"] is False
        assert report["validated_arrays"] == expected_names
        assert {"instance_X_body_geometry_R", "instance_X_body_geometry_r"} <= set(report["validated_arrays"])
        assert report["manifest_sha256"] == expected.manifest.sha256
        assert report["content_sha256"] == expected.manifest.content_sha256
        assert report["topology_sha256"] == expected.manifest.topology_sha256
        if case_id == "synthetic_dynamic_rigid_960x640_v1":
            assert {"replay_x_world_R", "replay_x_world_r"} <= set(report["validated_arrays"])
            assert expected.manifest.transform_frame_count > 1
        assert not (tmp_path / "ordinary_atexit.json").exists()
        assert not (tmp_path / "failure.json").exists()
        assert json.loads((tmp_path / "terminal_sync_entered.json").read_text()) == {"device_ordinal": 1}
        if fault == "runtime_terminal_sync_hang":
            assert result["worker_returncode"] == -9
            assert control["deadline_expired"] and control["kill_requested"]
            assert control["terminal_reason"] == "deadline_expired"
            # The last worker snapshot predates terminal entry. Do not fabricate
            # completion/release from a killed worker; supervisor proves reap.
            assert artifact["worker"]["record_kind"] == "prefix"
            assert worker["release_status"] == "not_authorized"
            assert not (tmp_path / "worker.json").exists()
        else:
            assert not control["deadline_expired"] and not control["kill_requested"]
            assert worker["emergency_sync_attempt_count"] == worker["terminal_arm_attempt_count"] == 1
            if fault == "runtime_success":
                assert result["worker_returncode"] == 0
                assert worker["disposition"] == "aborted_synced_exit"
                assert worker["guardian_state"] == "RELEASED"
                assert worker["release_status"] == "completed"
            else:
                assert result["worker_returncode"] == 74
                assert worker["emergency_sync_outcome"] == "failed"
                assert worker["guardian_state"] == "TRANSFERRED_TO_QUARANTINE"
                assert worker["custody"] == "full_quarantine"
                assert worker["release_status"] == "not_authorized"
        (tmp_path / "custody_artifact.json").write_text(json.dumps(artifact), encoding="utf-8")
        print(
            json.dumps(
                {
                    "fault": fault,
                    "case_id": case_id,
                    "parent_pid": os.getpid(),
                    "supervisor_pid": supervisor.pid,
                    "worker_pid": result["worker_pid"],
                    "worker_returncode": result["worker_returncode"],
                }
            )
        )
    finally:
        os.close(write_fd)
        if supervisor.poll() is None:
            try:
                supervisor.communicate(timeout=40)
            except subprocess.TimeoutExpired:
                supervisor.kill()
                supervisor.communicate(timeout=3)
