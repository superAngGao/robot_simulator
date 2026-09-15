from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from benchmarks.optical.closed_worker import (
    ClosedWorkerLaneProxy,
    build_closed_worker_runtime_manifest,
    write_closed_worker_runtime_manifest,
)


@pytest.mark.skipif(os.name != "posix", reason="reviewed closed-worker launcher is Linux-only")
@pytest.mark.parametrize("lane", ["warp", "cuda"])
def test_clean_spawn_worker_hello_open_and_single_close(lane, tmp_path):
    root = Path(__file__).resolve().parents[3]
    manifest = build_closed_worker_runtime_manifest(root, lane_id=lane, device_ordinal=0)
    manifest_path = tmp_path / f"{lane}-runtime.json"
    write_closed_worker_runtime_manifest(manifest_path, manifest)
    proxy = ClosedWorkerLaneProxy(
        manifest_path=manifest_path,
        artifact_root=tmp_path / f"{lane}-artifacts",
    )
    hello, opened = proxy.hello_and_open()
    assert hello["status"] == "ok"
    evidence = hello["payload"]["worker_evidence"]
    assert evidence["lane_id"] == lane
    assert evidence["callback_surface_enabled"] is False
    assert evidence["object_deserialization_enabled"] is False
    assert evidence["gpu_preparation_enabled"] is False
    assert evidence["cuda_capable_module_names"] == []
    assert opened["payload"] == {"result_capable": False, "state": "SESSION_OPEN_IDLE"}
    assert not hasattr(proxy, "submit_sample")
    closed = proxy.close_session()
    assert closed["payload"] == {"state": "CLOSE_ACK_PENDING", "worker_exit": True}
    supervisor = json.loads(proxy.supervisor_evidence_path.read_text(encoding="utf-8"))
    assert supervisor["worker_return_code"] == 0
    assert supervisor["parent_lost"] is False
    assert supervisor["watchdog_killed_worker"] is False
    assert supervisor["no_restart"] is True


@pytest.mark.skipif(os.name != "posix", reason="reviewed closed-worker launcher is Linux-only")
def test_product_gpu_prepare_is_rejected_then_worker_exits_without_sample_activity(tmp_path):
    root = Path(__file__).resolve().parents[3]
    manifest = build_closed_worker_runtime_manifest(root, lane_id="warp", device_ordinal=0)
    manifest_path = tmp_path / "runtime.json"
    write_closed_worker_runtime_manifest(manifest_path, manifest)
    proxy = ClosedWorkerLaneProxy(manifest_path=manifest_path, artifact_root=tmp_path / "artifacts")
    proxy.hello_and_open()
    response = proxy.request(
        "PREPARE_FROZEN_ROW",
        {
            "row_id": "r0_asymmetric_full_shadow",
            "case_id": "asymmetric_synthetic_7x3_v1",
            "repeat_index": 0,
            "sample_count": 1,
        },
    )
    assert response["status"] == "error"
    payload = response["payload"]
    assert payload["code"] == "GPU_PREPARATION_DISABLED_PENDING_C4C0_REVIEW"
    assert payload["measured_sample_gpu_submission_delta"] == 0
    assert payload["measured_sample_command_launch_delta"] == 0
    assert payload["measured_sample_terminal_event_record_delta"] == 0
    assert payload["termination_evidence"]["setup_operation_count"] == 0
    assert payload["termination_evidence"]["disposition"] == ("cancelled_before_submission_exit")
    assert proxy.wait_supervisor(timeout_seconds=10.0) == 0
    proxy.close_transport()


@pytest.mark.skipif(os.name != "posix", reason="reviewed closed-worker launcher is Linux-only")
def test_unknown_raw_launch_request_terminates_before_any_gpu_capable_import(tmp_path):
    root = Path(__file__).resolve().parents[3]
    manifest = build_closed_worker_runtime_manifest(root, lane_id="cuda", device_ordinal=1)
    manifest_path = tmp_path / "runtime.json"
    write_closed_worker_runtime_manifest(manifest_path, manifest)
    proxy = ClosedWorkerLaneProxy(manifest_path=manifest_path, artifact_root=tmp_path / "artifacts")
    proxy.hello_and_open()
    response = proxy.request("CALL_RAW_LAUNCH", {})
    assert response["status"] == "error"
    assert response["payload"]["code"] == "ClosedWorkerProtocolError"
    evidence = response["payload"]["termination_evidence"]
    assert evidence["setup_operation_count"] == 0
    assert evidence["measured_sample_submission_latched"] is False
    assert evidence["emergency_sync_attempt_count"] == 0
    assert evidence["disposition"] == "cancelled_before_submission_exit"
    assert proxy.wait_supervisor(timeout_seconds=10.0) == 0
    proxy.close_transport()


@pytest.mark.skipif(os.name != "posix", reason="reviewed closed-worker launcher is Linux-only")
def test_control_channel_eof_is_a_terminal_state_transition_not_cleanup_authority(tmp_path):
    root = Path(__file__).resolve().parents[3]
    manifest = build_closed_worker_runtime_manifest(root, lane_id="warp", device_ordinal=0)
    manifest_path = tmp_path / "runtime.json"
    write_closed_worker_runtime_manifest(manifest_path, manifest)
    proxy = ClosedWorkerLaneProxy(manifest_path=manifest_path, artifact_root=tmp_path / "artifacts")
    proxy.hello_and_open()
    journal_path = proxy.termination_journal_path
    proxy.close_transport()
    assert proxy.wait_supervisor(timeout_seconds=10.0) == 0
    supervisor = json.loads(proxy.supervisor_evidence_path.read_text(encoding="utf-8"))
    assert supervisor["no_restart"] is True
    assert supervisor["worker_return_code"] != 0
    assert journal_path.stat().st_size > 0
