"""Recorded bundle runs only in the fixed supervised sacrificial process."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.support.sacrificial_sources import canonical_bytes, frozen_source_manifest

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(os.name != "posix", reason="POSIX disposable supervisor")]


def run_case(
    tmp_path,
    *,
    profile="direct_light_full",
    shadows=True,
    diagnostics=True,
    resident_rays=None,
    fault="normal",
    position=0,
    target=0,
    ack_role=None,
    ack_fault="normal",
):
    root = Path(__file__).resolve().parents[2]
    payload = dict(
        sources=frozen_source_manifest(root),
        session=secrets.token_hex(16),
        target=target,
        profile=profile,
        shadows=shadows,
        diagnostics=diagnostics,
        resident_rays=resident_rays,
        fault=fault,
        position=position,
        ack_role=ack_role,
        ack_fault=ack_fault,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(canonical_bytes(payload))
    read_fd, write_fd = os.pipe()
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                str(root / "tools/optical_sacrificial_warp_supervisor.py"),
                str(manifest_path),
                str(tmp_path),
                str(read_fd),
            ],
            pass_fds=(read_fd,),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            if fault.startswith("hold_"):
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline and process.poll() is None:
                    journal_path = tmp_path / "journal.jsonl"
                    if journal_path.exists() and '"operation": "transport_hold"' in journal_path.read_text():
                        os.close(write_fd)
                        write_fd = None
                        break
                    time.sleep(0.02)
                else:
                    pytest.fail("child never reached requested real GPU transport hold")
            _, stderr = process.communicate(timeout=200)
        except BaseException:
            # Close liveness first so the real supervisor kills/reaps its child.
            if write_fd is not None:
                os.close(write_fd)
                write_fd = None
            process.communicate(timeout=10)
            raise
    finally:
        os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)
    assert process.returncode == 0, stderr
    supervisor = json.loads((tmp_path / "supervisor.json").read_bytes())
    journal = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines()]
    return supervisor, journal


@pytest.mark.parametrize("profile", ["direct_light_full", "rgb_preview", "render_only"])
@pytest.mark.parametrize("shadows,diagnostics", [(False, False), (False, True), (True, False), (True, True)])
def test_recorded_camera_matches_independent_cpu_oracle(tmp_path, profile, shadows, diagnostics):
    supervisor, journal = run_case(tmp_path, profile=profile, shadows=shadows, diagnostics=diagnostics)
    assert supervisor["worker_return_code"] == 0, (supervisor, journal)
    assert journal[0] == {"operation": "guardian_created", "imported_cuda_modules": []}
    inspected = next(item for item in journal if item["operation"] == "result_inspected")
    assert 0 < inspected["hit_count"] < inspected["ray_count"]
    assert inspected["distinct_hit_ids"] == inspected["distinct_hit_materials"] == 21
    if shadows and diagnostics:
        assert inspected["scalar_counts"]["shadow_traversal_occluded_count"] > 0
        assert inspected["scalar_counts"]["shadow_traversal_unoccluded_count"] > 0
    assert inspected["bundle_evidence"]["output_profile"] == profile
    released = next(item for item in journal if item["operation"] == "released")
    assert released["evidence"]["state"] == "RELEASED"
    assert released["evidence"]["attempted_launch_count"] == 5
    assert released["evidence"]["confirmed_launch_return_count"] == 5
    assert supervisor["terminal_deadline_armed"]
    assert_no_early_destruction(journal)


def test_recorded_render_targets_second_real_gpu_and_restores_ambient(tmp_path):
    import warp as wp

    if wp.get_cuda_device_count() < 2:
        pytest.skip("two real GPUs required; a skip is not dual-device evidence")
    supervisor, journal = run_case(tmp_path, target=1)
    assert supervisor["worker_return_code"] == 0, (supervisor, journal)
    inspected = next(item for item in journal if item["operation"] == "result_inspected")
    assert inspected["bundle_evidence"]["device_ordinal"] == 1
    assert inspected["ambient_device_ordinal"] == 0
    assert_no_early_destruction(journal)


def assert_no_early_destruction(journal):
    operations = [item["operation"] for item in journal]
    if "allocation_owner_finalized" in operations:
        assert "release_begin" in operations
        assert operations.index("release_begin") < operations.index("allocation_owner_finalized")


@pytest.mark.parametrize("resident_rays", [0, 1, 21])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_resident_and_empty_recorded_abi(tmp_path, resident_rays, diagnostics):
    supervisor, journal = run_case(tmp_path, resident_rays=resident_rays, diagnostics=diagnostics)
    assert supervisor["worker_return_code"] == 0, (supervisor, journal)
    inspected = next(item for item in journal if item["operation"] == "result_inspected")
    assert inspected["ray_count"] == resident_rays
    released = next(item for item in journal if item["operation"] == "released")
    assert released["evidence"]["confirmed_launch_return_count"] == (2 if resident_rays == 0 else 4)
    assert_no_early_destruction(journal)


@pytest.mark.parametrize("position", range(5))
@pytest.mark.parametrize("fault", ["pre_latch_failure", "post_latch_failure", "submit_then_raise"])
def test_every_command_partial_submission_lifetime(tmp_path, position, fault):
    supervisor, journal = run_case(tmp_path, fault=fault, position=position)
    operations = [item["operation"] for item in journal]
    clean = position == 0 and fault == "pre_latch_failure"
    assert supervisor["worker_return_code"] == (0 if clean else 74), (supervisor, journal)
    expected_attempts = position if fault == "pre_latch_failure" else position + 1
    assert operations.count("launch_latched") == expected_attempts
    assert "result_inspected" not in operations
    release = next(item for item in journal if item["operation"] == "release_begin")
    evidence = release["evidence"]
    assert evidence["attempted_launch_count"] == expected_attempts
    assert evidence["confirmed_launch_return_count"] == position
    assert evidence["device_quiescence_proven"]
    assert evidence["state"] == ("SEALED" if clean else "SYNCED")
    assert operations.count("emergency_sync_entered") == (0 if clean else 1)
    assert_no_early_destruction(journal)


@pytest.mark.parametrize("position", range(5))
@pytest.mark.parametrize("fault", ["record_failure", "warmup_failure"])
def test_setup_recording_and_warmup_failure(tmp_path, position, fault):
    supervisor, journal = run_case(tmp_path, fault=fault, position=position)
    operations = [item["operation"] for item in journal]
    assert supervisor["worker_return_code"] == 74, (supervisor, journal)
    assert "launch_latched" not in operations  # measured commands only, not setup GPU work
    assert "result_inspected" not in operations
    if fault == "record_failure":
        assert "emergency_sync_entered" not in operations
        assert "release_begin" not in operations
        assert journal[-1]["evidence"]["state"] == "QUARANTINED"
    else:
        assert operations.count("emergency_sync_entered") == 1
        assert "failure_release_authorized" in operations
    assert_no_early_destruction(journal)


@pytest.mark.parametrize(
    "fault",
    [
        "runtime_import_failure",
        "allocation_return_failure",
        "first_upload_failure",
        "compile_failure",
        "setup_barrier_raise",
        "setup_barrier_hang",
        "validation_raise",
        "validation_hang",
        "emergency_raise",
        "emergency_hang",
    ],
)
def test_setup_and_drain_failures_have_process_terminal_disposition(tmp_path, fault):
    supervisor, journal = run_case(tmp_path, fault=fault)
    operations = [item["operation"] for item in journal]
    assert "result_inspected" not in operations
    if fault.endswith("hang"):
        assert supervisor["killed"] and supervisor["reason"] == "guard_deadline"
        assert "release_begin" not in operations
    else:
        assert supervisor["worker_return_code"] == 74, (supervisor, journal)
        if fault == "first_upload_failure":
            assert "failure_release_authorized" in operations
        else:
            assert journal[-1]["evidence"]["state"] == "QUARANTINED"
            assert "release_begin" not in operations
    if fault.startswith(("setup_barrier", "validation")):
        assert "emergency_sync_entered" not in operations
    if fault.startswith("validation"):
        assert operations.count("launch_latched") == 5
    assert_no_early_destruction(journal)


@pytest.mark.parametrize("role", ["S", "V", "E"])
@pytest.mark.parametrize("fault", ["no_ack", "wrong_role", "wrong_sequence", "ack_eof"])
def test_real_gpu_graph_watchdog_rejects_missing_or_wrong_ack(tmp_path, role, fault):
    supervisor, journal = run_case(
        tmp_path,
        ack_role=role,
        ack_fault=fault,
        fault="post_latch_failure" if role == "E" else "normal",
    )
    operations = [item["operation"] for item in journal]
    assert supervisor["worker_return_code"] == 74, (supervisor, journal)
    forbidden = {"S": "setup_wait_entered", "V": "validation_drain_entered", "E": "emergency_sync_entered"}
    assert forbidden[role] not in operations
    assert "allocation_owner_finalized" not in operations
    assert "release_begin" not in operations
    assert "result_inspected" not in operations
    assert journal[-1]["evidence"]["state"] == "QUARANTINED"


@pytest.mark.parametrize("location", ["upload", "record", "sealed", "submit", "validated"])
def test_parent_loss_kills_real_gpu_child_without_unwinding(tmp_path, location):
    supervisor, journal = run_case(tmp_path, fault=f"hold_{location}")
    assert supervisor["killed"] and supervisor["reason"] == "parent_lost"
    operations = [item["operation"] for item in journal]
    assert "transport_hold" in operations
    assert "allocation_owner_finalized" not in operations
    assert "release_begin" not in operations
    assert "result_inspected" not in operations


@pytest.mark.parametrize("fault", ["journal_encoding_failure", "journal_write_failure"])
def test_post_submit_journal_failure_is_hard_exit_not_python_unwind(tmp_path, fault):
    supervisor, journal = run_case(tmp_path, fault=fault)
    assert supervisor["worker_return_code"] == 74, supervisor
    operations = [item["operation"] for item in journal]
    assert operations.count("launch_latched") == 1
    assert "result_inspected" not in operations
    assert "release_begin" not in operations
    assert "allocation_owner_finalized" not in operations
    assert "Traceback" not in supervisor["stderr"]
