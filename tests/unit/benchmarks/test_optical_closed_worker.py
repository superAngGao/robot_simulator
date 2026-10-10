from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
from dataclasses import replace
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
    assert {"optical_worker/lane_contract.py", "optical_worker/native_manifest.py"}.issubset(
        evidence["repository_module_paths"]
    )
    assert all(not path.startswith("benchmarks/") for path in evidence["repository_module_paths"])
    assert set(evidence["repository_module_paths"]).issubset(
        {source.relative_path for source in manifest.source_digests}
    )
    assert opened["payload"] == {"result_capable": False, "state": "SESSION_OPEN_IDLE"}
    assert not hasattr(proxy, "submit_sample")
    closed = proxy.close_session()
    assert closed["payload"] == {"state": "CLOSE_ACK_PENDING", "worker_exit": True}
    supervisor = json.loads(proxy.supervisor_evidence_path.read_text(encoding="utf-8"))
    assert supervisor["worker_return_code"] == 0
    assert supervisor["parent_lost"] is False
    assert supervisor["abort_watchdog_started"] is False
    assert supervisor["watchdog_killed_worker"] is False
    assert supervisor["no_restart"] is True


@pytest.mark.parametrize("removed", ["optical_worker/lane_contract.py", "optical_worker/native_manifest.py"])
def test_c0_bootstrap_rejects_missing_transitive_cpu_owner_before_import(tmp_path, removed):
    # Exercise the fixed bootstrap allowlist before any source hash/file import.
    from tools.optical_closed_worker import _REQUIRED_C4C0_SOURCE_PATHS, _load_and_prevalidate_manifest
    root = Path(__file__).resolve().parents[3]
    payload = {"repository_root": str(root), "source_digests": [
        {"relative_path": path, "sha256": "0" * 64}
        for path in _REQUIRED_C4C0_SOURCE_PATHS if path != removed]}
    path = tmp_path / "missing-owner.json"
    path.write_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    with pytest.raises(RuntimeError, match="frozen C4c0 allowlist"):
        _load_and_prevalidate_manifest(path)


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


@pytest.mark.skipif(os.name != "posix", reason="reviewed closed-worker launcher is Linux-only")
def test_idle_timeout_records_original_fault_and_reaps_worker(tmp_path):
    from optical_worker.protocol import (
        FRAME_HEADER_BYTES,
        MAX_RESPONSE_BYTES,
        decode_canonical_message,
        decode_frame_header,
    )

    root = Path(__file__).resolve().parents[3]
    manifest = build_closed_worker_runtime_manifest(root, lane_id="warp", device_ordinal=0)
    manifest_path = tmp_path / "idle-timeout-runtime.json"
    write_closed_worker_runtime_manifest(manifest_path, manifest)
    proxy = ClosedWorkerLaneProxy(manifest_path=manifest_path, artifact_root=tmp_path / "artifacts")
    try:
        proxy.hello_and_open()
        assert proxy.wait_supervisor(timeout_seconds=manifest.control_idle_timeout_ms / 1000.0 + 10.0) == 0
        supervisor = json.loads(proxy.supervisor_evidence_path.read_bytes())
        assert supervisor["worker_return_code"] == 64
        assert not supervisor["parent_lost"] and not supervisor["watchdog_killed_worker"]
        encoded = proxy.termination_journal_path.read_bytes()
        records = []
        while encoded:
            size = decode_frame_header(encoded[:FRAME_HEADER_BYTES], max_bytes=MAX_RESPONSE_BYTES)
            end = FRAME_HEADER_BYTES + size
            records.append(
                decode_canonical_message(encoded[FRAME_HEADER_BYTES:end], max_bytes=MAX_RESPONSE_BYTES)
            )
            encoded = encoded[end:]
        assert records[0] == {"kind": "worker_loop_fault", "fault": "control_idle_timeout",
                              "exception_type": "TimeoutError", "message": "control channel idle timeout"}
        assert records[-1]["kind"] == "transport_termination"
        assert records[-1]["evidence"]["disposition"] == "cancelled_before_submission_exit"
    finally:
        proxy.close_transport()
        proxy.wait_supervisor(timeout_seconds=10.0)


@pytest.mark.skipif(os.name != "posix", reason="reviewed closed-worker launcher is Linux-only")
def test_supervisor_watchdog_kills_hung_emergency_sync_while_parent_is_alive(tmp_path):
    root = Path(__file__).resolve().parents[3]
    supervisor_script = root / "tools" / "optical_closed_worker_supervisor.py"
    probe_root = tmp_path / "probe-root"
    probe_tools = probe_root / "tools"
    probe_tools.mkdir(parents=True)
    worker_script = probe_tools / "hung_abort_worker.py"
    worker_script.write_text(
        "import os,sys,time\n"
        "os.write(int(sys.argv[5]), b'A')\n"
        "assert os.read(int(sys.argv[6]), 2) == b'K'\n"
        "while True: time.sleep(1)\n",
        encoding="utf-8",
    )

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest_payload = {
        "abort_watchdog_ms": 100,
        "interpreter_executable": str(Path(sys.executable).resolve()),
        "repository_root": str(probe_root.resolve()),
        "worker_script_relative_path": "tools/hung_abort_worker.py",
        "source_digests": [
            {
                "relative_path": "tools/optical_closed_worker_supervisor.py",
                "sha256": digest(supervisor_script),
            },
            {
                "relative_path": "tools/hung_abort_worker.py",
                "sha256": digest(worker_script),
            },
        ],
    }
    manifest_path = tmp_path / "probe-manifest.json"
    manifest_path.write_bytes(
        json.dumps(
            manifest_payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    parent_socket, worker_socket = socket.socketpair()
    parent_liveness_read, parent_liveness_write = os.pipe()
    journal_path = tmp_path / "probe-journal.bin"
    evidence_path = tmp_path / "probe-supervisor.json"
    supervisor = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-S",
            str(supervisor_script),
            str(manifest_path),
            str(worker_socket.fileno()),
            str(parent_liveness_read),
            str(journal_path),
            str(evidence_path),
            str(worker_script),
        ],
        close_fds=True,
        pass_fds=(worker_socket.fileno(), parent_liveness_read),
    )
    worker_socket.close()
    os.close(parent_liveness_read)
    try:
        assert supervisor.wait(timeout=5.0) == 0
    finally:
        parent_socket.close()
        os.close(parent_liveness_write)
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["parent_lost"] is False
    assert evidence["abort_watchdog_started"] is True
    assert evidence["abort_signal_valid"] is True
    assert evidence["abort_ack_sent"] is True
    assert evidence["watchdog_reasons"] == ["emergency_sync"]
    assert evidence["watchdog_killed_worker"] is True


@pytest.mark.skipif(os.name != "posix", reason="process-terminal integration is Linux-only")
@pytest.mark.parametrize("acknowledge", [True, False])
def test_worker_loop_forces_quarantine_exit_without_running_owner_finalizer(
    tmp_path,
    acknowledge,
):
    from optical_worker.protocol import (
        MAX_REQUEST_BYTES,
        MAX_RESPONSE_BYTES,
        decode_canonical_message,
        decode_frame_header,
        encode_framed_message,
    )

    root = Path(__file__).resolve().parents[3]
    manifest = build_closed_worker_runtime_manifest(root, lane_id="cuda", device_ordinal=0)
    manifest = replace(manifest, abort_watchdog_ms=100)
    manifest_path = tmp_path / "forced-quarantine-runtime.json"
    write_closed_worker_runtime_manifest(manifest_path, manifest)
    parent_socket, child_socket = socket.socketpair()
    liveness_read, liveness_write = os.pipe()
    abort_read, abort_write = os.pipe()
    abort_ack_read, abort_ack_write = os.pipe()
    abort_ack_write_open = True
    journal_path = tmp_path / "forced-quarantine-journal.bin"
    journal_descriptor = os.open(journal_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    finalizer_marker = tmp_path / "ordinary-finalizer-ran"
    sync_marker = tmp_path / "emergency-sync-entered"
    script = r"""
import sys
from pathlib import Path
from optical_worker import worker as worker_module
from optical_worker.lifecycle import DeviceScopeDescriptor
from optical_worker.native_setup import NativeWorkerPreparationAborted
from optical_worker.protocol import ClosedWorkerRuntimeManifest, MAX_RESPONSE_BYTES, decode_canonical_message

manifest = ClosedWorkerRuntimeManifest.from_payload(
    decode_canonical_message(Path(sys.argv[1]).read_bytes(), max_bytes=MAX_RESPONSE_BYTES))
command_descriptor = int(sys.argv[2])
liveness_descriptor = int(sys.argv[3])
journal_descriptor = int(sys.argv[4])
abort_descriptor = int(sys.argv[5])
abort_ack_descriptor = int(sys.argv[6])
marker = Path(sys.argv[7])
sync_marker = Path(sys.argv[8])
original_handle = worker_module.ClosedWorkerSession.handle

class Owner:
    def __del__(self):
        marker.write_text('unsafe ordinary cleanup', encoding='utf-8')

def injected_handle(self, raw):
    if isinstance(raw, dict) and raw.get('operation') == 'PREPARE_FROZEN_ROW':
        lifecycle = self._lifecycle
        descriptor = DeviceScopeDescriptor.exact_target(
            operation_id='forced_sync_failure', leaf_kind='allocation', device_ordinal=0,
            scope_source='integration_probe', runtime_identity='forced_sync_failure_v1')
        permit = lifecycle.begin_device_leaf(descriptor)
        lifecycle.enter_device_leaf(permit)
        owner = Owner()
        lifecycle.adopt_returned_device_owner(
            owner, actual_device_ordinal=0, role='integration_device_owner',
            operation_id='forced_sync_failure')
        def fail_sync():
            sync_marker.write_text('sync entered', encoding='utf-8')
            raise RuntimeError('injected emergency sync failure')
        evidence = lifecycle.terminate(
            fault='setup_failure', detection_phase='forced_sync_failure',
            emergency_target_synchronize=fail_sync)
        raise NativeWorkerPreparationAborted(
            'injected native preparation abort', evidence=evidence)
    return original_handle(self, raw)

worker_module.ClosedWorkerSession.handle = injected_handle
worker_module.validate_repository_import_ledger = lambda manifest: ()
raise SystemExit(worker_module.run_worker_loop(
    command_descriptor=command_descriptor,
    liveness_descriptor=liveness_descriptor,
    journal_descriptor=journal_descriptor,
    abort_signal_descriptor=abort_descriptor,
    abort_ack_descriptor=abort_ack_descriptor,
    manifest=manifest))
"""
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(manifest_path),
            str(child_socket.fileno()),
            str(liveness_read),
            str(journal_descriptor),
            str(abort_write),
            str(abort_ack_read),
            str(finalizer_marker),
            str(sync_marker),
        ],
        cwd=root,
        close_fds=True,
        pass_fds=(
            child_socket.fileno(),
            liveness_read,
            journal_descriptor,
            abort_write,
            abort_ack_read,
        ),
    )

    child_socket.close()
    os.close(liveness_read)
    os.close(abort_write)
    os.close(abort_ack_read)
    os.close(journal_descriptor)

    def request(sequence: int, operation: str, arguments: dict[str, object]):
        parent_socket.sendall(
            encode_framed_message(
                {
                    "protocol_version": manifest.protocol_version,
                    "sequence": sequence,
                    "operation": operation,
                    "arguments": arguments,
                },
                max_bytes=MAX_REQUEST_BYTES,
            )
        )
        header = parent_socket.recv(4)
        size = decode_frame_header(header, max_bytes=MAX_RESPONSE_BYTES)
        chunks = bytearray()
        while len(chunks) < size:
            chunks.extend(parent_socket.recv(size - len(chunks)))
        return decode_canonical_message(bytes(chunks), max_bytes=MAX_RESPONSE_BYTES)

    try:
        request(
            0,
            "HELLO",
            {
                "manifest_sha256": manifest.sha256,
                "lane_id": "cuda",
                "device_ordinal": 0,
            },
        )
        request(
            1,
            "OPEN_FROZEN_SESSION",
            {
                "gate_sha256": manifest.gate_sha256,
                "workload_set_sha256": manifest.workload_set_sha256,
            },
        )
        parent_socket.sendall(
            encode_framed_message(
                {
                    "protocol_version": manifest.protocol_version,
                    "sequence": 2,
                    "operation": "PREPARE_FROZEN_ROW",
                    "arguments": {
                        "row_id": "r0_asymmetric_full_shadow",
                        "case_id": "asymmetric_synthetic_7x3_v1",
                        "repeat_index": 0,
                        "sample_count": 1,
                    },
                },
                max_bytes=MAX_REQUEST_BYTES,
            )
        )
        if acknowledge:
            assert os.read(abort_read, 1) == b"A"
            assert os.write(abort_ack_write, b"K") == 1
            os.close(abort_ack_write)
            abort_ack_write_open = False
        assert parent_socket.recv(1) == b""
        return_code = process.wait(timeout=10.0)
        if not acknowledge:
            assert os.read(abort_read, 1) == b"A"
    finally:
        parent_socket.close()
        os.close(liveness_write)
        if abort_ack_write_open:
            os.close(abort_ack_write)
    assert return_code == 74
    os.close(abort_read)
    assert not finalizer_marker.exists()
    assert sync_marker.exists() is acknowledge
    assert b"native_preparation_aborted" in journal_path.read_bytes()
