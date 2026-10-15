"""Parent-side manifest and proxy for the closed native-lane worker."""

from __future__ import annotations

import json
import os
import platform
import site
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from types import MappingProxyType

from optical_worker.manifest import (
    AST_POLICY_SHA256,
    current_environment_identity,
    sha256_file,
    validate_runtime_manifest_against_process,
)
from optical_worker.protocol import (
    CLOSED_WORKER_PROTOCOL_VERSION,
    CLOSED_WORKER_RUNTIME_MANIFEST_VERSION,
    CLOSED_WORKER_STATE_MACHINE_VERSION,
    CLOSED_WORKER_TRUST_MODEL_VERSION,
    MAX_NESTING_DEPTH,
    MAX_OBJECT_FIELDS,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    MAX_STRING_BYTES,
    ClosedNativeLaneWorkerEvidence,
    ClosedWorkerProtocolError,
    ClosedWorkerRuntimeManifest,
    FrozenSourceDigest,
    canonical_json_bytes,
    canonical_sha256,
    decode_canonical_message,
    decode_frame_header,
    encode_framed_message,
    validate_response_envelope,
)

CLOSED_WORKER_RUNTIME_CONTRACT_RELATIVE_PATH = (
    "benchmarks/results/q54_p12_4/closed_worker_runtime_contract.json"
)
CLOSED_WORKER_GATE_RELATIVE_PATH = "benchmarks/results/q54_p12_4/gate.json"


def _load_runtime_contract(root: Path) -> dict[str, object]:
    path = root / CLOSED_WORKER_RUNTIME_CONTRACT_RELATIVE_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version",
        "protocol_version",
        "trust_model_version",
        "state_machine_version",
        "control_idle_timeout_ms",
        "response_write_timeout_ms",
        "abort_watchdog_ms",
        "gpu_preparation_enabled",
        "worker_script_relative_path",
        "supervisor_script_relative_path",
        "trusted_source_paths",
        "request_limit_bytes",
        "response_limit_bytes",
        "nesting_depth_limit",
        "object_field_limit",
        "string_limit_utf8_bytes",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ClosedWorkerProtocolError("closed-worker runtime contract fields are invalid")
    expected = {
        "schema_version": "q54.p12.4.closed-worker-runtime-contract.v2",
        "protocol_version": CLOSED_WORKER_PROTOCOL_VERSION,
        "trust_model_version": CLOSED_WORKER_TRUST_MODEL_VERSION,
        "state_machine_version": CLOSED_WORKER_STATE_MACHINE_VERSION,
        "gpu_preparation_enabled": False,
        "request_limit_bytes": MAX_REQUEST_BYTES,
        "response_limit_bytes": MAX_RESPONSE_BYTES,
        "nesting_depth_limit": MAX_NESTING_DEPTH,
        "object_field_limit": MAX_OBJECT_FIELDS,
        "string_limit_utf8_bytes": MAX_STRING_BYTES,
    }
    for name, value in expected.items():
        if payload[name] != value:
            raise ClosedWorkerProtocolError(f"closed-worker contract {name} is invalid")
    for name in ("control_idle_timeout_ms", "response_write_timeout_ms", "abort_watchdog_ms"):
        value = payload[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ClosedWorkerProtocolError(f"closed-worker contract {name} is invalid")
    paths = payload["trusted_source_paths"]
    if not isinstance(paths, list) or paths != sorted(set(paths)):
        raise ClosedWorkerProtocolError("closed-worker trusted source list is not canonical")
    return payload


def _workload_set_sha256(gate: dict[str, object]) -> str:
    return canonical_sha256(
        {
            "cases": gate["cases"],
            "boundaries": gate["boundaries"],
            "mandatory_rows": gate["mandatory_rows"],
        }
    )


def build_closed_worker_runtime_manifest(
    repository_root: Path,
    *,
    lane_id: str,
    device_ordinal: int,
) -> ClosedWorkerRuntimeManifest:
    """Build a clean-tree manifest; no CUDA runtime is imported or queried."""

    root = repository_root.resolve()
    contract = _load_runtime_contract(root)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ClosedWorkerProtocolError("closed-worker manifest requires a clean repository")
    gate_path = root / CLOSED_WORKER_GATE_RELATIVE_PATH
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    sources = tuple(
        FrozenSourceDigest(relative_path=relative, sha256=sha256_file(root / relative))
        for relative in contract["trusted_source_paths"]
    )
    environment = current_environment_identity()
    site_paths = tuple(
        sorted(
            {
                str(Path(path).resolve())
                for path in site.getsitepackages()
                if Path(path).is_dir()
            }
        )
    )
    manifest = ClosedWorkerRuntimeManifest(
        schema_version=CLOSED_WORKER_RUNTIME_MANIFEST_VERSION,
        protocol_version=CLOSED_WORKER_PROTOCOL_VERSION,
        trust_model_version=CLOSED_WORKER_TRUST_MODEL_VERSION,
        state_machine_version=CLOSED_WORKER_STATE_MACHINE_VERSION,
        interpreter_executable=str(Path(sys.executable).resolve()),
        python_version=platform.python_version(),
        environment=MappingProxyType(environment),
        environment_sha256=canonical_sha256(environment),
        site_package_paths=site_paths,
        repository_root=str(root),
        repository_revision=revision,
        repository_clean=True,
        worker_script_relative_path=contract["worker_script_relative_path"],
        lane_id=lane_id,
        device_ordinal=device_ordinal,
        gate_relative_path=CLOSED_WORKER_GATE_RELATIVE_PATH,
        gate_sha256=sha256_file(gate_path),
        workload_set_sha256=_workload_set_sha256(gate),
        source_digests=sources,
        source_manifest_sha256=canonical_sha256([source.to_payload() for source in sources]),
        ast_policy_sha256=AST_POLICY_SHA256,
        control_idle_timeout_ms=contract["control_idle_timeout_ms"],
        response_write_timeout_ms=contract["response_write_timeout_ms"],
        abort_watchdog_ms=contract["abort_watchdog_ms"],
        gpu_preparation_enabled=False,
    )
    return ClosedWorkerRuntimeManifest.from_payload(manifest.to_payload())


def write_closed_worker_runtime_manifest(
    path: Path,
    manifest: ClosedWorkerRuntimeManifest,
) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(manifest.to_payload())
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class ClosedWorkerLaneProxy:
    """Opaque orchestration client; intentionally has no sample-runner API."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        artifact_root: Path,
    ) -> None:
        if os.name != "posix":
            raise ClosedWorkerProtocolError("the reviewed C4c0 launcher currently requires Linux")
        payload = decode_canonical_message(
            manifest_path.read_bytes(), max_bytes=MAX_RESPONSE_BYTES
        )
        self._manifest = ClosedWorkerRuntimeManifest.from_payload(payload)
        validate_runtime_manifest_against_process(self._manifest)
        root = Path(self._manifest.repository_root)
        contract = _load_runtime_contract(root)
        artifact_root.mkdir(parents=True, exist_ok=True)
        self._journal_path = (artifact_root / "transport-termination.bin").resolve()
        self._supervisor_evidence_path = (artifact_root / "supervisor-evidence.json").resolve()
        parent_socket, worker_socket = socket.socketpair()
        parent_liveness_read, parent_liveness_write = os.pipe()
        supervisor_script = root / contract["supervisor_script_relative_path"]
        worker_script = root / self._manifest.worker_script_relative_path
        self._supervisor = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                str(supervisor_script),
                str(manifest_path.resolve()),
                str(worker_socket.fileno()),
                str(parent_liveness_read),
                str(self._journal_path),
                str(self._supervisor_evidence_path),
                str(worker_script),
            ],
            close_fds=True,
            pass_fds=(worker_socket.fileno(), parent_liveness_read),
        )
        worker_socket.close()
        os.close(parent_liveness_read)
        self._socket = parent_socket
        self._liveness_write = parent_liveness_write
        self._next_sequence = 0
        self._closed = False

    @property
    def supervisor_process_id(self) -> int:
        return self._supervisor.pid

    @property
    def supervisor_evidence_path(self) -> Path:
        return self._supervisor_evidence_path

    @property
    def termination_journal_path(self) -> Path:
        return self._journal_path

    def request(self, operation: str, arguments: dict[str, object]) -> dict[str, object]:
        if self._closed:
            raise ClosedWorkerProtocolError("closed-worker proxy is closed")
        sequence = self._next_sequence
        self._next_sequence += 1
        frame = encode_framed_message(
            {
                "protocol_version": CLOSED_WORKER_PROTOCOL_VERSION,
                "sequence": sequence,
                "operation": operation,
                "arguments": arguments,
            },
            max_bytes=MAX_REQUEST_BYTES,
        )
        try:
            self._socket.settimeout(None)
            self._socket.sendall(frame)
            header = self._recv_exact(4)
            size = decode_frame_header(header, max_bytes=MAX_RESPONSE_BYTES)
            response = decode_canonical_message(self._recv_exact(size), max_bytes=MAX_RESPONSE_BYTES)
            canonical = validate_response_envelope(response)
            if canonical["sequence"] != sequence:
                raise ClosedWorkerProtocolError("worker response sequence mismatch")
            return canonical
        except BaseException:
            self._close_and_reap_after_request_failure()
            raise

    def hello_and_open(self) -> tuple[dict[str, object], dict[str, object]]:
        hello = self.request(
            "HELLO",
            {
                "manifest_sha256": self._manifest.sha256,
                "lane_id": self._manifest.lane_id,
                "device_ordinal": self._manifest.device_ordinal,
            },
        )
        worker_evidence = ClosedNativeLaneWorkerEvidence.from_payload(
            hello["payload"]["worker_evidence"]
        )
        if (
            hello["payload"]["worker_evidence_sha256"] != worker_evidence.sha256
            or worker_evidence.runtime_manifest_sha256 != self._manifest.sha256
            or worker_evidence.lane_id != self._manifest.lane_id
            or worker_evidence.device_ordinal != self._manifest.device_ordinal
            or not set(worker_evidence.repository_module_paths).issubset(
                {source.relative_path for source in self._manifest.source_digests}
            )
        ):
            raise ClosedWorkerProtocolError("live worker evidence does not bind the runtime manifest")
        opened = self.request(
            "OPEN_FROZEN_SESSION",
            {
                "gate_sha256": self._manifest.gate_sha256,
                "workload_set_sha256": self._manifest.workload_set_sha256,
            },
        )
        return hello, opened

    def close_session(self) -> dict[str, object]:
        response = self.request("CLOSE_SESSION", {})
        self.wait_supervisor(timeout_seconds=10.0)
        self.close_transport()
        return response

    def close_transport(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._socket.close()
        os.close(self._liveness_write)

    def wait_supervisor(self, *, timeout_seconds: float = 10.0) -> int:
        return self._supervisor.wait(timeout=timeout_seconds)

    def _close_and_reap_after_request_failure(self) -> None:
        self.close_transport()
        timeout = self._manifest.abort_watchdog_ms / 1000.0 + 10.0
        try:
            self._supervisor.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._supervisor.kill()
            self._supervisor.wait()

    def _recv_exact(self, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = self._socket.recv(remaining)
            if not chunk:
                raise ClosedWorkerProtocolError("worker response was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
