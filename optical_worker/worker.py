"""Closed-worker request loop for the C4c0 CPU/control-plane slice."""

from __future__ import annotations

import hashlib
import os
import secrets
import select
import sys
import time
from collections.abc import Mapping
from pathlib import Path

from .lifecycle import NativeWorkerSetupLifecycle, TransportTerminationEvidence
from .protocol import (
    CLOSED_WORKER_EVIDENCE_VERSION,
    CLOSED_WORKER_PROTOCOL_VERSION,
    FRAME_HEADER_BYTES,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    ClosedNativeLaneWorkerEvidence,
    ClosedWorkerProtocolError,
    ClosedWorkerRuntimeManifest,
    decode_canonical_message,
    decode_frame_header,
    encode_framed_message,
    validate_request,
)


class ClosedWorkerSession:
    """One connection, one session, and at most one frozen-row preparation."""

    def __init__(self, manifest: ClosedWorkerRuntimeManifest) -> None:
        self._manifest = manifest
        self._state = "EXPECT_HELLO"
        self._next_sequence = 0
        self._session_identity = f"session-{secrets.token_hex(16)}"
        self._lifecycle: NativeWorkerSetupLifecycle | None = None
        self._worker_evidence: ClosedNativeLaneWorkerEvidence | None = None
        self._termination: TransportTerminationEvidence | None = None
        self._close_ack_sequence: int | None = None

    @property
    def terminal(self) -> bool:
        return self._state in {"TERMINATED", "CLOSE_ACK_COMMITTED"}

    @property
    def hard_exit_required(self) -> bool:
        return self._termination is not None and self._termination.disposition in {
            "quarantined_process_exit",
            "multi_or_unknown_device_quarantine_exit",
        }

    @property
    def termination_evidence(self) -> TransportTerminationEvidence | None:
        return self._termination

    def handle(self, raw: object) -> dict[str, object]:
        request = validate_request(raw)
        sequence = request["sequence"]
        if sequence != self._next_sequence:
            raise ClosedWorkerProtocolError("request sequence is not the next connection sequence")
        self._next_sequence += 1
        operation = request["operation"]
        arguments = request["arguments"]
        if self._state == "EXPECT_HELLO":
            if operation != "HELLO":
                raise ClosedWorkerProtocolError("HELLO must be the first request")
            return self._hello(sequence, arguments)
        if self._state == "EXPECT_OPEN":
            if operation != "OPEN_FROZEN_SESSION":
                raise ClosedWorkerProtocolError("OPEN_FROZEN_SESSION must follow HELLO")
            return self._open(sequence, arguments)
        if self._state == "SESSION_OPEN_IDLE":
            if operation == "PREPARE_FROZEN_ROW":
                return self._reject_gpu_preparation(sequence)
            if operation == "CLOSE_SESSION":
                self._state = "CLOSE_ACK_PENDING"
                self._close_ack_sequence = sequence
                return self._ok(sequence, {"state": "CLOSE_ACK_PENDING", "worker_exit": True})
            raise ClosedWorkerProtocolError("operation is not legal in the idle C4c0 session")
        raise ClosedWorkerProtocolError("closed-worker session is already terminal or closing")

    def commit_written_response(self, sequence: int) -> None:
        if self._state == "CLOSE_ACK_PENDING":
            if sequence != self._close_ack_sequence:
                raise ClosedWorkerProtocolError("only the exact close acknowledgement can commit")
            assert self._lifecycle is not None
            self._lifecycle.commit_clean_close()
            self._state = "CLOSE_ACK_COMMITTED"

    def terminal_fault_response(
        self,
        *,
        sequence: int,
        code: str,
        message: str,
        fault: str,
        detection_phase: str,
    ) -> dict[str, object]:
        if self._lifecycle is not None and self._termination is None:
            self._termination = self._lifecycle.terminate(
                fault=fault,
                detection_phase=detection_phase,
                emergency_target_synchronize=None,
            )
        self._state = "TERMINATED"
        payload: dict[str, object] = {"code": code, "message": message, "worker_exit": True}
        if self._termination is not None:
            payload["termination_evidence"] = self._termination.to_payload()
            payload["termination_evidence_sha256"] = self._termination.sha256
        return self._error(sequence, payload)

    def _hello(self, sequence: int, arguments: Mapping[str, object]) -> dict[str, object]:
        if (
            arguments["manifest_sha256"] != self._manifest.sha256
            or arguments["lane_id"] != self._manifest.lane_id
            or arguments["device_ordinal"] != self._manifest.device_ordinal
        ):
            raise ClosedWorkerProtocolError("HELLO does not bind the runtime manifest")
        self._lifecycle = NativeWorkerSetupLifecycle(
            lane_id=self._manifest.lane_id,
            target_device_ordinal=self._manifest.device_ordinal,
            session_identity=self._session_identity,
        )
        executable_digest = hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
        evidence = ClosedNativeLaneWorkerEvidence(
            schema_version=CLOSED_WORKER_EVIDENCE_VERSION,
            protocol_version=self._manifest.protocol_version,
            trust_model_version=self._manifest.trust_model_version,
            state_machine_version=self._manifest.state_machine_version,
            runtime_manifest_sha256=self._manifest.sha256,
            source_manifest_sha256=self._manifest.source_manifest_sha256,
            ast_policy_sha256=self._manifest.ast_policy_sha256,
            gate_sha256=self._manifest.gate_sha256,
            workload_set_sha256=self._manifest.workload_set_sha256,
            lane_id=self._manifest.lane_id,
            device_ordinal=self._manifest.device_ordinal,
            process_id=os.getpid(),
            process_start_nonce=secrets.token_hex(16),
            worker_session_id=self._session_identity,
            interpreter_executable_sha256=executable_digest,
            callback_surface_enabled=False,
            object_deserialization_enabled=False,
            gpu_preparation_enabled=False,
            cuda_capable_module_names=tuple(
                sorted(name for name in ("numpy", "torch", "warp") if name in sys.modules)
            ),
        )
        self._worker_evidence = evidence
        self._state = "EXPECT_OPEN"
        return self._ok(
            sequence,
            {"worker_evidence": evidence.to_payload(), "worker_evidence_sha256": evidence.sha256},
        )

    def _open(self, sequence: int, arguments: Mapping[str, object]) -> dict[str, object]:
        if (
            arguments["gate_sha256"] != self._manifest.gate_sha256
            or arguments["workload_set_sha256"] != self._manifest.workload_set_sha256
        ):
            raise ClosedWorkerProtocolError("OPEN does not match the frozen gate/workload set")
        self._state = "SESSION_OPEN_IDLE"
        return self._ok(sequence, {"state": self._state, "result_capable": False})

    def _reject_gpu_preparation(self, sequence: int) -> dict[str, object]:
        if self._manifest.gpu_preparation_enabled:
            raise ClosedWorkerProtocolError("C4c0 must not activate product GPU preparation")
        assert self._lifecycle is not None
        self._termination = self._lifecycle.terminate(
            fault="request_protocol_failure",
            detection_phase="prepare_activation_gate",
            emergency_target_synchronize=None,
        )
        self._state = "TERMINATED"
        return self._error(
            sequence,
            {
                "code": "GPU_PREPARATION_DISABLED_PENDING_C4C0_REVIEW",
                "message": "C4c0 product PREPARE_FROZEN_ROW remains fail-closed",
                "worker_exit": True,
                "measured_sample_gpu_submission_delta": 0,
                "measured_sample_command_launch_delta": 0,
                "measured_sample_diagnostic_event_record_delta": 0,
                "measured_sample_terminal_event_record_delta": 0,
                "measured_sample_completion_ticket_delta": 0,
                "slot_state_sha256": "NO_SESSION_NO_SLOT",
                "termination_evidence": self._termination.to_payload(),
                "termination_evidence_sha256": self._termination.sha256,
            },
        )

    @staticmethod
    def _ok(sequence: int, payload: Mapping[str, object]) -> dict[str, object]:
        return {
            "protocol_version": CLOSED_WORKER_PROTOCOL_VERSION,
            "sequence": sequence,
            "status": "ok",
            "payload": dict(payload),
        }

    @staticmethod
    def _error(sequence: int, payload: Mapping[str, object]) -> dict[str, object]:
        return {
            "protocol_version": CLOSED_WORKER_PROTOCOL_VERSION,
            "sequence": sequence,
            "status": "error",
            "payload": dict(payload),
        }


def _read_exact(
    descriptor: int,
    size: int,
    *,
    liveness_descriptor: int,
    timeout_ms: int,
) -> bytes:
    deadline = time.monotonic() + timeout_ms / 1000.0
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        timeout = deadline - time.monotonic()
        if timeout <= 0:
            raise TimeoutError("control channel idle timeout")
        readable, _, _ = select.select([descriptor, liveness_descriptor], [], [], timeout)
        if liveness_descriptor in readable:
            os.read(liveness_descriptor, 1)
            raise BrokenPipeError("parent-liveness sentinel closed")
        if descriptor not in readable:
            raise TimeoutError("control channel idle timeout")
        chunk = os.read(descriptor, remaining)
        if not chunk:
            raise EOFError("control channel EOF")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _write_all(descriptor: int, payload: bytes, *, timeout_ms: int) -> None:
    deadline = time.monotonic() + timeout_ms / 1000.0
    view = memoryview(payload)
    while view:
        timeout = deadline - time.monotonic()
        if timeout <= 0:
            raise TimeoutError("response write timeout")
        _, writable, _ = select.select([], [descriptor], [], timeout)
        if descriptor not in writable:
            raise TimeoutError("response write timeout")
        count = os.write(descriptor, view)
        if count <= 0:
            raise BrokenPipeError("response channel closed")
        view = view[count:]


def _journal(descriptor: int, payload: Mapping[str, object]) -> None:
    try:
        encoded = encode_framed_message(dict(payload), max_bytes=MAX_RESPONSE_BYTES)
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    except BaseException:
        pass


def run_worker_loop(
    *,
    command_descriptor: int,
    liveness_descriptor: int,
    journal_descriptor: int,
    manifest: ClosedWorkerRuntimeManifest,
) -> int:
    session = ClosedWorkerSession(manifest)
    while True:
        sequence = 0
        try:
            header = _read_exact(
                command_descriptor,
                FRAME_HEADER_BYTES,
                liveness_descriptor=liveness_descriptor,
                timeout_ms=manifest.control_idle_timeout_ms,
            )
            size = decode_frame_header(header, max_bytes=MAX_REQUEST_BYTES)
            encoded = _read_exact(
                command_descriptor,
                size,
                liveness_descriptor=liveness_descriptor,
                timeout_ms=manifest.control_idle_timeout_ms,
            )
            raw = decode_canonical_message(encoded, max_bytes=MAX_REQUEST_BYTES)
            if isinstance(raw, Mapping):
                candidate = raw.get("sequence")
                if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0:
                    sequence = candidate
            response = session.handle(raw)
            frame = encode_framed_message(response, max_bytes=MAX_RESPONSE_BYTES)
            _write_all(
                command_descriptor,
                frame,
                timeout_ms=manifest.response_write_timeout_ms,
            )
            session.commit_written_response(response["sequence"])
            if session.terminal:
                return 0
            continue
        except TimeoutError as exc:
            fault = "control_idle_timeout" if "idle" in str(exc) else "response_write_timeout"
        except EOFError:
            exc = EOFError("control channel EOF")
            fault = "channel_eof"
        except BrokenPipeError as broken:
            exc = broken
            fault = "parent_lost" if "liveness" in str(broken) else "broken_pipe"
        except ClosedWorkerProtocolError as protocol_error:
            exc = protocol_error
            fault = "request_protocol_failure"
        except BaseException as unexpected:
            exc = unexpected
            fault = "dispatcher_failure"

        _journal(
            journal_descriptor,
            {
                "kind": "worker_loop_fault",
                "fault": fault,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        try:
            response = session.terminal_fault_response(
                sequence=sequence,
                code=type(exc).__name__,
                message=str(exc),
                fault=fault,
                detection_phase="request_loop",
            )
            frame = encode_framed_message(response, max_bytes=MAX_RESPONSE_BYTES)
            _write_all(
                command_descriptor,
                frame,
                timeout_ms=manifest.response_write_timeout_ms,
            )
        except BaseException as termination_error:
            _journal(
                journal_descriptor,
                {
                    "kind": "termination_handler_failure",
                    "exception_type": type(termination_error).__name__,
                    "message": str(termination_error),
                },
            )
        if session.termination_evidence is not None:
            _journal(
                journal_descriptor,
                {
                    "kind": "transport_termination",
                    "evidence": session.termination_evidence.to_payload(),
                    "evidence_sha256": session.termination_evidence.sha256,
                },
            )
        if session.hard_exit_required:
            os._exit(74)
        return 64
