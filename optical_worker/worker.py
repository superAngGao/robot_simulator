"""Closed-worker request loop for the C4c0 CPU/control-plane slice."""

from __future__ import annotations

import hashlib
import os
import secrets
import select
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .lifecycle import NativeWorkerSetupLifecycle, TransportTerminationEvidence
from .manifest import validate_repository_import_ledger
from .native_setup import NativeWorkerPreparationAborted
from .protocol import (
    CLOSED_WORKER_EVIDENCE_VERSION,
    FRAME_HEADER_BYTES,
    INTEGRATED_WORKER_EVIDENCE_VERSION,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    ClosedNativeLaneWorkerEvidence,
    ClosedWorkerProtocolError,
    ClosedWorkerRuntimeManifest,
    IntegratedWorkerRuntimeManifest,
    canonical_json_bytes,
    canonical_sha256,
    decode_canonical_message,
    decode_frame_header,
    encode_framed_message,
    validate_request,
)

_NATIVE_WORKERS = {}


def _no_legacy_abort(_device_ordinal):
    raise ClosedWorkerProtocolError("integrated workers cannot use the legacy A/K channel")


@dataclass(eq=False, slots=True)
class _WorkerSample:
    index: int
    prepared_token: str
    pending_token: str
    resolved_token: str
    plan: object = None
    active: object = None
    state: str = "PREPARED"
    submission_possible: bool = False


class ClosedWorkerSession:
    """One connection, one session, and at most one frozen-row preparation."""

    def __init__(
        self,
        manifest: ClosedWorkerRuntimeManifest,
        *,
        emergency_sync_notifier: Callable[[int], None],
    ) -> None:
        self._manifest = manifest
        self._state = "EXPECT_HELLO"
        self._next_sequence = 0
        self._session_identity = f"session-{secrets.token_hex(16)}"
        self._lifecycle: NativeWorkerSetupLifecycle | None = None
        self._worker_evidence: ClosedNativeLaneWorkerEvidence | None = None
        self._termination: TransportTerminationEvidence | None = None
        self._close_ack_sequence: int | None = None
        self._emergency_sync_notifier = emergency_sync_notifier
        self._native_source = None
        self._graph = None
        self._samples = ()
        self._tokens = {}
        self._active_sample = None
        self._next_sample = 0
        self._cancel_only = False
        self._native_operation = None
        self._handling = False
        self._idle_proof = None

    @classmethod
    def from_source_session(cls, manifest, source):
        """Bind the same serial dispatcher to its captured native source owner."""
        from .manifest import validate_integrated_worker_runtime_manifest

        if type(manifest) is not IntegratedWorkerRuntimeManifest:
            raise ClosedWorkerProtocolError("native worker requires the exact v3 manifest")
        manifest = IntegratedWorkerRuntimeManifest.from_payload(manifest.to_payload())
        source.check_live()
        validate_integrated_worker_runtime_manifest(manifest, source)
        controller = source.bound_controller()
        if source in _NATIVE_WORKERS:
            raise ClosedWorkerProtocolError("captured source session already bound its one worker")
        _NATIVE_WORKERS[source] = None
        worker = cls(manifest, emergency_sync_notifier=_no_legacy_abort)
        worker._native_source = source
        worker._lifecycle = controller
        worker._session_identity = controller._session_identity
        source._hold_runtime_owner(worker, "closed_worker_session")
        _NATIVE_WORKERS[source] = worker
        controller.bind_worker(worker)
        return worker

    def _require_native_binding(self):
        source = self._native_source
        if (source is None or _NATIVE_WORKERS.get(source) is not self
                or source.bound_controller() is not self._lifecycle):
            raise ClosedWorkerProtocolError("worker has no exact source/controller binding")
        self._lifecycle._check_running()
        return source

    @property
    def terminal(self) -> bool:
        return self._state in {"TERMINATED", "CLOSE_ACK_COMMITTED"}

    @property
    def hard_exit_required(self) -> bool:
        if self._termination is None:
            return False
        disposition = (self._termination.disposition if self._native_source is None
                       else self._termination.lifecycle.to_payload()["summary"]["disposition"])
        return disposition in {
            "quarantined_process_exit",
            "multi_or_unknown_device_quarantine_exit",
            "cleanup_failed_exit",
        }

    @property
    def termination_evidence(self) -> TransportTerminationEvidence | None:
        return self._termination

    def adopt_preparation_abort(self, aborted: NativeWorkerPreparationAborted) -> None:
        if self._native_source is not None:
            if aborted.evidence is not self._lifecycle.termination_evidence:
                raise ClosedWorkerProtocolError("native abort is not the actual controller disposition")
            self._termination = aborted.evidence
            self._state = "TERMINATED"
            return
        evidence = TransportTerminationEvidence.from_payload(aborted.evidence.to_payload())
        if (
            self._lifecycle is None
            or evidence != aborted.evidence
            or evidence.session_identity != self._session_identity
            or evidence.lane_id != self._manifest.lane_id
            or evidence.target_device_ordinal != self._manifest.device_ordinal
            or evidence.disposition
            not in {
                "aborted_synced_exit",
                "quarantined_process_exit",
                "multi_or_unknown_device_quarantine_exit",
            }
        ):
            raise ClosedWorkerProtocolError("native preparation abort evidence is not session-bound")
        self._termination = evidence
        self._state = "TERMINATED"

    def handle(self, raw: object) -> dict[str, object]:
        if self._handling:
            raise ClosedWorkerProtocolError("worker request dispatch is not reentrant")
        if self._native_source is not None:
            self._lifecycle._check_owner()
        self._handling = True
        try:
            return self._handle(raw)
        finally:
            self._handling = False

    def _handle(self, raw: object) -> dict[str, object]:
        request = validate_request(raw, expected_version=self._manifest.protocol_version)
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
        if self._native_source is not None:
            return self._handle_native(sequence, operation, arguments)
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
            if self._native_source is None:
                self._lifecycle.commit_clean_close()
            else:
                self._lifecycle.commit_clean_close(_worker=self, _proof=self._idle_proof)
                self._termination = self._lifecycle.termination_evidence
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
        if self._native_source is not None and self._termination is None:
            from .native_setup import _fail_optical_preparation

            _fail_optical_preparation(
                self._native_source, sys.exception() or ClosedWorkerProtocolError(message),
                source_integrity_failed=self._native_source.failed,
                fault=fault, detection_phase=detection_phase,
            )
            self._termination = self._lifecycle.termination_evidence
        elif self._lifecycle is not None and self._termination is None:
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
        if self._native_source is None:
            self._lifecycle = NativeWorkerSetupLifecycle(
                lane_id=self._manifest.lane_id,
                target_device_ordinal=self._manifest.device_ordinal,
                emergency_sync_notifier=self._emergency_sync_notifier,
                session_identity=self._session_identity,
            )
        else:
            self._require_native_binding().audit()
        repository_paths = validate_repository_import_ledger(self._manifest)
        executable_digest = hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
        evidence = ClosedNativeLaneWorkerEvidence(
            schema_version=(CLOSED_WORKER_EVIDENCE_VERSION if self._native_source is None
                            else INTEGRATED_WORKER_EVIDENCE_VERSION),
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
            gpu_preparation_enabled=self._native_source is not None,
            cuda_capable_module_names=tuple(
                sorted(name for name in ("numpy", "torch", "warp") if name in sys.modules)
            ),
            repository_module_paths=repository_paths,
            repository_import_ledger_sha256=canonical_sha256(list(repository_paths)),
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
        self._state = "SESSION_OPEN_IDLE" if self._native_source is None else "IDLE_UNPREPARED"
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

    def _ok(self, sequence: int, payload: Mapping[str, object]) -> dict[str, object]:
        return {
            "protocol_version": self._manifest.protocol_version,
            "sequence": sequence,
            "status": "ok",
            "payload": dict(payload),
        }

    def _error(self, sequence: int, payload: Mapping[str, object]) -> dict[str, object]:
        return {
            "protocol_version": self._manifest.protocol_version,
            "sequence": sequence,
            "status": "error",
            "payload": dict(payload),
        }

    def _handle_native(self, sequence, operation, arguments):
        self._require_native_binding()
        if operation == "PREPARE_FROZEN_ROW" and self._state == "IDLE_UNPREPARED":
            return self._prepare_native(sequence, arguments)
        if self._state not in {"PREPARED_IDLE", "SAMPLE_ACTIVE", "DRAINED_IDLE"}:
            raise ClosedWorkerProtocolError("operation is not legal in the current native worker state")
        if operation == "EXECUTE_PREPARED_SAMPLE":
            return self._execute_native(sequence, arguments["prepared_token"])
        if operation == "RESOLVE_PREPARED_SAMPLE":
            return self._resolve_native(sequence, arguments["pending_token"])
        if operation == "RELEASE_RESOLVED_SAMPLE":
            return self._release_native(sequence, arguments["resolved_token"])
        if operation == "CANCEL_PREPARED_SAMPLE":
            return self._cancel_native(sequence, arguments["prepared_token"])
        if operation == "CLOSE_SESSION":
            self._idle_proof = self._lifecycle.observe_post_reclaim_idle(self)
            self._state = "CLOSE_ACK_PENDING"
            self._close_ack_sequence = sequence
            return self._ok(sequence, {"state": self._state, "worker_exit": True,
                                      "post_reclaim_idle": self._idle_proof.to_payload()})
        raise ClosedWorkerProtocolError("native preparation is single-use and operations are state-bound")

    def _prepare_native(self, sequence, arguments):
        from .arena import select_native_worker_row

        source = self._require_native_binding()
        try:
            row = select_native_worker_row(
                row_id=arguments["row_id"], case_id=arguments["case_id"],
                sample_count=arguments["sample_count"], pass_mode=self._manifest.pass_mode,
            )
        except ValueError as error:
            raise ClosedWorkerProtocolError("stage_scope_rejected: " + str(error)) from error
        if arguments["case_id"] != self._manifest.case_id:
            raise ClosedWorkerProtocolError("PREPARE case differs from the frozen native session")
        count = row.sample_count
        payload = {"state": "PREPARED_IDLE", "prepared_tokens": [], "graph_sha256": "0" * 64}
        # Each fixed-width token contributes 34 JSON bytes plus a separator.
        if len(canonical_json_bytes(self._ok(sequence, payload))) + count * 35 - 1 > MAX_RESPONSE_BYTES:
            raise ClosedWorkerProtocolError("prepared response exceeds the fixed protocol capacity")
        self._state = "PREPARING"
        records = []
        for index in range(count):
            record = _WorkerSample(index, secrets.token_hex(16), secrets.token_hex(16), secrets.token_hex(16))
            records.append(record)
            for token, phase in ((record.prepared_token, "PREPARED"),
                                 (record.pending_token, "COMPLETION_RECORDED"),
                                 (record.resolved_token, "RESOLVED_HELD")):
                if token in self._tokens:
                    raise ClosedWorkerProtocolError("worker token collision")
                self._tokens[token] = record, phase
        self._samples = tuple(records)
        payload["prepared_tokens"] = [record.prepared_token for record in self._samples]
        source.configure_external_environment(
            source.runtime_observation.environment.to_payload(), self._manifest.runtime_environment_sha256,
        )
        source.materialize_native_source(
            arguments["case_id"], go2_model_dir=self._manifest.go2_model_dir,
            row_id=arguments["row_id"], sample_count=count, pass_mode=self._manifest.pass_mode,
        )
        binding = source.initialize_native_runtime()
        source.upload_native_input(binding)
        source.validate_native_input()
        if source.lane == "warp":
            source.prepare_warp_modules()
        else:
            source.prepare_cuda_factories()
        source.prepare_fixed_render_storage(
            traversal_diagnostics=self._manifest.pass_mode == "diagnostic_profiled",
        )
        source.prepare_fixed_render_graph(
            profile=row.row["output_profile"], shadows=row.row["shadows"], pass_mode=self._manifest.pass_mode,
        )
        source.run_fixed_render_warmups()
        self._graph = source.prepare_fixed_final_graph(sample_count=count)
        for record, plan in zip(self._samples, self._graph.samples, strict=True):
            record.plan = plan
        payload["graph_sha256"] = self._graph.sha256
        self._lifecycle.record_worker_transition(self, "worker_prepared", {
            "sample_count": count, "graph_sha256": self._graph.sha256,
            "token_registry_sha256": self._token_registry_sha256(),
        })
        self._state = "PREPARED_IDLE"
        return self._ok(sequence, payload)

    def _sample_for_token(self, token, phase):
        registered = self._tokens.get(token)
        if (registered is None or registered[1] != phase or registered[0].state != phase
                or self._samples[registered[0].index] is not registered[0]):
            raise ClosedWorkerProtocolError("sample token is foreign, consumed or in the wrong state")
        return registered[0]

    def _execute_native(self, sequence, token):
        record = self._sample_for_token(token, "PREPARED")
        if self._cancel_only or self._active_sample is not None or record.index != self._next_sample:
            raise ClosedWorkerProtocolError("EXECUTE requires the next FIFO sample and a free slot")
        active = self._graph.arena.acquire(record.plan)
        from optics.native_arena import NativeOpticalArenaBackpressure

        if isinstance(active, NativeOpticalArenaBackpressure):
            raise ClosedWorkerProtocolError("worker's single slot is not free")
        record.active, record.state = active, "ACQUIRED"
        self._active_sample = record
        self._native_operation = "execute"
        self._state = "SAMPLE_ACTIVE"
        try:
            self._lifecycle.record_worker_transition(self, "sample_acquired", {"sample_index": record.index})
            active.enqueue()
            active.finish()
            if active.state != "COMPLETION_RECORDED" or not record.submission_possible:
                raise ClosedWorkerProtocolError("pending token has no actual terminal recording")
            record.state = "COMPLETION_RECORDED"
            self._lifecycle.record_worker_transition(
                self, "sample_completion_recorded", {"sample_index": record.index},
            )
            self._next_sample += 1
        except BaseException:
            if not record.submission_possible and active.state == "ACQUIRED":
                active.cancel_before_enqueue()
            self._state = "POISONED"
            raise
        finally:
            self._native_operation = None
        return self._ok(sequence, {"pending_token": record.pending_token})

    def _resolve_native(self, sequence, token):
        record = self._sample_for_token(token, "COMPLETION_RECORDED")
        if self._active_sample is not record:
            raise ClosedWorkerProtocolError("pending token is not the active generation")
        record.active.resolve(_guarded_controller=self._lifecycle)
        record.state = "RESOLVED_HELD"
        self._lifecycle.record_worker_transition(self, "sample_resolved", {"sample_index": record.index})
        return self._ok(sequence, {"resolved_token": record.resolved_token})

    def _release_native(self, sequence, token):
        record = self._sample_for_token(token, "RESOLVED_HELD")
        if self._active_sample is not record:
            raise ClosedWorkerProtocolError("resolved token is not the active generation")
        record.active.release_device_result()
        record.active.reclaim()
        record.state = "RECLAIMED"
        self._active_sample = None
        self._lifecycle.record_worker_transition(self, "sample_reclaimed", {"sample_index": record.index})
        self._update_native_idle()
        return self._ok(sequence, {"state": "RECLAIMED"})

    def _cancel_native(self, sequence, token):
        record = self._sample_for_token(token, "PREPARED")
        if not self._cancel_only:
            self._graph.arena.begin_drain()
            self._cancel_only = True
        record.state = "CANCELLED"
        self._lifecycle.record_worker_transition(self, "sample_cancelled", {"sample_index": record.index})
        self._update_native_idle()
        return self._ok(sequence, {"state": "CANCELLED"})

    def _update_native_idle(self):
        self._state = ("SAMPLE_ACTIVE" if self._active_sample is not None else
                       "DRAINED_IDLE" if all(s.state in {"RECLAIMED", "CANCELLED"} for s in self._samples)
                       else "PREPARED_IDLE")

    def _token_registry_sha256(self):
        return canonical_sha256([
            {"index": s.index, "prepared": s.prepared_token, "pending": s.pending_token,
             "resolved": s.resolved_token, "state": s.state,
             "submission_possible": s.submission_possible} for s in self._samples
        ])


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
    abort_signal_descriptor: int,
    abort_ack_descriptor: int,
    manifest: ClosedWorkerRuntimeManifest,
    _source_session=None,
) -> int:
    abort_signaled = False

    def signal_abort_start(device_ordinal: int) -> None:
        nonlocal abort_signaled
        if abort_signaled or device_ordinal != manifest.device_ordinal:
            raise ClosedWorkerProtocolError("abort-start signal is duplicated or targets another device")
        if os.write(abort_signal_descriptor, b"A") != 1:
            raise BrokenPipeError("abort-start signal was not delivered")
        readable, _, _ = select.select(
            [abort_ack_descriptor],
            [],
            [],
            manifest.abort_watchdog_ms / 1000.0,
        )
        if abort_ack_descriptor not in readable:
            raise TimeoutError("abort-watchdog acknowledgement timeout")
        acknowledgement = os.read(abort_ack_descriptor, 2)
        if acknowledgement != b"K":
            raise BrokenPipeError("abort-watchdog acknowledgement is missing or invalid")
        abort_signaled = True

    validate_repository_import_ledger(manifest)
    session = (ClosedWorkerSession(manifest, emergency_sync_notifier=signal_abort_start)
               if _source_session is None else _source_session.open_worker(manifest))
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
            if _source_session is not None:
                _source_session.audit()
            validate_repository_import_ledger(manifest)
            frame = encode_framed_message(response, max_bytes=MAX_RESPONSE_BYTES)
            _write_all(
                command_descriptor,
                frame,
                timeout_ms=manifest.response_write_timeout_ms,
            )
            session.commit_written_response(response["sequence"])
            if session.terminal:
                if session.hard_exit_required:
                    os._exit(74)
                return 0
            continue
        except NativeWorkerPreparationAborted as aborted:
            session.adopt_preparation_abort(aborted)
            _journal(
                journal_descriptor,
                {
                    "kind": "native_preparation_aborted",
                    "evidence": aborted.evidence.to_payload(),
                    "evidence_sha256": aborted.evidence.sha256,
                },
            )
            if session.hard_exit_required:
                os._exit(74)
            return 66
        except TimeoutError as timeout:
            exc = timeout
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
