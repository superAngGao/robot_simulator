"""CPU guarded-wait state authority and nonblocking transport (not C0 adoption)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace

from .protocol import canonical_json_bytes, canonical_sha256, decode_canonical_message, encode_framed_message
from .wait_protocol import (
    MAX_BODY_BYTES,
    MAX_GUARDS,
    MAX_SEQUENCE,
    ROLES,
    GuardConfiguration,
    GuardFrameDecoder,
    GuardProtocolError,
    bounded_integer,
    checked_add,
    exact_fields,
    next_sequence,
    validate_message,
)

WAIT_EVIDENCE_VERSION = "q54.closed-worker.guarded-wait-evidence.v1"
TERMINAL_REASONS = frozenset(
    {
        "parent_lost",
        "input_eof",
        "invalid_message",
        "request_out_of_order",
        "ack_write_failed",
        "sequence_exhausted",
        "process_observation_failed",
    }
)
KILL_ERRORS = frozenset({"permission_denied", "process_missing", "other_os_error"})
CLOCK_ERRORS = frozenset({"backward", "out_of_domain", "deadline_overflow"})


def _enum(value: object, allowed: frozenset[str]) -> str:
    if type(value) is not str or value not in allowed:
        raise GuardProtocolError("unknown transition enum")
    return value


@dataclass(frozen=True, slots=True)
class _State:
    state: str = "IDLE"
    next_sequence: int | None = 0
    active: tuple[int, str] | None = None
    pending: tuple[str, int, str, int] | None = None
    operation_deadline_ns: int | None = None
    terminal_deadline_ns: int | None = None
    terminal_reason: str | None = None
    arm_count: int = 0
    normal_complete_count: int = 0
    deadline_expired: bool = False
    kill_requested: bool = False
    kill_error_kind: str | None = None
    observed_returncode: int | None = None
    clock_rejected: bool = False

    def deadlines(self) -> list[tuple[int, str]]:
        # Sorted ties prefer terminal, then ACK, then operation.
        ordered = [
            (self.terminal_deadline_ns, "terminal"),
            (None if self.pending is None else self.pending[3], "ack"),
            (self.operation_deadline_ns, "operation"),
        ]
        return [(value, name) for value, name in ordered if value is not None]

    def earliest(self) -> tuple[int, str] | None:
        return min(self.deadlines(), key=lambda pair: pair[0], default=None)

    def summary(self) -> dict:
        earliest = self.earliest()
        return {
            "state": self.state,
            "next_sequence": self.next_sequence,
            "active_wait": None
            if self.active is None
            else {
                "sequence": self.active[0],
                "role": self.active[1],
            },
            "pending_ack": None
            if self.pending is None
            else {
                "operation": self.pending[0],
                "sequence": self.pending[1],
                "role": self.pending[2],
                "deadline_ns": self.pending[3],
            },
            "operation_deadline_ns": self.operation_deadline_ns,
            "terminal_deadline_ns": self.terminal_deadline_ns,
            "effective_deadline_ns": None if earliest is None else earliest[0],
            "terminal_reason": self.terminal_reason,
            "arm_count": self.arm_count,
            "normal_complete_count": self.normal_complete_count,
            "deadline_expired": self.deadline_expired,
            "kill_requested": self.kill_requested,
            "kill_error_kind": self.kill_error_kind,
            "observed_returncode": self.observed_returncode,
            "complete": self.state == "EXITED",
        }


def _identity(details: dict) -> tuple[int, str]:
    return bounded_integer(details["sequence"], MAX_SEQUENCE), _enum(details["role"], ROLES)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GuardProtocolError(message)


def _transition(
    state: _State, event: dict, config: GuardConfiguration, *, expected_index: int, last_ns: int
) -> _State:
    exact_fields(event, {"index", "at_ns", "operation", "details"})
    _require(bounded_integer(event["index"]) == expected_index, "trace index drift")
    now = bounded_integer(event["at_ns"])
    _require(now >= last_ns, "trace time went backward")
    operation, details = event["operation"], event["details"]
    _require(type(operation) is str, "operation must be a string")
    if expected_index == 0:
        _require(operation == "initialized", "trace must begin with initialization")
        exact_fields(details, set())
        return state
    _require(state.state != "EXITED", "no event may follow observed exit")
    earliest = state.earliest()
    if operation in {"arm_committed", "ack_written", "normal_complete_received"}:
        _require(earliest is None or now < earliest[0], "expiry precedes protocol success")
    if operation == "arm_committed":
        exact_fields(details, {"sequence", "role"})
        sequence, role = _identity(details)
        _require(state.state == "IDLE" and sequence == state.next_sequence, "ARM is out of order")
        # Four normal events plus complete terminal tail fit before commitment.
        checked_add(expected_index, 4 + 6)
        return replace(
            state,
            state="ARM_ACK_PENDING",
            active=(sequence, role),
            next_sequence=next_sequence(sequence),
            arm_count=checked_add(state.arm_count, 1, MAX_GUARDS),
            operation_deadline_ns=checked_add(now, config.operation_timeout_ms * 1_000_000),
            pending=("ARMED", sequence, role, checked_add(now, config.ack_timeout_ms * 1_000_000)),
        )
    if operation == "ack_written":
        exact_fields(details, {"operation", "sequence", "role"})
        sequence, role = _identity(details)
        _require(
            state.pending is not None and state.pending[:3] == (details["operation"], sequence, role),
            "ACK does not bind exact pending response",
        )
        if state.state == "ARM_ACK_PENDING" and details["operation"] == "ARMED":
            return replace(
                state, state="ARMED_TERMINAL" if role == "emergency" else "ARMED_NORMAL", pending=None
            )
        _require(
            state.state == "DISARM_ACK_PENDING" and details["operation"] == "DISARMED",
            "ACK transport commit is out of order",
        )
        return replace(
            state,
            state="IDLE",
            active=None,
            pending=None,
            operation_deadline_ns=None,
            normal_complete_count=checked_add(state.normal_complete_count, 1, MAX_GUARDS),
        )
    if operation == "normal_complete_received":
        exact_fields(details, {"sequence", "role"})
        identity = _identity(details)
        _require(
            state.state == "ARMED_NORMAL" and identity == state.active and identity[1] != "emergency",
            "COMPLETE does not bind the outstanding normal wait",
        )
        return replace(
            state,
            state="DISARM_ACK_PENDING",
            pending=("DISARMED", *identity, checked_add(now, config.ack_timeout_ms * 1_000_000)),
        )
    if operation == "terminal_entered":
        exact_fields(details, {"reason"})
        reason = _enum(details["reason"], TERMINAL_REASONS)
        _require(state.state != "TERMINATING", "terminal transition cannot repeat")
        grace = checked_add(now, config.operation_timeout_ms * 1_000_000)
        deadline = min(grace, earliest[0]) if earliest is not None else grace
        return replace(
            state, state="TERMINATING", pending=None, terminal_reason=reason, terminal_deadline_ns=deadline
        )
    if operation == "deadline_expired":
        exact_fields(details, {"source"})
        _require(
            not state.deadline_expired
            and earliest is not None
            and now >= earliest[0]
            and details["source"] == earliest[1],
            "invalid/duplicate deadline expiry",
        )
        return replace(
            state,
            state="TERMINATING",
            pending=None,
            deadline_expired=True,
            terminal_reason=state.terminal_reason or "deadline_expired",
            terminal_deadline_ns=earliest[0],
        )
    if operation == "kill_requested":
        exact_fields(details, set())
        _require(
            state.state == "TERMINATING" and state.deadline_expired and not state.kill_requested,
            "kill attempt requires a new expired terminal obligation",
        )
        return replace(state, kill_requested=True)
    if operation == "kill_failed":
        exact_fields(details, {"error_kind"})
        error = _enum(details["error_kind"], KILL_ERRORS)
        _require(
            state.state == "TERMINATING" and state.kill_requested and state.kill_error_kind is None,
            "kill failure has no unique preceding attempt",
        )
        return replace(state, kill_error_kind=error)
    if operation == "process_exited":
        exact_fields(details, {"returncode"})
        code = bounded_integer(details["returncode"], 2**31 - 1, minimum=-(2**31))
        return replace(
            state,
            state="EXITED",
            active=None,
            pending=None,
            operation_deadline_ns=None,
            terminal_deadline_ns=None,
            observed_returncode=code,
        )
    if operation == "clock_rejected":
        exact_fields(details, {"reason"})
        _enum(details["reason"], CLOCK_ERRORS)
        _require(not state.clock_rejected, "clock rejection must be single-use")
        deadline = min(now, earliest[0]) if earliest is not None else now
        return replace(
            state,
            state="TERMINATING",
            pending=None,
            clock_rejected=True,
            terminal_reason=state.terminal_reason or "clock_fault",
            terminal_deadline_ns=deadline,
        )
    raise GuardProtocolError("unknown lifecycle event")


@dataclass(frozen=True, slots=True)
class GuardSupervisorEvidence:
    """Canonical consistency evidence; never a live process/ticket authority."""

    _encoded: bytes

    @classmethod
    def from_payload(cls, value: object, *, require_final: bool = False) -> GuardSupervisorEvidence:
        root = exact_fields(value, {"schema_version", "configuration", "trace", "summary"})
        _require(root["schema_version"] == WAIT_EVIDENCE_VERSION, "wrong wait evidence version")
        config = GuardConfiguration.from_payload(root["configuration"])
        trace = root["trace"]
        _require(type(trace) is list and bool(trace), "evidence needs initialization/full trace")
        state, last = _State(), 0
        for index, event in enumerate(trace):
            state = _transition(state, event, config, expected_index=index, last_ns=last)
            last = event["at_ns"]
        # Canonical bytes also distinguish True from1 in forged summaries.
        _require(
            canonical_json_bytes(root["summary"]) == canonical_json_bytes(state.summary()),
            "summary does not equal replayed state",
        )
        _require(not require_final or state.state == "EXITED", "snapshot is not final evidence")
        return cls(canonical_json_bytes(root))

    def to_payload(self) -> dict:
        return decode_canonical_message(self._encoded, max_bytes=len(self._encoded))

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_payload())


class GuardSupervisorCore:
    """Thread-owned CPU state machine; caller owns actual process custody/I/O."""

    def __init__(self, configuration: GuardConfiguration, *, clock=time.monotonic_ns) -> None:
        _require(type(configuration) is GuardConfiguration, "exact guard configuration required")
        self._configuration = GuardConfiguration.from_payload(configuration.to_payload())
        self._clock = clock
        self._thread = threading.current_thread()
        self._state = _State()
        self._trace: list[dict] = []
        self._last_ns = bounded_integer(clock())
        self._last_observed_ns = self._last_ns
        self._reading_clock = False
        self._append("initialized", {}, self._last_ns)

    @property
    def configuration(self) -> GuardConfiguration:
        return self._configuration

    @property
    def state(self) -> str:
        return self._state.state

    @property
    def pending_response(self) -> dict | None:
        pending = self._state.pending
        return None if pending is None else self.configuration.message(pending[0], pending[1], pending[2])

    @property
    def effective_deadline_ns(self) -> int | None:
        first = self._state.earliest()
        return None if first is None else first[0]

    @property
    def kill_required(self) -> bool:
        return self.state == "TERMINATING" and self._state.deadline_expired and not self._state.kill_requested

    def _check_thread(self) -> None:
        _require(threading.current_thread() is self._thread, "supervisor core belongs to another thread")

    def _append(self, operation: str, details: dict, now: int) -> None:
        event = {"index": len(self._trace), "at_ns": now, "operation": operation, "details": details}
        updated = _transition(
            self._state, event, self.configuration, expected_index=len(self._trace), last_ns=self._last_ns
        )
        self._trace.append(event)
        self._state = updated
        self._last_ns = now

    def _reject_clock(self, reason: str) -> int:
        if self.state != "EXITED" and not self._state.clock_rejected:
            self._append("clock_rejected", {"reason": reason}, self._last_observed_ns)
        return self._last_ns

    def _now(self) -> int:
        self._check_thread()
        _require(not self._reading_clock, "reentrant supervisor clock entry")
        self._reading_clock = True
        try:
            now = bounded_integer(self._clock())
        except BaseException:
            return self._reject_clock("out_of_domain")
        finally:
            self._reading_clock = False
        if now < self._last_observed_ns:
            return self._reject_clock("backward")
        self._last_observed_ns = now
        return now

    def tick(self) -> None:
        self._check_thread()
        if self.state == "EXITED":
            return
        now = self._now()
        first = self._state.earliest()
        if not self._state.deadline_expired and first is not None and now >= first[0]:
            self._append("deadline_expired", {"source": first[1]}, now)

    def fault(self, reason: str) -> None:
        self._check_thread()
        _enum(reason, TERMINAL_REASONS)
        self.tick()
        if self.state in {"TERMINATING", "EXITED"}:
            return
        now = self._now()
        if self.state == "TERMINATING":
            self.tick()
            return
        try:
            self._append("terminal_entered", {"reason": reason}, now)
        except GuardProtocolError:
            self._reject_clock("deadline_overflow")
            self.tick()

    def accept(self, raw: object) -> dict:
        self._check_thread()
        self.tick()
        _require(self.state not in {"TERMINATING", "EXITED"}, "supervisor is terminal")
        try:
            message = validate_message(raw, self.configuration)
        except BaseException:
            self.fault("invalid_message")
            raise
        if message["operation"] == "ARM" and self._state.next_sequence is None:
            self.fault("sequence_exhausted")
            raise GuardProtocolError("guard sequence exhausted")
        operation = {"ARM": "arm_committed", "COMPLETE": "normal_complete_received"}.get(message["operation"])
        if operation is None:
            self.fault("request_out_of_order")
            raise GuardProtocolError("supervisor accepts requests only")
        now = self._now()
        if self.state == "TERMINATING":
            self.tick()
            raise GuardProtocolError("clock rejected before request")
        try:
            # Precheck deadline arithmetic separately to distinguish ordering errors.
            checked_add(now, self.configuration.ack_timeout_ms * 1_000_000)
            if operation == "arm_committed":
                checked_add(now, self.configuration.operation_timeout_ms * 1_000_000)
        except GuardProtocolError:
            self._reject_clock("deadline_overflow")
            self.tick()
            raise
        try:
            self._append(operation, {"sequence": message["sequence"], "role": message["role"]}, now)
        except GuardProtocolError:
            self.fault("request_out_of_order")
            raise
        assert self.pending_response is not None
        return self.pending_response

    def acknowledge_written(self, response: object) -> None:
        self._check_thread()
        self.tick()
        try:
            observed = validate_message(response, self.configuration)
            _require(
                self.pending_response is not None and observed == self.pending_response,
                "ACK write does not bind pending output",
            )
            now = self._now()
            self._append(
                "ack_written", {name: observed[name] for name in ("operation", "sequence", "role")}, now
            )
        except BaseException:
            self.fault("ack_write_failed")
            raise

    def record_kill_request(self) -> None:
        self._check_thread()
        self.tick()
        self._append("kill_requested", {}, self._now())

    def record_kill_failure(self, error_kind: str) -> None:
        self._check_thread()
        self._append("kill_failed", {"error_kind": error_kind}, self._now())

    def observe_exit(self, returncode: int) -> None:
        self._check_thread()
        self._append("process_exited", {"returncode": returncode}, self._now())

    def evidence(self, *, require_final: bool = False) -> GuardSupervisorEvidence:
        self._check_thread()
        return GuardSupervisorEvidence.from_payload(
            {
                "schema_version": WAIT_EVIDENCE_VERSION,
                "configuration": self.configuration.to_payload(),
                "trace": list(self._trace),
                "summary": self._state.summary(),
            },
            require_final=require_final,
        )


class GuardSupervisorTransport:
    """One nonblocking socket and bounded input/output; process loop stays outside."""

    def __init__(self, channel, core: GuardSupervisorCore) -> None:
        import socket

        _require(
            type(channel) is socket.socket and type(core) is GuardSupervisorCore,
            "exact private socket and supervisor core required",
        )
        core._check_thread()
        self.channel = channel
        self.core = core
        channel.setblocking(False)
        self._decoder = GuardFrameDecoder()
        self._response: dict | None = None
        self._output = b""
        self._offset = 0

    @property
    def wants_write(self) -> bool:
        return self.core.pending_response is not None

    def service(self, *, readable: bool, writable: bool) -> None:
        self.core._check_thread()
        self.core.tick()
        try:
            if self.core.state in {"TERMINATING", "EXITED"}:
                return
            if readable:
                try:
                    chunk = self.channel.recv(self._decoder.remaining_capacity)
                except BlockingIOError:
                    chunk = None
                except OSError:
                    self.core.fault("invalid_message")
                    raise
                if chunk == b"":
                    self.core.fault("input_eof")
                    return
                if chunk is not None:
                    try:
                        message = self._decoder.feed(chunk)
                    except BaseException:
                        self.core.fault("invalid_message")
                        raise
                    if message is not None:
                        if self.core.pending_response is not None:
                            self.core.fault("request_out_of_order")
                            raise GuardProtocolError("request arrived before ACK write commit")
                        self.core.accept(message)
            if writable and self.core.pending_response is not None:
                if self._response is None:
                    self._response = self.core.pending_response
                    self._output = encode_framed_message(self._response, max_bytes=MAX_BODY_BYTES)
                    self._offset = 0
                self.core.tick()
                if self.core.state == "TERMINATING":
                    return
                try:
                    sent = self.channel.send(self._output[self._offset :])
                except BlockingIOError:
                    return
                except OSError:
                    self.core.fault("ack_write_failed")
                    raise
                if sent <= 0:
                    self.core.fault("ack_write_failed")
                    raise GuardProtocolError("ACK output failed")
                self._offset += sent
                if self._offset == len(self._output):
                    self.core.acknowledge_written(self._response)
                    self._response, self._output, self._offset = None, b"", 0
        except BaseException:
            if self.core.state not in {"TERMINATING", "EXITED"}:
                self.core.fault(
                    "ack_write_failed" if self.core.pending_response is not None else "invalid_message"
                )
            raise
        finally:
            self.core.tick()
            if self.core.state in {"TERMINATING", "EXITED"}:
                self._response, self._output, self._offset = None, b"", 0
