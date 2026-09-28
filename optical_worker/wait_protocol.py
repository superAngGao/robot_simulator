"""Inert CPU-only guarded-wait wire protocol; not adopted by the C0 worker."""

from __future__ import annotations

import select
import socket
import threading
import time
from dataclasses import dataclass

from .protocol import decode_canonical_message, decode_frame_header, encode_framed_message

WAIT_PROTOCOL_VERSION = "q54.closed-worker.guarded-wait.v1"
MAX_BODY_BYTES = 1024
I64_MAX = 2**63 - 1
MAX_GUARDS = (I64_MAX - 8) // 4
MAX_SEQUENCE = MAX_GUARDS - 1
ROLES = frozenset({"setup", "resolve", "emergency"})
OPERATIONS = frozenset({"ARM", "ARMED", "COMPLETE", "DISARMED"})


class GuardProtocolError(RuntimeError):
    """Fail-closed CPU protocol violation, not a GPU disposition."""


def bounded_integer(value: object, maximum: int = I64_MAX, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise GuardProtocolError("integer outside the exact protocol domain")
    return value


def checked_add(value: int, increment: int, maximum: int = I64_MAX) -> int:
    bounded_integer(value, maximum)
    bounded_integer(increment, maximum)
    return bounded_integer(value + increment, maximum)


def next_sequence(sequence: int) -> int | None:
    bounded_integer(sequence, MAX_SEQUENCE)
    return None if sequence == MAX_SEQUENCE else sequence + 1


def exact_fields(value: object, fields: set[str]) -> dict:
    if type(value) is not dict or set(value) != fields:
        raise GuardProtocolError("record fields differ from the frozen schema")
    return value


@dataclass(frozen=True, slots=True)
class GuardConfiguration:
    session: str
    device_ordinal: int
    ack_timeout_ms: int
    operation_timeout_ms: int

    def __post_init__(self) -> None:
        if (
            type(self.session) is not str
            or len(self.session) != 32
            or any(c not in "0123456789abcdef" for c in self.session)
        ):
            raise GuardProtocolError("session must be exactly 32 lowercase hex characters")
        bounded_integer(self.device_ordinal, 1023)
        bounded_integer(self.ack_timeout_ms, 2**31 - 1, minimum=1)
        bounded_integer(self.operation_timeout_ms, 2**31 - 1, minimum=1)

    @classmethod
    def from_payload(cls, value: object) -> GuardConfiguration:
        record = exact_fields(
            value,
            {
                "protocol_version",
                "session",
                "device_ordinal",
                "ack_timeout_ms",
                "operation_timeout_ms",
            },
        )
        if record["protocol_version"] != WAIT_PROTOCOL_VERSION:
            raise GuardProtocolError("wrong guarded-wait protocol version")
        return cls(
            **{
                name: record[name]
                for name in (
                    "session",
                    "device_ordinal",
                    "ack_timeout_ms",
                    "operation_timeout_ms",
                )
            }
        )

    def to_payload(self) -> dict:
        return {
            "protocol_version": WAIT_PROTOCOL_VERSION,
            "session": self.session,
            "device_ordinal": self.device_ordinal,
            "ack_timeout_ms": self.ack_timeout_ms,
            "operation_timeout_ms": self.operation_timeout_ms,
        }

    def message(self, operation: str, sequence: int, role: str) -> dict:
        return validate_message(
            {
                "protocol_version": WAIT_PROTOCOL_VERSION,
                "session": self.session,
                "device_ordinal": self.device_ordinal,
                "sequence": sequence,
                "role": role,
                "operation": operation,
            },
            self,
        )


def validate_message(value: object, configuration: GuardConfiguration) -> dict:
    record = exact_fields(
        value,
        {
            "protocol_version",
            "session",
            "device_ordinal",
            "sequence",
            "role",
            "operation",
        },
    )
    bounded_integer(record["device_ordinal"], 1023)
    bounded_integer(record["sequence"], MAX_SEQUENCE)
    if (
        record["protocol_version"] != WAIT_PROTOCOL_VERSION
        or record["session"] != configuration.session
        or record["device_ordinal"] != configuration.device_ordinal
        or type(record["role"]) is not str
        or record["role"] not in ROLES
        or type(record["operation"]) is not str
        or record["operation"] not in OPERATIONS
    ):
        raise GuardProtocolError("guard identity/role/operation mismatch")
    return dict(record)


class GuardFrameDecoder:
    """One bounded frame, never a queue of requests or a blocking reader."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._size: int | None = None

    @property
    def remaining_capacity(self) -> int:
        return MAX_BODY_BYTES + 4 - len(self._buffer)

    def feed(self, chunk: bytes) -> object | None:
        if type(chunk) is not bytes or not chunk or len(chunk) > self.remaining_capacity:
            raise GuardProtocolError("invalid guard input chunk/EOF")
        self._buffer.extend(chunk)
        if self._size is None and len(self._buffer) >= 4:
            self._size = decode_frame_header(bytes(self._buffer[:4]), max_bytes=MAX_BODY_BYTES)
        if self._size is None or len(self._buffer) < self._size + 4:
            return None
        if len(self._buffer) != self._size + 4:
            raise GuardProtocolError("only one guard frame may be buffered")
        result = decode_canonical_message(bytes(self._buffer[4:]), max_bytes=MAX_BODY_BYTES)
        self._buffer.clear()
        self._size = None
        return result


@dataclass(frozen=True, slots=True, eq=False)
class GuardWaitTicket:
    session: str
    device_ordinal: int
    role: str
    sequence: int


class GuardWaitClient:
    """One thread and one outstanding exact-object ticket on a private socket.

    No callable is executed here. A trusted fixed driver must hard-exit with
    its graph retained after any protocol/wait failure. This core cannot prove
    that driver's resource custody and does not authorize GPU preparation.
    """

    def __init__(self, channel: socket.socket, configuration: GuardConfiguration) -> None:
        if type(configuration) is not GuardConfiguration or type(channel) is not socket.socket:
            raise GuardProtocolError("client requires exact configuration/private socket")
        self._configuration = GuardConfiguration.from_payload(configuration.to_payload())
        self._channel = channel
        self._channel.setblocking(False)
        self._thread = threading.current_thread()
        self._state = "IDLE"
        self._sequence: int | None = 0
        self._active: GuardWaitTicket | None = None
        self._last_clock_ns: int | None = None

    @property
    def configuration(self) -> GuardConfiguration:
        return self._configuration

    @property
    def state(self) -> str:
        return self._state

    def _check_thread(self) -> None:
        if threading.current_thread() is not self._thread:
            raise GuardProtocolError("guard client is owned by another thread")

    def fail(self) -> None:
        self._check_thread()
        self._state = "FAILED"

    def close(self) -> None:
        self.fail()
        self._channel.close()

    def arm(self, role: str) -> GuardWaitTicket:
        self._check_thread()
        try:
            if self._state != "IDLE" or self._sequence is None:
                raise GuardProtocolError("guard is not idle or sequence is exhausted")
            request = self.configuration.message("ARM", self._sequence, role)
            self._state = "ARM_PENDING"
            self._exchange(request, "ARMED")
            if self._state != "ARM_PENDING":
                raise GuardProtocolError("nested failure invalidated pending ARM")
            ticket = GuardWaitTicket(
                self.configuration.session, self.configuration.device_ordinal, role, self._sequence
            )
            self._active = ticket
            self._sequence = next_sequence(self._sequence)
            self._state = "ARMED"
            return ticket
        except BaseException:
            self.fail()
            raise

    def complete(self, ticket: GuardWaitTicket) -> None:
        self._check_thread()
        try:
            if (
                self._state != "ARMED"
                or ticket is not self._active
                or type(ticket) is not GuardWaitTicket
                or ticket.role == "emergency"
            ):
                raise GuardProtocolError("only the exact live normal ticket may complete")
            self._state = "COMPLETE_PENDING"
            self._exchange(self.configuration.message("COMPLETE", ticket.sequence, ticket.role), "DISARMED")
            if self._state != "COMPLETE_PENDING":
                raise GuardProtocolError("nested failure invalidated pending COMPLETE")
            self._active = None
            self._state = "IDLE"
        except BaseException:
            self.fail()
            raise

    def _exchange(self, request: dict, response_operation: str) -> None:
        deadline = checked_add(self._clock_now(), self.configuration.ack_timeout_ms * 1_000_000)
        outgoing = encode_framed_message(request, max_bytes=MAX_BODY_BYTES)
        offset = 0
        decoder = GuardFrameDecoder()
        while offset < len(outgoing):
            self._wait(deadline, write=True)
            try:
                count = self._channel.send(outgoing[offset:])
            except BlockingIOError:
                continue
            if count <= 0:
                raise GuardProtocolError("guard send failed")
            offset += count
        while True:
            self._wait(deadline, write=False)
            try:
                chunk = self._channel.recv(decoder.remaining_capacity)
            except BlockingIOError:
                continue
            response = decoder.feed(chunk)
            if response is not None:
                # A readable ACK cannot win over its absolute deadline.
                if self._clock_now() >= deadline:
                    raise GuardProtocolError("guard ACK expired")
                response = validate_message(response, self.configuration)
                if response != dict(request, operation=response_operation):
                    raise GuardProtocolError("guard ACK is wrong/stale/foreign")
                return

    def _wait(self, deadline: int, *, write: bool) -> None:
        remaining = deadline - self._clock_now()
        if remaining <= 0:
            raise GuardProtocolError("guard ACK timeout")
        readable, writable, _ = select.select(
            [] if write else [self._channel],
            [self._channel] if write else [],
            [],
            remaining / 1e9,
        )
        if not (writable if write else readable) or self._clock_now() >= deadline:
            raise GuardProtocolError("guard ACK timeout")

    def _clock_now(self) -> int:
        now = bounded_integer(time.monotonic_ns())
        if self._last_clock_ns is not None and now < self._last_clock_ns:
            raise GuardProtocolError("client clock went backward")
        self._last_clock_ns = now
        return now
