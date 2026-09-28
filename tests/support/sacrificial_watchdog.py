"""Fixed S/V/E ACK protocol for disposable validation children only.

No callable, module name, native handle or resource object crosses this channel.
Product C4c0 A/K protocol and worker dispatcher are intentionally untouched.
"""

from __future__ import annotations

import json
import select
import socket
import struct
import time
from dataclasses import dataclass


class WatchdogProtocolError(RuntimeError):
    pass


def encode_message(message: dict) -> bytes:
    data = json.dumps(message, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("ascii")
    if len(data) > 1024:
        raise WatchdogProtocolError("guard message is too large")
    return struct.pack("!I", len(data)) + data


def _duplicates(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise WatchdogProtocolError("duplicate guard key")
        output[key] = value
    return output


def decode_message(data: bytes) -> dict:
    try:
        message = json.loads(data.decode("ascii"), object_pairs_hook=_duplicates)
        if type(message) is not dict or encode_message(message)[4:] != data:
            raise WatchdogProtocolError("noncanonical guard message")
    except (ValueError, UnicodeError, TypeError) as exc:
        raise WatchdogProtocolError("invalid guard message") from exc
    return message


def _read_exact(channel: socket.socket, count: int, deadline: float) -> bytes:
    result = bytearray()
    while len(result) < count:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([channel], [], [], remaining)[0]:
            raise WatchdogProtocolError("guard ACK timeout")
        chunk = channel.recv(count - len(result))
        if not chunk:
            raise WatchdogProtocolError("guard channel EOF")
        result.extend(chunk)
    return bytes(result)


def receive_message(channel: socket.socket, *, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    length = struct.unpack("!I", _read_exact(channel, 4, deadline))[0]
    if not 0 < length <= 1024:
        raise WatchdogProtocolError("invalid guard frame length")
    return decode_message(_read_exact(channel, length, deadline))


def send_message(channel: socket.socket, message: dict, *, timeout: float) -> None:
    data = encode_message(message)
    deadline = time.monotonic() + timeout
    sent = 0
    while sent < len(data):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([], [channel], [], remaining)[1]:
            raise WatchdogProtocolError("guard send timeout")
        count = channel.send(data[sent:])
        if count <= 0:
            raise WatchdogProtocolError("guard send failed")
        sent += count


@dataclass(frozen=True, slots=True)
class WaitTicket:
    session: str
    role: str
    sequence: int


class GuardedOperationChannel:
    def __init__(self, channel: socket.socket, *, session: str, ack_timeout: float = 1.0):
        if (
            type(session) is not str
            or len(session) != 32
            or any(c not in "0123456789abcdef" for c in session)
        ):
            raise WatchdogProtocolError("invalid session identity")
        if type(ack_timeout) not in (float, int) or not 0 < ack_timeout <= 30:
            raise WatchdogProtocolError("invalid ACK timeout")
        self._channel = channel
        self._session = session
        self._timeout = ack_timeout
        self._sequence = 0
        self._active: WaitTicket | None = None
        self._failed = False

    def arm(self, role: str) -> WaitTicket:
        if role not in ("S", "V", "E") or self._active is not None or self._failed:
            raise WatchdogProtocolError("cannot arm this guard operation")
        ticket = WaitTicket(self._session, role, self._sequence)
        request = dict(operation="ARM", session=ticket.session, role=role, sequence=ticket.sequence)
        self._exchange(request, "ARMED")
        self._active = ticket
        self._sequence += 1
        return ticket

    def complete(self, ticket: WaitTicket) -> None:
        if self._failed or ticket is not self._active or ticket.role != "S":
            raise WatchdogProtocolError("only the exact setup ticket can disarm")
        self._exchange(
            dict(operation="COMPLETE", session=ticket.session, role="S", sequence=ticket.sequence), "DISARMED"
        )
        self._active = None

    def _exchange(self, request, response_operation):
        try:
            send_message(self._channel, request, timeout=self._timeout)
            expected = dict(request, operation=response_operation)
            response = receive_message(self._channel, timeout=self._timeout)
            if type(response.get("sequence")) is not int or response != expected:
                raise WatchdogProtocolError("wrong, stale or foreign watchdog ACK")
        except BaseException:
            self._failed = True
            raise


class SupervisorGuardState:
    """Only the supervisor clock sets/disarms deadlines; ACK follows commit."""

    def __init__(self, *, session: str, operation_timeout: float):
        if (
            type(session) is not str
            or len(session) != 32
            or any(c not in "0123456789abcdef" for c in session)
        ):
            raise WatchdogProtocolError("invalid supervisor session identity")
        if type(operation_timeout) not in (float, int) or not 0 < operation_timeout <= 300:
            raise WatchdogProtocolError("invalid supervisor operation timeout")
        self.session = session
        self._timeout = operation_timeout
        self._sequence = 0
        self._active = None
        self.deadline = None
        self.history = []
        self.failed = False

    def accept(self, message: dict) -> dict:
        try:
            if (
                self.failed
                or type(message) is not dict
                or set(message) != {"operation", "session", "role", "sequence"}
                or message["session"] != self.session
                or type(message["sequence"]) is not int
                or message["role"] not in ("S", "V", "E")
            ):
                raise WatchdogProtocolError("invalid guard request")
            if self.deadline is not None and time.monotonic() >= self.deadline:
                raise WatchdogProtocolError("expired operation cannot disarm")
            if message["operation"] == "ARM":
                if self._active is not None or message["sequence"] != self._sequence:
                    raise WatchdogProtocolError("duplicate or out-of-order arm")
                self.deadline = time.monotonic() + self._timeout
                self._active = dict(message)
                self._sequence += 1
                self.history.append(dict(message, deadline=self.deadline))
                return dict(message, operation="ARMED")
            if message["operation"] == "COMPLETE":
                if self._active != dict(message, operation="ARM") or message["role"] != "S":
                    raise WatchdogProtocolError("invalid setup disarm")
                self.deadline = None
                self._active = None
                self.history.append(dict(message))
                return dict(message, operation="DISARMED")
            raise WatchdogProtocolError("unknown guard operation")
        except BaseException:
            self.failed = True
            raise
