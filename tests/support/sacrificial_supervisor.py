"""Single test-only supervision loop shared by fixed CPU and Warp children."""

from __future__ import annotations

import select
import socket
import struct
import time

from tests.support.sacrificial_watchdog import decode_message, encode_message


def supervise_child(child, server, parent_fd, state, *, startup_timeout, fault_role=None, fault="normal"):
    server.setblocking(False)
    buffer = bytearray()
    channel_open = True
    killed = False
    reason = None
    startup_deadline = time.monotonic() + startup_timeout
    try:
        while child.poll() is None:
            if time.monotonic() >= (state.deadline if state.deadline is not None else startup_deadline):
                reason = "guard_deadline" if state.deadline is not None else "harness_deadline"
                child.kill()
                killed = True
                break
            ready, _, _ = select.select([parent_fd] + ([server] if channel_open else []), [], [], 0.01)
            if parent_fd in ready:
                reason = "parent_lost"
                child.kill()
                killed = True
                break
            if server not in ready:
                continue
            data = server.recv(2048)
            if not data:
                channel_open = False
                continue
            buffer.extend(data)
            while len(buffer) >= 4:
                length = struct.unpack("!I", buffer[:4])[0]
                if not 0 < length <= 1024:
                    raise RuntimeError("invalid guard frame")
                if len(buffer) < length + 4:
                    break
                message = decode_message(bytes(buffer[4 : 4 + length]))
                del buffer[: 4 + length]
                response = state.accept(message)
                if message["role"] == fault_role and message["operation"] == "ARM":
                    if fault == "no_ack":
                        continue
                    if fault == "wrong_role":
                        response["role"] = "E" if fault_role != "E" else "V"
                    elif fault == "wrong_sequence":
                        response["sequence"] += 1
                    elif fault == "ack_eof":
                        server.shutdown(socket.SHUT_WR)
                        continue
                encoded = encode_message(response)
                if server.send(encoded) != len(encoded):
                    raise RuntimeError("incomplete watchdog ACK write")
    except BaseException as exc:
        reason = f"supervisor_protocol_failure:{type(exc).__name__}"
        if child.poll() is None:
            child.kill()
            killed = True
    finally:
        _, stderr = child.communicate(timeout=5)
        server.close()
    return dict(
        worker_return_code=child.returncode,
        killed=killed,
        reason=reason,
        guard_history=state.history,
        terminal_deadline_armed=state.deadline is not None,
        no_restart=True,
        stderr=(stderr or b"").decode("utf-8", errors="replace"),
    )
