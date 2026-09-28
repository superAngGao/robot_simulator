"""Test-only S/V/E supervisor. Never a product worker entrypoint.

Usage is a fixed CPU protocol scenario today; GPU drivers remain unconnected.
The child is fixed by repository path, not caller-supplied code or callbacks.
"""

from __future__ import annotations

import json
import os
import select
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path


def main():
    if os.name != "posix" or not sys.flags.isolated or not sys.flags.no_site or len(sys.argv) != 6:
        return 64
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from tests.support.sacrificial_watchdog import SupervisorGuardState, decode_message, encode_message

    output = Path(sys.argv[1]).resolve()
    role, fault, session = sys.argv[2:5]
    parent_fd = int(sys.argv[5])
    if (
        role not in ("S", "V", "E")
        or fault not in ("normal", "hang", "raise", "no_ack", "wrong_role", "wrong_sequence", "ack_eof")
        or not output.is_dir()
    ):
        return 64
    state = SupervisorGuardState(session=session, operation_timeout=0.6)
    server, child_channel = socket.socketpair()
    journal = os.open(output / "journal.jsonl", os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_APPEND, 0o600)
    child = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-S",
            str(root / "tests/support/sacrificial_protocol_child.py"),
            role,
            fault,
            session,
            str(child_channel.fileno()),
            str(journal),
        ],
        pass_fds=(child_channel.fileno(), journal),
        close_fds=True,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    child_channel.close()
    os.close(journal)
    server.setblocking(False)
    buffer = bytearray()
    channel_open = True
    killed = False
    reason = None
    # Harness ceiling, not a product latency/performance gate.
    startup_deadline = time.monotonic() + 10.0
    try:
        while child.poll() is None:
            if time.monotonic() >= (state.deadline if state.deadline is not None else startup_deadline):
                reason = "guard_deadline" if state.deadline is not None else "harness_deadline"
                child.kill()
                killed = True
                break
            watched = [parent_fd] + ([server] if channel_open else [])
            ready, _, _ = select.select(watched, [], [], 0.01)
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
                if message["role"] == role and message["operation"] == "ARM":
                    if fault == "no_ack":
                        continue
                    if fault == "wrong_role":
                        response["role"] = "E" if role != "E" else "V"
                    elif fault == "wrong_sequence":
                        response["sequence"] += 1
                    elif fault == "ack_eof":
                        server.shutdown(socket.SHUT_WR)
                        continue
                # A single <=1028-byte response cannot block this nonblocking endpoint.
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
    (output / "supervisor.json").write_text(
        json.dumps(
            dict(
                worker_return_code=child.returncode,
                killed=killed,
                reason=reason,
                guard_history=state.history,
                terminal_deadline_armed=state.deadline is not None,
                no_restart=True,
                stderr=stderr.decode("utf-8", errors="replace"),
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
