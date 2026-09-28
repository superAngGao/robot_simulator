"""Fixed CPU fault programs for C4c2b; never imported/adopted by the worker.

Run in an isolated interpreter. Only enumerated faults and private inherited
descriptors are accepted. No callback, module selector or executable payload.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import select
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from optical_worker.protocol import encode_framed_message  # noqa: E402
from optical_worker.wait_protocol import (  # noqa: E402
    MAX_BODY_BYTES,
    GuardConfiguration,
    GuardWaitClient,
)
from optical_worker.wait_supervision import (  # noqa: E402
    GuardSupervisorCore,
    GuardSupervisorTransport,
)

FAULTS = (
    "success",
    "normal_hang",
    "emergency_hang",
    "no_ack",
    "wrong_ack",
    "ack_eof",
    "send_failure",
    "stalled_disarm",
    "parent_loss",
    "trickle",
)


def configuration() -> GuardConfiguration:
    # Explicit CPU-test budgets, not product time/performance thresholds.
    return GuardConfiguration("a" * 32, 1, 350, 900)


def mark(directory: Path, name: str) -> None:
    (directory / name).write_text("1", encoding="ascii")


class Sentinel:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def __del__(self) -> None:
        mark(self.directory, "destructor")


def child(fault: str, fd: int, directory: Path) -> None:
    guardian = Sentinel(directory)
    atexit.register(mark, directory, "atexit")
    channel = socket.socket(fileno=fd)
    config = configuration()
    client = GuardWaitClient(channel, config)
    try:
        if fault == "send_failure":
            channel.close()
        if fault == "trickle":
            # A valid header trickles without a complete frame. The supervisor
            # must still react to real parent loss; no ARM has been committed.
            channel.send(b"\x00")
            mark(directory, "leaf")
            while True:
                time.sleep(0.02)
        role = "emergency" if fault == "emergency_hang" else "setup"
        ticket = client.arm(role)
        mark(directory, "leaf")
        if fault in {"normal_hang", "emergency_hang", "parent_loss"}:
            while True:
                time.sleep(0.02)
        if fault == "stalled_disarm":
            # Real peer refuses to consume the DISARMED frame. Bypass only the
            # client side to model the bad peer; use the real supervisor core.
            channel.setblocking(True)
            channel.sendall(
                encode_framed_message(config.message("COMPLETE", 0, "setup"), max_bytes=MAX_BODY_BYTES)
            )
            while True:
                time.sleep(0.02)
        client.complete(ticket)
        ticket = client.arm("resolve")
        client.complete(ticket)
        mark(directory, "completed")
        # Keep the graph alive until this deliberate process terminal point.
        assert guardian.directory == directory
        os._exit(0)
    except BaseException:
        mark(directory, "failed")
        assert guardian.directory == directory
        os._exit(74)


def supervise(fault: str, liveness_fd: int, directory: Path) -> dict:
    channel, child_channel = socket.socketpair()
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            str(Path(__file__).resolve()),
            "child",
            fault,
            str(child_channel.fileno()),
            str(directory),
        ],
        pass_fds=(child_channel.fileno(),),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    child_channel.close()
    core = GuardSupervisorCore(configuration())
    transport = GuardSupervisorTransport(channel, core)
    fault_injected = False
    parent_present = True
    start = time.monotonic()
    try:
        while True:
            # This outer bound is a failing test safeguard, never evidence of
            # successful kill/reap or a product watchdog replacement.
            if time.monotonic() - start > 10:
                raise RuntimeError("test supervisor exceeded its outer fail bound")
            code = process.poll()
            if code is not None:
                core.observe_exit(code)
                break
            core.tick()
            if core.kill_required:
                core.record_kill_request()
                try:
                    process.kill()
                except PermissionError:
                    core.record_kill_failure("permission_denied")
                except ProcessLookupError:
                    core.record_kill_failure("process_missing")
                except OSError:
                    core.record_kill_failure("other_os_error")
            pending = core.pending_response
            if pending is not None and not fault_injected:
                if fault == "wrong_ack":
                    wrong = dict(pending, session="b" * 32)
                    channel.send(encode_framed_message(wrong, max_bytes=MAX_BODY_BYTES))
                    fault_injected = True
                elif fault == "ack_eof":
                    channel.shutdown(socket.SHUT_WR)
                    fault_injected = True
                elif fault == "stalled_disarm" and pending["operation"] == "DISARMED":
                    # Fill the actual OS send buffer; the peer never reads.
                    # This models transport backpressure, not a mocked clock.
                    channel.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)
                    while True:
                        try:
                            channel.send(b"x" * 4096)
                        except BlockingIOError:
                            break
                    fault_injected = True
            readers = [channel] if core.state not in {"TERMINATING", "EXITED"} else []
            if parent_present:
                readers.append(liveness_fd)
            writers = [channel] if transport.wants_write else []
            if fault in {"no_ack", "wrong_ack", "ack_eof"}:
                writers = []
            deadline = core.effective_deadline_ns
            timeout = 0.01 if deadline is None else min(0.01, max(0, (deadline - time.monotonic_ns()) / 1e9))
            readable, writable, _ = select.select(readers, writers, [], timeout)
            if parent_present and liveness_fd in readable and os.read(liveness_fd, 1) == b"":
                parent_present = False
                core.fault("parent_lost")
            try:
                transport.service(readable=channel in readable, writable=channel in writable)
            except (RuntimeError, OSError, ValueError):
                # Production core has already entered terminal state. Retain
                # custody and continue actual observation; do not fake exit.
                if core.state != "TERMINATING":
                    raise
        process.wait(timeout=1)
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return {
            "evidence": core.evidence(require_final=True).to_payload(),
            "stderr": stderr,
            "fault_injected": fault_injected,
        }
    finally:
        # Only outer harness failure reaches this with a live process. Tests
        # fail, and actual OS kill/wait prevents orphaned sacrificial children.
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        channel.close()
        process.stderr.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("child", "supervisor"))
    parser.add_argument("fault", choices=FAULTS)
    parser.add_argument("fd", type=int)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.mode == "child":
        child(args.fault, args.fd, args.directory)
    else:
        print(json.dumps(supervise(args.fault, args.fd, args.directory), sort_keys=True))


if __name__ == "__main__":
    main()
