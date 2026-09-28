"""Test-only S/V/E supervisor. Never a product worker entrypoint.

Usage is a fixed CPU protocol scenario today; GPU drivers remain unconnected.
The child is fixed by repository path, not caller-supplied code or callbacks.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path


def main():
    if os.name != "posix" or not sys.flags.isolated or not sys.flags.no_site or len(sys.argv) != 6:
        return 64
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from tests.support.sacrificial_supervisor import supervise_child
    from tests.support.sacrificial_watchdog import SupervisorGuardState

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
    evidence = supervise_child(
        child, server, parent_fd, state, startup_timeout=10.0, fault_role=role, fault=fault
    )
    (output / "supervisor.json").write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
