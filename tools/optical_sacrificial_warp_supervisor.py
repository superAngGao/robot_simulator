"""Fixed Warp validation supervisor; not a product worker or benchmark runner."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path


def main():
    if os.name != "posix" or not sys.flags.isolated or not sys.flags.no_site or len(sys.argv) != 4:
        return 64
    # No repository helper may bootstrap from a previous timestamp cache.
    bootstrap_cache = Path(sys.argv[2]).resolve() / "supervisor-source-cache"
    bootstrap_cache.mkdir(exist_ok=False)
    sys.pycache_prefix = str(bootstrap_cache)
    sys.dont_write_bytecode = True
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from tests.support.sacrificial_sources import RepositoryImportLedger, canonical_bytes
    from tests.support.sacrificial_supervisor import supervise_child
    from tests.support.sacrificial_watchdog import SupervisorGuardState

    manifest_path = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    parent_fd = int(sys.argv[3])
    raw = manifest_path.read_bytes()
    payload = json.loads(raw)
    if (
        raw != canonical_bytes(payload)
        or type(payload) is not dict
        or set(payload)
        != {
            "sources",
            "session",
            "target",
            "profile",
            "shadows",
            "diagnostics",
            "resident_rays",
            "fault",
            "position",
            "ack_role",
            "ack_fault",
            "repetitions",
        }
    ):
        return 65
    ledger = RepositoryImportLedger(payload["sources"])
    if ledger.root != root:
        return 65
    if (
        type(payload["target"]) is not int
        or payload["target"] not in (0, 1, 2)
        or type(payload["shadows"]) is not bool
        or type(payload["diagnostics"]) is not bool
        or type(payload["repetitions"]) is not int
        or payload["repetitions"] not in (1, 2)
        or (payload["repetitions"] == 2 and payload["resident_rays"] not in (None, 0))
        or (
            payload["fault"] in ("auxiliary_submit_then_raise", "second_pass_submit_then_raise")
            and payload["repetitions"] != 2
        )
        or payload["profile"] not in ("direct_light_full", "rgb_preview", "render_only")
        or (
            payload["resident_rays"] is not None
            and (type(payload["resident_rays"]) is not int or payload["resident_rays"] not in (0, 1, 21))
        )
        or type(payload["position"]) is not int
        or not 0 <= payload["position"] < 5
        or payload["ack_role"] not in (None, "S", "V", "E")
        or payload["ack_fault"] not in ("normal", "no_ack", "wrong_role", "wrong_sequence", "ack_eof")
        or payload["fault"]
        not in (
            "normal",
            "audit_allocation",
            "audit_copy",
            "audit_materialization",
            "audit_numpy",
            "audit_bridge",
            "audit_dlpack",
            "audit_module",
            "audit_event",
            "audit_sync",
            "context_stream_drift",
            "context_ambient_drift",
            "auxiliary_submit_then_raise",
            "second_pass_submit_then_raise",
            "runtime_import_failure",
            "allocation_return_failure",
            "readback_storage_failure",
            "readback_transfer_failure",
            "readback_before_copy_failure",
            "readback_submit_then_raise",
            "readback_return_failure",
            "first_upload_failure",
            "compile_failure",
            "record_failure",
            "warmup_failure",
            "pre_latch_failure",
            "post_latch_failure",
            "submit_then_raise",
            "setup_barrier_hang",
            "setup_barrier_raise",
            "validation_hang",
            "validation_raise",
            "emergency_hang",
            "emergency_raise",
            "journal_encoding_failure",
            "journal_write_failure",
            "hold_upload",
            "hold_record",
            "hold_submit",
            "hold_sealed",
            "hold_validated",
        )
    ):
        return 65
    state = SupervisorGuardState(session=payload["session"], operation_timeout=5.0)
    server, child_channel = socket.socketpair()
    journal_fd = os.open(output / "journal.jsonl", os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_APPEND, 0o600)
    child = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-S",
            str(root / "tests/support/sacrificial_warp_child.py"),
            str(manifest_path),
            str(child_channel.fileno()),
            str(journal_fd),
        ],
        pass_fds=(child_channel.fileno(), journal_fd),
        close_fds=True,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    child_channel.close()
    os.close(journal_fd)
    evidence = supervise_child(
        child,
        server,
        parent_fd,
        state,
        startup_timeout=180.0,
        fault_role=payload["ack_role"],
        fault=payload["ack_fault"],
    )
    (output / "supervisor.json").write_bytes(canonical_bytes(evidence))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
