"""Stdlib-only parent-liveness and abort-watchdog supervisor."""

from __future__ import annotations

import hashlib
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

_SUPERVISOR_RELATIVE_PATH = "tools/optical_closed_worker_supervisor.py"


def _duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError("duplicate runtime-manifest key")
        result[key] = value
    return result


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path):
    encoded = path.read_bytes()
    payload = json.loads(
        encoded.decode("utf-8", errors="strict"),
        object_pairs_hook=_duplicates,
        parse_float=lambda _: (_ for _ in ()).throw(RuntimeError("manifest float forbidden")),
        parse_constant=lambda _: (_ for _ in ()).throw(RuntimeError("manifest constant forbidden")),
    )
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if canonical != encoded or not isinstance(payload, dict):
        raise RuntimeError("supervisor runtime manifest is not canonical")
    source_digests = payload.get("source_digests")
    if not isinstance(source_digests, list):
        raise RuntimeError("supervisor source manifest is invalid")
    expected = next(
        (
            item.get("sha256")
            for item in source_digests
            if isinstance(item, dict) and item.get("relative_path") == _SUPERVISOR_RELATIVE_PATH
        ),
        None,
    )
    if expected != _sha256(Path(__file__).resolve()):
        raise RuntimeError("supervisor source does not match the runtime manifest")
    if Path(payload.get("interpreter_executable", "")).resolve() != Path(sys.executable).resolve():
        raise RuntimeError("supervisor interpreter does not match the runtime manifest")
    return payload


def _atomic_json(path, payload):
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    if len(sys.argv) != 7 or os.name != "posix":
        return 64
    if not sys.flags.isolated or not sys.flags.no_site:
        return 65
    manifest_path = Path(sys.argv[1]).resolve()
    command_descriptor = int(sys.argv[2], 10)
    parent_liveness_descriptor = int(sys.argv[3], 10)
    journal_path = Path(sys.argv[4]).resolve()
    supervisor_evidence_path = Path(sys.argv[5]).resolve()
    worker_script = Path(sys.argv[6]).resolve()
    manifest = _load_manifest(manifest_path)
    abort_watchdog_ms = manifest["abort_watchdog_ms"]
    if not isinstance(abort_watchdog_ms, int) or isinstance(abort_watchdog_ms, bool):
        return 65

    journal_descriptor = os.open(
        journal_path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    worker_liveness_read, worker_liveness_write = os.pipe()
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-S",
            str(worker_script),
            str(manifest_path),
            str(command_descriptor),
            str(worker_liveness_read),
            str(journal_descriptor),
        ],
        close_fds=True,
        pass_fds=(command_descriptor, worker_liveness_read, journal_descriptor),
        start_new_session=True,
    )
    os.close(command_descriptor)
    os.close(worker_liveness_read)
    os.close(journal_descriptor)
    parent_lost = False
    watchdog_killed = False
    loss_deadline = None
    while process.poll() is None:
        readable, _, _ = select.select([parent_liveness_descriptor], [], [], 0.05)
        if parent_liveness_descriptor in readable and not parent_lost:
            os.read(parent_liveness_descriptor, 1)
            parent_lost = True
            os.close(worker_liveness_write)
            loss_deadline = time.monotonic() + abort_watchdog_ms / 1000.0
        if parent_lost and loss_deadline is not None and time.monotonic() >= loss_deadline:
            process.kill()
            watchdog_killed = True
            break
    return_code = process.wait()
    if not parent_lost:
        os.close(worker_liveness_write)
    _atomic_json(
        supervisor_evidence_path,
        {
            "schema_version": "q54.p12.4.closed-worker-supervisor-evidence.v1",
            "worker_process_id": process.pid,
            "worker_return_code": return_code,
            "parent_lost": parent_lost,
            "watchdog_killed_worker": watchdog_killed,
            "no_restart": True,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
