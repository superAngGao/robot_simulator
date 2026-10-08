"""Stdlib-only ``python -I -S`` bootstrap for one closed optical worker."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_REQUIRED_C4C0_SOURCE_PATHS = (
    "optical_worker/__init__.py",
    "optical_worker/lane_contract.py",
    "optical_worker/lifecycle.py",
    "optical_worker/manifest.py",
    "optical_worker/native_contract.py",
    "optical_worker/native_manifest.py",
    "optical_worker/native_setup.py",
    "optical_worker/protocol.py",
    "optical_worker/worker.py",
    "tools/optical_closed_worker.py",
    "tools/optical_closed_worker_supervisor.py",
)


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


def _load_and_prevalidate_manifest(path):
    encoded = path.read_bytes()
    if not encoded or len(encoded) > _MAX_MANIFEST_BYTES:
        raise RuntimeError("runtime manifest size is invalid")
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
    if canonical != encoded:
        raise RuntimeError("runtime manifest is not canonical JSON")
    if not isinstance(payload, dict):
        raise RuntimeError("runtime manifest root is invalid")
    root = Path(payload.get("repository_root", "")).resolve()
    if not root.is_dir():
        raise RuntimeError("runtime repository root is invalid")
    sources = payload.get("source_digests")
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("runtime source manifest is invalid")
    if tuple(source.get("relative_path") for source in sources if isinstance(source, dict)) != (
        _REQUIRED_C4C0_SOURCE_PATHS
    ):
        raise RuntimeError("runtime source allowlist is not the frozen C4c0 allowlist")
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"relative_path", "sha256"}:
            raise RuntimeError("runtime source entry is invalid")
        candidate = (root / source["relative_path"]).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("runtime source escapes repository") from exc
        if not candidate.is_file() or _sha256(candidate) != source["sha256"]:
            raise RuntimeError("runtime source hash mismatch")
    site_paths = payload.get("site_package_paths")
    if (
        not isinstance(site_paths, list)
        or site_paths != sorted(set(site_paths))
        or any(not isinstance(value, str) or not Path(value).is_absolute() for value in site_paths)
    ):
        raise RuntimeError("runtime site-package paths are invalid")
    return payload, root


def main():
    if len(sys.argv) != 7:
        return 64
    if not sys.flags.isolated or not sys.flags.no_site:
        return 65
    manifest_path = Path(sys.argv[1]).resolve()
    command_descriptor = int(sys.argv[2], 10)
    liveness_descriptor = int(sys.argv[3], 10)
    journal_descriptor = int(sys.argv[4], 10)
    abort_signal_descriptor = int(sys.argv[5], 10)
    abort_ack_descriptor = int(sys.argv[6], 10)
    payload, root = _load_and_prevalidate_manifest(manifest_path)
    sys.dont_write_bytecode = True
    stdlib_paths = tuple(sys.path)
    sys.path[:] = [str(root), *payload["site_package_paths"], *stdlib_paths]

    from optical_worker.manifest import validate_runtime_manifest_against_process
    from optical_worker.protocol import ClosedWorkerRuntimeManifest
    from optical_worker.worker import run_worker_loop

    manifest = ClosedWorkerRuntimeManifest.from_payload(payload)
    validate_runtime_manifest_against_process(manifest)
    return run_worker_loop(
        command_descriptor=command_descriptor,
        liveness_descriptor=liveness_descriptor,
        journal_descriptor=journal_descriptor,
        abort_signal_descriptor=abort_signal_descriptor,
        abort_ack_descriptor=abort_ack_descriptor,
        manifest=manifest,
    )


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException as exception:
        try:
            journal_descriptor = int(sys.argv[4], 10)
            payload = json.dumps(
                {
                    "kind": "bootstrap_failure",
                    "exception_type": type(exception).__name__,
                    "message": str(exception),
                },
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            os.write(journal_descriptor, len(payload).to_bytes(4, "big") + payload)
            os.fsync(journal_descriptor)
        except BaseException:
            pass
        os._exit(78)
    raise SystemExit(exit_code)
