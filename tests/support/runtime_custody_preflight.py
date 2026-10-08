"""Fixed CPU launch preflight; run separately with the pinned -I -S -B Python."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def main():
    request = json.load(sys.stdin)
    if set(request) != {"root", "revision", "lane", "catalog_sha256", "guard_configuration", "case_id"}:
        raise ValueError("unexpected runtime custody preflight input")
    root = Path(request["root"])
    spec = importlib.util.spec_from_file_location("source_preflight", root / "optical_worker/native_bootstrap.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.__name__ = "__main__"
    sys.modules["__main__"] = entry
    session = entry._install(root, request["revision"], request["lane"], request["catalog_sha256"])
    entry._verify(session)
    # Shared capture already checked the fixed stdlib source, path and exact
    # alias AST. No multiprocessing/backend import or second alias policy.
    descriptor = entry._INTERPRETER_ALIAS_CUSTODY[session][0]
    identity = list(descriptor[0])
    identity[1] = list(identity[1])
    executable = Path(sys.executable).resolve(strict=True)
    payload = {
        "schema_version": "q54.p12.4.runtime-custody-inputs.v1",
        "source": {"revision": session.revision, "catalog_sha256": session.catalog_sha256,
                   "snapshot_sha256": session.snapshot.sha256},
        "lane_id": session.lane, "configuration": request["guard_configuration"], "case_id": request["case_id"],
        "interpreter_alias": {"interpreter_identity": identity, "resolved_executable": executable.as_posix(),
                              "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                              "source_path": descriptor[1], "source_sha256": descriptor[3]},
    }
    if {"numpy", "torch", "warp", "multiprocessing"} & set(sys.modules):
        raise RuntimeError("runtime custody preflight imported a dependency")
    print(json.dumps(payload, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
