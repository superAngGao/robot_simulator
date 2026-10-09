"""Record systemd's main-process termination facts for one deployment build.

This is an ExecStopPost command, run with Python -I -S -B. It never launches,
signals or waits for a process, imports a backend, or publishes an artifact.
The launcher separately observes cgroup closure after systemd-run has exited.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath


_FIELDS = frozenset({"unit", "invocation_id", "service_result", "exit_code", "exit_status", "cgroup_path"})
_UNIT = re.compile(r"robot-sim-cuda-[0-9a-f]{32}-[012]\.service\Z")


def validate_status(value, *, unit: str) -> dict:
    """Validate a footer as descriptive facts, including unsuccessful outcomes."""
    if type(unit) is not str or _UNIT.fullmatch(unit) is None:
        raise ValueError("invalid fixed CUDA build unit")
    if type(value) is not dict or set(value) != _FIELDS or value["unit"] != unit:
        raise ValueError("build termination fields or unit mismatch")
    if any(item is not None and type(item) is not str for item in value.values()):
        raise ValueError("build termination facts must be strings or absent")
    invocation = value["invocation_id"]
    if invocation is not None and re.fullmatch(r"[0-9a-f]{32}", invocation) is None:
        raise ValueError("invalid systemd invocation ID")
    group = value["cgroup_path"]
    if group is not None:
        path = PurePosixPath(group)
        if (not path.is_absolute() or path.as_posix() != group or ".." in path.parts
                or path.name != unit or "\\" in group):
            raise ValueError("termination cgroup does not name the exact service")
    return dict(value)


def require_normal_exit(value, *, unit: str, waiter_returncode: int) -> dict:
    """Reject clean signals as well as failure; this does not prove group closure."""
    record = validate_status(value, unit=unit)
    if (type(waiter_returncode) is not int or waiter_returncode != 0
            or any(record[key] is None for key in _FIELDS)
            or (record["service_result"], record["exit_code"], record["exit_status"])
            != ("success", "exited", "0")):
        raise ValueError("CUDA build did not complete with normal main-process exit")
    return record


def record_status(output: Path, *, unit: str) -> None:
    """Read only this control process's manager environment and unified cgroup."""
    if output.name != "termination.json" or not output.is_absolute():
        raise ValueError("expected the private job termination.json path")
    parent = output.parent
    if parent.resolve(strict=True) != parent or any(p.is_symlink() for p in (parent, *parent.parents)):
        raise ValueError("noncanonical termination directory")
    info = parent.stat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.getuid():
        raise ValueError("termination directory is not private and owned")
    groups = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    if len(groups) != 1 or not groups[0].startswith("0::"):
        raise ValueError("expected one unified service cgroup")
    record = validate_status({
        "unit": unit,
        "invocation_id": os.environ.get("INVOCATION_ID"),
        "service_result": os.environ.get("SERVICE_RESULT"),
        "exit_code": os.environ.get("EXIT_CODE"),
        "exit_status": os.environ.get("EXIT_STATUS"),
        "cgroup_path": groups[0][3:],
    }, unit=unit)
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    with output.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    args = parser.parse_args()
    record_status(args.output, unit=args.unit)


if __name__ == "__main__":
    main()
