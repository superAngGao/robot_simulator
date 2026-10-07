"""Offline installed-wheel facts, never an import/admission permission generator.

Run this stdlib-only tool with -I -S -B. It reads explicitly named distributions;
it does not resolve dependencies, execute .pth files, import packages or initialize
CUDA. The output is authoring evidence for a later reviewed deployment contract.
Imports/probes/publications still need their own frozen, source-audited rules.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import sys
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

VERSION = "q54.p12.4.offline-wheel-inventory.v1"


def _name(value):
    if type(value) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("invalid explicit distribution name")
    return re.sub(r"[-_.]+", "-", value).lower()


def _contained_file(root, relative):
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or path.as_posix() != relative
        or not path.parts
        or any(part in {".", ".."} or ":" in part or "\\" in part for part in path.parts)
    ):
        raise ValueError(f"non-contained installed file: {relative!r}")
    current = root
    for part in path.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlinked installed file: {relative}")
    if not current.is_file() or not current.resolve(strict=True).is_relative_to(root):
        raise ValueError(f"missing/outside installed file: {relative}")
    return current


def _digest(path):
    before = path.stat()
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            count += len(chunk)
        finished = os.fstat(handle.fileno())
    after = path.stat()

    def identity(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    if (
        not identity(before) == identity(opened) == identity(finished) == identity(after)
        # Windows path-stat and fd-stat can expose different ctime semantics.
        # Compare each clock to itself, not the two APIs to each other.
        or before.st_ctime_ns != after.st_ctime_ns
        or opened.st_ctime_ns != finished.st_ctime_ns
        or count != before.st_size
    ):
        raise ValueError(f"installed file changed while reading: {path}")
    return digest.hexdigest(), count


def inventory(site_root, distribution_names):
    """Observe exact installed bytes; no dependency module body runs here."""
    root = Path(site_root)
    if not root.is_absolute() or root.resolve(strict=True) != root or not root.is_dir():
        raise ValueError("site root must be an existing resolved absolute directory")
    requested = tuple(sorted(_name(name) for name in distribution_names))
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("explicit nonempty unique distributions required")
    candidates = {}
    for metadata in sorted(root.glob("*.dist-info/METADATA")):
        metadata = _contained_file(root, metadata.relative_to(root).as_posix())
        parsed = BytesParser().parsebytes(metadata.read_bytes(), headersonly=True)
        names, versions = parsed.get_all("Name"), parsed.get_all("Version")
        if names is None or versions is None or len(names) != 1 or len(versions) != 1:
            raise ValueError(f"invalid distribution identity: {metadata}")
        key = _name(names[0])
        if key in requested:
            if key in candidates:
                raise ValueError(f"duplicate installed distribution: {key}")
            candidates[key] = (metadata, parsed)
    if set(candidates) != set(requested):
        raise ValueError(f"requested wheel metadata missing: {sorted(set(requested) - set(candidates))}")

    distributions, records, omitted = [], {}, []
    for key in requested:
        metadata, parsed = candidates[key]
        record_path = _contained_file(root, (metadata.parent / "RECORD").relative_to(root).as_posix())
        rows = list(csv.reader(io.StringIO(record_path.read_text(encoding="utf-8"))))
        names = set()
        for row in rows:
            if len(row) != 3 or not row[0] or row[0] in names:
                raise ValueError(f"invalid/duplicate wheel RECORD row: {key}")
            relative, expected_digest, expected_size = row
            names.add(relative)
            # Entry-point scripts/manuals live outside site-packages and are
            # never used by the isolated worker. Preserve exclusions explicitly.
            path = PurePosixPath(relative)
            if (
                not path.is_absolute()
                and path.parts[:3] == ("..", "..", "..")
                and (path.parts[3:4] == ("bin",) or path.parts[3:5] == ("share", "man"))
            ):
                reason = "external_entry_point" if path.parts[3] == "bin" else "external_manual"
                omitted.append({"distribution": key, "path": relative, "reason": reason})
                continue
            if "__pycache__" in path.parts or relative.endswith((".pyc", ".pyo")):
                omitted.append({"distribution": key, "path": relative, "reason": "bytecode_disabled"})
                continue
            installed = _contained_file(root, relative)
            digest, size = _digest(installed)
            if expected_digest:
                algorithm, separator, encoded = expected_digest.partition("=")
                actual = base64.urlsafe_b64encode(bytes.fromhex(digest)).decode("ascii").rstrip("=")
                if algorithm != "sha256" or separator != "=" or encoded != actual:
                    raise ValueError(f"installed bytes differ from wheel RECORD: {relative}")
            if expected_size and (not expected_size.isdecimal() or int(expected_size) != size):
                raise ValueError(f"installed size differs from wheel RECORD: {relative}")
            previous = records.get(relative)
            observed = {"path": relative, "sha256": digest, "size": size}
            if previous is not None and previous != observed:
                raise ValueError(f"shared wheel file changed: {relative}")
            records[relative] = observed
        required = {metadata.relative_to(root).as_posix(), record_path.relative_to(root).as_posix()}
        if not required <= names:
            raise ValueError(f"wheel RECORD omits its own identity files: {key}")
        distributions.append(
            {
                "name": parsed["Name"],
                "version": parsed["Version"],
                "metadata": metadata.relative_to(root).as_posix(),
                "record": record_path.relative_to(root).as_posix(),
                "requires_dist": parsed.get_all("Requires-Dist", []),
            }
        )
    executable = Path(sys.executable).resolve(strict=True)
    digest, size = _digest(executable)
    return {
        "schema_version": VERSION,
        "purpose": "offline_facts_only_not_runtime_admission",
        "site_root": root.as_posix(),
        "interpreter": {
            "path": executable.as_posix(),
            "sha256": digest,
            "size": size,
            "version": list(sys.version_info[:3]),
            "implementation": sys.implementation.name,
        },
        "distributions": distributions,
        "files": [records[name] for name in sorted(records)],
        "omitted": sorted(omitted, key=lambda item: (item["distribution"], item["path"])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-root", required=True)
    parser.add_argument("--distribution", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode:
        raise RuntimeError("inventory tool requires -I -S -B")
    if args.output.exists():
        raise FileExistsError("inventory will not overwrite existing evidence")
    result = inventory(args.site_root, args.distribution)
    data = json.dumps(
        result, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    with args.output.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "path": str(args.output),
                "sha256": hashlib.sha256(data).hexdigest(),
                "file_count": len(result["files"]),
                "bytes_observed": sum(row["size"] for row in result["files"]),
                "distribution_count": len(result["distributions"]),
            }
        )
    )


if __name__ == "__main__":
    main()
