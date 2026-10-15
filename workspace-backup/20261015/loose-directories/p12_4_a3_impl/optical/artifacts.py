"""Canonical, atomic artifacts for durable optical benchmark runs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ARTIFACT_MANIFEST_SCHEMA_VERSION = "q54.p12.4.artifact-manifest.v1"


@dataclass(frozen=True)
class ArtifactDigest:
    """Content identity for one file below a benchmark artifact root."""

    relative_path: str
    media_type: str
    size_bytes: int
    sha256: str
    record_count: int | None = None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "relative_path": self.relative_path,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }
        if self.record_count is not None:
            payload["record_count"] = self.record_count
        return payload


class BenchmarkArtifactWriter:
    """Write deterministic JSON artifacts without escaping the selected root."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._digests: dict[str, ArtifactDigest] = {}

    @property
    def root(self) -> Path:
        return self._root

    @property
    def digests(self) -> tuple[ArtifactDigest, ...]:
        return tuple(self._digests[name] for name in sorted(self._digests))

    def write_json(self, relative_path: str, payload: object) -> ArtifactDigest:
        encoded = _canonical_json_line(payload)
        return self._write(relative_path, encoded, media_type="application/json")

    def write_jsonl(
        self,
        relative_path: str,
        records: Iterable[Mapping[str, Any]],
    ) -> ArtifactDigest:
        encoded_records = [_canonical_json_line(record) for record in records]
        encoded = b"".join(encoded_records)
        return self._write(
            relative_path,
            encoded,
            media_type="application/x-ndjson",
            record_count=len(encoded_records),
        )

    def write_manifest(self, relative_path: str = "manifest.json") -> ArtifactDigest:
        """Write a manifest of existing artifacts, excluding the manifest itself."""

        payload = {
            "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "artifacts": [digest.to_payload() for digest in self.digests],
        }
        return self.write_json(relative_path, payload)

    def _write(
        self,
        relative_path: str,
        encoded: bytes,
        *,
        media_type: str,
        record_count: int | None = None,
    ) -> ArtifactDigest:
        target, normalized = self._resolve_target(relative_path)
        if normalized in self._digests:
            raise ValueError(f"artifact {normalized!r} has already been written")
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        digest = ArtifactDigest(
            relative_path=normalized,
            media_type=media_type,
            size_bytes=len(encoded),
            sha256=hashlib.sha256(encoded).hexdigest(),
            record_count=record_count,
        )
        self._digests[normalized] = digest
        return digest

    def _resolve_target(self, relative_path: str) -> tuple[Path, str]:
        relative = Path(relative_path)
        if relative.is_absolute() or not relative.parts:
            raise ValueError("artifact path must be a non-empty relative path")
        target = (self._root / relative).resolve()
        try:
            normalized = target.relative_to(self._root).as_posix()
        except ValueError as exc:
            raise ValueError("artifact path escapes the benchmark output root") from exc
        if normalized == ".":
            raise ValueError("artifact path must name a file")
        return target, normalized


def _canonical_json_line(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
