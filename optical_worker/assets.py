"""Exact external-asset manifest preflight for canonical optical cases."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .cases import get_case
from .workload_contract import (
    GO2_FROZEN_ASSET_AGGREGATE_SHA256,
    GO2_FROZEN_SOURCE_REVISION,
)

ASSET_MANIFEST_SCHEMA_VERSION = "q54.p12.4.external-asset-manifest.v1"
GO2_ASSET_REQUIREMENT = "mujoco_menagerie/unitree_go2@run_manifest"
GO2_SOURCE_REPOSITORY = "https://github.com/google-deepmind/mujoco_menagerie"
GO2_CANONICAL_REMOTE_URLS = frozenset(
    {
        GO2_SOURCE_REPOSITORY,
        f"{GO2_SOURCE_REPOSITORY}.git",
        "git@github.com:google-deepmind/mujoco_menagerie.git",
    }
)
DEFAULT_GO2_MODEL_DIR = Path("out/external/mujoco_menagerie/unitree_go2")
DEFAULT_ASSET_MANIFEST_NAME = "q54_asset_manifest.json"


class AssetPreflightError(RuntimeError):
    """A mandatory external asset is absent, incomplete, or mismatched."""


@dataclass(frozen=True)
class AssetFileEvidence:
    relative_path: str
    size_bytes: int
    sha256: str

    @classmethod
    def from_payload(cls, payload: object) -> "AssetFileEvidence":
        if not isinstance(payload, Mapping) or set(payload) != {
            "relative_path",
            "size_bytes",
            "sha256",
        }:
            raise AssetPreflightError("asset file manifest fields are invalid")
        relative_path = payload.get("relative_path")
        size_bytes = payload.get("size_bytes")
        sha256 = payload.get("sha256")
        if not isinstance(relative_path, str) or not relative_path:
            raise AssetPreflightError("asset file relative_path must be non-empty")
        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts or candidate.as_posix() != relative_path:
            raise AssetPreflightError("asset file relative_path must be normalized and root-scoped")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
            raise AssetPreflightError("asset file size_bytes must be a non-negative integer")
        if not _sha256_text(sha256):
            raise AssetPreflightError("asset file sha256 must be a lowercase digest")
        return cls(relative_path=relative_path, size_bytes=size_bytes, sha256=sha256)

    def to_payload(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class AssetManifestEvidence:
    schema_version: str
    requirement_id: str
    source_repository: str
    source_revision: str
    entrypoint: str
    files: tuple[AssetFileEvidence, ...]
    aggregate_sha256: str
    manifest_sha256: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "requirement_id": self.requirement_id,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "entrypoint": self.entrypoint,
            "files": [item.to_payload() for item in self.files],
            "aggregate_sha256": self.aggregate_sha256,
            "manifest_sha256": self.manifest_sha256,
        }


def preflight_case_asset(
    case_id: str,
    *,
    repo_root: Path,
    model_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> AssetManifestEvidence | None:
    """Validate every file in the exact external asset before lane timing."""

    case = get_case(case_id)
    requirement = case.required_asset_manifest
    if requirement is None:
        return None
    if requirement != GO2_ASSET_REQUIREMENT:
        raise AssetPreflightError(f"unsupported external asset requirement {requirement!r}")
    root = _resolve_within(repo_root, model_dir or DEFAULT_GO2_MODEL_DIR)
    manifest = _resolve_within(root, manifest_path or Path(DEFAULT_ASSET_MANIFEST_NAME))
    if not root.is_dir():
        raise AssetPreflightError(f"mandatory Go2 model directory is absent: {root}")
    if not manifest.is_file():
        raise AssetPreflightError(f"mandatory Go2 asset manifest is absent: {manifest}")
    try:
        raw = manifest.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssetPreflightError(f"cannot read Go2 asset manifest: {exc}") from exc
    expected_fields = {
        "schema_version",
        "requirement_id",
        "source_repository",
        "source_revision",
        "entrypoint",
        "files",
        "aggregate_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_fields:
        raise AssetPreflightError("Go2 asset manifest fields are invalid")
    if payload.get("schema_version") != ASSET_MANIFEST_SCHEMA_VERSION:
        raise AssetPreflightError("Go2 asset manifest schema version is invalid")
    if payload.get("requirement_id") != requirement:
        raise AssetPreflightError("Go2 asset manifest requirement does not match the case")
    if payload.get("source_repository") != GO2_SOURCE_REPOSITORY:
        raise AssetPreflightError("Go2 asset manifest repository is not canonical")
    source_revision = payload.get("source_revision")
    if not _git_commit_text(source_revision):
        raise AssetPreflightError("Go2 asset manifest source_revision must be a full Git commit")
    if source_revision != GO2_FROZEN_SOURCE_REVISION:
        raise AssetPreflightError("Go2 asset revision does not match the frozen workload source")
    if payload.get("entrypoint") != "scene.xml":
        raise AssetPreflightError("Go2 asset manifest entrypoint must be scene.xml")
    checkout_root = root.parent
    if root.name != "unitree_go2" or not (checkout_root / ".git").exists():
        raise AssetPreflightError("Go2 assets must remain in a canonical Git checkout")
    actual_revision = _git_output(checkout_root, "rev-parse", "HEAD")
    if actual_revision != source_revision:
        raise AssetPreflightError("Go2 manifest revision does not match checkout HEAD")
    remote_url = _git_output(checkout_root, "config", "--get", "remote.origin.url")
    if remote_url not in GO2_CANONICAL_REMOTE_URLS:
        raise AssetPreflightError("Go2 checkout origin is not the canonical repository")
    files_payload = payload.get("files")
    if not isinstance(files_payload, Sequence) or isinstance(files_payload, (str, bytes)):
        raise AssetPreflightError("Go2 asset manifest files must be an array")
    files = tuple(AssetFileEvidence.from_payload(item) for item in files_payload)
    paths = tuple(item.relative_path for item in files)
    if not files or tuple(sorted(paths)) != paths or len(set(paths)) != len(paths):
        raise AssetPreflightError("Go2 asset manifest files must be non-empty, unique, and sorted")
    if "scene.xml" not in paths:
        raise AssetPreflightError("Go2 asset manifest does not include scene.xml")

    manifest_relative = manifest.relative_to(root).as_posix()
    actual_paths = tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.resolve() != manifest.resolve()
        )
    )
    if actual_paths != paths:
        raise AssetPreflightError("Go2 asset manifest does not cover every and only model file")
    tracked_paths = tuple(
        sorted(
            line.removeprefix("unitree_go2/")
            for line in _git_output(checkout_root, "ls-files", "--", "unitree_go2").splitlines()
            if line
        )
    )
    if tracked_paths != paths:
        raise AssetPreflightError("Go2 manifest files do not match checkout-tracked model files")
    unstaged = _git_output(
        checkout_root,
        "diff",
        "--name-only",
        "--",
        "unitree_go2",
        allow_empty=True,
    )
    staged = _git_output(
        checkout_root,
        "diff",
        "--cached",
        "--name-only",
        "--",
        "unitree_go2",
        allow_empty=True,
    )
    if unstaged or staged:
        raise AssetPreflightError("Go2 checkout has modified tracked model files")
    untracked = tuple(
        line
        for line in _git_output(
            checkout_root,
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "unitree_go2",
            allow_empty=True,
        ).splitlines()
        if line
    )
    allowed_manifest = f"unitree_go2/{manifest_relative}"
    if untracked not in {(), (allowed_manifest,)}:
        raise AssetPreflightError("Go2 checkout contains untracked model files")
    verified: list[AssetFileEvidence] = []
    for expected in files:
        path = _resolve_within(root, Path(expected.relative_path))
        if path.is_symlink() or not path.is_file():
            raise AssetPreflightError(f"Go2 asset file is missing or symlinked: {expected.relative_path}")
        size = path.stat().st_size
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if size != expected.size_bytes or digest != expected.sha256:
            raise AssetPreflightError(f"Go2 asset file does not match manifest: {expected.relative_path}")
        verified.append(expected)
    aggregate = _canonical_sha256([item.to_payload() for item in verified])
    if payload.get("aggregate_sha256") != aggregate:
        raise AssetPreflightError("Go2 aggregate asset digest does not match file records")
    if aggregate != GO2_FROZEN_ASSET_AGGREGATE_SHA256:
        raise AssetPreflightError("Go2 asset content does not match the frozen workload source")
    return AssetManifestEvidence(
        schema_version=ASSET_MANIFEST_SCHEMA_VERSION,
        requirement_id=requirement,
        source_repository=GO2_SOURCE_REPOSITORY,
        source_revision=source_revision,
        entrypoint="scene.xml",
        files=tuple(verified),
        aggregate_sha256=aggregate,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _resolve_within(root: Path, candidate: Path) -> Path:
    root_resolved = root.resolve()
    resolved = candidate.resolve() if candidate.is_absolute() else (root_resolved / candidate).resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise AssetPreflightError(f"asset path escapes its root: {candidate}")
    return resolved


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_output(
    checkout_root: Path,
    *arguments: str,
    allow_empty: bool = False,
) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AssetPreflightError(f"cannot verify Go2 Git provenance: {exc}") from exc
    output = completed.stdout.strip()
    if not output and not allow_empty:
        raise AssetPreflightError(f"Go2 Git provenance query returned no value: {arguments}")
    return output


def _sha256_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _git_commit_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )
