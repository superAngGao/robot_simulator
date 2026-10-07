"""Frozen component source snapshots and actual repository-import checks.

CPU-only and untimed. This module neither loads inspected modules nor admits
GPU work. C2d must compose these checks with the reviewed versioned bootstrap
and lifecycle; C0's manifest/dispatcher is intentionally unchanged.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from .native_source_policy import NATIVE_AST_POLICY_SHA256, validate_native_source_bytes
from .protocol import (
    MAX_RESPONSE_BYTES,
    ClosedWorkerProtocolError,
    canonical_json_bytes,
    decode_canonical_message,
)

NATIVE_SOURCE_CATALOG_VERSION = "q54.p12.4.native-component-source-catalog.v6"
NATIVE_SOURCE_SNAPSHOT_VERSION = "q54.p12.4.native-component-source-snapshot.v1"
CATALOG_PATH = "optical_worker/data/native_sources/v9.json"

# Deliberate literal inventory, not filesystem discovery. Includes eager
# physics/sensing package initialization, even where the requested leaf is CPU.
# Deferred optics exports do not authorize loading the unused optical lane.
_COMMON = (
    "optical_worker/__init__.py",
    "optical_worker/assets.py",
    "optical_worker/camera_contract.py",
    "optical_worker/cases.py",
    "optical_worker/guarded_evidence.py",
    "optical_worker/guarded_lifecycle.py",
    "optical_worker/lane_contract.py",
    "optical_worker/lifecycle.py",
    "optical_worker/manifest.py",
    "optical_worker/native_bootstrap.py",
    "optical_worker/native_contract.py",
    "optical_worker/native_input.py",
    "optical_worker/native_readback.py",
    "optical_worker/native_setup.py",
    "optical_worker/native_source_policy.py",
    "optical_worker/native_sources.py",
    "optical_worker/protocol.py",
    "optical_worker/provenance.py",
    "optical_worker/render_parameters.py",
    "optical_worker/render_sequence_contract.py",
    "optical_worker/runtime_environment.py",
    "optical_worker/timing.py",
    "optical_worker/wait_protocol.py",
    "optical_worker/wait_supervision.py",
    "optical_worker/worker.py",
    "optical_worker/workload_contract.py",
    "optical_worker/workloads.py",
    "optics/__init__.py",
    "optics/camera_presets.py",
    "optics/device_bvh.py",
    "optics/device_channel.py",
    "optics/device_scene.py",
    "optics/execution.py",
    "optics/host_bvh.py",
    "optics/mjcf_visual.py",
    "optics/native_arena.py",
    "optics/output_contract.py",
    "optics/registry.py",
    "optics/scene.py",
    "optics/scene_contract.py",
    "physics/__init__.py",
    "physics/_robot_tree_base.py",
    "physics/collision.py",
    "physics/collision_filter.py",
    "physics/constraint_solver.py",
    "physics/constraint_solvers.py",
    "physics/contact.py",
    "physics/contact_tolerances.py",
    "physics/cuda_publication.py",
    "physics/dynamics_cache.py",
    "physics/dynamics_utils.py",
    "physics/force_source.py",
    "physics/geometry.py",
    "physics/integrator.py",
    "physics/joint.py",
    "physics/publish.py",
    "physics/robot_tree.py",
    "physics/spatial.py",
    "physics/step_pipeline.py",
    "physics/telemetry.py",
    "physics/terrain.py",
    "sensing/__init__.py",
    "sensing/builders.py",
    "sensing/optical.py",
    "sensing/readings.py",
    "sensing/state_sample.py",
    "sensing/surface_query.py",
    "tools/optical_closed_worker.py",
    "tools/optical_closed_worker_supervisor.py",
)
_WARP = ("optics/warp_native_render.py",)
_CUDA = (
    "optics/_cuda_lbvh_source.py",
    "optics/cuda_arena.py",
    "optics/cuda_direct_light.py",
    "optics/cuda_lbvh.py",
    "optics/cuda_scene.py",
    "optics/cuda_timeline.py",
)
_REPOSITORY_NAMES = frozenset(("optical_worker", "optics", "physics", "robot", "sensing",
                              "benchmarks", "examples", "tests", "tools"))


def native_source_paths(lane: str) -> tuple[str, ...]:
    if type(lane) is not str or lane not in {"warp", "cuda"}:
        raise ClosedWorkerProtocolError("native source lane must be exactly warp or cuda")
    return tuple(sorted((*_COMMON, *(_WARP if lane == "warp" else _CUDA))))


def _read_contained(root: Path, relative: str) -> bytes:
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ClosedWorkerProtocolError(f"native source redirects a path component: {relative}")
    try:
        path = current.resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ClosedWorkerProtocolError(f"native source is not a contained file: {relative}")
        return path.read_bytes()
    except OSError as exc:
        raise ClosedWorkerProtocolError(f"native source is unavailable: {relative}") from exc


def _load_catalog(root: Path, revision: str) -> tuple[bytes, dict]:
    if type(revision) is not str or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ClosedWorkerProtocolError("native source revision must be a complete Git commit identity")
    try:
        actual_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, timeout=20,
        ).stdout.decode("ascii").strip()
        reviewed_catalog = subprocess.run(
            ["git", "show", f"{revision}:{CATALOG_PATH}"], cwd=root,
            check=True, capture_output=True, timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise ClosedWorkerProtocolError("native source release cannot be verified") from exc
    if actual_revision != revision:
        raise ClosedWorkerProtocolError("native source revision differs from the configured release")
    encoded = _read_contained(root, CATALOG_PATH)
    if encoded != reviewed_catalog:
        raise ClosedWorkerProtocolError("native source catalog differs from the configured Git release")
    if not encoded.endswith(b"\n"):
        raise ClosedWorkerProtocolError("native source catalog needs its canonical LF terminator")
    value = decode_canonical_message(encoded[:-1], max_bytes=MAX_RESPONSE_BYTES)
    if (
        type(value) is not dict
        or set(value) != {"schema_version", "ast_policy_sha256", "lanes", "source_digests"}
        or value["schema_version"] != NATIVE_SOURCE_CATALOG_VERSION
        or value["ast_policy_sha256"] != NATIVE_AST_POLICY_SHA256
        or value["lanes"] != {lane: list(native_source_paths(lane)) for lane in ("warp", "cuda")}
        or type(value["source_digests"]) is not list
    ):
        raise ClosedWorkerProtocolError("native source catalog differs from its literal contract")
    entries = value["source_digests"]
    if any(type(entry) is not dict or set(entry) != {"relative_path", "sha256"} for entry in entries):
        raise ClosedWorkerProtocolError("native source digest entry has invalid fields")
    if [entry["relative_path"] for entry in entries] != sorted(set(_COMMON + _WARP + _CUDA)):
        raise ClosedWorkerProtocolError("native source digest entries are not the complete ordered inventory")
    for entry in entries:
        digest = entry["sha256"]
        if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ClosedWorkerProtocolError("native source catalog has an invalid digest")
    return encoded, value


@dataclass(frozen=True)
class NativeSourceFile:
    relative_path: str
    sha256: str
    source: bytes


@dataclass(frozen=True)
class NativeRepositorySourceSnapshot:
    """Immutable checked bytes, not loaded-code or execution authority.

    The repository/release and catalog are trusted configuration, not accepted
    from IPC. Their identity must be bound by C2d's runtime manifest. A parsed
    copy of this descriptive object cannot grant a lease or GPU permission.
    """

    repository_root: Path
    repository_revision: str
    lane: str
    catalog_sha256: str
    sources: tuple[NativeSourceFile, ...]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes({
            "schema_version": NATIVE_SOURCE_SNAPSHOT_VERSION,
            "repository_revision": self.repository_revision,
            "lane": self.lane,
            "catalog_sha256": self.catalog_sha256,
            "ast_policy_sha256": NATIVE_AST_POLICY_SHA256,
            "sources": [{"path": s.relative_path, "sha256": s.sha256} for s in self.sources],
        })).hexdigest()

    def verify_unchanged(self) -> None:
        """Untimed replay against the exact catalog and actual current bytes."""
        fresh = capture_native_source_snapshot(
            self.repository_root, self.lane, repository_revision=self.repository_revision,
        )
        if fresh != self:
            raise ClosedWorkerProtocolError("native source snapshot changed after capture")

    def check_loaded_modules(self) -> tuple[str, ...]:
        """Inspect actual sys.modules; do not accept a caller-supplied ledger.

        This is a post-import boundary audit, not a pre-exec loader and not
        proof of which bytecode an earlier Python import executed. C2d must
        bind actual source consumption before it opens its dispatcher.
        """
        self.verify_unchanged()
        root = self.repository_root
        expected = {_module_name(item.relative_path): item.relative_path for item in self.sources
                    if not item.relative_path.startswith("tools/")}
        allowed = {item.relative_path for item in self.sources}
        observed = set()
        for name, module in tuple(sys.modules.items()):
            repository_name = name.partition(".")[0] in _REPOSITORY_NAMES
            if module is None:
                if repository_name:
                    raise ClosedWorkerProtocolError("native repository import is unresolved")
                continue
            # Third-party runtimes may register non-module namespace objects.
            # They are not repository sources; never invoke their __getattr__.
            if not isinstance(module, ModuleType):
                if repository_name:
                    raise ClosedWorkerProtocolError("native repository module has an invalid type")
                continue
            attributes = vars(module)
            origin = attributes.get("__file__")
            if origin is None:
                if repository_name:
                    raise ClosedWorkerProtocolError("native repository module has no source file")
                continue
            if type(origin) is not str:
                raise ClosedWorkerProtocolError("loaded module file metadata has an invalid type")
            file_path = Path(origin)
            if not file_path.is_absolute():
                if repository_name:
                    raise ClosedWorkerProtocolError("native repository module file is not absolute")
                continue
            resolved = file_path.resolve()
            in_root = resolved.is_relative_to(root)
            if not in_root and not repository_name:
                continue  # Fixed installed dependencies remain trusted runtime.
            if not in_root:
                raise ClosedWorkerProtocolError("repository module imported from a different root")
            relative = resolved.relative_to(root).as_posix()
            if relative not in allowed:
                raise ClosedWorkerProtocolError(f"repository import outside lane source catalog: {relative}")
            if name == "__main__":
                if relative not in {"tools/optical_closed_worker.py",
                                    "tools/optical_closed_worker_supervisor.py",
                                    "optical_worker/native_bootstrap.py"}:
                    raise ClosedWorkerProtocolError("unlisted repository main script")
            elif expected.get(name) != relative or attributes.get("__name__") != name:
                raise ClosedWorkerProtocolError("native repository module name/path identity mismatch")
            spec = attributes.get("__spec__")
            if name != "__main__" and (spec is None or spec.origin != str(root / relative)):
                raise ClosedWorkerProtocolError("native repository module spec/source identity mismatch")
            # An alias/symlink spelling cannot be laundered by resolve().
            if file_path != root / relative:
                raise ClosedWorkerProtocolError("native repository module file identity is redirected")
            observed.add(relative)
        return tuple(sorted(observed))


def _module_name(relative: str) -> str:
    parts = relative[:-3].split("/")
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def capture_native_source_snapshot(
    repository_root: Path, lane: str, *, repository_revision: str,
) -> NativeRepositorySourceSnapshot:
    """Verify every literal lane source before returning any checked snapshot.

    The fixed catalog is tracked release data, not caller-selected paths or
    recomputed expected hashes. No dynamic discovery or alternate catalog path.
    Actual bytes are retained; no newline normalization or missing-file fallback.
    """
    paths = native_source_paths(lane)
    root = Path(repository_root).resolve(strict=True)
    encoded, catalog = _load_catalog(root, repository_revision)
    digests = {entry["relative_path"]: entry["sha256"] for entry in catalog["source_digests"]}
    sources = []
    for relative in paths:
        source = _read_contained(root, relative)
        digest = digests[relative]
        validate_native_source_bytes(relative, source, expected_sha256=digest)
        sources.append(NativeSourceFile(relative, digest, source))
    return NativeRepositorySourceSnapshot(
        root, repository_revision, lane, hashlib.sha256(encoded).hexdigest(), tuple(sources),
    )
