"""Direct-entry CPU source verifier. No worker IPC or GPU admission.

The configured release and this directly executed script are trusted bootstrap
inputs. This is not a sandbox for arbitrary code in the same process.
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from types import MappingProxyType

_BOOTSTRAP = "optical_worker/native_bootstrap.py"
_CATALOG = "optical_worker/data/native_sources/v2.json"
_MAX_BYTES = 4 * 1024 * 1024  # Rechecked against the sole protocol constant.
_ROOTS = frozenset(
    ("optical_worker", "optics", "physics", "robot", "sensing", "benchmarks", "examples", "tests", "tools")
)
_CORE = MappingProxyType(
    {
        "optical_worker": "optical_worker/__init__.py",
        "optical_worker.lifecycle": "optical_worker/lifecycle.py",
        "optical_worker.protocol": "optical_worker/protocol.py",
        "optical_worker.manifest": "optical_worker/manifest.py",
        "optical_worker.native_source_policy": "optical_worker/native_source_policy.py",
        "optical_worker.native_sources": "optical_worker/native_sources.py",
    }
)
_ACTIVE_SESSION = None


class SourceBootstrapError(RuntimeError):
    """Terminal CPU bootstrap failure; never a GPU quiescence claim."""


def _require(condition, message):
    if not condition:
        raise SourceBootstrapError(message)


def _hex(value, length):
    return type(value) is str and len(value) == length and all(c in "0123456789abcdef" for c in value)


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        _require(key not in value, "duplicate catalog key")
        value[key] = item
    return value


def _no_number(value):
    raise SourceBootstrapError("non-integral catalog number")


def _catalog_json(encoded):
    _require(0 < len(encoded) <= _MAX_BYTES and encoded.endswith(b"\n"), "catalog size/terminator")
    value = json.loads(
        encoded[:-1], object_pairs_hook=_pairs, parse_float=_no_number, parse_constant=_no_number
    )
    actual = (
        json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )
    _require(actual == encoded and type(value) is dict, "catalog is not canonical")
    return value


def _read(root, relative):
    _require(type(relative) is str and "\\" not in relative, "source path type/spelling")
    path = PurePosixPath(relative)
    _require(
        not path.is_absolute()
        and path.as_posix() == relative
        and bool(path.parts)
        and all(part not in {".", ".."} and ":" not in part for part in path.parts),
        "source path scope",
    )
    candidate = root
    for part in path.parts:
        candidate = candidate / part
        _require(not candidate.is_symlink(), "symlinked source component")
    resolved = candidate.resolve(strict=True)
    _require(resolved.is_relative_to(root) and resolved.is_file(), "source is not a contained file")
    return resolved.read_bytes()


def _capture(root, revision, lane, catalog_sha256):
    _require(
        _hex(revision, 40) and lane in {"warp", "cuda"} and _hex(catalog_sha256, 64),
        "invalid configured release",
    )
    head = (
        subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, timeout=20)
        .stdout.decode("ascii")
        .strip()
    )
    _require(head == revision, "configured revision differs from HEAD")
    released = subprocess.run(
        ["git", "show", f"{revision}:{_CATALOG}"], cwd=root, check=True, capture_output=True, timeout=20
    ).stdout
    encoded = _read(root, _CATALOG)
    _require(
        encoded == released and hashlib.sha256(encoded).hexdigest() == catalog_sha256,
        "catalog is not the configured release",
    )
    catalog = _catalog_json(encoded)
    entries = catalog.get("source_digests")
    lanes = catalog.get("lanes")
    _require(
        type(entries) is list and type(lanes) is dict and type(lanes.get(lane)) is list,
        "invalid initial catalog inventory",
    )
    digests = {}
    for entry in entries:
        _require(type(entry) is dict and set(entry) == {"relative_path", "sha256"}, "invalid digest entry")
        relative, digest = entry["relative_path"], entry["sha256"]
        _require(
            type(relative) is str and relative not in digests and _hex(digest, 64), "invalid digest identity"
        )
        digests[relative] = digest
    sources = {}
    for relative in lanes[lane]:
        _require(
            type(relative) is str
            and relative.endswith(".py")
            and relative not in sources
            and relative in digests,
            "invalid selected source",
        )
        source = _read(root, relative)
        _require(hashlib.sha256(source).hexdigest() == digests[relative], "source digest mismatch")
        sources[relative] = (source, digests[relative])
    _require(set(_CORE.values()) | {_BOOTSTRAP} <= set(sources), "verification core incomplete")
    return MappingProxyType(sources)


class _CapturedSourceLoader(importlib.abc.SourceLoader):
    def __init__(self, session, name):
        self.session, self.name = session, name
        self.relative = _CORE[name]
        self.origin = str(session.root / self.relative)
        self.source = session.sources[self.relative][0]

    def _check(self, fullname):
        self.session.check_live()
        _require(fullname == self.name and self.session.loaders.get(fullname) is self, "foreign loader/name")
        _require(
            self.source is self.session.sources[self.relative][0]
            and self.relative == _CORE[self.name]
            and self.origin == str(self.session.root / self.relative),
            "loader binding changed",
        )

    def get_filename(self, fullname):
        self._check(fullname)
        return self.origin

    def get_data(self, path):
        self._check(self.name)
        _require(path == self.origin, "loader cannot read another path or cache")
        return self.source

    def get_source(self, fullname):
        self._check(fullname)
        return self.source.decode("utf-8")

    def get_code(self, fullname):
        self._check(fullname)
        record = self.session.modules.get(fullname)
        _require(
            record is not None
            and record[1] is self
            and record[2] == "started"
            and sys.modules.get(fullname) is record[0],
            "code requested outside exact execution",
        )
        return self.source_to_code(self.source, self.origin)

    def exec_module(self, module):
        try:
            self._check(module.__name__)
            _require(
                self.name not in self.session.modules and sys.modules.get(self.name) is module,
                "module re-execution/replacement",
            )
            _require(
                module.__loader__ is self
                and module.__spec__.loader is self
                and module.__spec__.origin == self.origin,
                "module loader metadata mismatch",
            )
            self.session.modules[self.name] = (module, self, "started")
            self.session.trace.append(("started", self.name))
            super().exec_module(module)
            self.session.check_live()
            _require(self.session.modules[self.name] == (module, self, "started"), "execution record changed")
            self.session.modules[self.name] = (module, self, "completed")
            self.session.trace.append(("completed", self.name))
        except BaseException:
            self.session.failed = True
            self.session.trace.append(("failed", self.name))
            raise


class _CapturedSourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, session):
        self.session = session

    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] not in _ROOTS:
            return None
        try:
            self.session.check_live()
            _require(fullname in _CORE and target is None, "repository import outside CPU core or reload")
            _require(fullname not in self.session.loaders, "duplicate loader")
            loader = _CapturedSourceLoader(self.session, fullname)
            self.session.loaders[fullname] = loader
            return importlib.util.spec_from_loader(fullname, loader)
        except BaseException:
            self.session.failed = True
            raise


class _SourceSession:
    def __init__(self, root, revision, lane, catalog_sha256, sources):
        self.root, self.revision, self.lane = root, revision, lane
        self.catalog_sha256, self.sources = catalog_sha256, sources
        self.failed = False
        self.loaders, self.modules, self.trace = {}, {}, []
        self.finder = _CapturedSourceFinder(self)

    def check_live(self):
        _require(
            not self.failed and _ACTIVE_SESSION is self and sys.meta_path[0] is self.finder,
            "source session is failed/foreign or finder replaced",
        )

    def audit(self):
        try:
            self.check_live()
            _require(set(self.modules) == set(_CORE), "executed core incomplete")
            for name, (module, loader, state) in self.modules.items():
                _require(
                    state == "completed" and sys.modules.get(name) is module, "module identity/state drift"
                )
                loader._check(name)
                _require(
                    module.__name__ == name
                    and module.__loader__ is loader
                    and module.__spec__.loader is loader
                    and module.__spec__.origin == loader.origin
                    and module.__file__ == loader.origin,
                    "module metadata drift",
                )
                expected_path = [str(self.root / "optical_worker")] if name == "optical_worker" else None
                _require(module.__spec__.submodule_search_locations == expected_path, "package spec drift")
                if expected_path is not None:
                    _require(module.__path__ == expected_path, "package search path drift")
            _require(
                len(self.trace) == 2 * len(_CORE)
                and all(
                    self.trace.count((operation, name)) == 1
                    for name in _CORE
                    for operation in ("started", "completed")
                ),
                "execution trace drift",
            )
        except BaseException:
            self.failed = True
            raise


def _install(root, revision, lane, catalog_sha256):
    global _ACTIVE_SESSION
    _require(_ACTIVE_SESSION is None, "bootstrap is single-use")
    _ACTIVE_SESSION = object()  # Even a failed cold start cannot be retried.
    _require(
        sys.flags.isolated
        and sys.flags.no_site
        and sys.flags.dont_write_bytecode
        and sys.implementation.name == "cpython",
        "isolated CPython -I -S -B required",
    )
    _require(not any(name.partition(".")[0] in _ROOTS for name in sys.modules), "repository module preloaded")
    _require(
        not any(Path(path).resolve().is_relative_to(root) for path in sys.path), "repository on search path"
    )
    sources = _capture(root, revision, lane, catalog_sha256)
    session = _SourceSession(root, revision, lane, catalog_sha256, sources)
    _ACTIVE_SESSION = session
    sys.meta_path.insert(0, session.finder)
    return session


def _verify(session):
    try:
        session.check_live()
        from optical_worker.native_sources import _REPOSITORY_NAMES, capture_native_source_snapshot
        from optical_worker.protocol import MAX_RESPONSE_BYTES, canonical_json_bytes

        _require(
            _MAX_BYTES == MAX_RESPONSE_BYTES and _ROOTS == _REPOSITORY_NAMES, "bootstrap/core contract drift"
        )
        snapshot = capture_native_source_snapshot(
            session.root, session.lane, repository_revision=session.revision
        )
        _require(
            snapshot.catalog_sha256 == session.catalog_sha256
            and {entry.relative_path: (entry.source, entry.sha256) for entry in snapshot.sources}
            == session.sources,
            "captured sources differ from full verifier",
        )
        observed = snapshot.check_loaded_modules()
        _require(set(observed) == set(_CORE.values()) | {_BOOTSTRAP}, "actual repository ledger differs")
        _require(not {"numpy", "torch", "warp"}.intersection(sys.modules), "non-core runtime loaded")
        session.audit()
        return canonical_json_bytes(
            {
                "schema_version": "q54.p12.4.native-source-execution-check.v1",
                "status": "cpu_sources_verified",
                "lane": session.lane,
                "repository_revision": session.revision,
                "source_snapshot_sha256": snapshot.sha256,
                "catalog_sha256": snapshot.catalog_sha256,
                "repository_modules": list(observed),
                "execution_trace": [{"operation": op, "module": name} for op, name in session.trace],
            }
        )
    except BaseException:
        session.failed = True
        raise


def main():
    _require(__name__ == "__main__" and len(sys.argv) == 6, "direct source entry and five arguments required")
    root = Path(sys.argv[1]).resolve(strict=True)
    _require(Path(__file__) == root / _BOOTSTRAP, "bootstrap script is not the configured source entry")
    descriptor = int(sys.argv[5], 10)
    _require(descriptor >= 0, "invalid result descriptor")
    os.fstat(descriptor)
    session = _install(root, sys.argv[2], sys.argv[3], sys.argv[4])
    encoded = _verify(session)
    framed = len(encoded).to_bytes(4, "big") + encoded
    while framed:
        count = os.write(descriptor, framed)
        _require(count > 0, "result pipe closed")
        framed = framed[count:]
    return 0


if __name__ == "__main__":
    try:
        result = main()
    except BaseException as error:
        # CPU-only terminal failure. No partial success or GPU cleanup claim.
        print(f"source_bootstrap_failed: {type(error).__name__}: {str(error)[:512]}", file=sys.stderr)
        result = 78
    raise SystemExit(result)
