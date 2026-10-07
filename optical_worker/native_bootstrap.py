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
_CATALOG = "optical_worker/data/native_sources/v3.json"
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
_CUSTODY_PRELUDE = MappingProxyType(
    {
        "optical_worker.wait_protocol": "optical_worker/wait_protocol.py",
        "optical_worker.wait_supervision": "optical_worker/wait_supervision.py",
        "optical_worker.guarded_evidence": "optical_worker/guarded_evidence.py",
        "optical_worker.guarded_lifecycle": "optical_worker/guarded_lifecycle.py",
    }
)
_SETUP_DEFINITIONS = MappingProxyType(
    {
        "optical_worker.native_setup": "optical_worker/native_setup.py",
        "optical_worker.native_contract": "optical_worker/native_contract.py",
    }
)
# One-shot trusted-process identities, never reconstructed from a receipt.
# None is a permanent failed/in-progress binding tombstone, not permission to retry.
_SOURCE_CUSTODY = {}
_VERIFIED_SNAPSHOTS = {}


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
        self.relative = session.admitted_sources()[name]
        self.origin = str(session.root / self.relative)
        self.source = session.sources[self.relative][0]

    def _check(self, fullname):
        self.session.check_live()
        _require(fullname == self.name and self.session.loaders.get(fullname) is self, "foreign loader/name")
        _require(
            self.source is self.session.sources[self.relative][0]
            and self.relative == self.session.admitted_sources()[self.name]
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
            if self.name in _SETUP_DEFINITIONS:
                controller = self.session.bound_controller()
                controller._require_setup()
                controller.adopt_host_owner(
                    module, role="captured_repository_module", operation_id=self.name.replace(".", "_")
                )
            super().exec_module(module)
            self.session.check_live()
            _require(self.session.modules[self.name] == (module, self, "started"), "execution record changed")
            self.session.modules[self.name] = (module, self, "completed")
            self.session.trace.append(("completed", self.name))
        except BaseException as error:
            self.session.trace.append(("failed", self.name))
            self.session.fail(error)
            raise


class _CapturedSourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, session):
        self.session = session

    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in {"numpy", "torch", "warp"}:
            error = SourceBootstrapError("runtime imports remain closed in CPU source custody")
            self.session.fail(error)
            raise error
        if fullname.partition(".")[0] not in _ROOTS:
            return None
        try:
            self.session.check_live()
            _require(
                fullname in self.session.admitted_sources() and target is None,
                "repository import outside CPU core or reload",
            )
            if fullname in _SETUP_DEFINITIONS:
                self.session.bound_controller()._require_setup()
            _require(fullname not in self.session.loaders, "duplicate loader")
            loader = _CapturedSourceLoader(self.session, fullname)
            self.session.loaders[fullname] = loader
            return importlib.util.spec_from_loader(fullname, loader)
        except BaseException as error:
            self.session.fail(error)
            raise


class _SourceSession:
    def __init__(self, root, revision, lane, catalog_sha256, sources):
        self.root, self.revision, self.lane = root, revision, lane
        self.catalog_sha256, self.sources = catalog_sha256, sources
        # Retain the interpreter's trusted initial script owner separately. It
        # was NOT executed by our source loader; this is an identity hold, not
        # retroactive source-execution authority.
        self.bootstrap_owner = sys.modules["__main__"]
        self.bootstrap_loader = self.bootstrap_owner.__loader__
        self.bootstrap_spec = self.bootstrap_owner.__spec__
        self.bootstrap_package = self.bootstrap_owner.__package__
        _require(
            self.bootstrap_owner.__name__ == "__main__"
            and self.bootstrap_owner.__file__ == str(root / _BOOTSTRAP),
            "bootstrap owner is not the configured source entry",
        )
        self.failed = False
        self.failures = []
        self.phase = "core"
        self.snapshot = None
        self.loaders, self.modules, self.trace = {}, {}, []
        self.finder = _CapturedSourceFinder(self)

    def admitted_sources(self):
        if self.phase == "core":
            return _CORE
        if self.phase in {"prelude", "binding", "bound"}:
            return _CORE | _CUSTODY_PRELUDE
        if self.phase in {"importing", "imported"}:
            return _CORE | _CUSTODY_PRELUDE | _SETUP_DEFINITIONS
        raise SourceBootstrapError("invalid CPU source phase")

    def fail(self, error):
        self.failed = True
        if not any(value is error for value in self.failures):
            self.failures.append(error)
        binding = _SOURCE_CUSTODY.get(self)
        if binding is not None:
            controller = binding[0]
            # Retain the original swallowed/partial-bind error, not just a later
            # audit exception. Pre-guard adoption rejection is not auto-poison.
            controller._fail_guard("operation", error)
            if controller.termination_evidence is None and not controller._terminating:
                controller.terminate(
                    fault="setup_failure",
                    detection_phase="captured_source_failure",
                    emergency_target_synchronize=None,
                )

    def bound_controller(self):
        binding = _SOURCE_CUSTODY.get(self)
        _require(binding is not None, "source session has no exact custody binding")
        controller, guardian, launch_config, actual_config, channel, controller_type, config_type = binding
        _require(
            type(controller) is controller_type
            and controller.guardian is guardian
            and controller._guard_channel is channel
            and controller._guard_client.configuration is actual_config
            and type(launch_config) is config_type
            and type(actual_config) is config_type
            and launch_config is not actual_config
            and launch_config.to_payload() == actual_config.to_payload()
            and controller._lane_id == self.lane,
            "source custody identity changed",
        )
        return controller

    def bind_guarded_custody(self, channel, configuration_payload):
        try:
            self.check_live()
            _require(self.phase == "core" and self.snapshot is not None, "unverified binding")
            _require(self not in _SOURCE_CUSTODY, "source custody binding is single-use")
            _SOURCE_CUSTODY[self] = None
            self.phase = "prelude"
            from optical_worker.guarded_lifecycle import NativeGuardedSetupLifecycle
            from optical_worker.wait_protocol import GuardConfiguration

            self.audit()
            launch_config = GuardConfiguration.from_payload(configuration_payload)
            controller = NativeGuardedSetupLifecycle(
                lane_id=self.lane, channel=channel, configuration=launch_config
            )
            _SOURCE_CUSTODY[self] = (
                controller,
                controller.guardian,
                launch_config,
                controller._guard_client.configuration,
                channel,
                NativeGuardedSetupLifecycle,
                GuardConfiguration,
            )
            self.phase = "binding"
            for owner, role in (
                (self, "captured_source_session"),
                (self.snapshot, "source_snapshot"),
                (self.bootstrap_owner, "bootstrap_module"),
            ):
                controller.adopt_host_owner(owner, role=role, operation_id="source_custody_bind")
            for name, (module, _loader, _state) in self.modules.items():
                controller.adopt_host_owner(
                    module, role="captured_repository_module", operation_id=name.replace(".", "_")
                )
            self.bound_controller()
            self.phase = "bound"
            self.audit()
        except BaseException as error:
            self.fail(error)
            raise

    def import_setup_definitions(self):
        try:
            self.check_live()
            _require(self.phase == "bound", "definition import is single-use and ordered")
            controller = self.bound_controller()
            with controller.segment("setup"):
                self.phase = "importing"
                from optical_worker import native_setup

                _require(
                    native_setup is self.modules["optical_worker.native_setup"][0],
                    "setup definition module changed",
                )
                self.audit()
            self.phase = "imported"
        except BaseException as error:
            self.fail(error)
            raise

    def check_live(self):
        _require(
            not self.failed
            and not self.failures
            and _ACTIVE_SESSION is self
            and sys.meta_path[0] is self.finder,
            "source session is failed/foreign or finder replaced",
        )
        _require(
            sys.modules.get("__main__") is self.bootstrap_owner
            and self.bootstrap_owner.__name__ == "__main__"
            and self.bootstrap_owner.__file__ == str(self.root / _BOOTSTRAP)
            and self.bootstrap_owner.__loader__ is self.bootstrap_loader
            and self.bootstrap_owner.__spec__ is self.bootstrap_spec
            and self.bootstrap_owner.__package__ == self.bootstrap_package,
            "bootstrap owner identity/metadata drift",
        )
        verified = _VERIFIED_SNAPSHOTS.get(self)
        if verified is not None:
            snapshot, sources = verified
            _require(
                self.snapshot is snapshot
                and self.sources is sources
                and snapshot.repository_root == self.root
                and snapshot.repository_revision == self.revision
                and snapshot.lane == self.lane
                and snapshot.catalog_sha256 == self.catalog_sha256,
                "verified source snapshot identity drift",
            )
        binding = _SOURCE_CUSTODY.get(self)
        if binding is not None:
            controller = self.bound_controller()
            _require(
                not controller._terminating and not controller._replay.failed,
                "source custody is terminal or failed",
            )

    def audit(self):
        try:
            self.check_live()
            expected = self.admitted_sources()
            _require(set(self.modules) == set(expected), "executed core incomplete")
            for name, (module, loader, state) in self.modules.items():
                _require(
                    state == "completed" and sys.modules.get(name) is module, "module identity/state drift"
                )
                loader._check(name)
                expected_package = (
                    name if loader.relative.endswith("/__init__.py") else name.rpartition(".")[0]
                )
                _require(
                    module.__name__ == name
                    and module.__package__ == expected_package
                    and module.__loader__ is loader
                    and module.__spec__.name == name
                    and module.__spec__.parent == expected_package
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
                len(self.trace) == 2 * len(expected)
                and all(
                    self.trace.count((operation, name)) == 1
                    for name in expected
                    for operation in ("started", "completed")
                ),
                "execution trace drift",
            )
            if self.snapshot is not None:
                observed = self.snapshot.check_loaded_modules()
                _require(set(observed) == set(expected.values()) | {_BOOTSTRAP}, "source ledger drift")
            if _SOURCE_CUSTODY.get(self) is not None and self.phase != "binding":
                controller = self.bound_controller()
                controller.require_adopted(self, self.snapshot, self.bootstrap_owner)
                controller.require_adopted(*(record[0] for record in self.modules.values()))
        except BaseException as error:
            self.fail(error)
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
        _require(session.phase == "core", "CPU source verification cannot reopen a bound session")
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
        if session not in _VERIFIED_SNAPSHOTS:
            _VERIFIED_SNAPSHOTS[session] = (snapshot, session.sources)
        session.snapshot = _VERIFIED_SNAPSHOTS[session][0]
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
    except BaseException as error:
        session.fail(error)
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
