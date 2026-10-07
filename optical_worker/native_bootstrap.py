"""Direct-entry CPU source verifier. No worker IPC or GPU admission.

The configured release and this directly executed script are trusted bootstrap
inputs. This is not a sandbox for arbitrary code in the same process.
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from types import MappingProxyType, ModuleType

_BOOTSTRAP = "optical_worker/native_bootstrap.py"
_CATALOG = "optical_worker/data/native_sources/v9.json"
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
_EXTERNAL_CONTRACT_DEFINITIONS = MappingProxyType(
    {"optical_worker.runtime_environment": "optical_worker/runtime_environment.py"}
)
_MATERIALIZATION_SOURCES = MappingProxyType(
    {
        "optical_worker.assets": "optical_worker/assets.py",
        "optical_worker.camera_contract": "optical_worker/camera_contract.py",
        "optical_worker.cases": "optical_worker/cases.py",
        "optical_worker.lane_contract": "optical_worker/lane_contract.py",
        "optical_worker.native_input": "optical_worker/native_input.py",
        "optical_worker.provenance": "optical_worker/provenance.py",
        "optical_worker.render_parameters": "optical_worker/render_parameters.py",
        "optical_worker.render_sequence_contract": "optical_worker/render_sequence_contract.py",
        "optical_worker.timing": "optical_worker/timing.py",
        "optical_worker.workload_contract": "optical_worker/workload_contract.py",
        "optical_worker.workloads": "optical_worker/workloads.py",
        "optics": "optics/__init__.py",
        "optics.camera_presets": "optics/camera_presets.py",
        "optics.host_bvh": "optics/host_bvh.py",
        "optics.mjcf_visual": "optics/mjcf_visual.py",
        "optics.registry": "optics/registry.py",
        "optics.scene_contract": "optics/scene_contract.py",
        "physics": "physics/__init__.py",
        "physics._robot_tree_base": "physics/_robot_tree_base.py",
        "physics.collision": "physics/collision.py",
        "physics.collision_filter": "physics/collision_filter.py",
        "physics.constraint_solver": "physics/constraint_solver.py",
        "physics.constraint_solvers": "physics/constraint_solvers.py",
        "physics.contact": "physics/contact.py",
        "physics.contact_tolerances": "physics/contact_tolerances.py",
        "physics.cuda_publication": "physics/cuda_publication.py",
        "physics.dynamics_cache": "physics/dynamics_cache.py",
        "physics.dynamics_utils": "physics/dynamics_utils.py",
        "physics.force_source": "physics/force_source.py",
        "physics.geometry": "physics/geometry.py",
        "physics.integrator": "physics/integrator.py",
        "physics.joint": "physics/joint.py",
        "physics.publish": "physics/publish.py",
        "physics.robot_tree": "physics/robot_tree.py",
        "physics.spatial": "physics/spatial.py",
        "physics.step_pipeline": "physics/step_pipeline.py",
        "physics.telemetry": "physics/telemetry.py",
        "physics.terrain": "physics/terrain.py",
        "sensing": "sensing/__init__.py",
        "sensing.builders": "sensing/builders.py",
        "sensing.optical": "sensing/optical.py",
        "sensing.readings": "sensing/readings.py",
        "sensing.state_sample": "sensing/state_sample.py",
        "sensing.surface_query": "sensing/surface_query.py",
    }
)
_RUNTIME_PHASES = frozenset(
    {
        "materializing",
        "materialized",
        "initializing_runtime",
        "runtime_ready",
        "uploading",
        "uploaded",
        "validating_input",
        "input_validated",
    }
)
_VALIDATION_SOURCES = MappingProxyType(
    {"optical_worker.native_readback": "optical_worker/native_readback.py"}
)
# One-shot trusted-process identities, never reconstructed from a receipt.
# None is a permanent failed/in-progress binding tombstone, not permission to retry.
_SOURCE_CUSTODY = {}
_VERIFIED_SNAPSHOTS = {}
_EXTERNAL_CUSTODY = {}
_RUNTIME_BINDINGS = {}
_UPLOADED_INPUTS = {}
_IMPORT_TOPOLOGIES = {}
_GENERATED_STDLIB_SPECS = {}
_GENERATED_STDLIB_MODULES = {}


class SourceBootstrapError(RuntimeError):
    """Terminal CPU bootstrap failure; never a GPU quiescence claim."""


def _require(condition, message):
    if not condition:
        raise SourceBootstrapError(message)


def _generated_interpreter_identity():
    # Pure CPU startup facts; never ask sysconfig (which honors an override).
    return (
        sys.executable,
        tuple(sys.version_info),
        getattr(sys, "abiflags", ""),
        sys.platform,
        getattr(sys.implementation, "_multiarch", ""),
        getattr(sys, "_stdlib_dir", None),
        os.name,
    )


def _capture_generated_stdlib(root, paths):
    _require("_PYTHON_SYSCONFIGDATA_NAME" not in os.environ, "sysconfig name override forbidden")
    identity = _generated_interpreter_identity()
    if os.name != "posix" or sys.version_info[:2] != (3, 12):
        return (identity, None, None, None, None)
    _, _, abi, platform, multiarch, directory, _ = identity
    _require(
        all(type(v) is str and all(c.isalnum() or c in "_-" for c in v) for v in (abi, platform, multiarch)),
        "invalid generated interpreter name facts",
    )
    _require(type(directory) is str and bool(directory), "missing exact interpreter stdlib directory")
    base = Path(directory)
    _require(
        base.is_absolute()
        and base.resolve(strict=True) == base
        and directory in paths
        and not base.is_relative_to(root)
        and not {p.lower() for p in base.parts} & {"site-packages", "dist-packages"},
        "generated stdlib root is not an exact cold interpreter directory",
    )
    name = f"_sysconfigdata_{abi}_{platform}_{multiarch}"
    origin = base / (name + ".py")
    _require(
        origin.is_file() and not origin.is_symlink() and origin.resolve(strict=True) == origin,
        "generated stdlib source missing or redirected",
    )
    source = origin.read_bytes()
    # Standard CPython loader/cache trust, not held-source bytecode attestation.
    return (identity, name, str(origin), len(source), hashlib.sha256(source).hexdigest())


def _is_interpreter_name(name, names, generated=None):
    return name.partition(".")[0] in names or (
        generated is not None and generated[1] is not None and name == generated[1]
    )


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
            if self.name not in _CORE and self.name not in _CUSTODY_PRELUDE:
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
        try:
            topology = self.session.check_import_topology()
            if self.session.is_interpreter_name(fullname):
                if fullname == topology[6][1]:
                    return self.session.generated_stdlib_spec(fullname, path, target)
                # Terminal reporting may need unloaded stdlib modules after
                # poison. Topology integrity does not reopen session liveness.
                for owner, method in topology[4]:
                    spec = method(fullname, path, target)
                    self.session.check_import_topology()
                    if spec is not None:
                        return spec
                return None
        except BaseException as error:
            self.session.fail(error)
            raise
        external = _EXTERNAL_CUSTODY.get(self.session)
        if external is not None and fullname.partition(".")[0] not in _ROOTS:
            spec = external[0].find_spec(fullname, path, target)
            if type(spec) is _AbsentExternalSpec:
                try:
                    _require(spec.name == fullname, "absent resolution identity changed")
                    self.session.check_import_topology()
                except BaseException as error:
                    external[0].fail(error, fullname)
                    raise
                # No tail finder exists. Normal importlib returns None for a
                # metadata query and raises for an actual missing import.
                return None
            if type(spec) is _DisabledExternalToken:
                # The only deliberate rejection edge. Validation errors inside
                # consume_disabled still poison; arbitrary ImportError does not
                # receive an exemption from admission's failure handler.
                error = external[0].consume_disabled(spec, fullname)
                raise error
            return spec
        if fullname.partition(".")[0] in {"numpy", "torch", "warp"}:
            error = SourceBootstrapError("runtime imports remain closed in CPU source custody")
            self.session.fail(error)
            raise error
        if fullname.partition(".")[0] not in _ROOTS:
            error = SourceBootstrapError("external imports require configured custody")
            self.session.fail(error)
            raise error
        try:
            self.session.check_live()
            _require(
                fullname in self.session.admitted_sources() and target is None,
                "repository import outside CPU core or reload",
            )
            if fullname not in _CORE and fullname not in _CUSTODY_PRELUDE:
                self.session.bound_controller()._require_setup()
            _require(fullname not in self.session.loaders, "duplicate loader")
            loader = _CapturedSourceLoader(self.session, fullname)
            self.session.loaders[fullname] = loader
            return importlib.util.spec_from_loader(fullname, loader)
        except BaseException as error:
            self.session.fail(error)
            raise


class _ExternalSourceLoader(importlib.abc.SourceLoader):
    """One retained standard file loader; Python bytecode caches are not input."""

    def __init__(self, admission, name, delegate, origin, source):
        self.admission, self.name = admission, name
        self.delegate, self.origin, self.source = delegate, origin, source

    def _check(self):
        self.admission.check_live()
        self._check_binding()

    def _check_binding(self):
        # Exact local facts are checked at every supported boundary. The caller
        # already checks session/environment liveness before exec or audit;
        # do not repeat a whole environment traversal once per audited module.
        _require(self.admission.loaders.get(self.name) is self, "external loader identity changed")
        binding = self.admission.loader_bindings[self.name]
        _require(
            binding[0] is self and binding[1] is self.delegate
            and binding[2] == self.origin and binding[3] is self.source,
            "external loader binding changed",
        )
        delegate_type = (
            importlib.machinery.ExtensionFileLoader
            if self.source is None else importlib.machinery.SourceFileLoader
        )
        _require(
            type(self.delegate) is delegate_type
            and self.delegate.name == self.name and self.delegate.path == self.origin,
            "external delegate metadata drift",
        )

    def get_filename(self, fullname):
        self._check()
        _require(fullname == self.name, "external filename requested for another module")
        return self.origin

    def get_data(self, path):
        self._check()
        _require(path == self.origin and self.source is not None, "external source cannot read a cache")
        return self.source

    def get_code(self, fullname):
        self._check()
        record = self.admission.records.get(self.name)
        _require(
            fullname == self.name
            and self.source is not None
            and record is not None
            and record["state"] == "executing"
            and sys.modules.get(fullname) is record["module"],
            "external code requested outside exact execution",
        )
        return self.source_to_code(self.source, self.origin)

    def create_module(self, spec):
        try:
            self._check()
            _require(spec is self.admission.specs[self.name], "external spec identity changed")
            _require(self.name not in self.admission.records, "external module recreation")
            self.admission.check_self_alias_absence(self.name, "precreate")
            self.admission.record("create_started", self.name, "file_module")
            self.admission.stack.append(self.name)
            if type(self.delegate) is importlib.machinery.ExtensionFileLoader:
                module = self.delegate.create_module(spec)
                _require(module is not None, "native loader returned no module")
            else:
                module = ModuleType(self.name)
            # This registry is held by the adopted source session before any
            # explicit adoption that can fail. Preserve partial native owners.
            self.admission.records[self.name] = {"module": module, "state": "created", "kind": "file_module"}
            # Native return is already retained. Importlib has not initialized
            # spec/loader/package yet, so inspect only the raw intrinsic name.
            self.admission.check_module_name(self.name)
            self.admission.check_self_alias_absence(self.name, "postcreate")
            self.admission.hold(module)
            self.admission.record("created", self.name, "file_module")
            self.admission.stack.pop()
            return module
        except BaseException as error:
            self.admission.fail(error, self.name, native_create=self.source is None)
            raise

    def exec_module(self, module):
        try:
            self._check()
            record = self.admission.records.get(self.name)
            _require(
                record is not None
                and record["module"] is module
                and record["state"] == "created"
                and sys.modules.get(self.name) is module
                and module.__spec__ is self.admission.specs[self.name]
                and module.__loader__ is self,
                "external module execution identity/order changed",
            )
            # Importlib has initialized metadata now. Reject drift before the
            # delegated native exec, not merely after its side effects.
            self.admission.check_module(self.name)
            self.admission.check_self_alias_absence(self.name, "preexec")
            record["state"] = "executing"
            self.admission.record("exec_started", self.name, "file_module")
            self.admission.stack.append(self.name)
            if self.source is None:
                self.delegate.exec_module(module)
            else:
                super().exec_module(module)
            self.admission.check_live()
            self.admission.confirm_disabled_fallbacks(self.name)
            self.admission.reconcile_publications(self.name, admitting=True)
            self.admission.stack.pop()
            record["state"] = "completed"
            self.admission.check_module(self.name)
            self.admission.record("completed", self.name, "file_module")
        except BaseException as error:
            self.admission.fail(error, self.name, native_create=self.source is None)
            raise


class _ExternalAdmission:
    """Fixed external rules belonging to the existing captured source session.

    No independently installable finder, public module selector or import
    capability is provided. A parsed environment is not enough to construct
    this object: source custody and an actual setup guard must already exist.
    """

    def __init__(self, session, environment, expected_sha256):
        _require(type(session) is _SourceSession, "external admission needs the exact source session")
        session.check_live()
        controller = session.bound_controller()
        controller._require_setup()
        environment.verify_files(expected_sha256=expected_sha256)
        self.session, self.controller, self.environment = session, controller, environment
        self.expected_sha256 = expected_sha256
        payload = environment.to_payload()
        self.payload = payload
        self.roots = {r["id"]: r["path"] for r in payload["roots"]}
        self.search_roots = tuple(str(Path(r["path"])) for r in payload["roots"] if r["kind"] == "site")
        self.files = {(r["root"], r["path"]): r for r in payload["files"]}
        self.rules = {r["name"]: r for r in payload["modules"]}
        self.publications = {r["name"]: r for r in payload["publications"]}
        self.namespaces = {r["name"]: r for r in payload["namespaces"]}
        self.absent = {(r["name"], r["requester"], r["phase"]) for r in payload["absent_probes"]}
        self.disabled = {r["name"]: r for r in payload["disabled_optional_modules"]}
        self.disabled_records, self.disabled_tokens, self.disabled_bindings = {}, {}, {}
        self.materialization_case, self.materialization_finished = None, False
        self.phase = "materialization"
        self.stack, self.events, self.retained, self.failures = [], [], [], []
        self.failure_contexts, self.self_alias_checks = [], {}
        self.loaders, self.loader_bindings, self.specs, self.records = {}, {}, {}, {}
        self.baseline = dict(sys.modules)
        self.failed = False
        _require(
            not any(name in self.baseline for name in self.rules | self.publications | self.namespaces)
            and not any(self._has_module_prefix(name) for name in self.disabled),
            "external dependency was preloaded",
        )
        controller.adopt_host_owner(
            environment, role="runtime_environment", operation_id="runtime_environment"
        )
        controller.adopt_host_owner(
            self, role="runtime_import_custody", operation_id="runtime_import_custody"
        )

    def record(self, operation, name, kind):
        self.events.append(
            {
                "index": len(self.events),
                "operation": operation,
                "name": name,
                "kind": kind,
                "phase": self.phase,
            }
        )

    def hold(self, value):
        if not any(owner is value for owner in self.retained):
            self.retained.append(value)
            self.controller.adopt_host_owner(
                value,
                role="external_import_owner",
                operation_id="external_import_owner",
            )

    def check_live(self):
        _require(not self.failed, "external import session failed")
        self.session.check_live()
        binding = _EXTERNAL_CUSTODY.get(self.session)
        _require(
            binding is not None
            and binding[0] is self
            and binding[1] is self.environment
            and binding[2] == self.expected_sha256
            and binding[3] is self.controller,
            "external custody was not issued by this source session",
        )
        _require(self.session.bound_controller() is self.controller, "external controller identity changed")
        self.controller._require_setup()
        _require(
            os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0", "external backend plugin setting drift"
        )
        _require(self.environment.sha256 == self.expected_sha256, "external environment identity drift")
        _require(self.environment.to_payload() == self.payload, "external rule table drift")
        p = self.payload
        _require(
            self.roots == {r["id"]: r["path"] for r in p["roots"]}
            and self.search_roots == tuple(str(Path(r["path"])) for r in p["roots"] if r["kind"] == "site")
            and self.files == {(r["root"], r["path"]): r for r in p["files"]}
            and self.rules == {r["name"]: r for r in p["modules"]}
            and self.publications == {r["name"]: r for r in p["publications"]}
            and self.namespaces == {r["name"]: r for r in p["namespaces"]}
            and self.absent == {(r["name"], r["requester"], r["phase"]) for r in p["absent_probes"]},
            "external lookup index drift",
        )
        _require(
            self.disabled == {r["name"]: r for r in p["disabled_optional_modules"]},
            "disabled optional index drift",
        )

    def fail(self, error, name, *, native_create=False):
        first_failure = not self.failed
        self.failed = True
        if not any(value is error for value in self.failures):
            self.failures.append(error)
        if not any(
            e is error and n == name and native == native_create for e, n, native in self.failure_contexts
        ):
            self.failure_contexts.append((error, name, native_create))
        # Custody, never permission: keep any partial publication/root alive even
        # if importlib removes it or adoption subsequently rejects poisoned work.
        # A caught inner failure does not stop its already executing parent from
        # publishing another owner. Extend the held graph on EVERY failure; do
        # not call normal guardian adoption on a quarantined lifecycle.
        retained_ids = {id(owner) for owner in self.retained}
        for key, module in tuple(sys.modules.items()):
            if self.baseline.get(key) is not module and id(module) not in retained_ids:
                self.retained.append(module)
                retained_ids.add(id(module))
        if native_create and not self.controller._terminating:
            try:
                self.controller.mark_unknown_device_scope(
                    source="external_native_import",
                    reason="native_import_failed_before_complete_custody",
                )
            except BaseException as scope_error:
                self.failures.append(scope_error)
        if not first_failure:
            # Keep late native context above, even if the controller has already
            # committed process-terminal quarantine. Never rewrite that terminal
            # evidence or start another terminal operation during unwinding.
            return
        try:
            self.record("failed", name, "failure")
        except BaseException as record_error:
            self.failures.append(record_error)
        self.session.fail(error)

    def _paths(self, fullname):
        parent = fullname.rpartition(".")[0]
        if not parent:
            return self.search_roots
        _require(parent in self.records, "external parent was not admitted")
        parent_record = self.records[parent]
        if parent_record["kind"] == "namespace":
            return tuple(
                str(Path(self.roots[r["root"]]) / r["path"]) for r in self.namespaces[parent]["locations"]
            )
        rule = self.rules.get(parent)
        _require(rule is not None and rule["is_package"], "external parent is not a file package")
        return (str((Path(self.roots[rule["root"]]) / rule["path"]).parent),)

    def find_spec(self, fullname, path=None, target=None):
        if self.session.is_interpreter_name(fullname):
            return None  # The pinned interpreter's standard library is trusted.
        try:
            self.check_live()
            _require(target is None and fullname not in self.records, "external reload or duplicate import")
            expected_paths = self._paths(fullname)
            _require(
                path is None if not fullname.rpartition(".")[0] else tuple(path or ()) == expected_paths,
                "external package search path drift",
            )
            requester = self.stack[-1] if self.stack else None
            if fullname in self.disabled:
                return self._prepare_disabled(fullname, requester, expected_paths)
            rule = self.rules.get(fullname)
            namespace = self.namespaces.get(fullname)
            optional = (fullname, requester, self.phase) in self.absent
            _require(rule is not None or namespace is not None or optional, "unlisted external import")
            spec = importlib.machinery.PathFinder.find_spec(fullname, list(expected_paths))
            if optional:
                _require(
                    fullname not in sys.modules and spec is None, "expected absent dependency is present"
                )
                self.record("absent_probe", fullname, "absent_probe")
                # No-code absence, not a started-import exception exemption.
                return _AbsentExternalSpec(fullname)
            _require(spec is not None, "frozen external module missing")
            if namespace is not None:
                _require(namespace["phase"] == self.phase, "namespace phase mismatch")
                locations = [str(Path(self.roots[r["root"]]) / r["path"]) for r in namespace["locations"]]
                _require(
                    spec.loader is None
                    and spec.origin is None
                    and list(spec.submodule_search_locations) == locations,
                    "namespace identity drift",
                )
                # importlib creates a namespace object without executing a body.
                # A controlled wrapper adopts it before returning to its caller.
                loader = _ExternalNamespaceLoader(self, fullname, locations)
                spec = importlib.util.spec_from_loader(fullname, loader, is_package=True)
                spec.origin = None
                spec.submodule_search_locations = locations
                self.specs[fullname] = spec
                self.loaders[fullname] = loader
                return spec
            _require(rule["phase"] == self.phase, "file module phase mismatch")
            origin = str(Path(self.roots[rule["root"]]) / rule["path"])
            file_record = self.files[(rule["root"], rule["path"])]
            delegate_type = (
                importlib.machinery.SourceFileLoader
                if file_record["kind"] == "python"
                else importlib.machinery.ExtensionFileLoader
            )
            _require(
                type(spec.loader) is delegate_type and spec.origin == origin,
                "external file origin/loader drift",
            )
            _require(
                (spec.submodule_search_locations is not None) == rule["is_package"], "external package drift"
            )
            candidate = Path(origin)
            _require(
                not candidate.is_symlink() and candidate.resolve(strict=True) == candidate,
                "external path resolved differently",
            )
            data = candidate.read_bytes()
            _require(
                len(data) == file_record["size"]
                and hashlib.sha256(data).hexdigest() == file_record["sha256"],
                "external file bytes changed",
            )
            source = data if file_record["kind"] == "python" else None
            loader = _ExternalSourceLoader(self, fullname, spec.loader, origin, source)
            self.loaders[fullname] = loader
            self.loader_bindings[fullname] = (loader, spec.loader, origin, source)
            self.hold(spec.loader)
            spec.loader = loader
            self.specs[fullname] = spec
            self.hold(spec)
            self.record("file_resolved", fullname, "file_module")
            return spec
        except BaseException as error:
            self.fail(error, fullname)
            raise

    @staticmethod
    def _has_module_prefix(name):
        return any(key == name or key.startswith(name + ".") for key in sys.modules)

    def _disabled_resolution(self, name, *, retain=False):
        """Metadata-only resolution, also used by read-only segment audits."""
        rule = self.disabled[name]
        _require(not self._has_module_prefix(name), "disabled adapter was published or preloaded")
        _require(
            name not in self.records and name not in self.loaders, "disabled adapter has executable custody"
        )
        spec = importlib.machinery.PathFinder.find_spec(name, list(self._paths(name)))
        # Retain a found metadata owner before any further fallible check. This
        # does not create or execute the adapter module.
        if spec is not None and retain:
            self.hold(spec)
            self.hold(spec.loader)
        origin = str(Path(self.roots[rule["root"]]) / rule["path"])
        _require(
            spec is not None
            and type(spec.loader) is importlib.machinery.SourceFileLoader
            and spec.origin == origin
            and spec.submodule_search_locations is None,
            "disabled adapter spec/origin/kind drift",
        )
        candidate = Path(origin)
        _require(
            not candidate.is_symlink() and candidate.resolve(strict=True) == candidate,
            "disabled adapter path drift",
        )
        data = candidate.read_bytes()
        file_record = self.files[(rule["root"], rule["path"])]
        _require(
            len(data) == file_record["size"] and hashlib.sha256(data).hexdigest() == file_record["sha256"],
            "disabled adapter file bytes changed",
        )
        prerequisite = rule["absent_prerequisite"]
        _require(not self._has_module_prefix(prerequisite), "disabled prerequisite was preloaded")
        prerequisite_spec = importlib.machinery.PathFinder.find_spec(prerequisite, list(self.search_roots))
        if prerequisite_spec is not None and retain:
            self.hold(prerequisite_spec)
            self.hold(prerequisite_spec.loader)
        _require(prerequisite_spec is None, "disabled prerequisite is unexpectedly present")
        return spec

    def _prepare_disabled(self, name, requester, expected_paths):
        rule = self.disabled[name]
        _require(
            self.phase == rule["phase"] == "materialization" and requester == rule["requester"],
            "disabled optional requester/phase mismatch",
        )
        _require(name not in self.disabled_records, "disabled optional request is single-use")
        owner = self.records.get(requester)
        _require(
            owner is not None and owner["kind"] == "file_module" and owner["state"] == "executing",
            "disabled requester is not an executing file module",
        )
        self.check_module(requester)
        _require(tuple(expected_paths) == self._paths(name), "disabled search path changed")
        record = {
            "state": "unrequested",
            "requester": owner["module"],
            "name": name,
            "spec": None,
            "error": None,
            "fallback": None,
            "token": None,
        }
        # This held registry keeps partial resolution/denial construction alive.
        self.disabled_records[name] = record
        spec = self._disabled_resolution(name, retain=True)
        record["spec"] = spec
        record["state"] = "resolution_verified"
        error = ImportError(f"Frozen optional adapter disabled: {name}", name=name, path=spec.origin)
        record["error"] = error
        self.hold(error)
        token = _DisabledExternalToken()
        record["token"] = token
        self.disabled_tokens[token] = record
        self.disabled_bindings[name] = (record, owner["module"], spec, error, token, spec.loader)
        self.hold(token)
        return token

    def _check_disabled_identity(self, name):
        record = self.disabled_records[name]
        binding = self.disabled_bindings.get(name)
        _require(
            binding is not None
            and binding[0] is record
            and binding[1] is record["requester"]
            and binding[2] is record["spec"]
            and binding[3] is record["error"]
            and binding[4] is record["token"]
            and binding[5] is record["spec"].loader,
            "disabled construction identity changed",
        )
        rule = self.disabled[name]
        spec, error = record["spec"], record["error"]
        origin = str(Path(self.roots[rule["root"]]) / rule["path"])
        _require(
            spec.name == name
            and spec.origin == origin
            and spec.submodule_search_locations is None
            and type(spec.loader) is importlib.machinery.SourceFileLoader
            and type(error) is ImportError
            and error.args == (f"Frozen optional adapter disabled: {name}",)
            and error.name == name
            and error.path == origin,
            "disabled resolution/error identity drift",
        )

    def consume_disabled(self, token, name):
        try:
            self.check_live()
            record = self.disabled_tokens.get(token) if type(token) is _DisabledExternalToken else None
            _require(
                record is not None
                and self.disabled_records.get(name) is record
                and record["name"] == name
                and record["token"] is token
                and record["state"] == "resolution_verified",
                "unissued, foreign or consumed disabled token",
            )
            self._check_disabled_identity(name)
            rule = self.disabled[name]
            requester = rule["requester"]
            _require(
                self.phase == "materialization"
                and self.stack
                and self.stack[-1] == requester
                and self.records[requester]["module"] is record["requester"]
                and self.records[requester]["state"] == "executing",
                "disabled token requester drift",
            )
            self.check_module(requester)
            error = record["error"]
            del self.disabled_tokens[token]
            record["state"] = "rejection_issued"
            self._record_disabled(name, "disabled_optional")
            return error
        except BaseException as error:
            self.fail(error, name)
            raise

    def _disabled_fallback(self, name):
        self._check_disabled_identity(name)
        record, rule = self.disabled_records[name], self.disabled[name]
        requester = self.records[rule["requester"]]["module"]
        _require(requester is record["requester"], "disabled fallback owner identity changed")
        wrapper_owner = self.records.get("trimesh.exceptions")
        _require(
            wrapper_owner is not None
            and wrapper_owner["state"] == "completed"
            and wrapper_owner["kind"] == "file_module",
            "fallback class owner incomplete",
        )
        self.check_module("trimesh.exceptions")
        placeholder = vars(requester).get(rule["fallback_attribute"])
        # Retain before inspecting an object whose attribute access may raise.
        if placeholder is not None:
            self.hold(placeholder)
        wrapper_class = vars(wrapper_owner["module"]).get("ExceptionWrapper")
        _require(
            type(wrapper_class) is type and type(placeholder) is wrapper_class,
            "disabled fallback has wrong exact type",
        )
        fields = object.__getattribute__(placeholder, "__dict__")
        expected = (ImportError, record["error"].args)
        _require(
            type(fields) is dict
            and type(fields.get("exception")) is tuple
            and fields["exception"] == expected,
            "disabled fallback error differs from issued error",
        )
        if rule["requester"] == "trimesh.ray":
            _require(vars(requester).get("has_embree") is False, "disabled ray fallback flag drift")
        _require(
            not self._has_module_prefix(name) and name not in self.records and name not in self.loaders,
            "disabled adapter unexpectedly executed/published",
        )
        return placeholder

    def confirm_disabled_fallbacks(self, requester):
        for name, rule in self.disabled.items():
            if rule["requester"] != requester:
                continue
            # importlib can answer from sys.modules (even a None sentinel)
            # without consulting our finder. Check configured rows as well as
            # issued denials before accepting the parent's completed body.
            _require(not self._has_module_prefix(name), "disabled adapter was preloaded/published")
            record = self.disabled_records.get(name)
            if record is None:
                continue
            _require(record["state"] == "rejection_issued", "disabled fallback confirmation order")
            record["fallback"] = self._disabled_fallback(name)
            record["state"] = "fallback_confirmed"
            self._record_disabled(name, "disabled_fallback_confirmed")

    def _record_disabled(self, name, operation):
        record, rule = self.disabled_records[name], self.disabled[name]
        requester_rule = self.rules[rule["requester"]]

        def digest(value):
            return hashlib.sha256(
                json.dumps(
                    value, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()

        def owner_token(value):
            if value is None:
                return None
            for index, owner in enumerate(self.retained):
                if owner is value:
                    return f"external-owner-{index}"
            raise SourceBootstrapError("disabled evidence owner has no custody identity")

        self.events.append(
            {
                "index": len(self.events),
                "operation": operation,
                "name": name,
                "kind": "disabled_optional",
                "phase": self.phase,
                "environment_sha256": self.expected_sha256,
                "rule_sha256": digest(rule),
                "requester": rule["requester"],
                "requester_identity": owner_token(record["requester"]),
                "requester_source_sha256": self.files[(requester_rule["root"], requester_rule["path"])][
                    "sha256"
                ],
                "origin": record["spec"].origin,
                "source_sha256": self.files[(rule["root"], rule["path"])]["sha256"],
                "loader_kind": "SourceFileLoader",
                "present": True,
                "absent_prerequisite": rule["absent_prerequisite"],
                "prerequisite_spec": None,
                "request_id": name,
                "fallback_attribute": rule["fallback_attribute"] if record["fallback"] is not None else None,
                "fallback_identity": owner_token(record["fallback"]),
                "create_count": 0,
                "exec_count": 0,
                "body_count": 0,
                "native_create_count": 0,
                "error_type": "ImportError",
                "error_args": list(record["error"].args),
                "error_name": record["error"].name,
                "error_path": record["error"].path,
            }
        )

    def bind_materialization_case(self, case_id):
        self.check_live()
        _require(
            self.materialization_case is None and self.session.phase == "materializing",
            "materialization case is single-use and source-session owned",
        )
        self.materialization_case = case_id
        if case_id in {"go2_static_960x640_v1", "go2_static_1920x1080_v1"}:
            _require(
                set(self.disabled) == set(self.environment.required_go2_disabled_names),
                "Go2 requires both frozen disabled adapter rules",
            )

    def finish_materialization(self):
        self.check_live()
        _require(
            self.materialization_case is not None
            and not self.materialization_finished
            and self.session.phase == "materializing"
            and self.session.canonical_input.manifest.case_id == self.materialization_case,
            "materialization completion is not bound to the actual canonical source",
        )
        self._check_required_disabled()
        self.materialization_finished = True

    def _check_required_disabled(self):
        if self.materialization_case in {"go2_static_960x640_v1", "go2_static_1920x1080_v1"}:
            _require(
                set(self.disabled) == set(self.environment.required_go2_disabled_names),
                "Go2 disabled table drift",
            )
            for name in self.environment.required_go2_disabled_names:
                _require(
                    name in self.disabled_records
                    and self.disabled_records[name]["state"] == "fallback_confirmed",
                    "Go2 materialization lacks a required fallback confirmation",
                )

    def check_self_alias_absence(self, creator, boundary):
        rows = [r for r in self.publications.values() if r["creator"] == creator and not r["attributes"]]
        if not rows:
            return
        expected = {"precreate": (), "postcreate": ("precreate",), "preexec": ("precreate", "postcreate")}
        _require(
            boundary in expected and self.self_alias_checks.get(creator, ()) == expected[boundary],
            "creator-self absence boundary order changed",
        )
        _require(all(r["phase"] == self.phase for r in rows), "creator-self phase changed")
        _require(all(r["name"] not in sys.modules for r in rows), "creator-self alias published early")
        self.self_alias_checks[creator] = expected[boundary] + (boundary,)
        self.record("self_alias_" + boundary + "_absent", creator, "file_module")

    def reconcile_publications(self, publisher, *, admitting=False):
        publisher_record = self.records.get(publisher)
        if publisher_record is None:
            _require(
                not admitting
                and all(
                    r["name"] not in sys.modules and r["name"] not in self.records
                    for r in self.publications.values()
                    if r["publisher"] == publisher
                ),
                "unrequested publisher has a publication",
            )
            return
        if admitting:
            _require(
                publisher_record["kind"] == "file_module"
                and publisher_record["state"] == "executing"
                and self.stack
                and self.stack[-1] == publisher,
                "publication admission requires actual publisher execution",
            )
        else:
            _require(
                publisher_record["state"] == "completed", "publication audit before publisher completion"
            )
        for name, rule in self.publications.items():
            if rule["publisher"] != publisher:
                continue
            previous = self.records.get(name)
            _require((previous is None) == admitting, "publication cannot be admitted twice or during audit")
            _require(name in sys.modules, "required creator publication missing")
            _require(
                rule["phase"] == (previous["phase"] if previous is not None else self.phase),
                "publication phase mismatch",
            )
            root = self.records.get(rule["creator"])
            _require(
                root is not None and sys.modules.get(rule["creator"]) is root["module"], "creator root drift"
            )
            value = root["module"]
            relation = "creator_attribute" if rule["attributes"] else "creator_self"
            if relation == "creator_self":
                _require(
                    root is publisher_record
                    and self.self_alias_checks.get(publisher) == ("precreate", "postcreate", "preexec")
                    and type(self.loader_bindings[publisher][1]) is importlib.machinery.ExtensionFileLoader,
                    "creator-self native execution boundary changed",
                )
                self.check_module(publisher)
            for attribute in rule["attributes"]:
                # Do not invoke third-party __getattr__ and create new objects
                # while supposedly auditing a previously published object.
                value = vars(value).get(attribute)
                _require(value is not None, "creator attribute path missing")
            _require(
                value is sys.modules[name] and isinstance(value, ModuleType), "publication owner mismatch"
            )
            if previous is not None:
                _require(
                    previous["module"] is value
                    and previous["creator"] is root["module"]
                    and previous["publisher"] is publisher_record["module"]
                    and previous["relation"] == relation,
                    "published module replaced",
                )
                continue
            self.records[name] = {
                "kind": rule["kind"],
                "state": "completed",
                "module": value,
                "creator": root["module"],
                "publisher": publisher_record["module"],
                "phase": self.phase,
                "relation": relation,
            }
            self.hold(value)
            self.record("creator_publication", name, rule["kind"])

    def check_module_name(self, name):
        record = self.records[name]
        _require(record["kind"] == "file_module", "intrinsic name requires a file record")
        actual = vars(record["module"]).get("__name__")
        expected = self.environment.expected_file_module_name(name)
        _require(type(actual) is str and actual == expected, "external intrinsic module name drift")

    def check_module(self, name):
        record = self.records[name]
        _require(sys.modules.get(name) is record["module"], "external sys.modules identity drift")
        if record["kind"] == "namespace":
            module, spec, loader = record["module"], self.specs[name], self.loaders[name]
            locations = tuple(
                str(Path(self.roots[r["root"]]) / r["path"]) for r in self.namespaces[name]["locations"]
            )
            _require(
                module.__spec__ is spec
                and module.__loader__ is loader
                and spec.loader is loader
                and module.__name__ == name
                and module.__package__ == name
                and spec.name == name
                and spec.origin is None
                and tuple(module.__path__) == locations
                and tuple(spec.submodule_search_locations) == locations
                and loader.locations == locations
                and vars(module).get("__file__") is None,
                "namespace metadata drift",
            )
            return
        if record["kind"] != "file_module":
            return
        module, spec = record["module"], self.specs[name]
        loader = self.loaders[name]
        loader._check_binding()
        self.check_module_name(name)
        expected_package = name if self.rules[name]["is_package"] else name.rpartition(".")[0]
        _require(
            module.__spec__ is spec
            and module.__loader__ is loader
            and spec.loader is loader
            and spec.name == name
            and spec.parent == expected_package
            and module.__package__ == expected_package
            and module.__file__ == loader.origin
            and spec.origin == loader.origin,
            "external module metadata drift",
        )
        if self.rules[name]["is_package"]:
            _require(
                list(module.__path__) == [str(Path(loader.origin).parent)]
                and list(spec.submodule_search_locations) == list(module.__path__),
                "external package drift",
            )
        else:
            _require(
                spec.submodule_search_locations is None and "__path__" not in vars(module),
                "external nonpackage drift",
            )

    def audit(self):
        try:
            self.check_live()
            self.session.audit_generated_stdlib()
            _require(not self.stack, "external import is still in progress")
            self.environment.verify_files(expected_sha256=self.expected_sha256)
            for name, original in self.baseline.items():
                _require(sys.modules.get(name) is original, "pre-existing module identity changed")
            for publisher in {rule["publisher"] for rule in self.publications.values()}:
                self.reconcile_publications(publisher)
            for name, module in tuple(sys.modules.items()):
                if (
                    name in self.baseline
                    or name.partition(".")[0] in _ROOTS
                    or self.session.is_interpreter_name(name)
                ):
                    continue
                _require(
                    name in self.records and self.records[name]["module"] is module,
                    "unlisted module publication",
                )
            for name, record in self.records.items():
                _require(record["state"] == "completed", "external module incomplete")
                self.check_module(name)
                self.controller.require_adopted(record["module"])
            _require(not self.disabled_tokens, "disabled rejection token remains live at audit")
            for name in self.disabled:
                _require(not self._has_module_prefix(name), "disabled adapter was preloaded/published")
            for name, record in self.disabled_records.items():
                _require(record["state"] == "fallback_confirmed", "disabled denial incomplete at boundary")
                self._disabled_resolution(name)
                _require(
                    self._disabled_fallback(name) is record["fallback"], "disabled fallback storage replaced"
                )
            if self.materialization_finished:
                _require(
                    self.session.canonical_input.manifest.case_id == self.materialization_case,
                    "completed materialization case drift",
                )
                self._check_required_disabled()
        except BaseException as error:
            self.fail(error, "audit")
            raise


class _AbsentExternalSpec:
    """Internal no-spec sentinel; singleton importlib resolution executes no body."""

    def __init__(self, name):
        self.name = name


class _DisabledExternalToken:
    """Identity-only token. Its fields/type cannot confer rejection authority."""

    __slots__ = ()


class _ExternalNamespaceLoader(importlib.abc.Loader):
    def __init__(self, admission, name, locations):
        self.admission, self.name, self.locations = admission, name, tuple(locations)

    def create_module(self, spec):
        try:
            self.admission.check_live()
            _require(spec is self.admission.specs[self.name], "namespace spec replaced")
            _require(self.name not in self.admission.records, "namespace recreation")
            module = ModuleType(self.name)
            self.admission.records[self.name] = {"kind": "namespace", "state": "created", "module": module}
            self.admission.hold(module)
            return module
        except BaseException as error:
            self.admission.fail(error, self.name)
            raise

    def exec_module(self, module):
        try:
            self.admission.check_live()
            record = self.admission.records[self.name]
            _require(
                record["module"] is module
                and record["state"] == "created"
                and tuple(module.__path__) == self.locations,
                "namespace binding changed",
            )
            record["state"] = "completed"
            self.admission.check_module(self.name)
            self.admission.record("namespace", self.name, "namespace")
        except BaseException as error:
            self.admission.fail(error, self.name)
            raise


class _RuntimeSetupBinding:
    """Actual retained runtime objects; only the source session registers these.

    The exact registry, not construction or a parsed digest, is upload authority.
    It is one-shot and is not execution/final-graph readiness.
    """

    __slots__ = (
        "session",
        "controller",
        "module",
        "runtime",
        "target",
        "stream",
        "stream_identity",
        "source",
    )

    def __init__(self):
        raise TypeError("runtime bindings are issued only after guarded runtime initialization")

    def __setattr__(self, name, value):
        raise TypeError("runtime binding is frozen")

    def __delattr__(self, name):
        raise TypeError("runtime binding is frozen")


def _stream_identity(lane, stream, target):
    # Fixed metadata fields only; no ambient-device getter or CUDA query.
    if lane == "cuda":
        _require(
            target.type == "cuda"
            and type(target.index) is int
            and stream.device.type == "cuda"
            and type(stream.device.index) is int
            and stream.device.index == target.index,
            "CUDA stream target drift",
        )
        return (stream.device.index, int(stream.cuda_stream))
    _require(stream.device is target and target.is_cuda, "Warp stream target drift")
    return (int(target.ordinal), int(stream.cuda_stream))


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
        self.external = None
        self.runtime_owners = []
        self.render_source, self.canonical_input, self.prepared_input = None, None, None
        self.validation_report = None
        self.loaders, self.modules, self.trace = {}, {}, []
        self.finder = _CapturedSourceFinder(self)

    def admitted_sources(self):
        if self.phase == "core":
            return _CORE
        if self.phase in {"prelude", "binding", "bound"}:
            return _CORE | _CUSTODY_PRELUDE
        if self.phase in {"importing", "imported"}:
            return _CORE | _CUSTODY_PRELUDE | _SETUP_DEFINITIONS
        if self.phase in {"configuring_runtime", "runtime_configured"}:
            return _CORE | _CUSTODY_PRELUDE | _SETUP_DEFINITIONS | _EXTERNAL_CONTRACT_DEFINITIONS
        if self.phase in _RUNTIME_PHASES:
            admitted = (
                _CORE
                | _CUSTODY_PRELUDE
                | _SETUP_DEFINITIONS
                | _EXTERNAL_CONTRACT_DEFINITIONS
                | _MATERIALIZATION_SOURCES
            )
            if self.phase in {"validating_input", "input_validated"}:
                admitted = admitted | _VALIDATION_SOURCES
            return admitted
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

    def configure_external_environment(self, environment_payload, expected_sha256):
        """Internal trusted-launch configuration, not a worker IPC operation."""
        try:
            self.check_live()
            _require(
                self.phase == "imported" and self not in _EXTERNAL_CUSTODY,
                "external configuration is ordered and single-use",
            )
            _EXTERNAL_CUSTODY[self] = None
            controller = self.bound_controller()
            with controller.segment("setup"):
                self.phase = "configuring_runtime"
                from optical_worker.runtime_environment import RuntimeEnvironmentContract

                environment = RuntimeEnvironmentContract.from_payload(environment_payload)
                _require(
                    not any(
                        Path(row["path"]) == Path(path)
                        for row in environment.to_payload()["roots"]
                        if row["kind"] == "site"
                        for path in self.check_import_topology()[2]
                    ),
                    "external site root is on the interpreter search path",
                )
                admission = object.__new__(_ExternalAdmission)
                self.external = admission  # Retain partial construction before initialization.
                _EXTERNAL_CUSTODY[self] = (admission, environment, expected_sha256, controller)
                admission.__init__(self, environment, expected_sha256)
                os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
                self.audit()
                admission.audit()
            self.phase = "runtime_configured"
        except BaseException as error:
            self.fail(error)
            raise

    def materialize_native_source(self, case_id, *, go2_model_dir=None):
        """Fixed canonical CPU path; no supplied arrays, callbacks or readiness."""
        try:
            self.check_live()
            _require(self.phase == "runtime_configured", "canonical materialization is single-use")
            _require(
                case_id
                in {
                    "asymmetric_synthetic_7x3_v1",
                    "synthetic_dynamic_rigid_960x640_v1",
                    "go2_static_960x640_v1",
                    "go2_static_1920x1080_v1",
                },
                "case is outside C2d3 materialization",
            )
            controller = self.bound_controller()
            with controller.segment("setup"):
                self.phase = "materializing"
                self.external.bind_materialization_case(case_id)
                self._runtime_leaf("canonical_dependency_import", "runtime_import_or_init")
                from optical_worker.assets import preflight_case_asset
                from optical_worker.lane_contract import (
                    CANONICAL_NATIVE_REPLAY_INPUT,
                    FIXED_NATIVE_SCENE_INPUT,
                )
                from optical_worker.native_input import CanonicalNativeInput
                from optical_worker.render_parameters import resolve_render_source
                from optical_worker.workloads import (
                    build_canonical_fixed_fixture,
                    build_canonical_transform_replay,
                )

                is_go2 = case_id in {"go2_static_960x640_v1", "go2_static_1920x1080_v1"}
                _require(is_go2 == (go2_model_dir is not None), "Go2 path belongs only to frozen Go2 cases")
                model_dir = Path(go2_model_dir).resolve(strict=True) if is_go2 else None
                asset = (
                    preflight_case_asset(case_id, repo_root=model_dir.parent, model_dir=model_dir)
                    if is_go2
                    else None
                )
                if asset is not None:
                    self._hold_runtime_owner(asset, "canonical_asset")
                render = resolve_render_source(case_id, go2_model_dir=model_dir, asset_evidence=asset)
                self.render_source = render
                self._hold_runtime_owner(render, "canonical_render_source")
                packed = render.packed_scene
                if case_id == "synthetic_dynamic_rigid_960x640_v1":
                    replay = build_canonical_transform_replay(case_id)
                    self._hold_runtime_owner(replay, "canonical_replay")
                    source = CanonicalNativeInput.from_scene_contract(
                        packed, input_contract=CANONICAL_NATIVE_REPLAY_INPUT, replay=replay
                    )
                else:
                    fixture = build_canonical_fixed_fixture(packed)
                    self._hold_runtime_owner(fixture, "canonical_fixed_fixture")
                    source = CanonicalNativeInput.from_scene_contract(
                        packed, input_contract=FIXED_NATIVE_SCENE_INPUT, fixed_fixture=fixture
                    )
                self.canonical_input = source
                self._hold_runtime_owner(source, "canonical_native_input")
                self.external.finish_materialization()
                self.audit()
                self.external.audit()
            self.phase = "materialized"
            return source.manifest
        except BaseException as error:
            self.fail(error)
            raise

    def _hold_runtime_owner(self, owner, role, *, device_ordinal=None):
        # Retain in the already adopted session BEFORE a registration that may
        # raise. Failure custody therefore includes the last returned object.
        self.runtime_owners.append(owner)
        controller = self.bound_controller()
        if device_ordinal is None:
            if controller.guardian.contains_exact_owner(owner):
                # Imported native roots and upload-core results already have
                # their original custody record. Preserve that scope and role.
                controller.require_adopted(owner)
            else:
                controller.adopt_host_owner(owner, role=role, operation_id=role)
        else:
            controller.adopt_returned_device_owner(
                owner, actual_device_ordinal=device_ordinal, role=role, operation_id=role
            )

    def _runtime_leaf(self, operation, kind):
        from optical_worker.lifecycle import DeviceScopeDescriptor

        controller = self.bound_controller()
        descriptor = DeviceScopeDescriptor.exact_target(
            operation_id=operation,
            leaf_kind=kind,
            device_ordinal=controller.target_device_ordinal,
            scope_source="captured_runtime_admission",
            runtime_identity="frozen_runtime_environment_v2",
        )
        controller.enter_device_leaf(controller.begin_device_leaf(descriptor))

    def initialize_native_runtime(self):
        """Internal sacrificial-driver entry; not connected to product IPC."""
        try:
            self.check_live()
            _require(
                self.phase == "materialized" and self not in _RUNTIME_BINDINGS,
                "runtime initialization is ordered and single-use",
            )
            _RUNTIME_BINDINGS[self] = None
            controller = self.bound_controller()
            ordinal = controller.target_device_ordinal
            with controller.segment("setup"):
                self.phase = "initializing_runtime"
                self.external.audit()
                self.external.phase = "runtime"
                _require(os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0", "backend plugins enabled")
                try:
                    self._runtime_leaf("indexed_runtime_import", "runtime_import_or_init")
                    if self.lane == "cuda":
                        import torch as module

                        runtime = module._C
                        self._hold_runtime_owner(runtime, "torch_runtime_root")
                        target = module.device("cuda", ordinal)
                        self._hold_runtime_owner(target, "torch_target_descriptor")
                        self._runtime_leaf("indexed_runtime_stream", "allocation")
                        stream = module.cuda.Stream(device=target)
                        # Store before reading returned metadata as that read
                        # or its ordinal validation may itself fail.
                        self.runtime_owners.append(stream)
                        _require(type(stream.device.index) is int, "CUDA stream has no indexed device")
                        self._hold_runtime_owner(
                            stream, "torch_runtime_stream", device_ordinal=stream.device.index
                        )
                    else:
                        import warp as module

                        module.config.quiet = True
                        _require(module.config.quiet is True, "Warp quiet initialization required")
                        self._runtime_leaf("indexed_warp_init", "runtime_import_or_init")
                        module.init()
                        runtime = self.external.records["warp._src.context"]["module"].runtime
                        self._hold_runtime_owner(runtime, "warp_runtime_root")
                        target = module.get_device(f"cuda:{ordinal}")
                        self.runtime_owners.append(target)
                        _require(target.is_cuda, "Warp returned a non-CUDA target")
                        self._hold_runtime_owner(
                            target, "warp_runtime_target", device_ordinal=int(target.ordinal)
                        )
                        self._runtime_leaf("indexed_runtime_stream", "runtime_import_or_init")
                        stream = module.get_stream(target)
                        self.runtime_owners.append(stream)
                        self._hold_runtime_owner(
                            stream, "warp_runtime_stream", device_ordinal=int(stream.device.ordinal)
                        )
                except BaseException:
                    if not controller._terminating:
                        controller.mark_unknown_device_scope(
                            source="partial_runtime_initialization", reason="runtime_init_not_completed"
                        )
                    raise
                identity = _stream_identity(self.lane, stream, target)
                _require(identity[0] == ordinal, "runtime target differs from the guarded ordinal")
                self.external.audit()
                self.audit()
                binding = object.__new__(_RuntimeSetupBinding)
                for key, value in (
                    ("session", self),
                    ("controller", controller),
                    ("module", module),
                    ("runtime", runtime),
                    ("target", target),
                    ("stream", stream),
                    ("stream_identity", identity),
                    ("source", self.canonical_input),
                ):
                    object.__setattr__(binding, key, value)
                _RUNTIME_BINDINGS[self] = (
                    binding,
                    controller,
                    self.external,
                    module,
                    runtime,
                    target,
                    stream,
                    identity,
                    self.canonical_input,
                )
                self._hold_runtime_owner(binding, "runtime_setup_binding")
            self.phase = "runtime_ready"
            return binding
        except BaseException as error:
            self.fail(error)
            raise

    def _consume_runtime_binding(self, binding):
        self.check_live()
        controller = self.bound_controller()
        controller._require_setup()
        record = _RUNTIME_BINDINGS.get(self)
        # Burn even on identity/metadata rejection. The surrounding armed
        # segment handles poison and never falls back to C0 synchronization.
        _RUNTIME_BINDINGS[self] = None
        return self._validate_runtime_binding_record(binding, record)

    def _validate_runtime_binding_record(self, binding, record):
        self.check_live()
        controller = self.bound_controller()
        controller._require_setup()
        _require(
            type(binding) is _RuntimeSetupBinding and record is not None and record[0] is binding,
            "runtime binding is foreign, copied, unregistered or consumed",
        )
        _, original_controller, admission, module, runtime, target, stream, identity, source = record
        _require(
            binding.session is self
            and binding.controller is controller
            and controller is original_controller
            and self.external is admission
            and binding.module is module
            and binding.runtime is runtime
            and binding.target is target
            and binding.stream is stream
            and binding.stream_identity == identity
            and binding.source is source
            and self.canonical_input is source,
            "runtime binding resource identity drift",
        )
        expected_name = "torch" if self.lane == "cuda" else "warp"
        _require(
            admission.records[expected_name]["module"] is module and sys.modules.get(expected_name) is module,
            "runtime module replacement",
        )
        _require(
            _stream_identity(self.lane, stream, target) == identity
            and identity[0] == controller.target_device_ordinal,
            "runtime stream storage drift",
        )
        if self.lane == "cuda":
            _require(
                module._C is runtime and os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0",
                "Torch runtime/configuration drift",
            )
        else:
            _require(
                target.runtime is runtime
                and module.config.quiet is True
                and admission.records["warp._src.context"]["module"].runtime is runtime,
                "Warp runtime/configuration drift",
            )
        controller.require_adopted(binding, module, runtime, target, stream, self.canonical_input)
        return module, target, stream

    def upload_native_input(self, binding):
        try:
            self.check_live()
            _require(self.phase == "runtime_ready", "guarded upload is ordered and single-use")
            controller = self.bound_controller()
            with controller.segment("setup"):
                self.phase = "uploading"
                runtime_record = _RUNTIME_BINDINGS.get(self)
                module, target, stream = self._consume_runtime_binding(binding)
                self.audit()
                self.external.audit()
                from optical_worker.native_contract import assert_prepared_storage_registered
                from optical_worker.native_setup import (
                    _create_upload_context,
                    _upload_cuda_native_input,
                    _upload_warp_native_input,
                )

                self._runtime_leaf("indexed_upload_scope", "runtime_import_or_init")
                context = _create_upload_context(controller, module, target, stream)
                if self.lane == "cuda":
                    prepared = _upload_cuda_native_input(self.canonical_input, context=context)
                else:
                    prepared = _upload_warp_native_input(self.canonical_input, context=context)
                self.prepared_input = prepared
                self._hold_runtime_owner(prepared, "guarded_prepared_input")
                assert_prepared_storage_registered(prepared)
                _UPLOADED_INPUTS[self] = (runtime_record, prepared)
                self.audit()
                self.external.audit()
            self.phase = "uploaded"
            return prepared
        except BaseException as error:
            self.fail(error)
            raise

    def validate_native_input(self):
        """One supervised explicit D2H validation, never final readiness."""
        try:
            self.check_live()
            _require(self.phase == "uploaded", "input readback is ordered and single-use")
            controller = self.bound_controller()
            with controller.segment("setup"):
                self.phase = "validating_input"
                record = _UPLOADED_INPUTS.get(self)
                _require(
                    record is not None and record[1] is self.prepared_input,
                    "input readback requires this session's exact uploaded object",
                )
                runtime_record, prepared = record
                module, target, stream = self._validate_runtime_binding_record(
                    runtime_record[0], runtime_record
                )
                controller.require_adopted(prepared)
                self.external.audit()
                from optical_worker.native_readback import _readback_prepared_native_input

                report = _readback_prepared_native_input(
                    self.canonical_input,
                    prepared,
                    lifecycle=controller,
                    runtime=module,
                    target=target,
                    stream=stream,
                )
                self.validation_report = report
                self.audit()
                self.external.audit()
            self.phase = "input_validated"
            return report
        except BaseException as error:
            self.fail(error)
            raise

    def check_import_topology(self):
        """Pure CPU integrity, also usable by terminal stdlib after poison."""
        topology = _IMPORT_TOPOLOGIES.get(self)
        _require(topology is not None, "source import topology was not installed")
        chain, finder, paths, hooks, delegates, names, generated = topology
        _require("_PYTHON_SYSCONFIGDATA_NAME" not in os.environ, "sysconfig name override forbidden")
        _require(generated[0] == _generated_interpreter_identity(), "generated interpreter identity drift")
        _require(
            sys.meta_path is chain
            and len(chain) == 1
            and chain[0] is finder
            and finder is self.finder
            and tuple(sys.path) == paths
            and len(sys.path_hooks) == len(hooks)
            and all(actual is expected for actual, expected in zip(sys.path_hooks, hooks))
            and frozenset(sys.stdlib_module_names) | frozenset(sys.builtin_module_names) == names,
            "closed import topology changed",
        )
        _require(
            all(
                owner.find_spec.__self__ is owner and owner.find_spec.__func__ is method.__func__
                for owner, method in delegates
            ),
            "standard resolver binding changed",
        )
        return topology

    def is_interpreter_name(self, name):
        topology = _IMPORT_TOPOLOGIES[self]
        return _is_interpreter_name(name, topology[5], topology[6])

    def _check_generated_source(self):
        descriptor = self.check_import_topology()[6]
        if descriptor[1] is None:
            return descriptor
        path = Path(descriptor[2])
        _require(
            path.is_file() and not path.is_symlink() and path.resolve(strict=True) == path,
            "generated stdlib source missing or redirected",
        )
        data = path.read_bytes()
        _require(
            len(data) == descriptor[3] and hashlib.sha256(data).hexdigest() == descriptor[4],
            "generated stdlib source snapshot drift",
        )
        return descriptor

    def generated_stdlib_spec(self, fullname, path, target):
        descriptor = self._check_generated_source()
        _require(
            fullname == descriptor[1] and path is None and target is None and fullname not in sys.modules,
            "generated stdlib reload or foreign request",
        )
        delegate = self.check_import_topology()[4][2][1]
        spec = delegate(fullname, [str(Path(descriptor[2]).parent)], None)
        self.check_import_topology()
        _require(
            spec is not None
            and type(spec.loader) is importlib.machinery.SourceFileLoader
            and spec.name == fullname
            and spec.origin == descriptor[2]
            and spec.loader.name == fullname
            and spec.loader.path == descriptor[2]
            and spec.submodule_search_locations is None,
            "generated stdlib spec or loader drift",
        )
        self._check_generated_source()
        _GENERATED_STDLIB_SPECS[self] = (spec, spec.loader)
        return spec

    def audit_generated_stdlib(self):
        try:
            descriptor = self._check_generated_source()
            name = descriptor[1]
            if name is None:
                return
            previous = _GENERATED_STDLIB_MODULES.get(self)
            if name not in sys.modules:
                _require(previous is None, "generated stdlib module removed")
                return
            module = sys.modules[name]
            resolved = _GENERATED_STDLIB_SPECS.get(self)
            _require(
                resolved is not None and type(module) is ModuleType,
                "generated stdlib module not resolved by this session",
            )
            spec, loader = resolved
            _require(
                type(loader) is importlib.machinery.SourceFileLoader
                and module.__name__ == name
                and module.__package__ == ""
                and module.__spec__ is spec
                and module.__loader__ is loader
                and module.__file__ == descriptor[2]
                and spec.name == name
                and spec.origin == descriptor[2]
                and spec.loader is loader
                and spec.submodule_search_locations is None
                and loader.name == name
                and loader.path == descriptor[2]
                and not getattr(spec, "_initializing", False),
                "generated stdlib loaded metadata drift",
            )
            _require(previous is None or module is previous, "generated stdlib object replaced")
            if previous is None:
                _GENERATED_STDLIB_MODULES[self] = module
        except BaseException as error:
            self.fail(error)
            raise

    def check_live(self):
        self.check_import_topology()
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
        external = _EXTERNAL_CUSTODY.get(self)
        if external is not None:
            _require(self.external is external[0], "external source session identity drift")

    def audit(self):
        try:
            self.check_live()
            self.audit_generated_stdlib()
            expected = self.admitted_sources()
            if self.phase in _RUNTIME_PHASES:
                required = _CORE | _CUSTODY_PRELUDE | _SETUP_DEFINITIONS | _EXTERNAL_CONTRACT_DEFINITIONS
                _require(set(required) <= set(self.modules) <= set(expected), "executed source closure drift")
            else:
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
                expected_path = (
                    [str(Path(loader.origin).parent)] if loader.relative.endswith("/__init__.py") else None
                )
                _require(module.__spec__.submodule_search_locations == expected_path, "package spec drift")
                if expected_path is not None:
                    _require(module.__path__ == expected_path, "package search path drift")
            _require(
                len(self.trace) == 2 * len(self.modules)
                and all(
                    self.trace.count((operation, name)) == 1
                    for name in self.modules
                    for operation in ("started", "completed")
                ),
                "execution trace drift",
            )
            if self.snapshot is not None:
                observed = self.snapshot.check_loaded_modules()
                executed = {expected[name] for name in self.modules}
                _require(set(observed) == executed | {_BOOTSTRAP}, "source ledger drift")
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
    # Check before pathlib/subprocess or any other helper that can lazily
    # import stdlib. Even a subsequently caught import could invoke a rogue
    # startup finder before a later topology rejection.
    finders = (
        importlib.machinery.BuiltinImporter,
        importlib.machinery.FrozenImporter,
        importlib.machinery.PathFinder,
    )
    _require(
        len(sys.meta_path) == len(finders)
        and all(actual is expected for actual, expected in zip(sys.meta_path, finders)),
        "unexpected cold interpreter finder chain",
    )
    _require(
        sys.flags.isolated
        and sys.flags.no_site
        and sys.flags.dont_write_bytecode
        and sys.implementation.name == "cpython",
        "isolated CPython -I -S -B required",
    )
    _require("_PYTHON_SYSCONFIGDATA_NAME" not in os.environ, "sysconfig name override forbidden")
    names = frozenset(sys.stdlib_module_names) | frozenset(sys.builtin_module_names)
    _require(not any(name.partition(".")[0] in _ROOTS for name in sys.modules), "repository module preloaded")
    _require(
        all(name == "__main__" or _is_interpreter_name(name, names) for name in sys.modules),
        "non-standard-library module preloaded before source custody",
    )
    _require(
        not any(Path(path).resolve().is_relative_to(root) for path in sys.path), "repository on search path"
    )
    _require(
        all(
            type(path) is str
            and path
            and Path(path).is_absolute()
            and not {part.lower() for part in Path(path).parts}.intersection(
                {"site-packages", "dist-packages"}
            )
            for path in sys.path
        ),
        "unexpected cold interpreter search path",
    )
    paths, hooks = tuple(sys.path), tuple(sys.path_hooks)
    delegates = tuple((finder, finder.find_spec) for finder in finders)
    generated = _capture_generated_stdlib(root, paths)
    sources = _capture(root, revision, lane, catalog_sha256)
    _require(
        len(sys.meta_path) == len(finders)
        and all(actual is expected for actual, expected in zip(sys.meta_path, finders))
        and tuple(sys.path) == paths
        and tuple(sys.path_hooks) == hooks,
        "cold interpreter topology changed during capture",
    )
    session = _SourceSession(root, revision, lane, catalog_sha256, sources)
    _ACTIVE_SESSION = session
    chain = [session.finder]
    _IMPORT_TOPOLOGIES[session] = (chain, session.finder, paths, hooks, delegates, names, generated)
    sys.meta_path = chain
    session.check_import_topology()
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
