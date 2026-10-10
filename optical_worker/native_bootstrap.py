"""Direct-entry CPU source verifier. No worker IPC or GPU admission.

The configured release and this directly executed script are trusted bootstrap
inputs. This is not a sandbox for arbitrary code in the same process.
"""

from __future__ import annotations

import _thread
import ast
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import linecache
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from types import FunctionType, MappingProxyType, ModuleType

_BOOTSTRAP = "optical_worker/native_bootstrap.py"
_NATIVE_EXEC_OPERATION = "exec"  # Descriptive operation enum, never a dynamic Python callable.
_CATALOG = "optical_worker/data/native_sources/v17.json"
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
        "optical_worker.lane_contract": "optical_worker/lane_contract.py",
        "optical_worker.native_manifest": "optical_worker/native_manifest.py",
        "optical_worker.runtime_environment": "optical_worker/runtime_environment.py",
        "optical_worker.runtime_custody": "optical_worker/runtime_custody.py",
        "optical_worker.runtime_phases": "optical_worker/runtime_phases.py",
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
        "optics.query_numerics": "optics/query_numerics.py",
        "optics.scene": "optics/scene.py",
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
        "preparing_cuda_factories",
        "cuda_factories_ready",
        "preparing_warp_modules",
        "warp_modules_ready",
        "preparing_fixed_render_storage",
        "fixed_render_storage_ready",
        "preparing_fixed_render_graph",
        "fixed_render_graph_ready",
        "warming_fixed_render_graph",
        "fixed_render_warmups_ready",
    }
)
_VALIDATION_SOURCES = MappingProxyType(
    {"optical_worker.native_readback": "optical_worker/native_readback.py"}
)
_CUDA_ARTIFACT_SOURCES = MappingProxyType(
    {
        "optics.cuda_extension_loading": "optics/cuda_extension_loading.py",
        "optics.cuda_extension_recipes": "optics/cuda_extension_recipes.py",
        "optics._cuda_scene_source": "optics/_cuda_scene_source.py",
        "optics._cuda_lbvh_source": "optics/_cuda_lbvh_source.py",
        "optics._cuda_direct_light_source": "optics/_cuda_direct_light_source.py",
    }
)
_CUDA_FACTORY_SOURCES = MappingProxyType(
    {
        "optics.cuda_scene": "optics/cuda_scene.py",
        "optics.cuda_lbvh": "optics/cuda_lbvh.py",
        "optics.cuda_direct_light": "optics/cuda_direct_light.py",
        "optics.cuda_timeline": "optics/cuda_timeline.py",
        "optics.native_arena": "optics/native_arena.py",
        "optics.device_channel": "optics/device_channel.py",
        "optics.execution": "optics/execution.py",
        "optics.output_contract": "optics/output_contract.py",
        "optics.scene": "optics/scene.py",
    }
)
_WARP_SETUP_SOURCES = MappingProxyType(
    {"optical_worker.warp_setup": "optical_worker/warp_setup.py"}
)
_WARP_MODULE_SOURCES = MappingProxyType(
    {
        "optics.device_scene": "optics/device_scene.py",
        "optics.device_bvh": "optics/device_bvh.py",
        "optics.warp_native_render": "optics/warp_native_render.py",
        "optics.output_contract": "optics/output_contract.py",
    }
)
_WARP_RENDER_SOURCES = MappingProxyType(
    {"optics.execution": "optics/execution.py", "optics.scene": "optics/scene.py"}
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
_INTERPRETER_ALIAS_CUSTODY = {}


class SourceBootstrapError(RuntimeError):
    """Terminal CPU bootstrap failure; never a GPU quiescence claim."""


def _require(condition, message):
    if not condition:
        raise SourceBootstrapError(message)


def _raw_module(module):
    return ModuleType.__getattribute__(module, "__dict__")


def _raw_class(cls):
    return type.__getattribute__(cls, "__dict__")


def _same_facts(projection, facts):
    return (type(projection) is dict and all(type(key) is str for key in projection)
            and set(projection) == {key for key, _ in facts}
            and all(type(projection[key]) is type(value) and projection[key] == value
                    for key, value in facts))


def _capture_interpreter_alias(root, paths):
    main = sys.modules.get("__main__")
    _require(type(main) is ModuleType and "__mp_main__" not in sys.modules
             and "multiprocessing" not in sys.modules, "interpreter alias/main preloaded or missing")
    directory = getattr(sys, "_stdlib_dir", None)
    _require(type(directory) is str and bool(directory), "interpreter stdlib directory missing")
    base = Path(directory).resolve(strict=True)
    _require(str(base) in paths and not base.is_relative_to(root)
             and not {p.lower() for p in base.parts} & {"site-packages", "dist-packages"},
             "interpreter alias stdlib root drift")
    source_path = base / "multiprocessing" / "__init__.py"
    _require(source_path.resolve(strict=True) == source_path and not source_path.is_symlink(),
             "interpreter alias source redirected")
    source = source_path.read_bytes()
    expected = ast.parse("if '__main__' in sys.modules:\n sys.modules['__mp_main__'] = sys.modules['__main__']\n").body[0]
    matches = [node for node in ast.parse(source).body
               if ast.dump(node, include_attributes=False) == ast.dump(expected, include_attributes=False)]
    _require(len(matches) == 1, "interpreter alias statement differs from fixed source relation")
    return (_generated_interpreter_identity(), str(source_path), source,
            hashlib.sha256(source).hexdigest(), main)


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
        self.source_cache = None

    def _check(self, fullname):
        self.session.check_live()
        _require(fullname == self.name and self.session.loaders.get(fullname) is self, "foreign loader/name")
        _require(
            self.source is self.session.sources[self.relative][0]
            and self.relative == self.session.admitted_sources()[self.name]
            and self.origin == str(self.session.root / self.relative),
            "loader binding changed",
        )
        if self.source_cache is not None:
            entry = self.source_cache
            _require(
                linecache.cache.get(self.origin) is entry
                and type(entry) is tuple and len(entry) == 4
                and entry[0] == len(self.source) and entry[1] is None
                and type(entry[2]) is list and all(type(line) is str for line in entry[2])
                and "".join(entry[2]).encode("utf-8") == self.source
                and entry[3] == self.origin,
                "captured source linecache changed",
            )

    def _publish_source_cache(self):
        # Warp decorators call inspect while this module body executes. A held
        # get_source alone does not override inspect's readable on-disk origin.
        _require(self.source_cache is None, "captured source cache is single-use")
        lines = self.source.decode("utf-8").splitlines(keepends=True)
        self.source_cache = (len(self.source), None, lines, self.origin)
        linecache.cache[self.origin] = self.source_cache

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
            self._publish_source_cache()
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
        custody = _EXTERNAL_CUSTODY.get(self.session)
        admission = None if custody is None else custody[0]
        marker = None
        try:
            if admission is not None and "native_context" in vars(admission):
                marker = admission.enter_import_wrapper("request_entry", fullname)
            else:
                self.session.observe_interpreter_alias("request_entry")
            return self._find_spec(fullname, path, target)
        finally:
            if marker is not None:
                admission.leave_import_wrapper(marker)

    def _find_spec(self, fullname, path=None, target=None):
        try:
            topology = self.session.check_import_topology()
            if self.session.is_interpreter_name(fullname):
                if fullname == topology[6][1]:
                    return self.session.generated_stdlib_spec(fullname, path, target)
                # Terminal reporting may need unloaded stdlib modules after
                # poison. Topology integrity does not reopen session liveness.
                for owner, method in topology[4]:
                    search = path
                    if owner is importlib.machinery.PathFinder and path is None:
                        search = topology[2]
                    spec = method(fullname, search, target)
                    self.session.check_import_topology()
                    if spec is not None:
                        if fullname == "multiprocessing":
                            self.session.bind_interpreter_alias_resolution(spec, path, target)
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
        self.compiled_code = None

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
        _require(self.compiled_code is None, "external source compilation repeated")
        self.compiled_code = self.source_to_code(self.source, self.origin)
        return self.compiled_code


    def create_module(self, spec):
        marker, frame = None, None
        try:
            marker = self.admission.enter_import_wrapper("delegate_entry")
            self._check()
            _require(spec is self.admission.specs[self.name], "external spec identity changed")
            _require(self.name not in self.admission.records, "external module recreation")
            self.admission.check_self_alias_absence(self.name, "precreate")
            self.admission.record("create_started", self.name, "file_module")
            self.admission.stack.append(self.name)
            frame = self.admission.begin_native_call(self, "create", None, marker)
            if self.admission.rules[self.name]["root"] == "cuda_artifacts":
                module = self.admission.session._create_cuda_artifact(self, spec, frame)
            elif type(self.delegate) is importlib.machinery.ExtensionFileLoader:
                module = self.delegate.create_module(spec)
                _require(module is not None, "native loader returned no module")
                self._retain_created_module(module, frame)
            else:
                module = ModuleType(self.name)
                self._retain_created_module(module, frame)
            self.admission.return_native_call(frame, module)
            # Native return is already retained. Importlib has not initialized
            # spec/loader/package yet, so inspect only the raw intrinsic name.
            self.admission.check_module_name(self.name)
            self.admission.check_self_alias_absence(self.name, "postcreate")
            self.admission.hold(module)
            self.admission.record("created", self.name, "file_module")
            self.admission.complete_native_call(frame)
            self.admission.stack.pop()
            return module
        except BaseException as error:
            self.admission.fail_native_call(frame)
            self.admission.fail(error, self.name, native_create=self.source is None)
            raise
        finally:
            if marker is not None:
                self.admission.leave_import_wrapper(marker)

    def _retain_created_module(self, module, frame):
        # Called inside the indexed contexts for CUDA artifacts. A context exit,
        # journal append or later adoption may fail without losing this owner.
        self.admission.records[self.name] = {"module": module, "state": "created", "kind": "file_module"}
        _require(module is not None, "native loader returned no module")
        observation = self.admission.session.runtime_observation
        observation.append("returned_owner", {"producer": "file_create", "call_sequence": frame["sequence"],
                           "leaf_event_index": None, "object_token": observation.token(module)})

    def exec_module(self, module):
        marker, frame = None, None
        try:
            marker = self.admission.enter_import_wrapper("delegate_entry")
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
            frame = self.admission.begin_native_call(self, _NATIVE_EXEC_OPERATION, module, marker)
            if self.admission.rules[self.name]["root"] == "cuda_artifacts":
                self.admission.session._exec_cuda_artifact(self, module)
            elif self.source is None:
                self.delegate.exec_module(module)
            else:
                super().exec_module(module)
            self.admission.return_native_call(frame, module)
            self.admission.check_live()
            self.admission.confirm_disabled_fallbacks(self.name)
            self.admission.reconcile_publications(self.name, admitting=True)
            self.admission.check_module(self.name)
            self.admission.complete_native_call(frame)
            self.admission.stack.pop()
            record["state"] = "completed"
            self.admission.record("completed", self.name, "file_module")
        except BaseException as error:
            self.admission.fail_native_call(frame)
            self.admission.fail(error, self.name, native_create=self.source is None)
            raise
        finally:
            if marker is not None:
                self.admission.leave_import_wrapper(marker)


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
        self.search_roots = environment.external_search_roots()
        self.files = {(r["root"], r["path"]): r for r in payload["files"]}
        self.rules = {r["name"]: r for r in payload["modules"]}
        self.publications = {r["name"]: r for r in payload["publications"]}
        self.native_globals = {r["name"]: r for r in payload["native_globals"]}
        self.replacements = {r["name"]: r for r in payload["module_replacements"]}
        self.replacement_rules = tuple((r["name"], r["phase"], r["class_attribute"], r["relation"],
                                        r["native_root"], tuple(r["native_attributes"]))
                                       for r in payload["module_replacements"])
        self.native_global_rules = tuple(tuple(r[k] for k in ("name", "observer", "operation", "phase"))
                                         for r in payload["native_globals"])
        self.namespaces = {r["name"]: r for r in payload["namespaces"]}
        self.absent = {(r["name"], r["requester"], r["phase"]) for r in payload["absent_probes"]}
        self.disabled = {r["name"]: r for r in payload["disabled_optional_modules"]}
        self.disabled_records, self.disabled_tokens, self.disabled_bindings = {}, {}, {}
        self.materialization_case, self.materialization_finished = None, False
        self.phase = "materialization"
        self.stack, self.events, self.retained, self.failures = [], [], [], []
        self.failure_contexts, self.self_alias_checks = [], {}
        self.loaders, self.loader_bindings, self.specs, self.records = {}, {}, {}, {}
        self.native_thread = _thread.get_ident()
        self.native_context, self.native_calls, self.native_call_bindings = [], [], {}
        self.native_global_records, self.native_global_bindings = {}, {}
        self.native_global_events, self.native_global_event_bindings = [], []
        self.native_checkpoint_active = False
        self.replacement_records, self.replacement_bindings = {}, {}
        self.replacement_events, self.replacement_event_bindings = [], []
        self.replacement_objects = []
        self.replacement_checkpoint_active = False
        self.baseline = dict(sys.modules)
        self.failed = False
        _require(
            not any(name in self.baseline for name in self.rules | self.publications
                    | self.namespaces | self.native_globals)
            and not any(self._has_module_prefix(name) for name in self.disabled),
            "external dependency was preloaded",
        )
        controller.adopt_host_owner(
            environment, role="runtime_environment", operation_id="runtime_environment"
        )
        controller.adopt_host_owner(
            self, role="runtime_import_custody", operation_id="runtime_import_custody"
        )

    def _retain_replacements(self):
        values = {name: sys.modules[name] for name in self.replacements if name in sys.modules}
        for value in values.values():
            self._retain_replacement_owner(value)
        return values

    def _retain_replacement_owner(self, value):
        if not any(owner is value for owner in self.retained):
            self.retained.append(value)

    def _replacement_token(self, value):
        for index, original in enumerate(self.replacement_objects):
            if original is value:
                return index + 1
        self.replacement_objects.append(value)
        return len(self.replacement_objects)

    def _replacement_relation(self, name, original, exposed):
        # Intake has already retained the exposed object, even when invalid.
        _require(type(original) is ModuleType, "replacement original must be a file module")
        rule = self.replacements[name]
        raw = _raw_module(original)
        cls = raw.get(rule["class_attribute"])
        self._retain_replacement_owner(cls)
        _require(type(cls) is type and type(exposed) is cls and exposed is not original,
                 "replacement exact class/original mismatch")
        classes = [(cls, original)]
        if name not in {"torch._VF", "torch.backends.quantized"}:
            backend = self.records["torch.backends"]["module"]
            base = _raw_module(backend).get("PropModule")
            self._retain_replacement_owner(backend)
            self._retain_replacement_owner(base)
            _require(type(base) is type and type.__getattribute__(base, "__bases__") == (ModuleType,),
                     "replacement PropModule base drift")
            _require(type.__getattribute__(cls, "__bases__") == (base,), "replacement class base drift")
            classes.append((base, backend))
            expected_mro = (cls, base, ModuleType, object)
        else:
            _require(type.__getattribute__(cls, "__bases__") == (ModuleType,), "replacement direct base drift")
            expected_mro = (cls, ModuleType, object)
        _require(type.__getattribute__(cls, "__mro__") == expected_mro, "replacement MRO drift")
        functions = []
        for owner_class, owner_module in classes:
            class_raw = _raw_class(owner_class)
            for attribute in ("__init__", "__getattr__"):
                function = class_raw.get(attribute)
                if function is None:
                    continue
                self._retain_replacement_owner(function)
                _require(type(function) is FunctionType
                         and function.__globals__ is _raw_module(owner_module),
                         "replacement method defining globals drift")
                functions.append((owner_class, attribute, function, function.__code__, owner_module))
        exposed_raw = _raw_module(exposed)
        _require(exposed_raw.get("__name__") == name
                 and all(key in exposed_raw and exposed_raw[key] is None
                         for key in ("__loader__", "__spec__", "__package__", "__doc__"))
                 and not {"__file__", "__path__", "__cached__", "__getattr__", "__getattribute__"}
                 & exposed_raw.keys(), "replacement raw metadata drift")
        native_root, native_value = None, None
        if rule["relation"] == "original_backlink":
            value = exposed_raw.get("m")
            self._retain_replacement_owner(value)
            _require(value is original, "replacement original backlink mismatch")
        else:
            native_record = self.records.get(rule["native_root"])
            _require(native_record is not None, "replacement native root missing")
            native_root = native_record["module"]
            self._retain_replacement_owner(native_root)
            self.check_module(rule["native_root"])
            native_value = _raw_module(native_root).get(rule["native_attributes"][0])
            self._retain_replacement_owner(native_value)
            self._retain_replacement_owner(exposed_raw.get("vf"))
            _require(native_value is not None and exposed_raw.get("vf") is native_value,
                     "replacement native value mismatch")
        return cls, expected_mro, tuple(functions), native_root, native_value

    def _check_replacement(self, name, *, live=True):
        record = self.replacement_records.get(name)
        binding = self.replacement_bindings.get(name)
        _require(binding is not None and binding[0] is record, "replacement record identity drift")
        _, original_record, original, exposed, frame, relation, tokens, state = binding
        _require(self.records.get(name) is original_record and original_record["module"] is original
                 and record["original"] is original and record["module"] is exposed
                 and record["frame"] is frame and record["relation"] is relation
                 and _same_facts(record["tokens"], tokens) and record["state"] == state,
                 "replacement binding drift")
        self._check_native_call(frame)
        for key, value in (("original", original), ("replacement", exposed), ("class", relation[0]),
                           ("native_root", relation[3]), ("native_value", relation[4])):
            token = dict(tokens)[key]
            _require(token is None if value is None else
                     type(token) is int and 0 < token <= len(self.replacement_objects)
                     and self.replacement_objects[token - 1] is value, "replacement object token drift")
        if live:
            _require(sys.modules.get(name) is exposed, "replacement removed or replaced")
            current = self._replacement_relation(name, original, exposed)
            _require(current[0] is relation[0] and current[1] == relation[1]
                     and current[3] is relation[3] and current[4] is relation[4]
                     and len(current[2]) == len(relation[2])
                     and all(a[1] == b[1] and all(a[i] is b[i] for i in (0, 2, 3, 4))
                             for a, b in zip(current[2], relation[2])), "replacement class/function relation drift")
        return binding

    def _replacement_event(self, operation, name, checkpoint, request_name):
        record = self.replacement_records[name]
        event = {"index": len(self.replacement_events), "operation": operation, "name": name,
                 "phase": record["frame"]["phase"], "call_sequence": record["frame"]["sequence"],
                 "checkpoint": checkpoint, "request_name": request_name,
                 "relation": self.replacements[name]["relation"], **record["tokens"]}
        self.replacement_event_bindings.append((event, tuple(event.items()), record))
        self.replacement_events.append(event)
        observation = self.session.runtime_observation
        used = sorted({value for value in record["tokens"].values() if value is not None})
        observation.append("module_replacement", {"event_index": event["index"], "token_bindings": [
            {"local_token": token, "object_token": observation.token(self.replacement_objects[token - 1])}
            for token in used]})

    def _check_replacement_events(self):
        _require(len(self.replacement_events) == len(self.replacement_event_bindings),
                 "replacement event history drift")
        for event, binding in zip(self.replacement_events, self.replacement_event_bindings):
            _require(event is binding[0] and _same_facts(event, binding[1])
                     and self.replacement_records.get(event["name"]) is binding[2],
                     "replacement event facts drift")
            self._check_replacement(event["name"], live=not self.failed)

    def replacement_checkpoint(self, checkpoint, *, request_name=None, returned_frame=None):
        values = self._retain_replacements()
        self._check_native_thread()
        _require(not self.failed and not self.replacement_checkpoint_active,
                 "replacement checkpoint reentry/failure")
        self.replacement_checkpoint_active = True
        try:
            _require(self.replacements == {r["name"]: r for r in self.payload["module_replacements"]},
                     "replacement table drift")
            _require(tuple((r["name"], r["phase"], r["class_attribute"], r["relation"],
                            r["native_root"], tuple(r["native_attributes"]))
                           for r in self.replacements.values()) == self.replacement_rules,
                     "replacement immutable rule drift")
            frame = returned_frame if returned_frame is not None else (
                self.native_context[-1] if self.native_context else None)
            for name, rule in self.replacements.items():
                previous = self.replacement_records.get(name)
                original_record = self.records.get(name)
                if previous is not None:
                    self._check_replacement(name)
                    if previous["state"] == "observed_pending":
                        pending = previous["frame"]
                        _require(any(value is pending for value in self.native_context)
                                 or pending is returned_frame
                                 or (pending["state"] == "returned" and any(
                                     value is pending["wrapper"] for value in self.native_context)),
                                 "pending replacement frame disappeared")
                    continue
                _require(name not in self.replacement_bindings, "replacement record removed")
                if name not in values:
                    _require(original_record is None or original_record["state"] == "created",
                             "executing replacement key removed")
                    continue
                _require(original_record is not None, "replacement key preloaded")
                original = original_record["module"]
                if values[name] is original:
                    continue
                _require(checkpoint in {"request_entry", "exec_return"} and type(frame) is dict,
                         "replacement observed outside own Python exec")
                self._check_native_call(frame)
                _require(frame["loader"].source is not None and frame["loader"].name == name
                         and frame["module"] is original and frame["operation"] == "exec"
                         and frame["phase"] == rule["phase"] == self.phase
                         and frame["state"] == ("returned" if returned_frame is not None else "started"),
                         "replacement observed in wrong interval")
                relation = self._replacement_relation(name, original, values[name])
                tokens = {key: None if value is None else self._replacement_token(value)
                          for key, value in (("original", original), ("replacement", values[name]),
                                             ("class", relation[0]), ("native_root", relation[3]),
                                             ("native_value", relation[4]))}
                record = {"original": original, "module": values[name], "frame": frame,
                          "relation": relation, "tokens": tokens, "state": "observed_pending"}
                self.replacement_records[name] = record
                self.replacement_bindings[name] = (
                    record, original_record, original, values[name], frame, relation,
                    tuple(tokens.items()), "observed_pending")
                self.controller.adopt_host_owner(values[name], role="external_import_owner",
                                                 operation_id="module_replacement_observation")
                _require(not self.failed, "replacement adoption swallowed failure")
                self._replacement_event("observed_pending", name, checkpoint, request_name)
                _require(not self.failed, "replacement observation swallowed failure")
        finally:
            self.replacement_checkpoint_active = False

    def complete_replacement(self, frame):
        name = frame["loader"].name
        if frame["operation"] != "exec" or name not in self.replacements:
            return
        binding = self._check_replacement(name)
        record = self.replacement_records[name]
        _require(record["frame"] is frame and record["state"] == "observed_pending",
                 "required replacement missing or already completed")
        self._check_replacement_events()
        self._replacement_event("admitted", name, "exec_return", None)
        _require(not self.failed, "replacement admission swallowed failure")
        record["state"] = "admitted"
        self.replacement_bindings[name] = (*binding[:-1], "admitted")

    def exposed_module(self, name):
        if name in self.replacement_records:
            self._check_replacement(name)
            return self.replacement_records[name]["module"]
        return self.records[name]["module"]

    def audit_replacements(self):
        self.replacement_checkpoint("audit")
        self._check_replacement_events()
        backend_children = ("cpu", "cuda", "cudnn", "cusparselt", "kleidiai", "mha", "miopen",
                            "mkl", "mkldnn", "mps", "nnpack", "openmp", "opt_einsum", "quantized")
        for name in self.replacements:
            original = self.records.get(name)
            replacement = self.replacement_records.get(name)
            if original is None:
                _require(replacement is None, "unexecuted replacement has evidence")
                continue
            _require(original["state"] == "completed" and replacement is not None
                     and replacement["state"] == "admitted"
                     and replacement["frame"]["state"] == "completed", "replacement incomplete at audit")
            self.controller.require_adopted(replacement["module"])
            parent, _, attribute = name.rpartition(".")
            _require(_raw_module(self.exposed_module(parent)).get(attribute) is replacement["module"],
                     "replacement parent binding drift")
            for child, child_record in self.records.items():
                if child.rpartition(".")[0] != name or child_record["state"] != "completed":
                    continue
                short = child.rpartition(".")[2]
                exposed = self.exposed_module(child)
                _require(_raw_module(replacement["module"]).get(short) is exposed,
                         "replacement child binding drift")
                if name == "torch.backends" and short in backend_children:
                    _require(_raw_module(original["module"]).get(short) is exposed,
                             "replacement original from-import binding drift")

    def replacement_observations(self):
        try:
            self._retain_replacements()
            if not self.failed:
                if self.native_context:
                    self.replacement_checkpoint("evidence")
                else:
                    self.audit_replacements()
            _require(set(self.replacement_records) == set(self.replacement_bindings),
                     "replacement evidence records drift")
            for name in self.replacement_records:
                self._check_replacement(name, live=not self.failed)
            self._check_replacement_events()
            imports = self._native_global_observations()
            from optical_worker.runtime_environment import (
                MODULE_REPLACEMENT_OBSERVATIONS_VERSION,
                validate_module_replacement_observations,
            )

            payload = {"schema_version": MODULE_REPLACEMENT_OBSERVATIONS_VERSION,
                       "environment_sha256": self.expected_sha256,
                       "events": [dict(event) for event in self.replacement_events]}
            # As with native-global facts, preserve an immutable failed prefix
            # even if failure interrupted the admission/call completion commit.
            # Such a prefix is not a completed custody receipt.
            if not self.failed:
                validate_module_replacement_observations(payload, self.environment, imports,
                                                        require_complete=not self.native_context)
            return payload
        except BaseException as error:
            self.fail(error, "replacement_evidence")
            raise

    def _retain_native_globals(self):
        # Retention is deliberately first and remains usable after poisoning.
        values = {name: sys.modules[name] for name in self.native_globals if name in sys.modules}
        for value in values.values():
            if not any(owner is value for owner in self.retained):
                self.retained.append(value)
        return values

    def _check_native_thread(self):
        binding = _EXTERNAL_CUSTODY.get(self.session)
        _require(binding is not None and binding[0] is self
                 and _thread.get_ident() == self.native_thread == binding[4], "foreign import thread")

    def _check_native_call(self, frame):
        sequence = frame["sequence"]
        binding = self.native_call_bindings.get(sequence)
        _require(binding is not None and binding[0] is frame
                 and sequence < len(self.native_calls) and self.native_calls[sequence] is frame,
                 "native call identity drift")
        _, loader, module, operation, phase, parent, absent, marker, spec, state = binding
        # Hold a substituted spec before rejecting it, without adopting or
        # granting it the identity of the spec used by the actual delegate.
        for value in (frame["spec"], self.specs.get(loader.name)):
            if value is not spec:
                self.retained.append(value)
        _require(frame["loader"] is loader and frame["module"] is module
                 and frame["operation"] == operation and frame["phase"] == phase
                 and frame["parent"] is parent and frame["absent"] == absent
                 and frame["wrapper"] is marker and frame["spec"] is spec and frame["state"] == state,
                 "native call binding drift")
        loader._check_binding()
        _require(self.specs[loader.name] is spec, "native call spec drift")
        if module is not None:
            _require(self.records[loader.name]["module"] is module, "native call root drift")
        return binding

    def _native_call_state(self, frame, state, module):
        binding = self._check_native_call(frame)
        frame["module"], frame["state"] = module, state
        self.native_call_bindings[frame["sequence"]] = (
            binding[0], binding[1], module, *binding[3:-1], state
        )
        self._observe_file_call(frame)

    def _observe_file_call(self, frame):
        observation = self.session.runtime_observation
        observation.append("file_call", {
            "sequence": frame["sequence"], "parent_sequence": None if frame["parent"] is None else frame["parent"]["sequence"],
            "observer": frame["loader"].name, "operation": frame["operation"], "phase": frame["phase"], "state": frame["state"],
            "module_token": None if frame["module"] is None else observation.token(frame["module"]),
            "spec_token": observation.token(frame["spec"]), "loader_token": observation.token(frame["loader"].delegate),
        })

    def enter_import_wrapper(self, checkpoint, request_name=None):
        try:
            self.session.retain_import_boundary()
            if self.failed:
                self._retain_native_globals()
                self._retain_replacements()
                self.session.observe_interpreter_alias(checkpoint)
                return None  # Terminal stdlib reporting cannot reopen admission.
            self.session.observe_interpreter_alias(checkpoint)
            self.replacement_checkpoint(checkpoint, request_name=request_name)
            self.native_global_checkpoint(checkpoint, request_name=request_name)
            marker = object()
            self.native_context.append(marker)
            return marker
        except BaseException as error:
            self.fail(error, "native_global_checkpoint")
            raise

    def leave_import_wrapper(self, marker):
        if self.failed:
            # Only unwind our own context; custody lives in the separate registry.
            if any(value is marker for value in self.native_context):
                index = next(i for i, value in enumerate(self.native_context) if value is marker)
                del self.native_context[index:]
            return
        try:
            _require(bool(self.native_context) and self.native_context[-1] is marker,
                     "native wrapper nesting drift")
            self.native_context.pop()
        except BaseException as error:
            self.fail(error, "native_global_checkpoint")
            raise

    def begin_native_call(self, loader, operation, module, marker):
        self._check_native_thread()
        _require(not self.failed and self.native_context[-1] is marker, "native call wrapper drift")
        parents = [f for f in self.native_context[:-1] if type(f) is dict]
        parent = parents[-1] if parents else None
        if parent is not None:
            self._check_native_call(parent)
        absent = tuple(sorted(name for name, rule in self.native_globals.items()
                              if rule["observer"] == loader.name and operation == "exec"))
        _require(all(name not in sys.modules and name not in self.native_global_records for name in absent),
                 "native global missing observer-entry absence")
        frame = {"sequence": len(self.native_calls), "loader": loader, "spec": self.specs[loader.name],
                 "module": module, "operation": operation, "phase": self.phase, "parent": parent,
                 "absent": absent, "wrapper": marker, "state": "started"}
        self.native_calls.append(frame)
        self.native_call_bindings[frame["sequence"]] = (
            frame, loader, module, operation, self.phase, parent, absent, marker, frame["spec"], "started"
        )
        self.native_context.append(frame)
        self._observe_file_call(frame)
        return frame

    def return_native_call(self, frame, module):
        self.session.retain_import_boundary()
        self._check_native_thread()
        _require(self.native_context[-1] is frame, "native return nesting drift")
        self._native_call_state(frame, "returned", module)
        self.native_context.pop()  # Subsequent wrapper imports are not delegate work.
        self.session.observe_interpreter_alias("exec_return")
        self.replacement_checkpoint("exec_return" if frame["operation"] == "exec" else "create_return",
                                    returned_frame=frame)
        self.native_global_checkpoint("exec_return" if frame["operation"] == "exec" else "create_return",
                                      returned_frame=frame)

    def fail_native_call(self, frame):
        if frame is None:
            return
        # Preserve the frame even if metadata drift prevents a normal state commit.
        binding = self.native_call_bindings.get(frame.get("sequence"))
        if binding is not None and binding[0] is frame:
            previous = frame["state"]
            frame["state"] = "failed"
            self.native_call_bindings[frame["sequence"]] = (*binding[:-1], "failed")
            if previous != "failed":
                try:
                    self._observe_file_call(frame)
                except BaseException as observation_error:
                    self.failures.append(observation_error)
        if self.native_context and self.native_context[-1] is frame:
            self.native_context.pop()

    def _global_event(self, operation, name, record, checkpoint, request_name):
        frame = record["frame"]
        event = {
            "index": len(self.native_global_events), "operation": operation, "name": name,
            "phase": frame["phase"], "observer": frame["loader"].name, "observer_call": "exec",
            "call_sequence": frame["sequence"],
            "parent_call_sequence": None if frame["parent"] is None else frame["parent"]["sequence"],
            "absence_checkpoint": "observer_exec_entry", "checkpoint": checkpoint,
            "request_name": request_name, "object_token": record["token"],
        }
        # The projection is mutable descriptive data. Preserve the actual
        # issued facts and roots separately before exposing that projection.
        self.native_global_event_bindings.append(
            (event, tuple(event.items()), name, record, record["module"], frame))
        self.native_global_events.append(event)
        observation = self.session.runtime_observation
        if operation == "observed_pending":
            adopted = self.controller._guard_events[-1]
            _require(adopted["operation"] == "owner_adopted"
                     and adopted["details"]["operation_id"] == "native_global_observation",
                     "native global observation lacks original adoption")
            observation.native_global_adoptions[record["token"]] = adopted["index"]
        observation.append("native_global", {"event_index": event["index"],
                           "token_bindings": [{"local_token": record["token"], "object_token": observation.token(record["module"])}],
                           "adoption_event_index": observation.native_global_adoptions[record["token"]]})

    def _check_global_events(self, *, live=True):
        _require(len(self.native_global_events) == len(self.native_global_event_bindings),
                 "native global event history length drift")
        for event, binding in zip(self.native_global_events, self.native_global_event_bindings):
            issued, facts, name, record, module, frame = binding
            _require(event is issued and _same_facts(event, facts),
                     "native global issued event drift")
            self._check_global_record(name, record, live=live)
            _require(self.native_global_records.get(name) is record and record["module"] is module
                     and record["frame"] is frame and event["object_token"] == record["token"]
                     and event["call_sequence"] == frame["sequence"], "native global event root drift")

    @staticmethod
    def _check_global_metadata(name, value):
        _require(type(value) is ModuleType, "native global is not an exact module")
        raw = vars(value)
        metadata = ("__file__", "__spec__", "__loader__", "__package__")
        _require(raw.get("__name__") == name and "__path__" not in raw
                 and all(raw.get(key) is None for key in metadata),
                 "native global metadata drift")

    def _check_global_record(self, name, record, *, live=True):
        binding = self.native_global_bindings.get(name)
        _require(binding is not None and binding[0] is record
                 and binding[1] is record["module"] and binding[2] is record["frame"]
                 and binding[3] == record["token"] and binding[4] == record["state"],
                 "native global record drift")
        self._check_native_call(record["frame"])
        _require(self.records.get(name) is record, "native global admission record drift")
        if live:
            _require(sys.modules.get(name) is record["module"], "native global object removed or replaced")
            self._check_global_metadata(name, record["module"])
        return binding

    def native_global_checkpoint(self, checkpoint, *, request_name=None, returned_frame=None):
        values = self._retain_native_globals()
        self._check_native_thread()
        _require(not self.failed and not self.native_checkpoint_active,
                 "native global checkpoint reentry/failure")
        self.native_checkpoint_active = True
        try:
            _require(self.native_globals == {r["name"]: r for r in self.payload["native_globals"]},
                     "native global table drift")
            _require(tuple(tuple(r[k] for k in ("name", "observer", "operation", "phase"))
                           for r in self.native_globals.values()) == self.native_global_rules,
                     "native global rule binding drift")
            current = self.native_context[-1] if self.native_context else None
            frame = returned_frame if returned_frame is not None else current
            for name, rule in self.native_globals.items():
                previous = self.native_global_records.get(name)
                if previous is not None:
                    self._check_global_record(name, previous)
                    if previous["state"] == "observed_pending":
                        pending = previous["frame"]
                        _require(any(value is pending for value in self.native_context)
                                 or pending is returned_frame
                                 or (pending["state"] == "returned" and any(
                                     value is pending["wrapper"] for value in self.native_context)),
                                 "pending native interval disappeared")
                    continue
                _require(name not in self.native_global_bindings and name not in self.records,
                         "native global record removed")
                if name not in values:
                    continue
                _require(checkpoint in {"request_entry", "exec_return"} and type(frame) is dict,
                         "native global observed outside native exec")
                self._check_native_call(frame)
                loader = frame["loader"]
                _require(loader.source is None and loader.name == rule["observer"]
                         and frame["operation"] == rule["operation"] == "exec"
                         and frame["phase"] == rule["phase"] == self.phase
                         and name in frame["absent"]
                         and frame["state"] == ("returned" if returned_frame is not None else "started"),
                         "native global observed in wrong interval")
                value = values[name]
                self._check_global_metadata(name, value)
                record = {"kind": "native_global", "state": "observed_pending", "module": value,
                          "frame": frame, "token": len(self.native_global_bindings) + 1}
                self.native_global_records[name] = self.records[name] = record
                self.native_global_bindings[name] = (
                    record, value, frame, record["token"], "observed_pending")
                # Already strongly retained above; do not let hold()'s dedup skip guardian adoption.
                self.controller.adopt_host_owner(value, role="external_import_owner",
                                                 operation_id="native_global_observation")
                _require(not self.failed, "native global adoption swallowed failure")
                self._global_event("observed_pending", name, record, checkpoint, request_name)
        finally:
            self.native_checkpoint_active = False

    def complete_native_call(self, frame):
        self._check_native_thread()
        self._check_native_call(frame)
        _require(not self.failed and frame["state"] == "returned"
                 and self.native_context[-1] is frame["wrapper"], "native call completed out of order")
        self.complete_replacement(frame)
        self._check_global_events()
        for name in frame["absent"]:
            record = self.native_global_records.get(name)
            _require(record is not None and record["frame"] is frame
                     and record["state"] == "observed_pending", "required native global missing at return")
            binding = self._check_global_record(name, record)
            self._global_event("admitted", name, record, "exec_return", None)
            _require(not self.failed, "native global recording swallowed failure")
            record["state"] = "completed"
            self.native_global_bindings[name] = (*binding[:-1], "completed")
        self._native_call_state(frame, "completed", frame["module"])

    def audit_native_globals(self):
        self.native_global_checkpoint("audit")
        _require(not self.native_context, "native import context still active")
        for frame in self.native_calls:
            self._check_native_call(frame)
        self._check_global_events()
        for name, rule in self.native_globals.items():
            record = self.native_global_records.get(name)
            observer = self.records.get(rule["observer"])
            if observer is None:
                _require(record is None and name not in sys.modules, "unrequested native global present")
            else:
                _require(observer["state"] == "completed" and record is not None
                         and record["state"] == "completed" and record["frame"]["state"] == "completed",
                         "native global still pending or missing")
                self._check_global_record(name, record)
                self.controller.require_adopted(record["module"])

    def native_global_observations(self):
        try:
            return self._native_global_observations()
        except BaseException as error:
            self.fail(error, "native_global_evidence")
            raise

    def _native_global_observations(self):
        # Issued facts remain checked after poison. Importlib may have removed
        # live module keys during unwind; that does not rewrite held history.
        self._check_native_thread()
        _require(len(self.native_calls) == len(self.native_call_bindings), "native call history drift")
        for frame in self.native_calls:
            self._check_native_call(frame)
        for name, record in self.native_global_records.items():
            self._check_global_record(name, record, live=not self.failed)
        self._check_global_events(live=not self.failed)
        calls = [{"sequence": frame["sequence"],
                  "parent_sequence": None if frame["parent"] is None else frame["parent"]["sequence"],
                  "observer": frame["loader"].name, "operation": frame["operation"],
                  "phase": frame["phase"], "absent_names": list(frame["absent"]), "state": frame["state"]}
                 for frame in self.native_calls]
        from optical_worker.runtime_environment import (
            NATIVE_GLOBAL_OBSERVATIONS_VERSION,
            validate_native_global_observations,
        )

        payload = {"schema_version": NATIVE_GLOBAL_OBSERVATIONS_VERSION,
                   "environment_sha256": self.expected_sha256, "calls": calls,
                   "events": [dict(event) for event in self.native_global_events]}
        if not self.failed:
            validate_native_global_observations(payload, self.environment,
                                                require_complete=not self.native_context)
        return payload

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
        self._observe_external_event(self.events[-1])

    def _observe_external_event(self, event):
        observation = self.session.runtime_observation
        name, kind = event["name"], event["kind"]
        if kind == "file_module":
            loader, record = self.loaders[name], self.records.get(name)
            provenance = {"origin": loader.origin, "delegate_kind": type(loader.delegate).__name__,
                          "spec_token": observation.token(self.specs[name]), "loader_token": observation.token(loader.delegate),
                          "module_token": None if record is None else observation.token(record["module"])}
        elif kind == "absent_probe":
            requester = self.stack[-1]
            provenance = {"requester": requester, "requester_token": observation.token(self.records[requester]["module"])}
        elif kind in {"created_module", "module_alias"}:
            record = self.records[name]
            provenance = {"module_token": observation.token(record["module"]), "creator_token": observation.token(record["creator"]),
                          "publisher_token": observation.token(record["publisher"]), "relation": record["relation"],
                          "attributes": list(self.publications[name]["attributes"])}
        elif kind == "namespace":
            provenance = {"module_token": observation.token(self.records[name]["module"]),
                          "locations": list(self.loaders[name].locations)}
        elif kind == "disabled_optional":
            record = self.disabled_records[name]
            tokens = {event["requester_identity"]: observation.token(record["requester"])}
            if record["fallback"] is not None:
                tokens[event["fallback_identity"]] = observation.token(record["fallback"])
            provenance = {"requester_token": observation.token(record["requester"]),
                          "fallback_token": None if record["fallback"] is None else observation.token(record["fallback"]),
                          "token_bindings": [{"local_token": local, "object_token": tokens[local]} for local in sorted(tokens)]}
        else:
            _require(kind == "failure", "runtime external observation kind")
            provenance = {"exception_token": observation.token(self.failures[0])}
        observation.append("external", {"event": event, "provenance": provenance})

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
            and self.search_roots == self.environment.external_search_roots()
            and self.files == {(r["root"], r["path"]): r for r in p["files"]}
            and self.rules == {r["name"]: r for r in p["modules"]}
            and self.publications == {r["name"]: r for r in p["publications"]}
            and self.native_globals == {r["name"]: r for r in p["native_globals"]}
            and self.replacements == {r["name"]: r for r in p["module_replacements"]}
            and self.namespaces == {r["name"]: r for r in p["namespaces"]}
            and self.absent == {(r["name"], r["requester"], r["phase"]) for r in p["absent_probes"]},
            "external lookup index drift",
        )
        _require(
            self.disabled == {r["name"]: r for r in p["disabled_optional_modules"]},
            "disabled optional index drift",
        )

    def fail(self, error, name, *, native_create=False, notify_session=True):
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
        if notify_session:
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
        if parent in self.replacement_records:
            self._check_replacement(parent)
            self.check_module(parent)
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
            if rule["root"] == "cuda_artifacts":
                self.session._cuda_artifact_runtime(fullname)
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
        self._observe_external_event(self.events[-1])

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
        if name in self.replacement_records:
            self._check_replacement(name)
        else:
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
        raw = _raw_module(module)
        loader = self.loaders[name]
        loader._check_binding()
        self.check_module_name(name)
        expected_package = name if self.rules[name]["is_package"] else name.rpartition(".")[0]
        _require(
            raw.get("__spec__") is spec
            and raw.get("__loader__") is loader
            and spec.loader is loader
            and spec.name == name
            and spec.parent == expected_package
            and raw.get("__package__") == expected_package
            and raw.get("__file__") == loader.origin
            and spec.origin == loader.origin,
            "external module metadata drift",
        )
        if self.rules[name]["is_package"]:
            _require(
                list(raw.get("__path__", ())) == [str(Path(loader.origin).parent)]
                and list(spec.submodule_search_locations) == list(raw.get("__path__", ())),
                "external package drift",
            )
        else:
            _require(
                spec.submodule_search_locations is None and "__path__" not in vars(module),
                "external nonpackage drift",
            )

    def audit(self):
        try:
            self.session.retain_import_boundary()
            self.check_live()
            self.session.audit_generated_stdlib()
            self.session.observe_interpreter_alias("audit", final=True)
            _require(not self.stack, "external import is still in progress")
            self.audit_native_globals()
            self.audit_replacements()
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
                    or name == "__mp_main__" and self.session.interpreter_alias_admitted()
                ):
                    continue
                _require(
                    name in self.records and self.exposed_module(name) is module,
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








class _RuntimeObservation:
    """Session-held CPU facts; this table never authorizes a native operation."""

    def __init__(self, session):
        self.session = session
        self.journal, self.issued, self.owners = [], [], {}
        self.first, self.adoptions = {}, {}
        self.native_global_adoptions = {}
        self.environment, self.inputs, self.last_validated = None, None, None
        self.final_attempted, self.final_selection = False, None
        self.failures = []

    def token(self, owner):
        key = id(owner)
        previous = self.owners.get(key)
        if previous is not None:
            _require(previous[1] is owner, "runtime observation object identity reused")
            return previous[0]
        token = len(self.owners) + 1
        self.owners[key] = (token, owner)
        return token

    def append(self, kind, details):
        controller = self.session.bound_controller()
        # Copy only this bounded occurrence, not the growing import journal.
        encoded = json.dumps(details, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        copied = json.loads(encoded)
        index = len(self.journal)
        event = {"index": index, "kind": kind,
                 "guard": None if controller._guard_reference is None else dict(controller._guard_reference),
                 "lifecycle_event_index": len(controller._guard_events) - 1, "details": copied}

        def references(value):
            if type(value) is dict:
                for key, item in value.items():
                    # These are component-local facts or non-token descriptors.
                    if key in {"event", "manifest", "signature", "local_token"}:
                        continue
                    if (key.endswith("_token") or kind == "cuda_extensions_bound" and key == "token") and item is not None:
                        _require(type(item) is int and item > 0, "runtime observation token type")
                        self.first.setdefault(item, index)
                    else:
                        references(item)
            elif type(value) is list:
                for item in value:
                    references(item)

        # Keep issuance bytes before publishing the mutable diagnostic view.
        immutable = json.dumps(event, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        self.issued.append(immutable)
        self.journal.append(event)
        references(copied)

    def freeze_adoptions(self):
        guardian = self.session.bound_controller().guardian
        if guardian._state == "RELEASED":
            return  # Keep the exact matches frozen while the live table existed.
        _require(len(guardian._owners) == len(guardian._records), "runtime guardian owner table mismatch")
        for owner, record in zip(guardian._owners, guardian._records):
            observed = self.owners.get(id(owner))
            if observed is not None and observed[1] is owner:
                previous = self.adoptions.get(observed[0])
                _require(previous is None or previous == record.sequence, "runtime owner adoption changed")
                self.adoptions[observed[0]] = record.sequence

    def payload(self):
        session = self.session
        controller = session.bound_controller()
        self.freeze_adoptions()
        _require(len(self.journal) == len(self.issued), "runtime journal issuance interrupted")
        for event, encoded in zip(self.journal, self.issued):
            _require(json.dumps(event, allow_nan=False, ensure_ascii=True, sort_keys=True,
                                separators=(",", ":")).encode() == encoded, "runtime issued journal drift")
        lifecycle = controller.evidence().to_payload()
        components = session._runtime_component_projection()
        phases = [e for e in self.journal if e["kind"] == "phase_completed"]
        external = [e["details"]["event"] for e in self.journal if e["kind"] == "external"]
        kinds = {e["kind"] for e in self.journal}
        return {
            "schema_version": "q54.p12.4.runtime-custody.v3",
            "source": {"revision": session.revision, "catalog_sha256": session.catalog_sha256,
                       "snapshot_sha256": session.snapshot.sha256},
            "environment_sha256": self.environment.sha256 if "environment_bound" in kinds else None,
            "configuration": controller._guard_client.configuration.to_payload(), "lane_id": session.lane,
            "observation_cut": {"journal_length": len(self.journal),
                                "lifecycle_event_count": len(lifecycle["transition_trace"])},
            "journal": [json.loads(encoded) for encoded in self.issued],
            "objects": [{"token": token, "first_journal_index": self.first[token],
                         "guardian_sequence": self.adoptions.get(token)}
                        for token, _ in self.owners.values() if token in self.first],
            "components": components, "lifecycle": lifecycle,
            "summary": {
                "last_completed_phase": phases[-1]["details"]["phase"] if phases else None,
                "external_event_count": len(external),
                "completed_file_count": sum(e["kind"] == "file_module" and e["operation"] == "completed" for e in external),
                "successful_guard_sequences": [controller._guard_events[e["details"]["disarmed_event_index"]]["guard"]["sequence"]
                                               for e in phases],
                "binding_observed": "runtime_binding" in kinds,
                "prepared_storage_observed": "prepared_storage" in kinds,
                "input_validation_observed": "input_validated" in kinds,
                "final_execution_readiness_issued": False,
            },
        }

    def checkpoint(self):
        _require(self.environment is not None and self.inputs is not None, "runtime custody inputs missing")
        module = self.session.modules["optical_worker.runtime_custody"][0]
        record = module.RuntimeCustodyRecord.from_payload(
            self.payload(), environment=self.environment, trusted_launch_inputs=self.inputs)
        self.last_validated = record
        return record

    def final(self):
        if self.final_attempted:
            return self.final_selection
        self.final_attempted = True
        record, selection = self.last_validated, "prior_validated_cut"
        try:
            record = self.checkpoint()
            selection = "current"
        except BaseException as error:
            self.failures.append(error)
        if record is None:
            selection = "unavailable"
        observed = len(self.session.bound_controller()._guard_events)
        sidecar = {"schema_version": "q54.p12.4.runtime-custody-selection.v1", "selection": selection,
                   "runtime_record_sha256": None if record is None else record.sha256,
                   "selected_lifecycle_event_count": None if record is None else record.to_payload()["observation_cut"]["lifecycle_event_count"],
                   "observed_lifecycle_event_count": observed, "failure_diagnostic_present": False}
        self.final_selection = record, sidecar
        return self.final_selection


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
        self.warp_preparation_started = False
        self.runtime_observation = _RuntimeObservation(self)
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
            if self.phase in {"validating_input", "input_validated", "preparing_cuda_factories", "cuda_factories_ready",
                              "preparing_warp_modules", "warp_modules_ready",
                              "preparing_fixed_render_storage", "fixed_render_storage_ready",
                              "preparing_fixed_render_graph", "fixed_render_graph_ready",
                              "warming_fixed_render_graph", "fixed_render_warmups_ready"}:
                admitted = admitted | _VALIDATION_SOURCES
            if self.lane == "cuda" and self.phase not in {"materializing", "materialized"}:
                admitted = admitted | _CUDA_ARTIFACT_SOURCES
            if self.lane == "cuda" and self.phase in {"preparing_cuda_factories", "cuda_factories_ready",
                                                    "preparing_fixed_render_storage", "fixed_render_storage_ready",
                                                    "preparing_fixed_render_graph", "fixed_render_graph_ready",
                                                    "warming_fixed_render_graph", "fixed_render_warmups_ready"}:
                admitted = admitted | _CUDA_FACTORY_SOURCES
            if self.lane == "warp" and self.phase not in {"materializing", "materialized"}:
                admitted = admitted | _WARP_SETUP_SOURCES
            if self.lane == "warp" and self.phase in {"preparing_warp_modules", "warp_modules_ready",
                                                    "preparing_fixed_render_storage", "fixed_render_storage_ready",
                                                    "preparing_fixed_render_graph", "fixed_render_graph_ready",
                                                    "warming_fixed_render_graph", "fixed_render_warmups_ready"}:
                admitted = admitted | _WARP_MODULE_SOURCES
            if self.lane == "warp" and self.phase in {"preparing_fixed_render_graph", "fixed_render_graph_ready",
                                                    "warming_fixed_render_graph", "fixed_render_warmups_ready"}:
                admitted = admitted | _WARP_RENDER_SOURCES
            if self.phase in {"warming_fixed_render_graph", "fixed_render_warmups_ready"}:
                admitted = admitted | {"optical_worker.warmup": "optical_worker/warmup.py"}
                if self.lane == "warp":
                    admitted = admitted | {"optics.warp_timeline": "optics/warp_timeline.py"}
            return admitted
        raise SourceBootstrapError("invalid CPU source phase")





    def retain_import_boundary(self):
        # Intake only: no validation, adoption, event or terminal operation.
        # The registry's actual admission is already part of source custody;
        # do not trust the mutable session.external projection here.
        external = _EXTERNAL_CUSTODY.get(self)
        if external is not None:
            admission = external[0]
            raw = vars(admission)
            if "retained" in raw:
                if "replacements" in raw:
                    admission._retain_replacements()
                if "native_globals" in raw:
                    admission._retain_native_globals()
        if self in _INTERPRETER_ALIAS_CUSTODY:
            self._retain_interpreter_alias()

    def _retain_failure(self, error):
        self.failed = True
        if not any(value is error for value in self.failures):
            self.failures.append(error)
        if _SOURCE_CUSTODY.get(self) is not None:
            try:
                observation = self.runtime_observation
                observation.freeze_adoptions()
                observation.append("failure", {"source": "source_session", "exception_token": observation.token(error)})
            except BaseException as observation_error:
                self.failures.append(observation_error)
        # Commit poison and the original error before ancillary intake. The
        # available graph must still be held before any terminal operation.
        self.retain_import_boundary()
        external = _EXTERNAL_CUSTODY.get(self)
        if external is not None and "failures" in vars(external[0]) and not external[0].failed:
            external[0].fail(error, "source_session", notify_session=False)

    def fail(self, error):
        self._retain_failure(error)
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
            from optical_worker import runtime_custody

            self.audit()
            _require(runtime_custody is self.modules["optical_worker.runtime_custody"][0], "runtime custody definition drift")
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
            observation = self.runtime_observation
            observation.append("source_bound", {
                "session_token": observation.token(self), "bootstrap_token": observation.token(self.bootstrap_owner),
                "controller_token": observation.token(controller),
            })
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
            self._runtime_phase_completed("setup_definitions")
        except BaseException as error:
            self.fail(error)
            raise

    def configure_runtime_custody(self, environment_payload, trusted_launch_inputs):
        """Bind upstream CPU facts before C2d3 work; source-only custody stays C2d2."""
        try:
            self.check_live()
            _require(self.phase == "bound" and self.runtime_observation.environment is None,
                     "runtime custody inputs are single-use before setup")
            from optical_worker.runtime_custody import validate_runtime_custody_inputs
            from optical_worker.runtime_environment import RuntimeEnvironmentContract

            _require(type(environment_payload) is dict
                     and environment_payload.get("schema_version") == "q54.p12.4.runtime-environment.v5",
                     "worker runtime requires the ordinary v5 environment")
            environment = RuntimeEnvironmentContract.from_payload(environment_payload)
            validate_runtime_custody_inputs(trusted_launch_inputs, environment)
            observation = self.runtime_observation
            observation.environment = environment
            observation.inputs = json.loads(json.dumps(trusted_launch_inputs, allow_nan=False))
            self.audit()
            return observation.checkpoint()
        except BaseException as error:
            self.fail(error)
            raise

    def _runtime_phase_completed(self, phase):
        observation = self.runtime_observation
        controller = self.bound_controller()
        event = controller._guard_events[-1]
        _require(event["operation"] == "normal_disarmed", "runtime phase lacks actual DISARMED")
        observation.append("phase_completed", {
            "phase": phase, "disarmed_event_index": event["index"],
            "materialization_case_id": self.canonical_input.manifest.case_id if phase == "canonical_materialization" else None,
        })
        if observation.environment is not None:
            observation.checkpoint()

    def _runtime_component_projection(self):
        """Project only already-held issued facts, including after terminal exit intent."""
        descriptor, thread, _, resolution, observed = _INTERPRETER_ALIAS_CUSTODY[self]
        _require(thread == _thread.get_ident(), "runtime alias projection foreign thread")
        if observed is not None and observed[1] is not None:
            _require(len(self.interpreter_alias_events) == 1 and self.interpreter_alias_events[0] is observed[1]
                     and _same_facts(observed[1], observed[2]), "runtime alias issued facts drift")
        else:
            _require(not self.interpreter_alias_events, "runtime alias invented issued facts")
        identity = list(descriptor[0])
        identity[1] = list(identity[1])
        schema = self.modules["optical_worker.runtime_environment"][0]
        alias = {"schema_version": schema.INTERPRETER_ALIAS_OBSERVATIONS_VERSION,
                 "interpreter_identity": identity, "source_path": descriptor[1], "source_sha256": descriptor[3],
                 "resolution": None if resolution is None else {"name": "multiprocessing", "origin": descriptor[1],
                     "loader": "SourceFileLoader", "search_locations": [str(Path(descriptor[1]).parent)]},
                 "events": [dict(event) for event in self.interpreter_alias_events]}
        native, replacements = None, None
        admission = self.external
        if admission is not None:
            # A partially initialized or corrupt producer must raise here. It
            # is not converted to a fictitious absent component after failure.
            admission._check_native_thread()
            _require(len(admission.native_calls) == len(admission.native_call_bindings), "runtime native call history drift")
            for frame in admission.native_calls:
                admission._check_native_call(frame)
            admission._check_global_events(live=False)
            _require(set(admission.replacement_records) == set(admission.replacement_bindings)
                     and len(admission.replacement_events) == len(admission.replacement_event_bindings),
                     "runtime replacement history drift")
            for name in admission.replacement_records:
                admission._check_replacement(name, live=False)
            for event, binding in zip(admission.replacement_events, admission.replacement_event_bindings):
                _require(event is binding[0] and _same_facts(event, binding[1])
                         and admission.replacement_records.get(event["name"]) is binding[2],
                         "runtime replacement issued facts drift")
            calls = [{"sequence": frame["sequence"],
                      "parent_sequence": None if frame["parent"] is None else frame["parent"]["sequence"],
                      "observer": frame["loader"].name, "operation": frame["operation"], "phase": frame["phase"],
                      "absent_names": list(frame["absent"]), "state": frame["state"]} for frame in admission.native_calls]
            native = {"schema_version": schema.NATIVE_GLOBAL_OBSERVATIONS_VERSION,
                      "environment_sha256": admission.expected_sha256, "calls": calls,
                      "events": [dict(event) for event in admission.native_global_events]}
            replacements = {"schema_version": schema.MODULE_REPLACEMENT_OBSERVATIONS_VERSION,
                            "environment_sha256": admission.expected_sha256,
                            "events": [dict(event) for event in admission.replacement_events]}
        return {"native_globals": native, "module_replacements": replacements, "interpreter_alias": alias}

    def configure_external_environment(self, environment_payload, expected_sha256):
        """Internal trusted-launch configuration, not a worker IPC operation."""
        try:
            self.check_live()
            _require(self.runtime_observation.last_validated is not None,
                     "external configuration requires separately bound runtime custody inputs")
            _require(
                self.phase == "imported" and self not in _EXTERNAL_CUSTODY,
                "external configuration is ordered and single-use",
            )
            _EXTERNAL_CUSTODY[self] = None
            controller = self.bound_controller()
            with controller.segment("setup"):
                self.phase = "configuring_runtime"
                from optical_worker.runtime_environment import RuntimeEnvironmentContract

                _require(type(environment_payload) is dict
                         and environment_payload.get("schema_version") == "q54.p12.4.runtime-environment.v5",
                         "worker runtime requires the ordinary v5 environment")
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
                _EXTERNAL_CUSTODY[self] = (
                    admission, environment, expected_sha256, controller, _thread.get_ident())
                admission.__init__(self, environment, expected_sha256)
                observation = self.runtime_observation
                _require(environment.sha256 == observation.environment.sha256, "runtime environment differs from launch inputs")
                observation.append("environment_bound", {
                    "environment_sha256": environment.sha256, "environment_token": observation.token(environment),
                    "admission_token": observation.token(admission),
                })
                os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
                self.audit()
                admission.audit()
            self.phase = "runtime_configured"
            self._runtime_phase_completed("external_environment")
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
            self._runtime_phase_completed("canonical_materialization")
            return source.manifest
        except BaseException as error:
            self.fail(error)
            raise

    def _hold_runtime_owner(self, owner, role, *, device_ordinal=None):
        # Retain in the already adopted session BEFORE a registration that may
        # raise. Failure custody therefore includes the last returned object.
        self.runtime_owners.append(owner)
        controller = self.bound_controller()
        if role == "torch_target_descriptor":
            self._observe_runtime_return("runtime_target", owner)
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

    def _observe_runtime_return(self, producer, owner):
        controller = self.bound_controller()
        leaf = next(event["index"] for event in reversed(controller._guard_events)
                    if event["operation"] == "low_level_leaf_entered")
        observation = self.runtime_observation
        observation.append("returned_owner", {"producer": producer, "call_sequence": None,
                           "leaf_event_index": leaf, "object_token": observation.token(owner)})

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
                        self._observe_runtime_return("runtime_stream", stream)
                        _require(type(stream.device.index) is int, "CUDA stream has no indexed device")
                        self._hold_runtime_owner(
                            stream, "torch_runtime_stream", device_ordinal=stream.device.index
                        )
                    else:
                        import warp as module

                        from optical_worker.warp_setup import configure_warp_cache

                        configure_warp_cache(self, module)
                        self._runtime_leaf("indexed_warp_init", "runtime_import_or_init")
                        module.init()
                        runtime = self.external.records["warp._src.context"]["module"].runtime
                        self._hold_runtime_owner(runtime, "warp_runtime_root")
                        target = module.get_device(f"cuda:{ordinal}")
                        self.runtime_owners.append(target)
                        self._observe_runtime_return("runtime_target", target)
                        _require(target.is_cuda, "Warp returned a non-CUDA target")
                        self._hold_runtime_owner(
                            target, "warp_runtime_target", device_ordinal=int(target.ordinal)
                        )
                        self._runtime_leaf("indexed_runtime_stream", "runtime_import_or_init")
                        stream = module.get_stream(target)
                        self.runtime_owners.append(stream)
                        self._observe_runtime_return("runtime_stream", stream)
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
                observation = self.runtime_observation
                observation.append("runtime_binding", {
                    "binding_token": observation.token(binding), "controller_token": observation.token(controller),
                    "module_token": observation.token(module), "runtime_token": observation.token(runtime),
                    "target_token": observation.token(target), "stream_token": observation.token(stream),
                    "source_token": observation.token(self.canonical_input), "stream_identity": list(identity),
                })
                if self.lane == "cuda":
                    try:
                        self._bind_cuda_artifacts()
                    except BaseException:
                        if not controller._terminating and not controller._replay.unknown:
                            try:
                                controller.mark_unknown_device_scope(
                                    source="partial_cuda_artifact_binding", reason="cuda_artifact_set_not_completed")
                            except BaseException as scope_error:
                                self.runtime_owners.append(scope_error)
                        raise
            self.phase = "runtime_ready"
            self._runtime_phase_completed("runtime_initialization")
            return binding
        except BaseException as error:
            self.fail(error)
            raise

    def _cuda_artifact_runtime(self, name):
        from optical_worker.native_setup import _CUDA_EXTENSION_SETS
        from optics.cuda_extension_recipes import cuda_extension_recipe

        cuda_extension_recipe(name)
        self.check_live()
        _require(self.lane == "cuda" and self.phase == "initializing_runtime"
                 and self not in _CUDA_EXTENSION_SETS,
                 "CUDA artifact import outside original runtime initialization")
        controller = self.bound_controller()
        _require(controller._guard_reference["sequence"] == 3, "CUDA artifact import outside guard3")
        rule = self.external.rules.get(name)
        _require(rule is not None and rule["root"] == "cuda_artifacts"
                 and rule["phase"] == "runtime" and not rule["is_package"],
                 "CUDA artifact lacks its deployed ordinary file rule")
        record = _RUNTIME_BINDINGS.get(self)
        _require(record is not None, "CUDA artifact has no live runtime binding")
        return self._validate_runtime_binding_record(record[0], record)

    def _cuda_artifact_leaf(self, name, operation):
        from optical_worker.lifecycle import DeviceScopeDescriptor

        controller = self.bound_controller()
        descriptor = DeviceScopeDescriptor.exact_target(
            operation_id=name + "." + operation,
            leaf_kind="runtime_import_or_init",
            device_ordinal=controller.target_device_ordinal,
            scope_source="captured_runtime_admission",
            runtime_identity="python_extension_" + operation + "_v1",
        )
        controller.enter_device_leaf(controller.begin_device_leaf(descriptor))

    def _create_cuda_artifact(self, loader, spec, frame):
        from optical_worker.native_setup import _cuda_artifact_create_exact

        module, target, stream = self._cuda_artifact_runtime(loader.name)
        self._cuda_artifact_leaf(loader.name, "create")
        return _cuda_artifact_create_exact(module, target, stream, loader, spec, frame)

    def _exec_cuda_artifact(self, loader, extension):
        from optical_worker.native_setup import _cuda_artifact_exec_exact

        module, target, stream = self._cuda_artifact_runtime(loader.name)
        self._cuda_artifact_leaf(loader.name, _NATIVE_EXEC_OPERATION)
        _cuda_artifact_exec_exact(module, target, stream, loader.delegate, extension)

    def _bind_cuda_artifacts(self):
        from optical_worker.native_setup import CudaExtensionSet, _CUDA_EXTENSION_SETS
        from optics.cuda_extension_recipes import CUDA_EXTENSION_RECIPES, cuda_extension_entries

        for recipe in CUDA_EXTENSION_RECIPES:
            self._cuda_artifact_runtime(recipe.name)
            _require(recipe.name not in sys.modules and recipe.name not in self.external.records,
                     "CUDA artifact was preloaded")
        import robot_sim_cuda_scene_v3
        import robot_sim_cuda_lbvh_native_v3
        import robot_sim_cuda_direct_light_v10

        modules = (robot_sim_cuda_scene_v3, robot_sim_cuda_lbvh_native_v3, robot_sim_cuda_direct_light_v10)
        controller, admission = self.bound_controller(), self.external
        runtime_record = _RUNTIME_BINDINGS[self]
        owner = object.__new__(CudaExtensionSet)
        self.runtime_owners.append(owner)
        rows, evidence = [], []
        observation = self.runtime_observation
        for recipe, module in zip(CUDA_EXTENSION_RECIPES, modules, strict=True):
            admission.check_module(recipe.name)
            record, wrapper = admission.records[recipe.name], admission.loaders[recipe.name]
            _require(record["state"] == "completed" and record["module"] is module,
                     "CUDA artifact has not completed ordinary admission")
            spec, delegate = admission.specs[recipe.name], wrapper.delegate
            controller.require_adopted(module, spec, delegate)
            self.runtime_owners.extend(vars(module).get(name) for name in recipe.entries)
            entries = cuda_extension_entries(module, recipe.name)
            entry_rows = []
            for name, entry in zip(recipe.entries, entries, strict=True):
                controller.adopt_host_owner(entry, role="cuda_artifact_entry",
                                            operation_id=recipe.name + ".entry." + name)
                entry_rows.append({"name": name, "token": observation.token(entry)})
            rows.append((recipe.name, module, spec, delegate, tuple(entries)))
            evidence.append({"name": recipe.name, "module_token": observation.token(module),
                             "spec_token": observation.token(spec), "loader_token": observation.token(delegate),
                             "entries": entry_rows})
        rows = tuple(rows)
        identities = [value for _, module, spec, delegate, entries in rows for value in (module, spec, delegate, *entries)]
        _require(len({id(value) for value in identities}) == len(identities), "CUDA artifact owners alias")
        for key, value in (("session", self), ("controller", controller), ("binding", runtime_record[0]),
                           ("admission", admission), ("recipes", rows)):
            object.__setattr__(owner, key, value)
        controller.adopt_host_owner(owner, role="cuda_artifact_set", operation_id="cuda_artifacts.bind_set")
        observation.append("cuda_extensions_bound", {"owner_token": observation.token(owner),
                           "binding_token": observation.token(runtime_record[0]), "recipes": evidence})
        _CUDA_EXTENSION_SETS[self] = (owner, runtime_record, rows)

    def require_cuda_extension(self, owner, recipe_name):
        """Return only the module issued to this exact live source/runtime graph."""
        try:
            from optical_worker.native_setup import CudaExtensionSet, _CUDA_EXTENSION_SETS
            from optics.cuda_extension_recipes import cuda_extension_recipe, cuda_extension_entries

            recipe = cuda_extension_recipe(recipe_name)
            self.check_live()
            registered = _CUDA_EXTENSION_SETS.get(self)
            _require(type(owner) is CudaExtensionSet and registered is not None and registered[0] is owner,
                     "CUDA extension set is foreign, copied or unregistered")
            _, runtime_record, rows = registered
            controller = self.bound_controller()
            self._validate_runtime_binding_record(runtime_record[0], runtime_record)
            _require(owner.session is self and owner.controller is controller and owner.admission is self.external
                     and owner.binding is runtime_record[0] and owner.recipes is rows,
                     "CUDA extension set binding drift")
            controller.require_adopted(owner)
            for name, module, spec, delegate, entries in rows:
                self.external.check_module(name)
                _require(self.external.records[name]["state"] == "completed"
                         and self.external.records[name]["module"] is module
                         and self.external.specs[name] is spec and self.external.loaders[name].delegate is delegate,
                         "CUDA extension set file identity drift")
                actual_entries = cuda_extension_entries(module, name)
                _require(all(actual is original for actual, original in zip(actual_entries, entries, strict=True)),
                         "CUDA extension set entry was replaced")
                controller.require_adopted(module, spec, delegate, *entries)
            return next(row[1] for row in rows if row[0] == recipe.name)
        except BaseException as error:
            self.fail(error)
            raise

    def cuda_extensions(self):
        from optical_worker.native_setup import _CUDA_EXTENSION_SETS

        self.check_live()
        registered = _CUDA_EXTENSION_SETS.get(self)
        _require(registered is not None, "this session has no issued CUDA extension set")
        self.require_cuda_extension(registered[0], "robot_sim_cuda_scene_v3")
        return registered[0]

    def _consume_runtime_binding(self, binding):
        self.check_live()
        controller = self.bound_controller()
        controller._require_setup()
        record = _RUNTIME_BINDINGS.get(self)
        # Burn even on identity/metadata rejection. The surrounding armed
        # segment handles poison and never falls back to C0 synchronization.
        _RUNTIME_BINDINGS[self] = None
        validated = self._validate_runtime_binding_record(binding, record)
        observation = self.runtime_observation
        observation.append("binding_consumed", {"binding_token": observation.token(binding)})
        return validated

    def _validate_runtime_binding_record(self, binding, record):
        self.bound_controller()._require_setup()
        return self._validate_runtime_binding_resources(binding, record)

    def _validate_runtime_binding_resources(self, binding, record):
        """CPU identity checks, including at a healthy IDLE failure boundary."""
        self.check_live()
        controller = self.bound_controller()
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
                from optical_worker.native_contract import observe_prepared_storage
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
                storage = observe_prepared_storage(prepared)
                _UPLOADED_INPUTS[self] = (runtime_record, prepared)
                observation = self.runtime_observation
                arrays = []
                for entry in storage:
                    signature = entry.signature
                    arrays.append({"name": entry.name, "array_token": observation.token(entry.array),
                                   "storage_owner_token": None if entry.storage_owner is None else observation.token(entry.storage_owner),
                                   "signature": {"device_ordinal": signature.device_ordinal, "base_pointer": signature.base_pointer,
                                       "storage_identity": signature.storage_identity, "storage_nbytes": signature.storage_nbytes,
                                       "active_start": signature.active_start, "active_count": signature.active_count,
                                       "dtype": signature.dtype, "shape": list(signature.shape), "strides": list(signature.strides)}})
                from optical_worker.protocol import canonical_sha256

                observation.append("prepared_storage", {
                    "prepared_token": observation.token(prepared), "source_token": observation.token(prepared.source),
                    "ready_event_token": observation.token(prepared.ready_event), "manifest": prepared.manifest.to_payload(),
                    "arrays": arrays, "storage_sha256": canonical_sha256(arrays),
                })
                self.audit()
                self.external.audit()
            self.phase = "uploaded"
            self._runtime_phase_completed("native_upload")
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
                observation = self.runtime_observation
                observation.append("input_validated", {
                    "prepared_token": observation.token(prepared), "manifest_sha256": prepared.manifest.sha256,
                    "content_sha256": report.content_sha256, "topology_sha256": report.topology_sha256,
                    "validated_arrays": list(report.validated_arrays),
                })
                self.audit()
                self.external.audit()
            self.phase = "input_validated"
            self._runtime_phase_completed("input_validation")
            return report
        except BaseException as error:
            self.fail(error)
            raise



    def prepare_cuda_factories(self):
        """Parent setup foundation only: no slots, warmups or execution readiness."""
        try:
            self.check_live()
            _require(self.lane == "cuda" and self.phase == "input_validated",
                     "CUDA factory preparation requires this session's validated input, once")
            controller = self.bound_controller()
            with controller.segment("setup"):
                self.phase = "preparing_cuda_factories"
                record = _UPLOADED_INPUTS.get(self)
                _require(record is not None and record[1] is self.prepared_input
                         and self.validation_report is not None,
                         "CUDA factories require the original uploaded and validated input")
                runtime_record, prepared = record
                module, target, stream = self._validate_runtime_binding_record(runtime_record[0], runtime_record)
                controller.require_adopted(prepared)
                extensions = self.cuda_extensions()
                from optical_worker.native_setup import _prepare_cuda_factory_owners

                try:
                    factories = _prepare_cuda_factory_owners(self, prepared, module, target, stream, extensions)
                except BaseException:
                    if not controller._terminating and not controller._replay.unknown:
                        try:
                            controller.mark_unknown_device_scope(
                                source="partial_cuda_factory_preparation", reason="cuda_factories_not_completed")
                        except BaseException as scope_error:
                            self.runtime_owners.append(scope_error)
                    raise
                self.cuda_factories = factories
                self.audit()
                self.external.audit()
            self.phase = "cuda_factories_ready"
            # The six input phases remain unchanged. This is a later parent
            # setup guard, not another input phase or a final readiness record.
            self.runtime_observation.checkpoint()
            return factories
        except BaseException as error:
            self.fail(error)
            raise

    def _fail_warp_preparation(self, error, runtime_record):
        controller = self.bound_controller()
        if (controller.termination_evidence is not None or controller._terminating
                or controller._replay.failed or controller._guard_client.state != "IDLE"):
            self.fail(error)
            return
        synchronize = None
        try:
            _require(not controller._replay.compromised, "existing scope requires quarantine")
            _require(self.lane == "warp" and runtime_record is not None,
                     "no retained Warp runtime for idle disposal")
            module, target, _stream = self._validate_runtime_binding_resources(runtime_record[0], runtime_record)
            setup = self.modules["optical_worker.native_setup"][0]
            _require(sys.modules.get("optical_worker.native_setup") is setup, "setup module identity drift")

            def synchronize():
                setup._warp_synchronize_target_exact(module, target)
        except BaseException as custody_error:
            self.runtime_owners.append(custody_error)
            controller.mark_unknown_device_scope(
                source="warp_preparation_failure", reason="runtime_custody_not_validated")
        self._retain_failure(error)
        # Preserve the actual IDLE history. The existing lifecycle alone chooses
        # quarantine or its one emergency target synchronization and release.
        controller.terminate(fault="setup_failure", detection_phase="warp_preparation_idle_failure",
                             emergency_target_synchronize=synchronize)

    def prepare_warp_modules(self):
        """Compile existing optical modules; does not issue execution readiness."""
        uploaded = _UPLOADED_INPUTS.get(self)
        runtime_record = None if uploaded is None else uploaded[0]
        try:
            self.check_live()
            _require(self.lane == "warp" and self.phase == "input_validated"
                     and not self.warp_preparation_started,
                     "Warp module preparation requires this session's validated input, once")
            self.warp_preparation_started = True
            controller = self.bound_controller()
            # This must precede adopted-resource APIs: their compromised-scope
            # rejection poisons normal work, even when no segment is active.
            _require(controller._guard_client.state == "IDLE" and not controller._context_active
                     and controller._guard_reference is None and not controller._replay.compromised,
                     "Warp module preparation requires healthy IDLE target scope")
            _require(uploaded is not None and uploaded[1] is self.prepared_input
                     and self.validation_report is not None,
                     "Warp modules require the original uploaded and validated input")
            binding = runtime_record[0]
            self._validate_runtime_binding_resources(binding, runtime_record)
            controller.require_adopted(self.prepared_input)
            records = []
            with controller.segment("setup"):
                self.phase = "preparing_warp_modules"
                self.audit()
                self.external.audit()
                from optical_worker import warp_setup

                warp_setup.validate_compilation_inputs(self)
                warp_setup.validate_warp_cache(self, binding.module)
                self._runtime_leaf("warp_optical_imports", "runtime_import_or_init")
                from optics import device_scene, device_bvh, warp_native_render

                for index, imported in enumerate((device_scene, device_bvh, warp_native_render)):
                    python_module, module, definitions = warp_setup.definition_owners(self, index)
                    _require(python_module is imported, "Warp captured module identity drift")
                    self._hold_runtime_owner(module, "warp_imported_module")
                    self._hold_runtime_owner(definitions, "warp_imported_definitions")
                self.audit()
                self.external.audit()
            for index in range(3):
                with controller.segment("setup"):
                    self._validate_runtime_binding_record(binding, runtime_record)
                    self.audit()
                    self.external.audit()
                    records.append(warp_setup.compile_module(self, binding, index))
                    self.audit()
                    self.external.audit()
            self.warp_modules = warp_setup.issue_module_set(self, binding, tuple(records))
            self.phase = "warp_modules_ready"
            self.runtime_observation.checkpoint()
            return self.warp_modules
        except BaseException as error:
            self._fail_warp_preparation(error, runtime_record)
            raise

    def prepare_fixed_render_storage(self, *, traversal_diagnostics=False):
        """One supervised R0 storage construction, without execution readiness."""
        try:
            self.check_live()
            expected_phase = "cuda_factories_ready" if self.lane == "cuda" else "warp_modules_ready"
            _require(self.phase == expected_phase and type(traversal_diagnostics) is bool,
                     "fixed render storage requires completed compilation, once")
            _require(self.prepared_input.manifest.input_contract == "fixed_native_scene_v1",
                     "fixed render storage requires canonical R0 input")
            controller = self.bound_controller()
            with controller.segment("setup"):
                self.phase = "preparing_fixed_render_storage"
                uploaded = _UPLOADED_INPUTS.get(self)
                _require(uploaded is not None and uploaded[1] is self.prepared_input
                         and self.validation_report is not None,
                         "fixed render storage requires the original validated upload")
                record, prepared = uploaded
                module, target, stream = self._validate_runtime_binding_record(record[0], record)
                controller.require_adopted(prepared)
                self.audit()
                self.external.audit()
                if self.lane == "warp":
                    from optical_worker import warp_setup

                    warp_setup.validate_module_set(self, self.warp_modules)
                else:
                    self.cuda_extensions()
                from optical_worker.native_setup import _prepare_fixed_render_storage

                storage = _prepare_fixed_render_storage(
                    self, module, target, stream, traversal_diagnostics=traversal_diagnostics)
                self._hold_runtime_owner(storage, "fixed_render_storage")
                self.fixed_render_storage = storage
                if self.lane == "warp":
                    warp_setup.validate_module_set(self, self.warp_modules)
                self.audit()
                self.external.audit()
            self.phase = "fixed_render_storage_ready"
            self.runtime_observation.checkpoint()
            return storage
        except BaseException as error:
            self.fail(error)
            raise

    def prepare_fixed_render_graph(self, *, profile, shadows, pass_mode):
        """Prepare provisional frame commands/events, without executing optical work."""
        try:
            self.check_live()
            _require(self.phase == "fixed_render_storage_ready", "fixed render graph requires its storage, once")
            _require(type(profile) is str and profile in {"direct_light_full", "rgb_preview", "render_only"}
                     and type(shadows) is bool and type(pass_mode) is str
                     and pass_mode in {"primary_unprofiled", "diagnostic_profiled"},
                     "invalid fixed render graph modes")
            controller = self.bound_controller()
            with controller.segment("setup"):
                self.phase = "preparing_fixed_render_graph"
                uploaded = _UPLOADED_INPUTS.get(self)
                _require(uploaded is not None and uploaded[1] is self.prepared_input
                         and self.validation_report is not None,
                         "fixed render graph requires the original validated upload")
                record, prepared = uploaded
                module, target, stream = self._validate_runtime_binding_record(record[0], record)
                controller.require_adopted(prepared, self.fixed_render_storage)
                self.audit()
                self.external.audit()
                if self.lane == "warp":
                    from optical_worker import warp_setup
                    compiled = warp_setup.validate_module_set(self, self.warp_modules)
                else:
                    compiled = self.cuda_extensions()
                from optical_worker.native_input import certify_native_readiness
                from optical_worker.native_setup import _prepare_fixed_render_graph

                readiness = certify_native_readiness(prepared, self.validation_report,
                                                     compile_complete=True, warmup_frames_completed=0)
                self._hold_runtime_owner(readiness, "provisional_input_readiness")
                graph = _prepare_fixed_render_graph(self, module, target, stream, readiness=readiness,
                    profile=profile, shadows=shadows, pass_mode=pass_mode)
                self._hold_runtime_owner(graph, "provisional_fixed_render_graph")
                if self.lane == "warp":
                    _require(warp_setup.validate_module_set(self, self.warp_modules) is compiled,
                             "render recording changed its compiled modules")
                else:
                    _require(self.cuda_extensions() is compiled, "render binding changed its CUDA extensions")
                self.audit()
                self.external.audit()
            self.fixed_render_graph = graph
            self.phase = "fixed_render_graph_ready"
            self.runtime_observation.checkpoint()
            return graph
        except BaseException as error:
            self.fail(error)
            raise

    def run_fixed_render_warmups(self):
        """Execute and observe ten setup-only frames; no final/sample authority."""
        try:
            self.check_live()
            _require(self.phase == "fixed_render_graph_ready", "fixed render warmups require their graph, once")
            with self.bound_controller().segment("setup"):
                self.phase = "warming_fixed_render_graph"
                uploaded = _UPLOADED_INPUTS.get(self)
                _require(uploaded is not None and uploaded[1] is self.prepared_input,
                         "warmup requires the original uploaded input")
                record, prepared = uploaded
                runtime, target, stream = self._validate_runtime_binding_record(record[0], record)
                from optical_worker.warmup import _prepare_fixed_render_warmups, _run_fixed_render_warmups

                authority = _prepare_fixed_render_warmups(self, runtime, target, stream)
            evidence = _run_fixed_render_warmups(authority)
            self.fixed_render_warmups = evidence
            self.phase = "fixed_render_warmups_ready"
            self.runtime_observation.checkpoint()
            return evidence
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

    def _retain_interpreter_alias(self):
        binding = _INTERPRETER_ALIAS_CUSTODY.get(self)
        _require(binding is not None, "interpreter alias custody missing")
        retained = binding[2]
        for name in ("__main__", "__mp_main__", "multiprocessing"):
            if name in sys.modules and not any(value is sys.modules[name] for value in retained):
                retained.append(sys.modules[name])
        publisher = sys.modules.get("multiprocessing")
        if type(publisher) is ModuleType:
            raw = _raw_module(publisher)
            for value in (raw.get("__spec__"), raw.get("__loader__")):
                if not any(owner is value for owner in retained):
                    retained.append(value)
        return binding

    def _check_interpreter_alias_source(self):
        binding = self._retain_interpreter_alias()
        descriptor, thread, _, resolution, observed = binding
        _require(thread == _thread.get_ident(), "interpreter alias foreign thread")
        _require(descriptor[0] == _generated_interpreter_identity()
                 and sys.modules.get("__main__") is descriptor[4]
                 and self.bootstrap_owner is descriptor[4], "interpreter alias main/identity drift")
        path = Path(descriptor[1])
        _require(path.resolve(strict=True) == path and not path.is_symlink()
                 and path.read_bytes() == descriptor[2]
                 and hashlib.sha256(descriptor[2]).hexdigest() == descriptor[3],
                 "interpreter alias source drift")
        return binding

    def bind_interpreter_alias_resolution(self, spec, path, target):
        descriptor, thread, retained, resolution, observed = self._check_interpreter_alias_source()
        retained.append(spec)
        _require(resolution is None and observed is None and path is None and target is None
                 and "multiprocessing" not in sys.modules and "__mp_main__" not in sys.modules,
                 "interpreter alias resolution is not cold/single-use")
        _require(type(spec) is importlib.machinery.ModuleSpec, "interpreter alias spec type")
        loader = vars(spec).get("loader")
        retained.append(loader)
        _require(type(loader) is importlib.machinery.SourceFileLoader
                 and spec.name == "multiprocessing" and spec.origin == descriptor[1]
                 and loader.name == spec.name and loader.path == descriptor[1]
                 and list(spec.submodule_search_locations) == [str(Path(descriptor[1]).parent)],
                 "interpreter alias standard resolution drift")
        _INTERPRETER_ALIAS_CUSTODY[self] = (descriptor, thread, retained, (spec, loader), None)

    def observe_interpreter_alias(self, checkpoint, *, final=False):
        try:
            self.retain_import_boundary()
            binding = self._retain_interpreter_alias()
            if self.failed:
                return  # Terminal diagnostics enlarge custody, never observations.
            descriptor, thread, retained, resolution, observed = self._check_interpreter_alias_source()
            publisher = sys.modules.get("multiprocessing")
            alias_present = "__mp_main__" in sys.modules
            if resolution is None:
                _require(publisher is None and not alias_present and observed is None,
                         "interpreter alias appeared before standard resolution")
                return
            spec, loader = resolution
            if publisher is None:
                _require(not final and "multiprocessing" not in sys.modules
                         and not alias_present and observed is None, "interpreter alias publisher disappeared")
                return
            _require(type(publisher) is ModuleType, "interpreter alias publisher type")
            raw = _raw_module(publisher)
            retained.extend((raw.get("__spec__"), raw.get("__loader__")))
            _require(raw.get("__name__") == "multiprocessing" and raw.get("__spec__") is spec
                     and raw.get("__loader__") is loader and raw.get("__file__") == descriptor[1]
                     and raw.get("__package__") == "multiprocessing"
                     and raw.get("__path__") == [str(Path(descriptor[1]).parent)]
                     and spec.name == "multiprocessing" and spec.origin == descriptor[1]
                     and spec.loader is loader and loader.name == "multiprocessing"
                     and loader.path == descriptor[1]
                     and list(spec.submodule_search_locations) == raw["__path__"],
                     "interpreter alias publisher metadata drift")
            initializing = vars(spec).get("_initializing")
            _require(type(initializing) is bool, "interpreter alias publisher initializing metadata")
            if observed is not None:
                _require(observed[0] is publisher and alias_present
                         and sys.modules["__mp_main__"] is descriptor[4], "interpreter alias identity drift")
                if observed[1] is not None:
                    _require(not initializing and len(self.interpreter_alias_events) == 1
                             and self.interpreter_alias_events[0] is observed[1]
                             and _same_facts(observed[1], observed[2]), "interpreter alias facts drift")
                    return
            if not alias_present:
                _require(initializing and not final, "completed interpreter publisher lacks alias")
                return
            _require(sys.modules["__mp_main__"] is descriptor[4], "interpreter alias is not cold main")
            if initializing:
                _require(not final, "interpreter alias publisher incomplete at audit")
                _INTERPRETER_ALIAS_CUSTODY[self] = (descriptor, thread, retained, resolution, (publisher, None, None))
                return
            _require(checkpoint in {"request_entry", "exec_return", "audit"},
                     "interpreter alias first observation checkpoint")
            event = {"index": 0, "operation": "interpreter_alias_observed", "checkpoint": checkpoint,
                     "alias": "__mp_main__", "target": "__main__", "publisher": "multiprocessing",
                     "main_token": 1, "publisher_token": 2}
            _require(not self.interpreter_alias_events, "interpreter alias projection already exists")
            self.interpreter_alias_events.append(event)
            _INTERPRETER_ALIAS_CUSTODY[self] = (
                descriptor, thread, retained, resolution, (publisher, event, tuple(event.items())))
            observation = self.runtime_observation
            observation.append("interpreter_alias", {
                "event_index": event["index"], "token_bindings": [
                    {"local_token": 1, "object_token": observation.token(descriptor[4])},
                    {"local_token": 2, "object_token": observation.token(publisher)}],
                "publisher_token": observation.token(publisher), "spec_token": observation.token(resolution[0]),
                "loader_token": observation.token(resolution[1]),
            })
        except BaseException as error:
            self.fail(error)
            raise

    def interpreter_alias_admitted(self):
        binding = _INTERPRETER_ALIAS_CUSTODY[self]
        return binding[4] is not None and binding[4][1] is not None

    def interpreter_alias_observations(self):
        try:
            # Read-only checkpoint: issuance cannot first observe an alias.
            # After poison, retain and validate historical facts without
            # pretending importlib's unwind preserved live module keys.
            self.observe_interpreter_alias("evidence")
            descriptor, thread, _, resolution, observed = self._retain_interpreter_alias()
            _require(thread == _thread.get_ident(), "interpreter alias evidence foreign thread")
            if observed is not None and observed[1] is not None:
                _require(len(self.interpreter_alias_events) == 1
                         and self.interpreter_alias_events[0] is observed[1]
                         and _same_facts(observed[1], observed[2]), "interpreter alias evidence drift")
            else:
                _require(not self.interpreter_alias_events, "invented interpreter alias evidence")
            identity = list(descriptor[0])
            identity[1] = list(identity[1])
            # Also usable before the external-contract phase. Replay lives in
            # runtime_environment; issuance must not import that later phase.
            payload = {"schema_version": "q54.p12.4.interpreter-alias-observations.v1",
                    "interpreter_identity": identity,
                    "source_path": descriptor[1], "source_sha256": descriptor[3],
                    "resolution": None if resolution is None else {
                        "name": "multiprocessing", "origin": descriptor[1], "loader": "SourceFileLoader",
                        "search_locations": [str(Path(descriptor[1]).parent)]},
                    "events": [dict(event) for event in self.interpreter_alias_events]}
            return payload
        except BaseException as error:
            self.fail(error)
            raise

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
            self.retain_import_boundary()
            self.check_live()
            self.audit_generated_stdlib()
            self.observe_interpreter_alias("audit", final=True)
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
                _require(loader.source_cache is not None, "captured source cache missing")
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
    interpreter_alias = _capture_interpreter_alias(root, paths)
    sources = _capture(root, revision, lane, catalog_sha256)
    _require(
        len(sys.meta_path) == len(finders)
        and all(actual is expected for actual, expected in zip(sys.meta_path, finders))
        and tuple(sys.path) == paths
        and tuple(sys.path_hooks) == hooks,
        "cold interpreter topology changed during capture",
    )
    session = _SourceSession(root, revision, lane, catalog_sha256, sources)
    session.interpreter_alias_events = []
    _INTERPRETER_ALIAS_CUSTODY[session] = (
        interpreter_alias, _thread.get_ident(), [interpreter_alias[4]], None, None)
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
