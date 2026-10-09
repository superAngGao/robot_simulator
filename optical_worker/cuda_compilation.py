"""Fixed captured CUDA JIT prefix, with retained direct loading and no readiness."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType, ModuleType

from optics.cuda_extension_recipes import CUDA_EXTENSION_RECIPES, cuda_extension_kwargs

from .compilation_contract import (
    CUDA_COMPILATION_SELECTION_VERSION, CUDA_COMPILATION_VERSION, CUDA_COMPILE_SEQUENCES,
    CUDA_COMPILE_STEPS, cuda_native_operation, cuda_role_anchor,
)
from .cuda_compilation_evidence import CudaCompilationRecord
from .lifecycle import DeviceScopeDescriptor
from .protocol import canonical_json_bytes, canonical_sha256
from .wait_protocol import exact_fields

_COMPILATIONS = {}
_COMPILED_OWNERS = {}
_ACTIVE_HOOK = None
_INJECTION_VARIABLES = frozenset({
    "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "PYTORCH_NVCC", "CFLAGS", "CXXFLAGS", "CPPFLAGS",
    "LDFLAGS", "CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "LD_PRELOAD",
})
_CONDA = "/home/ang.gao/miniforge3/envs/robot_sim"
_CPP_EXTENSION_SHA256 = "52ba4de5deedd894e2bdafc9b7f40526dbc2cc1e4c560bbae3b38d4a5dcab0c6"


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _regular_bytes(path):
    path = Path(path)
    _require(not path.is_symlink() and stat.S_ISREG(path.stat().st_mode), "build input/output is not a regular file")
    data = path.read_bytes()
    return data, {"path": str(path), "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _private_directory(path):
    path = Path(path)
    _require(path.is_absolute() and str(path) == str(path.resolve(strict=True)), "private build directory is not canonical")
    for parent in (path, *path.parents):
        _require(not parent.is_symlink(), "private build directory has symlink ancestor")
    info = path.stat()
    _require(stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o700 and info.st_uid == os.getuid(),
             "build directory is not private and run-owned")
    return path


class _Toolchain:
    def __init__(self, payload, session):
        p = exact_fields(payload, {"schema_version", "source_sha256", "environment_sha256", "environment",
                                   "roots", "files", "symlinks", "tools", "run_root", "pch_paths"})
        _require(p["schema_version"] == "q54.p12.4.cuda-toolchain.v1"
                 and p["source_sha256"] == session.snapshot.sha256
                 and p["environment_sha256"] == session.runtime_observation.environment.sha256,
                 "compiler deployment differs from captured source/environment")
        self.encoded = canonical_json_bytes(p)
        self.sha256 = canonical_sha256(p)
        self.payload = p
        self.root = _private_directory(p["run_root"])
        _require(type(p["roots"]) is list and type(p["files"]) is list and p["files"]
                 and type(p["symlinks"]) is list, "compiler inventory missing")
        roots = [Path(path) for path in p["roots"]]
        _require(len(set(roots)) == len(roots) and all(path.is_absolute() and path.is_dir() for path in roots),
                 "compiler inventory root mismatch")
        paths = set()
        for row in p["files"]:
            exact_fields(row, {"path", "size", "sha256"})
            _require(type(row["path"]) is str and type(row["size"]) is int and row["size"] >= 0
                     and type(row["sha256"]) is str and len(row["sha256"]) == 64
                     and all(c in "0123456789abcdef" for c in row["sha256"]), "compiler inventory file shape")
            path = Path(row["path"])
            _require(path.is_absolute() and path not in paths and path.resolve(strict=True) == path
                     and any(path.is_relative_to(root) for root in roots), "compiler inventory file escaped roots")
            paths.add(path)
            data, observed = _regular_bytes(path)
            _require(observed == row, "compiler inventory file changed")
        tools = exact_fields(p["tools"], {"nvcc", "ninja", "cxx", "cc", "cc1plus", "collect2", "as", "ld"})
        _require(all(type(path) is str for path in tools.values()) and tools["nvcc"] == _CONDA + "/bin/nvcc"
                 and tools["cxx"] == "/usr/bin/x86_64-linux-gnu-g++-13"
                 and tools["cc"] == "/usr/bin/x86_64-linux-gnu-gcc-13"
                 and all(Path(path) in paths for path in tools.values()), "compiler tools differ from fixed deployment")
        _require(Path(_CONDA + "/bin/nvcc.profile") in paths, "nvcc implicit profile missing from inventory")
        for row in p["symlinks"]:
            exact_fields(row, {"path", "target"})
            path = Path(row["path"])
            _require(path.is_absolute() and path.is_symlink() and os.readlink(path) == row["target"]
                     and path.resolve(strict=True) in paths, "compiler symlink chain drift")
        expected_pch = [_CONDA + "/lib/python3.12/site-packages/torch/include/torch/extension.h" + suffix
                        for suffix in (".gch", ".sign")]
        _require(p["pch_paths"] == expected_pch, "shared PCH path mismatch")
        self.audit()

    def audit(self):
        _require(canonical_json_bytes(self.payload) == self.encoded, "frozen compiler deployment changed")
        environment = self.payload["environment"]
        _require(type(environment) is dict and all(type(k) is str and (v is None or type(v) is str) for k, v in environment.items()),
                 "compiler environment shape")
        required = {"CUDA_HOME": _CONDA, "CXX": self.payload["tools"]["cxx"], "CC": self.payload["tools"]["cc"],
                    "TORCH_CUDA_ARCH_LIST": "9.0", "MAX_JOBS": "1", "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
                    "TMPDIR": str(self.root / "tmp"), "TORCH_EXTENSIONS_DIR": str(self.root / "torch")}
        _require(all(environment.get(key) == value for key, value in required.items())
                 and {"PATH", "LD_LIBRARY_PATH"} <= set(environment)
                 and all(os.environ.get(key) == value for key, value in environment.items())
                 and not (_INJECTION_VARIABLES & os.environ.keys()), "compiler environment changed or injected")
        for path in self.payload["pch_paths"]:
            _require(not os.path.lexists(path), "shared Torch PCH must remain absent; it is never removed here")
        _private_directory(self.root)
        _private_directory(self.root / "tmp")
        _private_directory(self.root / "torch")


@dataclass(frozen=True, eq=False, slots=True)
class CompiledCudaOwner:
    recipe_id: str
    module: object
    spec: object
    loader: object
    entries: tuple
    guard_sequence: int


def _require_compiled_owner(owner, session):
    record = _COMPILED_OWNERS.get(owner)
    _require(type(owner) is CompiledCudaOwner and record is not None and record[0] is session,
             "compiled owner is not issued to this source session")
    compiler, index, identities = record[1:]
    compiler.audit()
    _require(owner is compiler.owners[index] and identities == tuple(id(v) for v in
             (owner.module, owner.spec, owner.loader, *owner.entries)), "compiled owner identity changed")
    compiler.audit_module(index)
    return owner


def _torch_compile_exact(torch, target, stream, compiler, kwargs):
    with torch.cuda.device(target):
        with torch.cuda.stream(stream):
            properties = torch.cuda.get_device_properties(target)
            compiler._retain(properties)
            _require((properties.major, properties.minor) == (9, 0), "compiler target is not the frozen H200 architecture")
            # cpp_extension itself performs native availability checks. Its
            # first import belongs inside the same observed indexed compile
            # leaf, not in an unobserved gap before that boundary.
            compiler.install_hook()
            return compiler.dependency.load_inline(**kwargs)


def _cuda_extension_create_exact(torch, target, stream, delegate, spec, compiler):
    with torch.cuda.device(target):
        with torch.cuda.stream(stream):
            module = delegate.create_module(spec)
            # Retain before context restoration, adoption, metadata initialization
            # or another native call can fail.
            compiler.module_returned(module)
            return module


def _cuda_extension_exec_exact(torch, target, stream, delegate, module):
    with torch.cuda.device(target):
        with torch.cuda.stream(stream):
            delegate.exec_module(module)


class _Compilation:
    def __init__(self, session, root, runtime_record, module, target, stream):
        self.session, self.root, self.runtime_record = session, root, runtime_record
        self.module, self.target, self.stream = module, target, stream
        self.controller = session.bound_controller()
        self.thread = threading.current_thread()
        self.observation = session.runtime_observation
        self.toolchain = None
        self.dependency = self.original_hook = self.original_load_inline = None
        self.hook = None
        self.index = None
        self.owners = []
        self.live = []
        self.steps = [[] for _ in CUDA_EXTENSION_RECIPES]
        self.rows = [{"recipe_id": recipe.name, "guard_sequence": CUDA_COMPILE_SEQUENCES[index],
                      "build_directory": None, "source_files": [], "generated_files": [], "artifact": None,
                      "entries": [], "object_bindings": dict.fromkeys(("module_token", "spec_token", "loader_token", "compiled_owner_token"))}
                     for index, recipe in enumerate(CUDA_EXTENSION_RECIPES)]
        self.failed_flag = False
        self.failure_objects = []
        self.last_validated = None
        self.final_selection = None
        self.completed = 0

    def initialize(self):
        self.toolchain = _Toolchain(self.root.deployment, self.session)
        self.observation.token(self.toolchain)
        self.audit()

    def _bind_dependency(self):
        _require(self.dependency is None, "build dependency binds once")
        from torch.utils import cpp_extension

        self.dependency = cpp_extension
        self.original_hook = cpp_extension._import_module_from_library
        self.original_load_inline = cpp_extension.load_inline
        admitted = self.session.external.records.get("torch.utils.cpp_extension")
        loader = self.session.external.loaders["torch.utils.cpp_extension"]
        _require(admitted is not None and admitted["module"] is cpp_extension and admitted["state"] == "completed"
                 and hashlib.sha256(loader.source).hexdigest() == _CPP_EXTENSION_SHA256,
                 "cpp_extension is not the admitted pinned dependency")
        for function, name in ((self.original_hook, "_import_module_from_library"), (self.original_load_inline, "load_inline")):
            _require(type(function) is FunctionType and function.__globals__ is vars(cpp_extension)
                     and function.__name__ == name and function.__code__.co_name == name
                     and function.__code__.co_filename == loader.origin, "build dependency function was replaced")
            self.observation.token(function)
        self.function_codes = self.original_hook.__code__, self.original_load_inline.__code__
        self.observation.token(cpp_extension)
        self.toolchain.audit()
        self.audit()

    def install_hook(self):
        global _ACTIVE_HOOK
        self.audit()
        _require(_ACTIVE_HOOK is None and self.hook is None, "overlapping or repeated compiler hook")
        if self.dependency is None:
            self._bind_dependency()
        recipe = CUDA_EXTENSION_RECIPES[self.index]
        _require(self.dependency._import_module_from_library is self.original_hook
                 and recipe.name not in sys.modules and self.dependency.JIT_EXTENSION_VERSIONER.get_version(recipe.name) is None,
                 "earlier hook replacement, generated module or extension cache")
        self.toolchain.audit()
        self.hook = self._direct_load
        self._retain(self.hook)
        _ACTIVE_HOOK = self
        self.dependency._import_module_from_library = self.hook

    def audit(self):
        _require(_COMPILATIONS.get(self.session) is self and self.root.collector is self
                 and threading.current_thread() is self.thread and not self.failed_flag, "compiler identity/thread/failure")
        self.session.check_live()
        self.root.audit_retention()
        self.session._validate_runtime_binding_record(self.runtime_record[0], self.runtime_record)
        _require(self.session.bound_controller() is self.controller, "compiler controller replaced")
        if self.dependency is not None:
            _require(self.dependency.load_inline is self.original_load_inline
                     and (self.original_hook.__code__, self.original_load_inline.__code__) == self.function_codes,
                     "build function code identity changed")

    def _retain(self, owner):
        return self.observation.token(owner)

    def _adopt(self, owner, field, entry=None):
        token = self._retain(owner)
        recipe = CUDA_EXTENSION_RECIPES[self.index]
        role, operation = cuda_role_anchor(recipe.name, field, entry)
        self.controller.adopt_host_owner(owner, role=role, operation_id=operation)
        self.observation.freeze_adoptions()
        _require(token in self.observation.adoptions, "compiled role lacks actual direct adoption")
        return token

    def _step(self, operation, exception=None):
        index = self.index
        row = self.rows[index]
        if operation != "failed":
            _require(not self.failed_flag and operation == CUDA_COMPILE_STEPS[len(self.steps[index])], "compiler step order")
        details = {"recipe_id": row["recipe_id"], "operation": operation,
                   **row["object_bindings"], "entry_tokens": list(row["entries"]),
                   "exception_token": None if exception is None else self._retain(exception)}
        self.observation.append("compile_step", details)
        if operation != "failed":
            self.steps[index].append(operation)

    def _enter(self, site):
        self.audit()
        name, identity = cuda_native_operation(CUDA_EXTENSION_RECIPES[self.index].name, site)
        permit = self.controller.begin_device_leaf(DeviceScopeDescriptor.exact_target(
            operation_id=name, leaf_kind="compile_or_jit", device_ordinal=self.controller.target_device_ordinal,
            scope_source="closed_worker_allowlist", runtime_identity=identity,
        ))
        self.controller.enter_device_leaf(permit)

    def compile_recipe(self, index):
        global _ACTIVE_HOOK
        self.audit()
        _require(type(index) is int and index == self.completed and not self.steps[index], "repeated or wrong compiler recipe")
        self.index = index
        recipe = CUDA_EXTENSION_RECIPES[index]
        _require(self.controller._guard_reference == {"sequence": CUDA_COMPILE_SEQUENCES[index], "role": "setup"},
                 "compiler called outside its original ARMED segment")
        self.toolchain.audit()
        _require(_ACTIVE_HOOK is None and (self.dependency is None or self.dependency._import_module_from_library is self.original_hook)
                 and recipe.name not in sys.modules,
                 "compile overlap, earlier hook replacement, module or cache")
        source = recipe.source_module
        held = self.session.modules.get(source.__name__)
        _require(held is not None and held[0] is source and held[2] == "completed", "recipe source is not captured")
        self._retain(source.CPP_SOURCE)
        self._retain(source.CUDA_SOURCE)
        directory = self.toolchain.root / recipe.name
        _require(not os.path.lexists(directory), "recipe output directory already exists")
        directory.mkdir(mode=0o700)
        _private_directory(directory)
        row = self.rows[index]
        row["build_directory"] = str(directory)
        source_bytes = held[1].source
        row["source_files"] = [{"path": held[1].origin, "size": len(source_bytes), "sha256": hashlib.sha256(source_bytes).hexdigest()}]
        self.live.append({"source": source, "source_bytes": source_bytes, "generated": {}, "module": None,
                          "delegate": None, "spec": None, "wrapper": None, "entries": (), "create_returned": False,
                          "publication": "direct"})
        self._step("build_entered")
        kwargs = cuda_extension_kwargs(recipe.name)
        kwargs["build_directory"] = str(directory)
        self._retain(kwargs)
        self.hook = None
        failure = None
        try:
            self._enter("cuda_compile")
            result = _torch_compile_exact(self.module, self.target, self.stream, self, kwargs)
            self._retain(result)
            _require(result is self.live[index]["module"] and len(self.owners) == index + 1,
                     "load_inline did not return its admitted module")
            self.audit_module(index)
        except BaseException as error:
            failure = error
        finally:
            self._finish_build(failure)
        self._step("hook_restored")

    def _finish_build(self, failure):
        """Restore once, audit every exit, and preserve the earliest failure."""
        global _ACTIVE_HOOK
        errors = [] if failure is None else [failure]
        if self.hook is not None:
            try:
                _require(_ACTIVE_HOOK is self and self.dependency._import_module_from_library is self.hook,
                         "direct helper changed during compilation")
            except BaseException as error:
                errors.append(error)
            try:
                self.dependency._import_module_from_library = self.original_hook
            except BaseException as error:
                errors.append(error)
            finally:
                if _ACTIVE_HOOK is self:
                    _ACTIVE_HOOK = None
        try:
            self.toolchain.audit()
        except BaseException as error:
            errors.append(error)
        for error in errors:
            try:
                self.failed(error)
            except BaseException as diagnostic:
                # The already rooted collector holds both exceptions even if
                # journal issuance fails. A partial journal grants no success.
                self.failure_objects.append(diagnostic)
        if errors:
            raise errors[0]

    def _direct_load(self, module_name, path, is_python_module):
        self.audit()
        recipe = CUDA_EXTENSION_RECIPES[self.index]
        row = self.rows[self.index]
        _require(_ACTIVE_HOOK is self and self.dependency._import_module_from_library is self.hook
                 and module_name == recipe.name and type(is_python_module) is bool and is_python_module
                 and path == row["build_directory"] and module_name not in sys.modules
                 and self.steps[self.index] == ["build_entered"], "unexpected generated direct loader entry")
        directory = _private_directory(path)
        allowed = {"main.cpp", "cuda.cu", "build.ninja", "main.o", "cuda.cuda.o", ".ninja_log", ".ninja_deps", "lock", recipe.name + ".so"}
        live = self.live[self.index]
        for file in sorted(directory.iterdir()):
            _require(file.name in allowed, "unlisted compiler output")
            data, metadata = _regular_bytes(file)
            live["generated"][file.name] = data
            self._retain(data)
            if file.name == recipe.name + ".so":
                row["artifact"] = metadata
            else:
                row["generated_files"].append(metadata)
        _require(row["artifact"] is not None and {"main.cpp", "cuda.cu", "build.ninja"} <= live["generated"].keys(),
                 "compiler did not generate required fresh files")
        self._step("generated")
        module = self.session.load_compiled_extension(self, row["artifact"])
        self.audit_module(self.index)
        entries = tuple(vars(module)[name] for name in recipe.entries)
        for name, entry in zip(recipe.entries, entries):
            _require(callable(entry) and getattr(entry, "__module__", None) == recipe.name
                     and getattr(entry, "__name__", None) == name,
                     "compiled entry is not bound to exact module/name")
        tokens = [{"name": name, "entry_token": self._adopt(entry, "entry_token", name)}
                  for name, entry in zip(recipe.entries, entries)]
        live["entries"] = entries
        row["entries"] = tokens
        self._step("entries_bound")
        owner = CompiledCudaOwner(recipe.name, module, live["spec"], live["delegate"], entries, CUDA_COMPILE_SEQUENCES[self.index])
        token = self._adopt(owner, "compiled_owner_token")
        self.owners.append(owner)
        _COMPILED_OWNERS[owner] = (self.session, self, self.index, tuple(id(v) for v in (module, live["spec"], live["delegate"], *entries)))
        row["object_bindings"]["compiled_owner_token"] = token
        self._step("owner_registered")
        return module

    def create_entered(self, spec, delegate, wrapper):
        live = self.live[self.index]
        self._retain(wrapper)
        live.update(spec=spec, delegate=delegate, wrapper=wrapper)
        spec_token = self._adopt(spec, "spec_token")
        loader_token = self._adopt(delegate, "loader_token")
        self.rows[self.index]["object_bindings"].update(spec_token=spec_token, loader_token=loader_token)
        self._step("create_entered")

    def module_returned(self, module):
        self._retain(module)
        live = self.live[self.index]
        live["module"] = module
        live["create_returned"] = type(module) is ModuleType
        _require(type(module) is ModuleType and vars(module).get("__name__") == CUDA_EXTENSION_RECIPES[self.index].name,
                 "native create returned wrong module")
        token = self._adopt(module, "module_token")
        self.rows[self.index]["object_bindings"]["module_token"] = token
        self._step("module_returned")

    def audit_module(self, index):
        live = self.live[index]
        module, spec, delegate, wrapper = (live[key] for key in ("module", "spec", "delegate", "wrapper"))
        recipe, row = CUDA_EXTENSION_RECIPES[index], self.rows[index]
        _require(type(module) is ModuleType and vars(module).get("__name__") == recipe.name
                 and live["publication"] == "direct"
                 and vars(module).get("__spec__") is spec and vars(module).get("__loader__") is wrapper
                 and spec.loader is wrapper and wrapper.delegate is delegate and wrapper.compiler is self
                 and spec.name == recipe.name and spec.origin == row["artifact"]["path"]
                 and delegate.name == recipe.name and delegate.path == spec.origin
                 and vars(module).get("__file__") == spec.origin and recipe.name not in sys.modules,
                 "direct extension identity/metadata/publication drift")
        for name, entry in zip(recipe.entries, live["entries"]):
            _require(vars(module).get(name) is entry, "compiled entry rebound")

    def failed(self, error):
        self.failed_flag = True
        if not any(old is error for old in self.failure_objects):
            self.failure_objects.append(error)
            self._retain(error)
            if self.index is not None:
                self._step("failed", error)

    def guard_completed(self, index):
        _require(index == self.index == self.completed and self.controller._guard_events[-1]["operation"] == "normal_disarmed",
                 "compiler guard did not complete")
        self._step("guard_completed")
        self.completed += 1

    def payload(self):
        self.root.audit_retention()
        self.observation.freeze_adoptions()
        observation = self.observation
        _require(len(observation.journal) == len(observation.issued), "compiler journal issuance interrupted")
        journal = [json.loads(encoded) for encoded in observation.issued]
        _require(journal == observation.journal, "compiler issued journal changed")
        upstream = self.root.input_payload
        j = len(upstream["journal"])
        _require(journal[:j] == upstream["journal"], "compiler rewrote input journal")
        from .cuda_compilation_evidence import _references

        tokens = set().union(*(_references(e["details"]) for e in journal[j:])) if len(journal) > j else set()
        retained = {token: event for token, owner, event in self.root.issued}
        objects = []
        for token in sorted(tokens):
            old = self.root.input_objects.get(token)
            objects.append({"token": token, "origin": "input" if old is not None else "continuation",
                            "first_journal_index": observation.first[token] if old is None else old["first_journal_index"],
                            "guardian_sequence": observation.adoptions.get(token) if old is None else old["guardian_sequence"],
                            "retention_event_index": None if old is not None else retained.get(token)})
        lifecycle = self.controller.evidence().to_payload()
        return {"schema_version": CUDA_COMPILATION_VERSION, "session": self.controller._guard_client.configuration.session,
                "input_evidence_sha256": self.root.input_record.sha256, "source_sha256": self.session.snapshot.sha256,
                "environment_sha256": observation.environment.sha256, "toolchain_sha256": self.toolchain.sha256,
                "target_ordinal": self.controller.target_device_ordinal,
                "observation_cut": {"input_journal_length": j, "input_lifecycle_event_count": len(upstream["lifecycle"]["transition_trace"]),
                                    "input_source_trace_length": len(self.root.input_trace), "journal_length": len(journal),
                                    "lifecycle_event_count": len(lifecycle["transition_trace"]), "source_trace_length": len(self.session.trace)},
                "recipes": self.rows, "journal": journal[j:], "objects": objects,
                "components": self.session._runtime_component_projection(), "lifecycle": lifecycle,
                "input_source_trace": [list(row) for row in self.root.input_trace], "source_trace": [list(row) for row in self.session.trace],
                "summary": {"completed_recipes": self.completed, "failed": self.failed_flag or self.session.failed or self.session.external.failed,
                            "boundary": "compiled_cuda" if self.completed == 3 and not self.failed_flag else "incomplete"}}

    def checkpoint(self):
        record = CudaCompilationRecord.from_payload(self.payload(), input_record=self.root.input_record,
                    input_source_trace=[list(row) for row in self.root.input_trace],
                    environment=self.observation.environment, trusted_launch_inputs=self.observation.inputs,
                    toolchain_sha256=self.toolchain.sha256)
        self.last_validated = record
        return record

    def final(self):
        if self.final_selection is not None:
            return self.final_selection
        record, selection = self.last_validated, "prior_validated_cut"
        diagnostic = False
        try:
            record, selection = self.checkpoint(), "current"
        except BaseException as error:
            self.failure_objects.append(error)
            diagnostic = True
        if record is None:
            selection = "unavailable"
        self.final_selection = record, {
            "schema_version": CUDA_COMPILATION_SELECTION_VERSION, "selection": selection,
            "record_sha256": None if record is None else record.sha256,
            "selected_lifecycle_event_count": None if record is None else record.to_payload()["observation_cut"]["lifecycle_event_count"],
            "observed_lifecycle_event_count": len(self.controller._guard_events), "failure_diagnostic_present": diagnostic,
        }
        return self.final_selection

    def compiled_owners(self):
        _require(self.completed == 3 and not self.failed_flag, "compiled owners incomplete")
        return tuple(_require_compiled_owner(owner, self.session) for owner in self.owners)


def _create_compilation(session, root, runtime_record, module, target, stream):
    _require(session not in _COMPILATIONS, "compiler creation is single-use")
    compiler = _Compilation(session, root, runtime_record, module, target, stream)
    root.collector = compiler
    _COMPILATIONS[session] = compiler
    session.runtime_observation.token(compiler)
    compiler.initialize()
    return compiler
