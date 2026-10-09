"""One private pinned-dependency initialization; importing this module is inert."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources.abc
import io
import os
import sys
import threading
from email.parser import BytesParser
from pathlib import Path
from types import CodeType, FunctionType

from .compilation_contract import setuptools_event_fields, setuptools_role_anchor
from .runtime_environment import _contained, _hash_file


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _code_children(code):
    yield code
    for value in code.co_consts:
        if type(value) is CodeType:
            yield from _code_children(value)


class _ReadOnlyResource(io.BufferedIOBase):
    """Immutable bytes plus a cursor; no writable buffer or filesystem handle."""

    def __init__(self, owner, data, index):
        self._owner, self._data, self._index, self._position = owner, data, index, 0

    def _check(self):
        try:
            _require(not self.closed, "read of closed resource")
            self._owner.check_resource_handle(self)
        except BaseException as error:
            self._owner.fail(error)
            raise

    def _require(self, condition, message):
        if not condition:
            error = ValueError(message)
            self._owner.fail(error)
            raise error

    def readable(self):
        return not self.closed

    def seekable(self):
        return not self.closed

    def writable(self):
        return False

    def read(self, size=-1):
        self._check()
        self._require(type(size) is int, "invalid resource read size")
        end = len(self._data) if size < 0 else min(len(self._data), self._position + size)
        value = self._data[self._position:end]
        self._position = max(self._position, end)
        return value

    read1 = read

    def readinto(self, buffer):
        try:
            view = memoryview(buffer).cast("B")
            self._require(not view.readonly, "resource readinto needs a writable buffer")
            value = self.read(len(view))
            view[:len(value)] = value
            return len(value)
        except BaseException as error:
            self._owner.fail(error)
            raise

    def tell(self):
        self._check()
        return self._position

    def seek(self, offset, whence=0):
        self._check()
        self._require(type(offset) is int and type(whence) is int and whence in (0, 1, 2), "invalid resource seek")
        position = (0, self._position, len(self._data))[whence] + offset
        self._require(position >= 0, "negative resource position")
        self._position = position
        return position


class _ResourceReader(importlib.resources.abc.ResourceReader):
    def __init__(self, owner):
        self.owner = owner

    def open_resource(self, name):
        return self.owner.open_resource(self, name)

    def resource_path(self, name):
        error = ValueError("setuptools resource cannot export a filesystem path")
        self.owner.fail(error)
        raise error

    def is_resource(self, name):
        self.owner.check_resource_reader(self)
        return name == "Lorem ipsum.txt"

    def contents(self):
        self.owner.check_resource_reader(self)
        return iter(("Lorem ipsum.txt",))


class _Distribution(importlib.metadata.Distribution):
    def __init__(self, owner):
        self.owner = owner

    def read_text(self, filename):
        return self.owner.read_metadata(self, filename)

    def locate_file(self, path):
        error = ValueError("setuptools metadata cannot export a filesystem path")
        self.owner.fail(error)
        raise error


class _AliasLoader:
    def __init__(self, owner, delegate):
        self.owner, self.delegate = owner, delegate

    def create_module(self, spec):
        return self.owner.alias_create(self, spec)

    def exec_module(self, module):
        self.owner.alias_exec(self, module)


class _SetuptoolsInitialization:
    """Issued/retained by SourceSession before this constructor can fail."""

    def __init__(self, session, run_root):
        self.session = session
        self.root, self.admission, self.controller, self.thread = session.setuptools_binding(self)
        self.observation = session.runtime_observation
        self.state, self.errors = "NEW", []
        self.vendor_active = False
        self.helpers, self.installed, self.restore_attempted = {}, [], set()
        self.originals, self.wrappers, self.hack = {}, {}, None
        self.finder = self.producer = self.alias_spec = self.delegate = self.alias_wrapper = None
        self.source = self.source_spec = self.source_loader = None
        self.alias_methods = self.hack_methods = None
        self.function_bindings = []
        self.reader = self.distribution = self.resource_bytes = self.metadata_bytes = None
        self.handles, self.metadata_reads = [], 0
        self.run_root = Path(run_root)
        cwd = self.run_root.stat()
        self.cwd_identity = cwd.st_dev, cwd.st_ino
        self.original_paths = tuple(sys.path)
        self.path_object = sys.path
        self.descriptor = self.admission.environment.setuptools_initialization
        _require(self.descriptor is not None, "setuptools deployment was not frozen")
        self.vendor_path = self.admission.roots[self.descriptor["vendor_root"]]
        self.active_paths = self.original_paths + (self.vendor_path,)
        self.token = self.observation.token(self)

    def _retain(self, value):
        return self.observation.token(value)

    def _emit(self, action, **fields):
        details = {"action": action, "initialization_token": self.token, **fields}
        _require(set(details) == setuptools_event_fields(action), "setuptools observation fields")
        self.observation.append("setuptools_initialization", details)

    def _helper(self, action, field, value, index=None, *, existing=False):
        token = self._retain(value)
        role, operation = setuptools_role_anchor(action, field, index)
        previous = self.helpers.get(operation)
        if previous is not None:
            _require(previous[0] == token and previous[1] is value, "setuptools helper identity changed")
            adoption = self.controller.guardian.records[self.observation.adoptions[token]]
            _require(adoption.role == role and adoption.operation_id == operation,
                     "setuptools helper adoption anchor changed")
            return token
        _require(not existing, "setuptools helper lacks its original adoption")
        _require(value is not self and all(held[1] is not value for held in self.helpers.values()),
                 "setuptools helper roles cannot share an object")
        _require(not self.controller.guardian.contains_exact_owner(value), "setuptools helper was already adopted")
        self.controller.adopt_host_owner(value, role=role, operation_id=operation)
        self.observation.freeze_adoptions()
        self.helpers[operation] = token, value
        return token

    def check(self, *, first_guard=True):
        binding = self.session.setuptools_binding(self)
        _require(all(actual is expected for actual, expected in zip(binding[:3],
                    (self.root, self.admission, self.controller))) and binding[3] == self.thread,
                 "setuptools issued binding changed")
        _require(threading.get_ident() == self.thread and self.state != "FAILED", "setuptools failed or wrong thread")
        self.session.check_live()
        if first_guard:
            self.controller._require_setup()
            _require(self.controller._guard_reference == {"sequence": 6, "role": "setup"}, "setuptools initialization guard")
        _require(os.environ.get("SETUPTOOLS_USE_DISTUTILS") == "local", "setuptools selection changed")
        current = self.run_root.stat()
        _require(Path.cwd() == self.run_root and self.cwd_identity == (current.st_dev, current.st_ino)
                 and not os.path.lexists(self.run_root / "pybuilddir.txt"), "setuptools cwd/CPython-build branch changed")
        _require(sys.path is self.path_object and tuple(sys.path) == (self.active_paths if self.vendor_active else self.original_paths),
                 "setuptools path transition drift")

    def context(self, name):
        self.check()
        record = self.admission.records.get(name)
        _require(self.admission.stack and self.admission.stack[-1] == name and record is not None
                 and record["state"] == "executing" and sys.modules.get(name) is record["module"],
                 "setuptools helper outside its exact source execution")
        return record["module"]

    def check_import(self):
        self.check(first_guard=self.state != "COMPLETE")
        self.controller._require_setup()
        _require(self.state not in {"NEW", "FAILED"}
                 and self.controller._guard_reference in ({"sequence": i, "role": "setup"} for i in (6, 7, 8)),
                 "setuptools dependency import outside compilation")

    def alias_path(self):
        self.check_import()
        _require(self.state in {"ALIAS_EXECUTED", "ALIAS_PUBLISHED", "SHIM_CLOSED", "RESTORED", "COMPLETE"}
                 and sys.modules.get("distutils") is self.source, "setuptools alias parent is not published")
        self.check_source_metadata()
        return (str(Path(self.source_loader.origin).parent),)

    def begin(self):
        try:
            self.check()
            _require(self.state == "NEW" and self.vendor_path not in sys.path, "setuptools initialization repeated")
            _require(not any(name.split(".")[0] in {"setuptools", "_distutils_hack", "distutils", "backports"}
                             for name in sys.modules.keys() | self.admission.records.keys()), "setuptools dependency preloaded")
            self.session.activate_setuptools_vendor(self)
            self.state = "ENTERED"
            self._emit("begun", vendor_root=self.descriptor["vendor_root"], original_paths=list(self.original_paths),
                       active_paths=list(self.active_paths))
        except BaseException as error:
            self.fail(error)
            raise

    def fail(self, error):
        first = self.state != "FAILED"
        self.state = "FAILED"
        if not any(original is error for original in self.errors):
            self.errors.append(error)
        try:
            token = self._retain(error)
            if first:
                self._emit("failed", exception_token=token)
        except BaseException as diagnostic:
            self.errors.append(diagnostic)
        if not self.session.failed:
            self.session.fail(error)

    def _function(self, function, code, globals_):
        _require(type(function) is FunctionType and function.__globals__ is globals_
                 and any(function.__code__ is child for child in _code_children(code)), "setuptools function code/globals drift")
        self._remember_function(function)

    def _remember_function(self, function):
        closure = function.__closure__
        self.function_bindings.append((function, function.__code__, function.__globals__, closure,
                                       () if closure is None else tuple(cell.cell_contents for cell in closure)))

    def _check_functions(self):
        for function, code, globals_, closure, contents in self.function_bindings:
            _require(function.__code__ is code and function.__globals__ is globals_
                     and function.__closure__ is closure
                     and (closure is None or all(cell.cell_contents is value for cell, value in zip(closure, contents))),
                     "setuptools function code/globals/closure changed")

    def after_file_complete(self, name, module, loader):
        try:
            if name == "_distutils_hack":
                self.check()
                _require(self.state == "ENTERED", "setuptools hook install order")
                self.hack = module
                raw = vars(module)
                for name_ in ("insert_shim", "_remove_shim"):
                    function = raw.get(name_)
                    self._retain(function)
                    self._function(function, loader.compiled_code, raw)
                    self.originals[name_] = function
                self.finder = raw.get("DISTUTILS_FINDER")
                self._retain(self.finder)
                cls = raw.get("DistutilsMetaFinder")
                _require(type(cls) is type and type(self.finder) is cls and not vars(self.finder), "setuptools finder identity")
                methods = vars(cls)
                self.producer = methods["spec_for_distutils"]
                is_cpython = methods["is_cpython"].__func__
                for function in (self.producer, is_cpython):
                    self._retain(function)
                    self._function(function, loader.compiled_code, raw)
                self.hack_methods = cls, self.producer, is_cpython

                def insert_shim():
                    self.enter_shim()

                def remove_shim():
                    self.leave_shim()

                self.wrappers = {"insert_shim": insert_shim, "_remove_shim": remove_shim}
                for function in self.wrappers.values():
                    self._retain(function)
                    self._remember_function(function)
                fields = {"hack_module_token": self._retain(module),
                          "finder_token": self._helper("hooked", "finder_token", self.finder),
                          "producer_token": self._helper("hooked", "producer_token", self.producer)}
                for name_, stem in (("insert_shim", "insert"), ("_remove_shim", "remove")):
                    fields[stem + "_original_token"] = self._helper("hooked", stem + "_original_token", self.originals[name_])
                    fields[stem + "_wrapper_token"] = self._helper("hooked", stem + "_wrapper_token", self.wrappers[name_])
                    self.installed.append(name_)  # Also own a possibly partial assignment.
                    setattr(module, name_, self.wrappers[name_])
                self.state = "HOOKED"
                self._emit("hooked", **fields)
        except BaseException as error:
            self.fail(error)
            raise

    def finish_file(self, name, failure):
        # This is the sole restoration owner, including partial hook installation.
        if name != "_distutils_hack.override" and not (name == "_distutils_hack" and failure is not None):
            return
        if failure is not None:
            self.fail(failure)
        try:
            self.restore()
        except BaseException as error:
            self.fail(error)
            if failure is None:
                raise

    def check_hooks(self, *, restored=False):
        self._check_functions()
        self._helper("hooked", "finder_token", self.finder, existing=True)
        self._helper("hooked", "producer_token", self.producer, existing=True)
        for name, stem in (("insert_shim", "insert"), ("_remove_shim", "remove")):
            self._helper("hooked", stem + "_original_token", self.originals[name], existing=True)
            self._helper("hooked", stem + "_wrapper_token", self.wrappers[name], existing=True)
        raw = vars(self.hack)
        cls, producer, is_cpython = self.hack_methods
        _require(raw.get("DISTUTILS_FINDER") is self.finder and raw.get("DistutilsMetaFinder") is cls
                 and type(self.finder) is cls and not vars(self.finder)
                 and vars(cls).get("spec_for_distutils") is producer
                 and vars(cls).get("is_cpython").__func__ is is_cpython, "setuptools spec producer changed")
        expected = self.originals if restored else self.wrappers
        _require(all(raw.get(name) is value for name, value in expected.items()), "setuptools shim binding changed")

    def enter_shim(self):
        try:
            module = self.context("_distutils_hack.override")
            _require(self.state == "HOOKED", "setuptools shim entry order")
            self.check_hooks()
            self.state = "SHIM_ACTIVE"
            self._emit("shim_entered", override_module_token=self._retain(module))
        except BaseException as error:
            self.fail(error)
            raise

    def alias_specification(self, path, target):
        try:
            self.context("_distutils_hack.override")
            self.check_hooks()
            _require(self.state == "SHIM_ACTIVE" and self.alias_spec is None and path is None and target is None,
                     "setuptools alias resolution order")
            spec = self.producer(self.finder)
            self._retain(spec)
            self.check()
            self.source = self.admission.records["setuptools._distutils"]["module"]
            self.source_spec = self.admission.specs["setuptools._distutils"]
            self.source_loader = self.admission.loaders["setuptools._distutils"]
            self.admission.check_module("setuptools._distutils")
            _require(spec is not None and spec.name == "distutils" and spec.origin == self.source_loader.origin
                     and spec.submodule_search_locations is None, "setuptools alias spec metadata")
            delegate = spec.loader
            self._retain(delegate)
            methods = vars(type(delegate))
            create, execute = methods["create_module"], methods["exec_module"]
            for function in (create, execute):
                self._retain(function)
                self._function(function, self.producer.__code__, vars(self.hack))
            _require(create.__code__.co_freevars == ("mod",) and len(create.__closure__) == 1
                     and create.__closure__[0].cell_contents is self.source and execute.__closure__ is None,
                     "setuptools alias loader closure")
            self.alias_methods = create, execute
            wrapper = _AliasLoader(self, delegate)
            self._retain(wrapper)
            self.alias_spec, self.delegate, self.alias_wrapper = spec, delegate, wrapper
            spec.loader = wrapper
            fields = self.alias_bindings()
            fields.update(source_spec_token=self._retain(self.source_spec),
                          source_loader_token=self._retain(self.source_loader.delegate))
            self._emit("alias_spec", **fields)
            return spec
        except BaseException as error:
            self.fail(error)
            raise

    def alias_bindings(self, *, existing=False):
        return {"source_module_token": self._retain(self.source),
                "alias_spec_token": self._helper("alias_spec", "alias_spec_token", self.alias_spec, existing=existing),
                "delegate_token": self._helper("alias_spec", "delegate_token", self.delegate, existing=existing),
                "wrapper_token": self._helper("alias_spec", "wrapper_token", self.alias_wrapper, existing=existing)}

    def check_alias_methods(self):
        self._check_functions()
        self.alias_bindings(existing=True)
        _require(type(self.alias_wrapper) is _AliasLoader and self.alias_wrapper.owner is self
                 and self.alias_wrapper.delegate is self.delegate
                 and vars(type(self.delegate)).get("create_module") is self.alias_methods[0]
                 and vars(type(self.delegate)).get("exec_module") is self.alias_methods[1], "setuptools alias method changed")

    def alias_create(self, wrapper, spec):
        try:
            self.check()
            _require(self.state == "SHIM_ACTIVE" and wrapper is self.alias_wrapper and spec is self.alias_spec,
                     "setuptools alias create order")
            self.check_alias_methods()
            result = self.alias_methods[0](self.delegate, spec)
            self._retain(result)
            _require(result is self.source, "setuptools alias returned a different module")
            self.state = "ALIAS_CREATED"
            self.check_source_metadata(intermediate=True)
            self._emit("alias_created", **self.alias_bindings())
            return result
        except BaseException as error:
            self.fail(error)
            raise

    def check_source_metadata(self, *, intermediate=False):
        raw, spec = vars(self.source), self.alias_spec
        _require(sys.modules.get("setuptools._distutils") is self.source and raw.get("__name__") == "distutils"
                 and raw.get("__spec__") is (self.source_spec if intermediate else spec)
                 and raw.get("__loader__") is self.source_loader and raw.get("__package__") == "setuptools._distutils"
                 and raw.get("__file__") == self.source_loader.origin
                 and raw.get("__path__") == [str(Path(self.source_loader.origin).parent)], "setuptools alias source metadata")
        self.source_loader._check_binding()
        _require(self.source_spec.name == "setuptools._distutils" and self.source_spec.loader is self.source_loader
                 and self.source_spec.origin == self.source_loader.origin
                 and self.source_spec.submodule_search_locations == raw["__path__"]
                 and spec.name == "distutils" and spec.parent == "" and spec.loader is self.alias_wrapper
                 and spec.origin == self.source_loader.origin and spec.submodule_search_locations is None,
                 "setuptools original/alias spec changed")

    def alias_exec(self, wrapper, module):
        try:
            self.check()
            _require(self.state == "ALIAS_CREATED" and wrapper is self.alias_wrapper and module is self.source
                     and sys.modules.get("distutils") is module, "setuptools alias exec order/publication")
            self.check_alias_methods()
            self.check_source_metadata()
            self.alias_methods[1](self.delegate, module)
            self.check_source_metadata()
            self.state = "ALIAS_EXECUTED"
            self._emit("alias_executed", **self.alias_bindings())
        except BaseException as error:
            self.fail(error)
            raise

    def leave_shim(self):
        if self.state == "FAILED":
            self.session.retain_import_boundary()
            if "distutils" in sys.modules:
                self._retain(sys.modules["distutils"])
            return
        try:
            self.context("_distutils_hack.override")
            _require(self.state == "ALIAS_EXECUTED" and sys.modules.get("distutils") is self.source,
                     "setuptools alias publication order")
            self.check_source_metadata()
            self.check_hooks()
            self.state = "ALIAS_PUBLISHED"
            self.admission.records["distutils"] = {"kind": "python_package_alias", "state": "completed", "module": self.source}
            self._emit("alias_published", **self.alias_bindings())
            self.state = "SHIM_CLOSED"
            self._emit("shim_left")
        except BaseException as error:
            self.fail(error)
            raise

    def restore(self):
        errors = []
        for name in self.installed:
            if name in self.restore_attempted:
                continue
            self.restore_attempted.add(name)
            try:
                _require(vars(self.hack).get(name) is self.wrappers[name], "setuptools hook changed before restoration")
            except BaseException as error:
                errors.append(error)
            try:
                setattr(self.hack, name, self.originals[name])
                _require(vars(self.hack).get(name) is self.originals[name], "setuptools hook restoration failed")
            except BaseException as error:
                errors.append(error)
        try:
            self.session.check_import_topology()
            if self.hack_methods is not None:
                self.check_hooks(restored=True)
        except BaseException as error:
            errors.append(error)
        for error in errors:
            self.fail(error)
        if self.state not in {"FAILED", "RESTORED", "COMPLETE"}:
            _require(self.state == "SHIM_CLOSED", "setuptools restoration without closed shim")
            self.state = "RESTORED"
            self._emit("hooks_restored")
        if errors:
            raise errors[0]

    def _file(self, descriptor, kind):
        row = self.admission.files[(descriptor["root"], descriptor["path"])]
        _require(row["kind"] == kind, "setuptools file kind")
        path = _contained(self.admission.roots[descriptor["root"]], descriptor["path"])
        data = path.read_bytes()
        self._retain(data)
        _require(len(data) == row["size"] and hashlib.sha256(data).hexdigest() == row["sha256"], "setuptools file bytes changed")
        return data, row

    def resource_reader(self, name, loader):
        try:
            self.context("jaraco.text")
            _require(name == "jaraco.text" and self.admission.loaders[name] is loader, "setuptools resource loader")
            if self.reader is None:
                self.reader = _ResourceReader(self)
                self._helper("resource_read", "reader_token", self.reader, 0)
            return self.reader
        except BaseException as error:
            self.fail(error)
            raise

    def check_resource_reader(self, reader):
        try:
            self.context("jaraco.text")
            _require(reader is self.reader and type(reader) is _ResourceReader and reader.owner is self, "setuptools reader identity")
        except BaseException as error:
            self.fail(error)
            raise

    def open_resource(self, reader, name):
        try:
            self.check_resource_reader(reader)
            _require(type(name) is str and name == "Lorem ipsum.txt", "unlisted setuptools resource")
            data, row = self._file(self.descriptor["resource"], "data")
            index = len(self.handles)
            if self.resource_bytes is None:
                self.resource_bytes = data
            _require(data == self.resource_bytes, "setuptools resource changed after capture")
            handle = _ReadOnlyResource.__new__(_ReadOnlyResource)
            self._retain(handle)
            _ReadOnlyResource.__init__(handle, self, self.resource_bytes, index)
            self.handles.append(handle)
            self._emit("resource_read", read_index=index, package_token=self._retain(self.context("jaraco.text")),
                       reader_token=self._helper("resource_read", "reader_token", reader, index),
                       bytes_token=self._helper("resource_read", "bytes_token", self.resource_bytes, index),
                       handle_token=self._helper("resource_read", "handle_token", handle, index),
                       file_root=row["root"], file_path=row["path"], sha256=row["sha256"])
            return handle
        except BaseException as error:
            self.fail(error)
            raise

    def check_resource_handle(self, handle):
        try:
            self.context("jaraco.text")
            _require(type(handle) is _ReadOnlyResource and type(handle._index) is int
                     and 0 <= handle._index < len(self.handles) and self.handles[handle._index] is handle
                     and handle._owner is self and handle._data is self.resource_bytes, "setuptools resource handle identity")
        except BaseException as error:
            self.fail(error)
            raise

    def find_distributions(self, context):
        try:
            self.context("setuptools.version")
            _require(type(context) is importlib.metadata.DistributionFinder.Context
                     and vars(context) == {"name": "setuptools"} and tuple(context.path) == self.active_paths,
                     "setuptools metadata query context")
            if self.distribution is None:
                self.distribution = _Distribution(self)
                self._helper("metadata_read", "distribution_token", self.distribution, 0)
            return (self.distribution,)
        except BaseException as error:
            self.fail(error)
            raise

    def read_metadata(self, distribution, filename):
        try:
            self.context("setuptools.version")
            _require(distribution is self.distribution and type(distribution) is _Distribution
                     and distribution.owner is self, "setuptools metadata owner identity")
            descriptor = self.descriptor["metadata"]
            if filename == "METADATA":
                path = _contained(self.admission.roots[descriptor["root"]], descriptor["path"])
                _require(not os.path.lexists(path.parent / "METADATA"), "unexpected setuptools METADATA file")
                return None
            _require(filename == "PKG-INFO", "unlisted setuptools metadata file")
            data, row = self._file(descriptor, "metadata")
            if self.metadata_bytes is None:
                self.metadata_bytes = data
            _require(data == self.metadata_bytes, "setuptools metadata changed after capture")
            headers = BytesParser().parsebytes(data, headersonly=True)
            _require(headers.get_all("Name") == ["setuptools"] and headers.get_all("Version") == ["83.0.0"],
                     "setuptools distribution identity changed")
            index = self.metadata_reads
            self._emit("metadata_read", read_index=index,
                       distribution_token=self._helper("metadata_read", "distribution_token", distribution, index),
                       bytes_token=self._helper("metadata_read", "bytes_token", self.metadata_bytes, index),
                       file_root=row["root"], file_path=row["path"], sha256=row["sha256"], version=headers["Version"])
            self.metadata_reads += 1
            return self.metadata_bytes.decode("utf-8")
        except BaseException as error:
            self.fail(error)
            raise

    def complete(self, dependency):
        try:
            self.check()
            _require(self.state == "RESTORED" and self.handles and self.metadata_reads, "setuptools initialization incomplete")
            self.admission.audit()
            self.check_hooks(restored=True)
            self.check_source_metadata()
            fields = {}
            for field, name in (("setuptools_module_token", "setuptools"),
                                ("build_ext_module_token", "setuptools.command.build_ext"),
                                ("cpp_extension_module_token", "torch.utils.cpp_extension")):
                record = self.admission.records[name]
                _require(record["state"] == "completed", "setuptools completion before actual source return")
                self.admission.check_module(name)
                fields[field] = self._retain(record["module"])
            _require(self.admission.records["torch.utils.cpp_extension"]["module"] is dependency
                     and vars(self.admission.records["setuptools"]["module"]).get("__version__") == "83.0.0"
                     and vars(self.admission.records["setuptools.version"]["module"]).get("__version__") == "83.0.0",
                     "setuptools version or compiler dependency changed")
            self.state = "COMPLETE"
            self._emit("completed", **fields)
        except BaseException as error:
            self.fail(error)
            raise

    def audit_completed(self):
        try:
            self.check(first_guard=False)
            _require(self.state == "COMPLETE" and self.handles and self.metadata_reads,
                     "setuptools initialization is not complete")
            self.check_hooks(restored=True)
            self.check_alias_methods()
            self.check_source_metadata()
            _require(sys.modules.get("distutils") is self.source
                     and self.restore_attempted == {"insert_shim", "_remove_shim"},
                     "setuptools alias/restoration changed")
            _require(type(self.reader) is _ResourceReader and self.reader.owner is self
                     and type(self.distribution) is _Distribution and self.distribution.owner is self,
                     "setuptools retained resource/metadata owner changed")
            self._helper("resource_read", "reader_token", self.reader, 0, existing=True)
            self._helper("resource_read", "bytes_token", self.resource_bytes, 0, existing=True)
            self._helper("metadata_read", "distribution_token", self.distribution, 0, existing=True)
            self._helper("metadata_read", "bytes_token", self.metadata_bytes, 0, existing=True)
            for index, handle in enumerate(self.handles):
                _require(type(handle) is _ReadOnlyResource and handle._owner is self and handle._index == index
                         and handle._data is self.resource_bytes, "setuptools retained resource handle changed")
                self._helper("resource_read", "handle_token", handle, index, existing=True)
            for key, kind, held in (("resource", "data", self.resource_bytes), ("metadata", "metadata", self.metadata_bytes)):
                descriptor = self.descriptor[key]
                row = self.admission.files[descriptor["root"], descriptor["path"]]
                path = _contained(self.admission.roots[descriptor["root"]], descriptor["path"])
                expected = row["sha256"], row["size"]
                _require(row["kind"] == kind and _hash_file(path) == expected
                         and (hashlib.sha256(held).hexdigest(), len(held)) == expected,
                         "setuptools retained bytes changed")
            for token, value in self.helpers.values():
                observed = self.observation.owners.get(id(value))
                _require(observed is not None and observed[0] == token and observed[1] is value,
                         "setuptools helper token changed")
                self.controller.require_adopted(value)
        except BaseException as error:
            self.fail(error)
            raise
