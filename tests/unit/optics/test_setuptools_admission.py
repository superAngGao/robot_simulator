"""Production helpers with a stdlib-only outer session, not dependency integration.

The real guardian, token registry and compilation retention run here. Source
records/outer custody are descriptive stand-ins; no setuptools/Torch is imported,
and these tests cannot issue or prove a complete initialization transaction.
"""

import hashlib
import importlib.machinery
import importlib.metadata
import importlib.resources
import io
import json
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from optical_worker import native_bootstrap as bootstrap
from optical_worker import setuptools_admission as admission_helpers
from optical_worker.compilation_contract import setuptools_initialization_descriptor
from optical_worker.guarded_lifecycle import NativeGuardedSetupLifecycle
from optical_worker.setuptools_admission import _SetuptoolsInitialization
from tests.support.guarded_peer import Peer


class _OuterSession:
    def __init__(self, controller, admission):
        self.controller, self.admission = controller, admission
        self.failed, self.failures, self.trace = False, [], []
        self.runtime_observation = bootstrap._RuntimeObservation(self)

    def bound_controller(self):
        return self.controller

    def setuptools_binding(self, owner):
        return self.root, self.admission, self.controller, threading.get_ident()

    def check_live(self):
        if self.failed:
            raise ValueError("outer session failed")

    def check_import_topology(self):
        pass  # This helper fixture does not install the production finder.

    def fail(self, error):
        self.failed = True
        self.failures.append(error)


@pytest.fixture
def initialization(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SETUPTOOLS_USE_DISTUTILS", "local")
    peer = Peer()
    controller = NativeGuardedSetupLifecycle(lane_id="cuda", channel=peer.channel, configuration=peer.config)
    descriptor = setuptools_initialization_descriptor()
    admission = SimpleNamespace(environment=SimpleNamespace(setuptools_initialization=descriptor),
                                roots={"site": str(tmp_path), "setuptools_vendor": str(tmp_path / "vendor")},
                                files={}, loaders={}, records={}, stack=[])
    for key, kind, content in (("resource", "data", "Lorem ipsum.\nΔοκιμή\n".encode()),
                               ("metadata", "metadata", b"Name: setuptools\nVersion: 83.0.0\n\n")):
        row = descriptor[key]
        path = Path(admission.roots[row["root"]]) / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        admission.files[row["root"], row["path"]] = {
            "root": row["root"], "path": row["path"], "kind": kind,
            "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    session = _OuterSession(controller, admission)
    try:
        for _ in range(6):
            with controller.segment():
                pass
        with controller.segment():
            encoded = b'{"objects":[]}'
            input_record = SimpleNamespace(encoded=encoded, to_payload=lambda: json.loads(encoded))
            session.root = bootstrap._CompileCustodyRoot(session, input_record, (), {})
            controller.adopt_host_owner(session.root, role="cuda_compile_custody", operation_id="cuda_compile_custody")
            bootstrap._CUDA_COMPILATIONS[id(session)] = session, session.root
            observation = session.runtime_observation
            session.root.root_token = observation.token(session.root)
            owner = object.__new__(_SetuptoolsInitialization)
            observation.token(owner)
            _SetuptoolsInitialization.__init__(owner, session, tmp_path)
            controller.adopt_host_owner(owner, role="setuptools_initialization", operation_id="setuptools_initialization")
            observation.freeze_adoptions()
            owner.state = "ENTERED"  # Descriptive entry, not a topology activation.
            yield owner
            session.root.audit_retention()
    finally:
        bootstrap._CUDA_COMPILATIONS.pop(id(session), None)
        peer.close()


def _executing(owner, name, monkeypatch):
    module = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    owner.admission.records[name] = {"module": module, "state": "executing"}
    owner.admission.stack[:] = [name]
    return module


def test_actual_resource_adapter_reads_utf8_and_distinct_retained_handles(initialization, monkeypatch):
    owner = initialization
    package = _executing(owner, "jaraco.text", monkeypatch)
    loader = SimpleNamespace(get_resource_reader=lambda name: owner.resource_reader(name, loader))
    owner.admission.loaders["jaraco.text"] = loader
    package.__spec__ = importlib.machinery.ModuleSpec("jaraco.text", loader, is_package=True)
    resource = importlib.resources.files(package).joinpath("Lorem ipsum.txt")
    assert resource.read_text(encoding="utf-8") == "Lorem ipsum.\nΔοκιμή\n"
    assert resource.read_bytes() == owner.resource_bytes
    assert len(owner.handles) == 2 and owner.handles[0] is not owner.handles[1]
    assert all(handle.closed for handle in owner.handles)
    events = [e for e in owner.observation.journal if e["kind"] == "setuptools_initialization"]
    assert [e["details"]["read_index"] for e in events] == [0, 1]
    assert events[0]["details"]["bytes_token"] == events[1]["details"]["bytes_token"]
    assert events[0]["details"]["handle_token"] != events[1]["details"]["handle_token"]
    assert all(owner.controller.guardian.contains_exact_owner(handle) for handle in owner.handles)


@pytest.mark.parametrize("operation", ["traversal", "path", "changed_file", "wrong_context", "closed_read"])
def test_resource_denials_poison_even_if_caught(initialization, monkeypatch, operation):
    owner = initialization
    _executing(owner, "jaraco.text", monkeypatch)
    loader = object()
    owner.admission.loaders["jaraco.text"] = loader
    reader = owner.resource_reader("jaraco.text", loader)
    handle = reader.open_resource("Lorem ipsum.txt")
    if operation == "changed_file":
        descriptor = owner.descriptor["resource"]
        (Path(owner.admission.roots[descriptor["root"]]) / descriptor["path"]).write_bytes(b"changed")
    if operation == "wrong_context":
        owner.admission.stack[:] = ["other"]
    if operation == "closed_read":
        handle.close()
    with pytest.raises(ValueError):
        if operation == "traversal":
            reader.open_resource("../Lorem ipsum.txt")
        elif operation == "path":
            reader.resource_path("Lorem ipsum.txt")
        elif operation == "changed_file":
            reader.open_resource("Lorem ipsum.txt")
        else:
            handle.read()
    assert owner.state == "FAILED" and owner.session.failed and owner.errors
    assert owner.controller.guardian.contains_exact_owner(handle)
    with pytest.raises(ValueError):
        reader.open_resource("Lorem ipsum.txt")
    handle.close()


def test_readonly_resource_cursor_and_binary_readinto(initialization, monkeypatch):
    owner = initialization
    _executing(owner, "jaraco.text", monkeypatch)
    loader = object()
    owner.admission.loaders["jaraco.text"] = loader
    with owner.resource_reader("jaraco.text", loader).open_resource("Lorem ipsum.txt") as handle:
        assert handle.read(5) == b"Lorem"
        assert handle.tell() == 5
        assert handle.seek(-5, 1) == 0
        buffer = bytearray(6)
        assert handle.readinto(buffer) == 6 and buffer == b"Lorem "
        assert handle.seek(1000) == 1000 and handle.read() == b"" and handle.tell() == 1000
        assert not handle.writable()
        with pytest.raises(io.UnsupportedOperation):
            handle.write(b"x")


def test_named_metadata_uses_stdlib_parser_and_reuses_exact_bytes(initialization, monkeypatch):
    owner = initialization
    _executing(owner, "setuptools.version", monkeypatch)
    # Only the observed query path is set up here; no production topology claim.
    owner.active_paths = tuple(sys.path)
    distribution, = owner.find_distributions(importlib.metadata.DistributionFinder.Context(name="setuptools"))
    assert distribution.version == "83.0.0" and distribution.metadata["Name"] == "setuptools"
    assert owner.metadata_reads == 2
    events = [e["details"] for e in owner.observation.journal
              if e["kind"] == "setuptools_initialization"]
    assert events[0]["bytes_token"] == events[1]["bytes_token"]
    with pytest.raises(ValueError, match="unlisted"):
        distribution.read_text("entry_points.txt")
    assert owner.state == "FAILED" and owner.session.failed


_HOOK_SOURCE = """
def insert_shim():
    raise AssertionError('the original shim must never run')
def _remove_shim():
    raise AssertionError('the original shim must never run')
class DistutilsMetaFinder:
    def spec_for_distutils(self):
        return None
    @staticmethod
    def is_cpython():
        return False
DISTUTILS_FINDER = DistutilsMetaFinder()
"""


def _hook_module():
    module = ModuleType("_distutils_hack")
    code = compile(_HOOK_SOURCE, "frozen-standin.py", "exec")
    exec(code, vars(module))
    return module, SimpleNamespace(compiled_code=code)


def test_restoration_once_and_outer_finally_are_idempotent(initialization):
    owner = initialization
    module, loader = _hook_module()
    originals = module.insert_shim, module._remove_shim
    owner.after_file_complete("_distutils_hack", module, loader)
    assert owner.state == "HOOKED" and module.insert_shim is not originals[0]
    owner.state = "SHIM_CLOSED"  # This test exercises restoration, not alias execution.
    owner.finish_file("_distutils_hack.override", None)
    owner.finish_file("_distutils_hack.override", None)
    assert owner.state == "RESTORED"
    assert module.insert_shim is originals[0] and module._remove_shim is originals[1]
    assert owner.restore_attempted == {"insert_shim", "_remove_shim"}
    assert sum(e["kind"] == "setuptools_initialization" and e["details"]["action"] == "hooks_restored"
               for e in owner.observation.journal) == 1


@pytest.mark.parametrize("target", ["original", "producer", "wrapper_closure"])
def test_changed_function_identity_facts_fail_restoration(initialization, target):
    owner = initialization
    module, loader = _hook_module()
    owner.after_file_complete("_distutils_hack", module, loader)
    if target == "wrapper_closure":
        owner.wrappers["insert_shim"].__closure__[0].cell_contents = object()
    else:
        function = owner.originals["insert_shim"] if target == "original" else owner.producer
        function.__code__ = (lambda: None).__code__
    with pytest.raises(ValueError, match="code/globals/closure"):
        owner.finish_file("_distutils_hack.override", None)
    assert owner.state == "FAILED" and owner.session.failed
    assert all(vars(module)[name] is function for name, function in owner.originals.items())
    owner.finish_file("_distutils_hack.override", owner.errors[0])
    assert owner.state == "FAILED"


def test_unwind_preserves_first_error_after_wrong_current_hook(initialization):
    owner = initialization
    module, loader = _hook_module()
    owner.after_file_complete("_distutils_hack", module, loader)
    original = RuntimeError("original source failure")
    module.insert_shim = object()
    owner.finish_file("_distutils_hack.override", original)
    assert owner.errors[0] is original and owner.state == "FAILED"
    assert any("changed before restoration" in str(error) for error in owner.errors)
    assert module.insert_shim is owner.originals["insert_shim"]
    assert module._remove_shim is owner.originals["_remove_shim"]


def test_coherent_original_slot_swap_still_mismatches_actual_adoption(initialization):
    owner = initialization
    module, loader = _hook_module()
    owner.after_file_complete("_distutils_hack", module, loader)
    owner.originals["insert_shim"], owner.originals["_remove_shim"] = (
        owner.originals["_remove_shim"], owner.originals["insert_shim"])
    left, right = "setuptools_initialization.insert_original", "setuptools_initialization.remove_original"
    owner.helpers[left], owner.helpers[right] = owner.helpers[right], owner.helpers[left]
    owner.state = "SHIM_CLOSED"
    before = owner.controller.guardian.records
    with pytest.raises(ValueError, match="adoption anchor"):
        owner.finish_file("_distutils_hack.override", None)
    assert owner.state == "FAILED" and owner.controller.guardian.records == before


def test_failed_adoption_is_not_retried_by_restoration_audit(initialization, monkeypatch):
    owner = initialization
    module, loader = _hook_module()
    original = RuntimeError("partial helper adoption")
    adopt = owner.controller.adopt_host_owner
    attempted = []

    def reject(value, *, role, operation_id):
        attempted.append(value)
        if operation_id == "setuptools_initialization.remove_original":
            raise original
        return adopt(value, role=role, operation_id=operation_id)

    monkeypatch.setattr(owner.controller, "adopt_host_owner", reject)
    with pytest.raises(RuntimeError) as raised:
        owner.after_file_complete("_distutils_hack", module, loader)
    assert raised.value is original
    count = len(attempted)
    owner.finish_file("_distutils_hack", original)
    assert len(attempted) == count and owner.errors[0] is original and owner.state == "FAILED"
    assert any(value is owner.originals["_remove_shim"] for value in owner.root.held)
    assert not owner.controller.guardian.contains_exact_owner(owner.originals["_remove_shim"])


def test_recompiled_lookalike_function_is_not_from_held_code_tree(initialization):
    owner = initialization
    module, loader = _hook_module()
    other = {}
    exec(compile(_HOOK_SOURCE, "frozen-standin.py", "exec"), other)
    module.insert_shim.__code__ = other["insert_shim"].__code__
    with pytest.raises(ValueError, match="code/globals drift"):
        owner.after_file_complete("_distutils_hack", module, loader)
    assert owner.state == "FAILED" and owner.installed == []


@pytest.mark.parametrize("after_assignment", [False, True])
def test_partial_installation_restores_each_attempt_and_retains_original_error(initialization, monkeypatch, after_assignment):
    owner = initialization
    module, loader = _hook_module()
    original = RuntimeError("partial second hook installation")
    assignments = []

    def assign(target, name, value):
        assignments.append((name, value))
        if name == "_remove_shim" and value is owner.wrappers[name]:
            if after_assignment:
                setattr(target, name, value)
            raise original
        setattr(target, name, value)

    monkeypatch.setattr(admission_helpers, "setattr", assign, raising=False)
    with pytest.raises(RuntimeError) as raised:
        owner.after_file_complete("_distutils_hack", module, loader)
    assert raised.value is original
    owner.finish_file("_distutils_hack", original)
    owner.finish_file("_distutils_hack", original)
    assert owner.state == "FAILED" and owner.errors[0] is original
    assert len(assignments) == 4
    assert all(vars(module)[name] is value for name, value in owner.originals.items())
    assert len(owner.restore_attempted) == 2


def test_copied_initialization_cannot_reuse_source_session_registration(initialization, monkeypatch):
    owner = initialization
    session = owner.session
    binding = (owner, type(owner), owner.root, owner.admission, owner.controller, owner.thread,
               owner.original_paths, owner.path_object)
    monkeypatch.setitem(bootstrap._SETUPTOOLS_CUSTODY, session, binding)
    monkeypatch.setitem(bootstrap._EXTERNAL_CUSTODY, session,
                        (owner.admission, owner.admission.environment, "fixture", owner.controller))
    assert bootstrap._SourceSession.setuptools_binding(session, owner)[0] is owner.root
    copied = object.__new__(type(owner))
    vars(copied).update(vars(owner))
    with pytest.raises(bootstrap.SourceBootstrapError, match="not issued"):
        bootstrap._SourceSession.setuptools_binding(session, copied)
    monkeypatch.setitem(bootstrap._CUDA_COMPILATIONS, id(session), (session, object()))
    with pytest.raises(bootstrap.SourceBootstrapError, match="custody changed"):
        bootstrap._SourceSession.setuptools_binding(session, owner)
    # The fixture's final retention audit still refers to its original root.
    bootstrap._CUDA_COMPILATIONS[id(session)] = session, owner.root


@pytest.mark.parametrize("changed", ["path_object", "original_paths"])
def test_topology_check_rejects_copied_path_or_changed_interpreter_projection(initialization, monkeypatch, changed):
    owner = initialization
    session = owner.session
    session.finder = object()
    paths = list(owner.original_paths)
    owner.path_object = paths
    machine = SimpleNamespace(meta_path=[session.finder], path=paths, path_hooks=[],
                              stdlib_module_names=frozenset({"fractions"}), builtin_module_names=())
    monkeypatch.setattr(bootstrap, "sys", machine)
    monkeypatch.setattr(bootstrap, "_generated_interpreter_identity", lambda: "identity")
    monkeypatch.delenv("_PYTHON_SYSCONFIGDATA_NAME", raising=False)
    topology = (machine.meta_path, session.finder, owner.original_paths, (), (),
                machine.stdlib_module_names, ("identity",))
    monkeypatch.setitem(bootstrap._IMPORT_TOPOLOGIES, session, topology)
    monkeypatch.setitem(bootstrap._SETUPTOOLS_CUSTODY, session,
                        (owner, type(owner), owner.root, owner.admission, owner.controller, owner.thread,
                         owner.original_paths, paths))
    assert bootstrap._SourceSession.check_import_topology(session) is topology
    if changed == "path_object":
        machine.path = list(paths)
    else:
        owner.original_paths += ("/foreign",)
    with pytest.raises(bootstrap.SourceBootstrapError, match="setuptools.*path"):
        bootstrap._SourceSession.check_import_topology(session)


@pytest.mark.parametrize("poisoned", [False, True])
def test_stdlib_resolution_uses_original_path_even_after_poison(tmp_path, monkeypatch, poisoned):
    original, vendor = tmp_path / "interpreter", tmp_path / "vendor"
    original.mkdir()
    vendor.mkdir()
    (original / "fractions.py").write_text("INTERPRETER = True\n")
    (vendor / "fractions.py").write_text("raise AssertionError('vendor shadow executed')\n")
    monkeypatch.setattr(sys, "path", [str(original), str(vendor)])
    owner = SimpleNamespace(original_paths=(str(original),))
    session = SimpleNamespace(failed=poisoned, setuptools_owner=lambda: owner,
                              is_interpreter_name=lambda name: name == "fractions")
    topology = (None, None, tuple(sys.path), (),
                ((importlib.machinery.PathFinder, importlib.machinery.PathFinder.find_spec),),
                frozenset({"fractions"}), (None, "_generated_stdlib"))
    session.check_import_topology = lambda: topology
    finder = bootstrap._CapturedSourceFinder(session)
    spec = finder._find_spec("fractions")
    assert spec.origin == str(original / "fractions.py")
    assert session.failed is poisoned
