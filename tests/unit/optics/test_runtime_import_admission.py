"""External importer mechanics, NOT cold-worker/GPU lifecycle acceptance.

The source-custody and guardian prerequisites are unit doubles. Real importlib
file resolution, create/exec, module publication and exception paths are used.
No backend import or native binary is executed by this suite.
"""

from __future__ import annotations

import hashlib
import importlib
import py_compile
import sys
from pathlib import Path

import pytest

from optical_worker import native_bootstrap as bootstrap
from optical_worker.runtime_environment import RUNTIME_ENVIRONMENT_VERSION, RuntimeEnvironmentContract


class _Owners:
    def __init__(self):
        self.owners = []
        self._terminating = False

    def _require_setup(self):
        pass

    def adopt_host_owner(self, owner, **_):
        self.owners.append(owner)

    def require_adopted(self, owner):
        assert any(value is owner for value in self.owners)


@pytest.fixture
def importer(tmp_path, monkeypatch):
    sessions = []
    original_meta_path = list(sys.meta_path)

    def create(sources, *, absent=(), publications=(), namespace=False):
        root = tmp_path.resolve()
        files, modules = [], []
        for name, source in sources.items():
            is_package = name == "c2ext"
            relative = name.replace(".", "/") + ("/__init__.py" if is_package else ".py")
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            data = source.encode("utf-8")
            path.write_bytes(data)
            files.append(
                {
                    "root": "site",
                    "path": relative,
                    "kind": "python",
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            modules.append(
                {
                    "name": name,
                    "root": "site",
                    "path": relative,
                    "is_package": is_package,
                    "phase": "materialization",
                }
            )
        if namespace:
            (root / "c2namespace").mkdir(exist_ok=True)
        executable = Path(sys.executable).resolve()
        payload = {
            "schema_version": RUNTIME_ENVIRONMENT_VERSION,
            "interpreter": {
                "path": executable.as_posix(),
                "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                "implementation": sys.implementation.name,
                "version": list(sys.version_info[:3]),
            },
            "roots": [{"id": "site", "path": root.as_posix(), "kind": "site"}],
            "files": sorted(files, key=lambda r: (r["root"], r["path"])),
            "distributions": [],
            "modules": sorted(modules, key=lambda r: r["name"]),
            "absent_probes": list(absent),
            "publications": list(publications),
            "namespaces": (
                [
                    {
                        "name": "c2namespace",
                        "phase": "materialization",
                        "locations": [{"root": "site", "path": "c2namespace"}],
                    }
                ]
                if namespace
                else []
            ),
            "settings": {"TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "warp_quiet": True},
        }
        environment = RuntimeEnvironmentContract.from_payload(payload)
        controller = _Owners()
        session = object.__new__(bootstrap._SourceSession)
        session.failed, session.failures = False, []

        def check():
            bootstrap._require(not session.failed, "unit source session failed")

        def fail(error):
            session.failed = True
            session.failures.append(error)

        session.check_live = check
        session.bound_controller = lambda: controller
        session.fail = fail
        admission = object.__new__(bootstrap._ExternalAdmission)
        monkeypatch.setitem(
            bootstrap._EXTERNAL_CUSTODY, session, (admission, environment, environment.sha256, controller)
        )
        admission.__init__(session, environment, environment.sha256)
        finder = bootstrap._CapturedSourceFinder(session)
        sys.meta_path.insert(0, finder)
        sessions.append(admission)
        return admission, root

    yield create
    sys.meta_path[:] = original_meta_path
    for name in tuple(sys.modules):
        if name.split(".")[0] in {"c2ext", "c2namespace", "c2missing"}:
            del sys.modules[name]
    for admission in sessions:
        # Unit doubles have no CUDA resources/quarantine. Never use this test
        # teardown as production release authority.
        admission.retained.clear()


def test_real_file_import_retains_exact_objects_and_executes_held_source(importer):
    admission, _ = importer({"c2ext": "from . import child\n", "c2ext.child": "value = 17\n"})
    module = importlib.import_module("c2ext")
    admission.audit()
    assert module.child.value == 17
    assert admission.records["c2ext.child"]["module"] is module.child
    for name in ("c2ext", "c2ext.child"):
        assert [e["operation"] for e in admission.events if e["name"] == name] == [
            "file_resolved",
            "create_started",
            "created",
            "exec_started",
            "completed",
        ]
        admission.controller.require_adopted(sys.modules[name])


def test_existing_bytecode_cache_is_not_executable_input(importer):
    admission, root = importer({"c2ext": "value = 'trusted'\n"})
    path = root / "c2ext/__init__.py"
    good = path.read_bytes()
    path.write_text("value = 'hostile'\n", encoding="utf-8")
    py_compile.compile(
        str(path), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH
    )
    path.write_bytes(good)
    delegate = importlib.machinery.SourceFileLoader("c2ext", str(path))
    assert "hostile" in delegate.get_code("c2ext").co_consts
    assert importlib.import_module("c2ext").value == "trusted"
    admission.audit()


def test_changed_source_after_capture_executes_held_bytes_but_fails_segment_audit(importer):
    admission, root = importer({"c2ext": "value = 'trusted'\n"})
    spec = admission.find_spec("c2ext")
    module = importlib.util.module_from_spec(spec)
    sys.modules["c2ext"] = module
    (root / "c2ext/__init__.py").write_text("raise AssertionError('must not execute')\n", encoding="utf-8")
    spec.loader.exec_module(module)
    assert module.value == "trusted"
    with pytest.raises(ValueError):
        admission.audit()
    assert admission.failed and admission.session.failed


def test_optional_absence_executes_no_loader_and_is_not_poison(importer):
    admission, _ = importer(
        {"c2ext": "try:\n import c2missing\nexcept ModuleNotFoundError:\n value = 9\n"},
        absent=[{"name": "c2missing", "requester": "c2ext", "phase": "materialization"}],
    )
    assert importlib.import_module("c2ext").value == 9
    admission.audit()
    assert not admission.failed
    assert "c2missing" not in admission.loaders
    assert "c2missing" not in admission.records
    assert [e["operation"] for e in admission.events if e["name"] == "c2missing"] == ["absent_probe"]


@pytest.mark.parametrize("reason", ["present", "wrong_requester", "unlisted"])
def test_optional_probe_cannot_hide_environment_drift(importer, reason):
    rules = (
        []
        if reason == "unlisted"
        else [{"name": "c2missing", "requester": "c2ext", "phase": "materialization"}]
    )
    admission, root = importer({"c2ext": "import c2missing\n"}, absent=rules)
    if reason == "present":
        (root / "c2missing.py").write_text("raise AssertionError('must not execute')\n", encoding="utf-8")
    with pytest.raises(bootstrap.SourceBootstrapError):
        importlib.import_module("c2missing" if reason == "wrong_requester" else "c2ext")
    assert admission.failed and admission.session.failed
    assert "c2missing" not in admission.records


def test_swallowed_started_import_error_is_sticky_and_retains_partial_module(importer):
    admission, _ = importer(
        {
            "c2ext": "try:\n from . import child\nexcept ImportError:\n pass\n",
            "c2ext.child": "partial_owner = object()\nraise ImportError('after body started')\n",
        }
    )
    with pytest.raises(bootstrap.SourceBootstrapError):
        importlib.import_module("c2ext")
    partial = admission.records["c2ext.child"]["module"]
    assert partial.partial_owner is not None
    assert any(owner is partial for owner in admission.retained)
    assert "c2ext.child" not in sys.modules
    assert admission.failed and admission.failures
    with pytest.raises(bootstrap.SourceBootstrapError):
        admission.audit()


def _publication():
    return {
        "name": "c2ext.alias",
        "kind": "module_alias",
        "creator": "c2ext",
        "publisher": "c2ext",
        "attributes": ["alias"],
        "phase": "materialization",
    }


def test_creator_publication_has_no_invented_file_execution(importer):
    admission, _ = importer(
        {
            "c2ext": "import sys, types\nalias = types.ModuleType('alias')\n"
            "sys.modules['c2ext.alias'] = alias\n"
        },
        publications=[_publication()],
    )
    module = importlib.import_module("c2ext")
    admission.audit()
    assert admission.records["c2ext.alias"]["creator"] is module
    assert admission.records["c2ext.alias"]["module"] is module.alias
    assert [e["operation"] for e in admission.events if e["name"] == "c2ext.alias"] == ["creator_publication"]
    assert "c2ext.alias" not in admission.loaders


@pytest.mark.parametrize("drift", ["missing", "wrong_owner", "replacement", "extra"])
def test_publication_requires_exact_held_creator_and_complete_table(importer, drift):
    body = "import sys, types\nalias = types.ModuleType('alias')\n"
    if drift != "missing":
        body += (
            "sys.modules['c2ext.alias'] = types.ModuleType('wrong')\n"
            if drift == "wrong_owner"
            else "sys.modules['c2ext.alias'] = alias\n"
        )
    admission, _ = importer({"c2ext": body}, publications=[_publication()])
    with pytest.raises(bootstrap.SourceBootstrapError):
        module = importlib.import_module("c2ext")
        if drift == "replacement":
            sys.modules["c2ext.alias"] = type(module)("alias")
        if drift == "extra":
            sys.modules["c2ext.extra"] = type(module)("extra")
        admission.audit()
    assert admission.failed


@pytest.mark.parametrize("drift", [None, "path", "spec", "file"])
def test_namespace_has_no_body_and_rejects_metadata_drift(importer, drift):
    admission, _ = importer({"c2ext": "value = 1\n"}, namespace=True)
    module = importlib.import_module("c2namespace")
    admission.audit()
    assert [e["operation"] for e in admission.events if e["name"] == "c2namespace"] == ["namespace"]
    if drift is None:
        return
    if drift == "path":
        module.__path__.append("/unapproved")
    elif drift == "spec":
        module.__spec__.name = "other"
    else:
        module.__file__ = "/fake.py"
    with pytest.raises(bootstrap.SourceBootstrapError):
        admission.audit()
    assert admission.failed


@pytest.mark.parametrize("drift", ["name", "loader", "path", "lookup"])
def test_file_metadata_and_rule_index_drift_poison(importer, drift):
    admission, _ = importer({"c2ext": "value = 1\n"})
    module = importlib.import_module("c2ext")
    if drift == "name":
        module.__spec__.name = "other"
    elif drift == "loader":
        module.__loader__ = object()
    elif drift == "path":
        module.__path__.append("/unapproved")
    else:
        admission.rules = {}
    with pytest.raises(bootstrap.SourceBootstrapError):
        admission.audit()
    assert admission.failed
