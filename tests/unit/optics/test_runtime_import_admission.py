"""External importer mechanics, NOT cold-worker/GPU lifecycle acceptance.

The source-custody and guardian prerequisites are unit doubles. Real importlib
file resolution, create/exec, module publication and exception paths are used.
No backend import or native binary is executed by this suite.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import os
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
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    sessions = []
    original_meta_path = list(sys.meta_path)
    original_modules = dict(sys.modules)

    def create(sources, *, absent=(), publications=(), namespace=False, disabled=(), packages=(), native=()):
        root = tmp_path.resolve()
        files, modules = [], []
        for name, source in sources.items():
            is_package = name == "c2ext" or name in packages
            relative = name.replace(".", "/") + ("/__init__.py" if is_package else ".py")
            if name in native:
                assert not is_package
                relative = name.replace(".", "/") + importlib.machinery.EXTENSION_SUFFIXES[0]
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            data = source.encode("utf-8")
            path.write_bytes(data)
            files.append(
                {
                    "root": "site",
                    "path": relative,
                    "kind": "extension" if name in native else "python",
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            if name in {row["name"] for row in disabled}:
                continue
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
            "disabled_optional_modules": list(disabled),
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
        generated = bootstrap._capture_generated_stdlib(root, tuple(sys.path))
        if generated[1] is not None:
            # Before baseline capture: a cold fixture cannot keep an old data
            # module in its baseline and then upload a replacement identity.
            monkeypatch.delitem(sys.modules, generated[1], raising=False)
        admission = object.__new__(bootstrap._ExternalAdmission)
        monkeypatch.setitem(
            bootstrap._EXTERNAL_CUSTODY, session, (admission, environment, environment.sha256, controller)
        )
        admission.__init__(session, environment, environment.sha256)
        finder = bootstrap._CapturedSourceFinder(session)
        session.finder = finder
        chain = [finder]
        delegates = tuple(
            (owner, owner.find_spec)
            for owner in (
                importlib.machinery.BuiltinImporter,
                importlib.machinery.FrozenImporter,
                importlib.machinery.PathFinder,
            )
        )
        monkeypatch.setitem(
            bootstrap._IMPORT_TOPOLOGIES,
            session,
            (
                chain,
                finder,
                tuple(sys.path),
                tuple(sys.path_hooks),
                delegates,
                frozenset(sys.stdlib_module_names) | frozenset(sys.builtin_module_names),
                generated,
            ),
        )
        sys.meta_path = chain
        sessions.append((admission, generated[1]))
        return admission, root

    yield create
    sys.meta_path[:] = original_meta_path
    for name in tuple(sys.modules):
        if name.split(".")[0] in {
            "c2ext", "c2namespace", "c2missing", "trimesh", "embreex", "shapely", "scipy", "uarray",
        }:
            if name in original_modules:
                sys.modules[name] = original_modules[name]
            else:
                del sys.modules[name]
    for admission, generated_name in sessions:
        # Unit doubles have no CUDA resources/quarantine. Never use this test
        # teardown as production release authority.
        admission.retained.clear()
        bootstrap._GENERATED_STDLIB_SPECS.pop(admission.session, None)
        bootstrap._GENERATED_STDLIB_MODULES.pop(admission.session, None)
        if generated_name is not None and generated_name not in original_modules:
            sys.modules.pop(generated_name, None)


@pytest.mark.parametrize(
    "name,requester",
    [
        ("mapbox_earcut", "trimesh.creation"),
        ("manifold3d", "trimesh.creation"),
        ("triangle", "trimesh.creation"),
        ("collada", "trimesh.exchange.dae"),
    ],
)
def test_metadata_absence_uses_normal_importlib_without_poison(importer, name, requester):
    sources = {
        "trimesh": "",
        "trimesh.exchange": "",
        "trimesh.util": (
            "def has_module(name):\n from importlib.util import find_spec\n"
            " return find_spec(name) is not None\n"
        ),
        requester: f"from trimesh.util import has_module\navailable = has_module({name!r})\n",
    }
    admission, _ = importer(
        sources,
        packages=("trimesh", "trimesh.exchange"),
        absent=[
            {"name": name, "requester": requester, "phase": "materialization"},
        ],
    )
    actual = importlib.import_module(requester)
    assert actual.available is False
    assert not admission.failed and not admission.session.failed
    assert name not in admission.records and name not in admission.loaders and name not in sys.modules
    assert [(e["kind"], e["operation"]) for e in admission.events if e["name"] == name] == [
        ("absent_probe", "absent_probe"),
    ]
    admission.audit()


def test_same_absence_supports_repeated_query_and_normal_import(importer):
    admission, _ = importer(
        {
            "c2ext": (
                "from importlib.util import find_spec\n"
                "assert find_spec('c2missing') is None\nassert find_spec('c2missing') is None\n"
                "try:\n import c2missing\nexcept ModuleNotFoundError as error:\n missing_name = error.name\n"
            )
        },
        absent=[{"name": "c2missing", "requester": "c2ext", "phase": "materialization"}],
    )
    module = importlib.import_module("c2ext")
    assert module.missing_name == "c2missing"
    assert len([e for e in admission.events if e["kind"] == "absent_probe"]) == 3
    assert "c2missing" not in admission.records and "c2missing" not in admission.loaders
    assert not admission.failed and not admission.session.failed


@pytest.mark.parametrize("mutation", ["tail", "path", "hook", "delegate", "resolver_tail"])
def test_resolution_topology_drift_never_calls_added_finder(importer, monkeypatch, mutation):
    admission, _ = importer(
        {"c2ext": "from importlib.util import find_spec\nvalue = find_spec('c2missing')\n"},
        absent=[{"name": "c2missing", "requester": "c2ext", "phase": "materialization"}],
    )
    calls = []

    class Witness:
        def find_spec(self, *args):
            calls.append(args)
            raise AssertionError("alternate finder executed")

    old_chain, old_paths, old_hooks = list(sys.meta_path), list(sys.path), list(sys.path_hooks)
    if mutation == "tail":
        sys.meta_path.append(Witness())
    elif mutation == "path":
        sys.path.append("unexpected_external_root")
    elif mutation == "hook":
        sys.path_hooks.append(Witness().find_spec)
    elif mutation == "delegate":
        monkeypatch.setattr(importlib.machinery.PathFinder, "find_spec", classmethod(Witness().find_spec))
    else:
        original_record = admission.record

        def record(operation, name, kind):
            original_record(operation, name, kind)
            if kind == "absent_probe":
                sys.meta_path.append(Witness())

        admission.record = record
    try:
        with pytest.raises(bootstrap.SourceBootstrapError):
            importlib.import_module("c2ext")
    finally:
        sys.meta_path[:] = old_chain
        sys.path[:] = old_paths
        sys.path_hooks[:] = old_hooks
        monkeypatch.undo()
    assert calls == []
    assert admission.session.failed
    assert "c2missing" not in admission.records and "c2missing" not in admission.loaders


def test_poison_does_not_prevent_stdlib_terminal_import_or_reopen_external(importer):
    admission, _ = importer({"c2ext": ""})
    name = next(
        n for n in ("quopri", "colorsys", "html.parser", "urllib.robotparser") if n not in sys.modules
    )
    original_error = RuntimeError("failed_before_terminal_reporting")
    admission.session.fail(original_error)
    module = importlib.import_module(name)
    assert module.__name__ == name
    assert admission.session.failed and admission.session.failures[0] is original_error
    with pytest.raises(bootstrap.SourceBootstrapError):
        importlib.import_module("c2ext")
    assert "c2ext" not in admission.loaders and "c2ext" not in admission.records


@pytest.mark.skipif(os.name != "posix" or sys.version_info[:2] != (3, 12),
                    reason="real interpreter data module requires CPython3.12 POSIX")
@pytest.mark.parametrize("mutation", ["none", "replacement", "late_override"])
def test_external_audit_generated_module_is_not_an_external_publication(importer, monkeypatch, mutation):
    admission, _ = importer({"c2ext": ""})
    session = admission.session
    name = bootstrap._IMPORT_TOPOLOGIES[session][6][1]
    module = importlib.import_module(name)
    admission.audit()
    assert bootstrap._GENERATED_STDLIB_MODULES[session] is module
    assert not admission.events and name not in admission.records
    assert importlib.import_module(name) is module
    if mutation == "replacement":
        replacement = type(module)(name)
        replacement.__dict__.update(module.__dict__)
        monkeypatch.setitem(sys.modules, name, replacement)
    elif mutation == "late_override":
        monkeypatch.setenv("_PYTHON_SYSCONFIGDATA_NAME", "")
    try:
        if mutation == "none":
            admission.audit()
            assert not admission.failed and not session.failed
        else:
            with pytest.raises(bootstrap.SourceBootstrapError):
                admission.audit()
            assert admission.failed and session.failed
    finally:
        if mutation == "late_override":
            monkeypatch.delenv("_PYTHON_SYSCONFIGDATA_NAME")
    assert not admission.events or all(event["kind"] == "failure" for event in admission.events)


def _disabled_sources():
    # Tiny dependency fixtures, never used for canonical Go2 acceptance. The
    # real production finder/loader is used unchanged by the importer fixture.
    return {
        "trimesh": "from . import ray, path\n",
        "trimesh.exceptions": (
            "class ExceptionWrapper:\n"
            " def __init__(self, error): self.exception = (type(error), error.args)\n"
            " def __getattribute__(self, name):\n"
            "  kind, args = object.__getattribute__(self, 'exception')\n"
            "  raise kind(*args)\n"
        ),
        "trimesh.ray": (
            "try:\n from . import ray_pyembree\n has_embree = True\n"
            "except BaseException as error:\n from ..exceptions import ExceptionWrapper\n"
            " ray_pyembree = ExceptionWrapper(error)\n has_embree = False\n"
        ),
        "trimesh.path": "from . import path\n",
        "trimesh.path.path": (
            "try:\n from . import polygons\n"
            "except BaseException as error:\n from ..exceptions import ExceptionWrapper\n"
            " polygons = ExceptionWrapper(error)\n"
        ),
        "trimesh.path.polygons": "raise AssertionError('ADAPTER BODY MUST NOT RUN')\n",
        "trimesh.ray.ray_pyembree": "raise AssertionError('ADAPTER BODY MUST NOT RUN')\n",
    }


def _disabled_rows():
    return [
        {
            "name": name,
            "path": name.replace(".", "/") + ".py",
            "root": "site",
            "requester": requester,
            "phase": "materialization",
            "absent_prerequisite": dependency,
            "fallback_attribute": attribute,
        }
        for name, requester, dependency, attribute in (
            ("trimesh.path.polygons", "trimesh.path.path", "shapely", "polygons"),
            ("trimesh.ray.ray_pyembree", "trimesh.ray", "embreex", "ray_pyembree"),
        )
    ]


def _disabled_importer(importer, sources=None):
    return importer(
        _disabled_sources() if sources is None else sources,
        disabled=_disabled_rows(),
        packages=("trimesh", "trimesh.ray", "trimesh.path"),
    )


def test_disabled_modules_are_present_but_never_executed_and_unused_rows_are_legal(importer):
    admission, _ = _disabled_importer(importer)
    admission.audit()
    assert not admission.disabled_records and not admission.disabled_tokens and not admission.events
    importlib.import_module("trimesh")
    admission.audit()
    for name in admission.disabled:
        assert name not in sys.modules and name not in admission.records and name not in admission.loaders
        record = admission.disabled_records[name]
        assert record["state"] == "fallback_confirmed"
        assert type(record["error"]) is ImportError and record["error"].__traceback__ is not None
        events = [e for e in admission.events if e["name"] == name]
        assert [e["operation"] for e in events] == ["disabled_optional", "disabled_fallback_confirmed"]
        assert all(e["present"] and e["loader_kind"] == "SourceFileLoader" for e in events)
        assert all(
            e["create_count"] == e["exec_count"] == e["body_count"] == e["native_create_count"] == 0
            for e in events
        )
        assert events[0]["fallback_identity"] is None and events[1]["fallback_identity"] is not None
        admission.controller.require_adopted(record["fallback"])
        admission.controller.require_adopted(record["error"])
    before = list(admission.events), list(admission.retained), list(admission.controller.owners)
    admission.audit()
    admission.audit()
    assert before == (admission.events, admission.retained, admission.controller.owners)


@pytest.mark.parametrize(
    "drift",
    [
        "v1",
        "missing_table",
        "extra_field",
        "duplicate",
        "order",
        "third_row",
        "requester",
        "prerequisite",
        "attribute",
        "phase",
        "path",
        "root",
        "executable_name",
        "executable_alias",
        "absent_name",
        "owner_phase",
        "file_kind",
    ],
)
def test_disabled_schema_has_no_caller_defined_policy(importer, drift):
    admission, _ = _disabled_importer(importer)
    p = admission.environment.to_payload()
    rows = p["disabled_optional_modules"]
    row = rows[0]
    if drift == "v1":
        p["schema_version"] = "q54.p12.4.runtime-environment.v1"
    elif drift == "missing_table":
        del p["disabled_optional_modules"]
    elif drift == "extra_field":
        row["ignore_errors"] = True
    elif drift == "duplicate":
        rows.insert(0, copy.deepcopy(row))
    elif drift == "order":
        rows.reverse()
    elif drift == "third_row":
        row["name"] = "trimesh.path.invented"
    elif drift in {"requester", "prerequisite", "attribute", "phase", "path", "root"}:
        key = {"prerequisite": "absent_prerequisite", "attribute": "fallback_attribute"}.get(drift, drift)
        row[key] = "different"
    elif drift in {"executable_name", "executable_alias"}:
        p["modules"].append(
            {
                "name": row["name"] if drift == "executable_name" else "trimesh.path.alias",
                "root": row["root"],
                "path": row["path"],
                "phase": "materialization",
                "is_package": False,
            }
        )
        p["modules"].sort(key=lambda r: r["name"])
    elif drift == "absent_name":
        p["absent_probes"] = [{"name": row["name"], "requester": row["requester"], "phase": row["phase"]}]
    elif drift == "owner_phase":
        next(r for r in p["modules"] if r["name"] == row["requester"])["phase"] = "runtime"
    else:
        next(r for r in p["files"] if r["path"] == row["path"])["kind"] = "extension"
    with pytest.raises((ValueError, TypeError)):
        RuntimeEnvironmentContract.from_payload(p)
    assert not admission.events and not admission.records and not admission.disabled_records


@pytest.mark.parametrize("drift", ["copy", "unissued", "pending"])
def test_live_denial_tokens_cannot_be_copied_or_left_unconsumed(importer, monkeypatch, drift):
    admission, _ = _disabled_importer(importer)
    consume = admission.consume_disabled

    def intercepted(token, name):
        if drift == "pending":
            return admission.disabled_records[name]["error"]
        invalid = copy.copy(token) if drift == "copy" else bootstrap._DisabledExternalToken()
        return consume(invalid, name)

    monkeypatch.setattr(admission, "consume_disabled", intercepted)
    with pytest.raises(bootstrap.SourceBootstrapError):
        importlib.import_module("trimesh")
    assert admission.failed and admission.session.failed
    assert admission.disabled_tokens
    assert all(r["state"] == "resolution_verified" for r in admission.disabled_records.values())
    assert not any(e["operation"] == "disabled_optional" for e in admission.events)
    assert "trimesh.ray.ray_pyembree" not in admission.loaders


@pytest.mark.parametrize(
    "drift", ["error_args", "error_name", "error_path", "error_identity", "spec", "loader"]
)
def test_confirmed_denial_custody_is_revalidated_on_audit(importer, drift):
    admission, _ = _disabled_importer(importer)
    importlib.import_module("trimesh")
    record = admission.disabled_records["trimesh.ray.ray_pyembree"]
    if drift.startswith("error_"):
        if drift == "error_identity":
            record["error"] = ImportError(
                *record["error"].args, name=record["error"].name, path=record["error"].path
            )
        else:
            setattr(
                record["error"],
                drift.removeprefix("error_"),
                ("wrong",) if drift == "error_args" else "wrong",
            )
    elif drift == "spec":
        record["spec"] = copy.copy(record["spec"])
    else:
        record["spec"].loader = copy.copy(record["spec"].loader)
    before = list(admission.events)
    with pytest.raises(bootstrap.SourceBootstrapError):
        admission.audit()
    assert admission.failed and admission.session.failed
    assert admission.events[:-1] == before
    assert admission.events[-1]["operation"] == "failed"


@pytest.mark.parametrize(
    "drift",
    [
        "bytes",
        "missing",
        "present_prerequisite",
        "preloaded_prerequisite",
        "preloaded_none",
        "preloaded_descendant",
        "wrong_phase",
    ],
)
def test_disabled_resolver_rejects_drift_before_adapter_execution(importer, monkeypatch, drift):
    admission, root = _disabled_importer(importer)
    name = "trimesh.ray.ray_pyembree"
    if drift == "bytes":
        (root / "trimesh/ray/ray_pyembree.py").write_text("wrong", encoding="utf-8")
    elif drift == "missing":
        (root / "trimesh/ray/ray_pyembree.py").unlink()
    elif drift == "present_prerequisite":
        (root / "embreex.py").write_text("raise AssertionError('MUST NOT LOAD')", encoding="utf-8")
    elif drift == "preloaded_prerequisite":
        monkeypatch.setitem(sys.modules, "embreex", None)
    elif drift in {"preloaded_none", "preloaded_descendant"}:
        monkeypatch.setitem(sys.modules, name + (".child" if drift == "preloaded_descendant" else ""), None)
    else:
        admission.phase = "runtime"
    with pytest.raises(bootstrap.SourceBootstrapError):
        importlib.import_module("trimesh")
    assert admission.failed and admission.session.failed
    assert name not in admission.records and name not in admission.loaders
    with pytest.raises(bootstrap.SourceBootstrapError):
        admission.audit()


def test_disabled_wrong_live_requester_fails_through_actual_finder(importer):
    sources = _disabled_sources()
    sources["trimesh.ray"] = ""
    sources["trimesh.path.path"] = "try:\n from ..ray import ray_pyembree\nexcept BaseException:\n pass\n"
    admission, _ = _disabled_importer(importer, sources)
    with pytest.raises(bootstrap.SourceBootstrapError):
        importlib.import_module("trimesh")
    assert admission.failed and admission.session.failed
    assert "trimesh.ray.ray_pyembree" not in admission.loaders
    assert not admission.disabled_records


@pytest.mark.parametrize("suffix", ["", ".child"])
def test_unrequested_disabled_publication_is_rejected_by_audit(importer, monkeypatch, suffix):
    admission, _ = _disabled_importer(importer)
    name = "trimesh.ray.ray_pyembree"
    monkeypatch.setitem(sys.modules, name + suffix, None)
    with pytest.raises(bootstrap.SourceBootstrapError):
        admission.audit()
    assert admission.failed and admission.session.failed
    assert not admission.disabled_records and not admission.disabled_tokens
    assert name not in admission.loaders and name not in admission.records


@pytest.mark.parametrize("drift", ["type", "args", "flag", "body_error"])
def test_parent_fallback_failures_stay_sticky(importer, drift):
    sources = _disabled_sources()
    sources["trimesh.ray"] += {
        "type": "ray_pyembree = object()\n",
        "args": (
            "object.__getattribute__(ray_pyembree, '__dict__')['exception'] = (ImportError, ('wrong',))\n"
        ),
        "flag": "has_embree = True\n",
        "body_error": "raise ImportError('Frozen optional adapter disabled: trimesh.ray.ray_pyembree')\n",
    }[drift]
    admission, _ = _disabled_importer(importer, sources)
    with pytest.raises((bootstrap.SourceBootstrapError, ImportError)):
        importlib.import_module("trimesh")
    assert admission.failed and admission.session.failed
    record = admission.disabled_records["trimesh.ray.ray_pyembree"]
    assert record["state"] == "rejection_issued"
    assert any(value is record["error"] for value in admission.retained)
    assert "trimesh.ray.ray_pyembree" not in admission.loaders


@pytest.mark.parametrize(
    "drift", ["copied", "unissued", "consumed", "replacement_fallback", "repeat_request"]
)
def test_disabled_authority_and_confirmed_identity_cannot_be_reused(importer, drift):
    admission, _ = _disabled_importer(importer)
    importlib.import_module("trimesh")
    name = "trimesh.ray.ray_pyembree"
    record = admission.disabled_records[name]
    with pytest.raises(bootstrap.SourceBootstrapError):
        if drift in {"copied", "unissued", "consumed"}:
            token = record["token"]
            if drift == "copied":
                token = copy.copy(token)
            elif drift == "unissued":
                token = bootstrap._DisabledExternalToken()
            admission.consume_disabled(token, name)
        elif drift == "replacement_fallback":
            wrapper_type = type(record["fallback"])
            vars(record["requester"])["ray_pyembree"] = wrapper_type(record["error"])
            admission.audit()
        else:
            admission.find_spec(name, admission._paths(name))
    assert admission.failed and admission.session.failed


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


def test_failure_intake_keeps_late_owner_and_each_error_without_readoption(importer, monkeypatch):
    admission, _ = importer(
        {
            "c2ext": "import sys, types\ntry:\n from . import child\nexcept ImportError: pass\n"
            "sys.modules['c2ext.late'] = types.ModuleType('late')\n"
            "raise ValueError('outer_after_late_publication')\n",
            "c2ext.child": "raise ImportError('inner_started_failure')\n",
        }
    )
    with pytest.raises(ValueError, match="outer_after_late_publication") as failure:
        importlib.import_module("c2ext")
    late = sys.modules.pop("c2ext.late")
    assert any(x is late for x in admission.retained)
    assert any(x is failure.value for x in admission.failures)
    assert any(str(x) == "inner_started_failure" for x in admission.failures)
    assert "c2ext.late" not in admission.records
    owners, events, notifications = (
        list(admission.controller.owners),
        list(admission.events),
        list(admission.session.failures),
    )
    admission.controller._terminating = True
    monkeypatch.setattr(admission.controller, "adopt_host_owner", lambda *a, **k: pytest.fail("readoption"))
    admission.fail(failure.value, "c2ext", native_create=True)
    admission.fail(failure.value, "c2ext", native_create=True)
    assert admission.controller.owners == owners and admission.events == events
    assert admission.session.failures == notifications
    assert (
        sum(e is failure.value and n == "c2ext" and native for e, n, native in admission.failure_contexts)
        == 1
    )
    assert any(x is late for x in admission.retained)


def test_failure_recording_fault_cannot_skip_sticky_notification(importer, monkeypatch):
    admission, _ = importer({"c2ext": "pass\n"})
    original, recording = ValueError("original"), RuntimeError("recording")

    def reject(*args):
        raise recording

    monkeypatch.setattr(admission, "record", reject)
    admission.fail(original, "c2ext")
    assert admission.failed and admission.session.failed
    assert admission.failures == [original, recording]


def _native_alias_importer(importer, monkeypatch, fault=None):
    # Fixed fake native delegates exercise production importlib/registry checks,
    # not native exception safety or the mandatory real SciPy extension matrix.
    calls = []

    def create(delegate, spec):
        calls.append("create")
        module = type(sys)(spec.name)
        if fault == "create_publication":
            sys.modules["c2ext.alias"] = module
        return module

    def execute(delegate, module):
        calls.append("exec")
        if fault != "missing":
            sys.modules["c2ext.alias"] = module
        if fault == "wrong_owner":
            replacement = type(sys)(module.__name__)
            vars(replacement).update(vars(module))
            sys.modules["c2ext.alias"] = replacement
        if fault == "qualified":
            sys.modules[module.__name__] = type(sys)(module.__name__)
        if fault == "exec_failure":
            raise ImportError("native_exec_fixture_failure")

    monkeypatch.setattr(importlib.machinery.ExtensionFileLoader, "create_module", create)
    monkeypatch.setattr(importlib.machinery.ExtensionFileLoader, "exec_module", execute)
    row = _publication()
    row.update(creator="c2ext._native", publisher="c2ext._native", attributes=[])
    admission, _ = importer(
        {"c2ext": "pass\n", "c2ext._native": "fixture only"}, native=("c2ext._native",), publications=[row]
    )
    importlib.import_module("c2ext")
    return admission, calls


def test_native_self_alias_is_exact_returned_root_and_audits_are_read_only(importer, monkeypatch):
    admission, calls = _native_alias_importer(importer, monkeypatch)
    # Configuration does not invent observations for an unrequested publisher.
    admission.audit()
    assert "c2ext.alias" not in admission.records and calls == []
    module = importlib.import_module("c2ext._native")
    record = admission.records["c2ext.alias"]
    assert record["module"] is record["creator"] is record["publisher"] is module
    assert record["relation"] == "creator_self"
    assert module.__name__ == module.__spec__.name == "c2ext._native"
    assert "c2ext.alias" not in admission.loaders and "c2ext.alias" not in admission.specs
    before = list(admission.events), list(admission.retained), dict(admission.records)
    admission.audit()
    admission.audit()
    assert before == (admission.events, admission.retained, admission.records)
    assert calls == ["create", "exec"]


def _intrinsic_name_importer(importer, monkeypatch, fault=None, name="scipy._lib._uarray._uarray"):
    # Fixed synthetic delegates exercise the real wrapper, not native execution
    # or captured-worker custody. Real pinned SciPy proof remains mandatory.
    calls, returned = [], []

    def create(delegate, spec):
        calls.append("create")
        module = type(sys)(admission.environment.expected_file_module_name(spec.name))
        returned.append(module)
        if fault == "missing":
            del module.__name__
        elif fault in {"none", "empty", "integer", "qualified", "arbitrary"}:
            module.__name__ = {"none": None, "empty": "", "integer": 3,
                               "qualified": name, "arbitrary": "other._uarray"}[fault]
        elif fault == "lookalike":
            module.__name__ = "uarray._uarray"
        return module

    def execute(delegate, module):
        calls.append("exec")
        if fault == "exec_changed":
            module.__name__ = name

    monkeypatch.setattr(importlib.machinery.ExtensionFileLoader, "create_module", create)
    monkeypatch.setattr(importlib.machinery.ExtensionFileLoader, "exec_module", execute)
    parents = tuple(".".join(name.split(".")[:i]) for i in range(1, len(name.split("."))))
    admission, _ = importer(
        {**dict.fromkeys(parents, ""), name: "fixture bytes; never native execution"},
        packages=parents, native=(name,),
    )
    importlib.import_module(parents[-1])
    return admission, name, calls, returned


def test_intrinsic_name_preserves_qualified_identity_and_readonly_audits(importer, monkeypatch):
    admission, name, calls, returned = _intrinsic_name_importer(importer, monkeypatch)
    module = importlib.import_module(name)
    assert module is returned[0] is admission.records[name]["module"]
    assert module.__name__ == "uarray._uarray"
    assert module.__spec__.name == name
    assert module.__spec__ is admission.specs[name]
    assert module.__loader__ is admission.loaders[name]
    assert module.__package__ == "scipy._lib._uarray"
    assert "uarray" not in sys.modules and "uarray._uarray" not in sys.modules
    live_checks = []
    original_check = admission.check_live

    def count_check():
        live_checks.append(True)
        original_check()

    monkeypatch.setattr(admission, "check_live", count_check)
    before = list(admission.events), list(admission.retained), dict(admission.records)
    admission.audit()
    admission.audit()
    # Each audit revalidates the environment once, then every actual module's
    # identity/metadata. It must not redo the full graph once per module.
    assert len(live_checks) == 2
    assert before == (admission.events, admission.retained, admission.records)
    assert calls == ["create", "exec"]


@pytest.mark.parametrize(
    "fault", ["missing", "none", "empty", "integer", "qualified", "arbitrary", "exec_changed"]
)
def test_native_name_failure_retains_actual_return_before_unwind(importer, monkeypatch, fault):
    admission, name, calls, returned = _intrinsic_name_importer(importer, monkeypatch, fault)
    with pytest.raises(bootstrap.SourceBootstrapError, match="intrinsic module name drift"):
        importlib.import_module(name)
    assert admission.failed and admission.session.failed
    assert admission.records[name]["module"] is returned[0]
    assert name not in sys.modules
    assert calls == (["create", "exec"] if fault == "exec_changed" else ["create"])
    assert not any(e["name"] == name and e["operation"] == "completed" for e in admission.events)
    with pytest.raises(bootstrap.SourceBootstrapError):
        admission.audit()


@pytest.mark.parametrize(
    "drift", ["name", "spec", "loader", "package", "file", "origin", "delegate", "delegate_name",
              "delegate_path", "nonpackage"]
)
def test_intrinsic_name_full_metadata_checked_before_exec(importer, monkeypatch, drift):
    admission, name, calls, returned = _intrinsic_name_importer(importer, monkeypatch)
    spec = admission.find_spec(name, sys.modules["scipy._lib._uarray"].__path__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if drift == "name":
        module.__name__ = name
    elif drift == "spec":
        module.__spec__ = copy.copy(spec)
    elif drift == "loader":
        module.__loader__ = copy.copy(module.__loader__)
    elif drift == "package":
        module.__package__ = "uarray"
    elif drift == "file":
        module.__file__ = "wrong.so"
    elif drift == "origin":
        spec.origin = "wrong.so"
    elif drift == "delegate":
        spec.loader.delegate = copy.copy(spec.loader.delegate)
    elif drift == "delegate_name":
        spec.loader.delegate.name = "uarray._uarray"
    elif drift == "delegate_path":
        spec.loader.delegate.path = "wrong.so"
    else:
        module.__path__ = []
    with pytest.raises(bootstrap.SourceBootstrapError):
        spec.loader.exec_module(module)
    assert calls == ["create"] and admission.failed and admission.session.failed
    assert admission.records[name]["module"] is returned[0]


@pytest.mark.parametrize("name", ["scipy._lib.other._uarray", "scipy._lib._uarray.other"])
def test_native_name_relation_has_no_prefix_or_suffix_exemption(importer, monkeypatch, name):
    admission, name, calls, returned = _intrinsic_name_importer(importer, monkeypatch, "lookalike", name)
    with pytest.raises(bootstrap.SourceBootstrapError, match="intrinsic module name drift"):
        importlib.import_module(name)
    assert calls == ["create"] and admission.failed
    assert admission.records[name]["module"] is returned[0]


@pytest.mark.parametrize("drift", ["name", "replacement", "short_alias", "short_none"])
def test_intrinsic_name_late_drift_and_alias_do_not_gain_permission(importer, monkeypatch, drift):
    admission, name, calls, returned = _intrinsic_name_importer(importer, monkeypatch)
    module = importlib.import_module(name)
    if drift == "name":
        module.__name__ = name
    elif drift == "replacement":
        replacement = type(sys)(module.__name__)
        vars(replacement).update(vars(module))
        sys.modules[name] = replacement
    else:
        sys.modules["uarray._uarray"] = module if drift == "short_alias" else None
    with pytest.raises(bootstrap.SourceBootstrapError):
        admission.audit()
    assert admission.failed and admission.session.failed
    assert admission.records[name]["module"] is returned[0]
    assert calls == ["create", "exec"]


@pytest.mark.parametrize(
    "fault",
    [
        "precreate_none",
        "create_publication",
        "preexec_none",
        "missing",
        "wrong_owner",
        "qualified",
        "exec_failure",
        "deleted_record",
        "extra",
        "alias_replaced",
        "phase",
    ],
)
def test_native_self_alias_drift_fails_closed_at_real_loader_boundary(importer, monkeypatch, fault):
    admission, calls = _native_alias_importer(importer, monkeypatch, fault)
    with pytest.raises((bootstrap.SourceBootstrapError, ImportError)):
        if fault == "precreate_none":
            sys.modules["c2ext.alias"] = None
        if fault == "phase":
            admission.phase = "runtime"
        if fault == "preexec_none":
            spec = admission.find_spec("c2ext._native", sys.modules["c2ext"].__path__)
            module = importlib.util.module_from_spec(spec)
            sys.modules["c2ext._native"] = module
            sys.modules["c2ext.alias"] = None
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module("c2ext._native")
        if fault == "deleted_record":
            del admission.records["c2ext.alias"]
        elif fault == "extra":
            sys.modules["c2ext.extra"] = module
        elif fault == "alias_replaced":
            sys.modules["c2ext.alias"] = type(sys)(module.__name__)
        admission.audit()
    assert admission.failed and admission.session.failed
    assert calls == (
        []
        if fault in {"precreate_none", "phase"}
        else ["create"]
        if fault in {"create_publication", "preexec_none"}
        else ["create", "exec"]
    )
    if calls:
        assert admission.records["c2ext._native"]["module"] is not None
    if fault == "deleted_record":
        assert "c2ext.alias" not in admission.records


def test_attribute_publication_record_cannot_be_recreated_by_audit(importer):
    admission, _ = importer(
        {
            "c2ext": "import sys, types\nalias = types.ModuleType('alias')\n"
            "sys.modules['c2ext.alias'] = alias\n"
        },
        publications=[_publication()],
    )
    importlib.import_module("c2ext")
    del admission.records["c2ext.alias"]
    with pytest.raises(bootstrap.SourceBootstrapError, match="during audit"):
        admission.audit()
    assert "c2ext.alias" not in admission.records
    assert sum(e["operation"] == "creator_publication" for e in admission.events) == 1


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
