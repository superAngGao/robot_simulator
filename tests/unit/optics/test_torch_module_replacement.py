"""Production importer with small Python packages/native doubles; no GPU proof."""
import copy
import hashlib
import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest

from optical_worker.native_bootstrap import SourceBootstrapError
from optical_worker import native_bootstrap as bootstrap
from optical_worker.runtime_environment import validate_module_replacement_observations
from tests.unit.optics.test_runtime_import_admission import importer as importer_fixture

importer = importer_fixture
CLASSES = {
    "torch._VF": "VFModule", "torch.backends": "GenericModule",
    "torch.backends.cudnn": "CudnnModule", "torch.backends.miopen": "MiopenModule",
    "torch.backends.mkldnn": "MkldnnModule", "torch.backends.opt_einsum": "OptEinsumModule",
    "torch.backends.quantized": "QuantizedEngine",
}


def _setup(importer, monkeypatch, modify=None, native_globals=()):
    common = (
        "import sys\nfrom types import ModuleType\n"
        "class PropModule(ModuleType):\n"
        " def __init__(self, m, name):\n  super().__init__(name)\n  self.m=m\n"
        " def __getattr__(self, name):\n  return self.m.__getattribute__(name)\n"
    )
    sources = {"torch": "import torch._C\nfrom torch import _VF, backends\n",
               "torch._C": "fixed native double",
               "torch._VF": (
                   "import sys, torch\nfrom types import ModuleType\n"
                   "class VFModule(ModuleType):\n"
                   " def __init__(self,name):\n  super().__init__(name)\n  self.vf=torch._C._VariableFunctions\n"
                   " def __getattr__(self,name):\n  return getattr(self.vf,name)\n"
                   "sys.modules[__name__]=VFModule(__name__)\n"
               )}
    for name, cls in CLASSES.items():
        if name == "torch._VF":
            continue
        prelude = common if name == "torch.backends" else "import sys\nfrom torch.backends import PropModule\n"
        if name.endswith("quantized"):
            source = common.replace("PropModule", cls)
        else:
            source = prelude + f"class {cls}(PropModule):\n def __init__(self,m,name):\n  super().__init__(m,name)\n"
        sources[name] = source + f"sys.modules[__name__]={cls}(sys.modules[__name__],__name__)\n"
    sources["torch.backends"] += "from torch.backends import cudnn, miopen, mkldnn, opt_einsum, quantized\n"
    if modify is not None:
        modify(sources)
    rows = [{"name": name, "phase": "runtime", "class_attribute": cls,
             "relation": "native_attribute" if name == "torch._VF" else "original_backlink",
             "native_root": "torch._C" if name == "torch._VF" else None,
             "native_attributes": ["_VariableFunctions"] if name == "torch._VF" else []}
            for name, cls in sorted(CLASSES.items())]
    def create(delegate, spec):
        result = ModuleType(spec.name)
        result._VariableFunctions = object()
        return result
    monkeypatch.setattr(importlib.machinery.ExtensionFileLoader, "create_module", create)
    monkeypatch.setattr(importlib.machinery.ExtensionFileLoader, "exec_module", lambda *args: None)
    admission, _ = importer(sources, native=("torch._C",),
                            packages=("torch", *(name for name in CLASSES if name != "torch._VF")),
                            replacements=rows, native_globals=native_globals, phase="runtime")
    return admission


def test_seven_replacements_preserve_original_exec_and_pending_parent(importer, monkeypatch):
    admission = _setup(importer, monkeypatch)
    importlib.import_module("torch")
    admission.audit()
    for name in CLASSES:
        record = admission.replacement_records[name]
        assert record["state"] == "admitted"
        assert record["original"] is admission.records[name]["module"]
        assert record["original"] is not record["module"] is sys.modules[name]
        assert record["frame"]["module"] is record["original"]
        assert admission.controller.owners.count(record["module"]) == 1
    events = admission.replacement_observations()
    backend = [e for e in events["events"] if e["name"] == "torch.backends"]
    assert backend[0]["checkpoint"] == "request_entry"
    assert backend[0]["index"] < events["events"][-2]["index"] < backend[1]["index"]
    validate_module_replacement_observations(events, admission.environment,
                                            admission.native_global_observations(), require_complete=True)
    admission.audit()
    assert admission.replacement_observations() == events


@pytest.mark.parametrize("mutation", ["same_class", "class", "backlink", "native_value", "shadow", "record", "event"])
def test_replacement_identity_drift_fails_closed(importer, monkeypatch, mutation):
    admission = _setup(importer, monkeypatch)
    importlib.import_module("torch")
    name = "torch.backends.cudnn"
    record = admission.replacement_records[name]
    original, exposed = record["original"], record["module"]
    if mutation == "same_class":
        sys.modules[name] = type(exposed)(original, name)
    elif mutation == "class":
        vars(original)["CudnnModule"] = type("CudnnModule", (type(exposed).__bases__[0],), {})
    elif mutation == "backlink":
        exposed.m = ModuleType(name)
    elif mutation == "native_value":
        vars(sys.modules["torch._VF"])["vf"] = object()
    elif mutation == "shadow":
        vars(exposed)["__path__"] = list(original.__path__)
    elif mutation == "record":
        admission.replacement_records[name] = dict(record)
    else:
        admission.replacement_events[0]["class"] = 999
    with pytest.raises(SourceBootstrapError):
        admission.audit()
    assert admission.failed and admission.session.failed
    assert any(value is exposed for value in admission.retained)


@pytest.mark.parametrize("boundary", ["request", "return", "frame", "audit", "malformed", "native_global"])
def test_available_publication_held_at_first_real_session_termination(importer, monkeypatch, boundary):
    main = sys.modules["__main__"]
    monkeypatch.setitem(sys.modules, "__main__", main)
    def modify(sources):
        if boundary in {"request", "return", "malformed"}:
            if boundary == "malformed":
                sources["torch._VF"] += "sys.modules[__name__] = object()\n"
            sources["torch._VF"] += "sys.modules['__main__'] = ModuleType('__main__')\n"
            if boundary == "request":
                sources["torch._VF"] += "import torch.backends\n"
    rules = [{"name": "c2global", "observer": "torch._C", "operation": "exec", "phase": "runtime"}]
    admission = _setup(importer, monkeypatch, modify, native_globals=rules if boundary == "native_global" else ())
    session = admission.session
    session.bootstrap_owner = main
    session.interpreter_alias_events = []
    path = Path(sys._stdlib_dir).resolve() / "multiprocessing" / "__init__.py"
    source = path.read_bytes()
    descriptor = (bootstrap._generated_interpreter_identity(), str(path), source,
                  hashlib.sha256(source).hexdigest(), main)
    monkeypatch.setitem(bootstrap._INTERPRETER_ALIAS_CUSTODY, session,
                        (descriptor, bootstrap._thread.get_ident(), [main], None, None))
    monkeypatch.setattr(session, "observe_interpreter_alias",
                        bootstrap._SourceSession.observe_interpreter_alias.__get__(session))
    observations = []
    name = "c2global" if boundary == "native_global" else "torch._VF"
    def terminate_at_entry(**kwargs):
        value = sys.modules.get(name)
        observations.append((value, any(owner is value for owner in admission.retained)))
        admission.controller.termination_evidence = object()
    admission.controller.termination_evidence = None
    admission.controller._fail_guard = lambda *args: None
    admission.controller.terminate = terminate_at_entry
    monkeypatch.setitem(bootstrap._SOURCE_CUSTODY, session, (admission.controller,))
    monkeypatch.setattr(session, "fail", bootstrap._SourceSession.fail.__get__(session))
    if boundary == "native_global":
        def native_exec(delegate, module):
            monkeypatch.setitem(sys.modules, name, ModuleType(name))
            sys.modules["__main__"] = ModuleType("__main__")
        monkeypatch.setattr(importlib.machinery.ExtensionFileLoader, "exec_module", native_exec)
    elif boundary == "frame":
        returning = admission.return_native_call
        def bad_frame(frame, module):
            if frame["loader"].name == name and frame["operation"] == "exec":
                frame["spec"] = object()
            return returning(frame, module)
        monkeypatch.setattr(admission, "return_native_call", bad_frame)
    if boundary == "audit":
        importlib.import_module("torch")
        sys.modules[name] = object()
        sys.modules["__main__"] = ModuleType("__main__")
    count = len(admission.replacement_events)
    with pytest.raises(SourceBootstrapError):
        session.audit() if boundary == "audit" else importlib.import_module("torch")
    assert observations and observations[0][0] is not None and observations[0][1] is True
    assert session.failed
    # The controller endpoint is a CPU double: this proves earliest custody,
    # not native quiescence or a real process quarantine/reap.
    after = len(admission.replacement_events)
    assert after == count
    session.observe_interpreter_alias("audit", final=True)
    assert len(admission.replacement_events) == after
    assert admission.failed and admission.session.failed


def test_replay_rejects_replacement_token_substitution(importer, monkeypatch):
    admission = _setup(importer, monkeypatch)
    importlib.import_module("torch")
    evidence = copy.deepcopy(admission.replacement_observations())
    evidence["events"][1]["replacement"] = 999
    with pytest.raises(ValueError):
        validate_module_replacement_observations(evidence, admission.environment,
                                                admission.native_global_observations())


@pytest.mark.parametrize("projection", ["event_index", "event_original", "record_token"])
@pytest.mark.parametrize("poisoned", [False, True])
def test_equal_value_type_rewrite_rejects_replacement_evidence(importer, monkeypatch, projection, poisoned):
    admission = _setup(importer, monkeypatch)
    importlib.import_module("torch")
    if poisoned:
        admission.fail(RuntimeError("original failure"), "torch")
    if projection == "event_index":
        admission.replacement_events[0]["index"] = False
    elif projection == "event_original":
        event = admission.replacement_events[0]
        event["original"] = float(event["original"])
    else:
        tokens = admission.replacement_records["torch._VF"]["tokens"]
        tokens["original"] = float(tokens["original"])
    with pytest.raises(SourceBootstrapError):
        admission.replacement_observations()
    assert admission.failed and admission.session.failed


@pytest.mark.parametrize("mutation", ["original", "spec", "function", "code", "parent", "original_child",
                                     "token", "coherent", "removed", "none", "eighth"])
def test_replacement_custody_and_parent_drift(importer, monkeypatch, mutation):
    admission = _setup(importer, monkeypatch)
    importlib.import_module("torch")
    name = "torch.backends.cudnn"
    r = admission.replacement_records[name]
    original, exposed = r["original"], r["module"]
    if mutation == "original":
        admission.records[name]["module"] = ModuleType(name)
    elif mutation == "spec":
        original.__spec__ = copy.copy(original.__spec__)
    elif mutation == "function":
        type(exposed).__init__ = lambda self, m, name: None
    elif mutation == "code":
        type(exposed).__init__.__code__ = type(exposed).__init__.__code__.replace(co_name="changed")
    elif mutation == "parent":
        vars(sys.modules["torch.backends"])["cudnn"] = original
    elif mutation == "original_child":
        vars(admission.records["torch.backends"]["module"])["cudnn"] = original
    elif mutation == "token":
        admission.replacement_objects[r["tokens"]["replacement"] - 1] = original
    elif mutation == "coherent":
        alternate = type(exposed)(original, name)
        sys.modules[name] = r["module"] = alternate
        admission.replacement_objects[r["tokens"]["replacement"] - 1] = alternate
    elif mutation == "removed":
        del sys.modules[name]
    elif mutation == "none":
        sys.modules[name] = None
    else:
        sys.modules["torch.unlisted"] = ModuleType("torch.unlisted")
    with pytest.raises(SourceBootstrapError):
        admission.audit()
    assert admission.failed and admission.session.failed
    assert any(value is exposed for value in admission.retained)


@pytest.mark.parametrize("fault", ["raise", "missing", "backlink", "subclass", "shadow", "late",
                                  "adopt", "pending_event", "admitted_event"])
def test_replacement_partial_failure_keeps_available_objects(importer, monkeypatch, fault):
    name = "torch.backends.cudnn"
    def modify(sources):
        if fault == "raise":
            sources[name] += "raise RuntimeError('after replacement')\n"
        elif fault == "missing":
            sources[name] = sources[name].split("sys.modules[__name__]=")[0]
        elif fault == "backlink":
            sources[name] += "sys.modules[__name__].m = object()\n"
        elif fault == "subclass":
            sources[name] += "class Other(CudnnModule): pass\nsys.modules[__name__] = Other(sys.modules[__name__].m, __name__)\n"
        elif fault == "shadow":
            sources[name] += "sys.modules[__name__].__path__ = __path__\n"
        elif fault == "late":
            # A child's body replaces a suspended parent's pending object.
            sources[name] += "sys.modules['torch.backends'] = type(sys.modules['torch.backends'])(sys.modules['torch.backends'].m, 'torch.backends')\n"
    admission = _setup(importer, monkeypatch, modify)
    observed = []
    retain = admission._retain_replacements
    def capture():
        values = retain()
        observed.extend(value for key, value in values.items() if key == name)
        return values
    monkeypatch.setattr(admission, "_retain_replacements", capture)
    if fault == "adopt":
        adopt = admission.controller.adopt_host_owner
        def fail_adoption(value, **kwargs):
            if kwargs.get("operation_id") == "module_replacement_observation" and vars(value).get("__name__") == name:
                raise RuntimeError("replacement adoption fault")
            return adopt(value, **kwargs)
        monkeypatch.setattr(admission.controller, "adopt_host_owner", fail_adoption)
    elif fault.endswith("event"):
        emit = admission._replacement_event
        def fail_event(operation, module, checkpoint, request):
            if module == name and operation == ("observed_pending" if fault == "pending_event" else "admitted"):
                raise RuntimeError("replacement event fault")
            return emit(operation, module, checkpoint, request)
        monkeypatch.setattr(admission, "_replacement_event", fail_event)
    with pytest.raises((SourceBootstrapError, RuntimeError)):
        importlib.import_module("torch")
    assert admission.failed and admission.session.failed
    assert observed and all(any(value is held for held in admission.retained) for value in observed)
    before = len(admission.replacement_events)
    admission.enter_import_wrapper("request_entry", "traceback")
    assert len(admission.replacement_events) == before
    with pytest.raises(SourceBootstrapError):
        admission.audit()
