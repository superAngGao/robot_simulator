"""Actual importer mechanics with fixed native doubles, not native/GPU proof."""

import copy
import gc
import importlib
import sys
import threading
import weakref
from types import ModuleType

import pytest

from optical_worker.native_bootstrap import SourceBootstrapError
from optical_worker.runtime_environment import (
    RuntimeEnvironmentContract,
    validate_native_global_observations,
)
from tests.unit.optics.test_runtime_import_admission import importer as importer_fixture

importer = importer_fixture


def _setup(importer, monkeypatch, execute, *, extra=(), create=None, sources=None):
    names = ("c2ext._parent", "c2ext._child")
    rows = [{"name": "c2ext.shared", "observer": names[0], "operation": "exec", "phase": "materialization"}]
    rows.extend(extra)
    admission, _ = importer(
        {"c2ext": "pass\n", **dict.fromkeys(names, "fixed native bytes"), **(sources or {})},
        native=names, native_globals=sorted(rows, key=lambda row: row["name"]),
    )
    monkeypatch.setattr(importlib.machinery.ExtensionFileLoader, "create_module",
                        create or (lambda delegate, spec: ModuleType(spec.name)))
    monkeypatch.setattr(importlib.machinery.ExtensionFileLoader, "exec_module",
                        lambda delegate, module: execute(admission, delegate.name))
    importlib.import_module("c2ext")
    return admission


def _publish(name="c2ext.shared"):
    value = ModuleType(name)
    sys.modules[name] = value
    return value


@pytest.mark.parametrize("checkpoint", ["return", "stdlib", "nested"])
def test_exact_interval_pending_then_once_only_admission(importer, monkeypatch, checkpoint):
    def execute(admission, name):
        if name.endswith("_parent"):
            value = _publish()
            if checkpoint == "stdlib":
                # Exercise the actual common finder before its stdlib branch.
                spec = admission.session.finder.find_spec("fractions")
                assert spec is not None
            elif checkpoint == "nested":
                importlib.import_module("c2ext._child")
            if checkpoint != "return":
                assert admission.native_global_records["c2ext.shared"]["state"] == "observed_pending"
                assert admission.records["c2ext.shared"]["module"] is value
        else:
            assert admission.native_global_records["c2ext.shared"]["frame"]["loader"].name == "c2ext._parent"

    admission = _setup(importer, monkeypatch, execute)
    admission.audit()  # Unrequested rows stay absent.
    importlib.import_module("c2ext._parent")
    record = admission.native_global_records["c2ext.shared"]
    assert record["state"] == "completed"
    assert "creator" not in record and "publisher" not in record
    assert "c2ext.shared" not in admission.loaders
    before = admission.native_global_observations()
    admission.audit()
    admission.audit()
    assert admission.native_global_observations() == before
    assert [row["operation"] for row in before["events"]] == ["observed_pending", "admitted"]
    assert before["events"][0]["checkpoint"] == ("exec_return" if checkpoint == "return" else "request_entry")
    validate_native_global_observations(before, admission.environment, require_complete=True)


def test_nested_observers_keep_distinct_intervals_and_parent_pending(importer, monkeypatch):
    def execute(admission, name):
        if name.endswith("_parent"):
            _publish()
            importlib.import_module("c2ext._child")
        else:
            _publish("c2ext.child_shared")

    row = {"name": "c2ext.child_shared", "observer": "c2ext._child",
           "operation": "exec", "phase": "materialization"}
    admission = _setup(importer, monkeypatch, execute, extra=[row])
    importlib.import_module("c2ext._parent")
    admission.audit()
    evidence = admission.native_global_observations()
    assert [(e["name"], e["operation"]) for e in evidence["events"]] == [
        ("c2ext.shared", "observed_pending"), ("c2ext.child_shared", "observed_pending"),
        ("c2ext.child_shared", "admitted"), ("c2ext.shared", "admitted"),
    ]
    validate_native_global_observations(evidence, admission.environment, require_complete=True)


@pytest.mark.parametrize("fault", ["early", "none", "type", "create", "child", "python", "missing", "phase",
                                    "wrapper", "removed_pending", "replaced_pending", "copied_frame"])
def test_invalid_interval_or_identity_is_sticky_and_retained(importer, monkeypatch, fault):
    def create(delegate, spec):
        if fault == "create" and spec.name.endswith("_parent"):
            _publish()
        return ModuleType(spec.name)

    def execute(admission, name):
        if name.endswith("_child"):
            if fault == "child":
                _publish()
            return
        if fault == "child":
            importlib.import_module("c2ext._child")
            return
        if fault == "python":
            importlib.import_module("c2ext.python")
            return
        if fault == "missing":
            return
        value = _publish()
        if fault == "none":
            sys.modules["c2ext.shared"] = None
        elif fault == "type":
            sys.modules["c2ext.shared"] = object()
        elif fault == "phase":
            admission.phase = "runtime"
        elif fault in {"removed_pending", "replaced_pending", "copied_frame"}:
            admission.session.finder.find_spec("fractions")
            if fault == "removed_pending":
                del sys.modules["c2ext.shared"]
            elif fault == "replaced_pending":
                _publish()
            else:
                record = admission.native_global_records["c2ext.shared"]
                record["frame"] = dict(record["frame"])
            assert any(owner is value for owner in admission.retained)
        elif fault == "wrapper":
            del sys.modules["c2ext.shared"]
            original = admission.check_live

            def check():
                _publish()
                admission.session.finder.find_spec("fractions")
                original()

            monkeypatch.setattr(admission, "check_live", check)

    admission = _setup(importer, monkeypatch, execute, create=create, sources={
        "c2ext.python": "import sys\nfrom types import ModuleType\n"
                        "sys.modules['c2ext.shared'] = ModuleType('c2ext.shared')\n"
                        "import c2ext._child\n",
    })
    if fault == "early":
        _publish()
    with pytest.raises(SourceBootstrapError):
        importlib.import_module("c2ext._parent")
    assert admission.failed and admission.session.failed
    assert not any(event["operation"] == "admitted" for event in admission.native_global_events)
    if "c2ext.shared" in sys.modules:
        assert any(owner is sys.modules["c2ext.shared"] for owner in admission.retained)


@pytest.mark.parametrize("fault", ["removed", "replace", "record", "copied_record", "metadata", "thread"])
def test_audit_cannot_restore_or_accept_changed_authority(importer, monkeypatch, fault):
    admission = _setup(importer, monkeypatch, lambda admission, name: _publish())
    importlib.import_module("c2ext._parent")
    original = sys.modules["c2ext.shared"]
    count = len(admission.native_global_events)
    if fault == "removed":
        del sys.modules["c2ext.shared"]
    elif fault == "replace":
        _publish()
    elif fault == "record":
        del admission.records["c2ext.shared"]
    elif fault == "copied_record":
        admission.native_global_records["c2ext.shared"] = dict(
            admission.native_global_records["c2ext.shared"])
    elif fault == "metadata":
        original.__file__ = "invented"
    elif fault == "thread":
        failures = []

        def foreign():
            try:
                admission.session.finder.find_spec("fractions")
            except SourceBootstrapError as error:
                failures.append(error)

        thread = threading.Thread(target=foreign)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive() and len(failures) == 1
    with pytest.raises(SourceBootstrapError):
        admission.audit()
    assert admission.failed and len(admission.native_global_events) == count
    assert any(owner is original for owner in admission.retained)


@pytest.mark.parametrize("fault", ["adopt", "event", "native", "nested_swallowed"])
def test_partial_failure_keeps_exact_owners_and_terminal_stdlib(importer, monkeypatch, fault):
    refs = []

    def execute(admission, name):
        if name.endswith("_child"):
            raise ValueError("original child failure")
        value = _publish()
        refs.append(weakref.ref(value))
        if fault == "native":
            raise ValueError("original native failure")
        if fault == "nested_swallowed":
            try:
                importlib.import_module("c2ext._child")
            except ValueError:
                late = _publish()
                refs.append(weakref.ref(late))

    admission = _setup(importer, monkeypatch, execute)
    if fault == "adopt":
        original = admission.controller.adopt_host_owner

        def adopt(owner, **kwargs):
            if kwargs.get("operation_id") == "native_global_observation":
                raise ValueError("original adopt failure")
            return original(owner, **kwargs)

        monkeypatch.setattr(admission.controller, "adopt_host_owner", adopt)
    elif fault == "event":
        def fail_event(*args):
            raise ValueError("original event failure")

        monkeypatch.setattr(admission, "_global_event", fail_event)
    with pytest.raises((ValueError, SourceBootstrapError)):
        importlib.import_module("c2ext._parent")
    assert admission.failed and admission.failures
    sys.modules.pop("c2ext.shared", None)
    gc.collect()
    assert refs and all(ref() is not None and any(x is ref() for x in admission.retained) for ref in refs)
    # Existing poisoned stdlib topology remains usable without re-adoption.
    assert admission.session.finder.find_spec("fractions") is not None
    assert not any(event["operation"] == "admitted" for event in admission.native_global_events)


def test_swallowed_checkpoint_reentry_cannot_complete_adoption(importer, monkeypatch):
    admission = _setup(importer, monkeypatch, lambda admission, name: _publish())
    original = admission.controller.adopt_host_owner

    def adopt(owner, **kwargs):
        if kwargs.get("operation_id") == "native_global_observation":
            try:
                admission.session.finder.find_spec("fractions")
            except SourceBootstrapError:
                pass
        return original(owner, **kwargs)

    monkeypatch.setattr(admission.controller, "adopt_host_owner", adopt)
    with pytest.raises(SourceBootstrapError, match="swallowed failure"):
        importlib.import_module("c2ext._parent")
    assert admission.failed and not admission.native_global_events
    assert any(value is sys.modules["c2ext.shared"] for value in admission.retained)


@pytest.mark.parametrize("mutation", ["early_parent", "pending_child", "erase_child"])
def test_nested_evidence_cannot_turn_parent_success_into_child_success(importer, monkeypatch, mutation):
    def execute(admission, name):
        _publish("c2ext.shared" if name.endswith("_parent") else "c2ext.child_shared")
        if name.endswith("_parent"):
            importlib.import_module("c2ext._child")

    row = {"name": "c2ext.child_shared", "observer": "c2ext._child",
           "operation": "exec", "phase": "materialization"}
    admission = _setup(importer, monkeypatch, execute, extra=[row])
    importlib.import_module("c2ext._parent")
    payload = admission.native_global_observations()
    if mutation == "early_parent":
        event = payload["events"].pop()
        payload["events"].insert(1, event)
    elif mutation == "pending_child":
        payload["events"].pop(2)
    else:
        payload["events"] = [event for event in payload["events"] if event["name"] == "c2ext.shared"]
    for index, event in enumerate(payload["events"]):
        event["index"] = index
    with pytest.raises(ValueError):
        validate_native_global_observations(payload, admission.environment, require_complete=True)


@pytest.mark.parametrize("fault", ["version", "creator", "python", "phase", "collision", "operation"])
def test_schema_has_no_alternate_global_permission(importer, monkeypatch, fault):
    admission = _setup(importer, monkeypatch, lambda admission, name: None)
    payload = admission.environment.to_payload()
    row = payload["native_globals"][0]
    if fault == "version":
        payload["schema_version"] = "q54.p12.4.runtime-environment.v3"
    elif fault == "creator":
        row["creator"] = row.pop("observer")
    elif fault == "python":
        row["observer"] = "c2ext"
    elif fault == "phase":
        row["phase"] = "runtime"
    elif fault == "collision":
        row["name"] = "c2ext._child"
    elif fault == "operation":
        row["operation"] = "create"
    with pytest.raises(ValueError):
        RuntimeEnvironmentContract.from_payload(payload)


@pytest.mark.parametrize("fault", [
    "remove", "swap", "token", "phase", "observer", "call", "checkpoint", "pending",
])
def test_evidence_rejects_fabricated_admissions(importer, monkeypatch, fault):
    admission = _setup(importer, monkeypatch, lambda admission, name: _publish())
    importlib.import_module("c2ext._parent")
    payload = copy.deepcopy(admission.native_global_observations())
    event = payload["events"][-1]
    if fault == "remove":
        payload["events"] = []
    elif fault == "swap":
        payload["events"].reverse()
    elif fault == "pending":
        payload["events"].pop()
    elif fault == "token":
        event["object_token"] += 1
    elif fault == "phase":
        event["phase"] = "runtime"
    elif fault == "observer":
        event["observer"] = "c2ext._child"
    elif fault == "call":
        payload["calls"][event["call_sequence"]]["state"] = "failed"
    elif fault == "checkpoint":
        event["checkpoint"] = "request_entry"
    with pytest.raises(ValueError):
        validate_native_global_observations(payload, admission.environment, require_complete=True)


@pytest.mark.parametrize("boundary", ["pending", "audit", "issuance"])
def test_coherent_spec_copy_cannot_replace_original_execution_spec(importer, monkeypatch, boundary):
    specs = []

    def replace(admission):
        original = admission.specs["c2ext._parent"]
        replacement = copy.copy(original)
        specs.extend((original, replacement))
        admission.specs["c2ext._parent"] = replacement
        sys.modules["c2ext._parent"].__spec__ = replacement
        for frame in admission.native_calls:
            if frame["loader"].name == "c2ext._parent":
                frame["spec"] = replacement

    def execute(admission, name):
        _publish()
        if boundary == "pending":
            admission.session.finder.find_spec("fractions")
            replace(admission)

    admission = _setup(importer, monkeypatch, execute)
    if boundary == "pending":
        with pytest.raises(SourceBootstrapError, match="binding drift"):
            importlib.import_module("c2ext._parent")
        assert not any(e["operation"] == "admitted" for e in admission.native_global_events)
    else:
        importlib.import_module("c2ext._parent")
        replace(admission)
        with pytest.raises(SourceBootstrapError, match="binding drift"):
            (admission.audit if boundary == "audit" else admission.native_global_observations)()
    assert admission.failed and admission.session.failed
    assert all(any(owner is spec for owner in admission.retained) for spec in specs)
    assert admission.session.finder.find_spec("fractions") is not None


@pytest.mark.parametrize("boundary", ["pending", "audit", "issuance"])
@pytest.mark.parametrize("fault", ["paired_token", "checkpoint", "remove", "reorder", "copy"])
def test_live_event_projection_cannot_rewrite_issued_facts(importer, monkeypatch, boundary, fault):
    def mutate(admission):
        events = admission.native_global_events
        if fault == "paired_token":
            for event in events:
                event["object_token"] += 100
        elif fault == "checkpoint":
            events[0]["checkpoint"], events[0]["request_name"] = "exec_return", None
        elif fault == "remove":
            events.clear()
        elif fault == "reorder":
            if len(events) == 1:
                events.append(events[0])
            else:
                events.reverse()
        else:
            events[:] = [dict(event) for event in events]

    def execute(admission, name):
        _publish()
        admission.session.finder.find_spec("fractions")
        if boundary == "pending":
            mutate(admission)

    admission = _setup(importer, monkeypatch, execute)
    if boundary == "pending":
        with pytest.raises(SourceBootstrapError, match="event"):
            importlib.import_module("c2ext._parent")
        assert not any(e["operation"] == "admitted" for e in admission.native_global_events)
    else:
        importlib.import_module("c2ext._parent")
        mutate(admission)
        count = len(admission.native_global_events)
        with pytest.raises(SourceBootstrapError, match="event"):
            (admission.audit if boundary == "audit" else admission.native_global_observations)()
        assert len(admission.native_global_events) == count
    assert admission.failed and admission.session.failed
    assert admission.native_global_records["c2ext.shared"]["token"] == 1
    assert any(owner is sys.modules["c2ext.shared"] for owner in admission.retained)


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("fault", ["create_phase", "parent_return", "bool_parent"])
def test_replay_rejects_impossible_call_phase_and_return_order(importer, monkeypatch, complete, fault):
    def execute(admission, name):
        _publish("c2ext.shared" if name.endswith("_parent") else "c2ext.child_shared")
        if name.endswith("_parent"):
            importlib.import_module("c2ext._child")

    row = {"name": "c2ext.child_shared", "observer": "c2ext._child",
           "operation": "exec", "phase": "materialization"}
    admission = _setup(importer, monkeypatch, execute, extra=[row])
    importlib.import_module("c2ext._parent")
    payload = admission.native_global_observations()
    if fault == "create_phase":
        next(c for c in payload["calls"] if c["observer"] == "c2ext._parent"
             and c["operation"] == "create")["phase"] = "runtime"
    elif fault == "parent_return":
        payload["events"][0]["checkpoint"], payload["events"][0]["request_name"] = "exec_return", None
    else:
        # Parent sequence is an integer identity, never a boolean flag.
        payload["events"][1]["parent_call_sequence"] = True
    if not complete:
        payload["events"].pop()
        parent = next(c for c in payload["calls"] if c["observer"] == "c2ext._parent"
                      and c["operation"] == "exec")
        parent["state"] = "failed"
    with pytest.raises(ValueError):
        validate_native_global_observations(payload, admission.environment, require_complete=complete)


@pytest.mark.parametrize("reverse", [False, True])
def test_unrelated_observer_returns_do_not_close_sibling_intervals(importer, monkeypatch, reverse):
    def execute(admission, name):
        _publish("c2ext.shared" if name.endswith("_parent") else "c2ext.child_shared")

    row = {"name": "c2ext.child_shared", "observer": "c2ext._child",
           "operation": "exec", "phase": "materialization"}
    admission = _setup(importer, monkeypatch, execute, extra=[row])
    names = ["c2ext._parent", "c2ext._child"]
    for name in reversed(names) if reverse else names:
        importlib.import_module(name)
    admission.audit()
    validate_native_global_observations(admission.native_global_observations(), admission.environment,
                                        require_complete=True)


@pytest.mark.parametrize("failure", ["issuance", "audit", "import"])
@pytest.mark.parametrize("drift", ["tokens", "erase", "order", "frame"])
def test_poisoned_reporting_never_exports_compromised_issued_history(importer, monkeypatch, failure, drift):
    def execute(admission, name):
        if name.endswith("_child"):
            raise ValueError("unrelated import failed")
        _publish()

    admission = _setup(importer, monkeypatch, execute)
    importlib.import_module("c2ext._parent")
    if failure == "import":
        with pytest.raises(ValueError, match="unrelated"):
            importlib.import_module("c2ext._child")
    if drift == "tokens":
        for event in admission.native_global_events:
            event["object_token"] += 100
    elif drift == "erase":
        admission.native_global_events.clear()
    elif drift == "order":
        admission.native_global_events.reverse()
    else:
        admission.native_global_records["c2ext.shared"]["frame"]["phase"] = "runtime"
    if failure != "import":
        with pytest.raises(SourceBootstrapError):
            (admission.audit if failure == "audit" else admission.native_global_observations)()
    count = len(admission.native_global_events)
    for _ in range(2):
        with pytest.raises(SourceBootstrapError):
            admission.native_global_observations()
    assert admission.failed and admission.session.failed
    assert len(admission.native_global_events) == count
    assert admission.session.finder.find_spec("fractions") is not None
    assert any(owner is sys.modules["c2ext.shared"] for owner in admission.retained)


@pytest.mark.parametrize("after_return", [False, True])
def test_legitimate_failed_prefix_survives_importlib_unwind(importer, monkeypatch, after_return):
    def execute(admission, name):
        _publish()
        if not after_return:
            admission.session.finder.find_spec("fractions")
            raise ValueError("failed before return")

    admission = _setup(importer, monkeypatch, execute)
    if after_return:
        original = admission.reconcile_publications

        def fail_after_return(name, **kwargs):
            if name == "c2ext._parent":
                raise ValueError("failed after return")
            return original(name, **kwargs)

        monkeypatch.setattr(admission, "reconcile_publications", fail_after_return)
    with pytest.raises(ValueError, match="failed"):
        importlib.import_module("c2ext._parent")
    assert "c2ext._parent" not in sys.modules
    owner = sys.modules.pop("c2ext.shared")
    for _ in range(2):
        payload = admission.native_global_observations()
        validate_native_global_observations(payload, admission.environment)
        assert (payload["events"][0]["object_token"]
                == admission.native_global_records["c2ext.shared"]["token"])
        assert [e["operation"] for e in payload["events"]] == ["observed_pending"]
    assert any(value is owner for value in admission.retained)
    assert admission.session.finder.find_spec("fractions") is not None


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("checkpoint", ["return", "request", "mixed"])
def test_two_globals_in_one_interval_have_consistent_return_facts(importer, monkeypatch, prefix, checkpoint):
    def execute(admission, name):
        _publish()
        if checkpoint == "mixed":
            admission.session.finder.find_spec("fractions")
        _publish("c2ext.shared_two")
        if checkpoint == "request":
            admission.session.finder.find_spec("fractions")

    row = {"name": "c2ext.shared_two", "observer": "c2ext._parent",
           "operation": "exec", "phase": "materialization"}
    admission = _setup(importer, monkeypatch, execute, extra=[row])
    importlib.import_module("c2ext._parent")
    payload = admission.native_global_observations()
    sequence = payload["events"][0]["call_sequence"]
    if prefix:
        payload["events"] = payload["events"][:2]
        payload["calls"][sequence]["state"] = "failed"
    validate_native_global_observations(payload, admission.environment, require_complete=not prefix)
    # Contradict the same delegate's known return without changing ancestry.
    payload["events"][0]["checkpoint"], payload["events"][0]["request_name"] = "exec_return", None
    payload["events"][1]["checkpoint"], payload["events"][1]["request_name"] = "request_entry", "fractions"
    with pytest.raises(ValueError, match="after return"):
        validate_native_global_observations(payload, admission.environment, require_complete=not prefix)


def test_return_observation_cannot_describe_a_still_started_call(importer, monkeypatch):
    admission = _setup(importer, monkeypatch, lambda admission, name: _publish())
    importlib.import_module("c2ext._parent")
    payload = admission.native_global_observations()
    payload["events"] = payload["events"][:1]
    payload["calls"][payload["events"][0]["call_sequence"]]["state"] = "started"
    with pytest.raises(ValueError, match="started call"):
        validate_native_global_observations(payload, admission.environment)


@pytest.mark.parametrize("mode", ["complete", "prefix", "failed_child"])
def test_all_same_call_first_observations_precede_its_first_admission(importer, monkeypatch, mode):
    def execute(admission, name):
        if name.endswith("_child"):
            raise ValueError("later unrelated child failed")
        _publish()
        _publish("c2ext.shared_two")

    row = {"name": "c2ext.shared_two", "observer": "c2ext._parent",
           "operation": "exec", "phase": "materialization"}
    admission = _setup(importer, monkeypatch, execute, extra=[row])
    importlib.import_module("c2ext._parent")
    if mode == "failed_child":
        with pytest.raises(ValueError, match="later unrelated"):
            importlib.import_module("c2ext._child")
    payload = admission.native_global_observations()
    validate_native_global_observations(payload, admission.environment, require_complete=mode == "complete")
    events = payload["events"]
    payload["events"] = [events[index] for index in (0, 2, 1, 3)]
    for index, event in enumerate(payload["events"]):
        event["index"] = index
    with pytest.raises(ValueError, match="same-call admission"):
        validate_native_global_observations(payload, admission.environment,
                                            require_complete=mode == "complete")


@pytest.mark.parametrize("at_request", [False, True])
def test_three_globals_at_one_return_can_follow_completed_child(importer, monkeypatch, at_request):
    def execute(admission, name):
        if name.endswith("_child"):
            _publish("c2ext.child_shared")
            return
        importlib.import_module("c2ext._child")
        _publish()
        if at_request:
            admission.session.finder.find_spec("fractions")
        _publish("c2ext.shared_two")
        _publish("c2ext.shared_three")

    rows = [{"name": name, "observer": observer, "operation": "exec", "phase": "materialization"}
            for name, observer in [("c2ext.child_shared", "c2ext._child"),
                                   ("c2ext.shared_two", "c2ext._parent"),
                                   ("c2ext.shared_three", "c2ext._parent")]]
    admission = _setup(importer, monkeypatch, execute, extra=rows)
    importlib.import_module("c2ext._parent")
    admission.audit()
    payload = admission.native_global_observations()
    assert payload["events"][1]["operation"] == "admitted"
    assert payload["events"][2]["operation"] == "observed_pending"
    validate_native_global_observations(payload, admission.environment, require_complete=True)
