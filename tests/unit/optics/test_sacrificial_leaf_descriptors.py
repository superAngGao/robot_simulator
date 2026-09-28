"""CPU descriptor mechanics, not a substitute for supervised real-GPU controls."""

import sys
from inspect import getattr_static
from unittest.mock import patch

import pytest

from tests.support.sacrificial_leaf_audit import (
    NativeEventCallGuard,
    RecordedLeafAudit,
    RecordedLeafContamination,
    validate_event_surface,
)


@pytest.mark.parametrize("kind", ["method", "property", "classmethod", "staticmethod"])
def test_leaf_hooks_preserve_descriptor_binding_and_restore_before_body(kind):
    calls = []

    class Owner:
        def method(self):
            calls.append(self)
            return self

        @property
        def property(self):
            calls.append(self)
            return self

        @classmethod
        def classmethod(cls):
            calls.append(cls)
            return cls

        @staticmethod
        def staticmethod():
            calls.append("static")
            return "static"

    instance = Owner()
    expected = Owner if kind == "classmethod" else "static" if kind == "staticmethod" else instance
    original = getattr_static(Owner, kind)
    audit = object.__new__(RecordedLeafAudit)
    audit._active = None
    audit.rejected = []

    def invoke():
        value = getattr(instance, kind)
        return value if kind == "property" else value()

    try:
        with patch.object(Owner, kind, audit._replacement(kind, original)):
            assert type(getattr_static(Owner, kind)) is type(original)
            assert invoke() is expected
            assert calls == [expected]
            calls.clear()
            audit._active = 0
            with pytest.raises(RecordedLeafContamination, match=kind):
                invoke()
            assert calls == []
            assert audit.rejected == [kind]
            raise RuntimeError("fixed exceptional exit")
    except RuntimeError as exc:
        assert str(exc) == "fixed exceptional exit"
    assert getattr_static(Owner, kind) is original
    assert invoke() is expected


def test_constructor_static_descriptor_rejects_before_new_and_restores_exactly():
    calls = []

    class Owner:
        def __new__(cls):
            calls.append(cls)
            return super().__new__(cls)

    original = getattr_static(Owner, "__new__")
    audit = object.__new__(RecordedLeafAudit)
    audit._active = 0
    audit.rejected = []
    with patch.object(Owner, "__new__", audit._replacement("Owner.__new__", original)):
        with pytest.raises(RecordedLeafContamination, match="Owner.__new__"):
            Owner()
        assert calls == []
    assert getattr_static(Owner, "__new__") is original
    assert type(Owner()) is Owner and calls == [Owner]


def test_new_public_runtime_event_operation_is_not_silently_outside_inventory():
    class Event:
        @classmethod
        def from_ipc_handle(cls):
            pass

        def wait(self):
            pass

        @property
        def is_complete(self):
            return True

    targets = {"runtime.Event.from_ipc_handle", "runtime.Event.wait", "runtime.Event.is_complete"}
    assert validate_event_surface((("runtime.Event", Event),), targets) == tuple(sorted(targets))
    for missing in targets:
        with pytest.raises(RecordedLeafContamination, match="uncovered installed"):
            validate_event_surface((("runtime.Event", Event),), targets - {missing})


@pytest.mark.parametrize("action", ["constructor", "bound", "unbound", "cached", "class_method"])
def test_native_call_monitor_denies_before_immutable_body_and_restores(action):
    sink = []
    cached = sink.append
    alias = list
    active = False
    rejected = []

    def deny(target):
        rejected.append(target)
        raise RecordedLeafContamination(target)

    guard = NativeEventCallGuard(
        (("fixture.List", list), ("fixture.Dict", dict)), active=lambda: active, deny=deny
    )
    before = tuple((sys.monitoring.get_tool(i), sys.monitoring.get_events(i)) for i in range(6))
    try:
        with guard.installed():
            active = True
            try:
                if action == "constructor":
                    alias()
                elif action == "bound":
                    sink.append(1)
                elif action == "unbound":
                    list.append(sink, 1)
                elif action == "cached":
                    cached(1)
                else:
                    dict.fromkeys(("key",))
            finally:
                active = False
    except RecordedLeafContamination:
        pass
    else:
        pytest.fail("immutable call body escaped")
    assert sink == []
    assert rejected == [
        "fixture.List"
        if action == "constructor"
        else "fixture.Dict.fromkeys"
        if action == "class_method"
        else "fixture.List.append"
    ]
    assert guard.calls == guard.restoration_checks == 1
    assert tuple((sys.monitoring.get_tool(i), sys.monitoring.get_events(i)) for i in range(6)) == before


def test_native_monitor_leaves_other_tool_and_profile_intact_on_normal_exit():
    monitoring = sys.monitoring
    external = next(i for i in range(6) if monitoring.get_tool(i) is None and monitoring.get_events(i) == 0)
    monitoring.use_tool_id(external, "external_observer_fixture")
    profile = sys.getprofile()

    def observer(*args):
        return None

    previous = monitoring.register_callback(external, monitoring.events.CALL, observer)
    monitoring.set_events(external, monitoring.events.CALL)
    guard = NativeEventCallGuard((("fixture.Dict", dict),), active=lambda: True, deny=lambda name: None)
    try:
        with guard.installed():
            assert abs(-2) == 2
        assert guard.calls == 0 and guard.restoration_checks == 1
        assert monitoring.get_tool(external) == "external_observer_fixture"
        assert monitoring.get_events(external) == monitoring.events.CALL
        assert sys.getprofile() is profile
    finally:
        monitoring.set_events(external, 0)
        removed = monitoring.register_callback(external, monitoring.events.CALL, previous)
        monitoring.free_tool_id(external)
    assert removed is observer


def test_no_monitoring_slot_fails_closed_without_replacing_another_tool():
    monitoring = sys.monitoring
    owned = []
    try:
        for tool in range(6):
            if monitoring.get_tool(tool) is None:
                monitoring.use_tool_id(tool, f"occupied_{tool}")
                owned.append(tool)
        before = tuple(monitoring.get_tool(i) for i in range(6))
        guard = NativeEventCallGuard((("fixture.Dict", dict),), active=lambda: True, deny=lambda name: None)
        with pytest.raises(RecordedLeafContamination, match="no unused"):
            with guard.installed():
                pytest.fail("entered without observation")
        assert tuple(monitoring.get_tool(i) for i in range(6)) == before
        assert guard.calls == guard.restoration_checks == 0
    finally:
        for tool in owned:
            monitoring.free_tool_id(tool)


@pytest.mark.parametrize("operation", ["register_callback", "set_events"])
def test_partial_native_monitor_install_restores_owned_slot(monkeypatch, operation):
    monitoring = sys.monitoring
    before = tuple((monitoring.get_tool(i), monitoring.get_events(i)) for i in range(6))
    original = getattr(monitoring, operation)
    called = False

    def fail_once(*args):
        nonlocal called
        if not called:
            called = True
            raise RuntimeError("fixed monitoring installation failure")
        return original(*args)

    guard = NativeEventCallGuard((("fixture.Dict", dict),), active=lambda: True, deny=lambda name: None)
    with monkeypatch.context() as patching:
        patching.setattr(monitoring, operation, fail_once)
        with pytest.raises(RuntimeError, match="fixed monitoring installation"):
            with guard.installed():
                pytest.fail("installation should fail before entry")
    assert guard._tool is None and guard.calls == 0
    assert guard.restoration_checks == 1
    assert tuple((monitoring.get_tool(i), monitoring.get_events(i)) for i in range(6)) == before
