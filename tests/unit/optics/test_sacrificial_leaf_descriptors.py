"""CPU descriptor mechanics, not a substitute for supervised real-GPU controls."""

from inspect import getattr_static
from unittest.mock import patch

import pytest

from tests.support.sacrificial_leaf_audit import (
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
