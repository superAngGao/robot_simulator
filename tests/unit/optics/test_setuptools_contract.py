"""Actual guardian/adoption joins for the fixed helper vocabulary, not a transaction."""

import itertools
from types import SimpleNamespace

import pytest

from optical_worker.historical.compilation_contract import (
    SETUPTOOLS_ALIAS_ACTIONS,
    SETUPTOOLS_HELPER_BINDINGS,
    setuptools_event_fields,
    setuptools_role_anchor,
)
from optical_worker.lifecycle import HOST_OWNER, NativeWorkerResourceGuardian, WorkerSetupProtocolError
from optical_worker.runtime_custody import _RuntimeReplay


def test_real_guardian_helper_anchors_reject_all_coherent_slot_exchanges():
    guardian = NativeWorkerResourceGuardian(session_identity="setuptools_contract_cpu")
    slots = [(action, field, 0 if action.endswith("_read") else None)
             for action, field, _ in SETUPTOOLS_HELPER_BINDINGS]
    slots += [("resource_read", "handle_token", index) for index in (0, 1)]
    owners, objects, adoptions = [], {}, {}
    for token, slot in enumerate(slots, 1):
        owner = object()
        owners.append(owner)
        role, operation = setuptools_role_anchor(*slot)
        record = guardian.adopt(owner, role=role, device_scope=HOST_OWNER, operation_id=operation)
        objects[token] = {"guardian_sequence": record.sequence}
        adoptions[record.sequence] = {"index": record.sequence, "details": record.to_payload()}
    # Descriptive event fixture only; these are actual guardian-produced records.
    # No setup guard, complete compiler record or live initialization is claimed.
    replay = SimpleNamespace(objects=objects, adoptions=adoptions, lci=len(adoptions))
    frozen = guardian.records
    for token, slot in enumerate(slots, 1):
        role, operation = setuptools_role_anchor(*slot)
        _RuntimeReplay.adopted(replay, token, roles={role}, scope=HOST_OWNER, operation=operation)
    for left, right in itertools.combinations(range(len(slots)), 2):
        role, operation = setuptools_role_anchor(*slots[left])
        with pytest.raises(ValueError, match="operation mismatch"):
            _RuntimeReplay.adopted(replay, right + 1, roles={role}, scope=HOST_OWNER, operation=operation)
    assert len(slots) == 15 and guardian.records == frozen
    assert all(actual is expected for actual, expected in zip(guardian._owners, owners))
    with pytest.raises(WorkerSetupProtocolError, match="already adopted"):
        guardian.adopt(owners[0], role="external_import_owner", device_scope=HOST_OWNER,
                       operation_id="external_import_owner")
    assert guardian.records == frozen


def test_alias_repetitions_reuse_the_same_operation_without_a_second_adoption():
    for field in ("alias_spec_token", "delegate_token", "wrapper_token"):
        original = setuptools_role_anchor("alias_spec", field)
        assert all(setuptools_role_anchor(action, field) == original for action in SETUPTOOLS_ALIAS_ACTIONS)
    assert setuptools_role_anchor("metadata_read", "bytes_token", 0) != setuptools_role_anchor(
        "resource_read", "bytes_token", 0)
    assert setuptools_role_anchor("metadata_read", "bytes_token", 1) == setuptools_role_anchor(
        "metadata_read", "bytes_token", 0)
    assert setuptools_role_anchor("resource_read", "handle_token", 1) != setuptools_role_anchor(
        "resource_read", "handle_token", 0)


@pytest.mark.parametrize("action,field", [
    ("hooked", "hack_module_token"), ("shim_entered", "override_module_token"),
    ("alias_spec", "source_module_token"), ("alias_spec", "source_spec_token"),
    ("alias_spec", "source_loader_token"), ("resource_read", "package_token"),
    ("completed", "cpp_extension_module_token"), ("failed", "exception_token"),
])
def test_ordinary_and_failed_owners_cannot_receive_a_new_helper_anchor(action, field):
    with pytest.raises(ValueError, match="ordinary-source or failed-owner"):
        setuptools_role_anchor(action, field, 0 if action == "resource_read" else None)


@pytest.mark.parametrize("index", [None, True, -1, 0.0, "0"])
def test_resource_occurrence_cannot_alias_by_boolean_or_coercion(index):
    with pytest.raises(ValueError, match="read occurrence"):
        setuptools_role_anchor("resource_read", "handle_token", index)


def test_vocabulary_is_closed_and_does_not_implicitly_issue_import_roles():
    with pytest.raises(ValueError, match="unknown.*action"):
        setuptools_event_fields("other_dependency")
    with pytest.raises(ValueError, match="unknown.*field"):
        setuptools_role_anchor("hooked", "other_token")
    with pytest.raises(ValueError, match="unexpected.*occurrence"):
        setuptools_role_anchor("hooked", "finder_token", 0)
    fields = setuptools_event_fields("alias_created")
    assert "source_spec_token" not in fields and "initialization_token" in fields
    with pytest.raises(AttributeError):
        fields.add("permission")
