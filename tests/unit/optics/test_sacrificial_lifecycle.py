from __future__ import annotations

import gc
import subprocess
import sys
import weakref
from dataclasses import replace
from pathlib import Path

import pytest

from optical_worker.lifecycle import DeviceScopeDescriptor
from tests.support.sacrificial_lifecycle import LifetimeError, SacrificialRenderGuardian, State


class Owner:
    pass


class Handshake:
    def __init__(self, *, fail_arm=False, fail_complete=False):
        self.roles = []
        self.fail_arm = fail_arm
        self.fail_complete = fail_complete
        self.completed = 0

    def arm(self, role):
        self.roles.append(role)
        if self.fail_arm:
            raise RuntimeError("no ACK")
        return object()

    def complete(self, ticket):
        if self.fail_complete:
            raise RuntimeError("no disarm ACK")
        self.completed += 1


def descriptor(*, target=1, returns_owner=False):
    return DeviceScopeDescriptor.exact_target(
        operation_id="test_leaf",
        leaf_kind="allocation",
        device_ordinal=target,
        scope_source="fixed_test",
        runtime_identity="warp_1_12",
        unreturned_resource_possible=returns_owner,
    )


def setup():
    guardian = SacrificialRenderGuardian(session_id="test_session", target_ordinal=1)
    owner = Owner()
    reference = weakref.ref(owner)
    permit = guardian.begin_setup_leaf(descriptor(returns_owner=True))
    guardian.finish_setup_leaf(permit, owners=(owner,))
    return guardian, reference


def seal(guardian, n=5):
    guardian.setup_barrier(Handshake(), lambda: None)
    guardian.seal(command_count=n)


def launches(guardian, n=5):
    for position in range(n):
        guardian.before_launch(position)
        guardian.launch_returned(position)


def test_guardian_strong_ownership_and_successful_release():
    guardian, reference = setup()
    gc.collect()
    assert reference() is not None
    with pytest.raises(LifetimeError):
        guardian.seal(command_count=5)
    seal(guardian)
    launches(guardian)
    with pytest.raises(LifetimeError):
        guardian.release()
    guardian.validation_drain(Handshake(), lambda: None)
    with pytest.raises(LifetimeError):
        guardian.release()
    guardian.inspection_completed()
    guardian.release()
    gc.collect()
    assert reference() is None
    assert guardian.state is State.RELEASED
    with pytest.raises(LifetimeError):
        guardian.release()


def test_leaf_owner_transfer_precedes_next_leaf_and_rejects_copied_permit():
    guardian = SacrificialRenderGuardian(session_id="test_session", target_ordinal=1)
    permit = guardian.begin_setup_leaf(descriptor(returns_owner=True))
    with pytest.raises(LifetimeError):
        guardian.begin_setup_leaf(descriptor())
    with pytest.raises(LifetimeError):
        guardian.finish_setup_leaf(replace(permit), owners=(Owner(),))
    with pytest.raises(LifetimeError):
        guardian.setup_barrier(Handshake(), lambda: pytest.fail("active leaf cannot sync"))
    guardian.finish_setup_leaf(permit, owners=(Owner(),))
    with pytest.raises(LifetimeError):
        guardian.finish_setup_leaf(permit, owners=())


@pytest.mark.parametrize("position", range(5))
def test_each_launch_is_latched_before_return_and_retained_until_emergency(position):
    guardian, reference = setup()
    seal(guardian)
    launches(guardian, position)
    guardian.before_launch(position)
    evidence = guardian.snapshot()
    assert evidence["possible_submission_latched"] is True
    assert evidence["attempted_launch_count"] == position + 1
    assert evidence["confirmed_launch_return_count"] == position
    guardian.poison("submit_then_raise")
    gc.collect()
    assert reference() is not None
    guardian.emergency_drain(Handshake(), lambda: None)
    assert reference() is not None
    guardian.release()
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("kind", ["foreign", "unknown_device", "unknown_owner"])
def test_compromised_scope_is_quarantine_only(kind):
    guardian, reference = setup()
    if kind == "unknown_owner":
        permit = guardian.begin_setup_leaf(descriptor(returns_owner=True))
        guardian.fail_setup_leaf(permit, partial_owners=(Owner(),))
    else:
        scope = (
            descriptor(target=0)
            if kind == "foreign"
            else DeviceScopeDescriptor.unknown(
                operation_id="test_leaf",
                leaf_kind="allocation",
                scope_source="fixed_test",
                runtime_identity="warp_1_12",
            )
        )
        with pytest.raises(LifetimeError):
            guardian.begin_setup_leaf(scope)
    with pytest.raises(LifetimeError):
        guardian.emergency_drain(Handshake(), lambda: pytest.fail("must not synchronize target"))
    assert guardian.state is State.QUARANTINED
    assert reference() is not None
    with pytest.raises(LifetimeError):
        guardian.release()
    guardian.quarantine("repeated")
    assert guardian.state is State.QUARANTINED


@pytest.mark.parametrize("role", ["S", "V", "E"])
def test_arm_failure_never_calls_wait_and_retains_graph(role):
    guardian, reference = setup()
    if role != "S":
        seal(guardian)
        if role == "V":
            launches(guardian)
        else:
            guardian.before_launch(0)
            guardian.poison("test")
    operation = {"S": guardian.setup_barrier, "V": guardian.validation_drain, "E": guardian.emergency_drain}[
        role
    ]
    with pytest.raises(RuntimeError, match="no ACK"):
        operation(Handshake(fail_arm=True), lambda: pytest.fail("unacknowledged wait"))
    assert guardian.state is State.QUARANTINED
    assert guardian.snapshot()["blocking_role_entered"] is None
    assert reference() is not None


@pytest.mark.parametrize("role", ["S", "V", "E"])
def test_wait_raising_has_no_second_sync_or_release(role):
    guardian, reference = setup()
    if role != "S":
        seal(guardian)
        if role == "V":
            launches(guardian)
        else:
            guardian.before_launch(0)
            guardian.poison("test")

    def fail():
        raise RuntimeError("wait failed")

    operation = {"S": guardian.setup_barrier, "V": guardian.validation_drain, "E": guardian.emergency_drain}[
        role
    ]
    with pytest.raises(RuntimeError, match="wait failed"):
        operation(Handshake(), fail)
    assert guardian.state is State.QUARANTINED
    assert reference() is not None
    with pytest.raises(LifetimeError):
        guardian.emergency_drain(Handshake(), lambda: pytest.fail("second sync"))
    with pytest.raises(LifetimeError):
        guardian.release()


def test_clean_cancel_requires_sealed_quiescence_without_any_launch():
    guardian, reference = setup()
    with pytest.raises(LifetimeError):
        guardian.release()
    seal(guardian)
    guardian.release()
    gc.collect()
    assert reference() is None


def test_duplicate_launch_and_return_do_not_advance_counts():
    guardian, _ = setup()
    seal(guardian, n=2)
    for wrong in (True, -1, 1):
        with pytest.raises(LifetimeError):
            guardian.before_launch(wrong)
    guardian.before_launch(0)
    with pytest.raises(LifetimeError):
        guardian.before_launch(1)
    guardian.launch_returned(0)
    with pytest.raises(LifetimeError):
        guardian.launch_returned(0)
    assert guardian.snapshot()["attempted_launch_count"] == 1
    assert guardian.snapshot()["confirmed_launch_return_count"] == 1
    guardian.poison("test_cleanup")
    guardian.emergency_drain(Handshake(), lambda: None)
    guardian.release()


def test_no_setup_owner_transfer_means_irreversible_unknown_ownership():
    guardian, reference = setup()
    permit = guardian.begin_setup_leaf(descriptor(returns_owner=True))
    with pytest.raises(LifetimeError, match="no owner"):
        guardian.finish_setup_leaf(permit, owners=())
    assert guardian.snapshot()["ownership_scope_unknown"]
    with pytest.raises(LifetimeError):
        guardian.emergency_drain(Handshake(), lambda: pytest.fail("ownership unknown"))
    del guardian
    gc.collect()
    assert reference() is not None  # The production process quarantine registry owns it.


def test_setup_disarm_failure_never_seals_or_releases():
    guardian, reference = setup()
    with pytest.raises(RuntimeError, match="disarm ACK"):
        guardian.setup_barrier(Handshake(fail_complete=True), lambda: None)
    assert guardian.state is State.QUARANTINED
    assert not guardian.snapshot()["setup_quiescence_proven"]
    with pytest.raises(LifetimeError):
        guardian.seal(command_count=5)
    assert reference() is not None


def test_diagnostic_snapshot_cannot_mutate_lifecycle_trace():
    guardian, _ = setup()
    snapshot = guardian.snapshot()
    declaration = next(item for item in snapshot["trace"] if item["operation"] == "setup_leaf_declared")
    declaration["scope"]["device_ordinals"][:] = [0]
    assert guardian.snapshot()["possible_dirty_device_ordinals"] == [1]
    recorded = next(
        item for item in guardian.snapshot()["trace"] if item["operation"] == "setup_leaf_declared"
    )
    assert recorded["scope"]["device_ordinals"] == [1]
    seal(guardian)
    guardian.release()


def test_guardian_import_and_creation_are_cpu_only():
    root = Path(__file__).resolve().parents[3]
    code = """
import sys
sys.path.insert(0, sys.argv[1])
class DenyCuda:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'warp', 'torch', 'numpy', 'optics'}:
            raise AssertionError('CUDA-capable import before guardian: ' + fullname)
sys.meta_path.insert(0, DenyCuda())
from tests.support.sacrificial_lifecycle import SacrificialRenderGuardian
guardian = SacrificialRenderGuardian(session_id='cpu_test', target_ordinal=1)
assert guardian.resource_count == 0
assert guardian.snapshot()['possible_dirty_device_ordinals'] == []
"""
    completed = subprocess.run([sys.executable, "-I", "-S", "-c", code, str(root)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
