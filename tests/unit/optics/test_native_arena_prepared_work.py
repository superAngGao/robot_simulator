"""Prepared arena binding checks; full worker allocation proof is separate."""
import copy
import sys
from dataclasses import replace

import pytest

from optics import native_arena as native
from tests.unit.optics.test_native_optical_arena import _arena, _manifest, _plan, _spec


def test_acquire_uses_prepared_identity_without_hash_payload_or_weakref_creation(monkeypatch):
    arena = _arena()
    plan = _plan(arena)

    def forbidden(*args, **kwargs):
        raise AssertionError("acquire rebuilt a prepared binding")

    with monkeypatch.context() as patch:
        patch.setattr(native, "_canonical_sha256", forbidden)
        patch.setattr(native.weakref, "ref", forbidden)
        for record_type in (native.NativeOpticalArenaSpec, native.NativeOpticalArenaResourceManifest,
                            native.C4PreparedSamplePlanEvidence, native.C4ActiveLeaseBindingEvidence):
            patch.setattr(record_type, "to_payload", forbidden)
        lease = arena.acquire(plan)
    assert lease.state == "LEASED"
    lease.cancel_before_enqueue()


@pytest.mark.parametrize("replacement", ["plan", "plan_evidence", "lease_evidence", "key"])
def test_equal_copies_cannot_replace_the_prepared_live_binding(replacement):
    arena = _arena()
    plan = _plan(arena)
    if replacement == "plan":
        plan = copy.copy(plan)
    elif replacement == "plan_evidence":
        object.__setattr__(plan, "_evidence", replace(plan.evidence))
    elif replacement == "lease_evidence":
        object.__setattr__(plan._live_lease, "_evidence", replace(plan._live_lease.evidence))
    else:
        other = _plan(arena)
        object.__setattr__(plan, "_registry_key", other._registry_key)
    with pytest.raises(native.NativeOpticalArenaProtocolError):
        arena.acquire(plan)
    assert len(arena.lifecycle_evidence().transitions) == 1


def test_acquire_work_does_not_walk_resource_owners():
    counts = []
    for count in (1, 120, 600):
        spec = _spec()
        manifest, owners = _manifest(spec)
        tokens = tuple(f"owner:{index}" for index in range(count))
        manifest = replace(manifest, session_resource_tokens=tokens,
                           session_resource_owner_tokens=tokens, session_resource_nbytes=(1,) * count,
                           total_resource_nbytes=count + sum(
                               slot.total_resource_nbytes for slot in manifest.slots))
        del owners["session-owner"]
        owners.update((token, object()) for token in tokens)
        arena = native.NativeOpticalArena(spec=spec, resource_manifest=manifest, resource_owners=owners)
        plan = _plan(arena)
        lines = 0

        def observed(frame, event, arg):
            nonlocal lines
            if event == "line" and frame.f_code.co_filename == native.__file__:
                lines += 1
            return observed

        previous = sys.gettrace()
        try:
            sys.settrace(observed)
            lease = arena.acquire(plan)
        finally:
            sys.settrace(previous)
        counts.append(lines)
        lease.cancel_before_enqueue()
    assert counts[0] > 0
    assert counts == [counts[0]] * 3
