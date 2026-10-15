import gc
import sys
import weakref

sys.path.insert(0, sys.argv[1])
from optical_worker.lifecycle import DeviceScopeDescriptor
from tests.support.sacrificial_lifecycle import LifetimeError, SacrificialRenderGuardian, State


class Handshake:
    def arm(self, role):
        return object()

    def complete(self, ticket):
        pass


def descriptor(returns_owner):
    return DeviceScopeDescriptor.exact_target(
        operation_id='review', leaf_kind='allocation', device_ordinal=1,
        scope_source='review', runtime_identity='cpu',
        unreturned_resource_possible=returns_owner)


def rejected(call):
    try:
        call()
    except LifetimeError:
        return
    raise AssertionError('forbidden action accepted')


class Owner:
    pass


def transfer(guardian, permit, mode, position):
    a, b = Owner(), Owner()
    refs = weakref.ref(a), weakref.ref(b)
    batch = [a, b]
    batch.insert(position, None)
    try:
        if mode == 'finish':
            guardian.finish_setup_leaf(permit, owners=tuple(batch))
        else:
            guardian.fail_setup_leaf(permit, partial_owners=tuple(batch))
    except RuntimeError:
        return refs
    raise AssertionError('invalid transfer accepted')


for mode in ('finish', 'fail'):
    for position in range(3):
        guardian = SacrificialRenderGuardian(session_id='review', target_ordinal=1)
        permit = guardian.begin_setup_leaf(descriptor(mode == 'finish'))
        refs = transfer(guardian, permit, mode, position)
        gc.collect()
        assert guardian.state is State.POISONED
        assert guardian.snapshot()['ownership_scope_unknown']
        assert guardian._active is None
        assert all(ref() is not None for ref in refs)
        rejected(guardian.release)
        rejected(lambda: guardian.finish_setup_leaf(permit, owners=()))
        rejected(lambda: guardian.begin_setup_leaf(descriptor(False)))
        rejected(lambda: guardian.setup_barrier(Handshake(), lambda: None))
        rejected(lambda: guardian.emergency_drain(Handshake(), lambda: None))
        assert guardian.state is State.QUARANTINED
        guardian.quarantine('repeated')
        rejected(guardian.release)
        del guardian
        gc.collect()
        assert all(ref() is not None for ref in refs)
        print('PASS invalid', mode, position, 'retained; permit cleared; recovery/release denied')


def valid_transfer(guardian, mode):
    owner = Owner()
    ref = weakref.ref(owner)
    permit = guardian.begin_setup_leaf(descriptor(mode == 'finish'))
    if mode == 'finish':
        guardian.finish_setup_leaf(permit, owners=(owner, owner))
    else:
        guardian.fail_setup_leaf(permit, partial_owners=(owner, owner))
    return ref


for mode in ('finish', 'fail'):
    guardian = SacrificialRenderGuardian(session_id='review', target_ordinal=1)
    ref = valid_transfer(guardian, mode)
    gc.collect()
    assert ref() is not None
    assert guardian.resource_count == 2  # One capsule plus one deduplicated owner.
    if mode == 'finish':
        guardian.setup_barrier(Handshake(), lambda: None)
        guardian.seal(command_count=2)
    else:
        guardian.emergency_drain(Handshake(), lambda: None)
    guardian.release()
    gc.collect()
    assert ref() is None
    assert guardian.resource_count == 0
    rejected(guardian.release)
    print('PASS valid', mode, 'capsule and duplicate owner released exactly once')
