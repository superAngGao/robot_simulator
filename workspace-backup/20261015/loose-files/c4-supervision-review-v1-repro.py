import gc
import sys

sys.path.insert(0, sys.argv[1])
from optical_worker.lifecycle import DeviceScopeDescriptor
from tests.support.sacrificial_lifecycle import SacrificialRenderGuardian


for operation in ('finish', 'fail'):
    markers = []

    class Owner:
        def __init__(self, name):
            self.name = name

        def __del__(self):
            markers.append(self.name)

    guardian = SacrificialRenderGuardian(session_id='review_repro', target_ordinal=1)
    permit = guardian.begin_setup_leaf(DeviceScopeDescriptor.exact_target(
        operation_id='partial_owner_repro', leaf_kind='allocation', device_ordinal=1,
        scope_source='review', runtime_identity='cpu', unreturned_resource_possible=True))
    try:
        if operation == 'finish':
            guardian.finish_setup_leaf(permit, owners=(Owner('first'), None, Owner('last')))
        else:
            guardian.fail_setup_leaf(permit, partial_owners=(Owner('first'), None, Owner('last')))
    except Exception as exc:
        print(operation, 'exception', type(exc).__name__, str(exc))
    gc.collect()
    print(operation, 'before_quarantine', guardian.state.value,
          'unknown_owner', guardian.snapshot()['ownership_scope_unknown'],
          'resource_count', guardian.resource_count, 'finalizers', markers)
    guardian.quarantine('review_cleanup')
    gc.collect()
    print(operation, 'after_quarantine', guardian.state.value,
          'resource_count', guardian.resource_count, 'finalizers', markers)
