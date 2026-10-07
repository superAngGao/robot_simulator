"""Cold CPU source/guardian composition; fixed instrumentation is test-only."""

from pathlib import Path

import pytest

from tests.unit.optics.test_native_bootstrap import _internal
from tests.unit.optics.test_native_sources import release as _release_fixture

release = _release_fixture
PEER = Path(__file__).resolve().parents[2] / "support" / "guarded_peer.py"


def _probe(release, lane, fault):
    # The isolated instrumenting driver is not an admitted product source or
    # callback. All repository production modules use the REAL held-byte loader.
    return _internal(
        release,
        lane=lane,
        after=rf"""
import socket
import weakref
from dataclasses import replace
entry._verify(session)
assert len(session.modules) == 6
assert session.lane == {lane!r}
fault = {fault!r}
original_error = RuntimeError('fixed_source_fault_' + fault)
entry.original_error = original_error
channel, remote = socket.socketpair()
configuration = {{'protocol_version': 'q54.closed-worker.guarded-wait.v1',
 'session': 'b'*32, 'device_ordinal': 1, 'ack_timeout_ms': 1000, 'operation_timeout_ms': 5000}}
original_exec = entry._CapturedSourceLoader.exec_module
original_compile = entry._CapturedSourceLoader.source_to_code
observed = []
partial = []

def source_compile(loader, data, path, *args, **kwargs):
    if loader.name in entry._SETUP_DEFINITIONS:
        controller = session.bound_controller()
        module = session.modules[loader.name][0]
        assert controller.guardian.contains_exact_owner(module)
        assert controller._guard_client.state == 'ARMED'
        observed.append(loader.name)
        partial.append(weakref.ref(module))
        if fault == 'compile' or (fault == 'caught_nested' and loader.name.endswith('native_contract')):
            raise original_error
        if fault == 'body' and loader.name.endswith('native_setup'):
            # Deliberate test-only body mutation, never normal frozen execution.
            data += b'\nimport sys\nraise sys.modules["__main__"].original_error\n'
        if fault == 'caught_nested' and loader.name.endswith('native_setup'):
            data = b'try:\n from . import native_contract\nexcept RuntimeError:\n pass\n'
    return original_compile(loader, data, path, *args, **kwargs)

def observe_exec(loader, module):
    original_exec(loader, module)
    if loader.name == 'optical_worker.guarded_lifecycle':
        kind = module.NativeGuardedSetupLifecycle
        original_adopt = kind.adopt_host_owner
        def adopt(self, owner, **kwargs):
            if ((fault == 'adopt_first' and kwargs['role'] == 'captured_source_session') or
                (fault == 'adopt_late' and kwargs['role'] == 'source_snapshot') or
                (fault == 'module_adopt' and kwargs['operation_id'] == 'optical_worker_native_setup')):
                raise original_error
            return original_adopt(self, owner, **kwargs)
        kind.adopt_host_owner = adopt

entry._CapturedSourceLoader.source_to_code = source_compile
entry._CapturedSourceLoader.exec_module = observe_exec
binding_error = None
try:
    session.bind_guarded_custody(channel, configuration)
except BaseException as error:
    binding_error = error
controller = entry._SOURCE_CUSTODY[session][0]
assert len(session.modules) == 10
assert controller is session.bound_controller()
assert entry._SOURCE_CUSTODY[session][2] is not controller._guard_client.configuration
assert entry._SOURCE_CUSTODY[session][3] is controller._guard_client.configuration

# Test-only peer executes outside the captured release; no test module is
# installed in sys.modules or admitted by the repository source finder.
peer_spec = importlib.util.spec_from_file_location('fixed_test_peer', {str(PEER)!r})
peer_module = importlib.util.module_from_spec(peer_spec)
peer_spec.loader.exec_module(peer_module)
peer = peer_module.Peer(configuration=controller._guard_client.configuration,
                        channel_pair=(channel, remote))
try:
    if fault in ('adopt_first', 'adopt_late'):
        assert binding_error is original_error
    else:
        assert binding_error is None
        if fault == 'configuration_copy':
            controller._guard_client._configuration = replace(controller._guard_client.configuration)
        elif fault == 'snapshot_copy':
            session.snapshot = replace(session.snapshot)
        elif fault == 'audit':
            original_audit = session.audit
            def audit():
                if session.phase == 'importing' and len(session.modules) == 12:
                    raise original_error
                return original_audit()
            session.audit = audit
        try:
            session.import_setup_definitions()
            if fault == 'duplicate_import':
                session.import_setup_definitions()
            elif fault == 'duplicate_bind':
                session.bind_guarded_custody(channel, configuration)
            elif fault == 'runtime_import':
                import torch
        except BaseException:
            assert fault != 'success'
        else:
            assert fault == 'success', fault
    if fault == 'success':
        assert observed == ['optical_worker.native_setup', 'optical_worker.native_contract']
        assert len(session.modules) == 12
        assert not set(sys.modules) & {{'numpy', 'torch', 'warp'}}
        session.audit()
        assert all(controller.guardian.contains_exact_owner(x[0]) for x in session.modules.values())
        final = controller.terminate(fault='setup_failure', detection_phase='fixed_cpu_test',
                                     emergency_target_synchronize=None).lifecycle.to_payload()
        assert final['summary']['disposition'] == 'cancelled_before_submission_exit'
        assert final['summary']['normal_complete_count'] == 1
        assert final['summary']['terminal_arm_attempt_count'] == 1
    else:
        assert session.failed
        final = controller.termination_evidence.lifecycle.to_payload()
        assert final['summary']['custody'] == 'full_quarantine'
        assert final['summary']['terminal_arm_attempt_count'] == 0
        completed = fault in ('duplicate_import', 'duplicate_bind', 'runtime_import')
        assert final['summary']['normal_complete_count'] == int(completed)
        injected = ('adopt_first', 'adopt_late', 'module_adopt', 'compile', 'body', 'caught_nested', 'audit')
        if fault in injected:
            assert any(value is original_error for value in controller._failure_holds)
        assert all(ref() is not None for ref in partial)
        # Restoring the ordinary flag cannot erase retained failures or tickets.
        session.failed = False
        try:
            session.import_setup_definitions()
        except BaseException:
            pass
        else:
            raise AssertionError('failed source session revived')
        assert controller.evidence().to_payload() == final
    assert final['summary']['emergency_sync_attempt_count'] == 0
    assert final['summary']['entered_leaf_count'] == 0
    assert not set(sys.modules) & {{'numpy', 'torch', 'warp'}}
    print('source_custody_checked')
finally:
    peer.close()
""",
    )


@pytest.mark.parametrize(
    "fault",
    [
        "success",
        "adopt_first",
        "adopt_late",
        "module_adopt",
        "compile",
        "body",
        "caught_nested",
        "audit",
        "configuration_copy",
        "snapshot_copy",
        "duplicate_import",
        "duplicate_bind",
        "runtime_import",
    ],
)
@pytest.mark.parametrize("lane", ["warp", "cuda"])
def test_actual_captured_source_custody(release, fault, lane):
    result = _probe(release, lane, fault)
    assert result[0] == 0, result[2].decode()
    assert result[1].splitlines() == [b"source_custody_checked"]
