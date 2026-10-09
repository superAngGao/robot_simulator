"""Cold production source session; no backend import or multiprocessing spawn."""
import copy
import json
from pathlib import Path
from textwrap import indent

import pytest

from optical_worker.runtime_environment import validate_interpreter_alias_observations

from tests.unit.optics.test_native_bootstrap import _internal as _source_internal
from tests.unit.optics.test_native_sources import release as _release_fixture

release = _release_fixture
PEER = Path(__file__).resolve().parents[2] / "support" / "guarded_peer.py"


def _internal(release, before="", after=""):
    # The aggregate observes aliases through the real bound controller. Keep
    # the original cold bootstrap and assertions, adding its current custody
    # prerequisite and actual guard peer rather than disabling observation.
    prefix = "\nentry._verify(session)\n"
    if not after:
        return _source_internal(release, before=before)
    assert after.startswith(prefix)
    body = indent(after[len(prefix):] + "\nalias_assertions_completed = True\n", "        ")
    return _source_internal(release, before=before, after=f"""
entry._verify(session)
import socket
channel, remote = socket.socketpair()
session.bind_guarded_custody(channel, {{
 'protocol_version': 'q54.closed-worker.guarded-wait.v1', 'session': 'c'*32,
 'device_ordinal': 1, 'ack_timeout_ms': 1000, 'operation_timeout_ms': 5000}})
controller = session.bound_controller()
from optical_worker.lifecycle import WorkerSetupProtocolError
peer_spec = importlib.util.spec_from_file_location('fixed_peer', {str(PEER)!r})
peer_module = importlib.util.module_from_spec(peer_spec)
peer_spec.loader.exec_module(peer_module)
peer = peer_module.Peer(configuration=controller._guard_client.configuration,
                        channel_pair=(channel, remote))
alias_assertions_completed = False
try:
    try:
        with controller.segment('setup'):
{indent(body, '    ')}
    except WorkerSetupProtocolError:
        # Expected poison must prevent normal guard completion. A failure in
        # any original test assertion/body is never swallowed by this wrapper.
        assert alias_assertions_completed and session.failed
        assert controller.termination_evidence is not None
    assert alias_assertions_completed
    if controller.termination_evidence is None:
        controller.terminate(fault='setup_failure', detection_phase='alias_cpu_fixture',
                             emergency_target_synchronize=None)
    summary = controller.termination_evidence.lifecycle.to_payload()['summary']
    assert summary['entered_leaf_count'] == 0
    assert summary['emergency_sync_attempt_count'] == 0
    # Healthy CPU-only cancellation still supervises release/exit. A poisoned
    # normal guard must quarantine directly, without another ARM.
    assert summary['terminal_arm_attempt_count'] == (0 if session.failed else 1)
    assert summary['normal_complete_count'] == (0 if session.failed else 1)
    assert summary['disposition'] == (
        'quarantined_process_exit' if session.failed else 'cancelled_before_submission_exit')
    assert not {{'numpy', 'torch', 'warp'}} & set(sys.modules)
finally:
    peer.close()
""")


@pytest.mark.parametrize("trap", ["class", "getattribute"])
@pytest.mark.parametrize("entrypoint", ["audit", "fail", "poisoned"])
def test_invalid_publisher_cannot_call_metadata_or_recover_failure(release, trap, entrypoint):
    result = _internal(release, after=f"""
entry._verify(session)
calls = []
original = RuntimeError('original failure')
class BadPublisher:
    @property
    def __class__(self):
        calls.append('class')
        raise RuntimeError('publisher callback')
    def __getattribute__(self, name):
        if {trap!r} == 'getattribute':
            calls.append(name)
            raise RuntimeError('publisher callback')
        return object.__getattribute__(self, name)
value = BadPublisher()
if {entrypoint!r} == 'poisoned': session.fail(original)
sys.modules['multiprocessing'] = value
if {entrypoint!r} == 'audit':
    try:
        session.audit()
    except entry.SourceBootstrapError as error:
        original = error
    else:
        raise AssertionError('invalid publisher accepted')
else:
    session.fail(original)
    session.observe_interpreter_alias('audit', final=True)
assert not calls
assert session.failed and any(error is original for error in session.failures)
assert any(owner is value for owner in entry._INTERPRETER_ALIAS_CUSTODY[session][2])
assert not session.interpreter_alias_events
del sys.modules['multiprocessing']
try:
    session.audit()
except entry.SourceBootstrapError:
    pass
else:
    raise AssertionError('poisoned source session reused')
assert not calls and not session.interpreter_alias_events
print('callback_free_sticky_failure')
""")
    assert result[0] == 0, result[2].decode()
    assert result[1].strip() == b"callback_free_sticky_failure"


def test_exact_cold_main_alias_is_observed_once(release):
    result = _internal(release, after="""
entry._verify(session)
assert '__mp_main__' not in sys.modules
import multiprocessing
session.audit()
assert sys.modules['__mp_main__'] is session.bootstrap_owner
assert session.interpreter_alias_admitted()
assert not session.is_interpreter_name('__mp_main__')
before = session.interpreter_alias_observations()
session.audit()
assert session.interpreter_alias_observations() == before
assert len(before['events']) == 1
assert not {'torch','numpy','warp'}.intersection(sys.modules)
print(entry.json.dumps(before))
""")
    assert result[0] == 0, result[2].decode()
    payload = json.loads(result[1])
    validate_interpreter_alias_observations(payload, require_observed=True)
    for mutation in ("extra", "duplicate", "no_resolution", "source", "token", "checkpoint", "operation"):
        changed = copy.deepcopy(payload)
        if mutation == "extra": changed["extra"] = True
        elif mutation == "duplicate": changed["events"].append(dict(changed["events"][0]))
        elif mutation == "no_resolution": changed["resolution"] = None
        elif mutation == "source": changed["source_path"] += ".other"
        elif mutation == "token": changed["events"][0]["main_token"] = True
        elif mutation == "checkpoint": changed["events"][0]["checkpoint"] = "evidence"
        else: changed["events"][0]["operation"] = "exec"
        with pytest.raises(ValueError):
            validate_interpreter_alias_observations(changed, require_observed=True)


@pytest.mark.parametrize("mutation", ["preload", "copied_main", "main", "publisher", "spec", "removed", "event", "event_copy"])
def test_alias_identity_and_projection_mutations_poison(release, mutation):
    before = "sys.modules['__mp_main__'] = sys.modules['__main__']" if mutation == "preload" else ""
    if before:
        result = _internal(release, before=before)
        assert result[0] != 0 and b"preloaded" in result[2]
        return
    result = _internal(release, after=f"""
entry._verify(session)
import multiprocessing
session.audit()
mutation = {mutation!r}
if mutation == 'copied_main': sys.modules['__mp_main__'] = types.ModuleType('__main__')
elif mutation == 'main': sys.modules['__main__'] = types.ModuleType('__main__')
elif mutation == 'publisher': sys.modules['multiprocessing'] = types.ModuleType('multiprocessing')
elif mutation == 'spec': multiprocessing.__spec__ = object()
elif mutation == 'removed': del sys.modules['__mp_main__']
elif mutation == 'event': session.interpreter_alias_events[0]['main_token'] = 2
elif mutation == 'event_copy': session.interpreter_alias_events[0] = dict(session.interpreter_alias_events[0])
try:
    session.audit()
except entry.SourceBootstrapError:
    assert session.failed and session.failures
else:
    raise AssertionError('alias mutation accepted')
print('alias_rejected')
""")
    assert result[0] == 0, result[2].decode()
    assert result[1].strip() == b"alias_rejected"


@pytest.mark.parametrize("mutation", ["injected", "partial", "issuance", "poison", "descriptor"])
def test_alias_boundaries_do_not_repair_missing_or_failed_observation(release, mutation):
    result = _internal(release, after=f"""
entry._verify(session)
mutation = {mutation!r}
if mutation == 'injected':
    sys.modules['__mp_main__'] = session.bootstrap_owner
elif mutation == 'poison':
    session.fail(RuntimeError('earlier failure'))
    sys.modules['__mp_main__'] = session.bootstrap_owner
    session.observe_interpreter_alias('audit', final=True)
    assert not session.interpreter_alias_observations()['events']
else:
    import multiprocessing
    session.audit()
    if mutation == 'partial': multiprocessing.__spec__._initializing = True
    elif mutation == 'issuance': sys.modules['__mp_main__'] = types.ModuleType('__main__')
    else:
        descriptor, *rest = entry._INTERPRETER_ALIAS_CUSTODY[session]
        entry._INTERPRETER_ALIAS_CUSTODY[session] = ((*descriptor[:3], '0'*64, descriptor[4]), *rest)
if mutation != 'poison':
    try:
        session.interpreter_alias_observations() if mutation == 'issuance' else session.audit()
    except entry.SourceBootstrapError:
        assert session.failed
    else:
        raise AssertionError('alias boundary accepted mutation')
print('boundary_checked')
""")
    assert result[0] == 0, result[2].decode()
    assert result[1].strip() == b"boundary_checked"


@pytest.mark.parametrize("key,value", [("main_token", True), ("main_token", 1.0),
                                     ("publisher_token", 2.0), ("index", False)])
@pytest.mark.parametrize("poisoned", [False, True])
def test_equal_value_type_rewrite_rejects_live_and_failed_alias_evidence(release, key, value, poisoned):
    result = _internal(release, after=f"""
entry._verify(session)
import multiprocessing
session.audit()
if {poisoned!r}: session.fail(RuntimeError('original failure'))
session.interpreter_alias_events[0][{key!r}] = {value!r}
try:
    session.interpreter_alias_observations()
except entry.SourceBootstrapError:
    assert session.failed
else:
    raise AssertionError('equal-value type rewrite accepted')
print('typed_fact_rejected')
""")
    assert result[0] == 0, result[2].decode()
    assert result[1].strip() == b"typed_fact_rejected"


@pytest.mark.parametrize("state", ["removed", "none", "initializing_removed", "resolution_only"])
def test_resolved_but_unobserved_publisher_cannot_complete_audit(release, state):
    result = _internal(release, after=f"""
entry._verify(session)
state = {state!r}
if state == 'resolution_only':
    session.finder.find_spec('multiprocessing')
    session.observe_interpreter_alias('request_entry')  # legitimate pre-publication interval
else:
    import multiprocessing
    assert not session.interpreter_alias_events
    if state == 'initializing_removed': multiprocessing.__spec__._initializing = True
    del sys.modules['multiprocessing']
    del sys.modules['__mp_main__']
    if state == 'none': sys.modules['multiprocessing'] = None
assert entry._INTERPRETER_ALIAS_CUSTODY[session][3] is not None
try:
    session.audit()
except entry.SourceBootstrapError:
    assert session.failed and not session.interpreter_alias_events
else:
    raise AssertionError('incomplete publisher completed final audit')
print('incomplete_publisher_rejected')
""")
    assert result[0] == 0, result[2].decode()
    assert result[1].strip() == b"incomplete_publisher_rejected"
