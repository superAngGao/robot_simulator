"""Actual cold collector failure cuts; native loader bodies are CPU doubles."""

import hashlib
import importlib.machinery
import sys
from pathlib import Path

import pytest

from optical_worker.runtime_environment import RUNTIME_ENVIRONMENT_VERSION, RuntimeEnvironmentContract
from tests.unit.optics.test_native_bootstrap import _internal
from tests.unit.optics.test_native_sources import release as _release_fixture
from tests.unit.optics.test_runtime_import_cold import PEER, _custody_inputs

release = _release_fixture


@pytest.mark.parametrize("fault", ["return_before_adopt", "before_call_complete", "after_call_complete",
                                 "after_observer_append", "partial_observer_append"])
def test_original_native_failure_windows_select_honest_cut(release, tmp_path, fault):
    root = tmp_path.resolve()
    package = root / "c2dep"
    package.mkdir()
    sources = [("c2dep", "c2dep/__init__.py", "python", b"from . import _native\n"),
               ("c2dep._native", "c2dep/_native" + importlib.machinery.EXTENSION_SUFFIXES[0], "extension", b"CPU native double")]
    files, modules = [], []
    for name, relative, kind, data in sources:
        (root / relative).write_bytes(data)
        files.append({"root": "site", "path": relative, "kind": kind, "size": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()})
        modules.append({"name": name, "root": "site", "path": relative, "is_package": name == "c2dep",
                        "phase": "materialization"})
    executable = Path(sys.executable).resolve()
    environment = RuntimeEnvironmentContract.from_payload({
        "schema_version": RUNTIME_ENVIRONMENT_VERSION,
        "interpreter": {"path": executable.as_posix(), "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                        "implementation": "cpython", "version": list(sys.version_info[:3])},
        "roots": [{"id": "site", "path": root.as_posix(), "kind": "site"}],
        "files": sorted(files, key=lambda row: row["path"]), "modules": sorted(modules, key=lambda row: row["name"]),
        "distributions": [], "absent_probes": [], "disabled_optional_modules": [], "publications": [],
        "module_replacements": [], "namespaces": [],
        "native_globals": [{"name": "c2dep.shared", "observer": "c2dep._native", "operation": "exec", "phase": "materialization"}],
        "settings": {"TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "warp_quiet": True},
    })
    inputs = _custody_inputs(release, "cuda", "a")
    result = _internal(release, lane="cuda", after=f'''
import socket
entry._verify(session)
channel, remote = socket.socketpair()
session.bind_guarded_custody(channel, {inputs['configuration']!r})
controller = session.bound_controller()
peer_spec = importlib.util.spec_from_file_location('fixed_peer', {str(PEER)!r})
peer_module = importlib.util.module_from_spec(peer_spec)
peer_spec.loader.exec_module(peer_module)
peer = peer_module.Peer(configuration=controller._guard_client.configuration, channel_pair=(channel, remote))
try:
    session.configure_runtime_custody({environment.to_payload()!r}, {inputs!r})
    session.import_setup_definitions()
    session.configure_external_environment({environment.to_payload()!r}, {environment.sha256!r})
    observation, admission = session.runtime_observation, session.external
    preceding = observation.last_validated
    fault = {fault!r}
    original_error = RuntimeError('fixed_' + fault)
    returned = []
    def create(delegate, spec):
        module = types.ModuleType(spec.name)
        returned.append(module)
        return module
    def execute(delegate, module):
        sys.modules['c2dep.shared'] = types.ModuleType('c2dep.shared')
    importlib.machinery.ExtensionFileLoader.create_module = create
    importlib.machinery.ExtensionFileLoader.exec_module = execute
    if fault == 'return_before_adopt':
        original = controller.adopt_host_owner
        def adopt(owner, **kwargs):
            if returned and owner is returned[0]:
                raise original_error
            return original(owner, **kwargs)
        controller.adopt_host_owner = adopt
    elif fault == 'before_call_complete':
        original = admission._native_call_state
        def state(frame, value, module):
            if frame['loader'].name == 'c2dep._native' and frame['operation'] == 'exec' and value == 'completed':
                raise original_error
            return original(frame, value, module)
        admission._native_call_state = state
    elif fault == 'after_call_complete':
        original = admission.record
        def record(operation, name, kind):
            if name == 'c2dep._native' and operation == 'completed':
                raise original_error
            return original(operation, name, kind)
        admission.record = record
    else:
        original = observation.append
        def append(kind, details):
            if kind == 'native_global':
                if fault == 'after_observer_append':
                    original(kind, details)
                else:
                    observation.issued.append(b'interrupted commit')
                raise original_error
            return original(kind, details)
        observation.append = append
    try:
        with controller.segment('setup'):
            import c2dep
    except RuntimeError as error:
        assert error is original_error
    else:
        raise AssertionError('fixed failure did not execute')
    assert session.failed and admission.failed and returned
    assert any(owner is returned[0] for owner in admission.retained)
    assert any(owner is original_error for owner in session.failures)
    selected, selection = observation.final()
    expected = 'current' if fault in {{'return_before_adopt', 'after_observer_append'}} else 'prior_validated_cut'
    assert selection['selection'] == expected, (selection, observation.failures)
    assert selected is not None
    if expected == 'prior_validated_cut':
        assert selected is preceding and selected.encoded == preceding.encoded
        assert observation.failures
    else:
        assert selected.to_payload()['lifecycle'] == controller.termination_evidence.lifecycle.to_payload()
        token = observation.token(returned[0])
        row = next(row for row in selected.to_payload()['objects'] if row['token'] == token)
        if fault == 'return_before_adopt':
            assert row['guardian_sequence'] is None
        assert any(event['kind'] == 'returned_owner' and event['details']['object_token'] == token
                   for event in selected.to_payload()['journal'])
    assert observation.final()[0] is selected
    assert controller.termination_evidence.lifecycle.to_payload()['summary']['release_status'] == 'not_authorized'
    assert not {{'numpy', 'torch', 'warp'}} & set(sys.modules)
    print('actual_failure_cut_checked')
finally:
    peer.close()
''')
    assert result[0] == 0, result[2].decode()
    assert result[1].splitlines() == [b"actual_failure_cut_checked"]
