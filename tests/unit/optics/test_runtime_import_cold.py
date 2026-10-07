"""Real captured-session composition with fixed CPU-only external dependencies.

This checks the actual cold loader/controller wiring, not CUDA initialization or
the required real NumPy/Torch/Warp admission matrix. The test peer and injected
fixture are outside product source admission and are never a worker IPC request.
"""

import hashlib
import sys
from pathlib import Path

import pytest

from optical_worker.runtime_environment import RUNTIME_ENVIRONMENT_VERSION, RuntimeEnvironmentContract
from tests.unit.optics.test_native_bootstrap import _internal
from tests.unit.optics.test_native_sources import release as _release_fixture

release = _release_fixture
PEER = Path(__file__).resolve().parents[2] / "support" / "guarded_peer.py"


def _environment(root, fault):
    package = root / "c2dep"
    package.mkdir()
    source = (
        "import sys\nfrom types import ModuleType\n"
        "try:\n import c2missing\nexcept ModuleNotFoundError:\n pass\n"
        "created = ModuleType('c2dep.created')\n"
        "sys.modules['c2dep.created'] = created\n"
        "sys.modules['c2dep.alias'] = created\n"
    )
    if fault == "started_import":
        source += "raise ImportError('body_has_started')\n"
    elif fault == "unexpected_absence":
        source += "try:\n import c2unexpected\nexcept ImportError:\n pass\n"
    elif fault == "wrong_alias":
        source += "sys.modules['c2dep.alias'] = ModuleType('c2dep.alias')\n"
    data = source.encode()
    (package / "__init__.py").write_bytes(data)
    executable = Path(sys.executable).resolve(strict=True)
    return RuntimeEnvironmentContract.from_payload(
        {
            "schema_version": RUNTIME_ENVIRONMENT_VERSION,
            "interpreter": {
                "path": executable.as_posix(),
                "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                "implementation": sys.implementation.name,
                "version": list(sys.version_info[:3]),
            },
            "roots": [{"id": "site", "path": root.as_posix(), "kind": "site"}],
            "files": [
                {
                    "root": "site",
                    "path": "c2dep/__init__.py",
                    "kind": "python",
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size": len(data),
                }
            ],
            "distributions": [],
            "modules": [
                {
                    "name": "c2dep",
                    "root": "site",
                    "path": "c2dep/__init__.py",
                    "is_package": True,
                    "phase": "materialization",
                }
            ],
            "absent_probes": [{"name": "c2missing", "requester": "c2dep", "phase": "materialization"}],
            "publications": [
                {
                    "name": name,
                    "kind": kind,
                    "creator": "c2dep",
                    "publisher": "c2dep",
                    "attributes": ["created"],
                    "phase": "materialization",
                }
                for name, kind in (("c2dep.alias", "module_alias"), ("c2dep.created", "created_module"))
            ],
            "namespaces": [],
            "settings": {"TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "warp_quiet": True},
        }
    )


@pytest.mark.parametrize("lane", ["warp", "cuda"])
@pytest.mark.parametrize(
    "fault", ["success", "started_import", "unexpected_absence", "wrong_alias", "identity_drift"]
)
def test_real_cold_session_adopts_external_objects_and_fails_closed(release, tmp_path, lane, fault):
    environment = _environment(tmp_path.resolve(), fault)
    result = _internal(
        release,
        lane=lane,
        after=f"""
import socket
entry._verify(session)
channel, remote = socket.socketpair()
session.bind_guarded_custody(channel, {{
 'protocol_version': 'q54.closed-worker.guarded-wait.v1', 'session': 'd'*32,
 'device_ordinal': 1, 'ack_timeout_ms': 1000, 'operation_timeout_ms': 5000}})
controller = session.bound_controller()
peer_spec = importlib.util.spec_from_file_location('fixed_peer', {str(PEER)!r})
peer_module = importlib.util.module_from_spec(peer_spec)
peer_spec.loader.exec_module(peer_module)
peer = peer_module.Peer(configuration=controller._guard_client.configuration,
                        channel_pair=(channel, remote))
try:
    session.import_setup_definitions()
    session.configure_external_environment({environment.to_payload()!r}, {environment.sha256!r})
    assert session.phase == 'runtime_configured'
    admission = session.external
    assert controller is admission.controller
    assert admission.session is session
    fault = {fault!r}
    try:
        with controller.segment():
            import c2dep
            if fault == 'identity_drift':
                sys.modules['c2dep.alias'] = types.ModuleType('c2dep.alias')
            session.audit()
            admission.audit()
    except BaseException:
        assert fault != 'success'
        assert session.failed and admission.failed
        terminal = controller.termination_evidence.lifecycle.to_payload()
        assert terminal['summary']['custody'] == 'full_quarantine'
        assert terminal['summary']['normal_complete_count'] == 2
        assert terminal['summary']['emergency_sync_attempt_count'] == 0
        assert terminal['summary']['terminal_arm_attempt_count'] == 0
        assert admission.records['c2dep']['module'] is not None
        assert controller.guardian.contains_exact_owner(admission.records['c2dep']['module'])
        assert admission.failures
    else:
        assert fault == 'success'
        controller.require_adopted(c2dep, c2dep.created)
        assert sys.modules['c2dep.alias'] is sys.modules['c2dep.created'] is c2dep.created
        assert 'c2missing' not in sys.modules
        terminal = controller.terminate(fault='setup_failure', detection_phase='cpu_external_fixture',
                                         emergency_target_synchronize=None).lifecycle.to_payload()
        assert terminal['summary']['disposition'] == 'cancelled_before_submission_exit'
        assert terminal['summary']['normal_complete_count'] == 3
    assert terminal['summary']['entered_leaf_count'] == 0
    assert not {{'numpy','warp','torch','benchmarks'}}.intersection(sys.modules)
    print('cold_external_composition_checked')
finally:
    peer.close()
""",
    )
    assert result[0] == 0, result[2].decode()
    assert result[1].splitlines() == [b"cold_external_composition_checked"]
