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
from tests.unit.optics.test_runtime_import_admission import _disabled_rows, _disabled_sources
from tests.support.runtime_custody_launch import capture_runtime_custody_inputs
from optical_worker.native_sources import CATALOG_PATH

release = _release_fixture
PEER = Path(__file__).resolve().parents[2] / "support" / "guarded_peer.py"


def _custody_inputs(release, lane, session_character):
    root, revision = release
    return capture_runtime_custody_inputs({
        "root": str(root), "revision": revision, "lane": lane,
        "catalog_sha256": hashlib.sha256((root / CATALOG_PATH).read_bytes()).hexdigest(),
        "case_id": "asymmetric_synthetic_7x3_v1",
        "guard_configuration": {"protocol_version": "q54.closed-worker.guarded-wait.v1",
            "session": session_character * 32, "device_ordinal": 1, "ack_timeout_ms": 1000, "operation_timeout_ms": 5000},
    })


def _optional_environment(root):
    # Dependency fixtures only. They exercise production cold-session custody;
    # they do not replace the separately required actual Go2 materialization.
    p = _environment(root, "success").to_payload()
    p["files"], p["modules"] = [], []
    p["absent_probes"], p["publications"] = [], []
    p["disabled_optional_modules"] = _disabled_rows()
    disabled = {r["name"] for r in p["disabled_optional_modules"]}
    for name, source in _disabled_sources().items():
        package = name in {"trimesh", "trimesh.ray", "trimesh.path"}
        relative = name.replace(".", "/") + ("/__init__.py" if package else ".py")
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = source.encode()
        path.write_bytes(data)
        p["files"].append(
            {
                "root": "site",
                "path": relative,
                "kind": "python",
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            }
        )
        if name not in disabled:
            p["modules"].append(
                {
                    "name": name,
                    "root": "site",
                    "path": relative,
                    "is_package": package,
                    "phase": "materialization",
                }
            )
    p["files"].sort(key=lambda row: (row["root"], row["path"]))
    p["modules"].sort(key=lambda row: row["name"])
    return RuntimeEnvironmentContract.from_payload(p)


@pytest.mark.parametrize("lane", ["warp", "cuda"])
@pytest.mark.parametrize("fault", ["success", "unused", "adoption", "issuance", "confirmation"])
def test_optional_configuration_and_import_use_distinct_guarded_segments(release, tmp_path, lane, fault):
    environment = _optional_environment(tmp_path.resolve())
    inputs = _custody_inputs(release, lane, "e")
    result = _internal(
        release,
        lane=lane,
        after=f"""
import socket
entry._verify(session)
channel, remote = socket.socketpair()
session.bind_guarded_custody(channel, {{
 'protocol_version': 'q54.closed-worker.guarded-wait.v1', 'session': 'e'*32,
 'device_ordinal': 1, 'ack_timeout_ms': 1000, 'operation_timeout_ms': 5000}})
controller = session.bound_controller()
peer_spec = importlib.util.spec_from_file_location('fixed_peer', {str(PEER)!r})
peer_module = importlib.util.module_from_spec(peer_spec)
peer_spec.loader.exec_module(peer_module)
peer = peer_module.Peer(configuration=controller._guard_client.configuration,
                        channel_pair=(channel, remote))
try:
    session.configure_runtime_custody({environment.to_payload()!r}, {inputs!r})
    session.import_setup_definitions()
    session.configure_external_environment({environment.to_payload()!r}, {environment.sha256!r})
    admission = session.external
    assert session.phase == 'runtime_configured'
    assert admission.session is session and admission.controller is controller
    assert not admission.disabled_records and not admission.events
    assert not any(n.startswith('trimesh') for n in sys.modules)
    fault = {fault!r}
    if fault == 'adoption':
        original_adopt = controller.adopt_host_owner
        def adopt(owner, **kwargs):
            if type(owner) is ImportError:
                raise RuntimeError('denial_adoption_fault')
            return original_adopt(owner, **kwargs)
        controller.adopt_host_owner = adopt
    elif fault in {{'issuance', 'confirmation'}}:
        original_record = admission._record_disabled
        def record(name, operation):
            selected = ('disabled_optional' if fault == 'issuance'
                        else 'disabled_fallback_confirmed')
            if operation == selected:
                raise RuntimeError('denial_evidence_fault')
            return original_record(name, operation)
        admission._record_disabled = record
    try:
        with controller.segment():
            if fault != 'unused':
                import trimesh
            session.audit()
            admission.audit()
    except BaseException:
        assert fault in {{'adoption','issuance','confirmation'}}
        assert session.failed and admission.failed
        evidence = controller.termination_evidence.lifecycle.to_payload()
        assert evidence['summary']['custody'] == 'full_quarantine'
        assert evidence['summary']['normal_complete_count'] == 2
        assert evidence['summary']['emergency_sync_attempt_count'] == 0
        assert evidence['summary']['terminal_arm_attempt_count'] == 0
        record = admission.disabled_records['trimesh.ray.ray_pyembree']
        assert record['error'] is not None and record['requester'] is not None
        # Even a failed adoption leaves the error strongly held by the
        # admission graph, itself already adopted before any import.
        assert any(x is record['error'] for x in admission.retained)
        assert controller.guardian.contains_exact_owner(admission)
    else:
        assert fault in {{'success','unused'}}
        if fault == 'success':
            for name in admission.disabled:
                assert admission.disabled_records[name]['state'] == 'fallback_confirmed'
                assert [e['operation'] for e in admission.events if e['name'] == name] == [
                    'disabled_optional','disabled_fallback_confirmed']
        else:
            assert not admission.disabled_records and not admission.events
        runtime_record = session.runtime_observation.checkpoint()
        assert runtime_record.to_payload()['summary']['external_event_count'] == len(admission.events)
        session.runtime_observation.freeze_adoptions()
        evidence = controller.terminate(fault='setup_failure', detection_phase='cpu_optional_fixture',
                                         emergency_target_synchronize=None).lifecycle.to_payload()
        assert evidence['summary']['disposition'] == 'cancelled_before_submission_exit'
        assert evidence['summary']['normal_complete_count'] == 3
    selected, selection = session.runtime_observation.final()
    assert selected is not None and selection['selection'] == 'current', session.runtime_observation.failures
    assert selected.to_payload()['lifecycle'] == evidence
    assert evidence['summary']['entered_leaf_count'] == 0
    assert not {{'numpy','warp','torch','benchmarks'}}.intersection(sys.modules)
    assert all(name not in admission.loaders and name not in admission.records
               for name in admission.disabled)
    print('cold_optional_segments_checked')
finally:
    peer.close()
""",
    )
    assert result[0] == 0, result[2].decode()
    assert result[1].splitlines() == [b"cold_optional_segments_checked"]


def _environment(root, fault):
    package = root / "c2dep"
    package.mkdir()
    source = (
        "import sys\nfrom types import ModuleType\n"
        "from importlib.util import find_spec\nassert find_spec('c2missing') is None\n"
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
            "disabled_optional_modules": [],
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
            "native_globals": [],
            "module_replacements": [],
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
    inputs = _custody_inputs(release, lane, "d")
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
    session.configure_runtime_custody({environment.to_payload()!r}, {inputs!r})
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
        assert len([e for e in admission.events if e['kind'] == 'absent_probe']) == 2
        runtime_record = session.runtime_observation.checkpoint()
        assert runtime_record.to_payload()['summary']['external_event_count'] == len(admission.events)
        session.runtime_observation.freeze_adoptions()
        terminal = controller.terminate(fault='setup_failure', detection_phase='cpu_external_fixture',
                                         emergency_target_synchronize=None).lifecycle.to_payload()
        assert terminal['summary']['disposition'] == 'cancelled_before_submission_exit'
        assert terminal['summary']['normal_complete_count'] == 3
    selected, selection = session.runtime_observation.final()
    assert selected is not None and selection['selection'] == 'current', session.runtime_observation.failures
    assert selected.to_payload()['lifecycle'] == terminal
    assert terminal['summary']['entered_leaf_count'] == 0
    assert not {{'numpy','warp','torch','benchmarks'}}.intersection(sys.modules)
    print('cold_external_composition_checked')
finally:
    peer.close()
""",
    )
    assert result[0] == 0, result[2].decode()
    assert result[1].splitlines() == [b"cold_external_composition_checked"]
