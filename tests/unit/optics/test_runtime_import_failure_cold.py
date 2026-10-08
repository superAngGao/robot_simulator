"""Actual cold session failure custody; CPU fixtures, not native/GPU acceptance."""

import hashlib
from pathlib import Path

import pytest

from optical_worker.runtime_environment import RuntimeEnvironmentContract
from tests.unit.optics.test_native_bootstrap import _internal
from tests.unit.optics.test_native_sources import release as _release_fixture
from tests.unit.optics.test_runtime_import_cold import _environment, _custody_inputs

release = _release_fixture
PEER = Path(__file__).resolve().parents[2] / "support/guarded_peer.py"


@pytest.mark.parametrize("lane", ["warp", "cuda"])
def test_late_parent_publication_survives_guarded_failure(release, tmp_path, lane):
    root = tmp_path.resolve()
    payload = _environment(root, "success").to_payload()
    payload["files"], payload["modules"] = [], []
    payload["absent_probes"], payload["publications"] = [], []
    sources = {
        "c2dep": (
            "import sys, types\n"
            "try:\n from . import child\nexcept ImportError: pass\n"
            "sys.modules['c2dep.late'] = types.ModuleType('late')\n"
            "raise ValueError('outer_after_late_publication')\n"
        ),
        "c2dep.child": "raise ImportError('inner_started_failure')\n",
    }
    for name, source in sources.items():
        relative = name.replace(".", "/") + ("/__init__.py" if name == "c2dep" else ".py")
        data = source.encode()
        (root / relative).write_bytes(data)
        payload["files"].append(
            {
                "root": "site",
                "path": relative,
                "kind": "python",
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            }
        )
        payload["modules"].append(
            {
                "name": name,
                "root": "site",
                "path": relative,
                "is_package": name == "c2dep",
                "phase": "materialization",
            }
        )
    payload["files"].sort(key=lambda row: (row["root"], row["path"]))
    environment = RuntimeEnvironmentContract.from_payload(payload)
    inputs = _custody_inputs(release, lane, "b")
    result = _internal(
        release,
        lane=lane,
        after=f"""
import socket, weakref, gc
entry._verify(session)
channel, remote = socket.socketpair()
session.bind_guarded_custody(channel, {{
 'protocol_version': 'q54.closed-worker.guarded-wait.v1', 'session': 'b'*32,
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
    try:
        with controller.segment():
            import c2dep
    except ValueError as error:
        assert str(error) == 'outer_after_late_publication'
        observed_error = error
    else:
        raise AssertionError('outer failure absent')
    evidence = controller.termination_evidence.lifecycle.to_payload()
    summary = evidence['summary']
    assert summary['custody'] == 'full_quarantine'
    assert summary['normal_complete_count'] == 2
    assert summary['emergency_sync_attempt_count'] == 0
    assert summary['terminal_arm_attempt_count'] == 0
    assert summary['entered_leaf_count'] == 0
    assert controller.guardian.contains_exact_owner(admission)
    late = sys.modules['c2dep.late']
    assert 'c2dep.late' not in admission.records
    assert any(x is late for x in admission.retained), 'late owner missing from held graph'
    assert any(x is observed_error for x in admission.failures), 'outer error omitted'
    assert any(str(x) == 'inner_started_failure' for x in admission.failures)
    reference = weakref.ref(late)
    del sys.modules['c2dep.late']
    del late
    gc.collect()
    assert reference() is not None
    before = controller.evidence().to_payload()
    admission.fail(observed_error, 'c2dep')
    admission.fail(observed_error, 'c2dep', native_create=True)
    assert controller.evidence().to_payload() == before
    assert not {{'numpy','torch','warp','benchmarks'}}.intersection(sys.modules)
    print('late_custody_checked')
finally:
    peer.close()
""",
    )
    assert result[0] == 0, result[2].decode()
    assert result[1].splitlines() == [b"late_custody_checked"]
