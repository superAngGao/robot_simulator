"""Actual CPU collector retention; no native loader or complete input fixture."""

import json
from types import SimpleNamespace

import pytest

from optical_worker import native_bootstrap as bootstrap
from optical_worker.compilation_contract import cuda_role_anchor
from optical_worker.guarded_lifecycle import NativeGuardedSetupLifecycle
from optical_worker.lifecycle import WorkerSetupProtocolError
from tests.support.guarded_peer import Peer


class Session:
    def __init__(self, controller):
        self.controller = controller
        self.trace = []
        self.runtime_observation = bootstrap._RuntimeObservation(self)

    def bound_controller(self):
        return self.controller


def root_for(session):
    # Minimal upstream descriptor for this low-level retention test only. The
    # full compilation parser does not accept it as a six-phase input record.
    encoded = b'{"objects":[]}'
    record = SimpleNamespace(encoded=encoded, to_payload=lambda: json.loads(encoded))
    root = bootstrap._CompileCustodyRoot(session, record, (), {})
    session.controller.adopt_host_owner(root, role="cuda_compile_custody", operation_id="cuda_compile_custody")
    bootstrap._CUDA_COMPILATIONS[id(session)] = session, root
    root.root_token = session.runtime_observation.token(root)
    session.runtime_observation.append("compile_root_registered", {"root_token": root.root_token})
    return root


def test_actual_token_registry_retains_once_before_role_adoption():
    peer = Peer()
    controller = NativeGuardedSetupLifecycle(lane_id="cuda", channel=peer.channel, configuration=peer.config)
    session = Session(controller)
    try:
        for _ in range(6):
            with controller.segment():
                pass
        with controller.segment():
            root = root_for(session)
            module = object()
            obs = session.runtime_observation
            token = obs.token(module)
            assert root.held == [module] and root.held[0] is module
            retained = obs.journal[-1]
            assert retained["kind"] == "compile_owner_retained" and retained["details"]["owner_token"] == token
            role, operation = cuda_role_anchor("robot_sim_cuda_scene_v1", "module_token")
            controller.adopt_host_owner(module, role=role, operation_id=operation)
            obs.freeze_adoptions()
            sequence = obs.adoptions[token]
            assert controller.guardian._owners[sequence] is module
            assert controller._guard_events[-1]["operation"] == "owner_adopted"
            assert retained["lifecycle_event_index"] < controller._guard_events[-1]["index"]
            assert obs.token(module) == token and len(root.held) == 1
            root.audit_retention()
            root.held[0] = object()
            with pytest.raises(bootstrap.SourceBootstrapError, match="identity"):
                root.audit_retention()
            root.held[0] = module
    finally:
        bootstrap._CUDA_COMPILATIONS.pop(id(session), None)
        peer.close()


def test_failed_role_adoption_preserves_original_return_through_adopted_root():
    peer = Peer()
    controller = NativeGuardedSetupLifecycle(lane_id="cuda", channel=peer.channel, configuration=peer.config)
    session = Session(controller)
    returned = object()
    try:
        with pytest.raises(WorkerSetupProtocolError):
            with controller.segment():
                root = root_for(session)
                token = session.runtime_observation.token(returned)
                controller.adopt_host_owner(returned, role="INVALID", operation_id="invalid")
        obs = session.runtime_observation
        obs.freeze_adoptions()
        assert root.held[0] is returned and controller.guardian.contains_exact_owner(root)
        assert token not in obs.adoptions and not controller.guardian.contains_exact_owner(returned)
        assert all(e["kind"] != "compile_step" for e in obs.journal)
        assert controller.evidence().to_payload()["summary"]["guarded_segment_failed"]
        root.audit_retention()
    finally:
        bootstrap._CUDA_COMPILATIONS.pop(id(session), None)
        peer.close()
