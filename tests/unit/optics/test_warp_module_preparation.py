"""Real lifecycle/entry tests; simulated Warp leaves do not prove GPU behavior."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from optical_worker import native_bootstrap as bootstrap, native_setup, warp_setup
from optical_worker.guarded_lifecycle import NativeGuardedSetupLifecycle
from optical_worker.wait_protocol import GuardConfiguration
from tests.support.guarded_peer import Peer


@pytest.fixture
def prepared_session(monkeypatch):
    peer = Peer()
    controller = NativeGuardedSetupLifecycle(lane_id="warp", channel=peer.channel, configuration=peer.config)
    session = object.__new__(bootstrap._SourceSession)
    session.lane, session.phase = "warp", "input_validated"
    session.failed, session.failures, session.runtime_owners = False, [], []
    session.warp_preparation_started = False
    session.canonical_input, session.prepared_input, session.validation_report = object(), object(), object()
    session.runtime_observation = bootstrap._RuntimeObservation(session)
    session.modules = {"optical_worker.native_setup": (native_setup, None, "complete")}
    wp = ModuleType("warp")
    runtime = object()
    wp.config = SimpleNamespace(quiet=True)
    target = SimpleNamespace(is_cuda=True, ordinal=controller.target_device_ordinal, runtime=runtime)
    stream = SimpleNamespace(device=target, cuda_stream=412)
    session.external = SimpleNamespace(
        audit=lambda: controller._require_setup(), records={"warp": {"module": wp},
            "warp._src.context": {"module": SimpleNamespace(runtime=runtime)}})
    monkeypatch.setitem(sys.modules, "warp", wp)
    monkeypatch.setitem(bootstrap._SOURCE_CUSTODY, session,
        (controller, controller.guardian, peer.config, controller._guard_client.configuration,
         peer.channel, NativeGuardedSetupLifecycle, GuardConfiguration))

    def check():
        bootstrap._require(not session.failed, "source session failed")
        bootstrap._require(not controller._terminating and not controller._replay.failed, "terminal source")

    session.check_live = check
    session.audit = lambda: controller._require_setup()
    binding = object.__new__(bootstrap._RuntimeSetupBinding)
    identity = (target.ordinal, 412)
    for name, value in dict(session=session, controller=controller, module=wp, runtime=runtime,
                            target=target, stream=stream, stream_identity=identity,
                            source=session.canonical_input).items():
        object.__setattr__(binding, name, value)
    runtime_record = (binding, controller, session.external, wp, runtime, target, stream,
                      identity, session.canonical_input)
    monkeypatch.setitem(bootstrap._UPLOADED_INPUTS, session, (runtime_record, session.prepared_input))
    for value in (session, wp, runtime, target, stream, binding, session.canonical_input, session.prepared_input):
        controller.adopt_host_owner(value, role="unit_owner", operation_id="unit_owner")
    # Model completed upload: subsequent IDLE failure has outstanding exact-target work.
    with controller.segment("setup"):
        session._runtime_leaf("unit_upload", "allocation")
    syncs = []
    monkeypatch.setattr(native_setup, "_warp_synchronize_target_exact",
                        lambda actual_wp, actual_target: syncs.append((actual_wp, actual_target)))
    checkpoints = []
    monkeypatch.setattr(session.runtime_observation, "checkpoint", lambda: checkpoints.append(True))
    imports = []
    import optics
    for name in ("device_scene", "device_bvh", "warp_native_render"):
        module = ModuleType("optics." + name)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        monkeypatch.setattr(optics, name, module, raising=False)
        imports.append(module)
    monkeypatch.setattr(warp_setup, "validate_compilation_inputs", lambda actual: controller._require_setup())
    monkeypatch.setattr(warp_setup, "definition_owners", lambda actual, index: (imports[index], imports[index], ()))
    records = []
    cache = SimpleNamespace(records=records)
    controller.adopt_host_owner(cache, role="unit_cache", operation_id="unit_cache")
    monkeypatch.setattr(warp_setup, "validate_warp_cache", lambda actual, actual_wp: cache)

    def compile_one(actual, actual_binding, index):
        assert actual is session and actual_binding is binding and index == len(records)
        controller._require_setup()
        session._runtime_leaf("unit_compile", "runtime_import_or_init")
        record = SimpleNamespace(guard=(controller._guard_reference["sequence"], "setup"))
        session._hold_runtime_owner(record, "unit_compile_result")
        records.append(record)
        return record

    monkeypatch.setattr(warp_setup, "compile_module", compile_one)
    try:
        yield SimpleNamespace(session=session, controller=controller, binding=binding, records=records,
                              syncs=syncs, checkpoints=checkpoints, compile_one=compile_one)
    finally:
        if controller.termination_evidence is None:
            controller.terminate(fault="setup_failure", detection_phase="unit_end",
                                 emergency_target_synchronize=lambda: None)
        warp_setup._MODULE_SETS.pop(session, None)
        peer.close()


def test_import_then_three_nonnested_compile_guards_issue_one_held_set(prepared_session):
    p = prepared_session
    owner = p.session.prepare_warp_modules()
    assert owner.session is p.session and owner.binding is p.binding
    assert owner.modules == tuple(p.records)
    assert p.session.phase == "warp_modules_ready" and p.checkpoints == [True]
    p.controller.require_adopted(owner, *p.records)
    summary = p.controller.evidence().to_payload()["summary"]
    assert summary["normal_complete_count"] == 5  # upload, imports, three compiles
    assert p.syncs == []


@pytest.mark.parametrize("boundary", ["entry", "issuance", "repeat"])
def test_idle_failure_preserves_successful_guards_and_uses_existing_emergency(prepared_session, monkeypatch, boundary):
    p = prepared_session
    failure = RuntimeError("CPU issuance failed")
    if boundary == "entry":
        p.session.validation_report = None
        expected = 1
    elif boundary == "issuance":
        def issue(*args):
            assert p.controller._guard_client.state == "IDLE"
            raise failure
        monkeypatch.setattr(warp_setup, "issue_module_set", issue)
        expected = 5
    else:
        p.session.prepare_warp_modules()
        expected = 5
    with pytest.raises((bootstrap.SourceBootstrapError, RuntimeError)) as caught:
        p.session.prepare_warp_modules()
    if boundary == "issuance":
        assert caught.value is failure
    summary = p.controller.termination_evidence.lifecycle.to_payload()["summary"]
    assert summary["normal_complete_count"] == expected
    assert summary["emergency_sync_attempt_count"] == 1
    assert summary["disposition"] == "aborted_synced_exit"
    assert p.syncs == [(p.binding.module, p.binding.target)]
    assert not any(e["operation"] == "guard_failed" for e in p.controller._guard_events)
    assert p.session.failed


@pytest.mark.parametrize("compromised", ["unknown", "missing_owner", "binding_drift"])
def test_idle_failure_with_unprovable_custody_quarantines_without_sync(prepared_session, compromised):
    p = prepared_session
    if compromised == "unknown":
        p.controller.mark_unknown_device_scope(source="unit_fault", reason="unknown_native_effect")
    elif compromised == "missing_owner":
        owners = p.controller.guardian._owners
        index = next(i for i, value in enumerate(owners) if value is p.binding.stream)
        owners[index] = object()
    else:
        object.__setattr__(p.binding, "stream", object())
    p.session.validation_report = None
    with pytest.raises(bootstrap.SourceBootstrapError):
        p.session.prepare_warp_modules()
    summary = p.controller.termination_evidence.lifecycle.to_payload()["summary"]
    assert summary["emergency_sync_attempt_count"] == 0
    assert p.controller.guardian.state == "TRANSFERRED_TO_QUARANTINE" and p.syncs == []
    assert not any(e["operation"] == "guard_failed" for e in p.controller._guard_events)


@pytest.mark.parametrize("stage", ["import", "first_compile", "last_compile"])
def test_guarded_failure_preserves_original_error_and_partial_roots(prepared_session, monkeypatch, stage):
    p = prepared_session
    failure = RuntimeError("partial Warp work")
    if stage == "import":
        def definitions(*args):
            p.controller._require_setup()
            raise failure
        monkeypatch.setattr(warp_setup, "definition_owners", definitions)
        expected = 1
    else:
        selected = 0 if stage == "first_compile" else 2
        def compile_one(*args):
            result = p.compile_one(*args)
            if args[2] == selected:
                raise failure
            return result
        monkeypatch.setattr(warp_setup, "compile_module", compile_one)
        expected = selected + 2
    with pytest.raises(RuntimeError) as caught:
        p.session.prepare_warp_modules()
    assert caught.value is failure
    summary = p.controller.termination_evidence.lifecycle.to_payload()["summary"]
    assert summary["normal_complete_count"] == expected
    assert summary["emergency_sync_attempt_count"] == 0
    assert p.controller.guardian.state == "TRANSFERRED_TO_QUARANTINE" and p.syncs == []
    assert p.checkpoints == [] and p.session not in warp_setup._MODULE_SETS
    assert all(any(owner is record for owner in p.session.runtime_owners) for record in p.records)
