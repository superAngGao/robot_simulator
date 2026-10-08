"""Binding/state unit tests with the real guard protocol, simulated CUDA leaves.

No real backend/native module is imported. Source/import custody is unit-double
scaffolding; the complete cold/GPU driver remains a separate mandatory gate.
"""

from __future__ import annotations

import gc
import sys
import weakref
from types import ModuleType, SimpleNamespace

import pytest

from optical_worker import native_bootstrap as bootstrap
from optical_worker import native_setup
from optical_worker.guarded_lifecycle import NativeGuardedSetupLifecycle
from optical_worker.wait_protocol import GuardConfiguration
from tests.support.guarded_peer import Peer


@pytest.fixture
def runtime_session(monkeypatch):
    peer = Peer()
    controller = NativeGuardedSetupLifecycle(lane_id="cuda", channel=peer.channel, configuration=peer.config)
    module = ModuleType("torch")
    module._C = ModuleType("torch._C")
    module.device = lambda kind, ordinal: SimpleNamespace(type=kind, index=ordinal)
    module.cuda = SimpleNamespace(Stream=lambda *, device: SimpleNamespace(device=device, cuda_stream=412))
    monkeypatch.setitem(sys.modules, "torch", module)
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    session = object.__new__(bootstrap._SourceSession)
    session.runtime_observation = bootstrap._RuntimeObservation(session)
    session.lane, session.phase = "cuda", "materialized"
    session.failed, session.failures = False, []
    session.runtime_owners = []
    session.canonical_input = object()
    materialization_steps = []
    session.external = SimpleNamespace(
        audit=lambda: None, phase="materialization", records={"torch": {"module": module}},
        materialization_steps=materialization_steps,
        bind_materialization_case=lambda case: materialization_steps.append(("bind", case)),
        finish_materialization=lambda: materialization_steps.append(("finish", None)),
    )

    def check():
        bootstrap._require(not session.failed, "unit source session failed")
        controller._check_running()

    session.check_live, session.audit = check, check
    monkeypatch.setitem(
        bootstrap._SOURCE_CUSTODY,
        session,
        (
            controller,
            controller.guardian,
            peer.config,
            controller._guard_client.configuration,
            peer.channel,
            NativeGuardedSetupLifecycle,
            GuardConfiguration,
        ),
    )
    controller.adopt_host_owner(session, role="unit_source", operation_id="unit_source")
    controller.adopt_host_owner(module, role="unit_module", operation_id="unit_module")
    controller.adopt_host_owner(
        session.canonical_input, role="unit_source_arrays", operation_id="unit_source_arrays"
    )
    try:
        yield session, controller, module, peer
    finally:
        if controller.termination_evidence is None:
            controller.terminate(
                fault="setup_failure", detection_phase="unit_end", emergency_target_synchronize=lambda: None
            )
        bootstrap._RUNTIME_BINDINGS.pop(session, None)
        bootstrap._UPLOADED_INPUTS.pop(session, None)
        peer.close()


def test_initialization_mints_exact_adopted_single_use_binding(runtime_session):
    session, controller, module, _ = runtime_session
    binding = session.initialize_native_runtime()
    assert type(binding) is bootstrap._RuntimeSetupBinding
    assert session.phase == "runtime_ready"
    assert binding.module is module and binding.runtime is module._C
    assert binding.source is session.canonical_input
    assert binding.stream_identity == (controller.target_device_ordinal, 412)
    controller.require_adopted(binding, binding.stream, binding.target, binding.runtime)
    with controller.segment():
        assert session._consume_runtime_binding(binding) == (module, binding.target, binding.stream)
    assert bootstrap._RUNTIME_BINDINGS[session] is None
    with pytest.raises(bootstrap.SourceBootstrapError, match="consumed"):
        with controller.segment():
            session._consume_runtime_binding(binding)
    assert (
        controller.termination_evidence.lifecycle.to_payload()["summary"]["emergency_sync_attempt_count"] == 0
    )


@pytest.mark.parametrize(
    "field", ["session", "controller", "module", "runtime", "target", "stream", "source"]
)
def test_object_setattr_binding_resource_drift_rejects_before_upload(runtime_session, field):
    session, controller, _, _ = runtime_session
    binding = session.initialize_native_runtime()
    object.__setattr__(binding, field, object())
    with pytest.raises(bootstrap.SourceBootstrapError, match="identity drift"):
        with controller.segment():
            session._consume_runtime_binding(binding)
    assert bootstrap._RUNTIME_BINDINGS[session] is None
    assert controller.termination_evidence is not None


def test_file_loader_owned_native_root_is_not_adopted_twice(runtime_session):
    session, controller, module, _ = runtime_session
    controller.adopt_host_owner(module._C, role="external_import_owner", operation_id="external_import_owner")
    before = controller.guardian.resource_count
    binding = session.initialize_native_runtime()
    assert binding.runtime is module._C
    assert controller.guardian.resource_count == before + 3  # target, stream, binding only
    before = controller.guardian.resource_count
    session._hold_runtime_owner(module._C, "alternate_root_role")
    assert controller.guardian.resource_count == before


def test_handmade_same_type_binding_does_not_copy_registration(runtime_session):
    session, controller, _, _ = runtime_session
    original = session.initialize_native_runtime()
    impostor = object.__new__(bootstrap._RuntimeSetupBinding)
    for name in bootstrap._RuntimeSetupBinding.__slots__:
        object.__setattr__(impostor, name, getattr(original, name))
    with pytest.raises(bootstrap.SourceBootstrapError, match="copied"):
        with controller.segment():
            session._consume_runtime_binding(impostor)
    assert bootstrap._RUNTIME_BINDINGS[session] is None


@pytest.mark.parametrize("drift", ["handle", "device", "module", "runtime", "source", "plugin"])
def test_actual_resource_drift_rejects_even_with_unchanged_binding(runtime_session, monkeypatch, drift):
    session, controller, module, _ = runtime_session
    binding = session.initialize_native_runtime()
    if drift == "handle":
        binding.stream.cuda_stream += 1
    elif drift == "device":
        binding.stream.device = SimpleNamespace(type="cuda", index=0)
    elif drift == "module":
        monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    elif drift == "runtime":
        module._C = ModuleType("torch._C")
    elif drift == "source":
        session.canonical_input = object()
    else:
        monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
    with pytest.raises(bootstrap.SourceBootstrapError):
        with controller.segment():
            session._consume_runtime_binding(binding)
    assert bootstrap._RUNTIME_BINDINGS[session] is None


def test_binding_has_no_public_mutable_or_reconstructible_constructor(runtime_session):
    session, _, _, _ = runtime_session
    with pytest.raises(TypeError, match="issued"):
        bootstrap._RuntimeSetupBinding()
    binding = session.initialize_native_runtime()
    with pytest.raises(TypeError, match="frozen"):
        binding.stream = object()
    with pytest.raises(TypeError, match="frozen"):
        del binding.source


def test_partial_runtime_init_marks_unknown_and_never_arms_fallback_sync(runtime_session):
    session, controller, module, _ = runtime_session
    failure = RuntimeError("native init may have started before stream return")

    def fail_stream(*, device):
        controller.require_adopted(device, module._C)
        raise failure

    module.cuda.Stream = fail_stream
    with pytest.raises(RuntimeError, match="may have started"):
        session.initialize_native_runtime()
    evidence = controller.termination_evidence.lifecycle.to_payload()
    assert evidence["summary"]["disposition"] == "multi_or_unknown_device_quarantine_exit"
    assert evidence["summary"]["emergency_sync_attempt_count"] == 0
    assert any(error is failure for error in session.failures)
    assert bootstrap._RUNTIME_BINDINGS[session] is None
    assert session.phase != "runtime_ready"


def test_returned_stream_survives_adoption_failure_in_original_session_graph(runtime_session, monkeypatch):
    session, controller, module, _ = runtime_session
    returned, destroyed = [], []

    class Stream:
        def __init__(self, device):
            self.device, self.cuda_stream = device, 417

        def __del__(self):
            destroyed.append(True)

    def create_stream(*, device):
        stream = Stream(device)
        returned.append(weakref.ref(stream))
        return stream

    def reject_adoption(*_, **__):
        raise RuntimeError("injected owner-registration failure")

    module.cuda.Stream = create_stream
    monkeypatch.setattr(controller, "adopt_returned_device_owner", reject_adoption)
    with pytest.raises(RuntimeError, match="owner-registration"):
        session.initialize_native_runtime()
    gc.collect()
    assert not destroyed
    assert returned[0]() is not None
    assert any(owner is returned[0]() for owner in session.runtime_owners)
    assert controller.guardian.state == "TRANSFERRED_TO_QUARANTINE"
    assert (
        controller.termination_evidence.lifecycle.to_payload()["summary"]["emergency_sync_attempt_count"] == 0
    )


def test_upload_entry_rejects_stream_rebind_without_calling_shared_core(runtime_session, monkeypatch):
    session, controller, _, _ = runtime_session
    binding = session.initialize_native_runtime()
    before = sum(
        event["operation"] == "descriptor_observed"
        for event in controller.evidence().to_payload()["transition_trace"]
    )
    called = []
    monkeypatch.setattr(native_setup, "_upload_cuda_native_input", lambda *a, **k: called.append(True))
    binding.stream.cuda_stream += 1
    with pytest.raises(bootstrap.SourceBootstrapError, match="stream storage drift"):
        session.upload_native_input(binding)
    assert not called
    after = sum(
        event["operation"] == "descriptor_observed"
        for event in controller.evidence().to_payload()["transition_trace"]
    )
    assert after == before
    assert session.phase != "uploaded"


@pytest.mark.parametrize("drift", ["prepared", "stream", "source"])
def test_readback_entry_rechecks_uploaded_and_runtime_identity_before_helper(
    runtime_session, monkeypatch, drift
):
    from optical_worker import native_readback

    session, controller, _, _ = runtime_session
    binding = session.initialize_native_runtime()
    record = bootstrap._RUNTIME_BINDINGS[session]
    with controller.segment():
        session._consume_runtime_binding(binding)
    prepared = object()
    controller.adopt_host_owner(prepared, role="unit_prepared", operation_id="unit_prepared")
    session.prepared_input, session.phase = prepared, "uploaded"
    bootstrap._UPLOADED_INPUTS[session] = (record, prepared)
    called = []
    monkeypatch.setattr(
        native_readback, "_readback_prepared_native_input", lambda *a, **k: called.append(True)
    )
    if drift == "prepared":
        session.prepared_input = object()
    elif drift == "stream":
        binding.stream.cuda_stream += 1
    else:
        session.canonical_input = object()
    with pytest.raises(bootstrap.SourceBootstrapError):
        session.validate_native_input()
    assert not called
    assert session.phase != "input_validated"
    assert controller.termination_evidence is not None


def test_readback_session_orders_one_validation_without_reminting_upload(runtime_session, monkeypatch):
    """Session ordering only; helper math/custody is tested against its real body."""
    from optical_worker import native_readback

    session, controller, module, _ = runtime_session
    binding = session.initialize_native_runtime()
    record = bootstrap._RUNTIME_BINDINGS[session]
    with controller.segment():
        session._consume_runtime_binding(binding)
    prepared = SimpleNamespace(manifest=SimpleNamespace(sha256="a" * 64))
    report = SimpleNamespace(content_sha256="b" * 64, topology_sha256="c" * 64, validated_arrays=("unit",))
    controller.adopt_host_owner(prepared, role="unit_prepared", operation_id="unit_prepared")
    session.prepared_input, session.phase = prepared, "uploaded"
    bootstrap._UPLOADED_INPUTS[session] = (record, prepared)
    calls = []

    def helper(source, actual_prepared, *, lifecycle, runtime, target, stream):
        controller._require_setup()
        assert source is session.canonical_input and actual_prepared is prepared
        assert lifecycle is controller and runtime is module
        assert target is binding.target and stream is binding.stream
        assert session.phase == "validating_input"
        calls.append(True)
        controller.adopt_host_owner(report, role="unit_report", operation_id="unit_report")
        return report

    monkeypatch.setattr(native_readback, "_readback_prepared_native_input", helper)
    assert session.validate_native_input() is report
    assert session.phase == "input_validated" and session.validation_report is report
    assert bootstrap._RUNTIME_BINDINGS[session] is None
    completed = controller.evidence().to_payload()["summary"]["normal_complete_count"]
    assert completed == 3  # initialization, unit consumption, validation
    with pytest.raises(bootstrap.SourceBootstrapError, match="single-use"):
        session.validate_native_input()
    assert calls == [True]
    assert controller.evidence().to_payload()["summary"]["normal_complete_count"] == completed


@pytest.mark.parametrize("phase", ["materialized", "runtime_ready"])
def test_readback_session_cannot_precede_its_upload(runtime_session, monkeypatch, phase):
    from optical_worker import native_readback

    session, controller, _, _ = runtime_session
    if phase == "runtime_ready":
        session.initialize_native_runtime()
    calls = []
    monkeypatch.setattr(
        native_readback, "_readback_prepared_native_input", lambda *a, **k: calls.append(True)
    )
    before = controller.evidence().to_payload()["summary"]["normal_complete_count"]
    with pytest.raises(bootstrap.SourceBootstrapError, match="single-use"):
        session.validate_native_input()
    assert not calls
    assert controller.evidence().to_payload()["summary"]["normal_complete_count"] == before


@pytest.mark.parametrize("case_id", ["asymmetric_synthetic_7x3_v1", "synthetic_dynamic_rigid_960x640_v1"])
def test_session_materializes_actual_canonical_source_without_readiness(runtime_session, case_id):
    from optical_worker.workload_contract import frozen_workload_content

    session, controller, _, _ = runtime_session
    session.phase = "runtime_configured"
    manifest = session.materialize_native_source(case_id)
    expected = frozen_workload_content(case_id)
    assert session.phase == "materialized"
    assert session.external.materialization_steps == [("bind", case_id), ("finish", None)]
    assert manifest is session.canonical_input.manifest
    assert manifest.scene_contract_sha256 == expected.scene_contract_sha256
    assert session.render_source.packed_scene is session.canonical_input.packed_scene_owner
    assert manifest.fixed_render_fixture_sha256 == expected.fixed_render_fixture_sha256
    assert manifest.transform_replay_sha256 == expected.transform_replay_sha256
    controller.require_adopted(session.render_source, session.canonical_input)
    assert session not in bootstrap._RUNTIME_BINDINGS
    assert not hasattr(session.canonical_input, "ready_event")


@pytest.mark.parametrize("case_id", ["physics_body_triangle_960x640_v1", "invented_case"])
def test_shipping_or_unknown_case_cannot_enter_materialization(runtime_session, case_id):
    session, controller, _, _ = runtime_session
    session.phase = "runtime_configured"
    original_source = session.canonical_input
    with pytest.raises(bootstrap.SourceBootstrapError, match="outside C2d3"):
        session.materialize_native_source(case_id)
    assert session.canonical_input is original_source
    assert session.external.materialization_steps == []
    trace = controller.evidence().to_payload()["transition_trace"]
    assert not any(event["operation"] == "descriptor_observed" for event in trace)


@pytest.mark.parametrize("fail_copy", [False, True])
def test_shared_cuda_copy_enters_device_before_stream_and_restores_both(fail_copy):
    trace = []
    stream, source = SimpleNamespace(device=object()), object()

    class Scope:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            trace.append("enter_" + self.name)

        def __exit__(self, *_):
            trace.append("exit_" + self.name)

    def device_scope(value):
        assert value is stream.device
        return Scope("device")

    def stream_scope(value):
        assert value is stream
        assert trace == ["enter_device"]
        return Scope("stream")

    def copy(value, *, non_blocking):
        assert value is source and non_blocking is True
        trace.append("copy")
        if fail_copy:
            raise RuntimeError("copy failed after possible submission")

    torch = SimpleNamespace(cuda=SimpleNamespace(device=device_scope, stream=stream_scope))
    destination = SimpleNamespace(copy_=copy)
    if fail_copy:
        with pytest.raises(RuntimeError, match="possible submission"):
            native_setup._torch_copy_exact(torch, destination, source, stream)
    else:
        native_setup._torch_copy_exact(torch, destination, source, stream)
    assert trace == ["enter_device", "enter_stream", "copy", "exit_stream", "exit_device"]
