# Imported fixtures are registered by pytest.
# ruff: noqa: F401, F811
"""Final graph composition over real CUDA C3/arena control, with CPU native leaves."""

from dataclasses import replace

import pytest

from tests.unit.optics.test_native_optical_arena import _spec
from tests.unit.optics.test_native_render_preparation import graph_session
from tests.unit.optics.test_native_render_warmup import executable_graph
from tests.unit.optics.test_prepared_fixed_render_inputs import fixed_inputs, fixed_storage_session

pytestmark = [
    pytest.mark.parametrize("fail_at", [None]),
    pytest.mark.parametrize("diagnostic", [False, True]),
]


def _final_spec(x):
    m = x.session.prepared_input.manifest
    first = x.session.fixed_render_storage[1]
    return replace(
        _spec(boundary="R0"),
        case_id=m.case_id,
        case_sha256=m.case_sha256,
        native_input_manifest_sha256=m.sha256,
        native_readiness_sha256=x.graph.readiness.sha256,
        diagnostic_mode=x.graph.pass_mode,
        scene_contract_sha256=m.scene_contract_sha256,
        fixed_render_fixture_sha256=m.fixed_render_fixture_sha256,
        primitive_count=int(first.triangle_v0_world.shape[0]),
        node_capacity=first.num_bvh_nodes,
        execution_stream_token=f"torch-stream:1:{x.f.stream.cuda_stream}",
    )


@pytest.mark.parametrize("fixed_inputs", ["cuda"], indirect=True)
def test_final_cuda_graph_reuses_storage_and_builds_actual_c3_plans(executable_graph, monkeypatch):
    from optical_worker import cuda_arena as adapter
    from optical_worker import native_setup

    x = executable_graph
    observed = x.session.run_fixed_render_warmups()
    # The event double has completed real guard waits; expose its opaque native
    # handle metadata now for the production arena's initialized-event check.
    x.graph.completion_event.cuda_event = id(x.graph.completion_event)
    original_storage = x.session.fixed_render_storage
    original_readiness = x.graph.readiness.to_payload()
    calls_before = tuple(x.calls)
    allocated_before = tuple(x.f.state.arrays)
    # CPU tensor doubles use NumPy dtypes; only native dtype spelling is adapted.
    original_signature = adapter._storage_signature

    class TensorMetadata:
        def __init__(self, tensor):
            self.tensor = tensor
            self.dtype = "torch." + str(tensor.dtype)

        def __getattr__(self, name):
            return getattr(self.tensor, name)

    monkeypatch.setattr(
        adapter,
        "_storage_signature",
        lambda name, tensor, *, owner_token: original_signature(
            name, TensorMetadata(tensor), owner_token=owner_token
        ),
    )
    from optics import cuda_arena

    validate_signature = cuda_arena._validate_c3_storage_signature
    monkeypatch.setattr(
        cuda_arena,
        "_validate_c3_storage_signature",
        lambda signature, *, tensor, owner_token: validate_signature(
            signature, tensor=TensorMetadata(tensor), owner_token=owner_token
        ),
    )
    monkeypatch.setattr(
        adapter,
        "validate_prepared_native_input",
        lambda *args: pytest.fail("captured final C3 performed a second unguarded readback"),
    )
    with x.controller.segment("setup"):
        graph = native_setup._prepare_fixed_final_graph(
            x.session, x.p.runtime, x.p.device, x.f.stream, spec=_final_spec(x), sample_count=3
        )
    assert native_setup._FINAL_RENDER_GRAPHS[x.session] is graph
    assert native_setup._FINAL_READINESS_ORIGINS[x.session] is graph.origin
    assert graph.origin.warmups is observed
    assert graph.origin.readiness is not x.graph.readiness
    assert graph.origin.readiness.warmup_frames_completed == 10
    assert x.graph.readiness.to_payload() == original_readiness
    assert graph.factory is not x.graph.factory
    assert graph.factory.context is x.graph.factory.context
    assert x.session.fixed_render_storage is original_storage
    assert all(bundle.buffers is original_storage[-1] for _, bundle in graph.frames)
    assert all(
        sample._active._transaction._completion_event is x.graph.completion_event for sample in graph.samples
    )
    assert len(graph.samples) == 3 and len(graph.sha256) == 64
    assert [sample._target.evidence.render_sample_index for sample in graph.samples] == [0, 1, 2]
    assert [sample._control_plan.evidence.expected_generation for sample in graph.samples] == [1, 2, 3]
    assert tuple(x.calls) == calls_before and tuple(x.f.state.arrays) == allocated_before
    assert x.controller.guardian.state == "CONSTRUCTING"
    assert x.controller.quiescence_evidence is None
    assert graph.arena.lifecycle_evidence().terminal_event_record_count == 0


@pytest.mark.parametrize("fixed_inputs", ["cuda"], indirect=True)
@pytest.mark.parametrize("attack", ["copied_warmup", "mode", "count", "unobserved"])
def test_final_graph_rejects_invalid_upstream_before_constructing(executable_graph, attack):
    from optical_worker import native_setup

    x = executable_graph
    if attack != "unobserved":
        evidence = x.session.run_fixed_render_warmups()
        if attack == "copied_warmup":
            x.session.fixed_render_warmups = replace(evidence)
    else:
        x.session.fixed_render_warmups = None
    spec = _final_spec(x)
    if attack == "mode":
        spec = replace(spec, shadows=False)
    before = tuple(x.calls)
    with pytest.raises((ValueError, TypeError)):
        with x.controller.segment("setup"):
            native_setup._prepare_fixed_final_graph(
                x.session,
                x.p.runtime,
                x.p.device,
                x.f.stream,
                spec=spec,
                sample_count=True if attack == "count" else 1,
            )
    assert tuple(x.calls) == before
    assert x.session not in native_setup._FINAL_RENDER_GRAPHS


def _adapt_warp_cpu_metadata(monkeypatch):
    from types import SimpleNamespace as NS

    from optical_worker.provenance import NativeStorageSignature
    from optics import warp_arena
    from optics import warp_native_render as render

    factory_type = render.WarpDirectLightRenderCommandFactory
    prepare = factory_type.prepare

    def prepare_final(cls, **kwargs):
        factory = prepare(**kwargs)
        factory.prepared_input = kwargs["prepared_input"]
        factory.readiness = kwargs["readiness"]
        return factory

    monkeypatch.setattr(factory_type, "prepare", classmethod(prepare_final))
    seal = factory_type.seal_camera

    def seal_final(factory, **kwargs):
        bundle = seal(factory, **kwargs)
        binding = bundle.binding
        binding.buffers = kwargs["buffers"]
        binding.transform_frame_index = None
        binding.query = kwargs["camera"].query
        binding.camera = kwargs["camera"]
        binding.output_profile = kwargs["output_profile"]
        binding.shadows = kwargs["shadows"]
        binding.bvh = kwargs["bvh"]
        bundle.evidence = NS(sha256="b" * 64)
        return bundle

    monkeypatch.setattr(factory_type, "seal_camera", seal_final)

    # Resource arithmetic uses CPU backing allocations in this test; no real
    # Warp allocation/recorded-command validation is claimed by the doubles.
    def cpu_storage(name, array, device, owner):
        return NativeStorageSignature(
            name=name,
            backend="warp",
            device_ordinal=device.ordinal,
            storage_token=f"cpu-allocation:{id(array):x}",
            dtype=array.dtype.str,
            shape=array.shape,
            nbytes=array.capacity,
            active_start=0,
            active_count=array.capacity,
            owner_token=owner,
        )

    monkeypatch.setattr(warp_arena, "_storage_signature", cpu_storage)


@pytest.mark.parametrize("fixed_inputs", ["warp"], indirect=True)
def test_final_warp_graph_consumes_same_slot_and_new_readiness(executable_graph, monkeypatch):
    from optical_worker import native_setup

    x = executable_graph
    observed = x.session.run_fixed_render_warmups()
    _adapt_warp_cpu_metadata(monkeypatch)
    m = x.session.prepared_input.manifest
    bvh = x.session.fixed_render_storage[2]
    spec = replace(
        _spec(boundary="R0"),
        lane="warp",
        renderer_id="warp_bvh_direct_light",
        case_id=m.case_id,
        case_sha256=m.case_sha256,
        native_input_manifest_sha256=m.sha256,
        native_readiness_sha256=x.graph.readiness.sha256,
        diagnostic_mode=x.graph.pass_mode,
        scene_contract_sha256=m.scene_contract_sha256,
        fixed_render_fixture_sha256=m.fixed_render_fixture_sha256,
        primitive_count=bvh.num_primitives,
        node_capacity=bvh.num_nodes,
        execution_stream_token=f"warp-stream:1:{x.f.stream.cuda_stream}",
    )
    calls_before, arrays_before = tuple(x.calls), tuple(x.f.state.arrays)
    with x.controller.segment("setup"):
        graph = native_setup._prepare_fixed_final_graph(
            x.session, x.p.runtime, x.p.device, x.f.stream, spec=spec, sample_count=3
        )
    assert graph.origin.warmups is observed
    assert graph.factory is not x.graph.factory and graph.factory.readiness is graph.origin.readiness
    assert graph.origin.readiness.warmup_frames_completed == 10
    assert x.graph.readiness.warmup_frames_completed == 0
    assert all(bundle.binding.buffers is x.session.fixed_render_storage[-1] for _, bundle in graph.frames)
    assert all(sample._transaction._terminal.event is x.graph.completion_event for sample in graph.samples)
    assert [sample._plan.evidence.expected_generation for sample in graph.samples] == [1, 2, 3]
    assert [sample._transaction.parameters.to_payload()["frame_id"] for sample in graph.samples] == [1, 2, 3]
    assert tuple(x.calls) == calls_before and tuple(x.f.state.arrays) == arrays_before
    assert x.controller.guardian.state == "CONSTRUCTING" and x.controller.quiescence_evidence is None
    assert len(graph.sha256) == 64


@pytest.fixture
def final_session(executable_graph, monkeypatch):
    from optical_worker import cuda_arena as adapter
    from optics import cuda_arena

    x = executable_graph
    x.session.run_fixed_render_warmups()
    x.session.failed = False
    from optical_worker.arena import _admit_fixed_row

    x.session.phase = "materializing"
    x.session.fixed_row = _admit_fixed_row(
        x.session,
        row_id="r0_asymmetric_full_shadow",
        case_id=x.session.prepared_input.manifest.case_id,
        sample_count=2,
        pass_mode=x.graph.pass_mode,
    )
    x.session.phase = "fixed_render_warmups_ready"
    if x.session.lane == "warp":
        _adapt_warp_cpu_metadata(monkeypatch)
    else:
        x.graph.completion_event.cuda_event = id(x.graph.completion_event)
        original_signature = adapter._storage_signature
        validate_signature = cuda_arena._validate_c3_storage_signature

        class TensorMetadata:
            def __init__(self, tensor):
                self.tensor = tensor
                self.dtype = "torch." + str(tensor.dtype)

            def __getattr__(self, name):
                return getattr(self.tensor, name)

        monkeypatch.setattr(
            adapter,
            "_storage_signature",
            lambda name, tensor, *, owner_token: original_signature(
                name, TensorMetadata(tensor), owner_token=owner_token
            ),
        )
        monkeypatch.setattr(
            cuda_arena,
            "_validate_c3_storage_signature",
            lambda signature, *, tensor, owner_token: validate_signature(
                signature, tensor=TensorMetadata(tensor), owner_token=owner_token
            ),
        )
    return x


@pytest.mark.parametrize("fixed_inputs", ["cuda", "warp"], indirect=True)
@pytest.mark.parametrize("failure", [None, "sync", "disarmed", "stream_drift", "event_drift"])
def test_final_barrier_admits_only_after_exact_sync_and_ack(final_session, monkeypatch, failure):
    from optical_worker import native_setup
    from optical_worker.guarded_evidence import GuardedLifecycleEvidence

    x = final_session
    barrier_calls = []
    complete = x.controller._guard_client.complete
    original_prepare = native_setup._prepare_fixed_final_graph

    def prepare(*args, **kwargs):
        graph = original_prepare(*args, **kwargs)
        with pytest.raises(ValueError, match="acknowledged final"):
            graph.arena.acquire(graph.samples[0])
        return graph

    monkeypatch.setattr(native_setup, "_prepare_fixed_final_graph", prepare)

    def synchronize(target):
        assert target is x.p.device and x.f.state.device is target and x.f.state.stream is x.f.stream
        assert x.controller.guardian.state == "SEALED"
        assert x.controller.quiescence_evidence is None
        graph = native_setup._FINAL_RENDER_GRAPHS[x.session]
        with pytest.raises(ValueError, match="acknowledged final"):
            graph.arena.acquire(graph.samples[0])
        barrier_calls.append("sync")
        if failure == "sync":
            raise RuntimeError("injected final sync failure")
        if failure == "stream_drift":
            x.f.stream.cuda_stream += 1
        if failure == "event_drift":
            x.graph.completion_event.cuda_event += 1

    if x.session.lane == "cuda":
        x.p.runtime.cuda.synchronize = synchronize
    else:
        x.p.runtime.synchronize_device = synchronize

    def acknowledge(ticket):
        if x.controller._final_barrier_ticket is ticket:
            assert x.controller.quiescence_evidence is None
            assert x.controller._final_admission is None
            if failure == "disarmed":
                raise RuntimeError("injected lost final DISARMED")
            barrier_calls.append("ack")
        complete(ticket)

    monkeypatch.setattr(x.controller._guard_client, "complete", acknowledge)
    try:
        if failure is None:
            graph = x.session.prepare_fixed_final_graph(sample_count=2)
            assert barrier_calls == ["sync", "ack"]
            assert x.session.phase == "fixed_final_graph_admitted"
            assert x.controller.state == "SETUP_QUIESCENT"
            assert x.controller._final_admission is graph
            active = graph.arena.acquire(graph.samples[0])
            from optical_worker.lifecycle import WorkerSetupProtocolError

            lease = active._lease if x.session.lane == "warp" else active._control_lease
            calls_before = tuple(x.calls)
            with pytest.raises(WorkerSetupProtocolError, match="measured submission remains closed"):
                lease.begin_enqueue()
            assert lease.state == "LEASED" and tuple(x.calls) == calls_before
            active.cancel_before_enqueue()
            evidence = x.controller.evidence().to_payload()
            assert evidence["schema_version"] == "q54.p12.4.setup-lifecycle.v4"
            assert (
                evidence["summary"]["final_setup"]["quiescence_sha256"] == x.session.final_quiescence.sha256
            )
            legacy = dict(evidence, schema_version="q54.p12.4.setup-lifecycle.v3")
            # CPU supervisor story verifies version propagation through the
            # combined artifact validator; it makes no real-process exit claim.
            from optical_worker.guarded_evidence import validate_custody_artifact
            from tests.unit.optics.test_guarded_evidence_fixtures import artifact

            operations = {"normal_arm_started": "ARM", "normal_complete_started": "COMPLETE"}
            steps = [
                (operations[event["operation"]], event["guard"]["sequence"], "setup", True)
                for event in evidence["transition_trace"]
                if event["operation"] in operations
            ]
            validate_custody_artifact(artifact(evidence, steps))
            from optical_worker.wait_protocol import GuardProtocolError

            with pytest.raises(GuardProtocolError):
                GuardedLifecycleEvidence.from_payload(legacy)
            x.controller.mark_unknown_device_scope(source="test_final_scope", reason="late_unknown_scope")
            with pytest.raises(ValueError, match="acknowledged final"):
                graph.arena.acquire(graph.samples[1])
        else:
            with pytest.raises(Exception):
                x.session.prepare_fixed_final_graph(sample_count=2)
            assert barrier_calls.count("sync") == 1
            assert x.controller._final_admission is None
            assert x.controller.quiescence_evidence is None
            assert x.controller.guardian.state == "TRANSFERRED_TO_QUARANTINE"
            assert x.controller._final_sealed_coverage is not None
            assert (
                x.controller.termination_evidence.lifecycle.to_payload()["summary"][
                    "emergency_sync_attempt_count"
                ]
                == 0
            )
        assert x.f.state.device == 0 and x.f.state.stream is None
    finally:
        monkeypatch.setattr(x.controller._guard_client, "complete", complete)


@pytest.mark.parametrize("fixed_inputs", ["cuda", "warp"], indirect=True)
@pytest.mark.parametrize("fault", ["channel_eof", "parent_lost"])
@pytest.mark.parametrize("cancel_acquired", [False, True])
def test_quiescent_termination_uses_existing_barrier(final_session, fault, cancel_acquired):
    from copy import deepcopy

    from optical_worker.guarded_evidence import replay_lifecycle
    from optical_worker.wait_protocol import GuardProtocolError

    x = final_session
    syncs = []
    def synchronize(target):
        assert target is x.p.device
        syncs.append("final")
    if x.session.lane == "cuda":
        x.p.runtime.cuda.synchronize = synchronize
    else:
        x.p.runtime.synchronize_device = synchronize
    graph = x.session.prepare_fixed_final_graph(sample_count=2)
    if cancel_acquired:
        graph.arena.acquire(graph.samples[0]).cancel_before_enqueue()
    def unnecessary_sync():
        pytest.fail("quiescent final graph synchronized again")
    terminal = x.controller.terminate(fault=fault, detection_phase="test_final_idle",
        emergency_target_synchronize=unnecessary_sync).lifecycle.to_payload()
    assert syncs == ["final"]
    summary = terminal["summary"]
    assert summary["disposition"] == "cancelled_before_submission_exit"
    assert summary["release_status"] == "completed"
    assert summary["emergency_sync_attempt_count"] == 0
    assert summary["possible_dirty_device_ordinals"] == [1]
    assert summary["final_setup"]["quiescence_sha256"] == x.session.final_quiescence.sha256
    assert x.f.state.device == 0 and x.f.state.stream is None
    # A new v4 story cannot ask for an unnecessary target synchronization.
    forged = deepcopy(terminal["transition_trace"])
    index = next(i for i, event in enumerate(forged) if event["operation"] == "terminal_arm_started")
    forged[index]["details"]["purpose"] = "target_sync"
    with pytest.raises(GuardProtocolError, match="wrong terminal purpose"):
        replay_lifecycle(terminal["configuration"], terminal["lane_id"], forged[:index + 1],
                         version=terminal["schema_version"])


@pytest.mark.parametrize("fixed_inputs", ["cuda", "warp"], indirect=True)
@pytest.mark.parametrize("change", ["missing_proof", "latched", "unknown", "later_guard", "cleanup_ack"])
def test_final_termination_keeps_uncertain_state_conservative(final_session, monkeypatch, change):
    x = final_session
    if x.session.lane == "cuda":
        x.p.runtime.cuda.synchronize = lambda target: None
    else:
        x.p.runtime.synchronize_device = lambda target: None
    x.session.prepare_fixed_final_graph(sample_count=2)
    if change == "missing_proof":
        x.controller._quiescence = None
    elif change == "latched":
        x.controller._measured_sample_submission_latched = True
    elif change == "unknown":
        x.controller.mark_unknown_device_scope(source="test_final_scope", reason="late_unknown_scope")
    elif change == "later_guard":
        with x.controller.segment("resolve"):
            pass
    else:
        def failed_arm(role):
            raise RuntimeError("cleanup guard acknowledgement unavailable")
        monkeypatch.setattr(x.controller._guard_client, "arm", failed_arm)
    syncs = []
    terminal = x.controller.terminate(fault="channel_eof", detection_phase="test_final_idle",
        emergency_target_synchronize=lambda: syncs.append("emergency")).lifecycle.to_payload()
    summary = terminal["summary"]
    if change == "later_guard":
        assert syncs == ["emergency"]
        assert summary["disposition"] == "aborted_synced_exit"
        assert summary["release_status"] == "completed"
    else:
        assert syncs == []
        assert "quarantine" in summary["disposition"]
        assert summary["release_status"] != "completed"
        assert x.controller.guardian.state == "TRANSFERRED_TO_QUARANTINE"
    assert summary["possible_dirty_device_ordinals"] == [1]
