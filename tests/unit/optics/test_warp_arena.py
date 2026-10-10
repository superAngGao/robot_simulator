# Imported fixtures are registered by pytest.
# ruff: noqa: F401, F811
"""Real FIFO/lease/C3 coordination; native commands and allocations are CPU doubles."""

from contextlib import contextmanager
from copy import copy
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from optical_worker.native_input import NativeInputValidationReport, certify_native_readiness
from optical_worker.provenance import NativeStorageSignature
from optical_worker.render_parameters import bind_render_parameters
from optics import warp_arena, warp_native_render
from optics.native_arena import NativeOpticalArenaProtocolError
from tests.unit.optics.test_native_optical_arena import _spec
from tests.unit.optics.test_prepared_fixed_render_inputs import fixed_inputs

pytestmark = pytest.mark.parametrize("fixed_inputs", ["warp"], indirect=True)


@pytest.fixture
def build_arena(fixed_inputs, monkeypatch):
    p = fixed_inputs
    m = p.prepared.manifest
    values = {
        name: getattr(m, name)
        for name in NativeInputValidationReport.__dataclass_fields__
        if name not in {"backend", "manifest_sha256", "validated_arrays"}
    }
    report = NativeInputValidationReport(
        backend="warp", manifest_sha256=m.sha256, validated_arrays=tuple(a.name for a in m.arrays), **values
    )
    readiness = certify_native_readiness(
        p.prepared, report, compile_complete=True, warmup_frames_completed=10
    )
    calls = []
    state = NS(device=0, stream=None, fail=None, wait=None)
    stream = NS(device=p.device, cuda_stream=17)

    class Event:
        def __init__(self):
            self.device = p.device
            self.cuda_event = id(self)

    def record(event):
        assert state.device is p.device and state.stream is stream
        calls.append(("record", event))
        if state.fail == "record":
            raise RuntimeError("record failed after possible submission")

    stream.record_event = record

    @contextmanager
    def device_scope(device):
        previous = state.device
        state.device = device
        try:
            yield
        finally:
            state.device = previous

    @contextmanager
    def stream_scope(selected):
        previous = state.stream
        state.stream = selected
        try:
            yield
        finally:
            state.stream = previous

    # Warp's scopes are reusable; a generator contextmanager isn't.
    class Scope:
        def __init__(self, creator, value):
            self.creator, self.value = creator, value

        def __enter__(self):
            self.active = self.creator(self.value)
            return self.active.__enter__()

        def __exit__(self, *args):
            return self.active.__exit__(*args)

    def wait(event):
        calls.append(("wait", event))
        if state.wait is not None:
            state.wait()
        if state.fail == "wait":
            raise RuntimeError("wait failed")

    runtime = NS(
        Event=Event,
        synchronize_event=wait,
        ScopedDevice=lambda d: Scope(device_scope, d),
        ScopedStream=lambda s: Scope(stream_scope, s),
    )
    factory = NS(prepared_input=p.prepared, readiness=readiness, device=p.device, stream=stream)
    output = object()
    buffers = NS(arrays=(("rgb", output),))

    def signature(name, array, device, owner):
        return NativeStorageSignature(
            name=name,
            backend="warp",
            device_ordinal=1,
            storage_token=f"cpu-allocation:{id(array):x}",
            dtype="float32",
            shape=(21, 3),
            nbytes=252,
            active_start=0,
            active_count=252,
            owner_token=owner,
        )

    monkeypatch.setattr(warp_arena, "_storage_signature", signature)

    class Bundle:
        def __init__(self, parameters, index, diagnostic):
            self.parameters = parameters
            self.binding = NS(
                factory=factory,
                buffers=buffers,
                transform_frame_index=None,
                traversal_diagnostics=diagnostic,
                output_profile=NS(value="direct_light_full"),
                shadows=True,
                query=NS(frame_id=index + 1, num_rays=21),
                camera=NS(width=7, height=3),
                bvh=NS(num_primitives=5, num_nodes=9),
            )
            ids = (
                "raygen",
                "_reset_bvh_primary_counters_kernel",
                "first_hit",
                "_reset_shadow_primary_counters_kernel",
                "shade",
            )

            def launch(name, *, stream):
                assert state.device is p.device and state.stream is stream
                calls.append(("launch", name))
                if state.fail == "launch":
                    raise RuntimeError("launch failed after possible submission")

            from functools import partial

            self.descriptors = tuple(NS(entry_id=name) for name in ids)
            self.commands = tuple(NS(launch=partial(launch, name)) for name in ids)
            self.evidence = NS(sha256="b" * 64)

        def validate_resolved_parameters(self, actual):
            if actual is not self.parameters:
                raise ValueError("recorded scalar binding changed")

    monkeypatch.setattr(warp_native_render, "WarpDirectLightRenderCommandBundle", Bundle)

    def build(count=2, diagnostic=False):
        mode = "diagnostic_profiled" if diagnostic else "primary_unprofiled"
        spec = replace(
            _spec(boundary="R0"),
            lane="warp",
            renderer_id="warp_bvh_direct_light",
            case_id=m.case_id,
            case_sha256=m.case_sha256,
            native_input_manifest_sha256=m.sha256,
            native_readiness_sha256=readiness.sha256,
            scene_contract_sha256=m.scene_contract_sha256,
            fixed_render_fixture_sha256=m.fixed_render_fixture_sha256,
            diagnostic_mode=mode,
            execution_stream_token="warp-stream:1:17",
        )
        frames = []
        for index in range(count):
            parameters = bind_render_parameters(
                p.render.sequence,
                sample_index=index,
                profile=spec.output_profile,
                shadows=True,
                pass_mode=mode,
                role_to_mask=p.render.packed_scene.role_to_mask,
            )
            frames.append((parameters, Bundle(parameters, index, diagnostic)))
        from optical_worker.native_setup import _render_phase_names

        events = tuple(
            (name, Event(), Event()) for name in _render_phase_names(boundary="R0", diagnostic=diagnostic)
        )
        arena = warp_arena.WarpNativeOpticalArena(
            spec=spec,
            runtime=runtime,
            frames=tuple(frames),
            completion_event=Event(),
            diagnostic_events=events,
        )
        plans = tuple(
            arena.prepare_sample(render_sample_index=i, planned_acquisition_offset=i) for i in range(count)
        )
        return NS(
            arena=arena,
            plans=plans,
            calls=calls,
            state=state,
            runtime=runtime,
            device=p.device,
            stream=stream,
            frames=frames,
            output=output,
        )

    return build


@pytest.mark.parametrize("diagnostic", [False, True])
def test_shared_phase_order_and_two_generations(build_arena, diagnostic):
    x = build_arena(diagnostic=diagnostic)
    assert x.calls == []
    for plan in x.plans:
        with pytest.raises(NativeOpticalArenaProtocolError):
            plan._transaction.enqueue()
        active = x.arena.acquire(plan)
        active.enqueue()
        active.finish()
        assert x.state.device == 0 and x.state.stream is None
        active.resolve()
        slots = x.arena.lifecycle_evidence().slots
        assert slots[0].transaction_hold_count == 0 and slots[0].device_result_hold_count == 1
        active.release_device_result()
        active.reclaim()
    evidence = x.arena.lifecycle_evidence()
    assert evidence.slot_reclaim_count == evidence.completion_resolution_count == 2
    assert [v for kind, v in x.calls if kind == "launch"] == [
        "raygen",
        "_reset_bvh_primary_counters_kernel",
        "first_hit",
        "_reset_shadow_primary_counters_kernel",
        "shade",
    ] * 2
    assert sum(kind == "record" for kind, _ in x.calls) == 2 * (7 if diagnostic else 1)


@pytest.mark.parametrize("state", ["prepared", "pending", "resolved"])
def test_drain_invalidates_future_plans_without_burning_generations(build_arena, state):
    x = build_arena(count=3)
    active = None
    if state != "prepared":
        active = x.arena.acquire(x.plans[0])
        active.enqueue()
        active.finish()
        if state == "resolved":
            active.resolve()
    before = x.arena.lifecycle_evidence()
    calls = list(x.calls)
    x.arena.begin_drain()
    drained = x.arena.lifecycle_evidence()
    assert drained.arena_state == "DRAINING"
    assert drained.slots == before.slots
    assert x.calls == calls
    assert not x.arena._control._prepared_registry
    for plan in x.plans[1:]:
        with pytest.raises(NativeOpticalArenaProtocolError):
            x.arena.acquire(plan)
        with pytest.raises(NativeOpticalArenaProtocolError):
            plan._transaction.enqueue()
    with pytest.raises(NativeOpticalArenaProtocolError):
        x.arena.begin_drain()
    if active is not None:
        with pytest.raises(NativeOpticalArenaProtocolError):
            x.arena.close()
        if state == "pending":
            active.resolve()
        active.release_device_result()
        active.reclaim()
    x.arena.close()
    result = x.arena.lifecycle_evidence()
    assert result.arena_state == "CLOSED"
    assert result.slots[0].generation == before.slots[0].generation
    assert result.slots[0].state == "FREE"
    assert result.final_drain_complete and result.final_reclaim_complete
    assert result.completion_resolution_count == result.slot_reclaim_count == int(active is not None)
    assert not any(t.operation == "cancelled_generation_burned" for t in result.transitions)


def test_copies_cancellation_and_burned_generation(build_arena):
    x = build_arena()
    with pytest.raises(NativeOpticalArenaProtocolError):
        x.arena.acquire(copy(x.plans[0]))
    active = x.arena.acquire(x.plans[0])
    with pytest.raises(NativeOpticalArenaProtocolError):
        copy(active).enqueue()
    with pytest.raises(NativeOpticalArenaProtocolError):
        copy(active._transaction).enqueue()
    active.cancel_before_enqueue()
    with pytest.raises(NativeOpticalArenaProtocolError):
        active.enqueue()
    with pytest.raises(NativeOpticalArenaProtocolError):
        x.arena.acquire(x.plans[0])
    next_active = x.arena.acquire(x.plans[1])
    assert next_active._lease.evidence.generation == 2
    assert x.calls == []


@pytest.mark.parametrize("failure", ["launch", "record", "wait"])
def test_native_failure_retains_both_holds_and_restores_scope(build_arena, failure):
    x = build_arena()
    active = x.arena.acquire(x.plans[0])
    x.state.fail = failure
    with pytest.raises(RuntimeError):
        active.enqueue()
        active.finish()
        active.resolve()
    assert active.state == "POISONED"
    assert x.state.device == 0 and x.state.stream is None
    snapshot = x.arena.lifecycle_evidence()
    assert snapshot.completion_resolution_count == snapshot.slot_reclaim_count == 0
    assert snapshot.slots[0].transaction_hold_count == snapshot.slots[0].device_result_hold_count == 1


@pytest.mark.parametrize("boundary", ["before_wait", "during_wait"])
@pytest.mark.parametrize("mutation", ["stream", "event", "event_handle", "parameters"])
def test_resolution_rechecks_original_scope_before_releasing(build_arena, boundary, mutation):
    x = build_arena()
    active = x.arena.acquire(x.plans[0])
    active.enqueue()
    active.finish()

    def mutate():
        if mutation == "stream":
            x.stream.cuda_stream += 1
        elif mutation == "event":
            active.completion.event.device = NS(ordinal=0)
        elif mutation == "event_handle":
            active.completion.event.cuda_event += 1
        else:
            x.frames[0][1].parameters = object()

    if boundary == "before_wait":
        mutate()
    else:
        x.state.wait = mutate
    with pytest.raises((ValueError, NativeOpticalArenaProtocolError)):
        active.resolve()
    snapshot = x.arena.lifecycle_evidence()
    assert snapshot.completion_resolution_count == 0
    assert snapshot.slots[0].transaction_hold_count == snapshot.slots[0].device_result_hold_count == 1


@pytest.mark.parametrize("failure", [None, "wait", "disarmed", "ack_drift"])
def test_guarded_wait_requires_ack_before_resolution(build_arena, monkeypatch, failure):
    from optical_worker.guarded_lifecycle import NativeGuardedSetupLifecycle
    from optical_worker.lifecycle import DeviceScopeDescriptor
    from tests.support.guarded_peer import Peer

    x = build_arena(count=1)
    active = x.arena.acquire(x.plans[0])
    active.enqueue()
    active.finish()
    peer = Peer()
    controller = NativeGuardedSetupLifecycle(lane_id="warp", channel=peer.channel, configuration=peer.config)
    try:
        with controller.segment("setup"):
            descriptor = DeviceScopeDescriptor.exact_target(
                operation_id="cpu_event_fixture",
                leaf_kind="event_record",
                device_ordinal=1,
                scope_source="cpu_test",
                runtime_identity="cpu_test_event",
            )
            controller.enter_device_leaf(controller.begin_device_leaf(descriptor))
            for owner in (active, active._transaction, x.stream, active.completion.event, x.runtime):
                controller.adopt_host_owner(owner, role="resolution_owner", operation_id="cpu_event_fixture")
        controller.seal_guardian()
        complete = controller._guard_client.complete

        def ack(ticket):
            snapshot = x.arena.lifecycle_evidence()
            assert snapshot.completion_resolution_count == 0
            assert snapshot.slots[0].transaction_hold_count == 1
            if failure == "disarmed":
                raise RuntimeError("lost DISARMED")
            complete(ticket)
            if failure == "ack_drift":
                x.stream.cuda_stream += 1

        monkeypatch.setattr(controller._guard_client, "complete", ack)
        if failure == "wait":
            x.state.fail = "wait"
        if failure is None:
            active.resolve(_guarded_controller=controller)
            assert x.arena.lifecycle_evidence().completion_resolution_count == 1
        else:
            with pytest.raises((RuntimeError, NativeOpticalArenaProtocolError)):
                active.resolve(_guarded_controller=controller)
            snapshot = x.arena.lifecycle_evidence()
            assert snapshot.completion_resolution_count == 0
            assert snapshot.slots[0].transaction_hold_count == snapshot.slots[0].device_result_hold_count == 1
            if failure != "ack_drift":
                assert controller.guardian.state == "TRANSFERRED_TO_QUARANTINE"
                assert (
                    controller.termination_evidence.lifecycle.to_payload()["summary"][
                        "emergency_sync_attempt_count"
                    ]
                    == 0
                )
    finally:
        monkeypatch.setattr(controller._guard_client, "complete", complete)
        if controller.termination_evidence is None:
            controller.terminate(
                fault="setup_failure",
                detection_phase="cpu_test_exit",
                emergency_target_synchronize=lambda: None,
            )
        peer.close()


def test_acquire_work_does_not_scale_with_frame_catalog(build_arena, monkeypatch):
    import sys

    counts = []
    for count in (1, 120, 600):
        x = build_arena(count=count)
        # Preparing every distinct frame's C3/lease above is intentionally untimed.
        # Count executed Python lines in the two production arena modules during
        # actual acquisition, rather than comparing noisy wall-clock durations.
        work = []

        def trace(frame, event, arg):
            if event == "line" and frame.f_code.co_filename.replace("\\", "/").endswith(
                ("optics/warp_arena.py", "optics/native_arena.py")
            ):
                work.append((frame.f_code.co_filename, frame.f_lineno))
            return trace

        previous = sys.gettrace()
        try:
            sys.settrace(trace)
            active = x.arena.acquire(x.plans[0])
        finally:
            sys.settrace(previous)
        counts.append(len(work))
        assert x.calls == []
        active.cancel_before_enqueue()
    assert counts[0] > 0 and counts[0] == counts[1] == counts[2], counts
