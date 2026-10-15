from __future__ import annotations

import gc
import subprocess
import sys
import weakref

import pytest

torch = pytest.importorskip("torch")
wp = pytest.importorskip("warp")

from physics.cuda_publication import (  # noqa: E402
    CudaConsumerResourceGuardian,
    CudaPublicationPoisonedError,
)
from physics.gpu_engine import GpuEngine  # noqa: E402
from physics.merged_model import merge_models  # noqa: E402
from physics.publish import ConsumerState  # noqa: E402
from tests.gpu.test_gpu_engine_api import _ball_model, _make_engine  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def _device_consumer(consumer_id: str = "cuda_camera") -> ConsumerState:
    return ConsumerState(
        consumer_id=consumer_id,
        consumer_kind="render_backed_sensing",
        qos_mode="lossless",
        access_mode="borrow",
        consumer_location="device",
    )


def _make_engine_on_device(device: str):
    merged = merge_models(robots={"ball": _ball_model()})
    engine = GpuEngine(merged, num_envs=1, device=device)
    q0, _ = merged.tree.default_state()
    q0[6] = 0.5
    engine.reset(q0=q0)
    return engine, merged


def test_torch_stream_binding_orders_zero_copy_read_and_slot_reuse():
    engine, _ = _make_engine(num_envs=1)
    first = engine.step()
    frame = engine.latest_published_frame()
    assert frame is not None
    consumer = _device_consumer()
    engine.register_consumer(consumer)

    stream = torch.cuda.Stream(device="cuda:0")
    destination = torch.empty(1, dtype=torch.float32, device="cuda:0")
    guardian = CudaConsumerResourceGuardian.seal(
        device_ordinal=0,
        cuda_stream_owner=stream,
        resources=(destination,),
    )
    binding = engine.bind_cuda_device_consumer(
        consumer.consumer_id,
        device_ordinal=0,
        cuda_stream_handle=stream.cuda_stream,
        consumer_resource_guardian=guardian,
    )

    lease = binding.borrow(frame.frame_id)
    with torch.cuda.stream(stream):
        q_view = wp.to_torch(lease.frame.q_wp)
        lease.retain_ephemeral(q_view)
        destination.copy_(q_view[:, 0])
    evidence = lease.complete(last_reader_operation="torch_zero_copy_q_read")

    assert evidence.publish_wait_enqueued is True
    assert evidence.consumer_completion_recorded is True
    assert isinstance(consumer.device_done_event, wp.Event)
    assert not isinstance(consumer.device_done_event, torch.cuda.Event)

    # Frame 3 reuses frame 0's slot.  The physics producer must wait the
    # publication-owned done event before overwriting that slot.
    engine.step()
    engine.step()
    engine.step()
    torch.cuda.synchronize(0)
    assert destination.item() == pytest.approx(float(first.q_new[0]), abs=1.0e-6)
    assert binding.evidence(frame.frame_id).producer_reuse_wait_enqueued is True
    assert binding.retained_done_event_count == 0

    binding.close()
    engine.unregister_consumer(consumer.consumer_id)
    engine.close()
    guardian.close()


def test_bound_consumer_rejects_legacy_api_rebind_reset_and_early_unregister():
    engine, _ = _make_engine(num_envs=1)
    engine.step()
    frame = engine.latest_published_frame()
    assert frame is not None
    consumer = _device_consumer("lifecycle")
    engine.register_consumer(consumer)
    stream = torch.cuda.Stream(device="cuda:0")
    resource = torch.empty(1, device="cuda:0")
    guardian = CudaConsumerResourceGuardian.seal(
        device_ordinal=0,
        cuda_stream_owner=stream,
        resources=(resource,),
    )
    with pytest.raises(TypeError, match="device_ordinal"):
        engine.bind_cuda_device_consumer(
            consumer.consumer_id,
            device_ordinal="0",
            cuda_stream_handle=stream.cuda_stream,
            consumer_resource_guardian=guardian,
        )
    assert guardian.state == "sealed_unbound"
    binding = engine.bind_cuda_device_consumer(
        consumer.consumer_id,
        device_ordinal=0,
        cuda_stream_handle=stream.cuda_stream,
        consumer_resource_guardian=guardian,
    )

    with pytest.raises(RuntimeError, match="already bound"):
        engine.bind_cuda_device_consumer(
            consumer.consumer_id,
            device_ordinal=0,
            cuda_stream_handle=stream.cuda_stream,
            consumer_resource_guardian=guardian,
        )
    with pytest.raises(RuntimeError, match="binding.borrow"):
        engine.borrow_device_frame(consumer.consumer_id, frame.frame_id)
    with pytest.raises(RuntimeError, match="lease.complete"):
        engine.complete_device_consumer(consumer.consumer_id, frame.frame_id)
    with pytest.raises(RuntimeError, match="while binding is open"):
        engine.unregister_consumer(consumer.consumer_id)
    run_id_before_rejected_reset = engine._physics_run_id
    frame_id_before_rejected_reset = engine._publish_frame_id
    with pytest.raises(RuntimeError, match="binding state exists"):
        engine.reset()
    assert engine._physics_run_id == run_id_before_rejected_reset
    assert engine._publish_frame_id == frame_id_before_rejected_reset
    assert engine.latest_published_frame() is frame

    lease = binding.borrow(frame.frame_id)
    lease.complete(last_reader_operation="no_consumer_reader_enqueued")
    binding.close()
    engine.unregister_consumer(consumer.consumer_id)
    engine.reset()
    engine.close()
    guardian.close()


def test_legacy_incomplete_borrow_cannot_bind_or_unregister():
    engine, _ = _make_engine(num_envs=1)
    engine.step()
    frame = engine.latest_published_frame()
    assert frame is not None
    consumer = _device_consumer("legacy_incomplete")
    engine.register_consumer(consumer)
    legacy_stream = wp.Stream(wp.get_device("cuda:0"))
    engine.borrow_device_frame(consumer.consumer_id, frame.frame_id, stream=legacy_stream)

    cuda_stream = torch.cuda.Stream(device="cuda:0")
    resource = torch.empty(1, device="cuda:0")
    guardian = CudaConsumerResourceGuardian.seal(
        device_ordinal=0,
        cuda_stream_owner=cuda_stream,
        resources=(resource,),
    )
    with pytest.raises(RuntimeError, match="fresh consumer"):
        engine.bind_cuda_device_consumer(
            consumer.consumer_id,
            device_ordinal=0,
            cuda_stream_handle=cuda_stream.cuda_stream,
            consumer_resource_guardian=guardian,
        )
    with pytest.raises(RuntimeError, match="incomplete device borrow"):
        engine.unregister_consumer(consumer.consumer_id)

    assert guardian.state == "sealed_unbound"
    assert consumer.latest_seen_frame_id == frame.frame_id
    assert consumer.device_completed_frame_id == -1
    assert consumer.device_done_event is None
    engine.complete_device_consumer(consumer.consumer_id, frame.frame_id, stream=legacy_stream)
    engine.unregister_consumer(consumer.consumer_id)
    engine.close()
    guardian.close()


def test_legacy_completed_borrow_cannot_bind_and_retains_done_event():
    engine, _ = _make_engine(num_envs=1)
    engine.step()
    frame = engine.latest_published_frame()
    assert frame is not None
    consumer = _device_consumer("legacy_completed")
    engine.register_consumer(consumer)
    legacy_stream = wp.Stream(wp.get_device("cuda:0"))
    engine.borrow_device_frame(consumer.consumer_id, frame.frame_id, stream=legacy_stream)
    legacy_done_event = engine.complete_device_consumer(
        consumer.consumer_id,
        frame.frame_id,
        stream=legacy_stream,
    )

    cuda_stream = torch.cuda.Stream(device="cuda:0")
    resource = torch.empty(1, device="cuda:0")
    guardian = CudaConsumerResourceGuardian.seal(
        device_ordinal=0,
        cuda_stream_owner=cuda_stream,
        resources=(resource,),
    )
    with pytest.raises(RuntimeError, match="fresh consumer"):
        engine.bind_cuda_device_consumer(
            consumer.consumer_id,
            device_ordinal=0,
            cuda_stream_handle=cuda_stream.cuda_stream,
            consumer_resource_guardian=guardian,
        )

    assert guardian.state == "sealed_unbound"
    assert consumer.device_completed_frame_id == frame.frame_id
    assert consumer.device_done_event is legacy_done_event
    assert consumer.consumer_id not in engine._cuda_consumer_bindings
    engine.unregister_consumer(consumer.consumer_id)
    assert consumer.device_done_event is legacy_done_event
    engine.close()
    guardian.close()


def test_legacy_unregister_enqueues_done_wait_before_fresh_id_can_bind(monkeypatch):
    engine, _ = _make_engine(num_envs=1)
    engine.step()
    frame = engine.latest_published_frame()
    assert frame is not None
    legacy_consumer = _device_consumer("legacy_to_remove")
    engine.register_consumer(legacy_consumer)
    legacy_stream = wp.Stream(wp.get_device("cuda:0"))
    engine.borrow_device_frame(legacy_consumer.consumer_id, frame.frame_id, stream=legacy_stream)
    legacy_done_event = engine.complete_device_consumer(
        legacy_consumer.consumer_id,
        frame.frame_id,
        stream=legacy_stream,
    )

    producer_waits = []
    original_wait = engine._wait_on_event

    def _observe_wait(event, *, stream=None):
        producer_waits.append(event)
        return original_wait(event, stream=stream)

    def _forbidden_sync(*_args, **_kwargs):
        raise AssertionError("legacy unregister attempted host/global synchronization")

    monkeypatch.setattr(engine, "_wait_on_event", _observe_wait)
    monkeypatch.setattr(torch.cuda, "synchronize", _forbidden_sync)
    monkeypatch.setattr(wp, "synchronize", _forbidden_sync)
    monkeypatch.setattr(wp, "synchronize_device", _forbidden_sync)
    monkeypatch.setattr(wp, "synchronize_event", _forbidden_sync)
    engine.unregister_consumer(legacy_consumer.consumer_id)
    assert producer_waits == [legacy_done_event]
    assert legacy_consumer.device_done_event is legacy_done_event

    fresh_consumer = _device_consumer("fresh_cuda_consumer")
    engine.register_consumer(fresh_consumer)
    cuda_stream = torch.cuda.Stream(device="cuda:0")
    resource = torch.empty(1, device="cuda:0")
    guardian = CudaConsumerResourceGuardian.seal(
        device_ordinal=0,
        cuda_stream_owner=cuda_stream,
        resources=(resource,),
    )
    binding = engine.bind_cuda_device_consumer(
        fresh_consumer.consumer_id,
        device_ordinal=0,
        cuda_stream_handle=cuda_stream.cuda_stream,
        consumer_resource_guardian=guardian,
    )
    binding.close()
    engine.unregister_consumer(fresh_consumer.consumer_id)
    engine.close()
    guardian.close()


def test_normal_hot_path_never_calls_host_or_global_synchronize(monkeypatch):
    engine, _ = _make_engine(num_envs=1)
    engine.step()
    frame = engine.latest_published_frame()
    assert frame is not None
    consumer = _device_consumer("no_host_sync")
    engine.register_consumer(consumer)
    stream = torch.cuda.Stream(device="cuda:0")
    resource = torch.empty(1, device="cuda:0")
    guardian = CudaConsumerResourceGuardian.seal(
        device_ordinal=0,
        cuda_stream_owner=stream,
        resources=(resource,),
    )
    binding = engine.bind_cuda_device_consumer(
        consumer.consumer_id,
        device_ordinal=0,
        cuda_stream_handle=stream.cuda_stream,
        consumer_resource_guardian=guardian,
    )

    def _forbidden_sync(*_args, **_kwargs):
        raise AssertionError("normal CUDA publication path attempted host/global synchronization")

    monkeypatch.setattr(torch.cuda, "synchronize", _forbidden_sync)
    monkeypatch.setattr(wp, "synchronize", _forbidden_sync)
    monkeypatch.setattr(wp, "synchronize_device", _forbidden_sync)
    monkeypatch.setattr(wp, "synchronize_event", _forbidden_sync)
    lease = binding.borrow(frame.frame_id)
    lease.complete(last_reader_operation="no_consumer_reader_enqueued")
    binding.close()
    engine.unregister_consumer(consumer.consumer_id)
    engine.close()
    guardian.close()


def test_engine_close_immediately_aborts_final_drain_poison(monkeypatch):
    engine, _ = _make_engine(num_envs=1)
    engine.step()
    frame = engine.latest_published_frame()
    assert frame is not None
    consumer = _device_consumer("close_poison")
    engine.register_consumer(consumer)
    stream = torch.cuda.Stream(device="cuda:0")
    resource = torch.empty(1, device="cuda:0")
    guardian = CudaConsumerResourceGuardian.seal(
        device_ordinal=0,
        cuda_stream_owner=stream,
        resources=(resource,),
    )
    binding = engine.bind_cuda_device_consumer(
        consumer.consumer_id,
        device_ordinal=0,
        cuda_stream_handle=stream.cuda_stream,
        consumer_resource_guardian=guardian,
    )
    binding.borrow(frame.frame_id).complete(last_reader_operation="no_consumer_reader_enqueued")

    runtime = binding._runtime
    original_emergency_sync = runtime.emergency_synchronize
    emergency_sync_count = 0

    def _fail_final_drain(_event):
        raise RuntimeError("injected final-drain wait failure")

    def _count_emergency_sync():
        nonlocal emergency_sync_count
        emergency_sync_count += 1
        original_emergency_sync()

    monkeypatch.setattr(runtime, "producer_wait_event", _fail_final_drain)
    monkeypatch.setattr(runtime, "emergency_synchronize", _count_emergency_sync)

    engine.close()
    assert engine._closed is True
    assert binding.state == "aborted_synced"
    assert binding.abort_disposition is not None
    assert binding.abort_disposition.disposition == "emergency_sync_succeeded"
    assert emergency_sync_count == 1
    assert guardian.state == "released_after_emergency_sync"

    engine.close()
    assert emergency_sync_count == 1
    engine.unregister_consumer(consumer.consumer_id)
    guardian.close()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_target_device_binding_restores_deliberately_wrong_ambient_device():
    original_device = torch.cuda.current_device()
    engine = None
    guardian = None
    try:
        engine, _ = _make_engine_on_device("cuda:1")
        engine.step()
        frame = engine.latest_published_frame()
        assert frame is not None
        consumer = _device_consumer("cuda_camera_1")
        engine.register_consumer(consumer)
        stream = torch.cuda.Stream(device="cuda:1")
        destination = torch.empty(1, dtype=torch.float32, device="cuda:1")
        guardian = CudaConsumerResourceGuardian.seal(
            device_ordinal=1,
            cuda_stream_owner=stream,
            resources=(destination,),
        )

        torch.cuda.set_device(0)
        assert torch.cuda.current_device() == 0
        binding = engine.bind_cuda_device_consumer(
            consumer.consumer_id,
            device_ordinal=1,
            cuda_stream_handle=stream.cuda_stream,
            consumer_resource_guardian=guardian,
        )
        assert torch.cuda.current_device() == 0

        lease = binding.borrow(frame.frame_id)
        assert torch.cuda.current_device() == 0
        with torch.cuda.stream(stream):
            q_view = wp.to_torch(lease.frame.q_wp)
            lease.retain_ephemeral(q_view)
            destination.copy_(q_view[:, 0])
        assert torch.cuda.current_device() == 0
        lease.complete(last_reader_operation="torch_zero_copy_q_read")
        assert torch.cuda.current_device() == 0
        binding.close()
        assert torch.cuda.current_device() == 0

        engine.unregister_consumer(consumer.consumer_id)
        engine.close()
        guardian.close()
        guardian = None
    finally:
        torch.cuda.set_device(original_device)
        if engine is not None and not getattr(engine, "_closed", True):
            engine.close()
        if guardian is not None and guardian.state not in {"bound", "quarantined", "closed"}:
            guardian.close()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_injected_target_device_failure_restores_wrong_ambient_device():
    original_device = torch.cuda.current_device()
    engine = None
    guardian = None
    try:
        engine, _ = _make_engine_on_device("cuda:1")
        engine.step()
        frame = engine.latest_published_frame()
        assert frame is not None
        consumer = _device_consumer("cuda_camera_failure_1")
        engine.register_consumer(consumer)
        stream = torch.cuda.Stream(device="cuda:1")
        resource = torch.empty(1, device="cuda:1")
        guardian = CudaConsumerResourceGuardian.seal(
            device_ordinal=1,
            cuda_stream_owner=stream,
            resources=(resource,),
        )
        torch.cuda.set_device(0)
        binding = engine.bind_cuda_device_consumer(
            consumer.consumer_id,
            device_ordinal=1,
            cuda_stream_handle=stream.cuda_stream,
            consumer_resource_guardian=guardian,
        )
        assert torch.cuda.current_device() == 0
        lease = binding.borrow(frame.frame_id)
        assert torch.cuda.current_device() == 0

        def _fail_record(_stream, _event):
            raise RuntimeError("injected record failure")

        binding._runtime.record_event = _fail_record
        with pytest.raises(CudaPublicationPoisonedError):
            lease.complete(last_reader_operation="no_consumer_reader_enqueued")
        assert torch.cuda.current_device() == 0
        disposition = binding.abort_poisoned()
        assert disposition.disposition == "emergency_sync_succeeded"
        assert torch.cuda.current_device() == 0
        guardian.close()
        guardian = None
        engine.unregister_consumer(consumer.consumer_id)
        engine.close()
    finally:
        torch.cuda.set_device(original_device)
        if engine is not None and not getattr(engine, "_closed", True):
            engine.close()
        if guardian is not None and guardian.state not in {"bound", "quarantined", "closed"}:
            guardian.close()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_cross_device_guardian_is_rejected_before_binding_or_stream_wrap():
    engine, _ = _make_engine_on_device("cuda:1")
    consumer = _device_consumer("mismatch")
    engine.register_consumer(consumer)
    stream = torch.cuda.Stream(device="cuda:0")
    destination = torch.empty(1, device="cuda:0")
    guardian = CudaConsumerResourceGuardian.seal(
        device_ordinal=0,
        cuda_stream_owner=stream,
        resources=(destination,),
    )

    with pytest.raises(ValueError, match="does not match engine"):
        engine.bind_cuda_device_consumer(
            consumer.consumer_id,
            device_ordinal=0,
            cuda_stream_handle=stream.cuda_stream,
            consumer_resource_guardian=guardian,
        )

    assert guardian.state == "sealed_unbound"
    guardian.close()
    engine.close()


def test_record_failure_keeps_cross_owner_resources_until_emergency_sync():
    engine, _ = _make_engine(num_envs=1)
    engine.step()
    frame = engine.latest_published_frame()
    assert frame is not None
    consumer = _device_consumer("record_failure")
    engine.register_consumer(consumer)
    stream = torch.cuda.Stream(device="cuda:0")
    destination = torch.empty(1, device="cuda:0")
    destination_ref = weakref.ref(destination)
    guardian = CudaConsumerResourceGuardian.seal(
        device_ordinal=0,
        cuda_stream_owner=stream,
        resources=(destination,),
    )
    binding = engine.bind_cuda_device_consumer(
        consumer.consumer_id,
        device_ordinal=0,
        cuda_stream_handle=stream.cuda_stream,
        consumer_resource_guardian=guardian,
    )
    lease = binding.borrow(frame.frame_id)
    with torch.cuda.stream(stream):
        q_view = wp.to_torch(lease.frame.q_wp)
        lease.retain_ephemeral(q_view)
        destination.copy_(q_view[:, 0])

    def _fail_record(_stream, _event):
        raise RuntimeError("injected record failure")

    binding._runtime.record_event = _fail_record
    del destination
    gc.collect()
    with pytest.raises(CudaPublicationPoisonedError, match="record"):
        lease.complete(last_reader_operation="torch_zero_copy_q_read")
    assert destination_ref() is not None
    assert frame.slot_meta.poisoned is True
    assert consumer.device_completed_frame_id == -1

    disposition = binding.abort_poisoned()
    assert disposition.disposition == "emergency_sync_succeeded"
    assert guardian.state == "released_after_emergency_sync"
    guardian.close()
    gc.collect()
    assert destination_ref() is None
    engine.unregister_consumer(consumer.consumer_id)
    engine.close()


def test_failed_emergency_sync_quarantine_is_verified_in_isolated_process():
    script = r"""
import gc
import os
import weakref

import torch
import warp as wp

from physics.cuda_publication import (
    CudaConsumerResourceGuardian,
    CudaPublicationPoisonedError,
    CudaPublicationQuarantinedError,
)
from physics.publish import ConsumerState
from tests.gpu.test_gpu_engine_api import _make_engine

engine, _ = _make_engine(num_envs=1)
engine.step()
frame = engine.latest_published_frame()
consumer = ConsumerState(
    consumer_id="quarantine_child",
    consumer_kind="render_backed_sensing",
    qos_mode="lossless",
    access_mode="borrow",
    consumer_location="device",
)
engine.register_consumer(consumer)
stream = torch.cuda.Stream(device="cuda:0")
destination = torch.empty(1, device="cuda:0")
destination_ref = weakref.ref(destination)
guardian = CudaConsumerResourceGuardian.seal(
    device_ordinal=0,
    cuda_stream_owner=stream,
    resources=(destination,),
)
binding = engine.bind_cuda_device_consumer(
    consumer.consumer_id,
    device_ordinal=0,
    cuda_stream_handle=stream.cuda_stream,
    consumer_resource_guardian=guardian,
)
lease = binding.borrow(frame.frame_id)
with torch.cuda.stream(stream):
    q_view = wp.to_torch(lease.frame.q_wp)
    lease.retain_ephemeral(q_view)
    destination.copy_(q_view[:, 0])

def fail_record(_stream, _event):
    raise RuntimeError("record failed")

def fail_sync():
    raise RuntimeError("sync failed")

binding._runtime.record_event = fail_record
try:
    lease.complete(last_reader_operation="torch_zero_copy_q_read")
except CudaPublicationPoisonedError:
    pass
else:
    raise AssertionError("record failure did not poison")
binding._runtime.emergency_synchronize = fail_sync
disposition = binding.abort_poisoned()
assert disposition.disposition == "quarantined_until_process_exit"
assert guardian.state == "quarantined"
binding_id = binding.binding_id
registry = engine._cuda_publication_quarantine
engine.unregister_consumer(consumer.consumer_id)
engine.close()
try:
    guardian.close()
except CudaPublicationQuarantinedError:
    pass
else:
    raise AssertionError("quarantined guardian allowed ordinary close")
del destination
del q_view
del lease
del binding
del guardian
del stream
del engine
gc.collect()
assert destination_ref() is not None
assert registry.contains_binding(binding_id)
os._exit(0)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr
