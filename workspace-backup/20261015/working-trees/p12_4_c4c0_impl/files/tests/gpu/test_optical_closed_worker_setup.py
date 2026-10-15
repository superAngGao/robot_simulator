from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmarks.optical.native import CanonicalNativeInput
from optical_worker.lifecycle import DeviceScopeDescriptor, NativeWorkerSetupLifecycle
from optical_worker.native_setup import (
    NativeWorkerPreparationAborted,
    prepare_cuda_native_input_for_closed_worker,
    prepare_warp_native_input_for_closed_worker,
    seal_and_quiesce_cuda_setup,
    seal_and_quiesce_warp_setup,
)

pytestmark = pytest.mark.gpu


def _source() -> CanonicalNativeInput:
    return CanonicalNativeInput.from_arrays(
        case_id="asymmetric_synthetic_7x3_v1",
        materializer_id="closed_worker_setup_probe_v1",
        arrays={
            "float_values": np.arange(18, dtype=np.float32).reshape(6, 3),
            "integer_values": np.array([7, 3, 11, 5], dtype=np.int32),
        },
        array_semantics={
            "float_values": "closed-worker float upload probe",
            "integer_values": "closed-worker integer upload probe",
        },
        topology_arrays=("integer_values",),
        semantics={"purpose": "closed-worker guarded setup fault matrix"},
    )


def _require_both_backends():
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    if not torch.cuda.is_available() or wp.get_cuda_device_count() == 0:
        pytest.skip("CUDA is required")
    return torch, wp


def _enter(
    lifecycle: NativeWorkerSetupLifecycle,
    descriptor: DeviceScopeDescriptor,
) -> None:
    permit = lifecycle.begin_device_leaf(descriptor)
    lifecycle.enter_device_leaf(permit)


def _noop_abort_notifier(device_ordinal: int) -> None:
    assert device_ordinal >= 0


@pytest.mark.parametrize("lane", ["warp", "cuda"])
def test_guarded_a2_register_before_copy_and_final_quiescence(lane):
    torch, wp = _require_both_backends()
    target = 1 if min(torch.cuda.device_count(), wp.get_cuda_device_count()) > 1 else 0
    ambient = 0
    torch.cuda.set_device(ambient)
    source = _source()
    lifecycle = NativeWorkerSetupLifecycle(
        lane_id=lane,
        target_device_ordinal=target,
        emergency_sync_notifier=_noop_abort_notifier,
        session_identity=f"{lane}-a2-success",
    )
    if lane == "cuda":
        prepared = prepare_cuda_native_input_for_closed_worker(source, lifecycle=lifecycle)
        evidence = seal_and_quiesce_cuda_setup(
            lifecycle,
            a2_readiness_sha256="a" * 64,
            prepared_graph_sha256="b" * 64,
        )
    else:
        prepared = prepare_warp_native_input_for_closed_worker(source, lifecycle=lifecycle)
        evidence = seal_and_quiesce_warp_setup(
            lifecycle,
            a2_readiness_sha256="a" * 64,
            prepared_graph_sha256="b" * 64,
        )
    assert prepared.device_ordinal == target
    assert tuple(prepared.arrays) == tuple(array.name for array in source.manifest.arrays)
    assert evidence.possible_dirty_device_ordinals == (target,)
    assert evidence.proven_quiescent_device_ordinals == (target,)
    assert evidence.unknown_device_scope is False
    assert lifecycle.guardian.state == "SEALED"
    assert torch.cuda.current_device() == ambient


@pytest.mark.parametrize("lane", ["warp", "cuda"])
@pytest.mark.parametrize("failure_point", ["first_upload", "ready_event_wait"])
def test_real_a2_failure_retains_partial_graph_until_target_abort_sync(
    lane, failure_point, monkeypatch
):
    torch, wp = _require_both_backends()
    import optical_worker.native_setup as setup_module

    target = 0
    source = _source()
    lifecycle = NativeWorkerSetupLifecycle(
        lane_id=lane,
        target_device_ordinal=target,
        emergency_sync_notifier=_noop_abort_notifier,
        session_identity=f"{lane}-{failure_point}",
    )
    sync_observations: list[tuple[int, int]] = []

    if lane == "cuda":
        original_copy = setup_module._torch_copy_exact
        original_wait = setup_module._torch_wait_event_host_exact
        original_sync = torch.cuda.synchronize

        def fail_copy(*args, **kwargs):
            original_copy(*args, **kwargs)
            raise RuntimeError("injected failure after CUDA H2D enqueue")

        def fail_wait(event):
            del event
            raise RuntimeError("injected CUDA ready-event wait failure")

        def checked_sync(device=None):
            sync_observations.append(
                (lifecycle.guardian.resource_count, lifecycle.target_device_ordinal)
            )
            return original_sync(device)

        monkeypatch.setattr(
            setup_module,
            "_torch_copy_exact",
            fail_copy if failure_point == "first_upload" else original_copy,
        )
        monkeypatch.setattr(
            setup_module,
            "_torch_wait_event_host_exact",
            fail_wait if failure_point == "ready_event_wait" else original_wait,
        )
        monkeypatch.setattr(torch.cuda, "synchronize", checked_sync)
        prepare = prepare_cuda_native_input_for_closed_worker
    else:
        original_copy = setup_module._warp_copy_exact
        original_wait = setup_module._warp_wait_event_host_exact
        original_sync = wp.synchronize_device

        def fail_copy(*args, **kwargs):
            original_copy(*args, **kwargs)
            raise RuntimeError("injected failure after Warp H2D enqueue")

        def fail_wait(warp, event):
            del warp, event
            raise RuntimeError("injected Warp ready-event wait failure")

        def checked_sync(device=None):
            sync_observations.append(
                (lifecycle.guardian.resource_count, lifecycle.target_device_ordinal)
            )
            return original_sync(device)

        monkeypatch.setattr(
            setup_module,
            "_warp_copy_exact",
            fail_copy if failure_point == "first_upload" else original_copy,
        )
        monkeypatch.setattr(
            setup_module,
            "_warp_wait_event_host_exact",
            fail_wait if failure_point == "ready_event_wait" else original_wait,
        )
        monkeypatch.setattr(wp, "synchronize_device", checked_sync)
        prepare = prepare_warp_native_input_for_closed_worker

    with pytest.raises(NativeWorkerPreparationAborted) as caught:
        prepare(source, lifecycle=lifecycle)
    evidence = caught.value.evidence
    assert sync_observations and sync_observations[0][0] > 0
    assert evidence.disposition == "aborted_synced_exit"
    assert evidence.emergency_sync_attempt_count == 1
    assert evidence.emergency_sync_attempted_ordinal == target
    assert evidence.resource_release_authorized is True
    assert lifecycle.guardian.state == "RELEASED"


@pytest.mark.parametrize("lane", ["warp", "cuda"])
def test_real_foreign_device_return_uses_process_quarantine_without_target_sync(lane):
    torch, wp = _require_both_backends()
    if min(torch.cuda.device_count(), wp.get_cuda_device_count()) < 2:
        pytest.skip("two CUDA devices are required")
    root = Path(__file__).resolve().parents[2]
    script = f"""
import json, os
from optical_worker.lifecycle import DeviceScopeDescriptor, NativeWorkerSetupLifecycle
lane = {lane!r}
lifecycle = NativeWorkerSetupLifecycle(
    lane_id=lane, target_device_ordinal=1,
    emergency_sync_notifier=lambda ordinal: None,
    session_identity=lane+'-foreign')
permit = lifecycle.begin_device_leaf(DeviceScopeDescriptor.exact_target(
    operation_id='lying_helper', leaf_kind='allocation', device_ordinal=1,
    scope_source='frozen_helper_contract', runtime_identity='fault_probe_v1'))
lifecycle.enter_device_leaf(permit)
if lane == 'cuda':
    import torch
    torch.cuda.set_device(0)
    owner = torch.empty((16,), device='cuda:0')
    actual = owner.device.index
else:
    import warp as wp
    wp.init()
    owner = wp.empty(shape=(16,), dtype=wp.float32, device='cuda:0')
    actual = int(owner.device.ordinal)
try:
    lifecycle.adopt_returned_device_owner(
        owner, actual_device_ordinal=actual, role='foreign_buffer', operation_id='lying_helper')
except Exception:
    pass
sync_calls = []
evidence = lifecycle.terminate(
    fault='device_scope_compromised', detection_phase='returned_owner_validation',
    emergency_target_synchronize=lambda: sync_calls.append(1))
print(json.dumps({{
    'disposition': evidence.disposition,
    'dirty': list(evidence.possible_dirty_device_ordinals),
    'sync_calls': sync_calls,
    'held': lifecycle.guardian.contains_exact_owner(owner),
    'release': evidence.resource_release_authorized,
}}, sort_keys=True), flush=True)
os._exit(0)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload == {
        "dirty": [0, 1],
        "disposition": "multi_or_unknown_device_quarantine_exit",
        "held": True,
        "release": False,
        "sync_calls": [],
    }


@pytest.mark.parametrize("lane", ["warp", "cuda"])
def test_real_warmup_interruption_retains_registered_operands_until_abort_sync(lane):
    torch, wp = _require_both_backends()
    source = _source()
    lifecycle = NativeWorkerSetupLifecycle(
        lane_id=lane,
        target_device_ordinal=0,
        emergency_sync_notifier=_noop_abort_notifier,
        session_identity=f"{lane}-warmup-failure",
    )
    if lane == "cuda":
        prepared = prepare_cuda_native_input_for_closed_worker(source, lifecycle=lifecycle)
        target = torch.device("cuda", 0)
        _enter(lifecycle,
            DeviceScopeDescriptor.exact_target(
                operation_id="cuda_warmup_stream",
                leaf_kind="allocation",
                device_ordinal=0,
                scope_source="closed_worker_allowlist",
                runtime_identity="torch_cuda_stream_explicit_device_v1",
            )
        )
        stream = torch.cuda.Stream(device=target)
        lifecycle.adopt_returned_device_owner(
            stream,
            actual_device_ordinal=0,
            role="cuda_warmup_stream",
            operation_id="cuda_warmup_stream",
        )
        destination = next(iter(prepared.arrays.values()))
        lifecycle.require_adopted(destination, stream)
        _enter(lifecycle,
            DeviceScopeDescriptor.exact_target(
                operation_id="cuda_warmup_launch",
                leaf_kind="warmup_launch",
                device_ordinal=0,
                scope_source="closed_worker_allowlist",
                runtime_identity="torch_fill_explicit_stream_v1",
            )
        )
        with torch.cuda.stream(stream):
            destination.zero_()

        def sync() -> None:
            torch.cuda.synchronize(target)
    else:
        prepared = prepare_warp_native_input_for_closed_worker(source, lifecycle=lifecycle)
        target = wp.get_device("cuda:0")
        _enter(lifecycle,
            DeviceScopeDescriptor.exact_target(
                operation_id="warp_warmup_stream",
                leaf_kind="allocation",
                device_ordinal=0,
                scope_source="closed_worker_allowlist",
                runtime_identity="warp_stream_explicit_device_v1",
            )
        )
        stream = wp.Stream(device=target)
        lifecycle.adopt_returned_device_owner(
            stream,
            actual_device_ordinal=0,
            role="warp_warmup_stream",
            operation_id="warp_warmup_stream",
        )
        destination = next(iter(prepared.arrays.values()))
        host = np.zeros(destination.shape, dtype=np.float32)
        lifecycle.adopt_host_owner(
            host,
            role="warp_warmup_host_staging",
            operation_id="warp_warmup_source",
        )
        source_array = wp.array(host, dtype=destination.dtype, device="cpu", copy=False)
        lifecycle.adopt_host_owner(
            source_array,
            role="warp_warmup_cpu_array",
            operation_id="warp_warmup_source",
        )
        lifecycle.require_adopted(destination, source_array, stream)
        _enter(lifecycle,
            DeviceScopeDescriptor.exact_target(
                operation_id="warp_warmup_launch",
                leaf_kind="warmup_launch",
                device_ordinal=0,
                scope_source="closed_worker_allowlist",
                runtime_identity="warp_copy_explicit_stream_v1",
            )
        )
        wp.copy(destination, source_array, stream=stream)

        def sync() -> None:
            wp.synchronize_device(target)

    resource_count_during_sync: list[int] = []

    def checked_sync() -> None:
        resource_count_during_sync.append(lifecycle.guardian.resource_count)
        sync()

    evidence = lifecycle.terminate(
        fault="setup_failure",
        detection_phase="warmup_launch",
        emergency_target_synchronize=checked_sync,
    )
    assert resource_count_during_sync[0] > 0
    assert evidence.setup_work_counts["warmup_launch"] == 1
    assert evidence.disposition == "aborted_synced_exit"
    assert lifecycle.guardian.state == "RELEASED"
