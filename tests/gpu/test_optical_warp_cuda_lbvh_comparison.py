from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.optical.eligibility import (
    CANONICAL_NATIVE_REPLAY_INPUT,
    verified_runtime_entry_point_audit,
)
from benchmarks.optical.instrumentation import (
    instrument_runtime_entry_points,
    invoke_instrumented_entry_point,
)
from benchmarks.optical.lanes.warp import (
    WARP_CUDA_LBVH_COMPARISON_SCHEMA_VERSION,
    WarpCudaLbvhComparisonFactory,
    allocate_warp_cuda_lbvh_comparison_storage,
)
from benchmarks.optical.native import CanonicalNativeInput, prepare_warp_native_input
from benchmarks.optical.timing import RuntimeOperationTrace, native_timed_submission
from benchmarks.optical.workloads import (
    build_canonical_packed_scene,
    build_canonical_transform_replay,
)
from optics.cuda_lbvh import CUDA_LBVH_ACCELERATION_ID, CUDA_LBVH_OPERATION_ID
from optics.device import stage_optical_compute_result_to_host
from optics.device_bvh import build_device_bvh_from_snapshot
from optics.device_scene import allocate_warp_world_scene_buffers
from optics.render_api import OpticalRaySensorSpec
from optics.warp_execution import GpuDeviceBvhOpticalExecutor

pytestmark = pytest.mark.gpu

_SCENE_ENTRY = "benchmarks.optical.lanes.warp.WarpCudaLbvhComparisonCommandBundle.enqueue_scene_update"
_ACCELERATION_ENTRY = "benchmarks.optical.lanes.warp.WarpCudaLbvhComparisonCommandBundle.enqueue_acceleration"


def test_warp_static_comparison_uses_one_stream_and_zero_timed_bridge():
    setup = _comparison_setup()
    bundle = setup.bundle
    evidence = bundle.evidence
    assert evidence.schema_version == WARP_CUDA_LBVH_COMPARISON_SCHEMA_VERSION
    assert evidence.acceleration_class == "production_static"
    assert evidence.acceleration_id == CUDA_LBVH_ACCELERATION_ID
    assert evidence.operation_id == CUDA_LBVH_OPERATION_ID
    assert evidence.warp_stream_handle == evidence.torch_stream_handle
    assert evidence.warp_stream_handle == int(setup.stream.cuda_stream)
    assert evidence.setup_stable_view_count == len(bundle.stable_views) == 20
    assert evidence.bridge_creation_count == 0
    assert evidence.warp_scene_stage_evidence_sha256 == bundle.scene_bundle.evidence.sha256
    assert evidence.cuda_lbvh_stage_evidence_sha256 == bundle.acceleration_stage.evidence.sha256
    assert bundle.warp_bvh.ready_event is None
    for stable in bundle.stable_views:
        warp_ptr = 0 if stable.warp_array.ptr is None else int(stable.warp_array.ptr)
        assert warp_ptr == int(stable.torch_tensor.data_ptr())

    bundle.arm()
    audits = (
        verified_runtime_entry_point_audit(
            operation="scene_update",
            entry_point=_SCENE_ENTRY,
        ),
        verified_runtime_entry_point_audit(
            operation="acceleration",
            entry_point=_ACCELERATION_ENTRY,
        ),
    )
    assert "optics.cuda_lbvh.load_inline" in audits[1].registered_runtime_hooks
    assert "warp.to_torch" in audits[1].registered_runtime_hooks
    trace = RuntimeOperationTrace(row_id="unit_warp_static_r2", lane="warp", boundary="R2")
    with instrument_runtime_entry_points(audits) as observation:
        with native_timed_submission(
            row_id=trace.row_id,
            lane=trace.lane,
            boundary=trace.boundary,
            operation_trace=trace,
        ):
            invoke_instrumented_entry_point(_SCENE_ENTRY, bundle)
            invoke_instrumented_entry_point(_ACCELERATION_ENTRY, bundle)

    assert observation.violation_hook_id is None
    assert "warp.to_torch" not in observation.invoked_hook_ids
    assert "warp.from_torch" not in observation.invoked_hook_ids
    assert bundle.state == "acceleration_enqueued"
    assert [record.operation for record in trace.evidence().records] == [
        "scene_update",
        "acceleration",
    ]
    done = setup.stream.record_event()
    setup.wp.synchronize_event(done)
    _assert_warp_renderer_parity(setup)


def test_warp_static_comparison_rejects_order_mutation_and_wrong_class():
    setup = _comparison_setup()
    bundle = setup.bundle
    with pytest.raises(RuntimeError, match="requires state 'scene_enqueued'"):
        bundle.enqueue_acceleration()
    with pytest.raises(AttributeError, match="immutable"):
        setup.factory.acceleration_class = "production_dynamic"
    with pytest.raises(AttributeError, match="immutable"):
        bundle._sealed = False
    with pytest.raises(ValueError, match="must come from prepare"):
        WarpCudaLbvhComparisonFactory(
            prepared=setup.factory.prepared,
            acceleration_class="production_static",
            stream=setup.factory.stream,
            external_stream=setup.factory.external_stream,
            context=setup.factory.context,
            scene_factory=setup.factory.scene_factory,
            cuda_lbvh_factory=setup.factory.cuda_lbvh_factory,
        )

    tampered = _comparison_setup().bundle
    object.__setattr__(tampered, "stable_views", tuple(reversed(tampered.stable_views)))
    with pytest.raises(ValueError, match="stable-view ABI"):
        tampered.arm()

    tampered_evidence = _comparison_setup().bundle
    object.__setattr__(
        tampered_evidence.evidence,
        "native_input_manifest_sha256",
        "0" * 64,
    )
    with pytest.raises(ValueError, match="evidence changed"):
        tampered_evidence.arm()

    tampered_traversal = _comparison_setup().bundle
    object.__setattr__(
        tampered_traversal.warp_bvh,
        "left",
        tampered_traversal.warp_bvh.right,
    )
    with pytest.raises(ValueError, match="traversal view"):
        tampered_traversal.arm()

    with pytest.raises(ValueError, match="restricted to production_static"):
        WarpCudaLbvhComparisonFactory.prepare(
            setup.prepared,
            stream=setup.wp.Stream(device=setup.scene.device),
            acceleration_class="production_dynamic",
        )


def test_warp_static_comparison_restores_wrong_ambient_torch_device():
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    setup = _prepared_setup()
    previous = torch.cuda.current_device()
    try:
        torch.cuda.set_device(1)
        assert torch.cuda.current_device() == 1
        factory = WarpCudaLbvhComparisonFactory.prepare(
            setup.prepared,
            stream=setup.stream,
            acceleration_class="production_static",
        )
        assert torch.cuda.current_device() == 1
        world = allocate_warp_world_scene_buffers(
            factory.scene_factory.local_scene,
            stream=setup.stream,
        )
        storage = allocate_warp_cuda_lbvh_comparison_storage(
            factory.cuda_lbvh_factory.requirements,
            device=factory.scene_factory.local_scene.device,
            stream=setup.stream,
        )
        bundle = factory.seal(transform_frame_index=1, world=world, storage=storage)
        assert torch.cuda.current_device() == 1
        bundle.arm()
        bundle.enqueue_scene_update()
        assert torch.cuda.current_device() == 1
        bundle.enqueue_acceleration()
        assert torch.cuda.current_device() == 1
        done = setup.stream.record_event()
        setup.wp.synchronize_event(done)
        assert torch.cuda.current_device() == 1
    finally:
        torch.cuda.set_device(previous)


def _prepared_setup():
    wp = pytest.importorskip("warp")
    if wp.get_cuda_device_count() == 0:
        pytest.skip("CUDA is required")
    packed = build_canonical_packed_scene("synthetic_dynamic_rigid_960x640_v1")
    replay = build_canonical_transform_replay("synthetic_dynamic_rigid_960x640_v1")
    source = CanonicalNativeInput.from_scene_contract(
        packed,
        input_contract=CANONICAL_NATIVE_REPLAY_INPUT,
        replay=replay,
    )
    prepared = prepare_warp_native_input(source, device="cuda:0")
    stream = wp.Stream(device=f"cuda:{prepared.device_ordinal}")
    return SimpleNamespace(wp=wp, prepared=prepared, stream=stream)


def _comparison_setup():
    setup = _prepared_setup()
    factory = WarpCudaLbvhComparisonFactory.prepare(
        setup.prepared,
        stream=setup.stream,
        acceleration_class="production_static",
    )
    scene = factory.scene_factory.local_scene
    world = allocate_warp_world_scene_buffers(scene, stream=setup.stream)
    storage = allocate_warp_cuda_lbvh_comparison_storage(
        factory.cuda_lbvh_factory.requirements,
        device=scene.device,
        stream=setup.stream,
    )
    bundle = factory.seal(transform_frame_index=1, world=world, storage=storage)
    return SimpleNamespace(
        **setup.__dict__,
        factory=factory,
        scene=scene,
        world=world,
        storage=storage,
        bundle=bundle,
    )


def _assert_warp_renderer_parity(setup) -> None:
    snapshot = setup.world.as_snapshot(
        setup.scene,
        frame_id=1,
        sim_time=1.0,
        env_idx=0,
    )
    fresh_bvh = build_device_bvh_from_snapshot(
        snapshot,
        leaf_size=4,
        split_strategy="sort",
        device=setup.scene.device,
        stream=setup.stream,
    )
    setup.wp.synchronize_event(fresh_bvh.ready_event)
    rays = OpticalRaySensorSpec(
        frame_id=1,
        sim_time=1.0,
        env_idx=0,
        sensor_id="warp_cuda_lbvh_c3e",
        origins_world=[[0.05, 0.05, 2.0], [2.0, 2.0, 2.0]],
        directions_world=[[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
        max_distance=10.0,
    )
    executor = GpuDeviceBvhOpticalExecutor(device=setup.scene.device, stream=setup.stream)
    comparison = executor.execute(snapshot, setup.bundle.warp_bvh, rays)
    oracle = executor.execute(snapshot, fresh_bvh, rays)
    done = setup.stream.record_event()
    setup.wp.synchronize_event(done)
    comparison_host = stage_optical_compute_result_to_host(comparison)
    oracle_host = stage_optical_compute_result_to_host(oracle)
    for name in (
        "hit_mask",
        "range_m",
        "position_world",
        "normal_world",
        "numeric_instance_id",
    ):
        np.testing.assert_allclose(
            comparison_host.channel(name),
            oracle_host.channel(name),
            rtol=0.0,
            atol=1.0e-6,
            equal_nan=True,
        )
