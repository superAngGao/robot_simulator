from __future__ import annotations

import pytest
from types import SimpleNamespace

from optics.cuda_arena import (
    CUDA_NATIVE_ARENA_PLAN_SCHEMA_VERSION,
    CUDA_NATIVE_ARENA_RESOURCE_SCHEMA_VERSION,
    CudaArenaLogicalResourceView,
    CudaArenaPlanEvidence,
    CudaArenaResourceRecord,
    CudaNativeOpticalArena,
)


def _view(**changes) -> CudaArenaLogicalResourceView:
    values = {
        "logical_name": "slot0.render.rgb",
        "dtype": "torch.float32",
        "shape": (2, 3),
        "stride": (3, 1),
        "active_offset_bytes": 0,
        "active_nbytes": 24,
    }
    values.update(changes)
    return CudaArenaLogicalResourceView(**values)


def test_cuda_resource_record_serializes_every_logical_view_and_byte_range():
    first = _view()
    second = _view(
        logical_name="slot0.render.rgb_alias",
        shape=(1, 3),
        active_offset_bytes=12,
        active_nbytes=12,
    )
    record = CudaArenaResourceRecord(
        schema_version=CUDA_NATIVE_ARENA_RESOURCE_SCHEMA_VERSION,
        resource_kind="cuda_storage",
        device_ordinal=0,
        storage_token="cuda-storage:0:1:2",
        logical_views=(first, second),
        nbytes=24,
    )

    assert record.logical_names == (
        "slot0.render.rgb",
        "slot0.render.rgb_alias",
    )
    assert record.logical_view("slot0.render.rgb_alias") is second
    assert record.to_payload()["logical_views"][1]["active_offset_bytes"] == 12


def test_cuda_resource_record_rejects_duplicate_or_out_of_bounds_views():
    with pytest.raises(ValueError, match="unique"):
        CudaArenaResourceRecord(
            schema_version=CUDA_NATIVE_ARENA_RESOURCE_SCHEMA_VERSION,
            resource_kind="cuda_storage",
            device_ordinal=0,
            storage_token="cuda-storage:0:1:2",
            logical_views=(_view(), _view()),
            nbytes=24,
        )
    with pytest.raises(ValueError, match="exceeds"):
        CudaArenaResourceRecord(
            schema_version=CUDA_NATIVE_ARENA_RESOURCE_SCHEMA_VERSION,
            resource_kind="cuda_storage",
            device_ordinal=0,
            storage_token="cuda-storage:0:1:2",
            logical_views=(_view(active_offset_bytes=4),),
            nbytes=24,
        )


def test_cuda_plan_evidence_rejects_noncanonical_sequence_types():
    values = {
        "schema_version": CUDA_NATIVE_ARENA_PLAN_SCHEMA_VERSION,
        "arena_spec_sha256": "1" * 64,
        "slot_id": 0,
        "slot_storage_manifest_sha256": "2" * 64,
        "slot_resource_contract_sha256": "2" * 64,
        "transform_frame_index": 0,
        "stage_names": ["cuda_scene_update"],
        "ordered_runtime_stage_evidence_sha256": "3" * 64,
        "terminal_event_token": "event:0",
        "diagnostic_phase_names": (),
        "plan_binding_sha256": "4" * 64,
    }
    with pytest.raises(ValueError, match="stage names"):
        CudaArenaPlanEvidence(**values)


@pytest.mark.parametrize("transform", [None, 3])
def test_camera_sample_selection_is_independent_of_transform_replay(transform):
    # Exercise the real selector and FIFO planner without requiring GPU storage.
    targets = {(transform, sample): object() for sample in (0, 1)}
    session = SimpleNamespace(
        _validate_configuration=lambda: None,
        _slots=(SimpleNamespace(targets={}), SimpleNamespace(targets=targets)),
        _control=SimpleNamespace(lifecycle_evidence=lambda: SimpleNamespace(
            next_acquire_slot_id=1, slots=(SimpleNamespace(generation=4), SimpleNamespace(generation=3)))),
    )
    for sample in (0, 1):
        assert CudaNativeOpticalArena.plan_target(
            session, transform_frame_index=transform, render_sample_index=sample) is targets[transform, sample]
    with pytest.raises(ValueError, match="sealed graph"):
        CudaNativeOpticalArena.plan_target(session, transform_frame_index=transform, render_sample_index=2)


@pytest.mark.parametrize("selector", ["render_sample_index", "transform_frame_index"])
@pytest.mark.parametrize("index", [True, 1.0, -1])
def test_frame_selection_rejects_noninteger_or_negative_indices(selector, index):
    indices = dict(transform_frame_index=None, render_sample_index=None)
    indices[selector] = index
    with pytest.raises(ValueError, match="indices"):
        CudaNativeOpticalArena.plan_target(object(), **indices)


def test_legacy_cuda_provenance_adapter_uses_the_production_implementation():
    from benchmarks.optical import cuda_arena as legacy
    from optical_worker import cuda_arena as production

    assert legacy.build_cuda_c3_binding_from_arena_target is production.build_cuda_c3_binding_from_arena_target
