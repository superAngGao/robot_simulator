from __future__ import annotations

import pytest

from optics.cuda_arena import (
    CUDA_NATIVE_ARENA_PLAN_SCHEMA_VERSION,
    CUDA_NATIVE_ARENA_RESOURCE_SCHEMA_VERSION,
    CudaArenaLogicalResourceView,
    CudaArenaPlanEvidence,
    CudaArenaResourceRecord,
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
