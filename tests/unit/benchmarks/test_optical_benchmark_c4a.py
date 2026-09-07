from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest

from benchmarks.optical.arena import native_optical_arena_spec_from_frozen_row
from benchmarks.optical.eligibility import (
    CANONICAL_NATIVE_REPLAY_INPUT,
    FIXED_NATIVE_SCENE_INPUT,
)
from benchmarks.optical.native import (
    CanonicalNativeInput,
    NativeReadinessEvidence,
    WarpPreparedNativeInput,
)
from benchmarks.optical.schema import load_frozen_gate
from benchmarks.optical.workloads import (
    build_canonical_fixed_fixture,
    build_canonical_packed_scene,
    build_canonical_transform_replay,
)
from optics.native_arena import NATIVE_OPTICAL_ARENA_SPEC_SCHEMA_VERSION


def test_frozen_r0_row_derives_exact_native_only_arena_policy():
    gate = load_frozen_gate()
    row_id = "r0_asymmetric_full_shadow"
    prepared = _prepared(row_id=row_id, gate=gate)
    spec = native_optical_arena_spec_from_frozen_row(
        gate=gate,
        row_id=row_id,
        prepared=prepared,
        readiness=_readiness(prepared),
        execution_stream_token="warp-stream:0:11",
        primitive_count=21,
        node_capacity=41,
    )

    assert spec.schema_version == NATIVE_OPTICAL_ARENA_SPEC_SCHEMA_VERSION
    assert spec.boundary == "R0"
    assert spec.width == 7
    assert spec.height == 3
    assert spec.num_rays == 21
    assert spec.payload == "native_only"
    assert spec.delivery_policy == "native_only"
    assert spec.payload_contract_id is None
    assert spec.ring_depth == 1
    assert spec.fixed_render_fixture_sha256 == prepared.manifest.fixed_render_fixture_sha256
    assert spec.transform_replay_sha256 is None


def test_frozen_r1_row_binds_replay_and_lane_acceleration_identity():
    gate = load_frozen_gate()
    row_id = "r1_dynamic_full_shadow"
    prepared = _prepared(row_id=row_id, gate=gate)
    readiness = _readiness(prepared)
    spec = native_optical_arena_spec_from_frozen_row(
        gate=gate,
        row_id=row_id,
        prepared=prepared,
        readiness=readiness,
        execution_stream_token="warp-stream:0:12",
        primitive_count=1,
        node_capacity=1,
    )

    assert spec.boundary == "R1"
    assert spec.acceleration_class == "production_dynamic"
    assert spec.acceleration_id == "cpu_median_leaf4_topology_warp_refit_v1"
    assert spec.transform_replay_sha256 == prepared.manifest.transform_replay_sha256
    assert spec.transform_frame_count == 120
    assert spec.native_readiness_sha256 == readiness.sha256


def test_adapter_rejects_readiness_or_gate_drift_before_resource_seal():
    gate = load_frozen_gate()
    row_id = "r1_dynamic_render_only_no_shadow"
    prepared = _prepared(row_id=row_id, gate=gate)
    wrong_readiness = replace(_readiness(prepared), manifest_sha256="0" * 64)
    with pytest.raises(ValueError, match="readiness manifest"):
        native_optical_arena_spec_from_frozen_row(
            gate=gate,
            row_id=row_id,
            prepared=prepared,
            readiness=wrong_readiness,
            execution_stream_token="warp-stream:0:13",
            primitive_count=1,
            node_capacity=1,
        )

    changed_gate = {**gate, "schema_version": "forged"}
    with pytest.raises(ValueError, match="schema_version"):
        native_optical_arena_spec_from_frozen_row(
            gate=changed_gate,
            row_id=row_id,
            prepared=prepared,
            readiness=_readiness(prepared),
            execution_stream_token="warp-stream:0:13",
            primitive_count=1,
            node_capacity=1,
        )


def _prepared(*, row_id: str, gate):
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == row_id)
    packed = build_canonical_packed_scene(row["case_id"])
    if row["boundary"] == "R0":
        source = CanonicalNativeInput.from_scene_contract(
            packed,
            input_contract=FIXED_NATIVE_SCENE_INPUT,
            fixed_fixture=build_canonical_fixed_fixture(packed),
        )
    else:
        source = CanonicalNativeInput.from_scene_contract(
            packed,
            input_contract=CANONICAL_NATIVE_REPLAY_INPUT,
            replay=build_canonical_transform_replay(row["case_id"]),
        )
    return WarpPreparedNativeInput(
        manifest=source.manifest,
        device_ordinal=0,
        arrays=MappingProxyType({name: object() for name in source.arrays}),
        ready_event=object(),
        source=source,
    )


def _readiness(prepared) -> NativeReadinessEvidence:
    manifest = prepared.manifest
    return NativeReadinessEvidence(
        schema_version=manifest.schema_version,
        backend=prepared.backend,
        device_ordinal=prepared.device_ordinal,
        manifest_sha256=manifest.sha256,
        content_sha256=manifest.content_sha256,
        topology_sha256=manifest.topology_sha256,
        input_contract=manifest.input_contract,
        scene_contract_schema_version=manifest.scene_contract_schema_version,
        scene_contract_sha256=manifest.scene_contract_sha256,
        scene_id=manifest.scene_id,
        transform_replay_sha256=manifest.transform_replay_sha256,
        transform_frame_count=manifest.transform_frame_count,
        fixed_render_fixture_sha256=manifest.fixed_render_fixture_sha256,
        producer_ready_event_complete=True,
        direct_validation_complete=True,
        allocation_complete=True,
        compile_complete=True,
        warmup_frames_completed=1,
    )
