"""Frozen-gate adapter for the production native optical arena contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from optics.native_arena import (
    NATIVE_OPTICAL_ARENA_LAYOUT_SCHEMA_VERSION,
    NATIVE_OPTICAL_ARENA_SPEC_SCHEMA_VERSION,
    NativeOpticalArenaSpec,
)

from .cases import get_case
from .eligibility import EXACT_LANE_BINDINGS, INPUT_CONTRACT_BY_BOUNDARY
from .native import NativeReadinessEvidence, PreparedNativeInput


def native_optical_arena_spec_from_frozen_row(
    *,
    gate: Mapping[str, Any],
    row_id: str,
    prepared: PreparedNativeInput,
    readiness: NativeReadinessEvidence,
    execution_stream_token: str,
    primitive_count: int,
    node_capacity: int,
    diagnostic_mode: str = "primary_unprofiled",
) -> NativeOpticalArenaSpec:
    """Derive one production arena spec after exact frozen-row validation.

    Physical slot allocation and C3 stage binding are C4b/C4c work.  This
    adapter only freezes the logical policy and provenance that those builders
    must satisfy.
    """

    # Local import avoids making the production arena depend on benchmark
    # schema code while still requiring the caller's gate to be canonical.
    from .schema import validate_frozen_gate

    validate_frozen_gate(gate)
    rows = {str(row["row_id"]): row for row in gate["mandatory_rows"] if isinstance(row, Mapping)}
    row = rows.get(row_id)
    if row is None:
        raise KeyError(f"unknown frozen native optical row {row_id!r}")
    boundary = str(row["boundary"])
    if boundary not in {"R0", "R1", "R2"}:
        raise ValueError("C4a native arena adapter only accepts R0/R1/R2 rows")
    if not prepared.manifest.result_capable:
        raise ValueError("native optical arena requires result-capable prepared input")
    manifest = prepared.manifest
    canonical_readiness = NativeReadinessEvidence.from_payload(readiness.to_payload())
    expected_case = gate["cases"][row["case_id"]]
    expected_input_contract = INPUT_CONTRACT_BY_BOUNDARY[boundary]
    expected_lane = prepared.backend
    expected_binding = EXACT_LANE_BINDINGS[expected_lane]
    expected_acceleration = expected_binding["acceleration_by_class"][row["acceleration_class"]]

    checks = {
        "case_id": (manifest.case_id, row["case_id"]),
        "case_sha256": (manifest.case_sha256, expected_case["sha256"]),
        "input_contract": (manifest.input_contract, expected_input_contract),
        "scene_contract_sha256": (
            manifest.scene_contract_sha256,
            expected_case["scene_contract_sha256"],
        ),
        "readiness backend": (canonical_readiness.backend, expected_lane),
        "readiness device": (canonical_readiness.device_ordinal, prepared.device_ordinal),
        "readiness manifest": (canonical_readiness.manifest_sha256, manifest.sha256),
        "readiness content": (canonical_readiness.content_sha256, manifest.content_sha256),
        "readiness topology": (
            canonical_readiness.topology_sha256,
            manifest.topology_sha256,
        ),
        "readiness input contract": (
            canonical_readiness.input_contract,
            manifest.input_contract,
        ),
        "readiness scene contract": (
            canonical_readiness.scene_contract_sha256,
            manifest.scene_contract_sha256,
        ),
        "readiness replay": (
            canonical_readiness.transform_replay_sha256,
            manifest.transform_replay_sha256,
        ),
        "readiness fixture": (
            canonical_readiness.fixed_render_fixture_sha256,
            manifest.fixed_render_fixture_sha256,
        ),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"native optical arena {label} does not match its frozen row")

    if boundary == "R0":
        if manifest.fixed_render_fixture_sha256 != expected_case["fixed_render_fixture_sha256"]:
            raise ValueError("R0 arena fixture does not match frozen workload content")
        payload = "native_only"
        delivery_policy = "native_only"
        payload_contract_id = None
        ring_depth = 1
    else:
        if manifest.transform_replay_sha256 != expected_case["transform_replay_sha256"]:
            raise ValueError("dynamic arena replay does not match frozen workload content")
        if boundary == "R1":
            payload = "native_only"
            delivery_policy = "native_only"
            payload_contract_id = None
            ring_depth = 1
        else:
            payload = str(row["payload"])
            delivery_policy = str(row["delivery_policy"])
            payload_contract_id = str(row["payload_contract_id"])
            ring_depth = int(row["ring_depth"])

    case = get_case(str(row["case_id"]))
    return NativeOpticalArenaSpec(
        schema_version=NATIVE_OPTICAL_ARENA_SPEC_SCHEMA_VERSION,
        row_id=str(row["row_id"]),
        case_id=manifest.case_id,
        case_sha256=manifest.case_sha256,
        boundary=boundary,
        lane=expected_lane,
        device_ordinal=int(prepared.device_ordinal),
        execution_stream_token=execution_stream_token,
        input_contract=expected_input_contract,
        native_input_manifest_sha256=manifest.sha256,
        native_readiness_sha256=canonical_readiness.sha256,
        scene_contract_sha256=str(manifest.scene_contract_sha256),
        transform_replay_sha256=manifest.transform_replay_sha256,
        transform_frame_count=manifest.transform_frame_count,
        fixed_render_fixture_sha256=manifest.fixed_render_fixture_sha256,
        renderer_id=str(expected_binding["renderer_id"]),
        acceleration_class=str(row["acceleration_class"]),
        acceleration_id=str(expected_acceleration),
        width=case.width,
        height=case.height,
        num_rays=case.width * case.height,
        primitive_count=primitive_count,
        node_capacity=node_capacity,
        output_profile=str(row["output_profile"]),
        shadows=bool(row["shadows"]),
        diagnostic_mode=diagnostic_mode,
        payload=payload,
        delivery_policy=delivery_policy,
        payload_contract_id=payload_contract_id,
        ring_depth=ring_depth,
        ordered_delivery=True,
        arena_layout_schema_version=NATIVE_OPTICAL_ARENA_LAYOUT_SCHEMA_VERSION,
    )
