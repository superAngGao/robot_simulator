"""Fail-closed P12.4a3a lane identities, audits, and preflight evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .timing import (
    FORBIDDEN_TIMED_OPERATIONS,
    RUNTIME_OPERATION_KINDS,
    RuntimeOperationTraceEvidence,
)

ELIGIBILITY_SCHEMA_VERSION = "q54.p12.4.lane-eligibility.v1"
ENTRY_POINT_AUDIT_SCHEMA_VERSION = "q54.p12.4.entry-point-audit.v1"
INPUT_CONTRACT_SCHEMA_VERSION = "q54.p12.4.input-contract.v1"

FIXED_NATIVE_SCENE_INPUT = "fixed_native_scene_v1"
CANONICAL_NATIVE_REPLAY_INPUT = "canonical_native_replay_v1"
GPU_ENGINE_PUBLISHED_FRAME_INPUT = "gpu_engine_published_frame_v1"

INPUT_CONTRACT_BY_BOUNDARY = {
    "R0": FIXED_NATIVE_SCENE_INPUT,
    "R1": CANONICAL_NATIVE_REPLAY_INPUT,
    "R2": CANONICAL_NATIVE_REPLAY_INPUT,
    "S": GPU_ENGINE_PUBLISHED_FRAME_INPUT,
}

EXACT_LANE_BINDINGS = {
    "warp": {
        "renderer_id": "warp_bvh_direct_light",
        "acceleration_by_class": {
            "fixed_bvh_fixture": "fixed_bvh_fixture",
            "production_static": "cuda_morton_karras_lbvh_rebuild_v1",
            "production_dynamic": "cpu_median_leaf4_topology_warp_refit_v1",
            "production_physics": "cpu_median_leaf4_topology_warp_refit_v1",
        },
    },
    "cuda": {
        "renderer_id": "cuda_direct_light",
        "acceleration_by_class": {
            "fixed_bvh_fixture": "fixed_bvh_fixture",
            "production_static": "cuda_morton_karras_lbvh_rebuild_v1",
            "production_dynamic": "cuda_morton_karras_lbvh_rebuild_v1",
            "production_physics": "cuda_morton_karras_lbvh_rebuild_v1",
        },
    },
}

EXACT_ACCELERATION_CONTRACTS = {
    "warp": {
        "fixed_bvh_fixture": {
            "topology_origin": "cpu_median_longest_axis_leaf4_source_order_v1",
            "topology_lifetime": "prepared_input_immutable_v1",
            "timed_scene_update_operation": "none",
            "timed_acceleration_operation": "none",
            "operation_classification": "none",
        },
        "cpu_median_leaf4_topology_warp_refit_v1": {
            "topology_origin": "cpu_median_longest_axis_leaf4_source_order_v1",
            "topology_lifetime": "prepared_lane_session_immutable_v1",
            "timed_scene_update_operation": "warp_local_scene_transform_and_aabb_out_v1",
            "timed_acceleration_operation": "warp_bvh_leaf_internal_refit_out_v1",
            "operation_classification": "refit",
        },
        "cuda_morton_karras_lbvh_rebuild_v1": {
            "topology_origin": "cuda_morton30_karras_source_index_tiebreak_v1",
            "topology_lifetime": "transaction_output_arena_lease_v1",
            "timed_scene_update_operation": "warp_local_scene_transform_and_aabb_out_v1",
            "timed_acceleration_operation": ("cuda_lbvh_scene_bounds_morton_sort_topology_bounds_out_v1"),
            "operation_classification": "rebuild",
        },
    },
    "cuda": {
        "fixed_bvh_fixture": {
            "topology_origin": "cpu_median_longest_axis_leaf4_source_order_v1",
            "topology_lifetime": "prepared_input_immutable_v1",
            "timed_scene_update_operation": "none",
            "timed_acceleration_operation": "none",
            "operation_classification": "none",
        },
        "cuda_morton_karras_lbvh_rebuild_v1": {
            "topology_origin": "cuda_morton30_karras_source_index_tiebreak_v1",
            "topology_lifetime": "transaction_output_arena_lease_v1",
            "timed_scene_update_operation": "torch_cuda_local_scene_transform_and_aabb_out_v1",
            "timed_acceleration_operation": ("cuda_lbvh_scene_bounds_morton_sort_topology_bounds_out_v1"),
            "operation_classification": "rebuild",
        },
    },
}


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EntryPointAudit:
    """Static audit of one concrete product call used by a candidate lane."""

    operation: str
    entry_point: str
    instrumented: bool
    contaminations: tuple[str, ...] = ()
    schema_version: str = ENTRY_POINT_AUDIT_SCHEMA_VERSION

    @classmethod
    def from_payload(cls, payload: object) -> "EntryPointAudit":
        expected = {
            "operation",
            "entry_point",
            "instrumented",
            "contaminations",
            "schema_version",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("entry-point audit fields are invalid")
        operation = payload.get("operation")
        entry_point = payload.get("entry_point")
        instrumented = payload.get("instrumented")
        contaminations = payload.get("contaminations")
        if payload.get("schema_version") != ENTRY_POINT_AUDIT_SCHEMA_VERSION:
            raise ValueError("entry-point audit schema version is invalid")
        if operation not in RUNTIME_OPERATION_KINDS:
            raise ValueError("entry-point audit operation is invalid")
        if not isinstance(entry_point, str) or not entry_point:
            raise ValueError("entry-point audit entry_point must be non-empty")
        if not isinstance(instrumented, bool):
            raise ValueError("entry-point audit instrumented must be boolean")
        if not isinstance(contaminations, Sequence) or isinstance(contaminations, (str, bytes)):
            raise ValueError("entry-point audit contaminations must be an array")
        contamination_tuple = tuple(contaminations)
        if len(set(contamination_tuple)) != len(contamination_tuple) or any(
            value not in FORBIDDEN_TIMED_OPERATIONS for value in contamination_tuple
        ):
            raise ValueError("entry-point audit contaminations are invalid")
        return cls(
            operation=operation,
            entry_point=entry_point,
            instrumented=instrumented,
            contaminations=contamination_tuple,
            schema_version=ENTRY_POINT_AUDIT_SCHEMA_VERSION,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "entry_point": self.entry_point,
            "instrumented": self.instrumented,
            "contaminations": list(self.contaminations),
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class LaneEligibilityEvidence:
    """Lane-owned proof that a row may or may not enter timed submission."""

    schema_version: str
    row_id: str
    case_id: str
    boundary: str
    lane: str
    input_contract: str
    renderer_id: str
    acceleration_id: str
    required_operations: tuple[str, ...]
    entry_point_audits: tuple[EntryPointAudit, ...]
    asset_manifest_sha256: str | None
    eligible: bool
    reason_codes: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: object) -> "LaneEligibilityEvidence":
        expected = {
            "schema_version",
            "row_id",
            "case_id",
            "boundary",
            "lane",
            "input_contract",
            "renderer_id",
            "acceleration_id",
            "required_operations",
            "entry_point_audits",
            "asset_manifest_sha256",
            "eligible",
            "reason_codes",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("lane eligibility fields are invalid")
        if payload.get("schema_version") != ELIGIBILITY_SCHEMA_VERSION:
            raise ValueError("lane eligibility schema version is invalid")
        for name in (
            "row_id",
            "case_id",
            "boundary",
            "lane",
            "input_contract",
            "renderer_id",
            "acceleration_id",
        ):
            if not isinstance(payload.get(name), str) or not payload[name]:
                raise ValueError(f"lane eligibility {name} must be non-empty")
        required = payload.get("required_operations")
        audits = payload.get("entry_point_audits")
        reasons = payload.get("reason_codes")
        if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
            raise ValueError("lane eligibility required_operations must be an array")
        required_tuple = tuple(required)
        if len(set(required_tuple)) != len(required_tuple) or any(
            operation not in RUNTIME_OPERATION_KINDS for operation in required_tuple
        ):
            raise ValueError("lane eligibility required_operations are invalid")
        if not isinstance(audits, Sequence) or isinstance(audits, (str, bytes)):
            raise ValueError("lane eligibility entry_point_audits must be an array")
        audit_tuple = tuple(EntryPointAudit.from_payload(item) for item in audits)
        if not isinstance(reasons, Sequence) or isinstance(reasons, (str, bytes)):
            raise ValueError("lane eligibility reason_codes must be an array")
        reason_tuple = tuple(reasons)
        if len(set(reason_tuple)) != len(reason_tuple) or any(
            not isinstance(reason, str) or not reason for reason in reason_tuple
        ):
            raise ValueError("lane eligibility reason_codes are invalid")
        eligible = payload.get("eligible")
        if not isinstance(eligible, bool) or eligible == bool(reason_tuple):
            raise ValueError("lane eligibility eligible/reason_codes are inconsistent")
        asset_hash = payload.get("asset_manifest_sha256")
        if asset_hash is not None and not _sha256_text(asset_hash):
            raise ValueError("lane eligibility asset manifest hash is invalid")
        return cls(
            schema_version=ELIGIBILITY_SCHEMA_VERSION,
            row_id=payload["row_id"],
            case_id=payload["case_id"],
            boundary=payload["boundary"],
            lane=payload["lane"],
            input_contract=payload["input_contract"],
            renderer_id=payload["renderer_id"],
            acceleration_id=payload["acceleration_id"],
            required_operations=required_tuple,
            entry_point_audits=audit_tuple,
            asset_manifest_sha256=asset_hash,
            eligible=eligible,
            reason_codes=reason_tuple,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "row_id": self.row_id,
            "case_id": self.case_id,
            "boundary": self.boundary,
            "lane": self.lane,
            "input_contract": self.input_contract,
            "renderer_id": self.renderer_id,
            "acceleration_id": self.acceleration_id,
            "required_operations": list(self.required_operations),
            "entry_point_audits": [audit.to_payload() for audit in self.entry_point_audits],
            "asset_manifest_sha256": self.asset_manifest_sha256,
            "eligible": self.eligible,
            "reason_codes": list(self.reason_codes),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_payload())


class LaneIneligibleError(RuntimeError):
    """Raised before timing when a lane cannot satisfy the frozen row."""


def required_runtime_operations(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the minimum high-level instrumented operations for one row."""

    boundary = row.get("boundary")
    operations: list[str]
    if boundary == "R0":
        operations = ["render"]
    elif boundary == "R1":
        operations = ["scene_update", "acceleration", "render", "reclaim"]
    elif boundary == "R2":
        operations = ["scene_update", "acceleration", "render"]
        if row.get("payload") == "rgb8":
            operations.append("pack")
        operations.extend(("delivery", "reclaim"))
    elif boundary == "S":
        operations = [
            "physics_step",
            "publication",
            "stream_wait",
            "view_binding",
            "scene_update",
            "consumer_completion",
            "acceleration",
            "render",
        ]
        if row.get("payload") == "rgb8":
            operations.append("pack")
        operations.extend(("delivery", "reclaim"))
    else:
        raise ValueError(f"unsupported benchmark boundary {boundary!r}")
    return tuple(operations)


def evaluate_lane_eligibility(
    *,
    row: Mapping[str, Any],
    lane: Literal["warp", "cuda"],
    entry_point_audits: Sequence[EntryPointAudit],
    asset_manifest_sha256: str | None,
) -> LaneEligibilityEvidence:
    """Evaluate exact identities and audit coverage without entering timing."""

    if lane not in EXACT_LANE_BINDINGS:
        raise ValueError("lane must be 'warp' or 'cuda'")
    boundary = str(row.get("boundary", ""))
    input_contract = INPUT_CONTRACT_BY_BOUNDARY.get(boundary)
    if input_contract is None:
        raise ValueError(f"unsupported benchmark boundary {boundary!r}")
    binding = EXACT_LANE_BINDINGS[lane]
    acceleration_class = str(row.get("acceleration_class", ""))
    acceleration_id = binding["acceleration_by_class"].get(acceleration_class)
    if acceleration_id is None:
        raise ValueError(f"unsupported acceleration class {acceleration_class!r}")
    audits = tuple(entry_point_audits)
    required = required_runtime_operations(row)
    reasons: list[str] = []
    for operation in required:
        matching = tuple(audit for audit in audits if audit.operation == operation)
        if not matching:
            reasons.append(f"missing_entry_point:{operation}")
            continue
        if not any(audit.instrumented for audit in matching):
            reasons.append(f"uninstrumented_entry_point:{operation}")
    for audit in audits:
        if audit.operation not in required:
            continue
        for contamination in audit.contaminations:
            reasons.append(f"timed_contamination:{contamination}:{audit.entry_point}")
    required_asset = row.get("required_asset_manifest")
    if required_asset is not None and asset_manifest_sha256 is None:
        reasons.append(f"asset_manifest_unavailable:{required_asset}")
    reasons = sorted(set(reasons))
    return LaneEligibilityEvidence(
        schema_version=ELIGIBILITY_SCHEMA_VERSION,
        row_id=str(row.get("row_id", "")),
        case_id=str(row.get("case_id", "")),
        boundary=boundary,
        lane=lane,
        input_contract=input_contract,
        renderer_id=str(binding["renderer_id"]),
        acceleration_id=str(acceleration_id),
        required_operations=required,
        entry_point_audits=audits,
        asset_manifest_sha256=asset_manifest_sha256,
        eligible=not reasons,
        reason_codes=tuple(reasons),
    )


def assert_eligible_for_submission(evidence: LaneEligibilityEvidence) -> None:
    if not evidence.eligible:
        reasons = ", ".join(evidence.reason_codes)
        raise LaneIneligibleError(
            f"lane {evidence.lane!r} is ineligible for row {evidence.row_id!r}: {reasons}"
        )


def validate_trace_entry_points(
    evidence: RuntimeOperationTraceEvidence,
    eligibility: LaneEligibilityEvidence,
) -> None:
    """Reject trace records not backed by an instrumented preflight audit."""

    instrumented = {
        (audit.operation, audit.entry_point) for audit in eligibility.entry_point_audits if audit.instrumented
    }
    unaudited = [
        (record.sequence, record.operation, record.entry_point)
        for record in evidence.records
        if (record.operation, record.entry_point) not in instrumented
    ]
    if unaudited:
        raise ValueError(f"operation trace contains unaudited entry points: {unaudited}")


def _sha256_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
