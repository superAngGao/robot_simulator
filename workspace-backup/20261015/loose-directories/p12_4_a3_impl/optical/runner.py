"""Backend-neutral single-lane execution core for Q54 optical benchmarks."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .eligibility import (
    LaneEligibilityEvidence,
    LaneIneligibleError,
    assert_eligible_for_submission,
    evaluate_lane_eligibility,
    required_runtime_operations,
    validate_trace_entry_points,
)
from .native import NativeReadinessEvidence, PreparedNativeInput
from .schema import (
    RESULT_SCHEMA_VERSION,
    ExecutionReceipt,
    canonical_gate_sha256,
    validate_frozen_gate,
    validate_result_record,
)
from .timing import (
    RuntimeOperationTrace,
    RuntimeOperationTraceEvidence,
    active_timed_boundary,
    native_timed_submission,
    validate_required_operation_order,
)

SINGLE_LANE_RUNNER_VERSION = "q54.p12.4.single-lane-runner.v1"


@dataclass(frozen=True)
class LaneSampleRequest:
    """Coordinator-owned identity; it contains no observed execution values."""

    run_id: str
    repeat_index: int
    sample_index: int
    profile_mode: str = "primary_unprofiled"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        for name, value in (
            ("repeat_index", self.repeat_index),
            ("sample_index", self.sample_index),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.profile_mode not in {"primary_unprofiled", "diagnostic_profiled"}:
            raise ValueError("profile_mode is invalid")


@dataclass(frozen=True)
class ResolvedLaneSample:
    """Lane-produced observation resolved after native timed submission."""

    receipt: ExecutionReceipt
    metrics_ms: Mapping[str, float]
    artifact_hashes: Mapping[str, str]
    profile_phases_ms: Mapping[str, float] | None = None


class PendingLaneSample(Protocol):
    """Native completion handle returned without a host wait in timed code."""

    def resolve(self) -> ResolvedLaneSample:
        """Wait/read native completion and build the lane receipt outside timing."""


class SingleLane(Protocol):
    """Minimal lane surface required by the durable runner."""

    lane_id: str

    def preflight_sample(
        self,
        *,
        row: Mapping[str, Any],
        prepared: PreparedNativeInput,
    ) -> LaneEligibilityEvidence:
        """Audit exact calls/assets and fail closed before timed submission."""

    def submit_sample(
        self,
        *,
        row: Mapping[str, Any],
        prepared: PreparedNativeInput,
        request: LaneSampleRequest,
    ) -> PendingLaneSample:
        """Enqueue one sample and return without host/global synchronization."""


@dataclass
class SubmittedLaneSample:
    """Unresolved sample that may be queued with peers before ordered drain."""

    row: Mapping[str, Any]
    lane_id: str
    prepared: PreparedNativeInput
    readiness: NativeReadinessEvidence
    eligibility: LaneEligibilityEvidence
    operation_trace: RuntimeOperationTraceEvidence
    request: LaneSampleRequest
    pending: PendingLaneSample
    _resolved: bool = field(default=False, init=False, repr=False)

    def mark_resolving(self) -> None:
        if self._resolved:
            raise RuntimeError("submitted lane sample has already been resolved")
        self._resolved = True


class SingleLaneSampleRunner:
    """Run and validate one native sample without paired-backend policy."""

    def __init__(self, gate: Mapping[str, Any], *, execution_phase: str) -> None:
        validate_frozen_gate(gate)
        self._gate = copy.deepcopy(dict(gate))
        authorization = self._gate["phase_authorization"].get(execution_phase)
        if not isinstance(authorization, Mapping):
            raise ValueError(f"unsupported benchmark execution phase {execution_phase!r}")
        self._execution_phase = execution_phase
        self._executable_boundaries = frozenset(authorization["executable_boundaries"])
        self._rows = {str(row["row_id"]): row for row in self._gate["mandatory_rows"]}
        self._gate_sha256 = canonical_gate_sha256(self._gate)

    def run_sample(
        self,
        *,
        row_id: str,
        lane: SingleLane,
        prepared: PreparedNativeInput,
        readiness: NativeReadinessEvidence,
        request: LaneSampleRequest,
    ) -> dict[str, object]:
        """Submit and immediately resolve one synchronous/native-only sample."""

        submitted = self.submit_sample(
            row_id=row_id,
            lane=lane,
            prepared=prepared,
            readiness=readiness,
            request=request,
        )
        return self.resolve_sample(submitted)

    def submit_sample(
        self,
        *,
        row_id: str,
        lane: SingleLane,
        prepared: PreparedNativeInput,
        readiness: NativeReadinessEvidence,
        request: LaneSampleRequest,
    ) -> SubmittedLaneSample:
        """Enqueue one sample and return its unresolved native completion.

        Ordered-async campaigns may submit several samples before calling
        :meth:`resolve_sample` in order and including their final drain/reclaim
        in the enclosing batch metric.
        """

        row = self._rows.get(row_id)
        if row is None:
            raise KeyError(f"unknown frozen benchmark row {row_id!r}")
        if row["boundary"] not in self._executable_boundaries:
            raise LaneIneligibleError(
                f"boundary {row['boundary']!r} is not executable during {self._execution_phase}"
            )
        if lane.lane_id not in {"warp", "cuda"}:
            raise ValueError("single-lane runner requires lane_id 'warp' or 'cuda'")
        if prepared.backend != lane.lane_id:
            raise ValueError("prepared native input backend does not match the lane")
        self._validate_readiness(prepared=prepared, readiness=readiness)
        expected_case_hash = self._gate["cases"][row["case_id"]]["sha256"]
        if prepared.manifest.case_id != row["case_id"]:
            raise ValueError("prepared native input case does not match the row")
        if prepared.manifest.case_sha256 != expected_case_hash:
            raise ValueError("prepared native input case hash does not match the frozen gate")

        submitted_row = copy.deepcopy(row)
        eligibility = lane.preflight_sample(row=submitted_row, prepared=prepared)
        if not isinstance(eligibility, LaneEligibilityEvidence):
            raise TypeError("lane preflight_sample must return LaneEligibilityEvidence")
        self._validate_eligibility(
            row=submitted_row,
            lane_id=lane.lane_id,
            eligibility=eligibility,
        )
        assert_eligible_for_submission(eligibility)
        operation_trace = RuntimeOperationTrace(
            row_id=row_id,
            lane=lane.lane_id,
            boundary=str(row["boundary"]),
        )
        with native_timed_submission(
            row_id=row_id,
            lane=lane.lane_id,
            boundary=str(row["boundary"]),
            operation_trace=operation_trace,
        ):
            pending = lane.submit_sample(row=submitted_row, prepared=prepared, request=request)
        if active_timed_boundary() is not None:
            raise AssertionError("native timed context leaked after lane submission")
        if not callable(getattr(pending, "resolve", None)):
            raise TypeError("lane submit_sample must return a pending native sample")
        trace_evidence = operation_trace.evidence()
        try:
            validate_required_operation_order(trace_evidence, eligibility.required_operations)
            validate_trace_entry_points(trace_evidence, eligibility)
        except ValueError as exc:
            raise LaneIneligibleError(str(exc)) from exc
        return SubmittedLaneSample(
            row=submitted_row,
            lane_id=lane.lane_id,
            prepared=prepared,
            readiness=readiness,
            eligibility=eligibility,
            operation_trace=trace_evidence,
            request=request,
            pending=pending,
        )

    def resolve_sample(self, submitted: SubmittedLaneSample) -> dict[str, object]:
        """Resolve a previously submitted sample exactly once outside timing."""

        if active_timed_boundary() is not None:
            raise RuntimeError("native sample resolution is forbidden inside timed submission")
        submitted.mark_resolving()
        sample = submitted.pending.resolve()
        self._validate_lane_provenance(
            row=submitted.row,
            prepared=submitted.prepared,
            readiness=submitted.readiness,
            eligibility=submitted.eligibility,
            operation_trace=submitted.operation_trace,
            sample=sample,
        )
        receipt_payload = sample.receipt.to_payload()
        record: dict[str, object] = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "gate_sha256": self._gate_sha256,
            "run_id": submitted.request.run_id,
            "repeat_index": submitted.request.repeat_index,
            "sample_index": submitted.request.sample_index,
            "profile_mode": submitted.request.profile_mode,
            "metrics_ms": dict(sample.metrics_ms),
            "artifact_hashes": dict(sample.artifact_hashes),
            "receipt": receipt_payload,
            "receipt_sha256": sample.receipt.sha256,
        }
        if sample.profile_phases_ms is not None:
            record["profile_phases_ms"] = dict(sample.profile_phases_ms)
        validate_result_record(record, self._gate)
        return record

    @staticmethod
    def _validate_lane_provenance(
        *,
        row: Mapping[str, Any],
        prepared: PreparedNativeInput,
        readiness: NativeReadinessEvidence,
        eligibility: LaneEligibilityEvidence,
        operation_trace: RuntimeOperationTraceEvidence,
        sample: ResolvedLaneSample,
    ) -> None:
        receipt = sample.receipt
        if receipt.lane != prepared.backend:
            raise ValueError("lane receipt backend does not match prepared native input")
        if receipt.native_input_manifest_sha256 != prepared.manifest.sha256:
            raise ValueError("lane receipt does not identify the prepared input manifest")
        if receipt.native_input_content_sha256 != prepared.manifest.content_sha256:
            raise ValueError("lane receipt does not identify the prepared input content")
        if receipt.topology_sha256 != prepared.manifest.topology_sha256:
            raise ValueError("lane receipt does not identify the prepared topology")
        if receipt.native_readiness != readiness:
            raise ValueError("lane receipt native readiness does not match certification")
        if receipt.native_readiness_sha256 != readiness.sha256:
            raise ValueError("lane receipt does not identify the certified native readiness")
        if receipt.row_id != row["row_id"]:
            raise ValueError("lane receipt row does not match the submitted row")
        if receipt.eligibility != eligibility or receipt.eligibility_sha256 != eligibility.sha256:
            raise ValueError("lane receipt eligibility does not match preflight evidence")
        if (
            receipt.operation_trace != operation_trace
            or receipt.operation_trace_sha256 != operation_trace.sha256
        ):
            raise ValueError("lane receipt operation trace does not match timed submission")

    def _validate_eligibility(
        self,
        *,
        row: Mapping[str, Any],
        lane_id: str,
        eligibility: LaneEligibilityEvidence,
    ) -> None:
        binding = self._gate["lane_bindings"][lane_id]
        expected = {
            "row_id": row["row_id"],
            "case_id": row["case_id"],
            "boundary": row["boundary"],
            "lane": lane_id,
            "input_contract": row["input_contract"],
            "renderer_id": binding["renderer_id"],
            "acceleration_id": binding["acceleration_by_class"][row["acceleration_class"]],
            "required_operations": required_runtime_operations(row),
        }
        for name, value in expected.items():
            if getattr(eligibility, name) != value:
                raise ValueError(f"lane eligibility {name} does not match the frozen row")
        recalculated = evaluate_lane_eligibility(
            row=row,
            lane=lane_id,
            entry_point_audits=eligibility.entry_point_audits,
            asset_manifest_sha256=eligibility.asset_manifest_sha256,
        )
        if eligibility != recalculated:
            raise ValueError("lane eligibility is not the result of the frozen preflight evaluation")
        required_asset = row.get("required_asset_manifest")
        if required_asset is not None and eligibility.asset_manifest_sha256 is None:
            raise LaneIneligibleError("lane eligibility lacks mandatory exact asset evidence")
        if required_asset is None and eligibility.asset_manifest_sha256 is not None:
            raise ValueError("lane eligibility carries unexpected asset evidence")

    def _validate_readiness(
        self,
        *,
        prepared: PreparedNativeInput,
        readiness: NativeReadinessEvidence,
    ) -> None:
        if readiness.schema_version != self._gate["native_input_contract"]["schema_version"]:
            raise ValueError("native readiness schema does not match the frozen gate")
        expected = {
            "backend": prepared.backend,
            "device_ordinal": prepared.device_ordinal,
            "manifest_sha256": prepared.manifest.sha256,
            "content_sha256": prepared.manifest.content_sha256,
            "topology_sha256": prepared.manifest.topology_sha256,
            "producer_ready_event_complete": True,
            "direct_validation_complete": True,
            "allocation_complete": True,
            "compile_complete": True,
        }
        for name, value in expected.items():
            if getattr(readiness, name) != value:
                raise ValueError(f"native readiness {name} does not match prepared input")
        required_warmups = int(self._gate["measurement"]["warmup_frames"])
        if readiness.warmup_frames_completed < required_warmups:
            raise ValueError(
                "native readiness does not prove the frozen warmup frame count "
                f"({readiness.warmup_frames_completed} < {required_warmups})"
            )
