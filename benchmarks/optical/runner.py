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
from .instrumentation import instrument_runtime_entry_points
from .native import (
    NativeReadinessEvidence,
    PreparedNativeInput,
    prepared_storage_signatures,
    prepared_transform_frame_signatures,
)
from .receipts import C4_RESULT_BINDING_CAPABILITY, NativeExecutionBindingEvidence
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
        binding: NativeExecutionBindingEvidence,
        request: LaneSampleRequest,
    ) -> PendingLaneSample:
        """Enqueue one sample and return without host/global synchronization."""

    def bind_sample(
        self,
        *,
        row: Mapping[str, Any],
        prepared: PreparedNativeInput,
        readiness: NativeReadinessEvidence,
    ) -> NativeExecutionBindingEvidence:
        """Freeze exact stage/storage provenance before timed submission."""


@dataclass
class SubmittedLaneSample:
    """Unresolved sample that may be queued with peers before ordered drain."""

    row: Mapping[str, Any]
    lane_id: str
    prepared: PreparedNativeInput
    readiness: NativeReadinessEvidence
    eligibility: LaneEligibilityEvidence
    operation_trace: RuntimeOperationTraceEvidence
    native_execution_binding: NativeExecutionBindingEvidence
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
        activation = self._gate["native_result_activation"]
        if (
            execution_phase in activation["blocked_result_phases"]
            and activation["result_capable"] is not True
        ):
            raise LaneIneligibleError(
                f"{execution_phase} result submission is closed until versioned C4/B3 approval"
            )

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
        self._validate_prepared_input_for_row(row=row, prepared=prepared)
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
        bind_sample = getattr(lane, "bind_sample", None)
        if not callable(bind_sample):
            raise LaneIneligibleError("lane does not expose pre-enqueue native binding")
        binding = bind_sample(
            row=submitted_row,
            prepared=prepared,
            readiness=readiness,
        )
        if not isinstance(binding, NativeExecutionBindingEvidence):
            raise TypeError("lane bind_sample must return NativeExecutionBindingEvidence")
        try:
            binding = NativeExecutionBindingEvidence.from_payload(binding.to_payload())
        except (AttributeError, TypeError, ValueError) as exc:
            raise LaneIneligibleError(
                f"native execution binding failed canonical pre-enqueue validation: {exc}"
            ) from exc
        if binding.submission_capability != C4_RESULT_BINDING_CAPABILITY:
            raise LaneIneligibleError(
                "C3 one-shot native binding is not result-capable; a reviewed C4 binding is required"
            )
        self._validate_native_binding_pre_enqueue(
            row=submitted_row,
            prepared=prepared,
            readiness=readiness,
            binding=binding,
        )
        operation_trace = RuntimeOperationTrace(
            row_id=row_id,
            lane=lane.lane_id,
            boundary=str(row["boundary"]),
        )
        with instrument_runtime_entry_points(eligibility.entry_point_audits):
            with native_timed_submission(
                row_id=row_id,
                lane=lane.lane_id,
                boundary=str(row["boundary"]),
                operation_trace=operation_trace,
            ):
                pending = lane.submit_sample(
                    row=submitted_row,
                    prepared=prepared,
                    binding=binding,
                    request=request,
                )
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
            native_execution_binding=binding,
            request=request,
            pending=pending,
        )

    def _validate_prepared_input_for_row(
        self,
        *,
        row: Mapping[str, Any],
        prepared: PreparedNativeInput,
    ) -> None:
        """Bind the frozen row to exact input content before lane preflight."""

        manifest = prepared.manifest
        if not manifest.result_capable:
            raise LaneIneligibleError("generic native input cannot enter a result-capable lane")
        if manifest.input_contract != row["input_contract"]:
            raise LaneIneligibleError(
                "prepared native input contract does not match the frozen row"
            )
        case_contract = self._gate["cases"][row["case_id"]]
        if manifest.scene_contract_sha256 != case_contract["scene_contract_sha256"]:
            raise LaneIneligibleError(
                "prepared scene content does not match the frozen row workload"
            )
        boundary = row["boundary"]
        if boundary == "R0":
            valid_source = (
                manifest.transform_replay_sha256 is None
                and manifest.transform_frame_count == 0
                and manifest.fixed_render_fixture_sha256
                == case_contract["fixed_render_fixture_sha256"]
                and manifest.fixed_render_fixture_sha256 is not None
            )
        elif boundary in {"R1", "R2"}:
            valid_source = (
                manifest.transform_replay_sha256
                == case_contract["transform_replay_sha256"]
                and manifest.transform_replay_sha256 is not None
                and manifest.transform_frame_count > 0
                and manifest.fixed_render_fixture_sha256 is None
            )
        elif boundary == "S":
            valid_source = (
                manifest.transform_replay_sha256 is None
                and manifest.transform_frame_count == 0
                and manifest.fixed_render_fixture_sha256 is None
            )
        else:  # protected by the frozen gate validator
            valid_source = False
        if not valid_source:
            raise LaneIneligibleError(
                "prepared input provenance is incompatible with the frozen row boundary"
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
            native_execution_binding=submitted.native_execution_binding,
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
        native_execution_binding: NativeExecutionBindingEvidence,
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
        if (
            receipt.native_execution_binding != native_execution_binding
            or receipt.native_execution_binding_sha256 != native_execution_binding.sha256
        ):
            raise ValueError("lane receipt native binding does not match pre-enqueue evidence")
        if receipt.row_id != row["row_id"]:
            raise ValueError("lane receipt row does not match the submitted row")
        if receipt.eligibility != eligibility or receipt.eligibility_sha256 != eligibility.sha256:
            raise ValueError("lane receipt eligibility does not match preflight evidence")
        if (
            receipt.operation_trace != operation_trace
            or receipt.operation_trace_sha256 != operation_trace.sha256
        ):
            raise ValueError("lane receipt operation trace does not match timed submission")

    @staticmethod
    def _validate_native_binding_pre_enqueue(
        *,
        row: Mapping[str, Any],
        prepared: PreparedNativeInput,
        readiness: NativeReadinessEvidence,
        binding: NativeExecutionBindingEvidence,
    ) -> None:
        manifest = prepared.manifest
        if not manifest.result_capable:
            raise LaneIneligibleError("generic native input cannot enter a result-capable lane")
        expected = {
            "row_id": row["row_id"],
            "case_id": manifest.case_id,
            "case_sha256": manifest.case_sha256,
            "boundary": row["boundary"],
            "lane": prepared.backend,
            "input_contract": manifest.input_contract,
            "native_input_manifest_sha256": manifest.sha256,
            "native_readiness_sha256": readiness.sha256,
            "scene_contract_sha256": manifest.scene_contract_sha256,
            "transform_replay_sha256": manifest.transform_replay_sha256,
            "fixed_render_fixture_sha256": manifest.fixed_render_fixture_sha256,
        }
        for name, value in expected.items():
            if getattr(binding, name) != value:
                raise LaneIneligibleError(f"native execution binding {name} is not prepared-input-owned")
        if binding.transform_frame_index is not None and (
            binding.transform_frame_index >= manifest.transform_frame_count
        ):
            raise LaneIneligibleError("native execution binding replay frame index is out of range")
        expected_scene_storage = prepared_storage_signatures(
            prepared,
            manifest.scene_array_names,
        )
        expected_fixture_storage = prepared_storage_signatures(
            prepared,
            manifest.fixed_render_fixture_arrays,
        )
        if binding.runtime_scene_storage != expected_scene_storage:
            raise LaneIneligibleError("native binding substituted prepared scene storage")
        if binding.derived_fixture_storage != expected_fixture_storage:
            raise LaneIneligibleError("native binding substituted prepared fixture storage")
        stages = {stage.stage_name: stage.evidence for stage in binding.ordered_stage_evidence}
        if binding.boundary == "R0":
            if stages["render"].input_storage != expected_scene_storage + expected_fixture_storage:
                raise LaneIneligibleError("R0 render stage substituted prepared input storage")
        elif binding.boundary in {"R1", "R2"}:
            assert binding.transform_frame_index is not None
            expected_transform_storage = prepared_transform_frame_signatures(
                prepared,
                binding.transform_frame_index,
            )
            if stages["scene_update"].input_storage != (
                expected_scene_storage + expected_transform_storage
            ):
                raise LaneIneligibleError(
                    "scene-update stage substituted prepared scene or replay-frame storage"
                )

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
            "input_contract": prepared.manifest.input_contract,
            "scene_contract_schema_version": prepared.manifest.scene_contract_schema_version,
            "scene_contract_sha256": prepared.manifest.scene_contract_sha256,
            "scene_id": prepared.manifest.scene_id,
            "transform_replay_sha256": prepared.manifest.transform_replay_sha256,
            "transform_frame_count": prepared.manifest.transform_frame_count,
            "fixed_render_fixture_sha256": prepared.manifest.fixed_render_fixture_sha256,
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
