from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field, replace

import numpy as np
import pytest

import benchmarks.optical.schema as optical_schema
from benchmarks.optical.artifacts import BenchmarkArtifactWriter
from benchmarks.optical.eligibility import (
    LaneIneligibleError,
    evaluate_lane_eligibility,
    verified_runtime_entry_point_audit,
)
from benchmarks.optical.instrumentation import entry_point_hook_id
from benchmarks.optical.native import (
    CanonicalNativeInput,
    NativeInputValidationReport,
    NativeReadinessEvidence,
    WarpPreparedNativeInput,
    certify_native_readiness,
    prepared_storage_signatures,
)
from benchmarks.optical.receipts import (
    ACCELERATION_RECEIPT_SCHEMA_VERSION,
    NATIVE_STAGE_EVIDENCE_SCHEMA_VERSION,
    AccelerationReceipt,
    NativeStageEvidence,
    NativeStorageSignature,
    build_c3_one_shot_native_execution_binding,
)
from benchmarks.optical.runner import (
    LaneSampleRequest,
    ResolvedLaneSample,
    SingleLaneSampleRunner,
)
from benchmarks.optical.schema import (
    ExecutionReceipt,
    canonical_input_provenance_sha256,
    canonical_row_semantics_sha256,
    load_frozen_gate,
)
from benchmarks.optical.timing import (
    OPERATION_TRACE_SCHEMA_VERSION,
    RuntimeOperationRecord,
    RuntimeOperationTraceEvidence,
    TimedBoundaryViolation,
    active_timed_boundary,
    assert_operation_allowed,
    native_timed_submission,
)
from benchmarks.optical.workloads import (
    build_canonical_fixed_fixture,
    build_canonical_packed_scene,
)


def test_canonical_native_input_is_immutable_and_order_independent():
    first = _native_source(reverse=False)
    second = _native_source(reverse=True)

    assert first.manifest.sha256 == second.manifest.sha256
    assert first.manifest.content_sha256 == second.manifest.content_sha256
    assert first.manifest.topology_sha256 == second.manifest.topology_sha256
    assert [item.name for item in first.manifest.arrays] == [
        "bvh_child_l",
        "triangle_v0",
    ]
    with pytest.raises(ValueError, match="read-only"):
        first.arrays["bvh_child_l"][0] = 99


def test_canonical_native_input_rejects_incomplete_semantics_and_topology():
    with pytest.raises(ValueError, match="every and only"):
        CanonicalNativeInput.from_arrays(
            case_id="asymmetric_synthetic_7x3_v1",
            materializer_id="unit_fixture_v1",
            arrays={"a": np.zeros(1, dtype=np.int32)},
            array_semantics={},
            topology_arrays=("a",),
            semantics={"coordinate_system": "world"},
        )

    with pytest.raises(ValueError, match="topology arrays are missing"):
        CanonicalNativeInput.from_arrays(
            case_id="asymmetric_synthetic_7x3_v1",
            materializer_id="unit_fixture_v1",
            arrays={"a": np.zeros(1, dtype=np.int32)},
            array_semantics={"a": "topology"},
            topology_arrays=("missing",),
            semantics={"coordinate_system": "world"},
        )


def test_timed_boundary_rejects_contaminating_operations_and_restores_context():
    assert active_timed_boundary() is None
    with native_timed_submission(row_id="row", lane="cuda", boundary="R0"):
        assert active_timed_boundary() is not None
        with pytest.raises(TimedBoundaryViolation, match="host_upload is forbidden"):
            assert_operation_allowed("host_upload")
        with pytest.raises(TimedBoundaryViolation, match="nested"):
            with native_timed_submission(row_id="nested", lane="cuda", boundary="R0"):
                pass
    assert active_timed_boundary() is None


def test_timed_boundary_restores_context_after_lane_failure():
    with pytest.raises(RuntimeError, match="submit failed"):
        with native_timed_submission(row_id="row", lane="warp", boundary="R1"):
            raise RuntimeError("submit failed")
    assert active_timed_boundary() is None


def test_single_lane_runner_refuses_c3_before_timing_even_if_activation_is_flipped(monkeypatch):
    gate = _activated_gate(monkeypatch)
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r0_asymmetric_full_shadow")
    source = _runner_native_source()
    prepared = _prepared(source)
    trace: list[tuple[str, bool]] = []
    lane = _FakeLane(gate=gate, row=row, prepared=prepared, trace=trace)
    runner = SingleLaneSampleRunner(gate, execution_phase="P12.4a3b")

    with pytest.raises(LaneIneligibleError, match="C3 one-shot"):
        runner.run_sample(
            row_id=row["row_id"],
            lane=lane,
            prepared=prepared,
            readiness=_readiness(prepared),
            request=LaneSampleRequest(run_id="unit-run", repeat_index=1, sample_index=2),
        )

    assert trace == []


def test_c3_rejection_prevents_lane_from_fabricating_result_provenance(monkeypatch):
    gate = _activated_gate(monkeypatch)
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r0_asymmetric_full_shadow")
    source = _runner_native_source()
    prepared = _prepared(source)
    lane = _FakeLane(gate=gate, row=row, prepared=prepared, trace=[], bad_manifest=True)

    with pytest.raises(LaneIneligibleError, match="C3 one-shot"):
        SingleLaneSampleRunner(gate, execution_phase="P12.4a3b").run_sample(
            row_id=row["row_id"],
            lane=lane,
            prepared=prepared,
            readiness=_readiness(prepared),
            request=LaneSampleRequest(run_id="unit-run", repeat_index=0, sample_index=0),
        )


def test_c3_rejection_prevents_ordered_submit_or_drain(
    monkeypatch,
):
    gate = _activated_gate(monkeypatch)
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r0_asymmetric_full_shadow")
    source = _runner_native_source()
    prepared = _prepared(source)
    readiness = _readiness(prepared)
    trace: list[tuple[str, bool]] = []
    lane = _FakeLane(gate=gate, row=row, prepared=prepared, trace=trace)
    runner = SingleLaneSampleRunner(gate, execution_phase="P12.4a3b")

    with pytest.raises(LaneIneligibleError, match="C3 one-shot"):
        runner.submit_sample(
            row_id=row["row_id"],
            lane=lane,
            prepared=prepared,
            readiness=readiness,
            request=LaneSampleRequest(run_id="ordered", repeat_index=0, sample_index=0),
        )
    assert trace == []


def test_single_lane_runner_rejects_incomplete_warmup_evidence(monkeypatch):
    gate = _activated_gate(monkeypatch)
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r0_asymmetric_full_shadow")
    source = _runner_native_source()
    prepared = _prepared(source)
    validation = NativeInputValidationReport(
        backend="warp",
        manifest_sha256=prepared.manifest.sha256,
        content_sha256=prepared.manifest.content_sha256,
        topology_sha256=prepared.manifest.topology_sha256,
        validated_arrays=tuple(item.name for item in prepared.manifest.arrays),
        input_contract=prepared.manifest.input_contract,
        scene_contract_sha256=prepared.manifest.scene_contract_sha256,
        transform_replay_sha256=prepared.manifest.transform_replay_sha256,
        transform_frame_count=prepared.manifest.transform_frame_count,
        fixed_render_fixture_sha256=prepared.manifest.fixed_render_fixture_sha256,
    )
    incomplete_validation = NativeInputValidationReport(
        backend="warp",
        manifest_sha256=prepared.manifest.sha256,
        content_sha256=prepared.manifest.content_sha256,
        topology_sha256=prepared.manifest.topology_sha256,
        validated_arrays=(),
        input_contract=prepared.manifest.input_contract,
        scene_contract_sha256=prepared.manifest.scene_contract_sha256,
        transform_replay_sha256=prepared.manifest.transform_replay_sha256,
        transform_frame_count=prepared.manifest.transform_frame_count,
        fixed_render_fixture_sha256=prepared.manifest.fixed_render_fixture_sha256,
    )
    with pytest.raises(ValueError, match="every prepared input array"):
        certify_native_readiness(
            prepared,
            incomplete_validation,
            compile_complete=True,
            warmup_frames_completed=10,
        )
    readiness = certify_native_readiness(
        prepared,
        validation,
        compile_complete=True,
        warmup_frames_completed=9,
    )

    with pytest.raises(ValueError, match="warmup frame count"):
        SingleLaneSampleRunner(gate, execution_phase="P12.4a3b").run_sample(
            row_id=row["row_id"],
            lane=_FakeLane(gate=gate, row=row, prepared=prepared, trace=[]),
            prepared=prepared,
            readiness=readiness,
            request=LaneSampleRequest(run_id="unit-run", repeat_index=0, sample_index=0),
        )


def test_c3_rejection_prevents_lane_request_receipt_fabrication(monkeypatch):
    gate = _activated_gate(monkeypatch)
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r0_asymmetric_full_shadow")
    source = _runner_native_source()
    prepared = _prepared(source)
    lane = _FakeLane(gate=gate, row=row, prepared=prepared, trace=[], wrong_shadows=True)

    with pytest.raises(LaneIneligibleError, match="C3 one-shot"):
        SingleLaneSampleRunner(gate, execution_phase="P12.4a3b").run_sample(
            row_id=row["row_id"],
            lane=lane,
            prepared=prepared,
            readiness=_readiness(prepared),
            request=LaneSampleRequest(run_id="unit-run", repeat_index=0, sample_index=0),
        )


def test_artifact_writer_is_canonical_atomic_and_root_scoped(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = BenchmarkArtifactWriter(first_root)
    second = BenchmarkArtifactWriter(second_root)

    digest = first.write_jsonl("raw/results.jsonl", [{"b": 2, "a": 1}, {"value": "x"}])
    mirror = second.write_jsonl("raw/results.jsonl", [{"a": 1, "b": 2}, {"value": "x"}])
    assert digest.sha256 == mirror.sha256
    assert digest.record_count == 2
    assert digest.sha256 == hashlib.sha256((first_root / digest.relative_path).read_bytes()).hexdigest()
    manifest = first.write_manifest()
    manifest_payload = json.loads((first_root / manifest.relative_path).read_text(encoding="utf-8"))
    assert manifest_payload["artifacts"] == [digest.to_payload()]

    with pytest.raises(ValueError, match="already been written"):
        first.write_jsonl("raw/results.jsonl", [])
    with pytest.raises(ValueError, match="escapes"):
        first.write_json("../outside.json", {})
    with pytest.raises(ValueError, match="Out of range float"):
        first.write_json("nan.json", {"bad": float("nan")})


def _native_source(*, reverse: bool = False) -> CanonicalNativeInput:
    arrays = {
        "bvh_child_l": np.array([1, -1, -1], dtype=np.int32),
        "triangle_v0": np.array(
            [[-1.0, -0.5, 2.0], [0.25, -1.0, 3.0]],
            dtype=np.float32,
        ),
    }
    if reverse:
        arrays = dict(reversed(tuple(arrays.items())))
    return CanonicalNativeInput.from_arrays(
        case_id="asymmetric_synthetic_7x3_v1",
        materializer_id="unit_asymmetric_native_v1",
        arrays=arrays,
        array_semantics={
            "bvh_child_l": "fixed BVH left-child topology",
            "triangle_v0": "world-space triangle vertex zero",
        },
        topology_arrays=("bvh_child_l",),
        semantics={
            "coordinate_system": "world",
            "primitive_order": "canonical_source_order",
        },
    )


def _runner_native_source() -> CanonicalNativeInput:
    packed = build_canonical_packed_scene("asymmetric_synthetic_7x3_v1")
    fixture = build_canonical_fixed_fixture(packed)
    return CanonicalNativeInput.from_scene_contract(
        packed,
        input_contract="fixed_native_scene_v1",
        fixed_fixture=fixture,
    )


def _activated_gate(monkeypatch):
    gate = load_frozen_gate()
    activation = {
        **gate["native_result_activation"],
        "result_capable": True,
        "activation_commit": "f" * 40,
    }
    monkeypatch.setattr(optical_schema, "EXPECTED_NATIVE_RESULT_ACTIVATION", activation)
    gate["native_result_activation"] = activation
    return gate


def test_single_lane_runner_rejects_same_shape_substitute_storage_before_enqueue(monkeypatch):
    gate = _activated_gate(monkeypatch)
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r0_asymmetric_full_shadow")
    source = _runner_native_source()
    prepared = _prepared(source)
    trace: list[tuple[str, bool]] = []
    lane = _FakeLane(
        gate=gate,
        row=row,
        prepared=prepared,
        trace=trace,
        substitute_scene_storage=True,
    )

    with pytest.raises(LaneIneligibleError, match="substituted prepared scene storage"):
        SingleLaneSampleRunner(gate, execution_phase="P12.4a3b").run_sample(
            row_id=row["row_id"],
            lane=lane,
            prepared=prepared,
            readiness=_readiness(prepared),
            request=LaneSampleRequest(run_id="unit-run", repeat_index=0, sample_index=0),
        )
    assert trace == []


def test_runner_rejects_input_contract_mismatch_before_preflight_or_enqueue(monkeypatch):
    gate = _activated_gate(monkeypatch)
    row = next(row for row in gate["mandatory_rows"] if row["boundary"] == "R1")
    prepared = _prepared(_runner_native_source())
    trace: list[tuple[str, bool]] = []
    lane = _FakeLane(gate=gate, row=row, prepared=prepared, trace=trace)

    with pytest.raises(LaneIneligibleError, match="input contract"):
        SingleLaneSampleRunner(gate, execution_phase="P12.4a3b").submit_sample(
            row_id=row["row_id"],
            lane=lane,
            prepared=prepared,
            readiness=_readiness(prepared),
            request=LaneSampleRequest(run_id="wrong-contract", repeat_index=0, sample_index=0),
        )

    assert lane.preflight_called is False
    assert lane.bind_called is False
    assert trace == []


def test_runner_canonicalizes_binding_before_timed_enqueue(monkeypatch):
    gate = _activated_gate(monkeypatch)
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r0_asymmetric_full_shadow")
    prepared = _prepared(_runner_native_source())
    trace: list[tuple[str, bool]] = []
    lane = _FakeLane(
        gate=gate,
        row=row,
        prepared=prepared,
        trace=trace,
        invalid_direct_binding=True,
    )

    with pytest.raises(LaneIneligibleError, match="canonical pre-enqueue validation"):
        SingleLaneSampleRunner(gate, execution_phase="P12.4a3b").submit_sample(
            row_id=row["row_id"],
            lane=lane,
            prepared=prepared,
            readiness=_readiness(prepared),
            request=LaneSampleRequest(run_id="invalid-binding", repeat_index=0, sample_index=0),
        )

    assert lane.preflight_called is True
    assert lane.bind_called is True
    assert trace == []


def test_runner_rejects_c3_one_shot_binding_before_timed_enqueue(monkeypatch):
    gate = _activated_gate(monkeypatch)
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r0_asymmetric_full_shadow")
    prepared = _prepared(_runner_native_source())
    trace: list[tuple[str, bool]] = []
    lane = _FakeLane(
        gate=gate,
        row=row,
        prepared=prepared,
        trace=trace,
    )

    with pytest.raises(LaneIneligibleError, match="C3 one-shot"):
        SingleLaneSampleRunner(gate, execution_phase="P12.4a3b").submit_sample(
            row_id=row["row_id"],
            lane=lane,
            prepared=prepared,
            readiness=_readiness(prepared),
            request=LaneSampleRequest(run_id="one-shot", repeat_index=0, sample_index=0),
        )

    assert trace == []


@dataclass(frozen=True)
class _FakeWarpArray:
    ptr: int


def _prepared(source: CanonicalNativeInput) -> WarpPreparedNativeInput:
    arrays = {
        manifest.name: _FakeWarpArray(0 if manifest.nbytes == 0 else 0x1000 + index * 0x100)
        for index, manifest in enumerate(source.manifest.arrays)
    }
    return WarpPreparedNativeInput(
        manifest=source.manifest,
        device_ordinal=0,
        arrays=arrays,
        ready_event=object(),
        source=source,
    )


def _readiness(prepared: WarpPreparedNativeInput):
    validation = NativeInputValidationReport(
        backend="warp",
        manifest_sha256=prepared.manifest.sha256,
        content_sha256=prepared.manifest.content_sha256,
        topology_sha256=prepared.manifest.topology_sha256,
        validated_arrays=tuple(item.name for item in prepared.manifest.arrays),
        input_contract=prepared.manifest.input_contract,
        scene_contract_sha256=prepared.manifest.scene_contract_sha256,
        transform_replay_sha256=prepared.manifest.transform_replay_sha256,
        transform_frame_count=prepared.manifest.transform_frame_count,
        fixed_render_fixture_sha256=prepared.manifest.fixed_render_fixture_sha256,
    )
    return certify_native_readiness(
        prepared,
        validation,
        compile_complete=True,
        warmup_frames_completed=10,
    )


@dataclass
class _FakePending:
    sample: ResolvedLaneSample
    trace: list[tuple[str, bool]]

    def resolve(self) -> ResolvedLaneSample:
        self.trace.append(("resolve", active_timed_boundary() is not None))
        return self.sample


@dataclass
class _FakeLane:
    gate: dict[str, object]
    row: dict[str, object]
    prepared: WarpPreparedNativeInput
    trace: list[tuple[str, bool]]
    bad_manifest: bool = False
    wrong_shadows: bool = False
    substitute_scene_storage: bool = False
    invalid_direct_binding: bool = False
    lane_id: str = "warp"
    preflight_called: bool = field(default=False, init=False)
    bind_called: bool = field(default=False, init=False)
    readiness: NativeReadinessEvidence = field(init=False)
    readiness_sha256: str = field(init=False)

    def __post_init__(self):
        self.readiness = _readiness(self.prepared)
        self.readiness_sha256 = self.readiness.sha256

    def bind_sample(self, *, row, prepared, readiness):
        self.bind_called = True
        scene_storage = prepared_storage_signatures(
            prepared, prepared.manifest.scene_array_names
        )
        if self.substitute_scene_storage:
            scene_storage = (
                replace(scene_storage[0], storage_token="warp:0xdeadbeef"),
                *scene_storage[1:],
            )
        fixture_storage = prepared_storage_signatures(
            prepared, prepared.manifest.fixed_render_fixture_arrays
        )
        render_output = NativeStorageSignature(
            name="render_output",
            backend="warp",
            device_ordinal=0,
            storage_token="warp:0x9000",
            dtype="<f4",
            shape=(21, 3),
            nbytes=252,
            active_start=0,
            active_count=252,
            owner_token="unit-output-slot",
        )
        render = NativeStageEvidence(
            schema_version=NATIVE_STAGE_EVIDENCE_SCHEMA_VERSION,
            stage_name="render",
            operation_id="warp_bvh_direct_light",
            output_profile=row["output_profile"],
            shadows=row["shadows"],
            scene_contract_sha256=prepared.manifest.scene_contract_sha256,
            input_contract=prepared.manifest.input_contract,
            transform_replay_sha256=None,
            transform_frame_index=None,
            fixed_render_fixture_sha256=prepared.manifest.fixed_render_fixture_sha256,
            topology_sha256=prepared.manifest.topology_sha256,
            input_storage=scene_storage + fixture_storage,
            output_storage=(render_output,),
            dispatch_plan_sha256="d" * 64,
        )
        binding = build_c3_one_shot_native_execution_binding(
            row_id=row["row_id"],
            case_id=row["case_id"],
            case_sha256=prepared.manifest.case_sha256,
            boundary=row["boundary"],
            lane="warp",
            renderer_id="warp_bvh_direct_light",
            acceleration_class=row["acceleration_class"],
            acceleration_id="fixed_bvh_fixture",
            output_profile=row["output_profile"],
            shadows=row["shadows"],
            input_contract=prepared.manifest.input_contract,
            native_input_manifest_sha256=prepared.manifest.sha256,
            native_readiness_sha256=readiness.sha256,
            scene_contract_sha256=prepared.manifest.scene_contract_sha256,
            transform_replay_sha256=None,
            transform_frame_index=None,
            fixed_render_fixture_sha256=prepared.manifest.fixed_render_fixture_sha256,
            runtime_scene_storage=scene_storage,
            derived_fixture_storage=fixture_storage,
            stage_evidence=(render,),
        )
        if self.invalid_direct_binding:
            return replace(binding, dataflow_binding_sha256="0" * 64)
        return binding

    def submit_sample(self, *, row, prepared, binding, request):
        del request
        self.trace.append(("submit", active_timed_boundary() is not None))
        for operation in _required_operations(row):
            _invoke_test_entry_point(operation)
        operation_trace = RuntimeOperationTraceEvidence(
            schema_version=OPERATION_TRACE_SCHEMA_VERSION,
            row_id=row["row_id"],
            lane=self.lane_id,
            boundary=row["boundary"],
            records=tuple(
                RuntimeOperationRecord(
                    sequence=index,
                    operation=operation,
                    entry_point=_TEST_ENTRY_POINTS[operation],
                    capture_source="runtime_hook",
                    runtime_hook_id=entry_point_hook_id(_TEST_ENTRY_POINTS[operation]),
                )
                for index, operation in enumerate(_required_operations(row))
            ),
        )
        native_execution_binding = binding
        lane_binding = self.gate["lane_bindings"][self.lane_id]
        acceleration_id = lane_binding["acceleration_by_class"][row["acceleration_class"]]
        contract = self.gate["acceleration_contracts"]["by_lane_and_identity"][self.lane_id][acceleration_id]
        classification = contract["operation_classification"]
        scene_count, refit_count, rebuild_count = {
            "none": (0, 0, 0),
            "refit": (1, 1, 0),
            "rebuild": (1, 0, 1),
        }[classification]
        eligibility = self.preflight_sample(row=row, prepared=prepared)
        acceleration = AccelerationReceipt(
            acceleration_schema_version=ACCELERATION_RECEIPT_SCHEMA_VERSION,
            scene_contract_sha256=prepared.manifest.scene_contract_sha256,
            acceleration_stage_evidence_sha256=None,
            acceleration_id=acceleration_id,
            workload_class=row["acceleration_class"],
            topology_origin=contract["topology_origin"],
            topology_lifetime=contract["topology_lifetime"],
            topology_validation_sha256=prepared.manifest.topology_sha256,
            timed_scene_update_operation=contract["timed_scene_update_operation"],
            timed_acceleration_operation=contract["timed_acceleration_operation"],
            operation_classification=classification,
            scene_update_count=scene_count,
            refit_count=refit_count,
            rebuild_count=rebuild_count,
            primitive_count=1,
            node_count=1,
            leaf_count=1,
            scene_update_kernel_launch_count=0 if classification == "none" else 1,
            acceleration_kernel_launch_count=0 if classification == "none" else 1,
            device_allocation_count=0,
            host_materialization_count=0,
            host_upload_bytes=0,
            device_to_host_bytes=0,
            global_synchronize_count=0,
            bridge_creation_count=0,
            completion_event_recorded=True,
            instrumentation_trace_sha256=operation_trace.sha256,
        )
        receipt = ExecutionReceipt(
            producer="warp_lane",
            row_semantics_sha256=canonical_row_semantics_sha256(row, self.gate),
            row_id=row["row_id"],
            case_id=row["case_id"],
            case_sha256=self.gate["cases"][row["case_id"]]["sha256"],
            native_input_manifest_sha256=("f" * 64 if self.bad_manifest else prepared.manifest.sha256),
            native_input_content_sha256=prepared.manifest.content_sha256,
            topology_sha256=prepared.manifest.topology_sha256,
            native_readiness=self.readiness,
            native_readiness_sha256=self.readiness_sha256,
            input_contract=row["input_contract"],
            input_provenance_sha256=canonical_input_provenance_sha256(
                input_contract=row["input_contract"],
                case_id=row["case_id"],
                case_sha256=self.gate["cases"][row["case_id"]]["sha256"],
                native_input_manifest_sha256=prepared.manifest.sha256,
                native_input_content_sha256=prepared.manifest.content_sha256,
                topology_sha256=prepared.manifest.topology_sha256,
                scene_contract_sha256=prepared.manifest.scene_contract_sha256,
                transform_replay_sha256=None,
                transform_frame_index=None,
                fixed_render_fixture_sha256=prepared.manifest.fixed_render_fixture_sha256,
                native_execution_binding_sha256=native_execution_binding.sha256,
                asset_manifest_sha256=eligibility.asset_manifest_sha256,
                shipping_publication_sha256=None,
            ),
            scene_contract_sha256=prepared.manifest.scene_contract_sha256,
            transform_replay_sha256=None,
            transform_frame_index=None,
            fixed_render_fixture_sha256=prepared.manifest.fixed_render_fixture_sha256,
            native_execution_binding=native_execution_binding,
            native_execution_binding_sha256=native_execution_binding.sha256,
            eligibility=eligibility,
            eligibility_sha256=eligibility.sha256,
            operation_trace=operation_trace,
            operation_trace_sha256=operation_trace.sha256,
            acceleration=acceleration,
            acceleration_sha256=acceleration.sha256,
            shipping_publication=None,
            shipping_publication_sha256=None,
            boundary=row["boundary"],
            boundary_version=self.gate["boundaries"][row["boundary"]],
            lane=self.lane_id,
            renderer_id=lane_binding["renderer_id"],
            acceleration_id=acceleration_id,
            output_profile=row["output_profile"],
            shadows=not row["shadows"] if self.wrong_shadows else row["shadows"],
            payload=row.get("payload", "none"),
            delivery_policy=row.get("delivery_policy", "native_only"),
            ring_depth=row.get("ring_depth", 0),
            completion_endpoint=row["completion_endpoint"],
            observed_per_ray_channels=(),
            observed_scalar_channels=(),
            observed_payload_bytes=0,
            lifecycle=tuple((name, True) for name in row["required_lifecycle"]),
            device_uuid="GPU-unit-test",
            counters={},
            memory={},
        )
        sample = ResolvedLaneSample(
            receipt=receipt,
            metrics_ms={metric: 1.0 for metric in row["primary_metrics"]},
            artifact_hashes={},
        )
        return _FakePending(sample=sample, trace=self.trace)

    def preflight_sample(self, *, row, prepared):
        del prepared
        self.preflight_called = True
        return evaluate_lane_eligibility(
            row=row,
            lane="warp",
            entry_point_audits=tuple(
                verified_runtime_entry_point_audit(
                    operation=operation,
                    entry_point=_TEST_ENTRY_POINTS[operation],
                )
                for operation in _required_operations(row)
            ),
            asset_manifest_sha256=("a" * 64 if row["required_asset_manifest"] else None),
        )


def _required_operations(row):
    if row["boundary"] == "R0":
        return ("render",)
    if row["boundary"] == "R1":
        return ("scene_update", "acceleration", "render", "reclaim")
    raise AssertionError("unit fake only supports native R0/R1")


_TEST_ENTRY_POINTS = {
    "scene_update": "json.dumps",
    "acceleration": "hashlib.sha256",
    "render": "math.isfinite",
    "reclaim": "time.perf_counter",
}


def _invoke_test_entry_point(operation):
    if operation == "scene_update":
        json.dumps({})
    elif operation == "acceleration":
        hashlib.sha256(b"")
    elif operation == "render":
        math.isfinite(1.0)
    elif operation == "reclaim":
        time.perf_counter()
    else:  # pragma: no cover - guarded by the fake-lane boundary restriction.
        raise AssertionError(f"unsupported fake operation {operation}")
