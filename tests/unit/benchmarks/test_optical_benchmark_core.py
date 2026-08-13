from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field

import numpy as np
import pytest

from benchmarks.optical.artifacts import BenchmarkArtifactWriter
from benchmarks.optical.eligibility import (
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
)
from benchmarks.optical.receipts import (
    ACCELERATION_RECEIPT_SCHEMA_VERSION,
    AccelerationReceipt,
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


def test_single_lane_runner_submits_inside_timing_and_resolves_outside():
    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r0_asymmetric_full_shadow")
    source = _native_source()
    prepared = WarpPreparedNativeInput(
        manifest=source.manifest,
        device_ordinal=0,
        arrays={"placeholder": object()},
        ready_event=object(),
    )
    trace: list[tuple[str, bool]] = []
    lane = _FakeLane(gate=gate, row=row, prepared=prepared, trace=trace)
    runner = SingleLaneSampleRunner(gate, execution_phase="P12.4a3b")

    record = runner.run_sample(
        row_id=row["row_id"],
        lane=lane,
        prepared=prepared,
        readiness=_readiness(prepared),
        request=LaneSampleRequest(run_id="unit-run", repeat_index=1, sample_index=2),
    )

    assert trace == [("submit", True), ("resolve", False)]
    assert record["run_id"] == "unit-run"
    assert record["repeat_index"] == 1
    assert record["sample_index"] == 2
    assert record["receipt"]["native_input_manifest_sha256"] == source.manifest.sha256


def test_single_lane_runner_rejects_lane_fabricated_native_provenance():
    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r0_asymmetric_full_shadow")
    source = _native_source()
    prepared = WarpPreparedNativeInput(
        manifest=source.manifest,
        device_ordinal=0,
        arrays={"placeholder": object()},
        ready_event=object(),
    )
    lane = _FakeLane(gate=gate, row=row, prepared=prepared, trace=[], bad_manifest=True)

    with pytest.raises(ValueError, match="prepared input manifest"):
        SingleLaneSampleRunner(gate, execution_phase="P12.4a3b").run_sample(
            row_id=row["row_id"],
            lane=lane,
            prepared=prepared,
            readiness=_readiness(prepared),
            request=LaneSampleRequest(run_id="unit-run", repeat_index=0, sample_index=0),
        )


def test_single_lane_runner_allows_ordered_submit_then_drain_and_rejects_double_resolve():
    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r0_asymmetric_full_shadow")
    source = _native_source()
    prepared = WarpPreparedNativeInput(
        manifest=source.manifest,
        device_ordinal=0,
        arrays={"placeholder": object()},
        ready_event=object(),
    )
    readiness = _readiness(prepared)
    trace: list[tuple[str, bool]] = []
    lane = _FakeLane(gate=gate, row=row, prepared=prepared, trace=trace)
    runner = SingleLaneSampleRunner(gate, execution_phase="P12.4a3b")

    submitted = [
        runner.submit_sample(
            row_id=row["row_id"],
            lane=lane,
            prepared=prepared,
            readiness=readiness,
            request=LaneSampleRequest(run_id="ordered", repeat_index=0, sample_index=index),
        )
        for index in range(2)
    ]
    assert trace == [("submit", True), ("submit", True)]

    records = [runner.resolve_sample(sample) for sample in submitted]
    assert trace == [
        ("submit", True),
        ("submit", True),
        ("resolve", False),
        ("resolve", False),
    ]
    assert [record["sample_index"] for record in records] == [0, 1]
    with pytest.raises(RuntimeError, match="already been resolved"):
        runner.resolve_sample(submitted[0])


def test_single_lane_runner_rejects_incomplete_warmup_evidence():
    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r0_asymmetric_full_shadow")
    source = _native_source()
    prepared = WarpPreparedNativeInput(
        manifest=source.manifest,
        device_ordinal=0,
        arrays={"placeholder": object()},
        ready_event=object(),
    )
    validation = NativeInputValidationReport(
        backend="warp",
        manifest_sha256=prepared.manifest.sha256,
        content_sha256=prepared.manifest.content_sha256,
        topology_sha256=prepared.manifest.topology_sha256,
        validated_arrays=tuple(item.name for item in prepared.manifest.arrays),
    )
    incomplete_validation = NativeInputValidationReport(
        backend="warp",
        manifest_sha256=prepared.manifest.sha256,
        content_sha256=prepared.manifest.content_sha256,
        topology_sha256=prepared.manifest.topology_sha256,
        validated_arrays=(),
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


def test_single_lane_runner_does_not_copy_wrong_actual_request_from_gate():
    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r0_asymmetric_full_shadow")
    source = _native_source()
    prepared = WarpPreparedNativeInput(
        manifest=source.manifest,
        device_ordinal=0,
        arrays={"placeholder": object()},
        ready_event=object(),
    )
    lane = _FakeLane(gate=gate, row=row, prepared=prepared, trace=[], wrong_shadows=True)

    with pytest.raises(ValueError, match="receipt shadows does not match"):
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


def _readiness(prepared: WarpPreparedNativeInput):
    validation = NativeInputValidationReport(
        backend="warp",
        manifest_sha256=prepared.manifest.sha256,
        content_sha256=prepared.manifest.content_sha256,
        topology_sha256=prepared.manifest.topology_sha256,
        validated_arrays=tuple(item.name for item in prepared.manifest.arrays),
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
    lane_id: str = "warp"
    readiness: NativeReadinessEvidence = field(init=False)
    readiness_sha256: str = field(init=False)

    def __post_init__(self):
        self.readiness = _readiness(self.prepared)
        self.readiness_sha256 = self.readiness.sha256

    def submit_sample(self, *, row, prepared, request):
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
        binding = self.gate["lane_bindings"][self.lane_id]
        acceleration_id = binding["acceleration_by_class"][row["acceleration_class"]]
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
                native_input_manifest_sha256=prepared.manifest.sha256,
                native_input_content_sha256=prepared.manifest.content_sha256,
                topology_sha256=prepared.manifest.topology_sha256,
                asset_manifest_sha256=eligibility.asset_manifest_sha256,
                shipping_publication_sha256=None,
            ),
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
            renderer_id=binding["renderer_id"],
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
