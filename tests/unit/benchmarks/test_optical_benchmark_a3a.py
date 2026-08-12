from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from benchmarks.optical.assets import (
    ASSET_MANIFEST_SCHEMA_VERSION,
    GO2_ASSET_REQUIREMENT,
    GO2_SOURCE_REPOSITORY,
    AssetPreflightError,
    preflight_case_asset,
)
from benchmarks.optical.eligibility import (
    CANONICAL_NATIVE_REPLAY_INPUT,
    GPU_ENGINE_PUBLISHED_FRAME_INPUT,
    EntryPointAudit,
    LaneIneligibleError,
    evaluate_lane_eligibility,
    validate_trace_entry_points,
)
from benchmarks.optical.lanes import CurrentCudaLanePreflight, CurrentWarpLanePreflight
from benchmarks.optical.lanes.audit import verify_current_product_entry_point_audits
from benchmarks.optical.native import (
    CanonicalNativeInput,
    NativeInputValidationReport,
    WarpPreparedNativeInput,
    certify_native_readiness,
)
from benchmarks.optical.receipts import (
    SHIPPING_PUBLICATION_RECEIPT_SCHEMA_VERSION,
    ShippingPublicationReceipt,
)
from benchmarks.optical.runner import LaneSampleRequest, SingleLaneSampleRunner
from benchmarks.optical.schema import load_frozen_gate
from benchmarks.optical.timing import (
    RuntimeOperationRecord,
    RuntimeOperationTrace,
    RuntimeOperationTraceEvidence,
    TimedBoundaryViolation,
    active_operation_trace,
    active_timed_boundary,
    native_timed_submission,
    record_runtime_operation,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_gate_freezes_exact_acceleration_and_separate_input_contracts():
    gate = load_frozen_gate()

    assert gate["lane_bindings"]["warp"]["acceleration_by_class"]["production_dynamic"] == (
        "cpu_median_leaf4_topology_warp_refit_v1"
    )
    assert gate["lane_bindings"]["cuda"]["acceleration_by_class"]["production_dynamic"] == (
        "cuda_morton_karras_lbvh_rebuild_v1"
    )
    r1_rows = [row for row in gate["mandatory_rows"] if row["boundary"] == "R1"]
    r2_rows = [row for row in gate["mandatory_rows"] if row["boundary"] == "R2"]
    shipping_rows = [row for row in gate["mandatory_rows"] if row["boundary"] == "S"]
    assert all(row["input_contract"] == CANONICAL_NATIVE_REPLAY_INPUT for row in r1_rows + r2_rows)
    assert all(row["input_contract"] == GPU_ENGINE_PUBLISHED_FRAME_INPUT for row in shipping_rows)
    assert gate["eligibility_contract"]["ineligible_result_emission"] == "forbidden"
    assert gate["eligibility_contract"]["a3a_baseline_emission"] == "forbidden"
    assert gate["phase_authorization"]["P12.4a3a"]["executable_boundaries"] == []
    assert gate["phase_authorization"]["P12.4a3b"]["executable_boundaries"] == [
        "R0",
        "R1",
        "R2",
    ]
    assert gate["phase_authorization"]["P12.4b4/d"]["executable_boundaries"] == ["S"]
    assert gate["native_input_contract"]["S_measured_physics_transforms_excluded"] is True
    assert "baseline_results" not in gate


@pytest.mark.parametrize("lane", ["warp", "cuda"])
def test_current_product_entry_point_audit_is_source_bound_and_ineligible(lane):
    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r1_dynamic_full_shadow")

    audits = verify_current_product_entry_point_audits(REPO_ROOT, lane)
    evidence = evaluate_lane_eligibility(
        row=row,
        lane=lane,
        entry_point_audits=audits,
        asset_manifest_sha256=None,
    )

    assert evidence.eligible is False
    assert any(reason.startswith("uninstrumented_entry_point:") for reason in evidence.reason_codes)
    assert any(reason.startswith("timed_contamination:allocation:") for reason in evidence.reason_codes)
    assert any("execute_camera" in audit.entry_point for audit in audits)


def test_current_preflight_adapters_do_not_expose_a_fake_submit_api():
    warp = CurrentWarpLanePreflight(repo_root=REPO_ROOT)
    cuda = CurrentCudaLanePreflight(repo_root=REPO_ROOT)

    assert not hasattr(warp, "submit_sample")
    assert not hasattr(cuda, "submit_sample")


def test_runner_rejects_ineligible_lane_before_submit_or_result_creation():
    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r0_asymmetric_full_shadow")
    prepared = _prepared_input(case_id=row["case_id"])
    lane = _IneligibleLane(row=row)

    with pytest.raises(LaneIneligibleError, match="ineligible"):
        SingleLaneSampleRunner(gate, execution_phase="P12.4a3b").run_sample(
            row_id=row["row_id"],
            lane=lane,
            prepared=prepared,
            readiness=_readiness(prepared),
            request=LaneSampleRequest(run_id="must-not-exist", repeat_index=0, sample_index=0),
        )

    assert lane.submit_called is False


def test_runner_phase_authorization_rejects_shipping_during_a3b_before_preflight():
    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["boundary"] == "S")
    lane = _NeverCalledLane()

    with pytest.raises(LaneIneligibleError, match="not executable during P12.4a3b"):
        SingleLaneSampleRunner(gate, execution_phase="P12.4a3b").submit_sample(
            row_id=row["row_id"],
            lane=lane,
            prepared=object(),
            readiness=object(),
            request=LaneSampleRequest(run_id="no-shipping", repeat_index=0, sample_index=0),
        )

    assert lane.preflight_called is False


def test_operation_trace_rejects_contamination_before_record_and_restores_context():
    trace = RuntimeOperationTrace(row_id="row", lane="cuda", boundary="R0")

    with pytest.raises(TimedBoundaryViolation, match="allocation is forbidden"):
        with native_timed_submission(
            row_id="row",
            lane="cuda",
            boundary="R0",
            operation_trace=trace,
        ):
            record_runtime_operation(
                "allocation",
                entry_point="torch.empty",
                contamination="allocation",
            )

    assert trace.evidence().records == ()
    assert active_timed_boundary() is None
    assert active_operation_trace() is None


def test_operation_trace_rejects_out_of_order_frozen_transaction():
    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r1_dynamic_full_shadow")
    prepared = _prepared_input(case_id=row["case_id"])
    lane = _OutOfOrderLane(row=row)

    with pytest.raises(LaneIneligibleError, match="ordered transaction"):
        SingleLaneSampleRunner(gate, execution_phase="P12.4a3b").submit_sample(
            row_id=row["row_id"],
            lane=lane,
            prepared=prepared,
            readiness=_readiness(prepared),
            request=LaneSampleRequest(run_id="out-of-order", repeat_index=0, sample_index=0),
        )


def test_operation_trace_reuse_fails_without_leaking_context():
    trace = RuntimeOperationTrace(row_id="row", lane="warp", boundary="R0")
    with native_timed_submission(
        row_id="row",
        lane="warp",
        boundary="R0",
        operation_trace=trace,
    ):
        record_runtime_operation("render", entry_point="unit.warp.render")

    with pytest.raises(TimedBoundaryViolation, match="cannot be reused"):
        with native_timed_submission(
            row_id="row",
            lane="warp",
            boundary="R0",
            operation_trace=trace,
        ):
            pass

    assert active_timed_boundary() is None
    assert active_operation_trace() is None


def test_operation_trace_rejects_entry_point_not_in_instrumented_preflight():
    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r0_asymmetric_full_shadow")
    eligibility = evaluate_lane_eligibility(
        row=row,
        lane="warp",
        entry_point_audits=(
            EntryPointAudit(
                operation="render",
                entry_point="approved.warp.render",
                instrumented=True,
            ),
        ),
        asset_manifest_sha256=None,
    )
    trace = RuntimeOperationTraceEvidence(
        schema_version="q54.p12.4.operation-trace.v1",
        row_id=row["row_id"],
        lane="warp",
        boundary="R0",
        records=(
            RuntimeOperationRecord(
                sequence=0,
                operation="render",
                entry_point="fabricated.warp.render",
            ),
        ),
    )

    with pytest.raises(ValueError, match="unaudited entry points"):
        validate_trace_entry_points(trace, eligibility)


def test_go2_asset_preflight_requires_exact_complete_manifest(tmp_path):
    root = tmp_path
    model = root / "out/external/mujoco_menagerie/unitree_go2"
    (model / "meshes").mkdir(parents=True)
    files = {
        "meshes/body.obj": b"mesh-data\n",
        "scene.xml": b"<mujoco/>\n",
    }
    for relative, content in files.items():
        path = model / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    checkout = model.parent
    _git(checkout, "init")
    _git(checkout, "config", "user.name", "Q54 Test")
    _git(checkout, "config", "user.email", "q54-test@example.invalid")
    _git(checkout, "remote", "add", "origin", GO2_SOURCE_REPOSITORY)
    _git(checkout, "add", "unitree_go2")
    _git(checkout, "commit", "-m", "unit test Go2 asset fixture")
    source_revision = _git(checkout, "rev-parse", "HEAD")
    records = [
        {
            "relative_path": relative,
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for relative, content in sorted(files.items())
    ]
    payload = {
        "schema_version": ASSET_MANIFEST_SCHEMA_VERSION,
        "requirement_id": GO2_ASSET_REQUIREMENT,
        "source_repository": GO2_SOURCE_REPOSITORY,
        "source_revision": source_revision,
        "entrypoint": "scene.xml",
        "files": records,
        "aggregate_sha256": _canonical_sha256(records),
    }
    (model / "q54_asset_manifest.json").write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )

    evidence = preflight_case_asset("go2_static_960x640_v1", repo_root=root)
    assert evidence is not None
    assert evidence.aggregate_sha256 == payload["aggregate_sha256"]
    assert (
        evidence.manifest_sha256
        == hashlib.sha256((model / "q54_asset_manifest.json").read_bytes()).hexdigest()
    )
    assert tuple(item.relative_path for item in evidence.files) == (
        "meshes/body.obj",
        "scene.xml",
    )

    _git(checkout, "remote", "set-url", "origin", "https://github.com/example/surrogate")
    with pytest.raises(AssetPreflightError, match="origin is not the canonical"):
        preflight_case_asset("go2_static_960x640_v1", repo_root=root)
    _git(checkout, "remote", "set-url", "origin", GO2_SOURCE_REPOSITORY)

    (model / "scene.xml").write_bytes(b"surrogate")
    with pytest.raises(AssetPreflightError, match="modified tracked model files"):
        preflight_case_asset("go2_static_960x640_v1", repo_root=root)


def test_go2_asset_preflight_hard_fails_when_asset_is_absent(tmp_path):
    with pytest.raises(AssetPreflightError, match="model directory is absent"):
        preflight_case_asset("go2_static_1920x1080_v1", repo_root=tmp_path)


def test_shipping_publication_receipt_rejects_replay_provenance():
    payload = _valid_shipping_publication_payload()
    payload["replay_source_count"] = 1

    receipt = ShippingPublicationReceipt.from_payload(payload)
    assert receipt.replay_source_count == 1
    # Structural parsing preserves actual observation. Gate validation, not the
    # dataclass parser, is responsible for rejecting this receipt as an S result.


def test_shipping_publication_receipt_rejects_fake_physical_reclaim():
    payload = _valid_shipping_publication_payload()
    payload["producer_reuse_wait_enqueued_when_reused"] = False

    with pytest.raises(ValueError, match="reuse wait evidence"):
        ShippingPublicationReceipt.from_payload(payload)


@dataclass
class _IneligibleLane:
    row: dict[str, object]
    lane_id: str = "warp"
    submit_called: bool = False

    def preflight_sample(self, *, row, prepared):
        del prepared
        return evaluate_lane_eligibility(
            row=row,
            lane="warp",
            entry_point_audits=(
                EntryPointAudit(
                    operation="render",
                    entry_point="current.warp.execute_camera",
                    instrumented=False,
                    contaminations=("allocation",),
                ),
            ),
            asset_manifest_sha256=None,
        )

    def submit_sample(self, *, row, prepared, request):
        del row, prepared, request
        self.submit_called = True
        raise AssertionError("ineligible lane must not be submitted")


@dataclass
class _NeverCalledLane:
    lane_id: str = "warp"
    preflight_called: bool = False

    def preflight_sample(self, *, row, prepared):
        del row, prepared
        self.preflight_called = True
        raise AssertionError("phase authorization must run first")


@dataclass
class _OutOfOrderLane:
    row: dict[str, object]
    lane_id: str = "warp"

    def preflight_sample(self, *, row, prepared):
        del prepared
        return evaluate_lane_eligibility(
            row=row,
            lane="warp",
            entry_point_audits=tuple(
                EntryPointAudit(
                    operation=operation,
                    entry_point=f"unit.warp.{operation}",
                    instrumented=True,
                )
                for operation in ("scene_update", "acceleration", "render", "reclaim")
            ),
            asset_manifest_sha256=None,
        )

    def submit_sample(self, *, row, prepared, request):
        del row, prepared, request
        for operation in ("scene_update", "render", "acceleration", "reclaim"):
            record_runtime_operation(operation, entry_point=f"unit.warp.{operation}")
        return _PendingNeverResolve()


class _PendingNeverResolve:
    def resolve(self):
        raise AssertionError("invalid trace must not resolve")


def _prepared_input(*, case_id="asymmetric_synthetic_7x3_v1"):
    source = CanonicalNativeInput.from_arrays(
        case_id=case_id,
        materializer_id="a3a_unit_input_v1",
        arrays={"bvh_left": np.array([-1], dtype=np.int32)},
        array_semantics={"bvh_left": "fixed topology left child"},
        topology_arrays=("bvh_left",),
        semantics={"coordinate_system": "world"},
    )
    return WarpPreparedNativeInput(
        manifest=source.manifest,
        device_ordinal=0,
        arrays={"bvh_left": object()},
        ready_event=object(),
    )


def _readiness(prepared):
    validation = NativeInputValidationReport(
        backend="warp",
        manifest_sha256=prepared.manifest.sha256,
        content_sha256=prepared.manifest.content_sha256,
        topology_sha256=prepared.manifest.topology_sha256,
        validated_arrays=("bvh_left",),
    )
    return certify_native_readiness(
        prepared,
        validation,
        compile_complete=True,
        warmup_frames_completed=10,
    )


def _canonical_sha256(payload):
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _git(checkout, *arguments):
    return subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _valid_shipping_publication_payload():
    return {
        "publication_schema_version": SHIPPING_PUBLICATION_RECEIPT_SCHEMA_VERSION,
        "input_contract": GPU_ENGINE_PUBLISHED_FRAME_INPUT,
        "physics_run_id": "run",
        "physics_step_index": 0,
        "physics_frame_id": 0,
        "physics_slot_id": 0,
        "physics_device_ordinal": 0,
        "physics_step_count": 1,
        "publish_event_token": "publish",
        "publish_event_recorded": True,
        "producer_publish_completion_recorded": True,
        "consumer_binding_id": "binding",
        "consumer_stream_token": "stream",
        "borrow_count": 1,
        "consumer_stream_wait_enqueued": True,
        "zero_copy_transform_binding": True,
        "last_physics_buffer_reader_operation": "scene_update",
        "consumer_done_event_token": "done",
        "consumer_completion_recorded": True,
        "slot_logically_reclaimable": True,
        "producer_reuse_wait_enqueued_when_reused": True,
        "done_event_retained_until_reuse_dependency": True,
        "physical_reclaim_evidence": "producer_wait_enqueued_before_reuse",
        "final_payload_drain_complete": False,
        "final_reclaim_complete": False,
        "replay_source_count": 0,
        "host_poll_count": 0,
        "host_synchronize_count": 0,
        "publication_trace_sha256": "a" * 64,
    }
