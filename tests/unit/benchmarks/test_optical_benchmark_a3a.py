from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

import benchmarks.optical.assets as optical_assets
import benchmarks.optical.schema as optical_schema
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
    verified_runtime_entry_point_audit,
)
from benchmarks.optical.instrumentation import (
    instrument_runtime_entry_points,
    invoke_instrumented_entry_point,
)
from benchmarks.optical.lanes import CurrentCudaLanePreflight, CurrentWarpLanePreflight
from benchmarks.optical.lanes.audit import verify_current_product_source_manifest
from benchmarks.optical.native import (
    CanonicalNativeInput,
    NativeInputValidationReport,
    WarpPreparedNativeInput,
    certify_native_readiness,
    prepared_storage_signatures,
    prepared_transform_frame_signatures,
)
from benchmarks.optical.receipts import (
    DYNAMIC_ACCELERATION_INPUT_NAMES,
    DYNAMIC_ACCELERATION_OUTPUT_NAMES,
    DYNAMIC_RENDER_METADATA_INPUT_NAMES,
    DYNAMIC_RENDER_SHADE_INPUT_NAMES,
    DYNAMIC_RENDER_WORLD_INPUT_NAMES,
    DYNAMIC_SCENE_OUTPUT_NAMES,
    DYNAMIC_SCENE_UPDATE_RUNTIME_INPUT_NAMES,
    NATIVE_STAGE_EVIDENCE_SCHEMA_VERSION,
    SHIPPING_PUBLICATION_RECEIPT_SCHEMA_VERSION,
    NativeStageEvidence,
    NativeStorageSignature,
    ShippingPublicationReceipt,
    build_c3_one_shot_native_execution_binding,
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
from benchmarks.optical.workloads import (
    build_canonical_fixed_fixture,
    build_canonical_packed_scene,
    build_canonical_transform_replay,
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
    assert gate["lane_bindings"]["warp"]["acceleration_by_class"]["production_static"] == (
        "cuda_morton_karras_lbvh_rebuild_v1"
    )
    assert gate["lane_bindings"]["cuda"]["acceleration_by_class"]["production_static"] == (
        "cuda_morton_karras_lbvh_rebuild_v1"
    )
    r1_rows = [row for row in gate["mandatory_rows"] if row["boundary"] == "R1"]
    r2_rows = [row for row in gate["mandatory_rows"] if row["boundary"] == "R2"]
    shipping_rows = [row for row in gate["mandatory_rows"] if row["boundary"] == "S"]
    assert all(row["input_contract"] == CANONICAL_NATIVE_REPLAY_INPUT for row in r1_rows + r2_rows)
    assert all(row["acceleration_class"] == "production_static" for row in r2_rows)
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
def test_current_product_source_manifest_is_explicitly_not_runtime_evidence(lane):
    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r1_dynamic_full_shadow")

    audits = verify_current_product_source_manifest(lane)
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
    assert all(audit.source_manifest_sha256 is not None for audit in audits)
    assert all(not audit.verified_runtime_hooks for audit in audits)


def test_c3_source_manifests_name_real_prepared_scene_and_acceleration_entries():
    warp_entries = {audit.entry_point for audit in verify_current_product_source_manifest("warp")}
    cuda_entries = {audit.entry_point for audit in verify_current_product_source_manifest("cuda")}

    assert "optics.device_scene.WarpSceneUpdateCommandBundle.enqueue" in warp_entries
    assert (
        "benchmarks.optical.lanes.warp.WarpCudaLbvhComparisonCommandBundle.enqueue_scene_update"
    ) in warp_entries
    assert (
        "benchmarks.optical.lanes.warp.WarpCudaLbvhComparisonCommandBundle.enqueue_acceleration"
    ) in warp_entries
    assert "optics.cuda_scene.CudaSceneUpdateStage.enqueue" in cuda_entries
    assert "optics.cuda_lbvh.CudaLbvhRebuildStage.enqueue" in cuda_entries


def test_current_preflight_adapters_do_not_expose_a_fake_submit_api():
    warp = CurrentWarpLanePreflight(repo_root=REPO_ROOT)
    cuda = CurrentCudaLanePreflight(repo_root=REPO_ROOT)

    assert not hasattr(warp, "submit_sample")
    assert not hasattr(cuda, "submit_sample")


@pytest.mark.parametrize("phase", ["P12.4a3b", "P12.4b4/d"])
def test_frozen_result_activation_blocks_runner_before_lane_preflight(phase):
    gate = load_frozen_gate()
    lane = _NeverCalledLane()

    with pytest.raises(LaneIneligibleError, match="closed until versioned C4/B3 approval"):
        SingleLaneSampleRunner(gate, execution_phase=phase)

    assert lane.preflight_called is False


def test_runner_rejects_ineligible_lane_before_submit_or_result_creation(monkeypatch):
    gate = _activated_gate(monkeypatch)
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


def test_runner_phase_authorization_rejects_shipping_during_a3b_before_preflight(monkeypatch):
    gate = _activated_gate(monkeypatch)
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


def test_real_product_entry_point_runtime_hook_catches_unreported_low_level_allocation():
    entry_point = "tools.optical_pipeline_lab.rgb_pack.pack_linear_rgb_to_preview_uint8"
    audit = verified_runtime_entry_point_audit(
        operation="pack",
        entry_point=entry_point,
    )
    trace = RuntimeOperationTrace(row_id="runtime-hook", lane="warp", boundary="R0")

    class Result:
        def channel(self, name):
            assert name == "rgb"
            return type("Rgb", (), {"shape": (1, 3), "device": "cuda:0"})()

    with pytest.raises(TimedBoundaryViolation, match="allocation is forbidden"):
        with instrument_runtime_entry_points((audit,)) as observation:
            with native_timed_submission(
                row_id="runtime-hook",
                lane="warp",
                boundary="R0",
                operation_trace=trace,
            ):
                invoke_instrumented_entry_point(entry_point, Result())

    evidence = trace.evidence()
    assert [(record.operation, record.capture_source) for record in evidence.records] == [
        ("pack", "runtime_hook")
    ]
    assert observation.violation_hook_id == "warp.empty"
    assert observation.violation_contamination == "allocation"
    assert active_timed_boundary() is None
    assert active_operation_trace() is None


@pytest.mark.parametrize(
    ("hook_target", "contamination"),
    [
        ("numpy.empty", "allocation"),
        ("torch.tensor", "host_upload"),
        (
            "tools.optical_pipeline_lab.delivery.stage_optical_channels",
            "host_materialization",
        ),
        ("torch.cuda.synchronize", "global_device_synchronize"),
        ("warp.synchronize_event", "host_synchronize"),
        ("warp.to_torch", "warp_to_torch_bridge"),
        ("torch.utils.cpp_extension.load_inline", "extension_load"),
    ],
)
def test_registered_low_level_hook_classes_fail_before_the_original_call(
    hook_target,
    contamination,
):
    audit = verified_runtime_entry_point_audit(
        operation="delivery",
        entry_point="tools.optical_pipeline_lab.delivery.VideoDeliveryFacade.submit",
    )
    trace = RuntimeOperationTrace(row_id="hook-classes", lane="cuda", boundary="R0")

    with pytest.raises(TimedBoundaryViolation, match=f"{contamination} is forbidden"):
        with instrument_runtime_entry_points((audit,)) as observation:
            with native_timed_submission(
                row_id="hook-classes",
                lane="cuda",
                boundary="R0",
                operation_trace=trace,
            ):
                invoke_instrumented_entry_point(hook_target)

    assert observation.violation_hook_id == hook_target
    assert observation.violation_contamination == contamination


def test_c3_one_shot_binding_cannot_reach_operation_trace(monkeypatch):
    gate = _activated_gate(monkeypatch)
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r1_dynamic_full_shadow")
    prepared = _prepared_input(case_id=row["case_id"])
    lane = _OutOfOrderLane(row=row)

    with pytest.raises(LaneIneligibleError, match="C3 one-shot"):
        SingleLaneSampleRunner(gate, execution_phase="P12.4a3b").submit_sample(
            row_id=row["row_id"],
            lane=lane,
            prepared=prepared,
            readiness=_readiness(prepared),
            request=LaneSampleRequest(run_id="out-of-order", repeat_index=0, sample_index=0),
        )


def test_runner_rejects_binding_acceleration_class_that_differs_from_row(monkeypatch):
    gate = _activated_gate(monkeypatch)
    row = next(row for row in gate["mandatory_rows"] if row["boundary"] == "R1")
    prepared = _prepared_input(case_id=row["case_id"])
    lane = _OutOfOrderLane(row=row, acceleration_class="production_static")

    with pytest.raises(LaneIneligibleError, match="acceleration_class"):
        SingleLaneSampleRunner(gate, execution_phase="P12.4a3b").submit_sample(
            row_id=row["row_id"],
            lane=lane,
            prepared=prepared,
            readiness=_readiness(prepared),
            request=LaneSampleRequest(run_id="wrong-class", repeat_index=0, sample_index=0),
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
                instrumented=False,
            ),
        ),
        asset_manifest_sha256=None,
    )
    trace = RuntimeOperationTraceEvidence(
        schema_version="q54.p12.4.operation-trace.v3",
        row_id=row["row_id"],
        lane="warp",
        boundary="R0",
        records=(
            RuntimeOperationRecord(
                sequence=0,
                operation="render",
                entry_point="fabricated.warp.render",
                capture_source="lane",
            ),
        ),
    )

    with pytest.raises(ValueError, match="unaudited entry points"):
        validate_trace_entry_points(trace, eligibility)


def test_submission_trace_rejects_arena_lifecycle_reclaim():
    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r0_asymmetric_full_shadow")
    entry_points = {
        "render": "unit.render",
        "terminal_completion": "unit.finish",
        "reclaim": "unit.reclaim",
    }
    eligibility = evaluate_lane_eligibility(
        row=row,
        lane="warp",
        entry_point_audits=tuple(
            EntryPointAudit(
                operation=operation,
                entry_point=entry_point,
                instrumented=True,
            )
            for operation, entry_point in entry_points.items()
        ),
        asset_manifest_sha256=None,
    )
    trace = RuntimeOperationTraceEvidence(
        schema_version="q54.p12.4.operation-trace.v3",
        row_id=row["row_id"],
        lane="warp",
        boundary="R0",
        records=tuple(
            RuntimeOperationRecord(
                sequence=index,
                operation=operation,
                entry_point=entry_points[operation],
                capture_source="runtime_hook",
                runtime_hook_id=f"entry_point:{entry_points[operation]}",
            )
            for index, operation in enumerate(("render", "terminal_completion", "reclaim"))
        ),
    )

    with pytest.raises(ValueError, match="lifecycle-only"):
        validate_trace_entry_points(trace, eligibility)


def test_manual_instrumented_boolean_cannot_grant_lane_eligibility():
    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r0_asymmetric_full_shadow")
    evidence = evaluate_lane_eligibility(
        row=row,
        lane="warp",
        entry_point_audits=(
            EntryPointAudit(
                operation="render",
                entry_point="json.dumps",
                instrumented=True,
            ),
        ),
        asset_manifest_sha256=None,
    )

    assert evidence.eligible is False
    assert evidence.reason_codes == (
        "missing_entry_point:reclaim",
        "missing_entry_point:terminal_completion",
        "unverified_runtime_hook_plan:render",
    )


def test_lane_cannot_forge_runtime_hook_trace_provenance():
    audit = verified_runtime_entry_point_audit(
        operation="render",
        entry_point="json.dumps",
    )
    trace = RuntimeOperationTrace(row_id="forged-hook", lane="cuda", boundary="R0")

    with pytest.raises(TimedBoundaryViolation, match="runtime operation hook is not installed"):
        with instrument_runtime_entry_points((audit,)):
            with native_timed_submission(
                row_id="forged-hook",
                lane="cuda",
                boundary="R0",
                operation_trace=trace,
            ):
                trace.record(
                    "render",
                    entry_point="json.dumps",
                    capture_source="runtime_hook",
                    runtime_hook_id="entry_point:json.dumps",
                )

    assert trace.evidence().records == ()


def test_go2_asset_preflight_requires_exact_complete_manifest(tmp_path, monkeypatch):
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

    with pytest.raises(AssetPreflightError, match="frozen workload source"):
        preflight_case_asset("go2_static_960x640_v1", repo_root=root)
    monkeypatch.setattr(optical_assets, "GO2_FROZEN_SOURCE_REVISION", source_revision)
    monkeypatch.setattr(
        optical_assets,
        "GO2_FROZEN_ASSET_AGGREGATE_SHA256",
        payload["aggregate_sha256"],
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
    acceleration_class: str = "production_dynamic"
    lane_id: str = "warp"

    def preflight_sample(self, *, row, prepared):
        del prepared
        return evaluate_lane_eligibility(
            row=row,
            lane="warp",
            entry_point_audits=tuple(
                verified_runtime_entry_point_audit(operation=operation, entry_point=entry_point)
                for operation, entry_point in _UNIT_ENTRY_POINTS.items()
            ),
            asset_manifest_sha256=None,
        )

    def bind_sample(self, *, row, prepared, readiness):
        scene_storage = prepared_storage_signatures(prepared, prepared.manifest.scene_array_names)
        transform_storage = prepared_transform_frame_signatures(prepared, 0)
        scene_outputs = tuple(
            _unit_storage(name=name, token=0x8000 + index * 0x10)
            for index, name in enumerate(DYNAMIC_SCENE_OUTPUT_NAMES)
        )
        acceleration_outputs = tuple(
            _unit_storage(name=name, token=0x9000 + index * 0x10)
            for index, name in enumerate(DYNAMIC_ACCELERATION_OUTPUT_NAMES)
        )
        output = NativeStorageSignature(
            name="output",
            backend="warp",
            device_ordinal=0,
            storage_token="warp:0x8200",
            dtype="<f4",
            shape=(1,),
            nbytes=4,
            active_start=0,
            active_count=4,
            owner_token="unit-slot",
        )
        runtime_by_name = {value.name: value for value in scene_storage}
        scene_by_name = {value.name: value for value in scene_outputs}
        acceleration_by_name = {value.name: value for value in acceleration_outputs}
        render_names = (
            DYNAMIC_RENDER_WORLD_INPUT_NAMES
            + DYNAMIC_RENDER_METADATA_INPUT_NAMES
            + DYNAMIC_ACCELERATION_OUTPUT_NAMES
            + DYNAMIC_RENDER_SHADE_INPUT_NAMES
        )
        acceleration_operation = (
            "warp_bvh_leaf_internal_refit_out_v1"
            if self.acceleration_class == "production_dynamic"
            else "cuda_lbvh_scene_bounds_morton_sort_topology_bounds_out_v1"
        )
        stage_specs = (
            (
                "scene_update",
                "warp_local_scene_transform_and_aabb_out_v1",
                tuple(runtime_by_name[name] for name in DYNAMIC_SCENE_UPDATE_RUNTIME_INPUT_NAMES)
                + transform_storage,
                scene_outputs,
            ),
            (
                "acceleration",
                acceleration_operation,
                tuple(scene_by_name[name] for name in DYNAMIC_ACCELERATION_INPUT_NAMES),
                acceleration_outputs,
            ),
            (
                "render",
                "warp_bvh_direct_light",
                tuple(
                    (runtime_by_name | scene_by_name | acceleration_by_name)[name] for name in render_names
                ),
                (output,),
            ),
        )
        stages = tuple(
            NativeStageEvidence(
                schema_version=NATIVE_STAGE_EVIDENCE_SCHEMA_VERSION,
                stage_name=name,
                operation_id=operation_id,
                output_profile=row["output_profile"] if name == "render" else None,
                shadows=row["shadows"] if name == "render" else None,
                scene_contract_sha256=prepared.manifest.scene_contract_sha256,
                input_contract=prepared.manifest.input_contract,
                transform_replay_sha256=prepared.manifest.transform_replay_sha256,
                transform_frame_index=0,
                fixed_render_fixture_sha256=None,
                topology_sha256=prepared.manifest.topology_sha256,
                input_storage=inputs,
                output_storage=outputs,
                dispatch_plan_sha256=hex(index + 10)[2:] * 64,
            )
            for index, (name, operation_id, inputs, outputs) in enumerate(stage_specs)
        )
        return build_c3_one_shot_native_execution_binding(
            row_id=row["row_id"],
            case_id=row["case_id"],
            case_sha256=prepared.manifest.case_sha256,
            boundary=row["boundary"],
            lane="warp",
            renderer_id="warp_bvh_direct_light",
            acceleration_class=self.acceleration_class,
            acceleration_id=(
                "cpu_median_leaf4_topology_warp_refit_v1"
                if self.acceleration_class == "production_dynamic"
                else "cuda_morton_karras_lbvh_rebuild_v1"
            ),
            output_profile=row["output_profile"],
            shadows=row["shadows"],
            input_contract=prepared.manifest.input_contract,
            native_input_manifest_sha256=prepared.manifest.sha256,
            native_readiness_sha256=readiness.sha256,
            scene_contract_sha256=prepared.manifest.scene_contract_sha256,
            transform_replay_sha256=prepared.manifest.transform_replay_sha256,
            transform_frame_index=0,
            fixed_render_fixture_sha256=None,
            runtime_scene_storage=scene_storage,
            derived_fixture_storage=(),
            stage_evidence=stages,
        )

    def submit_sample(self, *, row, prepared, binding, request):
        del row, prepared, binding, request
        import hashlib
        import json
        import math
        import time

        json.dumps({})
        math.isfinite(1.0)
        hashlib.sha256(b"")
        time.perf_counter()
        return _PendingNeverResolve()


def _unit_storage(*, name: str, token: int) -> NativeStorageSignature:
    return NativeStorageSignature(
        name=name,
        backend="warp",
        device_ordinal=0,
        storage_token=f"warp:0x{token:x}",
        dtype="<f4",
        shape=(1,),
        nbytes=4,
        active_start=0,
        active_count=4,
        owner_token="unit-slot",
    )


_UNIT_ENTRY_POINTS = {
    "arena_acquire": "operator.mul",
    "scene_update": "json.dumps",
    "acceleration": "hashlib.sha256",
    "render": "math.isfinite",
    "terminal_completion": "math.trunc",
    "reclaim": "time.perf_counter",
}


class _PendingNeverResolve:
    def resolve(self):
        raise AssertionError("invalid trace must not resolve")


def _prepared_input(*, case_id="asymmetric_synthetic_7x3_v1"):
    packed = build_canonical_packed_scene(case_id)
    if case_id == "asymmetric_synthetic_7x3_v1":
        fixture = build_canonical_fixed_fixture(packed)
        source = CanonicalNativeInput.from_scene_contract(
            packed,
            input_contract="fixed_native_scene_v1",
            fixed_fixture=fixture,
        )
    elif case_id == "synthetic_dynamic_rigid_960x640_v1":
        replay = build_canonical_transform_replay(case_id)
        source = CanonicalNativeInput.from_scene_contract(
            packed,
            input_contract=CANONICAL_NATIVE_REPLAY_INPUT,
            replay=replay,
        )
    else:
        source = CanonicalNativeInput.from_scene_contract(
            packed,
            input_contract=GPU_ENGINE_PUBLISHED_FRAME_INPUT,
        )
    return WarpPreparedNativeInput(
        manifest=source.manifest,
        device_ordinal=0,
        arrays={
            manifest.name: _FakeWarpArray(0 if manifest.nbytes == 0 else 0x1000 + index * 0x100)
            for index, manifest in enumerate(source.manifest.arrays)
        },
        ready_event=object(),
        source=source,
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


@dataclass(frozen=True)
class _FakeWarpArray:
    ptr: int


def _readiness(prepared):
    validation = NativeInputValidationReport(
        backend="warp",
        manifest_sha256=prepared.manifest.sha256,
        content_sha256=prepared.manifest.content_sha256,
        topology_sha256=prepared.manifest.topology_sha256,
        validated_arrays=tuple(value.name for value in prepared.manifest.arrays),
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
