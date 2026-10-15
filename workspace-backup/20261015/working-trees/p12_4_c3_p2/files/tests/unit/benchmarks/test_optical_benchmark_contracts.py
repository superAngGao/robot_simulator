from __future__ import annotations

import copy
import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest

import benchmarks.optical as optical_benchmarks
from benchmarks.optical import (
    ACCELERATION_RECEIPT_SCHEMA_VERSION,
    BOUNDARY_VERSIONS,
    DYNAMIC_ACCELERATION_INPUT_NAMES,
    DYNAMIC_ACCELERATION_OUTPUT_NAMES,
    DYNAMIC_RENDER_METADATA_INPUT_NAMES,
    DYNAMIC_RENDER_SHADE_INPUT_NAMES,
    DYNAMIC_RENDER_WORLD_INPUT_NAMES,
    DYNAMIC_SCENE_OUTPUT_NAMES,
    DYNAMIC_SCENE_UPDATE_RUNTIME_INPUT_NAMES,
    NATIVE_INPUT_SCHEMA_VERSION,
    NATIVE_STAGE_EVIDENCE_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    SHIPPING_PUBLICATION_RECEIPT_SCHEMA_VERSION,
    AccelerationReceipt,
    ExecutionReceipt,
    NativeReadinessEvidence,
    NativeStageEvidence,
    NativeStorageSignature,
    ObservedChannel,
    ShippingPublicationReceipt,
    build_c3_one_shot_native_execution_binding,
    canonical_cases,
    canonical_execution_receipt_sha256,
    canonical_gate_sha256,
    canonical_input_provenance_sha256,
    canonical_row_semantics_sha256,
    case_hashes,
    evaluate_lane_eligibility,
    load_frozen_gate,
    validate_frozen_gate,
    validate_repository_inventory,
    validate_result_record,
    verified_runtime_entry_point_audit,
)
from benchmarks.optical.instrumentation import entry_point_hook_id
from benchmarks.optical.timing import (
    OPERATION_TRACE_SCHEMA_VERSION,
    RuntimeOperationRecord,
    RuntimeOperationTraceEvidence,
)
from optics import (
    DIRECT_LIGHT_FULL_HOST_PER_RAY_CHANNEL_SPECS,
    HOST_DIAGNOSTIC_OVERLAY_CONTRACTS,
    HOST_PAYLOAD_CONTRACTS,
    PRIMARY_HOST_SCALAR_CHANNEL_SPECS,
    SHADOW_TRAVERSAL_HOST_SCALAR_CHANNEL_SPECS,
    CudaDeviceBvhDirectLightOpticalExecutor,
    GpuDeviceBvhDirectLightOpticalExecutor,
    OpticalOutputProfile,
    host_diagnostic_overlay_contract_schema,
    host_payload_contract_schema,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = REPO_ROOT / "benchmarks" / "results" / "q54_p12_4"

EXPECTED_CASE_HASHES = {
    "asymmetric_synthetic_7x3_v1": "3f6ce0ac1d59d0c0d0e5ea57f497c308a9d5c4b2a5c19462c900627fd99d0181",
    "go2_static_1920x1080_v1": "a6df8bd3f2020be74cef32a285ec8a697879e4e1dbc7dd6e7c4c6fe09816392a",
    "go2_static_960x640_v1": "da4045e2d05e7cb731cd5da7690c74a07708359e038512d3da3e6aae5c3f5723",
    "physics_body_triangle_960x640_v1": "c6d0b6f8aa1313cddae6be84122420507bdccafc010e705eae53fa331e5734c6",
    "synthetic_dynamic_rigid_960x640_v1": "7ebdbb64c57f904c732cf10eadfaa96008fcb126497f6e8b66add74d29c077ab",
}

_TEST_ENTRY_POINTS = {
    "physics_step": "time.process_time",
    "publication": "json.loads",
    "stream_wait": "math.ceil",
    "view_binding": "operator.add",
    "scene_update": "json.dumps",
    "consumer_completion": "math.floor",
    "acceleration": "hashlib.sha256",
    "render": "math.isfinite",
    "pack": "operator.sub",
    "delivery": "time.monotonic",
    "reclaim": "time.perf_counter",
}


def test_c3_public_api_has_no_forgeable_c4_capability():
    parameters = inspect.signature(
        optical_benchmarks.build_c3_one_shot_native_execution_binding
    ).parameters

    assert "submission_capability" not in parameters
    assert not hasattr(optical_benchmarks, "C4_RESULT_BINDING_CAPABILITY")
    assert not hasattr(optical_benchmarks, "build_native_execution_binding")
    assert hasattr(optical_benchmarks, "C3OneShotNativeExecutionBindingEvidence")


@pytest.mark.parametrize("lane", ["warp", "cuda"])
def test_every_native_mandatory_row_constructs_its_exact_lane_acceleration(lane):
    gate = load_frozen_gate()
    rows = [
        row for row in gate["mandatory_rows"] if row["boundary"] in {"R0", "R1", "R2"}
    ]

    for row in rows:
        receipt = ExecutionReceipt.from_payload(
            _valid_result_record(gate, row, lane=lane)["receipt"]
        )
        native_binding = receipt.native_execution_binding
        expected_acceleration_id = gate["lane_bindings"][lane]["acceleration_by_class"][
            row["acceleration_class"]
        ]
        acceleration_contract = gate["acceleration_contracts"]["by_lane_and_identity"][lane][
            expected_acceleration_id
        ]

        assert native_binding.acceleration_class == row["acceleration_class"]
        assert native_binding.acceleration_id == expected_acceleration_id
        if row["boundary"] in {"R1", "R2"}:
            stage_operations = {
                stage.stage_name: stage.evidence.operation_id
                for stage in native_binding.ordered_stage_evidence
            }
            assert stage_operations["scene_update"] == acceleration_contract[
                "timed_scene_update_operation"
            ]
            assert stage_operations["acceleration"] == acceleration_contract[
                "timed_acceleration_operation"
            ]

    warp_static_rows = [
        row for row in rows if lane == "warp" and row["acceleration_class"] == "production_static"
    ]
    if lane == "warp":
        assert warp_static_rows
        assert all(row["boundary"] == "R2" for row in warp_static_rows)


def test_canonical_case_ids_and_hashes_are_frozen():
    cases = canonical_cases()

    assert len(cases) == len({case.case_id for case in cases}) == 5
    assert case_hashes() == EXPECTED_CASE_HASHES
    for case in cases:
        assert case.width > 0
        assert case.height > 0
        assert case.render_contract["ray_layout"] == "row_major_py_times_width_plus_px"
        assert case.transform_sequence["frames"] > 0


@pytest.mark.parametrize("forbidden_field", ["evidence_sha256", "native_execution_binding_sha256"])
def test_native_stage_payload_cannot_self_reference_or_reference_outer_binding(
    forbidden_field: str,
):
    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["boundary"] == "R1")
    receipt = ExecutionReceipt.from_payload(_valid_result_record(gate, row)["receipt"])
    payload = receipt.native_execution_binding.ordered_stage_evidence[0].evidence.to_payload()
    payload[forbidden_field] = "f" * 64

    with pytest.raises(ValueError, match="fields are invalid"):
        NativeStageEvidence.from_payload(payload)


def test_native_binding_rejects_same_shape_unproduced_stage_storage():
    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["boundary"] == "R1")
    receipt = ExecutionReceipt.from_payload(_valid_result_record(gate, row)["receipt"])
    binding = receipt.native_execution_binding
    stages = [stage.evidence for stage in binding.ordered_stage_evidence]
    render = stages[2]
    substitute = replace(render.input_storage[-1], storage_token="cuda:0xdeadbeef")
    stages[2] = replace(render, input_storage=render.input_storage[:-1] + (substitute,))

    with pytest.raises(ValueError, match="complete ordered storage contract"):
        _rebuild_c3_binding(binding, stages)


@pytest.mark.parametrize(
    ("stage_name", "storage_name"),
    [
        ("render", "triangle_v0_world"),
        ("acceleration", "triangle_aabb_min"),
        ("render", "bvh_left"),
        ("render", "triangle_numeric_instance_id"),
        ("render", "material_albedo_rgb"),
        ("render", "light_kind"),
    ],
)
def test_dynamic_binding_rejects_each_missing_required_storage(stage_name, storage_name):
    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r1_dynamic_full_shadow")
    binding = ExecutionReceipt.from_payload(
        _valid_result_record(gate, row)["receipt"]
    ).native_execution_binding
    stages = [stage.evidence for stage in binding.ordered_stage_evidence]
    stage_index = next(index for index, stage in enumerate(stages) if stage.stage_name == stage_name)
    stage = stages[stage_index]
    stages[stage_index] = replace(
        stage,
        input_storage=tuple(value for value in stage.input_storage if value.name != storage_name),
    )

    with pytest.raises(ValueError, match="complete ordered storage contract"):
        _rebuild_c3_binding(binding, stages)


def test_dynamic_binding_rejects_extra_and_reordered_render_storage():
    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r1_dynamic_full_shadow")
    binding = ExecutionReceipt.from_payload(
        _valid_result_record(gate, row)["receipt"]
    ).native_execution_binding
    stages = [stage.evidence for stage in binding.ordered_stage_evidence]
    render = stages[2]
    extra = next(
        value for value in binding.runtime_scene_storage if value.name == "triangle_vertices_local"
    )

    stages[2] = replace(render, input_storage=render.input_storage + (extra,))
    with pytest.raises(ValueError, match="complete ordered storage contract"):
        _rebuild_c3_binding(binding, stages)

    stages[2] = replace(
        render,
        input_storage=(render.input_storage[1], render.input_storage[0]) + render.input_storage[2:],
    )
    with pytest.raises(ValueError, match="complete ordered storage contract"):
        _rebuild_c3_binding(binding, stages)


def test_dynamic_binding_contract_is_bound_to_operation_and_profile():
    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r1_dynamic_full_shadow")
    binding = ExecutionReceipt.from_payload(
        _valid_result_record(gate, row)["receipt"]
    ).native_execution_binding
    stages = [stage.evidence for stage in binding.ordered_stage_evidence]
    stages[1] = replace(stages[1], operation_id="unit_fake_refit")

    with pytest.raises(ValueError, match="operation sequence"):
        _rebuild_c3_binding(binding, stages)

    with pytest.raises(ValueError, match="render stage identity"):
        _rebuild_c3_binding(
            binding,
            [stage.evidence for stage in binding.ordered_stage_evidence],
            output_profile="render_only",
        )


def test_warp_r2_binding_rejects_acceleration_class_identity_drift():
    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["boundary"] == "R2")
    binding = ExecutionReceipt.from_payload(
        _valid_result_record(gate, row, lane="warp")["receipt"]
    ).native_execution_binding

    assert binding.acceleration_class == "production_static"
    assert binding.acceleration_id == "cuda_morton_karras_lbvh_rebuild_v1"
    with pytest.raises(ValueError, match="acceleration identity does not match"):
        _rebuild_c3_binding(
            binding,
            [stage.evidence for stage in binding.ordered_stage_evidence],
            acceleration_class="production_dynamic",
        )


def test_result_rejects_c3_one_shot_binding_type():
    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["boundary"] == "R0")
    record = _valid_result_record(gate, row)
    with pytest.raises(ValueError, match="C3 one-shot binding"):
        validate_result_record(record, gate)


def test_frozen_gate_validates_and_contains_no_observed_results():
    gate = load_frozen_gate()

    assert gate["boundaries"] == BOUNDARY_VERSIONS
    assert gate["approved_proposal_commit"] == "46a5d33"
    assert gate["approved_dependency_order_commit"] == ("57073b8cabe39d7e4104d8cef77fce70fc17956c")
    assert gate["frozen_before_result_inspection"] is True
    assert "observed_results" not in gate
    assert "baseline_results" not in gate


def test_gate_rejects_case_drift():
    gate = load_frozen_gate()
    changed = copy.deepcopy(gate)
    changed["cases"]["go2_static_1920x1080_v1"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="case definitions and frozen scene/replay/fixture"):
        validate_frozen_gate(changed)


def test_gate_rejects_frozen_workload_content_drift():
    gate = load_frozen_gate()
    changed = copy.deepcopy(gate)
    changed["cases"]["synthetic_dynamic_rigid_960x640_v1"][
        "transform_replay_sha256"
    ] = "0" * 64

    with pytest.raises(ValueError, match="case definitions and frozen scene/replay/fixture"):
        validate_frozen_gate(changed)


def test_result_record_is_bound_to_gate_row_case_and_unprofiled_primary_mode():
    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r0_asymmetric_full_shadow")
    record = _valid_result_record(gate, row)

    with pytest.raises(ValueError, match="C3 one-shot binding"):
        validate_result_record(record, gate)
    changed = copy.deepcopy(record)
    changed["profile_phases_ms"] = {"raygen_kernel_ms": 0.1}
    with pytest.raises(ValueError, match="must not contain profile phase"):
        validate_result_record(changed, gate)
    changed = copy.deepcopy(record)
    changed["gate_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="does not match the frozen gate"):
        validate_result_record(changed, gate)
    changed = copy.deepcopy(record)
    _replace_receipt(changed, row_semantics_sha256="a" * 64)
    with pytest.raises(ValueError, match="row_semantics_sha256"):
        validate_result_record(changed, gate)

    changed = copy.deepcopy(record)
    changed["receipt"]["shadows"] = False
    with pytest.raises(ValueError, match="receipt_sha256"):
        validate_result_record(changed, gate)

    changed = copy.deepcopy(record)
    changed["shadows"] = True
    with pytest.raises(ValueError, match="coordinator result contains non-metadata fields"):
        validate_result_record(changed, gate)


@pytest.mark.parametrize(
    ("field", "wrong_value", "message"),
    [
        ("output_profile", "rgb_preview", "output_profile"),
        ("shadows", False, "shadows"),
        ("renderer_id", "warp_bvh_direct_light", "renderer_id"),
        ("acceleration_id", "cpu_bvh", "acceleration_id"),
        ("completion_endpoint", "kernel_launch", "completion_endpoint"),
    ],
)
def test_result_record_rejects_wrong_execution_semantics(field, wrong_value, message):
    gate = load_frozen_gate()
    record = _valid_r0_result(gate)
    _replace_receipt(record, **{field: wrong_value})

    with pytest.raises(ValueError, match=message):
        validate_result_record(record, gate)


def test_result_record_rejects_incomplete_native_readiness_and_bad_artifact_hash():
    gate = load_frozen_gate()
    record = _valid_r0_result(gate)
    readiness_payload = copy.deepcopy(record["receipt"]["native_readiness"])
    readiness_payload["warmup_frames_completed"] = gate["measurement"]["warmup_frames"] - 1
    readiness = NativeReadinessEvidence.from_payload(readiness_payload)
    _replace_receipt(
        record,
        native_readiness=readiness_payload,
        native_readiness_sha256=readiness.sha256,
    )
    with pytest.raises(ValueError, match="warmup"):
        validate_result_record(record, gate)

    record = _valid_r0_result(gate)
    record["artifact_hashes"] = {"native_input": "not-a-digest"}
    with pytest.raises(ValueError, match="artifact_hashes"):
        validate_result_record(record, gate)


def test_result_record_rejects_wrong_r2_payload_ring_or_reclaim_evidence():
    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r2_go2_rgb8_async_shadow")
    record = _valid_result_record(gate, row)

    with pytest.raises(ValueError, match="C3 one-shot binding"):
        validate_result_record(record, gate)
    changes = (
        ("payload", "rgb", "payload"),
        ("ring_depth", 1, "ring_depth"),
        ("observed_payload_bytes", row["expected_payload_bytes"] - 1, "observed_payload_bytes"),
    )
    for field, value, message in changes:
        changed = copy.deepcopy(record)
        _replace_receipt(changed, **{field: value})
        with pytest.raises(ValueError, match=message):
            validate_result_record(changed, gate)

    changed = copy.deepcopy(record)
    lifecycle = dict(changed["receipt"]["lifecycle"])
    lifecycle["final_reclaim_complete"] = False
    _replace_receipt(changed, lifecycle=lifecycle)
    with pytest.raises(ValueError, match="lifecycle"):
        validate_result_record(changed, gate)

    changed = copy.deepcopy(record)
    per_ray = copy.deepcopy(changed["receipt"]["observed_per_ray_channels"])
    per_ray[0]["dtype"] = "float32"
    _replace_receipt(changed, observed_per_ray_channels=per_ray)
    with pytest.raises(ValueError, match="observed_per_ray_channels"):
        validate_result_record(changed, gate)

    changed = copy.deepcopy(record)
    scalar = copy.deepcopy(changed["receipt"]["observed_scalar_channels"])
    scalar.pop()
    _replace_receipt(changed, observed_scalar_channels=scalar)
    with pytest.raises(ValueError, match="observed_scalar_channels"):
        validate_result_record(changed, gate)


def test_gate_covers_only_the_supported_r2_payload_policy_matrix():
    gate = load_frozen_gate()
    rows = [row for row in gate["mandatory_rows"] if row["boundary"] == "R2"]

    assert {(row["payload"], row["delivery_policy"]) for row in rows} == {
        ("none", "device_only"),
        ("rgb", "sync_host"),
        ("rgb", "torch_async_ordered"),
        ("full", "sync_host"),
        ("rgb8", "sync_host"),
        ("rgb8", "torch_async_ordered"),
    }
    assert not any(
        row["payload"] == "full" and row["delivery_policy"] == "torch_async_ordered" for row in rows
    )
    for row in rows:
        if row["sync_or_async"] == "async":
            assert row["final_drain_required"] is True
            assert row["transaction_terminal_endpoint"] == "final_output_reclaim_and_drain"
        else:
            assert row["final_drain_required"] is False


def test_gate_freezes_c0_lifecycle_and_valid_shipping_rows():
    gate = load_frozen_gate()
    c0 = gate["c0_blocking_acceptance"]
    shipping_rows = [row for row in gate["mandatory_rows"] if row["boundary"] == "S"]

    assert c0["duplicate_complete"] == "reject"
    assert c0["complete_without_active_borrow"] == "reject"
    assert c0["out_of_order_frame_completion"] == "reject"
    assert c0["close_with_active_lease"] == "reject"
    assert c0["destroyed_or_reused_stream_handle"].startswith("fail_closed")
    assert c0["event_record_failure"].startswith("poison_consumer")
    assert c0["cuda_fatal_failure"].startswith("poison_consumer")
    assert {(row["payload"], row["delivery_policy"]) for row in shipping_rows} == {
        ("none", "device_only"),
        ("rgb", "sync_host"),
        ("rgb", "torch_async_ordered"),
        ("full", "sync_host"),
        ("rgb8", "sync_host"),
        ("rgb8", "torch_async_ordered"),
    }
    assert all(row["ring_depth"] > 0 for row in shipping_rows)


def test_gate_rejects_invalid_lifecycle_or_absolute_row_binding():
    gate = load_frozen_gate()

    changed = copy.deepcopy(gate)
    changed["c0_blocking_acceptance"]["duplicate_complete"] = "ignore"
    with pytest.raises(ValueError, match="c0_blocking_acceptance"):
        validate_frozen_gate(changed)

    changed = copy.deepcopy(gate)
    changed["absolute_product_gate"]["throughput_row_id"] = "r0_asymmetric_full_shadow"
    with pytest.raises(ValueError, match="throughput_row_id"):
        validate_frozen_gate(changed)


def test_full_payload_contract_is_derived_from_formal_product_channel_specs():
    gate = load_frozen_gate()
    cases = {case.case_id: case for case in canonical_cases()}
    per_ray_names = [spec.name for spec in DIRECT_LIGHT_FULL_HOST_PER_RAY_CHANNEL_SPECS]
    scalar_names = [spec.name for spec in PRIMARY_HOST_SCALAR_CHANNEL_SPECS]

    for row in gate["mandatory_rows"]:
        if row["boundary"] not in {"R2", "S"} or row["payload"] != "full":
            continue
        case = cases[row["case_id"]]
        pixels = case.width * case.height
        per_ray_bytes = sum(
            spec.expected_nbytes(pixels) for spec in DIRECT_LIGHT_FULL_HOST_PER_RAY_CHANNEL_SPECS
        )
        scalar_bytes = sum(spec.expected_nbytes(pixels) for spec in PRIMARY_HOST_SCALAR_CHANNEL_SPECS)
        assert row["expected_per_ray_channels"] == per_ray_names
        assert row["expected_scalar_channels"] == scalar_names
        assert row["expected_per_ray_payload_bytes"] == per_ray_bytes
        assert row["expected_scalar_payload_bytes"] == scalar_bytes == 16
        assert row["expected_payload_bytes"] == per_ray_bytes + scalar_bytes

    assert "material_index" in per_ray_names
    assert "material_index" in OpticalOutputProfile.DIRECT_LIGHT_FULL.guaranteed_channels
    assert "material_index" in CudaDeviceBvhDirectLightOpticalExecutor.capabilities
    assert "material_index" in GpuDeviceBvhDirectLightOpticalExecutor.capabilities


def test_every_gate_payload_is_mechanically_derived_from_the_product_contract():
    gate = load_frozen_gate()

    assert gate["payload_channel_contracts"] == host_payload_contract_schema()
    assert set(gate["payload_channel_contracts"]) == set(HOST_PAYLOAD_CONTRACTS)
    assert gate["diagnostic_overlay_contracts"] == host_diagnostic_overlay_contract_schema()
    assert set(gate["diagnostic_overlay_contracts"]) == set(HOST_DIAGNOSTIC_OVERLAY_CONTRACTS)
    traversal_overlay = HOST_DIAGNOSTIC_OVERLAY_CONTRACTS["shadow_traversal_counters_v1"]
    assert traversal_overlay.scalar == SHADOW_TRAVERSAL_HOST_SCALAR_CHANNEL_SPECS
    assert len(traversal_overlay.scalar) == 9
    assert sum(spec.expected_nbytes(1) for spec in traversal_overlay.scalar) == 36


def test_gate_freezes_lane_receipt_provenance_before_runner_implementation():
    provenance = load_frozen_gate()["receipt_provenance"]

    assert provenance["receipt_producer"] == "backend_lane"
    assert provenance["producer_by_lane"] == {"warp": "warp_lane", "cuda": "cuda_lane"}
    assert provenance["coordinator_may_populate"] == [
        "run_id",
        "repeat_index",
        "sample_index",
        "timing",
    ]
    assert {"execution", "native_input", "observed_payload", "lifecycle"}.issubset(
        provenance["coordinator_must_not_populate"]
    )
    assert {"eligibility", "acceleration", "publication", "operation_trace"}.issubset(
        provenance["coordinator_must_not_populate"]
    )
    assert provenance["a2_a3_lane_test_requirements"]


def test_memory_contract_has_no_unfounded_numeric_budget():
    gate = load_frozen_gate()
    memory = gate["memory_measurement"]

    assert memory["acceptance"] == "report_only_no_numeric_threshold"
    assert memory["process_model"] == "clean_single_lane_shipping_process"
    assert memory["losing_lane_loaded"] is False
    assert memory["primary_metric"].endswith("_mib")
    assert memory["sampling_start"] == "before_physics_initialization"
    assert memory["sampling_end"] == "after_final_reclaim"
    assert not any("memory" in key.lower() for key in gate["absolute_product_gate"])
    serialized = json.dumps(gate, sort_keys=True)
    assert "max_process_device_memory" not in serialized


def test_shipping_rows_keep_ready_and_reclaim_endpoints_distinct():
    gate = load_frozen_gate()
    rows = [row for row in gate["mandatory_rows"] if row["boundary"] == "S"]

    assert rows
    for row in rows:
        endpoints = row["endpoints"]
        assert endpoints["consumer_completion_recorded_metric"]
        assert endpoints["slot_reclaimable_metric"]
        assert endpoints["host_ready_metric"] != endpoints["slot_reclaimable_metric"]
        if row["delivery_policy"] == "torch_async_ordered":
            assert endpoints["final_payload_drain_metric"]
            assert endpoints["final_reclaim_metric"]
            assert endpoints["final_reclaim_metric"] in row["primary_metrics"]


def test_result_rejects_replay_provenance_in_shipping_receipt():
    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "s_physics_rgb8_sync_shadow")
    record = _valid_result_record(gate, row)
    publication_payload = copy.deepcopy(record["receipt"]["shipping_publication"])
    publication_payload["replay_source_count"] = 1
    publication = ShippingPublicationReceipt.from_payload(publication_payload)
    receipt = copy.deepcopy(record["receipt"])
    receipt["shipping_publication"] = publication_payload
    receipt["shipping_publication_sha256"] = publication.sha256
    receipt["input_provenance_sha256"] = canonical_input_provenance_sha256(
        input_contract=receipt["input_contract"],
        case_id=receipt["case_id"],
        case_sha256=receipt["case_sha256"],
        native_input_manifest_sha256=receipt["native_input_manifest_sha256"],
        native_input_content_sha256=receipt["native_input_content_sha256"],
        topology_sha256=receipt["topology_sha256"],
        scene_contract_sha256=receipt["scene_contract_sha256"],
        transform_replay_sha256=receipt["transform_replay_sha256"],
        transform_frame_index=receipt["transform_frame_index"],
        fixed_render_fixture_sha256=receipt["fixed_render_fixture_sha256"],
        native_execution_binding_sha256=receipt["native_execution_binding_sha256"],
        asset_manifest_sha256=receipt["eligibility"]["asset_manifest_sha256"],
        shipping_publication_sha256=publication.sha256,
    )
    record["receipt"] = receipt
    record["receipt_sha256"] = canonical_execution_receipt_sha256(receipt)

    with pytest.raises(ValueError, match="forbidden replay/host operation"):
        validate_result_record(record, gate)


def test_result_rejects_native_replay_receipt_as_shipping_input_even_if_values_match():
    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "s_physics_device_only_no_shadow")
    record = _valid_result_record(gate, row)
    receipt = copy.deepcopy(record["receipt"])
    receipt["input_contract"] = "canonical_native_replay_v1"
    record["receipt"] = receipt
    record["receipt_sha256"] = canonical_execution_receipt_sha256(receipt)

    with pytest.raises(ValueError, match="input_contract"):
        validate_result_record(record, gate)


def test_go2_result_requires_the_exact_preflight_manifest_artifact():
    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r0_go2_960_render_only_shadow")
    record = _valid_result_record(gate, row)
    record["artifact_hashes"] = {}

    with pytest.raises(ValueError, match="asset manifest artifact"):
        validate_result_record(record, gate)


def test_inventory_and_cleanup_ledger_match_current_repository():
    inventory = _read_json(RESULT_DIR / "architecture_inventory.json")
    cleanup = _read_json(RESULT_DIR / "cleanup_ledger.json")

    validate_repository_inventory(REPO_ROOT, inventory, cleanup)
    assert inventory["observed_commit"] == "46a5d33"
    assert cleanup["must_complete"] is True
    assert all(entry["must_resolve_by"] == "P12.4e" for entry in cleanup["entries"])


def test_production_packages_do_not_import_benchmark_code():
    production_roots = (
        "optics",
        "physics",
        "rendering",
        "rl_env",
        "robot",
        "sensing",
    )
    paths = [path for root in production_roots for path in (REPO_ROOT / root).rglob("*.py")]
    paths.extend(path for path in (REPO_ROOT / "simulator.py", REPO_ROOT / "scene.py") if path.exists())

    offenders = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        if "import benchmarks" in source or "from benchmarks" in source:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _valid_r0_result(gate):
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r0_asymmetric_full_shadow")
    return _valid_result_record(gate, row)


def _valid_result_record(gate, row, *, lane="cuda"):
    binding = gate["lane_bindings"][lane]
    per_ray, scalar, observed_bytes = _observed_payload(row)
    audits = tuple(
        verified_runtime_entry_point_audit(
            operation=operation,
            entry_point=_TEST_ENTRY_POINTS[operation],
        )
        for operation in _required_operations_for_test(row)
    )
    eligibility = evaluate_lane_eligibility(
        row=row,
        lane=lane,
        entry_point_audits=audits,
        asset_manifest_sha256=("a" * 64 if row["required_asset_manifest"] else None),
    )
    operation_trace = RuntimeOperationTraceEvidence(
        schema_version=OPERATION_TRACE_SCHEMA_VERSION,
        row_id=row["row_id"],
        lane=lane,
        boundary=row["boundary"],
        records=tuple(
            RuntimeOperationRecord(
                sequence=index,
                operation=operation,
                entry_point=_TEST_ENTRY_POINTS[operation],
                capture_source="runtime_hook",
                runtime_hook_id=entry_point_hook_id(_TEST_ENTRY_POINTS[operation]),
            )
            for index, operation in enumerate(eligibility.required_operations)
        ),
    )
    acceleration_id = binding["acceleration_by_class"][row["acceleration_class"]]
    acceleration_contract = gate["acceleration_contracts"]["by_lane_and_identity"][lane][acceleration_id]
    classification = acceleration_contract["operation_classification"]
    scene_count, refit_count, rebuild_count = {
        "none": (0, 0, 0),
        "refit": (1, 1, 0),
        "rebuild": (1, 0, 1),
    }[classification]
    case_contract = gate["cases"][row["case_id"]]
    scene_contract_sha256 = case_contract["scene_contract_sha256"]
    fixed_fixture_sha256 = (
        case_contract["fixed_render_fixture_sha256"] if row["boundary"] == "R0" else None
    )
    replay_sha256 = (
        case_contract["transform_replay_sha256"]
        if row["boundary"] in {"R1", "R2"}
        else None
    )
    replay_index = 0 if replay_sha256 is not None else None
    if row["boundary"] == "R0":
        runtime_storage = (
            _storage(name="scene", lane=lane, token="cuda:0x1000", owner="prepared-scene"),
        )
    else:
        runtime_names = tuple(
            dict.fromkeys(
                DYNAMIC_SCENE_UPDATE_RUNTIME_INPUT_NAMES
                + DYNAMIC_RENDER_METADATA_INPUT_NAMES
                + DYNAMIC_RENDER_SHADE_INPUT_NAMES
            )
        )
        runtime_storage = tuple(
            _storage(
                name=name,
                lane=lane,
                token=f"cuda:0x1{index:03x}",
                owner="prepared-scene",
            )
            for index, name in enumerate(runtime_names)
        )
    fixture_storage = (
        (_storage(name="fixture", lane=lane, token="cuda:0x2000", owner="prepared-fixture"),)
        if fixed_fixture_sha256 is not None
        else ()
    )
    native_stages = _native_stages(
        row=row,
        lane=lane,
        scene_contract_sha256=scene_contract_sha256,
        replay_sha256=replay_sha256,
        replay_index=replay_index,
        fixed_fixture_sha256=fixed_fixture_sha256,
        topology_sha256="3" * 64,
        runtime_storage=runtime_storage,
        fixture_storage=fixture_storage,
        renderer_id=binding["renderer_id"],
        acceleration_contract=acceleration_contract,
    )
    acceleration = AccelerationReceipt(
        acceleration_schema_version=ACCELERATION_RECEIPT_SCHEMA_VERSION,
        scene_contract_sha256=scene_contract_sha256,
        acceleration_stage_evidence_sha256=(
            None
            if classification == "none"
            else next(stage.sha256 for stage in native_stages if stage.stage_name == "acceleration")
        ),
        acceleration_id=acceleration_id,
        workload_class=row["acceleration_class"],
        topology_origin=acceleration_contract["topology_origin"],
        topology_lifetime=acceleration_contract["topology_lifetime"],
        topology_validation_sha256="3" * 64,
        timed_scene_update_operation=acceleration_contract["timed_scene_update_operation"],
        timed_acceleration_operation=acceleration_contract["timed_acceleration_operation"],
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
    publication = _shipping_publication(row, operation_trace)
    readiness = NativeReadinessEvidence(
        schema_version=NATIVE_INPUT_SCHEMA_VERSION,
        backend=lane,
        device_ordinal=0,
        manifest_sha256="1" * 64,
        content_sha256="2" * 64,
        topology_sha256="3" * 64,
        input_contract=row["input_contract"],
        scene_contract_schema_version="q54.optical-scene-contract.v1",
        scene_contract_sha256=scene_contract_sha256,
        scene_id=next(case.scene_id for case in canonical_cases() if case.case_id == row["case_id"]),
        transform_replay_sha256=replay_sha256,
        transform_frame_count=120 if replay_sha256 is not None else 0,
        fixed_render_fixture_sha256=fixed_fixture_sha256,
        producer_ready_event_complete=True,
        direct_validation_complete=True,
        allocation_complete=True,
        compile_complete=True,
        warmup_frames_completed=10,
    )
    native_binding = build_c3_one_shot_native_execution_binding(
        row_id=row["row_id"],
        case_id=row["case_id"],
        case_sha256=gate["cases"][row["case_id"]]["sha256"],
        boundary=row["boundary"],
        lane=lane,
        renderer_id=gate["lane_bindings"][lane]["renderer_id"],
        acceleration_class=row["acceleration_class"],
        acceleration_id=gate["lane_bindings"][lane]["acceleration_by_class"][
            row["acceleration_class"]
        ],
        output_profile=row["output_profile"],
        shadows=row["shadows"],
        input_contract=row["input_contract"],
        native_input_manifest_sha256="1" * 64,
        native_readiness_sha256=readiness.sha256,
        scene_contract_sha256=scene_contract_sha256,
        transform_replay_sha256=replay_sha256,
        transform_frame_index=replay_index,
        fixed_render_fixture_sha256=fixed_fixture_sha256,
        runtime_scene_storage=runtime_storage,
        derived_fixture_storage=fixture_storage,
        stage_evidence=native_stages,
    )
    receipt = ExecutionReceipt(
        producer=f"{lane}_lane",
        row_semantics_sha256=canonical_row_semantics_sha256(row, gate),
        row_id=row["row_id"],
        case_id=row["case_id"],
        case_sha256=gate["cases"][row["case_id"]]["sha256"],
        native_input_manifest_sha256="1" * 64,
        native_input_content_sha256="2" * 64,
        topology_sha256="3" * 64,
        native_readiness=readiness,
        native_readiness_sha256=readiness.sha256,
        input_contract=row["input_contract"],
        input_provenance_sha256=canonical_input_provenance_sha256(
            input_contract=row["input_contract"],
            case_id=row["case_id"],
            case_sha256=gate["cases"][row["case_id"]]["sha256"],
            native_input_manifest_sha256="1" * 64,
            native_input_content_sha256="2" * 64,
            topology_sha256="3" * 64,
            scene_contract_sha256=scene_contract_sha256,
            transform_replay_sha256=replay_sha256,
            transform_frame_index=replay_index,
            fixed_render_fixture_sha256=fixed_fixture_sha256,
            native_execution_binding_sha256=native_binding.sha256,
            asset_manifest_sha256=eligibility.asset_manifest_sha256,
            shipping_publication_sha256=(None if publication is None else publication.sha256),
        ),
        scene_contract_sha256=scene_contract_sha256,
        transform_replay_sha256=replay_sha256,
        transform_frame_index=replay_index,
        fixed_render_fixture_sha256=fixed_fixture_sha256,
        native_execution_binding=native_binding,
        native_execution_binding_sha256=native_binding.sha256,
        eligibility=eligibility,
        eligibility_sha256=eligibility.sha256,
        operation_trace=operation_trace,
        operation_trace_sha256=operation_trace.sha256,
        acceleration=acceleration,
        acceleration_sha256=acceleration.sha256,
        shipping_publication=publication,
        shipping_publication_sha256=None if publication is None else publication.sha256,
        boundary=row["boundary"],
        boundary_version=gate["boundaries"][row["boundary"]],
        lane=lane,
        renderer_id=binding["renderer_id"],
        acceleration_id=acceleration_id,
        output_profile=row["output_profile"],
        shadows=row["shadows"],
        payload=row.get("payload", "none"),
        delivery_policy=row.get("delivery_policy", "native_only"),
        ring_depth=row.get("ring_depth", 0),
        completion_endpoint=row["completion_endpoint"],
        observed_per_ray_channels=tuple(ObservedChannel.from_payload(value) for value in per_ray),
        observed_scalar_channels=tuple(ObservedChannel.from_payload(value) for value in scalar),
        observed_payload_bytes=observed_bytes,
        lifecycle=tuple((name, True) for name in row["required_lifecycle"]),
        device_uuid="GPU-unit-test",
        counters={},
        memory={},
    )
    receipt_payload = receipt.to_payload()
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "gate_sha256": canonical_gate_sha256(gate),
        "run_id": "unit-test",
        "repeat_index": 0,
        "sample_index": 0,
        "profile_mode": "primary_unprofiled",
        "metrics_ms": {metric: 1.25 for metric in row["primary_metrics"]},
        "artifact_hashes": (
            {"asset_manifest": eligibility.asset_manifest_sha256}
            if eligibility.asset_manifest_sha256 is not None
            else {}
        ),
        "receipt": receipt_payload,
        "receipt_sha256": receipt.sha256,
    }


def _required_operations_for_test(row):
    boundary = row["boundary"]
    if boundary == "R0":
        return ("render",)
    if boundary == "R1":
        return ("scene_update", "acceleration", "render", "reclaim")
    if boundary == "R2":
        operations = ["scene_update", "acceleration", "render"]
        if row["payload"] == "rgb8":
            operations.append("pack")
        return (*operations, "delivery", "reclaim")
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
    if row["payload"] == "rgb8":
        operations.append("pack")
    return (*operations, "delivery", "reclaim")


def _storage(*, name: str, lane: str, token: str, owner: str) -> NativeStorageSignature:
    return NativeStorageSignature(
        name=name,
        backend=lane,
        device_ordinal=0,
        storage_token=token,
        dtype="<f4",
        shape=(1,),
        nbytes=4,
        active_start=0,
        active_count=4,
        owner_token=owner,
    )


def _native_stages(
    *,
    row,
    lane,
    scene_contract_sha256,
    replay_sha256,
    replay_index,
    fixed_fixture_sha256,
    topology_sha256,
    runtime_storage,
    fixture_storage,
    renderer_id,
    acceleration_contract,
):
    scene_operation_id = acceleration_contract["timed_scene_update_operation"]
    acceleration_operation_id = acceleration_contract["timed_acceleration_operation"]
    scene_outputs = tuple(
        _storage(
            name=name,
            lane=lane,
            token=f"cuda:0x3{index:03x}",
            owner="scene-slot",
        )
        for index, name in enumerate(DYNAMIC_SCENE_OUTPUT_NAMES)
    )
    acceleration_outputs = tuple(
        _storage(
            name=name,
            lane=lane,
            token=f"cuda:0x4{index:03x}",
            owner="acceleration-slot",
        )
        for index, name in enumerate(DYNAMIC_ACCELERATION_OUTPUT_NAMES)
    )
    output = _storage(name="render", lane=lane, token="cuda:0x5000", owner="slot")
    if row["boundary"] == "R0":
        specs = (
            (
                "render",
                renderer_id,
                runtime_storage + fixture_storage,
                (output,),
            ),
        )
    else:
        transform_storage = (
            _storage(
                name="replay_x_world_R",
                lane=lane,
                token="cuda:0x2100",
                owner="prepared-replay",
            ),
            _storage(
                name="replay_x_world_r",
                lane=lane,
                token="cuda:0x2200",
                owner="prepared-replay",
            ),
        )
        runtime_by_name = {value.name: value for value in runtime_storage}
        scene_by_name = {value.name: value for value in scene_outputs}
        acceleration_by_name = {value.name: value for value in acceleration_outputs}
        scene_input = tuple(
            runtime_by_name[name] for name in DYNAMIC_SCENE_UPDATE_RUNTIME_INPUT_NAMES
        ) + transform_storage
        acceleration_input = tuple(
            scene_by_name[name] for name in DYNAMIC_ACCELERATION_INPUT_NAMES
        )
        render_names = (
            DYNAMIC_RENDER_WORLD_INPUT_NAMES
            + DYNAMIC_RENDER_METADATA_INPUT_NAMES
            + DYNAMIC_ACCELERATION_OUTPUT_NAMES
            + DYNAMIC_RENDER_SHADE_INPUT_NAMES
        )
        all_by_name = runtime_by_name | scene_by_name | acceleration_by_name
        render_input = tuple(all_by_name[name] for name in render_names)
        specs = (
            (
                "scene_update",
                scene_operation_id,
                scene_input,
                scene_outputs,
            ),
            (
                "acceleration",
                acceleration_operation_id,
                acceleration_input,
                acceleration_outputs,
            ),
            ("render", renderer_id, render_input, (output,)),
        )
    return tuple(
        NativeStageEvidence(
            schema_version=NATIVE_STAGE_EVIDENCE_SCHEMA_VERSION,
            stage_name=name,
            operation_id=operation_id,
            output_profile=row["output_profile"] if name == "render" else None,
            shadows=row["shadows"] if name == "render" else None,
            scene_contract_sha256=scene_contract_sha256,
            input_contract=row["input_contract"],
            transform_replay_sha256=replay_sha256,
            transform_frame_index=replay_index,
            fixed_render_fixture_sha256=fixed_fixture_sha256,
            topology_sha256=topology_sha256,
            input_storage=inputs,
            output_storage=outputs,
            dispatch_plan_sha256=hex(index + 10)[2:] * 64,
        )
        for index, (name, operation_id, inputs, outputs) in enumerate(specs)
    )


def _rebuild_c3_binding(
    binding,
    stages,
    *,
    output_profile=None,
    acceleration_class=None,
):
    return build_c3_one_shot_native_execution_binding(
        row_id=binding.row_id,
        case_id=binding.case_id,
        case_sha256=binding.case_sha256,
        boundary=binding.boundary,
        lane=binding.lane,
        renderer_id=binding.renderer_id,
        acceleration_class=(
            binding.acceleration_class
            if acceleration_class is None
            else acceleration_class
        ),
        acceleration_id=binding.acceleration_id,
        output_profile=binding.output_profile if output_profile is None else output_profile,
        shadows=binding.shadows,
        input_contract=binding.input_contract,
        native_input_manifest_sha256=binding.native_input_manifest_sha256,
        native_readiness_sha256=binding.native_readiness_sha256,
        scene_contract_sha256=binding.scene_contract_sha256,
        transform_replay_sha256=binding.transform_replay_sha256,
        transform_frame_index=binding.transform_frame_index,
        fixed_render_fixture_sha256=binding.fixed_render_fixture_sha256,
        runtime_scene_storage=binding.runtime_scene_storage,
        derived_fixture_storage=binding.derived_fixture_storage,
        stage_evidence=stages,
    )


def _shipping_publication(row, operation_trace):
    if row["boundary"] != "S":
        return None
    lifecycle = set(row["required_lifecycle"])
    return ShippingPublicationReceipt(
        publication_schema_version=SHIPPING_PUBLICATION_RECEIPT_SCHEMA_VERSION,
        input_contract="gpu_engine_published_frame_v1",
        physics_run_id="unit-physics-run",
        physics_step_index=0,
        physics_frame_id=0,
        physics_slot_id=0,
        physics_device_ordinal=0,
        physics_step_count=1,
        publish_event_token="publish-0",
        publish_event_recorded=True,
        producer_publish_completion_recorded=True,
        consumer_binding_id="binding-0",
        consumer_stream_token="stream-0",
        borrow_count=1,
        consumer_stream_wait_enqueued=True,
        zero_copy_transform_binding=True,
        last_physics_buffer_reader_operation="scene_update",
        consumer_done_event_token="done-0",
        consumer_completion_recorded=True,
        slot_logically_reclaimable=True,
        producer_reuse_wait_enqueued_when_reused=True,
        done_event_retained_until_reuse_dependency=True,
        physical_reclaim_evidence="producer_wait_enqueued_before_reuse",
        final_payload_drain_complete="final_drain_complete" in lifecycle,
        final_reclaim_complete="final_reclaim_complete" in lifecycle,
        replay_source_count=0,
        host_poll_count=0,
        host_synchronize_count=0,
        publication_trace_sha256=operation_trace.sha256,
    )


def _observed_payload(row):
    if row["boundary"] not in {"R2", "S"}:
        return [], [], 0
    case = next(case for case in canonical_cases() if case.case_id == row["case_id"])
    pixels = case.width * case.height
    return HOST_PAYLOAD_CONTRACTS[row["payload_contract_id"]].expected_observation(pixels)


def _replace_receipt(record, **changes):
    receipt = copy.deepcopy(record["receipt"])
    receipt.update(changes)
    record["receipt"] = receipt
    record["receipt_sha256"] = canonical_execution_receipt_sha256(receipt)
