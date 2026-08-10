from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from benchmarks.optical import (
    BOUNDARY_VERSIONS,
    ExecutionReceipt,
    ObservedChannel,
    canonical_cases,
    canonical_execution_receipt_sha256,
    canonical_gate_sha256,
    canonical_row_semantics_sha256,
    case_hashes,
    load_frozen_gate,
    validate_frozen_gate,
    validate_repository_inventory,
    validate_result_record,
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


def test_canonical_case_ids_and_hashes_are_frozen():
    cases = canonical_cases()

    assert len(cases) == len({case.case_id for case in cases}) == 5
    assert case_hashes() == EXPECTED_CASE_HASHES
    for case in cases:
        assert case.width > 0
        assert case.height > 0
        assert case.render_contract["ray_layout"] == "row_major_py_times_width_plus_px"
        assert case.transform_sequence["frames"] > 0


def test_frozen_gate_validates_and_contains_no_observed_results():
    gate = load_frozen_gate()

    assert gate["boundaries"] == BOUNDARY_VERSIONS
    assert gate["approved_proposal_commit"] == "46a5d33"
    assert gate["frozen_before_result_inspection"] is True
    assert "observed_results" not in gate
    assert "baseline_results" not in gate


def test_gate_rejects_case_drift():
    gate = load_frozen_gate()
    changed = copy.deepcopy(gate)
    changed["cases"]["go2_static_1920x1080_v1"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="case ids/hashes"):
        validate_frozen_gate(changed)


def test_result_record_is_bound_to_gate_row_case_and_unprofiled_primary_mode():
    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r0_asymmetric_full_shadow")
    record = _valid_result_record(gate, row)

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


def test_result_record_rejects_wrong_r2_payload_ring_or_reclaim_evidence():
    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == "r2_go2_rgb8_async_shadow")
    record = _valid_result_record(gate, row)

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
    assert {"execution", "observed_payload", "lifecycle"}.issubset(
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


def _valid_result_record(gate, row):
    lane = "cuda"
    binding = gate["lane_bindings"][lane]
    per_ray, scalar, observed_bytes = _observed_payload(row)
    receipt = ExecutionReceipt(
        producer="cuda_lane",
        row_semantics_sha256=canonical_row_semantics_sha256(row, gate),
        row_id=row["row_id"],
        case_id=row["case_id"],
        case_sha256=gate["cases"][row["case_id"]]["sha256"],
        boundary=row["boundary"],
        boundary_version=gate["boundaries"][row["boundary"]],
        lane=lane,
        renderer_id=binding["renderer_id"],
        acceleration_id=binding["acceleration_by_class"][row["acceleration_class"]],
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
        "schema_version": "q54.p12.4.result.v3",
        "gate_sha256": canonical_gate_sha256(gate),
        "run_id": "unit-test",
        "repeat_index": 0,
        "sample_index": 0,
        "profile_mode": "primary_unprofiled",
        "metrics_ms": {metric: 1.25 for metric in row["primary_metrics"]},
        "artifact_hashes": {},
        "receipt": receipt_payload,
        "receipt_sha256": receipt.sha256,
    }


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
