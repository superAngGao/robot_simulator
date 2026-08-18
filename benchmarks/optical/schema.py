"""Versioned schema and validation for the Q54 P12.4 benchmark gate."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from optics.output_contract import (
    HOST_PAYLOAD_CONTRACTS,
    host_diagnostic_overlay_contract_schema,
    host_payload_contract_schema,
)
from optics.scene_contract import SCENE_CONTRACT_SCHEMA_VERSION

from .assets import ASSET_MANIFEST_SCHEMA_VERSION, GO2_ASSET_REQUIREMENT
from .cases import CASE_SCHEMA_VERSION, case_hashes, cases_by_id
from .eligibility import (
    CANONICAL_NATIVE_REPLAY_INPUT,
    ELIGIBILITY_SCHEMA_VERSION,
    EXACT_ACCELERATION_CONTRACTS,
    EXACT_LANE_BINDINGS,
    FIXED_NATIVE_SCENE_INPUT,
    GPU_ENGINE_PUBLISHED_FRAME_INPUT,
    INPUT_CONTRACT_BY_BOUNDARY,
    INPUT_CONTRACT_SCHEMA_VERSION,
    LaneEligibilityEvidence,
    evaluate_lane_eligibility,
    required_runtime_operations,
    validate_trace_entry_points,
)
from .instrumentation import RUNTIME_HOOK_REGISTRY_VERSION, runtime_hook_registry_sha256
from .native import (
    FIXED_RENDER_FIXTURE_SCHEMA_VERSION,
    NATIVE_INPUT_SCHEMA_VERSION,
    TRANSFORM_REPLAY_SCHEMA_VERSION,
    NativeReadinessEvidence,
)
from .receipts import (
    ACCELERATION_RECEIPT_SCHEMA_VERSION,
    NATIVE_EXECUTION_BINDING_SCHEMA_VERSION,
    NATIVE_STAGE_EVIDENCE_SCHEMA_VERSION,
    SHIPPING_PUBLICATION_RECEIPT_SCHEMA_VERSION,
    AccelerationReceipt,
    NativeExecutionBindingEvidence,
    ShippingPublicationReceipt,
)
from .timing import (
    FORBIDDEN_TIMED_OPERATIONS,
    NATIVE_TIMING_CONTRACT_VERSION,
    OPERATION_TRACE_SCHEMA_VERSION,
    RuntimeOperationTraceEvidence,
    validate_required_operation_order,
)

GATE_SCHEMA_VERSION = "q54.p12.4.gate.v7"
RESULT_SCHEMA_VERSION = "q54.p12.4.result.v7"
ROW_SEMANTICS_SCHEMA_VERSION = "q54.p12.4.row-semantics.v4"
INVENTORY_SCHEMA_VERSION = "q54.p12.4.architecture-inventory.v1"
CLEANUP_SCHEMA_VERSION = "q54.p12.4.cleanup-ledger.v1"
PUBLICATION_BINDING_CONTRACT_VERSION = "q54.p12.4.publication-cuda-binding.v1"

BOUNDARY_VERSIONS = {
    "R0": "q54.p12.4.r0-native-renderer.v1",
    "R1": "q54.p12.4.r1-native-transaction.v1",
    "R2": "q54.p12.4.r2-payload-transaction.v1",
    "D": "q54.p12.4.delivery-diagnostic.v1",
    "I": "q54.p12.4.physics-handoff-diagnostic.v1",
    "S": "q54.p12.4.physics-to-image-shipping.v1",
}

ALLOWED_R2_PAYLOAD_POLICIES = frozenset(
    {
        ("none", "device_only"),
        ("rgb", "sync_host"),
        ("rgb", "torch_async_ordered"),
        ("full", "sync_host"),
        ("rgb8", "sync_host"),
        ("rgb8", "torch_async_ordered"),
    }
)

ALLOWED_SHIPPING_PAYLOAD_POLICIES = ALLOWED_R2_PAYLOAD_POLICIES

EXPECTED_PAYLOAD_CHANNEL_CONTRACTS = host_payload_contract_schema()
EXPECTED_DIAGNOSTIC_OVERLAY_CONTRACTS = host_diagnostic_overlay_contract_schema()

EXPECTED_MEMORY_MEASUREMENT = {
    "acceptance": "report_only_no_numeric_threshold",
    "process_model": "clean_single_lane_shipping_process",
    "included_components": [
        "physics_engine",
        "one_candidate_optical_backend",
        "candidate_selected_acceleration",
        "delivery_and_readback_rings",
        "allocator_reserved_device_memory",
    ],
    "losing_lane_loaded": False,
    "primary_metric": "peak_nvml_process_device_memory_mib",
    "sampling_start": "before_physics_initialization",
    "sampling_end": "after_final_reclaim",
    "diagnostics": [
        "torch_allocated_mib",
        "torch_reserved_mib",
        "warp_allocator_or_pool_mib_when_available",
    ],
    "forbidden": [
        "baseline_subtraction",
        "empty_cache_before_peak_read",
        "same_process_dual_lane_residency",
    ],
    "mandatory_workload_oom": "hard_failure",
}

EXPECTED_LANE_BINDINGS = EXACT_LANE_BINDINGS

EXPECTED_INPUT_CONTRACTS = {
    "schema_version": INPUT_CONTRACT_SCHEMA_VERSION,
    "by_boundary": INPUT_CONTRACT_BY_BOUNDARY,
    "contracts": {
        FIXED_NATIVE_SCENE_INPUT: {
            "timed_start": "before_native_renderer_submission",
            "measured_transform_owner": "prepared_candidate_native_world_scene",
            "replay_source_allowed": False,
            "physics_publication_required": False,
        },
        CANONICAL_NATIVE_REPLAY_INPUT: {
            "timed_start": "before_candidate_native_scene_update",
            "measured_transform_owner": "candidate_native_replay_arena",
            "replay_source_allowed": True,
            "physics_publication_required": False,
        },
        GPU_ENGINE_PUBLISHED_FRAME_INPUT: {
            "timed_start": "immediately_before_real_gpu_engine_step",
            "measured_transform_owner": "physics_publish_ring_slot",
            "replay_source_allowed": False,
            "physics_publication_required": True,
        },
    },
}

EXPECTED_ACCELERATION_CONTRACTS = {
    "schema_version": ACCELERATION_RECEIPT_SCHEMA_VERSION,
    "by_lane_and_identity": EXACT_ACCELERATION_CONTRACTS,
    "forbidden_primary_counts": {
        "device_allocation_count": 0,
        "host_materialization_count": 0,
        "host_upload_bytes": 0,
        "device_to_host_bytes": 0,
        "global_synchronize_count": 0,
        "bridge_creation_count": 0,
    },
}

EXPECTED_ELIGIBILITY_CONTRACT = {
    "schema_version": ELIGIBILITY_SCHEMA_VERSION,
    "operation_trace_schema_version": OPERATION_TRACE_SCHEMA_VERSION,
    "runtime_hook_registry_version": RUNTIME_HOOK_REGISTRY_VERSION,
    "runtime_hook_registry_sha256": runtime_hook_registry_sha256(),
    "preflight_position": "before_timed_submission",
    "runtime_hook_installation_owner": "single_lane_runner",
    "high_level_trace_capture": "installed_entry_point_wrapper_only",
    "low_level_forbidden_capture": "mandatory_registry_wrapper_fail_fast",
    "source_manifest_grants_instrumented": False,
    "ineligible_result_emission": "forbidden",
    "required_runtime_operations": {
        "R0": ["render"],
        "R1": ["scene_update", "acceleration", "render", "reclaim"],
        "R2": "derived_from_payload_and_delivery_policy",
        "S": "derived_from_publication_payload_and_delivery_policy",
    },
    "a3a_baseline_emission": "forbidden",
    "first_native_baseline_authorization": "P12.4a4",
}

EXPECTED_PHASE_AUTHORIZATION = {
    "P12.4a3a": {
        "executable_boundaries": [],
        "result_emission": "forbidden",
        "result_inspection": "forbidden",
    },
    "P12.4a3b": {
        "executable_boundaries": ["R0", "R1", "R2"],
        "result_emission": "sealed_native_artifacts_only",
        "result_inspection": "forbidden_until_P12.4a4",
    },
    "P12.4a4": {
        "executable_boundaries": [],
        "result_emission": "forbidden",
        "result_inspection": "native_common_subset_only",
    },
    "P12.4b4/d": {
        "executable_boundaries": ["S"],
        "result_emission": "shipping_artifacts",
        "result_inspection": "shipping_gate_review",
    },
}

EXPECTED_ASSET_MANIFEST_CONTRACT = {
    "schema_version": ASSET_MANIFEST_SCHEMA_VERSION,
    "go2_requirement_id": GO2_ASSET_REQUIREMENT,
    "default_model_dir": "out/external/mujoco_menagerie/unitree_go2",
    "default_manifest_name": "q54_asset_manifest.json",
    "absent_or_mismatched": "hard_failure_before_timing",
    "surrogate_allowed": False,
    "coverage": "every_and_only_regular_model_files_excluding_manifest",
    "source_checkout": "canonical_mujoco_menagerie_git_origin",
    "source_revision": "manifest_revision_equals_checkout_HEAD",
    "tracked_file_rule": "manifest_files_equal_clean_commit_tracked_unitree_go2_files",
}

EXPECTED_SHIPPING_PUBLICATION_RECEIPT_CONTRACT = {
    "schema_version": SHIPPING_PUBLICATION_RECEIPT_SCHEMA_VERSION,
    "required_input_contract": GPU_ENGINE_PUBLISHED_FRAME_INPUT,
    "replay_source_count": 0,
    "host_poll_count": 0,
    "host_synchronize_count": 0,
    "logical_reclaim_distinct_from_physical_safety": True,
    "physical_safety_evidence": [
        "producer_wait_enqueued_before_reuse",
        "done_event_retained_no_reuse",
    ],
}

EXPECTED_FIXED_R0_TOPOLOGY_CONTRACT = {
    "fixture_id": "cpu_median_longest_axis_leaf4_source_order_v1",
    "primitive_order": "canonical_source_order",
    "lane_rule": ("both lanes receive independently allocated copies of the same normalized topology arrays"),
    "run_manifest_requirement": "record and compare topology_sha256 before timing",
}

EXPECTED_C0_BLOCKING_ACCEPTANCE = {
    "contract_version": PUBLICATION_BINDING_CONTRACT_VERSION,
    "duplicate_complete": "reject",
    "complete_without_active_borrow": "reject",
    "out_of_order_frame_completion": "reject",
    "close_with_active_lease": "reject",
    "destroyed_or_reused_stream_handle": "fail_closed_without_logical_completion",
    "ordinary_exception": ("record_done_after_successfully_enqueued_readers_then_advance_logical_completion"),
    "event_record_failure": "poison_consumer_and_slot_then_abort",
    "cuda_fatal_failure": "poison_consumer_and_slot_then_abort",
    "logical_reclaim_semantics": ("never_report_gpu_event_as_physically_complete_without_observation"),
    "forbidden": [
        "host_polling",
        "host_synchronize",
        "private_ring_state_mutation",
        "torch_event_in_ring",
        "warp_optical_adapter",
    ],
}

EXPECTED_RECEIPT_PROVENANCE = {
    "receipt_producer": "backend_lane",
    "producer_by_lane": {"warp": "warp_lane", "cuda": "cuda_lane"},
    "coordinator_may_populate": [
        "run_id",
        "repeat_index",
        "sample_index",
        "timing",
    ],
    "coordinator_must_not_populate": [
        "execution",
        "eligibility",
        "native_input",
        "native_execution_binding",
        "stage_evidence",
        "acceleration",
        "publication",
        "operation_trace",
        "observed_payload",
        "lifecycle",
        "counters",
        "memory",
    ],
    "receipt_hash": "canonical_sha256_of_execution_receipt_content",
    "a2_a3_lane_test_requirements": [
        "wrong_actual_request_is_visible_in_receipt",
        "wrong_actual_ring_depth_is_visible_in_receipt",
        "missing_actual_endpoint_is_visible_in_receipt",
        "gate_expectations_are_not_copied_as_observations",
    ],
}

EXPECTED_NATIVE_INPUT_CONTRACT = {
    "schema_version": NATIVE_INPUT_SCHEMA_VERSION,
    "timing_contract_version": NATIVE_TIMING_CONTRACT_VERSION,
    "scope": "R0_R1_R2_and_S_static_optical_inputs_only",
    "logical_source": "PackedOpticalSceneHost_single_materialization_source",
    "scene_contract_schema_version": SCENE_CONTRACT_SCHEMA_VERSION,
    "transform_replay_schema_version": TRANSFORM_REPLAY_SCHEMA_VERSION,
    "fixed_render_fixture_schema_version": FIXED_RENDER_FIXTURE_SCHEMA_VERSION,
    "result_capable_constructor": "CanonicalNativeInput.from_scene_contract",
    "generic_from_arrays_result_capable": False,
    "S_measured_physics_transforms_excluded": True,
    "warp_preparation": "direct_host_to_warp_cuda_allocation",
    "cuda_preparation": "direct_host_to_torch_cuda_allocation",
    "allocation_rule": "independent_runtime_owned_allocations",
    "R0_array_ownership": {
        "packed_scene": "local_geometry_metadata_material_light_role_source",
        "fixed_fixture": "derived_world_geometry_and_normalized_bvh_only",
        "name_and_semantic_sets": "strictly_disjoint",
        "upload_count_per_logical_array_per_candidate": 1,
        "packed_owner_rule": "exact_object_identity_not_content_equality",
    },
    "native_ready_requirements": [
        "device_resident",
        "dtype_shape_layout_semantics_validated",
        "producer_ready_event_complete",
        "compile_warmup_and_allocation_complete",
    ],
    "timed_forbidden_operations": sorted(FORBIDDEN_TIMED_OPERATIONS),
    "receipt_fields": [
        "native_input_manifest_sha256",
        "native_input_content_sha256",
        "topology_sha256",
        "native_readiness",
        "native_readiness_sha256",
        "input_contract",
        "input_provenance_sha256",
        "scene_contract_sha256",
        "transform_replay_sha256",
        "transform_frame_index",
        "fixed_render_fixture_sha256",
        "native_execution_binding",
        "native_execution_binding_sha256",
        "eligibility",
        "eligibility_sha256",
    ],
}

EXPECTED_NATIVE_EXECUTION_BINDING_CONTRACT = {
    "schema_version": NATIVE_EXECUTION_BINDING_SCHEMA_VERSION,
    "stage_schema_version": NATIVE_STAGE_EVIDENCE_SCHEMA_VERSION,
    "construction_order": [
        "source_and_prepared_storage",
        "independent_stage_evidence",
        "native_execution_binding",
        "execution_receipt",
    ],
    "stage_payload_contains_own_sha256": False,
    "stage_payload_references_outer_binding": False,
    "binding_references_stage_sha256": True,
    "acceleration_receipt_references_acceleration_stage_sha256": True,
    "pre_enqueue_validation": [
        "row_case_boundary_lane_and_input_contract",
        "native_manifest_and_readiness",
        "scene_replay_or_fixed_fixture_provenance",
        "exact_prepared_scene_storage",
        "exact_prepared_derived_fixture_storage",
        "ordered_stage_evidence_and_dataflow_sha256",
    ],
    "a3b_result_capable_before_C4_B3_review": False,
}

REQUIRED_SHIPPING_ENDPOINT_FIELDS = frozenset(
    {
        "device_ready_metric",
        "host_ready_metric",
        "consumer_completion_recorded_metric",
        "slot_reclaimable_metric",
        "final_payload_drain_metric",
        "final_reclaim_metric",
    }
)

RESULT_RECORD_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "gate_sha256",
        "run_id",
        "repeat_index",
        "sample_index",
        "profile_mode",
        "metrics_ms",
        "artifact_hashes",
        "receipt",
        "receipt_sha256",
    }
)

RESULT_RECORD_ALLOWED_FIELDS = RESULT_RECORD_REQUIRED_FIELDS | frozenset({"profile_phases_ms"})

EXECUTION_RECEIPT_REQUIRED_FIELDS = frozenset(
    {
        "producer",
        "row_semantics_sha256",
        "row_id",
        "case_id",
        "case_sha256",
        "native_input_manifest_sha256",
        "native_input_content_sha256",
        "topology_sha256",
        "native_readiness",
        "native_readiness_sha256",
        "input_contract",
        "input_provenance_sha256",
        "scene_contract_sha256",
        "transform_replay_sha256",
        "transform_frame_index",
        "fixed_render_fixture_sha256",
        "native_execution_binding",
        "native_execution_binding_sha256",
        "eligibility",
        "eligibility_sha256",
        "operation_trace",
        "operation_trace_sha256",
        "acceleration",
        "acceleration_sha256",
        "shipping_publication",
        "shipping_publication_sha256",
        "boundary",
        "boundary_version",
        "lane",
        "renderer_id",
        "acceleration_id",
        "output_profile",
        "shadows",
        "payload",
        "delivery_policy",
        "ring_depth",
        "completion_endpoint",
        "observed_per_ray_channels",
        "observed_scalar_channels",
        "observed_payload_bytes",
        "lifecycle",
        "device_uuid",
        "counters",
        "memory",
    }
)


@dataclass(frozen=True)
class ObservedChannel:
    """One channel observation measured by a backend lane."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int

    @classmethod
    def from_payload(cls, payload: object) -> "ObservedChannel":
        if not isinstance(payload, Mapping):
            raise ValueError("receipt channel observation must be an object")
        if set(payload) != {"name", "dtype", "shape", "nbytes"}:
            raise ValueError("receipt channel observation fields are invalid")
        shape = payload.get("shape")
        if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
            raise ValueError("receipt channel shape must be an array")
        dimensions = tuple(shape)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in dimensions):
            raise ValueError("receipt channel shape dimensions must be non-negative integers")
        nbytes = payload.get("nbytes")
        if not isinstance(nbytes, int) or isinstance(nbytes, bool) or nbytes < 0:
            raise ValueError("receipt channel nbytes must be a non-negative integer")
        name = payload.get("name")
        dtype = payload.get("dtype")
        if not isinstance(name, str) or not isinstance(dtype, str):
            raise ValueError("receipt channel name and dtype must be strings")
        return cls(name=name, dtype=dtype, shape=dimensions, nbytes=nbytes)

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "nbytes": self.nbytes,
        }


@dataclass(frozen=True)
class ExecutionReceipt:
    """Lane-produced evidence of the request and lifecycle actually executed.

    The paired coordinator may carry this immutable payload and attach sample
    identity or timing, but must not synthesize its execution, payload, or
    lifecycle fields from the gate row.
    """

    producer: str
    row_semantics_sha256: str
    row_id: str
    case_id: str
    case_sha256: str
    native_input_manifest_sha256: str
    native_input_content_sha256: str
    topology_sha256: str
    native_readiness: NativeReadinessEvidence
    native_readiness_sha256: str
    input_contract: str
    input_provenance_sha256: str
    scene_contract_sha256: str
    transform_replay_sha256: str | None
    transform_frame_index: int | None
    fixed_render_fixture_sha256: str | None
    native_execution_binding: NativeExecutionBindingEvidence
    native_execution_binding_sha256: str
    eligibility: LaneEligibilityEvidence
    eligibility_sha256: str
    operation_trace: RuntimeOperationTraceEvidence
    operation_trace_sha256: str
    acceleration: AccelerationReceipt
    acceleration_sha256: str
    shipping_publication: ShippingPublicationReceipt | None
    shipping_publication_sha256: str | None
    boundary: str
    boundary_version: str
    lane: str
    renderer_id: str
    acceleration_id: str
    output_profile: str
    shadows: bool
    payload: str
    delivery_policy: str
    ring_depth: int
    completion_endpoint: str
    observed_per_ray_channels: tuple[ObservedChannel, ...]
    observed_scalar_channels: tuple[ObservedChannel, ...]
    observed_payload_bytes: int
    lifecycle: tuple[tuple[str, bool], ...]
    device_uuid: str
    counters: Mapping[str, object]
    memory: Mapping[str, object]

    @classmethod
    def from_payload(cls, payload: object) -> "ExecutionReceipt":
        if not isinstance(payload, Mapping):
            raise ValueError("execution receipt must be an object")
        missing = EXECUTION_RECEIPT_REQUIRED_FIELDS.difference(payload)
        extra = set(payload).difference(EXECUTION_RECEIPT_REQUIRED_FIELDS)
        if missing or extra:
            raise ValueError(
                f"execution receipt fields are invalid; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        lifecycle = payload.get("lifecycle")
        if not isinstance(lifecycle, Mapping) or any(
            not isinstance(name, str) or not isinstance(value, bool) for name, value in lifecycle.items()
        ):
            raise ValueError("execution receipt lifecycle must be a boolean object")
        per_ray = payload.get("observed_per_ray_channels")
        scalar = payload.get("observed_scalar_channels")
        if not isinstance(per_ray, Sequence) or isinstance(per_ray, (str, bytes)):
            raise ValueError("execution receipt per-ray observations must be an array")
        if not isinstance(scalar, Sequence) or isinstance(scalar, (str, bytes)):
            raise ValueError("execution receipt scalar observations must be an array")
        ring_depth = payload.get("ring_depth")
        observed_bytes = payload.get("observed_payload_bytes")
        for name, value in (("ring_depth", ring_depth), ("observed_payload_bytes", observed_bytes)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"execution receipt {name} must be a non-negative integer")
        if not isinstance(payload.get("shadows"), bool):
            raise ValueError("execution receipt shadows must be a boolean")
        counters = payload.get("counters")
        memory = payload.get("memory")
        if not isinstance(counters, Mapping) or not isinstance(memory, Mapping):
            raise ValueError("execution receipt counters and memory must be objects")
        string_fields = (
            "producer",
            "row_semantics_sha256",
            "row_id",
            "case_id",
            "case_sha256",
            "native_input_manifest_sha256",
            "native_input_content_sha256",
            "topology_sha256",
            "input_contract",
            "input_provenance_sha256",
            "scene_contract_sha256",
            "native_execution_binding_sha256",
            "eligibility_sha256",
            "operation_trace_sha256",
            "acceleration_sha256",
            "boundary",
            "boundary_version",
            "lane",
            "renderer_id",
            "acceleration_id",
            "output_profile",
            "payload",
            "delivery_policy",
            "completion_endpoint",
            "device_uuid",
        )
        if any(not isinstance(payload.get(name), str) for name in string_fields):
            raise ValueError("execution receipt identity and execution fields must be strings")
        for name in ("transform_replay_sha256", "fixed_render_fixture_sha256"):
            value = payload.get(name)
            if value is not None and not _sha256_text(value):
                raise ValueError(f"execution receipt {name} is invalid")
        transform_frame_index = payload.get("transform_frame_index")
        if transform_frame_index is not None and (
            not isinstance(transform_frame_index, int)
            or isinstance(transform_frame_index, bool)
            or transform_frame_index < 0
        ):
            raise ValueError("execution receipt transform_frame_index is invalid")
        if (payload.get("transform_replay_sha256") is None) != (transform_frame_index is None):
            raise ValueError("execution receipt replay hash/index must appear together")
        readiness = NativeReadinessEvidence.from_payload(payload.get("native_readiness"))
        if payload.get("native_readiness_sha256") != readiness.sha256:
            raise ValueError("native readiness SHA-256 does not match its evidence content")
        eligibility = LaneEligibilityEvidence.from_payload(payload.get("eligibility"))
        if payload.get("eligibility_sha256") != eligibility.sha256:
            raise ValueError("lane eligibility SHA-256 does not match its evidence content")
        operation_trace = RuntimeOperationTraceEvidence.from_payload(payload.get("operation_trace"))
        if payload.get("operation_trace_sha256") != operation_trace.sha256:
            raise ValueError("operation trace SHA-256 does not match its evidence content")
        acceleration = AccelerationReceipt.from_payload(payload.get("acceleration"))
        if payload.get("acceleration_sha256") != acceleration.sha256:
            raise ValueError("acceleration SHA-256 does not match its receipt content")
        native_binding = NativeExecutionBindingEvidence.from_payload(
            payload.get("native_execution_binding")
        )
        if payload.get("native_execution_binding_sha256") != native_binding.sha256:
            raise ValueError("native execution binding SHA-256 does not match its content")
        publication_payload = payload.get("shipping_publication")
        publication_hash = payload.get("shipping_publication_sha256")
        shipping_publication: ShippingPublicationReceipt | None
        if publication_payload is None:
            if publication_hash is not None:
                raise ValueError("shipping publication SHA-256 requires a receipt")
            shipping_publication = None
        else:
            shipping_publication = ShippingPublicationReceipt.from_payload(publication_payload)
            if publication_hash != shipping_publication.sha256:
                raise ValueError("shipping publication SHA-256 does not match its receipt content")
        return cls(
            producer=payload["producer"],
            row_semantics_sha256=payload["row_semantics_sha256"],
            row_id=payload["row_id"],
            case_id=payload["case_id"],
            case_sha256=payload["case_sha256"],
            native_input_manifest_sha256=payload["native_input_manifest_sha256"],
            native_input_content_sha256=payload["native_input_content_sha256"],
            topology_sha256=payload["topology_sha256"],
            native_readiness=readiness,
            native_readiness_sha256=payload["native_readiness_sha256"],
            input_contract=payload["input_contract"],
            input_provenance_sha256=payload["input_provenance_sha256"],
            scene_contract_sha256=payload["scene_contract_sha256"],
            transform_replay_sha256=payload["transform_replay_sha256"],
            transform_frame_index=transform_frame_index,
            fixed_render_fixture_sha256=payload["fixed_render_fixture_sha256"],
            native_execution_binding=native_binding,
            native_execution_binding_sha256=payload["native_execution_binding_sha256"],
            eligibility=eligibility,
            eligibility_sha256=payload["eligibility_sha256"],
            operation_trace=operation_trace,
            operation_trace_sha256=payload["operation_trace_sha256"],
            acceleration=acceleration,
            acceleration_sha256=payload["acceleration_sha256"],
            shipping_publication=shipping_publication,
            shipping_publication_sha256=publication_hash,
            boundary=payload["boundary"],
            boundary_version=payload["boundary_version"],
            lane=payload["lane"],
            renderer_id=payload["renderer_id"],
            acceleration_id=payload["acceleration_id"],
            output_profile=payload["output_profile"],
            shadows=payload["shadows"],
            payload=payload["payload"],
            delivery_policy=payload["delivery_policy"],
            ring_depth=ring_depth,
            completion_endpoint=payload["completion_endpoint"],
            observed_per_ray_channels=tuple(ObservedChannel.from_payload(value) for value in per_ray),
            observed_scalar_channels=tuple(ObservedChannel.from_payload(value) for value in scalar),
            observed_payload_bytes=observed_bytes,
            lifecycle=tuple((name, value) for name, value in lifecycle.items()),
            device_uuid=payload["device_uuid"],
            counters=dict(counters),
            memory=dict(memory),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "producer": self.producer,
            "row_semantics_sha256": self.row_semantics_sha256,
            "row_id": self.row_id,
            "case_id": self.case_id,
            "case_sha256": self.case_sha256,
            "native_input_manifest_sha256": self.native_input_manifest_sha256,
            "native_input_content_sha256": self.native_input_content_sha256,
            "topology_sha256": self.topology_sha256,
            "native_readiness": self.native_readiness.to_payload(),
            "native_readiness_sha256": self.native_readiness_sha256,
            "input_contract": self.input_contract,
            "input_provenance_sha256": self.input_provenance_sha256,
            "scene_contract_sha256": self.scene_contract_sha256,
            "transform_replay_sha256": self.transform_replay_sha256,
            "transform_frame_index": self.transform_frame_index,
            "fixed_render_fixture_sha256": self.fixed_render_fixture_sha256,
            "native_execution_binding": self.native_execution_binding.to_payload(),
            "native_execution_binding_sha256": self.native_execution_binding_sha256,
            "eligibility": self.eligibility.to_payload(),
            "eligibility_sha256": self.eligibility_sha256,
            "operation_trace": self.operation_trace.to_payload(),
            "operation_trace_sha256": self.operation_trace_sha256,
            "acceleration": self.acceleration.to_payload(),
            "acceleration_sha256": self.acceleration_sha256,
            "shipping_publication": (
                None if self.shipping_publication is None else self.shipping_publication.to_payload()
            ),
            "shipping_publication_sha256": self.shipping_publication_sha256,
            "boundary": self.boundary,
            "boundary_version": self.boundary_version,
            "lane": self.lane,
            "renderer_id": self.renderer_id,
            "acceleration_id": self.acceleration_id,
            "output_profile": self.output_profile,
            "shadows": self.shadows,
            "payload": self.payload,
            "delivery_policy": self.delivery_policy,
            "ring_depth": self.ring_depth,
            "completion_endpoint": self.completion_endpoint,
            "observed_per_ray_channels": [channel.to_payload() for channel in self.observed_per_ray_channels],
            "observed_scalar_channels": [channel.to_payload() for channel in self.observed_scalar_channels],
            "observed_payload_bytes": self.observed_payload_bytes,
            "lifecycle": dict(self.lifecycle),
            "device_uuid": self.device_uuid,
            "counters": dict(self.counters),
            "memory": dict(self.memory),
        }

    @property
    def sha256(self) -> str:
        return canonical_execution_receipt_sha256(self.to_payload())


def default_gate_path() -> Path:
    return Path(__file__).resolve().parents[1] / "results" / "q54_p12_4" / "gate.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def load_frozen_gate(path: Path | None = None) -> dict[str, Any]:
    payload = load_json(default_gate_path() if path is None else path)
    validate_frozen_gate(payload)
    return payload


def canonical_gate_sha256(gate: Mapping[str, Any]) -> str:
    """Hash the semantic gate content independent of JSON whitespace."""

    encoded = json.dumps(
        gate,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_execution_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    """Hash only lane-produced receipt content, excluding coordinator metadata."""

    encoded = json.dumps(
        receipt,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_input_provenance_sha256(
    *,
    input_contract: str,
    case_id: str,
    case_sha256: str,
    native_input_manifest_sha256: str,
    native_input_content_sha256: str,
    topology_sha256: str,
    scene_contract_sha256: str,
    transform_replay_sha256: str | None,
    transform_frame_index: int | None,
    fixed_render_fixture_sha256: str | None,
    native_execution_binding_sha256: str,
    asset_manifest_sha256: str | None,
    shipping_publication_sha256: str | None,
) -> str:
    """Bind the measured input source independently from numerical equality."""

    payload = {
        "schema_version": INPUT_CONTRACT_SCHEMA_VERSION,
        "input_contract": input_contract,
        "case_id": case_id,
        "case_sha256": case_sha256,
        "native_input_manifest_sha256": native_input_manifest_sha256,
        "native_input_content_sha256": native_input_content_sha256,
        "topology_sha256": topology_sha256,
        "scene_contract_sha256": scene_contract_sha256,
        "transform_replay_sha256": transform_replay_sha256,
        "transform_frame_index": transform_frame_index,
        "fixed_render_fixture_sha256": fixed_render_fixture_sha256,
        "native_execution_binding_sha256": native_execution_binding_sha256,
        "asset_manifest_sha256": asset_manifest_sha256,
        "shipping_publication_sha256": shipping_publication_sha256,
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_row_semantics_sha256(row: Mapping[str, Any], gate: Mapping[str, Any]) -> str:
    """Hash one row plus every shared contract that changes its execution meaning."""

    case_id = str(row.get("case_id", ""))
    boundary = str(row.get("boundary", ""))
    payload = {
        "schema_version": ROW_SEMANTICS_SCHEMA_VERSION,
        "row": row,
        "case_sha256": gate.get("cases", {}).get(case_id, {}).get("sha256"),
        "boundary_version": gate.get("boundaries", {}).get(boundary),
        "lane_bindings": gate.get("lane_bindings"),
        "payload_channel_contracts": gate.get("payload_channel_contracts"),
        "diagnostic_overlay_contracts": gate.get("diagnostic_overlay_contracts"),
        "native_input_contract": gate.get("native_input_contract"),
        "native_execution_binding_contract": gate.get("native_execution_binding_contract"),
        "input_contracts": gate.get("input_contracts"),
        "eligibility_contract": gate.get("eligibility_contract"),
        "acceleration_contracts": gate.get("acceleration_contracts"),
        "asset_manifest_contract": gate.get("asset_manifest_contract"),
        "shipping_publication_receipt_contract": gate.get("shipping_publication_receipt_contract"),
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_result_record(record: Mapping[str, Any], gate: Mapping[str, Any]) -> None:
    """Validate one raw R0/R1/R2/S sample against the frozen gate."""

    validate_frozen_gate(gate)
    errors: list[str] = []
    missing = RESULT_RECORD_REQUIRED_FIELDS.difference(record)
    if missing:
        errors.append(f"missing result fields: {sorted(missing)}")
    extra = set(record).difference(RESULT_RECORD_ALLOWED_FIELDS)
    if extra:
        errors.append(f"coordinator result contains non-metadata fields: {sorted(extra)}")
    if record.get("schema_version") != RESULT_SCHEMA_VERSION:
        errors.append("result schema_version is not current")

    receipt: ExecutionReceipt | None = None
    receipt_payload = record.get("receipt")
    if isinstance(receipt_payload, Mapping):
        receipt_hash = record.get("receipt_sha256")
        if not _sha256_text(receipt_hash):
            errors.append("result receipt_sha256 must be a lowercase SHA-256 digest")
        elif receipt_hash != canonical_execution_receipt_sha256(receipt_payload):
            errors.append("result receipt_sha256 does not match lane-produced receipt content")
        try:
            receipt = ExecutionReceipt.from_payload(receipt_payload)
        except ValueError as exc:
            errors.append(str(exc))
    else:
        errors.append("result receipt must be an object produced by a backend lane")

    rows = {row["row_id"]: row for row in gate["mandatory_rows"]}
    row = rows.get(receipt.row_id) if receipt is not None else None
    if row is None:
        errors.append("receipt row_id is not present in the frozen gate")
    else:
        if receipt.case_id != row["case_id"]:
            errors.append("receipt case_id does not match its gate row")
        if receipt.boundary != row["boundary"]:
            errors.append("receipt boundary does not match its gate row")
        expected_boundary_version = gate["boundaries"][row["boundary"]]
        if receipt.boundary_version != expected_boundary_version:
            errors.append("receipt boundary_version does not match the frozen gate")
        expected_case_hash = gate["cases"][row["case_id"]]["sha256"]
        if receipt.case_sha256 != expected_case_hash:
            errors.append("receipt case_sha256 does not match the frozen gate")
        for field in (
            "native_input_manifest_sha256",
            "native_input_content_sha256",
            "topology_sha256",
            "native_readiness_sha256",
            "input_provenance_sha256",
            "scene_contract_sha256",
            "native_execution_binding_sha256",
            "eligibility_sha256",
            "operation_trace_sha256",
            "acceleration_sha256",
        ):
            if not _sha256_text(getattr(receipt, field)):
                errors.append(f"receipt {field} must be a lowercase SHA-256 digest")
        expected_row_hash = canonical_row_semantics_sha256(row, gate)
        if receipt.row_semantics_sha256 != expected_row_hash:
            errors.append("receipt row_semantics_sha256 does not match its gate row")

        lane = receipt.lane
        readiness = receipt.native_readiness
        if readiness.backend != lane:
            errors.append("receipt native readiness backend does not match its lane")
        if readiness.manifest_sha256 != receipt.native_input_manifest_sha256:
            errors.append("receipt native readiness manifest does not match input provenance")
        if readiness.content_sha256 != receipt.native_input_content_sha256:
            errors.append("receipt native readiness content does not match input provenance")
        if readiness.topology_sha256 != receipt.topology_sha256:
            errors.append("receipt native readiness topology does not match input provenance")
        if readiness.input_contract != receipt.input_contract:
            errors.append("receipt native readiness input contract does not match execution")
        if readiness.scene_contract_sha256 != receipt.scene_contract_sha256:
            errors.append("receipt native readiness scene contract does not match execution")
        if readiness.transform_replay_sha256 != receipt.transform_replay_sha256:
            errors.append("receipt native readiness replay contract does not match execution")
        if readiness.fixed_render_fixture_sha256 != receipt.fixed_render_fixture_sha256:
            errors.append("receipt native readiness fixed fixture does not match execution")
        if receipt.transform_frame_index is not None and (
            receipt.transform_frame_index >= readiness.transform_frame_count
        ):
            errors.append("receipt transform frame index is outside the prepared replay")
        if readiness.warmup_frames_completed < int(gate["measurement"]["warmup_frames"]):
            errors.append("receipt native readiness does not prove the frozen warmup count")
        lane_binding = gate["lane_bindings"].get(lane, {}) if isinstance(lane, str) else {}
        expected_renderer = lane_binding.get("renderer_id")
        acceleration_class = row.get("acceleration_class")
        expected_acceleration = lane_binding.get("acceleration_by_class", {}).get(acceleration_class)
        if receipt.renderer_id != expected_renderer:
            errors.append("receipt renderer_id does not match its lane binding")
        if receipt.acceleration_id != expected_acceleration:
            errors.append("receipt acceleration_id does not match its row/lane binding")
        expected_producer = gate["receipt_provenance"]["producer_by_lane"].get(lane)
        if receipt.producer != expected_producer:
            errors.append("receipt producer does not match its backend lane")

        expected_input_contract = row.get("input_contract")
        if receipt.input_contract != expected_input_contract:
            errors.append("receipt input_contract does not match its gate row")
        if row["boundary"] == "R0":
            if (
                receipt.transform_replay_sha256 is not None
                or receipt.transform_frame_index is not None
                or receipt.fixed_render_fixture_sha256 is None
            ):
                errors.append("R0 receipt must bind only fixed-render-fixture provenance")
        elif row["boundary"] in {"R1", "R2"}:
            if (
                receipt.transform_replay_sha256 is None
                or receipt.transform_frame_index is None
                or receipt.fixed_render_fixture_sha256 is not None
            ):
                errors.append("R1/R2 receipt must bind only replay provenance")
        elif row["boundary"] == "S" and (
            receipt.transform_replay_sha256 is not None
            or receipt.transform_frame_index is not None
            or receipt.fixed_render_fixture_sha256 is not None
        ):
            errors.append("S receipt must not carry replay or fixed-fixture provenance")

        native_binding = receipt.native_execution_binding
        expected_binding_identity = {
            "row_id": receipt.row_id,
            "case_id": receipt.case_id,
            "case_sha256": receipt.case_sha256,
            "boundary": receipt.boundary,
            "lane": receipt.lane,
            "input_contract": receipt.input_contract,
            "native_input_manifest_sha256": receipt.native_input_manifest_sha256,
            "native_readiness_sha256": receipt.native_readiness_sha256,
            "scene_contract_sha256": receipt.scene_contract_sha256,
            "transform_replay_sha256": receipt.transform_replay_sha256,
            "transform_frame_index": receipt.transform_frame_index,
            "fixed_render_fixture_sha256": receipt.fixed_render_fixture_sha256,
        }
        for name, expected in expected_binding_identity.items():
            if getattr(native_binding, name) != expected:
                errors.append(f"receipt native execution binding {name} does not match execution")
        eligibility = receipt.eligibility
        expected_eligibility_identity = {
            "row_id": receipt.row_id,
            "case_id": receipt.case_id,
            "boundary": receipt.boundary,
            "lane": receipt.lane,
            "input_contract": receipt.input_contract,
            "renderer_id": receipt.renderer_id,
            "acceleration_id": receipt.acceleration_id,
        }
        for name, expected in expected_eligibility_identity.items():
            if getattr(eligibility, name) != expected:
                errors.append(f"receipt eligibility {name} does not match execution")
        expected_operations = required_runtime_operations(row)
        if eligibility.required_operations != expected_operations:
            errors.append("receipt eligibility required operations do not match its gate row")
        if not eligibility.eligible or eligibility.reason_codes:
            errors.append("receipt eligibility does not authorize timed submission")
        recalculated_eligibility = evaluate_lane_eligibility(
            row=row,
            lane=lane,
            entry_point_audits=eligibility.entry_point_audits,
            asset_manifest_sha256=eligibility.asset_manifest_sha256,
        )
        if eligibility != recalculated_eligibility:
            errors.append("receipt eligibility was not produced by the frozen evaluation")
        required_asset = row.get("required_asset_manifest")
        if required_asset is not None and not _sha256_text(eligibility.asset_manifest_sha256):
            errors.append("receipt eligibility does not prove the required asset manifest")
        if required_asset is None and eligibility.asset_manifest_sha256 is not None:
            errors.append("receipt eligibility carries an unexpected asset manifest")
        receipt_artifacts = record.get("artifact_hashes")
        asset_artifact_sha256 = (
            receipt_artifacts.get("asset_manifest") if isinstance(receipt_artifacts, Mapping) else None
        )
        if required_asset is not None and asset_artifact_sha256 != eligibility.asset_manifest_sha256:
            errors.append("result asset manifest artifact does not match eligibility evidence")
        if required_asset is None and asset_artifact_sha256 is not None:
            errors.append("result carries an unexpected asset manifest artifact")

        operation_trace = receipt.operation_trace
        if (
            operation_trace.row_id != receipt.row_id
            or operation_trace.lane != receipt.lane
            or operation_trace.boundary != receipt.boundary
        ):
            errors.append("receipt operation trace identity does not match execution")
        try:
            validate_required_operation_order(operation_trace, expected_operations)
            validate_trace_entry_points(operation_trace, eligibility)
        except ValueError as exc:
            errors.append(str(exc))

        acceleration = receipt.acceleration
        if acceleration.acceleration_id != receipt.acceleration_id:
            errors.append("receipt acceleration identity does not match execution")
        if acceleration.scene_contract_sha256 != receipt.scene_contract_sha256:
            errors.append("receipt acceleration scene contract does not match execution")
        if acceleration.workload_class != acceleration_class:
            errors.append("receipt acceleration workload class does not match its gate row")
        exact_acceleration = (
            gate["acceleration_contracts"]["by_lane_and_identity"].get(lane, {}).get(receipt.acceleration_id)
        )
        if exact_acceleration is None:
            errors.append("receipt acceleration has no frozen exact contract")
        else:
            for name, expected in exact_acceleration.items():
                if getattr(acceleration, name) != expected:
                    errors.append(f"receipt acceleration {name} does not match its exact contract")
        if acceleration.topology_validation_sha256 != receipt.topology_sha256:
            errors.append("receipt acceleration topology validation does not match native input")
        if acceleration.instrumentation_trace_sha256 != receipt.operation_trace_sha256:
            errors.append("receipt acceleration does not bind the actual operation trace")
        if acceleration.completion_event_recorded is not True:
            errors.append("receipt acceleration does not prove completion-event recording")
        forbidden_counts = gate["acceleration_contracts"]["forbidden_primary_counts"]
        for name, expected in forbidden_counts.items():
            if getattr(acceleration, name) != expected:
                errors.append(f"receipt acceleration {name} must equal {expected}")
        classification = acceleration.operation_classification
        expected_counts = {
            "none": (0, 0, 0),
            "refit": (1, 1, 0),
            "rebuild": (1, 0, 1),
        }.get(classification)
        if (
            expected_counts is not None
            and (
                acceleration.scene_update_count,
                acceleration.refit_count,
                acceleration.rebuild_count,
            )
            != expected_counts
        ):
            errors.append("receipt acceleration operation counts do not match classification")
        if min(acceleration.primitive_count, acceleration.node_count, acceleration.leaf_count) <= 0:
            errors.append("receipt acceleration primitive/node/leaf counts must be positive")
        if acceleration.node_count < acceleration.leaf_count:
            errors.append("receipt acceleration node_count cannot be smaller than leaf_count")
        if classification == "none":
            if acceleration.acceleration_stage_evidence_sha256 is not None:
                errors.append("fixed R0 acceleration must not bind stage evidence")
            if (
                acceleration.scene_update_kernel_launch_count != 0
                or acceleration.acceleration_kernel_launch_count != 0
            ):
                errors.append("fixed R0 acceleration must not claim update/acceleration kernels")
        else:
            acceleration_stages = {
                stage.stage_name: stage for stage in native_binding.ordered_stage_evidence
            }
            acceleration_stage = acceleration_stages.get("acceleration")
            if (
                acceleration_stage is None
                or acceleration.acceleration_stage_evidence_sha256
                != acceleration_stage.evidence_sha256
            ):
                errors.append("dynamic acceleration does not bind its stage evidence")
            if (
                acceleration.scene_update_kernel_launch_count <= 0
                or acceleration.acceleration_kernel_launch_count <= 0
            ):
                errors.append("dynamic acceleration must prove scene and acceleration kernel launches")

        publication = receipt.shipping_publication
        if row["boundary"] == "S":
            if publication is None:
                errors.append("shipping receipt requires publication/lifecycle evidence")
            else:
                if publication.input_contract != receipt.input_contract:
                    errors.append("shipping publication input contract does not match execution")
                for name in (
                    "publish_event_recorded",
                    "producer_publish_completion_recorded",
                    "consumer_stream_wait_enqueued",
                    "zero_copy_transform_binding",
                    "consumer_completion_recorded",
                    "slot_logically_reclaimable",
                    "done_event_retained_until_reuse_dependency",
                ):
                    if getattr(publication, name) is not True:
                        errors.append(f"shipping publication {name} must be true")
                if publication.physics_step_count != 1 or publication.borrow_count != 1:
                    errors.append("shipping publication must prove one measured step and borrow")
                if (
                    publication.replay_source_count != 0
                    or publication.host_poll_count != 0
                    or publication.host_synchronize_count != 0
                ):
                    errors.append("shipping publication contains a forbidden replay/host operation")
                if publication.publication_trace_sha256 != receipt.operation_trace_sha256:
                    errors.append("shipping publication does not bind the actual operation trace")
                lifecycle = dict(receipt.lifecycle)
                if lifecycle.get("final_drain_complete", False) != publication.final_payload_drain_complete:
                    errors.append("shipping publication final drain does not match lifecycle")
                if lifecycle.get("final_reclaim_complete", False) != publication.final_reclaim_complete:
                    errors.append("shipping publication final reclaim does not match lifecycle")
        elif publication is not None or receipt.shipping_publication_sha256 is not None:
            errors.append("non-shipping receipt must not contain publication evidence")

        expected_input_provenance = canonical_input_provenance_sha256(
            input_contract=receipt.input_contract,
            case_id=receipt.case_id,
            case_sha256=receipt.case_sha256,
            native_input_manifest_sha256=receipt.native_input_manifest_sha256,
            native_input_content_sha256=receipt.native_input_content_sha256,
            topology_sha256=receipt.topology_sha256,
            scene_contract_sha256=receipt.scene_contract_sha256,
            transform_replay_sha256=receipt.transform_replay_sha256,
            transform_frame_index=receipt.transform_frame_index,
            fixed_render_fixture_sha256=receipt.fixed_render_fixture_sha256,
            native_execution_binding_sha256=receipt.native_execution_binding_sha256,
            asset_manifest_sha256=eligibility.asset_manifest_sha256,
            shipping_publication_sha256=receipt.shipping_publication_sha256,
        )
        if receipt.input_provenance_sha256 != expected_input_provenance:
            errors.append("receipt input provenance does not match its bound evidence")

        expected_execution = {
            "output_profile": row["output_profile"],
            "shadows": row["shadows"],
            "payload": row.get("payload", "none"),
            "delivery_policy": row.get("delivery_policy", "native_only"),
            "ring_depth": row.get("ring_depth", 0),
            "completion_endpoint": row["completion_endpoint"],
        }
        for field, expected in expected_execution.items():
            if not _exact_json_equal(getattr(receipt, field), expected):
                errors.append(f"receipt {field} does not match its gate row")

        lifecycle = dict(receipt.lifecycle)
        expected_lifecycle = {name: True for name in row["required_lifecycle"]}
        if not _exact_json_equal(lifecycle, expected_lifecycle):
            errors.append("receipt lifecycle does not prove every required row endpoint")

        if row["boundary"] in {"R2", "S"} and record.get("profile_mode") == "primary_unprofiled":
            expected_per_ray, expected_scalar, expected_bytes = _expected_payload_observation(
                row,
            )
            observed_per_ray = [channel.to_payload() for channel in receipt.observed_per_ray_channels]
            observed_scalar = [channel.to_payload() for channel in receipt.observed_scalar_channels]
            if not _exact_json_equal(observed_per_ray, expected_per_ray):
                errors.append("receipt observed_per_ray_channels do not match the payload contract")
            if not _exact_json_equal(observed_scalar, expected_scalar):
                errors.append("receipt observed_scalar_channels do not match the payload contract")
            if not _exact_json_equal(receipt.observed_payload_bytes, expected_bytes):
                errors.append("receipt observed_payload_bytes do not match the payload contract")

        metrics = record.get("metrics_ms")
        if not isinstance(metrics, Mapping):
            errors.append("result metrics_ms must be an object")
        else:
            for metric in row["primary_metrics"]:
                value = metrics.get(metric)
                if not _finite_nonnegative_number(value):
                    errors.append(f"result primary metric {metric!r} must be finite and non-negative")

    if receipt is not None and receipt.lane not in {"warp", "cuda"}:
        errors.append("receipt lane must be 'warp' or 'cuda'")
    if record.get("profile_mode") not in {"primary_unprofiled", "diagnostic_profiled"}:
        errors.append("result profile_mode is invalid")
    if record.get("profile_mode") == "primary_unprofiled" and record.get("profile_phases_ms"):
        errors.append("primary_unprofiled result must not contain profile phase timings")
    artifact_hashes = record.get("artifact_hashes")
    if not isinstance(artifact_hashes, Mapping):
        errors.append("result artifact_hashes must be an object")
    elif any(
        not isinstance(name, str) or not name or not _sha256_text(digest)
        for name, digest in artifact_hashes.items()
    ):
        errors.append("result artifact_hashes must map non-empty names to SHA-256 digests")
    gate_sha256 = record.get("gate_sha256")
    if not _sha256_text(gate_sha256):
        errors.append("result gate_sha256 must be a lowercase SHA-256 digest")
    elif gate_sha256 != canonical_gate_sha256(gate):
        errors.append("result gate_sha256 does not match the frozen gate content")
    for field in ("repeat_index", "sample_index"):
        value = record.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"result {field} must be a non-negative integer")

    if errors:
        raise ValueError("Invalid Q54 P12.4 result record:\n- " + "\n- ".join(errors))


def validate_frozen_gate(payload: Mapping[str, Any]) -> None:
    """Reject incomplete or mutable gate definitions before benchmark use."""

    errors: list[str] = []
    _expect_equal(errors, payload, "schema_version", GATE_SCHEMA_VERSION)
    _expect_equal(errors, payload, "result_schema_version", RESULT_SCHEMA_VERSION)
    _expect_equal(errors, payload, "case_schema_version", CASE_SCHEMA_VERSION)
    _expect_equal(errors, payload, "status", "frozen")
    if payload.get("frozen_before_result_inspection") is not True:
        errors.append("frozen_before_result_inspection must be true")

    expected_hashes = case_hashes()
    configured_cases = payload.get("cases")
    if not isinstance(configured_cases, Mapping):
        errors.append("cases must be an object")
    else:
        observed_hashes = {
            str(case_id): value.get("sha256") if isinstance(value, Mapping) else None
            for case_id, value in configured_cases.items()
        }
        if observed_hashes != expected_hashes:
            errors.append("case ids/hashes do not match canonical cases.py definitions")

    if payload.get("boundaries") != BOUNDARY_VERSIONS:
        errors.append("boundary versions do not match schema.py")

    measurement = payload.get("measurement")
    if not isinstance(measurement, Mapping):
        errors.append("measurement must be an object")
    else:
        for field in ("warmup_frames", "measured_frames", "paired_repeats"):
            if not _positive_int(measurement.get(field)):
                errors.append(f"measurement.{field} must be a positive integer")
        if measurement.get("profiled_primary_passes") is not False:
            errors.append("primary passes must be unprofiled")
        if measurement.get("paired_order") != ["warp_cuda", "cuda_warp", "warp_cuda"]:
            errors.append("measurement.paired_order must use the frozen alternating order")
        if measurement.get("calibration_rows") != [
            "empty_runner_overhead",
            "native_completion_event_overhead",
            "payload_copy_only",
        ]:
            errors.append("measurement.calibration_rows do not match the frozen a1 controls")

    relative = payload.get("relative_gate")
    if not isinstance(relative, Mapping) or float(relative.get("max_cuda_over_warp_ratio", -1.0)) != 1.0:
        errors.append("relative_gate.max_cuda_over_warp_ratio must equal 1.00")
    elif not _positive_number(relative.get("noise_band_ratio")):
        errors.append("relative_gate.noise_band_ratio must be positive")

    absolute = payload.get("absolute_product_gate")
    if not isinstance(absolute, Mapping):
        errors.append("absolute_product_gate must be an object")
    else:
        for field in (
            "max_sync_host_ready_p90_ms",
            "max_sync_slot_reclaimable_p90_ms",
            "min_async_sustained_fps",
        ):
            if not _positive_number(absolute.get(field)):
                errors.append(f"absolute_product_gate.{field} must be positive")
        if any("memory" in str(field).lower() for field in absolute):
            errors.append("absolute_product_gate must not contain an unfounded memory threshold")

    if payload.get("payload_channel_contracts") != EXPECTED_PAYLOAD_CHANNEL_CONTRACTS:
        errors.append("payload_channel_contracts do not match the formal a1 host payloads")
    if payload.get("diagnostic_overlay_contracts") != EXPECTED_DIAGNOSTIC_OVERLAY_CONTRACTS:
        errors.append("diagnostic_overlay_contracts do not match the formal opt-in overlays")
    if payload.get("memory_measurement") != EXPECTED_MEMORY_MEASUREMENT:
        errors.append("memory_measurement does not match the clean single-lane contract")
    _validate_lane_bindings(errors, payload.get("lane_bindings"))

    rows = payload.get("mandatory_rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        errors.append("mandatory_rows must be a non-empty array")
        rows = []
    _validate_rows(errors, rows, expected_case_ids=frozenset(expected_hashes))

    rows_by_id = {
        str(row["row_id"]): row
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("row_id"), str)
    }
    if isinstance(absolute, Mapping):
        latency_row = rows_by_id.get(absolute.get("latency_row_id"))
        throughput_row = rows_by_id.get(absolute.get("throughput_row_id"))
        if (
            latency_row is None
            or latency_row.get("boundary") != "S"
            or latency_row.get("delivery_policy") != "sync_host"
        ):
            errors.append("absolute latency_row_id must reference a synchronous S row")
        if (
            throughput_row is None
            or throughput_row.get("boundary") != "S"
            or throughput_row.get("delivery_policy") != "torch_async_ordered"
        ):
            errors.append("absolute throughput_row_id must reference an asynchronous S row")

    if payload.get("fixed_r0_topology_contract") != EXPECTED_FIXED_R0_TOPOLOGY_CONTRACT:
        errors.append("fixed_r0_topology_contract does not match the frozen a1 contract")
    if payload.get("c0_blocking_acceptance") != EXPECTED_C0_BLOCKING_ACCEPTANCE:
        errors.append("c0_blocking_acceptance does not match the approved lifecycle contract")
    if payload.get("receipt_provenance") != EXPECTED_RECEIPT_PROVENANCE:
        errors.append("receipt_provenance does not freeze lane-generated evidence ownership")
    if payload.get("native_input_contract") != EXPECTED_NATIVE_INPUT_CONTRACT:
        errors.append("native_input_contract does not match the durable a2 contract")
    if payload.get("native_execution_binding_contract") != (
        EXPECTED_NATIVE_EXECUTION_BINDING_CONTRACT
    ):
        errors.append("native_execution_binding_contract does not match C3a provenance")
    if payload.get("input_contracts") != EXPECTED_INPUT_CONTRACTS:
        errors.append("input_contracts do not match the approved replay/shipping split")
    if payload.get("eligibility_contract") != EXPECTED_ELIGIBILITY_CONTRACT:
        errors.append("eligibility_contract does not match the approved a3a fail-closed gate")
    if payload.get("phase_authorization") != EXPECTED_PHASE_AUTHORIZATION:
        errors.append("phase_authorization does not preserve native/shipping review authority")
    if payload.get("acceleration_contracts") != EXPECTED_ACCELERATION_CONTRACTS:
        errors.append("acceleration_contracts do not match the approved exact identities")
    if payload.get("asset_manifest_contract") != EXPECTED_ASSET_MANIFEST_CONTRACT:
        errors.append("asset_manifest_contract does not enforce exact Go2 preflight")
    if payload.get("shipping_publication_receipt_contract") != EXPECTED_SHIPPING_PUBLICATION_RECEIPT_CONTRACT:
        errors.append("shipping publication receipt contract is not frozen")
    if payload.get("approved_dependency_order_commit") != ("57073b8cabe39d7e4104d8cef77fce70fc17956c"):
        errors.append("approved_dependency_order_commit is not the reviewed v3 commit")
    if payload.get("approved_c3_design_commit") != (
        "435ffb275674c04f6452cb41775533db0f47f1d3"
    ):
        errors.append("approved_c3_design_commit is not the reviewed C3 v4 commit")

    continuation = payload.get("continuation")
    expected_continuation = {
        "max_rerun_campaigns_per_row": 2,
        "max_cuda_optimization_slices": 5,
        "max_correctness_repair_slices_per_failure": 1,
        "continuation_deadline": "2026-10-15T23:59:59+08:00",
        "inconclusive_resolution": "warp_selected",
        "default_terminal_branch": "warp_selected",
    }
    if continuation != expected_continuation:
        errors.append("continuation policy does not match the approved v5 defaults")

    if errors:
        raise ValueError("Invalid frozen Q54 P12.4 gate:\n- " + "\n- ".join(errors))


def validate_repository_inventory(
    repo_root: Path,
    inventory: Mapping[str, Any],
    cleanup: Mapping[str, Any],
) -> None:
    """Validate that the a0 inventory and cleanup ledger remain actionable."""

    errors: list[str] = []
    _expect_equal(errors, inventory, "schema_version", INVENTORY_SCHEMA_VERSION)
    _expect_equal(errors, cleanup, "schema_version", CLEANUP_SCHEMA_VERSION)

    surfaces = inventory.get("surfaces")
    if not isinstance(surfaces, Sequence) or isinstance(surfaces, (str, bytes)):
        errors.append("inventory.surfaces must be an array")
        surfaces = []
    for surface in surfaces:
        if not isinstance(surface, Mapping):
            errors.append("every inventory surface must be an object")
            continue
        _validate_paths_and_symbols(repo_root, surface, errors, context="inventory")

    entries = cleanup.get("entries")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)) or not entries:
        errors.append("cleanup.entries must be a non-empty array")
        entries = []
    entry_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            errors.append("every cleanup entry must be an object")
            continue
        entry_id = str(entry.get("entry_id", ""))
        if not entry_id or entry_id in entry_ids:
            errors.append(f"cleanup entry id must be non-empty and unique: {entry_id!r}")
        entry_ids.add(entry_id)
        _validate_paths_and_symbols(repo_root, entry, errors, context=f"cleanup {entry_id}")
        for branch in ("cuda_selected", "warp_selected"):
            action = entry.get(branch)
            if not isinstance(action, Mapping) or action.get("action") not in {
                "delete",
                "retain",
                "collapse",
                "conditional",
                "replace_then_delete",
            }:
                errors.append(f"cleanup {entry_id}.{branch} has no terminal action")
        if entry.get("must_resolve_by") != "P12.4e":
            errors.append(f"cleanup {entry_id} must resolve by P12.4e")

    if errors:
        raise ValueError("Invalid Q54 P12.4 a0 inventory:\n- " + "\n- ".join(errors))


def _validate_rows(
    errors: list[str],
    rows: Sequence[object],
    *,
    expected_case_ids: frozenset[str],
) -> None:
    row_ids: set[str] = set()
    r2_pairs: set[tuple[str, str]] = set()
    shipping_pairs: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            errors.append("every mandatory row must be an object")
            continue
        row_id = str(row.get("row_id", ""))
        if not row_id or row_id in row_ids:
            errors.append(f"row_id must be non-empty and unique: {row_id!r}")
        row_ids.add(row_id)
        if row.get("case_id") not in expected_case_ids:
            errors.append(f"row {row_id} references an unknown case")
        boundary = row.get("boundary")
        if boundary not in {"R0", "R1", "R2", "S"}:
            errors.append(f"row {row_id} has unsupported boundary {boundary!r}")
        if row.get("mandatory") is not True:
            errors.append(f"row {row_id} must be marked mandatory")
        if not row.get("primary_metrics"):
            errors.append(f"row {row_id} must declare primary_metrics")
        for field in (
            "output_profile",
            "shadows",
            "acceleration_class",
            "input_contract",
            "required_asset_manifest",
            "completion_endpoint",
            "required_lifecycle",
        ):
            if field not in row:
                errors.append(f"row {row_id} is missing {field}")
        if not isinstance(row.get("shadows"), bool):
            errors.append(f"row {row_id} shadows must be a boolean")
        expected_input_contract = INPUT_CONTRACT_BY_BOUNDARY.get(str(boundary))
        if row.get("input_contract") != expected_input_contract:
            errors.append(f"row {row_id} input_contract does not match its boundary")
        case = cases_by_id().get(str(row.get("case_id")))
        if case is not None and row.get("required_asset_manifest") != case.required_asset_manifest:
            errors.append(f"row {row_id} required_asset_manifest does not match its case")
        lifecycle = row.get("required_lifecycle")
        if not isinstance(lifecycle, Sequence) or isinstance(lifecycle, (str, bytes)) or not lifecycle:
            errors.append(f"row {row_id} required_lifecycle must be a non-empty array")

        if boundary in {"R2", "S"}:
            if not _positive_int(row.get("ring_depth")):
                errors.append(f"row {row_id} ring_depth must be a positive integer")
            for field in (
                "payload_contract_id",
                "expected_per_ray_channels",
                "expected_scalar_channels",
                "expected_per_ray_payload_bytes",
                "expected_scalar_payload_bytes",
                "expected_payload_bytes",
            ):
                if field not in row:
                    errors.append(f"row {row_id} is missing {field}")
            _validate_row_payload_contract(errors, row, row_id=row_id)

        if boundary == "R2":
            pair = (str(row.get("payload")), str(row.get("delivery_policy")))
            if pair not in ALLOWED_R2_PAYLOAD_POLICIES:
                errors.append(f"row {row_id} uses unsupported R2 payload/policy {pair}")
            r2_pairs.add(pair)
            for field in (
                "payload_ready_endpoint",
                "transaction_terminal_endpoint",
                "ring_depth",
                "sync_or_async",
                "native_output_reclaim_endpoint",
                "final_drain_required",
            ):
                if field not in row:
                    errors.append(f"row {row_id} is missing {field}")

        if boundary == "S":
            pair = (str(row.get("payload")), str(row.get("delivery_policy")))
            if pair not in ALLOWED_SHIPPING_PAYLOAD_POLICIES:
                errors.append(f"row {row_id} uses unsupported S payload/policy {pair}")
            shipping_pairs.add(pair)
            if row.get("publication_binding_contract_version") != PUBLICATION_BINDING_CONTRACT_VERSION:
                errors.append(f"row {row_id} has the wrong publication binding version")
            endpoints = row.get("endpoints")
            if not isinstance(endpoints, Mapping):
                errors.append(f"row {row_id} endpoints must be an object")
            else:
                missing = REQUIRED_SHIPPING_ENDPOINT_FIELDS.difference(endpoints)
                if missing:
                    errors.append(f"row {row_id} is missing shipping endpoints: {sorted(missing)}")

    if r2_pairs != ALLOWED_R2_PAYLOAD_POLICIES:
        errors.append("mandatory R2 rows must cover exactly the supported payload/policy matrix")
    if shipping_pairs != ALLOWED_SHIPPING_PAYLOAD_POLICIES:
        errors.append("mandatory S rows must cover exactly the supported payload/policy matrix")


def _validate_lane_bindings(errors: list[str], lane_bindings: object) -> None:
    if lane_bindings != EXPECTED_LANE_BINDINGS:
        errors.append("lane_bindings do not match the frozen renderer/acceleration identities")


def _validate_row_payload_contract(
    errors: list[str],
    row: Mapping[str, Any],
    *,
    row_id: str,
) -> None:
    contract_id = row.get("payload_contract_id")
    contract = HOST_PAYLOAD_CONTRACTS.get(str(contract_id))
    if contract is None:
        errors.append(f"row {row_id} uses an unknown payload_contract_id")
        return
    expected_per_ray_names = [spec.name for spec in contract.per_ray]
    expected_scalar_names = [spec.name for spec in contract.scalar]
    if row.get("expected_per_ray_channels") != expected_per_ray_names:
        errors.append(f"row {row_id} expected_per_ray_channels do not match its contract")
    if row.get("expected_scalar_channels") != expected_scalar_names:
        errors.append(f"row {row_id} expected_scalar_channels do not match its contract")

    case = cases_by_id().get(str(row.get("case_id")))
    if case is None:
        return
    per_ray, scalar, total = contract.expected_observation(case.width * case.height)
    per_ray_bytes = sum(int(channel["nbytes"]) for channel in per_ray)
    scalar_bytes = sum(int(channel["nbytes"]) for channel in scalar)
    if row.get("expected_per_ray_payload_bytes") != per_ray_bytes:
        errors.append(f"row {row_id} expected_per_ray_payload_bytes is incorrect")
    if row.get("expected_scalar_payload_bytes") != scalar_bytes:
        errors.append(f"row {row_id} expected_scalar_payload_bytes is incorrect")
    if row.get("expected_payload_bytes") != total:
        errors.append(f"row {row_id} expected_payload_bytes is incorrect")


def _expected_payload_observation(
    row: Mapping[str, Any],
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    contract_id = str(row["payload_contract_id"])
    contract = HOST_PAYLOAD_CONTRACTS[contract_id]
    case = cases_by_id()[str(row["case_id"])]
    return contract.expected_observation(case.width * case.height)


def _validate_paths_and_symbols(
    repo_root: Path,
    record: Mapping[str, Any],
    errors: list[str],
    *,
    context: str,
) -> None:
    paths = record.get("paths")
    if not isinstance(paths, Sequence) or isinstance(paths, (str, bytes)) or not paths:
        errors.append(f"{context} record must declare paths")
        return
    for relative in paths:
        path = repo_root / str(relative)
        if not path.exists():
            errors.append(f"{context} path does not exist: {relative}")
    symbols = record.get("symbols", {})
    if not isinstance(symbols, Mapping):
        errors.append(f"{context} symbols must be an object")
        return
    for relative, required_symbols in symbols.items():
        path = repo_root / str(relative)
        if not path.is_file():
            errors.append(f"{context} symbol file does not exist: {relative}")
            continue
        source = path.read_text(encoding="utf-8")
        for symbol in required_symbols:
            if str(symbol) not in source:
                errors.append(f"{context} symbol {symbol!r} missing from {relative}")


def _expect_equal(
    errors: list[str],
    payload: Mapping[str, Any],
    field: str,
    expected: object,
) -> None:
    if payload.get(field) != expected:
        errors.append(f"{field} must equal {expected!r}")


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _positive_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) > 0.0


def _finite_nonnegative_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _sha256_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_json_equal(observed: object, expected: object) -> bool:
    try:
        return json.dumps(observed, sort_keys=True) == json.dumps(expected, sort_keys=True)
    except (TypeError, ValueError):
        return False
