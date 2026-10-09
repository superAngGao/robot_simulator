"""Lane-owned acceleration and shipping publication receipt contracts."""

from __future__ import annotations

import hashlib
import json
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .lane_contract import (
    EXACT_ACCELERATION_CONTRACTS,
    EXACT_LANE_BINDINGS,
    GPU_ENGINE_PUBLISHED_FRAME_INPUT,
)

ACCELERATION_RECEIPT_SCHEMA_VERSION = "q54.p12.4.acceleration-receipt.v2"
SHIPPING_PUBLICATION_RECEIPT_SCHEMA_VERSION = "q54.p12.4.shipping-publication-receipt.v1"
NATIVE_STAGE_EVIDENCE_SCHEMA_VERSION = "q54.p12.4.native-stage-evidence.v3"
NATIVE_EXECUTION_BINDING_SCHEMA_VERSION = "q54.p12.4.c3-one-shot-binding.v6"
DYNAMIC_DATAFLOW_CONTRACT_SCHEMA_VERSION = "q54.p12.4.dynamic-dataflow.v2"


def _validate_render_record(
    payload,
    *,
    expected_sha256,
    scene_contract_sha256,
    output_profile,
    shadows,
    case_id=None,
    case_sha256=None,
):
    # Lazy import avoids a provenance/native-input/materializer import cycle;
    # production module import itself retains its dependency-light boundary.
    from .cases import get_case
    from .render_parameters import BoundRenderParameters
    from .render_sequence_contract import RESOLVED_RENDER_SEQUENCE_SHA256
    from .workload_contract import frozen_workload_content

    matching = [
        name
        for name in RESOLVED_RENDER_SEQUENCE_SHA256
        if frozen_workload_content(name).scene_contract_sha256 == scene_contract_sha256
    ]
    if len(matching) != 1 or (case_id is not None and case_id != matching[0]):
        raise ValueError("render parameter scene/case identity is unresolved or inconsistent")
    case = get_case(matching[0])
    if case_sha256 is not None and case_sha256 != case.sha256:
        raise ValueError("render parameter case hash mismatch")
    record = BoundRenderParameters.from_payload(
        payload,
        case_id=case.case_id,
        profile=output_profile,
        shadows=shadows,
    )
    if expected_sha256 != record.sha256:
        raise ValueError("bound render parameters digest mismatch")
    return record


_CUDA_ARENA_C3_BINDING_AUTHORITIES: dict[
    int,
    tuple[weakref.ReferenceType[object], object, object, object],
] = {}


def _register_cuda_arena_c3_binding_authority(
    binding: object,
    *,
    target: object,
    prepared_input: object,
    native_readiness: object,
) -> None:
    """Register one adapter-issued, identity-bound C3 arena receipt.

    Canonical serialization proves the receipt's contents; this registry
    separately proves that the CUDA arena adapter derived those contents from
    the exact live a2 authorities and target allocations.
    """

    binding_id = id(binding)

    def discard(reference: weakref.ReferenceType[object]) -> None:
        current = _CUDA_ARENA_C3_BINDING_AUTHORITIES.get(binding_id)
        if current is not None and current[0] is reference:
            _CUDA_ARENA_C3_BINDING_AUTHORITIES.pop(binding_id, None)

    reference = weakref.ref(binding, discard)
    _CUDA_ARENA_C3_BINDING_AUTHORITIES[binding_id] = (
        reference,
        target,
        prepared_input,
        native_readiness,
    )


def _assert_cuda_arena_c3_binding_authority(
    binding: object,
    *,
    target: object,
    prepared_input: object,
    native_readiness: object,
) -> None:
    authority = _CUDA_ARENA_C3_BINDING_AUTHORITIES.get(id(binding))
    if (
        authority is None
        or authority[0]() is not binding
        or authority[1] is not target
        or authority[2] is not prepared_input
        or authority[3] is not native_readiness
    ):
        raise ValueError(
            "CUDA arena C3 binding was not issued by the canonical arena adapter "
            "for these exact a2 authorities"
        )


# These names are the logical allocation ABI between C3 stages.  They mirror
# DeviceOpticalSceneSnapshot, DeviceOpticalBvh, CudaFirstHitInputs, and
# CudaShadeInputs.  The tuple order is part of the contract.
DYNAMIC_SCENE_UPDATE_RUNTIME_INPUT_NAMES = (
    "geometry_vertices_local",
    "triangle_vertex_indices",
    "triangle_instance_index",
    "plane_normal_local",
    "plane_point_local",
    "plane_instance_index",
    "instance_body_index",
    "instance_X_body_geometry_R",
    "instance_X_body_geometry_r",
)
DYNAMIC_TRANSFORM_INPUT_NAMES = ("replay_x_world_R", "replay_x_world_r")
DYNAMIC_SCENE_OUTPUT_NAMES = (
    "triangle_v0_world",
    "triangle_e1_world",
    "triangle_e2_world",
    "triangle_normal_world",
    "plane_normal_world",
    "plane_point_world",
    "triangle_aabb_min",
    "triangle_aabb_max",
)
DYNAMIC_ACCELERATION_INPUT_NAMES = ("triangle_aabb_min", "triangle_aabb_max")
DYNAMIC_ACCELERATION_OUTPUT_NAMES = (
    "bvh_bounds_min",
    "bvh_bounds_max",
    "bvh_left",
    "bvh_right",
    "bvh_start",
    "bvh_count",
    "bvh_prim_ids",
)
DYNAMIC_RENDER_WORLD_INPUT_NAMES = (
    "plane_normal_world",
    "plane_point_world",
    "triangle_v0_world",
    "triangle_e1_world",
    "triangle_e2_world",
    "triangle_normal_world",
)
DYNAMIC_RENDER_METADATA_INPUT_NAMES = (
    "plane_numeric_instance_id",
    "plane_source_order_key",
    "plane_role_mask",
    "plane_material_index",
    "triangle_numeric_instance_id",
    "triangle_source_order_key",
    "triangle_role_mask",
    "triangle_material_index",
)
DYNAMIC_RENDER_SHADE_INPUT_NAMES = (
    "material_albedo_rgb",
    "light_kind",
    "light_position_or_direction_world",
    "light_intensity",
    "light_color_rgb",
)

_SUPPORTED_OUTPUT_PROFILES = frozenset({"direct_light_full", "rgb_preview", "render_only"})


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
class NativeStorageSignature:
    """Exact runtime allocation identity used in a native transaction."""

    name: str
    backend: str
    device_ordinal: int
    storage_token: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    active_start: int
    active_count: int
    owner_token: str

    @classmethod
    def from_payload(cls, payload: object) -> "NativeStorageSignature":
        expected = set(cls.__dataclass_fields__)
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("native storage signature fields are invalid")
        for name in ("name", "backend", "storage_token", "dtype", "owner_token"):
            if not isinstance(payload.get(name), str) or not payload[name]:
                raise ValueError(f"native storage signature {name} must be non-empty")
        if payload.get("backend") not in {"warp", "cuda"}:
            raise ValueError("native storage signature backend is invalid")
        for name in ("device_ordinal", "nbytes", "active_start", "active_count"):
            value = payload.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"native storage signature {name} must be non-negative")
        shape = payload.get("shape")
        if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
            raise ValueError("native storage signature shape must be an array")
        dimensions = tuple(shape)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in dimensions):
            raise ValueError("native storage signature shape is invalid")
        if payload["active_start"] + payload["active_count"] > payload["nbytes"]:
            raise ValueError("native storage active range exceeds its allocation")
        return cls(
            name=payload["name"],
            backend=payload["backend"],
            device_ordinal=payload["device_ordinal"],
            storage_token=payload["storage_token"],
            dtype=payload["dtype"],
            shape=dimensions,
            nbytes=payload["nbytes"],
            active_start=payload["active_start"],
            active_count=payload["active_count"],
            owner_token=payload["owner_token"],
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "backend": self.backend,
            "device_ordinal": self.device_ordinal,
            "storage_token": self.storage_token,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "nbytes": self.nbytes,
            "active_start": self.active_start,
            "active_count": self.active_count,
            "owner_token": self.owner_token,
        }


@dataclass(frozen=True)
class NativeStageEvidence:
    """Independent stage facts; the payload intentionally has no self-hash."""

    schema_version: str
    stage_name: str
    operation_id: str
    output_profile: str | None
    shadows: bool | None
    scene_contract_sha256: str
    input_contract: str
    transform_replay_sha256: str | None
    transform_frame_index: int | None
    fixed_render_fixture_sha256: str | None
    topology_sha256: str | None
    input_storage: tuple[NativeStorageSignature, ...]
    output_storage: tuple[NativeStorageSignature, ...]
    dispatch_plan_sha256: str
    bound_render_parameters: object | None
    bound_render_parameters_sha256: str | None

    @classmethod
    def from_payload(cls, payload: object) -> "NativeStageEvidence":
        expected = set(cls.__dataclass_fields__)
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("native stage evidence fields are invalid")
        if payload.get("schema_version") != NATIVE_STAGE_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("native stage evidence schema version is invalid")
        for name in ("stage_name", "operation_id", "input_contract"):
            if not isinstance(payload.get(name), str) or not payload[name]:
                raise ValueError(f"native stage evidence {name} must be non-empty")
        if payload["stage_name"] == "render":
            if payload.get("output_profile") not in _SUPPORTED_OUTPUT_PROFILES:
                raise ValueError("native render stage output profile is invalid")
            if not isinstance(payload.get("shadows"), bool):
                raise ValueError("native render stage shadows must be boolean")
            bound_render = _validate_render_record(
                payload.get("bound_render_parameters"),
                expected_sha256=payload.get("bound_render_parameters_sha256"),
                scene_contract_sha256=payload.get("scene_contract_sha256"),
                output_profile=payload["output_profile"],
                shadows=payload["shadows"],
            )
        elif payload.get("output_profile") is not None or payload.get("shadows") is not None:
            raise ValueError("non-render native stage cannot carry render identity")
        else:
            if (
                payload.get("bound_render_parameters") is not None
                or payload.get("bound_render_parameters_sha256") is not None
            ):
                raise ValueError("non-render stage cannot carry bound render parameters")
            bound_render = None
        for name in (
            "scene_contract_sha256",
            "transform_replay_sha256",
            "fixed_render_fixture_sha256",
            "topology_sha256",
            "dispatch_plan_sha256",
        ):
            value = payload.get(name)
            if value is not None and not _sha256_text(value):
                raise ValueError(f"native stage evidence {name} is invalid")
        if not _sha256_text(payload.get("scene_contract_sha256")):
            raise ValueError("native stage evidence scene contract digest is required")
        if not _sha256_text(payload.get("dispatch_plan_sha256")):
            raise ValueError("native stage evidence dispatch plan digest is required")
        frame_index = payload.get("transform_frame_index")
        if frame_index is not None and (
            not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0
        ):
            raise ValueError("native stage evidence transform frame index is invalid")
        if (payload.get("transform_replay_sha256") is None) != (frame_index is None):
            raise ValueError("native stage evidence replay hash/index must appear together")
        input_storage = _storage_sequence(payload.get("input_storage"), label="input")
        output_storage = _storage_sequence(payload.get("output_storage"), label="output")
        return cls(
            schema_version=NATIVE_STAGE_EVIDENCE_SCHEMA_VERSION,
            stage_name=payload["stage_name"],
            operation_id=payload["operation_id"],
            output_profile=payload["output_profile"],
            shadows=payload["shadows"],
            scene_contract_sha256=payload["scene_contract_sha256"],
            input_contract=payload["input_contract"],
            transform_replay_sha256=payload["transform_replay_sha256"],
            transform_frame_index=frame_index,
            fixed_render_fixture_sha256=payload["fixed_render_fixture_sha256"],
            topology_sha256=payload["topology_sha256"],
            input_storage=input_storage,
            output_storage=output_storage,
            dispatch_plan_sha256=payload["dispatch_plan_sha256"],
            bound_render_parameters=bound_render,
            bound_render_parameters_sha256=payload["bound_render_parameters_sha256"],
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "stage_name": self.stage_name,
            "operation_id": self.operation_id,
            "output_profile": self.output_profile,
            "shadows": self.shadows,
            "scene_contract_sha256": self.scene_contract_sha256,
            "input_contract": self.input_contract,
            "transform_replay_sha256": self.transform_replay_sha256,
            "transform_frame_index": self.transform_frame_index,
            "fixed_render_fixture_sha256": self.fixed_render_fixture_sha256,
            "topology_sha256": self.topology_sha256,
            "input_storage": [value.to_payload() for value in self.input_storage],
            "output_storage": [value.to_payload() for value in self.output_storage],
            "dispatch_plan_sha256": self.dispatch_plan_sha256,
            "bound_render_parameters": None
            if self.bound_render_parameters is None
            else self.bound_render_parameters.to_payload(),
            "bound_render_parameters_sha256": self.bound_render_parameters_sha256,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_payload())


@dataclass(frozen=True)
class NativeStageBinding:
    """One stage payload plus its externally computed canonical digest."""

    stage_name: str
    evidence: NativeStageEvidence
    evidence_sha256: str

    @classmethod
    def from_payload(cls, payload: object) -> "NativeStageBinding":
        if not isinstance(payload, Mapping) or set(payload) != {
            "stage_name",
            "evidence",
            "evidence_sha256",
        }:
            raise ValueError("native stage binding fields are invalid")
        evidence = NativeStageEvidence.from_payload(payload["evidence"])
        if payload.get("stage_name") != evidence.stage_name:
            raise ValueError("native stage binding name does not match its evidence")
        if payload.get("evidence_sha256") != evidence.sha256:
            raise ValueError("native stage binding digest does not match its evidence")
        return cls(
            stage_name=evidence.stage_name,
            evidence=evidence,
            evidence_sha256=evidence.sha256,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "stage_name": self.stage_name,
            "evidence": self.evidence.to_payload(),
            "evidence_sha256": self.evidence_sha256,
        }


@dataclass(frozen=True)
class C3OneShotNativeExecutionBindingEvidence:
    """C3 one-shot aggregate binding stage facts to prepared allocations.

    This type is deliberately not a result-submission capability.  C4 must
    introduce a separate evidence type whose constructor proves arena, slot,
    generation, lease, and ring ownership.
    """

    schema_version: str
    row_id: str
    case_id: str
    case_sha256: str
    boundary: str
    lane: str
    renderer_id: str
    acceleration_class: str
    acceleration_id: str
    output_profile: str
    shadows: bool
    dataflow_contract_id: str
    input_contract: str
    native_input_manifest_sha256: str
    native_readiness_sha256: str
    scene_contract_sha256: str
    transform_replay_sha256: str | None
    transform_frame_index: int | None
    fixed_render_fixture_sha256: str | None
    runtime_scene_storage: tuple[NativeStorageSignature, ...]
    runtime_scene_storage_sha256: str
    derived_fixture_storage: tuple[NativeStorageSignature, ...]
    derived_fixture_storage_sha256: str | None
    ordered_stage_evidence: tuple[NativeStageBinding, ...]
    dataflow_binding_sha256: str
    bound_render_parameters: object
    bound_render_parameters_sha256: str

    @classmethod
    def from_payload(cls, payload: object) -> "C3OneShotNativeExecutionBindingEvidence":
        expected = set(cls.__dataclass_fields__)
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("native execution binding fields are invalid")
        if payload.get("schema_version") != NATIVE_EXECUTION_BINDING_SCHEMA_VERSION:
            raise ValueError("native execution binding schema version is invalid")
        for name in (
            "row_id",
            "case_id",
            "boundary",
            "lane",
            "renderer_id",
            "acceleration_class",
            "acceleration_id",
            "output_profile",
            "dataflow_contract_id",
            "input_contract",
        ):
            if not isinstance(payload.get(name), str) or not payload[name]:
                raise ValueError(f"native execution binding {name} must be non-empty")
        if payload.get("lane") not in {"warp", "cuda"}:
            raise ValueError("native execution binding lane is invalid")
        if payload.get("output_profile") not in _SUPPORTED_OUTPUT_PROFILES:
            raise ValueError("native execution binding output profile is invalid")
        if not isinstance(payload.get("shadows"), bool):
            raise ValueError("native execution binding shadows must be boolean")
        bound_render = _validate_render_record(
            payload.get("bound_render_parameters"),
            expected_sha256=payload.get("bound_render_parameters_sha256"),
            scene_contract_sha256=payload.get("scene_contract_sha256"),
            output_profile=payload["output_profile"],
            shadows=payload["shadows"],
            case_id=payload["case_id"],
            case_sha256=payload["case_sha256"],
        )
        for name in (
            "case_sha256",
            "native_input_manifest_sha256",
            "native_readiness_sha256",
            "scene_contract_sha256",
            "runtime_scene_storage_sha256",
            "dataflow_binding_sha256",
        ):
            if not _sha256_text(payload.get(name)):
                raise ValueError(f"native execution binding {name} is invalid")
        for name in (
            "transform_replay_sha256",
            "fixed_render_fixture_sha256",
            "derived_fixture_storage_sha256",
        ):
            value = payload.get(name)
            if value is not None and not _sha256_text(value):
                raise ValueError(f"native execution binding {name} is invalid")
        frame_index = payload.get("transform_frame_index")
        if frame_index is not None and (
            not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0
        ):
            raise ValueError("native execution binding transform frame index is invalid")
        if (payload.get("transform_replay_sha256") is None) != (frame_index is None):
            raise ValueError("native execution binding replay hash/index must appear together")
        if frame_index is not None:
            from .cases import get_case

            frame_count = int(get_case(payload["case_id"]).transform_sequence["frames"])
            if frame_index != (bound_render.to_payload()["frame_id"] - 1) % frame_count:
                raise ValueError("render sample and transform replay frame differ")
        runtime_scene_storage = _storage_sequence(payload.get("runtime_scene_storage"), label="runtime scene")
        if not runtime_scene_storage:
            raise ValueError("native execution binding runtime scene storage is empty")
        if payload.get("runtime_scene_storage_sha256") != canonical_storage_manifest_sha256(
            runtime_scene_storage
        ):
            raise ValueError("native execution binding runtime scene storage digest is invalid")
        derived_fixture_storage = _storage_sequence(
            payload.get("derived_fixture_storage"), label="derived fixture"
        )
        derived_storage_hash = payload.get("derived_fixture_storage_sha256")
        if derived_fixture_storage:
            if derived_storage_hash != canonical_storage_manifest_sha256(derived_fixture_storage):
                raise ValueError("native execution binding derived fixture storage digest is invalid")
        elif derived_storage_hash is not None:
            raise ValueError("empty derived fixture storage must not carry a digest")
        if (payload.get("fixed_render_fixture_sha256") is None) != (not derived_fixture_storage):
            raise ValueError("native execution binding fixed fixture storage is inconsistent")
        stages_payload = payload.get("ordered_stage_evidence")
        if not isinstance(stages_payload, Sequence) or isinstance(stages_payload, (str, bytes)):
            raise ValueError("native execution binding stages must be an array")
        stages: list[NativeStageBinding] = []
        for entry in stages_payload:
            stages.append(NativeStageBinding.from_payload(entry))
        if len({stage.stage_name for stage in stages}) != len(stages):
            raise ValueError("native execution binding stage names must be unique")
        stage_names = tuple(stage.stage_name for stage in stages)
        boundary = payload["boundary"]
        if boundary == "R0" and stage_names != ("render",):
            raise ValueError("R0 native execution binding must contain only render evidence")
        if boundary in {"R1", "R2"} and stage_names[:3] != (
            "scene_update",
            "acceleration",
            "render",
        ):
            raise ValueError("dynamic native execution binding has the wrong stage prefix")
        all_storage = runtime_scene_storage + derived_fixture_storage
        for stage in stages:
            evidence = stage.evidence
            if evidence.stage_name == "render" and (
                evidence.bound_render_parameters != bound_render
                or evidence.bound_render_parameters_sha256 != bound_render.sha256
            ):
                raise ValueError("render stage parameters differ from aggregate binding")
            if (
                evidence.scene_contract_sha256 != payload["scene_contract_sha256"]
                or evidence.input_contract != payload["input_contract"]
                or evidence.transform_replay_sha256 != payload["transform_replay_sha256"]
                or evidence.transform_frame_index != frame_index
                or evidence.fixed_render_fixture_sha256 != payload["fixed_render_fixture_sha256"]
            ):
                raise ValueError("native stage evidence provenance does not match its binding")
            all_storage += evidence.input_storage + evidence.output_storage
        _validate_stage_dataflow(
            boundary=boundary,
            lane=payload["lane"],
            renderer_id=payload["renderer_id"],
            acceleration_class=payload["acceleration_class"],
            acceleration_id=payload["acceleration_id"],
            output_profile=payload["output_profile"],
            shadows=payload["shadows"],
            dataflow_contract_id=payload["dataflow_contract_id"],
            runtime_scene_storage=runtime_scene_storage,
            derived_fixture_storage=derived_fixture_storage,
            ordered_stage_evidence=tuple(stages),
        )
        if any(
            storage.backend != payload["lane"] or storage.device_ordinal != all_storage[0].device_ordinal
            for storage in all_storage
        ):
            raise ValueError("native execution binding storage lane/device is inconsistent")
        expected_dataflow_sha256 = canonical_dataflow_binding_sha256(
            runtime_scene_storage=runtime_scene_storage,
            derived_fixture_storage=derived_fixture_storage,
            ordered_stage_evidence=tuple(stages),
        )
        if payload.get("dataflow_binding_sha256") != expected_dataflow_sha256:
            raise ValueError("native execution binding dataflow digest is invalid")
        return cls(
            schema_version=NATIVE_EXECUTION_BINDING_SCHEMA_VERSION,
            row_id=payload["row_id"],
            case_id=payload["case_id"],
            case_sha256=payload["case_sha256"],
            boundary=payload["boundary"],
            lane=payload["lane"],
            renderer_id=payload["renderer_id"],
            acceleration_class=payload["acceleration_class"],
            acceleration_id=payload["acceleration_id"],
            output_profile=payload["output_profile"],
            shadows=payload["shadows"],
            dataflow_contract_id=payload["dataflow_contract_id"],
            input_contract=payload["input_contract"],
            native_input_manifest_sha256=payload["native_input_manifest_sha256"],
            native_readiness_sha256=payload["native_readiness_sha256"],
            scene_contract_sha256=payload["scene_contract_sha256"],
            transform_replay_sha256=payload["transform_replay_sha256"],
            transform_frame_index=frame_index,
            fixed_render_fixture_sha256=payload["fixed_render_fixture_sha256"],
            runtime_scene_storage=runtime_scene_storage,
            runtime_scene_storage_sha256=payload["runtime_scene_storage_sha256"],
            derived_fixture_storage=derived_fixture_storage,
            derived_fixture_storage_sha256=payload["derived_fixture_storage_sha256"],
            ordered_stage_evidence=tuple(stages),
            dataflow_binding_sha256=payload["dataflow_binding_sha256"],
            bound_render_parameters=bound_render,
            bound_render_parameters_sha256=bound_render.sha256,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "row_id": self.row_id,
            "case_id": self.case_id,
            "case_sha256": self.case_sha256,
            "boundary": self.boundary,
            "lane": self.lane,
            "renderer_id": self.renderer_id,
            "acceleration_class": self.acceleration_class,
            "acceleration_id": self.acceleration_id,
            "output_profile": self.output_profile,
            "shadows": self.shadows,
            "dataflow_contract_id": self.dataflow_contract_id,
            "input_contract": self.input_contract,
            "native_input_manifest_sha256": self.native_input_manifest_sha256,
            "native_readiness_sha256": self.native_readiness_sha256,
            "scene_contract_sha256": self.scene_contract_sha256,
            "transform_replay_sha256": self.transform_replay_sha256,
            "transform_frame_index": self.transform_frame_index,
            "fixed_render_fixture_sha256": self.fixed_render_fixture_sha256,
            "runtime_scene_storage": [value.to_payload() for value in self.runtime_scene_storage],
            "runtime_scene_storage_sha256": self.runtime_scene_storage_sha256,
            "derived_fixture_storage": [value.to_payload() for value in self.derived_fixture_storage],
            "derived_fixture_storage_sha256": self.derived_fixture_storage_sha256,
            "ordered_stage_evidence": [value.to_payload() for value in self.ordered_stage_evidence],
            "dataflow_binding_sha256": self.dataflow_binding_sha256,
            "bound_render_parameters": self.bound_render_parameters.to_payload(),
            "bound_render_parameters_sha256": self.bound_render_parameters_sha256,
        }

    def _assert_cuda_arena_adapter_authority(
        self,
        *,
        target: object,
        prepared_input: object,
        native_readiness: object,
    ) -> None:
        _assert_cuda_arena_c3_binding_authority(
            self,
            target=target,
            prepared_input=prepared_input,
            native_readiness=native_readiness,
        )

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_payload())


@dataclass(frozen=True)
class AccelerationReceipt:
    """Actual scene-update and acceleration operations observed by a lane."""

    acceleration_schema_version: str
    scene_contract_sha256: str
    acceleration_stage_evidence_sha256: str | None
    acceleration_id: str
    workload_class: str
    topology_origin: str
    topology_lifetime: str
    topology_validation_sha256: str
    timed_scene_update_operation: str
    timed_acceleration_operation: str
    operation_classification: str
    scene_update_count: int
    refit_count: int
    rebuild_count: int
    primitive_count: int
    node_count: int
    leaf_count: int
    scene_update_kernel_launch_count: int
    acceleration_kernel_launch_count: int
    device_allocation_count: int
    host_materialization_count: int
    host_upload_bytes: int
    device_to_host_bytes: int
    global_synchronize_count: int
    bridge_creation_count: int
    completion_event_recorded: bool
    instrumentation_trace_sha256: str

    @classmethod
    def from_payload(cls, payload: object) -> "AccelerationReceipt":
        expected = set(cls.__dataclass_fields__)
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("acceleration receipt fields are invalid")
        if payload.get("acceleration_schema_version") != ACCELERATION_RECEIPT_SCHEMA_VERSION:
            raise ValueError("acceleration receipt schema version is invalid")
        for name in (
            "acceleration_id",
            "workload_class",
            "topology_origin",
            "topology_lifetime",
            "timed_scene_update_operation",
            "timed_acceleration_operation",
        ):
            if not isinstance(payload.get(name), str) or not payload[name]:
                raise ValueError(f"acceleration receipt {name} must be non-empty")
        if payload.get("operation_classification") not in {"none", "refit", "rebuild"}:
            raise ValueError("acceleration receipt operation_classification is invalid")
        for name in (
            "scene_update_count",
            "refit_count",
            "rebuild_count",
            "primitive_count",
            "node_count",
            "leaf_count",
            "scene_update_kernel_launch_count",
            "acceleration_kernel_launch_count",
            "device_allocation_count",
            "host_materialization_count",
            "host_upload_bytes",
            "device_to_host_bytes",
            "global_synchronize_count",
            "bridge_creation_count",
        ):
            value = payload.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"acceleration receipt {name} must be a non-negative integer")
        if not isinstance(payload.get("completion_event_recorded"), bool):
            raise ValueError("acceleration receipt completion_event_recorded must be boolean")
        for name in (
            "scene_contract_sha256",
            "topology_validation_sha256",
            "instrumentation_trace_sha256",
        ):
            if not _sha256_text(payload.get(name)):
                raise ValueError(f"acceleration receipt {name} must be a lowercase SHA-256 digest")
        stage_sha = payload.get("acceleration_stage_evidence_sha256")
        if stage_sha is not None and not _sha256_text(stage_sha):
            raise ValueError("acceleration stage evidence digest is invalid")
        if payload.get("operation_classification") == "none":
            if stage_sha is not None:
                raise ValueError("fixed R0 acceleration must not invent stage evidence")
        elif stage_sha is None:
            raise ValueError("dynamic acceleration must bind acceleration-stage evidence")
        return cls(**{name: payload[name] for name in expected})

    def to_payload(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_payload())


@dataclass(frozen=True)
class ShippingPublicationReceipt:
    """Actual physics producer/consumer handoff and reclaim evidence for S."""

    publication_schema_version: str
    input_contract: str
    physics_run_id: str
    physics_step_index: int
    physics_frame_id: int
    physics_slot_id: int
    physics_device_ordinal: int
    physics_step_count: int
    publish_event_token: str
    publish_event_recorded: bool
    producer_publish_completion_recorded: bool
    consumer_binding_id: str
    consumer_stream_token: str
    borrow_count: int
    consumer_stream_wait_enqueued: bool
    zero_copy_transform_binding: bool
    last_physics_buffer_reader_operation: str
    consumer_done_event_token: str
    consumer_completion_recorded: bool
    slot_logically_reclaimable: bool
    producer_reuse_wait_enqueued_when_reused: bool
    done_event_retained_until_reuse_dependency: bool
    physical_reclaim_evidence: str
    final_payload_drain_complete: bool
    final_reclaim_complete: bool
    replay_source_count: int
    host_poll_count: int
    host_synchronize_count: int
    publication_trace_sha256: str

    @classmethod
    def from_payload(cls, payload: object) -> "ShippingPublicationReceipt":
        expected = set(cls.__dataclass_fields__)
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("shipping publication receipt fields are invalid")
        if payload.get("publication_schema_version") != SHIPPING_PUBLICATION_RECEIPT_SCHEMA_VERSION:
            raise ValueError("shipping publication receipt schema version is invalid")
        if payload.get("input_contract") != GPU_ENGINE_PUBLISHED_FRAME_INPUT:
            raise ValueError("shipping publication receipt input contract is invalid")
        for name in (
            "physics_run_id",
            "publish_event_token",
            "consumer_binding_id",
            "consumer_stream_token",
            "last_physics_buffer_reader_operation",
            "consumer_done_event_token",
        ):
            if not isinstance(payload.get(name), str) or not payload[name]:
                raise ValueError(f"shipping publication receipt {name} must be non-empty")
        for name in (
            "physics_step_index",
            "physics_frame_id",
            "physics_slot_id",
            "physics_device_ordinal",
            "physics_step_count",
            "borrow_count",
            "replay_source_count",
            "host_poll_count",
            "host_synchronize_count",
        ):
            value = payload.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"shipping publication receipt {name} must be a non-negative integer")
        for name in (
            "publish_event_recorded",
            "producer_publish_completion_recorded",
            "consumer_stream_wait_enqueued",
            "zero_copy_transform_binding",
            "consumer_completion_recorded",
            "slot_logically_reclaimable",
            "producer_reuse_wait_enqueued_when_reused",
            "done_event_retained_until_reuse_dependency",
            "final_payload_drain_complete",
            "final_reclaim_complete",
        ):
            if not isinstance(payload.get(name), bool):
                raise ValueError(f"shipping publication receipt {name} must be boolean")
        physical = payload.get("physical_reclaim_evidence")
        if physical not in {"producer_wait_enqueued_before_reuse", "done_event_retained_no_reuse"}:
            raise ValueError("shipping publication physical_reclaim_evidence is invalid")
        if physical == "producer_wait_enqueued_before_reuse" and not payload.get(
            "producer_reuse_wait_enqueued_when_reused"
        ):
            raise ValueError("shipping publication producer reuse wait evidence is inconsistent")
        if not payload.get("done_event_retained_until_reuse_dependency"):
            raise ValueError("shipping publication done event retention must be proven")
        if not _sha256_text(payload.get("publication_trace_sha256")):
            raise ValueError("shipping publication trace digest is invalid")
        return cls(**{name: payload[name] for name in expected})

    def to_payload(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_payload())


def _sha256_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _storage_sequence(payload: object, *, label: str) -> tuple[NativeStorageSignature, ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError(f"native stage {label} storage must be an array")
    values = tuple(NativeStorageSignature.from_payload(value) for value in payload)
    if len({value.name for value in values}) != len(values):
        raise ValueError(f"native stage {label} storage names must be unique")
    return values


def _validate_stage_dataflow(
    *,
    boundary: str,
    lane: str,
    renderer_id: str,
    acceleration_class: str,
    acceleration_id: str,
    output_profile: str,
    shadows: bool,
    dataflow_contract_id: str,
    runtime_scene_storage: tuple[NativeStorageSignature, ...],
    derived_fixture_storage: tuple[NativeStorageSignature, ...],
    ordered_stage_evidence: tuple[NativeStageBinding, ...],
) -> None:
    """Validate concrete storage arrows, not merely their aggregate digest."""

    initial = runtime_scene_storage + derived_fixture_storage
    if len({value.name for value in initial}) != len(initial):
        raise ValueError("native execution binding initial storage names overlap")
    evidence = tuple(stage.evidence for stage in ordered_stage_evidence)
    if boundary == "R0":
        render = evidence[0]
        expected_renderer = EXACT_LANE_BINDINGS[lane]["renderer_id"]
        if renderer_id != expected_renderer or render.operation_id != expected_renderer:
            raise ValueError("R0 render operation does not match its lane renderer")
        if acceleration_class != "fixed_bvh_fixture":
            raise ValueError("R0 acceleration class must be fixed_bvh_fixture")
        expected_acceleration_id = EXACT_LANE_BINDINGS[lane]["acceleration_by_class"][acceleration_class]
        if acceleration_id != expected_acceleration_id:
            raise ValueError("R0 acceleration identity does not match its class")
        if render.output_profile != output_profile or render.shadows != shadows:
            raise ValueError("R0 render stage identity does not match its binding")
        expected_contract_id = canonical_dataflow_contract_id(
            boundary=boundary,
            lane=lane,
            renderer_id=renderer_id,
            acceleration_class=acceleration_class,
            acceleration_id=acceleration_id,
            output_profile=output_profile,
            shadows=shadows,
        )
        if dataflow_contract_id != expected_contract_id:
            raise ValueError("R0 dataflow contract identity is invalid")
        if render.input_storage != initial:
            raise ValueError("R0 render must read the exact scene/fixture storage union")
        if not render.output_storage:
            raise ValueError("R0 render output storage is empty")
        return
    if boundary not in {"R1", "R2"}:
        return

    scene, acceleration, render = evidence[:3]
    expected_lane_binding = EXACT_LANE_BINDINGS[lane]
    expected_acceleration_id = expected_lane_binding["acceleration_by_class"].get(acceleration_class)
    if expected_acceleration_id is None:
        raise ValueError("dynamic acceleration class is not defined for its lane")
    acceleration_contract = EXACT_ACCELERATION_CONTRACTS[lane][expected_acceleration_id]
    if renderer_id != expected_lane_binding["renderer_id"]:
        raise ValueError("dynamic renderer identity does not match its lane")
    if acceleration_id != expected_acceleration_id:
        raise ValueError("dynamic acceleration identity does not match its lane")
    expected_operations = (
        acceleration_contract["timed_scene_update_operation"],
        acceleration_contract["timed_acceleration_operation"],
        expected_lane_binding["renderer_id"],
    )
    if tuple(stage.operation_id for stage in (scene, acceleration, render)) != expected_operations:
        raise ValueError("dynamic stage operation sequence is not the frozen lane contract")
    if render.output_profile != output_profile or render.shadows != shadows:
        raise ValueError("dynamic render stage identity does not match its binding")
    expected_contract_id = canonical_dataflow_contract_id(
        boundary=boundary,
        lane=lane,
        renderer_id=renderer_id,
        acceleration_class=acceleration_class,
        acceleration_id=acceleration_id,
        output_profile=output_profile,
        shadows=shadows,
    )
    if dataflow_contract_id != expected_contract_id:
        raise ValueError("dynamic dataflow contract identity is invalid")

    runtime_by_name = _storage_by_name(runtime_scene_storage, label="runtime scene")
    expected_transform = _require_named_storage(
        scene.input_storage,
        DYNAMIC_TRANSFORM_INPUT_NAMES,
        label="dynamic scene transform input",
        allow_other_names=True,
    )
    expected_scene_input = (
        _select_storage(
            runtime_by_name,
            DYNAMIC_SCENE_UPDATE_RUNTIME_INPUT_NAMES,
            label="dynamic scene runtime input",
        )
        + expected_transform
    )
    _require_exact_storage(
        scene.input_storage,
        expected_scene_input,
        label="dynamic scene-update input",
    )
    _require_exact_storage_names(
        scene.output_storage,
        DYNAMIC_SCENE_OUTPUT_NAMES,
        label="dynamic scene-update output",
    )
    scene_by_name = _storage_by_name(scene.output_storage, label="dynamic scene output")
    expected_acceleration_input = _select_storage(
        scene_by_name,
        DYNAMIC_ACCELERATION_INPUT_NAMES,
        label="dynamic acceleration input",
    )
    _require_exact_storage(
        acceleration.input_storage,
        expected_acceleration_input,
        label="dynamic acceleration input",
    )
    _require_exact_storage_names(
        acceleration.output_storage,
        DYNAMIC_ACCELERATION_OUTPUT_NAMES,
        label="dynamic acceleration output",
    )
    acceleration_by_name = _storage_by_name(acceleration.output_storage, label="dynamic acceleration output")
    render_names = (
        DYNAMIC_RENDER_WORLD_INPUT_NAMES
        + DYNAMIC_RENDER_METADATA_INPUT_NAMES
        + DYNAMIC_ACCELERATION_OUTPUT_NAMES
        + DYNAMIC_RENDER_SHADE_INPUT_NAMES
    )
    combined_by_name = runtime_by_name | scene_by_name | acceleration_by_name
    expected_render_input = _select_storage(
        combined_by_name,
        render_names,
        label="dynamic render input",
    )
    _require_exact_storage(
        render.input_storage,
        expected_render_input,
        label="dynamic render input",
    )
    if not render.output_storage:
        raise ValueError("dynamic render output storage is empty")

    available = runtime_scene_storage + scene.output_storage + acceleration.output_storage
    available += render.output_storage
    for later in evidence[3:]:
        if any(value not in available for value in later.input_storage):
            raise ValueError(f"native stage {later.stage_name} reads unbound storage")
        if not later.output_storage:
            raise ValueError(f"native stage {later.stage_name} output storage is empty")
        available += later.output_storage


def _storage_by_name(
    storage: Sequence[NativeStorageSignature], *, label: str
) -> dict[str, NativeStorageSignature]:
    result = {value.name: value for value in storage}
    if len(result) != len(storage):
        raise ValueError(f"{label} storage names must be unique")
    return result


def _select_storage(
    storage_by_name: Mapping[str, NativeStorageSignature],
    names: Sequence[str],
    *,
    label: str,
) -> tuple[NativeStorageSignature, ...]:
    missing = tuple(name for name in names if name not in storage_by_name)
    if missing:
        raise ValueError(f"{label} is missing required storage {missing!r}")
    return tuple(storage_by_name[name] for name in names)


def _require_exact_storage(
    actual: Sequence[NativeStorageSignature],
    expected: Sequence[NativeStorageSignature],
    *,
    label: str,
) -> None:
    if tuple(actual) != tuple(expected):
        raise ValueError(f"{label} must match the complete ordered storage contract")


def _require_exact_storage_names(
    actual: Sequence[NativeStorageSignature],
    expected_names: Sequence[str],
    *,
    label: str,
) -> None:
    if tuple(value.name for value in actual) != tuple(expected_names):
        raise ValueError(f"{label} must match the complete ordered storage-name contract")


def _require_named_storage(
    actual: Sequence[NativeStorageSignature],
    expected_names: Sequence[str],
    *,
    label: str,
    allow_other_names: bool = False,
) -> tuple[NativeStorageSignature, ...]:
    by_name = _storage_by_name(actual, label=label)
    if not allow_other_names and tuple(by_name) != tuple(expected_names):
        raise ValueError(f"{label} contains missing or extra storage")
    return _select_storage(by_name, expected_names, label=label)


def dynamic_scene_update_input_storage(
    runtime_scene_storage: Sequence[NativeStorageSignature],
    transform_storage: Sequence[NativeStorageSignature],
) -> tuple[NativeStorageSignature, ...]:
    """Return the exact ordered C3 scene-update input allocation tuple."""

    runtime_by_name = _storage_by_name(runtime_scene_storage, label="runtime scene")
    transform_by_name = _storage_by_name(transform_storage, label="replay transform")
    return _select_storage(
        runtime_by_name,
        DYNAMIC_SCENE_UPDATE_RUNTIME_INPUT_NAMES,
        label="dynamic scene runtime input",
    ) + _select_storage(
        transform_by_name,
        DYNAMIC_TRANSFORM_INPUT_NAMES,
        label="dynamic scene transform input",
    )


def canonical_dataflow_contract_id(
    *,
    boundary: str,
    lane: str,
    renderer_id: str,
    acceleration_class: str,
    acceleration_id: str,
    output_profile: str,
    shadows: bool,
) -> str:
    """Hash the frozen stage/operation/profile identity, independent of storage tokens."""

    identity_sha256 = _canonical_sha256(
        {
            "boundary": boundary,
            "lane": lane,
            "renderer_id": renderer_id,
            "acceleration_class": acceleration_class,
            "acceleration_id": acceleration_id,
            "output_profile": output_profile,
            "shadows": shadows,
        }
    )
    return f"{DYNAMIC_DATAFLOW_CONTRACT_SCHEMA_VERSION}:{identity_sha256}"


def canonical_storage_manifest_sha256(values: Sequence[NativeStorageSignature]) -> str:
    return _canonical_sha256([value.to_payload() for value in values])


def canonical_dataflow_binding_sha256(
    *,
    runtime_scene_storage: Sequence[NativeStorageSignature],
    derived_fixture_storage: Sequence[NativeStorageSignature],
    ordered_stage_evidence: Sequence[NativeStageBinding],
) -> str:
    return _canonical_sha256(
        {
            "runtime_scene_storage": [value.to_payload() for value in runtime_scene_storage],
            "derived_fixture_storage": [value.to_payload() for value in derived_fixture_storage],
            "ordered_stage_evidence": [value.to_payload() for value in ordered_stage_evidence],
        }
    )


def build_c3_one_shot_native_execution_binding(
    *,
    row_id: str,
    case_id: str,
    case_sha256: str,
    boundary: str,
    lane: str,
    renderer_id: str,
    acceleration_class: str,
    acceleration_id: str,
    output_profile: str,
    shadows: bool,
    input_contract: str,
    native_input_manifest_sha256: str,
    native_readiness_sha256: str,
    scene_contract_sha256: str,
    transform_replay_sha256: str | None,
    transform_frame_index: int | None,
    fixed_render_fixture_sha256: str | None,
    runtime_scene_storage: Sequence[NativeStorageSignature],
    derived_fixture_storage: Sequence[NativeStorageSignature],
    stage_evidence: Sequence[NativeStageEvidence],
    bound_render_parameters: object,
) -> C3OneShotNativeExecutionBindingEvidence:
    """Build C3 one-shot evidence; this constructor can never authorize results."""

    runtime_storage = tuple(runtime_scene_storage)
    derived_storage = tuple(derived_fixture_storage)
    stages = tuple(
        NativeStageBinding(
            stage_name=evidence.stage_name,
            evidence=evidence,
            evidence_sha256=evidence.sha256,
        )
        for evidence in stage_evidence
    )
    binding = C3OneShotNativeExecutionBindingEvidence(
        schema_version=NATIVE_EXECUTION_BINDING_SCHEMA_VERSION,
        row_id=row_id,
        case_id=case_id,
        case_sha256=case_sha256,
        boundary=boundary,
        lane=lane,
        renderer_id=renderer_id,
        acceleration_class=acceleration_class,
        acceleration_id=acceleration_id,
        output_profile=output_profile,
        shadows=shadows,
        dataflow_contract_id=canonical_dataflow_contract_id(
            boundary=boundary,
            lane=lane,
            renderer_id=renderer_id,
            acceleration_class=acceleration_class,
            acceleration_id=acceleration_id,
            output_profile=output_profile,
            shadows=shadows,
        ),
        input_contract=input_contract,
        native_input_manifest_sha256=native_input_manifest_sha256,
        native_readiness_sha256=native_readiness_sha256,
        scene_contract_sha256=scene_contract_sha256,
        transform_replay_sha256=transform_replay_sha256,
        transform_frame_index=transform_frame_index,
        fixed_render_fixture_sha256=fixed_render_fixture_sha256,
        runtime_scene_storage=runtime_storage,
        runtime_scene_storage_sha256=canonical_storage_manifest_sha256(runtime_storage),
        derived_fixture_storage=derived_storage,
        derived_fixture_storage_sha256=(
            canonical_storage_manifest_sha256(derived_storage) if derived_storage else None
        ),
        ordered_stage_evidence=stages,
        dataflow_binding_sha256=canonical_dataflow_binding_sha256(
            runtime_scene_storage=runtime_storage,
            derived_fixture_storage=derived_storage,
            ordered_stage_evidence=stages,
        ),
        bound_render_parameters=bound_render_parameters,
        bound_render_parameters_sha256=bound_render_parameters.sha256,
    )
    return C3OneShotNativeExecutionBindingEvidence.from_payload(binding.to_payload())
