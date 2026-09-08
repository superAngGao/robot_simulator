"""C3 evidence adapter for the sealed C4b CUDA arena graph.

The durable optical implementation does not import the benchmark package.
This adapter is the one direction in which benchmark provenance may inspect a
sealed implementation graph and construct the canonical, non-result-capable
C3 receipt consumed by that graph's C4 preflight.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from optics.cuda_arena import CudaArenaPlanTarget

from .receipts import (
    DYNAMIC_ACCELERATION_INPUT_NAMES,
    DYNAMIC_ACCELERATION_OUTPUT_NAMES,
    DYNAMIC_RENDER_METADATA_INPUT_NAMES,
    DYNAMIC_RENDER_SHADE_INPUT_NAMES,
    DYNAMIC_RENDER_WORLD_INPUT_NAMES,
    DYNAMIC_SCENE_UPDATE_RUNTIME_INPUT_NAMES,
    DYNAMIC_TRANSFORM_INPUT_NAMES,
    NATIVE_STAGE_EVIDENCE_SCHEMA_VERSION,
    NativeStageEvidence,
    NativeStorageSignature,
    build_c3_one_shot_native_execution_binding,
)


def build_cuda_c3_binding_from_arena_target(target: CudaArenaPlanTarget):
    """Build C3 evidence from the target's actual immutable CUDA allocations."""

    if not isinstance(target, CudaArenaPlanTarget):
        raise TypeError("CUDA C3 arena adapter requires CudaArenaPlanTarget")
    target._transaction_template.validate_bound()
    spec = target._spec
    session_owner = spec.native_input_manifest_sha256
    slot_owner = target.evidence.slot_storage_manifest_sha256
    stages = target._transaction_template.stages

    if spec.boundary == "R0":
        render = stages[0]
        render_evidence = render._cuda_transaction_stage_evidence()
        runtime_scene = tuple(
            _storage_signature(name, tensor, owner_token=session_owner)
            for name, tensor in target._runtime_scene_arrays.items()
        )
        derived_fixture = tuple(
            _storage_signature(name, tensor, owner_token=session_owner)
            for name, tensor in target._derived_fixture_arrays.items()
        )
        stage_evidence = (
            _stage_evidence(
                name="render",
                operation_id=spec.renderer_id,
                target=target,
                inputs=runtime_scene + derived_fixture,
                outputs=_render_output_signatures(target, slot_owner),
                dispatch_plan_sha256=_canonical_sha256(_plain_payload(render_evidence)),
            ),
        )
    else:
        scene, acceleration, render = stages[:3]
        render_evidence = render._cuda_transaction_stage_evidence()
        runtime_scene = tuple(
            _storage_signature(name, tensor, owner_token=session_owner)
            for name, tensor in target._runtime_scene_arrays.items()
        )
        runtime_by_name = {value.name: value for value in runtime_scene}
        transform_by_name = {
            "replay_x_world_R": _storage_signature(
                "replay_x_world_R", scene.transforms.x_world_R, owner_token=session_owner
            ),
            "replay_x_world_r": _storage_signature(
                "replay_x_world_r", scene.transforms.x_world_r, owner_token=session_owner
            ),
        }
        scene_outputs = tuple(
            _storage_signature(name, tensor, owner_token=slot_owner)
            for name, tensor in scene.outputs.tensor_items()
        )
        scene_by_name = {value.name: value for value in scene_outputs}
        acceleration_tensors = {
            "bvh_bounds_min": acceleration.outputs.node_bounds_min,
            "bvh_bounds_max": acceleration.outputs.node_bounds_max,
            "bvh_left": acceleration.outputs.left,
            "bvh_right": acceleration.outputs.right,
            "bvh_start": acceleration.outputs.start,
            "bvh_count": acceleration.outputs.count,
            "bvh_prim_ids": acceleration.outputs.sorted_primitive_ids,
        }
        acceleration_outputs = _ordered_signatures(
            acceleration_tensors,
            DYNAMIC_ACCELERATION_OUTPUT_NAMES,
            slot_owner,
        )
        acceleration_by_name = {value.name: value for value in acceleration_outputs}
        scene_inputs = tuple(
            runtime_by_name[name] for name in DYNAMIC_SCENE_UPDATE_RUNTIME_INPUT_NAMES
        ) + tuple(transform_by_name[name] for name in DYNAMIC_TRANSFORM_INPUT_NAMES)
        acceleration_inputs = tuple(scene_by_name[name] for name in DYNAMIC_ACCELERATION_INPUT_NAMES)
        render_names = (
            DYNAMIC_RENDER_WORLD_INPUT_NAMES
            + DYNAMIC_RENDER_METADATA_INPUT_NAMES
            + DYNAMIC_ACCELERATION_OUTPUT_NAMES
            + DYNAMIC_RENDER_SHADE_INPUT_NAMES
        )
        render_inputs_by_name = runtime_by_name | scene_by_name | acceleration_by_name
        render_inputs = tuple(render_inputs_by_name[name] for name in render_names)
        stage_evidence = (
            _stage_evidence(
                name="scene_update",
                operation_id=scene.evidence.operation_id,
                target=target,
                inputs=scene_inputs,
                outputs=scene_outputs,
                dispatch_plan_sha256=scene.evidence.dispatch_plan_sha256,
            ),
            _stage_evidence(
                name="acceleration",
                operation_id=acceleration.evidence.operation_id,
                target=target,
                inputs=acceleration_inputs,
                outputs=acceleration_outputs,
                dispatch_plan_sha256=acceleration.evidence.dispatch_plan_sha256,
            ),
            _stage_evidence(
                name="render",
                operation_id=spec.renderer_id,
                target=target,
                inputs=render_inputs,
                outputs=_render_output_signatures(target, slot_owner),
                dispatch_plan_sha256=_canonical_sha256(_plain_payload(render_evidence)),
            ),
        )
        derived_fixture = ()

    return build_c3_one_shot_native_execution_binding(
        row_id=spec.row_id,
        case_id=spec.case_id,
        case_sha256=spec.case_sha256,
        boundary=spec.boundary,
        lane="cuda",
        renderer_id=spec.renderer_id,
        acceleration_class=spec.acceleration_class,
        acceleration_id=spec.acceleration_id,
        output_profile=spec.output_profile,
        shadows=spec.shadows,
        input_contract=spec.input_contract,
        native_input_manifest_sha256=spec.native_input_manifest_sha256,
        native_readiness_sha256=spec.native_readiness_sha256,
        scene_contract_sha256=spec.scene_contract_sha256,
        transform_replay_sha256=spec.transform_replay_sha256,
        transform_frame_index=target.evidence.transform_frame_index,
        fixed_render_fixture_sha256=spec.fixed_render_fixture_sha256,
        runtime_scene_storage=runtime_scene,
        derived_fixture_storage=derived_fixture,
        stage_evidence=stage_evidence,
    )


def _stage_evidence(
    *,
    name: str,
    operation_id: str,
    target: CudaArenaPlanTarget,
    inputs: tuple[NativeStorageSignature, ...],
    outputs: tuple[NativeStorageSignature, ...],
    dispatch_plan_sha256: str,
) -> NativeStageEvidence:
    spec = target._spec
    return NativeStageEvidence(
        schema_version=NATIVE_STAGE_EVIDENCE_SCHEMA_VERSION,
        stage_name=name,
        operation_id=operation_id,
        output_profile=spec.output_profile if name == "render" else None,
        shadows=spec.shadows if name == "render" else None,
        scene_contract_sha256=spec.scene_contract_sha256,
        input_contract=spec.input_contract,
        transform_replay_sha256=spec.transform_replay_sha256,
        transform_frame_index=target.evidence.transform_frame_index,
        fixed_render_fixture_sha256=spec.fixed_render_fixture_sha256,
        topology_sha256=None,
        input_storage=inputs,
        output_storage=outputs,
        dispatch_plan_sha256=dispatch_plan_sha256,
    )


def _render_output_signatures(
    target: CudaArenaPlanTarget,
    owner_token: str,
) -> tuple[NativeStorageSignature, ...]:
    return tuple(
        _storage_signature(name, tensor, owner_token=owner_token)
        for name, tensor in target._render_buffers.tensor_items()
    )


def _ordered_signatures(
    tensors: Mapping[str, object],
    names: tuple[str, ...],
    owner_token: str,
) -> tuple[NativeStorageSignature, ...]:
    missing = tuple(name for name in names if name not in tensors)
    if missing:
        raise ValueError(f"CUDA arena is missing C3 storage {missing!r}")
    return tuple(_storage_signature(name, tensors[name], owner_token=owner_token) for name in names)


def _storage_signature(
    name: str,
    tensor: Any,
    *,
    owner_token: str,
) -> NativeStorageSignature:
    storage = tensor.untyped_storage()
    storage_nbytes = int(storage.nbytes())
    base_pointer = int(storage.data_ptr()) if storage_nbytes else 0
    active_start = int(tensor.data_ptr()) - base_pointer if storage_nbytes else 0
    dtype = {
        "torch.bool": "|b1",
        "torch.uint8": "|u1",
        "torch.int32": "<i4",
        "torch.int64": "<i8",
        "torch.float32": "<f4",
        "torch.float64": "<f8",
    }.get(str(tensor.dtype))
    if dtype is None:
        raise ValueError(f"CUDA arena C3 storage {name!r} has unsupported dtype")
    token = f"cuda:0x{base_pointer:x}" if storage_nbytes else f"cuda:empty:{name}:{owner_token}"
    return NativeStorageSignature(
        name=name,
        backend="cuda",
        device_ordinal=int(tensor.device.index),
        storage_token=token,
        dtype=dtype,
        shape=tuple(int(value) for value in tensor.shape),
        nbytes=storage_nbytes,
        active_start=active_start,
        active_count=int(tensor.numel()) * int(tensor.element_size()),
        owner_token=owner_token,
    )


def _plain_payload(value: object) -> object:
    fields = getattr(value, "__dataclass_fields__", None)
    if fields is not None:
        return {name: _plain_payload(getattr(value, name)) for name in fields}
    if isinstance(value, tuple):
        return [_plain_payload(item) for item in value]
    return value


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
