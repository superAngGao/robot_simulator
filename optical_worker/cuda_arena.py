"""Production C3 binding over the exact sealed CUDA arena allocations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from optical_worker.render_parameters import bind_render_parameters, load_frozen_render_sequence
from optics.cuda_arena import CudaArenaPlanTarget

from .native_contract import (
    CudaPreparedNativeInput,
    NativeReadinessEvidence,
)
from .native_input import (
    assert_cuda_arena_a2_authority,
    prepared_storage_signatures,
    validate_prepared_native_input,
)
from .provenance import (
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
    _register_cuda_arena_c3_binding_authority,
    build_c3_one_shot_native_execution_binding,
)


def build_cuda_c3_binding_from_arena_target(
    target: CudaArenaPlanTarget,
    *,
    prepared_input: CudaPreparedNativeInput,
    native_readiness: NativeReadinessEvidence,
    _captured_session=None,
):
    """Bind exact a2 allocations/readiness to one immutable C4 target."""

    if not isinstance(target, CudaArenaPlanTarget):
        raise TypeError("CUDA C3 arena adapter requires CudaArenaPlanTarget")
    assert_cuda_arena_a2_authority(prepared_input, native_readiness)
    if prepared_input.source is None:
        raise ValueError("CUDA C3 arena adapter requires the canonical a2 source owner")
    if _captured_session is None:
        validate_prepared_native_input(prepared_input.source, prepared_input)
    else:
        from .native_setup import _require_final_readiness_origin

        _require_final_readiness_origin(_captured_session, prepared_input, native_readiness)
    _validate_target_a2_allocations(target, prepared_input)
    spec = target._spec
    if (
        prepared_input.manifest.sha256 != spec.native_input_manifest_sha256
        or native_readiness.sha256 != spec.native_readiness_sha256
    ):
        raise ValueError("CUDA C3 arena a2 authority does not match the target specification")
    slot_owner = target.evidence.slot_storage_manifest_sha256
    runtime_evidence = target._runtime_stage_evidence
    render_abi = runtime_evidence[0 if spec.boundary == "R0" else 2]
    bound = bind_render_parameters(
        load_frozen_render_sequence(spec.case_id),
        sample_index=(target.evidence.render_sample_index
                      if target.evidence.render_sample_index is not None
                      else target.evidence.transform_frame_index or 0),
        profile=spec.output_profile,
        shadows=spec.shadows,
        pass_mode=spec.diagnostic_mode,
        role_to_mask=prepared_input.source.packed_scene_owner.role_to_mask,
    )
    render_abi.validate_render_parameters(bound.to_payload())

    if spec.boundary == "R0":
        render_evidence = runtime_evidence[0]
        runtime_scene = prepared_storage_signatures(
            prepared_input,
            tuple(target._runtime_scene_arrays),
        )
        derived_fixture = prepared_storage_signatures(
            prepared_input,
            tuple(target._derived_fixture_arrays),
        )
        stage_evidence = (
            _stage_evidence(
                name="render",
                bound_render_parameters=bound,
                operation_id=spec.renderer_id,
                target=target,
                inputs=runtime_scene + derived_fixture,
                outputs=_render_output_signatures(target, slot_owner),
                dispatch_plan_sha256=_canonical_sha256(_plain_payload(render_evidence)),
            ),
        )
    else:
        scene_evidence, acceleration_evidence, render_evidence = runtime_evidence[:3]
        runtime_scene = prepared_storage_signatures(
            prepared_input,
            tuple(target._runtime_scene_arrays),
        )
        runtime_by_name = {value.name: value for value in runtime_scene}
        transform_by_name = {
            name: _storage_signature(
                name,
                target._transform_arrays[name],
                owner_token=spec.native_input_manifest_sha256,
            )
            for name in DYNAMIC_TRANSFORM_INPUT_NAMES
        }
        scene_outputs = tuple(
            _storage_signature(name, tensor, owner_token=slot_owner)
            for name, tensor in target._scene_output_arrays.items()
        )
        scene_by_name = {value.name: value for value in scene_outputs}
        acceleration_outputs = _ordered_signatures(
            target._acceleration_output_arrays,
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
                operation_id=scene_evidence.operation_id,
                target=target,
                inputs=scene_inputs,
                outputs=scene_outputs,
                dispatch_plan_sha256=scene_evidence.dispatch_plan_sha256,
            ),
            _stage_evidence(
                name="acceleration",
                operation_id=acceleration_evidence.operation_id,
                target=target,
                inputs=acceleration_inputs,
                outputs=acceleration_outputs,
                dispatch_plan_sha256=acceleration_evidence.dispatch_plan_sha256,
            ),
            _stage_evidence(
                name="render",
                bound_render_parameters=bound,
                operation_id=spec.renderer_id,
                target=target,
                inputs=render_inputs,
                outputs=_render_output_signatures(target, slot_owner),
                dispatch_plan_sha256=_canonical_sha256(_plain_payload(render_evidence)),
            ),
        )
        derived_fixture = ()

    binding = build_c3_one_shot_native_execution_binding(
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
        bound_render_parameters=bound,
    )
    _register_cuda_arena_c3_binding_authority(
        binding,
        target=target,
        prepared_input=prepared_input,
        native_readiness=native_readiness,
    )
    return binding


def _stage_evidence(
    *,
    name: str,
    operation_id: str,
    target: CudaArenaPlanTarget,
    inputs: tuple[NativeStorageSignature, ...],
    outputs: tuple[NativeStorageSignature, ...],
    dispatch_plan_sha256: str,
    bound_render_parameters=None,
) -> NativeStageEvidence:
    spec = target._spec
    return NativeStageEvidence(
        schema_version=NATIVE_STAGE_EVIDENCE_SCHEMA_VERSION,
        stage_name=name,
        bound_render_parameters=bound_render_parameters,
        bound_render_parameters_sha256=(
            None if bound_render_parameters is None else bound_render_parameters.sha256
        ),
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


def _validate_target_a2_allocations(
    target: CudaArenaPlanTarget,
    prepared_input: CudaPreparedNativeInput,
) -> None:
    manifest = prepared_input.manifest
    prepared = prepared_input.arrays
    if target._spec.boundary == "R0":
        _require_exact_objects(
            target._runtime_scene_arrays,
            prepared,
            tuple(manifest.scene_array_names),
            label="packed scene",
        )
        _require_exact_objects(
            target._derived_fixture_arrays,
            prepared,
            tuple(manifest.fixed_render_fixture_arrays),
            label="fixed fixture",
        )
        return

    _require_exact_objects(
        target._runtime_scene_arrays,
        prepared,
        tuple(manifest.scene_array_names),
        label="packed scene",
    )
    frame_index = target.evidence.transform_frame_index
    if frame_index is None:
        raise ValueError("dynamic CUDA C3 target lacks a replay frame")
    for name in DYNAMIC_TRANSFORM_INPUT_NAMES:
        expected = prepared[name][frame_index : frame_index + 1]
        actual = target._transform_arrays.get(name)
        if actual is None or _tensor_identity(actual) != _tensor_identity(expected):
            raise ValueError(f"CUDA C3 target {name!r} is not the approved a2 replay view")


def _require_exact_objects(
    actual: Mapping[str, object],
    prepared: Mapping[str, object],
    names: tuple[str, ...],
    *,
    label: str,
) -> None:
    if set(actual) != set(names):
        raise ValueError(f"CUDA C3 target {label} allocation names are not exact")
    for name in names:
        if prepared.get(name) is not actual[name]:
            raise ValueError(f"CUDA C3 target {label} {name!r} is not the exact a2 allocation")


def _tensor_identity(tensor: Any) -> tuple[object, ...]:
    storage = tensor.untyped_storage()
    base = int(storage.data_ptr()) if storage.nbytes() else 0
    return (
        int(tensor.device.index),
        str(tensor.dtype),
        tuple(int(value) for value in tensor.shape),
        tuple(int(value) for value in tensor.stride()),
        base,
        int(storage.nbytes()),
        int(tensor.data_ptr()) - base if storage.nbytes() else 0,
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
