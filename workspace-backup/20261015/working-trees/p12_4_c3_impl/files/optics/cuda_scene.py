"""Torch-owned CUDA materialization of the canonical optical scene contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from .scene_contract import PackedOpticalSceneHost


@dataclass(frozen=True)
class CudaOpticalLocalScene:
    """Torch-owned local scene arrays used by the native CUDA data plane."""

    device_ordinal: int
    arrays: Mapping[str, Any]
    role_to_mask: Mapping[str, int]
    num_instances: int
    num_materials: int
    num_lights: int
    num_triangles: int
    num_planes: int
    max_body_index: int
    scene_contract_sha256: str
    ready_event: Any | None
    resources: tuple[object, ...]


def build_cuda_optical_scene(
    packed_scene: PackedOpticalSceneHost,
    *,
    device: object = "cuda:0",
) -> CudaOpticalLocalScene:
    """Independently upload a packed host scene into Torch CUDA allocations."""

    import torch

    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("CUDA optical scene requires a CUDA device")
    ordinal = torch.cuda.current_device() if target.index is None else int(target.index)
    target = torch.device("cuda", ordinal)
    arrays: dict[str, Any] = {}
    with torch.cuda.device(target):
        stream = torch.cuda.current_stream(target)
        for name in sorted(packed_scene.arrays):
            arrays[name] = torch.tensor(
                np.array(packed_scene.arrays[name], copy=True, order="C"),
                device=target,
            ).contiguous()
        ready_event = torch.cuda.Event(enable_timing=False)
        ready_event.record(stream)
    return cuda_optical_scene_from_arrays(
        packed_scene,
        arrays,
        device_ordinal=ordinal,
        ready_event=ready_event,
        owner=(),
    )


def cuda_optical_scene_from_arrays(
    packed_scene: PackedOpticalSceneHost,
    arrays: Mapping[str, Any],
    *,
    device_ordinal: int,
    ready_event: Any | None,
    owner: tuple[object, ...],
) -> CudaOpticalLocalScene:
    """Wrap exact caller-owned Torch arrays without allocation or copying."""

    import torch

    if not isinstance(device_ordinal, int) or isinstance(device_ordinal, bool) or device_ordinal < 0:
        raise ValueError("device_ordinal must be a non-negative integer")
    if set(arrays) != set(packed_scene.arrays):
        raise ValueError("CUDA scene arrays do not match the packed scene contract")
    selected: dict[str, Any] = {}
    for name in sorted(packed_scene.arrays):
        tensor = arrays[name]
        host = packed_scene.arrays[name]
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
            raise ValueError(f"CUDA scene array {name!r} must be a Torch CUDA tensor")
        if tensor.device.index != device_ordinal:
            raise ValueError(f"CUDA scene array {name!r} is on the wrong device")
        if tuple(tensor.shape) != tuple(host.shape) or not tensor.is_contiguous():
            raise ValueError(f"CUDA scene array {name!r} shape/layout does not match")
        if _torch_dtype_numpy(torch, tensor.dtype) != host.dtype:
            raise ValueError(f"CUDA scene array {name!r} dtype does not match")
        selected[name] = tensor
    return CudaOpticalLocalScene(
        device_ordinal=device_ordinal,
        arrays=MappingProxyType(selected),
        role_to_mask=packed_scene.role_to_mask,
        num_instances=packed_scene.num_instances,
        num_materials=packed_scene.num_materials,
        num_lights=packed_scene.num_lights,
        num_triangles=packed_scene.num_triangles,
        num_planes=packed_scene.num_planes,
        max_body_index=packed_scene.max_body_index,
        scene_contract_sha256=packed_scene.scene_contract_sha256,
        ready_event=ready_event,
        resources=tuple(owner) + (packed_scene,),
    )


def _torch_dtype_numpy(torch: Any, dtype: Any) -> np.dtype[Any]:
    mapping = {
        torch.bool: np.dtype(np.bool_),
        torch.int8: np.dtype(np.int8),
        torch.int16: np.dtype(np.int16),
        torch.int32: np.dtype(np.int32),
        torch.int64: np.dtype(np.int64),
        torch.uint8: np.dtype(np.uint8),
        torch.float16: np.dtype(np.float16),
        torch.float32: np.dtype(np.float32),
        torch.float64: np.dtype(np.float64),
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported CUDA scene Torch dtype {dtype}") from exc
