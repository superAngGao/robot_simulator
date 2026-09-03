"""Caller-owned native CUDA LBVH rebuild and legacy compatibility facade."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from functools import lru_cache

from ._cuda_lbvh_source import CPP_SOURCE, CUDA_SOURCE
from .cuda_scene import CudaWorldSceneBuffers
from .device_bvh import DeviceBvhBuildStats, DeviceOpticalBvh
from .device_scene import DeviceOpticalSceneSnapshot

try:  # pragma: no cover - exercised in CUDA extension environments.
    import torch
    from torch.utils.cpp_extension import load_inline

    _HAS_TORCH_CUDA_LBVH_DEPS = True
except Exception:  # pragma: no cover - keep CPU-only imports working.
    torch = None
    load_inline = None
    _HAS_TORCH_CUDA_LBVH_DEPS = False

try:  # pragma: no cover - compatibility facade only.
    import warp as wp

    _HAS_WARP_COMPAT = True
except Exception:  # pragma: no cover - core CUDA path does not require Warp.
    wp = None
    _HAS_WARP_COMPAT = False


CUDA_LBVH_ACCELERATION_ID = "cuda_morton_karras_lbvh_rebuild_v1"
CUDA_LBVH_OPERATION_ID = "cuda_lbvh_scene_bounds_morton_sort_topology_bounds_out_v1"
CUDA_LBVH_TOPOLOGY_ORIGIN_ID = "cuda_morton30_karras_source_index_tiebreak_v1"
CUDA_LBVH_TOPOLOGY_LIFETIME = "transaction_output_arena_lease_v1"
CUDA_LBVH_EXTENSION_ABI_VERSION = "q54.p12.4.cuda-lbvh-extension.v3"
CUDA_LBVH_WORKSPACE_SCHEMA_VERSION = "q54.p12.4.cuda-lbvh-workspace.v1"
CUDA_LBVH_STAGE_SCHEMA_VERSION = "q54.p12.4.cuda-lbvh-stage.v1"
CUDA_LBVH_MAX_PRIMITIVES = 357_913_941
_PREPARED_EXTENSION_TOKEN = object()
_CUDA_LBVH_EXTENSION_SOURCE_SHA256 = hashlib.sha256(
    (CUDA_LBVH_EXTENSION_ABI_VERSION + CPP_SOURCE + CUDA_SOURCE).encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class CudaLbvhWorkspaceRequirements:
    """Frozen CUB workspace sizing result for one device/count/source tuple."""

    schema_version: str
    extension_abi_version: str
    extension_source_sha256: str
    device_ordinal: int
    compute_capability: tuple[int, int]
    primitive_count: int
    node_capacity: int
    reduction_temp_bytes: int
    radix_sort_temp_bytes: int
    required_bytes: int
    alignment_bytes: int

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "extension_abi_version": self.extension_abi_version,
            "extension_source_sha256": self.extension_source_sha256,
            "device_ordinal": self.device_ordinal,
            "compute_capability": list(self.compute_capability),
            "primitive_count": self.primitive_count,
            "node_capacity": self.node_capacity,
            "reduction_temp_bytes": self.reduction_temp_bytes,
            "radix_sort_temp_bytes": self.radix_sort_temp_bytes,
            "required_bytes": self.required_bytes,
            "alignment_bytes": self.alignment_bytes,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_payload())


@dataclass(frozen=True)
class CudaLbvhWorkspace:
    """Caller-owned aligned opaque workspace and its exact requirement."""

    storage: object
    requirements: CudaLbvhWorkspaceRequirements
    resources: tuple[object, ...] = ()

    @property
    def coverage_sha256(self) -> str:
        return _canonical_sha256(
            {
                "requirements_sha256": self.requirements.sha256,
                "storage": _tensor_manifest("workspace", self.storage),
            }
        )


@dataclass(frozen=True)
class CudaLbvhBuffers:
    """Caller-owned mutable outputs for one native CUDA LBVH rebuild."""

    device_ordinal: int
    primitive_count: int
    node_count: int
    scene_bounds_min: object
    scene_bounds_max: object
    morton_keys: object
    primitive_ids: object
    sorted_morton_keys: object
    sorted_primitive_ids: object
    left: object
    right: object
    parent: object
    start: object
    count: object
    range_start: object
    range_end: object
    split: object
    node_bounds_min: object
    node_bounds_max: object
    child_counter: object
    resources: tuple[object, ...] = ()

    def tensor_items(self) -> tuple[tuple[str, object], ...]:
        return tuple(
            (name, getattr(self, name))
            for name in (
                "scene_bounds_min",
                "scene_bounds_max",
                "morton_keys",
                "primitive_ids",
                "sorted_morton_keys",
                "sorted_primitive_ids",
                "left",
                "right",
                "parent",
                "start",
                "count",
                "range_start",
                "range_end",
                "split",
                "node_bounds_min",
                "node_bounds_max",
                "child_counter",
            )
        )


@dataclass(frozen=True)
class CudaLbvhRebuildStageEvidence:
    """Independent acceleration facts derived from one sealed native stage."""

    schema_version: str
    operation_id: str
    acceleration_id: str
    topology_origin_id: str
    topology_lifetime: str
    extension_abi_version: str
    extension_source_sha256: str
    extension_entry: str | None
    device_ordinal: int
    stream_handle: int
    native_input_manifest_sha256: str
    scene_contract_sha256: str
    input_contract: str
    primitive_count: int
    node_count: int
    leaf_count: int
    workspace_requirements_sha256: str
    workspace_coverage_sha256: str
    input_storage_sha256: str
    output_storage_sha256: str
    extension_dispatch_count: int
    kernel_dispatch_count: int
    cub_dispatch_count: int
    async_reset_count: int
    dispatch_plan_sha256: str

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.__dict__)


def cuda_lbvh_workspace_requirements(
    primitive_count: int,
    device_ordinal: int,
) -> CudaLbvhWorkspaceRequirements:
    """Query CUB scratch size outside native timing for one indexed device."""

    _require_torch_cuda_lbvh()
    _validate_primitive_count(primitive_count)
    if not isinstance(device_ordinal, int) or isinstance(device_ordinal, bool):
        raise TypeError("device_ordinal must be an integer")
    if device_ordinal < 0 or device_ordinal >= torch.cuda.device_count():
        raise ValueError("device_ordinal is out of range")
    module = _load_cuda_lbvh_extension()
    with torch.cuda.device(device_ordinal):
        reduction_bytes, radix_bytes, alignment_bytes = module.cuda_lbvh_workspace_requirements(
            primitive_count, device_ordinal
        )
        properties = torch.cuda.get_device_properties(device_ordinal)
    reduction_bytes = int(reduction_bytes)
    radix_bytes = int(radix_bytes)
    alignment_bytes = int(alignment_bytes)
    if min(reduction_bytes, radix_bytes) < 0 or alignment_bytes <= 0:
        raise RuntimeError("CUDA LBVH extension returned invalid workspace requirements")
    return CudaLbvhWorkspaceRequirements(
        schema_version=CUDA_LBVH_WORKSPACE_SCHEMA_VERSION,
        extension_abi_version=CUDA_LBVH_EXTENSION_ABI_VERSION,
        extension_source_sha256=_CUDA_LBVH_EXTENSION_SOURCE_SHA256,
        device_ordinal=device_ordinal,
        compute_capability=(int(properties.major), int(properties.minor)),
        primitive_count=primitive_count,
        node_capacity=_node_count(primitive_count),
        reduction_temp_bytes=reduction_bytes,
        radix_sort_temp_bytes=radix_bytes,
        required_bytes=max(reduction_bytes, radix_bytes),
        alignment_bytes=alignment_bytes,
    )


def allocate_cuda_lbvh_workspace(
    requirements: CudaLbvhWorkspaceRequirements,
) -> CudaLbvhWorkspace:
    """Allocate aligned scratch outside timing; C4 supplies final arena storage."""

    _require_torch_cuda_lbvh()
    _validate_requirements(requirements)
    device = torch.device("cuda", requirements.device_ordinal)
    required = requirements.required_bytes
    alignment = requirements.alignment_bytes
    with torch.cuda.device(device):
        if required == 0:
            owner = torch.empty((0,), dtype=torch.uint8, device=device)
            storage = owner
        else:
            owner = torch.empty((required + alignment - 1,), dtype=torch.uint8, device=device)
            offset = (-int(owner.data_ptr())) % alignment
            storage = owner[offset : offset + required]
    workspace = CudaLbvhWorkspace(
        storage=storage,
        requirements=requirements,
        resources=(owner,),
    )
    _validate_workspace(workspace, requirements)
    return workspace


def allocate_cuda_lbvh_buffers(
    *,
    primitive_count: int,
    device: object,
) -> CudaLbvhBuffers:
    """Allocate one setup/test output graph outside native timed submission."""

    _require_torch_cuda_lbvh()
    _validate_primitive_count(primitive_count)
    target = _indexed_cuda_device(device)
    node_count = _node_count(primitive_count)
    internal_count = max(primitive_count - 1, 0)
    scene_shape = (3,) if primitive_count > 0 else (0,)
    with torch.cuda.device(target):
        values = {
            "scene_bounds_min": torch.empty(scene_shape, dtype=torch.float32, device=target),
            "scene_bounds_max": torch.empty(scene_shape, dtype=torch.float32, device=target),
            "morton_keys": torch.empty((primitive_count,), dtype=torch.int64, device=target),
            "primitive_ids": torch.empty((primitive_count,), dtype=torch.int32, device=target),
            "sorted_morton_keys": torch.empty((primitive_count,), dtype=torch.int64, device=target),
            "sorted_primitive_ids": torch.empty((primitive_count,), dtype=torch.int32, device=target),
            "left": torch.empty((node_count,), dtype=torch.int32, device=target),
            "right": torch.empty((node_count,), dtype=torch.int32, device=target),
            "parent": torch.empty((node_count,), dtype=torch.int32, device=target),
            "start": torch.empty((node_count,), dtype=torch.int32, device=target),
            "count": torch.empty((node_count,), dtype=torch.int32, device=target),
            "range_start": torch.empty((internal_count,), dtype=torch.int32, device=target),
            "range_end": torch.empty((internal_count,), dtype=torch.int32, device=target),
            "split": torch.empty((internal_count,), dtype=torch.int32, device=target),
            "node_bounds_min": torch.empty((node_count, 3), dtype=torch.float32, device=target),
            "node_bounds_max": torch.empty((node_count, 3), dtype=torch.float32, device=target),
            "child_counter": torch.empty((internal_count,), dtype=torch.int32, device=target),
        }
    buffers = CudaLbvhBuffers(
        device_ordinal=int(target.index),
        primitive_count=primitive_count,
        node_count=node_count,
        resources=(),
        **values,
    )
    _validate_buffers(buffers, target)
    return buffers


class CudaLbvhRebuildStageFactory:
    """Prepare one immutable native LBVH operation identity outside timing."""

    __slots__ = (
        "_entry",
        "_frozen_signature",
        "_sealed",
        "context",
        "input_contract",
        "module",
        "native_input_manifest_sha256",
        "primitive_count",
        "requirements",
        "scene_contract_sha256",
    )

    def __init__(
        self,
        *,
        context: object,
        primitive_count: int,
        requirements: CudaLbvhWorkspaceRequirements,
        module: object,
        native_input_manifest_sha256: str,
        scene_contract_sha256: str,
        input_contract: str,
        _prepared_extension_token: object | None = None,
    ) -> None:
        from .cuda_timeline import CudaOpticalExecutionContext

        self._sealed = False
        if _prepared_extension_token is not _PREPARED_EXTENSION_TOKEN:
            raise ValueError("CUDA LBVH factory module must come from prepare()")
        if not isinstance(context, CudaOpticalExecutionContext):
            raise TypeError("CUDA LBVH factory requires CudaOpticalExecutionContext")
        _validate_primitive_count(primitive_count)
        _validate_requirements(requirements)
        if requirements.primitive_count != primitive_count:
            raise ValueError("CUDA LBVH requirements primitive count mismatch")
        if requirements.device_ordinal != context.device.index:
            raise ValueError("CUDA LBVH requirements are on the wrong device")
        if (
            not _sha256_text(native_input_manifest_sha256)
            or not _sha256_text(scene_contract_sha256)
            or not input_contract
        ):
            raise ValueError("CUDA LBVH provenance must be non-empty")
        entry = getattr(module, "cuda_lbvh_rebuild_out", None)
        if not callable(entry):
            raise ValueError("CUDA LBVH extension is missing its fixed rebuild entry")
        context._validate_identity()
        self.context = context
        self.primitive_count = primitive_count
        self.requirements = requirements
        self.module = module
        self.native_input_manifest_sha256 = native_input_manifest_sha256
        self.scene_contract_sha256 = scene_contract_sha256
        self.input_contract = input_contract
        self._entry = entry
        self._frozen_signature = self._factory_signature()
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False) or hasattr(self, "_frozen_signature"):
            raise AttributeError("CUDA LBVH rebuild factory is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("CUDA LBVH rebuild factory is immutable")

    @classmethod
    def prepare(
        cls,
        *,
        context: object,
        primitive_count: int,
        native_input_manifest_sha256: str,
        scene_contract_sha256: str,
        input_contract: str,
    ) -> "CudaLbvhRebuildStageFactory":
        """Load the extension and freeze workspace requirements outside timing."""

        requirements = cuda_lbvh_workspace_requirements(
            primitive_count,
            int(getattr(context.device, "index", -1)),
        )
        return cls(
            context=context,
            primitive_count=primitive_count,
            requirements=requirements,
            module=_load_cuda_lbvh_extension(),
            native_input_manifest_sha256=native_input_manifest_sha256,
            scene_contract_sha256=scene_contract_sha256,
            input_contract=input_contract,
            _prepared_extension_token=_PREPARED_EXTENSION_TOKEN,
        )

    def bind(
        self,
        *,
        world: CudaWorldSceneBuffers,
        outputs: CudaLbvhBuffers,
        workspace: CudaLbvhWorkspace,
    ) -> "CudaLbvhRebuildStage":
        """Seal exact caller-owned AABB, workspace, and output allocations."""

        self._assert_stable()
        _validate_world_aabbs(world, self.primitive_count, self.context.device)
        _validate_buffers(outputs, self.context.device)
        if outputs.primitive_count != self.primitive_count:
            raise ValueError("CUDA LBVH output primitive count mismatch")
        _validate_workspace(workspace, self.requirements)
        _validate_lbvh_storage_disjointness(world, outputs, workspace)
        entry = self._entry if self.primitive_count > 0 else None
        arguments = _lbvh_abi_arguments(world, outputs, workspace) if entry else ()
        evidence = _build_stage_evidence(
            factory=self,
            world=world,
            outputs=outputs,
            workspace=workspace,
            extension_entry=entry,
        )
        stage = CudaLbvhRebuildStage(
            factory=self,
            world=world,
            outputs=outputs,
            workspace=workspace,
            extension_entry=entry,
            extension_args=arguments,
            evidence=evidence,
        )
        stage._validate_cuda_transaction_stage(self.context)
        stage._freeze()
        return stage

    def _factory_signature(self) -> tuple[object, ...]:
        return (
            id(self.context),
            self.context.device,
            self.context.stream_handle,
            id(self.context.stream),
            self.primitive_count,
            self.requirements,
            self.requirements.sha256,
            id(self.module),
            id(self._entry),
            CUDA_LBVH_EXTENSION_ABI_VERSION,
            _CUDA_LBVH_EXTENSION_SOURCE_SHA256,
            self.native_input_manifest_sha256,
            self.scene_contract_sha256,
            self.input_contract,
        )

    def _assert_stable(self) -> None:
        self.context._validate_identity()
        if self._factory_signature() != self._frozen_signature:
            raise ValueError("CUDA LBVH rebuild factory changed after construction")


class CudaLbvhRebuildStage:
    """One immutable allocation-free native CUDA LBVH transaction stage."""

    cuda_transaction_stage_name = "cuda_lbvh_rebuild"

    __slots__ = (
        "_binding_signature",
        "_extension_args",
        "_extension_entry",
        "_sealed",
        "evidence",
        "factory",
        "outputs",
        "workspace",
        "world",
    )

    def __init__(
        self,
        *,
        factory: CudaLbvhRebuildStageFactory,
        world: CudaWorldSceneBuffers,
        outputs: CudaLbvhBuffers,
        workspace: CudaLbvhWorkspace,
        extension_entry: object | None,
        extension_args: tuple[object, ...],
        evidence: CudaLbvhRebuildStageEvidence,
    ) -> None:
        self._sealed = False
        self.factory = factory
        self.world = world
        self.outputs = outputs
        self.workspace = workspace
        self._extension_entry = extension_entry
        self._extension_args = extension_args
        self.evidence = evidence
        self._binding_signature: tuple[object, ...] | None = None

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False) or getattr(self, "_binding_signature", None) is not None:
            raise AttributeError("CUDA LBVH rebuild stage is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("CUDA LBVH rebuild stage is immutable")

    def enqueue(self, transaction: object) -> None:
        """Invoke only the frozen native entry on the transaction stream."""

        with transaction.enqueue_stage(self):
            if self._extension_entry is not None:
                transaction.record_diagnostic_start("acceleration_kernel")
                self._extension_entry(*self._extension_args)
                transaction.record_diagnostic_end("acceleration_kernel")

    def _validate_cuda_transaction_stage(self, context: object) -> None:
        if context is not self.factory.context:
            raise ValueError("CUDA LBVH stage execution context identity mismatch")
        self.factory._assert_stable()
        _validate_world_aabbs(self.world, self.factory.primitive_count, context.device)
        _validate_buffers(self.outputs, context.device)
        _validate_workspace(self.workspace, self.factory.requirements)
        _validate_lbvh_storage_disjointness(self.world, self.outputs, self.workspace)
        expected_entry = self.factory._entry if self.factory.primitive_count > 0 else None
        if self._extension_entry is not expected_entry:
            raise ValueError("CUDA LBVH extension entry identity changed")
        expected_arguments = (
            _lbvh_abi_arguments(self.world, self.outputs, self.workspace)
            if expected_entry is not None
            else ()
        )
        if not _identity_arguments_match(self._extension_args, expected_arguments):
            raise ValueError("CUDA LBVH fixed ABI binding changed")
        expected_evidence = _build_stage_evidence(
            factory=self.factory,
            world=self.world,
            outputs=self.outputs,
            workspace=self.workspace,
            extension_entry=expected_entry,
        )
        if self.evidence != expected_evidence:
            raise ValueError("CUDA LBVH stage evidence changed from its bound ABI")

    def _freeze(self) -> None:
        if self._sealed or self._binding_signature is not None:
            raise RuntimeError("CUDA LBVH rebuild stage seal is one-way")
        object.__setattr__(self, "_binding_signature", self._stage_signature())
        object.__setattr__(self, "_sealed", True)

    def _assert_cuda_transaction_stage_stable(self, context: object) -> None:
        if context is not self.factory.context:
            raise ValueError("CUDA LBVH rebuild stage execution context changed")
        if self._binding_signature is None or self._stage_signature() != self._binding_signature:
            raise ValueError("CUDA LBVH rebuild stage binding changed after validation")

    def _cuda_transaction_stage_evidence(self) -> CudaLbvhRebuildStageEvidence:
        return self.evidence

    def _stage_signature(self) -> tuple[object, ...]:
        return (
            self.factory._factory_signature(),
            id(self.world),
            id(self.outputs),
            id(self.workspace),
            id(self._extension_entry),
            id(self._extension_args),
            self.evidence,
            _storage_manifest_sha256(_world_aabb_items(self.world)),
            _storage_manifest_sha256(self.outputs.tensor_items()),
            self.workspace.coverage_sha256,
            tuple(id(resource) for resource in self.world.resources),
            tuple(id(resource) for resource in self.outputs.resources),
            tuple(id(resource) for resource in self.workspace.resources),
        )


def _build_stage_evidence(
    *,
    factory: CudaLbvhRebuildStageFactory,
    world: CudaWorldSceneBuffers,
    outputs: CudaLbvhBuffers,
    workspace: CudaLbvhWorkspace,
    extension_entry: object | None,
) -> CudaLbvhRebuildStageEvidence:
    plan = _dispatch_plan(factory.primitive_count)
    return CudaLbvhRebuildStageEvidence(
        schema_version=CUDA_LBVH_STAGE_SCHEMA_VERSION,
        operation_id=CUDA_LBVH_OPERATION_ID,
        acceleration_id=CUDA_LBVH_ACCELERATION_ID,
        topology_origin_id=CUDA_LBVH_TOPOLOGY_ORIGIN_ID,
        topology_lifetime=CUDA_LBVH_TOPOLOGY_LIFETIME,
        extension_abi_version=CUDA_LBVH_EXTENSION_ABI_VERSION,
        extension_source_sha256=_CUDA_LBVH_EXTENSION_SOURCE_SHA256,
        extension_entry=("cuda_lbvh_rebuild_out" if extension_entry is not None else None),
        device_ordinal=int(factory.context.device.index),
        stream_handle=factory.context.stream_handle,
        native_input_manifest_sha256=factory.native_input_manifest_sha256,
        scene_contract_sha256=factory.scene_contract_sha256,
        input_contract=factory.input_contract,
        primitive_count=factory.primitive_count,
        node_count=_node_count(factory.primitive_count),
        leaf_count=factory.primitive_count,
        workspace_requirements_sha256=factory.requirements.sha256,
        workspace_coverage_sha256=workspace.coverage_sha256,
        input_storage_sha256=_storage_manifest_sha256(_world_aabb_items(world)),
        output_storage_sha256=_storage_manifest_sha256(outputs.tensor_items()),
        extension_dispatch_count=1 if factory.primitive_count > 0 else 0,
        kernel_dispatch_count=_kernel_dispatch_count(factory.primitive_count),
        cub_dispatch_count=7 if factory.primitive_count > 0 else 0,
        async_reset_count=(9 if factory.primitive_count > 1 else 5) if factory.primitive_count > 0 else 0,
        dispatch_plan_sha256=_canonical_sha256(plan),
    )


def _dispatch_plan(primitive_count: int) -> tuple[dict[str, object], ...]:
    if primitive_count == 0:
        return ()
    operations: list[dict[str, object]] = [
        {"operation": "scene_bounds_cub_reduce", "count": 6},
        {"operation": "topology_integer_async_reset", "count": 5},
        {"operation": "morton30_source_index", "count": 1},
        {"operation": "cub_radix_sort_pairs", "count": 1},
        {"operation": "leaf_metadata", "count": 1},
    ]
    if primitive_count > 1:
        operations.extend(
            (
                {"operation": "internal_integer_async_reset", "count": 4},
                {"operation": "karras_internal_topology", "count": 1},
            )
        )
    operations.extend(
        (
            {"operation": "node_bounds_initialize", "count": 1},
            {"operation": "leaf_bounds", "count": 1},
        )
    )
    if primitive_count > 1:
        operations.append({"operation": "internal_bounds", "count": 1})
    return tuple(operations)


def _kernel_dispatch_count(primitive_count: int) -> int:
    if primitive_count == 0:
        return 0
    return 6 if primitive_count > 1 else 4


def _lbvh_abi_arguments(
    world: CudaWorldSceneBuffers,
    outputs: CudaLbvhBuffers,
    workspace: CudaLbvhWorkspace,
) -> tuple[object, ...]:
    return (
        world.triangle_aabb_min,
        world.triangle_aabb_max,
        workspace.storage,
        outputs.scene_bounds_min,
        outputs.scene_bounds_max,
        outputs.morton_keys,
        outputs.primitive_ids,
        outputs.sorted_morton_keys,
        outputs.sorted_primitive_ids,
        outputs.left,
        outputs.right,
        outputs.parent,
        outputs.start,
        outputs.count,
        outputs.range_start,
        outputs.range_end,
        outputs.split,
        outputs.node_bounds_min,
        outputs.node_bounds_max,
        outputs.child_counter,
    )


def _identity_arguments_match(
    actual: tuple[object, ...],
    expected: tuple[object, ...],
) -> bool:
    return len(actual) == len(expected) and all(
        actual_value is expected_value for actual_value, expected_value in zip(actual, expected, strict=True)
    )


def _validate_primitive_count(primitive_count: int) -> None:
    if not isinstance(primitive_count, int) or isinstance(primitive_count, bool):
        raise TypeError("primitive_count must be an integer")
    if primitive_count < 0 or primitive_count > CUDA_LBVH_MAX_PRIMITIVES:
        raise ValueError("primitive_count exceeds the int32 flattened-node range")


def _node_count(primitive_count: int) -> int:
    return 0 if primitive_count == 0 else primitive_count * 2 - 1


def _indexed_cuda_device(device: object):
    target = torch.device(device)
    if target.type != "cuda" or target.index is None:
        raise ValueError("CUDA LBVH requires an indexed CUDA device")
    if target.index < 0 or target.index >= torch.cuda.device_count():
        raise ValueError("CUDA LBVH device index is out of range")
    return target


def _validate_requirements(requirements: CudaLbvhWorkspaceRequirements) -> None:
    if not isinstance(requirements, CudaLbvhWorkspaceRequirements):
        raise TypeError("requirements must be CudaLbvhWorkspaceRequirements")
    _validate_primitive_count(requirements.primitive_count)
    if (
        requirements.schema_version != CUDA_LBVH_WORKSPACE_SCHEMA_VERSION
        or requirements.extension_abi_version != CUDA_LBVH_EXTENSION_ABI_VERSION
        or requirements.extension_source_sha256 != _CUDA_LBVH_EXTENSION_SOURCE_SHA256
        or requirements.node_capacity != _node_count(requirements.primitive_count)
        or len(requirements.compute_capability) != 2
        or requirements.reduction_temp_bytes < 0
        or requirements.radix_sort_temp_bytes < 0
        or requirements.required_bytes
        != max(requirements.reduction_temp_bytes, requirements.radix_sort_temp_bytes)
        or requirements.alignment_bytes <= 0
    ):
        raise ValueError("CUDA LBVH workspace requirements are stale or invalid")


def _validate_workspace(
    workspace: CudaLbvhWorkspace,
    requirements: CudaLbvhWorkspaceRequirements,
) -> None:
    if not isinstance(workspace, CudaLbvhWorkspace):
        raise TypeError("workspace must be CudaLbvhWorkspace")
    _validate_requirements(requirements)
    if workspace.requirements != requirements:
        raise ValueError("CUDA LBVH workspace requirement identity mismatch")
    _validate_tensor(
        workspace.storage,
        name="workspace",
        device=torch.device("cuda", requirements.device_ordinal),
        dtype=torch.uint8,
        exact=(requirements.required_bytes,),
    )
    if requirements.required_bytes > 0 and (
        int(workspace.storage.data_ptr()) % requirements.alignment_bytes != 0
    ):
        raise ValueError("CUDA LBVH workspace pointer is not sufficiently aligned")


def _validate_world_aabbs(
    world: CudaWorldSceneBuffers,
    primitive_count: int,
    device: object,
) -> None:
    if not isinstance(world, CudaWorldSceneBuffers):
        raise TypeError("world must be CudaWorldSceneBuffers")
    target = torch.device(device)
    if world.device_ordinal != target.index:
        raise ValueError("CUDA LBVH world buffers are on the wrong device")
    _validate_tensor(
        world.triangle_aabb_min,
        name="triangle_aabb_min",
        device=target,
        dtype=torch.float32,
        exact=(primitive_count, 3),
    )
    _validate_tensor(
        world.triangle_aabb_max,
        name="triangle_aabb_max",
        device=target,
        dtype=torch.float32,
        exact=(primitive_count, 3),
    )


def _validate_buffers(buffers: CudaLbvhBuffers, device: object) -> None:
    if not isinstance(buffers, CudaLbvhBuffers):
        raise TypeError("outputs must be CudaLbvhBuffers")
    _validate_primitive_count(buffers.primitive_count)
    target = torch.device(device)
    if buffers.device_ordinal != target.index:
        raise ValueError("CUDA LBVH outputs are on the wrong device")
    expected_node_count = _node_count(buffers.primitive_count)
    if buffers.node_count != expected_node_count:
        raise ValueError("CUDA LBVH output node count is invalid")
    internal_count = max(buffers.primitive_count - 1, 0)
    scene_shape = (3,) if buffers.primitive_count > 0 else (0,)
    expected = {
        "scene_bounds_min": (torch.float32, scene_shape),
        "scene_bounds_max": (torch.float32, scene_shape),
        "morton_keys": (torch.int64, (buffers.primitive_count,)),
        "primitive_ids": (torch.int32, (buffers.primitive_count,)),
        "sorted_morton_keys": (torch.int64, (buffers.primitive_count,)),
        "sorted_primitive_ids": (torch.int32, (buffers.primitive_count,)),
        "left": (torch.int32, (expected_node_count,)),
        "right": (torch.int32, (expected_node_count,)),
        "parent": (torch.int32, (expected_node_count,)),
        "start": (torch.int32, (expected_node_count,)),
        "count": (torch.int32, (expected_node_count,)),
        "range_start": (torch.int32, (internal_count,)),
        "range_end": (torch.int32, (internal_count,)),
        "split": (torch.int32, (internal_count,)),
        "node_bounds_min": (torch.float32, (expected_node_count, 3)),
        "node_bounds_max": (torch.float32, (expected_node_count, 3)),
        "child_counter": (torch.int32, (internal_count,)),
    }
    for name, (dtype, shape) in expected.items():
        _validate_tensor(
            getattr(buffers, name),
            name=name,
            device=target,
            dtype=dtype,
            exact=shape,
        )


def _validate_tensor(
    value: object,
    *,
    name: str,
    device: object,
    dtype: object,
    exact: tuple[int, ...],
) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if value.device != torch.device(device):
        raise ValueError(f"{name} must be on {torch.device(device)}")
    if value.dtype != dtype:
        raise ValueError(f"{name} has the wrong dtype")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tuple(value.shape) != exact:
        raise ValueError(f"{name} has the wrong shape")


def _validate_lbvh_storage_disjointness(
    world: CudaWorldSceneBuffers,
    outputs: CudaLbvhBuffers,
    workspace: CudaLbvhWorkspace,
) -> None:
    inputs = _world_aabb_items(world) + (("workspace", workspace.storage),)
    nonempty_inputs = tuple(
        (name, *_tensor_storage_span(value)) for name, value in inputs if int(value.numel()) > 0
    )
    for index, (name, start, end) in enumerate(nonempty_inputs):
        for other_name, other_start, other_end in nonempty_inputs[:index]:
            if start < other_end and other_start < end:
                raise ValueError(f"CUDA LBVH inputs {name!r} and {other_name!r} overlap")
    seen_outputs: list[tuple[str, int, int]] = []
    for name, value in outputs.tensor_items():
        if int(value.numel()) == 0:
            continue
        start, end = _tensor_storage_span(value)
        for input_name, input_start, input_end in nonempty_inputs:
            if start < input_end and input_start < end:
                raise ValueError(f"CUDA LBVH output {name!r} overlaps input {input_name!r}")
        for other_name, other_start, other_end in seen_outputs:
            if start < other_end and other_start < end:
                raise ValueError(f"CUDA LBVH outputs {name!r} and {other_name!r} overlap")
        seen_outputs.append((name, start, end))


def _world_aabb_items(
    world: CudaWorldSceneBuffers,
) -> tuple[tuple[str, object], ...]:
    return (
        ("triangle_aabb_min", world.triangle_aabb_min),
        ("triangle_aabb_max", world.triangle_aabb_max),
    )


def _tensor_storage_span(value: object) -> tuple[int, int]:
    start = int(value.data_ptr())
    return start, start + int(value.numel()) * int(value.element_size())


def _tensor_manifest(name: str, value: object) -> dict[str, object]:
    return {
        "name": name,
        "device_ordinal": int(value.device.index),
        "ptr": int(value.data_ptr()),
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "nbytes": int(value.numel()) * int(value.element_size()),
    }


def _storage_manifest_sha256(values: tuple[tuple[str, object], ...]) -> str:
    return _canonical_sha256([_tensor_manifest(name, value) for name, value in values])


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def build_cuda_lbvh_from_snapshot(
    snapshot: DeviceOpticalSceneSnapshot,
    *,
    device=None,
    stream=None,
) -> DeviceOpticalBvh:
    """Allocating Warp/Torch compatibility facade; never native-benchmark eligible."""

    total_start = time.perf_counter()
    _require_torch_cuda_lbvh()
    _require_warp_compat()
    if snapshot.triangle_aabb_min is None or snapshot.triangle_aabb_max is None:
        raise ValueError("build_cuda_lbvh_from_snapshot requires AABB snapshot buffers")
    wp.init()
    resolved_device = snapshot.scene.device if device is None else wp.get_device(device)
    if resolved_device != snapshot.scene.device:
        raise ValueError("DeviceOpticalBvh device must match DeviceOpticalSceneSnapshot device")

    detail_ms: list[tuple[str, float]] = []
    if snapshot.ready_event is not None:
        wait_start = time.perf_counter()
        wp.synchronize_event(snapshot.ready_event)
        detail_ms.append(("wait_snapshot_ready", (time.perf_counter() - wait_start) * 1000.0))

    convert_start = time.perf_counter()
    aabb_min_t = wp.to_torch(snapshot.triangle_aabb_min).contiguous()
    aabb_max_t = wp.to_torch(snapshot.triangle_aabb_max).contiguous()
    detail_ms.append(("warp_to_torch_aabb", (time.perf_counter() - convert_start) * 1000.0))
    ordinal = int(resolved_device.ordinal)
    target = torch.device("cuda", ordinal)
    empty_triangle = torch.empty((aabb_min_t.shape[0], 3), dtype=torch.float32, device=target)
    empty_plane = torch.empty((0, 3), dtype=torch.float32, device=target)
    world = CudaWorldSceneBuffers(
        device_ordinal=ordinal,
        triangle_v0_world=empty_triangle,
        triangle_e1_world=torch.empty_like(empty_triangle),
        triangle_e2_world=torch.empty_like(empty_triangle),
        triangle_normal_world=torch.empty_like(empty_triangle),
        plane_normal_world=empty_plane,
        plane_point_world=torch.empty_like(empty_plane),
        triangle_aabb_min=aabb_min_t,
        triangle_aabb_max=aabb_max_t,
        resources=(snapshot,),
    )
    primitive_count = int(aabb_min_t.shape[0])
    scene_contract_sha256 = snapshot.scene.scene_contract_sha256 or _canonical_sha256(
        {
            "compatibility_scene": id(snapshot.scene),
            "primitive_count": primitive_count,
        }
    )
    native_manifest_sha256 = _canonical_sha256(
        {
            "compatibility_only": True,
            "frame_id": snapshot.frame_id,
            "scene_contract_sha256": scene_contract_sha256,
            "aabb_storage": [
                _tensor_manifest("triangle_aabb_min", aabb_min_t),
                _tensor_manifest("triangle_aabb_max", aabb_max_t),
            ],
        }
    )
    from .cuda_timeline import CudaOpticalExecutionContext

    with torch.cuda.device(target):
        torch_stream = torch.cuda.Stream(device=target)
        completion_event = torch.cuda.Event(enable_timing=False)
        completion_event.record(torch_stream)
    context = CudaOpticalExecutionContext(device=target, stream=torch_stream)
    factory = CudaLbvhRebuildStageFactory.prepare(
        context=context,
        primitive_count=primitive_count,
        native_input_manifest_sha256=native_manifest_sha256,
        scene_contract_sha256=scene_contract_sha256,
        input_contract="legacy_warp_snapshot_bridge_v1",
    )
    outputs = allocate_cuda_lbvh_buffers(primitive_count=primitive_count, device=target)
    workspace = allocate_cuda_lbvh_workspace(factory.requirements)
    stage = factory.bind(world=world, outputs=outputs, workspace=workspace)
    build_start = time.perf_counter()
    transaction = context.begin_transaction(stages=(stage,), completion_event=completion_event)
    stage.enqueue(transaction)
    completion = transaction.finish()
    completion.event.synchronize()
    build_ms = (time.perf_counter() - build_start) * 1000.0
    detail_ms.extend(
        (
            ("native_cuda_lbvh_transaction_and_wait", build_ms),
            ("compatibility_allocations_and_bridge", (build_start - total_start) * 1000.0),
        )
    )

    metadata_start = time.perf_counter()
    num_nodes = outputs.node_count
    node_depth_t = torch.zeros((num_nodes,), dtype=torch.int32, device=target)
    level_ranges_t = torch.empty((0, 2), dtype=torch.int32, device=target)
    detail_ms.append(("metadata_tensor_alloc", (time.perf_counter() - metadata_start) * 1000.0))

    wrap_start = time.perf_counter()
    bounds_min = wp.from_torch(outputs.node_bounds_min, dtype=wp.float32)
    bounds_max = wp.from_torch(outputs.node_bounds_max, dtype=wp.float32)
    left = wp.from_torch(outputs.left, dtype=wp.int32)
    right = wp.from_torch(outputs.right, dtype=wp.int32)
    start = wp.from_torch(outputs.start, dtype=wp.int32)
    count = wp.from_torch(outputs.count, dtype=wp.int32)
    node_depth = wp.from_torch(node_depth_t, dtype=wp.int32)
    level_ranges = wp.from_torch(level_ranges_t, dtype=wp.int32)
    prim_ids = wp.from_torch(outputs.sorted_primitive_ids, dtype=wp.int32)
    detail_ms.append(("torch_to_warp_views", (time.perf_counter() - wrap_start) * 1000.0))

    event_start = time.perf_counter()
    ready_event = (stream or wp.get_stream(resolved_device)).record_event()
    detail_ms.append(("record_ready_event", (time.perf_counter() - event_start) * 1000.0))

    stats = DeviceBvhBuildStats(
        num_nodes=num_nodes,
        num_leaves=primitive_count,
        max_depth=0,
        leaf_size=1,
        sah_quality_cost=0.0,
        host_build_ms=0.0,
        upload_ms=0.0,
        total_ms=(time.perf_counter() - total_start) * 1000.0,
        split_strategy="cuda_lbvh",
        supports_refit=False,
        level_ranges=(),
        detail_ms=tuple(detail_ms),
    )
    resources = (
        outputs,
        workspace,
        stage,
        completion,
        node_depth_t,
        level_ranges_t,
        aabb_min_t,
        aabb_max_t,
    )
    stats = replace(stats, host_build_ms=build_ms)
    scene = snapshot.scene
    return DeviceOpticalBvh(
        device=resolved_device,
        frame_id=snapshot.frame_id,
        env_idx=snapshot.env_idx,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        left=left,
        right=right,
        start=start,
        count=count,
        node_depth=node_depth,
        level_ranges=level_ranges,
        prim_ids=prim_ids,
        primitive_global_id=scene.triangle_primitive_global_id,
        primitive_instance_index=scene.triangle_instance_index,
        primitive_index_within_instance=scene.triangle_primitive_index_within_instance,
        primitive_geometry_index=scene.triangle_geometry_index,
        primitive_geometry_primitive_index=scene.triangle_geometry_primitive_index,
        primitive_source_order_key=scene.triangle_source_order_key,
        stats=stats,
        topology_origin_id=CUDA_LBVH_TOPOLOGY_ORIGIN_ID,
        scene_contract_sha256=scene.scene_contract_sha256,
        ready_event=ready_event,
        resources=resources,
    )


@lru_cache(maxsize=1)
def _load_cuda_lbvh_extension():
    _require_torch_cuda_lbvh()
    return load_inline(
        name="robot_sim_cuda_lbvh_native_v3",
        cpp_sources=[CPP_SOURCE],
        cuda_sources=[CUDA_SOURCE],
        with_cuda=True,
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        verbose=False,
    )


def _require_torch_cuda_lbvh() -> None:
    if not _HAS_TORCH_CUDA_LBVH_DEPS:
        raise ImportError("CUDA LBVH rebuild requires Torch CUDA extension tooling")
    if not torch.cuda.is_available():
        raise ImportError("CUDA LBVH rebuild requires Torch CUDA availability")


def _require_warp_compat() -> None:
    if not _HAS_WARP_COMPAT:
        raise ImportError("legacy CUDA LBVH compatibility requires Warp")
