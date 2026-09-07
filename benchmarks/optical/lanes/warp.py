"""Fail-closed Warp audit adapter and temporary static comparison binding."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..assets import AssetPreflightError, preflight_case_asset
from ..eligibility import (
    LaneEligibilityEvidence,
    evaluate_lane_eligibility,
)
from ..native import PreparedNativeInput
from ..timing import assert_outside_timed_boundary
from .audit import verify_current_product_source_manifest

WARP_CUDA_LBVH_COMPARISON_SCHEMA_VERSION = "q54.p12.4.warp-cuda-lbvh-comparison.v1"
_PREPARED_COMPARISON_TOKEN = object()

_LBVH_ARRAY_SPECS = (
    ("scene_bounds_min", "float32", "scene"),
    ("scene_bounds_max", "float32", "scene"),
    ("morton_keys", "int64", "primitive"),
    ("primitive_ids", "int32", "primitive"),
    ("sorted_morton_keys", "int64", "primitive"),
    ("sorted_primitive_ids", "int32", "primitive"),
    ("left", "int32", "node"),
    ("right", "int32", "node"),
    ("parent", "int32", "node"),
    ("start", "int32", "node"),
    ("count", "int32", "node"),
    ("range_start", "int32", "internal"),
    ("range_end", "int32", "internal"),
    ("split", "int32", "internal"),
    ("node_bounds_min", "float32", "node_vec3"),
    ("node_bounds_max", "float32", "node_vec3"),
    ("child_counter", "int32", "internal"),
)

_WORLD_VIEW_NAMES = (
    "triangle_aabb_min",
    "triangle_aabb_max",
)


@dataclass(frozen=True)
class WarpCudaLbvhComparisonStorage:
    """Warp-owned one-shot LBVH outputs and scratch prepared outside timing."""

    device: object
    primitive_count: int
    node_count: int
    arrays: Mapping[str, object]
    workspace: object
    workspace_owner: object
    node_depth: object
    level_ranges: object
    resources: tuple[object, ...] = ()


@dataclass(frozen=True)
class WarpTorchStableView:
    """Strongly owned setup-time Torch view of one Warp CUDA allocation."""

    name: str
    warp_array: object
    torch_tensor: object


@dataclass(frozen=True)
class WarpCudaLbvhComparisonEvidence:
    """Independent facts for the temporary Warp-renderer/static-LBVH binding."""

    schema_version: str
    operation_id: str
    acceleration_id: str
    acceleration_class: str
    topology_origin_id: str
    topology_lifetime: str
    device_ordinal: int
    warp_stream_handle: int
    torch_stream_handle: int
    native_input_manifest_sha256: str
    scene_contract_sha256: str
    input_contract: str
    transform_replay_sha256: str
    transform_frame_index: int
    primitive_count: int
    node_count: int
    stable_view_names: tuple[str, ...]
    stable_view_alias_sha256: str
    warp_scene_stage_evidence_sha256: str
    cuda_lbvh_stage_evidence_sha256: str
    setup_stable_view_count: int
    bridge_creation_count: int

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "stable_view_names": list(self.stable_view_names),
        }


class CurrentWarpLanePreflight:
    """Audit current Warp calls; intentionally has no sample submission API."""

    lane_id = "warp"

    def __init__(self, *, repo_root: Path) -> None:
        self._repo_root = repo_root.resolve()

    def preflight_sample(
        self,
        *,
        row: Mapping[str, Any],
        prepared: PreparedNativeInput,
    ) -> LaneEligibilityEvidence:
        if prepared.backend != self.lane_id:
            raise ValueError("Warp preflight requires Warp-owned native input")
        asset_hash: str | None = None
        try:
            asset = preflight_case_asset(str(row["case_id"]), repo_root=self._repo_root)
        except AssetPreflightError:
            asset = None
        if asset is not None:
            asset_hash = asset.manifest_sha256
        return evaluate_lane_eligibility(
            row=row,
            lane="warp",
            entry_point_audits=verify_current_product_source_manifest("warp"),
            asset_manifest_sha256=asset_hash,
        )


class WarpCudaLbvhComparisonFactory:
    """Freeze the setup-only Warp/Torch interop graph for static R2 rows.

    The factory is comparison infrastructure, not a product executor and not a
    C4 result capability.  C4 may later use it to seal final arena slots, but
    only after those slots and their leases exist.
    """

    __slots__ = (
        "_frozen_signature",
        "_sealed",
        "acceleration_class",
        "context",
        "cuda_lbvh_factory",
        "external_stream",
        "prepared",
        "scene_factory",
        "stream",
    )

    def __init__(
        self,
        *,
        prepared: PreparedNativeInput,
        acceleration_class: str,
        stream: object,
        external_stream: object,
        context: object,
        scene_factory: object,
        cuda_lbvh_factory: object,
        _prepared_token: object | None = None,
    ) -> None:
        self._sealed = False
        if _prepared_token is not _PREPARED_COMPARISON_TOKEN:
            raise ValueError("Warp CUDA-LBVH comparison factory must come from prepare()")
        self.prepared = prepared
        self.acceleration_class = acceleration_class
        self.stream = stream
        self.external_stream = external_stream
        self.context = context
        self.scene_factory = scene_factory
        self.cuda_lbvh_factory = cuda_lbvh_factory
        self._frozen_signature = self._factory_signature()
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False) or hasattr(self, "_frozen_signature"):
            raise AttributeError("Warp CUDA-LBVH comparison factory is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Warp CUDA-LBVH comparison factory is immutable")

    @classmethod
    def prepare(
        cls,
        prepared: PreparedNativeInput,
        *,
        stream: object,
        acceleration_class: str,
    ) -> "WarpCudaLbvhComparisonFactory":
        """Create the external-stream and extension binding outside timing."""

        assert_outside_timed_boundary("Warp CUDA-LBVH comparison factory preparation")
        torch, wp = _require_warp_torch_cuda()
        manifest = prepared.manifest
        if prepared.backend != "warp" or prepared.source is None:
            raise ValueError("Warp CUDA-LBVH comparison requires Warp-owned native input")
        if manifest.input_contract != "canonical_native_replay_v1":
            raise ValueError("Warp CUDA-LBVH comparison requires canonical replay input")
        if acceleration_class != "production_static":
            raise ValueError("Warp CUDA-LBVH comparison is restricted to production_static")
        if manifest.scene_contract_sha256 is None or manifest.transform_replay_sha256 is None:
            raise ValueError("Warp CUDA-LBVH comparison provenance is incomplete")
        packed = prepared.source.packed_scene_owner
        if packed is None or packed.scene_contract_sha256 != manifest.scene_contract_sha256:
            raise ValueError("Warp CUDA-LBVH comparison packed scene does not match its manifest")
        _validate_warp_stream_identity(stream, prepared.device_ordinal, wp=wp)
        if prepared.ready_event is not None:
            stream.wait_event(prepared.ready_event)

        target = torch.device("cuda", prepared.device_ordinal)
        with torch.cuda.device(target):
            external_stream = torch.cuda.ExternalStream(
                int(stream.cuda_stream),
                device=target,
            )
        from optics.cuda_lbvh import CudaLbvhRebuildStageFactory
        from optics.cuda_timeline import CudaOpticalExecutionContext

        context = CudaOpticalExecutionContext(device=target, stream=external_stream)
        from ..native import build_prepared_warp_scene_update_factory

        scene_factory = build_prepared_warp_scene_update_factory(prepared, stream=stream)
        cuda_lbvh_factory = CudaLbvhRebuildStageFactory.prepare(
            context=context,
            primitive_count=packed.num_triangles,
            native_input_manifest_sha256=manifest.sha256,
            scene_contract_sha256=manifest.scene_contract_sha256,
            input_contract=manifest.input_contract,
        )
        return cls(
            prepared=prepared,
            acceleration_class=acceleration_class,
            stream=stream,
            external_stream=external_stream,
            context=context,
            scene_factory=scene_factory,
            cuda_lbvh_factory=cuda_lbvh_factory,
            _prepared_token=_PREPARED_COMPARISON_TOKEN,
        )

    def seal(
        self,
        *,
        transform_frame_index: int,
        world: object,
        storage: WarpCudaLbvhComparisonStorage,
    ) -> "WarpCudaLbvhComparisonCommandBundle":
        """Create stable views and seal one explicit one-shot storage graph."""

        assert_outside_timed_boundary("Warp CUDA-LBVH comparison slot sealing")
        self._assert_stable()
        _, wp = _require_warp_torch_cuda()
        _validate_comparison_storage(
            storage,
            requirements=self.cuda_lbvh_factory.requirements,
            device_ordinal=self.prepared.device_ordinal,
            wp=wp,
        )
        from ..native import build_prepared_warp_transform_inputs

        transforms = build_prepared_warp_transform_inputs(
            self.prepared,
            transform_frame_index,
        )
        scene_bundle = self.scene_factory.seal(transforms=transforms, outputs=world)
        stable_views = _create_stable_views(world=world, storage=storage, wp=wp)
        view_by_name = {value.name: value.torch_tensor for value in stable_views}

        from optics.cuda_lbvh import (
            CudaLbvhBuffers,
            CudaLbvhWorkspace,
            CudaWorldAabbBuffers,
        )

        torch_world = CudaWorldAabbBuffers(
            device_ordinal=self.prepared.device_ordinal,
            resources=(world, stable_views),
            **{name: view_by_name[f"world.{name}"] for name in _WORLD_VIEW_NAMES},
        )
        torch_outputs = CudaLbvhBuffers(
            device_ordinal=self.prepared.device_ordinal,
            primitive_count=storage.primitive_count,
            node_count=storage.node_count,
            resources=(storage, stable_views),
            **{name: view_by_name[f"lbvh.{name}"] for name, _, _ in _LBVH_ARRAY_SPECS},
        )
        torch_workspace = CudaLbvhWorkspace(
            storage=view_by_name["lbvh.workspace"],
            requirements=self.cuda_lbvh_factory.requirements,
            resources=(storage.workspace_owner, storage.workspace, stable_views),
        )
        acceleration_stage = self.cuda_lbvh_factory.bind(
            world=torch_world,
            outputs=torch_outputs,
            workspace=torch_workspace,
        )
        warp_bvh = _build_warp_lbvh_view(
            factory=self,
            storage=storage,
            frame_index=transform_frame_index,
        )
        evidence = _build_comparison_evidence(
            factory=self,
            scene_bundle=scene_bundle,
            acceleration_stage=acceleration_stage,
            stable_views=stable_views,
            transform_frame_index=transform_frame_index,
            primitive_count=storage.primitive_count,
            node_count=storage.node_count,
        )
        return WarpCudaLbvhComparisonCommandBundle(
            factory=self,
            scene_bundle=scene_bundle,
            acceleration_stage=acceleration_stage,
            warp_bvh=warp_bvh,
            stable_views=stable_views,
            evidence=evidence,
        )

    def _factory_signature(self) -> tuple[object, ...]:
        return (
            id(self.prepared),
            self.prepared.manifest.sha256,
            self.prepared.device_ordinal,
            self.acceleration_class,
            id(self.stream),
            int(self.stream.cuda_stream),
            id(self.external_stream),
            int(self.external_stream.cuda_stream),
            id(self.context),
            self.context.stream_handle,
            id(self.scene_factory),
            id(self.cuda_lbvh_factory),
            self.cuda_lbvh_factory._factory_signature(),
        )

    def _assert_stable(self) -> None:
        _, wp = _require_warp_torch_cuda()
        _validate_warp_stream_identity(
            self.stream,
            self.prepared.device_ordinal,
            wp=wp,
        )
        self.context._validate_identity()
        if int(self.external_stream.cuda_stream) != int(self.stream.cuda_stream):
            raise ValueError("Warp and Torch comparison stream handles differ")
        self.cuda_lbvh_factory._assert_stable()
        if self._factory_signature() != self._frozen_signature:
            raise ValueError("Warp CUDA-LBVH comparison factory changed after construction")


class WarpCudaLbvhComparisonCommandBundle:
    """One-shot Warp scene-update then CUDA-LBVH native command sequence."""

    __slots__ = (
        "_binding_signature",
        "_sealed",
        "_state",
        "acceleration_stage",
        "evidence",
        "factory",
        "scene_bundle",
        "stable_views",
        "warp_bvh",
    )

    def __init__(
        self,
        *,
        factory: WarpCudaLbvhComparisonFactory,
        scene_bundle: object,
        acceleration_stage: object,
        warp_bvh: object,
        stable_views: tuple[WarpTorchStableView, ...],
        evidence: WarpCudaLbvhComparisonEvidence,
    ) -> None:
        self._sealed = False
        self.factory = factory
        self.scene_bundle = scene_bundle
        self.acceleration_stage = acceleration_stage
        self.warp_bvh = warp_bvh
        self.stable_views = stable_views
        self.evidence = evidence
        self._state = "open"
        self._binding_signature = self._bundle_signature()
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False) or hasattr(self, "_binding_signature"):
            raise AttributeError("Warp CUDA-LBVH comparison bundle is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Warp CUDA-LBVH comparison bundle is immutable")

    @property
    def state(self) -> str:
        return self._state

    def arm(self) -> None:
        """Perform the final full-graph validation before the timed boundary."""

        assert_outside_timed_boundary("Warp CUDA-LBVH comparison pre-enqueue validation")
        if self._state != "open":
            raise RuntimeError("Warp CUDA-LBVH comparison bundle can only be armed once")
        self.factory._assert_stable()
        self.scene_bundle.validate_bound()
        self.acceleration_stage.validate_bound()
        _validate_bound_graph(self)
        expected_evidence = _build_comparison_evidence(
            factory=self.factory,
            scene_bundle=self.scene_bundle,
            acceleration_stage=self.acceleration_stage,
            stable_views=self.stable_views,
            transform_frame_index=self.scene_bundle.transforms.transform_frame_index,
            primitive_count=self.acceleration_stage.outputs.primitive_count,
            node_count=self.acceleration_stage.outputs.node_count,
        )
        if self.evidence != expected_evidence:
            raise ValueError("Warp CUDA-LBVH comparison evidence changed after sealing")
        if self._bundle_signature() != self._binding_signature:
            raise ValueError("Warp CUDA-LBVH comparison binding changed after sealing")
        object.__setattr__(self, "_state", "armed")

    def enqueue_scene_update(self) -> None:
        """Submit the frozen Warp scene command first on the shared stream."""

        self._require_state("armed", operation="scene update")
        try:
            with self.factory.context.activate():
                self.scene_bundle.enqueue()
        except BaseException:
            object.__setattr__(self, "_state", "poisoned")
            raise
        object.__setattr__(self, "_state", "scene_enqueued")

    def enqueue_acceleration(self) -> None:
        """Submit the CUDA LBVH command second without recording completion."""

        self._require_state("scene_enqueued", operation="acceleration")
        try:
            self.acceleration_stage.enqueue_bound_stream()
        except BaseException:
            object.__setattr__(self, "_state", "poisoned")
            raise
        object.__setattr__(self, "_state", "acceleration_enqueued")

    def _require_state(self, expected: str, *, operation: str) -> None:
        if self._state != expected:
            raise RuntimeError(
                f"Warp CUDA-LBVH comparison {operation} requires state {expected!r}; found {self._state!r}"
            )

    def _bundle_signature(self) -> tuple[object, ...]:
        return (
            self.factory._factory_signature(),
            id(self.scene_bundle),
            self.scene_bundle.evidence.sha256,
            id(self.acceleration_stage),
            self.acceleration_stage.evidence.sha256,
            id(self.warp_bvh),
            _warp_bvh_binding_signature(self.warp_bvh),
            tuple((value.name, id(value.warp_array), id(value.torch_tensor)) for value in self.stable_views),
            _stable_view_alias_sha256(self.stable_views),
            self.evidence.sha256,
        )


def _build_comparison_evidence(
    *,
    factory: WarpCudaLbvhComparisonFactory,
    scene_bundle: object,
    acceleration_stage: object,
    stable_views: tuple[WarpTorchStableView, ...],
    transform_frame_index: int,
    primitive_count: int,
    node_count: int,
) -> WarpCudaLbvhComparisonEvidence:
    manifest = factory.prepared.manifest
    replay_sha256 = manifest.transform_replay_sha256
    scene_sha256 = manifest.scene_contract_sha256
    input_contract = manifest.input_contract
    if replay_sha256 is None or scene_sha256 is None or input_contract is None:
        raise ValueError("Warp CUDA-LBVH comparison evidence provenance is incomplete")
    stage_evidence = acceleration_stage.evidence
    return WarpCudaLbvhComparisonEvidence(
        schema_version=WARP_CUDA_LBVH_COMPARISON_SCHEMA_VERSION,
        operation_id=stage_evidence.operation_id,
        acceleration_id=stage_evidence.acceleration_id,
        acceleration_class=factory.acceleration_class,
        topology_origin_id=stage_evidence.topology_origin_id,
        topology_lifetime=stage_evidence.topology_lifetime,
        device_ordinal=factory.prepared.device_ordinal,
        warp_stream_handle=int(factory.stream.cuda_stream),
        torch_stream_handle=int(factory.external_stream.cuda_stream),
        native_input_manifest_sha256=manifest.sha256,
        scene_contract_sha256=scene_sha256,
        input_contract=input_contract,
        transform_replay_sha256=replay_sha256,
        transform_frame_index=transform_frame_index,
        primitive_count=primitive_count,
        node_count=node_count,
        stable_view_names=tuple(value.name for value in stable_views),
        stable_view_alias_sha256=_stable_view_alias_sha256(stable_views),
        warp_scene_stage_evidence_sha256=scene_bundle.evidence.sha256,
        cuda_lbvh_stage_evidence_sha256=stage_evidence.sha256,
        setup_stable_view_count=len(stable_views),
        bridge_creation_count=0,
    )


def _validate_bound_graph(bundle: WarpCudaLbvhComparisonCommandBundle) -> None:
    factory = bundle.factory
    scene_bundle = bundle.scene_bundle
    stage = bundle.acceleration_stage
    if scene_bundle.stream is not factory.stream:
        raise ValueError("Warp comparison scene command uses a different stream owner")
    if stage.factory is not factory.cuda_lbvh_factory or stage.factory.context is not factory.context:
        raise ValueError("Warp comparison acceleration stage uses a different factory/context")
    if not stage.world.resources or stage.world.resources[0] is not scene_bundle.outputs:
        raise ValueError("Warp comparison AABB views do not own the scene output graph")
    if not stage.outputs.resources or not isinstance(
        stage.outputs.resources[0], WarpCudaLbvhComparisonStorage
    ):
        raise ValueError("Warp comparison LBVH views do not own their Warp storage graph")
    storage = stage.outputs.resources[0]
    if (
        len(stage.workspace.resources) < 2
        or stage.workspace.resources[0] is not storage.workspace_owner
        or stage.workspace.resources[1] is not storage.workspace
    ):
        raise ValueError("Warp comparison workspace view lost its Warp owners")

    expected_names = (
        tuple(f"world.{name}" for name in _WORLD_VIEW_NAMES)
        + tuple(f"lbvh.{name}" for name, _, _ in _LBVH_ARRAY_SPECS)
        + ("lbvh.workspace",)
    )
    if tuple(value.name for value in bundle.stable_views) != expected_names:
        raise ValueError("Warp comparison stable-view ABI is incomplete or reordered")
    view_by_name = {value.name: value for value in bundle.stable_views}
    for name in _WORLD_VIEW_NAMES:
        view = view_by_name[f"world.{name}"]
        if view.warp_array is not getattr(scene_bundle.outputs, name) or view.torch_tensor is not getattr(
            stage.world, name
        ):
            raise ValueError("Warp comparison world-AABB stable view was rebound")
    for name, _, _ in _LBVH_ARRAY_SPECS:
        view = view_by_name[f"lbvh.{name}"]
        if view.warp_array is not storage.arrays[name] or view.torch_tensor is not getattr(
            stage.outputs, name
        ):
            raise ValueError("Warp comparison LBVH stable view was rebound")
    workspace_view = view_by_name["lbvh.workspace"]
    if (
        workspace_view.warp_array is not storage.workspace
        or workspace_view.torch_tensor is not stage.workspace.storage
    ):
        raise ValueError("Warp comparison workspace stable view was rebound")
    if bundle.warp_bvh.resources != (factory, storage):
        raise ValueError("Warp comparison traversal view lost its resource owners")
    expected_warp_arrays = {
        "bounds_min": storage.arrays["node_bounds_min"],
        "bounds_max": storage.arrays["node_bounds_max"],
        "left": storage.arrays["left"],
        "right": storage.arrays["right"],
        "start": storage.arrays["start"],
        "count": storage.arrays["count"],
        "prim_ids": storage.arrays["sorted_primitive_ids"],
    }
    if any(getattr(bundle.warp_bvh, name) is not value for name, value in expected_warp_arrays.items()):
        raise ValueError("Warp comparison traversal view does not use the CUDA LBVH outputs")
    if bundle.warp_bvh.ready_event is not None:
        raise ValueError("C3 Warp comparison binding must not publish an intermediate event")
    if (
        scene_bundle.evidence.native_input_manifest_sha256 != factory.prepared.manifest.sha256
        or scene_bundle.evidence.scene_contract_sha256 != factory.prepared.manifest.scene_contract_sha256
        or scene_bundle.evidence.transform_replay_sha256 != factory.prepared.manifest.transform_replay_sha256
        or scene_bundle.evidence.transform_frame_index != bundle.evidence.transform_frame_index
    ):
        raise ValueError("Warp comparison scene provenance does not match its prepared input")


def _warp_bvh_binding_signature(value: object) -> tuple[object, ...]:
    return (
        id(value.device),
        value.frame_id,
        value.env_idx,
        tuple(
            id(getattr(value, name))
            for name in (
                "bounds_min",
                "bounds_max",
                "left",
                "right",
                "start",
                "count",
                "prim_ids",
            )
        ),
        value.num_nodes,
        value.num_primitives,
        value.topology_origin_id,
        value.scene_contract_sha256,
        value.topology_sha256,
        value.ready_event,
        tuple(id(resource) for resource in value.resources),
    )


def allocate_warp_cuda_lbvh_comparison_storage(
    requirements: object,
    *,
    device: object,
    stream: object | None = None,
) -> WarpCudaLbvhComparisonStorage:
    """Allocate one Warp-owned C3 proof graph; C4 supplies final arena slots."""

    assert_outside_timed_boundary("Warp CUDA-LBVH comparison storage allocation")
    _, wp = _require_warp_torch_cuda()
    from optics.cuda_lbvh import CudaLbvhWorkspaceRequirements

    if not isinstance(requirements, CudaLbvhWorkspaceRequirements):
        raise TypeError("requirements must be CudaLbvhWorkspaceRequirements")
    resolved_device = wp.get_device(device)
    if int(resolved_device.ordinal) != requirements.device_ordinal:
        raise ValueError("Warp CUDA-LBVH comparison storage is on the wrong device")
    if stream is not None:
        _validate_warp_stream_identity(stream, requirements.device_ordinal, wp=wp)
    primitive_count = requirements.primitive_count
    node_count = requirements.node_capacity
    internal_count = max(primitive_count - 1, 0)
    shapes = {
        "scene": (3,) if primitive_count else (0,),
        "primitive": (primitive_count,),
        "node": (node_count,),
        "internal": (internal_count,),
        "node_vec3": (node_count, 3),
    }
    dtypes = {"float32": wp.float32, "int32": wp.int32, "int64": wp.int64}
    scope = wp.ScopedStream(stream) if stream is not None else nullcontext()
    with scope:
        arrays = {
            name: wp.empty(shapes[shape_id], dtype=dtypes[dtype_id], device=resolved_device)
            for name, dtype_id, shape_id in _LBVH_ARRAY_SPECS
        }
        required = requirements.required_bytes
        alignment = requirements.alignment_bytes
        if required:
            workspace_owner = wp.empty(
                (required + alignment - 1,),
                dtype=wp.uint8,
                device=resolved_device,
            )
            offset = (-int(workspace_owner.ptr)) % alignment
            workspace = workspace_owner[offset : offset + required]
        else:
            workspace_owner = wp.empty((0,), dtype=wp.uint8, device=resolved_device)
            workspace = workspace_owner
        node_depth = wp.empty((node_count,), dtype=wp.int32, device=resolved_device)
        level_ranges = wp.empty(
            (1 if node_count else 0, 2),
            dtype=wp.int32,
            device=resolved_device,
        )
    result = WarpCudaLbvhComparisonStorage(
        device=resolved_device,
        primitive_count=primitive_count,
        node_count=node_count,
        arrays=MappingProxyType(arrays),
        workspace=workspace,
        workspace_owner=workspace_owner,
        node_depth=node_depth,
        level_ranges=level_ranges,
        resources=tuple(arrays.values()) + (workspace_owner, workspace, node_depth, level_ranges),
    )
    _validate_comparison_storage(
        result,
        requirements=requirements,
        device_ordinal=requirements.device_ordinal,
        wp=wp,
    )
    return result


def _create_stable_views(
    *,
    world: object,
    storage: WarpCudaLbvhComparisonStorage,
    wp: object,
) -> tuple[WarpTorchStableView, ...]:
    views: list[WarpTorchStableView] = []
    for name in _WORLD_VIEW_NAMES:
        value = getattr(world, name)
        views.append(
            WarpTorchStableView(
                name=f"world.{name}",
                warp_array=value,
                torch_tensor=wp.to_torch(value),
            )
        )
    for name, _, _ in _LBVH_ARRAY_SPECS:
        value = storage.arrays[name]
        views.append(
            WarpTorchStableView(
                name=f"lbvh.{name}",
                warp_array=value,
                torch_tensor=wp.to_torch(value),
            )
        )
    views.append(
        WarpTorchStableView(
            name="lbvh.workspace",
            warp_array=storage.workspace,
            torch_tensor=wp.to_torch(storage.workspace),
        )
    )
    result = tuple(views)
    _validate_stable_views(result)
    return result


def _build_warp_lbvh_view(
    *,
    factory: WarpCudaLbvhComparisonFactory,
    storage: WarpCudaLbvhComparisonStorage,
    frame_index: int,
) -> object:
    from optics.cuda_lbvh import CUDA_LBVH_TOPOLOGY_ORIGIN_ID
    from optics.device_bvh import DeviceBvhBuildStats, DeviceOpticalBvh

    scene = factory.scene_factory.local_scene
    stats = DeviceBvhBuildStats(
        num_nodes=storage.node_count,
        num_leaves=storage.primitive_count,
        max_depth=0,
        leaf_size=1,
        sah_quality_cost=0.0,
        split_strategy="cuda_lbvh",
        supports_refit=False,
        level_ranges=((0, storage.node_count),) if storage.node_count else (),
    )
    return DeviceOpticalBvh(
        device=storage.device,
        frame_id=frame_index,
        env_idx=0,
        bounds_min=storage.arrays["node_bounds_min"],
        bounds_max=storage.arrays["node_bounds_max"],
        left=storage.arrays["left"],
        right=storage.arrays["right"],
        start=storage.arrays["start"],
        count=storage.arrays["count"],
        node_depth=storage.node_depth,
        level_ranges=storage.level_ranges,
        prim_ids=storage.arrays["sorted_primitive_ids"],
        primitive_global_id=scene.triangle_primitive_global_id,
        primitive_instance_index=scene.triangle_instance_index,
        primitive_index_within_instance=scene.triangle_primitive_index_within_instance,
        primitive_geometry_index=scene.triangle_geometry_index,
        primitive_geometry_primitive_index=scene.triangle_geometry_primitive_index,
        primitive_source_order_key=scene.triangle_source_order_key,
        stats=stats,
        topology_origin_id=CUDA_LBVH_TOPOLOGY_ORIGIN_ID,
        scene_contract_sha256=scene.scene_contract_sha256,
        topology_sha256=None,
        ready_event=None,
        resources=(factory, storage),
    )


def _validate_warp_stream_identity(stream: object, device_ordinal: int, *, wp: object) -> None:
    handle = getattr(stream, "cuda_stream", None)
    if not isinstance(handle, int) or handle == 0:
        raise ValueError("Warp CUDA-LBVH comparison requires a live non-default stream")
    stream_device = wp.get_device(getattr(stream, "device", None))
    if not stream_device.is_cuda or int(stream_device.ordinal) != device_ordinal:
        raise ValueError("Warp CUDA-LBVH comparison stream is on the wrong device")


def _validate_comparison_storage(
    storage: WarpCudaLbvhComparisonStorage,
    *,
    requirements: object,
    device_ordinal: int,
    wp: object,
) -> None:
    if not isinstance(storage, WarpCudaLbvhComparisonStorage):
        raise TypeError("storage must be WarpCudaLbvhComparisonStorage")
    if int(wp.get_device(storage.device).ordinal) != device_ordinal:
        raise ValueError("Warp CUDA-LBVH comparison storage device drifted")
    if (
        storage.primitive_count != requirements.primitive_count
        or storage.node_count != requirements.node_capacity
    ):
        raise ValueError("Warp CUDA-LBVH comparison storage count mismatch")
    if tuple(storage.arrays) != tuple(name for name, _, _ in _LBVH_ARRAY_SPECS):
        raise ValueError("Warp CUDA-LBVH comparison storage ABI is incomplete or reordered")
    if tuple(storage.workspace.shape) != (requirements.required_bytes,):
        raise ValueError("Warp CUDA-LBVH comparison workspace has the wrong size")
    if requirements.required_bytes and int(storage.workspace.ptr) % requirements.alignment_bytes:
        raise ValueError("Warp CUDA-LBVH comparison workspace is not sufficiently aligned")
    retained = tuple(id(value) for value in storage.resources)
    if id(storage.workspace_owner) not in retained or id(storage.workspace) not in retained:
        raise ValueError("Warp CUDA-LBVH comparison workspace owners are not retained")


def _validate_stable_views(views: tuple[WarpTorchStableView, ...]) -> None:
    names = tuple(value.name for value in views)
    if len(names) != len(set(names)):
        raise ValueError("Warp/Torch stable view names must be unique")
    for value in views:
        tensor = value.torch_tensor
        warp_array = value.warp_array
        if int(tensor.data_ptr()) != _warp_array_ptr(warp_array):
            raise ValueError(f"stable view {value.name!r} does not alias its Warp owner")
        if tuple(tensor.shape) != tuple(warp_array.shape):
            raise ValueError(f"stable view {value.name!r} shape differs from its Warp owner")
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"stable view {value.name!r} must be contiguous CUDA storage")


def _stable_view_alias_sha256(views: tuple[WarpTorchStableView, ...]) -> str:
    _validate_stable_views(views)
    return _canonical_sha256(
        [
            {
                "name": value.name,
                "warp_ptr": _warp_array_ptr(value.warp_array),
                "torch_ptr": int(value.torch_tensor.data_ptr()),
                "device_ordinal": int(value.torch_tensor.device.index),
                "dtype": str(value.torch_tensor.dtype),
                "shape": list(value.torch_tensor.shape),
                "stride": list(value.torch_tensor.stride()),
                "nbytes": int(value.torch_tensor.numel()) * int(value.torch_tensor.element_size()),
            }
            for value in views
        ]
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _warp_array_ptr(value: object) -> int:
    pointer = getattr(value, "ptr", None)
    return 0 if pointer is None else int(pointer)


def _require_warp_torch_cuda() -> tuple[object, object]:
    try:
        import torch
        import warp as wp
    except ImportError as exc:  # pragma: no cover - exercised in optional environments.
        raise ImportError("Warp CUDA-LBVH comparison requires Warp and PyTorch") from exc
    if not torch.cuda.is_available() or wp.get_cuda_device_count() <= 0:
        raise ImportError("Warp CUDA-LBVH comparison requires CUDA")
    return torch, wp
