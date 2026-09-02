"""Device BVH buffers for GPU optical traversal."""

from __future__ import annotations

import hashlib
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace

import numpy as np

from .device_scene import (
    DeviceOpticalSceneSnapshot,
    WarpWorldSceneBuffers,
    _canonical_sha256,
    _sha256_text,
    _validate_disjoint_warp_storage,
    _validate_warp_array,
    _validate_warp_stream,
    _warp_storage_manifest_sha256,
    _warp_world_scene_arrays,
)

try:  # pragma: no cover - exercised in GPU environments.
    import warp as wp

    _HAS_WARP = True
except Exception:  # pragma: no cover - keeps CPU-only imports working.
    wp = None
    _HAS_WARP = False

_CENTROID_EPS = 1.0e-12
WARP_BVH_REFIT_OPERATION_ID = "warp_bvh_leaf_internal_refit_out_v1"
WARP_BVH_REFIT_COMMAND_SCHEMA_VERSION = "q54.p12.4.warp-bvh-refit-command.v1"
WARP_BVH_TOPOLOGY_SCHEMA_VERSION = "q54.p12.4.warp-bvh-topology.v1"


@dataclass(frozen=True)
class DeviceBvhBuildStats:
    """Host-side diagnostics for a device BVH build."""

    num_nodes: int
    num_leaves: int
    max_depth: int
    leaf_size: int
    sah_quality_cost: float
    device_to_host_ms: float = 0.0
    host_build_ms: float = 0.0
    upload_ms: float = 0.0
    total_ms: float = 0.0
    split_strategy: str = "sort"
    supports_refit: bool = True
    level_ranges: tuple[tuple[int, int], ...] = ()
    detail_ms: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class DeviceOpticalBvh:
    """Device buffers for a triangle BVH over one optical scene snapshot.

    The first L5C.2 bridge builds this tree on the CPU from per-frame triangle
    AABBs, then uploads the flat node arrays for GPU traversal.
    """

    device: object
    frame_id: int
    env_idx: int
    bounds_min: object
    bounds_max: object
    left: object
    right: object
    start: object
    count: object
    node_depth: object
    level_ranges: object
    prim_ids: object
    primitive_global_id: object
    primitive_instance_index: object
    primitive_index_within_instance: object
    primitive_geometry_index: object
    primitive_geometry_primitive_index: object
    primitive_source_order_key: object
    stats: DeviceBvhBuildStats
    scene_contract_sha256: str | None = None
    topology_sha256: str | None = None
    ready_event: object | None = None
    resources: tuple[object, ...] = ()

    @property
    def num_nodes(self) -> int:
        return self.stats.num_nodes

    @property
    def num_primitives(self) -> int:
        return int(self.prim_ids.shape[0])


@dataclass(frozen=True)
class WarpBvhRefitCommandEvidence:
    """Independent facts captured while a Warp refit bundle is sealed."""

    schema_version: str
    operation_id: str
    device_ordinal: int
    stream_handle: int
    native_input_manifest_sha256: str
    scene_contract_sha256: str
    input_contract: str
    native_input_topology_sha256: str
    topology_sha256: str
    primitive_count: int
    node_count: int
    leaf_count: int
    input_storage_sha256: str
    output_storage_sha256: str
    dispatch_count: int
    dispatch_plan_sha256: str

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.__dict__)


class WarpBvhRefitCommandBundle:
    """Immutable pre-recorded Warp refit commands for one storage graph."""

    __slots__ = (
        "_commands",
        "_poisoned",
        "_sealed",
        "bvh",
        "evidence",
        "stream",
        "world",
    )

    def __init__(
        self,
        *,
        commands: tuple[object, ...],
        evidence: WarpBvhRefitCommandEvidence,
        world: WarpWorldSceneBuffers,
        bvh: DeviceOpticalBvh,
        stream: object,
    ) -> None:
        self._commands = commands
        self._poisoned = False
        self.evidence = evidence
        self.world = world
        self.bvh = bvh
        self.stream = stream
        self._sealed = True

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("Warp BVH-refit command bundle is immutable")
        object.__setattr__(self, name, value)

    @property
    def dispatch_count(self) -> int:
        return len(self._commands)

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def enqueue(self) -> None:
        """Launch only the frozen leaf/internal commands on the frozen stream."""

        if self._poisoned:
            raise RuntimeError("Warp BVH-refit command bundle is poisoned")
        try:
            for command in self._commands:
                command.launch(stream=self.stream)
        except Exception:
            object.__setattr__(self, "_poisoned", True)
            raise


class WarpBvhRefitCommandFactory:
    """Seal immutable topology and caller-owned slot bounds into Warp commands."""

    __slots__ = (
        "input_contract",
        "native_input_manifest_sha256",
        "native_input_topology_sha256",
        "scene_contract_sha256",
        "stream",
        "topology",
    )

    def __init__(
        self,
        topology: DeviceOpticalBvh,
        *,
        stream: object,
        native_input_manifest_sha256: str,
        native_input_topology_sha256: str,
        scene_contract_sha256: str,
        input_contract: str,
    ) -> None:
        _require_warp()
        if (
            not _sha256_text(native_input_manifest_sha256)
            or not _sha256_text(native_input_topology_sha256)
            or not _sha256_text(scene_contract_sha256)
            or not input_contract
        ):
            raise ValueError("Warp BVH-refit provenance must be non-empty")
        _validate_warp_stream(stream, device=topology.device)
        _validate_warp_bvh_storage(topology)
        if topology.scene_contract_sha256 != scene_contract_sha256:
            raise ValueError("Warp BVH topology belongs to a different scene contract")
        if not _sha256_text(topology.topology_sha256):
            raise ValueError("Warp BVH lacks canonical topology evidence")
        if topology.ready_event is not None:
            stream.wait_event(topology.ready_event)
        self.topology = topology
        self.stream = stream
        self.native_input_manifest_sha256 = native_input_manifest_sha256
        self.native_input_topology_sha256 = native_input_topology_sha256
        self.scene_contract_sha256 = scene_contract_sha256
        self.input_contract = input_contract

    def seal(
        self,
        *,
        world: WarpWorldSceneBuffers,
        bvh: DeviceOpticalBvh | None = None,
    ) -> WarpBvhRefitCommandBundle:
        """Validate and record one immutable refit plan outside timing."""

        bvh = self.topology if bvh is None else bvh
        if not isinstance(world, WarpWorldSceneBuffers):
            raise TypeError("world must be WarpWorldSceneBuffers")
        _validate_matching_warp_bvh_topology(self.topology, bvh)
        if bvh.ready_event is not None and bvh is not self.topology:
            self.stream.wait_event(bvh.ready_event)
        if wp.get_device(world.device) != wp.get_device(bvh.device):
            raise ValueError("Warp BVH and world-scene output devices must match")
        for name in ("triangle_aabb_min", "triangle_aabb_max"):
            _validate_warp_array(
                getattr(world, name),
                name=name,
                dtype=wp.float32,
                ndim=2,
                shape=(bvh.num_primitives, 3),
                device=bvh.device,
            )
        input_arrays = _warp_refit_input_arrays(world, bvh)
        output_arrays = _warp_refit_output_arrays(bvh)
        _validate_disjoint_warp_storage(
            _warp_world_scene_arrays(world) + input_arrays,
            output_arrays,
        )

        commands: list[object] = []
        dispatches: list[dict[str, object]] = []
        if bvh.num_nodes > 0:
            commands.append(
                wp.launch(
                    _refit_bvh_leaf_bounds_kernel,
                    dim=bvh.num_nodes,
                    inputs=(
                        world.triangle_aabb_min,
                        world.triangle_aabb_max,
                        bvh.start,
                        bvh.count,
                        bvh.prim_ids,
                        bvh.bounds_min,
                        bvh.bounds_max,
                    ),
                    device=bvh.device,
                    stream=self.stream,
                    record_tape=False,
                    record_cmd=True,
                )
            )
            dispatches.append({"kernel": "refit_bvh_leaf_bounds", "dim": bvh.num_nodes})
            for depth in range(bvh.stats.max_depth - 1, -1, -1):
                level_start, level_end = bvh.stats.level_ranges[depth]
                level_count = int(level_end - level_start)
                if level_count <= 0:
                    continue
                commands.append(
                    wp.launch(
                        _refit_bvh_internal_bounds_kernel,
                        dim=level_count,
                        inputs=(
                            int(level_start),
                            bvh.left,
                            bvh.right,
                            bvh.count,
                            bvh.bounds_min,
                            bvh.bounds_max,
                        ),
                        device=bvh.device,
                        stream=self.stream,
                        record_tape=False,
                        record_cmd=True,
                    )
                )
                dispatches.append(
                    {
                        "kernel": "refit_bvh_internal_bounds",
                        "depth": depth,
                        "level_start": int(level_start),
                        "dim": level_count,
                    }
                )

        evidence = WarpBvhRefitCommandEvidence(
            schema_version=WARP_BVH_REFIT_COMMAND_SCHEMA_VERSION,
            operation_id=WARP_BVH_REFIT_OPERATION_ID,
            device_ordinal=int(bvh.device.ordinal),
            stream_handle=int(self.stream.cuda_stream),
            native_input_manifest_sha256=self.native_input_manifest_sha256,
            scene_contract_sha256=self.scene_contract_sha256,
            input_contract=self.input_contract,
            native_input_topology_sha256=self.native_input_topology_sha256,
            topology_sha256=bvh.topology_sha256,
            primitive_count=bvh.num_primitives,
            node_count=bvh.num_nodes,
            leaf_count=bvh.stats.num_leaves,
            input_storage_sha256=_warp_storage_manifest_sha256(input_arrays),
            output_storage_sha256=_warp_storage_manifest_sha256(output_arrays),
            dispatch_count=len(commands),
            dispatch_plan_sha256=_canonical_sha256(dispatches),
        )
        return WarpBvhRefitCommandBundle(
            commands=tuple(commands),
            evidence=evidence,
            world=world,
            bvh=bvh,
            stream=self.stream,
        )


def _validate_warp_bvh_storage(bvh: DeviceOpticalBvh) -> None:
    if not isinstance(bvh, DeviceOpticalBvh):
        raise TypeError("bvh must be DeviceOpticalBvh")
    if not bvh.stats.supports_refit:
        raise ValueError("Warp BVH topology does not support refit")
    device = bvh.device
    nodes = bvh.num_nodes
    primitives = bvh.num_primitives
    for name in ("bounds_min", "bounds_max"):
        _validate_warp_array(
            getattr(bvh, name),
            name=f"bvh_{name}",
            dtype=wp.float32,
            ndim=2,
            shape=(nodes, 3),
            device=device,
        )
    for name in ("left", "right", "start", "count", "node_depth"):
        _validate_warp_array(
            getattr(bvh, name),
            name=f"bvh_{name}",
            dtype=wp.int32,
            ndim=1,
            shape=(nodes,),
            device=device,
        )
    _validate_warp_array(
        bvh.level_ranges,
        name="bvh_level_ranges",
        dtype=wp.int32,
        ndim=2,
        device=device,
    )
    _validate_warp_array(
        bvh.prim_ids,
        name="bvh_prim_ids",
        dtype=wp.int32,
        ndim=1,
        shape=(primitives,),
        device=device,
    )
    for name in (
        "primitive_global_id",
        "primitive_instance_index",
        "primitive_index_within_instance",
        "primitive_geometry_index",
        "primitive_geometry_primitive_index",
    ):
        _validate_warp_array(
            getattr(bvh, name),
            name=name,
            dtype=wp.int32,
            ndim=1,
            shape=(primitives,),
            device=device,
        )
    _validate_warp_array(
        bvh.primitive_source_order_key,
        name="primitive_source_order_key",
        dtype=wp.int64,
        ndim=1,
        shape=(primitives,),
        device=device,
    )
    expected_levels = 0 if nodes == 0 else bvh.stats.max_depth + 1
    if tuple(bvh.level_ranges.shape) != (expected_levels, 2):
        raise ValueError("Warp BVH level-range storage does not match topology stats")
    if len(bvh.stats.level_ranges) != expected_levels:
        raise ValueError("Warp BVH host level ranges do not match topology stats")
    if nodes == 0:
        if primitives != 0 or bvh.stats.num_leaves != 0:
            raise ValueError("empty Warp BVH topology counts are inconsistent")
    elif bvh.stats.num_leaves <= 0:
        raise ValueError("non-empty Warp BVH must contain a leaf")


def _validate_matching_warp_bvh_topology(
    source: DeviceOpticalBvh,
    target: DeviceOpticalBvh,
) -> None:
    _validate_warp_bvh_storage(target)
    if wp.get_device(target.device) != wp.get_device(source.device):
        raise ValueError("Warp BVH slot and topology devices do not match")
    if target.scene_contract_sha256 != source.scene_contract_sha256:
        raise ValueError("Warp BVH slot belongs to a different scene contract")
    if target.topology_sha256 != source.topology_sha256:
        raise ValueError("Warp BVH slot topology hash does not match the factory")
    if _warp_storage_manifest_sha256(
        _warp_bvh_immutable_arrays(target)
    ) != _warp_storage_manifest_sha256(_warp_bvh_immutable_arrays(source)):
        raise ValueError("Warp BVH slot replaced immutable topology storage")
    source_facts = (
        source.stats.num_nodes,
        source.stats.num_leaves,
        source.stats.max_depth,
        source.stats.leaf_size,
        source.stats.level_ranges,
    )
    target_facts = (
        target.stats.num_nodes,
        target.stats.num_leaves,
        target.stats.max_depth,
        target.stats.leaf_size,
        target.stats.level_ranges,
    )
    if target_facts != source_facts:
        raise ValueError("Warp BVH slot topology facts do not match the factory")


def _warp_bvh_immutable_arrays(
    bvh: DeviceOpticalBvh,
) -> tuple[tuple[str, object], ...]:
    return (
        ("bvh_left", bvh.left),
        ("bvh_right", bvh.right),
        ("bvh_start", bvh.start),
        ("bvh_count", bvh.count),
        ("bvh_node_depth", bvh.node_depth),
        ("bvh_level_ranges", bvh.level_ranges),
        ("bvh_prim_ids", bvh.prim_ids),
        ("primitive_global_id", bvh.primitive_global_id),
        ("primitive_instance_index", bvh.primitive_instance_index),
        ("primitive_index_within_instance", bvh.primitive_index_within_instance),
        ("primitive_geometry_index", bvh.primitive_geometry_index),
        (
            "primitive_geometry_primitive_index",
            bvh.primitive_geometry_primitive_index,
        ),
        ("primitive_source_order_key", bvh.primitive_source_order_key),
    )


def _warp_refit_input_arrays(
    world: WarpWorldSceneBuffers,
    bvh: DeviceOpticalBvh,
) -> tuple[tuple[str, object], ...]:
    return (
        ("triangle_aabb_min", world.triangle_aabb_min),
        ("triangle_aabb_max", world.triangle_aabb_max),
    ) + _warp_bvh_immutable_arrays(bvh)


def _warp_refit_output_arrays(
    bvh: DeviceOpticalBvh,
) -> tuple[tuple[str, object], ...]:
    return (
        ("bvh_bounds_min", bvh.bounds_min),
        ("bvh_bounds_max", bvh.bounds_max),
    )


def build_device_bvh_from_snapshot(
    snapshot: DeviceOpticalSceneSnapshot,
    *,
    leaf_size: int = 4,
    split_strategy: str = "sort",
    device=None,
    stream=None,
) -> DeviceOpticalBvh:
    """Build a median-split triangle BVH and upload flat buffers to device."""
    total_start = time.perf_counter()
    _require_warp()
    if leaf_size <= 0:
        raise ValueError("leaf_size must be > 0")
    if split_strategy not in {"sort", "partition"}:
        raise ValueError("split_strategy must be 'sort' or 'partition'")
    if snapshot.triangle_aabb_min is None or snapshot.triangle_aabb_max is None:
        raise ValueError("build_device_bvh_from_snapshot requires AABB snapshot buffers")
    wp.init()
    resolved_device = snapshot.scene.device if device is None else wp.get_device(device)
    if resolved_device != snapshot.scene.device:
        raise ValueError("DeviceOpticalBvh device must match DeviceOpticalSceneSnapshot device")

    device_to_host_start = time.perf_counter()
    _synchronize_event(snapshot.ready_event)
    scene = snapshot.scene
    aabb_min = np.asarray(snapshot.triangle_aabb_min.numpy(), dtype=np.float32)
    aabb_max = np.asarray(snapshot.triangle_aabb_max.numpy(), dtype=np.float32)
    source_keys = np.asarray(scene.triangle_source_order_key.numpy(), dtype=np.int64)
    primitive_global_id = np.asarray(scene.triangle_primitive_global_id.numpy(), dtype=np.int32)
    primitive_instance_index = np.asarray(scene.triangle_instance_index.numpy(), dtype=np.int32)
    primitive_index_within_instance = np.asarray(
        scene.triangle_primitive_index_within_instance.numpy(),
        dtype=np.int32,
    )
    primitive_geometry_index = np.asarray(scene.triangle_geometry_index.numpy(), dtype=np.int32)
    primitive_geometry_primitive_index = np.asarray(
        scene.triangle_geometry_primitive_index.numpy(),
        dtype=np.int32,
    )
    device_to_host_ms = (time.perf_counter() - device_to_host_start) * 1000.0

    host_build_start = time.perf_counter()
    host_bvh = build_host_median_bvh(
        aabb_min,
        aabb_max,
        source_keys,
        leaf_size=int(leaf_size),
        split_strategy=split_strategy,
    )
    topology_sha256 = host_bvh_topology_sha256(host_bvh)
    host_build_ms = (time.perf_counter() - host_build_start) * 1000.0

    upload_start = time.perf_counter()
    with _scoped_stream(stream):
        bounds_min = _wp_array(host_bvh["bounds_min"], dtype=wp.float32, device=resolved_device)
        bounds_max = _wp_array(host_bvh["bounds_max"], dtype=wp.float32, device=resolved_device)
        left = _wp_array(host_bvh["left"], dtype=wp.int32, device=resolved_device)
        right = _wp_array(host_bvh["right"], dtype=wp.int32, device=resolved_device)
        start = _wp_array(host_bvh["start"], dtype=wp.int32, device=resolved_device)
        count = _wp_array(host_bvh["count"], dtype=wp.int32, device=resolved_device)
        node_depth = _wp_array(host_bvh["node_depth"], dtype=wp.int32, device=resolved_device)
        level_ranges = _wp_array(host_bvh["level_ranges"], dtype=wp.int32, device=resolved_device)
        prim_ids = _wp_array(host_bvh["prim_ids"], dtype=wp.int32, device=resolved_device)
        primitive_global_id_wp = _wp_array(primitive_global_id, dtype=wp.int32, device=resolved_device)
        primitive_instance_index_wp = _wp_array(
            primitive_instance_index, dtype=wp.int32, device=resolved_device
        )
        primitive_index_within_instance_wp = _wp_array(
            primitive_index_within_instance,
            dtype=wp.int32,
            device=resolved_device,
        )
        primitive_geometry_index_wp = _wp_array(
            primitive_geometry_index, dtype=wp.int32, device=resolved_device
        )
        primitive_geometry_primitive_index_wp = _wp_array(
            primitive_geometry_primitive_index,
            dtype=wp.int32,
            device=resolved_device,
        )
        primitive_source_order_key_wp = _wp_array(source_keys, dtype=wp.int64, device=resolved_device)
        ready_event = (stream or wp.get_stream(resolved_device)).record_event()
    upload_ms = (time.perf_counter() - upload_start) * 1000.0
    stats = replace(
        host_bvh["stats"],
        device_to_host_ms=device_to_host_ms,
        host_build_ms=host_build_ms,
        upload_ms=upload_ms,
        total_ms=(time.perf_counter() - total_start) * 1000.0,
        split_strategy=split_strategy,
    )

    resources = (
        bounds_min,
        bounds_max,
        left,
        right,
        start,
        count,
        node_depth,
        level_ranges,
        prim_ids,
        primitive_global_id_wp,
        primitive_instance_index_wp,
        primitive_index_within_instance_wp,
        primitive_geometry_index_wp,
        primitive_geometry_primitive_index_wp,
        primitive_source_order_key_wp,
    )
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
        primitive_global_id=primitive_global_id_wp,
        primitive_instance_index=primitive_instance_index_wp,
        primitive_index_within_instance=primitive_index_within_instance_wp,
        primitive_geometry_index=primitive_geometry_index_wp,
        primitive_geometry_primitive_index=primitive_geometry_primitive_index_wp,
        primitive_source_order_key=primitive_source_order_key_wp,
        stats=stats,
        scene_contract_sha256=scene.scene_contract_sha256,
        topology_sha256=topology_sha256,
        ready_event=ready_event,
        resources=resources,
    )


def refit_device_bvh_from_snapshot(
    snapshot: DeviceOpticalSceneSnapshot,
    bvh: DeviceOpticalBvh,
    *,
    stream=None,
) -> DeviceOpticalBvh:
    """Refit an existing BVH topology from per-frame triangle AABBs on the GPU."""
    _require_warp()
    if snapshot.triangle_aabb_min is None or snapshot.triangle_aabb_max is None:
        raise ValueError("refit_device_bvh_from_snapshot requires AABB snapshot buffers")
    if snapshot.scene.device != bvh.device:
        raise ValueError("DeviceOpticalBvh device must match DeviceOpticalSceneSnapshot device")
    if snapshot.scene.num_triangles != bvh.num_primitives:
        raise ValueError("DeviceOpticalBvh primitive count must match DeviceOpticalSceneSnapshot")
    if not bvh.stats.supports_refit:
        raise ValueError("DeviceOpticalBvh topology does not support GPU refit")

    with _scoped_stream(stream):
        _wait_on_event(snapshot.ready_event, stream=stream, device=bvh.device)
        _wait_on_event(bvh.ready_event, stream=stream, device=bvh.device)
        if bvh.num_nodes > 0:
            wp.launch(
                _refit_bvh_leaf_bounds_kernel,
                dim=bvh.num_nodes,
                inputs=[
                    snapshot.triangle_aabb_min,
                    snapshot.triangle_aabb_max,
                    bvh.start,
                    bvh.count,
                    bvh.prim_ids,
                    bvh.bounds_min,
                    bvh.bounds_max,
                ],
                device=bvh.device,
                stream=stream,
            )
            for depth in range(bvh.stats.max_depth - 1, -1, -1):
                level_start, level_end = bvh.stats.level_ranges[depth]
                wp.launch(
                    _refit_bvh_internal_bounds_kernel,
                    dim=level_end - level_start,
                    inputs=[
                        int(level_start),
                        bvh.left,
                        bvh.right,
                        bvh.count,
                        bvh.bounds_min,
                        bvh.bounds_max,
                    ],
                    device=bvh.device,
                    stream=stream,
                )
        ready_event = (stream or wp.get_stream(bvh.device)).record_event()

    return replace(
        bvh,
        frame_id=snapshot.frame_id,
        env_idx=snapshot.env_idx,
        ready_event=ready_event,
    )


@dataclass
class _HostNode:
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    left: int = -1
    right: int = -1
    start: int = -1
    count: int = 0
    depth: int = 0


def build_host_median_bvh(
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    source_keys: np.ndarray,
    *,
    leaf_size: int,
    split_strategy: str = "sort",
) -> dict[str, object]:
    """Build the canonical CPU median/longest-axis normalized BVH arrays."""
    if split_strategy not in {"sort", "partition"}:
        raise ValueError("split_strategy must be 'sort' or 'partition'")
    num_primitives = int(aabb_min.shape[0])
    if num_primitives != int(aabb_max.shape[0]) or num_primitives != int(source_keys.shape[0]):
        raise ValueError("BVH AABB and source-key buffers must have matching lengths")
    if num_primitives == 0:
        stats = DeviceBvhBuildStats(
            num_nodes=0,
            num_leaves=0,
            max_depth=0,
            leaf_size=leaf_size,
            sah_quality_cost=0.0,
            level_ranges=(),
        )
        return {
            "bounds_min": np.empty((0, 3), dtype=np.float32),
            "bounds_max": np.empty((0, 3), dtype=np.float32),
            "left": np.empty(0, dtype=np.int32),
            "right": np.empty(0, dtype=np.int32),
            "start": np.empty(0, dtype=np.int32),
            "count": np.empty(0, dtype=np.int32),
            "node_depth": np.empty(0, dtype=np.int32),
            "level_ranges": np.empty((0, 2), dtype=np.int32),
            "prim_ids": np.empty(0, dtype=np.int32),
            "stats": stats,
        }

    centroids = (aabb_min.astype(np.float64) + aabb_max.astype(np.float64)) * 0.5
    raw_nodes: list[_HostNode] = []
    leaf_prim_ids: list[int] = []

    def build_recursive(prim_ids: np.ndarray, depth: int) -> int:
        node_index = len(raw_nodes)
        node_min = np.min(aabb_min[prim_ids], axis=0)
        node_max = np.max(aabb_max[prim_ids], axis=0)
        raw_nodes.append(
            _HostNode(
                bounds_min=node_min.astype(np.float32),
                bounds_max=node_max.astype(np.float32),
                depth=depth,
            )
        )
        centroid_min = np.min(centroids[prim_ids], axis=0)
        centroid_max = np.max(centroids[prim_ids], axis=0)
        extent = centroid_max - centroid_min
        should_leaf = len(prim_ids) <= leaf_size or float(np.max(extent)) <= _CENTROID_EPS
        if should_leaf:
            ordered = prim_ids[np.argsort(source_keys[prim_ids], kind="stable")]
            raw_nodes[node_index].start = len(leaf_prim_ids)
            raw_nodes[node_index].count = int(len(ordered))
            leaf_prim_ids.extend(int(prim_id) for prim_id in ordered)
            return node_index

        axis = int(np.argmax(extent))
        if split_strategy == "sort":
            order = np.lexsort((source_keys[prim_ids], centroids[prim_ids, axis]))
        else:
            order = np.argpartition(centroids[prim_ids, axis], len(prim_ids) // 2)
        ordered = prim_ids[order]
        mid = len(ordered) // 2
        if mid <= 0 or mid >= len(ordered):
            ordered = prim_ids[np.argsort(source_keys[prim_ids], kind="stable")]
            raw_nodes[node_index].start = len(leaf_prim_ids)
            raw_nodes[node_index].count = int(len(ordered))
            leaf_prim_ids.extend(int(prim_id) for prim_id in ordered)
            return node_index

        raw_nodes[node_index].left = build_recursive(ordered[:mid], depth + 1)
        raw_nodes[node_index].right = build_recursive(ordered[mid:], depth + 1)
        return node_index

    root_old_index = build_recursive(np.arange(num_primitives, dtype=np.int32), 0)
    ordered_old_indices: list[int] = []
    queue = [root_old_index]
    while queue:
        old_index = queue.pop(0)
        ordered_old_indices.append(old_index)
        node = raw_nodes[old_index]
        if node.count == 0:
            queue.append(node.left)
            queue.append(node.right)

    old_to_new = {old_index: new_index for new_index, old_index in enumerate(ordered_old_indices)}
    nodes: list[_HostNode] = []
    for old_index in ordered_old_indices:
        old_node = raw_nodes[old_index]
        nodes.append(
            _HostNode(
                bounds_min=old_node.bounds_min,
                bounds_max=old_node.bounds_max,
                left=old_to_new.get(old_node.left, -1),
                right=old_to_new.get(old_node.right, -1),
                start=old_node.start,
                count=old_node.count,
                depth=old_node.depth,
            )
        )

    bounds_min = np.asarray([node.bounds_min for node in nodes], dtype=np.float32)
    bounds_max = np.asarray([node.bounds_max for node in nodes], dtype=np.float32)
    left = np.asarray([node.left for node in nodes], dtype=np.int32)
    right = np.asarray([node.right for node in nodes], dtype=np.int32)
    start = np.asarray([node.start for node in nodes], dtype=np.int32)
    count = np.asarray([node.count for node in nodes], dtype=np.int32)
    node_depth = np.asarray([node.depth for node in nodes], dtype=np.int32)
    max_depth = int(np.max(node_depth))
    level_ranges = np.empty((max_depth + 1, 2), dtype=np.int32)
    for depth in range(max_depth + 1):
        indices = np.flatnonzero(node_depth == depth)
        level_ranges[depth, 0] = int(indices[0])
        level_ranges[depth, 1] = int(indices[-1] + 1)

    root_area = _surface_area(bounds_min[0], bounds_max[0])
    sah_quality_cost = 0.0
    num_leaves = 0
    if root_area > 0.0:
        for node in nodes:
            if node.count > 0:
                num_leaves += 1
                sah_quality_cost += _surface_area(node.bounds_min, node.bounds_max) / root_area * node.count

    stats = DeviceBvhBuildStats(
        num_nodes=len(nodes),
        num_leaves=num_leaves,
        max_depth=max_depth,
        leaf_size=leaf_size,
        sah_quality_cost=float(sah_quality_cost),
        split_strategy=split_strategy,
        level_ranges=tuple((int(start), int(end)) for start, end in level_ranges),
    )
    return {
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "left": left,
        "right": right,
        "start": start,
        "count": count,
        "node_depth": node_depth,
        "level_ranges": level_ranges,
        "prim_ids": np.asarray(leaf_prim_ids, dtype=np.int32),
        "stats": stats,
    }


def host_bvh_topology_sha256(host_bvh: dict[str, object]) -> str:
    """Hash the complete normalized CPU topology before its one-time upload."""

    digest = hashlib.sha256(WARP_BVH_TOPOLOGY_SCHEMA_VERSION.encode("ascii"))
    for name in (
        "left",
        "right",
        "start",
        "count",
        "node_depth",
        "level_ranges",
        "prim_ids",
    ):
        value = np.asarray(host_bvh[name])
        if not value.flags.c_contiguous:
            raise ValueError(f"host BVH topology array {name!r} must be contiguous")
        digest.update(name.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(str(tuple(int(dimension) for dimension in value.shape)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _surface_area(bounds_min: np.ndarray, bounds_max: np.ndarray) -> float:
    extent = np.maximum(
        np.asarray(bounds_max, dtype=np.float64) - np.asarray(bounds_min, dtype=np.float64), 0.0
    )
    return float(2.0 * (extent[0] * extent[1] + extent[1] * extent[2] + extent[2] * extent[0]))


def _wp_array(values: np.ndarray, *, dtype, device):
    return wp.array(values, dtype=dtype, device=device)


def _scoped_stream(stream):
    if stream is None:
        return nullcontext()
    return wp.ScopedStream(stream)


def _synchronize_event(event) -> None:
    if event is None:
        return
    wp.synchronize_event(event)


def _wait_on_event(event, *, stream, device) -> None:
    if event is None:
        return
    (stream or wp.get_stream(device)).wait_event(event)


def _require_warp() -> None:
    if not _HAS_WARP:
        raise ImportError("DeviceOpticalBvh requires the optional warp package")


if _HAS_WARP:

    @wp.kernel
    def _refit_bvh_leaf_bounds_kernel(
        triangle_aabb_min: wp.array2d(dtype=wp.float32),
        triangle_aabb_max: wp.array2d(dtype=wp.float32),
        bvh_start: wp.array(dtype=wp.int32),
        bvh_count: wp.array(dtype=wp.int32),
        bvh_prim_ids: wp.array(dtype=wp.int32),
        bvh_bounds_min: wp.array2d(dtype=wp.float32),
        bvh_bounds_max: wp.array2d(dtype=wp.float32),
    ):
        node = wp.tid()
        count = bvh_count[node]
        if count > 0:
            start = bvh_start[node]
            first_prim = bvh_prim_ids[start]
            min_x = triangle_aabb_min[first_prim, 0]
            min_y = triangle_aabb_min[first_prim, 1]
            min_z = triangle_aabb_min[first_prim, 2]
            max_x = triangle_aabb_max[first_prim, 0]
            max_y = triangle_aabb_max[first_prim, 1]
            max_z = triangle_aabb_max[first_prim, 2]
            for offset in range(1, count):
                prim = bvh_prim_ids[start + offset]
                min_x = wp.min(min_x, triangle_aabb_min[prim, 0])
                min_y = wp.min(min_y, triangle_aabb_min[prim, 1])
                min_z = wp.min(min_z, triangle_aabb_min[prim, 2])
                max_x = wp.max(max_x, triangle_aabb_max[prim, 0])
                max_y = wp.max(max_y, triangle_aabb_max[prim, 1])
                max_z = wp.max(max_z, triangle_aabb_max[prim, 2])
            bvh_bounds_min[node, 0] = min_x
            bvh_bounds_min[node, 1] = min_y
            bvh_bounds_min[node, 2] = min_z
            bvh_bounds_max[node, 0] = max_x
            bvh_bounds_max[node, 1] = max_y
            bvh_bounds_max[node, 2] = max_z

    @wp.kernel
    def _refit_bvh_internal_bounds_kernel(
        level_start: int,
        bvh_left: wp.array(dtype=wp.int32),
        bvh_right: wp.array(dtype=wp.int32),
        bvh_count: wp.array(dtype=wp.int32),
        bvh_bounds_min: wp.array2d(dtype=wp.float32),
        bvh_bounds_max: wp.array2d(dtype=wp.float32),
    ):
        node = level_start + wp.tid()
        if bvh_count[node] == 0:
            left = bvh_left[node]
            right = bvh_right[node]
            if left >= 0 and right >= 0:
                bvh_bounds_min[node, 0] = wp.min(bvh_bounds_min[left, 0], bvh_bounds_min[right, 0])
                bvh_bounds_min[node, 1] = wp.min(bvh_bounds_min[left, 1], bvh_bounds_min[right, 1])
                bvh_bounds_min[node, 2] = wp.min(bvh_bounds_min[left, 2], bvh_bounds_min[right, 2])
                bvh_bounds_max[node, 0] = wp.max(bvh_bounds_max[left, 0], bvh_bounds_max[right, 0])
                bvh_bounds_max[node, 1] = wp.max(bvh_bounds_max[left, 1], bvh_bounds_max[right, 1])
                bvh_bounds_max[node, 2] = wp.max(bvh_bounds_max[left, 2], bvh_bounds_max[right, 2])

else:

    def _refit_bvh_leaf_bounds_kernel(*args, **kwargs):  # pragma: no cover
        raise ImportError("DeviceOpticalBvh requires the optional warp package")

    def _refit_bvh_internal_bounds_kernel(*args, **kwargs):  # pragma: no cover
        raise ImportError("DeviceOpticalBvh requires the optional warp package")
