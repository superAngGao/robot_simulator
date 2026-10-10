"""Device BVH buffers for GPU optical traversal."""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass, replace

import numpy as np

from .device_scene import (
    DeviceOpticalSceneSnapshot,
    WarpWorldSceneBuffers,
    _sha256_text,
    _validate_disjoint_warp_storage,
    _validate_warp_array,
    _validate_warp_stream,
    _warp_storage_manifest_sha256,
    _warp_world_scene_arrays,
)
from .host_bvh import (
    _CENTROID_EPS as _CENTROID_EPS,
)
from .host_bvh import (
    WARP_BVH_TOPOLOGY_SCHEMA_VERSION as WARP_BVH_TOPOLOGY_SCHEMA_VERSION,
)
from .host_bvh import (
    DeviceBvhBuildStats as DeviceBvhBuildStats,
)
from .host_bvh import (
    _canonical_sha256 as _canonical_sha256,
)
from .host_bvh import (
    _HostNode as _HostNode,
)
from .host_bvh import (
    _surface_area as _surface_area,
)
from .host_bvh import (
    build_host_median_bvh as build_host_median_bvh,
)
from .host_bvh import (
    host_bvh_topology_sha256 as host_bvh_topology_sha256,
)

try:  # pragma: no cover - exercised in GPU environments.
    import warp as wp

    _HAS_WARP = True
except Exception:  # pragma: no cover - keeps CPU-only imports working.
    wp = None
    _HAS_WARP = False

WARP_BVH_REFIT_OPERATION_ID = "warp_bvh_leaf_internal_refit_out_v1"
WARP_BVH_REFIT_ACCELERATION_ID = "cpu_median_leaf4_topology_warp_refit_v1"
WARP_BVH_REFIT_COMMAND_SCHEMA_VERSION = "q54.p12.4.warp-bvh-refit-command.v2"
WARP_BVH_TOPOLOGY_ORIGIN_ID = "cpu_median_longest_axis_leaf4_source_order_v1"


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
    topology_origin_id: str | None = None
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
    acceleration_id: str
    device_ordinal: int
    stream_handle: int
    native_input_manifest_sha256: str
    scene_contract_sha256: str
    input_contract: str
    native_input_topology_sha256: str
    topology_origin_id: str
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
        "_frozen_device_ordinal",
        "_frozen_immutable_storage_sha256",
        "_frozen_input_contract",
        "_frozen_native_input_manifest_sha256",
        "_frozen_native_input_topology_sha256",
        "_frozen_scene_contract_sha256",
        "_frozen_stream_handle",
        "_frozen_topology_facts",
        "_frozen_topology_origin_id",
        "_frozen_topology_sha256",
        "_sealed",
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
        _context=None,
    ) -> None:
        from optical_worker.native_setup import _native_resource_scope

        _require_warp()
        with _native_resource_scope(wp, topology.device, lane="warp", stream=stream, context=_context,
                owner_role="refit_factory_custody", scope_id="refit_factory_scope") as (lifecycle, owners, operation):
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
            if topology.topology_origin_id != WARP_BVH_TOPOLOGY_ORIGIN_ID:
                raise ValueError("Warp refit requires the frozen leaf-4 topology origin")
            if topology.stats.leaf_size != 4 or topology.stats.split_strategy != "sort":
                raise ValueError("Warp refit requires leaf_size=4 and split_strategy='sort'")
            if topology.ready_event is not None:
                with operation("refit_topology_ready_wait", "event_wait"):
                    if lifecycle is not None:
                        lifecycle.require_adopted(topology.ready_event, stream)
                        if topology.ready_event.device is not topology.device:
                            raise ValueError("refit topology event differs from its target")
                    stream.wait_event(topology.ready_event)
            self.topology = topology
            self.stream = stream
            self.native_input_manifest_sha256 = native_input_manifest_sha256
            self.native_input_topology_sha256 = native_input_topology_sha256
            self.scene_contract_sha256 = scene_contract_sha256
            self.input_contract = input_contract
            self._frozen_device_ordinal = int(topology.device.ordinal)
            self._frozen_stream_handle = int(stream.cuda_stream)
            self._frozen_scene_contract_sha256 = topology.scene_contract_sha256
            self._frozen_topology_origin_id = topology.topology_origin_id
            self._frozen_topology_sha256 = topology.topology_sha256
            self._frozen_immutable_storage_sha256 = _warp_storage_manifest_sha256(
                _warp_bvh_immutable_arrays(topology)
            )
            self._frozen_topology_facts = _warp_bvh_topology_facts(topology)
            self._frozen_native_input_manifest_sha256 = native_input_manifest_sha256
            self._frozen_native_input_topology_sha256 = native_input_topology_sha256
            self._frozen_input_contract = input_contract
            self._sealed = True

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("Warp BVH-refit command factory is immutable")
        object.__setattr__(self, name, value)

    def seal(
        self,
        *,
        world: WarpWorldSceneBuffers,
        bvh: DeviceOpticalBvh | None = None,
        _context=None,
    ) -> WarpBvhRefitCommandBundle:
        """Validate and record one immutable refit plan outside timing."""

        self._validate_frozen_state()
        bvh = self.topology if bvh is None else bvh
        if not isinstance(world, WarpWorldSceneBuffers):
            raise TypeError("world must be WarpWorldSceneBuffers")
        _validate_matching_warp_bvh_topology(self.topology, bvh)
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

        from optical_worker.native_setup import _warp_command_recording

        ready_event = bvh.ready_event if bvh is not self.topology else None
        with _warp_command_recording(wp, bvh.device, self.stream, context=_context,
                                     ready_event=ready_event) as record:
            commands: list[object] = []
            dispatches: list[dict[str, object]] = []
            if bvh.num_nodes > 0:
                commands.append(
                    record(
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
                    )
                )
                dispatches.append({"kernel": "refit_bvh_leaf_bounds", "dim": bvh.num_nodes})
                for depth in range(bvh.stats.max_depth - 1, -1, -1):
                    level_start, level_end = bvh.stats.level_ranges[depth]
                    level_count = int(level_end - level_start)
                    if level_count <= 0:
                        continue
                    commands.append(
                        record(
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
            acceleration_id=WARP_BVH_REFIT_ACCELERATION_ID,
            device_ordinal=int(bvh.device.ordinal),
            stream_handle=int(self.stream.cuda_stream),
            native_input_manifest_sha256=self.native_input_manifest_sha256,
            scene_contract_sha256=self.scene_contract_sha256,
            input_contract=self.input_contract,
            native_input_topology_sha256=self.native_input_topology_sha256,
            topology_origin_id=bvh.topology_origin_id,
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

    def _validate_frozen_state(self) -> None:
        topology = self.topology
        _validate_warp_stream(self.stream, device=topology.device)
        if int(topology.device.ordinal) != self._frozen_device_ordinal:
            raise ValueError("Warp refit factory device drifted after construction")
        if int(self.stream.cuda_stream) != self._frozen_stream_handle:
            raise ValueError("Warp refit factory stream drifted after construction")
        if topology.scene_contract_sha256 != self._frozen_scene_contract_sha256:
            raise ValueError("Warp refit factory scene contract drifted after construction")
        if topology.topology_origin_id != self._frozen_topology_origin_id:
            raise ValueError("Warp refit factory topology origin drifted after construction")
        if topology.topology_sha256 != self._frozen_topology_sha256:
            raise ValueError("Warp refit factory topology hash drifted after construction")
        if (
            _warp_storage_manifest_sha256(_warp_bvh_immutable_arrays(topology))
            != self._frozen_immutable_storage_sha256
        ):
            raise ValueError("Warp refit factory topology storage drifted after construction")
        if _warp_bvh_topology_facts(topology) != self._frozen_topology_facts:
            raise ValueError("Warp refit factory topology facts drifted after construction")
        if (
            self.native_input_manifest_sha256 != self._frozen_native_input_manifest_sha256
            or self.native_input_topology_sha256 != self._frozen_native_input_topology_sha256
            or self.scene_contract_sha256 != self._frozen_scene_contract_sha256
            or self.input_contract != self._frozen_input_contract
        ):
            raise ValueError("Warp refit factory provenance drifted after construction")


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
    if target.topology_origin_id != source.topology_origin_id:
        raise ValueError("Warp BVH slot topology origin does not match the factory")
    if _warp_storage_manifest_sha256(_warp_bvh_immutable_arrays(target)) != _warp_storage_manifest_sha256(
        _warp_bvh_immutable_arrays(source)
    ):
        raise ValueError("Warp BVH slot replaced immutable topology storage")
    if _warp_bvh_topology_facts(target) != _warp_bvh_topology_facts(source):
        raise ValueError("Warp BVH slot topology facts do not match the factory")


def _warp_bvh_topology_facts(bvh: DeviceOpticalBvh) -> tuple[object, ...]:
    return (
        bvh.stats.num_nodes,
        bvh.stats.num_leaves,
        bvh.stats.max_depth,
        bvh.stats.leaf_size,
        bvh.stats.split_strategy,
        bvh.stats.supports_refit,
        bvh.stats.level_ranges,
    )


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
        topology_origin_id=(
            WARP_BVH_TOPOLOGY_ORIGIN_ID if int(leaf_size) == 4 and split_strategy == "sort" else None
        ),
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
