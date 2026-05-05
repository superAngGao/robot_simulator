"""Device BVH buffers for GPU optical traversal."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np

from .device_scene import DeviceOpticalSceneSnapshot

try:  # pragma: no cover - exercised in GPU environments.
    import warp as wp

    _HAS_WARP = True
except Exception:  # pragma: no cover - keeps CPU-only imports working.
    wp = None
    _HAS_WARP = False

_CENTROID_EPS = 1.0e-12


@dataclass(frozen=True)
class DeviceBvhBuildStats:
    """Host-side diagnostics for a device BVH build."""

    num_nodes: int
    num_leaves: int
    max_depth: int
    leaf_size: int
    sah_quality_cost: float


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
    resources: tuple[object, ...] = ()

    @property
    def num_nodes(self) -> int:
        return self.stats.num_nodes

    @property
    def num_primitives(self) -> int:
        return int(self.prim_ids.shape[0])


def build_device_bvh_from_snapshot(
    snapshot: DeviceOpticalSceneSnapshot,
    *,
    leaf_size: int = 4,
    device=None,
    stream=None,
) -> DeviceOpticalBvh:
    """Build a median-split triangle BVH and upload flat buffers to device."""
    _require_warp()
    if leaf_size <= 0:
        raise ValueError("leaf_size must be > 0")
    if snapshot.triangle_aabb_min is None or snapshot.triangle_aabb_max is None:
        raise ValueError("build_device_bvh_from_snapshot requires AABB snapshot buffers")
    wp.init()
    resolved_device = snapshot.scene.device if device is None else wp.get_device(device)
    if resolved_device != snapshot.scene.device:
        raise ValueError("DeviceOpticalBvh device must match DeviceOpticalSceneSnapshot device")

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

    host_bvh = _build_host_bvh(
        aabb_min,
        aabb_max,
        source_keys,
        leaf_size=int(leaf_size),
    )

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
        stats=host_bvh["stats"],
        resources=resources,
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


def _build_host_bvh(
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    source_keys: np.ndarray,
    *,
    leaf_size: int,
) -> dict[str, object]:
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
        order = np.lexsort((source_keys[prim_ids], centroids[prim_ids, axis]))
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


def _require_warp() -> None:
    if not _HAS_WARP:
        raise ImportError("DeviceOpticalBvh requires the optional warp package")
