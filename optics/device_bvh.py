"""Device BVH buffers for GPU optical traversal."""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass, replace

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
    ready_event: object | None = None
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
    host_bvh = _build_host_bvh(
        aabb_min,
        aabb_max,
        source_keys,
        leaf_size=int(leaf_size),
        split_strategy=split_strategy,
    )
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


def _build_host_bvh(
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    source_keys: np.ndarray,
    *,
    leaf_size: int,
    split_strategy: str = "sort",
) -> dict[str, object]:
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
