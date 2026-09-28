"""CPU-only canonical median BVH and normalized topology contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

_CENTROID_EPS = 1.0e-12


WARP_BVH_TOPOLOGY_SCHEMA_VERSION = "q54.p12.4.warp-bvh-topology.v2"


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
    if leaf_size <= 0:
        raise ValueError("leaf_size must be > 0")
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
            split_strategy=split_strategy,
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

    stats = host_bvh.get("stats")
    if not isinstance(stats, DeviceBvhBuildStats):
        raise ValueError("host BVH topology requires canonical build stats")
    digest = hashlib.sha256(WARP_BVH_TOPOLOGY_SCHEMA_VERSION.encode("ascii"))
    digest.update(
        _canonical_sha256(
            {
                "builder": "cpu_median_longest_axis_source_order_v1",
                "leaf_size": stats.leaf_size,
                "split_strategy": stats.split_strategy,
            }
        ).encode("ascii")
    )
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


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
