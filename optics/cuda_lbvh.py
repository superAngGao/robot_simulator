"""Optional CUDA LBVH builder for optical device BVHs."""

from __future__ import annotations

import time
from dataclasses import replace
from functools import lru_cache

from .device_bvh import DeviceBvhBuildStats, DeviceOpticalBvh
from .device_scene import DeviceOpticalSceneSnapshot

try:  # pragma: no cover - exercised in CUDA extension environments.
    import torch
    import warp as wp
    from torch.utils.cpp_extension import load_inline

    _HAS_CUDA_LBVH_DEPS = True
except Exception:  # pragma: no cover - keeps CPU-only imports working.
    torch = None
    wp = None
    load_inline = None
    _HAS_CUDA_LBVH_DEPS = False


from examples.cuda_lbvh_extension_spike import CPP_SOURCE, CUDA_SOURCE


def build_cuda_lbvh_from_snapshot(
    snapshot: DeviceOpticalSceneSnapshot,
    *,
    device=None,
    stream=None,
) -> DeviceOpticalBvh:
    """Build a leaf-size-1 CUDA LBVH and expose it as a `DeviceOpticalBvh`.

    This first integrated CUDA path supports traversal parity only. It does not
    yet fill true `node_depth` / `level_ranges`, so GPU refit is explicitly
    disabled on the returned BVH.
    """
    total_start = time.perf_counter()
    _require_cuda_lbvh_deps()
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

    load_start = time.perf_counter()
    module = _load_cuda_lbvh_extension()
    detail_ms.append(("load_extension", (time.perf_counter() - load_start) * 1000.0))

    convert_start = time.perf_counter()
    aabb_min_t = wp.to_torch(snapshot.triangle_aabb_min).contiguous()
    aabb_max_t = wp.to_torch(snapshot.triangle_aabb_max).contiguous()
    detail_ms.append(("warp_to_torch_aabb", (time.perf_counter() - convert_start) * 1000.0))

    scene_bounds_start = _record_torch_event()
    scene_min_t = torch.min(aabb_min_t, dim=0).values.contiguous()
    scene_max_t = torch.max(aabb_max_t, dim=0).values.contiguous()
    scene_bounds_end = _record_torch_event()

    build_start = time.perf_counter()
    morton_start = _record_torch_event()
    sorted_keys_t, sorted_prim_ids_t = module.morton_sort_aabbs(
        aabb_min_t,
        aabb_max_t,
        scene_min_t,
        scene_max_t,
    )
    morton_end = _record_torch_event()
    topology_start = _record_torch_event()
    (
        left_t,
        right_t,
        _parent_t,
        start_t,
        count_t,
        _range_start_t,
        _range_end_t,
        _split_t,
        bounds_min_t,
        bounds_max_t,
    ) = module.build_lbvh_topology_and_bounds(
        sorted_keys_t,
        sorted_prim_ids_t,
        aabb_min_t,
        aabb_max_t,
    )
    topology_end = _record_torch_event()
    sync_start = time.perf_counter()
    torch.cuda.synchronize()
    sync_ms = (time.perf_counter() - sync_start) * 1000.0
    build_ms = (time.perf_counter() - build_start) * 1000.0
    detail_ms.extend(
        (
            ("scene_bounds_device", _elapsed_torch_ms(scene_bounds_start, scene_bounds_end)),
            ("morton_sort_device", _elapsed_torch_ms(morton_start, morton_end)),
            ("topology_bounds_device", _elapsed_torch_ms(topology_start, topology_end)),
            ("cuda_build_wait", build_ms),
            ("torch_synchronize_wait", sync_ms),
        )
    )

    metadata_start = time.perf_counter()
    num_primitives = int(sorted_prim_ids_t.shape[0])
    num_nodes = int(left_t.shape[0])
    node_depth_t = torch.zeros((num_nodes,), dtype=torch.int32, device=sorted_prim_ids_t.device)
    level_ranges_t = torch.empty((0, 2), dtype=torch.int32, device=sorted_prim_ids_t.device)
    detail_ms.append(("metadata_tensor_alloc", (time.perf_counter() - metadata_start) * 1000.0))

    wrap_start = time.perf_counter()
    bounds_min = wp.from_torch(bounds_min_t, dtype=wp.float32)
    bounds_max = wp.from_torch(bounds_max_t, dtype=wp.float32)
    left = wp.from_torch(left_t, dtype=wp.int32)
    right = wp.from_torch(right_t, dtype=wp.int32)
    start = wp.from_torch(start_t, dtype=wp.int32)
    count = wp.from_torch(count_t, dtype=wp.int32)
    node_depth = wp.from_torch(node_depth_t, dtype=wp.int32)
    level_ranges = wp.from_torch(level_ranges_t, dtype=wp.int32)
    prim_ids = wp.from_torch(sorted_prim_ids_t, dtype=wp.int32)
    detail_ms.append(("torch_to_warp_views", (time.perf_counter() - wrap_start) * 1000.0))

    event_start = time.perf_counter()
    ready_event = (stream or wp.get_stream(resolved_device)).record_event()
    detail_ms.append(("record_ready_event", (time.perf_counter() - event_start) * 1000.0))

    stats = DeviceBvhBuildStats(
        num_nodes=num_nodes,
        num_leaves=num_primitives,
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
        bounds_min_t,
        bounds_max_t,
        left_t,
        right_t,
        start_t,
        count_t,
        node_depth_t,
        level_ranges_t,
        sorted_prim_ids_t,
        sorted_keys_t,
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
        ready_event=ready_event,
        resources=resources,
    )


@lru_cache(maxsize=1)
def _load_cuda_lbvh_extension():
    _require_cuda_lbvh_deps()
    return load_inline(
        name="robot_sim_cuda_lbvh_spike_v2",
        cpp_sources=[CPP_SOURCE],
        cuda_sources=[CUDA_SOURCE],
        with_cuda=True,
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        verbose=False,
    )


def _require_cuda_lbvh_deps() -> None:
    if not _HAS_CUDA_LBVH_DEPS:
        raise ImportError("CUDA LBVH builder requires torch, warp, and torch CUDA extension tooling")
    if not torch.cuda.is_available():
        raise ImportError("CUDA LBVH builder requires torch CUDA availability")


def _record_torch_event():
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def _elapsed_torch_ms(start, end) -> float:
    return float(start.elapsed_time(end))
