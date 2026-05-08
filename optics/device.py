"""Device-oriented helpers for optical execution results and workloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .execution import OpticalComputeResult
from .registry import OpticalPlaneGeometry, OpticalTriangleMeshGeometry
from .scene import OpticalSceneSnapshot, transform_directions, transform_points

MAX_PRIMITIVES_PER_INSTANCE = 2**32
DEVICE_FLOAT32_RECOMMENDED_SCENE_SCALE_M = 1000.0
_BUILD_EPS_F32 = 1e-8


@dataclass(frozen=True)
class HostOpticalPrimitiveWorkload:
    """Host-packed primitive workload ready to upload to a device executor."""

    triangles_world: np.ndarray
    triangle_normals_world: np.ndarray
    triangle_numeric_instance_id: np.ndarray
    triangle_source_order_key: np.ndarray
    plane_normal_world: np.ndarray
    plane_point_world: np.ndarray
    plane_numeric_instance_id: np.ndarray
    plane_source_order_key: np.ndarray


def build_host_optical_primitive_workload(
    snapshot: OpticalSceneSnapshot,
    *,
    sensor_role: str,
) -> HostOpticalPrimitiveWorkload:
    """Pack snapshot primitives visible to `sensor_role` into flat host arrays.

    L5A does role filtering on the host so the first Warp kernel can stay
    focused on first-hit math and result schema. Source-order keys are packed
    int64 values preserving CPU lexicographic order:
    `(instance_index, primitive_index_within_instance)`.
    """
    triangles_world: list[np.ndarray] = []
    triangle_normals_world: list[np.ndarray] = []
    triangle_numeric_instance_id: list[int] = []
    triangle_source_order_key: list[int] = []
    plane_normal_world: list[np.ndarray] = []
    plane_point_world: list[np.ndarray] = []
    plane_numeric_instance_id: list[int] = []
    plane_source_order_key: list[int] = []

    for instance_index, instance in enumerate(snapshot.instances):
        if sensor_role not in instance.roles:
            continue
        geometry = instance.geometry
        if isinstance(geometry, OpticalPlaneGeometry):
            point_world = transform_points(
                instance.X_world_geometry,
                np.asarray(geometry.point_local)[None, :],
            )[0]
            normal_world = transform_directions(
                instance.X_world_geometry,
                np.asarray(geometry.normal_local)[None, :],
            )[0]
            normal_world = normal_world / np.linalg.norm(normal_world)
            plane_point_world.append(point_world.astype(np.float32))
            plane_normal_world.append(normal_world.astype(np.float32))
            plane_numeric_instance_id.append(int(instance.numeric_instance_id))
            plane_source_order_key.append(pack_source_order_key(instance_index, 0))
            continue

        if isinstance(geometry, OpticalTriangleMeshGeometry):
            vertices_world = transform_points(instance.X_world_geometry, geometry.vertices_local)
            for primitive_index, triangle in enumerate(np.asarray(geometry.triangles, dtype=np.int64)):
                tri_world = vertices_world[triangle]
                edge1 = tri_world[1] - tri_world[0]
                edge2 = tri_world[2] - tri_world[0]
                normal = np.cross(edge1, edge2)
                normal_norm = np.linalg.norm(normal)
                if normal_norm <= _BUILD_EPS_F32:
                    continue
                triangles_world.append(tri_world.astype(np.float32))
                triangle_normals_world.append((normal / normal_norm).astype(np.float32))
                triangle_numeric_instance_id.append(int(instance.numeric_instance_id))
                triangle_source_order_key.append(pack_source_order_key(instance_index, primitive_index))

    return HostOpticalPrimitiveWorkload(
        triangles_world=_array_or_empty(triangles_world, shape=(0, 3, 3), dtype=np.float32),
        triangle_normals_world=_array_or_empty(triangle_normals_world, shape=(0, 3), dtype=np.float32),
        triangle_numeric_instance_id=np.asarray(triangle_numeric_instance_id, dtype=np.int32),
        triangle_source_order_key=np.asarray(triangle_source_order_key, dtype=np.int64),
        plane_normal_world=_array_or_empty(plane_normal_world, shape=(0, 3), dtype=np.float32),
        plane_point_world=_array_or_empty(plane_point_world, shape=(0, 3), dtype=np.float32),
        plane_numeric_instance_id=np.asarray(plane_numeric_instance_id, dtype=np.int32),
        plane_source_order_key=np.asarray(plane_source_order_key, dtype=np.int64),
    )


def pack_source_order_key(instance_index: int, primitive_index_within_instance: int) -> np.int64:
    """Pack CPU lexicographic source order into one signed int64 key."""
    instance_index = int(instance_index)
    primitive_index_within_instance = int(primitive_index_within_instance)
    if instance_index < 0:
        raise ValueError("instance_index must be >= 0")
    if primitive_index_within_instance < 0:
        raise ValueError("primitive_index_within_instance must be >= 0")
    if primitive_index_within_instance >= MAX_PRIMITIVES_PER_INSTANCE:
        raise ValueError("primitive_index_within_instance exceeds MAX_PRIMITIVES_PER_INSTANCE")
    key = instance_index * MAX_PRIMITIVES_PER_INSTANCE + primitive_index_within_instance
    if key > np.iinfo(np.int64).max:
        raise ValueError("packed source-order key exceeds int64 range")
    return np.int64(key)


def stage_optical_compute_result_to_host(result: OpticalComputeResult) -> OpticalComputeResult:
    """Stage a device `OpticalComputeResult` into canonical host NumPy channels."""
    if result.location != "device":
        raise ValueError("stage_optical_compute_result_to_host requires a device result")
    _synchronize_ready_event(result.ready_event)

    channels: dict[str, object] = {}
    for name, value in result.channels.items():
        channels[name] = _stage_channel_to_host(name, value)

    return OpticalComputeResult(
        frame_id=result.frame_id,
        sim_time=result.sim_time,
        env_idx=result.env_idx,
        sensor_id=result.sensor_id,
        location="host",
        channels=channels,
        output_profile=result.output_profile,
        ready_event=None,
        resources=(),
    )


def stage_optical_channels(result: OpticalComputeResult, channels: Sequence[str]) -> dict[str, np.ndarray]:
    """Stage selected device result channels into host NumPy arrays.

    This helper is intentionally narrow: callers choose explicit channel names,
    while this function centralizes the ready-event synchronization and canonical
    dtype conversion used by full-result staging.
    """
    if result.location != "device":
        raise ValueError("stage_optical_channels requires a device result")
    _synchronize_ready_event(result.ready_event)

    staged: dict[str, np.ndarray] = {}
    for name in channels:
        staged[name] = _stage_channel_to_host(name, result.channel(name))
    return staged


def _stage_channel_to_host(name: str, value: object) -> np.ndarray:
    array = _channel_to_numpy(value)
    if name == "hit_mask":
        return np.asarray(array, dtype=bool).copy()
    if name in ("range_m", "position_world", "normal_world", "rgb", "intensity"):
        return np.asarray(array, dtype=np.float64).copy()
    if name == "numeric_instance_id":
        return np.asarray(array, dtype=np.int64).copy()
    return np.asarray(array).copy()


def _channel_to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _synchronize_ready_event(ready_event: object | None) -> None:
    if ready_event is None:
        return
    try:
        import warp as wp
    except Exception:
        return
    wp.synchronize_event(ready_event)


def _array_or_empty(values: list[object], *, shape: tuple[int, ...], dtype) -> np.ndarray:
    if not values:
        return np.empty(shape, dtype=dtype)
    return np.asarray(values, dtype=dtype)
