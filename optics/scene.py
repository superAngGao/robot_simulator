"""Frame-aligned optical scene snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from physics.publish import CpuPublishedFrame
from physics.spatial import SpatialTransform

from .registry import (
    OpticalInstanceSpec,
    OpticalMaterialSpec,
    OpticalPlaneGeometry,
    OpticalTriangleMeshGeometry,
    OpticalWorldRegistry,
)


@dataclass(frozen=True)
class OpticalFrameInputs:
    """Frame-aligned producer inputs for optical scene construction.

    Phase A carries only a rigid-body `CpuPublishedFrame`. Future producers can
    add cloth, soft-body, fluid, particle, volume, or medium streams here. Any
    geometry realization performed by scene/cache must be sensor-independent:
    converting particles/level sets to a surface mesh is scene preparation;
    ray-direction-dependent volume integration remains executor work.
    """

    frame_id: int
    sim_time: float
    env_idx: int
    rigid: CpuPublishedFrame | None = None

    def __post_init__(self) -> None:
        if self.env_idx < 0:
            raise ValueError("env_idx must be >= 0")
        # Phase A has only the rigid producer; relax this when non-rigid
        # producers such as cloth or fluid are added.
        if self.rigid is None:
            raise ValueError("OpticalFrameInputs requires at least one producer stream")
        if self.rigid.frame_id != self.frame_id:
            raise ValueError("rigid.frame_id must match OpticalFrameInputs.frame_id")
        if self.rigid.sim_time != self.sim_time:
            raise ValueError("rigid.sim_time must match OpticalFrameInputs.sim_time")

    @classmethod
    def from_published_frame(
        cls,
        frame: CpuPublishedFrame,
        *,
        env_idx: int = 0,
    ) -> "OpticalFrameInputs":
        return cls(
            frame_id=frame.frame_id,
            sim_time=frame.sim_time,
            env_idx=env_idx,
            rigid=frame,
        )


@dataclass(frozen=True)
class OpticalInstanceSnapshot:
    instance_id: str
    numeric_instance_id: int
    geometry_id: str
    material: OpticalMaterialSpec
    geometry: OpticalPlaneGeometry | OpticalTriangleMeshGeometry
    X_world_geometry: SpatialTransform
    roles: frozenset[str]
    source_key: object | None = None
    body_index: int | None = None


@dataclass(frozen=True)
class CpuBvhNode:
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    left: int = -1
    right: int = -1
    start: int = 0
    count: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.count > 0


@dataclass(frozen=True)
class OpticalSceneAcceleration:
    kind: Literal["cpu_bvh"]
    triangles_world: np.ndarray
    triangle_normals_world: np.ndarray
    primitive_instance_indices: np.ndarray
    primitive_source_order_keys: np.ndarray
    primitive_aabb_min: np.ndarray
    primitive_aabb_max: np.ndarray
    primitive_indices: np.ndarray
    nodes: list[CpuBvhNode]


@dataclass(frozen=True)
class OpticalSceneSnapshot:
    """Immutable CPU snapshot for one frame/env in Phase A."""

    frame_id: int
    sim_time: float
    env_idx: int
    instances: tuple[OpticalInstanceSnapshot, ...]
    lights: tuple[object, ...] = ()
    acceleration: OpticalSceneAcceleration | None = None
    location: str = "host"
    ready_event: object | None = None


class OpticalSceneCache:
    """CPU scene cache that composes a registry with frame-aligned inputs."""

    def __init__(self, registry: OpticalWorldRegistry) -> None:
        self.registry = registry

    def snapshot_from_published_frame(
        self,
        frame: CpuPublishedFrame,
        *,
        env_idx: int = 0,
        acceleration: Literal["none", "cpu_bvh"] = "none",
    ) -> OpticalSceneSnapshot:
        if not isinstance(frame, CpuPublishedFrame):
            raise NotImplementedError("Phase A OpticalSceneCache supports CpuPublishedFrame only")
        return self.snapshot_from_frame_inputs(
            OpticalFrameInputs.from_published_frame(frame, env_idx=env_idx),
            acceleration=acceleration,
        )

    def snapshot_from_frame_inputs(
        self,
        inputs: OpticalFrameInputs,
        *,
        acceleration: Literal["none", "cpu_bvh"] = "none",
    ) -> OpticalSceneSnapshot:
        if inputs.rigid is None:
            raise NotImplementedError("Phase A OpticalSceneCache requires a rigid frame input")
        if inputs.env_idx != 0:
            raise NotImplementedError("Phase A OpticalSceneCache supports one CPU env only")
        if acceleration not in ("none", "cpu_bvh"):
            raise ValueError("acceleration must be 'none' or 'cpu_bvh'")

        geometry = self.registry.geometry
        materials = self.registry.materials
        instances = tuple(
            self._build_instance_snapshot(instance, inputs, geometry, materials)
            for instance in self.registry.instances
        )
        acceleration_payload = None
        if acceleration == "cpu_bvh":
            acceleration_payload = build_cpu_bvh_acceleration(instances)
        return OpticalSceneSnapshot(
            frame_id=inputs.frame_id,
            sim_time=inputs.sim_time,
            env_idx=inputs.env_idx,
            instances=instances,
            lights=tuple(self.registry.lights.values()),
            acceleration=acceleration_payload,
        )

    def _build_instance_snapshot(
        self,
        instance: OpticalInstanceSpec,
        inputs: OpticalFrameInputs,
        geometry: dict[str, OpticalPlaneGeometry | OpticalTriangleMeshGeometry],
        materials: dict[str, OpticalMaterialSpec],
    ) -> OpticalInstanceSnapshot:
        X_body = _world_transform_for_instance(inputs, instance)
        X_world_geometry = X_body @ instance.X_body_geometry
        if instance.numeric_instance_id is None:
            raise ValueError("OpticalInstanceSpec must have a registry-assigned numeric_instance_id")
        return OpticalInstanceSnapshot(
            instance_id=instance.instance_id,
            numeric_instance_id=instance.numeric_instance_id,
            geometry_id=instance.geometry_id,
            material=materials[instance.material_id],
            geometry=geometry[instance.geometry_id],
            X_world_geometry=X_world_geometry,
            roles=instance.roles,
            source_key=instance.source_key,
            body_index=instance.body_index,
        )


def _world_transform_for_instance(
    inputs: OpticalFrameInputs,
    instance: OpticalInstanceSpec,
) -> SpatialTransform:
    if instance.body_index is None:
        return SpatialTransform.identity()
    if inputs.rigid is None:
        raise ValueError("OpticalFrameInputs.rigid is required for body-bound optical geometry")
    X_world = inputs.rigid.X_world
    if X_world is None:
        raise ValueError("CpuPublishedFrame.X_world is required for body-bound optical geometry")
    if instance.body_index >= len(X_world):
        raise IndexError(f"body_index {instance.body_index} is out of range for frame.X_world")
    return X_world[instance.body_index]


def transform_points(X_world_geometry: SpatialTransform, points_local: object) -> np.ndarray:
    points = np.asarray(points_local, dtype=np.float64)
    return points @ X_world_geometry.R.T + X_world_geometry.r


def transform_directions(X_world_geometry: SpatialTransform, directions_local: object) -> np.ndarray:
    directions = np.asarray(directions_local, dtype=np.float64)
    return directions @ X_world_geometry.R.T


def build_cpu_bvh_acceleration(
    instances: tuple[OpticalInstanceSnapshot, ...],
    *,
    leaf_size: int = 4,
) -> OpticalSceneAcceleration:
    if leaf_size <= 0:
        raise ValueError("leaf_size must be > 0")

    packed = _pack_triangle_primitives(instances)
    triangles_world = packed["triangles_world"]
    primitive_count = int(triangles_world.shape[0])
    primitive_indices = np.empty(0, dtype=np.int64)
    nodes: list[CpuBvhNode] = []
    if primitive_count > 0:
        primitive_indices = _build_bvh_nodes(
            nodes,
            np.arange(primitive_count, dtype=np.int64),
            packed["primitive_aabb_min"],
            packed["primitive_aabb_max"],
            packed["centroids"],
            packed["primitive_source_order_keys"],
            leaf_size=leaf_size,
        )
    return OpticalSceneAcceleration(
        kind="cpu_bvh",
        triangles_world=triangles_world,
        triangle_normals_world=packed["triangle_normals_world"],
        primitive_instance_indices=packed["primitive_instance_indices"],
        primitive_source_order_keys=packed["primitive_source_order_keys"],
        primitive_aabb_min=packed["primitive_aabb_min"],
        primitive_aabb_max=packed["primitive_aabb_max"],
        primitive_indices=primitive_indices,
        nodes=nodes,
    )


def _pack_triangle_primitives(instances: tuple[OpticalInstanceSnapshot, ...]) -> dict[str, np.ndarray]:
    triangles_world: list[np.ndarray] = []
    normals_world: list[np.ndarray] = []
    instance_indices: list[int] = []
    source_order_keys: list[tuple[int, int]] = []
    aabb_min: list[np.ndarray] = []
    aabb_max: list[np.ndarray] = []
    centroids: list[np.ndarray] = []

    for instance_index, instance in enumerate(instances):
        geometry = instance.geometry
        if not isinstance(geometry, OpticalTriangleMeshGeometry):
            continue
        vertices_world = transform_points(instance.X_world_geometry, geometry.vertices_local)
        for triangle_index, triangle in enumerate(np.asarray(geometry.triangles, dtype=np.int64)):
            tri_world = vertices_world[triangle]
            edge1 = tri_world[1] - tri_world[0]
            edge2 = tri_world[2] - tri_world[0]
            normal = np.cross(edge1, edge2)
            normal_norm = np.linalg.norm(normal)
            if normal_norm <= 1e-12:
                continue
            triangles_world.append(tri_world.copy())
            normals_world.append((normal / normal_norm).copy())
            instance_indices.append(instance_index)
            source_order_keys.append((instance_index, triangle_index))
            aabb_min.append(np.min(tri_world, axis=0))
            aabb_max.append(np.max(tri_world, axis=0))
            centroids.append(np.mean(tri_world, axis=0))

    return {
        "triangles_world": _array_or_empty(triangles_world, shape=(0, 3, 3), dtype=np.float64),
        "triangle_normals_world": _array_or_empty(normals_world, shape=(0, 3), dtype=np.float64),
        "primitive_instance_indices": np.asarray(instance_indices, dtype=np.int64),
        "primitive_source_order_keys": _array_or_empty(source_order_keys, shape=(0, 2), dtype=np.int64),
        "primitive_aabb_min": _array_or_empty(aabb_min, shape=(0, 3), dtype=np.float64),
        "primitive_aabb_max": _array_or_empty(aabb_max, shape=(0, 3), dtype=np.float64),
        "centroids": _array_or_empty(centroids, shape=(0, 3), dtype=np.float64),
    }


def _array_or_empty(values: list[object], *, shape: tuple[int, ...], dtype) -> np.ndarray:
    if not values:
        return np.empty(shape, dtype=dtype)
    return np.asarray(values, dtype=dtype)


def _build_bvh_nodes(
    nodes: list[CpuBvhNode],
    ids: np.ndarray,
    primitive_aabb_min: np.ndarray,
    primitive_aabb_max: np.ndarray,
    centroids: np.ndarray,
    source_order_keys: np.ndarray,
    *,
    leaf_size: int,
) -> np.ndarray:
    ordered_ids: list[int] = []
    _append_bvh_node(
        nodes,
        np.asarray(ids, dtype=np.int64),
        ordered_ids,
        primitive_aabb_min,
        primitive_aabb_max,
        centroids,
        source_order_keys,
        leaf_size=leaf_size,
    )
    return np.asarray(ordered_ids, dtype=np.int64)


def _append_bvh_node(
    nodes: list[CpuBvhNode],
    ids: np.ndarray,
    ordered_ids: list[int],
    primitive_aabb_min: np.ndarray,
    primitive_aabb_max: np.ndarray,
    centroids: np.ndarray,
    source_order_keys: np.ndarray,
    *,
    leaf_size: int,
) -> int:
    node_index = len(nodes)
    bounds_min = np.min(primitive_aabb_min[ids], axis=0)
    bounds_max = np.max(primitive_aabb_max[ids], axis=0)
    nodes.append(CpuBvhNode(bounds_min=bounds_min.copy(), bounds_max=bounds_max.copy()))

    centroid_bounds_min = np.min(centroids[ids], axis=0)
    centroid_bounds_max = np.max(centroids[ids], axis=0)
    centroid_extent = centroid_bounds_max - centroid_bounds_min
    if ids.shape[0] <= leaf_size or np.max(centroid_extent) <= 1e-12:
        start = len(ordered_ids)
        ordered_ids.extend(int(primitive_id) for primitive_id in ids)
        nodes[node_index] = CpuBvhNode(
            bounds_min=bounds_min.copy(),
            bounds_max=bounds_max.copy(),
            start=start,
            count=int(ids.shape[0]),
        )
        return node_index

    axis = int(np.argmax(centroid_extent))
    order = np.lexsort((source_order_keys[ids, 1], source_order_keys[ids, 0], centroids[ids, axis]))
    sorted_ids = ids[order]
    split = sorted_ids.shape[0] // 2
    left = _append_bvh_node(
        nodes,
        sorted_ids[:split],
        ordered_ids,
        primitive_aabb_min,
        primitive_aabb_max,
        centroids,
        source_order_keys,
        leaf_size=leaf_size,
    )
    right = _append_bvh_node(
        nodes,
        sorted_ids[split:],
        ordered_ids,
        primitive_aabb_min,
        primitive_aabb_max,
        centroids,
        source_order_keys,
        leaf_size=leaf_size,
    )
    nodes[node_index] = CpuBvhNode(
        bounds_min=bounds_min.copy(),
        bounds_max=bounds_max.copy(),
        left=left,
        right=right,
    )
    return node_index
