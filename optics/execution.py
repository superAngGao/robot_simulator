"""Optical executor interfaces and the Phase-A CPU reference executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from sensing.optical import OpticalRaySensorSpec

from .registry import OpticalPlaneGeometry, OpticalTriangleMeshGeometry
from .scene import OpticalSceneSnapshot, transform_directions, transform_points

_BUILD_EPS = 1e-12
_DIR_EPS = 1e-12
_T_EPS = 1e-9


class MissingAccelerationError(RuntimeError):
    """Raised when an accelerated executor receives a snapshot without acceleration."""


@dataclass(frozen=True)
class OpticalComputeResult:
    frame_id: int
    sim_time: float
    env_idx: int
    sensor_id: str
    location: Literal["host", "device", "external"] = "host"
    channels: dict[str, object] = field(default_factory=dict)
    ready_event: object | None = None

    def channel(self, name: str) -> object:
        return self.channels[name]


class OpticalExecutor(Protocol):
    def execute(self, snapshot: OpticalSceneSnapshot, spec: OpticalRaySensorSpec) -> OpticalComputeResult:
        """Execute an optical sensor spec against a frame-aligned snapshot."""


class CpuReferenceOpticalExecutor:
    """Deterministic first-hit range + material-id executor.

    This deliberately does not compute direct lighting, reflection, refraction,
    exposure, or camera response.
    """

    capabilities = frozenset(
        {
            "range_m",
            "hit_mask",
            "position_world",
            "normal_world",
            "material_id",
            "instance_id",
            "numeric_instance_id",
        }
    )

    def execute(self, snapshot: OpticalSceneSnapshot, spec: OpticalRaySensorSpec) -> OpticalComputeResult:
        self._validate(snapshot, spec)
        workload = self._prepare_workload(spec)
        hits = self._intersect(snapshot, workload)
        channels = self._resolve_channels(hits)
        return self._build_result(snapshot, spec, channels)

    def _validate(self, snapshot: OpticalSceneSnapshot, spec: OpticalRaySensorSpec) -> None:
        if snapshot.frame_id != spec.frame_id:
            raise ValueError("snapshot.frame_id must match spec.frame_id")
        if snapshot.env_idx != spec.env_idx:
            raise ValueError("snapshot.env_idx must match spec.env_idx")

    def _prepare_workload(self, spec: OpticalRaySensorSpec) -> "_RayWorkload":
        return _RayWorkload(
            origins_world=np.asarray(spec.origins_world, dtype=np.float64),
            directions_world=np.asarray(spec.directions_world, dtype=np.float64),
            max_distance=float(spec.max_distance),
            sensor_role=spec.sensor_role,
        )

    def _intersect(self, snapshot: OpticalSceneSnapshot, workload: "_RayWorkload") -> "_ReferenceHitBatch":
        hits = _empty_reference_hits(workload.num_rays)
        for instance in snapshot.instances:
            if workload.sensor_role not in instance.roles:
                continue
            geometry = instance.geometry
            if isinstance(geometry, OpticalPlaneGeometry):
                instance_hits = _intersect_plane(
                    workload.origins_world,
                    workload.directions_world,
                    max_distance=workload.max_distance,
                    plane=geometry,
                    X_world_geometry=instance.X_world_geometry,
                )
            elif isinstance(geometry, OpticalTriangleMeshGeometry):
                instance_hits = _intersect_triangle_mesh(
                    workload.origins_world,
                    workload.directions_world,
                    max_distance=workload.max_distance,
                    mesh=geometry,
                    X_world_geometry=instance.X_world_geometry,
                )
            else:
                continue

            closer = instance_hits.hit_mask & (instance_hits.distance < hits.range_m)
            if not np.any(closer):
                continue
            hits.range_m[closer] = instance_hits.distance[closer]
            hits.hit_mask[closer] = True
            hits.position_world[closer] = instance_hits.position_world[closer]
            hits.normal_world[closer] = instance_hits.normal_world[closer]
            hits.material_id[closer] = instance.material.material_id
            hits.instance_id[closer] = instance.instance_id
            hits.numeric_instance_id[closer] = instance.numeric_instance_id
        return hits

    def _resolve_channels(self, hits: "_ReferenceHitBatch") -> dict[str, object]:
        return {
            "range_m": hits.range_m,
            "hit_mask": hits.hit_mask,
            "position_world": hits.position_world,
            "normal_world": hits.normal_world,
            "material_id": hits.material_id,
            "instance_id": hits.instance_id,
            "numeric_instance_id": hits.numeric_instance_id,
        }

    def _build_result(
        self,
        snapshot: OpticalSceneSnapshot,
        spec: OpticalRaySensorSpec,
        channels: dict[str, object],
    ) -> OpticalComputeResult:
        return OpticalComputeResult(
            frame_id=snapshot.frame_id,
            sim_time=snapshot.sim_time,
            env_idx=snapshot.env_idx,
            sensor_id=spec.sensor_id,
            channels=channels,
        )


class CpuBvhOpticalExecutor(CpuReferenceOpticalExecutor):
    """CPU BVH first-hit executor over snapshot-owned acceleration data."""

    def _intersect(self, snapshot: OpticalSceneSnapshot, workload: "_RayWorkload") -> "_ReferenceHitBatch":
        acceleration = snapshot.acceleration
        if acceleration is None or acceleration.kind != "cpu_bvh":
            raise MissingAccelerationError("snapshot does not contain CPU BVH acceleration")

        hits = _empty_reference_hits(workload.num_rays)
        source_order_keys = np.full((workload.num_rays, 2), np.iinfo(np.int64).max, dtype=np.int64)
        self._intersect_bvh_triangles(snapshot, workload, hits, source_order_keys)
        self._intersect_planes(snapshot, workload, hits, source_order_keys)
        return hits

    def _intersect_bvh_triangles(
        self,
        snapshot: OpticalSceneSnapshot,
        workload: "_RayWorkload",
        hits: "_ReferenceHitBatch",
        source_order_keys: np.ndarray,
    ) -> None:
        acceleration = snapshot.acceleration
        if acceleration is None or not acceleration.nodes:
            return

        for ray_index in range(workload.num_rays):
            origin = workload.origins_world[ray_index]
            direction = workload.directions_world[ray_index]
            stack = [0]
            while stack:
                node_index = stack.pop()
                node = acceleration.nodes[node_index]
                node_hit, _ = _intersect_aabb_scalar(
                    origin,
                    direction,
                    node.bounds_min,
                    node.bounds_max,
                    max_distance=hits.range_m[ray_index] + _T_EPS,
                )
                if not node_hit:
                    continue

                if node.is_leaf:
                    leaf_ids = acceleration.primitive_indices[node.start : node.start + node.count]
                    for primitive_id in leaf_ids:
                        instance_index = int(acceleration.primitive_instance_indices[primitive_id])
                        instance = snapshot.instances[instance_index]
                        if workload.sensor_role not in instance.roles:
                            continue
                        tri = acceleration.triangles_world[primitive_id]
                        hit = _intersect_triangle_scalar(
                            origin,
                            direction,
                            hits.range_m[ray_index] + _T_EPS,
                            tri[0],
                            tri[1],
                            tri[2],
                        )
                        if hit is None:
                            continue
                        source_order_key = acceleration.primitive_source_order_keys[primitive_id]
                        _update_reference_hit(
                            hits,
                            source_order_keys,
                            ray_index,
                            distance=hit.distance,
                            position_world=hit.position_world,
                            normal_world=hit.normal_world,
                            material_id=instance.material.material_id,
                            instance_id=instance.instance_id,
                            numeric_instance_id=instance.numeric_instance_id,
                            source_order_key=source_order_key,
                        )
                    continue

                left = acceleration.nodes[node.left]
                right = acceleration.nodes[node.right]
                left_hit, left_t = _intersect_aabb_scalar(
                    origin,
                    direction,
                    left.bounds_min,
                    left.bounds_max,
                    max_distance=hits.range_m[ray_index] + _T_EPS,
                )
                right_hit, right_t = _intersect_aabb_scalar(
                    origin,
                    direction,
                    right.bounds_min,
                    right.bounds_max,
                    max_distance=hits.range_m[ray_index] + _T_EPS,
                )
                if left_hit and right_hit:
                    if left_t <= right_t:
                        stack.extend([node.right, node.left])
                    else:
                        stack.extend([node.left, node.right])
                elif left_hit:
                    stack.append(node.left)
                elif right_hit:
                    stack.append(node.right)

    def _intersect_planes(
        self,
        snapshot: OpticalSceneSnapshot,
        workload: "_RayWorkload",
        hits: "_ReferenceHitBatch",
        source_order_keys: np.ndarray,
    ) -> None:
        for instance_index, instance in enumerate(snapshot.instances):
            if workload.sensor_role not in instance.roles:
                continue
            geometry = instance.geometry
            if not isinstance(geometry, OpticalPlaneGeometry):
                continue
            plane_hits = _intersect_plane(
                workload.origins_world,
                workload.directions_world,
                max_distance=workload.max_distance,
                plane=geometry,
                X_world_geometry=instance.X_world_geometry,
            )
            source_order_key = np.array([instance_index, 0], dtype=np.int64)
            for ray_index in np.flatnonzero(plane_hits.hit_mask):
                _update_reference_hit(
                    hits,
                    source_order_keys,
                    int(ray_index),
                    distance=float(plane_hits.distance[ray_index]),
                    position_world=plane_hits.position_world[ray_index],
                    normal_world=plane_hits.normal_world[ray_index],
                    material_id=instance.material.material_id,
                    instance_id=instance.instance_id,
                    numeric_instance_id=instance.numeric_instance_id,
                    source_order_key=source_order_key,
                )


@dataclass(frozen=True)
class _RayWorkload:
    origins_world: np.ndarray
    directions_world: np.ndarray
    max_distance: float
    sensor_role: str

    @property
    def num_rays(self) -> int:
        return int(self.origins_world.shape[0])


@dataclass(frozen=True)
class _HitBatch:
    hit_mask: np.ndarray
    distance: np.ndarray
    position_world: np.ndarray
    normal_world: np.ndarray


@dataclass(frozen=True)
class _ReferenceHitBatch:
    hit_mask: np.ndarray
    range_m: np.ndarray
    position_world: np.ndarray
    normal_world: np.ndarray
    material_id: np.ndarray
    instance_id: np.ndarray
    numeric_instance_id: np.ndarray


@dataclass(frozen=True)
class _ScalarHit:
    distance: float
    position_world: np.ndarray
    normal_world: np.ndarray


def _empty_reference_hits(num_rays: int) -> _ReferenceHitBatch:
    return _ReferenceHitBatch(
        hit_mask=np.zeros(num_rays, dtype=bool),
        range_m=np.full(num_rays, np.inf, dtype=np.float64),
        position_world=np.full((num_rays, 3), np.nan, dtype=np.float64),
        normal_world=np.full((num_rays, 3), np.nan, dtype=np.float64),
        material_id=np.full(num_rays, None, dtype=object),
        instance_id=np.full(num_rays, None, dtype=object),
        numeric_instance_id=np.zeros(num_rays, dtype=np.int64),
    )


def _empty_hits(num_rays: int) -> _HitBatch:
    return _HitBatch(
        hit_mask=np.zeros(num_rays, dtype=bool),
        distance=np.full(num_rays, np.inf, dtype=np.float64),
        position_world=np.full((num_rays, 3), np.nan, dtype=np.float64),
        normal_world=np.full((num_rays, 3), np.nan, dtype=np.float64),
    )


def _intersect_plane(
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    max_distance: float,
    plane: OpticalPlaneGeometry,
    X_world_geometry,
) -> _HitBatch:
    hits = _empty_hits(origins.shape[0])
    point_world = transform_points(X_world_geometry, np.asarray(plane.point_local)[None, :])[0]
    normal_world = transform_directions(X_world_geometry, np.asarray(plane.normal_local)[None, :])[0]
    normal_world = normal_world / np.linalg.norm(normal_world)

    denom = directions @ normal_world
    numer = (point_world - origins) @ normal_world
    valid = np.abs(denom) > 1e-12
    t = np.full(origins.shape[0], np.inf, dtype=np.float64)
    t[valid] = numer[valid] / denom[valid]
    mask = valid & (t >= 0.0) & (t <= max_distance)

    if np.any(mask):
        hits.hit_mask[mask] = True
        hits.distance[mask] = t[mask]
        hits.position_world[mask] = origins[mask] + directions[mask] * t[mask, None]
        hits.normal_world[mask] = normal_world
    return hits


def _intersect_triangle_mesh(
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    max_distance: float,
    mesh: OpticalTriangleMeshGeometry,
    X_world_geometry,
) -> _HitBatch:
    hits = _empty_hits(origins.shape[0])
    vertices_world = transform_points(X_world_geometry, mesh.vertices_local)

    for tri in np.asarray(mesh.triangles, dtype=np.int64):
        v0, v1, v2 = vertices_world[tri]
        tri_hits = _intersect_triangle(origins, directions, max_distance, v0, v1, v2)
        closer = tri_hits.hit_mask & (tri_hits.distance < hits.distance)
        if not np.any(closer):
            continue
        hits.hit_mask[closer] = True
        hits.distance[closer] = tri_hits.distance[closer]
        hits.position_world[closer] = tri_hits.position_world[closer]
        hits.normal_world[closer] = tri_hits.normal_world[closer]
    return hits


def _intersect_triangle(
    origins: np.ndarray,
    directions: np.ndarray,
    max_distance: float,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> _HitBatch:
    hits = _empty_hits(origins.shape[0])
    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normal = np.cross(edge1, edge2)
    normal_norm = np.linalg.norm(face_normal)
    if normal_norm <= 1e-12:
        return hits
    face_normal = face_normal / normal_norm

    pvec = np.cross(directions, edge2)
    det = pvec @ edge1
    valid = np.abs(det) > 1e-12
    inv_det = np.zeros_like(det)
    inv_det[valid] = 1.0 / det[valid]

    tvec = origins - v0
    u = (pvec * tvec).sum(axis=1) * inv_det
    qvec = np.cross(tvec, edge1)
    v = (qvec * directions).sum(axis=1) * inv_det
    t = (qvec @ edge2) * inv_det
    mask = valid & (u >= 0.0) & (v >= 0.0) & ((u + v) <= 1.0) & (t >= 0.0) & (t <= max_distance)

    if np.any(mask):
        hits.hit_mask[mask] = True
        hits.distance[mask] = t[mask]
        hits.position_world[mask] = origins[mask] + directions[mask] * t[mask, None]
        normals = np.repeat(face_normal[None, :], origins.shape[0], axis=0)
        away = (normals * directions).sum(axis=1) > 0.0
        normals[away] *= -1.0
        hits.normal_world[mask] = normals[mask]
    return hits


def _intersect_aabb_scalar(
    origin: np.ndarray,
    direction: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    *,
    max_distance: float,
) -> tuple[bool, float]:
    t_enter = 0.0
    t_exit = float(max_distance)
    for axis in range(3):
        ray_origin = float(origin[axis])
        ray_direction = float(direction[axis])
        lower = float(bounds_min[axis])
        upper = float(bounds_max[axis])
        if abs(ray_direction) <= _DIR_EPS:
            if ray_origin < lower or ray_origin > upper:
                return False, np.inf
            continue
        inv_direction = 1.0 / ray_direction
        t0 = (lower - ray_origin) * inv_direction
        t1 = (upper - ray_origin) * inv_direction
        if t0 > t1:
            t0, t1 = t1, t0
        t_enter = max(t_enter, t0)
        t_exit = min(t_exit, t1)
        if t_exit < t_enter:
            return False, np.inf
    if t_exit < max(t_enter, 0.0):
        return False, np.inf
    if t_enter > max_distance:
        return False, np.inf
    return True, t_enter


def _intersect_triangle_scalar(
    origin: np.ndarray,
    direction: np.ndarray,
    max_distance: float,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> _ScalarHit | None:
    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normal = np.cross(edge1, edge2)
    normal_norm = np.linalg.norm(face_normal)
    if normal_norm <= _BUILD_EPS:
        return None
    face_normal = face_normal / normal_norm

    pvec = np.cross(direction, edge2)
    det = float(pvec @ edge1)
    if abs(det) <= _BUILD_EPS:
        return None
    inv_det = 1.0 / det
    tvec = origin - v0
    u = float(pvec @ tvec) * inv_det
    if u < 0.0:
        return None
    qvec = np.cross(tvec, edge1)
    v = float(qvec @ direction) * inv_det
    if v < 0.0 or (u + v) > 1.0:
        return None
    t = float(qvec @ edge2) * inv_det
    if t < 0.0 or t > max_distance:
        return None

    normal_world = face_normal.copy()
    if float(normal_world @ direction) > 0.0:
        normal_world *= -1.0
    return _ScalarHit(
        distance=t,
        position_world=origin + direction * t,
        normal_world=normal_world,
    )


def _update_reference_hit(
    hits: _ReferenceHitBatch,
    source_order_keys: np.ndarray,
    ray_index: int,
    *,
    distance: float,
    position_world: np.ndarray,
    normal_world: np.ndarray,
    material_id: str,
    instance_id: str,
    numeric_instance_id: int,
    source_order_key: np.ndarray,
) -> None:
    current_distance = float(hits.range_m[ray_index])
    current_key = source_order_keys[ray_index]
    should_update = distance < current_distance - _T_EPS
    if not should_update and abs(distance - current_distance) <= _T_EPS:
        should_update = tuple(source_order_key) < tuple(current_key)
    if not should_update:
        return

    hits.hit_mask[ray_index] = True
    hits.range_m[ray_index] = distance
    hits.position_world[ray_index] = position_world
    hits.normal_world[ray_index] = normal_world
    hits.material_id[ray_index] = material_id
    hits.instance_id[ray_index] = instance_id
    hits.numeric_instance_id[ray_index] = numeric_instance_id
    source_order_keys[ray_index] = source_order_key
