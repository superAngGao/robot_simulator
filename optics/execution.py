"""Optical executor interfaces and the Phase-A CPU reference executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from sensing.optical import OpticalRaySensorSpec

from .registry import OpticalPlaneGeometry, OpticalTriangleMeshGeometry
from .scene import OpticalSceneSnapshot, transform_directions, transform_points


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
