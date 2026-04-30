"""Frame-aligned optical scene snapshots."""

from __future__ import annotations

from dataclasses import dataclass

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
class OpticalSceneSnapshot:
    """Immutable CPU snapshot for one frame/env in Phase A."""

    frame_id: int
    sim_time: float
    env_idx: int
    instances: tuple[OpticalInstanceSnapshot, ...]
    lights: tuple[object, ...] = ()
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
    ) -> OpticalSceneSnapshot:
        if not isinstance(frame, CpuPublishedFrame):
            raise NotImplementedError("Phase A OpticalSceneCache supports CpuPublishedFrame only")
        return self.snapshot_from_frame_inputs(
            OpticalFrameInputs.from_published_frame(frame, env_idx=env_idx)
        )

    def snapshot_from_frame_inputs(self, inputs: OpticalFrameInputs) -> OpticalSceneSnapshot:
        if inputs.rigid is None:
            raise NotImplementedError("Phase A OpticalSceneCache requires a rigid frame input")
        if inputs.env_idx != 0:
            raise NotImplementedError("Phase A OpticalSceneCache supports one CPU env only")

        geometry = self.registry.geometry
        materials = self.registry.materials
        instances = tuple(
            self._build_instance_snapshot(instance, inputs, geometry, materials)
            for instance in self.registry.instances
        )
        return OpticalSceneSnapshot(
            frame_id=inputs.frame_id,
            sim_time=inputs.sim_time,
            env_idx=inputs.env_idx,
            instances=instances,
            lights=tuple(self.registry.lights.values()),
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
