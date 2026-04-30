"""Long-lived optical world registry.

The registry owns optical world data: geometry handles, material bindings, and
future light/medium state. It does not own sensor specs or published frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Literal

import numpy as np

from physics.spatial import SpatialTransform


@dataclass(frozen=True)
class OpticalMaterialSpec:
    material_id: str
    albedo_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    extension: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.material_id:
            raise ValueError("material_id must be non-empty")
        if len(self.albedo_rgb) != 3:
            raise ValueError("albedo_rgb must contain exactly three values")
        object.__setattr__(self, "albedo_rgb", tuple(float(c) for c in self.albedo_rgb))
        object.__setattr__(self, "extension", MappingProxyType(dict(self.extension)))


@dataclass(frozen=True)
class OpticalLightSpec:
    light_id: str
    kind: Literal["point", "directional"]
    position_or_direction_world: object
    intensity: float = 1.0
    color_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.light_id:
            raise ValueError("light_id must be non-empty")
        if self.kind not in ("point", "directional"):
            raise ValueError("kind must be 'point' or 'directional'")
        vector = np.asarray(self.position_or_direction_world, dtype=np.float64)
        if vector.shape != (3,):
            raise ValueError("position_or_direction_world must have shape (3,)")
        if len(self.color_rgb) != 3:
            raise ValueError("color_rgb must contain exactly three values")
        object.__setattr__(self, "position_or_direction_world", vector.copy())
        object.__setattr__(self, "intensity", float(self.intensity))
        object.__setattr__(self, "color_rgb", tuple(float(c) for c in self.color_rgb))


@dataclass(frozen=True)
class OpticalPlaneGeometry:
    normal_local: object
    point_local: object

    def __post_init__(self) -> None:
        normal = np.asarray(self.normal_local, dtype=np.float64)
        point = np.asarray(self.point_local, dtype=np.float64)
        if normal.shape != (3,):
            raise ValueError("normal_local must have shape (3,)")
        if point.shape != (3,):
            raise ValueError("point_local must have shape (3,)")
        norm = np.linalg.norm(normal)
        if norm <= 1e-12:
            raise ValueError("normal_local must be non-zero")
        object.__setattr__(self, "normal_local", (normal / norm).copy())
        object.__setattr__(self, "point_local", point.copy())


@dataclass(frozen=True)
class OpticalTriangleMeshGeometry:
    vertices_local: object
    triangles: object

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_local, dtype=np.float64)
        triangles = np.asarray(self.triangles, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices_local must have shape (num_vertices, 3)")
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("triangles must have shape (num_triangles, 3)")
        if vertices.shape[0] == 0:
            raise ValueError("vertices_local must not be empty")
        if triangles.shape[0] == 0:
            raise ValueError("triangles must not be empty")
        if np.any(triangles < 0) or np.any(triangles >= vertices.shape[0]):
            raise ValueError("triangles contain out-of-range vertex indices")
        object.__setattr__(self, "vertices_local", vertices.copy())
        object.__setattr__(self, "triangles", triangles.copy())


@dataclass(frozen=True)
class OpticalInstanceSpec:
    instance_id: str
    geometry_id: str
    material_id: str
    numeric_instance_id: int | None = None
    body_index: int | None = None
    X_body_geometry: SpatialTransform = field(default_factory=SpatialTransform.identity)
    roles: frozenset[str] = frozenset({"rgb", "depth", "lidar", "segmentation"})
    source_key: object | None = None

    def __post_init__(self) -> None:
        if not self.instance_id:
            raise ValueError("instance_id must be non-empty")
        if not self.geometry_id:
            raise ValueError("geometry_id must be non-empty")
        if not self.material_id:
            raise ValueError("material_id must be non-empty")
        if self.numeric_instance_id is not None and self.numeric_instance_id <= 0:
            raise ValueError("numeric_instance_id must be > 0 when provided")
        if self.body_index is not None and self.body_index < 0:
            raise ValueError("body_index must be >= 0 when provided")
        roles = frozenset(str(role) for role in self.roles)
        if not roles:
            raise ValueError("roles must not be empty")
        if any(not role for role in roles):
            raise ValueError("roles must contain non-empty strings")
        object.__setattr__(self, "roles", roles)


class OpticalWorldRegistry:
    """Registry for persistent optical world state."""

    def __init__(self) -> None:
        self._materials: dict[str, OpticalMaterialSpec] = {}
        self._lights: dict[str, OpticalLightSpec] = {}
        self._geometry: dict[str, OpticalPlaneGeometry | OpticalTriangleMeshGeometry] = {}
        self._instances: dict[str, OpticalInstanceSpec] = {}
        self._next_numeric_instance_id = 1

    @property
    def materials(self) -> dict[str, OpticalMaterialSpec]:
        return dict(self._materials)

    @property
    def lights(self) -> dict[str, OpticalLightSpec]:
        return dict(self._lights)

    @property
    def geometry(self) -> dict[str, OpticalPlaneGeometry | OpticalTriangleMeshGeometry]:
        return dict(self._geometry)

    @property
    def instances(self) -> list[OpticalInstanceSpec]:
        return list(self._instances.values())

    def add_material(self, material: OpticalMaterialSpec) -> None:
        if material.material_id in self._materials:
            raise ValueError(f"Duplicate material_id: {material.material_id}")
        self._materials[material.material_id] = material

    def add_light(self, light: OpticalLightSpec) -> None:
        if light.light_id in self._lights:
            raise ValueError(f"Duplicate light_id: {light.light_id}")
        self._lights[light.light_id] = light

    def add_plane_geometry(self, geometry_id: str, *, normal_local: object, point_local: object) -> None:
        self._add_geometry(
            geometry_id,
            OpticalPlaneGeometry(normal_local=normal_local, point_local=point_local),
        )

    def add_triangle_mesh_geometry(
        self,
        geometry_id: str,
        *,
        vertices_local: object,
        triangles: object,
    ) -> None:
        self._add_geometry(
            geometry_id,
            OpticalTriangleMeshGeometry(vertices_local=vertices_local, triangles=triangles),
        )

    def add_instance(self, instance: OpticalInstanceSpec) -> OpticalInstanceSpec:
        if instance.instance_id in self._instances:
            raise ValueError(f"Duplicate instance_id: {instance.instance_id}")
        if instance.geometry_id not in self._geometry:
            raise KeyError(f"Unknown geometry_id: {instance.geometry_id}")
        if instance.material_id not in self._materials:
            raise KeyError(f"Unknown material_id: {instance.material_id}")

        numeric_id = instance.numeric_instance_id
        if numeric_id is None:
            numeric_id = self._allocate_numeric_instance_id()
            instance = replace(instance, numeric_instance_id=numeric_id)
        elif any(existing.numeric_instance_id == numeric_id for existing in self._instances.values()):
            raise ValueError(f"Duplicate numeric_instance_id: {numeric_id}")
        else:
            self._next_numeric_instance_id = max(self._next_numeric_instance_id, numeric_id + 1)

        self._instances[instance.instance_id] = instance
        return instance

    def _allocate_numeric_instance_id(self) -> int:
        while any(
            existing.numeric_instance_id == self._next_numeric_instance_id
            for existing in self._instances.values()
        ):
            self._next_numeric_instance_id += 1
        numeric_id = self._next_numeric_instance_id
        self._next_numeric_instance_id += 1
        return numeric_id

    def _add_geometry(
        self,
        geometry_id: str,
        geometry: OpticalPlaneGeometry | OpticalTriangleMeshGeometry,
    ) -> None:
        if not geometry_id:
            raise ValueError("geometry_id must be non-empty")
        if geometry_id in self._geometry:
            raise ValueError(f"Duplicate geometry_id: {geometry_id}")
        self._geometry[geometry_id] = geometry
