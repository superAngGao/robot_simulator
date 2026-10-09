"""CPU-safe canonical optical scene materialization.

This module is the single owner of registry-to-array packing.  It deliberately
imports neither Warp nor Torch; runtime builders independently upload the
immutable arrays exposed by :class:`PackedOpticalSceneHost`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .registry import (
    OpticalPlaneGeometry, OpticalTriangleMeshGeometry, OpticalWorldRegistry,
    _validated_triangle_indices,
)

SCENE_CONTRACT_SCHEMA_VERSION = "q54.optical-scene-contract.v3"
SCENE_MATERIALIZER_ID = "optical_registry_pack_v3"
_MAX_INT64_ROLE_BITS = 63
_MAX_PRIMITIVES_PER_INSTANCE = 2**32

SCENE_ARRAY_SEMANTICS = MappingProxyType(
    {
        "material_albedo_rgb": "shade.material.albedo_rgb",
        "light_kind": "shade.light.kind",
        "light_position_or_direction_world": "shade.light.position_or_direction_world",
        "light_intensity": "shade.light.intensity",
        "light_color_rgb": "shade.light.color_rgb",
        "geometry_vertices_local": "scene.geometry.vertices_local",
        "geometry_vertex_offsets": "scene.geometry.vertex_offsets",
        "triangle_vertex_indices": "scene.triangle.vertex_indices",
        "triangle_instance_index": "scene.triangle.instance_index",
        "triangle_primitive_global_id": "scene.triangle.primitive_global_id",
        "triangle_primitive_index_within_instance": (
            "scene.triangle.primitive_index_within_instance"
        ),
        "triangle_geometry_index": "scene.triangle.geometry_index",
        "triangle_geometry_primitive_index": "scene.triangle.geometry_primitive_index",
        "triangle_source_order_key": "scene.triangle.source_order_key",
        "triangle_role_mask": "scene.triangle.role_mask",
        "triangle_numeric_instance_id": "scene.triangle.numeric_instance_id",
        "triangle_material_index": "scene.triangle.material_index",
        "plane_normal_local": "scene.plane.normal_local",
        "plane_point_local": "scene.plane.point_local",
        "plane_instance_index": "scene.plane.instance_index",
        "plane_source_order_key": "scene.plane.source_order_key",
        "plane_role_mask": "scene.plane.role_mask",
        "plane_numeric_instance_id": "scene.plane.numeric_instance_id",
        "plane_material_index": "scene.plane.material_index",
        "instance_body_index": "scene.instance.body_index",
        "instance_X_body_geometry_R": "scene.instance.X_body_geometry.R",
        "instance_X_body_geometry_r": "scene.instance.X_body_geometry.r",
    }
)


@dataclass(frozen=True)
class PackedOpticalSceneHost:
    """Immutable logical scene arrays and their canonical provenance."""

    source_contract_id: str
    source_contract_sha256: str
    scene_id: str
    materializer_id: str
    arrays: Mapping[str, np.ndarray]
    array_semantics: Mapping[str, str]
    role_to_mask: Mapping[str, int]
    num_instances: int
    num_materials: int
    num_lights: int
    num_triangles: int
    num_planes: int
    max_body_index: int
    scene_contract_schema_version: str
    scene_contract_sha256: str

    @classmethod
    def from_registry(
        cls,
        registry: OpticalWorldRegistry,
        *,
        source_contract_id: str = "optical_world_registry_v1",
        source_contract_sha256: str | None = None,
        scene_id: str = "runtime_registry_scene",
        materializer_id: str = SCENE_MATERIALIZER_ID,
    ) -> "PackedOpticalSceneHost":
        if not isinstance(registry, OpticalWorldRegistry):
            raise TypeError("registry must be an OpticalWorldRegistry")
        for name, value in (
            ("source_contract_id", source_contract_id),
            ("scene_id", scene_id),
            ("materializer_id", materializer_id),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty")

        role_to_mask = _role_to_mask(registry)
        arrays, counts = _pack_registry_arrays(registry, role_to_mask)
        frozen_arrays = _freeze_arrays(arrays)
        array_payload = _array_manifest_payload(frozen_arrays)
        source_sha = source_contract_sha256
        if source_sha is None:
            source_sha = _sha256_payload(
                {
                    "source_contract_id": source_contract_id,
                    "scene_id": scene_id,
                    "array_manifests": array_payload,
                    "counts": counts,
                    "role_to_mask": dict(sorted(role_to_mask.items())),
                }
            )
        if not _sha256_text(source_sha):
            raise ValueError("source_contract_sha256 must be a lowercase SHA-256 digest")

        scene_payload = {
            "scene_contract_schema_version": SCENE_CONTRACT_SCHEMA_VERSION,
            "source_contract_id": source_contract_id,
            "source_contract_sha256": source_sha,
            "scene_id": scene_id,
            "materializer_id": materializer_id,
            "array_manifests": array_payload,
            "array_semantics": dict(sorted(SCENE_ARRAY_SEMANTICS.items())),
            "role_to_mask": dict(sorted(role_to_mask.items())),
            "counts": counts,
        }
        return cls(
            source_contract_id=source_contract_id,
            source_contract_sha256=source_sha,
            scene_id=scene_id,
            materializer_id=materializer_id,
            arrays=MappingProxyType(frozen_arrays),
            array_semantics=SCENE_ARRAY_SEMANTICS,
            role_to_mask=MappingProxyType(dict(sorted(role_to_mask.items()))),
            num_instances=counts["num_instances"],
            num_materials=counts["num_materials"],
            num_lights=counts["num_lights"],
            num_triangles=counts["num_triangles"],
            num_planes=counts["num_planes"],
            max_body_index=counts["max_body_index"],
            scene_contract_schema_version=SCENE_CONTRACT_SCHEMA_VERSION,
            scene_contract_sha256=_sha256_payload(scene_payload),
        )

    def array_manifest_payload(self) -> list[dict[str, object]]:
        return _array_manifest_payload(self.arrays)


def _role_to_mask(registry: OpticalWorldRegistry) -> dict[str, int]:
    roles = sorted({str(role) for instance in registry.instances for role in instance.roles})
    if len(roles) > _MAX_INT64_ROLE_BITS:
        raise ValueError("optical role bitmask supports at most 63 roles")
    return {role: 1 << index for index, role in enumerate(roles)}


def validate_indexed_geometry(scene: PackedOpticalSceneHost) -> None:
    """Validate source intervals before host gathering or native upload."""
    arrays = scene.arrays
    vertices = arrays["geometry_vertices_local"]
    offsets = arrays["geometry_vertex_offsets"]
    indices = arrays["triangle_vertex_indices"]
    geometry = arrays["triangle_geometry_index"]
    instances = arrays["triangle_instance_index"]
    if vertices.dtype != np.float64 or vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("geometry vertices must have float64 shape (V, 3)")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("geometry vertices must be finite")
    if len(vertices) > np.iinfo(np.int32).max:
        raise ValueError("geometry vertex count exceeds native array limits")
    if (offsets.dtype != np.int64 or offsets.ndim != 1 or len(offsets) == 0
            or offsets[0] != 0 or offsets[-1] != len(vertices)
            or np.any(offsets[1:] < offsets[:-1])):
        raise ValueError("invalid geometry vertex offsets")
    if indices.dtype != np.int64 or indices.shape != (scene.num_triangles, 3):
        raise ValueError("triangle indices must have int64 shape (N, 3)")
    if (geometry.dtype != np.int32 or geometry.shape != (scene.num_triangles,)
            or np.any(geometry < 0) or np.any(geometry >= len(offsets) - 1)):
        raise ValueError("invalid triangle geometry indices")
    if (instances.dtype != np.int32 or instances.shape != (scene.num_triangles,)
            or np.any(instances < 0) or np.any(instances >= scene.num_instances)):
        raise ValueError("invalid triangle instance indices")
    if (np.any(indices < offsets[geometry, None])
            or np.any(indices >= offsets[geometry + 1, None])):
        raise ValueError("triangle vertex index lies outside its geometry interval")
    for instance in np.unique(instances):
        if len(np.unique(geometry[instances == instance])) != 1:
            raise ValueError("one instance cannot refer to multiple geometries")


def _mask_for_roles(role_to_mask: Mapping[str, int], roles: object) -> int:
    mask = 0
    for role in roles:
        mask |= int(role_to_mask.get(str(role), 0))
    return mask


def _pack_registry_arrays(
    registry: OpticalWorldRegistry,
    role_to_mask: Mapping[str, int],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    geometry = registry.geometry
    instances = registry.instances
    materials = registry.materials
    lights = registry.lights
    geometry_index_by_id = {key: index for index, key in enumerate(geometry)}
    material_index_by_id = {key: index for index, key in enumerate(materials)}
    int32_max = np.iinfo(np.int32).max
    if max(len(instances), len(geometry), len(materials), len(lights)) > int32_max:
        raise ValueError("optical source count exceeds native int32 limits")

    material_albedo_rgb = [
        np.asarray(material.albedo_rgb, dtype=np.float32) for material in materials.values()
    ]
    light_kind: list[int] = []
    light_position_or_direction_world: list[np.ndarray] = []
    light_intensity: list[float] = []
    light_color_rgb: list[np.ndarray] = []
    for light in lights.values():
        if not light.enabled:
            continue
        light_kind.append(0 if light.kind == "directional" else 1)
        light_position_or_direction_world.append(
            np.asarray(light.position_or_direction_world, dtype=np.float32)
        )
        light_intensity.append(float(light.intensity))
        light_color_rgb.append(np.asarray(light.color_rgb, dtype=np.float32))

    instance_body_index: list[int] = []
    instance_x_R: list[np.ndarray] = []
    instance_x_r: list[np.ndarray] = []
    geometry_vertices: list[np.ndarray] = []
    geometry_offsets = [0]
    geometry_faces: dict[str, np.ndarray] = {}
    for geometry_id, source in geometry.items():
        if isinstance(source, OpticalTriangleMeshGeometry):
            vertices = np.asarray(source.vertices_local, dtype=np.float64)
            if vertices.ndim != 2 or vertices.shape[1] != 3:
                raise ValueError("source mesh vertices must have shape (V, 3)")
            faces = _validated_triangle_indices(source.triangles, len(vertices))
            geometry_faces[geometry_id] = faces + geometry_offsets[-1]
            geometry_vertices.append(vertices)
            geometry_offsets.append(geometry_offsets[-1] + len(vertices))
        elif isinstance(source, OpticalPlaneGeometry):
            geometry_offsets.append(geometry_offsets[-1])
        else:
            raise TypeError("unsupported optical source geometry")
    triangles: list[np.ndarray] = []
    triangle_instance_index: list[int] = []
    triangle_primitive_global_id: list[int] = []
    triangle_primitive_index_within_instance: list[int] = []
    triangle_geometry_index: list[int] = []
    triangle_geometry_primitive_index: list[int] = []
    triangle_source_order_key: list[int] = []
    triangle_role_mask: list[int] = []
    triangle_numeric_instance_id: list[int] = []
    triangle_material_index: list[int] = []
    plane_normals: list[np.ndarray] = []
    plane_points: list[np.ndarray] = []
    plane_instance_index: list[int] = []
    plane_source_order_key: list[int] = []
    plane_role_mask: list[int] = []
    plane_numeric_instance_id: list[int] = []
    plane_material_index: list[int] = []

    for instance_index, instance in enumerate(instances):
        if instance.numeric_instance_id is None:
            raise ValueError("OpticalInstanceSpec must have a registry-assigned numeric_instance_id")
        if instance.numeric_instance_id > int32_max or (
            instance.body_index is not None and instance.body_index > int32_max
        ):
            raise ValueError("optical instance identifier exceeds native int32 limits")
        instance_body_index.append(-1 if instance.body_index is None else int(instance.body_index))
        instance_x_R.append(np.asarray(instance.X_body_geometry.R, dtype=np.float64).reshape(9))
        instance_x_r.append(np.asarray(instance.X_body_geometry.r, dtype=np.float64))
        role_mask = _mask_for_roles(role_to_mask, instance.roles)
        material_index = material_index_by_id[instance.material_id]
        instance_geometry = geometry[instance.geometry_id]

        if isinstance(instance_geometry, OpticalPlaneGeometry):
            plane_normals.append(np.asarray(instance_geometry.normal_local, dtype=np.float64))
            plane_points.append(np.asarray(instance_geometry.point_local, dtype=np.float64))
            plane_instance_index.append(instance_index)
            plane_source_order_key.append(_pack_source_order_key(instance_index, 0))
            plane_role_mask.append(role_mask)
            plane_numeric_instance_id.append(int(instance.numeric_instance_id))
            plane_material_index.append(material_index)
            continue

        if isinstance(instance_geometry, OpticalTriangleMeshGeometry):
            for primitive_index, triangle in enumerate(geometry_faces[instance.geometry_id]):
                global_primitive_id = len(triangles)
                if global_primitive_id >= int32_max or primitive_index >= int32_max:
                    raise ValueError("optical triangle count exceeds native int32 limits")
                triangles.append(triangle)
                triangle_instance_index.append(instance_index)
                triangle_primitive_global_id.append(global_primitive_id)
                triangle_primitive_index_within_instance.append(primitive_index)
                triangle_geometry_index.append(geometry_index_by_id[instance.geometry_id])
                triangle_geometry_primitive_index.append(primitive_index)
                triangle_source_order_key.append(
                    _pack_source_order_key(instance_index, primitive_index)
                )
                triangle_role_mask.append(role_mask)
                triangle_numeric_instance_id.append(int(instance.numeric_instance_id))
                triangle_material_index.append(material_index)

    arrays = {
        "material_albedo_rgb": _array_or_empty(material_albedo_rgb, (0, 3), np.float32),
        "light_kind": np.asarray(light_kind, dtype=np.int32),
        "light_position_or_direction_world": _array_or_empty(
            light_position_or_direction_world, (0, 3), np.float32
        ),
        "light_intensity": np.asarray(light_intensity, dtype=np.float32),
        "light_color_rgb": _array_or_empty(light_color_rgb, (0, 3), np.float32),
        "geometry_vertices_local": (np.concatenate(geometry_vertices, axis=0)
                                    if geometry_vertices else np.empty((0, 3), dtype=np.float64)),
        "geometry_vertex_offsets": np.asarray(geometry_offsets, dtype=np.int64),
        "triangle_vertex_indices": _array_or_empty(triangles, (0, 3), np.int64),
        "triangle_instance_index": np.asarray(triangle_instance_index, dtype=np.int32),
        "triangle_primitive_global_id": np.asarray(
            triangle_primitive_global_id, dtype=np.int32
        ),
        "triangle_primitive_index_within_instance": np.asarray(
            triangle_primitive_index_within_instance, dtype=np.int32
        ),
        "triangle_geometry_index": np.asarray(triangle_geometry_index, dtype=np.int32),
        "triangle_geometry_primitive_index": np.asarray(
            triangle_geometry_primitive_index, dtype=np.int32
        ),
        "triangle_source_order_key": np.asarray(triangle_source_order_key, dtype=np.int64),
        "triangle_role_mask": np.asarray(triangle_role_mask, dtype=np.int64),
        "triangle_numeric_instance_id": np.asarray(
            triangle_numeric_instance_id, dtype=np.int32
        ),
        "triangle_material_index": np.asarray(triangle_material_index, dtype=np.int32),
        "plane_normal_local": _array_or_empty(plane_normals, (0, 3), np.float64),
        "plane_point_local": _array_or_empty(plane_points, (0, 3), np.float64),
        "plane_instance_index": np.asarray(plane_instance_index, dtype=np.int32),
        "plane_source_order_key": np.asarray(plane_source_order_key, dtype=np.int64),
        "plane_role_mask": np.asarray(plane_role_mask, dtype=np.int64),
        "plane_numeric_instance_id": np.asarray(
            plane_numeric_instance_id, dtype=np.int32
        ),
        "plane_material_index": np.asarray(plane_material_index, dtype=np.int32),
        "instance_body_index": np.asarray(instance_body_index, dtype=np.int32),
        "instance_X_body_geometry_R": _array_or_empty(instance_x_R, (0, 9), np.float64),
        "instance_X_body_geometry_r": _array_or_empty(instance_x_r, (0, 3), np.float64),
    }
    counts = {
        "num_instances": len(instances),
        "num_materials": len(materials),
        "num_lights": len(light_kind),
        "num_triangles": len(triangles),
        "num_planes": len(plane_normals),
        "max_body_index": max(instance_body_index, default=-1),
    }
    return arrays, counts


def _freeze_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(arrays) != set(SCENE_ARRAY_SEMANTICS):
        raise ValueError("packed scene arrays do not match the canonical semantic registry")
    frozen: dict[str, np.ndarray] = {}
    for name in sorted(arrays):
        value = np.array(arrays[name], copy=True, order="C")
        if value.dtype.hasobject:
            raise ValueError(f"packed scene array {name!r} cannot use object dtype")
        if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
            raise ValueError(f"packed scene array {name!r} must be finite")
        value.setflags(write=False)
        frozen[name] = value
    return frozen


def _array_manifest_payload(arrays: Mapping[str, np.ndarray]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for name in sorted(arrays):
        value = arrays[name]
        digest = hashlib.sha256()
        header = {
            "name": name,
            "semantic": SCENE_ARRAY_SEMANTICS[name],
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "nbytes": int(value.nbytes),
        }
        digest.update(_canonical_json_bytes(header))
        digest.update(value.tobytes(order="C"))
        payload.append({**header, "content_sha256": digest.hexdigest()})
    return payload


def _array_or_empty(values: list[object], shape: tuple[int, ...], dtype: object) -> np.ndarray:
    if not values:
        return np.empty(shape, dtype=dtype)
    return np.asarray(values, dtype=dtype)


def _pack_source_order_key(instance_index: int, primitive_index: int) -> np.int64:
    if instance_index < 0 or primitive_index < 0:
        raise ValueError("source-order indices must be non-negative")
    if primitive_index >= _MAX_PRIMITIVES_PER_INSTANCE:
        raise ValueError("primitive index exceeds the canonical source-order range")
    key = instance_index * _MAX_PRIMITIVES_PER_INSTANCE + primitive_index
    if key > np.iinfo(np.int64).max:
        raise ValueError("packed source-order key exceeds int64 range")
    return np.int64(key)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_payload(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
