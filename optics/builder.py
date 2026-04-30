"""Builders that convert model assets into optical registry records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from physics.geometry import (
    BodyCollisionGeometry,
    CollisionShape,
    HalfSpaceShape,
    MeshShape,
    ShapeInstance,
)
from physics.spatial import SpatialTransform
from robot.model import RobotModel

from .registry import OpticalInstanceSpec, OpticalMaterialSpec, OpticalWorldRegistry

OpticalSourcePolicy = Literal["collision_only"]


@dataclass(frozen=True)
class OpticalSourceKey:
    """Stable provenance key for an optical instance produced by a builder."""

    model_name: str
    body_name: str | None
    body_index: int | None
    geometry_role: str
    shape_index: int
    mesh_uri: str | None = None
    submesh_index: int | None = None


@dataclass(frozen=True)
class OpticalBindingDiagnostic:
    severity: Literal["warning", "error"]
    code: str
    message: str
    source_key: OpticalSourceKey | None = None


@dataclass(frozen=True)
class OpticalBindingBuildResult:
    registry: OpticalWorldRegistry
    source_to_instance_id: dict[OpticalSourceKey, str]
    instance_to_source: dict[str, OpticalSourceKey]
    diagnostics: list[OpticalBindingDiagnostic]


def build_optical_registry_from_robot_model(
    model: RobotModel,
    *,
    model_name: str = "main",
    source_policy: OpticalSourcePolicy = "collision_only",
    default_material_id: str = "default_collision",
    default_roles: frozenset[str] = frozenset({"depth", "lidar", "segmentation"}),
) -> OpticalBindingBuildResult:
    """Build an optical registry from a robot model.

    Phase A supports only `collision_only`: collision geometry is used as a
    conservative optical source for depth/LiDAR/segmentation-style sensors.
    Visual-preferred and explicit optical asset policies are intentionally left
    for later asset pipeline work.
    """

    if source_policy != "collision_only":
        raise NotImplementedError(f"Unsupported optical source_policy: {source_policy}")

    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec(default_material_id))
    source_to_instance_id: dict[OpticalSourceKey, str] = {}
    instance_to_source: dict[str, OpticalSourceKey] = {}
    diagnostics: list[OpticalBindingDiagnostic] = []

    for geom in model.geometries:
        _add_body_collision_geometry(
            registry,
            source_to_instance_id,
            instance_to_source,
            diagnostics,
            model,
            model_name=model_name,
            geom=geom,
            default_material_id=default_material_id,
            default_roles=default_roles,
        )

    return OpticalBindingBuildResult(
        registry=registry,
        source_to_instance_id=source_to_instance_id,
        instance_to_source=instance_to_source,
        diagnostics=diagnostics,
    )


def _add_body_collision_geometry(
    registry: OpticalWorldRegistry,
    source_to_instance_id: dict[OpticalSourceKey, str],
    instance_to_source: dict[str, OpticalSourceKey],
    diagnostics: list[OpticalBindingDiagnostic],
    model: RobotModel,
    *,
    model_name: str,
    geom: BodyCollisionGeometry,
    default_material_id: str,
    default_roles: frozenset[str],
) -> None:
    body_name = _body_name(model, geom.body_index)
    for shape_index, shape_instance in enumerate(geom.shapes):
        source_key = OpticalSourceKey(
            model_name=model_name,
            body_name=body_name,
            body_index=geom.body_index,
            geometry_role="collision",
            shape_index=shape_index,
            mesh_uri=_mesh_uri(shape_instance.shape),
        )
        ids = _make_ids(model_name, body_name, geom.body_index, shape_index)
        try:
            _add_shape_geometry(registry, ids["geometry_id"], shape_instance.shape)
        except NotImplementedError as exc:
            diagnostics.append(
                OpticalBindingDiagnostic(
                    severity="warning",
                    code="unsupported_collision_shape",
                    message=str(exc),
                    source_key=source_key,
                )
            )
            continue

        instance = registry.add_instance(
            OpticalInstanceSpec(
                instance_id=ids["instance_id"],
                geometry_id=ids["geometry_id"],
                material_id=default_material_id,
                body_index=geom.body_index,
                X_body_geometry=_shape_local_transform(shape_instance),
                roles=default_roles,
                source_key=source_key,
            )
        )
        source_to_instance_id[source_key] = instance.instance_id
        instance_to_source[instance.instance_id] = source_key


def _add_shape_geometry(registry: OpticalWorldRegistry, geometry_id: str, shape: CollisionShape) -> None:
    if isinstance(shape, HalfSpaceShape):
        registry.add_plane_geometry(
            geometry_id,
            normal_local=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            point_local=np.zeros(3, dtype=np.float64),
        )
        return

    if isinstance(shape, MeshShape):
        if shape.vertices is None or shape.faces is None:
            raise NotImplementedError("MeshShape optical binding requires loaded vertices and triangle faces")
        registry.add_triangle_mesh_geometry(
            geometry_id,
            vertices_local=shape.vertices,
            triangles=shape.faces,
        )
        return

    topology = shape.face_topology()
    if topology is None:
        raise NotImplementedError(f"{type(shape).__name__} has no optical triangle mesh binding yet")

    registry.add_triangle_mesh_geometry(
        geometry_id,
        vertices_local=topology.vertices,
        triangles=_triangulate_face_topology(topology.face_vertex_ids),
    )


def _triangulate_face_topology(face_vertex_ids: list[np.ndarray]) -> np.ndarray:
    triangles: list[list[int]] = []
    for face in face_vertex_ids:
        if len(face) < 3:
            continue
        root = int(face[0])
        for i in range(1, len(face) - 1):
            triangles.append([root, int(face[i]), int(face[i + 1])])
    if not triangles:
        raise NotImplementedError("Face topology did not produce any triangles")
    return np.asarray(triangles, dtype=np.int64)


def _shape_local_transform(shape_instance: ShapeInstance) -> SpatialTransform:
    return SpatialTransform.from_rpy(
        float(shape_instance.origin_rpy[0]),
        float(shape_instance.origin_rpy[1]),
        float(shape_instance.origin_rpy[2]),
        r=np.asarray(shape_instance.origin_xyz, dtype=np.float64),
    )


def _body_name(model: RobotModel, body_index: int) -> str | None:
    if body_index < 0 or body_index >= len(model.tree.bodies):
        return None
    return model.tree.bodies[body_index].name


def _mesh_uri(shape: CollisionShape) -> str | None:
    if isinstance(shape, MeshShape):
        return shape.filename
    return None


def _make_ids(model_name: str, body_name: str | None, body_index: int, shape_index: int) -> dict[str, str]:
    body_token = _safe_token(body_name or f"body_{body_index}")
    prefix = f"{_safe_token(model_name)}/{body_token}/collision/{shape_index}"
    return {
        "geometry_id": f"{prefix}/geometry",
        "instance_id": f"{prefix}/instance",
    }


def _safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in value)
