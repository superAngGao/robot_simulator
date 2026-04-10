"""
Extract a RenderScene from physics state.

Bridges physics types (CollisionShape, SpatialTransform, Terrain) to the
backend-agnostic RenderScene data structure. Imports from ``physics/``
(allowed direction); ``physics/`` never imports from ``rendering/``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np

from physics.geometry import (
    BoxShape,
    CapsuleShape,
    CollisionShape,
    ConvexHullShape,
    CylinderShape,
    HalfSpaceShape,
    MeshShape,
    SphereShape,
)
from physics.terrain import FlatTerrain, HalfSpaceTerrain

from .render_scene import ContactPoint, PositionedShape, RenderScene, TerrainInfo

if TYPE_CHECKING:
    from physics.engine import ContactInfo
    from physics.merged_model import MergedModel
    from physics.robot_tree import RobotTreeNumpy
    from physics.spatial import SpatialTransform
    from physics.terrain import Terrain


# ---------------------------------------------------------------------------
# Shape type dispatch
# ---------------------------------------------------------------------------


def _shape_to_type_params(shape: CollisionShape) -> tuple[str, dict]:
    """Map a CollisionShape to (shape_type, params) for RenderScene."""
    if isinstance(shape, BoxShape):
        return "box", {"size": tuple(shape.size)}
    elif isinstance(shape, SphereShape):
        return "sphere", {"radius": shape.radius}
    elif isinstance(shape, CylinderShape):
        return "cylinder", {"radius": shape.radius, "length": shape.length}
    elif isinstance(shape, CapsuleShape):
        return "capsule", {"radius": shape.radius, "length": shape.length}
    elif isinstance(shape, ConvexHullShape):
        return "convex_hull", {"vertices": shape.vertices.copy()}
    elif isinstance(shape, MeshShape):
        verts = shape.vertices.copy() if shape.vertices is not None else None
        return "mesh", {"vertices": verts, "filename": shape.filename}
    elif isinstance(shape, HalfSpaceShape):
        return "halfspace", {}
    return "unknown", {}


def _terrain_to_info(terrain: "Terrain") -> TerrainInfo:
    """Map a Terrain instance to TerrainInfo."""
    if isinstance(terrain, HalfSpaceTerrain):
        return TerrainInfo(
            terrain_type="halfspace",
            params={
                "normal": terrain.normal_world.copy(),
                "point": terrain.point_on_plane.copy(),
            },
        )
    elif isinstance(terrain, FlatTerrain):
        return TerrainInfo(
            terrain_type="flat",
            params={"z": terrain.height_at(0.0, 0.0)},
        )
    return TerrainInfo(terrain_type="unknown", params={})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_render_scene(
    merged: "MergedModel",
    X_world: List["SpatialTransform"],
    contacts: Optional[List["ContactInfo"]] = None,
    terrain: Optional["Terrain"] = None,
) -> RenderScene:
    """Build a RenderScene from merged model state.

    Args:
        merged  : MergedModel with collision shapes and tree.
        X_world : Body world-frame transforms (from FK or StepOutput).
        contacts: Contact info list (from engine.query_contacts()).
        terrain : Terrain instance (from merged.terrain or Scene).

    Returns:
        RenderScene ready for any rendering backend.
    """
    tree = merged.tree

    # --- Shapes ---
    shapes: list[PositionedShape] = []
    for body_idx, geom in enumerate(merged.collision_shapes):
        if geom is None or not geom.shapes:
            continue
        body_name = tree.bodies[body_idx].name if body_idx < len(tree.bodies) else f"body_{body_idx}"
        for si in geom.shapes:
            shape_type, params = _shape_to_type_params(si.shape)
            if shape_type in ("halfspace", "unknown"):
                continue
            X_shape = si.world_pose(X_world[body_idx])
            shapes.append(
                PositionedShape(
                    shape_type=shape_type,
                    params=params,
                    position=X_shape.r.copy(),
                    rotation=X_shape.R.copy(),
                    body_index=body_idx,
                    body_name=body_name,
                )
            )

    # --- Contacts ---
    contact_points: list[ContactPoint] = []
    if contacts:
        for c in contacts:
            contact_points.append(
                ContactPoint(
                    position=np.asarray(c.point, dtype=np.float64).copy(),
                    normal=np.asarray(c.normal, dtype=np.float64).copy(),
                    depth=float(c.depth),
                    body_i=c.body_i,
                    body_j=c.body_j,
                )
            )

    # --- Terrain ---
    if terrain is not None:
        terrain_info = _terrain_to_info(terrain)
    else:
        terrain_info = TerrainInfo(terrain_type="flat", params={"z": 0.0})

    # --- Skeleton links ---
    skeleton_links: list[tuple[np.ndarray, np.ndarray]] = []
    for body in tree.bodies:
        if body.parent >= 0:
            p_pos = X_world[body.parent].r.copy()
            c_pos = X_world[body.index].r.copy()
            skeleton_links.append((p_pos, c_pos))

    # --- Body positions ---
    body_positions = [X_world[i].r.copy() for i in range(len(tree.bodies))]
    body_names = [b.name for b in tree.bodies]

    return RenderScene(
        shapes=shapes,
        contacts=contact_points,
        terrain=terrain_info,
        skeleton_links=skeleton_links,
        body_positions=body_positions,
        body_names=body_names,
    )


def build_render_scene_from_tree(
    tree: "RobotTreeNumpy",
    q: np.ndarray,
    geometries=None,
    contacts=None,
    terrain=None,
) -> RenderScene:
    """Convenience builder from a single RobotTree + q.

    Runs forward kinematics internally, then extracts the scene.
    Compatible with the existing RobotViewer workflow.
    """
    from physics.geometry import BodyCollisionGeometry

    X_world = tree.forward_kinematics(q)

    # --- Shapes ---
    shapes: list[PositionedShape] = []
    if geometries:
        for geom in geometries:
            if not isinstance(geom, BodyCollisionGeometry) or not geom.shapes:
                continue
            body_idx = geom.body_index
            body_name = tree.bodies[body_idx].name if body_idx < len(tree.bodies) else f"body_{body_idx}"
            for si in geom.shapes:
                shape_type, params = _shape_to_type_params(si.shape)
                if shape_type in ("halfspace", "unknown"):
                    continue
                X_shape = si.world_pose(X_world[body_idx])
                shapes.append(
                    PositionedShape(
                        shape_type=shape_type,
                        params=params,
                        position=X_shape.r.copy(),
                        rotation=X_shape.R.copy(),
                        body_index=body_idx,
                        body_name=body_name,
                    )
                )

    # --- Contacts ---
    contact_points: list[ContactPoint] = []
    if contacts:
        for c in contacts:
            contact_points.append(
                ContactPoint(
                    position=np.asarray(c.point, dtype=np.float64).copy(),
                    normal=np.asarray(c.normal, dtype=np.float64).copy(),
                    depth=float(c.depth),
                    body_i=c.body_i,
                    body_j=c.body_j,
                )
            )

    # --- Terrain ---
    if terrain is not None:
        terrain_info = _terrain_to_info(terrain)
    else:
        terrain_info = TerrainInfo(terrain_type="flat", params={"z": 0.0})

    # --- Skeleton links ---
    skeleton_links: list[tuple[np.ndarray, np.ndarray]] = []
    for body in tree.bodies:
        if body.parent >= 0:
            skeleton_links.append((X_world[body.parent].r.copy(), X_world[body.index].r.copy()))

    body_positions = [X_world[i].r.copy() for i in range(len(tree.bodies))]
    body_names = [b.name for b in tree.bodies]

    return RenderScene(
        shapes=shapes,
        contacts=contact_points,
        terrain=terrain_info,
        skeleton_links=skeleton_links,
        body_positions=body_positions,
        body_names=body_names,
    )
