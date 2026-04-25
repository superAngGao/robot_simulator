"""
Extract a RenderScene from physics state.

Bridges physics types (CollisionShape, SpatialTransform, Terrain) to the
backend-agnostic RenderScene data structure. Imports from ``physics/``
(allowed direction); ``physics/`` never imports from ``rendering/``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np

from physics.engine import ContactInfo
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
from physics.publish import CpuPublishedFrame, GpuPublishedFrame
from physics.spatial import SpatialTransform
from physics.terrain import FlatTerrain, HalfSpaceTerrain

from .render_scene import ContactPoint, PositionedShape, RenderScene, TerrainInfo

if TYPE_CHECKING:
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
        topo = shape.face_topology()
        triangles = []
        for fvids in topo.face_vertex_ids:
            for k in range(1, len(fvids) - 1):
                triangles.append([fvids[0], fvids[k], fvids[k + 1]])
        faces = np.array(triangles, dtype=np.int32)
        return "convex_hull", {"vertices": topo.vertices.copy(), "faces": faces}
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


def build_render_scene_from_gpu(
    engine,
    env_idx: int = 0,
    include_contacts: bool = True,
) -> RenderScene:
    """Build a RenderScene from a GPU engine state.

    Reads ``engine.q_wp`` (Warp array), runs CPU forward kinematics on the
    merged model, and delegates to ``build_render_scene``.

    Args:
        engine          : GpuPhysicsEngine instance.
        env_idx         : Which parallel environment to visualise.
        include_contacts: If False, contacts list will be empty.

    Returns:
        RenderScene ready for any rendering backend.

    Raises:
        IndexError: If env_idx >= number of environments.
    """
    merged = engine.merged
    q_all = engine.q_wp.numpy()
    if env_idx >= q_all.shape[0]:
        raise IndexError(f"env_idx={env_idx} out of bounds for {q_all.shape[0]} environments")
    q_np = q_all[env_idx].astype(np.float64)
    X_world = merged.tree.forward_kinematics(q_np)
    contacts = engine.query_contacts(env_idx) if include_contacts else None
    terrain = getattr(merged, "terrain", None)
    return build_render_scene(merged, X_world, contacts=contacts, terrain=terrain)


def _transforms_from_gpu_published_frame(frame: GpuPublishedFrame, env_idx: int) -> list[SpatialTransform]:
    R_all = frame.x_world_R_wp.numpy()
    r_all = frame.x_world_r_wp.numpy()
    if env_idx >= R_all.shape[0]:
        raise IndexError(f"env_idx={env_idx} out of bounds for {R_all.shape[0]} environments")
    return [
        SpatialTransform(
            R_all[env_idx, body_idx].astype(np.float64), r_all[env_idx, body_idx].astype(np.float64)
        )
        for body_idx in range(R_all.shape[1])
    ]


def _contacts_from_gpu_published_frame(
    frame: GpuPublishedFrame,
    env_idx: int,
    engine=None,
) -> list[ContactInfo]:
    if frame.contact_count_wp is None:
        return [] if engine is None else engine.query_contacts(env_idx)

    count_all = frame.contact_count_wp.numpy()
    if env_idx >= count_all.shape[0]:
        raise IndexError(f"env_idx={env_idx} out of bounds for {count_all.shape[0]} environments")
    n_contacts = int(count_all[env_idx])
    if n_contacts == 0:
        return []

    if frame.contact_cache_ref is None:
        # Phase-1 fallback: if this frame did not materialize the dense contact
        # block, we fall back to `engine.query_contacts(env_idx)`. That returns
        # the engine's latest contact state, not a historical frozen copy tied
        # to `frame.frame_id`, so this path is debug-oriented and should not be
        # reused as a temporal-accuracy guarantee for future sensor pipelines.
        return [] if engine is None else engine.query_contacts(env_idx)

    cache = frame.contact_cache_ref
    bi = cache["contact_bi_wp"].numpy()[env_idx, :n_contacts]
    bj = cache["contact_bj_wp"].numpy()[env_idx, :n_contacts]
    depth = cache["contact_depth_wp"].numpy()[env_idx, :n_contacts]
    normal = cache["contact_normal_wp"].numpy()[env_idx, :n_contacts]
    point = cache["contact_point_wp"].numpy()[env_idx, :n_contacts]
    return [
        ContactInfo(
            body_i=int(bi[idx]),
            body_j=int(bj[idx]),
            depth=float(depth[idx]),
            normal=np.asarray(normal[idx], dtype=np.float64).copy(),
            point=np.asarray(point[idx], dtype=np.float64).copy(),
        )
        for idx in range(n_contacts)
    ]


def build_render_scene_from_published_frame(
    engine,
    frame: CpuPublishedFrame | GpuPublishedFrame | None = None,
    env_idx: int = 0,
    include_contacts: bool = True,
) -> RenderScene:
    """Build a RenderScene from an engine published frame.

    CPU frames use the already-published `X_world` / `contacts` directly.
    GPU frames consume published slot buffers and only fall back to
    `engine.query_contacts(env_idx)` when dense contact buffers were not
    materialized for that frame.
    """
    if frame is None:
        frame = engine.latest_published_frame()
    if frame is None:
        raise RuntimeError("No published frame is available yet.")

    merged = engine.merged
    terrain = getattr(merged, "terrain", None)

    if isinstance(frame, CpuPublishedFrame):
        contacts = frame.contacts if include_contacts else None
        return build_render_scene(merged, frame.X_world, contacts=contacts, terrain=terrain)

    if isinstance(frame, GpuPublishedFrame):
        X_world = _transforms_from_gpu_published_frame(frame, env_idx=env_idx)
        contacts = None
        if include_contacts:
            contacts = _contacts_from_gpu_published_frame(frame, env_idx=env_idx, engine=engine)
        return build_render_scene(merged, X_world, contacts=contacts, terrain=terrain)

    raise TypeError(f"Unsupported published frame type: {type(frame).__name__}")


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
