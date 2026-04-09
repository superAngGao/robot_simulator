"""
MergedModel — merge multiple RobotModels into a single multi-root kinematic tree.

All robots in a Scene become bodies in one flat tree. Each robot's root body
has parent=-1 (independent subtree). ABA/FK/RNEA naturally handle multiple
roots. Collision detection runs on the full body list without distinguishing
"self-collision" from "cross-robot collision."

Usage:
    merged = merge_models(
        robots={"arm": arm_model, "box": box_model},
        terrain=FlatTerrain(),
    )
    # merged.tree has all bodies from both robots
    # merged.robot_slices["arm"].q_slice gives arm's q range in merged state

Reference: MuJoCo — all bodies in one global list, "robots" are just subtrees.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from numpy.typing import NDArray

from .geometry import BodyCollisionGeometry
from .robot_tree import Body, RobotTreeNumpy
from .terrain import FlatTerrain, Terrain

if TYPE_CHECKING:
    from robot.model import RobotModel


@dataclass
class RobotSlice:
    """Index mapping for one robot within a MergedModel."""

    name: str
    body_start: int  # first body index in merged tree
    body_count: int  # number of bodies
    q_slice: slice  # position slice in merged q vector
    v_slice: slice  # velocity slice in merged qdot vector
    nu: int  # actuated DOFs for this robot
    actuated_v_indices: NDArray  # (nu,) indices into merged qdot for actuated joints


@dataclass
class MergedModel:
    """All robots merged into a single multi-root kinematic tree.

    Attributes:
        tree             : Single RobotTreeNumpy with all bodies (multiple roots).
        collision_shapes : Per-body collision geometries (None if no geometry).
        collision_pairs  : Candidate body-pair indices for collision (global).
        contact_points   : (body_idx, local_pos) for ground contact detection.
        robot_slices     : Per-robot index mapping into merged state.
        terrain          : Ground/terrain model.
        nq, nv, nb       : Total merged dimensions.
    """

    tree: RobotTreeNumpy
    collision_shapes: List[Optional[BodyCollisionGeometry]]
    collision_pairs: List[tuple[int, int]]
    contact_points: List[tuple[int, NDArray]]
    robot_slices: dict[str, RobotSlice]
    terrain: Terrain
    nq: int
    nv: int
    nb: int
    collision_filter: object = None  # Optional CollisionFilter for GPU broadphase


def merge_models(
    robots: dict[str, "RobotModel"],
    terrain: Optional[Terrain] = None,
    collision_filter=None,
) -> MergedModel:
    """Merge multiple RobotModels into a single MergedModel.

    Each robot becomes an independent subtree (root has parent=-1).
    Body indices, q/v indices are remapped to global merged space.

    Args:
        robots           : Named robot models to merge.
        terrain          : Terrain model (default: FlatTerrain).
        collision_filter : Optional collision filter (for excluding pairs).

    Returns:
        MergedModel with unified tree and index mappings.
    """
    terrain = terrain or FlatTerrain()

    merged_tree = RobotTreeNumpy(gravity=9.81)
    robot_slices: dict[str, RobotSlice] = {}
    collision_shapes: List[Optional[BodyCollisionGeometry]] = []
    contact_points: List[tuple[int, NDArray]] = []
    all_collision_pairs: List[tuple[int, int]] = []

    body_offset = 0  # global body index offset
    q_offset = 0
    v_offset = 0

    for robot_name, model in robots.items():
        src_tree = model.tree
        src_bodies = src_tree.bodies
        n_bodies = len(src_bodies)
        n_q = src_tree.nq
        n_v = src_tree.nv

        # Record slice for this robot
        # Actuated joints: find v-indices for non-FreeJoint, nv>0 bodies
        from .joint import FreeJoint as _Free

        actuated_v = []
        for b in src_bodies:
            if b.joint.nv > 0 and not isinstance(b.joint, _Free):
                for vi in range(b.v_idx.start, b.v_idx.stop):
                    actuated_v.append(v_offset + vi)
        nu = len(actuated_v)

        robot_slices[robot_name] = RobotSlice(
            name=robot_name,
            body_start=body_offset,
            body_count=n_bodies,
            q_slice=slice(q_offset, q_offset + n_q),
            v_slice=slice(v_offset, v_offset + n_v),
            nu=nu,
            actuated_v_indices=np.array(actuated_v, dtype=np.int32),
        )

        # Add bodies to merged tree with remapped indices
        for b in src_bodies:
            new_parent = b.parent + body_offset if b.parent >= 0 else -1
            new_body = Body(
                name=f"{robot_name}/{b.name}",
                index=body_offset + b.index,
                joint=deepcopy(b.joint),
                inertia=deepcopy(b.inertia),
                X_tree=deepcopy(b.X_tree),
                parent=new_parent,
                children=[],  # will be rebuilt by add_body
                q_idx=slice(0, 0),  # will be set by finalize
                v_idx=slice(0, 0),
            )
            merged_tree.add_body(new_body)

        # Contact points (ground contact detection bodies)
        contact_names = set(model.contact_body_names)
        for b in src_bodies:
            if b.name in contact_names:
                contact_points.append((body_offset + b.index, np.zeros(3, dtype=np.float64)))

        # Collision shapes (per-body geometry)
        geom_by_body: dict[int, BodyCollisionGeometry] = {}
        for g in model.geometries:
            geom_by_body[g.body_index] = g
        for i in range(n_bodies):
            if i in geom_by_body:
                # Remap body_index to global
                g = deepcopy(geom_by_body[i])
                g.body_index = body_offset + i
                collision_shapes.append(g)
            else:
                collision_shapes.append(None)

        # Intra-robot collision pairs (self-collision)
        parent_list = [b.parent for b in src_bodies]
        for i in range(n_bodies):
            for j in range(i + 1, n_bodies):
                # Skip parent-child pairs
                if parent_list[j] == i or parent_list[i] == j:
                    continue
                gi = body_offset + i
                gj = body_offset + j
                if collision_shapes[gi] is not None and collision_shapes[gj] is not None:
                    # Apply CollisionFilter at pair-build time (mirrors GPU
                    # static_data.collision_excluded built from same filter).
                    if collision_filter is not None and not collision_filter.should_collide(gi, gj):
                        continue
                    all_collision_pairs.append((gi, gj))

        body_offset += n_bodies
        q_offset += n_q
        v_offset += n_v

    # Cross-robot collision pairs
    robot_names = list(robots.keys())
    for ri in range(len(robot_names)):
        for rj in range(ri + 1, len(robot_names)):
            slice_i = robot_slices[robot_names[ri]]
            slice_j = robot_slices[robot_names[rj]]
            for bi in range(slice_i.body_start, slice_i.body_start + slice_i.body_count):
                for bj in range(slice_j.body_start, slice_j.body_start + slice_j.body_count):
                    if collision_shapes[bi] is not None and collision_shapes[bj] is not None:
                        if collision_filter is not None and not collision_filter.should_collide(bi, bj):
                            continue
                        all_collision_pairs.append((bi, bj))

    # Set gravity from first robot's tree (all should match)
    if robots:
        first_tree = next(iter(robots.values())).tree
        merged_tree._gravity = first_tree._gravity

    merged_tree.finalize()

    return MergedModel(
        tree=merged_tree,
        collision_shapes=collision_shapes,
        collision_pairs=all_collision_pairs,
        contact_points=contact_points,
        robot_slices=robot_slices,
        terrain=terrain,
        nq=merged_tree.nq,
        nv=merged_tree.nv,
        nb=merged_tree.num_bodies,
        collision_filter=collision_filter,
    )
