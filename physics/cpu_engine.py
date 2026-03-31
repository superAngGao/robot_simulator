"""
CpuEngine — CPU physics engine using StepPipeline + collision detection.

Operates on the MergedModel's unified tree. Collision detection runs on
all body pairs (intra-robot + cross-robot) uniformly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np
from numpy.typing import NDArray

from .constraint_solvers import wrap_solver
from .dynamics_cache import DynamicsCache
from .engine import PhysicsEngine, StepOutput
from .force_source import PassiveForceSource
from .gjk_epa import ground_contact_query
from .solvers.pgs_solver import ContactConstraint
from .solvers.pgs_split_impulse import PGSSplitImpulseSolver
from .step_pipeline import StepPipeline

if TYPE_CHECKING:
    from .merged_model import MergedModel


class CpuEngine(PhysicsEngine):
    """CPU physics engine with full collision detection.

    Uses StepPipeline for the two-stage dynamics pipeline and
    GJK/EPA-based collision detection on the merged body list.

    Args:
        merged : MergedModel (multi-root tree + collision data).
        solver : Contact solver (default: PGSSplitImpulseSolver).
        dt     : Default time step [s] (can be overridden in step()).
    """

    def __init__(
        self,
        merged: "MergedModel",
        solver=None,
        dt: float = 2e-4,
    ) -> None:
        super().__init__(merged)
        solver = solver or PGSSplitImpulseSolver(max_iter=60, erp=0.8, slop=0.005)
        wrapped = wrap_solver(solver)
        self._pipeline = StepPipeline(
            dt=dt,
            force_sources=[PassiveForceSource()],
            constraint_solver=wrapped,
        )
        self._dt = dt

    def step(
        self,
        q: NDArray,
        qdot: NDArray,
        tau: NDArray,
        dt: float | None = None,
    ) -> StepOutput:
        dt = dt or self._dt
        tree = self.merged.tree

        # Build DynamicsCache (FK + body_v)
        cache = DynamicsCache.from_tree(tree, q, qdot, dt)

        # Collision detection on merged body list
        contacts = self._detect_contacts(cache)

        # Run pipeline (smooth forces → constraint → integrate)
        self._pipeline.dt = dt
        q_new, qdot_new = self._pipeline.step(tree, q, qdot, tau, contacts, cache=cache)

        # Build output
        X_world = tree.forward_kinematics(q_new)
        v_bodies = tree.body_velocities(q_new, qdot_new)
        contact_active = np.array([True] * len(contacts) if contacts else [])

        return StepOutput(
            q_new=q_new,
            qdot_new=qdot_new,
            X_world=X_world,
            v_bodies=v_bodies,
            contact_active=contact_active,
            force_state=self._pipeline.last_force_state,
        )

    def _detect_contacts(self, cache: DynamicsCache) -> List[ContactConstraint]:
        """Detect all contacts using GJK/EPA: body-ground + body-body."""
        contacts: List[ContactConstraint] = []
        merged = self.merged
        X_world = cache.X_world
        terrain = merged.terrain

        # 1. Ground contacts (GJK/EPA per body, all shapes)
        for body_idx, _local_pos in merged.contact_points:
            geom = merged.collision_shapes[body_idx] if body_idx < len(merged.collision_shapes) else None
            if geom is None or not geom.shapes:
                continue
            X_body = X_world[body_idx]
            for si in geom.shapes:
                X_shape = si.world_pose(X_body)
                gz = terrain.height_at(X_shape.r[0], X_shape.r[1])
                manifold = ground_contact_query(si.shape, X_shape, ground_z=gz)
                if manifold is not None and manifold.depth > 1e-10:
                    for pt in manifold.points:
                        contacts.append(
                            ContactConstraint(
                                body_i=body_idx,
                                body_j=-1,
                                point=pt,
                                normal=manifold.normal.copy(),
                                tangent1=np.zeros(3),
                                tangent2=np.zeros(3),
                                depth=manifold.depth,
                                mu=0.8,
                                condim=3,
                            )
                        )

        # 2. Body-body contacts (analytical sphere approximation)
        # Note: GJK/EPA has poor depth accuracy for deep sphere-sphere penetration
        # (EPA returns ~0.00003 for actual 0.05 overlap). Analytical sphere approx
        # is more reliable for body-body until per-shape analytical functions are
        # ported to CPU.
        for bi, bj in merged.collision_pairs:
            shape_i = merged.collision_shapes[bi]
            shape_j = merged.collision_shapes[bj]
            if shape_i is None or shape_j is None or not shape_i.shapes or not shape_j.shapes:
                continue

            r_i = float(np.mean(shape_i.aabb_half_extents()))
            r_j = float(np.mean(shape_j.aabb_half_extents()))
            diff = X_world[bi].r - X_world[bj].r
            dist = np.linalg.norm(diff)
            overlap = (r_i + r_j) - dist

            if overlap > 0 and dist > 1e-10:
                normal = diff / dist
                contact_pt = X_world[bj].r + normal * r_j
                contacts.append(
                    ContactConstraint(
                        body_i=bi,
                        body_j=bj,
                        point=contact_pt,
                        normal=normal,
                        tangent1=np.zeros(3),
                        tangent2=np.zeros(3),
                        depth=overlap,
                        mu=0.8,
                        condim=3,
                    )
                )

        return contacts
