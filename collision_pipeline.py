"""
CollisionPipeline — unified collision detection for Scene.

Replaces the separate ContactModel.compute_forces() + SelfCollisionModel
with a single detect() call that outputs ContactConstraint objects for
the solver.

Three collision sources, all producing the same ContactConstraint type:
  1. Robot body vs terrain   (ground_contact_query)
  2. Robot body vs static geometry  (gjk_epa_query)
  3. Robot body vs robot body  (broad_phase + collision_filter + gjk_epa_query)

The pipeline does NOT solve constraints — it only detects contacts.
Solving is delegated to the ContactSolver (PGS/Jacobi/ADMM).

References:
  MuJoCo mj_collision: broad + narrow → contact list
  Bullet btCollisionDispatcher: dispatches all collision pairs
  Drake QueryObject::ComputeContactSurfaces
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from physics.geometry import CollisionShape
from physics.gjk_epa import gjk_epa_query, ground_contact_query, halfspace_convex_query
from physics.solvers.pgs_solver import ContactConstraint
from physics.spatial import SpatialTransform, Vec6
from physics.terrain import HalfSpaceTerrain

_MIN_NORMAL_NORM = 1e-8  # reject contacts with degenerate normals

if TYPE_CHECKING:
    from scene import Scene


class CollisionPipeline:
    """Detects all contacts in a Scene and returns ContactConstraint list.

    Args:
        scene: The built Scene (with registry and collision_filter).
    """

    def __init__(self, scene: "Scene") -> None:
        self._scene = scene

    def detect(
        self,
        all_X: list[SpatialTransform],
        all_v: list[Vec6],
    ) -> list[ContactConstraint]:
        """Run collision detection for all sources.

        Args:
            all_X: World-frame transforms, indexed by global body id.
                   Len = registry.total_bodies.
            all_v: Spatial velocities in body frame, same indexing.
                   Static bodies should have zero velocity.

        Returns:
            List of ContactConstraint (body indices are global).
        """
        contacts: list[ContactConstraint] = []

        def _valid_normal(n):
            return np.linalg.norm(n) > _MIN_NORMAL_NORM

        scene = self._scene
        reg = scene.registry
        filt = scene.collision_filter

        # ── 1. Robot bodies vs terrain ──
        for name, model in scene.robots.items():
            offset = reg.robot_offset[name]
            for geom in model.geometries:
                if not geom.shapes:
                    continue
                gid = offset + geom.body_index
                X_body = all_X[gid]
                for si in geom.shapes:
                    X_shape = si.world_pose(X_body)
                    if isinstance(scene.terrain, HalfSpaceTerrain):
                        manifold = halfspace_convex_query(
                            si.shape,
                            X_shape,
                            hs_normal_world=scene.terrain.normal_world,
                            hs_point_world=scene.terrain.point_on_plane,
                        )
                    else:
                        gz = scene.terrain.height_at(X_shape.r[0], X_shape.r[1])
                        manifold = ground_contact_query(si.shape, X_shape, ground_z=gz)
                    if manifold is not None and _valid_normal(manifold.normal):
                        for pi, pt in enumerate(manifold.points):
                            contacts.append(
                                ContactConstraint(
                                    body_i=gid,
                                    body_j=-1,  # -1 = ground/terrain
                                    point=pt,
                                    normal=manifold.normal.copy(),
                                    tangent1=np.zeros(3),
                                    tangent2=np.zeros(3),
                                    depth=manifold.depth_at(pi),
                                    mu=getattr(scene.terrain, "mu", 0.5),
                                    condim=3,
                                )
                            )

        # ── 2. Robot bodies vs static geometries ──
        for name, model in scene.robots.items():
            offset = reg.robot_offset[name]
            for geom in model.geometries:
                if not geom.shapes:
                    continue
                gid = offset + geom.body_index
                X_body = all_X[gid]

                for si_shape in geom.shapes:
                    X_shape = si_shape.world_pose(X_body)
                    for si_idx, sg in enumerate(scene.static_geometries):
                        sgid = reg.static_global_id(si_idx)
                        if filt is not None and not filt.should_collide(gid, sgid):
                            continue
                        manifold = gjk_epa_query(si_shape.shape, X_shape, sg.shape, sg.pose)
                        if manifold is not None and _valid_normal(manifold.normal):
                            for pi, pt in enumerate(manifold.points):
                                contacts.append(
                                    ContactConstraint(
                                        body_i=gid,
                                        body_j=sgid,
                                        point=pt,
                                        normal=manifold.normal.copy(),
                                        tangent1=np.zeros(3),
                                        tangent2=np.zeros(3),
                                        depth=manifold.depth_at(pi),
                                        mu=sg.mu,
                                        condim=sg.condim,
                                        mu_spin=sg.mu_spin,
                                        mu_roll=sg.mu_roll,
                                    )
                                )

        # ── 3. Robot body vs robot body (self + inter-robot) ──
        # Collect per-shape entries (gid, shape, X_shape_world)
        body_entries: list[tuple[int, "CollisionShape", SpatialTransform]] = []
        for name, model in scene.robots.items():
            offset = reg.robot_offset[name]
            for geom in model.geometries:
                if not geom.shapes:
                    continue
                gid = offset + geom.body_index
                for si in geom.shapes:
                    X_shape = si.world_pose(all_X[gid])
                    body_entries.append((gid, si.shape, X_shape))

        # Brute-force narrow phase (broad phase optimization can be added later)
        n = len(body_entries)
        for i in range(n):
            gid_i, shape_i, X_i = body_entries[i]
            for j in range(i + 1, n):
                gid_j, shape_j, X_j = body_entries[j]
                # Same-body shape filtering
                if gid_i == gid_j:
                    continue
                if filt is not None and not filt.should_collide(gid_i, gid_j):
                    continue
                manifold = gjk_epa_query(shape_i, X_i, shape_j, X_j)
                if manifold is not None and _valid_normal(manifold.normal):
                    for pi, pt in enumerate(manifold.points):
                        contacts.append(
                            ContactConstraint(
                                body_i=gid_i,
                                body_j=gid_j,
                                point=pt,
                                normal=manifold.normal.copy(),
                                tangent1=np.zeros(3),
                                tangent2=np.zeros(3),
                                depth=manifold.depth_at(pi),
                                mu=0.5,
                                condim=3,
                            )
                        )

        return contacts

    def gather_mass_properties(
        self,
    ) -> tuple[list[float], list[NDArray]]:
        """Return (inv_mass, inv_inertia) for all global body indices.

        Static bodies get inv_mass=0, inv_inertia=zeros (infinite mass).
        """
        scene = self._scene
        reg = scene.registry
        n = reg.total_bodies
        inv_mass = [0.0] * n
        inv_inertia = [np.zeros((3, 3))] * n

        for name, model in scene.robots.items():
            offset = reg.robot_offset[name]
            for bi, body in enumerate(model.tree.bodies):
                gid = offset + bi
                m = body.inertia.mass
                I_com = body.inertia.inertia
                c = body.inertia.com
                I_origin = I_com + m * (np.dot(c, c) * np.eye(3) - np.outer(c, c))
                inv_mass[gid] = 1.0 / m if m > 1e-10 else 0.0
                try:
                    inv_inertia[gid] = np.linalg.inv(I_origin)
                except np.linalg.LinAlgError:
                    inv_inertia[gid] = np.zeros((3, 3))

        # Static bodies already have inv_mass=0, inv_inertia=zeros (infinite mass)
        return inv_mass, inv_inertia
