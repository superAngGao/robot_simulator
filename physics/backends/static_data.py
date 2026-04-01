"""
StaticRobotData — flatten a RobotModel's object tree into contiguous arrays.

These arrays are constant for all N environments and all time steps.
They can be transferred to GPU once at initialisation and used as kernel
arguments without any per-step host-device copies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from physics.geometry import BoxShape, CapsuleShape, CylinderShape, SphereShape
from physics.joint import (
    FixedJoint,
    FreeJoint,
    PrismaticJoint,
    RevoluteJoint,
)

if TYPE_CHECKING:
    from robot.model import RobotModel

# Joint type enum values (used in GPU kernels)
JOINT_FREE = 0
JOINT_REVOLUTE = 1
JOINT_PRISMATIC = 2
JOINT_FIXED = 3

# Shape type enum values (used in GPU collision kernels)
SHAPE_NONE = 0
SHAPE_SPHERE = 1
SHAPE_BOX = 2
SHAPE_CYLINDER = 3
SHAPE_CAPSULE = 4


@dataclass
class StaticRobotData:
    """Flattened, contiguous representation of a RobotModel.

    All arrays use float32 (GPU-friendly) except integer index arrays.
    """

    # -- Dimensions --
    nb: int  # number of bodies
    nq: int  # total generalised position dim
    nv: int  # total generalised velocity dim
    nc: int  # number of contact points
    nu: int  # number of actuated DOFs

    # -- Per-body arrays (nb,) --
    joint_type: NDArray[np.int32]  # (nb,) JOINT_* enum
    joint_axis: NDArray[np.float32]  # (nb, 3)
    parent_idx: NDArray[np.int32]  # (nb,) -1 for root
    q_idx_start: NDArray[np.int32]  # (nb,)
    q_idx_len: NDArray[np.int32]  # (nb,)
    v_idx_start: NDArray[np.int32]  # (nb,)
    v_idx_len: NDArray[np.int32]  # (nb,)

    # -- Per-body transforms --
    X_tree_R: NDArray[np.float32]  # (nb, 3, 3)
    X_tree_r: NDArray[np.float32]  # (nb, 3)

    # -- Per-body inertia (precomputed 6x6 spatial inertia matrix) --
    inertia_mat: NDArray[np.float32]  # (nb, 6, 6)

    # -- Joint limits & damping --
    q_min: NDArray[np.float32]  # (nb,)
    q_max: NDArray[np.float32]  # (nb,)
    k_limit: NDArray[np.float32]  # (nb,)
    b_limit: NDArray[np.float32]  # (nb,)
    damping: NDArray[np.float32]  # (nb,)

    # -- Contact points --
    contact_body_idx: NDArray[np.int32]  # (nc,)
    contact_local_pos: NDArray[np.float32]  # (nc, 3)
    contact_k_normal: float = 5000.0
    contact_b_normal: float = 500.0
    contact_mu: float = 0.8
    contact_slip_eps: float = 1e-3
    contact_ground_z: float = 0.0

    # -- Self-collision --
    collision_body_idx: NDArray[np.int32] = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    collision_half_ext: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float32)
    )
    collision_pair_i: NDArray[np.int32] = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    collision_pair_j: NDArray[np.int32] = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    collision_k: float = 2000.0
    collision_b: float = 100.0

    # -- Controller indices --
    actuated_q_indices: NDArray[np.int32] = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    actuated_v_indices: NDArray[np.int32] = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    effort_limits: NDArray[np.float32] | None = None  # (nu,) or None

    # -- Gravity --
    gravity: float = 9.81

    # -- Default state --
    default_q: NDArray[np.float32] = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    default_qdot: NDArray[np.float32] = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    # -- Per-body inverse mass/inertia (for constraint solvers) --
    inv_mass_per_body: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )  # (nb,) scalar 1/m
    inv_inertia_per_body: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((0, 3, 3), dtype=np.float32)
    )  # (nb, 3, 3) inverse rotational inertia at body origin

    # -- Body-body collision pairs (for constraint solver) --
    collision_pair_body_i: NDArray[np.int32] = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )  # (n_pairs,) global body index
    collision_pair_body_j: NDArray[np.int32] = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )  # (n_pairs,) global body index
    body_collision_radius: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )  # (nb,) collision sphere radius per body (legacy, kept for backward compat)

    # -- Per-body collision shape descriptors (for analytical GPU collision) --
    body_shape_type: NDArray[np.int32] = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )  # (nb,) 0=None, 1=Sphere, 2=Box, 3=Cylinder, 4=Capsule
    body_shape_params: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((0, 4), dtype=np.float32)
    )  # (nb, 4) packed shape params: Sphere=[r,0,0,0], Box=[hx,hy,hz,0],
    #   Cylinder=[r,hl,0,0], Capsule=[r,hl,0,0]

    # -- Constraint solver parameters --
    contact_cfm: float = 1e-6  # Constraint Force Mixing (regularization)
    contact_erp_pos: float = 0.8  # Position correction ERP (split impulse)
    contact_slop: float = 0.005  # Allowed penetration before correction [m]
    solver_max_iter: int = 60  # Jacobi PGS iteration count
    solver_omega: float = 0.7  # Jacobi relaxation factor
    # Solimp impedance params for friction R regularization (Q25 fix)
    solimp_d0: float = 0.95
    solimp_dw: float = 0.99
    solimp_width: float = 0.001
    solimp_mid: float = 0.5
    solimp_power: float = 2.0

    # -- Root body index (for observation slicing) --
    root_body_idx: int = 0
    root_q_start: int = 0
    root_q_len: int = 7  # FreeJoint default

    @classmethod
    def from_model(cls, model: "RobotModel") -> "StaticRobotData":
        """Extract all constant data from a RobotModel into flat arrays."""
        tree = model.tree
        bodies = tree.bodies
        nb = len(bodies)

        # -- Per-body arrays --
        joint_type = np.zeros(nb, dtype=np.int32)
        joint_axis = np.zeros((nb, 3), dtype=np.float32)
        parent_idx = np.zeros(nb, dtype=np.int32)
        q_idx_start = np.zeros(nb, dtype=np.int32)
        q_idx_len = np.zeros(nb, dtype=np.int32)
        v_idx_start = np.zeros(nb, dtype=np.int32)
        v_idx_len = np.zeros(nb, dtype=np.int32)
        X_tree_R = np.zeros((nb, 3, 3), dtype=np.float32)
        X_tree_r = np.zeros((nb, 3), dtype=np.float32)
        inertia_mat = np.zeros((nb, 6, 6), dtype=np.float32)
        q_min_arr = np.full(nb, -np.inf, dtype=np.float32)
        q_max_arr = np.full(nb, np.inf, dtype=np.float32)
        k_limit_arr = np.zeros(nb, dtype=np.float32)
        b_limit_arr = np.zeros(nb, dtype=np.float32)
        damping_arr = np.zeros(nb, dtype=np.float32)

        for i, body in enumerate(bodies):
            j = body.joint
            parent_idx[i] = body.parent
            q_idx_start[i] = body.q_idx.start
            q_idx_len[i] = body.q_idx.stop - body.q_idx.start
            v_idx_start[i] = body.v_idx.start
            v_idx_len[i] = body.v_idx.stop - body.v_idx.start

            X_tree_R[i] = body.X_tree.R.astype(np.float32)
            X_tree_r[i] = body.X_tree.r.astype(np.float32)
            inertia_mat[i] = body.inertia.matrix().astype(np.float32)

            if isinstance(j, FreeJoint):
                joint_type[i] = JOINT_FREE
            elif isinstance(j, RevoluteJoint):
                joint_type[i] = JOINT_REVOLUTE
                joint_axis[i] = j._axis_vec.astype(np.float32)
                q_min_arr[i] = j.q_min
                q_max_arr[i] = j.q_max
                k_limit_arr[i] = j.k_limit
                b_limit_arr[i] = j.b_limit
                damping_arr[i] = j.damping
            elif isinstance(j, PrismaticJoint):
                joint_type[i] = JOINT_PRISMATIC
                joint_axis[i] = j._axis_vec.astype(np.float32)
                damping_arr[i] = j.damping
            elif isinstance(j, FixedJoint):
                joint_type[i] = JOINT_FIXED

        # -- Contact points (derived from contact_body_names + geometries) --
        # GPU backends use penalty contact at body origins for named contact links
        contact_body_names = set(model.contact_body_names)
        contact_bodies = [b for b in bodies if b.name in contact_body_names]
        nc = len(contact_bodies)
        contact_body_idx = (
            np.array([b.index for b in contact_bodies], dtype=np.int32)
            if nc > 0
            else np.zeros(0, dtype=np.int32)
        )
        contact_local_pos = np.zeros((max(nc, 0), 3), dtype=np.float32)
        # Default penalty contact params (GPU backends use these)
        contact_params = dict(
            contact_k_normal=5000.0,
            contact_b_normal=500.0,
            contact_mu=0.8,
            contact_slip_eps=1e-3,
            contact_ground_z=0.0,
        )
        # Override from model if old-style contact_model exists (backward compat)
        if hasattr(model, "contact_model"):
            from physics.contact import PenaltyContactModel

            if isinstance(model.contact_model, PenaltyContactModel):
                cps = model.contact_model.contact_points
                nc = len(cps)
                contact_body_idx = np.array([cp.body_index for cp in cps], dtype=np.int32)
                contact_local_pos = np.array([cp.position for cp in cps], dtype=np.float32).reshape(nc, 3)
                p = model.contact_model.params
                contact_params = dict(
                    contact_k_normal=p.k_normal,
                    contact_b_normal=p.b_normal,
                    contact_mu=p.mu,
                    contact_slip_eps=p.slip_eps,
                    contact_ground_z=p.ground_z,
                )

        # -- Self-collision (derived from geometries + parent list) --
        from physics.collision import AABBSelfCollision

        collision_kwargs = {}
        # Override from model if old-style self_collision exists (backward compat)
        if hasattr(model, "self_collision"):
            from physics.collision import AABBSelfCollision as _ASC

            if isinstance(model.self_collision, _ASC):
                sc = model.self_collision
                aabbs = sc._aabbs
                pairs = sc._pairs
                collision_kwargs = dict(
                    collision_body_idx=np.array([a.body_index for a in aabbs], dtype=np.int32),
                    collision_half_ext=np.array([a.half_extents for a in aabbs], dtype=np.float32).reshape(
                        len(aabbs), 3
                    )
                    if aabbs
                    else np.zeros((0, 3), dtype=np.float32),
                    collision_pair_i=np.array([p[0] for p in pairs], dtype=np.int32)
                    if pairs
                    else np.zeros(0, dtype=np.int32),
                    collision_pair_j=np.array([p[1] for p in pairs], dtype=np.int32)
                    if pairs
                    else np.zeros(0, dtype=np.int32),
                    collision_k=sc.k_contact,
                    collision_b=sc.b_contact,
                )

        if not collision_kwargs and model.geometries:
            # Build from geometries directly
            parent_list = [b.parent for b in bodies]
            sc = AABBSelfCollision.from_geometries(model.geometries, parent_list)
            aabbs = sc._aabbs
            pairs = sc._pairs
            collision_kwargs = dict(
                collision_body_idx=np.array([a.body_index for a in aabbs], dtype=np.int32),
                collision_half_ext=np.array([a.half_extents for a in aabbs], dtype=np.float32).reshape(
                    len(aabbs), 3
                )
                if aabbs
                else np.zeros((0, 3), dtype=np.float32),
                collision_pair_i=np.array([p[0] for p in pairs], dtype=np.int32)
                if pairs
                else np.zeros(0, dtype=np.int32),
                collision_pair_j=np.array([p[1] for p in pairs], dtype=np.int32)
                if pairs
                else np.zeros(0, dtype=np.int32),
                collision_k=2000.0,
                collision_b=100.0,
            )

        # -- Per-body inverse mass/inertia (for constraint solver) --
        inv_mass_per_body = np.zeros(nb, dtype=np.float32)
        inv_inertia_per_body = np.zeros((nb, 3, 3), dtype=np.float32)
        for i, body in enumerate(bodies):
            m = body.inertia.mass
            if m > 1e-10:
                inv_mass_per_body[i] = 1.0 / m
                I_com = body.inertia.inertia  # (3,3) at CoM
                c = body.inertia.com  # CoM offset
                # Parallel axis theorem: I_origin = I_com + m*(|c|²I - ccᵀ)
                I_origin = I_com + m * (np.dot(c, c) * np.eye(3) - np.outer(c, c))
                try:
                    inv_inertia_per_body[i] = np.linalg.inv(I_origin).astype(np.float32)
                except np.linalg.LinAlgError:
                    pass  # stays zero

        # -- Controller indices --
        actuated_bodies = [b for b in bodies if b.joint.nv > 0 and not isinstance(b.joint, FreeJoint)]
        actuated_q_indices = np.array(
            [i for b in actuated_bodies for i in range(b.q_idx.start, b.q_idx.stop)],
            dtype=np.int32,
        )
        actuated_v_indices = np.array(
            [i for b in actuated_bodies for i in range(b.v_idx.start, b.v_idx.stop)],
            dtype=np.int32,
        )
        nu = len(actuated_q_indices)
        effort_limits = None
        if model.effort_limits is not None:
            effort_limits = model.effort_limits.astype(np.float32)

        # -- Default state --
        q0, qdot0 = tree.default_state()

        # -- Root body info --
        root_body = bodies[0]
        root_q_start = root_body.q_idx.start
        root_q_len = root_body.q_idx.stop - root_body.q_idx.start

        return cls(
            nb=nb,
            nq=tree.nq,
            nv=tree.nv,
            nc=nc,
            nu=nu,
            joint_type=joint_type,
            joint_axis=joint_axis,
            parent_idx=parent_idx,
            q_idx_start=q_idx_start,
            q_idx_len=q_idx_len,
            v_idx_start=v_idx_start,
            v_idx_len=v_idx_len,
            X_tree_R=X_tree_R,
            X_tree_r=X_tree_r,
            inertia_mat=inertia_mat,
            q_min=q_min_arr,
            q_max=q_max_arr,
            k_limit=k_limit_arr,
            b_limit=b_limit_arr,
            damping=damping_arr,
            contact_body_idx=contact_body_idx,
            contact_local_pos=contact_local_pos,
            **contact_params,
            **collision_kwargs,
            inv_mass_per_body=inv_mass_per_body,
            inv_inertia_per_body=inv_inertia_per_body,
            actuated_q_indices=actuated_q_indices,
            actuated_v_indices=actuated_v_indices,
            effort_limits=effort_limits,
            gravity=tree._gravity,
            default_q=q0.astype(np.float32),
            default_qdot=qdot0.astype(np.float32),
            root_body_idx=0,
            root_q_start=root_q_start,
            root_q_len=root_q_len,
        )

    @classmethod
    def from_merged(cls, merged) -> "StaticRobotData":
        """Build StaticRobotData from a MergedModel (multi-robot).

        Args:
            merged: physics.merged_model.MergedModel instance.
        """
        tree = merged.tree
        bodies = tree.bodies
        nb = len(bodies)

        # Per-body arrays (same logic as from_model but on merged tree)
        joint_type = np.zeros(nb, dtype=np.int32)
        joint_axis = np.zeros((nb, 3), dtype=np.float32)
        parent_idx = np.zeros(nb, dtype=np.int32)
        q_idx_start = np.zeros(nb, dtype=np.int32)
        q_idx_len = np.zeros(nb, dtype=np.int32)
        v_idx_start = np.zeros(nb, dtype=np.int32)
        v_idx_len = np.zeros(nb, dtype=np.int32)
        X_tree_R = np.zeros((nb, 3, 3), dtype=np.float32)
        X_tree_r = np.zeros((nb, 3), dtype=np.float32)
        inertia_mat = np.zeros((nb, 6, 6), dtype=np.float32)
        q_min_arr = np.full(nb, -np.inf, dtype=np.float32)
        q_max_arr = np.full(nb, np.inf, dtype=np.float32)
        k_limit_arr = np.zeros(nb, dtype=np.float32)
        b_limit_arr = np.zeros(nb, dtype=np.float32)
        damping_arr = np.zeros(nb, dtype=np.float32)
        inv_mass_per_body = np.zeros(nb, dtype=np.float32)
        inv_inertia_per_body = np.zeros((nb, 3, 3), dtype=np.float32)
        body_coll_radius = np.full(nb, 0.05, dtype=np.float32)  # default sphere radius
        body_shape_type = np.zeros(nb, dtype=np.int32)  # SHAPE_NONE by default
        body_shape_params = np.zeros((nb, 4), dtype=np.float32)

        # Populate collision shape data from actual geometry
        # TODO(Q26-gpu): support multi-shape per body in GPU arrays
        if merged.collision_shapes:
            for i, geom in enumerate(merged.collision_shapes):
                if geom is None or not geom.shapes:
                    continue
                # Pick largest shape by half-extent volume as representative
                shape = max(geom.shapes, key=lambda s: float(np.prod(s.shape.half_extents_approx()))).shape
                he = shape.half_extents_approx()
                body_coll_radius[i] = float(np.mean(he))

                if isinstance(shape, SphereShape):
                    body_shape_type[i] = SHAPE_SPHERE
                    body_shape_params[i, 0] = shape.radius
                elif isinstance(shape, BoxShape):
                    body_shape_type[i] = SHAPE_BOX
                    body_shape_params[i, :3] = (shape._size / 2.0).astype(np.float32)
                elif isinstance(shape, CylinderShape):
                    body_shape_type[i] = SHAPE_CYLINDER
                    body_shape_params[i, 0] = shape._radius
                    body_shape_params[i, 1] = shape._length / 2.0
                elif isinstance(shape, CapsuleShape):
                    body_shape_type[i] = SHAPE_CAPSULE
                    body_shape_params[i, 0] = shape.radius
                    body_shape_params[i, 1] = shape.length / 2.0
                # ConvexHullShape/MeshShape: falls back to SHAPE_NONE + sphere radius

        for i, body in enumerate(bodies):
            j = body.joint
            parent_idx[i] = body.parent
            q_idx_start[i] = body.q_idx.start
            q_idx_len[i] = body.q_idx.stop - body.q_idx.start
            v_idx_start[i] = body.v_idx.start
            v_idx_len[i] = body.v_idx.stop - body.v_idx.start
            X_tree_R[i] = body.X_tree.R.astype(np.float32)
            X_tree_r[i] = body.X_tree.r.astype(np.float32)
            inertia_mat[i] = body.inertia.matrix().astype(np.float32)

            if isinstance(j, FreeJoint):
                joint_type[i] = JOINT_FREE
            elif isinstance(j, RevoluteJoint):
                joint_type[i] = JOINT_REVOLUTE
                joint_axis[i] = j._axis_vec.astype(np.float32)
                q_min_arr[i] = j.q_min
                q_max_arr[i] = j.q_max
                k_limit_arr[i] = j.k_limit
                b_limit_arr[i] = j.b_limit
                damping_arr[i] = j.damping
            elif isinstance(j, PrismaticJoint):
                joint_type[i] = JOINT_PRISMATIC
                joint_axis[i] = j._axis_vec.astype(np.float32)
                damping_arr[i] = j.damping
            elif isinstance(j, FixedJoint):
                joint_type[i] = JOINT_FIXED

            # Inverse mass/inertia
            m = body.inertia.mass
            if m > 1e-10:
                inv_mass_per_body[i] = 1.0 / m
                I_com = body.inertia.inertia
                c = body.inertia.com
                I_origin = I_com + m * (np.dot(c, c) * np.eye(3) - np.outer(c, c))
                try:
                    inv_inertia_per_body[i] = np.linalg.inv(I_origin).astype(np.float32)
                except np.linalg.LinAlgError:
                    pass

        # Contact points
        nc = len(merged.contact_points)
        contact_body_idx = (
            np.array([cp[0] for cp in merged.contact_points], dtype=np.int32)
            if nc > 0
            else np.zeros(0, dtype=np.int32)
        )
        contact_local_pos = (
            np.array([cp[1] for cp in merged.contact_points], dtype=np.float32).reshape(max(nc, 0), 3)
            if nc > 0
            else np.zeros((0, 3), dtype=np.float32)
        )

        # Body-body collision pairs
        n_pairs = len(merged.collision_pairs)
        coll_bi = (
            np.array([p[0] for p in merged.collision_pairs], dtype=np.int32)
            if n_pairs > 0
            else np.zeros(0, dtype=np.int32)
        )
        coll_bj = (
            np.array([p[1] for p in merged.collision_pairs], dtype=np.int32)
            if n_pairs > 0
            else np.zeros(0, dtype=np.int32)
        )

        # Actuated indices (from all robot slices)
        actuated_v_all = []
        for rs in merged.robot_slices.values():
            actuated_v_all.extend(rs.actuated_v_indices.tolist())
        actuated_v_indices = (
            np.array(actuated_v_all, dtype=np.int32) if actuated_v_all else np.zeros(0, dtype=np.int32)
        )
        # Corresponding q indices (for PD controller): find q_idx for each actuated v_idx
        actuated_q_indices = np.zeros_like(actuated_v_indices)
        for idx, vi in enumerate(actuated_v_indices):
            for body in bodies:
                if body.v_idx.start <= vi < body.v_idx.stop:
                    offset_in_joint = vi - body.v_idx.start
                    actuated_q_indices[idx] = body.q_idx.start + offset_in_joint
                    break
        nu = len(actuated_v_indices)

        q0, qdot0 = tree.default_state()

        root_body = bodies[0]
        root_q_start = root_body.q_idx.start
        root_q_len = root_body.q_idx.stop - root_body.q_idx.start

        return cls(
            nb=nb,
            nq=tree.nq,
            nv=tree.nv,
            nc=nc,
            nu=nu,
            joint_type=joint_type,
            joint_axis=joint_axis,
            parent_idx=parent_idx,
            q_idx_start=q_idx_start,
            q_idx_len=q_idx_len,
            v_idx_start=v_idx_start,
            v_idx_len=v_idx_len,
            X_tree_R=X_tree_R,
            X_tree_r=X_tree_r,
            inertia_mat=inertia_mat,
            q_min=q_min_arr,
            q_max=q_max_arr,
            k_limit=k_limit_arr,
            b_limit=b_limit_arr,
            damping=damping_arr,
            contact_body_idx=contact_body_idx,
            contact_local_pos=contact_local_pos,
            collision_pair_body_i=coll_bi,
            collision_pair_body_j=coll_bj,
            body_collision_radius=body_coll_radius,
            body_shape_type=body_shape_type,
            body_shape_params=body_shape_params,
            inv_mass_per_body=inv_mass_per_body,
            inv_inertia_per_body=inv_inertia_per_body,
            actuated_q_indices=actuated_q_indices,
            actuated_v_indices=actuated_v_indices,
            gravity=tree._gravity,
            default_q=q0.astype(np.float32),
            default_qdot=qdot0.astype(np.float32),
            root_body_idx=0,
            root_q_start=root_q_start,
            root_q_len=root_q_len,
        )
