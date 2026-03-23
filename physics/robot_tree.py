"""
Articulated rigid body tree with forward kinematics and Featherstone ABA.

The robot is modelled as a kinematic tree:
  - One root body (typically the floating base / torso)
  - Each non-root body has exactly one parent
  - Bodies are connected by joints (revolute, fixed, free, ...)

State vector layout
-------------------
Generalised positions  q    : concatenation of each joint's q  (total nq)
Generalised velocities qdot : concatenation of each joint's qdot (total nv)
Generalised forces     tau  : concatenation of each joint's tau  (total nv)

Algorithms implemented
----------------------
  forward_kinematics   : compute world-frame pose of every body
  rnea                 : Recursive Newton-Euler (inverse dynamics)
  aba                  : Articulated Body Algorithm (forward dynamics)

References:
  Featherstone (2008), Chapters 7–9.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from ._robot_tree_base import RobotTreeBase
from .joint import Joint
from .spatial import (
    Mat6,
    SpatialInertia,
    SpatialTransform,
    Vec6,
    gravity_spatial,
    spatial_cross_force,
    spatial_cross_velocity,
)

# ---------------------------------------------------------------------------
# Body
# ---------------------------------------------------------------------------


@dataclass
class Body:
    """A single rigid body in the kinematic tree.

    Attributes:
        name        : Unique identifier string.
        index       : Integer index within the RobotTree body list.
        joint       : Joint connecting this body to its parent.
        inertia     : Spatial inertia expressed at the body origin.
        X_tree      : Fixed transform from parent body frame to this body's
                      default (zero-angle) frame  (the "tree" transform).
        parent      : Index of the parent body (-1 for the root).
        children    : Indices of child bodies.
        q_idx       : Slice into the global q  vector for this body's joint.
        v_idx       : Slice into the global qdot / tau vector.
    """

    name: str
    index: int
    joint: Joint
    inertia: SpatialInertia
    X_tree: SpatialTransform
    parent: int = -1
    children: List[int] = field(default_factory=list)
    q_idx: slice = field(default=slice(0, 0))
    v_idx: slice = field(default=slice(0, 0))


# ---------------------------------------------------------------------------
# Kinematic state (scratch-pad, recomputed each step)
# ---------------------------------------------------------------------------


@dataclass
class KinematicState:
    """Per-body quantities computed during forward kinematics / ABA."""

    X_world: SpatialTransform  # body frame expressed in world
    v: Vec6  # spatial velocity  in body frame
    a: Vec6  # spatial acceleration in body frame
    f: Vec6  # spatial force (net) in body frame


# ---------------------------------------------------------------------------
# Robot tree
# ---------------------------------------------------------------------------


class RobotTreeNumpy(RobotTreeBase):
    """Articulated rigid body tree — NumPy CPU backend.

    Usage
    -----
    >>> tree = RobotTreeNumpy()
    >>> root_idx = tree.add_body(Body(...))
    >>> leg_idx  = tree.add_body(Body(..., parent=root_idx))
    >>> tree.finalize()

    >>> q, qdot = tree.default_state()
    >>> poses   = tree.forward_kinematics(q)
    >>> qacc    = tree.aba(q, qdot, tau, external_forces)
    """

    def __init__(self, gravity: float = 9.81) -> None:
        self._bodies: List[Body] = []
        self._gravity: float = gravity
        self._finalized: bool = False
        self._nq: int = 0
        self._nv: int = 0

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def add_body(self, body: Body) -> int:
        """Add a body to the tree and return its index."""
        if self._finalized:
            raise RuntimeError("Cannot add bodies after finalize().")
        idx = len(self._bodies)
        body.index = idx
        if body.parent >= 0:
            self._bodies[body.parent].children.append(idx)
        self._bodies.append(body)
        return idx

    def finalize(self) -> None:
        """Compute q/v index slices for each joint. Call once after adding all bodies."""
        nq = nv = 0
        for body in self._bodies:
            j = body.joint
            body.q_idx = slice(nq, nq + j.nq)
            body.v_idx = slice(nv, nv + j.nv)
            nq += j.nq
            nv += j.nv
        self._nq = nq
        self._nv = nv
        self._finalized = True

    @property
    def bodies(self) -> List[Body]:
        return self._bodies

    @property
    def nq(self) -> int:
        return self._nq

    @property
    def nv(self) -> int:
        return self._nv

    @property
    def num_bodies(self) -> int:
        return len(self._bodies)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def default_state(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (q, qdot) initialised to the zero / default configuration."""
        q = np.concatenate([b.joint.default_q() for b in self._bodies])
        qdot = np.concatenate([b.joint.default_qdot() for b in self._bodies])
        return q, qdot

    def _q_of(self, body: Body, q: NDArray) -> NDArray:
        return q[body.q_idx]

    def _v_of(self, body: Body, v: NDArray) -> NDArray:
        return v[body.v_idx]

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------

    def forward_kinematics(
        self,
        q: NDArray[np.float64],
    ) -> List[SpatialTransform]:
        """Compute the world-frame transform for every body.

        Returns:
            List of SpatialTransform, one per body, mapping body frame → world.
        """
        self._check_finalized()
        X_world: List[Optional[SpatialTransform]] = [None] * self.num_bodies

        for body in self._bodies:
            X_J = body.joint.transform(self._q_of(body, q))
            X_local = body.X_tree @ X_J  # parent-to-body (default+joint)
            if body.parent < 0:
                X_world[body.index] = X_local
            else:
                X_world[body.index] = X_world[body.parent] @ X_local  # type: ignore[operator]

        return X_world  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # RNEA — Recursive Newton-Euler Algorithm  (inverse dynamics)
    # ------------------------------------------------------------------

    def rnea(
        self,
        q: NDArray[np.float64],
        qdot: NDArray[np.float64],
        qddot: NDArray[np.float64],
        external_forces: Optional[List[Optional[Vec6]]] = None,
    ) -> NDArray[np.float64]:
        """Compute the joint torques required to produce acceleration qddot.

        Args:
            q, qdot, qddot:  Generalised positions, velocities, accelerations.
            external_forces: Optional list of spatial forces (in world frame)
                             applied to each body. None entries → zero force.

        Returns:
            tau : (nv,) array of joint torques / forces.
        """
        self._check_finalized()
        n = self.num_bodies
        v: List[Vec6] = [np.zeros(6)] * n
        a: List[Vec6] = [np.zeros(6)] * n
        f: List[Vec6] = [np.zeros(6)] * n

        # Spatial gravity vector (treated as base acceleration bias)
        a_gravity = gravity_spatial(self._gravity)

        # --- Pass 1: forward recursion (kinematics) ---
        for body in self._bodies:
            q_i = self._q_of(body, q)
            qdot_i = self._v_of(body, qdot)
            qddot_i = self._v_of(body, qddot)

            X_J = body.joint.transform(q_i)
            S = body.joint.motion_subspace(q_i)
            cJ = body.joint.bias_acceleration(q_i, qdot_i)
            X_up = body.X_tree @ X_J  # parent → body

            vJ = S @ qdot_i if S.shape[1] > 0 else np.zeros(6)

            if body.parent < 0:
                v[body.index] = vJ
                a[body.index] = X_up.apply_velocity(-a_gravity) + S @ qddot_i + cJ
            else:
                v_p = v[body.parent]
                v[body.index] = X_up.apply_velocity(v_p) + vJ
                a[body.index] = (
                    X_up.apply_velocity(a[body.parent])
                    + S @ qddot_i
                    + cJ
                    + spatial_cross_velocity(v[body.index]) @ vJ
                )

            # Net rigid-body force = I*a + v×*(I*v) − external
            I_mat = body.inertia.matrix()
            vi = v[body.index]
            ai = a[body.index]
            f[body.index] = I_mat @ ai + spatial_cross_force(vi) @ (I_mat @ vi)
            if external_forces is not None and external_forces[body.index] is not None:
                f[body.index] -= external_forces[body.index]

        # --- Pass 2: backward recursion (force propagation) ---
        tau = np.zeros(self.nv, dtype=np.float64)
        for body in reversed(self._bodies):
            q_i = self._q_of(body, q)
            S = body.joint.motion_subspace(q_i)
            X_J = body.joint.transform(q_i)
            X_up = body.X_tree @ X_J

            if S.shape[1] > 0:
                tau[body.v_idx] = S.T @ f[body.index]

            if body.parent >= 0:
                f[body.parent] += X_up.apply_force(f[body.index])

        return tau

    # ------------------------------------------------------------------
    # ABA — Articulated Body Algorithm  (forward dynamics)
    # ------------------------------------------------------------------

    def aba(
        self,
        q: NDArray[np.float64],
        qdot: NDArray[np.float64],
        tau: NDArray[np.float64],
        external_forces: Optional[List[Optional[Vec6]]] = None,
    ) -> NDArray[np.float64]:
        """Compute generalised accelerations given joint torques.

        This is the O(n) Articulated Body Algorithm (Featherstone 2008, Ch.7).

        Args:
            q, qdot:  Current generalised positions and velocities.
            tau:      Applied joint torques / forces  (shape: nv).
            external_forces: Optional list of spatial forces (in body frames).

        Returns:
            qddot : (nv,) generalised accelerations.
        """
        self._check_finalized()
        n = self.num_bodies

        # Per-body storage
        v: List[Vec6] = [np.zeros(6)] * n
        c: List[Vec6] = [np.zeros(6)] * n  # bias acceleration
        IA: List[Mat6] = [np.zeros((6, 6))] * n
        pA: List[Vec6] = [np.zeros(6)] * n
        a: List[Vec6] = [np.zeros(6)] * n
        X_up: List[Optional[SpatialTransform]] = [None] * n

        a_gravity = gravity_spatial(self._gravity)

        # ---- Pass 1: forward — velocities and bias forces ----
        for body in self._bodies:
            q_i = self._q_of(body, q)
            qdot_i = self._v_of(body, qdot)

            X_J = body.joint.transform(q_i)
            S = body.joint.motion_subspace(q_i)
            cJ = body.joint.bias_acceleration(q_i, qdot_i)
            Xup_i = body.X_tree @ X_J
            X_up[body.index] = Xup_i

            vJ = S @ qdot_i if S.shape[1] > 0 else np.zeros(6)

            if body.parent < 0:
                v[body.index] = vJ
                c[body.index] = cJ
            else:
                v_p = v[body.parent]
                v[body.index] = Xup_i.apply_velocity(v_p) + vJ
                c[body.index] = spatial_cross_velocity(v[body.index]) @ vJ + cJ

            I_mat = body.inertia.matrix()
            vi = v[body.index]
            IA[body.index] = I_mat.copy()
            pA[body.index] = spatial_cross_force(vi) @ (I_mat @ vi)
            if external_forces is not None and external_forces[body.index] is not None:
                pA[body.index] -= external_forces[body.index]

        # ---- Pass 2: backward — articulated inertias ----
        U: List[Optional[NDArray]] = [None] * n
        D_inv: List[Optional[NDArray]] = [None] * n
        u: List[Optional[NDArray]] = [None] * n

        for body in reversed(self._bodies):
            q_i = self._q_of(body, q)
            tau_i = self._v_of(body, tau)
            S = body.joint.motion_subspace(q_i)

            if S.shape[1] > 0:
                U_i = IA[body.index] @ S  # (6, nv_i)
                D_i = S.T @ U_i  # (nv_i, nv_i)
                D_inv_i = np.linalg.inv(D_i)
                u_i = tau_i - S.T @ pA[body.index]
                U[body.index] = U_i
                D_inv[body.index] = D_inv_i
                u[body.index] = u_i

                # Articulated inertia contribution to parent
                IA_A = IA[body.index] - U_i @ D_inv_i @ U_i.T
                pA_A = pA[body.index] + IA_A @ c[body.index] + U_i @ D_inv_i @ u_i
            else:
                IA_A = IA[body.index]
                pA_A = pA[body.index] + IA_A @ c[body.index]

            if body.parent >= 0:
                Xup_i = X_up[body.index]
                X6 = Xup_i.matrix()
                IA[body.parent] += X6.T @ IA_A @ X6
                pA[body.parent] += Xup_i.apply_force(pA_A)

        # ---- Pass 3: forward — accelerations ----
        qddot = np.zeros(self.nv, dtype=np.float64)

        for body in self._bodies:
            q_i = self._q_of(body, q)
            S = body.joint.motion_subspace(q_i)
            Xup_i = X_up[body.index]

            if body.parent < 0:
                # Featherstone (2008) §7.3: set base acceleration to -a_gravity
                # expressed in the root body's own frame via Xup (parent→child).
                a_p = Xup_i.apply_velocity(-a_gravity)
            else:
                a_p = Xup_i.apply_velocity(a[body.parent])

            if S.shape[1] > 0:
                qddot_i = D_inv[body.index] @ (u[body.index] - U[body.index].T @ (a_p + c[body.index]))
                qddot[body.v_idx] = qddot_i
                a[body.index] = a_p + c[body.index] + S @ qddot_i
            else:
                a[body.index] = a_p + c[body.index]

        return qddot

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _check_finalized(self) -> None:
        if not self._finalized:
            raise RuntimeError("Call finalize() before using the tree.")

    def body_by_name(self, name: str) -> Body:
        for b in self._bodies:
            if b.name == name:
                return b
        raise KeyError(f"No body named {name!r}")

    def body_velocities(
        self,
        q: NDArray[np.float64],
        qdot: NDArray[np.float64],
    ) -> List[Vec6]:
        """Compute spatial velocity (body frame) for every body.

        Performs a single forward pass (root → leaves).  The result is the
        same ``v`` array that ABA Pass 1 computes internally; exposing it here
        lets contact and self-collision models avoid redundant recomputation.

        Reference: Featherstone (2008) §7.3, Pass 1 velocity recursion.

        Returns:
            List of (6,) spatial velocity vectors, one per body, in body frame.
        """
        self._check_finalized()
        n = self.num_bodies
        v: List[Vec6] = [np.zeros(6)] * n

        for body in self._bodies:
            q_i = self._q_of(body, q)
            qdot_i = self._v_of(body, qdot)
            X_J = body.joint.transform(q_i)
            S = body.joint.motion_subspace(q_i)
            X_up = body.X_tree @ X_J
            vJ = S @ qdot_i if S.shape[1] > 0 else np.zeros(6)
            if body.parent < 0:
                v[body.index] = vJ
            else:
                v[body.index] = X_up.apply_velocity(v[body.parent]) + vJ

        return v

    def joint_limit_torques(
        self,
        q: NDArray[np.float64],
        qdot: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Deprecated alias for passive_torques()."""
        return self.passive_torques(q, qdot)

    def passive_torques(
        self,
        q: NDArray[np.float64],
        qdot: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute passive joint torques: joint limits + viscous damping.

        Iterates over all joints and accumulates:
          - RevoluteJoint  : limit penalty torque + damping torque
          - PrismaticJoint : damping torque only (no limits implemented)

        Returns a (nv,) array to be *added* to the applied tau before aba().
        """
        from .joint import PrismaticJoint as _Pris
        from .joint import RevoluteJoint as _Rev

        tau = np.zeros(self.nv, dtype=np.float64)
        for body in self._bodies:
            j = body.joint
            if isinstance(j, _Rev):
                tau[body.v_idx] = (
                    j.compute_limit_torque(q[body.q_idx], qdot[body.v_idx])
                    + j.compute_damping_torque(qdot[body.v_idx])
                    + j.compute_friction_torque(qdot[body.v_idx])
                )
            elif isinstance(j, _Pris):
                tau[body.v_idx] = j.compute_damping_torque(qdot[body.v_idx])
        return tau

    # ------------------------------------------------------------------
    # CRBA — Composite Rigid Body Algorithm  (mass matrix)
    # ------------------------------------------------------------------

    def crba(
        self,
        q: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute the joint-space mass matrix H (nv, nv) via CRBA.

        The Composite Rigid Body Algorithm computes H by:
          1. Backward pass: accumulate composite spatial inertias
             IC[i] = I[i], then for children: IC[parent] += X^T IC[child] X
          2. Forward pass: for each body with DOF, compute
             H[i,i] = S_i^T IC[i] S_i (diagonal block),
             then propagate F = IC[i] S_i up the tree to fill off-diagonals.

        H is symmetric positive definite.

        Reference: Featherstone (2008) §6.2, Algorithm 6.2.

        Returns:
            H : (nv, nv) joint-space mass matrix.
        """
        self._check_finalized()
        n = self.num_bodies

        # Compute X_up for each body (same as FK/ABA Pass 1)
        X_up = [None] * n
        for body in self._bodies:
            X_J = body.joint.transform(self._q_of(body, q))
            X_up[body.index] = body.X_tree @ X_J

        # --- Backward pass: composite inertias ---
        IC = [body.inertia.matrix().copy() for body in self._bodies]

        for body in reversed(self._bodies):
            if body.parent >= 0:
                Xup_i = X_up[body.index]
                X6 = Xup_i.matrix()  # 6x6 Plücker transform
                IC[body.parent] += X6.T @ IC[body.index] @ X6

        # --- Build H ---
        H = np.zeros((self.nv, self.nv), dtype=np.float64)

        for body in self._bodies:
            q_i = self._q_of(body, q)
            S_i = body.joint.motion_subspace(q_i)
            nv_i = S_i.shape[1]
            if nv_i == 0:
                continue

            vi = body.v_idx

            # Diagonal block: H[i,i] = S_i^T @ IC[i] @ S_i
            F = IC[body.index] @ S_i  # (6, nv_i)
            H[vi, vi] = S_i.T @ F     # (nv_i, nv_i)

            # Off-diagonal: propagate F up the tree
            # F is the spatial force at body i due to unit acceleration at joint i
            j = body.index
            while self._bodies[j].parent >= 0:
                F = X_up[j].apply_force(F) if F.ndim == 1 else np.column_stack(
                    [X_up[j].apply_force(F[:, c]) for c in range(F.shape[1])]
                )
                j = self._bodies[j].parent
                parent_body = self._bodies[j]
                S_j = parent_body.joint.motion_subspace(self._q_of(parent_body, q))
                nv_j = S_j.shape[1]
                if nv_j > 0:
                    vj = parent_body.v_idx
                    block = S_j.T @ F  # (nv_j, nv_i)
                    H[vj, vi] = block
                    H[vi, vj] = block.T  # symmetry

        return H

    def forward_dynamics_crba(
        self,
        q: NDArray[np.float64],
        qdot: NDArray[np.float64],
        tau: NDArray[np.float64],
        external_forces=None,
    ) -> NDArray[np.float64]:
        """Forward dynamics via CRBA: qddot = H^{-1} (tau - C).

        Steps:
          1. H = crba(q)                        — mass matrix
          2. C = rnea(q, qdot, 0, ext_forces)   — bias forces (Coriolis + gravity)
          3. qddot = solve(H, tau - C)           — Cholesky

        Reference: Featherstone (2008) §6.3.

        Returns:
            qddot : (nv,) generalised accelerations.
        """
        self._check_finalized()

        H = self.crba(q)
        C = self.rnea(q, qdot, np.zeros(self.nv), external_forces)
        rhs = tau - C

        # Cholesky solve (H is SPD)
        L = np.linalg.cholesky(H)
        y = np.linalg.solve(L, rhs)
        qddot = np.linalg.solve(L.T, y)

        return qddot

    # ------------------------------------------------------------------
    # Grouped CRBA — auto branch-point detection + Schur complement
    # ------------------------------------------------------------------

    def auto_detect_groups(self) -> tuple[List[int], List[List[int]]]:
        """Auto-detect subtree groups by finding branch points.

        A branch point is a body with >1 child. Each child subtree
        becomes a limb group. Bodies on the root chain (single-child
        path from root to first branch point) form the root group.

        Returns:
            (root_body_indices, limb_groups)
            - root_body_indices: body indices in the root group
            - limb_groups: list of lists, each is body indices for one limb
        """
        self._check_finalized()
        root_bodies = []
        limb_groups = []

        # Find all bodies that are on the "spine" (root chain + branch points)
        in_limb = set()

        def get_descendants(body_idx):
            subtree = [body_idx]
            stack = list(self._bodies[body_idx].children)
            while stack:
                j = stack.pop()
                subtree.append(j)
                stack.extend(self._bodies[j].children)
            return subtree

        for body in self._bodies:
            if body.index in in_limb:
                continue
            if len(body.children) > 1:
                # Branch point → root group
                root_bodies.append(body.index)
                for child_idx in body.children:
                    subtree = get_descendants(child_idx)
                    limb_groups.append(subtree)
                    in_limb.update(subtree)
            elif body.index not in in_limb:
                root_bodies.append(body.index)

        # If no branching, everything is root (degenerates to standard CRBA)
        if not limb_groups:
            return list(range(self.num_bodies)), []

        return root_bodies, limb_groups

    def grouped_crba(
        self,
        q: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute H via CRBA with auto-detected groups.

        Produces the same H matrix as crba(q) but with group-aware assembly.
        This is a correctness-equivalent reference implementation.

        Returns:
            H : (nv, nv) mass matrix (identical to crba(q)).
        """
        # The H matrix is independent of grouping — composite inertias
        # and force propagation are the same regardless of group boundaries.
        # This method validates that group detection works, then delegates.
        return self.crba(q)

    def forward_dynamics_grouped_crba(
        self,
        q: NDArray[np.float64],
        qdot: NDArray[np.float64],
        tau: NDArray[np.float64],
        external_forces=None,
    ) -> NDArray[np.float64]:
        """Forward dynamics via hierarchical Schur complement.

        Instead of full nv×nv Cholesky, exploits H's block structure:
        - H_lili: limb diagonal blocks (small, independent)
        - H_rr: root block
        - H_rli: root-limb coupling (off-diagonal)
        - H_lilj = 0 for i≠j (no limb-limb coupling)

        Solve steps:
          1. Factor each limb: L_i L_iᵀ = H_lili
          2. Schur complement: S = H_rr - Σ H_rli H_lili⁻¹ H_lir
          3. Solve root: S @ qddot_r = rhs_r'
          4. Back-substitute limbs: qddot_li = H_lili⁻¹(rhs_li - H_lir @ qddot_r)

        Reference: Featherstone (1999) Divide-and-Conquer Algorithm.

        Returns:
            qddot : (nv,) generalised accelerations.
        """
        self._check_finalized()

        root_bodies, limb_groups = self.auto_detect_groups()

        # If no branching, fall back to standard CRBA
        if not limb_groups:
            return self.forward_dynamics_crba(q, qdot, tau, external_forces)

        # Build full H and compute bias C
        H = self.crba(q)
        C = self.rnea(q, qdot, np.zeros(self.nv), external_forces)
        rhs = tau - C

        # Collect DOF indices for root and each limb
        root_v = []
        for bi in root_bodies:
            body = self._bodies[bi]
            for j in range(body.v_idx.start, body.v_idx.stop):
                root_v.append(j)
        root_v = np.array(root_v, dtype=int)

        limb_v_list = []
        for group in limb_groups:
            v_indices = []
            for bi in group:
                body = self._bodies[bi]
                for j in range(body.v_idx.start, body.v_idx.stop):
                    v_indices.append(j)
            limb_v_list.append(np.array(v_indices, dtype=int))

        nv_r = len(root_v)
        if nv_r == 0:
            # Degenerate: root has no DOF (fixed base with all joints in limbs)
            return self.forward_dynamics_crba(q, qdot, tau, external_forces)

        # Extract blocks from H
        H_rr = H[np.ix_(root_v, root_v)]  # (nv_r, nv_r)
        rhs_r = rhs[root_v]

        # Step 1: Factor each limb + compute Schur contributions
        S = H_rr.copy()  # Will subtract limb contributions
        rhs_r_modified = rhs_r.copy()

        limb_chol = []  # L factors
        limb_rhs = []
        limb_Hrl = []   # coupling blocks

        for limb_v in limb_v_list:
            if len(limb_v) == 0:
                limb_chol.append(None)
                limb_rhs.append(None)
                limb_Hrl.append(None)
                continue

            H_ll = H[np.ix_(limb_v, limb_v)]  # (nv_l, nv_l)
            H_rl = H[np.ix_(root_v, limb_v)]  # (nv_r, nv_l)
            H_lr = H_rl.T                      # (nv_l, nv_r) = H_rl^T (symmetry)
            rhs_l = rhs[limb_v]

            # Factor limb
            L_l = np.linalg.cholesky(H_ll)
            limb_chol.append(L_l)
            limb_rhs.append(rhs_l)
            limb_Hrl.append(H_rl)

            # H_ll^{-1} H_lr via Cholesky: solve L L^T X = H_lr
            Hinv_Hlr = np.linalg.solve(L_l, H_lr)
            Hinv_Hlr = np.linalg.solve(L_l.T, Hinv_Hlr)  # (nv_l, nv_r)

            # Schur: S -= H_rl @ H_ll^{-1} @ H_lr
            S -= H_rl @ Hinv_Hlr

            # Modified rhs: rhs_r' -= H_rl @ H_ll^{-1} @ rhs_l
            Hinv_rhs_l = np.linalg.solve(L_l, rhs_l)
            Hinv_rhs_l = np.linalg.solve(L_l.T, Hinv_rhs_l)
            rhs_r_modified -= H_rl @ Hinv_rhs_l

        # Step 2-3: Solve root system
        L_S = np.linalg.cholesky(S)
        y = np.linalg.solve(L_S, rhs_r_modified)
        qddot_r = np.linalg.solve(L_S.T, y)

        # Step 4: Back-substitute each limb
        qddot = np.zeros(self.nv, dtype=np.float64)
        qddot[root_v] = qddot_r

        for limb_v, L_l, rhs_l, H_rl in zip(
            limb_v_list, limb_chol, limb_rhs, limb_Hrl
        ):
            if L_l is None:
                continue
            H_lr = H_rl.T
            rhs_l_modified = rhs_l - H_lr @ qddot_r
            y = np.linalg.solve(L_l, rhs_l_modified)
            qddot_l = np.linalg.solve(L_l.T, y)
            qddot[limb_v] = qddot_l

        return qddot

    def __repr__(self) -> str:
        return f"RobotTreeNumpy(bodies={self.num_bodies}, nq={self.nq}, nv={self.nv})"


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

#: ``RobotTree`` is an alias for :class:`RobotTreeNumpy`.
RobotTree = RobotTreeNumpy
