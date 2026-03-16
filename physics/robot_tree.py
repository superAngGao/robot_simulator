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

from .spatial import (
    SpatialTransform,
    SpatialInertia,
    spatial_cross_velocity,
    spatial_cross_force,
    Vec6, Mat6,
    gravity_spatial,
)
from .joint import Joint, FreeJoint


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
    name:     str
    index:    int
    joint:    Joint
    inertia:  SpatialInertia
    X_tree:   SpatialTransform
    parent:   int                  = -1
    children: List[int]            = field(default_factory=list)
    q_idx:    slice                = field(default=slice(0, 0))
    v_idx:    slice                = field(default=slice(0, 0))


# ---------------------------------------------------------------------------
# Kinematic state (scratch-pad, recomputed each step)
# ---------------------------------------------------------------------------

@dataclass
class KinematicState:
    """Per-body quantities computed during forward kinematics / ABA."""
    X_world:  SpatialTransform    # body frame expressed in world
    v:        Vec6                # spatial velocity  in body frame
    a:        Vec6                # spatial acceleration in body frame
    f:        Vec6                # spatial force (net) in body frame


# ---------------------------------------------------------------------------
# Robot tree
# ---------------------------------------------------------------------------

class RobotTree:
    """Articulated rigid body tree.

    Usage
    -----
    >>> tree = RobotTree()
    >>> root_idx = tree.add_body(Body(...))
    >>> leg_idx  = tree.add_body(Body(..., parent=root_idx))
    >>> tree.finalize()

    >>> q, qdot = tree.default_state()
    >>> poses   = tree.forward_kinematics(q)
    >>> qacc    = tree.aba(q, qdot, tau, external_forces)
    """

    def __init__(self, gravity: float = 9.81) -> None:
        self._bodies:   List[Body] = []
        self._gravity:  float      = gravity
        self._finalized: bool      = False
        self.nq: int = 0
        self.nv: int = 0

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
        self.nq = nq
        self.nv = nv
        self._finalized = True

    @property
    def bodies(self) -> List[Body]:
        return self._bodies

    @property
    def num_bodies(self) -> int:
        return len(self._bodies)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def default_state(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (q, qdot) initialised to the zero / default configuration."""
        q    = np.concatenate([b.joint.default_q()    for b in self._bodies])
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
            X_local = body.X_tree @ X_J          # parent-to-body (default+joint)
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
        q:    NDArray[np.float64],
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
            q_i    = self._q_of(body, q)
            qdot_i = self._v_of(body, qdot)
            qddot_i= self._v_of(body, qddot)

            X_J  = body.joint.transform(q_i)
            S    = body.joint.motion_subspace(q_i)
            cJ   = body.joint.bias_acceleration(q_i, qdot_i)
            X_up = body.X_tree @ X_J         # parent → body

            vJ = S @ qdot_i if S.shape[1] > 0 else np.zeros(6)

            if body.parent < 0:
                v[body.index] = vJ
                a[body.index] = X_up.apply_velocity(a_gravity) + S @ qddot_i + cJ
            else:
                v_p = v[body.parent]
                v[body.index] = X_up.apply_velocity(v_p) + vJ
                a[body.index] = (X_up.apply_velocity(a[body.parent])
                                 + S @ qddot_i + cJ
                                 + spatial_cross_velocity(v[body.index]) @ vJ)

            # Net rigid-body force = I*a + v×*(I*v) − external
            I_mat = body.inertia.matrix()
            vi    = v[body.index]
            ai    = a[body.index]
            f[body.index] = (I_mat @ ai
                             + spatial_cross_force(vi) @ (I_mat @ vi))
            if external_forces is not None and external_forces[body.index] is not None:
                f[body.index] -= external_forces[body.index]

        # --- Pass 2: backward recursion (force propagation) ---
        tau = np.zeros(self.nv, dtype=np.float64)
        for body in reversed(self._bodies):
            q_i  = self._q_of(body, q)
            S    = body.joint.motion_subspace(q_i)
            X_J  = body.joint.transform(q_i)
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
        q:    NDArray[np.float64],
        qdot: NDArray[np.float64],
        tau:  NDArray[np.float64],
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
        v:    List[Vec6]  = [np.zeros(6)] * n
        c:    List[Vec6]  = [np.zeros(6)] * n    # bias acceleration
        IA:   List[Mat6]  = [np.zeros((6,6))] * n
        pA:   List[Vec6]  = [np.zeros(6)] * n
        a:    List[Vec6]  = [np.zeros(6)] * n
        X_up: List[Optional[SpatialTransform]] = [None] * n

        a_gravity = gravity_spatial(self._gravity)

        # ---- Pass 1: forward — velocities and bias forces ----
        for body in self._bodies:
            q_i    = self._q_of(body, q)
            qdot_i = self._v_of(body, qdot)

            X_J   = body.joint.transform(q_i)
            S     = body.joint.motion_subspace(q_i)
            cJ    = body.joint.bias_acceleration(q_i, qdot_i)
            Xup_i = body.X_tree @ X_J
            X_up[body.index] = Xup_i

            vJ = S @ qdot_i if S.shape[1] > 0 else np.zeros(6)

            if body.parent < 0:
                v[body.index] = vJ
                c[body.index] = cJ
            else:
                v_p = v[body.parent]
                v[body.index] = Xup_i.apply_velocity(v_p) + vJ
                c[body.index] = (spatial_cross_velocity(v[body.index]) @ vJ + cJ)

            I_mat = body.inertia.matrix()
            vi    = v[body.index]
            IA[body.index] = I_mat.copy()
            pA[body.index] = spatial_cross_force(vi) @ (I_mat @ vi)
            if external_forces is not None and external_forces[body.index] is not None:
                pA[body.index] -= external_forces[body.index]

        # ---- Pass 2: backward — articulated inertias ----
        U:     List[Optional[NDArray]] = [None] * n
        D_inv: List[Optional[NDArray]] = [None] * n
        u:     List[Optional[NDArray]] = [None] * n

        for body in reversed(self._bodies):
            q_i    = self._q_of(body, q)
            tau_i  = self._v_of(body, tau)
            S      = body.joint.motion_subspace(q_i)

            if S.shape[1] > 0:
                U_i     = IA[body.index] @ S                        # (6, nv_i)
                D_i     = S.T @ U_i                                  # (nv_i, nv_i)
                D_inv_i = np.linalg.inv(D_i)
                u_i     = tau_i - S.T @ pA[body.index]
                U[body.index]     = U_i
                D_inv[body.index] = D_inv_i
                u[body.index]     = u_i

                # Articulated inertia contribution to parent
                IA_A = IA[body.index] - U_i @ D_inv_i @ U_i.T
                pA_A = pA[body.index] + IA_A @ c[body.index] + U_i @ D_inv_i @ u_i
            else:
                IA_A = IA[body.index]
                pA_A = pA[body.index] + IA_A @ c[body.index]

            if body.parent >= 0:
                Xup_i = X_up[body.index]
                X6    = Xup_i.matrix()
                IA[body.parent] += X6.T @ IA_A @ X6
                pA[body.parent] += Xup_i.apply_force(pA_A)

        # ---- Pass 3: forward — accelerations ----
        qddot = np.zeros(self.nv, dtype=np.float64)

        for body in self._bodies:
            q_i    = self._q_of(body, q)
            S      = body.joint.motion_subspace(q_i)
            Xup_i  = X_up[body.index]

            if body.parent < 0:
                # Featherstone (2008) §7.3: set base acceleration to -a_gravity
                # so that gravity is implicitly included in all body accelerations.
                a_p = -a_gravity
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

    def joint_limit_torques(
        self,
        q:    NDArray[np.float64],
        qdot: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute restoring torques for all joints that have active limits.

        Returns a (nv,) array that should be *added* to the applied tau
        before calling aba(), so the limits are enforced every time-step.
        """
        from .joint import RevoluteJoint as _Rev
        tau_lim = np.zeros(self.nv, dtype=np.float64)
        for body in self._bodies:
            if isinstance(body.joint, _Rev):
                tau_lim[body.v_idx] = body.joint.compute_limit_torque(
                    q[body.q_idx], qdot[body.v_idx]
                )
        return tau_lim

    def __repr__(self) -> str:
        return (f"RobotTree(bodies={self.num_bodies}, "
                f"nq={self.nq}, nv={self.nv})")
