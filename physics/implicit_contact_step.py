"""
Contact-aware integrator: solver output directly drives integration.

Supports both acceleration-level (MuJoCoStyle) and velocity-level (PGS)
solvers, eliminating the impulse -> force -> ABA round-trip for ALL solvers.

Acceleration-level flow (MuJoCoStyleSolver):
    1. a_u = ABA(q, qdot, tau)
    2. f, J = solver.solve(contacts, tree, q, qdot, a_u, dt)
    3. a_c = ABA(q, qdot, tau + J^T @ f)
    4. qdot_new = qdot + dt * a_c
    5. q_new = integrate_q(q, qdot_new, dt)

Velocity-level flow (PGS / PGS-SI / Jacobi-PGS):
    1. a_u = ABA(q, qdot, tau)
    2. v_predicted = qdot + dt * a_u             (includes gravity!)
    3. body_v_pred = FK_velocities(v_predicted)
    4. impulse = solver.solve(contacts, body_v_pred, ...)
    5. qdot_new = v_predicted + H^{-1} J^T impulse_generalized
    6. q_new = integrate_q(q, qdot_new, dt)

Both paths: one ABA call for a_u, no force round-trip.

The velocity-level flow fixes the "gravity double-counting" bug in PGS-SI:
the solver sees v_predicted (already including gravity), so its impulse
directly corrects the post-gravity velocity. Result: vz = 0 at rest.

Reference:
    MuJoCo computation pipeline (mj_step1 + mj_step2).
    Bullet btSequentialImpulseConstraintSolver (operates on predicted velocity).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from .integrator import Integrator
from .solvers.pgs_solver import ContactConstraint


class ImplicitContactStep(Integrator):
    """Integrator with direct contact resolution (no impulse->force round-trip).

    Automatically selects acceleration-level or velocity-level flow based
    on solver type.

    Args:
        dt     : Time step [s].
        solver : Any contact solver. MuJoCoStyleSolver uses acceleration-level
                 flow; PGS/PGS-SI/Jacobi/ADMM use velocity-level flow.
    """

    def __init__(self, dt: float, solver) -> None:
        super().__init__(dt)
        self.solver = solver
        from .solvers.mujoco_qp import MuJoCoStyleSolver
        self._accel_level = isinstance(solver, MuJoCoStyleSolver)

    def step(
        self,
        tree,
        q: NDArray[np.float64],
        qdot: NDArray[np.float64],
        tau: NDArray[np.float64],
        ext_forces: Optional[List] = None,
        *,
        contacts: list[ContactConstraint] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Advance one step with coupled contact resolution."""
        # 1. Unconstrained acceleration (shared by both paths)
        a_u = tree.aba(q, qdot, tau, ext_forces)

        if not contacts:
            qdot_new = qdot + self.dt * a_u
            q_new = self._integrate_q(tree, q, qdot_new, self.dt)
            return q_new, qdot_new

        if self._accel_level:
            return self._step_accel(tree, q, qdot, tau, ext_forces, a_u, contacts)
        else:
            return self._step_velocity(tree, q, qdot, tau, ext_forces, a_u, contacts)

    # ------------------------------------------------------------------
    # Acceleration-level path (MuJoCoStyleSolver)
    # ------------------------------------------------------------------

    def _step_accel(self, tree, q, qdot, tau, ext_forces, a_u, contacts):
        """MuJoCo-style: solve QP at acceleration level, apply via ABA."""
        f, J = self.solver.solve(contacts, tree, q, qdot, a_u, self.dt)

        if f.size == 0:
            qdot_new = qdot + self.dt * a_u
            q_new = self._integrate_q(tree, q, qdot_new, self.dt)
            return q_new, qdot_new

        # a_c = ABA(q, qdot, tau + J^T @ f)  — exact, no approximation
        tau_contact = J.T @ f
        a_c = tree.aba(q, qdot, tau + tau_contact, ext_forces)

        qdot_new = qdot + self.dt * a_c
        q_new = self._integrate_q(tree, q, qdot_new, self.dt)
        self._check_finite(q_new)
        return q_new, qdot_new

    # ------------------------------------------------------------------
    # Velocity-level path (PGS / PGS-SI / Jacobi-PGS / ADMM)
    # ------------------------------------------------------------------

    def _step_velocity(self, tree, q, qdot, tau, ext_forces, a_u, contacts):
        """Predicted-velocity flow: PGS solves on v_predicted, not v_current.

        Fixes the gravity double-counting bug: the solver sees the velocity
        AFTER gravity has been applied, so its impulse directly corrects
        the post-gravity velocity.
        """
        # 2. Predicted velocity (includes gravity + all non-contact forces)
        v_predicted = qdot + self.dt * a_u

        # 3. Convert to body-frame predicted velocities for the solver
        #    We need body_v and body_X at the CURRENT configuration
        #    (contacts were detected at current q)
        X_world = tree.forward_kinematics(q)
        body_v_pred = self._qdot_to_body_v(tree, q, v_predicted, X_world)

        # 4. Gather mass properties for the solver
        inv_mass, inv_inertia = self._gather_mass(tree)

        # 5. Solve on predicted velocities
        impulses = self.solver.solve(
            contacts, body_v_pred, X_world, inv_mass, inv_inertia, dt=self.dt
        )

        # 6. Convert body-frame impulses to joint-space velocity correction
        #    Δqdot = H^{-1} @ generalized_impulse
        #    where generalized_impulse = Σ J_body_i^T @ impulse_i
        qdot_new = v_predicted.copy()
        gen_impulse = self._impulses_to_generalized(
            tree, q, qdot, X_world, impulses, contacts
        )
        if gen_impulse is not None:
            # H^{-1} @ gen_impulse via ABA trick:
            # ABA(q, 0, gen_impulse/dt, 0) * dt = H^{-1} @ gen_impulse
            # But simpler: just use CRBA + solve
            H = tree.crba(q)
            try:
                L = np.linalg.cholesky(H)
                dqdot = np.linalg.solve(L.T, np.linalg.solve(L, gen_impulse))
            except np.linalg.LinAlgError:
                dqdot = np.linalg.lstsq(H, gen_impulse, rcond=None)[0]
            qdot_new = v_predicted + dqdot

        # Apply position corrections (PGS-SI only)
        q_new = self._integrate_q(tree, q, qdot_new, self.dt)
        if hasattr(self.solver, "position_corrections"):
            for i in range(tree.num_bodies):
                pc = self.solver.position_corrections[i]
                if pc[0] == 0.0 and pc[1] == 0.0 and pc[2] == 0.0:
                    continue
                body = tree.bodies[i]
                if body.joint.nq == 7:  # FreeJoint
                    qs = body.q_idx.start if isinstance(body.q_idx, slice) else body.q_idx[0]
                    q_new[qs + 4 : qs + 7] += pc

        self._check_finite(q_new)
        return q_new, qdot_new

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _qdot_to_body_v(tree, q, qdot_gen, X_world):
        """Convert generalized velocity to per-body spatial velocity (body frame).

        Same forward recursion as tree.body_velocities() but with arbitrary qdot.
        """
        from .spatial import SpatialTransform
        n = tree.num_bodies
        v = [np.zeros(6) for _ in range(n)]
        for body in tree.bodies:
            q_i = q[body.q_idx]
            qdot_i = qdot_gen[body.v_idx]
            X_J = body.joint.transform(q_i)
            S = body.joint.motion_subspace(q_i)
            X_up = body.X_tree @ X_J
            vJ = S @ qdot_i if S.shape[1] > 0 else np.zeros(6)
            if body.parent < 0:
                v[body.index] = vJ
            else:
                v[body.index] = X_up.apply_velocity(v[body.parent]) + vJ
        return v

    @staticmethod
    def _gather_mass(tree):
        """Extract per-body inverse mass and inertia."""
        inv_mass = []
        inv_inertia = []
        for body in tree.bodies:
            m = body.inertia.mass
            I_com = body.inertia.inertia
            c = body.inertia.com
            I_origin = I_com + m * (np.dot(c, c) * np.eye(3) - np.outer(c, c))
            inv_mass.append(1.0 / m if m > 1e-10 else 0.0)
            try:
                inv_inertia.append(np.linalg.inv(I_origin))
            except np.linalg.LinAlgError:
                inv_inertia.append(np.zeros((3, 3)))
        return inv_mass, inv_inertia

    @staticmethod
    def _impulses_to_generalized(tree, q, qdot, X_world, impulses, contacts):
        """Convert body-frame spatial impulses to joint-space generalized impulse.

        Uses the contact Jacobian: gen_impulse = Σ J_contact^T @ impulse_contact_dir
        For efficiency, we compute J^T @ impulse for each contact body using
        the kinematic chain (same as RNEA backward pass).
        """
        nv = tree.nv
        gen_impulse = np.zeros(nv)
        has_impulse = False

        # Accumulate body-frame impulses
        body_imp = [np.zeros(6) for _ in range(tree.num_bodies)]
        for i, imp in enumerate(impulses):
            if np.any(imp != 0):
                body_imp[i] = imp
                has_impulse = True

        if not has_impulse:
            return None

        # Backward pass: propagate impulses to joint torques (like RNEA pass 2)
        f = [bi.copy() for bi in body_imp]
        for body in reversed(tree.bodies):
            q_i = q[body.q_idx]
            S = body.joint.motion_subspace(q_i)
            X_J = body.joint.transform(q_i)
            X_up = body.X_tree @ X_J

            if S.shape[1] > 0:
                gen_impulse[body.v_idx] = S.T @ f[body.index]

            if body.parent >= 0:
                f[body.parent] += X_up.apply_force(f[body.index])

        return gen_impulse

    @staticmethod
    def _check_finite(q):
        if not np.all(np.isfinite(q)):
            raise RuntimeError("ImplicitContactStep: state diverged (NaN/Inf).")
