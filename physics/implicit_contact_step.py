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

from .constraint_solvers import VelocityLevelAdapter
from .dynamics_cache import DynamicsCache
from .dynamics_utils import gather_inv_mass, impulses_to_generalized, qdot_to_body_v
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
        from .solvers.admm_qp import ADMMQPSolver

        self._accel_level = isinstance(solver, ADMMQPSolver)

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
        # Share the same articulated solve/apply mobility as CpuEngine.
        # Reuse a_u so external forces and the legacy force contract are preserved.
        cache = DynamicsCache.from_tree(tree, q, qdot, self.dt, compute_H=True)
        cache.qacc_smooth = a_u
        qacc = VelocityLevelAdapter(self.solver).solve(tree, cache, tau, contacts)
        qdot_new = qdot + self.dt * qacc

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
        """Compatibility delegate to shared dynamics utilities."""
        return qdot_to_body_v(tree, q, qdot_gen, X_world)

    @staticmethod
    def _gather_mass(tree):
        """Compatibility delegate to shared dynamics utilities."""
        return gather_inv_mass(tree)

    @staticmethod
    def _impulses_to_generalized(tree, q, qdot, X_world, impulses, contacts):
        """Compatibility delegate to shared dynamics utilities."""
        return impulses_to_generalized(tree, q, impulses, contacts)

    @staticmethod
    def _check_finite(q):
        if not np.all(np.isfinite(q)):
            raise RuntimeError("ImplicitContactStep: state diverged (NaN/Inf).")
