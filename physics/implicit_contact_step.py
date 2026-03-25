"""
Contact-aware integrator: solver output directly drives integration.

Eliminates the impulse -> force -> ABA round-trip by using the contact
solver's forces directly in the equation of motion.

Flow:
    1. a_u = ABA(q, qdot, tau)              — unconstrained acceleration
    2. f, J = solver.solve(contacts, ...)    — contact forces (acceleration-level)
    3. tau_c = J^T @ f                       — contact forces in joint space
    4. a_c = ABA(q, qdot, tau + tau_c)       — constrained acceleration
    5. qdot_new = qdot + dt * a_c
    6. q_new = integrate_q(q, qdot_new, dt)

Two ABA calls, zero information loss.

Reference:
    MuJoCo computation pipeline (mj_step1 + mj_step2).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from .integrator import Integrator
from .solvers.mujoco_qp import MuJoCoStyleSolver
from .solvers.pgs_solver import ContactConstraint


class ImplicitContactStep(Integrator):
    """Integrator with direct contact force application (no round-trip).

    Args:
        dt     : Time step [s].
        solver : MuJoCoStyleSolver instance.
    """

    def __init__(self, dt: float, solver: MuJoCoStyleSolver) -> None:
        super().__init__(dt)
        self.solver = solver

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
        """Advance one step with coupled contact resolution.

        Args:
            tree       : RobotTreeNumpy.
            q, qdot    : Current state.
            tau        : Applied joint torques (nv,) — should include passive_torques.
            ext_forces : Optional body-frame spatial forces (for non-contact forces).
            contacts   : Contact constraints from collision pipeline.

        Returns:
            (q_new, qdot_new)
        """
        # 1. Unconstrained acceleration
        a_u = tree.aba(q, qdot, tau, ext_forces)

        if not contacts:
            # No contacts: simple semi-implicit Euler
            qdot_new = qdot + self.dt * a_u
            q_new = self._integrate_q(tree, q, qdot_new, self.dt)
            return q_new, qdot_new

        # 2. Solve contact QP (acceleration-level, joint-space)
        f, J = self.solver.solve(contacts, tree, q, qdot, a_u, self.dt)

        if f.size == 0:
            qdot_new = qdot + self.dt * a_u
            q_new = self._integrate_q(tree, q, qdot_new, self.dt)
            return q_new, qdot_new

        # 3. Contact generalized forces: tau_contact = J^T @ f
        tau_contact = J.T @ f

        # 4. Constrained acceleration via ABA
        #    a_c = ABA(q, qdot, tau + tau_contact, ext_forces)
        #    This is exact: a_c = H^{-1}(tau + tau_c - C) = a_u + H^{-1} J^T f
        a_c = tree.aba(q, qdot, tau + tau_contact, ext_forces)

        # 5-6. Semi-implicit Euler integration
        qdot_new = qdot + self.dt * a_c
        q_new = self._integrate_q(tree, q, qdot_new, self.dt)

        if not np.all(np.isfinite(q_new)):
            raise RuntimeError(
                "ImplicitContactStep: state diverged (NaN/Inf)."
            )
        return q_new, qdot_new
