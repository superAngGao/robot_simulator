"""
Constraint solver adapters wrapping existing solvers into the ConstraintSolver
interface (qacc_smooth + contacts → qacc).

Two adapters for two solver families:
  AccelLevelAdapter   — wraps ADMMQPSolver (acceleration-level QP)
  VelocityLevelAdapter — wraps PGS / PGS-SI / Jacobi / ADMM (velocity-level)

These adapters translate between the legacy solver signatures and the new
unified ConstraintSolver.solve(tree, cache, tau_smooth, contacts) → qacc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .constraint_solver import ConstraintSolver
from .dynamics_utils import gather_inv_mass, impulses_to_generalized, qdot_to_body_v

if TYPE_CHECKING:
    pass


class AccelLevelAdapter(ConstraintSolver):
    """Wraps ADMMQPSolver for the new pipeline.

    Flow: inner.solve(contacts, tree, q, qdot, qacc_smooth, dt) → (f, J)
          qacc = ABA(q, qdot, tau_smooth + J^T @ f)
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    def solve(self, tree, cache, tau_smooth, contacts):
        if not contacts:
            return cache.qacc_smooth

        f, J = self._inner.solve(contacts, tree, cache.q, cache.qdot, cache.qacc_smooth, cache.dt)

        if f.size == 0:
            return cache.qacc_smooth

        tau_contact = J.T @ f
        return tree.aba(cache.q, cache.qdot, tau_smooth + tau_contact)


class VelocityLevelAdapter(ConstraintSolver):
    """Wraps PGS / PGS-SI / Jacobi / ADMM for the new pipeline.

    Flow: v_predicted = qdot + dt * qacc_smooth
          impulses = inner.solve(contacts, body_v_pred, X_world, ...)
          gen_impulse = J^T @ impulses (backward pass)
          qacc = qacc_smooth + (1/dt) * H^{-1} @ gen_impulse
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    def solve(self, tree, cache, tau_smooth, contacts):
        if not contacts:
            return cache.qacc_smooth

        # Predicted generalized velocity (includes gravity + smooth forces)
        v_predicted = cache.qdot + cache.dt * cache.qacc_smooth

        # Convert to body-frame spatial velocities for the solver
        body_v_pred = qdot_to_body_v(tree, cache.q, v_predicted, cache.X_world)

        # Gather mass properties
        inv_mass, inv_inertia = gather_inv_mass(tree)

        # Solve on predicted velocities
        impulses = self._inner.solve(
            contacts,
            body_v_pred,
            cache.X_world,
            inv_mass,
            inv_inertia,
            dt=cache.dt,
        )

        # Convert body-frame impulses to generalized
        gen_impulse = impulses_to_generalized(tree, cache.q, impulses)

        if gen_impulse is None:
            return cache.qacc_smooth

        # qacc = qacc_smooth + (1/dt) * H^{-1} @ gen_impulse
        try:
            dqdot = np.linalg.solve(cache.L.T, np.linalg.solve(cache.L, gen_impulse))
        except (np.linalg.LinAlgError, AttributeError):
            # Fallback if L is not available
            dqdot = np.linalg.lstsq(cache.H, gen_impulse, rcond=None)[0]

        return cache.qacc_smooth + dqdot / cache.dt

    @property
    def position_corrections(self):
        """Forward PGS-SI position corrections (side-channel)."""
        if hasattr(self._inner, "position_corrections"):
            return self._inner.position_corrections
        return None


def wrap_solver(solver) -> ConstraintSolver:
    """Auto-wrap a legacy solver into the ConstraintSolver interface."""
    from .solvers.admm_qp import ADMMQPSolver

    if isinstance(solver, ConstraintSolver):
        return solver
    if isinstance(solver, ADMMQPSolver):
        return AccelLevelAdapter(solver)
    return VelocityLevelAdapter(solver)
