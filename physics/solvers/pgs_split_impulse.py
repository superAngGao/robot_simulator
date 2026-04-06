"""
PGS with Baumgarte velocity bias for position correction.

Position correction is integrated into the PGS velocity solve via
Baumgarte stabilization: v_bias = erp * max(depth - slop, 0) / dt
on the normal constraint row.  This propagates corrections through
the contact Jacobian J^T to all generalized coordinates, correctly
handling articulated bodies (Bullet btMultiBodyConstraintSolver style).

The ``position_corrections`` attribute is kept for API compatibility
but is always zero — all correction flows through the velocity solve.

Reference:
  Bullet btMultiBodyConstraintSolver (Baumgarte for articulated bodies)
  Todorov (2014) — MuJoCo reference acceleration formulation
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..spatial import Vec6
from .pgs_solver import ContactConstraint, PGSContactSolver


class PGSSplitImpulseSolver:
    """PGS with Baumgarte velocity bias for position correction.

    Penetration correction is folded into the velocity constraint RHS
    as ``v_bias = erp * max(depth - slop, 0) / dt`` on the normal row.
    This propagates through J^T to all generalized coordinates, correctly
    handling articulated bodies.

    The ``position_corrections`` attribute is always zero (kept for API compat).

    Args:
        max_iter : Maximum PGS iterations for velocity solve.
        tolerance: Convergence tolerance on impulse change.
        erp      : Error Reduction Parameter for Baumgarte bias (0..1).
        slop     : Allowed penetration before correction kicks in [m].
        cfm      : Constraint Force Mixing (regularization).
        max_depenetration_vel: Upper bound on the Baumgarte velocity bias
                  magnitude [m/s] (PhysX ``maxDepenetrationVelocity`` style).
                  Because position correction is folded into the velocity
                  solve, the bias also becomes the post-solve velocity — keep
                  it small (default 1 m/s) to avoid ejection from deep initial
                  penetration.
    """

    def __init__(
        self,
        max_iter: int = 30,
        tolerance: float = 1e-6,
        erp: float = 10.0,  # 1/τ where τ=0.1s; v_ref = depth/τ (MuJoCo QP style)
        slop: float = 0.001,
        cfm: float = 1e-6,
        solimp: tuple[float, ...] = (0.95, 0.99, 0.001, 0.5, 2.0),
        friction_warmstart: bool = False,
        max_depenetration_vel: float = 1.0,
    ) -> None:
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.erp = erp
        self.slop = slop
        self.cfm = cfm
        self.max_depenetration_vel = max_depenetration_vel
        # Delegate to PGS with Baumgarte ERP bias
        self._vel_solver = PGSContactSolver(
            max_iter=max_iter,
            tolerance=tolerance,
            erp=erp,
            cfm=cfm,
            slop=slop,
            solimp=solimp,
            friction_warmstart=friction_warmstart,
            max_depenetration_vel=max_depenetration_vel,
        )
        self.position_corrections: list[NDArray] = []

    def solve(
        self,
        contacts: list[ContactConstraint],
        body_v: list[Vec6],
        body_X_world: list,
        inv_mass: list[float],
        inv_inertia: list[NDArray],
        dt: float,
    ) -> list[Vec6]:
        """Solve contact constraints with Baumgarte position correction.

        Returns velocity impulses (same interface as PGS).
        ``position_corrections`` is always zero — correction flows through velocity.
        """
        num_bodies = len(body_v)
        self.position_corrections = [np.zeros(3) for _ in range(num_bodies)]

        if not contacts:
            self._vel_solver._warm_cache.clear()
            return [np.zeros(6) for _ in range(num_bodies)]

        # Single-pass PGS with Baumgarte ERP bias in the normal constraint RHS.
        # Corrections propagate through J^T to all generalized coordinates.
        impulses = self._vel_solver.solve(contacts, body_v, body_X_world, inv_mass, inv_inertia, dt)

        return impulses
