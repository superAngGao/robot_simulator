"""
PGS with Split Impulse -- decoupled velocity/position solve.

Eliminates Baumgarte feedback divergence by separating:
  Pass 1: PGS velocity constraints (v_n >= 0, no Baumgarte bias)
  Pass 2: Direct position correction (mass-weighted penetration pushout)

The two passes are decoupled: position correction cannot feed back into
the velocity solve, eliminating the positive-feedback divergence that
occurs with standard Baumgarte ERP on stiff/multi-surface contacts.

Reference:
  Bullet btSequentialImpulseConstraintSolver (m_splitImpulse mode)
  Catto (2009) -- Modeling and Solving Constraints (GDC)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..spatial import Vec6
from .pgs_solver import ContactConstraint, PGSContactSolver


class PGSSplitImpulseSolver:
    """PGS with split impulse: velocity solve + direct position correction.

    The velocity PGS uses zero Baumgarte bias, preventing the
    positive-feedback divergence on stiff or multi-surface contacts.
    Position correction is computed separately and stored in
    ``position_corrections`` for the caller to apply after integration.

    Args:
        max_iter : Maximum PGS iterations for velocity solve.
        tolerance: Convergence tolerance on impulse change.
        erp      : Error Reduction Parameter for position correction (0..1).
                    Higher = faster penetration resolution, but may overshoot.
        slop     : Allowed penetration before correction kicks in [m].
        cfm      : Constraint Force Mixing (regularization).
    """

    def __init__(
        self,
        max_iter: int = 30,
        tolerance: float = 1e-6,
        erp: float = 0.8,
        slop: float = 0.005,
        cfm: float = 1e-6,
    ) -> None:
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.erp = erp
        self.slop = slop
        self.cfm = cfm
        # Delegate velocity solve to PGS with erp=0 (no Baumgarte)
        self._vel_solver = PGSContactSolver(
            max_iter=max_iter,
            tolerance=tolerance,
            erp=0.0,
            cfm=cfm,
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
        """Solve contact constraints with split impulse.

        Returns velocity impulses (same interface as PGS). Position
        corrections are stored in ``self.position_corrections`` as
        per-body world-frame translation deltas.
        """
        num_bodies = len(body_v)
        self.position_corrections = [np.zeros(3) for _ in range(num_bodies)]

        if not contacts:
            self._vel_solver._warm_cache.clear()
            return [np.zeros(6) for _ in range(num_bodies)]

        # Pass 1: velocity PGS (erp=0, no Baumgarte bias)
        impulses = self._vel_solver.solve(
            contacts, body_v, body_X_world, inv_mass, inv_inertia, dt
        )

        # Pass 2: direct position correction (decoupled from velocity)
        for c in contacts:
            contact_slop = c.slop if c.slop is not None else self.slop
            contact_erp = c.erp if c.erp is not None else self.erp
            effective_depth = c.depth - contact_slop
            if effective_depth <= 0.0:
                continue
            correction = contact_erp * effective_depth
            # Mass-weighted distribution between bodies
            total_inv = 0.0
            for bi in (c.body_i, c.body_j):
                if bi >= 0:
                    total_inv += inv_mass[bi]
            if total_inv < 1e-10:
                continue
            for bi, sign in ((c.body_i, 1.0), (c.body_j, -1.0)):
                if bi < 0:
                    continue
                frac = inv_mass[bi] / total_inv
                self.position_corrections[bi] += sign * correction * frac * c.normal

        return impulses
