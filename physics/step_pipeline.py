"""
Two-stage rigid body dynamics pipeline.

  Stage 1 (smooth):     ForceSource[] → tau_smooth → qacc_smooth
  Stage 2 (constraint): ConstraintSolver(qacc_smooth, contacts) → qacc
  Integration:          qdot_new = qdot + dt*qacc; q_new = tree.integrate_q(...)

Integration is inline (semi-implicit Euler), not a separate abstraction.
Different physics subsystems (future: soft body, fluid) each have their
own pipeline with their own integration scheme.

Reference: MuJoCo mj_step1 (smooth) + mj_step2 (constraint).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np
from numpy.typing import NDArray

from .constraint_solver import NullConstraintSolver
from .dynamics_cache import DynamicsCache, ForceState
from .force_source import ForceSource, PassiveForceSource

if TYPE_CHECKING:
    from .constraint_solver import ConstraintSolver
    from .robot_tree import RobotTreeNumpy
    from .solvers.pgs_solver import ContactConstraint


class StepPipeline:
    """Rigid body dynamics pipeline (MuJoCo mj_step equivalent).

    Args:
        dt                : Time step [s].
        force_sources     : List of ForceSource (default: [PassiveForceSource()]).
        constraint_solver : ConstraintSolver (default: NullConstraintSolver).
    """

    def __init__(
        self,
        dt: float,
        force_sources: Optional[List[ForceSource]] = None,
        constraint_solver: Optional["ConstraintSolver"] = None,
    ) -> None:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        self.dt = dt
        self.force_sources = force_sources if force_sources is not None else []
        self.constraint_solver = constraint_solver or NullConstraintSolver()
        self._last_force_state: Optional[ForceState] = None

    def step(
        self,
        tree: "RobotTreeNumpy",
        q: NDArray,
        qdot: NDArray,
        tau: NDArray,
        contacts: Optional[List["ContactConstraint"]] = None,
        *,
        cache: Optional[DynamicsCache] = None,
    ) -> tuple[NDArray, NDArray]:
        """Execute one step of the two-stage pipeline.

        Args:
            tree     : Articulated body tree.
            q, qdot  : Current generalized state.
            tau      : User-supplied actuator torques (nv,).
            contacts : Contact constraints from collision pipeline.
            cache    : Pre-built DynamicsCache (optional). If provided, FK and
                       body_v are reused. If None, built internally.

        Returns:
            (q_new, qdot_new) after one step.
        """
        contacts = contacts or []
        has_contacts = len(contacts) > 0

        # ── Build or reuse DynamicsCache ──
        if cache is None:
            cache = DynamicsCache.from_tree(tree, q, qdot, self.dt, compute_H=has_contacts)
        elif has_contacts and cache.H is None:
            # Cache was built without H; compute now
            cache.H = tree.crba(q)
            cache.L = np.linalg.cholesky(cache.H)

        nv = tree.nv

        # ── Stage 1: Smooth forces ──
        qfrc_passive = np.zeros(nv)
        qfrc_applied = np.zeros(nv)

        for src in self.force_sources:
            result = src.compute(tree, cache)
            if isinstance(src, PassiveForceSource):
                qfrc_passive = result
            else:
                qfrc_applied = qfrc_applied + result

        tau_smooth = tau + qfrc_passive + qfrc_applied

        # ── Compute qacc_smooth ──
        if not has_contacts:
            # Fast path: O(n) ABA, no mass matrix needed
            qacc_smooth = tree.aba(q, qdot, tau_smooth)
        else:
            # CRBA path: qacc_smooth = L^{-T} L^{-1} (tau_smooth - C)
            C = tree.rnea(q, qdot, np.zeros(nv))
            rhs = tau_smooth - C
            qacc_smooth = np.linalg.solve(cache.L.T, np.linalg.solve(cache.L, rhs))

        cache.qacc_smooth = qacc_smooth

        # ── Stage 2: Constraint solve ──
        if has_contacts:
            qacc = self.constraint_solver.solve(tree, cache, tau_smooth, contacts)
        else:
            qacc = qacc_smooth

        # ── Integration (inline semi-implicit Euler) ──
        qdot_new = qdot + self.dt * qacc
        q_new = tree.integrate_q(q, qdot_new, self.dt)

        # ── Position corrections (PGS-SI side-channel) ──
        pos_corr = getattr(self.constraint_solver, "position_corrections", None)
        if pos_corr is not None:
            for i in range(tree.num_bodies):
                pc = pos_corr[i] if i < len(pos_corr) else None
                if pc is None:
                    continue
                if pc[0] == 0.0 and pc[1] == 0.0 and pc[2] == 0.0:
                    continue
                body = tree.bodies[i]
                if body.joint.nq == 7:  # FreeJoint
                    qs = body.q_idx.start if isinstance(body.q_idx, slice) else body.q_idx[0]
                    q_new[qs + 4 : qs + 7] += pc

        # ── Finite check ──
        if not np.all(np.isfinite(q_new)):
            raise RuntimeError(
                "StepPipeline: state diverged (NaN/Inf). Try reducing dt or contact stiffness."
            )

        # ── Record ForceState ──
        self._last_force_state = ForceState(
            qfrc_passive=qfrc_passive,
            qfrc_actuator=tau.copy(),
            qfrc_applied=qfrc_applied,
            tau_smooth=tau_smooth,
            qacc_smooth=qacc_smooth,
            qacc=qacc,
        )

        return q_new, qdot_new

    @property
    def last_force_state(self) -> Optional[ForceState]:
        """Force breakdown from the most recent step() call."""
        return self._last_force_state
