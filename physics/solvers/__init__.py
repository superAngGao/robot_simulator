"""
Contact constraint solvers.

Public API (CPU production):
  ADMMQPSolver              — Acceleration-level QP with R-regularization (precision path)
  PGSSplitImpulseSolver     — PGS + split impulse (RL fast path)

Internal:
  PGSContactSolver          — Projected Gauss-Seidel (used internally by PGS-SI)

GPU reserved:
  JacobiPGSContactSolver    — Jacobi PGS (GPU-friendly, future Jacobi-PGS-SI)
  ADMMContactSolver         — Velocity-level ADMM (GPU-friendly, future ADMM-TC reference)

Deprecated alias:
  MuJoCoStyleSolver         — Use ADMMQPSolver instead
"""

from .admm import ADMMContactSolver
from .jacobi_pgs import JacobiPGSContactSolver
from .mujoco_qp import ADMMQPSolver, MuJoCoStyleSolver
from .pgs_solver import ContactConstraint, PGSContactSolver
from .pgs_split_impulse import PGSSplitImpulseSolver

__all__ = [
    "ContactConstraint",
    # Public API
    "ADMMQPSolver",
    "PGSSplitImpulseSolver",
    # Internal / GPU reserved
    "PGSContactSolver",
    "JacobiPGSContactSolver",
    "ADMMContactSolver",
    # Backward compat
    "MuJoCoStyleSolver",
]
