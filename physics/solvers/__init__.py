"""
Contact constraint solvers.

Public API (CPU production):
  ADMMQPSolver              — Acceleration-level QP with R-regularization (precision path)
  PGSSplitImpulseSolver     — PGS + split impulse (RL fast path)

Internal:
  PGSContactSolver          — Projected Gauss-Seidel (used internally by PGS-SI)

GPU solvers are implemented directly in Warp kernels:
  - Jacobi PGS + split impulse: physics/backends/warp/solver_kernels.py
  - ADMM (solref/solimp):       physics/backends/warp/admm_kernels.py

Deprecated alias:
  MuJoCoStyleSolver         — Use ADMMQPSolver instead
"""

from .admm_qp import ADMMQPSolver, MuJoCoStyleSolver
from .pgs_solver import ContactConstraint, PGSContactSolver
from .pgs_split_impulse import PGSSplitImpulseSolver

__all__ = [
    "ContactConstraint",
    # Public API
    "ADMMQPSolver",
    "PGSSplitImpulseSolver",
    # Internal
    "PGSContactSolver",
    # Backward compat
    "MuJoCoStyleSolver",
]
