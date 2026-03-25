"""
Contact constraint solvers.

  PGSContactSolver          — Projected Gauss-Seidel (CPU, serial)
  PGSSplitImpulseSolver     — PGS + split impulse (no Baumgarte divergence)
  JacobiPGSContactSolver    — Jacobi PGS (GPU-friendly, all rows parallel)
  ADMMContactSolver         — ADMM (GPU-friendly, supports compliant contact + adaptive rho)
  MuJoCoStyleSolver         — Acceleration-level QP with R-regularization (MuJoCo-compatible)
"""

from .admm import ADMMContactSolver
from .jacobi_pgs import JacobiPGSContactSolver
from .mujoco_qp import MuJoCoStyleSolver
from .pgs_solver import ContactConstraint, PGSContactSolver
from .pgs_split_impulse import PGSSplitImpulseSolver

__all__ = [
    "ContactConstraint",
    "PGSContactSolver",
    "PGSSplitImpulseSolver",
    "JacobiPGSContactSolver",
    "ADMMContactSolver",
    "MuJoCoStyleSolver",
]
