"""
Contact constraint solvers.

  PGSContactSolver     — Projected Gauss-Seidel (CPU, serial)
  (future) JacobiPGS   — Jacobi PGS (GPU-friendly, parallel)
  (future) ADMMSolver  — ADMM (GPU-friendly, implicit-compatible)
"""

from .pgs_solver import ContactConstraint, PGSContactSolver

__all__ = ["ContactConstraint", "PGSContactSolver"]
