"""
Contact constraint solvers.

  PGSContactSolver        — Projected Gauss-Seidel (CPU, serial)
  JacobiPGSContactSolver  — Jacobi PGS (GPU-friendly, all rows parallel)
  ADMMContactSolver       — ADMM (GPU-friendly, implicit-contact-compatible)
"""

from .admm import ADMMContactSolver
from .jacobi_pgs import JacobiPGSContactSolver
from .pgs_solver import ContactConstraint, PGSContactSolver

__all__ = [
    "ContactConstraint",
    "PGSContactSolver",
    "JacobiPGSContactSolver",
    "ADMMContactSolver",
]
