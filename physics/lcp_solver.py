# Backward-compatibility re-export. Import from physics.solvers instead.
from .solvers.pgs_solver import (  # noqa: F401
    CONDIM_VALID,
    ContactConstraint,
    PGSContactSolver,
    _build_contact_frame,
    _compute_angular_jacobian_row,
    _compute_linear_jacobian_row,
    _pgs_box_row,
    _row_to_contact,
)

__all__ = ["ContactConstraint", "PGSContactSolver", "CONDIM_VALID"]
