"""
Constraint solver interface for the rigid body dynamics pipeline.

Stage 2 of the two-stage pipeline: all solvers take qacc_smooth + contacts
and output qacc (nv,), regardless of internal formulation (acceleration-level
QP, velocity-level impulse, or penalty).

Reference: MuJoCo mj_step2 (constraint phase).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

from numpy.typing import NDArray

if TYPE_CHECKING:
    from .dynamics_cache import DynamicsCache
    from .robot_tree import RobotTreeNumpy
    from .solvers.pgs_solver import ContactConstraint


class ConstraintSolver(ABC):
    """Abstract base for constraint solvers.

    All implementations receive the shared DynamicsCache (with FK, body_v,
    H, L pre-computed) and return constrained generalized acceleration.
    """

    @abstractmethod
    def solve(
        self,
        tree: "RobotTreeNumpy",
        cache: "DynamicsCache",
        tau_smooth: NDArray,
        contacts: List["ContactConstraint"],
    ) -> NDArray:
        """Solve constraints and return constrained acceleration.

        Args:
            tree       : Articulated body tree.
            cache      : DynamicsCache (has X_world, body_v, H, L, qacc_smooth).
            tau_smooth : (nv,) total smooth forces.
            contacts   : Contact constraints from collision pipeline.

        Returns:
            qacc : (nv,) constrained generalized acceleration.
        """


class NullConstraintSolver(ConstraintSolver):
    """No-op solver: returns qacc_smooth unchanged.

    Used when there are no contacts in the current step (fast path).
    """

    def solve(self, tree, cache, tau_smooth, contacts):
        return cache.qacc_smooth
