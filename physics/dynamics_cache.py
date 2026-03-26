"""
Per-step shared computation cache and observable force breakdown.

DynamicsCache holds FK results, body velocities, mass matrix, and Cholesky
factor — computed once per step, shared by all force sources, the constraint
solver, and integration.

ForceState records which forces contributed to the step, enabling
observability ("what forces acted on this body?").

MuJoCo equivalents: mjData (xpos, xmat, cvel, qM, qLD) and
qfrc_passive / qfrc_actuator / qfrc_constraint.

Reference: MuJoCo computation pipeline (mj_step1 + mj_step2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from numpy.typing import NDArray

from .spatial import SpatialTransform, Vec6

if TYPE_CHECKING:
    from .robot_tree import RobotTreeNumpy


@dataclass
class DynamicsCache:
    """Per-step shared computation cache.

    Computed once at the start of each step, reused by force sources,
    the constraint solver, and the integrator.

    Attributes:
        q, qdot, dt : Current state and time step.
        X_world     : Per-body world transforms from FK.
        body_v      : Per-body spatial velocity (body frame).
        H           : (nv, nv) mass matrix from CRBA (contact path only).
        L           : Cholesky factor of H.
        qacc_smooth : (nv,) unconstrained acceleration (set by pipeline).
    """

    q: NDArray
    qdot: NDArray
    dt: float
    X_world: List[SpatialTransform]
    body_v: List[Vec6]
    H: Optional[NDArray] = None
    L: Optional[NDArray] = None
    qacc_smooth: Optional[NDArray] = None

    @classmethod
    def from_tree(
        cls,
        tree: "RobotTreeNumpy",
        q: NDArray,
        qdot: NDArray,
        dt: float,
        *,
        compute_H: bool = False,
    ) -> "DynamicsCache":
        """Build cache from current state.

        Args:
            tree      : Articulated body tree.
            q, qdot   : Current generalized state.
            dt        : Time step [s].
            compute_H : If True, eagerly compute mass matrix and Cholesky.
                        Set True when contacts are expected (CRBA path).
        """
        X_world = tree.forward_kinematics(q)
        body_v = tree.body_velocities(q, qdot)
        cache = cls(q=q, qdot=qdot, dt=dt, X_world=X_world, body_v=body_v)
        if compute_H:
            cache.H = tree.crba(q)
            cache.L = np.linalg.cholesky(cache.H)
        return cache


@dataclass
class ForceState:
    """Observable force breakdown for one simulation step.

    All arrays are (nv,) generalized forces / accelerations.

    Attributes:
        qfrc_passive  : Joint limits + damping + friction.
        qfrc_actuator : User-supplied joint torques.
        qfrc_applied  : External forces (body wrenches, springs, etc.).
        tau_smooth    : Sum of all smooth forces (passive + actuator + applied).
        qacc_smooth   : Unconstrained acceleration (before constraints).
        qacc          : Final constrained acceleration (after solver).
    """

    qfrc_passive: NDArray
    qfrc_actuator: NDArray
    qfrc_applied: NDArray
    tau_smooth: NDArray
    qacc_smooth: NDArray
    qacc: NDArray
