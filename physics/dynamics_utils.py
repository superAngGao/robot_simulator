"""
Shared helper functions for dynamics computation.

Extracted from ImplicitContactStep for reuse by constraint solver adapters
and the StepPipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np
from numpy.typing import NDArray

from .spatial import SpatialTransform, Vec6

if TYPE_CHECKING:
    from .robot_tree import RobotTreeNumpy
    from .solvers.pgs_solver import ContactConstraint


def qdot_to_body_v(
    tree: "RobotTreeNumpy",
    q: NDArray,
    qdot_gen: NDArray,
    X_world: List[SpatialTransform],
) -> List[Vec6]:
    """Convert generalized velocity to per-body spatial velocity (body frame).

    Same forward recursion as tree.body_velocities() but accepts arbitrary
    qdot (e.g. predicted velocity), not just the current one.

    Reference: Featherstone (2008) §7.3, Pass 1 velocity recursion.
    """
    n = tree.num_bodies
    v: List[Vec6] = [np.zeros(6) for _ in range(n)]
    for body in tree.bodies:
        q_i = q[body.q_idx]
        qdot_i = qdot_gen[body.v_idx]
        X_J = body.joint.transform(q_i)
        S = body.joint.motion_subspace(q_i)
        X_up = body.X_tree @ X_J
        vJ = S @ qdot_i if S.shape[1] > 0 else np.zeros(6)
        if body.parent < 0:
            v[body.index] = vJ
        else:
            v[body.index] = X_up.apply_velocity(v[body.parent]) + vJ
    return v


def gather_inv_mass(
    tree: "RobotTreeNumpy",
) -> tuple[List[float], List[NDArray]]:
    """Extract per-body inverse mass and inverse inertia.

    Returns:
        (inv_mass, inv_inertia) where:
          inv_mass[i]    : scalar 1/m for body i
          inv_inertia[i] : (3,3) inverse inertia at body origin
    """
    inv_mass: List[float] = []
    inv_inertia: List[NDArray] = []
    for body in tree.bodies:
        m = body.inertia.mass
        I_com = body.inertia.inertia
        c = body.inertia.com
        # Parallel axis theorem: I_origin = I_com + m*(|c|²I - ccᵀ)
        I_origin = I_com + m * (np.dot(c, c) * np.eye(3) - np.outer(c, c))
        inv_mass.append(1.0 / m if m > 1e-10 else 0.0)
        try:
            inv_inertia.append(np.linalg.inv(I_origin))
        except np.linalg.LinAlgError:
            inv_inertia.append(np.zeros((3, 3)))
    return inv_mass, inv_inertia


def impulses_to_generalized(
    tree: "RobotTreeNumpy",
    q: NDArray,
    impulses: List[Vec6],
    contacts: Optional[List["ContactConstraint"]] = None,
) -> Optional[NDArray]:
    """Convert body-frame spatial impulses to joint-space generalized impulse.

    Uses backward pass (RNEA pass 2 pattern) to propagate body impulses
    up the kinematic chain into joint-space generalized forces.

    Args:
        tree     : Articulated body tree.
        q        : Current generalized positions.
        impulses : Per-body spatial impulses (body frame).
        contacts : Unused, kept for API compatibility.

    Returns:
        (nv,) generalized impulse, or None if all impulses are zero.
    """
    nv = tree.nv
    gen_impulse = np.zeros(nv)
    has_impulse = False

    body_imp = [np.zeros(6) for _ in range(tree.num_bodies)]
    for i, imp in enumerate(impulses):
        if np.any(imp != 0):
            body_imp[i] = imp
            has_impulse = True

    if not has_impulse:
        return None

    # Backward pass: propagate impulses to joint torques
    f = [bi.copy() for bi in body_imp]
    for body in reversed(tree.bodies):
        q_i = q[body.q_idx]
        S = body.joint.motion_subspace(q_i)
        X_J = body.joint.transform(q_i)
        X_up = body.X_tree @ X_J

        if S.shape[1] > 0:
            gen_impulse[body.v_idx] = S.T @ f[body.index]

        if body.parent >= 0:
            f[body.parent] += X_up.apply_force(f[body.index])

    return gen_impulse
