"""
Numerical integrators for rigid body simulation.

Integrating the equations of motion:
    q̈ = f(t, q, q̇)   (from ABA)
    q̇  = dq/dt
    q  = ∫ q̇ dt

Integrators provided
--------------------
  SemiImplicitEuler : Semi-implicit (symplectic) Euler — first order,
                      energy-conserving, recommended for contact-rich sims.
  RK4               : Classical 4th-order Runge-Kutta — more accurate but
                      4× more expensive per step. Use for validation.

Quaternion handling
-------------------
The FreeJoint uses a quaternion for orientation. Naive Euler integration
drifts off the unit sphere, so we use the dedicated `FreeJoint.integrate_q`
method when a floating base is present.

Usage
-----
    integrator = SemiImplicitEuler(dt=0.001)
    q, qdot = integrator.step(tree, q, qdot, tau, contact_forces)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import numpy as np
from numpy.typing import NDArray

from .joint import FreeJoint
from .robot_tree import RobotTree

# ---------------------------------------------------------------------------
# Type alias for the dynamics callable
# ---------------------------------------------------------------------------

DynamicsFn = Callable[
    [NDArray[np.float64], NDArray[np.float64]],  # (q, qdot) → qddot
    NDArray[np.float64],
]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Integrator(ABC):
    """Abstract base class for numerical integrators."""

    def __init__(self, dt: float) -> None:
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}.")
        self.dt = dt

    @abstractmethod
    def step(
        self,
        tree: RobotTree,
        q: NDArray[np.float64],
        qdot: NDArray[np.float64],
        tau: NDArray[np.float64],
        ext_forces: Optional[List] = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Advance the state by one time step dt.

        Args:
            tree       : The articulated body tree.
            q          : Generalised positions  (nq,).
            qdot       : Generalised velocities (nv,).
            tau        : Applied joint torques   (nv,).
            ext_forces : Optional spatial external forces per body.

        Returns:
            (q_new, qdot_new) after one step.
        """

    def _integrate_q(
        self,
        tree: RobotTree,
        q: NDArray[np.float64],
        qdot: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        """Integrate positions, handling quaternion joints correctly."""
        q_new = q.copy()
        for body in tree.bodies:
            j = body.joint
            if j.nq == 0:
                continue
            if isinstance(j, FreeJoint):
                q_new[body.q_idx] = j.integrate_q(q[body.q_idx], qdot[body.v_idx], dt)
            else:
                q_new[body.q_idx] = q[body.q_idx] + qdot[body.v_idx] * dt
        return q_new


# ---------------------------------------------------------------------------
# Semi-implicit (symplectic) Euler
# ---------------------------------------------------------------------------


class SemiImplicitEuler(Integrator):
    """Semi-implicit Euler (symplectic Euler) integrator.

    Update rule:
        q̈       = ABA(q_n, q̇_n, τ, f_ext)
        q̇_{n+1} = q̇_n + dt * q̈
        q_{n+1}  = q_n + dt * q̇_{n+1}    ← uses updated velocity

    This ordering preserves a discrete energy invariant, making it more
    stable than explicit Euler for oscillatory / contact dynamics without
    requiring smaller time steps.
    """

    def step(
        self,
        tree: RobotTree,
        q: NDArray[np.float64],
        qdot: NDArray[np.float64],
        tau: NDArray[np.float64],
        ext_forces: Optional[List] = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        qddot = tree.aba(q, qdot, tau, ext_forces)
        qdot_new = qdot + self.dt * qddot
        q_new = self._integrate_q(tree, q, qdot_new, self.dt)  # use updated qdot
        if not np.all(np.isfinite(q_new)):
            raise RuntimeError(
                "SemiImplicitEuler: state diverged (NaN/Inf). Try reducing dt or contact stiffness k_normal."
            )
        return q_new, qdot_new


# ---------------------------------------------------------------------------
# RK4
# ---------------------------------------------------------------------------


class RK4(Integrator):
    """Classical 4th-order Runge-Kutta integrator.

    More accurate than semi-implicit Euler (O(dt⁴) local error vs O(dt²))
    but requires 4 ABA evaluations per step.  Recommended for:
      - Smooth, contact-free trajectories where accuracy matters.
      - Validating results from the faster semi-implicit Euler.

    Note: for stiff contact dynamics, semi-implicit Euler is often
    more robust despite lower formal order.
    """

    def step(
        self,
        tree: RobotTree,
        q: NDArray[np.float64],
        qdot: NDArray[np.float64],
        tau: NDArray[np.float64],
        ext_forces: Optional[List] = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        dt = self.dt

        def deriv(q_: NDArray, v_: NDArray):
            """Returns (dq/dt, dqdot/dt) = (qdot, qddot)."""
            qddot_ = tree.aba(q_, v_, tau, ext_forces)
            return v_, qddot_

        v1, a1 = deriv(q, qdot)
        v2, a2 = deriv(self._integrate_q(tree, q, v1, dt / 2), qdot + dt / 2 * a1)
        v3, a3 = deriv(self._integrate_q(tree, q, v2, dt / 2), qdot + dt / 2 * a2)
        v4, a4 = deriv(self._integrate_q(tree, q, v3, dt), qdot + dt * a3)

        qdot_new = qdot + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        # Weighted average of velocity increments for position update
        v_avg = (v1 + 2 * v2 + 2 * v3 + v4) / 6.0
        q_new = self._integrate_q(tree, q, v_avg, dt)

        return q_new, qdot_new


# ---------------------------------------------------------------------------
# Simulation loop helper
# ---------------------------------------------------------------------------


def simulate(
    tree: RobotTree,
    q0: NDArray[np.float64],
    qdot0: NDArray[np.float64],
    controller_fn: Callable[[float, NDArray, NDArray], NDArray],
    contact_fn: Optional[Callable] = None,
    dt: float = 1e-3,
    duration: float = 1.0,
    integrator: str = "semi_implicit",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Run a full simulation loop.

    Args:
        tree          : Articulated body tree.
        q0, qdot0     : Initial state.
        controller_fn : fn(t, q, qdot) → tau  (joint torques).
        contact_fn    : fn(q, qdot, X_world_list, v_list) → ext_forces.
                        If None, no contact forces are applied.
        dt            : Time step [s].
        duration      : Total simulation time [s].
        integrator    : "semi_implicit" or "rk4".

    Returns:
        times  : (N,)      time stamps.
        qs     : (N, nq)   generalised positions.
        qdots  : (N, nv)   generalised velocities.
    """
    integ: Integrator = SemiImplicitEuler(dt) if integrator == "semi_implicit" else RK4(dt)

    n_steps = int(duration / dt)
    times = np.zeros(n_steps, dtype=np.float64)
    qs = np.zeros((n_steps, tree.nq), dtype=np.float64)
    qdots = np.zeros((n_steps, tree.nv), dtype=np.float64)

    q, qdot = q0.copy(), qdot0.copy()

    for i in range(n_steps):
        t = i * dt
        times[i] = t
        qs[i] = q
        qdots[i] = qdot

        tau = controller_fn(t, q, qdot)

        ext_forces = None
        if contact_fn is not None:
            X_world = tree.forward_kinematics(q)
            # Velocity in body frame per body
            # (contact_fn builds its own velocity list from ABA internals)
            ext_forces = contact_fn(q, qdot, X_world)

        q, qdot = integ.step(tree, q, qdot, tau, ext_forces)

    return times, qs, qdots
