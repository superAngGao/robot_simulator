"""
Joint-level controllers for the RL environment.

PDController  : PD position controller with optional effort clipping.
TorqueController : Pass-through torque controller.

Reference: Isaac Lab ActuatorBase pattern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Controller(ABC):
    @abstractmethod
    def compute(
        self,
        action: NDArray,
        q: NDArray,
        qdot: NDArray,
    ) -> NDArray:
        """Compute generalised forces tau (nv,) from action."""
        ...


class PDController(Controller):
    """PD position controller.

    action = target joint angle offset relative to current q.
    tau = kp*(target - q) - kd*qdot, clipped to effort_limits if provided.

    Args:
        kp                  : Proportional gain.
        kd                  : Derivative gain.
        action_scale        : Multiplier applied to action before adding to q.
        actuated_q_indices  : int array indexing into q for actuated joints.
        actuated_v_indices  : int array indexing into qdot/tau for actuated joints.
        nv                  : Total size of the tau vector.
        effort_limits       : (nu,) array of max torque magnitudes, or None.
    """

    def __init__(
        self,
        kp: float,
        kd: float,
        action_scale: float,
        actuated_q_indices: NDArray,
        actuated_v_indices: NDArray,
        nv: int,
        effort_limits: NDArray | None = None,
    ) -> None:
        self.kp = kp
        self.kd = kd
        self.action_scale = action_scale
        self.actuated_q_indices = actuated_q_indices
        self.actuated_v_indices = actuated_v_indices
        self.nv = nv
        self.effort_limits = effort_limits

    def compute(self, action: NDArray, q: NDArray, qdot: NDArray) -> NDArray:
        target = q[self.actuated_q_indices] + action * self.action_scale
        tau_act = self.kp * (target - q[self.actuated_q_indices]) - self.kd * qdot[self.actuated_v_indices]
        if self.effort_limits is not None:
            tau_act = np.clip(tau_act, -self.effort_limits, self.effort_limits)
        tau = np.zeros(self.nv, dtype=np.float64)
        tau[self.actuated_v_indices] = tau_act
        return tau


class TorqueController(Controller):
    """Direct torque controller — action is tau, passed through with optional effort clip.

    Args:
        actuated_v_indices : int array indexing into tau for actuated joints.
        nv                 : Total size of the tau vector.
        effort_limits      : (nu,) array of max torque magnitudes, or None.
    """

    def __init__(
        self,
        actuated_v_indices: NDArray,
        nv: int,
        effort_limits: NDArray | None = None,
    ) -> None:
        self.actuated_v_indices = actuated_v_indices
        self.nv = nv
        self.effort_limits = effort_limits

    def compute(self, action: NDArray, q: NDArray, qdot: NDArray) -> NDArray:
        tau_act = np.asarray(action, dtype=np.float64)
        if self.effort_limits is not None:
            tau_act = np.clip(tau_act, -self.effort_limits, self.effort_limits)
        tau = np.zeros(self.nv, dtype=np.float64)
        tau[self.actuated_v_indices] = tau_act
        return tau
