"""
Non-constraint force sources for the rigid body dynamics pipeline.

All force sources output (nv,) generalized forces in joint space.
MuJoCo naming: qfrc_passive, qfrc_actuator, qfrc_applied, etc.

Pre-planned subclasses (not yet implemented):
  - ActuatorForceSource  : motor dynamics, gear ratio, PD servo
  - ExternalWrenchSource : body wrenches (wind, thrust) → J^T @ f
  - SpringForceSource    : tendons, springs across bodies
  - FluidForceSource     : viscous drag, lift

Reference: MuJoCo computation pipeline (mj_step1: smooth forces).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from numpy.typing import NDArray

if TYPE_CHECKING:
    from .dynamics_cache import DynamicsCache
    from .robot_tree import RobotTreeNumpy


class ForceSource(ABC):
    """Abstract base for non-constraint force sources.

    Each source computes a (nv,) generalized force contribution.
    The pipeline sums all sources into tau_smooth before the constraint stage.
    """

    @abstractmethod
    def compute(self, tree: "RobotTreeNumpy", cache: "DynamicsCache") -> NDArray:
        """Compute generalized force contribution.

        Args:
            tree  : Articulated body tree.
            cache : Shared DynamicsCache for this step.

        Returns:
            (nv,) array of generalized forces.
        """


class PassiveForceSource(ForceSource):
    """Joint limits + viscous damping + Coulomb friction.

    Wraps tree.passive_torques(). MuJoCo name: qfrc_passive.
    """

    def compute(self, tree: "RobotTreeNumpy", cache: "DynamicsCache") -> NDArray:
        return tree.passive_torques(cache.q, cache.qdot)
