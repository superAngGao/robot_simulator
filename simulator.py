"""
Simulator — orchestrates one physics step for a RobotModel.

Encapsulates the standard step sequence:
  passive torques → FK → body velocities → contact forces →
  self-collision forces → merge → integrate.

Reference: Drake System::CalcTimeDerivatives pattern (Drake docs §4.2).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from physics.integrator import Integrator
from robot.model import RobotModel


class Simulator:
    """Stateless single-step orchestrator for a RobotModel.

    Args:
        model      : Loaded robot (tree + contact + self-collision).
        integrator : Any Integrator instance (SemiImplicitEuler, RK4, …).
    """

    def __init__(self, model: RobotModel, integrator: Integrator) -> None:
        self.model = model
        self.integrator = integrator

    def step(
        self,
        q: NDArray[np.float64],
        qdot: NDArray[np.float64],
        tau: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Advance the simulation by one time step.

        Args:
            q    : Generalised positions  (nq,).
            qdot : Generalised velocities (nv,).
            tau  : Actuator torques       (nv,).

        Returns:
            (q_new, qdot_new) after one integrator step.
        """
        tree = self.model.tree

        tau_passive = tree.passive_torques(q, qdot)
        tau_total = tau + tau_passive

        X_world = tree.forward_kinematics(q)
        v_bodies = tree.body_velocities(q, qdot)

        contact_forces = self.model.contact_model.compute_forces(
            X_world,
            v_bodies,
            tree.num_bodies,
            dt=self.integrator.dt,
            tree=tree,
        )
        sc_forces = self.model.self_collision.compute_forces(X_world, v_bodies, tree.num_bodies)

        ext_forces = [cf + scf for cf, scf in zip(contact_forces, sc_forces)]

        return self.integrator.step(tree, q, qdot, tau_total, ext_forces)
