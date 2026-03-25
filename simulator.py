"""
Simulator — orchestrates one physics step for a Scene.

Supports multi-robot scenes with static geometry and unified collision
detection via CollisionPipeline.

Step sequence per robot:
  passive torques → FK → body velocities → (global) collision detect →
  (global) constraint solve → distribute forces → integrate.

References:
  Drake System::CalcTimeDerivatives pattern (Drake docs §4.2).
  Isaac Lab InteractiveScene step pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from collision_pipeline import CollisionPipeline
from physics.integrator import Integrator
from physics.solvers.mujoco_qp import MuJoCoStyleSolver
from physics.solvers.pgs_solver import PGSContactSolver
from physics.spatial import SpatialTransform
from robot.model import RobotModel

if TYPE_CHECKING:
    from physics.implicit_contact_step import ImplicitContactStep


class Simulator:
    """Multi-robot scene simulator with unified collision pipeline.

    Args:
        scene      : Built Scene (call scene.build() first).
        integrator : Any Integrator instance (SemiImplicitEuler, RK4).
        solver     : Contact solver (PGS, Jacobi, ADMM). If None, uses PGS(30).
    """

    def __init__(
        self,
        scene_or_model,
        integrator: Integrator,
        solver=None,
    ) -> None:
        # Auto-wrap RobotModel into Scene for backward compatibility
        if isinstance(scene_or_model, RobotModel):
            from scene import Scene

            scene_or_model = Scene.single_robot(scene_or_model)

        self.scene = scene_or_model
        self.integrator = integrator
        kw = self.scene.solver_kwargs if self.scene.solver_kwargs else {"max_iter": 30}
        self.solver = solver or PGSContactSolver(**kw)
        self.pipeline = CollisionPipeline(self.scene)
        # Direct integration path: available for ALL constraint solvers
        # (eliminates impulse->force->ABA round-trip)
        self._use_direct_path = isinstance(self.solver, (MuJoCoStyleSolver,))
        # Can also be enabled for PGS/PGS-SI/Jacobi via use_direct_step=True
        self._implicit_step = None

    def step(self, states_or_q, taus_or_qdot=None, tau_single=None):
        """Advance the simulation by one time step.

        Two calling conventions:
          Multi-robot:  step({"a": (q,qdot)}, {"a": tau}) → {"a": (q,qdot)}
          Single-robot: step(q, qdot, tau) → (q, qdot)   [backward compat]
        """
        # Detect single-robot backward-compat call: step(q, qdot, tau)
        if tau_single is not None:
            return self.step_single(states_or_q, taus_or_qdot, tau_single)
        if not isinstance(states_or_q, dict):
            # Assume step(q, qdot, tau) with tau=taus_or_qdot... but 2 args
            # This shouldn't happen in new code; raise for clarity
            raise TypeError(
                "step() requires either step(states_dict, taus_dict) or "
                "step(q, qdot, tau). Use step_single() for single-robot."
            )

        states = states_or_q
        taus = taus_or_qdot
        reg = self.scene.registry
        dt = self.integrator.dt

        # ── 1. FK + body velocities (per robot) ──
        all_X: list[SpatialTransform | None] = [None] * reg.total_bodies
        all_v: list[NDArray | None] = [None] * reg.total_bodies

        for name, model in self.scene.robots.items():
            q, qdot = states[name]
            offset = reg.robot_offset[name]
            X_list = model.tree.forward_kinematics(q)
            v_list = model.tree.body_velocities(q, qdot)
            for i in range(len(X_list)):
                all_X[offset + i] = X_list[i]
                all_v[offset + i] = v_list[i]

        # Static geometries: fixed pose, zero velocity
        for si, sg in enumerate(self.scene.static_geometries):
            gid = reg.static_global_id(si)
            all_X[gid] = sg.pose
            all_v[gid] = np.zeros(6)

        # ── 2. Unified collision detection ──
        contacts = self.pipeline.detect(all_X, all_v)

        # ── Branch: direct path (no round-trip) vs legacy ──
        if self._use_direct_path:
            return self._step_direct(states, taus, contacts, reg)

        # ── 3. Solve constraints (legacy path) ──
        if contacts:
            inv_mass, inv_inertia = self.pipeline.gather_mass_properties()
            impulses_global = self.solver.solve(contacts, all_v, all_X, inv_mass, inv_inertia, dt=dt)
        else:
            impulses_global = [np.zeros(6) for _ in range(reg.total_bodies)]

        # ── 4. Distribute forces and integrate (per robot) ──
        new_states: dict[str, tuple[NDArray, NDArray]] = {}

        for name, model in self.scene.robots.items():
            q, qdot = states[name]
            tau = taus[name]
            tree = model.tree
            offset = reg.robot_offset[name]
            nb = tree.num_bodies

            # Convert global impulses to per-robot ext_forces
            ext_forces = [impulses_global[offset + i] / dt for i in range(nb)]

            tau_total = tau + tree.passive_torques(q, qdot)
            q_new, qdot_new = self.integrator.step(tree, q, qdot, tau_total, ext_forces)
            new_states[name] = (q_new, qdot_new)

        # ── 5. Apply split-impulse position corrections (if solver provides them) ──
        if hasattr(self.solver, "position_corrections"):
            pos_corr = self.solver.position_corrections
            for name, model in self.scene.robots.items():
                q_new, qdot_new = new_states[name]
                offset = reg.robot_offset[name]
                tree = model.tree
                for i in range(tree.num_bodies):
                    pc = pos_corr[offset + i]
                    if pc[0] == 0.0 and pc[1] == 0.0 and pc[2] == 0.0:
                        continue
                    body = tree.bodies[i]
                    # Apply world-frame position correction to FreeJoint translation
                    if body.joint.nq == 7:  # FreeJoint
                        qs = body.q_idx.start if isinstance(body.q_idx, slice) else body.q_idx[0]
                        q_new[qs + 4 : qs + 7] += pc
                new_states[name] = (q_new, qdot_new)

        return new_states

    def _step_direct(self, states, taus, contacts, reg):
        """Direct path: solver operates on predicted state (no round-trip).

        Works for ALL solver types (MuJoCoStyle, PGS, PGS-SI, Jacobi, ADMM).
        """
        if self._implicit_step is None:
            from physics.implicit_contact_step import ImplicitContactStep
            self._implicit_step = ImplicitContactStep(self.integrator.dt, self.solver)

        new_states: dict[str, tuple[NDArray, NDArray]] = {}

        for name, model in self.scene.robots.items():
            q, qdot = states[name]
            tau = taus[name]
            tree = model.tree
            offset = reg.robot_offset[name]
            nb = tree.num_bodies

            # Filter contacts for this robot (remap global -> local indices)
            robot_contacts = []
            for c in contacts:
                bi_local = c.body_i - offset if 0 <= c.body_i - offset < nb else -1
                bj_local = c.body_j - offset if 0 <= c.body_j - offset < nb else -1
                if bi_local >= 0 or bj_local >= 0:
                    from physics.solvers.pgs_solver import ContactConstraint
                    cc = ContactConstraint(
                        body_i=bi_local if bi_local >= 0 else -1,
                        body_j=bj_local if bj_local >= 0 else -1,
                        point=c.point, normal=c.normal,
                        tangent1=c.tangent1, tangent2=c.tangent2,
                        depth=c.depth, mu=c.mu, condim=c.condim,
                        mu_spin=c.mu_spin, mu_roll=c.mu_roll,
                        restitution=c.restitution,
                        erp=c.erp, slop=c.slop,
                    )
                    robot_contacts.append(cc)

            tau_total = tau + tree.passive_torques(q, qdot)
            q_new, qdot_new = self._implicit_step.step(
                tree, q, qdot, tau_total, contacts=robot_contacts
            )
            new_states[name] = (q_new, qdot_new)

        return new_states

    # ------------------------------------------------------------------
    # Single-robot convenience API
    # ------------------------------------------------------------------

    def step_single(
        self,
        q: NDArray,
        qdot: NDArray,
        tau: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Single-robot shortcut (scene must have exactly one robot named 'main').

        Matches the old Simulator.step(q, qdot, tau) signature.
        """
        result = self.step({"main": (q, qdot)}, {"main": tau})
        return result["main"]

    @classmethod
    def from_model(
        cls,
        model: "RobotModel",
        integrator: Integrator,
        terrain=None,
        static_geometries=None,
        solver=None,
        **solver_kwargs,
    ) -> "Simulator":
        """Build a Simulator from a single RobotModel (backward compat).

        Automatically wraps the model in a Scene.
        """
        from scene import Scene

        scene = Scene(
            robots={"main": model},
            static_geometries=static_geometries or [],
            terrain=terrain or model._terrain if hasattr(model, "_terrain") else None,
            solver_kwargs=solver_kwargs,
        ).build()
        return cls(scene, integrator, solver)
