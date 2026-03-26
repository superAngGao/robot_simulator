"""
Simulator — multi-physics orchestrator for a Scene.

Coordinates per-robot rigid body dynamics pipelines with unified collision
detection via CollisionPipeline.

Step sequence:
  1. Per-robot DynamicsCache (FK + body_v) — computed once, shared
  2. Unified collision detection (CollisionPipeline)
  3. Per-robot StepPipeline.step() (smooth forces → constraint → integrate)

References:
  MuJoCo mj_step1 + mj_step2 pipeline.
  Drake System::CalcTimeDerivatives pattern (Drake docs §4.2).
  Isaac Lab InteractiveScene step pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from collision_pipeline import CollisionPipeline
from physics.constraint_solvers import wrap_solver
from physics.dynamics_cache import DynamicsCache
from physics.force_source import PassiveForceSource
from physics.solvers.pgs_solver import ContactConstraint, PGSContactSolver
from physics.spatial import SpatialTransform
from physics.step_pipeline import StepPipeline
from robot.model import RobotModel

if TYPE_CHECKING:
    from physics.dynamics_cache import ForceState
    from physics.integrator import Integrator


class Simulator:
    """Multi-robot scene simulator with unified collision pipeline.

    Args:
        scene_or_model : Built Scene or single RobotModel (auto-wrapped).
        integrator     : Any Integrator instance (for dt extraction).
        solver         : Contact solver. If None, uses PGS(max_iter=30).
    """

    def __init__(
        self,
        scene_or_model,
        integrator: "Integrator",
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
        self.collision_pipeline = CollisionPipeline(self.scene)

        # Build the rigid body dynamics pipeline
        wrapped_solver = wrap_solver(self.solver)
        self._pipeline = StepPipeline(
            dt=integrator.dt,
            force_sources=[PassiveForceSource()],
            constraint_solver=wrapped_solver,
        )

        # Per-robot force state from last step
        self._last_force_states: dict[str, ForceState] = {}

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
            raise TypeError(
                "step() requires either step(states_dict, taus_dict) or "
                "step(q, qdot, tau). Use step_single() for single-robot."
            )

        states = states_or_q
        taus = taus_or_qdot
        reg = self.scene.registry

        # ── 1. Per-robot DynamicsCache (FK + body_v, no H yet) ──
        caches: dict[str, DynamicsCache] = {}
        all_X: list[SpatialTransform | None] = [None] * reg.total_bodies
        all_v: list[NDArray | None] = [None] * reg.total_bodies

        for name, model in self.scene.robots.items():
            q, qdot = states[name]
            cache = DynamicsCache.from_tree(model.tree, q, qdot, self._pipeline.dt)
            caches[name] = cache
            offset = reg.robot_offset[name]
            for i in range(len(cache.X_world)):
                all_X[offset + i] = cache.X_world[i]
                all_v[offset + i] = cache.body_v[i]

        # Static geometries: fixed pose, zero velocity
        for si, sg in enumerate(self.scene.static_geometries):
            gid = reg.static_global_id(si)
            all_X[gid] = sg.pose
            all_v[gid] = np.zeros(6)

        # ── 2. Unified collision detection ──
        contacts = self.collision_pipeline.detect(all_X, all_v)

        # ── 3. Per-robot: filter contacts + StepPipeline ──
        new_states: dict[str, tuple[NDArray, NDArray]] = {}

        for name, model in self.scene.robots.items():
            q, qdot = states[name]
            tau = taus[name]
            tree = model.tree
            offset = reg.robot_offset[name]
            nb = tree.num_bodies

            # Filter contacts for this robot (remap global → local indices)
            robot_contacts = _filter_contacts(contacts, offset, nb)

            q_new, qdot_new = self._pipeline.step(tree, q, qdot, tau, robot_contacts, cache=caches[name])
            new_states[name] = (q_new, qdot_new)

            # Store force state
            if self._pipeline.last_force_state is not None:
                self._last_force_states[name] = self._pipeline.last_force_state

        return new_states

    @property
    def last_force_states(self) -> dict[str, "ForceState"]:
        """Per-robot force breakdown from the most recent step() call."""
        return self._last_force_states

    # Keep backward compat attribute name
    @property
    def pipeline(self):
        return self.collision_pipeline

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
        integrator: "Integrator",
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


def _filter_contacts(
    contacts: list[ContactConstraint],
    offset: int,
    nb: int,
) -> list[ContactConstraint]:
    """Filter and remap global contact indices to local per-robot indices."""
    robot_contacts: list[ContactConstraint] = []
    for c in contacts:
        bi_local = c.body_i - offset if 0 <= c.body_i - offset < nb else -1
        bj_local = c.body_j - offset if 0 <= c.body_j - offset < nb else -1
        if bi_local >= 0 or bj_local >= 0:
            cc = ContactConstraint(
                body_i=bi_local if bi_local >= 0 else -1,
                body_j=bj_local if bj_local >= 0 else -1,
                point=c.point,
                normal=c.normal,
                tangent1=c.tangent1,
                tangent2=c.tangent2,
                depth=c.depth,
                mu=c.mu,
                condim=c.condim,
                mu_spin=c.mu_spin,
                mu_roll=c.mu_roll,
                restitution=c.restitution,
                erp=c.erp,
                slop=c.slop,
            )
            robot_contacts.append(cc)
    return robot_contacts
