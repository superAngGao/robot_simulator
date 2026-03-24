"""
NumpyLoopBackend — CPU fallback using a Python for-loop over N environments.

Wraps the existing Simulator + Controller logic. Functionally identical
to the current VecEnv implementation but packaged as a BatchBackend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from physics.integrator import SemiImplicitEuler
from physics.joint import FreeJoint

# NumpyLoopBackend uses direct penalty contact (matching GPU kernels),
# not the Scene-based Simulator.
from .batch_backend import BatchBackend, StepResult
from .static_data import StaticRobotData

if TYPE_CHECKING:
    from rl_env.cfg import EnvCfg
    from rl_env.controllers import Controller
    from robot.model import RobotModel


class NumpyLoopBackend(BatchBackend):
    """CPU for-loop backend — one Simulator per environment.

    Args:
        model    : Shared RobotModel.
        cfg      : Environment configuration.
        num_envs : Number of parallel environments.
    """

    def __init__(
        self,
        model: "RobotModel",
        cfg: "EnvCfg",
        num_envs: int,
    ) -> None:
        self._model = model
        self._cfg = cfg
        self._num_envs = num_envs
        self._static = StaticRobotData.from_model(model)

        tree = model.tree

        # Per-env integrator (penalty contact, matching GPU backends)
        self._integrators = [SemiImplicitEuler(cfg.dt) for _ in range(num_envs)]

        # Build penalty contact model from static data (matches GPU kernel)
        from physics.contact import ContactParams, ContactPoint, PenaltyContactModel

        s = self._static
        self._penalty_contact = PenaltyContactModel(
            ContactParams(
                k_normal=s.contact_k_normal,
                b_normal=s.contact_b_normal,
                mu=s.contact_mu,
                slip_eps=s.contact_slip_eps,
                ground_z=s.contact_ground_z,
            )
        )
        for ci in range(s.nc):
            self._penalty_contact.add_contact_point(
                ContactPoint(
                    body_index=int(s.contact_body_idx[ci]),
                    position=s.contact_local_pos[ci].astype(np.float64),
                    name=f"contact_{ci}",
                )
            )

        # Build self-collision model from static data
        from physics.collision import AABBSelfCollision, BodyAABB, NullSelfCollision

        if len(s.collision_body_idx) > 0:
            sc = AABBSelfCollision(k_contact=s.collision_k, b_contact=s.collision_b)
            for ci_sc in range(len(s.collision_body_idx)):
                sc.add_body(
                    BodyAABB(
                        int(s.collision_body_idx[ci_sc]),
                        s.collision_half_ext[ci_sc].astype(np.float64),
                    )
                )
            parent_list = [b.parent for b in tree.bodies]
            sc.build_pairs(parent_list)
            self._self_collision = sc
        else:
            self._self_collision = NullSelfCollision()

        # Build controllers
        self._controllers = [self._make_controller(model, cfg, tree) for _ in range(num_envs)]

        # Batched state (numpy)
        self._q = np.zeros((num_envs, tree.nq), dtype=np.float64)
        self._qdot = np.zeros((num_envs, tree.nv), dtype=np.float64)

    # ------------------------------------------------------------------
    # BatchBackend interface
    # ------------------------------------------------------------------

    def reset_all(self, init_noise_scale: float = 0.0) -> StepResult:
        tree = self._model.tree
        q0, qdot0 = tree.default_state()

        for i in range(self._num_envs):
            self._q[i] = q0.copy()
            self._qdot[i] = qdot0.copy()
            if init_noise_scale > 0:
                self._q[i] += np.random.uniform(-init_noise_scale, init_noise_scale, q0.shape)

        return self._build_result()

    def reset_envs(
        self,
        env_ids: torch.Tensor,
        init_noise_scale: float = 0.0,
    ) -> None:
        tree = self._model.tree
        q0, qdot0 = tree.default_state()
        for idx in env_ids.tolist():
            self._q[idx] = q0.copy()
            self._qdot[idx] = qdot0.copy()
            if init_noise_scale > 0:
                self._q[idx] += np.random.uniform(-init_noise_scale, init_noise_scale, q0.shape)

    def step_batch(self, actions: torch.Tensor) -> StepResult:
        actions_np = actions.detach().cpu().numpy()

        for i in range(self._num_envs):
            action = actions_np[i]
            # Action clip
            if self._cfg.action_clip is not None:
                action = np.clip(action, -self._cfg.action_clip, self._cfg.action_clip)

            tau = self._controllers[i].compute(action, self._q[i], self._qdot[i])

            # Penalty contact step (matches GPU backend kernels)
            tree = self._model.tree
            tau_passive = tree.passive_torques(self._q[i], self._qdot[i])
            tau_total = tau + tau_passive
            X_world = tree.forward_kinematics(self._q[i])
            v_bodies = tree.body_velocities(self._q[i], self._qdot[i])
            contact_f = self._penalty_contact.compute_forces(X_world, v_bodies, tree.num_bodies)
            sc_f = self._self_collision.compute_forces(X_world, v_bodies, tree.num_bodies)
            ext_f = [cf + sf for cf, sf in zip(contact_f, sc_f)]
            self._q[i], self._qdot[i] = self._integrators[i].step(
                tree, self._q[i], self._qdot[i], tau_total, ext_f
            )

        return self._build_result()

    def get_obs_data(self, result: StepResult) -> dict[str, torch.Tensor]:
        s = self._static

        # Extract root body velocity from v_bodies
        base_lin_vel = result.v_bodies[:, s.root_body_idx, :3]  # (N, 3)
        base_ang_vel = result.v_bodies[:, s.root_body_idx, 3:6]  # (N, 3)

        # Root quaternion from q
        quat_start = s.root_q_start
        base_orientation = result.q[:, quat_start : quat_start + 4]  # (N, 4)

        # Actuated joint positions and velocities
        aq = torch.from_numpy(s.actuated_q_indices.astype(np.int64))
        av = torch.from_numpy(s.actuated_v_indices.astype(np.int64))
        joint_pos = result.q[:, aq]  # (N, nu)
        joint_vel = result.qdot[:, av]  # (N, nu)

        return {
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "base_orientation": base_orientation,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "contact_mask": result.contact_mask,
        }

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def nq(self) -> int:
        return self._static.nq

    @property
    def nv(self) -> int:
        return self._static.nv

    @property
    def num_bodies(self) -> int:
        return self._static.nb

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_result(self) -> StepResult:
        """Compute FK/velocities/contacts and pack into a StepResult."""
        tree = self._model.tree
        N = self._num_envs
        nb = tree.num_bodies
        nc = self._static.nc

        X_world_t = torch.zeros(N, nb, 12, dtype=torch.float32)
        v_bodies_t = torch.zeros(N, nb, 6, dtype=torch.float32)
        contact_mask_t = torch.zeros(N, nc, dtype=torch.bool)

        for i in range(N):
            X_world = tree.forward_kinematics(self._q[i])
            v_bodies = tree.body_velocities(self._q[i], self._qdot[i])

            for j, (X, v) in enumerate(zip(X_world, v_bodies)):
                X_world_t[i, j, :9] = torch.from_numpy(X.R.astype(np.float32).flatten())
                X_world_t[i, j, 9:] = torch.from_numpy(X.r.astype(np.float32))
                v_bodies_t[i, j] = torch.from_numpy(v.astype(np.float32))

            # Contact mask (legacy: uses contact_model if available)
            if hasattr(self._model, "contact_model"):
                active = self._model.contact_model.active_contacts(X_world)
                active_names = {name for name, _ in active}
                from physics.contact import PenaltyContactModel

                if isinstance(self._model.contact_model, PenaltyContactModel):
                    for ci, cp in enumerate(self._model.contact_model.contact_points):
                        if cp.name in active_names:
                            contact_mask_t[i, ci] = True

        return StepResult(
            q=torch.from_numpy(self._q.astype(np.float32)),
            qdot=torch.from_numpy(self._qdot.astype(np.float32)),
            X_world=X_world_t,
            v_bodies=v_bodies_t,
            contact_mask=contact_mask_t,
        )

    @staticmethod
    def _make_controller(model, cfg, tree) -> "Controller":
        from rl_env.controllers import PDController

        if cfg.controller is not None:
            return cfg.controller

        actuated_bodies = [b for b in tree.bodies if b.joint.nv > 0 and not isinstance(b.joint, FreeJoint)]
        actuated_q_indices = np.array(
            [i for b in actuated_bodies for i in range(b.q_idx.start, b.q_idx.stop)],
            dtype=np.intp,
        )
        actuated_v_indices = np.array(
            [i for b in actuated_bodies for i in range(b.v_idx.start, b.v_idx.stop)],
            dtype=np.intp,
        )

        return PDController(
            kp=cfg.kp,
            kd=cfg.kd,
            action_scale=cfg.action_scale,
            actuated_q_indices=actuated_q_indices,
            actuated_v_indices=actuated_v_indices,
            nv=tree.nv,
            effort_limits=model.effort_limits,
        )
