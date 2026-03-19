"""
Gymnasium-compatible single-robot RL environment.

Follows Isaac Lab's Env pattern:
  - Pre-computed static indices in __init__ (GPU-friendly for Phase 2e).
  - Per-step cache updated by _update_cache() after every sim.step().
  - Observation / reward / termination delegated to term managers.

Reference: Isaac Lab ManagerBasedEnv (Isaac Lab docs §3.1).
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray

from physics.integrator import SemiImplicitEuler
from physics.joint import FreeJoint
from robot.model import RobotModel
from simulator import Simulator

from .cfg import EnvCfg
from .controllers import PDController
from .managers import ObsManager, RewardManager, TerminationManager


class Env(gym.Env):
    """Single-robot Gymnasium environment.

    Args:
        model    : Loaded RobotModel (tree + contact + self-collision).
        cfg      : EnvCfg with all hyperparameters.
        reset_fn : Callable() -> (q, qdot).  None → tree.default_state().
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        model: RobotModel,
        cfg: EnvCfg,
        reset_fn: Callable | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self._reset_fn = reset_fn

        tree = model.tree

        # ------------------------------------------------------------------
        # Pre-compute static indices (computed once, reused every step)
        # ------------------------------------------------------------------

        # Root body index (body 0 is always the root after BFS ordering)
        self.root_body_idx: int = 0

        # Root body q slice (FreeJoint: 7 values [qx,qy,qz,qw,px,py,pz])
        root_body = tree.bodies[self.root_body_idx]
        self.root_q_slice: slice = root_body.q_idx

        # Actuated joint indices into q and qdot
        actuated_bodies = [b for b in tree.bodies if b.joint.nv > 0 and not isinstance(b.joint, FreeJoint)]
        self.actuated_q_indices: NDArray = np.array(
            [i for b in actuated_bodies for i in range(b.q_idx.start, b.q_idx.stop)],
            dtype=np.intp,
        )
        self.actuated_v_indices: NDArray = np.array(
            [i for b in actuated_bodies for i in range(b.v_idx.start, b.v_idx.stop)],
            dtype=np.intp,
        )

        self.contact_body_names: list[str] = list(model.contact_body_names)

        # ------------------------------------------------------------------
        # Simulator
        # ------------------------------------------------------------------
        self.sim = Simulator(model, SemiImplicitEuler(cfg.dt))

        # ------------------------------------------------------------------
        # Controller
        # ------------------------------------------------------------------
        if cfg.controller is not None:
            self.controller = cfg.controller
        else:
            self.controller = PDController(
                kp=cfg.kp,
                kd=cfg.kd,
                action_scale=cfg.action_scale,
                actuated_q_indices=self.actuated_q_indices,
                actuated_v_indices=self.actuated_v_indices,
                nv=tree.nv,
                effort_limits=model.effort_limits,
            )

        # ------------------------------------------------------------------
        # State cache (populated by _update_cache)
        # ------------------------------------------------------------------
        self.q: NDArray = np.zeros(tree.nq)
        self.qdot: NDArray = np.zeros(tree.nv)
        self.X_world: list = []
        self.v_bodies: list = []
        self.active_contacts: list = []

        # ------------------------------------------------------------------
        # Term managers
        # ------------------------------------------------------------------
        self.obs_manager = ObsManager(cfg.obs_cfg, self)
        self.reward_manager = RewardManager()
        self.termination_manager = TerminationManager()

        # ------------------------------------------------------------------
        # Gymnasium spaces (set after a dummy reset to know obs_dim)
        # ------------------------------------------------------------------
        nu = len(self.actuated_q_indices)
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(nu,), dtype=np.float32)

        # Bootstrap cache so obs_manager.obs_dim works before first reset
        self.q, self.qdot = tree.default_state()
        self._update_cache()
        obs_dim = self.obs_manager.obs_dim

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _update_cache(self) -> None:
        tree = self.model.tree
        self.X_world = tree.forward_kinematics(self.q)
        self.v_bodies = tree.body_velocities(self.q, self.qdot)
        self.active_contacts = self.model.contact_model.active_contacts(self.X_world)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        tree = self.model.tree

        if self._reset_fn is not None:
            self.q, self.qdot = self._reset_fn()
        else:
            self.q, self.qdot = tree.default_state()

        if self.cfg.init_noise_scale > 0:
            self.q = self.q + np.random.uniform(
                -self.cfg.init_noise_scale,
                self.cfg.init_noise_scale,
                self.q.shape,
            )

        self._update_cache()
        self._step_count = 0
        self.obs_manager.train()

        obs = self.obs_manager.compute()
        return obs, {}

    def step(self, action):
        # Action clip (training hyperparameter)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        if self.cfg.action_clip is not None:
            action = torch.clamp(action, -self.cfg.action_clip, self.cfg.action_clip)

        tau = self.controller.compute(action.numpy(), self.q, self.qdot)
        self.q, self.qdot = self.sim.step(self.q, self.qdot, tau)
        self._update_cache()
        self._step_count += 1

        obs = self.obs_manager.compute()
        reward = self.reward_manager.compute()
        terminated = self.termination_manager.compute()
        truncated = self._step_count >= self.cfg.episode_length

        return obs, reward, terminated, truncated, {}

    def render(self):
        pass  # rendering handled externally via RobotViewer
