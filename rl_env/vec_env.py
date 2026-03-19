"""
Vectorised environment — Python for-loop backend (Phase 2d).

Phase 2e: replace the for loop with a Warp kernel; the public API is unchanged.
"""

from __future__ import annotations

from typing import Callable

import torch

from robot.model import RobotModel

from .base_env import Env
from .cfg import EnvCfg


class VecEnv:
    """N independent Env instances stepped in a Python for loop.

    Args:
        model     : Shared RobotModel (read-only; each Env has its own state).
        cfg       : Shared EnvCfg.
        num_envs  : Number of parallel environments.
        reset_fn  : Optional callable() -> (q, qdot) for custom resets.
                    The same function is passed to every sub-env.
    """

    def __init__(
        self,
        model: RobotModel,
        cfg: EnvCfg,
        num_envs: int,
        reset_fn: Callable | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.envs = [Env(model, cfg, reset_fn) for _ in range(num_envs)]

    def reset(self) -> tuple[torch.Tensor, list[dict]]:
        """Reset all envs. Returns obs (N, obs_dim) and list of info dicts."""
        obs_list, info_list = [], []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)
        return torch.stack(obs_list, dim=0), info_list

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
        """Step all envs with actions (N, nu).

        Returns:
            obs        : (N, obs_dim)
            rewards    : (N,)
            terminated : (N,) bool
            truncated  : (N,) bool
            infos      : list of N dicts
        """
        obs_list, rew_list, term_list, trunc_list, info_list = [], [], [], [], []
        for i, env in enumerate(self.envs):
            obs, rew, term, trunc, info = env.step(actions[i])
            obs_list.append(obs)
            rew_list.append(rew)
            term_list.append(term)
            trunc_list.append(trunc)
            info_list.append(info)

        return (
            torch.stack(obs_list, dim=0),
            torch.tensor(rew_list, dtype=torch.float32),
            torch.tensor(term_list, dtype=torch.bool),
            torch.tensor(trunc_list, dtype=torch.bool),
            info_list,
        )
