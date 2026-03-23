"""
Vectorised environment — backend-agnostic batched simulation.

Delegates physics to a BatchBackend (CPU for-loop or GPU kernels).
The public API (reset / step) is unchanged from Phase 2d.
"""

from __future__ import annotations

from typing import Callable

import torch

from robot.model import RobotModel

from .cfg import EnvCfg, NoiseCfg, ObsTermCfg


class VecEnv:
    """N parallel environments backed by a BatchBackend.

    Args:
        model     : Shared RobotModel (read-only).
        cfg       : Shared EnvCfg.
        num_envs  : Number of parallel environments.
        backend   : ``"numpy"`` (CPU for-loop) or ``"warp"`` (GPU).
        reset_fn  : Ignored for backend-based VecEnv (kept for API compat).
    """

    def __init__(
        self,
        model: RobotModel,
        cfg: EnvCfg,
        num_envs: int,
        backend: str = "numpy",
        reset_fn: Callable | None = None,
    ) -> None:
        from physics.backends import get_backend

        self.num_envs = num_envs
        self._cfg = cfg
        self._backend = get_backend(backend, model, cfg, num_envs)
        self._obs_manager = BatchedObsManager(cfg.obs_cfg)
        self._step_counts = torch.zeros(num_envs, dtype=torch.int64)

        # Bootstrap obs_dim by doing a reset
        result = self._backend.reset_all(init_noise_scale=0.0)
        obs_data = self._backend.get_obs_data(result)
        self._obs_dim = self._obs_manager.compute(obs_data).shape[1]

    def reset(self) -> tuple[torch.Tensor, list[dict]]:
        """Reset all envs. Returns obs (N, obs_dim) and list of info dicts."""
        result = self._backend.reset_all(
            init_noise_scale=self._cfg.init_noise_scale,
        )
        self._step_counts.zero_()
        self._obs_manager.train()

        obs_data = self._backend.get_obs_data(result)
        obs = self._obs_manager.compute(obs_data)
        return obs, [{} for _ in range(self.num_envs)]

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
        result = self._backend.step_batch(actions)
        self._step_counts += 1

        obs_data = self._backend.get_obs_data(result)
        obs = self._obs_manager.compute(obs_data)

        # Stubs (Phase 3: use RewardManager / TerminationManager)
        rewards = torch.zeros(self.num_envs, dtype=torch.float32)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = self._step_counts >= self._cfg.episode_length

        return obs, rewards, terminated, truncated, [{} for _ in range(self.num_envs)]


# ---------------------------------------------------------------------------
# Batched observation manager
# ---------------------------------------------------------------------------


class BatchedObsManager:
    """Concatenates observation terms for N environments.

    Works with the ``dict[str, Tensor]`` returned by
    ``BatchBackend.get_obs_data()``. Each obs term function is mapped to
    a key in this dict via ``func.__name__``.

    Args:
        terms : Ordered dict of name -> ObsTermCfg.
    """

    def __init__(self, terms: dict[str, ObsTermCfg]) -> None:
        self._terms = terms
        self._training = True

    def train(self) -> None:
        self._training = True

    def eval(self) -> None:
        self._training = False

    def compute(self, obs_data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute batched observations (N, obs_dim).

        Args:
            obs_data : Pre-sliced dict from BatchBackend.get_obs_data().
        """
        parts = []
        for cfg in self._terms.values():
            key = cfg.func.__name__
            vec = obs_data[key]  # (N, dim)
            if self._training and cfg.noise is not None:
                vec = vec + _sample_noise_batch(cfg.noise, vec.shape, vec.device)
            parts.append(vec)
        return torch.cat(parts, dim=1) if parts else torch.zeros(0)


def _sample_noise_batch(
    cfg: NoiseCfg, shape: tuple, device: torch.device
) -> torch.Tensor:
    if cfg.noise_type == "gaussian":
        return torch.randn(shape, device=device) * cfg.std
    elif cfg.noise_type == "uniform":
        return torch.empty(shape, device=device).uniform_(cfg.low, cfg.high)
    else:
        raise ValueError(f"Unknown noise_type {cfg.noise_type!r}")
