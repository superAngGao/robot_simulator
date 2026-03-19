"""
Term managers for the RL environment.

ObsManager      : Computes and concatenates observation terms, applies noise.
RewardManager   : Stub — returns 0.0 (Phase 3).
TerminationManager : Stub — returns False (Phase 3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .cfg import NoiseCfg, ObsTermCfg


class TermManager(ABC):
    def __init__(self) -> None:
        self._training = True

    def train(self) -> None:
        self._training = True

    def eval(self) -> None:
        self._training = False

    @abstractmethod
    def compute(self): ...


class ObsManager(TermManager):
    """Concatenates all observation terms and optionally adds noise in train mode.

    Args:
        terms : Ordered dict of name → ObsTermCfg.
        env   : The Env instance (passed at construction, not per-compute).
    """

    def __init__(self, terms: dict[str, ObsTermCfg], env) -> None:
        super().__init__()
        self._terms = terms
        self._env = env

    def compute(self) -> torch.Tensor:
        parts = []
        for cfg in self._terms.values():
            vec = cfg.func(self._env, **cfg.params)  # (dim,)
            if self._training and cfg.noise is not None:
                vec = vec + _sample_noise(cfg.noise, vec.shape)
            parts.append(vec)
        return torch.cat(parts, dim=0) if parts else torch.zeros(0)

    @property
    def obs_dim(self) -> int:
        """Total observation dimension (sum of all term dims)."""
        total = 0
        for cfg in self._terms.values():
            total += cfg.func(self._env, **cfg.params).shape[0]
        return total


class RewardManager(TermManager):
    """Stub reward manager — always returns 0.0. Implement terms in Phase 3."""

    def compute(self) -> float:
        return 0.0


class TerminationManager(TermManager):
    """Stub termination manager — never terminates. Implement terms in Phase 3."""

    def compute(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Noise helpers
# ---------------------------------------------------------------------------


def _sample_noise(cfg: NoiseCfg, shape: torch.Size) -> torch.Tensor:
    if cfg.noise_type == "gaussian":
        return torch.randn(shape) * cfg.std
    elif cfg.noise_type == "uniform":
        return torch.empty(shape).uniform_(cfg.low, cfg.high)
    else:
        raise ValueError(f"Unknown noise_type {cfg.noise_type!r}")
