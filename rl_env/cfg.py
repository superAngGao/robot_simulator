"""
RL environment configuration dataclasses.

Pure data — no logic. All hyperparameters live here so Phase 2e can
swap backends without touching env code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class NoiseCfg:
    noise_type: str = "gaussian"  # "gaussian" | "uniform"
    std: float = 0.0
    low: float = 0.0
    high: float = 0.0


@dataclass
class ObsTermCfg:
    func: Callable  # fn(env, **params) -> torch.Tensor, shape (dim,)
    params: dict = field(default_factory=dict)
    noise: NoiseCfg | None = None


@dataclass
class EnvCfg:
    dt: float = 2e-4
    device: str = "cpu"
    episode_length: int = 1000
    obs_cfg: dict[str, ObsTermCfg] = field(default_factory=dict)
    # Controller hyperparameters
    kp: float = 20.0
    kd: float = 0.5
    action_scale: float = 0.5  # rad; NN output 1.0 → 0.5 rad offset
    action_clip: float | None = None  # None = no clip
    # Reset randomisation
    init_noise_scale: float = 0.0  # 0 = no noise
    # Custom controller (None → auto-build PDController)
    controller: Any | None = None
