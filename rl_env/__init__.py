"""rl_env — Gymnasium-compatible RL environments (Phase 2+).

VecEnv was removed in Q31 refactor. Batched RL training will use
the new Manager-based RLEnv backed by GpuEngine (coming next).
"""

from .base_env import Env
from .cfg import EnvCfg, NoiseCfg, ObsTermCfg
from .controllers import Controller, PDController, TorqueController

__all__ = [
    "Env",
    "EnvCfg",
    "ObsTermCfg",
    "NoiseCfg",
    "Controller",
    "PDController",
    "TorqueController",
]
