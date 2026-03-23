"""
GPU and CPU backend selection for batched physics simulation.

Usage::

    from physics.backends import get_backend

    backend = get_backend("numpy", model, cfg, num_envs)   # CPU fallback
    backend = get_backend("warp",  model, cfg, num_envs)   # NVIDIA Warp GPU
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .batch_backend import BatchBackend, StepResult

if TYPE_CHECKING:
    from rl_env.cfg import EnvCfg
    from robot.model import RobotModel

__all__ = ["get_backend", "BatchBackend", "StepResult"]


def get_backend(
    name: str,
    model: "RobotModel",
    cfg: "EnvCfg",
    num_envs: int,
) -> BatchBackend:
    """Factory: return a BatchBackend instance by name.

    Args:
        name     : ``"numpy"`` (CPU for-loop) or ``"warp"`` (NVIDIA Warp GPU).
        model    : Loaded RobotModel.
        cfg      : Environment configuration.
        num_envs : Number of parallel environments.
    """
    if name == "numpy":
        from .numpy_loop import NumpyLoopBackend

        return NumpyLoopBackend(model, cfg, num_envs)
    elif name == "warp":
        try:
            from .warp.warp_backend import WarpBatchBackend
        except ImportError as e:
            raise ImportError(
                "Warp backend requires `warp-lang` and `torch`. "
                "Install with: pip install warp-lang torch"
            ) from e
        return WarpBatchBackend(model, cfg, num_envs)
    else:
        raise ValueError(f"Unknown backend: {name!r}. Choose 'numpy' or 'warp'.")
