"""
GPU and CPU backend selection for batched physics simulation.

Usage::

    from physics.backends import get_backend

    backend = get_backend("numpy",    model, cfg, num_envs)   # CPU fallback
    backend = get_backend("tilelang", model, cfg, num_envs)   # PyTorch CUDA
    backend = get_backend("cuda",     model, cfg, num_envs)   # CUDA C++ fused
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
        name     : ``"numpy"`` | ``"tilelang"`` | ``"cuda"``
        model    : Loaded RobotModel.
        cfg      : Environment configuration.
        num_envs : Number of parallel environments.
    """
    if name == "numpy":
        from .numpy_loop import NumpyLoopBackend

        return NumpyLoopBackend(model, cfg, num_envs)
    elif name == "tilelang":
        try:
            from .tilelang.tilelang_backend import TileLangBatchBackend
        except ImportError as e:
            raise ImportError(
                "TileLang backend requires `tilelang` and `torch`. Install with: pip install tilelang torch"
            ) from e
        return TileLangBatchBackend(model, cfg, num_envs)
    elif name in ("cuda", "cuda_crba", "cuda_crba_tc", "cuda_grouped_schur"):
        try:
            from .cuda.cuda_backend import CudaBatchBackend
        except Exception as e:
            raise ImportError(
                f"CUDA backend requires PyTorch with CUDA and a C++ compiler. Error: {e}"
            ) from e
        if name == "cuda_grouped_schur":
            dynamics = "grouped_schur"
        elif name == "cuda_crba_tc":
            dynamics = "crba_tc"
        elif name == "cuda_crba":
            dynamics = "crba"
        else:
            dynamics = "aba"
        return CudaBatchBackend(model, cfg, num_envs, dynamics=dynamics)
    else:
        raise ValueError(f"Unknown backend: {name!r}. Choose 'numpy', 'tilelang', or 'cuda'.")
