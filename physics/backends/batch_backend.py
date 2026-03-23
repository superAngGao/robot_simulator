"""
BatchBackend ABC — the interface between VecEnv and batched physics.

Every backend (NumPy for-loop, Warp GPU, TileLang GPU) implements this
interface so that VecEnv is backend-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class StepResult:
    """Output of a single batched physics step.

    All tensors have a leading batch dimension N (number of environments).
    Device matches the backend (CPU for NumPy, CUDA for Warp).
    """

    q: torch.Tensor  # (N, nq)
    qdot: torch.Tensor  # (N, nv)
    X_world: torch.Tensor  # (N, num_bodies, 12)  — R(3x3) + r(3) flattened
    v_bodies: torch.Tensor  # (N, num_bodies, 6)
    contact_mask: torch.Tensor  # (N, num_contacts) bool


class BatchBackend(ABC):
    """Abstract batched physics backend for VecEnv."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def reset_all(
        self,
        init_noise_scale: float = 0.0,
    ) -> StepResult:
        """Reset all N environments to the default state (+ optional noise).

        Returns a StepResult with the initial state and cached FK data.
        """

    @abstractmethod
    def reset_envs(
        self,
        env_ids: torch.Tensor,
        init_noise_scale: float = 0.0,
    ) -> None:
        """Reset specific environments (for auto-reset on episode end).

        Args:
            env_ids : 1-D int tensor of environment indices to reset.
        """

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    @abstractmethod
    def step_batch(
        self,
        actions: torch.Tensor,
    ) -> StepResult:
        """Run one physics step for all N environments.

        The backend owns the state (q, qdot) internally and updates it
        in-place.  ``actions`` shape is ``(N, nu)``.

        Returns a StepResult with post-step state and cached FK data.
        """

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    @abstractmethod
    def get_obs_data(self, result: StepResult) -> dict[str, torch.Tensor]:
        """Pre-slice observation data from a StepResult.

        Returns a dict with keys like ``'base_lin_vel'``, ``'joint_pos'``,
        etc., each a ``(N, dim)`` tensor ready for concatenation by
        BatchedObsManager.
        """

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def device(self) -> str:
        """``'cpu'`` or ``'cuda:0'``."""

    @property
    @abstractmethod
    def num_envs(self) -> int:
        """Number of parallel environments."""

    @property
    @abstractmethod
    def nq(self) -> int:
        """Total generalised position dimension."""

    @property
    @abstractmethod
    def nv(self) -> int:
        """Total generalised velocity dimension."""

    @property
    @abstractmethod
    def num_bodies(self) -> int:
        """Number of rigid bodies in the robot tree."""
