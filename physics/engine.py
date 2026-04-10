"""
PhysicsEngine — unified interface for CPU and GPU physics stepping.

The single divergence point between CPU and GPU: everything above
(Scene, Simulator, VecEnv) is shared, everything inside step() is
engine-specific.

    CpuEngine  : Python FK → GJK/EPA detect → PGS-SI solve → integrate
    GpuEngine  : fused GPU kernel(s) doing the same pipeline

Reference: MuJoCo mj_step (unified engine), Isaac Lab (engine-agnostic sim).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from numpy.typing import NDArray

if TYPE_CHECKING:
    from .dynamics_cache import ForceState
    from .merged_model import MergedModel


@dataclass
class ContactInfo:
    """One detected contact (read-only snapshot, shared by CPU and GPU engines)."""

    body_i: int
    body_j: int  # -1 = ground contact
    depth: float
    normal: NDArray  # (3,)
    point: NDArray  # (3,)


@dataclass
class StepOutput:
    """Unified output from PhysicsEngine.step() — convergence point for CPU/GPU.

    All arrays use merged global indices. Simulator uses RobotSlice to
    split back into per-robot results.
    """

    q_new: NDArray  # (nq,) or (N, nq)
    qdot_new: NDArray  # (nv,) or (N, nv)
    X_world: object  # List[SpatialTransform] (CPU) or Tensor (GPU)
    v_bodies: object  # List[Vec6] (CPU) or Tensor (GPU)
    contact_active: object  # bool array/tensor
    force_state: Optional["ForceState"] = None


class PhysicsEngine(ABC):
    """Abstract physics engine — the CPU/GPU divergence point.

    Encapsulates the full physics step: FK → collision detect →
    force computation → constraint solve → integration.

    Args:
        merged : MergedModel containing the unified multi-root tree.
    """

    def __init__(self, merged: "MergedModel") -> None:
        self.merged = merged

    @abstractmethod
    def step(
        self,
        q: NDArray,
        qdot: NDArray,
        tau: NDArray,
        dt: float,
    ) -> StepOutput:
        """Advance physics by one time step.

        Args:
            q    : Merged generalized positions (nq,).
            qdot : Merged generalized velocities (nv,).
            tau  : Merged generalized forces (nv,). Only actuated DOFs
                   should be non-zero; passive forces are added internally.
            dt   : Time step [s].

        Returns:
            StepOutput with new state and diagnostics.
        """

    @abstractmethod
    def query_contacts(self, env_idx: int = 0) -> List[ContactInfo]:
        """Return detected contacts from the most recent step().

        Args:
            env_idx : Environment index (GPU multi-env). Ignored by CpuEngine.

        Returns:
            List of ContactInfo snapshots. body_j == -1 means ground contact.
        """
