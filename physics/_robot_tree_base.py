"""
Abstract base class defining the runtime interface for a robot kinematic tree.

RobotTreeBase specifies only the query / simulation interface (forward
kinematics, ABA, body velocities, etc.).  It intentionally omits
construction methods (add_body, finalize) because different backends may
build their trees differently (e.g. NumPy CPU, NVIDIA Warp GPU).

References:
  Featherstone (2008) §7 — articulated body algorithm interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

import numpy as np
from numpy.typing import NDArray

from .spatial import SpatialTransform, Vec6

if TYPE_CHECKING:
    from .robot_tree import Body


class RobotTreeBase(ABC):
    """Runtime interface for an articulated rigid body tree."""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def nq(self) -> int:
        """Total number of generalized position coordinates."""

    @property
    @abstractmethod
    def nv(self) -> int:
        """Total number of generalized velocity / force coordinates."""

    @property
    @abstractmethod
    def num_bodies(self) -> int:
        """Number of rigid bodies in the tree."""

    # ------------------------------------------------------------------
    # Runtime interface
    # ------------------------------------------------------------------

    @abstractmethod
    def forward_kinematics(self, q: NDArray[np.float64]) -> List[SpatialTransform]:
        """Return world-frame transform for every body."""

    @abstractmethod
    def aba(
        self,
        q: NDArray[np.float64],
        qdot: NDArray[np.float64],
        tau: NDArray[np.float64],
        external_forces=None,
    ) -> NDArray[np.float64]:
        """Compute generalised accelerations (Articulated Body Algorithm)."""

    @abstractmethod
    def body_velocities(
        self,
        q: NDArray[np.float64],
        qdot: NDArray[np.float64],
    ) -> List[Vec6]:
        """Return spatial velocity (body frame) for every body."""

    @abstractmethod
    def passive_torques(
        self,
        q: NDArray[np.float64],
        qdot: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return passive joint torques (limits + damping) as a (nv,) array."""

    @abstractmethod
    def default_state(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (q, qdot) for the zero / default configuration."""

    @abstractmethod
    def body_by_name(self, name: str) -> "Body":
        """Return the Body object with the given name."""

    # ------------------------------------------------------------------
    # CRBA (optional — concrete implementations may override)
    # ------------------------------------------------------------------

    def crba(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the joint-space mass matrix H (nv x nv) via CRBA.

        Reference: Featherstone (2008) §6.2.
        """
        raise NotImplementedError("CRBA not implemented for this backend.")

    def forward_dynamics_crba(
        self,
        q: NDArray[np.float64],
        qdot: NDArray[np.float64],
        tau: NDArray[np.float64],
        external_forces=None,
    ) -> NDArray[np.float64]:
        """Forward dynamics via CRBA: qddot = H^{-1} (tau - C).

        Equivalent to ABA but uses dense matrix factorisation.
        Better suited for GPU tensor-core acceleration on larger robots.
        """
        raise NotImplementedError("CRBA forward dynamics not implemented for this backend.")
