"""
Joint models for articulated rigid body systems.

Each joint type defines:
  - Its motion subspace matrix S  (6 x dof)
  - The joint transform X_J(q)    (Plücker transform from parent to child)
  - Velocity and acceleration contributions

Supported joint types:
  - RevoluteJoint  : 1 DOF rotation about a fixed axis
  - PrismaticJoint : 1 DOF translation along a fixed axis
  - FixedJoint     : 0 DOF rigid connection
  - FreeJoint      : 6 DOF floating base (for the root body)

References:
  Featherstone (2008), Chapter 4.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

from .spatial import (
    SpatialTransform,
    Vec3,
    Vec6,
    quat_to_rot,
    rot_x,
    rot_y,
    rot_z,
)

# ---------------------------------------------------------------------------
# Joint axis constants
# ---------------------------------------------------------------------------


class Axis(Enum):
    X = auto()
    Y = auto()
    Z = auto()


_AXIS_VEC: dict[Axis, Vec3] = {
    Axis.X: np.array([1.0, 0.0, 0.0]),
    Axis.Y: np.array([0.0, 1.0, 0.0]),
    Axis.Z: np.array([0.0, 0.0, 1.0]),
}

_AXIS_ROT = {
    Axis.X: rot_x,
    Axis.Y: rot_y,
    Axis.Z: rot_z,
}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Joint(ABC):
    """Abstract base class for all joint types."""

    #: Number of generalized position coordinates (may differ from DOF for
    #: quaternion-parameterised joints).
    nq: int
    #: Number of generalized velocity / force coordinates (true DOF).
    nv: int

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def transform(self, q: NDArray[np.float64]) -> SpatialTransform:
        """Return the joint transform X_J(q): child frame relative to parent."""

    @abstractmethod
    def motion_subspace(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the motion subspace matrix S(q) with shape (6, nv).

        The spatial velocity contributed by this joint is:  v_J = S @ q̇
        """

    @abstractmethod
    def bias_acceleration(
        self,
        q: NDArray[np.float64],
        qdot: NDArray[np.float64],
    ) -> Vec6:
        """Bias acceleration term c_J = Ṡ @ q̇  (often zero for simple joints)."""

    @abstractmethod
    def default_q(self) -> NDArray[np.float64]:
        """Return the zero / default configuration."""

    @abstractmethod
    def default_qdot(self) -> NDArray[np.float64]:
        """Return the zero velocity."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# Revolute joint  (1 DOF)
# ---------------------------------------------------------------------------


class RevoluteJoint(Joint):
    """Single-axis revolute (hinge) joint.

    q    : scalar angle [rad]
    qdot : scalar angular velocity [rad/s]

    Optional joint limits via a penalty spring-damper:
        q_min, q_max : angular limits [rad]
        k_limit      : spring stiffness  [N·m/rad]
        b_limit      : damping           [N·m·s/rad]

    The ``axis`` parameter accepts either an :class:`Axis` enum value
    (backward-compatible) or an arbitrary unit 3-vector ``NDArray``.
    Rotation is computed with Rodrigues' formula regardless of axis type.
    """

    nq = 1
    nv = 1

    def __init__(
        self,
        name: str,
        axis: Axis | NDArray[np.float64] = Axis.Z,
        q_min: float = -np.inf,
        q_max: float = np.inf,
        k_limit: float = 5_000.0,
        b_limit: float = 50.0,
        damping: float = 0.0,
    ) -> None:
        super().__init__(name)
        self.axis = axis
        if isinstance(axis, Axis):
            self._axis_vec: Vec3 = _AXIS_VEC[axis].copy()
        else:
            k = np.asarray(axis, dtype=np.float64).ravel()
            norm = np.linalg.norm(k)
            if norm < 1e-12:
                raise ValueError("Joint axis vector must be non-zero.")
            self._axis_vec = k / norm
        self._S: NDArray[np.float64] = np.zeros((6, 1), dtype=np.float64)
        # Motion subspace: pure rotation about axis (angular component in [3:])
        self._S[3:, 0] = self._axis_vec
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.k_limit = float(k_limit)
        self.b_limit = float(b_limit)
        self.damping = float(damping)

    def transform(self, q: NDArray[np.float64]) -> SpatialTransform:
        # Rodrigues' rotation formula: R = I cosθ + sinθ [k]× + (1−cosθ) k kᵀ
        # Works for arbitrary unit axes including X/Y/Z.
        angle = float(q[0])
        k = self._axis_vec
        c, s = np.cos(angle), np.sin(angle)
        K = np.array(
            [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
            dtype=np.float64,
        )
        R = c * np.eye(3) + s * K + (1.0 - c) * np.outer(k, k)
        return SpatialTransform(R, np.zeros(3))

    def motion_subspace(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._S  # constant for revolute joints

    def bias_acceleration(
        self,
        q: NDArray[np.float64],
        qdot: NDArray[np.float64],
    ) -> Vec6:
        return np.zeros(6, dtype=np.float64)  # Ṡ = 0 for fixed-axis joints

    def compute_limit_torque(
        self,
        q: NDArray[np.float64],
        qdot: NDArray[np.float64],
    ) -> float:
        """Return a penalty spring-damper torque when q violates [q_min, q_max].

        The torque pushes the joint back inside its limits:
          - Below q_min: positive torque (+ direction)
          - Above q_max: negative torque (- direction)
          - Damping opposes the velocity that would deepen the violation.
        Returns 0.0 when neither limit is active.
        """
        angle = float(q[0])
        omega = float(qdot[0])
        if angle < self.q_min:
            penetration = self.q_min - angle
            # damping only opposes motion that deepens penetration (ω < 0)
            return self.k_limit * penetration - self.b_limit * min(omega, 0.0)
        if angle > self.q_max:
            penetration = angle - self.q_max
            # damping only opposes motion that deepens penetration (ω > 0)
            return -(self.k_limit * penetration + self.b_limit * max(omega, 0.0))
        return 0.0

    def compute_damping_torque(self, qdot: NDArray[np.float64]) -> float:
        """Return viscous damping torque: −damping × qdot."""
        return -self.damping * float(qdot[0])

    def default_q(self) -> NDArray[np.float64]:
        return np.zeros(1, dtype=np.float64)

    def default_qdot(self) -> NDArray[np.float64]:
        return np.zeros(1, dtype=np.float64)


# ---------------------------------------------------------------------------
# Prismatic joint  (1 DOF)
# ---------------------------------------------------------------------------


class PrismaticJoint(Joint):
    """Single-axis prismatic (sliding) joint.

    q    : scalar displacement [m]
    qdot : scalar velocity [m/s]

    The ``axis`` parameter accepts either an :class:`Axis` enum value
    (backward-compatible) or an arbitrary unit 3-vector ``NDArray``.
    """

    nq = 1
    nv = 1

    def __init__(
        self,
        name: str,
        axis: Axis | NDArray[np.float64] = Axis.Z,
        damping: float = 0.0,
    ) -> None:
        super().__init__(name)
        self.axis = axis
        if isinstance(axis, Axis):
            self._axis_vec: Vec3 = _AXIS_VEC[axis].copy()
        else:
            k = np.asarray(axis, dtype=np.float64).ravel()
            norm = np.linalg.norm(k)
            if norm < 1e-12:
                raise ValueError("Joint axis vector must be non-zero.")
            self._axis_vec = k / norm
        self._S: NDArray[np.float64] = np.zeros((6, 1), dtype=np.float64)
        # Motion subspace: pure translation (linear component in [:3])
        self._S[:3, 0] = self._axis_vec
        self.damping = float(damping)

    def transform(self, q: NDArray[np.float64]) -> SpatialTransform:
        r = self._axis_vec * float(q[0])
        return SpatialTransform(np.eye(3), r)

    def motion_subspace(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._S

    def bias_acceleration(
        self,
        q: NDArray[np.float64],
        qdot: NDArray[np.float64],
    ) -> Vec6:
        return np.zeros(6, dtype=np.float64)

    def compute_damping_torque(self, qdot: NDArray[np.float64]) -> float:
        """Return viscous damping force: −damping × qdot."""
        return -self.damping * float(qdot[0])

    def default_q(self) -> NDArray[np.float64]:
        return np.zeros(1, dtype=np.float64)

    def default_qdot(self) -> NDArray[np.float64]:
        return np.zeros(1, dtype=np.float64)


# ---------------------------------------------------------------------------
# Fixed joint  (0 DOF)
# ---------------------------------------------------------------------------


class FixedJoint(Joint):
    """Zero-DOF rigid connection between two bodies."""

    nq = 0
    nv = 0

    def __init__(self, name: str, offset: SpatialTransform | None = None) -> None:
        super().__init__(name)
        self._X = offset if offset is not None else SpatialTransform.identity()

    def transform(self, q: NDArray[np.float64]) -> SpatialTransform:
        return self._X

    def motion_subspace(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros((6, 0), dtype=np.float64)

    def bias_acceleration(
        self,
        q: NDArray[np.float64],
        qdot: NDArray[np.float64],
    ) -> Vec6:
        return np.zeros(6, dtype=np.float64)

    def default_q(self) -> NDArray[np.float64]:
        return np.zeros(0, dtype=np.float64)

    def default_qdot(self) -> NDArray[np.float64]:
        return np.zeros(0, dtype=np.float64)


# ---------------------------------------------------------------------------
# Free joint  (6 DOF) — used for floating-base root body
# ---------------------------------------------------------------------------


class FreeJoint(Joint):
    """6-DOF free joint for the floating base of a robot.

    Position parameterisation (nq = 7):
        q = [qw, qx, qy, qz, x, y, z]   (quaternion + translation)

    Velocity parameterisation (nv = 6):
        qdot = [vx, vy, vz, ωx, ωy, ωz]  (spatial velocity, [linear; angular])
    """

    nq = 7
    nv = 6

    def __init__(self, name: str = "root") -> None:
        super().__init__(name)

    def transform(self, q: NDArray[np.float64]) -> SpatialTransform:
        quat = q[:4]
        pos = q[4:]
        R = quat_to_rot(quat)
        return SpatialTransform(R, pos)

    def motion_subspace(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.eye(6, dtype=np.float64)  # S = I_6

    def bias_acceleration(
        self,
        q: NDArray[np.float64],
        qdot: NDArray[np.float64],
    ) -> Vec6:
        return np.zeros(6, dtype=np.float64)

    def default_q(self) -> NDArray[np.float64]:
        # Identity quaternion (w=1) at origin
        q = np.zeros(7, dtype=np.float64)
        q[0] = 1.0  # qw
        return q

    def default_qdot(self) -> NDArray[np.float64]:
        return np.zeros(6, dtype=np.float64)

    def integrate_q(
        self,
        q: NDArray[np.float64],
        qdot: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        """Integrate quaternion state (avoids naive Euler on quaternions)."""
        vel = qdot[:3]
        omega = qdot[3:]
        qw, qx, qy, qz = q[:4]
        # Quaternion derivative: q̇ = 0.5 * Ω(ω) @ q
        dq = 0.5 * np.array(
            [
                -qx * omega[0] - qy * omega[1] - qz * omega[2],
                qw * omega[0] + qy * omega[2] - qz * omega[1],
                qw * omega[1] - qx * omega[2] + qz * omega[0],
                qw * omega[2] + qx * omega[1] - qy * omega[0],
            ]
        )
        quat_new = q[:4] + dq * dt
        quat_new /= np.linalg.norm(quat_new)
        pos_new = q[4:] + vel * dt
        return np.concatenate([quat_new, pos_new])
