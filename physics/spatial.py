"""
Spatial algebra for rigid body dynamics.

Implements 6D spatial vectors and transforms following Featherstone (2008).
Spatial vectors combine rotational and translational components into a unified
6D representation, enabling compact and efficient multi-body dynamics.

Conventions:
  - Spatial velocity:  v = [ω; v]  (angular first, then linear)
  - Spatial force:     f = [τ; f]  (torque first, then force)
  - Coordinate frames: right-handed, z-up
  - Rotation matrices: R such that v_b = R @ v_a transforms from frame a to b

References:
  Featherstone, R. (2008). Rigid Body Dynamics Algorithms. Springer.
  http://royfeatherstone.org/spatial/
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

Vec3 = NDArray[np.float64]  # shape (3,)
Vec6 = NDArray[np.float64]  # shape (6,)
Mat3 = NDArray[np.float64]  # shape (3, 3)
Mat6 = NDArray[np.float64]  # shape (6, 6)


# ---------------------------------------------------------------------------
# Rotation utilities
# ---------------------------------------------------------------------------


def skew(v: Vec3) -> Mat3:
    """Return the 3x3 skew-symmetric matrix of vector v.

    Satisfies: skew(v) @ u == np.cross(v, u)
    """
    x, y, z = v
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def rot_x(angle: float) -> Mat3:
    """Rotation matrix about the x-axis by `angle` radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(angle: float) -> Mat3:
    """Rotation matrix about the y-axis by `angle` radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(angle: float) -> Mat3:
    """Rotation matrix about the z-axis by `angle` radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def quat_to_rot(q: NDArray[np.float64]) -> Mat3:
    """Convert unit quaternion [w, x, y, z] to rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rot_to_quat(R: Mat3) -> NDArray[np.float64]:
    """Convert rotation matrix to unit quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# Spatial transforms  (Plücker coordinate transforms)
# ---------------------------------------------------------------------------


class SpatialTransform:
    """Rigid-body coordinate transform in Plücker coordinates.

    Represents the transform X from frame A to frame B, defined by:
      - R : 3x3 rotation matrix  (columns of R are axes of A expressed in B)
      - r : 3D translation vector (origin of A expressed in B)

    The 6x6 Plücker transform matrix is:
        X = [ R,       0  ]
            [ -R@skew(r), R ]

    Applied to spatial velocity:   v_B = X @ v_A
    Applied to spatial force:      f_A = X.T @ f_B  (dual transform)
    """

    __slots__ = ("R", "r")

    def __init__(self, R: Mat3, r: Vec3) -> None:
        self.R: Mat3 = np.asarray(R, dtype=np.float64)
        self.r: Vec3 = np.asarray(r, dtype=np.float64)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def identity(cls) -> "SpatialTransform":
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_rotation(cls, R: Mat3) -> "SpatialTransform":
        return cls(R, np.zeros(3))

    @classmethod
    def from_translation(cls, r: Vec3) -> "SpatialTransform":
        return cls(np.eye(3), np.asarray(r, dtype=np.float64))

    @classmethod
    def from_rpy(cls, roll: float, pitch: float, yaw: float, r: Vec3 | None = None) -> "SpatialTransform":
        R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
        return cls(R, np.zeros(3) if r is None else np.asarray(r, dtype=np.float64))

    # ------------------------------------------------------------------
    # Matrix form
    # ------------------------------------------------------------------

    def matrix(self) -> Mat6:
        """Return the 6x6 spatial velocity transform matrix.

        Satisfies: matrix() @ v == apply_velocity(v)
        and:       matrix().T @ f == apply_force(f)
        """
        E = self.R.T  # parent→child rotation
        r = self.r
        X = np.zeros((6, 6), dtype=np.float64)
        X[:3, :3] = E
        X[3:, :3] = -E @ skew(r)
        X[3:, 3:] = E
        return X

    def matrix_dual(self) -> Mat6:
        """Return the dual (force) transform matrix X* = X^{-T}."""
        return self.matrix().T

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def apply_velocity(self, v: Vec6) -> Vec6:
        """Transform a spatial velocity from parent frame to child frame.

        Convention: R is child→parent rotation, r is child origin in parent frame.
        SE3 formula: omega_child = R.T @ omega_parent
                     v_child     = R.T @ (v_parent + omega_parent × r)
        """
        R, r = self.R, self.r
        omega = v[:3]
        vel = v[3:]
        return np.concatenate(
            [
                R.T @ omega,
                R.T @ (vel + np.cross(omega, r)),
            ]
        )

    def apply_force(self, f: Vec6) -> Vec6:
        """Transform a spatial force from child frame to parent frame.

        Convention: R is child→parent rotation, r is child origin in parent frame.
        SE3 formula: f_parent   = R @ f_child
                     tau_parent = R @ tau_child + r × (R @ f_child)
        """
        R, r = self.R, self.r
        tau = f[:3]
        frc = f[3:]
        f_parent = R @ frc
        return np.concatenate(
            [
                R @ tau + np.cross(r, f_parent),
                f_parent,
            ]
        )

    def inverse(self) -> "SpatialTransform":
        """Return the inverse transform (child→parent becomes parent→child)."""
        Rt = self.R.T
        return SpatialTransform(Rt, -(Rt @ self.r))

    def compose(self, other: "SpatialTransform") -> "SpatialTransform":
        """Compose transforms: self * other  (apply other first, then self).

        SE3 formula: R = self.R @ other.R
                     r = self.r + self.R @ other.r
        """
        R = self.R @ other.R
        r = self.r + self.R @ other.r
        return SpatialTransform(R, r)

    def __matmul__(self, other: "SpatialTransform") -> "SpatialTransform":
        return self.compose(other)

    def __repr__(self) -> str:
        return f"SpatialTransform(R=\n{self.R},\nr={self.r})"


# ---------------------------------------------------------------------------
# Spatial inertia
# ---------------------------------------------------------------------------


class SpatialInertia:
    """Spatial (6x6) inertia matrix of a rigid body.

    Expressed at the body's center of mass frame:
        I_spatial = [ I_com,      0    ]
                    [   0,    m * E_3  ]

    where I_com is the 3x3 rotational inertia about the CoM.

    When expressed at an arbitrary point p (offset from CoM):
        I_spatial(p) = X(p)^T @ I_spatial_com @ X(p)
    """

    __slots__ = ("mass", "inertia", "com")

    def __init__(self, mass: float, inertia: Mat3, com: Vec3) -> None:
        """
        Args:
            mass:    Scalar mass [kg].
            inertia: 3x3 rotational inertia tensor about CoM [kg·m²].
            com:     Center of mass position in body frame [m].
        """
        self.mass = float(mass)
        self.inertia = np.asarray(inertia, dtype=np.float64)
        self.com = np.asarray(com, dtype=np.float64)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_box(cls, mass: float, lx: float, ly: float, lz: float) -> "SpatialInertia":
        """Uniform solid box with dimensions lx × ly × lz."""
        ixx = mass / 12.0 * (ly**2 + lz**2)
        iyy = mass / 12.0 * (lx**2 + lz**2)
        izz = mass / 12.0 * (lx**2 + ly**2)
        return cls(mass, np.diag([ixx, iyy, izz]), np.zeros(3))

    @classmethod
    def from_cylinder(cls, mass: float, radius: float, length: float) -> "SpatialInertia":
        """Uniform solid cylinder along z-axis."""
        ixy = mass / 12.0 * (3 * radius**2 + length**2)
        izz = 0.5 * mass * radius**2
        return cls(mass, np.diag([ixy, ixy, izz]), np.zeros(3))

    @classmethod
    def point_mass(cls, mass: float, pos: Vec3) -> "SpatialInertia":
        """Point mass at position pos."""
        return cls(mass, np.zeros((3, 3)), np.asarray(pos, dtype=np.float64))

    # ------------------------------------------------------------------
    # Matrix form
    # ------------------------------------------------------------------

    def matrix(self) -> Mat6:
        """Return the 6x6 spatial inertia matrix expressed at body origin."""
        m = self.mass
        c = self.com
        I = self.inertia
        C = skew(c)
        top_left = I + m * (C @ C.T)  # shifted inertia (parallel axis)
        top_right = m * C
        bot_left = m * C.T
        bot_right = m * np.eye(3)
        M = np.zeros((6, 6), dtype=np.float64)
        M[:3, :3] = top_left
        M[:3, 3:] = top_right
        M[3:, :3] = bot_left
        M[3:, 3:] = bot_right
        return M

    def __add__(self, other: "SpatialInertia") -> "SpatialInertia":
        """Combine two spatial inertias (e.g. for composite body)."""
        m1, m2 = self.mass, other.mass
        m = m1 + m2
        if m < 1e-12:
            return SpatialInertia(0.0, np.zeros((3, 3)), np.zeros(3))
        com = (m1 * self.com + m2 * other.com) / m
        # Parallel axis theorem for combined inertia
        d1 = self.com - com
        d2 = other.com - com
        I = (
            self.inertia
            + m1 * (np.dot(d1, d1) * np.eye(3) - np.outer(d1, d1))
            + other.inertia
            + m2 * (np.dot(d2, d2) * np.eye(3) - np.outer(d2, d2))
        )
        return SpatialInertia(m, I, com)

    def __repr__(self) -> str:
        return f"SpatialInertia(mass={self.mass:.3f}, com={self.com})"


# ---------------------------------------------------------------------------
# Spatial vector helpers
# ---------------------------------------------------------------------------


def spatial_cross_velocity(v: Vec6) -> Mat6:
    """Spatial cross-product operator for velocity vectors.

    Returns the 6x6 matrix vcross such that:
        vcross @ u == spatial_cross(v, u)

    Used in the equation of motion:  f = I*a + v x* (I*v)
    """
    omega = v[:3]
    vel = v[3:]
    top_left = skew(omega)
    top_right = np.zeros((3, 3))
    bot_left = skew(vel)
    bot_right = skew(omega)
    M = np.zeros((6, 6), dtype=np.float64)
    M[:3, :3] = top_left
    M[:3, 3:] = top_right
    M[3:, :3] = bot_left
    M[3:, 3:] = bot_right
    return M


def spatial_cross_force(v: Vec6) -> Mat6:
    """Spatial cross-product operator for force vectors (dual).

    Returns the 6x6 matrix such that:
        M @ f == v x* f   (force cross product)
    """
    return -spatial_cross_velocity(v).T


def gravity_spatial(g: float = 9.81) -> Vec6:
    """Spatial gravity vector (acceleration of free fall, z-up convention)."""
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, -g], dtype=np.float64)
