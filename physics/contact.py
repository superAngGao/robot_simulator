"""
Contact dynamics for legged robots using the penalty (spring-damper) method.

The penalty method models ground contact as a compliant surface:
  - When a contact point penetrates the ground (z < 0), a reaction force
    is generated proportional to penetration depth and velocity.
  - Friction is modelled with a regularised Coulomb cone to avoid
    discontinuities at zero slip velocity.

This is simpler than LCP / impulse methods but sufficient for Phase 1.
Parameters (stiffness, damping, friction) should be tuned to match the
real robot's foot-ground interaction.

Contact model
-------------
Normal force (z):
    F_n = max(0,  k_n * δ  +  b_n * δ̇)
    where δ = -z_contact  (penetration depth, positive when in contact)

Tangential force (x, y) — regularised Coulomb:
    v_slip = [vx, vy]  (slip velocity at contact point)
    F_t = -μ * F_n * v_slip / sqrt(|v_slip|² + ε²)

References:
  Mirtich & Canny (1995). Impulse-based simulation of rigid bodies.
  Azad & Featherstone (2014). A new penalty method for contacts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np

from .spatial import SpatialTransform, Vec3, Vec6
from .terrain import FlatTerrain, Terrain

# ---------------------------------------------------------------------------
# Contact parameters
# ---------------------------------------------------------------------------


@dataclass
class ContactParams:
    """Tunable parameters for the spring-damper contact model.

    Attributes:
        k_normal  : Normal stiffness  [N/m].   Typical: 1e4 – 1e5
        b_normal  : Normal damping    [N·s/m]. Typical: 1e2 – 1e3
        mu        : Coulomb friction coefficient.
        slip_eps  : Regularisation for Coulomb cone [m/s].
        ground_z  : Height of the ground plane [m] (default 0).
    """

    k_normal: float = 5_000.0
    b_normal: float = 500.0
    mu: float = 0.8
    slip_eps: float = 1e-3
    ground_z: float = 0.0


# ---------------------------------------------------------------------------
# Contact point definition
# ---------------------------------------------------------------------------


@dataclass
class ContactPoint:
    """A single contact candidate point (e.g. foot tip).

    Attributes:
        body_index  : Index of the body this point is attached to.
        position    : Position of the contact point in the body's local frame [m].
        name        : Optional label (e.g. "FL_foot").
    """

    body_index: int
    position: Vec3
    name: str = ""

    def world_position(self, X_world: SpatialTransform) -> Vec3:
        """Return contact point position in the world frame."""
        # X_world.R maps body→world, X_world.r is origin of body in world
        return X_world.R @ self.position + X_world.r

    def world_velocity(
        self,
        X_world: SpatialTransform,
        v_body: Vec6,
    ) -> Vec3:
        """Return contact point velocity in the world frame.

        Uses the rigid-body velocity formula:
            v_point = v_linear + ω × r_point
        """
        omega = X_world.R @ v_body[:3]  # angular vel in world
        v_lin = X_world.R @ v_body[3:]  # linear vel of body origin in world
        r_world = X_world.R @ self.position
        return v_lin + np.cross(omega, r_world)


# ---------------------------------------------------------------------------
# Abstract contact model
# ---------------------------------------------------------------------------


class ContactModel(ABC):
    """Abstract base class for ground-contact models."""

    @abstractmethod
    def compute_forces(
        self,
        X_world_list: List[SpatialTransform],
        v_body_list: List[Vec6],
        num_bodies: int,
    ) -> List[Vec6]:
        """Compute spatial contact forces for all bodies.

        Returns:
            List of spatial force vectors (6,) in body frame, one per body.
        """

    @abstractmethod
    def active_contacts(
        self,
        X_world_list: List[SpatialTransform],
    ) -> List[tuple[str, Vec3]]:
        """Return list of (name, world_position) for currently active contacts."""


# ---------------------------------------------------------------------------
# Penalty contact model (concrete)
# ---------------------------------------------------------------------------


class PenaltyContactModel(ContactModel):
    """Penalty-based contact model for a set of contact points.

    Computes the spatial forces acting on each body due to ground contact.
    Forces are expressed in the respective body frame (ready for ABA).

    Args:
        params  : Contact parameters (stiffness, damping, friction).
        terrain : Optional Terrain object.  If None, a FlatTerrain at
                  ``params.ground_z`` is used (backward-compatible default).
    """

    def __init__(
        self,
        params: ContactParams | None = None,
        terrain: Terrain | None = None,
    ) -> None:
        self.params = params or ContactParams()
        self._terrain: Terrain = terrain if terrain is not None else FlatTerrain(self.params.ground_z)
        self._contact_points: List[ContactPoint] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_contact_point(self, cp: ContactPoint) -> None:
        """Register a contact candidate point."""
        self._contact_points.append(cp)

    # ------------------------------------------------------------------
    # Force computation
    # ------------------------------------------------------------------

    def compute_forces(
        self,
        X_world_list: List[SpatialTransform],
        v_body_list: List[Vec6],
        num_bodies: int,
    ) -> List[Vec6]:
        """Compute spatial contact forces for all bodies.

        Args:
            X_world_list : World-frame transforms for each body (from FK).
            v_body_list  : Spatial velocities in body frame for each body.
            num_bodies   : Total number of bodies in the tree.

        Returns:
            List of spatial force vectors (6,) in body frame, one per body.
            Zero for bodies with no active contact.
        """
        forces: List[Vec6] = [np.zeros(6, dtype=np.float64) for _ in range(num_bodies)]
        p = self.params

        for cp in self._contact_points:
            bi = cp.body_index
            X = X_world_list[bi]
            v = v_body_list[bi]

            pos_world = cp.world_position(X)
            ground_z = self._terrain.height_at(pos_world[0], pos_world[1])
            depth = ground_z - pos_world[2]  # positive when penetrating

            if depth <= 0.0:
                continue  # not in contact

            vel_world = cp.world_velocity(X, v)

            # --- Normal force ---
            v_normal = vel_world[2]  # z-component of contact velocity
            F_normal = p.k_normal * depth - p.b_normal * v_normal
            F_normal = max(0.0, F_normal)  # unilateral constraint

            # --- Tangential (friction) force ---
            v_slip = vel_world[:2]  # x, y slip velocity
            slip_norm = np.sqrt(v_slip @ v_slip + p.slip_eps**2)
            F_tangent = -p.mu * F_normal * v_slip / slip_norm

            # Force in world frame
            F_world = np.array([F_tangent[0], F_tangent[1], F_normal], dtype=np.float64)

            # Convert to spatial force in body frame
            # Torque arm: r = pos_world − body_origin_world
            body_origin_world = X.r
            r_world = pos_world - body_origin_world

            torque_world = np.cross(r_world, F_world)

            # Spatial force in world frame = [torque; force]
            f_spatial_world = np.concatenate([torque_world, F_world])
            # Transform to body frame: X_world maps body→world,
            # so X_world.inverse() maps world→body.
            f_spatial_body = X.inverse().apply_force(f_spatial_world)

            forces[bi] += f_spatial_body

        return forces

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def active_contacts(
        self,
        X_world_list: List[SpatialTransform],
    ) -> List[tuple[str, Vec3]]:
        """Return list of (name, world_position) for currently active contacts."""
        active = []
        for cp in self._contact_points:
            X = X_world_list[cp.body_index]
            pos = cp.world_position(X)
            ground_z = self._terrain.height_at(pos[0], pos[1])
            if pos[2] <= ground_z:
                active.append((cp.name, pos))
        return active

    @property
    def contact_points(self) -> List[ContactPoint]:
        return list(self._contact_points)

    def __repr__(self) -> str:
        return (
            f"PenaltyContactModel(points={len(self._contact_points)}, "
            f"k={self.params.k_normal}, mu={self.params.mu})"
        )


# ---------------------------------------------------------------------------
# Null contact model (for debugging / no-contact scenarios)
# ---------------------------------------------------------------------------


class NullContactModel(ContactModel):
    """Contact model that always returns zero forces (no contact)."""

    def compute_forces(
        self,
        X_world_list: List[SpatialTransform],
        v_body_list: List[Vec6],
        num_bodies: int,
    ) -> List[Vec6]:
        return [np.zeros(6, dtype=np.float64) for _ in range(num_bodies)]

    def active_contacts(
        self,
        X_world_list: List[SpatialTransform],
    ) -> List[tuple[str, Vec3]]:
        return []

    def __repr__(self) -> str:
        return "NullContactModel()"
