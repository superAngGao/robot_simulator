"""
Self-collision detection and response models.

Provides an abstract SelfCollisionModel interface and concrete
implementations:
  - AABBSelfCollision : penalty-based AABB broad phase (moved from
                        self_collision.py — import from here going forward)
  - NullSelfCollision : no-op model for debugging / headless tests

Design notes for AABBSelfCollision
------------------------------------
- Each body's OBB (oriented bounding box, defined in the body frame) is
  conservatively projected to a world-frame AABB each step using:
      world_half[i] = Σ_j |R[i,j]| * local_half[j]
- Adjacent bodies in the kinematic tree (direct parent-child pairs) are
  automatically excluded — they are always touching by construction.
- The contact force is applied at each body's origin (zero moment arm),
  which is a valid first-order approximation for diffuse bodies.
- No friction is modelled for self-contact; only a normal penalty force.

References:
  Featherstone (2008) §2.1 — bounding-box conventions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from .spatial import SpatialTransform, Vec6

if TYPE_CHECKING:
    from .geometry import BodyCollisionGeometry


# ---------------------------------------------------------------------------
# Per-body AABB descriptor
# ---------------------------------------------------------------------------


@dataclass
class BodyAABB:
    """Axis-aligned bounding box (in local body frame) for a single body.

    Attributes:
        body_index   : Index of the body in the RobotTree body list.
        half_extents : Half-size along each local axis [m].  Shape (3,).
    """

    body_index: int
    half_extents: NDArray[np.float64]  # (3,)


# ---------------------------------------------------------------------------
# World-frame AABB helper
# ---------------------------------------------------------------------------


def _world_aabb(
    babb: BodyAABB,
    X_world: SpatialTransform,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (min_corner, max_corner) of the world-frame AABB.

    Converts the OBB (body-frame box rotated by X_world.R) to a
    conservative world AABB using the standard OBB projection formula:

        world_half[i] = Σ_j |R[i,j]| * local_half[j]
    """
    center = X_world.r.copy()
    world_half = np.abs(X_world.R) @ babb.half_extents
    return center - world_half, center + world_half


# ---------------------------------------------------------------------------
# Abstract self-collision model
# ---------------------------------------------------------------------------


class SelfCollisionModel(ABC):
    """Abstract base class for self-collision detection and response."""

    @abstractmethod
    def compute_forces(
        self,
        X_world_list: List[SpatialTransform],
        v_body_list: Optional[List[NDArray]],
        num_bodies: int,
    ) -> List[Vec6]:
        """Compute spatial self-collision penalty forces for all bodies.

        Returns:
            List of spatial force vectors (6,) in body frame, one per body.
        """

    @classmethod
    @abstractmethod
    def from_geometries(
        cls,
        geometries: "List[BodyCollisionGeometry]",
        parent_list: List[int],
        **kwargs,
    ) -> "SelfCollisionModel":
        """Construct a model from a list of BodyCollisionGeometry objects."""


# ---------------------------------------------------------------------------
# AABB-based self-collision model
# ---------------------------------------------------------------------------


class AABBSelfCollision(SelfCollisionModel):
    """Penalty-based self-collision model using body AABBs.

    Typical usage::

        sc = AABBSelfCollision(k_contact=2_000.0, b_contact=100.0)
        sc.add_body(BodyAABB(0, np.array([0.175, 0.10, 0.05])))  # torso
        sc.add_body(BodyAABB(3, np.array([0.01, 0.01, 0.10])))   # calf FL
        ...
        # Call once after all bodies are registered:
        sc.build_pairs(parent_list=[body.parent for body in tree.bodies])

        # Each simulation step:
        self_forces = sc.compute_forces(X_world_list, v_body_list, tree.num_bodies)

    Parameters
    ----------
    k_contact : Normal spring stiffness [N/m].
    b_contact : Normal damping [N·s/m].  Applied when the relative velocity
                of the body centres deepens the penetration.
    """

    def __init__(
        self,
        k_contact: float = 2_000.0,
        b_contact: float = 100.0,
    ) -> None:
        self.k_contact = float(k_contact)
        self.b_contact = float(b_contact)
        self._aabbs: List[BodyAABB] = []
        self._pairs: List[Tuple[int, int]] = []  # indices into self._aabbs

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_body(self, babb: BodyAABB) -> None:
        """Register a body's AABB."""
        self._aabbs.append(babb)

    def build_pairs(self, parent_list: List[int]) -> None:
        """Build the list of collision candidate pairs.

        Adjacent pairs (direct parent-child in the kinematic tree) are
        excluded because those bodies are geometrically attached and will
        always "overlap" near their joint.

        Args:
            parent_list: ``parent_list[i]`` is the parent body index of
                         body *i* in the tree (-1 for the root body).
        """
        # Build adjacency set (undirected edges in the kinematic tree)
        adjacent: Set[Tuple[int, int]] = set()
        for child_idx, parent_idx in enumerate(parent_list):
            if parent_idx >= 0:
                a = min(child_idx, parent_idx)
                b = max(child_idx, parent_idx)
                adjacent.add((a, b))

        n = len(self._aabbs)
        self._pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                bi = self._aabbs[i].body_index
                bj = self._aabbs[j].body_index
                edge = (min(bi, bj), max(bi, bj))
                if edge not in adjacent:
                    self._pairs.append((i, j))

    # ------------------------------------------------------------------
    # Force computation
    # ------------------------------------------------------------------

    def compute_forces(
        self,
        X_world_list: List[SpatialTransform],
        v_body_list: Optional[List[NDArray]] = None,
        num_bodies: int = 0,
    ) -> List[Vec6]:
        """Compute spatial self-collision penalty forces for all bodies.

        Args:
            X_world_list : World-frame transforms for every body.
            v_body_list  : Optional spatial velocities (body frame) for
                           velocity-dependent damping.  Pass ``None`` to
                           skip damping.
            num_bodies   : Total number of bodies in the tree.

        Returns:
            List of spatial force vectors (6,) in *body frame*, one per body.
            Non-colliding bodies receive a zero vector.
        """
        n = num_bodies or len(X_world_list)
        forces: List[Vec6] = [np.zeros(6, dtype=np.float64) for _ in range(n)]

        for i_idx, j_idx in self._pairs:
            babb_i = self._aabbs[i_idx]
            babb_j = self._aabbs[j_idx]
            bi = babb_i.body_index
            bj = babb_j.body_index

            Xi = X_world_list[bi]
            Xj = X_world_list[bj]

            min_i, max_i = _world_aabb(babb_i, Xi)
            min_j, max_j = _world_aabb(babb_j, Xj)

            # Per-axis overlap (positive = penetrating)
            overlap = np.minimum(max_i, max_j) - np.maximum(min_i, min_j)
            if np.any(overlap <= 0.0):
                continue  # separated on at least one axis → no contact

            # Axis of minimum penetration (MTV direction)
            k_axis = int(np.argmin(overlap))
            depth = overlap[k_axis]

            # Push i away from j along the MTV axis
            sep = Xi.r - Xj.r
            direction = np.zeros(3, dtype=np.float64)
            direction[k_axis] = 1.0 if sep[k_axis] >= 0.0 else -1.0

            # Spring force magnitude
            F_mag = self.k_contact * depth

            # Optional velocity damping along the contact direction
            if v_body_list is not None:
                vi_world = Xi.R @ v_body_list[bi][3:]  # linear vel of body i in world
                vj_world = Xj.R @ v_body_list[bj][3:]
                v_rel = np.dot(vi_world - vj_world, direction)
                if v_rel < 0.0:  # bodies approaching → add damping
                    F_mag -= self.b_contact * v_rel

            F_world = direction * F_mag

            # Spatial force: zero torque (applied at body origin)
            f_sw_i = np.concatenate([np.zeros(3), F_world])
            f_sw_j = np.concatenate([np.zeros(3), -F_world])

            forces[bi] += Xi.inverse().apply_force(f_sw_i)
            forces[bj] += Xj.inverse().apply_force(f_sw_j)

        return forces

    # ------------------------------------------------------------------
    # Class method constructor from geometry descriptors
    # ------------------------------------------------------------------

    @classmethod
    def from_geometries(
        cls,
        geometries: "List[BodyCollisionGeometry]",
        parent_list: List[int],
        **kwargs,
    ) -> "AABBSelfCollision":
        """Construct from BodyCollisionGeometry objects.

        Args:
            geometries  : One BodyCollisionGeometry per body to include.
            parent_list : parent_list[i] is the parent body index of body i.
            **kwargs    : Forwarded to ``__init__`` (e.g. k_contact, b_contact).
        """
        obj = cls(**kwargs)
        for geom in geometries:
            obj.add_body(BodyAABB(geom.body_index, geom.aabb_half_extents()))
        obj.build_pairs(parent_list)
        return obj

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def num_pairs(self) -> int:
        return len(self._pairs)

    def __repr__(self) -> str:
        return f"AABBSelfCollision(bodies={len(self._aabbs)}, pairs={len(self._pairs)}, k={self.k_contact})"


# ---------------------------------------------------------------------------
# Null self-collision model (for debugging / no-collision scenarios)
# ---------------------------------------------------------------------------


class NullSelfCollision(SelfCollisionModel):
    """Self-collision model that always returns zero forces (no collisions)."""

    def compute_forces(
        self,
        X_world_list: List[SpatialTransform],
        v_body_list: Optional[List[NDArray]] = None,
        num_bodies: int = 0,
    ) -> List[Vec6]:
        n = num_bodies or len(X_world_list)
        return [np.zeros(6, dtype=np.float64) for _ in range(n)]

    @classmethod
    def from_geometries(
        cls,
        geometries: "List[BodyCollisionGeometry]",
        parent_list: List[int],
        **kwargs,
    ) -> "NullSelfCollision":
        return cls()

    def __repr__(self) -> str:
        return "NullSelfCollision()"
