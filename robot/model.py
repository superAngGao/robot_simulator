"""
RobotModel dataclass — top-level container for a loaded robot description.

Bundles the kinematic tree, contact model, self-collision model, and
metadata produced by load_urdf() or manual construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from numpy.typing import NDArray

from physics.collision import SelfCollisionModel
from physics.collision_filter import CollisionFilter
from physics.contact import ContactModel
from physics.geometry import BodyCollisionGeometry
from physics.robot_tree import RobotTreeNumpy


@dataclass
class RobotModel:
    """All physics objects for one robot, ready for simulation.

    Attributes:
        tree                 : Kinematic tree (bodies, joints, ABA/FK).
        contact_model        : Ground-contact model (penalty or null).
        self_collision       : Self-collision model (AABB or null).
        collision_filter     : Collision filter (bitmask + exclude pairs).
        actuated_joint_names : Names of joints with nv > 0 and not FreeJoint.
        contact_body_names   : Body names used as contact points.
        geometries           : All BodyCollisionGeometry objects (one per link
                               that has non-mesh collision shapes).
    """

    tree: RobotTreeNumpy
    contact_model: ContactModel
    self_collision: SelfCollisionModel
    collision_filter: CollisionFilter | None = None
    actuated_joint_names: list[str] = field(default_factory=list)
    contact_body_names: list[str] = field(default_factory=list)
    geometries: list[BodyCollisionGeometry] = field(default_factory=list)
    effort_limits: NDArray | None = None  # shape (nu,), per actuated_joint_names order
