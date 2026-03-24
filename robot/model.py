"""
RobotModel dataclass — robot description (tree + geometry + metadata).

Pure robot description: does NOT contain contact/collision models.
Those are managed at the Scene level by CollisionPipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from numpy.typing import NDArray

from physics.geometry import BodyCollisionGeometry
from physics.robot_tree import RobotTreeNumpy


@dataclass
class RobotModel:
    """All physics objects for one robot, ready for simulation.

    Attributes:
        tree                 : Kinematic tree (bodies, joints, ABA/FK).
        actuated_joint_names : Names of joints with nv > 0 and not FreeJoint.
        contact_body_names   : Body names designated as contact points.
        geometries           : BodyCollisionGeometry objects (one per link
                               with non-mesh collision shapes).
        effort_limits        : Per actuated-joint effort limits, shape (nu,).
    """

    tree: RobotTreeNumpy
    actuated_joint_names: list[str] = field(default_factory=list)
    contact_body_names: list[str] = field(default_factory=list)
    geometries: list[BodyCollisionGeometry] = field(default_factory=list)
    effort_limits: NDArray | None = None
