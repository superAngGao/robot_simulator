"""
physics — Spatial algebra, joint models, articulated body dynamics, contact.
"""

from .spatial import (
    SpatialTransform,
    SpatialInertia,
    skew,
    rot_x, rot_y, rot_z,
    quat_to_rot, rot_to_quat,
    spatial_cross_velocity,
    spatial_cross_force,
    gravity_spatial,
)
from .joint import (
    Axis,
    Joint,
    RevoluteJoint,
    PrismaticJoint,
    FixedJoint,
    FreeJoint,
)
from .robot_tree import Body, RobotTree, KinematicState
from .contact import ContactParams, ContactPoint, ContactModel
from .integrator import SemiImplicitEuler, RK4, simulate
from .self_collision import BodyAABB, AABBSelfCollision

__all__ = [
    # spatial
    "SpatialTransform", "SpatialInertia",
    "skew", "rot_x", "rot_y", "rot_z",
    "quat_to_rot", "rot_to_quat",
    "spatial_cross_velocity", "spatial_cross_force", "gravity_spatial",
    # joints
    "Axis", "Joint",
    "RevoluteJoint", "PrismaticJoint", "FixedJoint", "FreeJoint",
    # tree
    "Body", "RobotTree", "KinematicState",
    # contact
    "ContactParams", "ContactPoint", "ContactModel",
    # integrators
    "SemiImplicitEuler", "RK4", "simulate",
    # self-collision
    "BodyAABB", "AABBSelfCollision",
]
