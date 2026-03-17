"""
physics — Spatial algebra, joint models, articulated body dynamics, contact.
"""

from ._robot_tree_base import RobotTreeBase
from .collision import (
    AABBSelfCollision,
    BodyAABB,
    NullSelfCollision,
    SelfCollisionModel,
)
from .contact import (
    ContactModel,
    ContactParams,
    ContactPoint,
    NullContactModel,
    PenaltyContactModel,
)
from .geometry import (
    BodyCollisionGeometry,
    BoxShape,
    CollisionShape,
    CylinderShape,
    MeshShape,
    ShapeInstance,
    SphereShape,
)
from .integrator import RK4, SemiImplicitEuler, simulate
from .joint import (
    Axis,
    FixedJoint,
    FreeJoint,
    Joint,
    PrismaticJoint,
    RevoluteJoint,
)
from .robot_tree import Body, KinematicState, RobotTree, RobotTreeNumpy
from .spatial import (
    SpatialInertia,
    SpatialTransform,
    gravity_spatial,
    quat_to_rot,
    rot_to_quat,
    rot_x,
    rot_y,
    rot_z,
    skew,
    spatial_cross_force,
    spatial_cross_velocity,
)
from .terrain import FlatTerrain, HeightmapTerrain, Terrain

__all__ = [
    # spatial
    "SpatialTransform",
    "SpatialInertia",
    "skew",
    "rot_x",
    "rot_y",
    "rot_z",
    "quat_to_rot",
    "rot_to_quat",
    "spatial_cross_velocity",
    "spatial_cross_force",
    "gravity_spatial",
    # joints
    "Axis",
    "Joint",
    "RevoluteJoint",
    "PrismaticJoint",
    "FixedJoint",
    "FreeJoint",
    # tree
    "Body",
    "RobotTree",
    "RobotTreeNumpy",
    "RobotTreeBase",
    "KinematicState",
    # contact
    "ContactParams",
    "ContactPoint",
    "ContactModel",
    "PenaltyContactModel",
    "NullContactModel",
    # integrators
    "SemiImplicitEuler",
    "RK4",
    "simulate",
    # self-collision / collision
    "BodyAABB",
    "AABBSelfCollision",
    "SelfCollisionModel",
    "NullSelfCollision",
    # geometry
    "CollisionShape",
    "BoxShape",
    "SphereShape",
    "CylinderShape",
    "MeshShape",
    "ShapeInstance",
    "BodyCollisionGeometry",
    # terrain
    "Terrain",
    "FlatTerrain",
    "HeightmapTerrain",
]
