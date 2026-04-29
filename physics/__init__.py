"""
physics — Rigid body dynamics pipeline, spatial algebra, joint models, contact.
"""

from ._robot_tree_base import RobotTreeBase
from .collision import (
    AABBSelfCollision,
    BodyAABB,
    NullSelfCollision,
    SelfCollisionModel,
)
from .collision_filter import CollisionFilter
from .constraint_solver import ConstraintSolver, NullConstraintSolver
from .constraint_solvers import AccelLevelAdapter, VelocityLevelAdapter, wrap_solver
from .contact import (
    ContactModel,
    ContactParams,
    ContactPoint,
    NullContactModel,
    PenaltyContactModel,
)
from .dynamics_cache import DynamicsCache, ForceState
from .force_source import ForceSource, PassiveForceSource
from .geometry import (
    BodyCollisionGeometry,
    BoxShape,
    CollisionShape,
    ConvexHullShape,
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
from .publish import (
    AckPolicy,
    BorrowedFrameLease,
    ConsumerPublishStats,
    ConsumerState,
    CpuPublishedFrame,
    DeviceConsumerStalledError,
    DeviceSnapshotSpec,
    GpuPublishedFrame,
    HostSnapshotSpec,
    LeaseExpiredError,
    OnRingFull,
    PublishedFrameCore,
    PublishedRing,
    PublishedSlotMeta,
    PublishedSlotStats,
    PublishPlan,
    PublishPolicy,
    PublishRuntimeStats,
    RingPressureStats,
    SlotReclaimedError,
    SlotReclaimer,
    SnapshotHandle,
    ViewPolicy,
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
from .step_pipeline import StepPipeline
from .telemetry import TelemetrySnapshot, build_telemetry_snapshot_from_published_frame
from .terrain import FlatTerrain, HeightmapTerrain, Terrain

__all__ = [
    # === New pipeline (Phase 2h) ===
    "StepPipeline",
    "TelemetrySnapshot",
    "build_telemetry_snapshot_from_published_frame",
    "DynamicsCache",
    "ForceState",
    "ForceSource",
    "PassiveForceSource",
    "ConstraintSolver",
    "NullConstraintSolver",
    "AccelLevelAdapter",
    "VelocityLevelAdapter",
    "wrap_solver",
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
    # publish / control plane
    "ViewPolicy",
    "PublishPolicy",
    "PublishPlan",
    "PublishedFrameCore",
    "PublishedRing",
    "BorrowedFrameLease",
    "CpuPublishedFrame",
    "GpuPublishedFrame",
    "HostSnapshotSpec",
    "DeviceSnapshotSpec",
    "LeaseExpiredError",
    "SlotReclaimedError",
    "OnRingFull",
    "ConsumerState",
    "ConsumerPublishStats",
    "DeviceConsumerStalledError",
    "AckPolicy",
    "PublishedSlotMeta",
    "PublishedSlotStats",
    "PublishRuntimeStats",
    "RingPressureStats",
    "SnapshotHandle",
    "SlotReclaimer",
    # contact (legacy — use ConstraintSolver + CollisionPipeline instead)
    "ContactParams",
    "ContactPoint",
    "ContactModel",
    "PenaltyContactModel",
    "NullContactModel",
    # integrators (legacy — StepPipeline integrates inline)
    "SemiImplicitEuler",
    "RK4",
    "simulate",
    # self-collision / collision
    "BodyAABB",
    "AABBSelfCollision",
    "SelfCollisionModel",
    "NullSelfCollision",
    "CollisionFilter",
    # geometry
    "CollisionShape",
    "BoxShape",
    "SphereShape",
    "CylinderShape",
    "CapsuleShape",
    "ConvexHullShape",
    "MeshShape",
    "ShapeInstance",
    "BodyCollisionGeometry",
    # terrain
    "Terrain",
    "FlatTerrain",
    "HeightmapTerrain",
]
