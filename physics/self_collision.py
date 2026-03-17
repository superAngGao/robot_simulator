# Backward-compatibility re-export. Import from physics.collision instead.
from .collision import AABBSelfCollision, BodyAABB, SelfCollisionModel  # noqa: F401

__all__ = ["AABBSelfCollision", "BodyAABB", "SelfCollisionModel"]
