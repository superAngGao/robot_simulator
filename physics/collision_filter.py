"""
Collision filter — decides which body pairs should be tested for collision.

Three independent filtering mechanisms (any one vetoes the pair):

1. **Auto-exclude**: parent-child pairs in the kinematic tree are always
   excluded (they share a joint and always "overlap").

2. **Bitmask filter**: each body has a ``group`` and a ``mask`` (uint32).
   Pair (i, j) passes iff ``(group_i & mask_j) != 0 AND (group_j & mask_i) != 0``.
   Default group=0xFFFFFFFF, mask=0xFFFFFFFF (collide with everything).

3. **Explicit exclude set**: user-declared ``(i, j)`` pairs that should
   never collide regardless of bitmask.

All three are static — computed once at model construction, not per step.

References:
  MuJoCo: contype / conaffinity bitmask filtering.
  Bullet: btBroadphaseProxy filter group/mask.
  Drake: CollisionFilterDeclaration (auto parent-child + user exclude).
"""

from __future__ import annotations

from typing import Iterable, Set, Tuple


def _ordered(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


class CollisionFilter:
    """Determines which body pairs are eligible for collision detection.

    Args:
        num_bodies: Total number of bodies in the robot tree.
    """

    def __init__(self, num_bodies: int = 0) -> None:
        self._num_bodies = num_bodies
        # Bitmask per body: (group, mask)
        self._groups: list[int] = [0xFFFFFFFF] * num_bodies
        self._masks: list[int] = [0xFFFFFFFF] * num_bodies
        # Explicit exclude set: ordered (min, max) tuples
        self._excluded: Set[Tuple[int, int]] = set()

    # ------------------------------------------------------------------
    # Auto-exclude (kinematic tree adjacency)
    # ------------------------------------------------------------------

    def auto_exclude_adjacent(self, parent_list: list[int]) -> None:
        """Exclude all direct parent-child pairs.

        Args:
            parent_list: parent_list[i] is the parent body index of body i
                         (-1 for the root).
        """
        for child_idx, parent_idx in enumerate(parent_list):
            if parent_idx >= 0:
                self._excluded.add(_ordered(child_idx, parent_idx))

    # ------------------------------------------------------------------
    # Bitmask API
    # ------------------------------------------------------------------

    def set_group(self, body_index: int, group: int) -> None:
        """Set the collision group bits for a body."""
        self._groups[body_index] = group & 0xFFFFFFFF

    def set_mask(self, body_index: int, mask: int) -> None:
        """Set the collision mask bits for a body."""
        self._masks[body_index] = mask & 0xFFFFFFFF

    def set_group_mask(self, body_index: int, group: int, mask: int) -> None:
        """Set both group and mask for a body."""
        self.set_group(body_index, group)
        self.set_mask(body_index, mask)

    # ------------------------------------------------------------------
    # Explicit exclude API
    # ------------------------------------------------------------------

    def exclude_pair(self, body_i: int, body_j: int) -> None:
        """Explicitly exclude a specific body pair from collision."""
        self._excluded.add(_ordered(body_i, body_j))

    def exclude_pairs(self, pairs: Iterable[tuple[int, int]]) -> None:
        """Exclude multiple body pairs."""
        for i, j in pairs:
            self._excluded.add(_ordered(i, j))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def should_collide(self, body_i: int, body_j: int) -> bool:
        """Return True if the pair should be tested for collision.

        False if any of the three mechanisms vetoes the pair.
        """
        if body_i == body_j:
            return False

        pair = _ordered(body_i, body_j)

        # Explicit exclude
        if pair in self._excluded:
            return False

        # Bitmask: both directions must pass
        if (self._groups[body_i] & self._masks[body_j]) == 0:
            return False
        if (self._groups[body_j] & self._masks[body_i]) == 0:
            return False

        return True

    def excluded_pairs(self) -> Set[Tuple[int, int]]:
        """Return the full set of excluded pairs (for BroadPhase integration).

        Includes auto-excluded + explicit excludes. Does NOT enumerate
        bitmask-excluded pairs (those are checked per-pair via should_collide).
        """
        return set(self._excluded)

    # ------------------------------------------------------------------
    # Bulk generation
    # ------------------------------------------------------------------

    def filter_pairs(self, candidate_pairs: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
        """Filter a list of candidate pairs, returning only allowed ones."""
        return [(i, j) for i, j in candidate_pairs if self.should_collide(i, j)]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def num_bodies(self) -> int:
        return self._num_bodies

    @property
    def num_excluded(self) -> int:
        return len(self._excluded)

    def __repr__(self) -> str:
        return f"CollisionFilter(bodies={self._num_bodies}, excluded={len(self._excluded)})"
