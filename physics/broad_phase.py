"""
Broad-phase collision detection using an AABB tree.

Filters body pairs that *might* collide (AABB overlap) before running
expensive narrow-phase (GJK/EPA). Reduces O(n²) all-pairs to O(n log n).

Two implementations:
  - BruteForceBroadPhase: O(n²) baseline (current behavior)
  - AABBTreeBroadPhase: balanced binary tree with incremental updates

References:
  Ericson (2004) — Real-Time Collision Detection, §6.
  Bullet: btDbvtBroadphase (Dynamic AABB Tree).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Set, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class AABB:
    """Axis-aligned bounding box."""
    min_pt: NDArray[np.float64]  # (3,)
    max_pt: NDArray[np.float64]  # (3,)

    def overlaps(self, other: "AABB") -> bool:
        return bool(np.all(self.min_pt <= other.max_pt) and np.all(other.min_pt <= self.max_pt))

    def merge(self, other: "AABB") -> "AABB":
        return AABB(
            np.minimum(self.min_pt, other.min_pt),
            np.maximum(self.max_pt, other.max_pt),
        )

    def surface_area(self) -> float:
        d = self.max_pt - self.min_pt
        return 2.0 * (d[0]*d[1] + d[1]*d[2] + d[0]*d[2])


class BroadPhase(ABC):
    """Abstract broad-phase collision detection."""

    @abstractmethod
    def update(self, body_aabbs: list[tuple[int, AABB]]) -> None:
        """Update with current frame's AABBs. body_aabbs: [(body_index, aabb), ...]"""

    @abstractmethod
    def query_pairs(self, excluded: Set[Tuple[int, int]]) -> list[tuple[int, int]]:
        """Return overlapping body pairs, excluding specified pairs."""


class BruteForceBroadPhase(BroadPhase):
    """O(n²) all-pairs broad-phase (baseline)."""

    def __init__(self) -> None:
        self._aabbs: list[tuple[int, AABB]] = []

    def update(self, body_aabbs: list[tuple[int, AABB]]) -> None:
        self._aabbs = body_aabbs

    def query_pairs(self, excluded: Set[Tuple[int, int]]) -> list[tuple[int, int]]:
        pairs = []
        n = len(self._aabbs)
        for i in range(n):
            bi, aabb_i = self._aabbs[i]
            for j in range(i + 1, n):
                bj, aabb_j = self._aabbs[j]
                edge = (min(bi, bj), max(bi, bj))
                if edge in excluded:
                    continue
                if aabb_i.overlaps(aabb_j):
                    pairs.append((bi, bj))
        return pairs


class AABBTreeBroadPhase(BroadPhase):
    """Balanced AABB tree for O(n log n) broad-phase.

    Builds a binary tree where:
    - Leaves hold individual body AABBs
    - Internal nodes hold merged AABBs of their children
    - Overlap queries traverse the tree, pruning non-overlapping subtrees
    """

    def __init__(self) -> None:
        self._aabbs: list[tuple[int, AABB]] = []
        self._root: "_Node | None" = None

    def update(self, body_aabbs: list[tuple[int, AABB]]) -> None:
        self._aabbs = body_aabbs
        if not body_aabbs:
            self._root = None
            return
        # Build balanced tree (top-down, split on longest axis)
        leaves = [_Node(aabb=aabb, body_idx=bi) for bi, aabb in body_aabbs]
        self._root = self._build(leaves)

    def query_pairs(self, excluded: Set[Tuple[int, int]]) -> list[tuple[int, int]]:
        if self._root is None or len(self._aabbs) < 2:
            return []
        pairs = []
        self._query_recursive(self._root, self._root, excluded, pairs)
        return pairs

    def _build(self, leaves: list["_Node"]) -> "_Node":
        if len(leaves) == 1:
            return leaves[0]
        if len(leaves) == 2:
            node = _Node(aabb=leaves[0].aabb.merge(leaves[1].aabb))
            node.left = leaves[0]
            node.right = leaves[1]
            return node

        # Find longest axis of combined AABB
        combined = leaves[0].aabb
        for leaf in leaves[1:]:
            combined = combined.merge(leaf.aabb)
        d = combined.max_pt - combined.min_pt
        axis = int(np.argmax(d))

        # Sort by centroid along axis, split in half
        leaves.sort(key=lambda n: (n.aabb.min_pt[axis] + n.aabb.max_pt[axis]) / 2)
        mid = len(leaves) // 2

        left = self._build(leaves[:mid])
        right = self._build(leaves[mid:])

        node = _Node(aabb=left.aabb.merge(right.aabb))
        node.left = left
        node.right = right
        return node

    def _query_recursive(
        self,
        a: "_Node",
        b: "_Node",
        excluded: Set[Tuple[int, int]],
        pairs: list,
    ) -> None:
        if not a.aabb.overlaps(b.aabb):
            return

        if a.is_leaf() and b.is_leaf():
            if a.body_idx != b.body_idx:
                edge = (min(a.body_idx, b.body_idx), max(a.body_idx, b.body_idx))
                if edge not in excluded:
                    pairs.append((a.body_idx, b.body_idx))
            return

        # Expand the larger node
        if a.is_leaf() or (not b.is_leaf() and a.aabb.surface_area() <= b.aabb.surface_area()):
            self._query_recursive(a, b.left, excluded, pairs)
            self._query_recursive(a, b.right, excluded, pairs)
        else:
            self._query_recursive(a.left, b, excluded, pairs)
            self._query_recursive(a.right, b, excluded, pairs)


@dataclass
class _Node:
    """AABB tree node."""
    aabb: AABB = None
    body_idx: int = -1  # >= 0 for leaves
    left: "_Node | None" = None
    right: "_Node | None" = None

    def is_leaf(self) -> bool:
        return self.body_idx >= 0
