"""Tests for broad-phase collision detection."""

from __future__ import annotations

import numpy as np
import pytest

from physics.broad_phase import AABB, AABBTreeBroadPhase, BruteForceBroadPhase


class TestAABB:
    def test_overlap(self):
        a = AABB(np.array([0, 0, 0.0]), np.array([2, 2, 2.0]))
        b = AABB(np.array([1, 1, 1.0]), np.array([3, 3, 3.0]))
        assert a.overlaps(b)

    def test_separated(self):
        a = AABB(np.array([0, 0, 0.0]), np.array([1, 1, 1.0]))
        b = AABB(np.array([2, 2, 2.0]), np.array([3, 3, 3.0]))
        assert not a.overlaps(b)

    def test_merge(self):
        a = AABB(np.array([0, 0, 0.0]), np.array([1, 1, 1.0]))
        b = AABB(np.array([2, 0, 0.0]), np.array([3, 1, 1.0]))
        m = a.merge(b)
        np.testing.assert_allclose(m.min_pt, [0, 0, 0])
        np.testing.assert_allclose(m.max_pt, [3, 1, 1])


class TestBruteForceBroadPhase:
    def test_overlapping_pair(self):
        bp = BruteForceBroadPhase()
        bp.update([
            (0, AABB(np.array([0, 0, 0.0]), np.array([2, 2, 2.0]))),
            (1, AABB(np.array([1, 1, 1.0]), np.array([3, 3, 3.0]))),
        ])
        pairs = bp.query_pairs(set())
        assert (0, 1) in pairs

    def test_separated_no_pair(self):
        bp = BruteForceBroadPhase()
        bp.update([
            (0, AABB(np.array([0, 0, 0.0]), np.array([1, 1, 1.0]))),
            (1, AABB(np.array([5, 5, 5.0]), np.array([6, 6, 6.0]))),
        ])
        pairs = bp.query_pairs(set())
        assert len(pairs) == 0

    def test_excluded_pair(self):
        bp = BruteForceBroadPhase()
        bp.update([
            (0, AABB(np.array([0, 0, 0.0]), np.array([2, 2, 2.0]))),
            (1, AABB(np.array([1, 1, 1.0]), np.array([3, 3, 3.0]))),
        ])
        pairs = bp.query_pairs({(0, 1)})
        assert len(pairs) == 0


class TestAABBTreeBroadPhase:
    def test_overlapping_pair(self):
        bp = AABBTreeBroadPhase()
        bp.update([
            (0, AABB(np.array([0, 0, 0.0]), np.array([2, 2, 2.0]))),
            (1, AABB(np.array([1, 1, 1.0]), np.array([3, 3, 3.0]))),
        ])
        pairs = bp.query_pairs(set())
        assert (0, 1) in pairs or (1, 0) in pairs

    def test_separated(self):
        bp = AABBTreeBroadPhase()
        bp.update([
            (0, AABB(np.array([0, 0, 0.0]), np.array([1, 1, 1.0]))),
            (1, AABB(np.array([5, 5, 5.0]), np.array([6, 6, 6.0]))),
        ])
        pairs = bp.query_pairs(set())
        assert len(pairs) == 0

    def test_many_bodies(self):
        """10 bodies in a line — only adjacent should overlap."""
        bp = AABBTreeBroadPhase()
        aabbs = []
        for i in range(10):
            aabbs.append((i, AABB(
                np.array([i * 0.9, 0, 0.0]),
                np.array([i * 0.9 + 1.0, 1, 1.0]),
            )))
        bp.update(aabbs)
        pairs = bp.query_pairs(set())
        # Adjacent boxes overlap (gap = 0.9 < size 1.0)
        assert len(pairs) >= 9  # at least adjacent pairs

    def test_excluded(self):
        bp = AABBTreeBroadPhase()
        bp.update([
            (0, AABB(np.array([0, 0, 0.0]), np.array([2, 2, 2.0]))),
            (1, AABB(np.array([1, 1, 1.0]), np.array([3, 3, 3.0]))),
        ])
        pairs = bp.query_pairs({(0, 1)})
        assert len(pairs) == 0

    def test_consistent_with_brute_force(self):
        """AABB tree should find same pairs as brute force."""
        np.random.seed(42)
        n = 20
        aabbs = []
        for i in range(n):
            center = np.random.rand(3) * 5
            half = np.random.rand(3) * 0.5 + 0.1
            aabbs.append((i, AABB(center - half, center + half)))

        bf = BruteForceBroadPhase()
        bf.update(aabbs)
        bf_pairs = set((min(a, b), max(a, b)) for a, b in bf.query_pairs(set()))

        tree = AABBTreeBroadPhase()
        tree.update(aabbs)
        tree_pairs = set((min(a, b), max(a, b)) for a, b in tree.query_pairs(set()))

        assert bf_pairs == tree_pairs, f"Mismatch: brute={len(bf_pairs)} tree={len(tree_pairs)}"
