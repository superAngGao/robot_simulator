"""
Tests for Phase 2g-3: Grouped CRBA with auto branch-point detection
and hierarchical Schur complement.
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.joint import FreeJoint, RevoluteJoint
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform

RNG = np.random.default_rng(123)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_quadruped():
    """Floating base + 4 legs × 2 revolute joints = nv=14."""
    tree = RobotTreeNumpy(gravity=9.81)
    tree.add_body(Body(name="base", index=0, joint=FreeJoint("root"),
        inertia=SpatialInertia.from_box(5.0, 0.3, 0.2, 0.1),
        X_tree=SpatialTransform.identity(), parent=-1))
    idx = 0
    for limb, off in [("FL", [0.15, 0.1, 0]), ("FR", [0.15, -0.1, 0]),
                       ("RL", [-0.15, 0.1, 0]), ("RR", [-0.15, -0.1, 0])]:
        parent = 0
        for j in range(2):
            idx += 1
            r = np.array(off) if j == 0 else np.array([0, 0, -0.2])
            tree.add_body(Body(name=f"{limb}_j{j}", index=idx,
                joint=RevoluteJoint(f"{limb}_j{j}", axis=np.array([0, 1, 0])),
                inertia=SpatialInertia(0.3, np.diag([0.001]*3), np.array([0, 0, -0.1])),
                X_tree=SpatialTransform(np.eye(3), r), parent=parent))
            parent = idx
    tree.finalize()
    return tree


def _make_humanoid():
    """Floating base + 4 limbs × 6 joints = nv=30."""
    tree = RobotTreeNumpy(gravity=9.81)
    tree.add_body(Body(name="base", index=0, joint=FreeJoint("root"),
        inertia=SpatialInertia.from_box(10.0, 0.3, 0.2, 0.4),
        X_tree=SpatialTransform.identity(), parent=-1))
    idx = 0
    axes = [np.array([0,1,0]), np.array([1,0,0]), np.array([0,0,1]),
            np.array([0,1,0]), np.array([1,0,0]), np.array([0,1,0])]
    offsets = [[0, 0.15, 0.15], [0, -0.15, 0.15], [0, 0.1, -0.2], [0, -0.1, -0.2]]
    for li, off in enumerate(offsets):
        parent = 0
        for j in range(6):
            idx += 1
            r = np.array(off) if j == 0 else np.array([0, 0, -0.08])
            tree.add_body(Body(name=f"L{li}_j{j}", index=idx,
                joint=RevoluteJoint(f"L{li}_j{j}", axis=axes[j], damping=0.05),
                inertia=SpatialInertia(0.2, np.diag([0.0005]*3), np.array([0, 0, -0.04])),
                X_tree=SpatialTransform(np.eye(3), r), parent=parent))
            parent = idx
    tree.finalize()
    return tree


def _make_chain():
    """Single chain: base + 5 revolute (no branching). nv=11."""
    tree = RobotTreeNumpy(gravity=9.81)
    tree.add_body(Body(name="base", index=0, joint=FreeJoint("root"),
        inertia=SpatialInertia.from_box(3.0, 0.2, 0.2, 0.2),
        X_tree=SpatialTransform.identity(), parent=-1))
    for i in range(5):
        tree.add_body(Body(name=f"link{i}", index=i+1,
            joint=RevoluteJoint(f"j{i}", axis=np.array([0, 1, 0])),
            inertia=SpatialInertia(0.5, np.diag([0.001]*3), np.array([0, 0, -0.1])),
            X_tree=SpatialTransform(np.eye(3), np.array([0, 0, -0.2])),
            parent=i))
    tree.finalize()
    return tree


# ---------------------------------------------------------------------------
# Tests: auto group detection
# ---------------------------------------------------------------------------


class TestAutoDetectGroups:
    def test_quadruped_finds_4_limbs(self):
        tree = _make_quadruped()
        root, limbs = tree.auto_detect_groups()
        assert len(limbs) == 4, f"Expected 4 limb groups, got {len(limbs)}"

    def test_quadruped_root_is_base(self):
        tree = _make_quadruped()
        root, limbs = tree.auto_detect_groups()
        assert 0 in root, "Root body should be in root group"

    def test_quadruped_limbs_cover_all(self):
        tree = _make_quadruped()
        root, limbs = tree.auto_detect_groups()
        all_bodies = set(root)
        for g in limbs:
            all_bodies.update(g)
        assert all_bodies == set(range(tree.num_bodies))

    def test_humanoid_finds_4_limbs(self):
        tree = _make_humanoid()
        root, limbs = tree.auto_detect_groups()
        assert len(limbs) == 4

    def test_chain_no_limbs(self):
        tree = _make_chain()
        root, limbs = tree.auto_detect_groups()
        assert len(limbs) == 0, "Chain should have no limb groups"
        assert len(root) == tree.num_bodies


# ---------------------------------------------------------------------------
# Tests: grouped CRBA == monolithic CRBA == ABA
# ---------------------------------------------------------------------------


class TestGroupedCRBACorrectness:
    def test_quadruped_grouped_eq_aba(self):
        tree = _make_quadruped()
        q, qdot = tree.default_state()
        tau = RNG.standard_normal(tree.nv) * 0.5

        qddot_aba = tree.aba(q, qdot, tau)
        qddot_grouped = tree.forward_dynamics_grouped_crba(q, qdot, tau)

        np.testing.assert_allclose(qddot_grouped, qddot_aba, atol=1e-10,
                                   err_msg="Grouped CRBA != ABA for quadruped")

    def test_humanoid_grouped_eq_aba(self):
        tree = _make_humanoid()
        q, qdot = tree.default_state()
        for body in tree.bodies:
            if body.joint.nv > 0 and not isinstance(body.joint, FreeJoint):
                q[body.q_idx] = RNG.uniform(-0.3, 0.3, body.joint.nq)
                qdot[body.v_idx] = RNG.standard_normal(body.joint.nv) * 0.2
        tau = RNG.standard_normal(tree.nv) * 2.0

        qddot_aba = tree.aba(q, qdot, tau)
        qddot_grouped = tree.forward_dynamics_grouped_crba(q, qdot, tau)

        np.testing.assert_allclose(qddot_grouped, qddot_aba, atol=1e-9,
                                   err_msg="Grouped CRBA != ABA for humanoid")

    def test_chain_falls_back_to_standard(self):
        tree = _make_chain()
        q, qdot = tree.default_state()
        tau = RNG.standard_normal(tree.nv) * 0.5

        qddot_aba = tree.aba(q, qdot, tau)
        qddot_grouped = tree.forward_dynamics_grouped_crba(q, qdot, tau)

        np.testing.assert_allclose(qddot_grouped, qddot_aba, atol=1e-10,
                                   err_msg="Grouped CRBA != ABA for chain (should fallback)")

    def test_with_external_forces(self):
        tree = _make_quadruped()
        q, qdot = tree.default_state()
        tau = RNG.standard_normal(tree.nv) * 0.5
        ext = [None] * tree.num_bodies
        ext[1] = RNG.standard_normal(6)
        ext[5] = RNG.standard_normal(6)

        qddot_aba = tree.aba(q, qdot, tau, ext)
        qddot_grouped = tree.forward_dynamics_grouped_crba(q, qdot, tau, ext)

        np.testing.assert_allclose(qddot_grouped, qddot_aba, atol=1e-10,
                                   err_msg="Grouped CRBA != ABA with external forces")

    def test_grouped_eq_monolithic_crba(self):
        """Grouped forward dynamics should match monolithic CRBA."""
        tree = _make_quadruped()
        q, qdot = tree.default_state()
        tau = RNG.standard_normal(tree.nv) * 0.5

        qddot_mono = tree.forward_dynamics_crba(q, qdot, tau)
        qddot_grouped = tree.forward_dynamics_grouped_crba(q, qdot, tau)

        np.testing.assert_allclose(qddot_grouped, qddot_mono, atol=1e-10,
                                   err_msg="Grouped CRBA != monolithic CRBA")
