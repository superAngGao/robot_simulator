"""
Unit tests for RobotTree: RNEA, FK, and defensive checks.

ABA is covered in test_aba_vs_pinocchio.py and test_spatial.py.
body_velocities is covered in test_body_velocities.py.
passive_torques is covered in test_joint_limits.py.

References:
  Featherstone (2008) §5 (RNEA), §3 (FK), §7 (ABA).
"""

import numpy as np
import pytest

try:
    import pinocchio as pin

    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False

from physics.joint import Axis, RevoluteJoint
from physics.robot_tree import Body, RobotTree
from physics.spatial import SpatialInertia, SpatialTransform, rot_z

ATOL = 1e-8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pendulum(gravity=9.81):
    """Single revolute-Y pendulum: mass=1, CoM [0,0,-0.5], I=0.1*eye."""
    tree = RobotTree(gravity=gravity)
    tree.add_body(
        Body(
            name="link0",
            index=0,
            joint=RevoluteJoint("j0", axis=Axis.Y),
            inertia=SpatialInertia(mass=1.0, inertia=0.1 * np.eye(3), com=np.array([0.0, 0.0, -0.5])),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()
    return tree


def _make_double_pendulum(gravity=9.81):
    """Two revolute joints (Y then X), with rotated X_tree on second link."""
    tree = RobotTree(gravity=gravity)
    tree.add_body(
        Body(
            name="link0",
            index=0,
            joint=RevoluteJoint("j0", axis=Axis.Y),
            inertia=SpatialInertia(mass=1.0, inertia=0.05 * np.eye(3), com=np.array([0.0, 0.0, -0.3])),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.add_body(
        Body(
            name="link1",
            index=1,
            joint=RevoluteJoint("j1", axis=Axis.X),
            inertia=SpatialInertia(mass=0.5, inertia=0.02 * np.eye(3), com=np.array([0.0, 0.0, -0.2])),
            X_tree=SpatialTransform(np.eye(3), np.array([0.0, 0.0, -0.6])),
            parent=0,
        )
    )
    tree.finalize()
    return tree


# ===========================================================================
# FK tests
# ===========================================================================


class TestForwardKinematics:
    def test_fk_identity_at_zero(self):
        """At q=0, FK result equals the accumulated X_tree transforms."""
        tree = _make_pendulum()
        q, _ = tree.default_state()
        X_world = tree.forward_kinematics(q)
        # Single body with identity X_tree and q=0 -> identity joint transform
        np.testing.assert_allclose(X_world[0].R, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(X_world[0].r, np.zeros(3), atol=1e-12)

    def test_fk_single_revolute(self):
        """Revolute-Y at q=pi/2: R should be rot_y(pi/2)."""
        tree = _make_pendulum()
        q = np.array([np.pi / 2])
        X_world = tree.forward_kinematics(q)
        np.testing.assert_allclose(X_world[0].R, np.eye(3) @ tree.bodies[0].joint.transform(q).R, atol=1e-10)

    def test_fk_chain_composes(self):
        """For a chain, FK[child] = FK[parent] @ X_tree_child @ X_J_child."""
        tree = _make_double_pendulum()
        q = np.array([0.3, -0.4])
        X_world = tree.forward_kinematics(q)

        # Manual computation for body 1
        b0, b1 = tree.bodies[0], tree.bodies[1]
        X_J0 = b0.joint.transform(q[b0.q_idx])
        X_J1 = b1.joint.transform(q[b1.q_idx])
        X_world_0 = b0.X_tree @ X_J0
        X_world_1 = X_world_0 @ (b1.X_tree @ X_J1)

        np.testing.assert_allclose(X_world[1].R, X_world_1.R, atol=1e-10)
        np.testing.assert_allclose(X_world[1].r, X_world_1.r, atol=1e-10)

    def test_fk_returns_correct_count(self):
        tree = _make_double_pendulum()
        q, _ = tree.default_state()
        X_world = tree.forward_kinematics(q)
        assert len(X_world) == tree.num_bodies


# ===========================================================================
# RNEA tests
# ===========================================================================


class TestRNEA:
    def test_rnea_zero_gravity_zero_accel(self):
        """No gravity + zero acceleration -> tau = 0."""
        tree = _make_pendulum(gravity=0.0)
        q, qdot = tree.default_state()
        qddot = np.zeros(tree.nv)
        tau = tree.rnea(q, qdot, qddot)
        np.testing.assert_allclose(tau, np.zeros(tree.nv), atol=1e-12)

    def test_rnea_static_equilibrium(self):
        """Static (qdot=0, qddot=0): RNEA gives gravity compensation torque."""
        tree = _make_pendulum()
        q = np.array([0.3])
        qdot = np.zeros(1)
        qddot = np.zeros(1)
        tau = tree.rnea(q, qdot, qddot)
        # Torque should be nonzero (gravity compensation)
        assert abs(tau[0]) > 0.01

    def test_rnea_aba_roundtrip(self):
        """tau -> ABA -> qddot -> RNEA(qddot) should recover tau."""
        tree = _make_double_pendulum()
        q = np.array([0.3, -0.4])
        qdot = np.array([1.0, -0.5])
        tau = np.array([0.5, -1.0])

        qddot = tree.aba(q, qdot, tau)
        tau_recovered = tree.rnea(q, qdot, qddot)

        np.testing.assert_allclose(tau_recovered, tau, atol=ATOL)

    @pytest.mark.skipif(not HAS_PINOCCHIO, reason="pinocchio not installed")
    def test_rnea_vs_pinocchio(self):
        """RNEA output matches Pinocchio for a single pendulum."""
        tree = _make_pendulum()

        model = pin.Model()
        model.gravity.linear = np.array([0.0, 0.0, -9.81])
        jid = model.addJoint(0, pin.JointModelRY(), pin.SE3.Identity(), "j0")
        model.appendBodyToJoint(
            jid,
            pin.Inertia(1.0, np.zeros(3), 0.1 * np.eye(3)),
            pin.SE3(np.eye(3), np.array([0.0, 0.0, -0.5])),
        )
        data = model.createData()

        for q_val in [0.0, 0.3, -0.5, np.pi / 4]:
            q = np.array([q_val])
            qdot = np.array([1.0])
            qddot = np.array([0.5])

            ours = tree.rnea(q, qdot, qddot)
            ref = pin.rnea(model, data, q, qdot, qddot).copy()
            np.testing.assert_allclose(ours, ref, atol=ATOL)

    @pytest.mark.skipif(not HAS_PINOCCHIO, reason="pinocchio not installed")
    def test_rnea_vs_pinocchio_rotated_x_tree(self):
        """RNEA with non-identity R in X_tree matches Pinocchio."""
        R_tree = rot_z(np.pi / 6)
        r_tree = np.array([0.0, 0.0, -0.4])

        tree = RobotTree(gravity=9.81)
        tree.add_body(
            Body(
                name="link0",
                index=0,
                joint=RevoluteJoint("j0", axis=Axis.Y),
                inertia=SpatialInertia(mass=1.5, inertia=0.08 * np.eye(3), com=np.array([0.0, 0.0, -0.25])),
                X_tree=SpatialTransform(R_tree, r_tree),
                parent=-1,
            )
        )
        tree.finalize()

        model = pin.Model()
        model.gravity.linear = np.array([0.0, 0.0, -9.81])
        jid = model.addJoint(0, pin.JointModelRY(), pin.SE3(R_tree, r_tree), "j0")
        model.appendBodyToJoint(
            jid,
            pin.Inertia(1.5, np.zeros(3), 0.08 * np.eye(3)),
            pin.SE3(np.eye(3), np.array([0.0, 0.0, -0.25])),
        )
        data = model.createData()

        q = np.array([0.3])
        qdot = np.array([1.0])
        qddot = np.array([-0.5])

        ours = tree.rnea(q, qdot, qddot)
        ref = pin.rnea(model, data, q, qdot, qddot).copy()
        np.testing.assert_allclose(ours, ref, atol=ATOL)


# ===========================================================================
# Defensive checks
# ===========================================================================


class TestDefensive:
    def test_not_finalized_raises(self):
        tree = RobotTree()
        tree.add_body(
            Body(
                name="b",
                index=0,
                joint=RevoluteJoint("j", axis=Axis.Y),
                inertia=SpatialInertia(mass=1.0, inertia=np.eye(3), com=np.zeros(3)),
                X_tree=SpatialTransform.identity(),
                parent=-1,
            )
        )
        q, qdot = np.zeros(1), np.zeros(1)
        with pytest.raises(RuntimeError):
            tree.forward_kinematics(q)
        with pytest.raises(RuntimeError):
            tree.aba(q, qdot, np.zeros(1))

    def test_add_after_finalize_raises(self):
        tree = RobotTree()
        tree.add_body(
            Body(
                name="b",
                index=0,
                joint=RevoluteJoint("j", axis=Axis.Y),
                inertia=SpatialInertia(mass=1.0, inertia=np.eye(3), com=np.zeros(3)),
                X_tree=SpatialTransform.identity(),
                parent=-1,
            )
        )
        tree.finalize()
        with pytest.raises(RuntimeError):
            tree.add_body(
                Body(
                    name="b2",
                    index=1,
                    joint=RevoluteJoint("j2", axis=Axis.Y),
                    inertia=SpatialInertia(mass=1.0, inertia=np.eye(3), com=np.zeros(3)),
                    X_tree=SpatialTransform.identity(),
                    parent=0,
                )
            )

    def test_body_by_name_missing_raises(self):
        tree = _make_pendulum()
        with pytest.raises(KeyError):
            tree.body_by_name("nonexistent")

    def test_body_by_name_found(self):
        tree = _make_pendulum()
        b = tree.body_by_name("link0")
        assert b.name == "link0"
