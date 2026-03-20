"""
Unit tests for PrismaticJoint, FixedJoint, and FreeJoint.

RevoluteJoint is covered in test_joint_limits.py. These tests cover the
remaining joint types that had zero direct test coverage.

References:
  Featherstone (2008) §4 — joint models.
"""

import numpy as np
import pytest

from physics.joint import Axis, FixedJoint, FreeJoint, PrismaticJoint, RevoluteJoint
from physics.spatial import SpatialTransform, rot_x, rot_z

ATOL = 1e-12

# ===========================================================================
# PrismaticJoint
# ===========================================================================


class TestPrismaticJoint:
    def test_nq_nv(self):
        j = PrismaticJoint("p", axis=Axis.Z)
        assert j.nq == 1
        assert j.nv == 1

    def test_transform_z_axis(self):
        """q=0.5 along Z -> translation [0, 0, 0.5]."""
        j = PrismaticJoint("p", axis=Axis.Z)
        X = j.transform(np.array([0.5]))
        np.testing.assert_allclose(X.R, np.eye(3), atol=ATOL)
        np.testing.assert_allclose(X.r, [0, 0, 0.5], atol=ATOL)

    def test_transform_x_axis(self):
        j = PrismaticJoint("p", axis=Axis.X)
        X = j.transform(np.array([-0.3]))
        np.testing.assert_allclose(X.r, [-0.3, 0, 0], atol=ATOL)

    def test_transform_arbitrary_axis(self):
        """Arbitrary unit axis [1,1,0]/sqrt(2)."""
        axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
        j = PrismaticJoint("p", axis=axis)
        X = j.transform(np.array([1.0]))
        np.testing.assert_allclose(X.r, axis, atol=ATOL)

    def test_transform_no_rotation(self):
        """Prismatic joint never rotates."""
        j = PrismaticJoint("p", axis=Axis.Y)
        X = j.transform(np.array([2.5]))
        np.testing.assert_allclose(X.R, np.eye(3), atol=ATOL)

    def test_motion_subspace(self):
        """S has linear component = axis, angular = 0."""
        j = PrismaticJoint("p", axis=Axis.Z)
        S = j.motion_subspace(np.array([0.0]))
        assert S.shape == (6, 1)
        np.testing.assert_allclose(S[:3, 0], [0, 0, 0], atol=ATOL)
        np.testing.assert_allclose(S[3:, 0], [0, 0, 1], atol=ATOL)

    def test_bias_acceleration_zero(self):
        j = PrismaticJoint("p", axis=Axis.X)
        c = j.bias_acceleration(np.array([0.5]), np.array([1.0]))
        np.testing.assert_allclose(c, np.zeros(6), atol=ATOL)

    def test_damping_torque(self):
        j = PrismaticJoint("p", axis=Axis.Z, damping=3.0)
        assert j.compute_damping_torque(np.array([2.0])) == pytest.approx(-6.0)
        assert j.compute_damping_torque(np.array([-1.0])) == pytest.approx(3.0)
        assert j.compute_damping_torque(np.array([0.0])) == pytest.approx(0.0)

    def test_zero_damping(self):
        j = PrismaticJoint("p", axis=Axis.Z, damping=0.0)
        assert j.compute_damping_torque(np.array([5.0])) == 0.0

    def test_default_state(self):
        j = PrismaticJoint("p", axis=Axis.Z)
        np.testing.assert_allclose(j.default_q(), [0.0], atol=ATOL)
        np.testing.assert_allclose(j.default_qdot(), [0.0], atol=ATOL)

    def test_zero_axis_raises(self):
        with pytest.raises(ValueError):
            PrismaticJoint("p", axis=np.array([0.0, 0.0, 0.0]))


# ===========================================================================
# FixedJoint
# ===========================================================================


class TestFixedJoint:
    def test_nq_nv_zero(self):
        j = FixedJoint("f")
        assert j.nq == 0
        assert j.nv == 0

    def test_transform_identity_default(self):
        j = FixedJoint("f")
        X = j.transform(np.zeros(0))
        np.testing.assert_allclose(X.R, np.eye(3), atol=ATOL)
        np.testing.assert_allclose(X.r, np.zeros(3), atol=ATOL)

    def test_transform_with_offset(self):
        offset = SpatialTransform(rot_z(0.5), np.array([1.0, 2.0, 3.0]))
        j = FixedJoint("f", offset=offset)
        X = j.transform(np.zeros(0))
        np.testing.assert_allclose(X.R, offset.R, atol=ATOL)
        np.testing.assert_allclose(X.r, offset.r, atol=ATOL)

    def test_motion_subspace_empty(self):
        j = FixedJoint("f")
        S = j.motion_subspace(np.zeros(0))
        assert S.shape == (6, 0)

    def test_bias_acceleration_zero(self):
        j = FixedJoint("f")
        c = j.bias_acceleration(np.zeros(0), np.zeros(0))
        np.testing.assert_allclose(c, np.zeros(6), atol=ATOL)

    def test_default_state_empty(self):
        j = FixedJoint("f")
        assert j.default_q().shape == (0,)
        assert j.default_qdot().shape == (0,)


# ===========================================================================
# FreeJoint
# ===========================================================================


class TestFreeJoint:
    def test_nq_nv(self):
        j = FreeJoint()
        assert j.nq == 7
        assert j.nv == 6

    def test_transform_identity_quat(self):
        """Default quaternion [1,0,0,0] at origin -> identity transform."""
        j = FreeJoint()
        q = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        X = j.transform(q)
        np.testing.assert_allclose(X.R, np.eye(3), atol=ATOL)
        np.testing.assert_allclose(X.r, np.zeros(3), atol=ATOL)

    def test_transform_translation_only(self):
        """Identity quaternion with position [1,2,3]."""
        j = FreeJoint()
        q = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
        X = j.transform(q)
        np.testing.assert_allclose(X.R, np.eye(3), atol=ATOL)
        np.testing.assert_allclose(X.r, [1.0, 2.0, 3.0], atol=ATOL)

    def test_transform_rotation(self):
        """90-deg rotation about Z via quaternion."""
        # quat for 90-deg about Z: [cos(45), 0, 0, sin(45)]
        c = np.cos(np.pi / 4)
        s = np.sin(np.pi / 4)
        j = FreeJoint()
        q = np.array([c, 0, 0, s, 0, 0, 0])
        X = j.transform(q)
        np.testing.assert_allclose(X.R, rot_z(np.pi / 2), atol=1e-10)

    def test_motion_subspace_identity(self):
        j = FreeJoint()
        S = j.motion_subspace(j.default_q())
        np.testing.assert_allclose(S, np.eye(6), atol=ATOL)

    def test_bias_acceleration_zero(self):
        j = FreeJoint()
        c = j.bias_acceleration(j.default_q(), j.default_qdot())
        np.testing.assert_allclose(c, np.zeros(6), atol=ATOL)

    def test_default_q(self):
        j = FreeJoint()
        q = j.default_q()
        assert q.shape == (7,)
        assert q[0] == 1.0  # qw = 1
        np.testing.assert_allclose(q[1:], 0.0, atol=ATOL)

    def test_integrate_q_preserves_norm(self):
        """Quaternion norm stays 1 after integration."""
        j = FreeJoint()
        q = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        qdot = np.array([0.5, -0.3, 1.0, 0.1, 0.2, -0.5])
        q_new = j.integrate_q(q, qdot, dt=0.01)
        quat_norm = np.linalg.norm(q_new[:4])
        assert abs(quat_norm - 1.0) < 1e-10

    def test_integrate_q_pure_rotation(self):
        """Only angular velocity, no linear -> position unchanged."""
        j = FreeJoint()
        q = np.array([1.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0])
        qdot = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # omega_x only
        q_new = j.integrate_q(q, qdot, dt=0.01)
        np.testing.assert_allclose(q_new[4:], [5.0, 6.0, 7.0], atol=ATOL)

    def test_integrate_q_pure_translation(self):
        """Only linear velocity -> quaternion unchanged (up to normalization)."""
        j = FreeJoint()
        q = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        qdot = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
        dt = 0.1
        q_new = j.integrate_q(q, qdot, dt)
        np.testing.assert_allclose(q_new[:4], [1, 0, 0, 0], atol=ATOL)
        np.testing.assert_allclose(q_new[4:], [0.1, 0.2, 0.3], atol=ATOL)


# ===========================================================================
# RevoluteJoint — arbitrary axis (supplement to test_joint_limits.py)
# ===========================================================================


class TestRevoluteArbitraryAxis:
    def test_zero_axis_raises(self):
        with pytest.raises(ValueError):
            RevoluteJoint("r", axis=np.array([0.0, 0.0, 0.0]))

    def test_arbitrary_axis_rotation(self):
        """Rotation about [1,0,0] by pi/2 should match rot_x(pi/2)."""
        j = RevoluteJoint("r", axis=np.array([1.0, 0.0, 0.0]))
        X = j.transform(np.array([np.pi / 2]))
        np.testing.assert_allclose(X.R, rot_x(np.pi / 2), atol=1e-10)

    def test_axis_normalized(self):
        """Non-unit axis gets normalized."""
        j = RevoluteJoint("r", axis=np.array([0.0, 3.0, 0.0]))
        np.testing.assert_allclose(j._axis_vec, [0, 1, 0], atol=ATOL)
