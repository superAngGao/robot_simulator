"""
Unit tests for physics/spatial.py — spatial algebra primitives.

Verification strategy:
  - Rotation utilities: compare against Pinocchio SE3 and scipy Rotation.
  - SpatialTransform: compare apply_velocity/apply_force against Pinocchio
    SE3.actInv(Motion) / SE3.act(Force).
  - SpatialInertia: compare matrix() against Pinocchio Inertia.matrix().
  - Spatial cross products: verify mathematical identities.
  - matrix()/matrix_dual(): verify consistency with apply_velocity/apply_force.

References:
  Featherstone (2008) §2 — spatial algebra conventions.
  Pinocchio docs — SE3, Motion, Force, Inertia.
"""

import numpy as np
import pytest

try:
    import pinocchio as pin

    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False

from physics.spatial import (
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

ATOL = 1e-12

# ---------------------------------------------------------------------------
# After Q15, we use [linear; angular] — same as Pinocchio. No permutation needed.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_rotation():
    """Random rotation matrix via QR decomposition."""
    M = np.random.randn(3, 3)
    Q, R = np.linalg.qr(M)
    # Ensure det(Q) = +1
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _random_transform():
    """Random SpatialTransform with non-identity R and non-zero r."""
    R = _random_rotation()
    r = np.random.randn(3)
    return SpatialTransform(R, r)


def _random_unit_quaternion():
    """Random unit quaternion [w, x, y, z]."""
    q = np.random.randn(4)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q  # canonical form: w >= 0
    return q


# ===========================================================================
# 1. Rotation utilities
# ===========================================================================


class TestSkew:
    def test_skew_cross_product(self):
        """skew(v) @ u == np.cross(v, u) for random vectors."""
        for _ in range(5):
            v = np.random.randn(3)
            u = np.random.randn(3)
            np.testing.assert_allclose(skew(v) @ u, np.cross(v, u), atol=ATOL)

    def test_skew_antisymmetric(self):
        """skew(v) is antisymmetric: S + S.T == 0."""
        v = np.random.randn(3)
        S = skew(v)
        np.testing.assert_allclose(S + S.T, np.zeros((3, 3)), atol=ATOL)


class TestRotationMatrices:
    def test_rot_identity_at_zero(self):
        """rot_x/y/z(0) == I."""
        for fn in [rot_x, rot_y, rot_z]:
            np.testing.assert_allclose(fn(0.0), np.eye(3), atol=ATOL)

    def test_rot_orthogonal(self):
        """R @ R.T == I and det(R) == +1 for various angles."""
        for fn in [rot_x, rot_y, rot_z]:
            for angle in [0.3, -0.7, np.pi / 2, np.pi]:
                R = fn(angle)
                np.testing.assert_allclose(R @ R.T, np.eye(3), atol=ATOL)
                assert abs(np.linalg.det(R) - 1.0) < ATOL

    def test_rot_x_90(self):
        """rot_x(pi/2): y-axis maps to z-axis."""
        R = rot_x(np.pi / 2)
        np.testing.assert_allclose(R @ [0, 1, 0], [0, 0, 1], atol=ATOL)

    def test_rot_y_90(self):
        """rot_y(pi/2): z-axis maps to x-axis."""
        R = rot_y(np.pi / 2)
        np.testing.assert_allclose(R @ [0, 0, 1], [1, 0, 0], atol=ATOL)

    def test_rot_z_90(self):
        """rot_z(pi/2): x-axis maps to y-axis."""
        R = rot_z(np.pi / 2)
        np.testing.assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=ATOL)


class TestQuaternionConversion:
    def test_identity_quaternion(self):
        """[1,0,0,0] -> I."""
        R = quat_to_rot(np.array([1, 0, 0, 0], dtype=np.float64))
        np.testing.assert_allclose(R, np.eye(3), atol=ATOL)

    def test_roundtrip(self):
        """rot_to_quat(quat_to_rot(q)) == +-q for random quaternions."""
        np.random.seed(42)
        for _ in range(10):
            q = _random_unit_quaternion()
            R = quat_to_rot(q)
            q2 = rot_to_quat(R)
            # Quaternion sign ambiguity: q and -q represent the same rotation
            if q2[0] < 0:
                q2 = -q2
            np.testing.assert_allclose(q2, q, atol=1e-10)

    def test_rot_to_quat_branch_coverage(self):
        """Exercise all 4 branches of rot_to_quat."""
        # Branch 1: trace > 0 (identity-like)
        R1 = rot_z(0.1)
        q1 = rot_to_quat(R1)
        np.testing.assert_allclose(quat_to_rot(q1), R1, atol=ATOL)

        # Branch 2: R[0,0] dominant (180 deg about x)
        R2 = np.diag([1.0, -1.0, -1.0])
        q2 = rot_to_quat(R2)
        np.testing.assert_allclose(quat_to_rot(q2), R2, atol=ATOL)

        # Branch 3: R[1,1] dominant (180 deg about y)
        R3 = np.diag([-1.0, 1.0, -1.0])
        q3 = rot_to_quat(R3)
        np.testing.assert_allclose(quat_to_rot(q3), R3, atol=ATOL)

        # Branch 4: R[2,2] dominant (180 deg about z)
        R4 = np.diag([-1.0, -1.0, 1.0])
        q4 = rot_to_quat(R4)
        np.testing.assert_allclose(quat_to_rot(q4), R4, atol=ATOL)

    @pytest.mark.skipif(not HAS_PINOCCHIO, reason="pinocchio not installed")
    def test_quat_to_rot_vs_pinocchio(self):
        """Compare quat_to_rot against Pinocchio for random quaternions."""
        np.random.seed(7)
        for _ in range(5):
            q = _random_unit_quaternion()
            R_ours = quat_to_rot(q)
            # Pinocchio uses [x,y,z,w] order
            q_pin = np.array([q[1], q[2], q[3], q[0]])
            R_pin = pin.Quaternion(q_pin).toRotationMatrix()
            np.testing.assert_allclose(R_ours, R_pin, atol=ATOL)


# ===========================================================================
# 2. SpatialTransform
# ===========================================================================


class TestSpatialTransformBasic:
    def test_identity_no_change_velocity(self):
        v = np.random.randn(6)
        v_out = SpatialTransform.identity().apply_velocity(v)
        np.testing.assert_allclose(v_out, v, atol=ATOL)

    def test_identity_no_change_force(self):
        f = np.random.randn(6)
        f_out = SpatialTransform.identity().apply_force(f)
        np.testing.assert_allclose(f_out, f, atol=ATOL)

    def test_inverse_compose_identity(self):
        """X @ X.inverse() should be identity."""
        np.random.seed(1)
        X = _random_transform()
        I = X @ X.inverse()
        np.testing.assert_allclose(I.R, np.eye(3), atol=ATOL)
        np.testing.assert_allclose(I.r, np.zeros(3), atol=ATOL)

    def test_compose_associative(self):
        """(A @ B) @ C == A @ (B @ C)."""
        np.random.seed(2)
        A = _random_transform()
        B = _random_transform()
        C = _random_transform()
        AB_C = (A @ B) @ C
        A_BC = A @ (B @ C)
        np.testing.assert_allclose(AB_C.R, A_BC.R, atol=ATOL)
        np.testing.assert_allclose(AB_C.r, A_BC.r, atol=ATOL)

    def test_apply_velocity_pure_translation(self):
        """Pure translation: omega unchanged (R=I), v gets omega x r term."""
        r = np.array([1.0, 2.0, 3.0])
        X = SpatialTransform.from_translation(r)
        v_lin = np.array([1.0, 0.0, 0.0])
        omega = np.array([0.0, 0.0, 1.0])
        v = np.concatenate([v_lin, omega])
        v_out = X.apply_velocity(v)
        # v_out[:3] = v_lin + omega x r
        expected_v = v_lin + np.cross(omega, r)
        np.testing.assert_allclose(v_out[:3], expected_v, atol=ATOL)
        # R=I, so omega_out = omega
        np.testing.assert_allclose(v_out[3:], omega, atol=ATOL)

    def test_apply_force_inverse_roundtrip(self):
        """X.apply_force(X.inverse().apply_force(f)) should NOT be identity.
        But X.inverse().apply_force(X.apply_force(f)) == f.
        (apply_force goes child->parent, so inverse reverses direction.)
        """
        np.random.seed(3)
        X = _random_transform()
        f = np.random.randn(6)
        # X.apply_force: child->parent; X.inverse().apply_force: parent->child
        f_roundtrip = X.inverse().apply_force(X.apply_force(f))
        np.testing.assert_allclose(f_roundtrip, f, atol=ATOL)


class TestSpatialTransformMatrix:
    def test_matrix_matches_apply_velocity(self):
        """matrix() @ v == apply_velocity(v) for random transforms and vectors."""
        np.random.seed(10)
        for _ in range(10):
            X = _random_transform()
            v = np.random.randn(6)
            np.testing.assert_allclose(X.matrix() @ v, X.apply_velocity(v), atol=ATOL)

    def test_matrix_transpose_matches_apply_force(self):
        """matrix().T @ f == apply_force(f) for random transforms and vectors."""
        np.random.seed(11)
        for _ in range(10):
            X = _random_transform()
            f = np.random.randn(6)
            np.testing.assert_allclose(X.matrix().T @ f, X.apply_force(f), atol=ATOL)

    def test_matrix_dual_equals_transpose(self):
        """matrix_dual() == matrix().T."""
        np.random.seed(12)
        X = _random_transform()
        np.testing.assert_allclose(X.matrix_dual(), X.matrix().T, atol=ATOL)

    def test_compose_matrix_reverse_order(self):
        """(A @ B).matrix() == B.matrix() @ A.matrix().

        Velocity transforms compose in reverse order relative to SE3:
        A @ B places B's child in A's parent, so the velocity path is
        A_parent -> A_child=B_parent -> B_child = M(B) @ M(A).
        """
        np.random.seed(13)
        for _ in range(5):
            A = _random_transform()
            B = _random_transform()
            np.testing.assert_allclose((A @ B).matrix(), B.matrix() @ A.matrix(), atol=ATOL)


@pytest.mark.skipif(not HAS_PINOCCHIO, reason="pinocchio not installed")
class TestSpatialTransformVsPinocchio:
    """Cross-validate SpatialTransform against Pinocchio SE3."""

    def _to_pin(self, X: SpatialTransform) -> "pin.SE3":
        return pin.SE3(X.R, X.r)

    def test_apply_velocity_vs_actInv(self):
        """Our apply_velocity == Pinocchio SE3.actInv(Motion).

        Both use [linear; angular] layout after Q15.
        """
        np.random.seed(20)
        for _ in range(10):
            X = _random_transform()
            v = np.random.randn(6)
            ours = X.apply_velocity(v)
            pin_result = self._to_pin(X).actInv(pin.Motion(v)).np.ravel()
            np.testing.assert_allclose(ours, pin_result, atol=ATOL)

    def test_apply_force_vs_act(self):
        """Our apply_force == Pinocchio SE3.act(Force).

        Both use [linear; angular] layout after Q15.
        """
        np.random.seed(21)
        for _ in range(10):
            X = _random_transform()
            f = np.random.randn(6)
            ours = X.apply_force(f)
            pin_result = self._to_pin(X).act(pin.Force(f)).np.ravel()
            np.testing.assert_allclose(ours, pin_result, atol=ATOL)

    def test_compose_vs_pinocchio(self):
        """Our compose matches Pinocchio SE3 multiplication."""
        np.random.seed(22)
        for _ in range(5):
            A = _random_transform()
            B = _random_transform()
            AB = A @ B
            AB_pin = self._to_pin(A) * self._to_pin(B)
            np.testing.assert_allclose(AB.R, AB_pin.rotation, atol=ATOL)
            np.testing.assert_allclose(AB.r, AB_pin.translation, atol=ATOL)

    def test_inverse_vs_pinocchio(self):
        """Our inverse matches Pinocchio SE3.inverse()."""
        np.random.seed(23)
        for _ in range(5):
            X = _random_transform()
            Xi = X.inverse()
            Xi_pin = self._to_pin(X).inverse()
            np.testing.assert_allclose(Xi.R, Xi_pin.rotation, atol=ATOL)
            np.testing.assert_allclose(Xi.r, Xi_pin.translation, atol=ATOL)


# ===========================================================================
# 3. SpatialInertia
# ===========================================================================


class TestSpatialInertia:
    def test_matrix_symmetric(self):
        I = SpatialInertia(mass=2.0, inertia=np.diag([0.1, 0.2, 0.3]), com=np.array([0.1, -0.2, 0.05]))
        M = I.matrix()
        np.testing.assert_allclose(M, M.T, atol=ATOL)

    def test_matrix_positive_definite(self):
        I = SpatialInertia(mass=2.0, inertia=np.diag([0.1, 0.2, 0.3]), com=np.zeros(3))
        eigvals = np.linalg.eigvalsh(I.matrix())
        assert np.all(eigvals > 0), f"Not positive definite: {eigvals}"

    def test_add_total_mass(self):
        I1 = SpatialInertia(mass=1.0, inertia=0.01 * np.eye(3), com=np.array([0.1, 0, 0]))
        I2 = SpatialInertia(mass=2.0, inertia=0.02 * np.eye(3), com=np.array([-0.1, 0, 0]))
        I12 = I1 + I2
        assert abs(I12.mass - 3.0) < ATOL

    def test_add_com_weighted(self):
        I1 = SpatialInertia(mass=1.0, inertia=0.01 * np.eye(3), com=np.array([3.0, 0, 0]))
        I2 = SpatialInertia(mass=2.0, inertia=0.01 * np.eye(3), com=np.array([0.0, 0, 0]))
        I12 = I1 + I2
        expected_com = (1.0 * np.array([3.0, 0, 0]) + 2.0 * np.array([0.0, 0, 0])) / 3.0
        np.testing.assert_allclose(I12.com, expected_com, atol=ATOL)

    def test_from_box_known_values(self):
        """Box 2x4x6, mass 12: Ixx = 12/12*(16+36) = 52."""
        I = SpatialInertia.from_box(12.0, 2.0, 4.0, 6.0)
        assert abs(I.mass - 12.0) < ATOL
        np.testing.assert_allclose(I.com, np.zeros(3), atol=ATOL)
        ixx = 12.0 / 12.0 * (4.0**2 + 6.0**2)  # 52
        iyy = 12.0 / 12.0 * (2.0**2 + 6.0**2)  # 40
        izz = 12.0 / 12.0 * (2.0**2 + 4.0**2)  # 20
        np.testing.assert_allclose(np.diag(I.inertia), [ixx, iyy, izz], atol=ATOL)

    def test_point_mass_zero_inertia(self):
        I = SpatialInertia.point_mass(5.0, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(I.inertia, np.zeros((3, 3)), atol=ATOL)
        assert abs(I.mass - 5.0) < ATOL

    def test_add_matrix_equals_sum_of_matrices(self):
        """(I1 + I2).matrix() == I1.matrix() + I2.matrix() when expressed at same origin."""
        I1 = SpatialInertia(mass=1.0, inertia=0.1 * np.eye(3), com=np.array([0.1, 0.2, 0.0]))
        I2 = SpatialInertia(mass=0.5, inertia=0.05 * np.eye(3), com=np.array([-0.1, 0.0, 0.3]))
        # The 6x6 matrix at origin should be additive
        np.testing.assert_allclose((I1 + I2).matrix(), I1.matrix() + I2.matrix(), atol=ATOL)


@pytest.mark.skipif(not HAS_PINOCCHIO, reason="pinocchio not installed")
class TestSpatialInertiaVsPinocchio:
    def test_matrix_vs_pinocchio(self):
        """Our SpatialInertia.matrix() matches Pinocchio Inertia.matrix().

        Both use [linear; angular] ordering after Q15.
        """
        mass = 2.5
        inertia = np.diag([0.1, 0.2, 0.3])
        com = np.array([0.05, -0.1, 0.15])

        ours = SpatialInertia(mass, inertia, com).matrix()
        pin_I = pin.Inertia(mass, com, inertia)
        np.testing.assert_allclose(ours, pin_I.matrix(), atol=ATOL)

    def test_matrix_zero_com(self):
        """With com=0, matrix should be block-diagonal."""
        mass = 3.0
        inertia = np.diag([0.5, 0.6, 0.7])
        com = np.zeros(3)
        ours = SpatialInertia(mass, inertia, com).matrix()
        pin_I = pin.Inertia(mass, com, inertia)
        np.testing.assert_allclose(ours, pin_I.matrix(), atol=ATOL)


# ===========================================================================
# 4. Spatial cross products
# ===========================================================================


class TestSpatialCross:
    def test_velocity_self_cross_zero(self):
        """v x v == 0 (antisymmetry)."""
        v = np.random.randn(6)
        result = spatial_cross_velocity(v) @ v
        np.testing.assert_allclose(result, np.zeros(6), atol=ATOL)

    def test_force_is_negative_transpose(self):
        """spatial_cross_force(v) == -spatial_cross_velocity(v).T."""
        np.random.seed(30)
        v = np.random.randn(6)
        np.testing.assert_allclose(spatial_cross_force(v), -spatial_cross_velocity(v).T, atol=ATOL)

    def test_gravity_spatial(self):
        """gravity_spatial(9.81) == [0,0,-9.81, 0,0,0]."""
        g = gravity_spatial(9.81)
        np.testing.assert_allclose(g, [0, 0, -9.81, 0, 0, 0], atol=ATOL)


# ===========================================================================
# 5. ABA with non-identity R in X_tree (validates matrix() fix)
# ===========================================================================


@pytest.mark.skipif(not HAS_PINOCCHIO, reason="pinocchio not installed")
class TestABAWithRotatedXTree:
    """ABA cross-validation with non-identity R in X_tree.

    This is the critical test that would have FAILED before the matrix() fix.
    """

    def test_single_pendulum_rotated_x_tree(self):
        """Single revolute pendulum with a 45-degree rotated X_tree."""
        from physics.joint import Axis, RevoluteJoint
        from physics.robot_tree import Body, RobotTree
        from physics.spatial import SpatialInertia

        # Our model: X_tree has a 30-deg rotation about Z + translation
        R_tree = rot_z(np.pi / 6)
        r_tree = np.array([0.0, 0.0, -0.4])
        X_tree = SpatialTransform(R_tree, r_tree)

        tree = RobotTree(gravity=9.81)
        tree.add_body(
            Body(
                name="link0",
                index=0,
                joint=RevoluteJoint("j0", axis=Axis.Y),
                inertia=SpatialInertia(mass=1.5, inertia=0.08 * np.eye(3), com=np.array([0.0, 0.0, -0.25])),
                X_tree=X_tree,
                parent=-1,
            )
        )
        tree.finalize()

        # Pinocchio model with matching rotated placement
        model = pin.Model()
        model.gravity.linear = np.array([0.0, 0.0, -9.81])
        placement = pin.SE3(R_tree, r_tree)
        jid = model.addJoint(0, pin.JointModelRY(), placement, "j0")
        body_inertia = pin.Inertia(1.5, np.zeros(3), 0.08 * np.eye(3))
        body_placement = pin.SE3(np.eye(3), np.array([0.0, 0.0, -0.25]))
        model.appendBodyToJoint(jid, body_inertia, body_placement)

        data = model.createData()

        for q_val in [0.0, 0.3, -0.5, np.pi / 4]:
            for qdot_val in [0.0, 1.0, -2.0]:
                for tau_val in [0.0, 0.5]:
                    q = np.array([q_val])
                    qdot = np.array([qdot_val])
                    tau = np.array([tau_val])

                    ours = tree.aba(q, qdot, tau)
                    ref = pin.aba(model, data, q, qdot, tau).copy()

                    np.testing.assert_allclose(
                        ours, ref, atol=1e-8, err_msg=f"q={q_val}, qdot={qdot_val}, tau={tau_val}"
                    )

    def test_double_pendulum_rotated_x_tree(self):
        """Double pendulum where BOTH joints have rotated X_tree."""
        from physics.joint import Axis, RevoluteJoint
        from physics.robot_tree import Body, RobotTree
        from physics.spatial import SpatialInertia

        R1 = rot_z(np.pi / 6)
        r1 = np.array([0.0, 0.0, 0.0])
        R2 = rot_x(np.pi / 4)
        r2 = np.array([0.0, 0.0, -0.5])

        tree = RobotTree(gravity=9.81)
        tree.add_body(
            Body(
                name="link0",
                index=0,
                joint=RevoluteJoint("j0", axis=Axis.Y),
                inertia=SpatialInertia(mass=1.0, inertia=0.05 * np.eye(3), com=np.array([0.0, 0.0, -0.2])),
                X_tree=SpatialTransform(R1, r1),
                parent=-1,
            )
        )
        tree.add_body(
            Body(
                name="link1",
                index=1,
                joint=RevoluteJoint("j1", axis=Axis.X),
                inertia=SpatialInertia(mass=0.5, inertia=0.02 * np.eye(3), com=np.array([0.0, 0.0, -0.15])),
                X_tree=SpatialTransform(R2, r2),
                parent=0,
            )
        )
        tree.finalize()

        # Pinocchio equivalent
        model = pin.Model()
        model.gravity.linear = np.array([0.0, 0.0, -9.81])

        p1 = pin.SE3(R1, r1)
        jid1 = model.addJoint(0, pin.JointModelRY(), p1, "j0")
        model.appendBodyToJoint(
            jid1,
            pin.Inertia(1.0, np.zeros(3), 0.05 * np.eye(3)),
            pin.SE3(np.eye(3), np.array([0.0, 0.0, -0.2])),
        )

        p2 = pin.SE3(R2, r2)
        jid2 = model.addJoint(jid1, pin.JointModelRX(), p2, "j1")
        model.appendBodyToJoint(
            jid2,
            pin.Inertia(0.5, np.zeros(3), 0.02 * np.eye(3)),
            pin.SE3(np.eye(3), np.array([0.0, 0.0, -0.15])),
        )

        data = model.createData()

        configs = [
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.3, -0.4, 1.0, -0.5, 0.5, -1.0),
            (np.pi / 4, -np.pi / 6, 0.0, 0.0, 0.0, 0.0),
            (-0.5, 0.8, -1.5, 2.0, 1.0, -0.5),
        ]
        for q0, q1, qd0, qd1, t0, t1 in configs:
            q = np.array([q0, q1])
            qdot = np.array([qd0, qd1])
            tau = np.array([t0, t1])

            ours = tree.aba(q, qdot, tau)
            ref = pin.aba(model, data, q, qdot, tau).copy()

            np.testing.assert_allclose(ours, ref, atol=1e-8, err_msg=f"q={q}, qdot={qdot}, tau={tau}")
