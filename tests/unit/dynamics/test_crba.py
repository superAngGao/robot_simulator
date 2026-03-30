"""
Tests for CRBA (Composite Rigid Body Algorithm).

Validates:
  1. Mass matrix H is symmetric positive definite
  2. H dimensions match nv x nv
  3. CRBA forward dynamics == ABA (identical qddot)
  4. H @ qddot + C == tau (inverse/forward dynamics roundtrip)
  5. Pinocchio comparison (if available)
"""

from __future__ import annotations

import os
import tempfile
import textwrap

import numpy as np
import pytest

from physics.joint import FreeJoint, RevoluteJoint
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from robot import load_urdf

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_simple_tree(gravity=9.81):
    """Floating base + 2 revolute joints."""
    tree = RobotTreeNumpy(gravity=gravity)
    tree.add_body(
        Body(
            name="base",
            index=0,
            joint=FreeJoint("root"),
            inertia=SpatialInertia.from_box(5.0, 0.3, 0.2, 0.1),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.add_body(
        Body(
            name="link1",
            index=1,
            joint=RevoluteJoint("j1", axis=np.array([0, 1, 0]), damping=0.1),
            inertia=SpatialInertia(0.5, np.diag([0.001, 0.001, 0.001]), np.array([0, 0, -0.1])),
            X_tree=SpatialTransform(np.eye(3), np.array([0.15, 0.0, 0.0])),
            parent=0,
        )
    )
    tree.add_body(
        Body(
            name="link2",
            index=2,
            joint=RevoluteJoint("j2", axis=np.array([0, 1, 0]), damping=0.05),
            inertia=SpatialInertia(0.3, np.diag([0.001, 0.001, 0.001]), np.array([0, 0, -0.1])),
            X_tree=SpatialTransform(np.eye(3), np.array([0.0, 0.0, -0.2])),
            parent=1,
        )
    )
    tree.finalize()
    return tree


def _make_quadruped():
    """Load quadruped from URDF."""
    urdf = """
    <robot name="quad">
      <link name="base"><inertial><mass value="5.0"/><origin xyz="0 0 0"/>
        <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.05"/></inertial></link>
      <link name="FL_hip"><inertial><mass value="0.5"/><origin xyz="0 0 -0.1"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
      <joint name="FL_hip_j" type="revolute"><parent link="base"/><child link="FL_hip"/>
        <origin xyz="0.15 0.1 0"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="20"/></joint>
      <link name="FL_calf"><inertial><mass value="0.3"/><origin xyz="0 0 -0.1"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
      <joint name="FL_calf_j" type="revolute"><parent link="FL_hip"/><child link="FL_calf"/>
        <origin xyz="0 0 -0.2"/><axis xyz="0 1 0"/><limit lower="-2" upper="0.5" effort="20"/></joint>
      <link name="FL_foot"><inertial><mass value="0.05"/><origin xyz="0 0 0"/>
        <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial></link>
      <joint name="FL_foot_j" type="fixed"><parent link="FL_calf"/><child link="FL_foot"/>
        <origin xyz="0 0 -0.2"/></joint>
      <link name="FR_hip"><inertial><mass value="0.5"/><origin xyz="0 0 -0.1"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
      <joint name="FR_hip_j" type="revolute"><parent link="base"/><child link="FR_hip"/>
        <origin xyz="0.15 -0.1 0"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="20"/></joint>
      <link name="FR_calf"><inertial><mass value="0.3"/><origin xyz="0 0 -0.1"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
      <joint name="FR_calf_j" type="revolute"><parent link="FR_hip"/><child link="FR_calf"/>
        <origin xyz="0 0 -0.2"/><axis xyz="0 1 0"/><limit lower="-2" upper="0.5" effort="20"/></joint>
      <link name="FR_foot"><inertial><mass value="0.05"/><origin xyz="0 0 0"/>
        <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial></link>
      <joint name="FR_foot_j" type="fixed"><parent link="FR_calf"/><child link="FR_foot"/>
        <origin xyz="0 0 -0.2"/></joint>
    </robot>
    """
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
    f.write(textwrap.dedent(urdf))
    f.close()
    model = load_urdf(f.name, floating_base=True, contact_links=[])
    os.unlink(f.name)
    return model.tree


# ---------------------------------------------------------------------------
# Tests: mass matrix properties
# ---------------------------------------------------------------------------


class TestMassMatrixProperties:
    """H must be symmetric positive definite with correct shape."""

    def test_shape(self):
        tree = _make_simple_tree()
        q, _ = tree.default_state()
        H = tree.crba(q)
        assert H.shape == (tree.nv, tree.nv)

    def test_symmetric(self):
        tree = _make_simple_tree()
        q, _ = tree.default_state()
        H = tree.crba(q)
        np.testing.assert_allclose(H, H.T, atol=1e-12)

    def test_positive_definite(self):
        tree = _make_simple_tree()
        q, _ = tree.default_state()
        H = tree.crba(q)
        eigvals = np.linalg.eigvalsh(H)
        assert np.all(eigvals > 0), f"H not PD: min eigenvalue = {eigvals.min()}"

    def test_nonzero_q(self):
        """H should still be SPD at non-zero configuration."""
        tree = _make_simple_tree()
        q, _ = tree.default_state()
        q[7] = 0.5  # revolute joint 1
        q[8] = -0.3  # revolute joint 2
        H = tree.crba(q)
        np.testing.assert_allclose(H, H.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(H) > 0)

    def test_quadruped_shape(self):
        tree = _make_quadruped()
        q, _ = tree.default_state()
        H = tree.crba(q)
        assert H.shape == (tree.nv, tree.nv)
        np.testing.assert_allclose(H, H.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(H) > 0)


# ---------------------------------------------------------------------------
# Tests: CRBA vs ABA consistency
# ---------------------------------------------------------------------------


class TestCRBAvsABA:
    """CRBA forward dynamics must produce identical qddot as ABA."""

    def test_zero_state(self):
        tree = _make_simple_tree()
        q, qdot = tree.default_state()
        tau = np.zeros(tree.nv)

        qddot_aba = tree.aba(q, qdot, tau)
        qddot_crba = tree.forward_dynamics_crba(q, qdot, tau)

        np.testing.assert_allclose(qddot_crba, qddot_aba, atol=1e-10, err_msg="CRBA != ABA at zero state")

    def test_nonzero_tau(self):
        tree = _make_simple_tree()
        q, qdot = tree.default_state()
        tau = RNG.standard_normal(tree.nv) * 0.5

        qddot_aba = tree.aba(q, qdot, tau)
        qddot_crba = tree.forward_dynamics_crba(q, qdot, tau)

        np.testing.assert_allclose(qddot_crba, qddot_aba, atol=1e-10, err_msg="CRBA != ABA with nonzero tau")

    def test_nonzero_state(self):
        tree = _make_simple_tree()
        q, qdot = tree.default_state()
        q[7] = 0.5
        q[8] = -0.3
        qdot[6] = 0.1
        qdot[7] = -0.2
        tau = RNG.standard_normal(tree.nv) * 0.5

        qddot_aba = tree.aba(q, qdot, tau)
        qddot_crba = tree.forward_dynamics_crba(q, qdot, tau)

        np.testing.assert_allclose(qddot_crba, qddot_aba, atol=1e-10, err_msg="CRBA != ABA at nonzero state")

    def test_quadruped_zero_state(self):
        tree = _make_quadruped()
        q, qdot = tree.default_state()
        tau = np.zeros(tree.nv)

        qddot_aba = tree.aba(q, qdot, tau)
        qddot_crba = tree.forward_dynamics_crba(q, qdot, tau)

        np.testing.assert_allclose(
            qddot_crba, qddot_aba, atol=1e-10, err_msg="CRBA != ABA for quadruped at zero state"
        )

    def test_quadruped_random(self):
        tree = _make_quadruped()
        q, qdot = tree.default_state()
        # Random joint angles
        for body in tree.bodies:
            if body.joint.nv > 0 and not isinstance(body.joint, FreeJoint):
                q[body.q_idx] = RNG.uniform(-0.5, 0.5, body.joint.nq)
                qdot[body.v_idx] = RNG.standard_normal(body.joint.nv) * 0.3
        tau = RNG.standard_normal(tree.nv) * 2.0

        qddot_aba = tree.aba(q, qdot, tau)
        qddot_crba = tree.forward_dynamics_crba(q, qdot, tau)

        np.testing.assert_allclose(
            qddot_crba, qddot_aba, atol=1e-9, err_msg="CRBA != ABA for quadruped random state"
        )

    def test_with_external_forces(self):
        tree = _make_simple_tree()
        q, qdot = tree.default_state()
        tau = RNG.standard_normal(tree.nv) * 0.5

        ext = [None] * tree.num_bodies
        ext[1] = RNG.standard_normal(6)  # force on link1

        qddot_aba = tree.aba(q, qdot, tau, ext)
        qddot_crba = tree.forward_dynamics_crba(q, qdot, tau, ext)

        np.testing.assert_allclose(
            qddot_crba, qddot_aba, atol=1e-10, err_msg="CRBA != ABA with external forces"
        )


# ---------------------------------------------------------------------------
# Tests: inverse/forward dynamics roundtrip
# ---------------------------------------------------------------------------


class TestRoundtrip:
    """H @ qddot + C == tau  (CRBA mass matrix + RNEA bias = applied torque)."""

    def test_roundtrip(self):
        tree = _make_simple_tree()
        q, qdot = tree.default_state()
        q[7] = 0.3
        qdot[6] = 0.1
        tau = RNG.standard_normal(tree.nv) * 0.5

        qddot = tree.forward_dynamics_crba(q, qdot, tau)
        H = tree.crba(q)
        C = tree.rnea(q, qdot, np.zeros(tree.nv))

        tau_reconstructed = H @ qddot + C
        np.testing.assert_allclose(tau_reconstructed, tau, atol=1e-10, err_msg="H @ qddot + C != tau")


# ---------------------------------------------------------------------------
# Tests: Pinocchio comparison (if available)
# ---------------------------------------------------------------------------


class TestPinocchioComparison:
    @pytest.fixture(autouse=True)
    def _skip_if_no_pinocchio(self):
        pytest.importorskip("pinocchio")

    def test_mass_matrix_vs_pinocchio(self):
        import pinocchio as pin

        tree = _make_simple_tree(gravity=9.81)
        q, _ = tree.default_state()

        H_ours = tree.crba(q)

        # Build equivalent Pinocchio model
        model = pin.Model()
        # Base (free joint)
        base_id = model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "root")
        # Must match SpatialInertia.from_box(5.0, 0.3, 0.2, 0.1) = m/12*(ly²+lz², lx²+lz², lx²+ly²)
        Ibox = np.diag([5.0 / 12 * (0.04 + 0.01), 5.0 / 12 * (0.09 + 0.01), 5.0 / 12 * (0.09 + 0.04)])
        model.appendBodyToJoint(base_id, pin.Inertia(5.0, np.zeros(3), Ibox), pin.SE3.Identity())
        # Link1 (revolute Y)
        j1_placement = pin.SE3(np.eye(3), np.array([0.15, 0.0, 0.0]))
        j1_id = model.addJoint(base_id, pin.JointModelRY(), j1_placement, "j1")
        model.appendBodyToJoint(
            j1_id,
            pin.Inertia(0.5, np.array([0, 0, -0.1]), np.diag([0.001, 0.001, 0.001])),
            pin.SE3.Identity(),
        )
        # Link2 (revolute Y)
        j2_placement = pin.SE3(np.eye(3), np.array([0.0, 0.0, -0.2]))
        j2_id = model.addJoint(j1_id, pin.JointModelRY(), j2_placement, "j2")
        model.appendBodyToJoint(
            j2_id,
            pin.Inertia(0.3, np.array([0, 0, -0.1]), np.diag([0.001, 0.001, 0.001])),
            pin.SE3.Identity(),
        )

        data = model.createData()

        # Pinocchio uses [x,y,z,qx,qy,qz,qw] for free joint (7 DOF)
        # Our convention: [qw,qx,qy,qz,x,y,z]
        q_pin = np.zeros(model.nq)
        q_pin[6] = 1.0  # qw (Pinocchio puts qw last in the 7-element vector)
        q_pin[7] = q[7]  # revolute j1
        q_pin[8] = q[8]  # revolute j2

        pin.crba(model, data, q_pin)
        H_pin = data.M.copy()
        # Make symmetric (Pinocchio only fills upper triangle)
        H_pin = np.triu(H_pin) + np.triu(H_pin, 1).T

        np.testing.assert_allclose(
            H_ours, H_pin, atol=1e-8, err_msg="CRBA mass matrix differs from Pinocchio"
        )
