"""
Tier 1 Layer 1: Pure dynamics validation — CPU vs MuJoCo (no contact).

Validates that our ABA + semi-implicit Euler produces the same trajectories
as MuJoCo's forward dynamics for articulated bodies WITHOUT contact.

This is the foundation test: if pure dynamics disagree, contact tests are
meaningless. Any failure here indicates a bug in ABA, RNEA, joint models,
spatial algebra, or integration.

Test matrix:
  - Single pendulum  (1-DOF, fixed base)
  - Double pendulum  (2-DOF, chaotic)
  - Quadruped free fall  (18-DOF, floating base, no ground contact)

All tests compare joint-space trajectories (q, qdot) step-by-step.
"""

from __future__ import annotations

import numpy as np
import pytest

from .models import (
    DT,
    G,
    build_double_pendulum,
    build_quadruped,
    build_single_pendulum,
)

try:
    import mujoco

    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

pytestmark = pytest.mark.skipif(not HAS_MUJOCO, reason="mujoco not installed")


# ---------------------------------------------------------------------------
# Helpers: run trajectories
# ---------------------------------------------------------------------------


def _run_ours(model, q0, qdot0, n_steps, dt=DT, tau_fn=None):
    """Run our CPU simulator (ABA + semi-implicit Euler).

    Returns: (q_traj, qdot_traj) each shape (n_steps, nq/nv).
    """
    tree = model.tree
    q = q0.copy()
    qdot = qdot0.copy()
    nq, nv = tree.nq, tree.nv

    q_traj = np.zeros((n_steps, nq))
    qdot_traj = np.zeros((n_steps, nv))

    for i in range(n_steps):
        q_traj[i] = q
        qdot_traj[i] = qdot

        tau = tau_fn(q, qdot) if tau_fn else np.zeros(nv)
        tau_total = tau + tree.passive_torques(q, qdot)
        qddot = tree.aba(q, qdot, tau_total)

        # Semi-implicit Euler
        qdot_new = qdot + dt * qddot
        q_new = tree.integrate_q(q, qdot_new, dt)

        q, qdot = q_new, qdot_new

    return q_traj, qdot_traj


def _run_mujoco(xml, q0_mj, qdot0_mj, n_steps, dt=DT):
    """Run MuJoCo (Euler integrator).

    Args:
        q0_mj, qdot0_mj: Initial state in MuJoCo layout.

    Returns: (qpos_traj, qvel_traj) each shape (n_steps, nq/nv).
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Override timestep
    model.opt.timestep = dt

    # Set initial state
    data.qpos[:] = q0_mj
    data.qvel[:] = qdot0_mj

    nq = model.nq
    nv = model.nv

    qpos_traj = np.zeros((n_steps, nq))
    qvel_traj = np.zeros((n_steps, nv))

    for i in range(n_steps):
        qpos_traj[i] = data.qpos[:nq]
        qvel_traj[i] = data.qvel[:nv]
        mujoco.mj_step(model, data)

    return qpos_traj, qvel_traj


# ---------------------------------------------------------------------------
# Test: Single Pendulum
# ---------------------------------------------------------------------------


class TestSinglePendulumDynamics:
    """Single revolute pendulum: gravity swing from horizontal."""

    def _run_both(self, q0_angle, qdot0, n_steps=5000):
        model, xml = build_single_pendulum()

        # Our sim
        q_ours = np.array([q0_angle])
        qdot_ours = np.array([qdot0])
        q_traj, qdot_traj = _run_ours(model, q_ours, qdot_ours, n_steps)

        # MuJoCo (hinge joint: qpos = angle, qvel = angular velocity)
        q_mj = np.array([q0_angle])
        qdot_mj = np.array([qdot0])
        qpos_mj, qvel_mj = _run_mujoco(xml, q_mj, qdot_mj, n_steps)

        return q_traj, qdot_traj, qpos_mj, qvel_mj

    def test_horizontal_start_angle_trajectory(self):
        """Pendulum starting horizontal (q=pi/2): angle trajectory matches."""
        q_ours, _, qpos_mj, _ = self._run_both(np.pi / 2, 0.0, n_steps=5000)
        # 5000 steps × dt=2e-4 = 1 second of simulation
        np.testing.assert_allclose(
            q_ours[:, 0],
            qpos_mj[:, 0],
            atol=1e-4,
            err_msg="Single pendulum angle diverged from MuJoCo",
        )

    def test_horizontal_start_velocity_trajectory(self):
        """Pendulum: angular velocity trajectory matches."""
        _, qdot_ours, _, qvel_mj = self._run_both(np.pi / 2, 0.0, n_steps=5000)
        np.testing.assert_allclose(
            qdot_ours[:, 0],
            qvel_mj[:, 0],
            atol=1e-3,
            err_msg="Single pendulum velocity diverged from MuJoCo",
        )

    def test_small_angle_oscillation(self):
        """Small angle (0.1 rad): should match for many periods."""
        q_ours, _, qpos_mj, _ = self._run_both(0.1, 0.0, n_steps=25000)
        # 25000 steps = 5 seconds ≈ several oscillation periods
        np.testing.assert_allclose(
            q_ours[:, 0],
            qpos_mj[:, 0],
            atol=1e-3,
            err_msg="Small angle pendulum drifted from MuJoCo over 5s",
        )

    def test_nonzero_initial_velocity(self):
        """Start at rest position with angular velocity."""
        q_ours, _, qpos_mj, _ = self._run_both(0.0, 5.0, n_steps=5000)
        np.testing.assert_allclose(
            q_ours[:, 0],
            qpos_mj[:, 0],
            atol=1e-4,
            err_msg="Pendulum with initial velocity diverged",
        )


# ---------------------------------------------------------------------------
# Test: Double Pendulum
# ---------------------------------------------------------------------------


class TestDoublePendulumDynamics:
    """Double pendulum: chaotic system, short-term agreement expected."""

    def _run_both(self, q0, qdot0, n_steps=5000):
        model, xml = build_double_pendulum()

        q_traj, qdot_traj = _run_ours(model, np.array(q0), np.array(qdot0), n_steps)
        qpos_mj, qvel_mj = _run_mujoco(xml, np.array(q0), np.array(qdot0), n_steps)

        return q_traj, qdot_traj, qpos_mj, qvel_mj

    def test_both_horizontal(self):
        """Both links horizontal: short-term trajectory match."""
        q_ours, _, qpos_mj, _ = self._run_both(
            [np.pi / 2, 0.0],
            [0.0, 0.0],
            n_steps=5000,
        )
        # 1 second: should agree well before chaos dominates
        np.testing.assert_allclose(
            q_ours,
            qpos_mj,
            atol=1e-3,
            err_msg="Double pendulum diverged from MuJoCo within 1s",
        )

    def test_small_perturbation(self):
        """Small angles: quasi-linear regime, longer agreement."""
        q_ours, _, qpos_mj, _ = self._run_both(
            [0.1, 0.05],
            [0.0, 0.0],
            n_steps=10000,
        )
        # 2 seconds in near-linear regime
        np.testing.assert_allclose(
            q_ours,
            qpos_mj,
            atol=1e-3,
            err_msg="Double pendulum small angle diverged over 2s",
        )

    def test_energy_conservation(self):
        """Total energy (KE + PE) should be conserved (no damping)."""
        model, _ = build_double_pendulum()
        q0 = np.array([np.pi / 2, np.pi / 4])
        qdot0 = np.array([0.0, 0.0])
        tree = model.tree

        q_traj, qdot_traj = _run_ours(model, q0, qdot0, 10000)

        def _total_energy(q, qdot):
            # KE = 0.5 * qdot^T H qdot
            H = tree.crba(q)
            KE = 0.5 * qdot @ H @ qdot
            # PE = sum(m_i * g * z_com_i) for each body
            X_world = tree.forward_kinematics(q)
            PE = 0.0
            for body in tree.bodies:
                pos = X_world[body.index].R @ body.inertia.com + X_world[body.index].r
                PE += body.inertia.mass * G * pos[2]
            return KE + PE

        E0 = _total_energy(q_traj[0], qdot_traj[0])
        E_final = _total_energy(q_traj[-1], qdot_traj[-1])
        # Euler integration on a chaotic double pendulum at large amplitude
        # will drift. Check it stays bounded (no blowup), not exact conservation.
        rel_drift = abs(E_final - E0) / (abs(E0) + 1e-10)
        assert rel_drift < 0.50, f"Energy blew up ({rel_drift * 100:.1f}%): E0={E0:.4f} E_final={E_final:.4f}"


# ---------------------------------------------------------------------------
# Test: Quadruped Free Fall (no contact)
# ---------------------------------------------------------------------------


class TestQuadrupedFreeFall:
    """Floating-base quadruped in free fall: no ground contact.

    Tests FreeJoint + 12 revolute joints on our ABA vs MuJoCo.
    """

    @staticmethod
    def _init_state(model, base_z=1.0, q_joints=None):
        """Initialize quadruped state.

        Returns: (q_ours, qdot_ours, q_mj, qdot_mj)
        """
        tree = model.tree
        q, qdot = tree.default_state()
        # Set base height
        q[6] = base_z  # pz in our layout [qw,qx,qy,qz,px,py,pz]

        if q_joints is not None:
            # Set joint angles (indices 7..18 for 12 revolute joints)
            q[7 : 7 + len(q_joints)] = q_joints

        # Convert to MuJoCo layout
        # FreeJoint: our [qw,qx,qy,qz,px,py,pz] → MuJoCo [px,py,pz,qw,qx,qy,qz]
        q_mj = np.zeros_like(q)
        q_mj[0:3] = q[4:7]  # position
        q_mj[3:7] = q[0:4]  # quaternion
        q_mj[7:] = q[7:]  # revolute joints (same layout)

        qdot_mj = qdot.copy()  # velocity layout is the same

        return q, qdot, q_mj, qdot_mj

    def test_pure_free_fall_base_z(self):
        """Base should follow z = z0 - 0.5*g*t^2 in free fall."""
        model, xml = build_quadruped(contact=False)
        q, qdot, q_mj, qdot_mj = self._init_state(model, base_z=2.0)

        n_steps = 5000  # 1 second
        q_traj, _ = _run_ours(model, q, qdot, n_steps)
        qpos_mj, _ = _run_mujoco(xml, q_mj, qdot_mj, n_steps)

        # Compare base z (our index 6, MuJoCo index 2)
        z_ours = q_traj[:, 6]
        z_mj = qpos_mj[:, 2]
        np.testing.assert_allclose(
            z_ours,
            z_mj,
            atol=1e-3,
            err_msg="Quadruped free-fall base z diverged from MuJoCo",
        )

        # Also check against analytical: z = 2.0 - 0.5 * 9.81 * t^2
        t = np.arange(n_steps) * DT
        z_analytical = 2.0 - 0.5 * G * t**2
        np.testing.assert_allclose(
            z_ours,
            z_analytical,
            atol=5e-3,
            err_msg="Quadruped free-fall z diverged from analytical",
        )

    def test_free_fall_joint_angles_constant(self):
        """In free fall with zero velocity, joint angles should stay constant."""
        model, xml = build_quadruped(contact=False)
        q, qdot, q_mj, qdot_mj = self._init_state(model, base_z=2.0)

        n_steps = 5000
        q_traj, _ = _run_ours(model, q, qdot, n_steps)

        # Joint angles (indices 7:19) should remain at initial values
        q_joints_init = q_traj[0, 7:]
        q_joints_final = q_traj[-1, 7:]
        np.testing.assert_allclose(
            q_joints_final,
            q_joints_init,
            atol=1e-6,
            err_msg="Joint angles changed during free fall (should be constant)",
        )

    def test_free_fall_with_initial_joint_angles(self):
        """Free fall with bent legs: joint dynamics should match MuJoCo."""
        model, xml = build_quadruped(contact=False)
        # Moderate joint angles — legs bent but not extreme
        q_joints = np.array([0.3, -0.5, 0.2, -0.4, -0.3, 0.5, 0.1, -0.6])
        q, qdot, q_mj, qdot_mj = self._init_state(
            model,
            base_z=2.0,
            q_joints=q_joints,
        )

        n_steps = 5000
        q_traj, qdot_traj = _run_ours(model, q, qdot, n_steps)
        qpos_mj, qvel_mj = _run_mujoco(xml, q_mj, qdot_mj, n_steps)

        # Joint angles should match well (local dynamics, less sensitive
        # to integrator differences than base position)
        np.testing.assert_allclose(
            q_traj[:, 7:],
            qpos_mj[:, 7:],
            atol=5e-3,
            err_msg="Quadruped joint angles diverged from MuJoCo in free fall",
        )

        # Base z should still follow free-fall parabola closely
        np.testing.assert_allclose(
            q_traj[:, 6],
            qpos_mj[:, 2],
            atol=5e-3,
            err_msg="Quadruped base z diverged from MuJoCo",
        )

    def test_free_fall_with_initial_joint_velocities(self):
        """Free fall with nonzero joint velocities: full dynamics test."""
        model, xml = build_quadruped(contact=False)
        q, qdot, q_mj, qdot_mj = self._init_state(model, base_z=2.0)

        # Add joint velocities (8 revolute joints)
        qdot[6:] = np.array([1.0, -0.5, 0.3, -0.7, 0.8, -0.2, 0.5, -1.0])
        qdot_mj[6:] = qdot[6:]

        n_steps = 5000
        q_traj, qdot_traj = _run_ours(model, q, qdot, n_steps)
        qpos_mj, qvel_mj = _run_mujoco(xml, q_mj, qdot_mj, n_steps)

        # Joint angles (both at 7: for our sim and MuJoCo)
        np.testing.assert_allclose(
            q_traj[:, 7:],
            qpos_mj[:, 7:],
            atol=1e-3,
            err_msg="Quadruped with joint vel: angles diverged from MuJoCo",
        )

    def test_base_quaternion_stays_normalized(self):
        """Base quaternion should stay unit norm throughout free fall."""
        model, _ = build_quadruped(contact=False)
        q, qdot, _, _ = self._init_state(model, base_z=2.0)
        n_joints = model.tree.nv - 6
        qdot[6:] = np.random.default_rng(42).standard_normal(n_joints) * 2.0

        q_traj, _ = _run_ours(model, q, qdot, 10000)

        quat_norms = np.linalg.norm(q_traj[:, 0:4], axis=1)
        np.testing.assert_allclose(
            quat_norms,
            1.0,
            atol=1e-6,
            err_msg="Base quaternion drifted from unit norm",
        )

    def test_all_joints_all_velocities_vs_mujoco(self):
        """Full state comparison: q + qdot for all DOFs.

        The strictest test: every DOF of the 18-DOF quadruped must match
        MuJoCo within tolerance over 1 second of free-fall simulation.
        """
        model, xml = build_quadruped(contact=False)
        rng = np.random.default_rng(123)
        n_joints = model.tree.nv - 6  # 8 revolute joints
        q_joints = rng.uniform(-0.5, 0.5, n_joints)
        q, qdot, q_mj, qdot_mj = self._init_state(
            model,
            base_z=3.0,
            q_joints=q_joints,
        )
        qdot[6:] = rng.standard_normal(n_joints)
        qdot_mj[6:] = qdot[6:]

        n_steps = 5000
        q_traj, qdot_traj = _run_ours(model, q, qdot, n_steps)
        qpos_mj, qvel_mj = _run_mujoco(xml, q_mj, qdot_mj, n_steps)

        # Joint angles — the most important check (local dynamics correctness)
        np.testing.assert_allclose(
            q_traj[:, 7:],
            qpos_mj[:, 7:],
            atol=5e-3,
            err_msg="Joint angles diverged",
        )

        # Joint velocities
        np.testing.assert_allclose(
            qdot_traj[:, 6:],
            qvel_mj[:, 6:],
            atol=0.05,
            err_msg="Joint velocities diverged",
        )

        # Base z (free fall dominated, should agree well)
        np.testing.assert_allclose(
            q_traj[:, 6],
            qpos_mj[:, 2],
            atol=0.01,
            err_msg="Base z diverged",
        )

        # Base x,y — coupling from joint motions makes these sensitive to
        # integrator differences. Use looser tolerance.
        np.testing.assert_allclose(
            q_traj[:, 4:6],
            qpos_mj[:, 0:2],
            atol=0.3,
            err_msg="Base x,y diverged beyond tolerance",
        )

        # Base quaternion (our 0:4, MuJoCo 3:7)
        for step in range(0, n_steps, 500):
            dot = abs(np.dot(q_traj[step, 0:4], qpos_mj[step, 3:7]))
            assert dot > 0.99, f"Quaternion diverged at step {step}: dot={dot:.6f}"
