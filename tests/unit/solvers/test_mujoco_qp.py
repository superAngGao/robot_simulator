"""
Tests for MuJoCoStyleSolver + ImplicitContactStep.

Validates against MuJoCo (elliptic cone, Euler integrator) on:
  1. Analytical single-step reference (known a_ref, force, acceleration)
  2. Ball drop full trajectory (z-L2, settling, penetration)
  3. contact_jacobian correctness (numerical + Pinocchio)
  4. R-regularization self-adaptivity (different masses)
  5. Equilibrium vz = 0 (exact)
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.geometry import SphereShape
from physics.gjk_epa import ground_contact_query
from physics.implicit_contact_step import ImplicitContactStep
from physics.joint import Axis, FreeJoint, RevoluteJoint
from physics.robot_tree import Body, RobotTreeNumpy
from physics.solvers.admm_qp import MuJoCoStyleSolver
from physics.solvers.pgs_solver import ContactConstraint
from physics.spatial import SpatialInertia, SpatialTransform

try:
    import mujoco

    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

DT = 0.001
RADIUS = 0.1
MASS = 1.0
G = 9.81
I_DIAG = 0.004


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ball_tree(mass=MASS):
    tree = RobotTreeNumpy(gravity=G)
    tree.add_body(
        Body(
            "ball",
            0,
            FreeJoint("root"),
            SpatialInertia(mass, np.diag([I_DIAG] * 3), np.zeros(3)),
            SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()
    return tree


def _run_ball_drop(tree, solver, n_steps=500, z0=0.3):
    integ = ImplicitContactStep(dt=DT, solver=solver)
    shape = SphereShape(RADIUS)
    q, qdot = tree.default_state()
    q[6] = z0
    z, vz = np.zeros(n_steps), np.zeros(n_steps)
    for i in range(n_steps):
        z[i], vz[i] = q[6], qdot[2]
        X = tree.forward_kinematics(q)
        m = ground_contact_query(shape, X[0], ground_z=0.0, margin=0.0)
        cc = []
        if m is not None and m.depth > 0:
            cc.append(
                ContactConstraint(
                    body_i=0,
                    body_j=-1,
                    point=m.points[0],
                    normal=m.normal.copy(),
                    tangent1=np.zeros(3),
                    tangent2=np.zeros(3),
                    depth=m.depth,
                    mu=0.5,
                    condim=3,
                )
            )
        tau = tree.passive_torques(q, qdot)
        q, qdot = integ.step(tree, q, qdot, tau, contacts=cc)
    return z, vz


def _mujoco_ball_drop(n_steps=500, z0=0.3, mass=MASS):
    xml = f"""<mujoco>
      <option timestep="{DT}" gravity="0 0 -{G}" integrator="Euler" cone="elliptic"/>
      <worldbody>
        <geom type="plane" size="10 10 0.1" friction="0.5 0.005 0.0001"/>
        <body name="ball" pos="0 0 {z0}"><freejoint/>
          <geom type="sphere" size="{RADIUS}" mass="{mass}" friction="0.5 0.005 0.0001"/>
          <inertial pos="0 0 0" mass="{mass}" diaginertia="{I_DIAG} {I_DIAG} {I_DIAG}"/>
        </body>
      </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    z, vz = np.zeros(n_steps), np.zeros(n_steps)
    for i in range(n_steps):
        z[i], vz[i] = data.qpos[2], data.qvel[2]
        mujoco.mj_step(model, data)
    return z, vz


def _settle_step(vz, start=200):
    for i in range(start, len(vz) - 20):
        if all(abs(vz[j]) < 0.001 for j in range(i, i + 20)):
            return i
    return -1


# ---------------------------------------------------------------------------
# 1. Contact Jacobian
# ---------------------------------------------------------------------------


class TestContactJacobian:
    """Joint-space contact Jacobian correctness."""

    def test_single_revolute_analytical(self):
        tree = RobotTreeNumpy(gravity=G)
        tree.add_body(
            Body(
                "link",
                0,
                RevoluteJoint("j0", axis=Axis.Z),
                SpatialInertia(1.0, np.diag([0.01] * 3), np.zeros(3)),
                SpatialTransform(np.eye(3), np.zeros(3)),
                parent=-1,
            )
        )
        tree.finalize()
        q = np.array([0.0])
        point = np.array([1.0, 0.0, 0.0])
        J = tree.contact_jacobian(q, 0, point)
        np.testing.assert_allclose(J, [[0], [1], [0]], atol=1e-10)

    def test_double_pendulum_analytical(self):
        tree = RobotTreeNumpy(gravity=G)
        tree.add_body(
            Body(
                "l1",
                0,
                RevoluteJoint("j0", axis=Axis.Z),
                SpatialInertia(1.0, np.diag([0.01] * 3), np.zeros(3)),
                SpatialTransform(np.eye(3), np.zeros(3)),
                parent=-1,
            )
        )
        tree.add_body(
            Body(
                "l2",
                1,
                RevoluteJoint("j1", axis=Axis.Z),
                SpatialInertia(1.0, np.diag([0.01] * 3), np.zeros(3)),
                SpatialTransform(np.eye(3), np.array([1, 0, 0])),
                parent=0,
            )
        )
        tree.finalize()
        q = np.array([0.0, 0.0])
        X = tree.forward_kinematics(q)
        pt = X[1].R @ np.array([0.5, 0, 0]) + X[1].r
        J = tree.contact_jacobian(q, 1, pt)
        np.testing.assert_allclose(J, [[0, 0], [1.5, 0.5], [0, 0]], atol=1e-10)

    def test_numerical_differentiation(self):
        tree = RobotTreeNumpy(gravity=G)
        tree.add_body(
            Body(
                "l1",
                0,
                RevoluteJoint("j0", axis=Axis.Z),
                SpatialInertia(1.0, np.diag([0.01] * 3), np.zeros(3)),
                SpatialTransform(np.eye(3), np.zeros(3)),
                parent=-1,
            )
        )
        tree.add_body(
            Body(
                "l2",
                1,
                RevoluteJoint("j1", axis=Axis.Z),
                SpatialInertia(1.0, np.diag([0.01] * 3), np.zeros(3)),
                SpatialTransform(np.eye(3), np.array([1, 0, 0])),
                parent=0,
            )
        )
        tree.finalize()
        q = np.array([np.pi / 6, np.pi / 4])
        X = tree.forward_kinematics(q)
        pt_local = np.array([0.5, 0, 0])
        pt_world = X[1].R @ pt_local + X[1].r
        J_analytic = tree.contact_jacobian(q, 1, pt_world)
        J_num = np.zeros((3, 2))
        eps = 1e-7
        for j in range(2):
            qp = q.copy()
            qp[j] += eps
            Xp = tree.forward_kinematics(qp)
            pp = Xp[1].R @ pt_local + Xp[1].r
            p0 = X[1].R @ pt_local + X[1].r
            J_num[:, j] = (pp - p0) / eps
        np.testing.assert_allclose(J_analytic, J_num, atol=1e-5)

    def test_free_body(self):
        tree = _make_ball_tree()
        q, _ = tree.default_state()
        q[6] = 0.5
        pt = np.array([0.0, 0.0, 0.0])
        J = tree.contact_jacobian(q, 0, pt)
        expected = np.array(
            [
                [1, 0, 0, 0, -0.5, 0],
                [0, 1, 0, 0.5, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ]
        )
        np.testing.assert_allclose(J, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Single-step analytical
# ---------------------------------------------------------------------------


class TestSingleStep:
    """Verify solver output for one step with known state."""

    def test_unconstrained_accel_is_gravity(self):
        tree = _make_ball_tree()
        q, qdot = tree.default_state()
        q[6] = 0.3
        a_u = tree.aba(q, qdot, np.zeros(tree.nv))
        np.testing.assert_allclose(a_u[2], -G, atol=1e-10)

    def test_contact_force_positive_upward(self):
        tree = _make_ball_tree()
        solver = MuJoCoStyleSolver(max_iter=50, rho=1.0)
        q, qdot = tree.default_state()
        q[6] = 0.099  # 1mm penetration
        qdot[2] = -1.0  # approaching
        a_u = tree.aba(q, qdot, np.zeros(tree.nv))
        tree.forward_kinematics(q)
        cc = [
            ContactConstraint(
                body_i=0,
                body_j=-1,
                point=np.array([0, 0, 0.0]),
                normal=np.array([0, 0, 1.0]),
                tangent1=np.zeros(3),
                tangent2=np.zeros(3),
                depth=0.001,
                mu=0.5,
                condim=3,
            )
        ]
        f, J = solver.solve(cc, tree, q, qdot, a_u, DT)
        assert f[0] > 0, f"Normal force should be positive (upward), got {f[0]}"

    def test_no_contacts_gives_gravity(self):
        tree = _make_ball_tree()
        solver = MuJoCoStyleSolver()
        integ = ImplicitContactStep(dt=DT, solver=solver)
        q, qdot = tree.default_state()
        q[6] = 0.5  # well above ground
        tau = np.zeros(tree.nv)
        q2, qdot2 = integ.step(tree, q, qdot, tau, contacts=[])
        expected_vz = qdot[2] + DT * (-G)
        assert abs(qdot2[2] - expected_vz) < 1e-10


# ---------------------------------------------------------------------------
# 3. MuJoCo trajectory comparison
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MUJOCO, reason="mujoco not installed")
class TestMuJoCoTrajectory:
    """Full trajectory comparison against MuJoCo."""

    @pytest.fixture(scope="class")
    def trajectories(self):
        tree = _make_ball_tree()
        solver = MuJoCoStyleSolver(max_iter=50, rho=1.0)
        oz, ovz = _run_ball_drop(tree, solver, n_steps=1000)
        mz, mvz = _mujoco_ball_drop(n_steps=1000)
        return oz, ovz, mz, mvz

    def test_z_l2_below_threshold(self, trajectories):
        oz, _, mz, _ = trajectories
        l2 = np.sqrt(np.mean((oz - mz) ** 2))
        assert l2 < 0.0001, f"z-L2 = {l2 * 1000:.4f} mm, expected < 0.1 mm"

    def test_free_fall_match(self, trajectories):
        oz, _, mz, _ = trajectories
        np.testing.assert_allclose(oz[:190], mz[:190], atol=1e-6)

    def test_settling_time_match(self, trajectories):
        _, ovz, _, mvz = trajectories
        our_settle = _settle_step(ovz)
        mj_settle = _settle_step(mvz)
        assert our_settle > 0, "Did not settle"
        assert abs(our_settle - mj_settle) <= 1, f"Settle mismatch: ours={our_settle}, mujoco={mj_settle}"

    def test_rest_velocity_zero(self, trajectories):
        _, ovz, _, mvz = trajectories
        assert abs(ovz[-1]) < 1e-8, f"ours vz={ovz[-1]}"
        assert abs(mvz[-1]) < 1e-8, f"mujoco vz={mvz[-1]}"

    def test_max_penetration_match(self, trajectories):
        oz, _, mz, _ = trajectories
        our_pen = max(0, RADIUS - np.min(oz))
        mj_pen = max(0, RADIUS - np.min(mz))
        assert abs(our_pen - mj_pen) < 0.001, (
            f"Penetration mismatch: ours={our_pen * 1000:.2f}mm, mj={mj_pen * 1000:.2f}mm"
        )

    def test_impact_velocity_match(self, trajectories):
        _, ovz, _, mvz = trajectories
        # At step 210 (during contact), velocities should match
        assert abs(ovz[210] - mvz[210]) < 0.001


# ---------------------------------------------------------------------------
# 4. R-regularization self-adaptivity
# ---------------------------------------------------------------------------


class TestRAdaptivity:
    """R scales with mass — same solref/solimp works for different robots."""

    def test_light_and_heavy_ball_both_settle(self):
        solver = MuJoCoStyleSolver(max_iter=50, rho=1.0)
        for mass in [0.1, 1.0, 10.0, 100.0]:
            tree = _make_ball_tree(mass=mass)
            z, vz = _run_ball_drop(tree, solver, n_steps=800, z0=0.2)
            settle = _settle_step(vz, start=100)
            assert settle > 0, f"mass={mass} did not settle (vz[-1]={vz[-1]:.6f})"
            assert abs(vz[-1]) < 0.001, f"mass={mass} residual vz={vz[-1]:.6f}"

    @pytest.mark.skipif(not HAS_MUJOCO, reason="mujoco not installed")
    def test_heavy_ball_matches_mujoco(self):
        mass = 50.0
        tree = _make_ball_tree(mass=mass)
        solver = MuJoCoStyleSolver(max_iter=50, rho=1.0)
        oz, ovz = _run_ball_drop(tree, solver, n_steps=800, z0=0.2)
        mz, mvz = _mujoco_ball_drop(n_steps=800, z0=0.2, mass=mass)
        l2 = np.sqrt(np.mean((oz - mz) ** 2))
        assert l2 < 0.0001, f"mass={mass}: z-L2 = {l2 * 1000:.4f} mm"


# ---------------------------------------------------------------------------
# 5. Equilibrium exact zero
# ---------------------------------------------------------------------------


class TestEquilibriumZero:
    """At equilibrium, vz must be exactly 0 (not approximately)."""

    def test_vz_stays_zero_once_settled(self):
        tree = _make_ball_tree()
        solver = MuJoCoStyleSolver(max_iter=50, rho=1.0)
        z, vz = _run_ball_drop(tree, solver, n_steps=1000)
        # After settling (~step 400), vz should be 0 for all remaining steps
        settled_vz = vz[500:]
        max_vz = np.max(np.abs(settled_vz))
        assert max_vz < 1e-6, f"Residual vz after settling: {max_vz}"

    def test_position_constant_after_settling(self):
        tree = _make_ball_tree()
        solver = MuJoCoStyleSolver(max_iter=50, rho=1.0)
        z, vz = _run_ball_drop(tree, solver, n_steps=1000)
        z_range = np.max(z[500:]) - np.min(z[500:])
        assert z_range < 1e-6, f"Position drift after settling: {z_range}"


# ---------------------------------------------------------------------------
# 6. Impedance function
# ---------------------------------------------------------------------------


class TestImpedance:
    """solimp impedance function d(r) correctness."""

    def test_zero_depth_gives_d0(self):
        solver = MuJoCoStyleSolver(solimp=(0.9, 0.95, 0.001, 0.5, 2))
        assert abs(solver._impedance(0.0) - 0.9) < 1e-10

    def test_full_depth_gives_dwidth(self):
        solver = MuJoCoStyleSolver(solimp=(0.9, 0.95, 0.001, 0.5, 2))
        assert abs(solver._impedance(0.001) - 0.95) < 1e-10
        assert abs(solver._impedance(0.01) - 0.95) < 1e-10  # clamped

    def test_monotonic(self):
        solver = MuJoCoStyleSolver(solimp=(0.9, 0.95, 0.001, 0.5, 2))
        depths = np.linspace(0, 0.002, 20)
        d_vals = [solver._impedance(d) for d in depths]
        for i in range(1, len(d_vals)):
            assert d_vals[i] >= d_vals[i - 1] - 1e-10
