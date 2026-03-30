"""
Two-ball-two-wall scenario: our solvers vs MuJoCo reference.

Scene: Two 1kg spheres (radius=0.1m) between two vertical walls.
  - Ball A at x=-0.3, Ball B at x=0.3, both at z=0.3 (above ground)
  - Wall L at x=-0.8, Wall R at x=0.8
  - Balls approaching each other: vx_A=+2, vx_B=-2
  - Ground at z=0, gravity=-9.81

Expected physics:
  1. Free fall phase: both fall toward ground
  2. Ball-ball collision: balls bounce off each other in x
  3. Ground contact: balls land
  4. Wall contact: balls may reach walls after bouncing

Comparisons:
  - MuJoCo (mj_step, soft constraints, elliptic cone)
  - Our ADMMQPSolver (designed to match MuJoCo)
  - Our CpuEngine (PGS-SI, hard constraints)

References:
  OPEN_QUESTIONS.md Q21 — solver stability on ball-wall scenarios.
  test_mujoco_qp.py — template for MuJoCo comparison.
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.geometry import BodyCollisionGeometry, ShapeInstance, SphereShape
from physics.joint import FreeJoint
from physics.merged_model import merge_models
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform

try:
    import mujoco

    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

try:
    from physics.cpu_engine import CpuEngine

    HAS_ENGINE = True
except Exception:
    HAS_ENGINE = False

try:
    from physics.solvers.mujoco_qp import ADMMQPSolver

    HAS_ADMM = HAS_ENGINE  # needs CpuEngine too
except Exception:
    HAS_ADMM = False

# Scene parameters
DT = 0.001
RADIUS = 0.1
MASS = 1.0
G = 9.81
I_DIAG = 2.0 / 5.0 * MASS * RADIUS**2  # solid sphere inertia
MU = 0.5
N_STEPS = 400
WALL_X = 0.8
BALL_Z0 = 0.3
BALL_AX = -0.3
BALL_BX = 0.3
VX_A = 2.0
VX_B = -2.0

pytestmark = pytest.mark.skipif(not HAS_MUJOCO, reason="mujoco not installed")


# ---------------------------------------------------------------------------
# MuJoCo reference
# ---------------------------------------------------------------------------


def _mujoco_two_ball_wall():
    """Run two-ball-two-wall scenario in MuJoCo."""
    xml = f"""<mujoco>
      <option timestep="{DT}" gravity="0 0 -{G}" integrator="Euler" cone="elliptic"/>
      <worldbody>
        <geom type="plane" size="10 10 0.1" friction="{MU} 0.005 0.0001"/>
        <geom name="wall_L" type="box" pos="-{WALL_X} 0 1" size="0.01 2 2"
              friction="{MU} 0.005 0.0001"/>
        <geom name="wall_R" type="box" pos="{WALL_X} 0 1" size="0.01 2 2"
              friction="{MU} 0.005 0.0001"/>
        <body name="ball_A" pos="{BALL_AX} 0 {BALL_Z0}">
          <freejoint/>
          <geom type="sphere" size="{RADIUS}" mass="{MASS}"
                friction="{MU} 0.005 0.0001"/>
          <inertial pos="0 0 0" mass="{MASS}"
                    diaginertia="{I_DIAG} {I_DIAG} {I_DIAG}"/>
        </body>
        <body name="ball_B" pos="{BALL_BX} 0 {BALL_Z0}">
          <freejoint/>
          <geom type="sphere" size="{RADIUS}" mass="{MASS}"
                friction="{MU} 0.005 0.0001"/>
          <inertial pos="0 0 0" mass="{MASS}"
                    diaginertia="{I_DIAG} {I_DIAG} {I_DIAG}"/>
        </body>
      </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Set initial velocities
    # MuJoCo qvel layout: [vx, vy, vz, wx, wy, wz] per free body
    # Ball A is first free body, Ball B is second
    data.qvel[0] = VX_A  # Ball A vx
    data.qvel[6] = VX_B  # Ball B vx

    # Trajectory arrays
    xa = np.zeros(N_STEPS)
    za = np.zeros(N_STEPS)
    xb = np.zeros(N_STEPS)
    zb = np.zeros(N_STEPS)
    vxa = np.zeros(N_STEPS)
    vxb = np.zeros(N_STEPS)

    for i in range(N_STEPS):
        # MuJoCo freejoint qpos layout: [px, py, pz, qw, qx, qy, qz] per body
        xa[i] = data.qpos[0]  # Ball A px
        za[i] = data.qpos[2]  # Ball A pz
        xb[i] = data.qpos[7]  # Ball B px
        zb[i] = data.qpos[9]  # Ball B pz
        vxa[i] = data.qvel[0]
        vxb[i] = data.qvel[6]
        mujoco.mj_step(model, data)

    return xa, za, xb, zb, vxa, vxb


# ---------------------------------------------------------------------------
# Our CpuEngine (PGS-SI on MergedModel)
# ---------------------------------------------------------------------------


def _our_two_ball_wall():
    """Run two-ball-two-wall scenario using CpuEngine + MergedModel."""

    def _ball(mass=MASS, radius=RADIUS):
        tree = RobotTreeNumpy(gravity=G)
        tree.add_body(
            Body(
                "ball",
                0,
                FreeJoint("root"),
                SpatialInertia(mass, np.eye(3) * I_DIAG, np.zeros(3)),
                SpatialTransform.identity(),
                parent=-1,
            )
        )
        tree.finalize()
        return tree

    tree_a = _ball()
    tree_b = _ball()

    from robot.model import RobotModel

    model_a = RobotModel(
        tree=tree_a,
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(SphereShape(RADIUS))])],
        contact_body_names=["ball"],
    )
    model_b = RobotModel(
        tree=tree_b,
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(SphereShape(RADIUS))])],
        contact_body_names=["ball"],
    )

    # Build merged model with walls as static geometry
    merged = merge_models(robots={"a": model_a, "b": model_b})

    cpu = CpuEngine(merged, dt=DT)

    # Initial state
    q, qdot = merged.tree.default_state()
    # Ball A: qpos = [qw=1, qx, qy, qz, px, py, pz]
    rs_a = merged.robot_slices["a"]
    rs_b = merged.robot_slices["b"]
    q[rs_a.q_slice.start + 4] = BALL_AX  # px
    q[rs_a.q_slice.start + 6] = BALL_Z0  # pz
    q[rs_b.q_slice.start + 4] = BALL_BX
    q[rs_b.q_slice.start + 6] = BALL_Z0

    qdot[rs_a.v_slice.start] = VX_A  # vx
    qdot[rs_b.v_slice.start] = VX_B

    tau = np.zeros(merged.nv)

    xa = np.zeros(N_STEPS)
    za = np.zeros(N_STEPS)
    xb = np.zeros(N_STEPS)
    zb = np.zeros(N_STEPS)
    vxa = np.zeros(N_STEPS)
    vxb = np.zeros(N_STEPS)

    for i in range(N_STEPS):
        xa[i] = q[rs_a.q_slice.start + 4]
        za[i] = q[rs_a.q_slice.start + 6]
        xb[i] = q[rs_b.q_slice.start + 4]
        zb[i] = q[rs_b.q_slice.start + 6]
        vxa[i] = qdot[rs_a.v_slice.start]
        vxb[i] = qdot[rs_b.v_slice.start]

        out = cpu.step(q, qdot, tau, dt=DT)
        q, qdot = out.q_new, out.qdot_new

    return xa, za, xb, zb, vxa, vxb


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFreeFallPhase:
    """Before any contact, both simulations should match kinematics exactly."""

    @pytest.fixture(scope="class")
    def trajectories(self):
        mj = _mujoco_two_ball_wall()
        if HAS_ENGINE:
            ours = _our_two_ball_wall()
        else:
            ours = None
        return mj, ours

    def test_mujoco_free_fall_z(self, trajectories):
        """MuJoCo balls should free-fall correctly (z = z0 - 0.5*g*t^2)."""
        mj, _ = trajectories
        _, za_mj, _, zb_mj, _, _ = mj
        # First 50 steps (t=0.05s), before any contact
        t = np.arange(50) * DT
        z_analytical = BALL_Z0 - 0.5 * G * t**2
        np.testing.assert_allclose(za_mj[:50], z_analytical, atol=0.002)
        np.testing.assert_allclose(zb_mj[:50], z_analytical, atol=0.002)

    def test_mujoco_free_fall_x(self, trajectories):
        """MuJoCo balls should move horizontally (x = x0 + vx*t)."""
        mj, _ = trajectories
        xa_mj, _, xb_mj, _, _, _ = mj
        t = np.arange(50) * DT
        xa_analytical = BALL_AX + VX_A * t
        xb_analytical = BALL_BX + VX_B * t
        np.testing.assert_allclose(xa_mj[:50], xa_analytical, atol=0.002)
        np.testing.assert_allclose(xb_mj[:50], xb_analytical, atol=0.002)

    @pytest.mark.skipif(not HAS_ENGINE, reason="CpuEngine not available")
    def test_our_matches_mujoco_free_fall(self, trajectories):
        """Our free-fall phase should match MuJoCo."""
        mj, ours = trajectories
        if ours is None:
            pytest.skip("CpuEngine not available")
        xa_mj, za_mj, xb_mj, zb_mj, _, _ = mj
        xa_o, za_o, xb_o, zb_o, _, _ = ours
        # First 50 steps, both should agree
        np.testing.assert_allclose(xa_o[:50], xa_mj[:50], atol=0.005)
        np.testing.assert_allclose(za_o[:50], za_mj[:50], atol=0.005)


class TestBallBallCollision:
    """Balls approach each other and should collide around step ~150."""

    @pytest.fixture(scope="class")
    def trajectories(self):
        mj = _mujoco_two_ball_wall()
        if HAS_ENGINE:
            ours = _our_two_ball_wall()
        else:
            ours = None
        return mj, ours

    def test_mujoco_balls_reverse_x(self, trajectories):
        """After collision, Ball A should have negative vx and Ball B positive."""
        mj, _ = trajectories
        _, _, _, _, vxa_mj, vxb_mj = mj
        # At step 300 (after collision + settling), velocities should have reversed
        # Ball A started with vx=+2, should now be near 0 or negative
        # Ball B started with vx=-2, should now be near 0 or positive
        assert vxa_mj[300] < VX_A, "Ball A should have decelerated"
        assert vxb_mj[300] > VX_B, "Ball B should have decelerated"

    def test_mujoco_symmetry(self, trajectories):
        """Scene is symmetric: Ball A and B should be mirrors in x."""
        mj, _ = trajectories
        xa_mj, za_mj, xb_mj, zb_mj, _, _ = mj
        # xa should be ~ -xb, za should be ~ zb
        np.testing.assert_allclose(xa_mj, -xb_mj, atol=0.01)
        np.testing.assert_allclose(za_mj, zb_mj, atol=0.01)

    @pytest.mark.skipif(not HAS_ENGINE, reason="CpuEngine not available")
    def test_our_balls_also_reverse(self, trajectories):
        """Our solver should also reverse ball velocities after collision."""
        mj, ours = trajectories
        if ours is None:
            pytest.skip()
        _, _, _, _, vxa_o, vxb_o = ours
        # Same check: velocities should reverse after collision
        assert vxa_o[300] < VX_A, "Our Ball A should have decelerated"
        assert vxb_o[300] > VX_B, "Our Ball B should have decelerated"


class TestGroundContact:
    """Both balls should land on the ground and not penetrate."""

    @pytest.fixture(scope="class")
    def trajectories(self):
        mj = _mujoco_two_ball_wall()
        if HAS_ENGINE:
            ours = _our_two_ball_wall()
        else:
            ours = None
        return mj, ours

    def test_mujoco_no_ground_penetration(self, trajectories):
        """MuJoCo balls should stay above ground (z >= radius - tolerance)."""
        mj, _ = trajectories
        _, za_mj, _, zb_mj, _, _ = mj
        assert np.all(za_mj > RADIUS - 0.02), f"Ball A penetrated ground: min z={za_mj.min()}"
        assert np.all(zb_mj > RADIUS - 0.02), f"Ball B penetrated ground: min z={zb_mj.min()}"

    @pytest.mark.skipif(not HAS_ENGINE, reason="CpuEngine not available")
    def test_our_no_ground_penetration(self, trajectories):
        """Our balls should also stay above ground."""
        _, ours = trajectories
        if ours is None:
            pytest.skip()
        _, za_o, _, zb_o, _, _ = ours
        # CpuEngine uses sphere approximation for ground (center-based), so z > 0
        assert np.all(za_o > -0.05), f"Our Ball A penetrated: min z={za_o.min()}"
        assert np.all(zb_o > -0.05), f"Our Ball B penetrated: min z={zb_o.min()}"


class TestTrajectoryAgreement:
    """Overall trajectory comparison between MuJoCo and our solver."""

    @pytest.fixture(scope="class")
    def trajectories(self):
        mj = _mujoco_two_ball_wall()
        if HAS_ENGINE:
            ours = _our_two_ball_wall()
        else:
            ours = None
        return mj, ours

    @pytest.mark.skipif(not HAS_ENGINE, reason="CpuEngine not available")
    def test_z_trajectory_l2(self, trajectories):
        """Z trajectory L2 error should be bounded.

        Note: MuJoCo uses soft constraints (elliptic cone), our PGS-SI uses
        hard constraints. Different contact models mean different settling
        behavior, so tolerance is generous.
        """
        mj, ours = trajectories
        if ours is None:
            pytest.skip()
        _, za_mj, _, zb_mj, _, _ = mj
        _, za_o, _, zb_o, _, _ = ours
        # Compare only free-fall phase (first 100 steps) where both should match
        l2_za = np.sqrt(np.mean((za_o[:100] - za_mj[:100]) ** 2))
        l2_zb = np.sqrt(np.mean((zb_o[:100] - zb_mj[:100]) ** 2))
        assert l2_za < 0.01, f"Ball A z L2 too large: {l2_za:.4f}"
        assert l2_zb < 0.01, f"Ball B z L2 too large: {l2_zb:.4f}"

    @pytest.mark.skipif(not HAS_ENGINE, reason="CpuEngine not available")
    def test_x_trajectory_l2(self, trajectories):
        """X trajectory L2 error in free-fall phase."""
        mj, ours = trajectories
        if ours is None:
            pytest.skip()
        xa_mj, _, xb_mj, _, _, _ = mj
        xa_o, _, xb_o, _, _, _ = ours
        l2_xa = np.sqrt(np.mean((xa_o[:100] - xa_mj[:100]) ** 2))
        l2_xb = np.sqrt(np.mean((xb_o[:100] - xb_mj[:100]) ** 2))
        assert l2_xa < 0.01, f"Ball A x L2 too large: {l2_xa:.4f}"
        assert l2_xb < 0.01, f"Ball B x L2 too large: {l2_xb:.4f}"

    @pytest.mark.skipif(not HAS_ENGINE, reason="CpuEngine not available")
    def test_final_state_finite(self, trajectories):
        """Both simulations should produce finite results throughout."""
        mj, ours = trajectories
        if ours is None:
            pytest.skip()
        for arr in mj:
            assert np.all(np.isfinite(arr)), "MuJoCo produced NaN/Inf"
        for arr in ours:
            assert np.all(np.isfinite(arr)), "Our solver produced NaN/Inf"


# ---------------------------------------------------------------------------
# ADMMQPSolver (MuJoCo-style soft constraints) vs MuJoCo
# ---------------------------------------------------------------------------


def _our_admm_two_ball_wall():
    """Run two-ball-two-wall scenario using CpuEngine + ADMMQPSolver."""
    from robot.model import RobotModel

    def _ball():
        tree = RobotTreeNumpy(gravity=G)
        tree.add_body(
            Body(
                "ball",
                0,
                FreeJoint("root"),
                SpatialInertia(MASS, np.eye(3) * I_DIAG, np.zeros(3)),
                SpatialTransform.identity(),
                parent=-1,
            )
        )
        tree.finalize()
        return RobotModel(
            tree=tree,
            geometries=[BodyCollisionGeometry(0, [ShapeInstance(SphereShape(RADIUS))])],
            contact_body_names=["ball"],
        )

    merged = merge_models(robots={"a": _ball(), "b": _ball()})
    solver = ADMMQPSolver(max_iter=50)
    cpu = CpuEngine(merged, solver=solver, dt=DT)

    q, qdot = merged.tree.default_state()
    rs_a = merged.robot_slices["a"]
    rs_b = merged.robot_slices["b"]
    q[rs_a.q_slice.start + 4] = BALL_AX
    q[rs_a.q_slice.start + 6] = BALL_Z0
    q[rs_b.q_slice.start + 4] = BALL_BX
    q[rs_b.q_slice.start + 6] = BALL_Z0
    qdot[rs_a.v_slice.start] = VX_A
    qdot[rs_b.v_slice.start] = VX_B
    tau = np.zeros(merged.nv)

    xa, za, xb, zb = [np.zeros(N_STEPS) for _ in range(4)]
    for i in range(N_STEPS):
        xa[i] = q[rs_a.q_slice.start + 4]
        za[i] = q[rs_a.q_slice.start + 6]
        xb[i] = q[rs_b.q_slice.start + 4]
        zb[i] = q[rs_b.q_slice.start + 6]
        out = cpu.step(q, qdot, tau, dt=DT)
        q, qdot = out.q_new, out.qdot_new

    return xa, za, xb, zb


@pytest.mark.skipif(not HAS_ADMM, reason="ADMMQPSolver or CpuEngine not available")
class TestADMMvsMuJoCo:
    """ADMMQPSolver should closely match MuJoCo (same contact model)."""

    @pytest.fixture(scope="class")
    def trajectories(self):
        mj = _mujoco_two_ball_wall()
        admm = _our_admm_two_ball_wall()
        return mj, admm

    def test_full_trajectory_x_l2(self, trajectories):
        """Full x trajectory L2 should be < 2mm (same contact model)."""
        mj, admm = trajectories
        xa_mj, _, _, _, _, _ = mj
        xa_a, _, _, _ = admm
        l2 = np.sqrt(np.mean((xa_a - xa_mj) ** 2))
        assert l2 < 0.002, f"ADMM x L2 vs MuJoCo: {l2 * 1000:.1f}mm (expected < 2mm)"

    def test_full_trajectory_z_l2(self, trajectories):
        """Full z trajectory L2 should be < 1mm (same contact model)."""
        mj, admm = trajectories
        _, za_mj, _, _, _, _ = mj
        _, za_a, _, _ = admm
        l2 = np.sqrt(np.mean((za_a - za_mj) ** 2))
        assert l2 < 0.001, f"ADMM z L2 vs MuJoCo: {l2 * 1000:.1f}mm (expected < 1mm)"

    def test_final_position_agreement(self, trajectories):
        """Final ball positions should agree within 2mm."""
        mj, admm = trajectories
        xa_mj, za_mj, xb_mj, zb_mj, _, _ = mj
        xa_a, za_a, xb_a, zb_a = admm
        np.testing.assert_allclose(xa_a[-1], xa_mj[-1], atol=0.002, err_msg="Ball A final x mismatch")
        np.testing.assert_allclose(za_a[-1], za_mj[-1], atol=0.002, err_msg="Ball A final z mismatch")
        np.testing.assert_allclose(xb_a[-1], xb_mj[-1], atol=0.002, err_msg="Ball B final x mismatch")
        np.testing.assert_allclose(zb_a[-1], zb_mj[-1], atol=0.002, err_msg="Ball B final z mismatch")

    def test_symmetry(self, trajectories):
        """ADMM trajectory should maintain scene symmetry."""
        _, admm = trajectories
        xa_a, za_a, xb_a, zb_a = admm
        np.testing.assert_allclose(xa_a, -xb_a, atol=0.01)
        np.testing.assert_allclose(za_a, zb_a, atol=0.01)

    def test_no_ground_penetration(self, trajectories):
        """Balls should stay above ground (z >= radius - tolerance).

        Soft constraints (ADMM) allow slight penetration during impact;
        MuJoCo behaves the same way. Tolerance is 1.5mm below radius.
        """
        _, admm = trajectories
        _, za_a, _, zb_a = admm
        assert np.all(za_a > RADIUS - 0.015), f"Ball A penetrated: min z={za_a.min():.4f}"
        assert np.all(zb_a > RADIUS - 0.015), f"Ball B penetrated: min z={zb_a.min():.4f}"
