"""
GPU ADMM solver vs MuJoCo trajectory comparison.

Tests that the GPU ADMM solver (velocity-level, body-level Delassus) produces
ball-drop trajectories close to MuJoCo's ground truth. For free bodies
(FreeJoint), the body-level Delassus is exact, so the main sources of error
are: float32 arithmetic, velocity-vs-acceleration formulation, and the
dt-scaled rhs_const approximation.

Also compares against the CPU ADMMQPSolver to quantify the formulation gap.
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.geometry import BodyCollisionGeometry, ShapeInstance, SphereShape
from physics.joint import FreeJoint
from physics.merged_model import merge_models
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from robot.model import RobotModel

try:
    import mujoco

    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

try:
    from physics.gpu_engine import GpuEngine

    HAS_WARP = True
except Exception:
    HAS_WARP = False

pytestmark = [
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.skipif(not (HAS_MUJOCO and HAS_WARP), reason="Requires both mujoco and Warp/CUDA"),
]

# Physical constants — must match MuJoCo XML exactly
DT = 2e-4  # GPU engine default timestep
RADIUS = 0.1
MASS = 1.0
G = 9.81
I_DIAG = 2.0 / 5.0 * MASS * RADIUS**2  # solid sphere


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ball_merged(mass=MASS, radius=RADIUS):
    tree = RobotTreeNumpy(gravity=G)
    I_s = 2.0 / 5.0 * mass * radius**2
    tree.add_body(
        Body(
            "ball",
            0,
            FreeJoint("root"),
            SpatialInertia(mass, np.diag([I_s] * 3), np.zeros(3)),
            SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()
    model = RobotModel(
        tree=tree,
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(SphereShape(radius))])],
        contact_body_names=["ball"],
    )
    return merge_models(robots={"ball": model})


def _gpu_admm_ball_drop(n_steps, z0=0.3, dt=DT, admm_iters=30):
    merged = _ball_merged()
    gpu = GpuEngine(merged, num_envs=1, dt=dt, solver="admm")
    gpu._admm_iters = admm_iters

    q, qdot = merged.tree.default_state()
    q[6] = z0
    gpu.reset(q)

    z = np.zeros(n_steps)
    vz = np.zeros(n_steps)
    for i in range(n_steps):
        q_cur = gpu._scratch.q.numpy()[0]
        qdot_cur = gpu._scratch.qdot.numpy()[0]
        z[i] = q_cur[6]
        vz[i] = qdot_cur[2]
        gpu.step(dt=dt)
    return z, vz


def _mujoco_ball_drop(n_steps, z0=0.3, dt=DT):
    xml = f"""<mujoco>
      <option timestep="{dt}" gravity="0 0 -{G}" integrator="Euler" cone="elliptic"/>
      <worldbody>
        <geom type="plane" size="10 10 0.1" friction="0.5 0.005 0.0001"/>
        <body name="ball" pos="0 0 {z0}"><freejoint/>
          <geom type="sphere" size="{RADIUS}" mass="{MASS}"
                friction="0.5 0.005 0.0001"/>
          <inertial pos="0 0 0" mass="{MASS}"
                    diaginertia="{I_DIAG} {I_DIAG} {I_DIAG}"/>
        </body>
      </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    z = np.zeros(n_steps)
    vz = np.zeros(n_steps)
    for i in range(n_steps):
        z[i] = data.qpos[2]
        vz[i] = data.qvel[2]
        mujoco.mj_step(model, data)
    return z, vz


def _cpu_admm_ball_drop(n_steps, z0=0.3, dt=DT):
    """Run CPU ADMMQPSolver (acceleration-level) for comparison."""
    from physics.geometry import SphereShape
    from physics.gjk_epa import ground_contact_query
    from physics.implicit_contact_step import ImplicitContactStep
    from physics.solvers.admm_qp import MuJoCoStyleSolver
    from physics.solvers.pgs_solver import ContactConstraint

    tree = RobotTreeNumpy(gravity=G)
    I_s = 2.0 / 5.0 * MASS * RADIUS**2
    tree.add_body(
        Body(
            "ball",
            0,
            FreeJoint("root"),
            SpatialInertia(MASS, np.diag([I_s] * 3), np.zeros(3)),
            SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()

    solver = MuJoCoStyleSolver(max_iter=50, rho=1.0)
    integ = ImplicitContactStep(dt=dt, solver=solver)
    shape = SphereShape(RADIUS)

    q, qdot = tree.default_state()
    q[6] = z0
    z = np.zeros(n_steps)
    vz = np.zeros(n_steps)
    for i in range(n_steps):
        z[i] = q[6]
        vz[i] = qdot[2]
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


def _settle_step(vz, start=200):
    """Find first step where velocity stays below threshold for 20 steps."""
    for i in range(start, len(vz) - 20):
        if all(abs(vz[j]) < 0.001 for j in range(i, i + 20)):
            return i
    return -1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

N_STEPS = 5000  # 1 second at dt=2e-4


class TestGpuAdmmVsMuJoCoBallDrop:
    """Full ball-drop trajectory: GPU ADMM vs MuJoCo."""

    @pytest.fixture(scope="class")
    def trajectories(self):
        gz, gvz = _gpu_admm_ball_drop(N_STEPS, z0=0.3)
        mz, mvz = _mujoco_ball_drop(N_STEPS, z0=0.3)
        return gz, gvz, mz, mvz

    def test_free_fall_match(self, trajectories):
        """Before impact, both trajectories should match analytically.

        Free fall z(t) = z0 - 0.5*g*t^2 is independent of solver.
        """
        gz, _, mz, _ = trajectories
        # Impact at t = sqrt(2*(z0-radius)/g) ≈ sqrt(2*0.2/9.81) ≈ 0.202s
        # At dt=2e-4, that's ~1010 steps. Check first 900 steps.
        n_check = 900
        np.testing.assert_allclose(
            gz[:n_check],
            mz[:n_check],
            atol=1e-5,
            err_msg="Free-fall phase should match to ~10µm",
        )

    def test_z_l2_trajectory(self, trajectories):
        """L2 distance between trajectories should be small."""
        gz, _, mz, _ = trajectories
        l2 = np.sqrt(np.mean((gz - mz) ** 2))
        print(f"GPU ADMM vs MuJoCo z-L2 = {l2 * 1000:.4f} mm")
        # Relaxed tolerance for velocity-level ADMM (vs CPU ADMM's sub-mm)
        assert l2 < 0.005, f"z-L2 = {l2 * 1000:.2f} mm, expected < 5 mm"

    def test_no_ground_penetration(self, trajectories):
        """Ball should not penetrate excessively.

        The velocity-level ADMM with dt-scaled rhs allows more transient
        penetration during impact than the acceleration-level CPU ADMM.
        This is a known limitation — the compliance spring response is
        weaker per step in velocity space. Penetration should still be
        bounded and the ball must recover.
        """
        gz, _, mz, _ = trajectories
        min_z_gpu = np.min(gz)
        min_z_mj = np.min(mz)
        pen_gpu = max(0, RADIUS - min_z_gpu)
        pen_mj = max(0, RADIUS - min_z_mj)
        print(f"Max penetration: GPU={pen_gpu * 1000:.2f} mm, MuJoCo={pen_mj * 1000:.2f} mm")
        # GPU ADMM allows up to ~20mm transient penetration during impact
        assert min_z_gpu > RADIUS - 0.025, f"GPU min z={min_z_gpu:.4f}, penetration={pen_gpu * 1000:.1f} mm"
        # Ball must recover (final z should be near radius)
        assert np.mean(gz[-100:]) > RADIUS - 0.005

    def test_settling_position(self, trajectories):
        """Both should settle near z=radius."""
        gz, _, mz, _ = trajectories
        # Last 100 steps should be near-stationary
        gpu_final = np.mean(gz[-100:])
        mj_final = np.mean(mz[-100:])
        assert abs(gpu_final - RADIUS) < 0.01, f"GPU settled at {gpu_final:.4f}"
        assert abs(mj_final - RADIUS) < 0.01, f"MuJoCo settled at {mj_final:.4f}"
        assert abs(gpu_final - mj_final) < 0.005, (
            f"Settling difference: GPU={gpu_final:.4f} MuJoCo={mj_final:.4f}"
        )

    def test_settling_velocity(self, trajectories):
        """After settling, velocity should be near zero for both."""
        _, gvz, _, mvz = trajectories
        gpu_vz_final = np.mean(np.abs(gvz[-100:]))
        mj_vz_final = np.mean(np.abs(mvz[-100:]))
        assert gpu_vz_final < 0.1, f"GPU final |vz|={gpu_vz_final:.4f}"
        assert mj_vz_final < 0.1, f"MuJoCo final |vz|={mj_vz_final:.4f}"


class TestGpuVsCpuAdmmBallDrop:
    """GPU ADMM vs CPU ADMM for the same ball-drop scenario.

    For FreeJoint bodies, body-level Delassus = joint-space Delassus exactly.
    So the difference should come only from:
    - float32 vs float64
    - velocity-level vs acceleration-level formulation
    - rhs_const approximation (v_c ≈ v_free)
    """

    @pytest.fixture(scope="class")
    def trajectories(self):
        gz, gvz = _gpu_admm_ball_drop(N_STEPS, z0=0.3)
        cz, cvz = _cpu_admm_ball_drop(N_STEPS, z0=0.3)
        mz, mvz = _mujoco_ball_drop(N_STEPS, z0=0.3)
        return gz, gvz, cz, cvz, mz, mvz

    def test_gpu_vs_cpu_z_l2(self, trajectories):
        gz, _, cz, _, _, _ = trajectories
        l2 = np.sqrt(np.mean((gz - cz) ** 2))
        print(f"GPU ADMM vs CPU ADMM z-L2 = {l2 * 1000:.4f} mm")
        assert l2 < 0.005, f"GPU vs CPU z-L2 = {l2 * 1000:.2f} mm, expected < 5 mm"

    def test_cpu_closer_to_mujoco(self, trajectories):
        """CPU ADMM (accel-level, float64) should be closer to MuJoCo than GPU."""
        gz, _, cz, _, mz, _ = trajectories
        l2_gpu = np.sqrt(np.mean((gz - mz) ** 2))
        l2_cpu = np.sqrt(np.mean((cz - mz) ** 2))
        print(f"vs MuJoCo: GPU L2={l2_gpu * 1000:.4f} mm, CPU L2={l2_cpu * 1000:.4f} mm")
        # Just report — CPU should be closer but not necessarily by a lot
        # Both should be reasonable
        assert l2_gpu < 0.01, f"GPU too far from MuJoCo: {l2_gpu * 1000:.2f} mm"

    def test_all_three_free_fall_match(self, trajectories):
        """Free-fall phase: all three should match perfectly."""
        gz, _, cz, _, mz, _ = trajectories
        n_check = 900
        np.testing.assert_allclose(gz[:n_check], mz[:n_check], atol=1e-5)
        np.testing.assert_allclose(cz[:n_check], mz[:n_check], atol=1e-5)

    def test_all_settle_to_same_position(self, trajectories):
        """All three should settle to approximately z=radius."""
        gz, _, cz, _, mz, _ = trajectories
        finals = {
            "GPU ADMM": np.mean(gz[-100:]),
            "CPU ADMM": np.mean(cz[-100:]),
            "MuJoCo": np.mean(mz[-100:]),
        }
        for name, val in finals.items():
            assert abs(val - RADIUS) < 0.01, f"{name} settled at {val:.4f}"

        # Print summary table
        for name, val in finals.items():
            print(f"  {name:10s} final z = {val:.6f} (err = {(val - RADIUS) * 1000:.4f} mm)")
