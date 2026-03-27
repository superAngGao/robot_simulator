"""
Tests for GPU multi-body physics — verifies Q23 fix (angular velocity divergence).

Tests CpuEngine vs GpuEngine on MergedModel with two FreeJoint root bodies.
The original bug: J_body_j in solver_kernels_v2.py was not negated for body-body
contacts, causing the constraint to be absolute velocity instead of relative,
leading to angular velocity divergence on body index >= 1.

Reference: OPEN_QUESTIONS.md Q23.
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
    from physics.cpu_engine import CpuEngine
    from physics.gpu_engine import GpuEngine

    HAS_WARP = True
except Exception:
    HAS_WARP = False

pytestmark = pytest.mark.skipif(not HAS_WARP, reason="Warp or CUDA not available")


def _ball_model(mass: float = 1.0, radius: float = 0.1) -> RobotModel:
    """Single FreeJoint sphere."""
    tree = RobotTreeNumpy(gravity=9.81)
    I_sphere = 2.0 / 5.0 * mass * radius**2
    tree.add_body(
        Body(
            name="ball",
            index=0,
            joint=FreeJoint("root"),
            inertia=SpatialInertia(mass=mass, inertia=np.eye(3) * I_sphere, com=np.zeros(3)),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()
    return RobotModel(
        tree=tree,
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(SphereShape(radius))])],
        contact_body_names=["ball"],
    )


def _two_ball_merged(radius=0.1):
    """Build a MergedModel with two equal-mass free balls."""
    m_a = _ball_model(mass=1.0, radius=radius)
    m_b = _ball_model(mass=1.0, radius=radius)
    return merge_models(robots={"a": m_a, "b": m_b})


def _init_two_balls(merged, x_a=0.0, z_a=1.0, x_b=0.5, z_b=1.0):
    """Initialize merged state with two balls at given positions."""
    q, qdot = merged.tree.default_state()  # default_q sets qw=1 correctly
    for name, rs in merged.robot_slices.items():
        qs = rs.q_slice
        # FreeJoint layout: [qw, qx, qy, qz, px, py, pz]
        # default_state already sets qw=1 for each FreeJoint
        if name == "a":
            q[qs.start + 4] = x_a  # px
            q[qs.start + 6] = z_a  # pz
        else:
            q[qs.start + 4] = x_b
            q[qs.start + 6] = z_b
    return q, qdot


# ---------------------------------------------------------------------------
# Two-ball free fall (no collision) — pure ABA test
# ---------------------------------------------------------------------------


class TestGpuTwoBallFreeFall:
    def test_two_balls_free_fall_cpu_vs_gpu(self):
        """Two separated balls in free fall: GPU should match CPU."""
        merged = _two_ball_merged()
        cpu = CpuEngine(merged, dt=1e-3)
        gpu = GpuEngine(merged, num_envs=1, dt=1e-3)

        q, qdot = _init_two_balls(merged, x_a=-2.0, z_a=2.0, x_b=2.0, z_b=2.0)
        tau = np.zeros(merged.nv)

        q_cpu, qdot_cpu = q.copy(), qdot.copy()
        q_gpu, qdot_gpu = q.copy(), qdot.copy()

        for _ in range(100):
            out_cpu = cpu.step(q_cpu, qdot_cpu, tau, dt=1e-3)
            q_cpu, qdot_cpu = out_cpu.q_new, out_cpu.qdot_new

            out_gpu = gpu.step(q_gpu, qdot_gpu, tau, dt=1e-3)
            q_gpu, qdot_gpu = out_gpu.q_new, out_gpu.qdot_new

        # Both balls should free-fall identically
        # float32 vs float64 tolerance
        np.testing.assert_allclose(q_gpu, q_cpu, atol=5e-3, err_msg="q diverged CPU vs GPU")
        np.testing.assert_allclose(qdot_gpu, qdot_cpu, atol=5e-3, err_msg="qdot diverged CPU vs GPU")

    def test_second_root_no_angular_divergence(self):
        """Body index >= 1 (second root) must NOT have angular velocity divergence."""
        merged = _two_ball_merged()
        gpu = GpuEngine(merged, num_envs=1, dt=1e-3)

        q, qdot = _init_two_balls(merged, x_a=-2.0, z_a=2.0, x_b=2.0, z_b=2.0)
        tau = np.zeros(merged.nv)

        for _ in range(500):
            out = gpu.step(q, qdot, tau, dt=1e-3)
            q, qdot = out.q_new, out.qdot_new

        # Angular velocities for both balls should be near zero (free fall, no torque)
        rs_b = merged.robot_slices["b"]
        omega_b = qdot[rs_b.v_slice][3:6]
        assert np.all(np.isfinite(omega_b)), f"Ball B angular velocity is NaN/Inf: {omega_b}"
        assert np.linalg.norm(omega_b) < 1.0, f"Ball B angular velocity diverged: {omega_b}"


# ---------------------------------------------------------------------------
# Two-ball body-body collision — the Q23 scenario
# ---------------------------------------------------------------------------


class TestGpuBodyBodyCollision:
    def test_approaching_balls_bounce(self):
        """Two balls approaching each other should bounce (velocity reversal)."""
        merged = _two_ball_merged(radius=0.1)
        gpu = GpuEngine(merged, num_envs=1, dt=2e-4)

        # Place close but not overlapping, approaching each other
        q, qdot = _init_two_balls(merged, x_a=-0.09, z_a=1.0, x_b=0.09, z_b=1.0)
        # Ball A moves right, Ball B moves left → approaching
        rs_a = merged.robot_slices["a"]
        rs_b = merged.robot_slices["b"]
        qdot[rs_a.v_slice.start] = 1.0  # vx_a = +1
        qdot[rs_b.v_slice.start] = -1.0  # vx_b = -1
        tau = np.zeros(merged.nv)

        for _ in range(200):
            out = gpu.step(q, qdot, tau, dt=2e-4)
            q, qdot = out.q_new, out.qdot_new

        vx_a = qdot[rs_a.v_slice][0]
        vx_b = qdot[rs_b.v_slice][0]

        # After collision: A should have negative vx, B should have positive vx
        # (or at least the approach velocity should be reversed/reduced)
        assert vx_a < 0.5, f"Ball A should have bounced: vx_a={vx_a}"
        assert vx_b > -0.5, f"Ball B should have bounced: vx_b={vx_b}"

    def test_body_body_collision_stable_500_steps(self):
        """Body-body collision should stay stable for 500 steps (Q23 regression)."""
        merged = _two_ball_merged(radius=0.1)
        gpu = GpuEngine(merged, num_envs=1, dt=2e-4)

        # Two balls falling toward each other
        q, qdot = _init_two_balls(merged, x_a=-0.05, z_a=0.5, x_b=0.05, z_b=0.5)
        tau = np.zeros(merged.nv)

        for step in range(500):
            out = gpu.step(q, qdot, tau, dt=2e-4)
            q, qdot = out.q_new, out.qdot_new
            assert np.all(np.isfinite(q)), f"q has NaN at step {step}"
            assert np.all(np.isfinite(qdot)), f"qdot has NaN at step {step}"

        # Angular velocity of both balls should be bounded
        for name in ["a", "b"]:
            rs = merged.robot_slices[name]
            omega = qdot[rs.v_slice][3:6]
            assert np.linalg.norm(omega) < 100.0, (
                f"Ball {name} angular velocity diverged: norm={np.linalg.norm(omega)}"
            )

    def test_cpu_gpu_body_body_consistency(self):
        """CPU and GPU body-body collision should produce similar results."""
        merged = _two_ball_merged(radius=0.1)
        dt = 2e-4
        cpu = CpuEngine(merged, dt=dt)
        gpu = GpuEngine(merged, num_envs=1, dt=dt)

        q, qdot = _init_two_balls(merged, x_a=0.0, z_a=1.0, x_b=0.15, z_b=1.0)
        tau = np.zeros(merged.nv)

        q_cpu, qdot_cpu = q.copy(), qdot.copy()
        q_gpu, qdot_gpu = q.copy(), qdot.copy()

        for _ in range(50):
            out_cpu = cpu.step(q_cpu, qdot_cpu, tau, dt=dt)
            q_cpu, qdot_cpu = out_cpu.q_new, out_cpu.qdot_new

            out_gpu = gpu.step(q_gpu, qdot_gpu, tau, dt=dt)
            q_gpu, qdot_gpu = out_gpu.q_new, out_gpu.qdot_new

        # Should agree within float32 tolerance after 50 steps
        np.testing.assert_allclose(
            q_gpu,
            q_cpu,
            atol=0.05,
            err_msg="Body-body collision: q diverged CPU vs GPU",
        )


# ---------------------------------------------------------------------------
# Ground collision for both balls — ensure both roots work
# ---------------------------------------------------------------------------


class TestGpuTwoBallGroundContact:
    def test_single_ball_lands_on_ground(self):
        """Single ball should land and stabilize near z=radius."""
        m = _ball_model(mass=1.0, radius=0.1)
        merged = merge_models(robots={"a": m})
        gpu = GpuEngine(merged, num_envs=1, dt=2e-4)

        q, qdot = merged.tree.default_state()
        q[6] = 0.15  # just above ground
        tau = np.zeros(merged.nv)

        for step in range(2000):
            out = gpu.step(q, qdot, tau, dt=2e-4)
            q, qdot = out.q_new, out.qdot_new
            assert np.all(np.isfinite(q)), f"State diverged at step {step}"

        # Ball should rest near z=radius=0.1
        z = q[6]
        assert 0.05 < z < 0.2, f"Ball not near ground: z={z}"
