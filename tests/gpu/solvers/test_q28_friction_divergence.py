"""
Tests for Q28 fix — GPU ADMM multi-body simultaneous contact angular divergence.

Root cause (shared with Q25): spurious friction at near-zero tangential velocity
creates torque through the contact moment arm (r_arm = radius). In multi-body
scenarios, body-body + ground contact coupling amplifies this via positive
feedback (exponential growth, vs Q25's linear growth in single-body PGS).

Fix: tangential friction dead zone — zero out tangential impulse when both
current and predicted tangential velocities are below a threshold.
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
    from physics.gpu_engine import GpuEngine

    HAS_WARP = True
except Exception:
    HAS_WARP = False

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not HAS_WARP, reason="Warp or CUDA not available"),
]


def _ball_model(mass: float = 1.0, radius: float = 0.1) -> RobotModel:
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
    m_a = _ball_model(mass=1.0, radius=radius)
    m_b = _ball_model(mass=1.0, radius=radius)
    return merge_models(robots={"a": m_a, "b": m_b})


def _init_two_balls(merged, x_a=0.0, z_a=1.0, x_b=0.5, z_b=1.0):
    q, qdot = merged.tree.default_state()
    for name, rs in merged.robot_slices.items():
        qs = rs.q_slice
        if name == "a":
            q[qs.start + 4] = x_a
            q[qs.start + 6] = z_a
        else:
            q[qs.start + 4] = x_b
            q[qs.start + 6] = z_b
    return q, qdot


class TestQ28AdmmMultiBodyStability:
    """Two balls on ground with body-body contact — ADMM must not diverge."""

    def test_two_balls_ground_plus_body_body_stable_2000_steps(self):
        """Q28 reproduction: two balls close together, both on ground.

        Ground contacts + body-body contact activate simultaneously.
        Before fix: angular velocity → ~7000 rad/s → NaN at ~1000 steps.
        After fix: stable for 2000+ steps, angular velocity bounded.
        """
        merged = _two_ball_merged(radius=0.1)
        gpu = GpuEngine(merged, num_envs=1, dt=2e-4, solver="admm")

        # Place both balls just above ground, close enough for body-body contact
        # Separation = 0.19 < 2*radius = 0.2 → body-body contact active immediately
        q, qdot = _init_two_balls(merged, x_a=-0.095, z_a=0.11, x_b=0.095, z_b=0.11)
        gpu.reset(q)

        max_omega = 0.0
        for step in range(2000):
            out = gpu.step(dt=2e-4)
            q_now = out.q_new
            qdot_now = out.qdot_new

            assert np.all(np.isfinite(q_now)), f"q NaN at step {step}"
            assert np.all(np.isfinite(qdot_now)), f"qdot NaN at step {step}"

            for name in ["a", "b"]:
                rs = merged.robot_slices[name]
                omega = qdot_now[rs.v_slice][3:6]
                omega_norm = np.linalg.norm(omega)
                max_omega = max(max_omega, omega_norm)

        # Angular velocity must stay bounded (< 10 rad/s for balls at rest)
        assert max_omega < 10.0, f"Angular velocity diverged: max |ω| = {max_omega:.1f} rad/s"

    def test_two_balls_approach_and_land_stable_3000_steps(self):
        """Two balls approaching each other, both land on ground.

        Tests the full Q28 scenario: fall → land → body-body contact activates.
        """
        merged = _two_ball_merged(radius=0.1)
        gpu = GpuEngine(merged, num_envs=1, dt=2e-4, solver="admm")

        # Start from height, approaching each other
        q, qdot = _init_two_balls(merged, x_a=-0.12, z_a=0.5, x_b=0.12, z_b=0.5)
        import warp as wp

        nv = merged.nv
        qdot_init = np.zeros(nv)
        for name, rs in merged.robot_slices.items():
            vs = rs.v_slice
            if name == "a":
                qdot_init[vs.start] = 0.3  # approach
            else:
                qdot_init[vs.start] = -0.3
        gpu.reset(q)
        wp.copy(
            gpu._scratch.qdot,
            wp.array(
                qdot_init.reshape(1, -1).astype(np.float32),
                dtype=wp.float32,
                device=gpu._device,
            ),
        )

        for step in range(3000):
            out = gpu.step(dt=2e-4)
            assert np.all(np.isfinite(out.q_new)), f"q NaN at step {step}"
            assert np.all(np.isfinite(out.qdot_new)), f"qdot NaN at step {step}"

        # Both balls should have settled near ground
        for name in ["a", "b"]:
            rs = merged.robot_slices[name]
            qs = rs.q_slice
            z = out.q_new[qs.start + 6]
            assert z > 0.05, f"Ball {name} penetrated ground: z={z:.4f}"

            omega = out.qdot_new[rs.v_slice][3:6]
            assert np.linalg.norm(omega) < 10.0, f"Ball {name} angular velocity diverged: {omega}"

    def test_single_ball_angular_velocity_bounded(self):
        """Q25 regression: single ball on ground, angular velocity must stay near zero."""
        m = _ball_model(mass=1.0, radius=0.1)
        merged = merge_models(robots={"ball": m})
        gpu = GpuEngine(merged, num_envs=1, dt=2e-4, solver="admm")

        q, qdot = merged.tree.default_state()
        q[6] = 0.11  # just above ground
        gpu.reset(q)

        max_omega = 0.0
        for step in range(5000):
            out = gpu.step(dt=2e-4)
            omega = out.qdot_new[3:6]
            max_omega = max(max_omega, np.linalg.norm(omega))

        assert max_omega < 1.0, f"Single ball angular velocity grew: max |ω| = {max_omega:.4f} rad/s"

    def test_admm_mujoco_precision_not_regressed(self):
        """Single ball drop: ADMM vs MuJoCo precision must stay sub-mm."""
        m = _ball_model(mass=1.0, radius=0.1)
        merged = merge_models(robots={"ball": m})
        gpu = GpuEngine(merged, num_envs=1, dt=2e-4, solver="admm")

        q, qdot = merged.tree.default_state()
        q[6] = 0.5
        gpu.reset(q)

        for _ in range(5000):
            out = gpu.step(dt=2e-4)

        z = out.q_new[6]
        # MuJoCo steady-state: z ≈ 0.1 - 0.000367 = 0.099633
        # We need sub-mm accuracy (|z - 0.099633| < 1mm)
        mujoco_z = 0.099633
        assert abs(z - mujoco_z) < 0.001, (
            f"MuJoCo precision regressed: z={z:.6f}, expected ~{mujoco_z:.6f}, "
            f"diff={abs(z - mujoco_z) * 1000:.3f} mm"
        )
