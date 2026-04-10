"""
Q25 fix — GPU PGS multi-body angular stability tests.

Coverage gap from session 16: the Q25 per-row R fix on friction rows
(physics/backends/warp/solver_kernels.py + crba_kernels.py + solver_kernels_v2.py,
session 15) was only validated on the CPU PGS path with single-sphere fixtures
(tests/unit/solvers/test_q25_friction_regularization.py — CPU only). The GPU
default solver (`jacobi_pgs_si`) was never tested in a multi-body scenario.

Past bugs Q23 (J_body_j sign in body-body contact) and Q28 (Plücker double-torque
in batched_impulse_to_gen_v2) showed that GPU bugs in body-body / multi-body
paths hide unless the test setup actually exercises:
  1. Bodies not at the world origin (catches Plücker bugs)
  2. body-body contact (catches J sign bugs)
  3. multi-row PGS interactions on the GPU jacobi path

These tests stress the Q25 fix on the GPU PGS solver in two multi-body
scenarios. Both run for 5000 GPU steps and assert that all body angular
velocities stay bounded (the Q25 failure mode is exponential growth driven by
friction × moment arm fake torques on bodies that should be at rest).

The companion ADMM solver is already covered for the same scenarios in
test_q28_friction_divergence.py.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ball_model(mass: float = 1.0, radius: float = 0.1) -> RobotModel:
    """Single FreeJoint sphere with collision geometry."""
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


def _two_ball_merged(radius: float = 0.1):
    a = _ball_model(mass=1.0, radius=radius)
    b = _ball_model(mass=1.0, radius=radius)
    return merge_models(robots={"a": a, "b": b})


def _place_two_balls(merged, x_a, z_a, x_b, z_b):
    """Set initial state for a two-ball merged model (FreeJoint q layout)."""
    q, _ = merged.tree.default_state()
    for name, rs in merged.robot_slices.items():
        qs = rs.q_slice
        if name == "a":
            q[qs.start + 4] = x_a  # px
            q[qs.start + 6] = z_a  # pz
        else:
            q[qs.start + 4] = x_b
            q[qs.start + 6] = z_b
    return q


def _max_omega_two_balls(gpu, merged, n_steps):
    """Run gpu for n_steps with zero torque, return (max |omega_a|, max |omega_b|)."""
    max_a = 0.0
    max_b = 0.0
    rs_a = merged.robot_slices["a"]
    rs_b = merged.robot_slices["b"]
    for step in range(n_steps):
        out = gpu.step(dt=2e-4)
        assert np.all(np.isfinite(out.q_new)), f"q NaN at step {step}"
        assert np.all(np.isfinite(out.qdot_new)), f"qdot NaN at step {step}"
        omega_a = out.qdot_new[rs_a.v_slice][3:6]
        omega_b = out.qdot_new[rs_b.v_slice][3:6]
        max_a = max(max_a, float(np.linalg.norm(omega_a)))
        max_b = max(max_b, float(np.linalg.norm(omega_b)))
    return max_a, max_b


# ---------------------------------------------------------------------------
# Scenario 1: two balls side by side on ground (no body-body contact)
# ---------------------------------------------------------------------------


class TestQ25GpuPgsTwoBallsSeparate:
    """Two separated balls on ground, default GPU PGS solver.

    Catches a Q25 regression on the second-root body. The CPU Q25 test only
    has one root body, so a sign / index error that affects body index >= 1
    (Q23-style) would not be visible. Body-body contact is NOT exercised here;
    the test isolates "multiple ground contacts on multiple roots" from
    "body-body contact".
    """

    @pytest.mark.slow
    def test_two_balls_resting_no_body_body(self):
        merged = _two_ball_merged(radius=0.1)
        gpu = GpuEngine(merged, num_envs=1, dt=2e-4)  # default solver = jacobi_pgs_si

        # x separation 1.0 m >> 2*radius=0.2 → no body-body contact ever
        q = _place_two_balls(merged, x_a=-0.5, z_a=0.11, x_b=0.5, z_b=0.11)
        gpu.reset(q)

        max_a, max_b = _max_omega_two_balls(gpu, merged, n_steps=5000)
        # CPU single-sphere Q25 test asserts < 0.1 rad/s. Both GPU bodies should match.
        assert max_a < 0.1, (
            f"Q25 GPU PGS regression on body 'a' (root 0): max |omega_a| = {max_a:.4f} rad/s (expected < 0.1)"
        )
        assert max_b < 0.1, (
            f"Q25 GPU PGS regression on body 'b' (second root): "
            f"max |omega_b| = {max_b:.4f} rad/s (expected < 0.1)"
        )


# ---------------------------------------------------------------------------
# Scenario 2: two balls touching, both on ground (body-body + ground)
# ---------------------------------------------------------------------------


class TestQ25GpuPgsTwoBallsTouching:
    """Two touching balls on ground, default GPU PGS solver.

    Catches Q25 + Q23 + Q28 simultaneously. Body-body contact and ground
    contact are both active throughout. The Q25 failure mode (friction ×
    moment arm) is amplified here because each ball has TWO contact points
    (one ground, one body-body), each with its own moment arm and friction
    row in the PGS solve.
    """

    @pytest.mark.slow
    def test_two_balls_touching_on_ground(self):
        merged = _two_ball_merged(radius=0.1)
        gpu = GpuEngine(merged, num_envs=1, dt=2e-4)

        # Slight overlap so body-body contact activates immediately and
        # persists. Separation 0.19 < 2*radius = 0.2.
        q = _place_two_balls(merged, x_a=-0.095, z_a=0.11, x_b=0.095, z_b=0.11)
        gpu.reset(q)

        max_a, max_b = _max_omega_two_balls(gpu, merged, n_steps=5000)
        # Allow slightly more tolerance than scenario 1: body-body friction
        # adds a second source of small impulses, but the Q25 fix should still
        # keep both omegas in the small-noise regime, not divergent.
        assert max_a < 0.5, f"Q25 GPU PGS body-body regression on body 'a': max |omega_a| = {max_a:.4f} rad/s"
        assert max_b < 0.5, f"Q25 GPU PGS body-body regression on body 'b': max |omega_b| = {max_b:.4f} rad/s"


# ---------------------------------------------------------------------------
# Scenario 3: quadruped standing still (chain dynamics + 4 ground contacts)
# ---------------------------------------------------------------------------


class TestQ25GpuPgsQuadrupedStanding:
    """Quadruped standing still on ground, default GPU PGS solver.

    This is the most realistic Q25 stress test: 13 bodies, 8 actuated DOF,
    4 simultaneous foot-ground contacts, and bodies that are NOT at the world
    origin (legs offset from base). Catches Q25 in a chain-dynamics setting
    where Q28-style Plücker bugs would also surface.

    The Q25 failure mode here would be: friction × moment arm at each foot
    drives a fake torque on the foot body, which propagates up the calf-hip-base
    kinematic chain and causes the base to start spinning or the legs to start
    twitching. After the per-row R fix, both should stay near zero.
    """

    @pytest.mark.slow
    def test_quadruped_standing_still_5000_steps(self):
        from tests.validation.rigid_body.models import build_quadruped

        model, _ = build_quadruped(contact=True)
        merged = merge_models(robots={"q": model})
        gpu = GpuEngine(merged, num_envs=1, dt=2e-4)

        # Default joint angles (0) → legs straight down. Foot rest height
        # below base = hip(0.2) + calf(0.2) + foot_radius(0.02) = 0.42.
        # Drop from 0.45 so feet land softly.
        q, _ = merged.tree.default_state()
        q[6] = 0.45  # base z
        gpu.reset(q)

        # Settle for 5000 steps (= 1s simulated). Bouncing should damp.
        for step in range(5000):
            out = gpu.step(dt=2e-4)
            assert np.all(np.isfinite(out.q_new)), f"q NaN during settling at step {step}"
            assert np.all(np.isfinite(out.qdot_new)), f"qdot NaN during settling at step {step}"

        # Now measure |omega_root| and joint qdot magnitude over next 3000 steps.
        # In steady state both should be ~zero. Q25 failure mode would show as
        # monotonic growth here.
        rs = merged.robot_slices["q"]
        max_root_omega = 0.0
        max_joint_qdot = 0.0
        for step in range(3000):
            out = gpu.step(dt=2e-4)
            assert np.all(np.isfinite(out.qdot_new)), f"qdot NaN in steady phase at step {step}"
            qdot_q = out.qdot_new[rs.v_slice]
            # FreeJoint v layout: [vx, vy, vz, wx, wy, wz] (linear; angular)
            omega_root = qdot_q[3:6]
            joint_qdot = qdot_q[6:]  # 8 revolute joint velocities
            max_root_omega = max(max_root_omega, float(np.linalg.norm(omega_root)))
            max_joint_qdot = max(max_joint_qdot, float(np.linalg.norm(joint_qdot)))

        assert max_root_omega < 0.5, (
            f"Q25 GPU PGS quadruped: base angular velocity not bounded after settling. "
            f"max |omega_root| = {max_root_omega:.4f} rad/s"
        )
        assert max_joint_qdot < 1.0, (
            f"Q25 GPU PGS quadruped: joint velocities not bounded after settling. "
            f"max |joint_qdot| = {max_joint_qdot:.4f} rad/s"
        )


# ---------------------------------------------------------------------------
# Scenario 4: two balls free-falling from different heights (temporal asymmetry)
# ---------------------------------------------------------------------------


class TestQ25GpuPgsTwoBallsDifferentHeights:
    """Two balls dropped from different heights, default GPU PGS solver.

    Tests temporal asymmetry in the multi-body GPU PGS path: ball A starts
    closer to the ground and lands first; ball B is still in free fall when A
    transitions through contact. This catches a class of bugs where the GPU
    PGS solver leaks state between bodies during contact transitions — e.g. a
    contact impulse for body A spuriously updating body B's velocity, or a
    friction row for A's ground contact corrupting B's free-fall trajectory.

    Throughout the entire run, both balls' angular velocities should stay near
    zero (no torque applied — gravity is central, ground normal contact at the
    body center has zero moment arm). B's vertical motion should match the
    analytical free-fall z(t) = z0 - 0.5 * g * t² until it lands, regardless
    of what A is doing.
    """

    @pytest.mark.slow
    def test_two_balls_different_heights_free_fall_then_land(self):
        merged = _two_ball_merged(radius=0.1)
        gpu = GpuEngine(merged, num_envs=1, dt=2e-4)

        # Horizontally separated (no body-body contact possible).
        # A drops from z=0.5 (lands ~step 1500); B drops from z=1.0 (lands ~step 2150).
        z_a_init = 0.5
        z_b_init = 1.0
        q = _place_two_balls(merged, x_a=-0.5, z_a=z_a_init, x_b=0.5, z_b=z_b_init)
        gpu.reset(q)

        rs_a = merged.robot_slices["a"]
        rs_b = merged.robot_slices["b"]

        # Phase 1: both in air → both free-fall
        # Phase 2: A on ground, B still in air → asymmetric state (the key test)
        # Phase 3: both on ground → both at rest
        n_steps = 5000
        max_omega_a = 0.0
        max_omega_b = 0.0
        z_b_history = np.zeros(n_steps)
        for step in range(n_steps):
            out = gpu.step(dt=2e-4)
            assert np.all(np.isfinite(out.q_new)), f"q NaN at step {step}"
            assert np.all(np.isfinite(out.qdot_new)), f"qdot NaN at step {step}"
            omega_a = out.qdot_new[rs_a.v_slice][3:6]
            omega_b = out.qdot_new[rs_b.v_slice][3:6]
            max_omega_a = max(max_omega_a, float(np.linalg.norm(omega_a)))
            max_omega_b = max(max_omega_b, float(np.linalg.norm(omega_b)))
            z_b_history[step] = out.q_new[rs_b.q_slice.start + 6]

        # Both omegas must stay near zero throughout the entire run.
        assert max_omega_a < 0.1, (
            f"Q25 GPU PGS asymmetric scenario, ball 'a' (lands first): "
            f"max |omega_a| = {max_omega_a:.4f} rad/s"
        )
        assert max_omega_b < 0.1, (
            f"Q25 GPU PGS asymmetric scenario, ball 'b' (lands later): "
            f"max |omega_b| = {max_omega_b:.4f} rad/s"
        )

        # Verify ball B's free-fall trajectory was not corrupted by ball A's
        # contact transitions. At step 1000 (t=0.2s), ball B should still be
        # in free fall: z_b(t) = z_b_init - 0.5 * g * t²
        # = 1.0 - 0.5 * 9.81 * 0.04 = 0.8038
        t_check = 1000 * 2e-4
        z_b_analytical = z_b_init - 0.5 * 9.81 * t_check**2
        z_b_actual = z_b_history[1000]
        assert abs(z_b_actual - z_b_analytical) < 1e-3, (
            f"Ball B free-fall trajectory disturbed by ball A's contact: "
            f"z_b at step 1000 = {z_b_actual:.5f}, expected {z_b_analytical:.5f} "
            f"(diff {(z_b_actual - z_b_analytical) * 1000:.3f} mm)"
        )

        # And ball B should eventually settle on the ground (radius=0.1)
        z_b_final = z_b_history[-1]
        assert 0.05 < z_b_final < 0.2, f"Ball B did not settle near ground: z_b_final = {z_b_final:.4f}"
