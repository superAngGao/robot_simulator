"""
Integration tests: objects on inclined planes with friction.

Validates HalfSpaceTerrain + halfspace_convex_query through the full
Simulator → CollisionPipeline → PGS solver stack.

Analytical reference (block on incline, angle θ, gravity g, friction μ):
  Going up:    a = -(g sinθ + μ g cosθ)   (gravity + friction both decelerate)
  Sliding down: a = g (sinθ - μ cosθ)     (gravity accelerates, friction decelerates)
  Static:      stays at rest if μ ≥ tanθ

Reference: Kleppner & Kolenkow, *An Introduction to Mechanics*, Ch. 3.
"""

from __future__ import annotations

import numpy as np

from physics.geometry import BodyCollisionGeometry, BoxShape, ShapeInstance, SphereShape
from physics.integrator import SemiImplicitEuler
from physics.joint import FreeJoint
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from physics.terrain import HalfSpaceTerrain
from robot.model import RobotModel
from scene import Scene
from simulator import Simulator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _incline_normal(theta: float) -> np.ndarray:
    """Outward normal for a plane tilted by *theta* around the Y axis.

    The incline rises in the +x direction: at x > 0 the surface is higher.
    Normal has a −x component and a +z component.
    """
    return np.array([-np.sin(theta), 0.0, np.cos(theta)])


def _along_incline(theta: float) -> np.ndarray:
    """Unit vector pointing *up the slope* (positive x, positive z)."""
    return np.array([np.cos(theta), 0.0, np.sin(theta)])


def _ball_on_incline(
    theta: float,
    mu: float,
    mass: float = 1.0,
    radius: float = 0.05,
    dt: float = 1e-3,
    v0_along: float = 0.0,
) -> tuple[Simulator, np.ndarray, np.ndarray]:
    """Build a sphere resting on an inclined plane.

    Returns (simulator, q0, qdot0) ready for stepping.
    """
    tree = RobotTreeNumpy(gravity=9.81)
    I_sphere = 2.0 / 5.0 * mass * radius**2
    tree.add_body(
        Body(
            name="ball",
            index=0,
            joint=FreeJoint("root"),
            inertia=SpatialInertia(
                mass=mass,
                inertia=np.eye(3) * I_sphere,
                com=np.zeros(3),
            ),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()

    model = RobotModel(
        tree=tree,
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(SphereShape(radius))])],
        contact_body_names=["ball"],
    )

    normal = _incline_normal(theta)
    terrain = HalfSpaceTerrain(normal=normal, point=np.zeros(3), mu=mu)
    scene = Scene.single_robot(model, terrain=terrain)

    q, qdot = tree.default_state()
    q[0] = 1.0  # qw = 1 (identity quaternion)

    # Place ball centre at radius above the plane surface through origin
    # so the sphere just touches the plane.
    pos = normal * (radius + 1e-4)  # tiny gap to avoid initial penetration
    q[4] = pos[0]
    q[5] = pos[1]
    q[6] = pos[2]

    # Set initial velocity along the incline
    if v0_along != 0.0:
        v_world = v0_along * _along_incline(theta)
        qdot[0] = v_world[0]
        qdot[1] = v_world[1]
        qdot[2] = v_world[2]

    sim = Simulator(scene, SemiImplicitEuler(dt=dt))
    return sim, q, qdot


def _velocity_along_incline(qdot: np.ndarray, theta: float) -> float:
    """Project world-frame linear velocity onto the up-slope direction."""
    return float(np.dot(qdot[:3], _along_incline(theta)))


def _run_steps(sim, q, qdot, n_steps):
    tau = np.zeros(sim.scene.robots["main"].tree.nv)
    for _ in range(n_steps):
        q, qdot = sim.step_single(q, qdot, tau)
    return q, qdot


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInclinePhysics:
    """Physics validation for objects on inclined planes."""

    def test_frictionless_slide_down(self):
        """μ ≈ 0: sphere accelerates at g sinθ down the slope."""
        theta = np.radians(30)
        dt = 1e-3
        n_steps = 200
        sim, q, qdot = _ball_on_incline(theta, mu=0.01, dt=dt)

        q, qdot = _run_steps(sim, q, qdot, n_steps)

        # Expected: v = a * t, a = g * sinθ ≈ 4.905
        t = n_steps * dt
        a_expected = 9.81 * np.sin(theta)
        v_expected = -a_expected * t  # negative = down-slope
        v_actual = _velocity_along_incline(qdot, theta)

        # Allow 15% tolerance (solver damping, small μ, discrete time)
        assert v_actual < 0, "Ball should slide down"
        assert abs(v_actual - v_expected) / abs(v_expected) < 0.15, (
            f"v_actual={v_actual:.4f}, v_expected={v_expected:.4f}"
        )

    def test_rolling_sphere_high_friction(self):
        """μ > (2/7)tanθ: sphere rolls without sliding.

        For a solid sphere rolling on an incline, translational
        acceleration is a = (5/7) g sinθ, independent of μ.
        Condition for rolling: μ ≥ (2/7) tanθ.
        """
        theta = np.radians(20)
        mu = 1.0  # >> (2/7)*tan(20°) ≈ 0.104
        dt = 1e-3
        n_steps = 500
        sim, q, qdot = _ball_on_incline(theta, mu=mu, dt=dt)

        q, qdot = _run_steps(sim, q, qdot, n_steps)

        t = n_steps * dt
        a_expected = (5.0 / 7.0) * 9.81 * np.sin(theta)
        v_expected = -a_expected * t
        v_actual = _velocity_along_incline(qdot, theta)

        assert v_actual < 0, "Ball should roll down"
        assert abs(v_actual - v_expected) / abs(v_expected) < 0.15, (
            f"v_actual={v_actual:.4f}, v_expected={v_expected:.4f}"
        )

    def test_sliding_sphere_low_friction(self):
        """μ < (2/7)tanθ: sphere slides, a = g(sinθ − μ cosθ).

        When friction is too low to sustain rolling without slip,
        kinetic friction applies: f = μN, giving the block-on-slope
        acceleration formula for translation.
        """
        theta = np.radians(30)
        mu = 0.1  # < (2/7)*tan(30°) ≈ 0.165 → sliding regime
        dt = 1e-3
        n_steps = 200
        sim, q, qdot = _ball_on_incline(theta, mu=mu, dt=dt)

        q, qdot = _run_steps(sim, q, qdot, n_steps)

        t = n_steps * dt
        a_expected = 9.81 * (np.sin(theta) - mu * np.cos(theta))
        v_expected = -a_expected * t
        v_actual = _velocity_along_incline(qdot, theta)

        assert v_actual < 0, "Ball should slide down"
        assert abs(v_actual - v_expected) / abs(v_expected) < 0.20, (
            f"v_actual={v_actual:.4f}, v_expected={v_expected:.4f}"
        )

    def test_up_then_down(self):
        """Initial upward velocity → decelerate, stop, slide back down.

        Phase 1 (going up):   a = -(g sinθ + μ g cosθ)
        Phase 2 (going down): a =   g sinθ - μ g cosθ   (if μ < tanθ)
        """
        theta = np.radians(30)
        mu = 0.2  # tan(30°) ≈ 0.577, so μ < tanθ → will slide back
        v0 = 1.0  # initial speed up the slope
        dt = 1e-3
        sim, q, qdot = _ball_on_incline(theta, mu=mu, dt=dt, v0_along=v0)

        g = 9.81
        a_up = g * np.sin(theta) + mu * g * np.cos(theta)
        t_stop = v0 / a_up  # time to reach zero velocity going up

        # --- Phase 1: run until just past the stop time ---
        n_phase1 = int(t_stop / dt) + 50
        q, qdot = _run_steps(sim, q, qdot, n_phase1)

        # Velocity should be near zero or slightly negative
        v1 = _velocity_along_incline(qdot, theta)
        assert v1 < 0.15, f"Should have stopped or reversed, v={v1:.4f}"

        # --- Phase 2: continue and verify it slides back down ---
        n_phase2 = 300
        q, qdot = _run_steps(sim, q, qdot, n_phase2)
        v2 = _velocity_along_incline(qdot, theta)
        assert v2 < -0.1, f"Should be sliding down, v={v2:.4f}"

    def test_box_on_incline(self):
        """BoxShape (not just sphere) works on inclined half-space."""
        theta = np.radians(30)
        mu = 0.01
        dt = 1e-3
        mass = 1.0
        half = 0.05

        tree = RobotTreeNumpy(gravity=9.81)
        tree.add_body(
            Body(
                name="box",
                index=0,
                joint=FreeJoint("root"),
                inertia=SpatialInertia(
                    mass=mass,
                    inertia=np.eye(3) * mass * (2 * half) ** 2 / 6.0,
                    com=np.zeros(3),
                ),
                X_tree=SpatialTransform.identity(),
                parent=-1,
            )
        )
        tree.finalize()

        model = RobotModel(
            tree=tree,
            geometries=[BodyCollisionGeometry(0, [ShapeInstance(BoxShape((2 * half,) * 3))])],
            contact_body_names=["box"],
        )

        normal = _incline_normal(theta)
        terrain = HalfSpaceTerrain(normal=normal, point=np.zeros(3), mu=mu)
        scene = Scene.single_robot(model, terrain=terrain)

        q, qdot = tree.default_state()
        q[0] = 1.0
        pos = normal * (half + 1e-4)
        q[4], q[5], q[6] = pos

        sim = Simulator(scene, SemiImplicitEuler(dt=dt))
        q, qdot = _run_steps(sim, q, qdot, 200)

        v = _velocity_along_incline(qdot, theta)
        assert v < -0.1, f"Box should slide down, v={v:.4f}"
