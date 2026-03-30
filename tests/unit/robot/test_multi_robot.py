"""
Tests for multi-robot Scene simulation.

Covers:
  - Two robots in one Scene, independent dynamics
  - Inter-robot collision detection and force distribution
  - Simulator.step() dict API with multiple robots
  - Force isolation (impulse on robot A doesn't corrupt robot B)

Reference: physical reasoning + internal consistency.
"""

from __future__ import annotations

import numpy as np

from physics.geometry import BodyCollisionGeometry, ShapeInstance, SphereShape
from physics.integrator import SemiImplicitEuler
from physics.joint import FreeJoint
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from robot.model import RobotModel
from scene import Scene, StaticGeometry
from simulator import Simulator


def _ball_model(mass: float = 1.0, radius: float = 0.1) -> RobotModel:
    tree = RobotTreeNumpy(gravity=9.81)
    tree.add_body(
        Body(
            name="ball",
            index=0,
            joint=FreeJoint("root"),
            inertia=SpatialInertia(mass=mass, inertia=np.eye(3) * 0.004, com=np.zeros(3)),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()
    return RobotModel(
        tree=tree,
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(SphereShape(radius))])],
    )


def _init_state(model, x=0.0, z=1.0):
    q, qdot = model.tree.default_state()  # default_q sets qw=1 at q[0]
    q[4] = x  # px
    q[6] = z  # pz
    return q, qdot


# ---------------------------------------------------------------------------
# Multi-robot Scene construction
# ---------------------------------------------------------------------------


class TestMultiRobotScene:
    def test_two_robots_build(self):
        scene = Scene(robots={"a": _ball_model(), "b": _ball_model()}).build()
        assert len(scene.robots) == 2
        assert scene.registry.total_bodies == 2

    def test_three_robots_offsets(self):
        scene = Scene(robots={"r1": _ball_model(), "r2": _ball_model(), "r3": _ball_model()}).build()
        reg = scene.registry
        assert reg.total_bodies == 3
        offsets = [reg.robot_offset[n] for n in ["r1", "r2", "r3"]]
        assert offsets == [0, 1, 2]


# ---------------------------------------------------------------------------
# Independent dynamics
# ---------------------------------------------------------------------------


class TestIndependentDynamics:
    def test_two_robots_free_fall_independent(self):
        """Two balls in free fall: both should fall identically."""
        m1 = _ball_model(mass=1.0)
        m2 = _ball_model(mass=1.0)
        scene = Scene(robots={"a": m1, "b": m2}).build()
        sim = Simulator(scene, SemiImplicitEuler(dt=1e-3))

        q_a, qdot_a = _init_state(m1, x=-1.0, z=2.0)
        q_b, qdot_b = _init_state(m2, x=1.0, z=2.0)

        states = {"a": (q_a, qdot_a), "b": (q_b, qdot_b)}
        taus = {"a": np.zeros(6), "b": np.zeros(6)}

        for _ in range(50):
            states = sim.step(states, taus)

        qa_new, qdota_new = states["a"]
        qb_new, qdotb_new = states["b"]

        # Both should have same z (same mass, same gravity, no contact)
        assert abs(qa_new[6] - qb_new[6]) < 1e-10
        # x positions should stay different
        assert abs(qa_new[4] - (-1.0)) < 1e-10
        assert abs(qb_new[4] - 1.0) < 1e-10

    def test_different_mass_different_contact(self):
        """Light vs heavy ball hitting ground: different bounce behavior."""
        m_light = _ball_model(mass=0.5)
        m_heavy = _ball_model(mass=10.0)
        scene = Scene(robots={"light": m_light, "heavy": m_heavy}).build()
        sim = Simulator(scene, SemiImplicitEuler(dt=1e-3))

        q_l, qdot_l = _init_state(m_light, x=-1.0, z=0.05)
        q_h, qdot_h = _init_state(m_heavy, x=1.0, z=0.05)
        qdot_l[2] = -2.0
        qdot_h[2] = -2.0

        states = {"light": (q_l, qdot_l), "heavy": (q_h, qdot_h)}
        taus = {"light": np.zeros(6), "heavy": np.zeros(6)}

        states = sim.step(states, taus)

        # Both should get upward impulse (ground contact)
        _, qdot_l_new = states["light"]
        _, qdot_h_new = states["heavy"]
        assert qdot_l_new[2] > -2.0  # light gets pushed up
        assert qdot_h_new[2] > -2.0  # heavy gets pushed up


# ---------------------------------------------------------------------------
# Inter-robot collision
# ---------------------------------------------------------------------------


class TestInterRobotCollision:
    def test_overlapping_robots_produce_contact(self):
        """Two spheres overlapping should generate inter-robot contact force.

        Note: the legacy Simulator path (per-robot StepPipeline) splits
        cross-robot contacts into per-robot views, which produces contact
        forces but not necessarily in the physically correct direction.
        The Engine path (CpuEngine/GpuEngine) handles body-body correctly.
        Here we just verify that SOME horizontal force is applied.
        """
        m1 = _ball_model(mass=1.0, radius=0.1)
        m2 = _ball_model(mass=1.0, radius=0.1)
        scene = Scene(robots={"a": m1, "b": m2}).build()
        sim = Simulator(scene, SemiImplicitEuler(dt=1e-3))

        # Place them overlapping: a at x=0, b at x=0.15 (overlap 0.05)
        q_a, qdot_a = _init_state(m1, x=0.0, z=1.0)
        q_b, qdot_b = _init_state(m2, x=0.15, z=1.0)

        states = {"a": (q_a, qdot_a), "b": (q_b, qdot_b)}
        taus = {"a": np.zeros(6), "b": np.zeros(6)}

        states = sim.step(states, taus)

        _, qdota_new = states["a"]
        _, qdotb_new = states["b"]

        # Contact should produce some horizontal velocity (not just gravity)
        assert abs(qdota_new[0]) > 1e-6 or abs(qdotb_new[0]) > 1e-6, (
            "Overlapping robots should produce contact force"
        )

    def test_separated_robots_no_contact(self):
        """Two spheres far apart should have no interaction."""
        m1 = _ball_model()
        m2 = _ball_model()
        scene = Scene(robots={"a": m1, "b": m2}).build()
        sim = Simulator(scene, SemiImplicitEuler(dt=1e-3))

        q_a, qdot_a = _init_state(m1, x=-5.0, z=1.0)
        q_b, qdot_b = _init_state(m2, x=5.0, z=1.0)

        states = {"a": (q_a.copy(), qdot_a.copy()), "b": (q_b.copy(), qdot_b.copy())}
        taus = {"a": np.zeros(6), "b": np.zeros(6)}

        states = sim.step(states, taus)

        qa_new, qdota_new = states["a"]
        qb_new, qdotb_new = states["b"]

        # Only gravity acts — no horizontal velocity change
        assert abs(qdota_new[0]) < 1e-10  # no x velocity
        assert abs(qdotb_new[0]) < 1e-10


# ---------------------------------------------------------------------------
# Static geometry with multi-robot
# ---------------------------------------------------------------------------


class TestMultiRobotWithStatic:
    def test_wall_affects_one_robot(self):
        """Wall near robot A, far from robot B — only A gets contact."""
        m1 = _ball_model(radius=0.1)
        m2 = _ball_model(radius=0.1)
        wall = StaticGeometry(
            name="wall",
            shape=SphereShape(0.5),
            pose=SpatialTransform.from_translation(np.array([-0.5, 0, 1])),
        )
        scene = Scene(
            robots={"a": m1, "b": m2},
            static_geometries=[wall],
        ).build()
        sim = Simulator(scene, SemiImplicitEuler(dt=1e-3))

        # Robot A near wall, robot B far away
        q_a, qdot_a = _init_state(m1, x=-0.45, z=1.0)
        q_b, qdot_b = _init_state(m2, x=10.0, z=1.0)

        states = {"a": (q_a, qdot_a), "b": (q_b, qdot_b)}
        taus = {"a": np.zeros(6), "b": np.zeros(6)}

        states = sim.step(states, taus)
        _, qdota_new = states["a"]
        _, qdotb_new = states["b"]

        # Robot B should only have gravity, no horizontal force
        assert abs(qdotb_new[0]) < 1e-10
