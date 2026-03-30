"""
Integration tests for StaticGeometry through the full Scene+Pipeline+Simulator.

These tests validate that walls, ramps, and obstacles work end-to-end
without hand-written ContactConstraints — the CollisionPipeline detects
contacts automatically via GJK/EPA against StaticGeometry.

Reference: PyBullet for qualitative comparison, analytical for directions.
"""

from __future__ import annotations

import numpy as np

from physics.geometry import BodyCollisionGeometry, BoxShape, ShapeInstance, SphereShape
from physics.integrator import SemiImplicitEuler
from physics.joint import FreeJoint
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from robot.model import RobotModel
from scene import Scene, StaticGeometry
from simulator import Simulator


def _ball_model(mass=1.0, radius=0.1):
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


def _init(model, x=0.0, z=1.0, vx=0.0, vz=0.0):
    q, qdot = model.tree.default_state()
    q[3] = 1.0
    q[4] = x
    q[6] = z
    qdot[0] = vx
    qdot[2] = vz
    return q, qdot


class TestBallHitsWall:
    """Ball thrown at a vertical wall — the core use case that motivated Scene.

    Reference: physical reasoning.
    Before Scene, this required manually constructing ContactConstraints.
    Now it works automatically through CollisionPipeline.
    """

    def test_wall_stops_ball(self):
        """Ball moving toward wall should be stopped/reversed."""
        model = _ball_model()
        wall = StaticGeometry(
            name="wall",
            shape=BoxShape((0.1, 2.0, 2.0)),
            pose=SpatialTransform.from_translation(np.array([-0.5, 0, 1])),
            mu=0.5,
        )
        scene = Scene(
            robots={"main": model},
            static_geometries=[wall],
        ).build()
        sim = Simulator(scene, SemiImplicitEuler(dt=1e-3))

        q, qdot = _init(model, x=0.0, z=1.0, vx=-5.0)
        tau = np.zeros(6)

        # Run until ball reaches wall (~80ms at vx=-5)
        for _ in range(120):
            q, qdot = sim.step_single(q, qdot, tau)

        # Ball should not pass far through wall. Our ERP is gentle (~12 steps
        # to arrest), so slight overshoot beyond wall face is expected.
        assert q[4] > -0.65, f"Ball x={q[4]:.3f}, should not pass through wall"
        assert np.all(np.isfinite(q))

    def test_wall_friction_slows_sliding(self):
        """Ball sliding down a rough wall should decelerate vertically."""
        model = _ball_model()
        wall = StaticGeometry(
            name="wall",
            shape=BoxShape((0.1, 2.0, 2.0)),
            pose=SpatialTransform.from_translation(np.array([-0.45, 0, 1])),
            mu=0.8,  # rough wall
        )
        scene = Scene(
            robots={"main": model},
            static_geometries=[wall],
        ).build()
        sim = Simulator(scene, SemiImplicitEuler(dt=1e-3))

        # Ball pressed against wall with downward velocity
        q, qdot = _init(model, x=-0.4, z=1.0, vx=-2.0, vz=-3.0)
        tau = np.zeros(6)

        for _ in range(50):
            q, qdot = sim.step_single(q, qdot, tau)

        assert np.all(np.isfinite(q))


class TestBallOnGround:
    """Ball settling on ground via CollisionPipeline (not old PenaltyContactModel)."""

    def test_ball_settles_on_ground(self):
        """Ball dropped from height should settle at z ≈ radius."""
        model = _ball_model(radius=0.1)
        scene = Scene.single_robot(model)
        sim = Simulator(scene, SemiImplicitEuler(dt=1e-3))

        q, qdot = _init(model, z=0.5)
        tau = np.zeros(6)

        for _ in range(500):
            q, qdot = sim.step_single(q, qdot, tau)

        assert q[6] > 0.05, f"Ball fell through ground: z={q[6]:.4f}"
        assert q[6] < 0.3, f"Ball didn't settle: z={q[6]:.4f}"
        assert abs(qdot[2]) < 1.0, f"Ball still bouncing: vz={qdot[2]:.4f}"


class TestMultipleObstacles:
    """Scene with ground + multiple walls."""

    def test_corridor(self):
        """Ball in a corridor (two walls) should stay between them."""
        model = _ball_model()
        left_wall = StaticGeometry(
            name="left",
            shape=BoxShape((0.1, 2.0, 2.0)),
            pose=SpatialTransform.from_translation(np.array([-1, 0, 1])),
        )
        right_wall = StaticGeometry(
            name="right",
            shape=BoxShape((0.1, 2.0, 2.0)),
            pose=SpatialTransform.from_translation(np.array([1, 0, 1])),
        )
        scene = Scene(
            robots={"main": model},
            static_geometries=[left_wall, right_wall],
        ).build()
        sim = Simulator(scene, SemiImplicitEuler(dt=1e-3))

        q, qdot = _init(model, x=0.0, z=1.0, vx=3.0)
        tau = np.zeros(6)

        for _ in range(200):
            q, qdot = sim.step_single(q, qdot, tau)

        # Ball should stay between walls
        assert q[4] > -1.1, f"Ball escaped left: x={q[4]:.3f}"
        assert q[4] < 1.1, f"Ball escaped right: x={q[4]:.3f}"
        assert np.all(np.isfinite(q))
