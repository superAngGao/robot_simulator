"""
Tests for CollisionPipeline — unified collision detection.

Covers the three detection paths:
  1. Robot body vs terrain (ground_contact_query)
  2. Robot body vs static geometry (gjk_epa_query)
  3. Robot body vs robot body (self + inter-robot)

Also: gather_mass_properties() correctness.

Reference: analytical geometry (penetration depth, normal direction).
"""

from __future__ import annotations

import numpy as np

from collision_pipeline import CollisionPipeline
from physics.geometry import BodyCollisionGeometry, BoxShape, ShapeInstance, SphereShape
from physics.joint import FreeJoint, RevoluteJoint
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from robot.model import RobotModel
from scene import Scene, StaticGeometry


def _sphere_model(name: str = "base", mass: float = 1.0, radius: float = 0.1) -> RobotModel:
    tree = RobotTreeNumpy(gravity=9.81)
    tree.add_body(
        Body(
            name=name,
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


def _fk_at(model: RobotModel, z: float, x: float = 0.0):
    """Return (X_world, v_bodies) for a free body at (x, 0, z)."""
    q, qdot = model.tree.default_state()
    q[3] = 1.0  # qw
    q[4] = x
    q[6] = z
    X = model.tree.forward_kinematics(q)
    v = model.tree.body_velocities(q, qdot)
    return X, v, q, qdot


# ---------------------------------------------------------------------------
# 1. Robot body vs terrain
# ---------------------------------------------------------------------------


class TestRobotVsTerrain:
    def test_sphere_penetrating_ground(self):
        """Sphere at z=0.05 with radius 0.1 → penetrates ground by 0.05."""
        model = _sphere_model(radius=0.1)
        scene = Scene.single_robot(model)
        pipeline = CollisionPipeline(scene)

        X, v, _, _ = _fk_at(model, z=0.05)
        all_X = list(X)
        all_v = list(v)

        contacts = pipeline.detect(all_X, all_v)
        assert len(contacts) >= 1
        c = contacts[0]
        assert c.body_j == -1  # ground
        assert c.normal[2] > 0.9  # upward normal
        assert c.depth > 0.04  # ~0.05 penetration

    def test_sphere_above_ground(self):
        """Sphere at z=1.0 → no ground contact."""
        model = _sphere_model(radius=0.1)
        scene = Scene.single_robot(model)
        pipeline = CollisionPipeline(scene)

        X, v, _, _ = _fk_at(model, z=1.0)
        contacts = pipeline.detect(list(X), list(v))
        assert len(contacts) == 0

    def test_no_geometry_no_contact(self):
        """Model with no collision geometry → no contacts."""
        tree = RobotTreeNumpy(gravity=9.81)
        tree.add_body(
            Body(
                name="base",
                index=0,
                joint=FreeJoint("root"),
                inertia=SpatialInertia(mass=1.0, inertia=np.eye(3), com=np.zeros(3)),
                X_tree=SpatialTransform.identity(),
                parent=-1,
            )
        )
        tree.finalize()
        model = RobotModel(tree=tree, geometries=[])
        scene = Scene.single_robot(model)
        pipeline = CollisionPipeline(scene)

        X, v, _, _ = _fk_at(model, z=0.01)
        contacts = pipeline.detect(list(X), list(v))
        assert len(contacts) == 0


# ---------------------------------------------------------------------------
# 2. Robot body vs static geometry
# ---------------------------------------------------------------------------


class TestRobotVsStatic:
    def test_sphere_hits_wall(self):
        """Sphere at x=-0.45 with radius 0.1, wall at x=-0.5 → contact."""
        model = _sphere_model(radius=0.1)
        wall = StaticGeometry(
            name="wall",
            shape=BoxShape((0.02, 2.0, 2.0)),
            pose=SpatialTransform.from_translation(np.array([-0.5, 0, 1])),
            mu=0.3,
        )
        scene = Scene(
            robots={"main": model},
            static_geometries=[wall],
        ).build()
        pipeline = CollisionPipeline(scene)

        X, v, _, _ = _fk_at(model, z=1.0, x=-0.45)
        # Add static body pose/vel
        all_X = list(X) + [wall.pose]
        all_v = list(v) + [np.zeros(6)]

        contacts = pipeline.detect(all_X, all_v)
        # Should have at least one robot-static contact
        static_contacts = [c for c in contacts if c.body_j >= 0 and c.body_j != -1]
        assert len(static_contacts) >= 1
        sc = static_contacts[0]
        assert sc.mu == 0.3  # inherits from StaticGeometry

    def test_sphere_far_from_wall(self):
        """Sphere far from wall → no contact."""
        model = _sphere_model(radius=0.1)
        wall = StaticGeometry(
            name="wall",
            shape=BoxShape((0.02, 2.0, 2.0)),
            pose=SpatialTransform.from_translation(np.array([-5, 0, 1])),
        )
        scene = Scene(
            robots={"main": model},
            static_geometries=[wall],
        ).build()
        pipeline = CollisionPipeline(scene)

        X, v, _, _ = _fk_at(model, z=1.0, x=0.0)
        all_X = list(X) + [wall.pose]
        all_v = list(v) + [np.zeros(6)]

        contacts = pipeline.detect(all_X, all_v)
        static_contacts = [c for c in contacts if c.body_j >= 0 and c.body_j != -1]
        assert len(static_contacts) == 0


# ---------------------------------------------------------------------------
# 3. Robot body vs robot body
# ---------------------------------------------------------------------------


class TestRobotVsRobot:
    def test_two_spheres_colliding(self):
        """Two sphere robots overlapping → inter-robot contact."""
        m1 = _sphere_model(name="ball_a", radius=0.1)
        m2 = _sphere_model(name="ball_b", radius=0.1)
        scene = Scene(robots={"a": m1, "b": m2}).build()
        pipeline = CollisionPipeline(scene)

        # ball_a at (0,0,1), ball_b at (0.15,0,1) → overlap 0.05
        X_a, v_a, _, _ = _fk_at(m1, z=1.0, x=0.0)
        X_b, v_b, _, _ = _fk_at(m2, z=1.0, x=0.15)
        all_X = list(X_a) + list(X_b)
        all_v = list(v_a) + list(v_b)

        contacts = pipeline.detect(all_X, all_v)
        # Should have body-body contact (not ground, not static)
        bb_contacts = [c for c in contacts if c.body_i >= 0 and c.body_j >= 0]
        assert len(bb_contacts) >= 1

    def test_two_spheres_separated(self):
        """Two sphere robots far apart → no inter-robot contact."""
        m1 = _sphere_model(name="ball_a", radius=0.1)
        m2 = _sphere_model(name="ball_b", radius=0.1)
        scene = Scene(robots={"a": m1, "b": m2}).build()
        pipeline = CollisionPipeline(scene)

        X_a, v_a, _, _ = _fk_at(m1, z=1.0, x=0.0)
        X_b, v_b, _, _ = _fk_at(m2, z=1.0, x=5.0)
        all_X = list(X_a) + list(X_b)
        all_v = list(v_a) + list(v_b)

        contacts = pipeline.detect(all_X, all_v)
        bb_contacts = [c for c in contacts if c.body_i >= 0 and c.body_j >= 0]
        assert len(bb_contacts) == 0

    def test_self_collision_filtered(self):
        """Parent-child within same robot should not produce contact."""
        tree = RobotTreeNumpy(gravity=9.81)
        tree.add_body(
            Body(
                name="base",
                index=0,
                joint=FreeJoint("root"),
                inertia=SpatialInertia(mass=1.0, inertia=np.eye(3) * 0.01, com=np.zeros(3)),
                X_tree=SpatialTransform.identity(),
                parent=-1,
            )
        )
        tree.add_body(
            Body(
                name="link",
                index=1,
                joint=RevoluteJoint("j1", axis=np.array([0, 1, 0])),
                inertia=SpatialInertia(mass=0.5, inertia=np.eye(3) * 0.005, com=np.zeros(3)),
                X_tree=SpatialTransform(R=np.eye(3), r=np.array([0, 0, -0.05])),
                parent=0,
            )
        )
        tree.finalize()
        model = RobotModel(
            tree=tree,
            geometries=[
                BodyCollisionGeometry(0, [ShapeInstance(SphereShape(0.1))]),
                BodyCollisionGeometry(1, [ShapeInstance(SphereShape(0.1))]),
            ],
        )
        scene = Scene.single_robot(model)
        pipeline = CollisionPipeline(scene)

        q, qdot = tree.default_state()
        q[3] = 1.0
        q[6] = 1.0
        X = tree.forward_kinematics(q)
        v = tree.body_velocities(q, qdot)

        contacts = pipeline.detect(list(X), list(v))
        # Parent-child overlap is filtered by auto-exclude
        bb_contacts = [c for c in contacts if c.body_i >= 0 and c.body_j >= 0]
        assert len(bb_contacts) == 0


# ---------------------------------------------------------------------------
# gather_mass_properties
# ---------------------------------------------------------------------------


class TestMassProperties:
    def test_robot_body_has_mass(self):
        model = _sphere_model(mass=2.0)
        scene = Scene.single_robot(model)
        pipeline = CollisionPipeline(scene)

        inv_mass, inv_inertia = pipeline.gather_mass_properties()
        assert abs(inv_mass[0] - 0.5) < 1e-10  # 1/2.0

    def test_static_body_infinite_mass(self):
        model = _sphere_model()
        wall = StaticGeometry(
            name="wall",
            shape=BoxShape((1, 1, 1)),
            pose=SpatialTransform.identity(),
        )
        scene = Scene(robots={"main": model}, static_geometries=[wall]).build()
        pipeline = CollisionPipeline(scene)

        inv_mass, inv_inertia = pipeline.gather_mass_properties()
        # Static body (global id 1) has inv_mass = 0
        assert inv_mass[1] == 0.0
        assert np.allclose(inv_inertia[1], np.zeros((3, 3)))
