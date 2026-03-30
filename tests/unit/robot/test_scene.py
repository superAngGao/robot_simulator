"""
Tests for Scene, StaticGeometry, and BodyRegistry.

Covers:
  - BodyRegistry index mapping (global ↔ local, single/multi robot)
  - Scene.build() auto-exclude, collision_filter creation
  - Scene.single_robot() convenience
  - StaticGeometry in Scene
  - Multi-robot Scene construction
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.collision_filter import CollisionFilter
from physics.geometry import BodyCollisionGeometry, BoxShape, ShapeInstance, SphereShape
from physics.joint import FreeJoint, RevoluteJoint
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from robot.model import RobotModel
from scene import BodyRegistry, Scene, StaticGeometry


def _one_body_model(name_prefix: str = "") -> RobotModel:
    tree = RobotTreeNumpy(gravity=9.81)
    tree.add_body(
        Body(
            name=f"{name_prefix}base",
            index=0,
            joint=FreeJoint("root"),
            inertia=SpatialInertia(mass=1.0, inertia=np.eye(3) * 0.01, com=np.zeros(3)),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()
    return RobotModel(
        tree=tree,
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(SphereShape(0.1))])],
    )


def _two_body_model() -> RobotModel:
    tree = RobotTreeNumpy(gravity=9.81)
    tree.add_body(
        Body(
            name="base",
            index=0,
            joint=FreeJoint("root"),
            inertia=SpatialInertia(mass=2.0, inertia=np.eye(3) * 0.05, com=np.zeros(3)),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.add_body(
        Body(
            name="link1",
            index=1,
            joint=RevoluteJoint("j1", axis=np.array([0, 1, 0])),
            inertia=SpatialInertia(mass=0.5, inertia=np.eye(3) * 0.005, com=np.zeros(3)),
            X_tree=SpatialTransform(R=np.eye(3), r=np.array([0, 0, -0.3])),
            parent=0,
        )
    )
    tree.finalize()
    return RobotModel(
        tree=tree,
        geometries=[
            BodyCollisionGeometry(0, [ShapeInstance(BoxShape((0.2, 0.2, 0.1)))]),
            BodyCollisionGeometry(1, [ShapeInstance(SphereShape(0.05))]),
        ],
    )


# ---------------------------------------------------------------------------
# BodyRegistry
# ---------------------------------------------------------------------------


class TestBodyRegistry:
    def test_single_robot_offsets(self):
        model = _two_body_model()
        reg = BodyRegistry({"main": model})
        assert reg.robot_offset["main"] == 0
        assert reg.robot_num_bodies["main"] == 2
        assert reg.total_bodies == 2

    def test_multi_robot_offsets(self):
        m1 = _two_body_model()  # 2 bodies
        m2 = _one_body_model()  # 1 body
        reg = BodyRegistry({"arm": m1, "ball": m2})
        assert reg.robot_offset["arm"] == 0
        assert reg.robot_offset["ball"] == 2
        assert reg.total_bodies == 3

    def test_static_offset(self):
        model = _one_body_model()
        reg = BodyRegistry({"main": model}, n_static=3)
        assert reg.static_offset == 1
        assert reg.total_bodies == 4
        assert reg.static_global_id(0) == 1
        assert reg.static_global_id(2) == 3

    def test_global_id(self):
        m1 = _two_body_model()
        m2 = _one_body_model()
        reg = BodyRegistry({"arm": m1, "ball": m2})
        assert reg.global_id("arm", 0) == 0
        assert reg.global_id("arm", 1) == 1
        assert reg.global_id("ball", 0) == 2

    def test_to_local(self):
        m1 = _two_body_model()
        m2 = _one_body_model()
        reg = BodyRegistry({"arm": m1, "ball": m2}, n_static=1)
        assert reg.to_local(0) == ("arm", 0)
        assert reg.to_local(1) == ("arm", 1)
        assert reg.to_local(2) == ("ball", 0)
        assert reg.to_local(3) == (None, 0)  # static

    def test_is_static(self):
        reg = BodyRegistry({"main": _one_body_model()}, n_static=2)
        assert not reg.is_static(0)
        assert reg.is_static(1)
        assert reg.is_static(2)


# ---------------------------------------------------------------------------
# Scene.build()
# ---------------------------------------------------------------------------


class TestSceneBuild:
    def test_build_creates_registry(self):
        scene = Scene(robots={"main": _one_body_model()}).build()
        assert scene._registry is not None
        assert scene.registry.total_bodies == 1

    def test_build_creates_collision_filter(self):
        scene = Scene(robots={"main": _one_body_model()}).build()
        assert scene.collision_filter is not None
        assert isinstance(scene.collision_filter, CollisionFilter)

    def test_auto_exclude_parent_child(self):
        """Parent-child pairs within a robot should be auto-excluded."""
        model = _two_body_model()  # base → link1
        scene = Scene(robots={"main": model}).build()
        f = scene.collision_filter
        assert not f.should_collide(0, 1)  # parent-child excluded

    def test_multi_robot_auto_exclude(self):
        """Each robot's parent-child excluded, but inter-robot not excluded."""
        m1 = _two_body_model()  # bodies 0,1 — parent-child excluded
        m2 = _one_body_model()  # body 2
        scene = Scene(robots={"arm": m1, "ball": m2}).build()
        f = scene.collision_filter
        assert not f.should_collide(0, 1)  # arm parent-child
        assert f.should_collide(0, 2)  # arm vs ball — allowed
        assert f.should_collide(1, 2)  # arm link vs ball — allowed

    def test_registry_not_accessible_before_build(self):
        scene = Scene(robots={"main": _one_body_model()})
        with pytest.raises(RuntimeError, match="build"):
            _ = scene.registry


# ---------------------------------------------------------------------------
# Scene.single_robot()
# ---------------------------------------------------------------------------


class TestSingleRobot:
    def test_creates_main_robot(self):
        scene = Scene.single_robot(_one_body_model())
        assert "main" in scene.robots
        assert scene.registry.total_bodies == 1

    def test_with_static_geometry(self):
        wall = StaticGeometry(
            name="wall",
            shape=BoxShape((0.1, 2.0, 2.0)),
            pose=SpatialTransform.from_translation(np.array([-1, 0, 1])),
        )
        scene = Scene.single_robot(_one_body_model(), static_geometries=[wall])
        assert len(scene.static_geometries) == 1
        assert scene.registry.total_bodies == 2  # 1 robot + 1 static


# ---------------------------------------------------------------------------
# StaticGeometry
# ---------------------------------------------------------------------------


class TestStaticGeometry:
    def test_default_friction(self):
        sg = StaticGeometry(
            name="floor",
            shape=BoxShape((10, 10, 0.1)),
            pose=SpatialTransform.identity(),
        )
        assert sg.mu == 0.5
        assert sg.condim == 3

    def test_custom_friction(self):
        sg = StaticGeometry(
            name="ice",
            shape=BoxShape((10, 10, 0.1)),
            pose=SpatialTransform.identity(),
            mu=0.05,
            condim=1,
        )
        assert sg.mu == 0.05
        assert sg.condim == 1


# ---------------------------------------------------------------------------
# Scene repr
# ---------------------------------------------------------------------------


class TestSceneRepr:
    def test_repr_includes_counts(self):
        scene = Scene.single_robot(_one_body_model())
        r = repr(scene)
        assert "robots=1" in r
        assert "static=0" in r
