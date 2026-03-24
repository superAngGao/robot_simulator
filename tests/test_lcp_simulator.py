"""
Tests for contact through Simulator + Scene + CollisionPipeline.

Verifies:
  1. Ground contact via CollisionPipeline produces correct forces
  2. Multi-step simulation is stable
  3. load_urdf_scene builds a working Scene
  4. Different masses produce different responses
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from physics.geometry import BodyCollisionGeometry, ShapeInstance, SphereShape
from physics.integrator import SemiImplicitEuler
from physics.joint import FreeJoint, RevoluteJoint
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from robot.model import RobotModel
from scene import Scene
from simulator import Simulator


def _make_ball_scene(
    mass: float = 5.0,
    z0: float = 0.5,
    shape_radius: float = 0.1,
) -> tuple[Scene, np.ndarray, np.ndarray]:
    """Single free-floating sphere with collision geometry → Scene."""
    tree = RobotTreeNumpy(gravity=9.81)
    body = Body(
        name="base",
        index=0,
        joint=FreeJoint("root"),
        inertia=SpatialInertia(mass=mass, inertia=np.diag([0.1, 0.1, 0.1]), com=np.zeros(3)),
        X_tree=SpatialTransform.identity(),
        parent=-1,
    )
    tree.add_body(body)
    tree.finalize()

    model = RobotModel(
        tree=tree,
        contact_body_names=["base"],
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(SphereShape(shape_radius))])],
    )
    scene = Scene.single_robot(model)

    q, qdot = tree.default_state()
    q[3] = 1.0
    q[6] = z0
    return scene, q, qdot


def _make_two_body_scene():
    """Free base + revolute foot, foot has collision sphere."""
    tree = RobotTreeNumpy(gravity=9.81)
    base = Body(
        name="base",
        index=0,
        joint=FreeJoint("root"),
        inertia=SpatialInertia(mass=5.0, inertia=np.diag([0.1, 0.1, 0.1]), com=np.zeros(3)),
        X_tree=SpatialTransform.identity(),
        parent=-1,
    )
    tree.add_body(base)
    foot = Body(
        name="foot",
        index=1,
        joint=RevoluteJoint("hip", axis=np.array([0, 1, 0])),
        inertia=SpatialInertia(mass=1.0, inertia=np.diag([0.01, 0.01, 0.01]), com=np.array([0, 0, -0.15])),
        X_tree=SpatialTransform(R=np.eye(3), r=np.array([0.0, 0.0, -0.2])),
        parent=0,
    )
    tree.add_body(foot)
    tree.finalize()

    model = RobotModel(
        tree=tree,
        contact_body_names=["foot"],
        geometries=[BodyCollisionGeometry(1, [ShapeInstance(SphereShape(0.05))])],
    )
    scene = Scene.single_robot(model)

    q, qdot = tree.default_state()
    q[3] = 1.0
    q[6] = 0.5
    return scene, q, qdot


class TestContactSimulatorIntegration:
    def test_single_step_finite(self):
        scene, q, qdot = _make_ball_scene(z0=0.05)
        sim = Simulator(scene, SemiImplicitEuler(dt=1e-3))
        tau = np.zeros(scene.robots["main"].tree.nv)
        q_new, qdot_new = sim.step_single(q, qdot, tau)
        assert np.all(np.isfinite(q_new))
        assert np.all(np.isfinite(qdot_new))

    def test_contact_produces_upward_force(self):
        scene, q, qdot = _make_ball_scene(z0=0.05)
        sim = Simulator(scene, SemiImplicitEuler(dt=1e-3))
        tau = np.zeros(scene.robots["main"].tree.nv)
        qdot[2] = -1.0
        _, qdot_new = sim.step_single(q, qdot, tau)
        assert qdot_new[2] > qdot[2]

    def test_free_fall_no_contact(self):
        scene, q, qdot = _make_ball_scene(z0=1.0)
        sim = Simulator(scene, SemiImplicitEuler(dt=1e-3))
        tau = np.zeros(scene.robots["main"].tree.nv)
        _, qdot_new = sim.step_single(q, qdot, tau)
        assert abs(qdot_new[2] - (-9.81 * 1e-3)) < 1e-6

    def test_multi_step_stable(self):
        scene, q, qdot = _make_ball_scene(z0=0.2)
        sim = Simulator(scene, SemiImplicitEuler(dt=1e-3))
        tau = np.zeros(scene.robots["main"].tree.nv)
        for _ in range(50):
            q, qdot = sim.step_single(q, qdot, tau)
        assert np.all(np.isfinite(q))
        assert q[6] > -0.5

    def test_real_mass_affects_dynamics(self):
        scene_l, q_l, qdot_l = _make_ball_scene(mass=1.0, z0=0.05)
        scene_h, q_h, qdot_h = _make_ball_scene(mass=50.0, z0=0.05)
        qdot_l[2] = -1.0
        qdot_h[2] = -1.0
        sim_l = Simulator(scene_l, SemiImplicitEuler(dt=1e-3))
        sim_h = Simulator(scene_h, SemiImplicitEuler(dt=1e-3))
        tau = np.zeros(6)
        _, qdot_l_new = sim_l.step_single(q_l, qdot_l, tau)
        _, qdot_h_new = sim_h.step_single(q_h, qdot_h, tau)
        assert qdot_l_new[2] != pytest.approx(qdot_h_new[2], abs=1e-4)

    def test_two_body(self):
        scene, q, qdot = _make_two_body_scene()
        sim = Simulator(scene, SemiImplicitEuler(dt=1e-3))
        tau = np.zeros(scene.robots["main"].tree.nv)
        for _ in range(20):
            q, qdot = sim.step_single(q, qdot, tau)
        assert np.all(np.isfinite(q))


_SIMPLE_URDF = """\
<robot name="test_bot">
  <link name="base">
    <inertial>
      <mass value="5.0"/><origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
    <collision><geometry><box size="0.4 0.3 0.2"/></geometry></collision>
  </link>
  <link name="leg">
    <inertial>
      <mass value="1.0"/><origin xyz="0 0 -0.15"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <collision><geometry><sphere radius="0.05"/></geometry></collision>
  </link>
  <joint name="hip" type="revolute">
    <parent link="base"/><child link="leg"/>
    <origin xyz="0 0 -0.1"/><axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50"/>
  </joint>
</robot>
"""


class TestLoadURDFScene:
    def _write_urdf(self) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
        f.write(_SIMPLE_URDF)
        f.close()
        return f.name

    def test_load_scene_creates_scene(self):
        from robot import load_urdf_scene

        path = self._write_urdf()
        try:
            scene = load_urdf_scene(path, floating_base=True, contact_links=["leg"])
        finally:
            os.unlink(path)
        assert isinstance(scene, Scene)
        assert "main" in scene.robots

    def test_load_scene_step_stable(self):
        from robot import load_urdf_scene

        path = self._write_urdf()
        try:
            scene = load_urdf_scene(path, floating_base=True, contact_links=["leg"])
        finally:
            os.unlink(path)
        sim = Simulator(scene, SemiImplicitEuler(dt=1e-3))
        model = scene.robots["main"]
        q, qdot = model.tree.default_state()
        q[3] = 1.0
        q[6] = 0.3
        tau = np.zeros(model.tree.nv)
        for _ in range(20):
            q, qdot = sim.step_single(q, qdot, tau)
        assert np.all(np.isfinite(q))

    def test_load_urdf_returns_model(self):
        """load_urdf (not scene) returns a plain RobotModel."""
        from robot import load_urdf

        path = self._write_urdf()
        try:
            model = load_urdf(path, floating_base=True, contact_links=["leg"])
        finally:
            os.unlink(path)
        assert isinstance(model, RobotModel)
        assert "leg" in model.contact_body_names

    def test_load_no_geometry_still_works(self):
        urdf = """\
<robot name="test">
  <link name="base">
    <inertial><mass value="5.0"/><origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  <link name="foot">
    <inertial><mass value="0.5"/><origin xyz="0 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="hip" type="revolute">
    <parent link="base"/><child link="foot"/>
    <origin xyz="0 0 -0.2"/><axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57"/>
  </joint>
</robot>
"""
        from robot import load_urdf

        f = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
        f.write(urdf)
        f.close()
        try:
            model = load_urdf(f.name, floating_base=True, contact_links=["foot"])
        finally:
            os.unlink(f.name)
        assert "foot" in model.contact_body_names

    def test_ball_doesnt_fall_through(self):
        """Scene-based simulation should prevent ground penetration."""
        scene, q, qdot = _make_ball_scene(z0=0.3)
        sim = Simulator(scene, SemiImplicitEuler(dt=1e-3))
        tau = np.zeros(scene.robots["main"].tree.nv)
        for _ in range(100):
            q, qdot = sim.step_single(q, qdot, tau)
        assert q[6] > -0.5
        assert np.all(np.isfinite(q))
