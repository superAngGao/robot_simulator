"""
Tests for Simulator with Scene API.

Uses a single free-floating body (no contact, no self-collision) to keep
tests fast and dependency-free. Validates the new Scene-based Simulator.
"""

import numpy as np

from physics.integrator import RK4, SemiImplicitEuler
from physics.joint import FreeJoint
from physics.robot_tree import Body, RobotTree
from physics.spatial import SpatialInertia, SpatialTransform
from robot.model import RobotModel
from scene import Scene
from simulator import Simulator


def _make_scene(gravity: float = 9.81):
    """Single free-floating body, 1 kg, unit inertia, no contact."""
    tree = RobotTree(gravity=gravity)
    body = Body(
        name="base",
        index=0,
        joint=FreeJoint("root"),
        inertia=SpatialInertia(mass=1.0, inertia=np.eye(3), com=np.zeros(3)),
        X_tree=SpatialTransform.identity(),
        parent=-1,
    )
    tree.add_body(body)
    tree.finalize()
    model = RobotModel(tree=tree)
    return Scene.single_robot(model)


def _initial_state(scene, z0: float = 1.0):
    tree = scene.robots["main"].tree
    q, qdot = tree.default_state()
    q[3] = 1.0  # qw = 1 (identity quaternion)
    q[6] = z0  # pz
    return q, qdot


# ---------------------------------------------------------------------------


def test_step_returns_valid_state():
    """Single step: output shapes correct and all values finite."""
    scene = _make_scene()
    sim = Simulator(scene, SemiImplicitEuler(dt=2e-4))
    q, qdot = _initial_state(scene)
    tau = np.zeros(scene.robots["main"].tree.nv)

    q_new, qdot_new = sim.step_single(q, qdot, tau)

    assert q_new.shape == q.shape
    assert qdot_new.shape == qdot.shape
    assert np.all(np.isfinite(q_new))
    assert np.all(np.isfinite(qdot_new))


def test_passive_torques_applied():
    """Simulator must include passive_torques in the dynamics.

    For a free body with no revolute joints, passive_torques is zero, so
    we verify the result matches a manual call that explicitly adds zeros.
    """
    scene = _make_scene()
    integrator = SemiImplicitEuler(dt=2e-4)
    sim = Simulator(scene, integrator)
    tree = scene.robots["main"].tree

    q, qdot = _initial_state(scene)
    tau = np.zeros(tree.nv)

    q_sim, qdot_sim = sim.step_single(q.copy(), qdot.copy(), tau.copy())

    # Manual equivalent (no contacts, no collision → zero ext_forces)
    tau_passive = tree.passive_torques(q, qdot)
    tau_total = tau + tau_passive
    ext_forces = [np.zeros(6) for _ in range(tree.num_bodies)]
    q_manual, qdot_manual = integrator.step(tree, q, qdot, tau_total, ext_forces)

    np.testing.assert_allclose(q_sim, q_manual, atol=1e-14)
    np.testing.assert_allclose(qdot_sim, qdot_manual, atol=1e-14)


def test_matches_manual_loop():
    """Simulator over 100 steps must exactly match a manual step loop."""
    scene = _make_scene()
    integrator = SemiImplicitEuler(dt=2e-4)
    sim = Simulator(scene, integrator)
    tree = scene.robots["main"].tree

    q0, qdot0 = _initial_state(scene)
    tau = np.zeros(tree.nv)

    # Simulator path
    q_s, qdot_s = q0.copy(), qdot0.copy()
    for _ in range(100):
        q_s, qdot_s = sim.step_single(q_s, qdot_s, tau)

    # Manual path (no contacts → zero ext_forces)
    q_m, qdot_m = q0.copy(), qdot0.copy()
    for _ in range(100):
        tau_passive = tree.passive_torques(q_m, qdot_m)
        ext = [np.zeros(6) for _ in range(tree.num_bodies)]
        q_m, qdot_m = integrator.step(tree, q_m, qdot_m, tau + tau_passive, ext)

    np.testing.assert_allclose(q_s, q_m, atol=1e-12)
    np.testing.assert_allclose(qdot_s, qdot_m, atol=1e-12)


def test_swap_integrator():
    """Simulator with RK4 integrator must run without error."""
    scene = _make_scene()
    sim = Simulator(scene, RK4(dt=2e-4))
    q, qdot = _initial_state(scene)
    tau = np.zeros(scene.robots["main"].tree.nv)

    for _ in range(50):
        q, qdot = sim.step_single(q, qdot, tau)

    assert np.all(np.isfinite(q))
    assert np.all(np.isfinite(qdot))
