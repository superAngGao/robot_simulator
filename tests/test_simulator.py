"""
Tests for Simulator (Layer 2 orchestrator).

Uses a single free-floating body (no contact, no self-collision) to keep
tests fast and dependency-free. Pattern mirrors test_free_fall.py.
"""

import numpy as np

from physics.collision import NullSelfCollision
from physics.contact import NullContactModel
from physics.integrator import RK4, SemiImplicitEuler
from physics.joint import FreeJoint
from physics.robot_tree import Body, RobotTree
from physics.spatial import SpatialInertia, SpatialTransform
from robot.model import RobotModel
from simulator import Simulator


def _make_model(gravity: float = 9.81) -> RobotModel:
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
    return RobotModel(
        tree=tree,
        contact_model=NullContactModel(),
        self_collision=NullSelfCollision(),
    )


def _initial_state(model: RobotModel, z0: float = 1.0):
    tree = model.tree
    q, qdot = tree.default_state()
    q[3] = 1.0  # qw = 1 (identity quaternion)
    q[6] = z0  # pz
    return q, qdot


# ---------------------------------------------------------------------------


def test_step_returns_valid_state():
    """Single step: output shapes correct and all values finite."""
    model = _make_model()
    sim = Simulator(model, SemiImplicitEuler(dt=2e-4))
    q, qdot = _initial_state(model)
    tau = np.zeros(model.tree.nv)

    q_new, qdot_new = sim.step(q, qdot, tau)

    assert q_new.shape == q.shape
    assert qdot_new.shape == qdot.shape
    assert np.all(np.isfinite(q_new))
    assert np.all(np.isfinite(qdot_new))


def test_passive_torques_applied():
    """Simulator must include passive_torques in the dynamics.

    For a free body with no revolute joints, passive_torques is zero, so
    we verify the result matches a manual call that explicitly adds zeros.
    The key invariant: sim.step(q, qdot, tau) == integrator.step(tree, q, qdot,
    tau + passive_torques, ext_forces).
    """
    model = _make_model()
    integrator = SemiImplicitEuler(dt=2e-4)
    sim = Simulator(model, integrator)
    tree = model.tree

    q, qdot = _initial_state(model)
    tau = np.zeros(tree.nv)

    q_sim, qdot_sim = sim.step(q.copy(), qdot.copy(), tau.copy())

    # Manual equivalent
    tau_passive = tree.passive_torques(q, qdot)
    tau_total = tau + tau_passive
    X_world = tree.forward_kinematics(q)
    v_bodies = tree.body_velocities(q, qdot)
    contact_forces = model.contact_model.compute_forces(X_world, v_bodies, tree.num_bodies)
    sc_forces = model.self_collision.compute_forces(X_world, v_bodies, tree.num_bodies)
    ext_forces = [cf + scf for cf, scf in zip(contact_forces, sc_forces)]
    q_manual, qdot_manual = integrator.step(tree, q, qdot, tau_total, ext_forces)

    np.testing.assert_allclose(q_sim, q_manual, atol=1e-14)
    np.testing.assert_allclose(qdot_sim, qdot_manual, atol=1e-14)


def test_matches_manual_loop():
    """Simulator over 100 steps must exactly match a manual step loop."""
    model = _make_model()
    integrator = SemiImplicitEuler(dt=2e-4)
    sim = Simulator(model, integrator)
    tree = model.tree

    q0, qdot0 = _initial_state(model)
    tau = np.zeros(tree.nv)

    # Simulator path
    q_s, qdot_s = q0.copy(), qdot0.copy()
    for _ in range(100):
        q_s, qdot_s = sim.step(q_s, qdot_s, tau)

    # Manual path
    q_m, qdot_m = q0.copy(), qdot0.copy()
    for _ in range(100):
        tau_passive = tree.passive_torques(q_m, qdot_m)
        X_world = tree.forward_kinematics(q_m)
        v_bodies = tree.body_velocities(q_m, qdot_m)
        cf = model.contact_model.compute_forces(X_world, v_bodies, tree.num_bodies)
        scf = model.self_collision.compute_forces(X_world, v_bodies, tree.num_bodies)
        ext = [a + b for a, b in zip(cf, scf)]
        q_m, qdot_m = integrator.step(tree, q_m, qdot_m, tau + tau_passive, ext)

    np.testing.assert_allclose(q_s, q_m, atol=1e-12)
    np.testing.assert_allclose(qdot_s, qdot_m, atol=1e-12)


def test_swap_integrator():
    """Simulator with RK4 integrator must run without error and return finite state."""
    model = _make_model()
    sim = Simulator(model, RK4(dt=2e-4))
    q, qdot = _initial_state(model)
    tau = np.zeros(model.tree.nv)

    for _ in range(50):
        q, qdot = sim.step(q, qdot, tau)

    assert np.all(np.isfinite(q))
    assert np.all(np.isfinite(qdot))
