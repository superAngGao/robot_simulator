"""
Unit tests for SemiImplicitEuler and RK4 integrators.

Tests cover:
  - Invalid dt raises ValueError
  - Single step output shape and finiteness
  - Free-fall under gravity: position/velocity match analytic solution
  - Energy conservation: pendulum amplitude stable over many steps (semi-implicit)
  - RK4 is more accurate than semi-implicit Euler at same dt (free-fall)
  - SemiImplicitEuler raises RuntimeError on divergence (large dt)
  - simulate() helper returns correct array shapes
  - FreeJoint quaternion stays normalised after many steps
"""

import numpy as np
import pytest

from physics.integrator import RK4, SemiImplicitEuler, simulate
from physics.joint import Axis, FreeJoint, RevoluteJoint
from physics.robot_tree import Body, RobotTree
from physics.spatial import SpatialInertia, SpatialTransform

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_pendulum(gravity=9.81):
    """Single revolute-Y pendulum: mass=1, CoM at [0,0,-0.5], I=0.1*eye."""
    tree = RobotTree(gravity=gravity)
    tree.add_body(
        Body(
            name="link",
            index=0,
            joint=RevoluteJoint("j0", axis=Axis.Y),
            inertia=SpatialInertia(mass=1.0, inertia=0.1 * np.eye(3), com=np.array([0.0, 0.0, -0.5])),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()
    return tree


def _make_free_body(gravity=9.81):
    """Single free-floating body: mass=1, I=eye, CoM at origin."""
    tree = RobotTree(gravity=gravity)
    tree.add_body(
        Body(
            name="base",
            index=0,
            joint=FreeJoint("base"),
            inertia=SpatialInertia(mass=1.0, inertia=np.eye(3), com=np.zeros(3)),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()
    return tree


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_invalid_dt_raises():
    with pytest.raises(ValueError):
        SemiImplicitEuler(dt=0.0)
    with pytest.raises(ValueError):
        SemiImplicitEuler(dt=-0.01)
    with pytest.raises(ValueError):
        RK4(dt=0.0)


# ---------------------------------------------------------------------------
# Output shape and finiteness
# ---------------------------------------------------------------------------


def test_semi_implicit_output_shape():
    tree = _make_pendulum()
    q, qdot = tree.default_state()
    tau = np.zeros(tree.nv)
    integ = SemiImplicitEuler(dt=1e-3)
    q_new, qdot_new = integ.step(tree, q, qdot, tau)
    assert q_new.shape == (tree.nq,)
    assert qdot_new.shape == (tree.nv,)
    assert np.all(np.isfinite(q_new))
    assert np.all(np.isfinite(qdot_new))


def test_rk4_output_shape():
    tree = _make_pendulum()
    q, qdot = tree.default_state()
    tau = np.zeros(tree.nv)
    integ = RK4(dt=1e-3)
    q_new, qdot_new = integ.step(tree, q, qdot, tau)
    assert q_new.shape == (tree.nq,)
    assert qdot_new.shape == (tree.nv,)
    assert np.all(np.isfinite(q_new))
    assert np.all(np.isfinite(qdot_new))


# ---------------------------------------------------------------------------
# Free-fall accuracy: analytic comparison
# ---------------------------------------------------------------------------
# A free body dropped from rest falls under gravity.
# Analytic: z(t) = -0.5 * g * t^2,  vz(t) = -g * t
# FreeJoint q layout: [qw, qx, qy, qz, px, py, pz]  (quaternion first)
# v layout: [wx, wy, wz, vx, vy, vz]


def _free_fall_analytic(g, t):
    z = -0.5 * g * t**2
    vz = -g * t
    return z, vz


def test_semi_implicit_free_fall_accuracy():
    """Semi-implicit Euler free-fall: z error < 1e-3 m after 0.1 s."""
    g = 9.81
    tree = _make_free_body(gravity=g)
    q, qdot = tree.default_state()
    tau = np.zeros(tree.nv)
    dt = 1e-4
    integ = SemiImplicitEuler(dt=dt)

    t_end = 0.1
    n_steps = int(t_end / dt)
    for _ in range(n_steps):
        q, qdot = integ.step(tree, q, qdot, tau)

    z_sim = q[6]  # pz in FreeJoint q layout
    vz_sim = qdot[2]  # vz in FreeJoint v layout [vx,vy,vz,ωx,ωy,ωz]
    z_ref, vz_ref = _free_fall_analytic(g, t_end)

    assert abs(z_sim - z_ref) < 1e-3, f"z error too large: {abs(z_sim - z_ref):.4f}"
    assert abs(vz_sim - vz_ref) < 1e-2, f"vz error too large: {abs(vz_sim - vz_ref):.4f}"


def test_rk4_free_fall_accuracy():
    """RK4 free-fall: z error < 1e-6 m after 0.1 s (much tighter than semi-implicit)."""
    g = 9.81
    tree = _make_free_body(gravity=g)
    q, qdot = tree.default_state()
    tau = np.zeros(tree.nv)
    dt = 1e-3
    integ = RK4(dt=dt)

    t_end = 0.1
    n_steps = int(t_end / dt)
    for _ in range(n_steps):
        q, qdot = integ.step(tree, q, qdot, tau)

    z_sim = q[6]
    vz_sim = qdot[2]
    z_ref, vz_ref = _free_fall_analytic(g, t_end)

    assert abs(z_sim - z_ref) < 1e-6, f"RK4 z error too large: {abs(z_sim - z_ref):.2e}"
    assert abs(vz_sim - vz_ref) < 1e-5, f"RK4 vz error too large: {abs(vz_sim - vz_ref):.2e}"


def test_rk4_more_accurate_than_semi_implicit():
    """RK4 has smaller free-fall error than semi-implicit Euler at same dt."""
    g = 9.81
    dt = 1e-3
    t_end = 0.5
    n_steps = int(t_end / dt)
    errors = {}
    for name, cls in [("semi", SemiImplicitEuler), ("rk4", RK4)]:
        tree = _make_free_body(gravity=g)
        tau = np.zeros(tree.nv)
        q, qdot = tree.default_state()
        integ = cls(dt=dt)
        for _ in range(n_steps):
            q, qdot = integ.step(tree, q, qdot, tau)
        z_ref, _ = _free_fall_analytic(g, t_end)
        errors[name] = abs(q[6] - z_ref)

    assert errors["rk4"] < errors["semi"], (
        f"RK4 should be more accurate: rk4={errors['rk4']:.2e}, semi={errors['semi']:.2e}"
    )


# ---------------------------------------------------------------------------
# Energy conservation: pendulum (semi-implicit Euler)
# ---------------------------------------------------------------------------


def test_pendulum_energy_bounded_semi_implicit():
    """Semi-implicit Euler pendulum: total energy stays bounded over 1000 steps."""
    g = 9.81
    L = 0.5  # CoM distance
    m = 1.0
    I_com = 0.1
    I_pivot = I_com + m * L**2  # parallel axis

    tree = _make_pendulum(gravity=g)
    q = np.array([0.5])  # initial angle
    qdot = np.zeros(1)
    tau = np.zeros(1)
    dt = 1e-3
    integ = SemiImplicitEuler(dt=dt)

    def energy(q_, qdot_):
        theta = q_[0]
        KE = 0.5 * I_pivot * qdot_[0] ** 2
        PE = -m * g * L * np.cos(theta)
        return KE + PE

    E0 = energy(q, qdot)
    E_max = E0

    for _ in range(1000):
        q, qdot = integ.step(tree, q, qdot, tau)
        E_max = max(E_max, energy(q, qdot))

    # Semi-implicit Euler is symplectic: energy should not grow unboundedly
    assert E_max < E0 + 0.01, f"Energy grew too much: E0={E0:.4f}, E_max={E_max:.4f}"


# ---------------------------------------------------------------------------
# Divergence detection
# ---------------------------------------------------------------------------


def test_semi_implicit_raises_on_nan_state(monkeypatch):
    """If ABA returns NaN, SemiImplicitEuler raises RuntimeError."""
    tree = _make_pendulum()
    q = np.array([0.5])
    qdot = np.zeros(1)
    tau = np.zeros(1)
    integ = SemiImplicitEuler(dt=1e-3)

    # Patch ABA to return NaN
    monkeypatch.setattr(tree, "aba", lambda *a, **kw: np.array([np.nan]))

    with pytest.raises(RuntimeError):
        integ.step(tree, q, qdot, tau)


# ---------------------------------------------------------------------------
# simulate() helper
# ---------------------------------------------------------------------------


def test_simulate_output_shapes():
    """simulate() returns arrays with correct shapes."""
    tree = _make_pendulum()
    q0, qdot0 = tree.default_state()
    dt = 1e-3
    duration = 0.1
    n_steps = int(duration / dt)

    times, qs, qdots = simulate(
        tree,
        q0,
        qdot0,
        controller_fn=lambda t, q, v: np.zeros(tree.nv),
        dt=dt,
        duration=duration,
    )

    assert times.shape == (n_steps,)
    assert qs.shape == (n_steps, tree.nq)
    assert qdots.shape == (n_steps, tree.nv)
    assert np.all(np.isfinite(qs))
    assert np.all(np.isfinite(qdots))


def test_simulate_rk4_option():
    """simulate() with integrator='rk4' runs without error."""
    tree = _make_pendulum()
    q0, qdot0 = tree.default_state()
    times, qs, qdots = simulate(
        tree,
        q0,
        qdot0,
        controller_fn=lambda t, q, v: np.zeros(tree.nv),
        dt=1e-3,
        duration=0.05,
        integrator="rk4",
    )
    assert np.all(np.isfinite(qs))


# ---------------------------------------------------------------------------
# FreeJoint quaternion normalisation
# ---------------------------------------------------------------------------


def test_freejoint_quaternion_stays_normalised():
    """After 500 steps, FreeJoint quaternion norm stays within 1e-6 of 1."""
    tree = _make_free_body()
    q, qdot = tree.default_state()
    # Give it some angular velocity ([vx,vy,vz,ωx,ωy,ωz])
    qdot[3] = 1.0  # wx
    qdot[4] = 0.5  # wy
    tau = np.zeros(tree.nv)
    integ = SemiImplicitEuler(dt=1e-3)

    for _ in range(500):
        q, qdot = integ.step(tree, q, qdot, tau)

    quat = q[:4]  # [qw, qx, qy, qz]
    norm = np.linalg.norm(quat)
    assert abs(norm - 1.0) < 1e-6, f"Quaternion norm drifted: {norm:.8f}"
