"""
Incline contact physics validation — plots velocity and forces vs analytical.

Generates two figures:
  1. Box sliding down from rest on a 30-degree incline (mu=0.3)
  2. Box going up then sliding back down (mu=0.2, v0=1.0 m/s)

Each figure has 3 subplots: velocity, normal force, friction force.

Usage:
    MPLBACKEND=Agg python tests/integration/contact_pipeline/plot_incline_validation.py

Output:
    tests/integration/contact_pipeline/incline_slide_down.png
    tests/integration/contact_pipeline/incline_up_then_down.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics.geometry import BodyCollisionGeometry, BoxShape, ShapeInstance
from physics.integrator import SemiImplicitEuler
from physics.joint import FreeJoint
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from physics.terrain import HalfSpaceTerrain
from robot.model import RobotModel
from scene import Scene
from simulator import Simulator

OUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

G = 9.81


def _incline_normal(theta: float) -> np.ndarray:
    return np.array([-np.sin(theta), 0.0, np.cos(theta)])


def _along_incline(theta: float) -> np.ndarray:
    """Unit vector pointing up the slope."""
    return np.array([np.cos(theta), 0.0, np.sin(theta)])


def _box_on_incline(
    theta: float,
    mu: float,
    mass: float = 1.0,
    half: float = 0.05,
    dt: float = 5e-4,
    v0_along: float = 0.0,
):
    """Build box on incline. Returns (sim, q0, qdot0, tree)."""
    tree = RobotTreeNumpy(gravity=G)
    side = 2 * half
    I_box = mass * side**2 / 6.0
    tree.add_body(
        Body(
            name="box",
            index=0,
            joint=FreeJoint("root"),
            inertia=SpatialInertia(mass=mass, inertia=np.eye(3) * I_box, com=np.zeros(3)),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()

    model = RobotModel(
        tree=tree,
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(BoxShape((side, side, side)))])],
        contact_body_names=["box"],
    )

    normal = _incline_normal(theta)
    terrain = HalfSpaceTerrain(normal=normal, point=np.zeros(3), mu=mu)
    scene = Scene.single_robot(model, terrain=terrain)

    q, qdot = tree.default_state()
    q[0] = 1.0  # identity quaternion
    # For an axis-aligned box on a tilted plane, the distance from box
    # center to the deepest corner along the plane normal is:
    #   h_proj = |nx|*half_x + |ny|*half_y + |nz|*half_z
    h_proj = np.sum(np.abs(normal) * half)
    pos = normal * (h_proj + 1e-4)
    q[4], q[5], q[6] = pos

    if v0_along != 0.0:
        v = v0_along * _along_incline(theta)
        qdot[0], qdot[1], qdot[2] = v

    sim = Simulator(scene, SemiImplicitEuler(dt=dt))
    return sim, q, qdot, mass


def _simulate(sim, q, qdot, n_steps, mass, theta):
    """Run simulation and record time histories."""
    tau = np.zeros(6)
    normal = _incline_normal(theta)
    along = _along_incline(theta)

    times = []
    vel_along = []  # velocity component along incline (+ = up)
    acc_along = []  # acceleration along incline
    f_normal_list = []  # normal contact force magnitude
    f_friction_list = []  # friction force magnitude (signed: + opposes motion)

    dt = sim._pipeline.dt
    for i in range(n_steps):
        q, qdot = sim.step_single(q, qdot, tau)
        t = (i + 1) * dt
        times.append(t)

        # Velocity along incline
        v_along = float(np.dot(qdot[:3], along))
        vel_along.append(v_along)

        # Extract contact force from ForceState
        fs = sim.last_force_states.get("main")
        if fs is not None:
            # For FreeJoint: qacc[:3] is world-frame linear acceleration
            # Contact acceleration = qacc - qacc_smooth
            a_contact = fs.qacc[:3] - fs.qacc_smooth[:3]
            f_contact = mass * a_contact  # world-frame contact force

            # Decompose into normal and tangential
            fn = float(np.dot(f_contact, normal))
            ft_vec = f_contact - fn * normal
            # Sign convention: positive friction opposes downhill motion
            ft_signed = float(np.dot(ft_vec, along))

            f_normal_list.append(fn)
            f_friction_list.append(ft_signed)

            # Net acceleration along incline
            a_total = float(np.dot(fs.qacc[:3], along))
            acc_along.append(a_total)
        else:
            f_normal_list.append(0.0)
            f_friction_list.append(0.0)
            acc_along.append(0.0)

    return (
        np.array(times),
        np.array(vel_along),
        np.array(acc_along),
        np.array(f_normal_list),
        np.array(f_friction_list),
        q,
        qdot,
    )


# ---------------------------------------------------------------------------
# Scenario 1: Slide down from rest
# ---------------------------------------------------------------------------


def plot_slide_down():
    theta = np.radians(30)
    mu = 0.3
    mass = 1.0
    dt = 5e-4
    n_steps = 1000
    sim, q, qdot, mass = _box_on_incline(theta, mu, mass=mass, dt=dt)

    times, vel, acc, fn, ft, _, _ = _simulate(sim, q, qdot, n_steps, mass, theta)

    # Analytical expectations
    a_analytical = -G * (np.sin(theta) - mu * np.cos(theta))  # negative = downhill
    v_analytical = a_analytical * times
    fn_analytical = mass * G * np.cos(theta)
    ft_analytical = mu * mass * G * np.cos(theta)  # friction points uphill

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(
        f"Box sliding down incline: $\\theta$={np.degrees(theta):.0f}deg, $\\mu$={mu}, m={mass}kg",
        fontsize=14,
    )

    # Velocity
    ax = axes[0]
    ax.plot(times, vel, "b-", linewidth=1.5, label="Simulated")
    ax.plot(times, v_analytical, "r--", linewidth=1.5, label="Analytical")
    ax.set_ylabel("Velocity along slope [m/s]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Velocity (+ = uphill)")

    # Normal force
    ax = axes[1]
    ax.plot(times, fn, "b-", linewidth=1.5, label="Simulated")
    ax.axhline(fn_analytical, color="r", linestyle="--", linewidth=1.5, label="Analytical")
    ax.set_ylabel("Normal force [N]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Contact normal force")

    # Friction force
    ax = axes[2]
    ax.plot(times, ft, "b-", linewidth=1.5, label="Simulated")
    ax.axhline(ft_analytical, color="r", linestyle="--", linewidth=1.5, label="Analytical")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Friction force along slope [N]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Friction force (+ = uphill)")

    fig.tight_layout()
    out = OUT_DIR / "incline_slide_down.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)

    # Print summary
    t_end = times[-1]
    print(f"\n--- Slide-down summary (t={t_end:.3f}s) ---")
    print(f"  v_sim  = {vel[-1]:.4f} m/s,  v_ana = {v_analytical[-1]:.4f} m/s")
    print(f"  Fn_sim = {np.mean(fn[len(fn) // 2 :]):.3f} N,  Fn_ana = {fn_analytical:.3f} N")
    print(f"  Ft_sim = {np.mean(ft[len(ft) // 2 :]):.3f} N,  Ft_ana = {ft_analytical:.3f} N")


# ---------------------------------------------------------------------------
# Scenario 2: Up then down
# ---------------------------------------------------------------------------


def plot_up_then_down():
    theta = np.radians(30)
    mu = 0.2
    mass = 1.0
    v0 = 1.0
    dt = 5e-4
    n_steps = 2000
    sim, q, qdot, mass = _box_on_incline(theta, mu, mass=mass, dt=dt, v0_along=v0)

    times, vel, acc, fn, ft, _, _ = _simulate(sim, q, qdot, n_steps, mass, theta)

    # Analytical
    a_up = -(G * np.sin(theta) + mu * G * np.cos(theta))
    a_down = -(G * np.sin(theta) - mu * G * np.cos(theta))
    t_stop = v0 / abs(a_up)

    v_analytical = np.where(
        times <= t_stop,
        v0 + a_up * times,
        a_down * (times - t_stop),
    )

    fn_analytical = mass * G * np.cos(theta)
    # Friction: uphill when going up (opposing), downhill when going down (opposing)
    ft_analytical = np.where(
        times <= t_stop,
        -mu * mass * G * np.cos(theta),  # friction points downhill when going up
        mu * mass * G * np.cos(theta),  # friction points uphill when going down
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(
        f"Box up-then-down: $\\theta$={np.degrees(theta):.0f}deg, $\\mu$={mu}, v0={v0}m/s",
        fontsize=14,
    )

    # Velocity
    ax = axes[0]
    ax.plot(times, vel, "b-", linewidth=1.5, label="Simulated")
    ax.plot(times, v_analytical, "r--", linewidth=1.5, label="Analytical")
    ax.axvline(t_stop, color="gray", linestyle=":", alpha=0.5, label=f"t_stop={t_stop:.3f}s")
    ax.set_ylabel("Velocity along slope [m/s]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Velocity (+ = uphill)")

    # Normal force
    ax = axes[1]
    ax.plot(times, fn, "b-", linewidth=1.5, label="Simulated")
    ax.axhline(fn_analytical, color="r", linestyle="--", linewidth=1.5, label="Analytical")
    ax.axvline(t_stop, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("Normal force [N]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Contact normal force")

    # Friction force
    ax = axes[2]
    ax.plot(times, ft, "b-", linewidth=1.5, label="Simulated")
    ax.plot(times, ft_analytical, "r--", linewidth=1.5, label="Analytical")
    ax.axvline(t_stop, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Friction force along slope [N]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Friction force (+ = uphill, opposing downhill motion)")

    fig.tight_layout()
    out = OUT_DIR / "incline_up_then_down.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)

    # Print summary
    print("\n--- Up-then-down summary ---")
    print(f"  t_stop_analytical = {t_stop:.4f}s")
    print(f"  a_up = {a_up:.3f} m/s^2,  a_down = {a_down:.3f} m/s^2")
    print(f"  v_end_sim = {vel[-1]:.4f},  v_end_ana = {v_analytical[-1]:.4f}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plot_slide_down()
    plot_up_then_down()
