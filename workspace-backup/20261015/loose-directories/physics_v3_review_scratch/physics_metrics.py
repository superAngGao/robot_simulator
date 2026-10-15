"""Small, fixture-specific numerical oracles for the physics audit."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from physics.cpu_engine import CpuEngine
from physics.merged_model import merge_models

from .models import G, build_double_pendulum, build_quadruped


def total_energy(tree, q, velocity):
    transforms = tree.forward_kinematics(q)
    potential = sum(
        body.inertia.mass * G
        * (transforms[body.index].R @ body.inertia.com + transforms[body.index].r)[2]
        for body in tree.bodies
    )
    return float(0.5 * velocity @ tree.crba(q) @ velocity + potential)


def pendulum_energy_history(dt, duration=2.0):
    """Run the actual CPU entrypoint; include initial AND final states."""
    model, _ = build_double_pendulum()
    merged = merge_models({"pendulum": model})
    engine = CpuEngine(merged, dt=dt)
    q = np.array([np.pi / 2, np.pi / 4])
    velocity = np.zeros(2)
    count = round(duration / dt)
    energies = np.empty(count + 1)
    for i in range(count + 1):
        energies[i] = total_energy(merged.tree, q, velocity)
        if i != count:
            out = engine.step(q, velocity, np.zeros(2))
            q, velocity = out.q_new, out.qdot_new
    reference = total_energy(merged.tree, np.zeros(2), np.zeros(2))
    return np.arange(count + 1) * dt, energies, reference


def energy_errors(energies, reference):
    """Normalize by available energy, independent of potential datum."""
    scale = energies[0] - reference
    if scale <= 0:
        raise ValueError("fixture requires positive energy above equilibrium")
    delta = energies - energies[0]
    return {
        "initial_j": float(energies[0]),
        "scale_j": float(scale),
        "endpoint_j": float(delta[-1]),
        "max_j": float(np.max(np.abs(delta))),
        "max_relative": float(np.max(np.abs(delta)) / scale),
    }


def quadruped_admm_equilibrium_depth(solver, dt):
    """Four coupled normal rows at the symmetric straight-leg equilibrium.

    This is a model-consistency oracle, not a second dynamics implementation.
    Normal geometric rows are formed independently of contact_jacobian.
    Symmetry gives f_n = M*g/4 and A*f = g. Thus R_ii*f_n = k*d*depth.
    """
    model, _ = build_quadruped(contact=True)
    tree = model.tree
    q, _ = tree.default_state()
    transforms = tree.forward_kinematics(q)
    feet = [b for b in tree.bodies if b.name.endswith("foot")]
    assert len(feet) == 4
    rows = np.zeros((4, tree.nv))
    for i, foot in enumerate(feet):
        x, y, _ = transforms[foot.index].r
        rows[i, :6] = [0, 0, 1, y, -x, 0]
    inertia = tree.crba(q)
    a = rows @ np.linalg.solve(inertia, rows.T)
    load = sum(b.inertia.mass for b in tree.bodies) * G / 4
    np.testing.assert_allclose(a @ np.full(4, load), G, atol=1e-10, rtol=0)
    d0, dmax, width, midpoint, power = solver.solimp
    timeconst, ratio = solver.solref
    k = 1 / (dmax * max(timeconst, 2 * dt) * ratio) ** 2

    def residual(depth):
        x = min(depth / width, 1.0)
        y = x**power / midpoint**(power - 1) if x <= midpoint else (
            1 - (1 - x)**power / (1 - midpoint)**(power - 1)
        )
        d = d0 + (dmax - d0) * y
        return (1 - d) / d * a[0, 0] * load - k * d * depth

    return brentq(residual, 0, 0.02, xtol=1e-14)
