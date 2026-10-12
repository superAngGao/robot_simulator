"""Reproduce the bounded audit: python -m tests.validation.rigid_body.audit_physics.

Writes raw energy curves and measured errors; these are numerical diagnostics,
not real-world calibration. CUDA execution is explicit via --gpu.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from physics.geometry import BoxShape, CylinderShape
from physics.gjk_epa import gjk_epa_query
from physics.merged_model import merge_models
from physics.solvers.admm_qp import ADMMQPSolver
from physics.solvers.pgs_split_impulse import PGSSplitImpulseSolver
from tests.unit.collision.test_convex_margin import _box_hull, _pose

from .models import DT, build_quadruped
from .physics_metrics import (
    energy_errors,
    first_freefall_penetration,
    pendulum_energy_history,
    quadruped_admm_equilibrium_depth,
)
from .test_contact_vs_mujoco import LEG_LENGTH, _run_cpu, _run_mujoco_quad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--sweep", action="store_true", help="also halve/double dt and double iterations")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []

    def emit(kind, **values):
        row = dict(kind=kind, **values)
        rows.append(row)
        print(json.dumps(row), flush=True)
        (args.output / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")

    curves = []
    for dt in [4e-4, 2e-4, 1e-4]:
        times, energy, reference = pendulum_energy_history(dt)
        metrics = energy_errors(energy, reference)
        emit("energy", dt=dt, **metrics)
        # Keep every integration state in the standalone curve artifact.
        np.savetxt(
            args.output / f"energy-{dt:g}.csv",
            np.c_[times, energy],
            delimiter=",",
            header="time_s,energy_j",
            comments="",
        )
        curves.append((dt, times, (energy - energy[0]) / metrics["scale_j"]))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for dt, times, error in curves:
        ax.plot(times, 100 * error, label=f"dt={1000 * dt:g} ms")
    ax.set(
        xlabel="Time (s)",
        ylabel="Energy error / initial available energy (%)",
        title="Double pendulum: semi-implicit Euler, CPU engine",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output / "energy-convergence.png", dpi=160)
    plt.close(fig)

    model, xml = build_quadruped(contact=True)
    q, velocity = model.tree.default_state()
    q[6] = 0.45
    configurations = [("admm", DT, 50), ("pgs", DT, 60)]
    if args.sweep:
        configurations += [("admm", DT, 100), ("pgs", DT, 120)]
        configurations += [
            (name, dt, iterations) for name, iterations in [("admm", 50), ("pgs", 60)] for dt in [4e-4, 1e-4]
        ]
    for name, dt, iterations in configurations:
        solver = (
            ADMMQPSolver(max_iter=iterations)
            if name == "admm"
            else PGSSplitImpulseSolver(max_iter=iterations, erp=0.8, slop=0.005)
        )
        z, vz, _, _ = _run_cpu(model, q, velocity, round(1 / dt), solver=solver, dt=dt)
        tail = round(0.1 / dt)
        expected = (
            LEG_LENGTH - quadruped_admm_equilibrium_depth(solver, dt)
            if name == "admm"
            else first_freefall_penetration(q[6], LEG_LENGTH, dt)[1]
        )
        emit(
            "settling",
            solver=name,
            dt=dt,
            iterations=iterations,
            mean_z=float(z[-tail:].mean()),
            reference_z=expected,
            reference_kind="coupled_soft_equilibrium" if name == "admm" else "first_discrete_penetration",
            error_m=float(z[-tail:].mean() - expected),
            tail_range_m=float(np.ptp(z[-tail:])),
            max_abs_vz=float(abs(vz[-tail:]).max()),
        )
    z, vz, _, _ = _run_mujoco_quad(xml, 0.45, 5000)
    emit(
        "settling",
        solver="mujoco",
        dt=DT,
        mean_z=float(z[-500:].mean()),
        max_abs_vz=float(abs(vz[-500:]).max()),
    )
    for shape in [CylinderShape(0.05, 0.1), _box_hull(0.05)]:
        for pen in [1e-5, 1e-4, 1e-3, 3e-3]:
            hit = gjk_epa_query(
                BoxShape((0.1, 0.1, 0.1)), _pose([0, 0, 0]), shape, _pose([0, 0, 0.1 - pen]), margin=0.001
            )
            emit(
                "unresolved_geometry",
                pair=type(shape).__name__,
                penetration_m=pen,
                detected=hit is not None,
                depth_m=None if hit is None else hit.depth,
            )
    if args.gpu:
        from .quadruped.test_gpu_vs_cpu import _run_gpu

        merged = merge_models({"quad": model})
        qs, vs = _run_gpu(merged, q, velocity, 5000)
        emit(
            "settling",
            solver="gpu_admm",
            dt=DT,
            mean_z=float(qs[-500:, 6].mean()),
            max_abs_vz=float(abs(vs[-500:, 2]).max()),
        )


if __name__ == "__main__":
    main()
