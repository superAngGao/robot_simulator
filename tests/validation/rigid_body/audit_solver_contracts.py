"""Minimal design/implementation counterexamples, including real CUDA kernels.

Run with CUDA_VISIBLE_DEVICES set to an available GPU. Report discrepancies;
this diagnostic deliberately does not mark unresolved model contracts passed.
"""

import json

import numpy as np

from physics.geometry import SphereShape
from physics.solvers.admm_qp import ADMMQPSolver
from physics.solvers.pgs_solver import ContactConstraint
from tests.integration.test_cpu_engine_shapes import _make_single_body_engine, _sphere_inertia


def emit(kind, **values):
    print(json.dumps(dict(kind=kind, **values)), flush=True)


def main():
    merged, _ = _make_single_body_engine(SphereShape(0.1), 1.0, _sphere_inertia(1.0, 0.1))
    tree = merged.tree
    q, v = tree.default_state()
    contact = ContactConstraint(
        0, -1, np.zeros(3), np.array([0.0, 0.0, 1.0]), np.zeros(3), np.zeros(3), 0.0, 1.0
    )
    # J directions are z,y,-x at this identity pose; A=I for a center contact.
    # Default impedance d=.9 yields Q=(10/9)I, rhs=[1,2,0].
    solver = ADMMQPSolver(max_iter=1000, tolerance=1e-12)
    force, _ = solver.solve([contact], tree, q, v, np.array([0.0, -2.0, -1.0, 0.0, 0.0, 0.0]), 0.001)
    expected = np.array([1.35, 1.35, 0.0])
    rhs = np.array([1.0, 2.0, 0.0])

    def objective(f):
        return float(0.5 / 0.9 * f @ f - rhs @ f)

    emit(
        "admm_stated_cone_qp",
        returned=force.tolist(),
        exact_minimizer=expected.tolist(),
        returned_objective=objective(force),
        exact_objective=objective(expected),
    )
    z = np.array([1.0, 2.0, 0.0])
    emit(
        "cone_projection",
        returned=solver._project_all(z, [contact], [0]).tolist(),
        euclidean_projection=[1.5, 1.5, 0.0],
    )

    import warp as wp

    from physics.backends.warp.colored_pgs_kernels import batched_colored_pgs_all_iters
    from physics.backends.warp.solver_kernels import batched_jacobi_pgs_step
    from physics.gpu_engine import GpuEngine

    device = "cuda:0"
    wp.init()
    w = wp.array(np.diag([1.0, 3.5, 3.5])[None].astype(np.float32), device=device)
    diag = wp.array(np.array([[1.0, 3.5 / 0.99, 3.5 / 0.99]], dtype=np.float32), device=device)
    vf = wp.array(np.array([[-2.0, 2.0, 0.0]], dtype=np.float32), device=device)
    active = wp.array(np.ones((1, 1), dtype=np.int32), device=device)
    old = wp.zeros((1, 3), dtype=wp.float32, device=device)
    new = wp.zeros_like(old)
    for _ in range(200):
        wp.launch(
            batched_jacobi_pgs_step,
            dim=1,
            device=device,
            inputs=[w, diag, vf, old, active, 1.0, 0.6, 1, 3],
            outputs=[new],
        )
        wp.copy(old, new)
    emit(
        "gpu_jacobi_compliance",
        impulse=new.numpy()[0].tolist(),
        soft_model_impulse=[2.0, -2 * 0.99 / 3.5, 0.0],
        hard_model_impulse=[2.0, -2 / 3.5, 0.0],
    )
    color = wp.array(np.zeros((1, 1), dtype=np.int32), device=device)
    count = wp.array(np.ones(1, dtype=np.int32), device=device)
    new.zero_()
    wp.launch(
        batched_colored_pgs_all_iters,
        dim=1,
        device=device,
        inputs=[w, diag, vf, new, active, color, count, 1.0, 200, 1, 3],
    )
    emit(
        "gpu_colored_compliance",
        impulse=new.numpy()[0].tolist(),
        soft_model_impulse=[2.0, -2 * 0.99 / 3.5, 0.0],
    )

    engine = GpuEngine(merged, num_envs=1, solver="admm", dt=0.001)
    q[6] = 0.099
    engine.reset(q)
    engine.step()
    scratch = engine._solver_scratch
    physical = np.diag(scratch.W.numpy()[0])[:3]
    r = scratch.admm_R_diag.numpy()[0, :3]
    emit(
        "gpu_admm_regularization",
        physical_W_diag=physical.tolist(),
        regularization=r.tolist(),
        R_over_A=(r / physical).tolist(),
        cpu_model_ratio=(1 - 0.95) / 0.95,
    )
    from physics.cpu_engine import CpuEngine
    from physics.terrain import FlatTerrain

    merged.terrain = FlatTerrain(z=0.5)
    cpu = CpuEngine(merged)
    elevated = GpuEngine(merged, num_envs=1, solver="admm", dt=0.001)
    q[6] = 0.599
    cpu.step(q, np.zeros(merged.nv), np.zeros(merged.nv))
    elevated.reset(q)
    elevated.step()
    emit(
        "flat_terrain_height",
        requested_z=0.5,
        gpu_static_z=elevated._static.contact_ground_z,
        cpu_contacts=len(cpu.query_contacts()),
        gpu_contacts=int(elevated._contact_count.numpy()[0]),
    )


if __name__ == "__main__":
    main()
