"""Real-engine normal-contact convergence, with the actual GPU Q and RHS."""
import json
import sys
import numpy as np
sys.path.insert(0, "/data2/shared/ang.gao/robot_sim_physics_correctness")
from physics.geometry import SphereShape
from physics.gpu_engine import GpuEngine
from tests.integration.test_cpu_engine_shapes import _make_single_body_engine, _sphere_inertia

for mass in [.1, 1., 50., 1000.]:
    for iterations in [30, 300]:
        merged, _ = _make_single_body_engine(SphereShape(.1), mass, _sphere_inertia(mass, .1))
        gpu = GpuEngine(merged, num_envs=1, dt=.0002, solver="admm")
        gpu._admm_iters = iterations
        gpu._admm_warmstart = False
        q, _ = merged.tree.default_state()
        q[6] = .1-1e-5
        gpu.reset(q)
        gpu.step(dt=.0002)
        active = np.flatnonzero(gpu._contact_active.numpy()[0])
        assert len(active) == 1
        row = int(active[0])*3
        sc = gpu._solver_scratch
        w = float(sc.W.numpy()[0, row, row])
        r = float(sc.admm_R_diag.numpy()[0, row])
        rhs = float(sc.admm_rhs_const.numpy()[0, row])
        impulse = float(sc.lambdas.numpy()[0, row])
        reference = rhs/(w+r)
        print(json.dumps(dict(mass=mass, iterations=iterations, W=w, R=r, rhs=rhs,
                              impulse=impulse, scalar_model_reference=reference,
                              relative_error=abs(impulse-reference)/reference)), flush=True)
