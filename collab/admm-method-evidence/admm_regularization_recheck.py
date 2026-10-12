"""Read-only investigation of per-row R, rho, and the declared ADMM QP."""
import json
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, "/data2/shared/ang.gao/robot_sim_physics_correctness")
from physics.geometry import SphereShape
from physics.solvers.admm_qp import ADMMQPSolver
from physics.solvers.pgs_solver import ContactConstraint
from tests.integration.test_cpu_engine_shapes import _make_single_body_engine, _sphere_inertia
from tests.validation.rigid_body.physics_metrics import quadruped_admm_equilibrium_depth


class Capture(ADMMQPSolver):
    def _factor_ARrho(self, ar, rho, n):
        self.captured_q = ar.copy()
        return super()._factor_ARrho(ar, rho, n)


def run(mass, offset, d, rho, adaptive, tangential_rhs):
    radius = 0.1
    merged, _ = _make_single_body_engine(SphereShape(radius), mass, _sphere_inertia(mass, radius))
    tree = merged.tree
    q, v = tree.default_state()
    c = ContactConstraint(0, -1, np.array([0., 0., -offset]), np.array([0., 0., 1.]),
                          np.zeros(3), np.zeros(3), 0., 1.)
    s = Capture(max_iter=10000, tolerance=1e-11, rho=rho, warmstart=False,
                adaptive_rho=adaptive, solimp=(d, d, .001, .5, 2.))
    rhs = np.array([1., tangential_rhs, 0.])
    f, j = s.solve([c], tree, q, v, np.array([0., -tangential_rhs, -1., 0., 0., 0.]), .001)
    a = j @ np.linalg.solve(tree.crba(q), j.T)
    intended_r = (1-d)/d * np.diag(a)
    np.testing.assert_allclose(s.captured_q, a + np.diag(intended_r), atol=1e-13, rtol=1e-13)
    if tangential_rhs == 0:
        exact = np.array([mass*d, 0., 0.])
        np.testing.assert_allclose(f, exact, atol=1e-9, rtol=0)
    else:
        # Both tangential fixtures have their exact cone optimum on f_n=f_t>0.
        n = rhs[:2].sum() / (s.captured_q[0, 0] + s.captured_q[1, 1])
        exact = np.array([n, n, 0.])
    objective = lambda force: float(.5 * force @ s.captured_q @ force - rhs @ force)
    return dict(mass=mass, offset=offset, d=d, rho=rho, adaptive=adaptive,
                physical_A_diag=np.diag(a).tolist(), physical_R_diag=intended_r.tolist(),
                returned=f.tolist(), exact=exact.tolist(),
                objective_gap=objective(f)-objective(exact),
                primal_residual=float(np.linalg.norm(s._prev_f-s._prev_s)))


rows = []
for mass in [.1, 1., 10.]:
    rows.append(dict(kind="normal_mass_scaling", **run(mass, .1, .9, 1., True, 0.)))
for offset, tangential_rhs in [(0., 2.), (.1, 8.)]:
    for d in [.5, .9, .99]:
        for rho in [.1, 1., 10.]:
            for adaptive in [False, True]:
                row = run(1., offset, d, rho, adaptive, tangential_rhs)
                assert row["objective_gap"] > .1
                assert row["primal_residual"] < 1e-9
                rows.append(dict(kind="cone_qp_with_per_row_R", **row))
s = ADMMQPSolver()
k = 1 / (s.solimp[1] * s.solref[0] * s.solref[1])**2
single = brentq(lambda p: k*s._impedance(p)**2*p - (1-s._impedance(p))*9.81, 0, .02)
rows.append(dict(kind="isolated_row_vs_coupled_four_feet",
                 isolated_row_depth_m=single,
                 actual_four_foot_model_depth_m=quadruped_admm_equilibrium_depth(s, .0002)))
print(json.dumps(rows, indent=2))
