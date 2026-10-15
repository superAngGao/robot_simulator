"""Exercise production ADMM; scalar rows isolate algorithm from cone bug."""
import copy
import json
import sys
import numpy as np

sys.path.insert(0, "/data2/shared/ang.gao/robot_sim_physics_correctness")
from physics.geometry import SphereShape
from physics.merged_model import merge_models
from physics.solvers.admm_qp import ADMMQPSolver
from physics.solvers.pgs_solver import ContactConstraint
from tests.integration.test_cpu_engine_shapes import _make_single_body_engine, _sphere_inertia


class Trace(ADMMQPSolver):
    def _factor_ARrho(self, ar, rho, n):
        self.last_ar = ar.copy()
        self.last_rho = rho
        self.factor_count = getattr(self, "factor_count", 0) + 1
        return super()._factor_ARrho(ar, rho, n)

    def _project_all(self, z, contacts, offsets):
        self.iterations = getattr(self, "iterations", 0) + 1
        return super()._project_all(z, contacts, offsets)


def ball(mass=1):
    merged, _ = _make_single_body_engine(SphereShape(.1), mass, _sphere_inertia(mass, .1))
    return merged.tree


def contact(index=0, condim=1):
    return ContactConstraint(index, -1, np.zeros(3), np.array([0., 0., 1.]),
                             np.zeros(3), np.zeros(3), 0., 1., condim=condim, mu_spin=1.)


def solve(s, tree, contacts, accel):
    q, v = tree.default_state()
    s.iterations = s.factor_count = 0
    f, j = s.solve(contacts, tree, q, v, accel, .001)
    return f, j


rows = []
for mass in [.1, 1., 50., 1000.]:
    for adaptive in [False, True]:
        s = Trace(max_iter=50, warmstart=False, adaptive_rho=adaptive)
        f, _ = solve(s, ball(mass), [contact()], np.array([0., 0., -1., 0., 0., 0.]))
        expected = .9*mass
        rows.append(dict(kind="positive_scalar_mass_scale", mass=mass, adaptive=adaptive,
                         f=float(f[0]), exact=expected, relative_error=float(abs(f[0]-expected)/expected),
                         iterations=s.iterations, factorizations=s.factor_count, final_rho=s.last_rho,
                         fixed_rho_theoretical_relative_error=(1/(1+1/(mass*.9)))**50))

tree = ball()
up = np.array([0., 0., 1., 0., 0., 0.])
for iterations in [1, 5, 50]:
    s = Trace(max_iter=iterations, adaptive_rho=True, warmstart=False)
    f, _ = solve(s, tree, [contact()], up)
    rows.append(dict(kind="truncated_normal_feasibility", budget=iterations, iterations=s.iterations,
                     returned=f.tolist(), projected=s._prev_s.tolist(), exact=[0.]))

s = Trace(max_iter=1000, tolerance=1e-12, adaptive_rho=True)
solve(s, tree, [contact()], up)
end_rho = s.last_rho
saved_u = s._prev_u.copy()
raw, rescaled = copy.deepcopy(s), copy.deepcopy(s)
raw.max_iter = rescaled.max_iter = 1
# Diagnostic reference only: preserve unscaled y=rho*u at next solve's rho.
rescaled._prev_u *= end_rho/rescaled.rho
raw_f, _ = solve(raw, tree, [contact()], up)
scaled_f, _ = solve(rescaled, tree, [contact()], up)
raw_full, scaled_full = copy.deepcopy(s), copy.deepcopy(s)
scaled_full._prev_u *= end_rho/scaled_full.rho
solve(raw_full, tree, [contact()], up)
solve(scaled_full, tree, [contact()], up)
rows.append(dict(kind="rho_warmstart_scale", previous_final_rho=end_rho, next_initial_rho=s.rho,
                 saved_u=saved_u.tolist(), cached_dual=(end_rho*saved_u).tolist(),
                 interpreted_next_dual=(s.rho*saved_u).tolist(),
                 next_one_step_returned=raw_f.tolist(), next_one_step_rescaled_reference=scaled_f.tolist(),
                 repeated_solve_iterations=raw_full.iterations,
                 rescaled_repeated_solve_iterations=scaled_full.iterations))

# Two independent free bodies: normal contact order reverses, row count remains 2.
from robot.model import RobotModel
merged = merge_models({name: RobotModel(tree=ball(), actuated_joint_names=[], contact_body_names=[])
                       for name in ["a", "b"]})
tree2 = merged.tree
accel = np.zeros(tree2.nv)
accel[2], accel[8] = -1., -10.
s = Trace(max_iter=1000, tolerance=1e-12, adaptive_rho=False)
first, _ = solve(s, tree2, [contact(0), contact(1)], accel)
raw, remapped = copy.deepcopy(s), copy.deepcopy(s)
raw.max_iter = remapped.max_iter = 1
for name in ["_prev_f", "_prev_s", "_prev_u"]:
    setattr(remapped, name, getattr(remapped, name)[::-1].copy())
raw_f, _ = solve(raw, tree2, [contact(1), contact(0)], accel)
mapped_f, _ = solve(remapped, tree2, [contact(1), contact(0)], accel)
rows.append(dict(kind="contact_order_warmstart", first_order=first.tolist(), exact_reversed=[9., .9],
                 next_one_step_returned=raw_f.tolist(), next_one_step_mapped_reference=mapped_f.tolist()))

s = Trace()
solve(s, ball(), [contact(condim=4)], np.array([0., 0., -1., 0., 0., 0.]))
rows.append(dict(kind="zero_angular_row_convexity", eigenvalues_Q=np.linalg.eigvalsh(s.last_ar).tolist()))
rows.append(dict(kind="per_row_R_condition_number", condition_A=float(np.linalg.cond(np.diag([1., 1e6]))),
                 condition_Q=float(np.linalg.cond(np.diag([1., 1e6])/.9))))
print(json.dumps(rows, indent=2))
