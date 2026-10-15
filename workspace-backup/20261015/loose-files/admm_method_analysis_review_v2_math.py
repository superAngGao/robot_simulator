import json
import numpy as np

with open('/tmp/admm_method_analysis_review_v2_cpu.json') as stream:
    rows = json.load(stream)
Q = np.diag([1., 3.5, 3.5]) / .9
c = np.array([1., 8., 0.])
f = np.array([1.8, 1.8, 0.])
outward = np.array([-1., 1., 0.])
grad = Q @ f - c
assert np.allclose(grad + outward, 0., atol=1e-14)
assert not np.allclose(grad, 0.)
results = {'force_gradient': grad.tolist(), 'force_KKT_residual': (grad + outward).tolist(), 'scaling': [], 'fixed_rho_actual_iteration_checks': []}
for dt in [.0002, .001, .3, 2.]:
    p = dt * f
    old_obj = .5 * f @ Q @ f - c @ f
    new_obj = .5 * p @ Q @ p - dt * c @ p
    residual = Q @ p - dt * c + dt * outward
    assert abs(new_obj - dt**2 * old_obj) < 1e-14
    assert np.linalg.norm(residual) < 1e-14
    results['scaling'].append({'dt': dt, 'impulse_KKT_residual_norm': float(np.linalg.norm(residual)), 'objective_scaling_error': float(new_obj - dt**2 * old_obj)})
for row in rows:
    if row['kind'] != 'positive_scalar_mass_scale':
        continue
    alpha = 1 / (1 + 1 / (.9 * row['mass']))
    assert row['fixed_rho_theoretical_relative_error'] == alpha**50
    if not row['adaptive']:
        expected = alpha**row['iterations']
        difference = abs(expected - row['relative_error'])
        assert difference < 1e-14
        results['fixed_rho_actual_iteration_checks'].append({'mass': row['mass'], 'iterations': row['iterations'], 'theory_at_actual_iterations': expected, 'measured_error': row['relative_error'], 'absolute_difference': difference})
print(json.dumps(results, indent=2))
