import json
import sys
import numpy as np

sys.path.insert(0, '/data2/shared/ang.gao/robot_sim_physics_correctness')
from physics.solvers.admm_qp import ADMMQPSolver
from physics.solvers.pgs_solver import ContactConstraint

data = json.load(open('/tmp/admm_bounded_independent_review_v1.json'))
z = np.array([1., 8., 0.])
contact = ContactConstraint(0, -1, np.zeros(3), np.array([0.,0.,1.]), np.zeros(3), np.zeros(3), 0., 1., condim=3)
actual = ADMMQPSolver._project_all(z, [contact], [0])
exact = np.array([4.5,4.5,0.])
assert np.array_equal(actual, [1.,1.,0.])
assert np.dot(z-actual, exact-actual) > 0
print('Projection counterexample:', dict(input=z.tolist(), production=actual.tolist(), exact=exact.tolist(), production_squared_distance=float(np.sum((z-actual)**2)), exact_squared_distance=float(np.sum((z-exact)**2)), variational_inequality_violation=float(np.dot(z-actual,exact-actual))))

for kind, exact_scalar in [('coupled_normal',18/43),('coupled_friction',162/181)]:
    row = next(r for r in data['frozen'] if r['kind']==kind and r['rho']==1 and r['budget']==1000)
    points=[np.array([-.1,0.,-.1]),np.array([.1,0.,-.1])]
    directions=[np.array([0.,0.,1.])]
    if kind=='coupled_friction':
        directions.extend([np.array([0.,1.,0.]),np.array([-1.,0.,0.])])
    J=np.array([np.concatenate([d,np.cross(p,d)]) for p in points for d in directions])
    invH=np.diag([1.,1.,1.,250.,250.,250.])
    A=J@invH@J.T
    Q=A+np.diag(np.diag(A)/9.)
    c=np.array([1.,8.,0.]*2) if kind=='coupled_friction' else np.ones(2)
    direction=np.array([1.,1.,0.]*2) if kind=='coupled_friction' else np.ones(2)
    scalar=float(c@direction/(direction@Q@direction))
    np.testing.assert_allclose(scalar,exact_scalar,rtol=0,atol=1e-15)
    f=exact_scalar*direction
    g=Q@f-c
    if kind=='coupled_friction':
        for b in [0,3]:
            assert g[b]>=0
            np.testing.assert_allclose(g[b:b+3],g[b]*np.array([1.,-1.,0.]),atol=2e-15)
    else:
        np.testing.assert_allclose(g,0.,atol=2e-15)
    np.testing.assert_allclose(row['Q'],Q,atol=1e-14)
    np.testing.assert_allclose(row['reference'],f,atol=1e-8)
    print('Independent analytic coupled optimum:',kind, f.tolist(), 'gradient',g.tolist(), 'lambda_min',float(np.linalg.eigvalsh(Q)[0]))

print('Trajectory balance maximum:',max(r['linear_impulse_balance_error'] for r in data['trajectories']))
print('Trajectory vx<=1 force error maximum:',max(r['worst_frozen_step']['max_error'] for r in data['trajectories'] if r['vx']<=1))
print('Missed penetrating steps:',sum(r['missed_penetrating_steps'] for r in data['trajectories']))
for row in data['gpu']:
    Q=np.array(row['Q']); c=np.array(row['rhs']); p=np.array(row['returned_impulse'])
    assert np.all(np.linalg.eigvalsh(Q)>0)
    ref=np.array(row['reference_impulse']); g=Q@ref-c
    if row['vx']:
        normal=np.concatenate([[-.8],ref[1:]/np.linalg.norm(ref[1:])])
        multiplier=-g@normal/(normal@normal)
        assert multiplier>=0
        np.testing.assert_allclose(g+multiplier*normal,0.,atol=1e-16)
    else:
        np.testing.assert_allclose(g,0.,atol=1e-16)
    assert abs(np.linalg.norm(p-ref)/np.linalg.norm(ref)-row['relative_error'])<1e-15
    print('GPU independently checked KKT:',row['mass'],row['vx'],row['budget'],row['relative_error'])
