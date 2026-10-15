import json, sys
import numpy as np
sys.path.insert(0, '/data2/shared/ang.gao/robot_sim_physics_correctness')
from physics.solvers.admm_qp import ADMMQPSolver
from physics.solvers.pgs_solver import ContactConstraint
c = ContactConstraint(0,-1,np.zeros(3),np.array([0.,0.,1.]),np.zeros(3),np.zeros(3),0.,1.)
v = np.array([1.,2.,0.])
got = ADMMQPSolver._project_all(v,[c],[0])
expected=np.array([1.5,1.5,0.])
q=np.diag([1.,3.5,3.5])/.9
rhs=np.array([1.,8.,0.])
x=np.array([1.8,1.8,0.])
g=q@x-rhs
print(json.dumps(dict(production_projection=got.tolist(),euclidean_projection=expected.tolist(),production_distance_squared=float(np.sum((got-v)**2)),minimum_distance_squared=float(np.sum((expected-v)**2)),analytic_cone_optimum=x.tolist(),gradient_at_optimum=g.tolist(),cone_multiplier=1.0)))
assert np.allclose(g,[1,-1,0])
assert np.sum((got-v)**2)>np.sum((expected-v)**2)
