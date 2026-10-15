import sys,json
sys.path.insert(0,'/data2/shared/ang.gao/robot_sim_physics_correctness')
import numpy as np
from physics.merged_model import merge_models
from tests.gpu.solvers.test_gpu_admm import _ball_model
from physics.solvers.admm_qp import ADMMQPSolver
from physics.solvers.pgs_solver import ContactConstraint
from physics.solvers.contact_regularization import reference_body_weights,validate_R_domain,impedance
for dw in (.00012,.00013,.00015):
    ratio=(1-dw)/dw
    weight=np.finfo(float).max/ratio/2
    for i in range(8): weight=np.nextafter(weight,0.)
    mass=1/weight
    merged=merge_models(robots={'a':_ball_model(mass),'b':_ball_model(mass)})
    tree=merged.tree;q,v=tree.default_state()
    imp=(.9999,dw,.001,.5,2)
    weights=reference_body_weights(tree)
    validate_R_domain(weights,imp,1,1)
    c=ContactConstraint(0,1,np.zeros(3),np.array([0.,0.,1.]),np.zeros(3),np.zeros(3),.01,.5,condim=1)
    s=ADMMQPSolver(regularization='mujoco',solimp=imp,max_iter=1)
    s.solve([c],tree,q,v,np.zeros(tree.nv),.002)
    print('CPU_EDGE',json.dumps(dict(dw=dw,mass=mass,weights=weights.tolist(),d=impedance(.01,imp),host_bound=ratio*(2*max(weights)),R=s.last_R_diag.tolist())))
