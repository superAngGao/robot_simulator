import sys,json
sys.path.insert(0,'/data2/shared/ang.gao/robot_sim_physics_correctness')
import numpy as np
from physics.gpu_engine import GpuEngine
from tests.gpu.solvers.test_gpu_admm import _ball_merged
from tests.reference.mujoco.test_admm_regularization import scalar_pair,contacts_from_mujoco
from physics.solvers.admm_qp import ADMMQPSolver
for width in (np.nextafter(1e-15,0.),1e-15,np.nextafter(1e-15,np.inf),1.000001e-15):
    imp=(.2,.8,float(width),.5,2)
    tree,m,d,q,v=scalar_pair(free=True,mass=2,condim=1,solimp=' '.join(map(str,imp)))
    s=ADMMQPSolver(regularization='mujoco',solimp=imp,max_iter=1)
    s.solve(contacts_from_mujoco(tree,m,d),tree,q,v,np.zeros(tree.nv),.002)
    e=GpuEngine(_ball_merged(mass=2.),solver='admm',regularization='mujoco',contact_condim=1,dt=.002)
    e._admm_solimp=imp;e._admm_iters=1
    e.reset(q);e.step()
    row=3*np.flatnonzero(e._contact_active.numpy()[0])[0]
    actual=float(e._solver_scratch.admm_R_diag.numpy()[0,row])
    expected=float(d.efc_R[0])
    print('WIDTH_SWITCH',json.dumps(dict(width=float(width),native_width=float(np.float32(width)),cpu=float(s.last_R_diag[0]),mujoco=expected,gpu=actual)))
    if width==np.nextafter(1e-15,np.inf):
        assert np.isclose(expected,.125) and np.isclose(actual,.5)
    e.close()
