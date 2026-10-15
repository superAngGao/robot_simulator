import sys
sys.path.insert(0,'/data2/shared/ang.gao/robot_sim_physics_correctness')
import numpy as np
from physics.gpu_engine import GpuEngine
from physics.merged_model import merge_models
from tests.gpu.solvers.test_gpu_admm import _ball_model
from physics.solvers.contact_regularization import validate_R_domain
for dw in (0.00012,0.00013,0.00015,0.00018):
    r=(np.float32(1)-np.float32(dw))/np.float32(dw)
    w=np.float32(np.finfo(np.float32).max/r/np.float32(2))
    w=np.nextafter(w,np.float32(0))
    mass=1/float(w)
    imp=(.9999,dw,.001,.5,2)
    validate_R_domain([w,w],imp,1,1,np.float32)
    m=merge_models(robots={'a':_ball_model(mass),'b':_ball_model(mass)})
    e=GpuEngine(m,solver='admm',regularization='mujoco',contact_condim=1,dt=.002)
    e._admm_solimp=imp
    q,v=m.tree.default_state();q[6]=.09;q[11]=.19;q[13]=.09
    e.reset(q)
    try:
        e.step()
        active=e._contact_active.numpy()[0].astype(bool)
        rows=3*np.flatnonzero(active)
        print('EDGE',dw,mass,e._solver_scratch.admm_R_diag.numpy()[0,rows],e._contact_bi.numpy()[0,active],e._contact_bj.numpy()[0,active])
    except ValueError as ex: print('REJECT',dw,str(ex))
    e.close()
