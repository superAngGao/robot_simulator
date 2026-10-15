import sys,json
sys.path.insert(0,'/data2/shared/ang.gao/robot_sim_physics_correctness')
import numpy as np
from physics.gpu_engine import GpuEngine
from physics.solvers.admm_qp import ADMMQPSolver
from physics.merged_model import merge_models
from tests.gpu.solvers.test_gpu_admm import _ball_model
from tests.reference.mujoco.test_admm_regularization import scalar_pair,contacts_from_mujoco
for width in (.001,.010001,.0101,.02):
    imp=(.9999,.00012,width,.5,2)
    tree,m,d,q,v=scalar_pair(free=True,mass=2,condim=1,solimp=' '.join(map(str,imp)))
    solver=ADMMQPSolver(regularization='mujoco',solimp=imp,max_iter=1)
    solver.solve(contacts_from_mujoco(tree,m,d),tree,q,v,np.zeros(tree.nv),.002)
    reference=float(d.efc_R[0]); cpu=float(solver.last_R_diag[0])
    assert np.isclose(cpu,reference,rtol=1e-10,atol=1e-12)
    merged=merge_models(robots={'a':_ball_model(2.)})
    e=GpuEngine(merged,solver='admm',regularization='mujoco',contact_condim=1,dt=.002)
    e._admm_solimp=imp;e._admm_iters=1
    qq,vv=merged.tree.default_state();qq[6]=.09;e.reset(qq);e.step()
    row=3*np.flatnonzero(e._contact_active.numpy()[0])[0]
    rr=float(e._solver_scratch.admm_R_diag.numpy()[0,row])
    depth=float(e._contact_depth.numpy()[0,row//3])
    # Evaluate the actual reference equations at GPU depth too, eliminating
    # tiny collision-depth differences as an explanation for an R discrepancy.
    x=min(1.,depth/width)
    y=.5*(x/.5)**2 if x<=.5 else 1.-.5*((1.-x)/.5)**2
    dimp=imp[0]+y*(imp[1]-imp[0]); expected=(1-dimp)/dimp*.5
    budget=3e-5*abs(expected)+2e-6*.5
    print('DECREASING_CURVE',json.dumps(dict(width=width,depth=depth,x=x,mujoco_R=reference,cpu_R=cpu,gpu_R=rr,reference_at_gpu_depth=expected,absolute_error=abs(rr-expected),budget=budget,exceeds_budget=abs(rr-expected)>budget)))
    e.close()
    if width in (.001,.010001):
        assert abs(rr-expected)>budget
        mass=4.89730620518322e-35
        merged=merge_models(robots={'a':_ball_model(mass),'b':_ball_model(mass)})
        e=GpuEngine(merged,solver='admm',regularization='mujoco',contact_condim=1,dt=.002)
        e._admm_solimp=imp
        qq,vv=merged.tree.default_state();qq[6]=.09;qq[11]=.19;qq[13]=.09;e.reset(qq);e.step()
        active=e._contact_active.numpy()[0].astype(bool)
        rr=e._solver_scratch.admm_R_diag.numpy()[0,3*np.flatnonzero(active)]
        assert np.any(np.isinf(rr))
        print('INTERIOR_OR_ENDPOINT_OVERFLOW',width,rr.tolist())
        e.close()
print('CONFIRMED: ordinary 2 kg R conformance failure plus endpoint and interior native overflow')
