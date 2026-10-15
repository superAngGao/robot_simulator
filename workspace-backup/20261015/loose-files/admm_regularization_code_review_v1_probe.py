import sys, json
sys.path.insert(0, '/data2/shared/ang.gao/robot_sim_physics_correctness')
import numpy as np
import warp as wp
from tests.unit.solvers.test_contact_regularization import fixture
from tests.gpu.solvers.test_gpu_admm import _ball_merged
from physics.solvers.admm_qp import ADMMQPSolver
from physics.solvers.contact_regularization import normalize_parameters, reference_coefficients
from physics.gpu_engine import GpuEngine

tree,q,v,cs=fixture()
s=ADMMQPSolver(regularization='mujoco')
s.solve(cs,tree,q,v,np.zeros(tree.nv),.002)
for field in ('q','v','au','depth','body'):
    qq=q.copy(); vv=v.copy(); au=np.zeros(tree.nv)
    old=s._prev_f.copy(); before=s.last_R_diag
    if field=='q': qq[4]=np.nan
    if field=='v': vv[0]=np.inf
    if field=='au': au[0]=np.nan
    if field=='depth': cs[0].depth=np.nan
    if field=='body': cs[0].body_i=99
    cs[0].tangent1[:]=7
    try: s.solve(cs,tree,qq,vv,au,.002)
    except ValueError: pass
    else: raise AssertionError(field)
    assert np.array_equal(old,s._prev_f) and np.array_equal(before,s.last_R_diag)
    assert np.all(cs[0].tangent1==7)
    cs[0].depth=0; cs[0].body_i=0
print('CPU_INVALID_STATE_PREMUTATION_PASS')

# Validated coefficients differ from the actual CPU evaluation expression.
s.solref=(.02,1e155)
print('CPU_EXTREME_VALIDATED',reference_coefficients(s.solref,s.solimp,.002))
cs[0].tangent1[:]=7
try:
    s.solve(cs,tree,q,v,np.zeros(tree.nv),.002)
    print('CPU_EXTREME_RESULT',s.last_forces.tolist())
except Exception as e:
    print('CPU_EXTREME_ERROR',type(e).__name__,str(e),'contact_mutated',not np.all(cs[0].tangent1==7))

for field in ('q','qdot','tau','zero_dt','large_finite_q'):
    merged=_ball_merged()
    e=GpuEngine(merged,solver='admm',regularization='mujoco',contact_condim=1,dt=.002)
    qq,vv=merged.tree.default_state(); qq[6]=.09
    e.reset(qq); e.step()
    before=e._solver_scratch.admm_f_prev.numpy().copy()
    kwargs={}
    if field=='q': qq[6]=np.nan; kwargs['q']=qq
    if field=='qdot': vv[2]=np.inf; kwargs['qdot']=vv
    if field=='tau': tt=np.zeros(merged.tree.nv); tt[2]=np.nan; kwargs['tau']=tt
    if field=='zero_dt': kwargs['dt']=0
    if field=='large_finite_q': qq[6]=1e100; kwargs['q']=qq
    try:
        e.step(**kwargs)
        print('GPU_INPUT_ACCEPTED',field,'state_finite',bool(np.all(np.isfinite(e._scratch.q.numpy()))),'cache_changed',not np.array_equal(before,e._solver_scratch.admm_f_prev.numpy()))
    except Exception as ex:
        print('GPU_INPUT_REJECTED',field,type(ex).__name__,str(ex))
    e.close()

# Inject stale tangent/inactive entries into otherwise real engine caches.
e=GpuEngine(_ball_merged(),solver='admm',regularization='mujoco',contact_condim=1,num_envs=2,dt=.002)
qq,vv=e._merged.tree.default_state() if hasattr(e,'_merged') else _ball_merged().tree.default_state()
qq[6]=.09; e.reset(qq); e.step()
sol=e._solver_scratch
for a in (sol.admm_f_prev,sol.admm_s_prev,sol.admm_u_prev):
    a.fill_(7.0)
e.step()
active=e._contact_active.numpy().astype(bool)
unused=np.repeat(~active,3,axis=1) | (np.arange(sol.admm_R_diag.shape[1])[None,:]%3!=0)
for a in (sol.admm_f_prev,sol.admm_s_prev,sol.admm_u_prev,sol.admm_R_diag,sol.admm_rhs_const,sol.lambdas):
    assert np.all(a.numpy()[unused]==0)
print('GPU_STALE_CACHE_MULTI_ENV_MASK_PASS')
e.close()

for entry in ('step_n_tau','step_n_zero_dt','reset','reset_envs'):
    merged=_ball_merged(); e=GpuEngine(merged,solver='admm',regularization='mujoco',contact_condim=1,dt=.002)
    qq,vv=merged.tree.default_state(); qq[6]=.09; e.reset(qq); e.step()
    try:
        if entry=='step_n_tau':
            tt=np.zeros(merged.tree.nv); tt[2]=np.nan; e.step_n(tau=tt)
        elif entry=='step_n_zero_dt': e.step_n(dt=0)
        elif entry=='reset': qq[6]=np.nan; e.reset(qq); e.step()
        else: qq[6]=np.nan; e.reset_envs(np.array([0]),qq); e.step()
        print('GPU_ENTRY_ACCEPTED',entry,'state_finite',bool(np.all(np.isfinite(e._scratch.q.numpy()))))
    except Exception as ex: print('GPU_ENTRY_REJECTED',entry,type(ex).__name__,str(ex))
    e.close()

# A real engine, nonzero normal/tangent velocity, two environments, scalar oracle.
for policy in ('per_row','mujoco'):
    merged=_ball_merged(); e=GpuEngine(merged,solver='admm',regularization=policy,contact_condim=1,num_envs=2,dt=.002)
    qq,vv=merged.tree.default_state(); qq[6]=.09
    qbatch=np.tile(qq,(2,1)); vbatch=np.tile(vv,(2,1)); vbatch[:,2]=[-.7,.2]; vbatch[:,0]=[.4,-.8]
    e._admm_solimp=(.9,.9,.001,.5,2); e._admm_iters=1500; e._admm_warmstart=False
    e.step(q=qbatch,qdot=vbatch)
    sol=e._solver_scratch; W=sol.W.numpy(); R=sol.admm_R_diag.numpy(); p=sol.lambdas.numpy()
    mass=merged.tree.bodies[0].inertia.mass
    for env in range(2):
        ids=np.flatnonzero(e._contact_active.numpy()[env]); assert len(ids)==1
        row=3*ids[0]; depth=float(e._contact_depth.numpy()[env,ids[0]])
        rhs=.002*(-2/(.9*.02)*vbatch[env,2] + 1/(.9**2*.02**2)*.9*depth+9.81)
        expected=max(0,rhs/(1/mass+(1-.9)/.9/mass))
        assert np.isclose(sol.admm_rhs_const.numpy()[env,row],rhs,rtol=3e-5,atol=2e-7)
        assert np.isclose(p[env,row],expected,rtol=5e-5,atol=1e-7)
    print('GPU_NONZERO_VELOCITY_SCALAR_ORACLE_PASS',policy)
    e.close()

from tests.gpu.solvers.test_gpu_admm import _ball_model
from physics.merged_model import merge_models
merged=merge_models(robots={'tiny':_ball_model(1e-35)})
e=GpuEngine(merged,solver='admm',regularization='mujoco',contact_condim=1,dt=.002)
e._admm_solimp=(1e-4,1e-4,.001,.5,2)
qq,vv=merged.tree.default_state(); qq[6]=.09
try:
    e.reset(qq); e.step()
    rr=e._solver_scratch.admm_R_diag.numpy()
    print('GPU_NATIVE_R_RANGE','weights_finite',bool(np.all(np.isfinite(e._admm_body_weights.numpy()))),'R_finite',bool(np.all(np.isfinite(rr))),'R_max',float(np.max(rr)))
except Exception as ex: print('GPU_NATIVE_R_REJECTED',type(ex).__name__,str(ex))
e.close()
