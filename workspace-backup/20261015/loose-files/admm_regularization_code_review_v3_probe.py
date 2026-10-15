"""Independent repair verification; original v1/v2 scripts remain unchanged."""
import sys, json, platform
from decimal import Decimal, localcontext
sys.path.insert(0, '/data2/shared/ang.gao/robot_sim_physics_correctness')
import numpy as np
import mujoco, scipy, warp
from physics.gpu_engine import GpuEngine
from physics.solvers.admm_qp import ADMMQPSolver
from physics.solvers.pgs_solver import ContactConstraint
from physics.merged_model import merge_models
from tests.gpu.solvers.test_gpu_admm import _ball_model, _ball_merged
from tests.reference.mujoco.test_admm_regularization import scalar_pair, contacts_from_mujoco

print('ENV', platform.python_version(), np.__version__, scipy.__version__, mujoco.__version__, warp.__version__)

def expected_r(depth, imp, weight):
    # Decimal polynomial evaluation, independent of either native implementation.
    with localcontext() as ctx:
        ctx.prec = 80
        a,b,width,mid,power = [Decimal(str(x)) for x in imp]
        x = min(Decimal(1), abs(Decimal(str(depth))) / width) if width > Decimal('1e-15') else None
        if x is None:
            d = (a+b)/2
        else:
            y = x if power == 1 else (x**power / mid**(power-1) if x <= mid else 1-(1-x)**power/(1-mid)**(power-1))
            d = a + y*(b-a)
        return float((1-d)/d*Decimal(str(weight)))

def snap(e):
    sc, so = e._scratch, e._solver_scratch
    arrays = [sc.q,sc.qdot,sc.tau_total,e._contact_active,e._contact_depth,
              so.admm_f_prev,so.admm_s_prev,so.admm_u_prev,so.admm_R_diag,
              so.admm_prev_n_active,so.lambdas] if hasattr(so,'admm_prev_n_active') else [
              sc.q,sc.qdot,sc.tau_total,e._contact_active,e._contact_depth,
              so.admm_f_prev,so.admm_s_prev,so.admm_u_prev,so.admm_R_diag,so.lambdas]
    return ([a.numpy().copy() for a in arrays], (e._physics_run_id,e._publish_frame_id,e._publish_step_index,e._publish_sim_time))

def unchanged(e, before):
    after=snap(e)
    for a,b in zip(before[0],after[0],strict=True): np.testing.assert_array_equal(a,b)
    assert before[1]==after[1]

widths=(np.nextafter(1e-15,0.),1e-15,np.nextafter(1e-15,np.inf),1.000001e-15,0.)
ordinary = [(a,b,w,.5,2) for a,b in ((.2,.8),(.8,.2)) for w in widths]
ordinary += [(.9999,.00012,w,.5,2) for w in (.001,.010001,.0101,.02)]
ordinary += [(.00012,.9999,w,.5,2) for w in (.001,.010001,.0101,.02)]
ordinary += [(.9999,.00012,.1,.001,100),(.9999,.00012,.02,.5,1)]
for condim in (1,3):
    for imp in ordinary:
        tree,m,d,q,v=scalar_pair(free=True,condim=condim,solimp=' '.join(map(str,imp)))
        s=ADMMQPSolver(regularization='mujoco',solimp=imp,max_iter=1)
        s.solve(contacts_from_mujoco(tree,m,d),tree,q,v,np.zeros(tree.nv),.002)
        np.testing.assert_allclose(s.last_R_diag,d.efc_R,rtol=1e-10,atol=1e-12)
        e=GpuEngine(_ball_merged(mass=2),solver='admm',regularization='mujoco',contact_condim=condim,dt=.002)
        e._admm_solimp=imp;e._admm_iters=1;e.reset(q);e.step()
        slots=np.flatnonzero(e._contact_active.numpy()[0]); assert len(slots)==1
        slot=int(slots[0]);depth=float(e._contact_depth.numpy()[0,slot])
        actual=e._solver_scratch.admm_R_diag.numpy()[0,3*slot:3*slot+condim]
        expected=expected_r(depth,imp,.5)
        budget=3e-5*abs(expected)+1e-6
        assert np.all(abs(actual.astype(float)-expected)<=budget),(imp,actual,expected,budget)
        d.qpos[2]=.1-depth;mujoco.mj_forward(m,d)
        np.testing.assert_allclose(actual,d.efc_R,rtol=3e-5,atol=1e-6)
        print('ORDINARY',json.dumps(dict(condim=condim,imp=list(map(float,imp)),R=actual.tolist(),expected=expected,error=float(max(abs(actual.astype(float)-expected))),budget=budget)))
        e.close()

# Exact old boundary and adjacent supported cases, both dimensions and depths.
for condim in (1,3):
    for width in (.001,.010001):
        for mass,accept in ((4.89730620518322e-35,False),(5e-35,True)):
            merged=merge_models(robots={'a':_ball_model(mass),'b':_ball_model(mass)})
            e=GpuEngine(merged,solver='admm',regularization='mujoco',contact_condim=condim,dt=.002)
            q,v=merged.tree.default_state();q[6]=q[13]=.09;q[11]=.19
            e._admm_iters=1;e.reset(q)
            # Seed ordinary valid state/cache on this same engine before rejection.
            e.step();before=snap(e)
            e._admm_solimp=(.9999,.00012,width,.5,2)
            if not accept:
                try: e.step(q=q,qdot=v)
                except ValueError: pass
                else: raise AssertionError('old GPU edge accepted')
                unchanged(e,before)
                e._admm_solimp=(.9,.95,.001,.5,2);e.step(q=q,qdot=v)
                assert np.all(np.isfinite(e._solver_scratch.admm_R_diag.numpy()))
                print('GPU_ATOMIC_REJECT_RECOVER',condim,width,mass)
            else:
                e.step(q=q,qdot=v)
                active=e._contact_active.numpy()[0].astype(bool)
                bi=e._contact_bi.numpy()[0];bj=e._contact_bj.numpy()[0]
                R=e._solver_scratch.admm_R_diag.numpy()[0];depths=e._contact_depth.numpy()[0]
                assert active.sum()==3 and np.any(bj[active]>=0) and np.any(bj[active]<0)
                for c in np.flatnonzero(active):
                    expected=expected_r(float(depths[c]),e._admm_solimp,(int(bi[c]>=0)+int(bj[c]>=0))/mass)
                    values=R[3*c:3*c+condim]
                    assert np.all(np.isfinite(values))
                    np.testing.assert_allclose(values,expected,rtol=3e-5)
                print('GPU_ADJACENT_FINITE',condim,width,mass,R[3*np.flatnonzero(active)].tolist())
            e.close()

for condim in (1,3):
    for width in (.001,.010001):
        for mass,accept in ((8.556863841944613e-305,False),(1e-304,True)):
            tree=merge_models(robots={'a':_ball_model(mass),'b':_ball_model(mass)}).tree
            q,v=tree.default_state()
            c=ContactConstraint(0,1,np.zeros(3),np.array([0.,0.,1.]),np.full(3,7.),np.zeros(3),.01,.5,condim=condim)
            s=ADMMQPSolver(regularization='mujoco',solimp=(.9999,.00013,width,.5,2),max_iter=1)
            if not accept:
                try: s.solve([c],tree,q,v,np.zeros(tree.nv),.002)
                except ValueError: pass
                else: raise AssertionError('old CPU edge accepted')
                np.testing.assert_array_equal(c.tangent1,7.)
                assert s._prev_f is None and s.last_R_diag.size==0
                s.solimp=(.9,.95,.001,.5,2);s.solve([c],tree,q,v,np.zeros(tree.nv),.002)
                assert np.all(np.isfinite(s.last_R_diag))
                print('CPU_ATOMIC_REJECT_RECOVER',condim,width,mass)
            else:
                s.solve([c],tree,q,v,np.zeros(tree.nv),.002)
                expected=expected_r(.01,s.solimp,2/mass)
                assert np.all(np.isfinite(s.last_R_diag))
                np.testing.assert_allclose(s.last_R_diag,expected,rtol=1e-10)
                print('CPU_ADJACENT_FINITE',condim,width,mass,s.last_R_diag.tolist())
print('PASS: F3/F4 original cases and expanded controls; F3 rejection atomic and recoverable')
