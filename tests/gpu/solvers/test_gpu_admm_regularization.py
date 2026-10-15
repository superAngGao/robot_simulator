"""Real-device tests of production regularization and condim1 subproblems."""
import numpy as np
import pytest

wp = pytest.importorskip("warp")
from physics.gpu_engine import GpuEngine
from physics.merged_model import merge_models
from physics.solvers.contact_regularization import impedance, normalize_parameters, reference_body_weights
from tests.gpu.solvers.test_gpu_admm import _ball_model, _ball_merged

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("policy,condim,imp,ratio", [
    ("per_row",1,(.9,.9,.001,.5,2),1), ("per_row",3,(.9,.9,.001,.5,2),1),
    ("mujoco",1,(.9,.9,.001,.5,2),1), ("mujoco",3,(.9,.95,.001,.5,2),10),
    ("mujoco",3,(.2,.8,0,.5,2),1e-16),
    ("mujoco",3,(-1,2,-1,2,.5),1),
    ("mujoco",3,(.2,.9,.1,.001,100),1),
])
def test_real_engine_R_and_normal_impulses(policy,condim,imp,ratio):
    # Independent roots with unequal masses; body-body plus ground contacts.
    merged = merge_models(robots={"a": _ball_model(2.), "b": _ball_model(5.)})
    engine = GpuEngine(merged,solver="admm",regularization=policy,impratio=ratio,contact_condim=condim,dt=.002)
    engine._admm_solimp = imp
    engine._admm_iters = 1500 if condim == 1 else 1
    engine._admm_warmstart = False
    q,v = merged.tree.default_state(); q[6]=.09; q[11]=.19; q[13]=.09
    engine.reset(q); engine.step()
    sol=engine._solver_scratch
    active=engine._contact_active.numpy()[0].astype(bool)
    assert active.sum() >= 3
    bi=engine._contact_bi.numpy()[0]; bj=engine._contact_bj.numpy()[0]
    assert np.any(active & (bi >= 0) & (bj >= 0))
    W=sol.W.numpy()[0]; R=sol.admm_R_diag.numpy()[0]; p=sol.lambdas.numpy()[0]
    depths=engine._contact_depth.numpy()[0]
    weights=reference_body_weights(merged.tree)
    normalized=normalize_parameters((.02,1),imp,.002)[1] if policy=="mujoco" else imp
    for c in np.flatnonzero(active):
        base=3*c
        d=impedance(float(depths[c]), normalized)
        if policy=="per_row":
            expected=(1-d)/d*np.abs(np.diag(W)[base:base+condim])
        else:
            rn=max(1e-15,(1-d)/d*sum(weights[i] for i in (bi[c],bj[c]) if i>=0))
            expected=np.r_[rn,np.full(condim-1,rn/max(1e-15,ratio))]
        np.testing.assert_allclose(R[base:base+condim],expected,rtol=3e-5,atol=2e-6)
    unused=np.repeat(~active,3)
    if condim==1: unused |= np.arange(len(R))%3 != 0
    for a in (R,p,sol.admm_rhs_const.numpy()[0],sol.admm_f_prev.numpy()[0],sol.admm_s_prev.numpy()[0],sol.admm_u_prev.numpy()[0]):
        np.testing.assert_array_equal(a[unused],0)
    if condim==1:
        rows=3*np.flatnonzero(active)
        A=W[np.ix_(rows,rows)]+np.diag(R[rows]); rhs=sol.admm_rhs_const.numpy()[0,rows]
        # Independent active-set enumeration for a small nonnegative QP.
        best=None
        for bits in range(1<<len(rows)):
            ids=np.array([i for i in range(len(rows)) if bits & (1<<i)],dtype=int)
            x=np.zeros(len(rows))
            if len(ids): x[ids]=np.linalg.solve(A[np.ix_(ids,ids)].astype(float),rhs[ids])
            g=A@x-rhs
            if np.all(x>=-1e-10) and np.all(g>=-1e-7) and np.max(np.abs(g*x))<1e-7:
                best=x; break
        assert best is not None
        np.testing.assert_allclose(p[rows],best,rtol=5e-5,atol=1e-7)
        n=3*(np.flatnonzero(active)[-1]+1); mat=sol.admm_AR_rho.numpy()[0,:n,:n].copy()
        mat-=np.eye(n)*engine._admm_rho
        np.testing.assert_array_equal(mat[unused[:n]],0)
        np.testing.assert_array_equal(mat[:,unused[:n]],0)
    # Warm cache from active contacts must not survive a now-empty solve.
    engine._admm_warmstart=True; q[6]=2.; q[13]=2.; q[11]=1.
    engine.step(q=q,qdot=v)
    assert not np.any(engine._contact_active.numpy())
    for a in (sol.admm_R_diag,sol.lambdas,sol.admm_s_prev,sol.admm_u_prev):
        np.testing.assert_array_equal(a.numpy(),0)
    engine.close()


def test_invalid_options_and_native_domain():
    merged=_ball_merged()
    for kw in ({"regularization":"bad"}, {"regularization":"mujoco"},
               {"solver":"admm","contact_condim":4}, {"solver":"admm","impratio":2}):
        with pytest.raises(ValueError): GpuEngine(merged,**kw)
    engine=GpuEngine(merged,solver="admm",regularization="mujoco",contact_condim=1)
    with pytest.raises(AttributeError): engine.impratio=2
    for ref in ((.02,1e-30),(.02,np.nan),(-1,1)):
        engine._admm_solref=ref
        with pytest.raises(ValueError): engine._step_physics()
    engine.close()
