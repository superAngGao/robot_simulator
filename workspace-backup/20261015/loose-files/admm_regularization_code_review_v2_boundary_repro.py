import sys, json
sys.path.insert(0,'/data2/shared/ang.gao/robot_sim_physics_correctness')
import numpy as np
import scipy, mujoco, warp
from physics.gpu_engine import GpuEngine
from physics.merged_model import merge_models
from tests.gpu.solvers.test_gpu_admm import _ball_model
from physics.solvers.contact_regularization import validate_R_domain
print('ENV',sys.version,np.__version__,scipy.__version__,mujoco.__version__,warp.__version__)
for condim in (1,3):
    dw=.00012
    endpoint_ratio=(np.float32(1)-np.float32(dw))/np.float32(dw)
    w=np.float32(np.finfo(np.float32).max/endpoint_ratio/np.float32(2))
    w=np.nextafter(w,np.float32(0))
    mass=1/float(w)
    imp=(.9999,dw,.001,.5,2)
    m=merge_models(robots={'a':_ball_model(mass),'b':_ball_model(mass)})
    e=GpuEngine(m,solver='admm',regularization='mujoco',contact_condim=condim,dt=.002)
    e._admm_solimp=imp
    q,v=m.tree.default_state();q[6]=.09;q[11]=.19;q[13]=.09
    e.reset(q)
    host_weights=e._admm_host_body_weights.copy()
    validate_R_domain(host_weights,imp,1,condim,np.float32)
    e._validate_admm_step(.002)
    native_weights=e._admm_body_weights.numpy()
    validated_bound=endpoint_ratio*(np.float32(2)*np.max(native_weights))
    interp_d=np.float32(imp[0])+np.float32(1)*(np.float32(dw)-np.float32(imp[0]))
    assert interp_d < np.float32(dw)
    before=e._solver_scratch.admm_R_diag.numpy().copy()
    output=e.step()
    active=e._contact_active.numpy()[0].astype(bool)
    bi=e._contact_bi.numpy()[0,active];bj=e._contact_bj.numpy()[0,active]
    rows=3*np.flatnonzero(active)
    R=e._solver_scratch.admm_R_diag.numpy()[0,rows]
    expected=(1-dw)/dw*float(w)*np.where(bj>=0,2,1)
    assert np.all(np.isfinite(expected)) and np.all(expected < np.finfo(np.float32).max)
    assert np.any(np.isinf(R[bj>=0]))
    assert np.all(np.isfinite(R[bj<0]))
    print('ADMITTED_NONFINITE_R',json.dumps(dict(condim=condim,mass=mass,weights=host_weights.tolist(),native_weights=native_weights.tolist(),d_endpoint=float(np.float32(dw)),d_interpolated=float(interp_d),host_bound=float(validated_bound),actual_R=R.tolist(),exact_endpoint_R=expected.tolist(),bodies=list(zip(bi.tolist(),bj.tolist())),depths=e._contact_depth.numpy()[0,active].tolist(),mutated_R=not np.array_equal(before,e._solver_scratch.admm_R_diag.numpy()),state_finite=bool(np.all(np.isfinite(output.q_new))))))
    e.close()
print('CONFIRMED: native endpoint-bound admission permits Inf R in condim1 and condim3')
