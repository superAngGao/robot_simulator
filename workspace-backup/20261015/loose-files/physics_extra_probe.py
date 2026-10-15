import json
import numpy as np
import mujoco
from tests.validation.rigid_body.models import build_double_pendulum,build_quadruped,G
from tests.validation.rigid_body.test_contact_vs_mujoco import _run_mujoco_quad
from tests.integration.contact_pipeline.test_incline_contact import _ball_on_incline,_run_steps,_velocity_along_incline
from physics.cpu_engine import CpuEngine
from physics.merged_model import merge_models
def emit(kind,**kw):print(json.dumps(dict(kind=kind,**kw)),flush=True)
model,xml=build_double_pendulum(); mm=mujoco.MjModel.from_xml_string(xml);md=mujoco.MjData(mm)
rng=np.random.default_rng(928);errs=[]
for _ in range(30):
    q=rng.uniform(-2,2,2);v=rng.uniform(-3,3,2);tau=rng.uniform(-1,1,2)
    md.qpos[:]=q;md.qvel[:]=v;md.qfrc_applied[:]=tau;mujoco.mj_forward(mm,md)
    errs.append(float(max(abs(model.tree.aba(q,v,tau)-md.qacc))))
emit('acceleration',max_error=max(errs),samples=30)
model,xml=build_quadruped(contact=True);z,v,_,_=_run_mujoco_quad(xml,.45,5000)
emit('mujoco_settling',mean_z=float(z[-500:].mean()),mean_abs_v=float(abs(v[-500:]).mean()))
for theta,mu,t,mode in [(30,0.,.2,'sliding_zero'),(30,.1,.2,'sliding'),(20,1.,.5,'rolling')]:
    theta=np.radians(theta)
    for dt in [.001,.0005]:
        sim,q,v=_ball_on_incline(theta,mu,dt=dt)
        q,v=_run_steps(sim,q,v,round(t/dt));actual=_velocity_along_incline(v,theta)
        a=(5/7*G*np.sin(theta)) if mode=='rolling' else G*(np.sin(theta)-mu*np.cos(theta))
        emit('incline',mode=mode,dt=dt,actual=actual,expected=-a*t,rel_error=abs(actual+a*t)/(a*t))
for dt in [.001,.0005]:
    model,_=build_quadruped(contact=False);merged=merge_models({'r':model});eng=CpuEngine(merged,dt=dt)
    q,v=merged.tree.default_state();q[6]=10.;t=.1
    for _ in range(round(t/dt)):
        out=eng.step(q,v,np.zeros(merged.nv));q,v=out.q_new,out.qdot_new
    emit('freefall',dt=dt,z_error=q[6]-(10-.5*G*t*t),euler_prediction=-.5*G*t*dt,v_error=v[2]+G*t)
