import sys
sys.path.insert(0, '/data2/shared/ang.gao/robot_sim_physics_correctness')
import numpy as np
import mujoco
from tests.validation.rigid_body.models import build_quadruped
from physics.solvers.admm_qp import ADMMQPSolver
from physics.solvers.pgs_solver import ContactConstraint
model,xml=build_quadruped(contact=True)
m=mujoco.MjModel.from_xml_string(xml)
m.geom_condim[:]=1
m.jnt_limited[:]=0
m.dof_damping[:]=0
m.opt.iterations=1000
m.opt.tolerance=1e-13
d=mujoco.MjData(m)
d.qpos[2]=.419
mujoco.mj_forward(m,d)
tree=model.tree
q,v=tree.default_state();q[6]=.419
names={b.name:b.index for b in tree.bodies}
def ours(mid):
    if mid==0:return -1
    return names[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,mid)]
contacts=[]
for i in range(d.ncon):
    c=d.contact[i]
    contacts.append(ContactConstraint(ours(m.geom_bodyid[c.geom2]),ours(m.geom_bodyid[c.geom1]),c.pos.copy(),c.frame[:3].copy(),np.zeros(3),np.zeros(3),-c.dist,0.,condim=1))
s=ADMMQPSolver(warmstart=False,adaptive_rho=False,max_iter=2000,tolerance=1e-12)
f,J=s.solve(contacts,tree,q,v,tree.aba(q,v,np.zeros(tree.nv)),.0002)
print('ncon',d.ncon,'nefc',d.nefc,'R',d.efc_R,'force',d.efc_force,'oldforce',f)
print('Jerr',np.max(abs(J-d.efc_J.reshape(d.nefc,m.nv))))
print('aref',d.efc_aref,'smooth',d.qacc_smooth)
