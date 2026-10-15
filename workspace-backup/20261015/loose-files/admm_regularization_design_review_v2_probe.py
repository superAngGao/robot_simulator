import sys, json
sys.path.insert(0, '/data2/shared/ang.gao/robot_sim_physics_correctness')
import numpy as np
import mujoco
from physics.robot_tree import Body, RobotTreeNumpy
from physics.joint import PrismaticJoint, FixedJoint
from physics.spatial import SpatialInertia, SpatialTransform, skew
from physics.dynamics_utils import body_velocity_jacobians

print('MuJoCo', mujoco.__version__)
def check(label, *, parent_levels=0, child=False, com=0., axis=(0,0,1), inertia=(.008,.009,.01)):
    tree=RobotTreeNumpy()
    prefix=''
    for i in range(parent_levels):
        tree.add_body(Body('fixed'+str(i),i,FixedJoint('fixed'),SpatialInertia(1., np.eye(3)*.01,np.zeros(3)),SpatialTransform.identity(),parent=i-1))
        prefix += '<body name="fixed%d"><inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/>' % i
    idx=tree.add_body(Body('slider',0,PrismaticJoint('z',axis=np.array(axis,dtype=float)),SpatialInertia(2.,np.diag(inertia),np.array([com,0,0])),SpatialTransform(np.eye(3),np.array([0.,0.,.09])),parent=parent_levels-1))
    inside=''
    if child:
        tree.add_body(Body('child',0,FixedJoint('fixed'),SpatialInertia(1.,np.eye(3)*.01,np.zeros(3)),SpatialTransform(np.eye(3),np.array([.3,0,0])),parent=idx))
        inside='<body name="child" pos=".3 0 0"><inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/></body>'
    xml=f'''<mujoco><compiler fusestatic="false"/><option timestep=".002" solver="Newton" tolerance="1e-14" iterations="1000"><flag diagexact="disable"/></option><default><geom condim="1" solimp=".9 .9 .001 .5 2" solref=".02 1"/></default><worldbody><geom type="plane" size="2 2 .1"/>{prefix}<body name="slider" pos="0 0 .09"><joint type="slide" axis="{' '.join(map(str,axis))}"/><inertial pos="{com} 0 0" mass="2" diaginertia="{' '.join(map(str,inertia))}"/><geom type="sphere" size=".1"/>{inside}</body>{'</body>'*parent_levels}</worldbody></mujoco>'''
    tree.finalize();q,_=tree.default_state();H=tree.crba(q);B=body_velocity_jacobians(tree,q)
    mm=mujoco.MjModel.from_xml_string(xml);dd=mujoco.MjData(mm);mujoco.mj_forward(mm,dd)
    weights=[]
    for b in tree.bodies:
        J=B[b.index,:3]-skew(b.inertia.com)@B[b.index,3:]
        trace=np.trace(J@np.linalg.solve(H,J.T))/3
        simple=isinstance(b.joint,PrismaticJoint) and not b.children and (b.parent<0 or (tree.bodies[b.parent].parent<0 and isinstance(tree.bodies[b.parent].joint,FixedJoint))) and np.all(np.abs(b.inertia.com)<1e-6) and np.count_nonzero(np.abs(b.joint._axis_vec)>1e-14)==1
        predicted=1/max(1e-15,b.inertia.mass) if simple else trace
        mid=mujoco.mj_name2id(mm,mujoco.mjtObj.mjOBJ_BODY,b.name)
        np.testing.assert_allclose(predicted,mm.body_invweight0[mid,0],rtol=1e-11,atol=1e-13)
        weights.append((b.name,bool(simple),int(mm.body_simple[mid]),predicted,float(mm.body_invweight0[mid,0])))
    expected_R=(1-.9)/.9*weights[idx][3]
    np.testing.assert_allclose(dd.efc_R[0],expected_R,rtol=1e-12)
    if label=='simple':
        np.testing.assert_allclose(dd.efc_force[0],67.658,rtol=1e-12)
    print(label,json.dumps(weights),'R',float(dd.efc_R[0]),'force',float(dd.efc_force[0]))

check('simple')
check('fixed_parent',parent_levels=1)
check('nested_fixed_parent',parent_levels=2)
check('fixed_child',child=True)
check('offset_com',com=.01)
check('below_com_cutoff',com=.999e-6)
check('at_com_cutoff',com=1e-6)
check('tilted_axis',axis=(.1,0,1))
check('below_axis_cutoff',axis=(.999e-14,0,1))
check('above_axis_cutoff',axis=(1.001e-14,0,1))

xml='''<mujoco><option timestep=".002" cone="elliptic"><flag diagexact="disable"/></option><default><geom condim="3" solimp=".9 .9 .001 .5 2" solref=".02 1"/></default><worldbody><geom type="plane" size="2 2 .1"/><body pos="0 0 .09"><freejoint/><inertial pos="0 0 0" mass="2" diaginertia=".008 .009 .01"/><geom type="sphere" size=".1"/></body></worldbody></mujoco>'''
for impratio in (1e-16,1e-15,2e-15,1.,10.):
    m=mujoco.MjModel.from_xml_string(xml);m.opt.impratio=impratio;d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    expected=np.array([1.,1/max(1e-15,impratio),1/max(1e-15,impratio)])*((1-.9)/.9*.5)
    np.testing.assert_allclose(d.efc_R,expected,rtol=1e-13)
    print('impratio',impratio,'R',d.efc_R.tolist())

from tests.validation.rigid_body.models import build_double_pendulum,build_quadruped
for build in (build_double_pendulum,build_quadruped):
    model,xml=build();tree=model.tree;q,_=tree.default_state();H=tree.crba(q);B=body_velocity_jacobians(tree,q)
    m=mujoco.MjModel.from_xml_string(xml);errors=[]
    for b in tree.bodies:
        J=B[b.index,:3]-skew(b.inertia.com)@B[b.index,3:]
        w=np.trace(J@np.linalg.solve(H,J.T))/3
        ref=m.body_invweight0[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,b.name),0]
        np.testing.assert_allclose(w,ref,rtol=1e-11,atol=1e-13);errors.append(abs(w-ref))
    print(build.__name__,'bodies',len(errors),'maximum_weight_error',max(errors))
model,xml=build_quadruped(contact=True);tree=model.tree
xml=xml.replace('cone="elliptic"/>','cone="elliptic"><flag diagexact="disable"/></option>')
m=mujoco.MjModel.from_xml_string(xml);m.geom_condim[:]=1;m.jnt_limited[:]=0;m.dof_damping[:]=0
m.opt.iterations=1000;m.opt.tolerance=1e-13
d=mujoco.MjData(m);d.qpos[2]=.419;mujoco.mj_forward(m,d)
q,v=tree.default_state();q[6]=.419
q0,_=tree.default_state();H0=tree.crba(q0);B=body_velocity_jacobians(tree,q0)
names={b.name:b.index for b in tree.bodies};weights={-1:0.}
for b in tree.bodies:
    jc=B[b.index,:3]-skew(b.inertia.com)@B[b.index,3:]
    weights[b.index]=np.trace(jc@np.linalg.solve(H0,jc.T))/3
def bid(mid):
    return -1 if mid==0 else names[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,mid)]
rows=[];R=[];aref=[]
for c in d.contact:
    i=bid(m.geom_bodyid[c.geom2]);j=bid(m.geom_bodyid[c.geom1])
    ji=tree.contact_jacobian(q,i,c.pos) if i>=0 else np.zeros((3,tree.nv))
    jj=tree.contact_jacobian(q,j,c.pos) if j>=0 else np.zeros((3,tree.nv))
    rows.append(c.frame[:3]@(ji-jj))
    # Penetration .001 reaches solimp width .001; reference defaults dw=.95.
    assert -c.dist >= .001-1e-15
    R.append((1-.95)/.95*(weights[i]+weights[j]))
    aref.append((-c.dist)/(.95*.02**2))
J=np.array(rows);H=tree.crba(q);A=J@np.linalg.solve(H,J.T)
M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,M)
Jref=d.efc_J.reshape(d.nefc,m.nv);Aref=Jref@np.linalg.solve(M,Jref.T)
au=tree.aba(q,v,np.zeros(tree.nv));rhs=np.array(aref)-J@au
f=np.linalg.solve(A+np.diag(R),rhs)
assert np.all(f>0)
np.testing.assert_allclose(J,Jref,atol=1e-12)
np.testing.assert_allclose(A,Aref,rtol=1e-11,atol=1e-12)
np.testing.assert_allclose(R,d.efc_R,rtol=1e-11)
np.testing.assert_allclose(aref,d.efc_aref,rtol=1e-11)
np.testing.assert_allclose(rhs,d.efc_aref-Jref@d.qacc_smooth,rtol=1e-11)
np.testing.assert_allclose(f,d.efc_force,rtol=1e-10,atol=1e-10)
np.testing.assert_allclose(au+np.linalg.solve(H,J.T@f),d.qacc,rtol=1e-10,atol=1e-10)
print('quad_frozen', 'ncon',d.ncon,'max_offdiag_A',float(np.max(np.abs(A-np.diag(np.diag(A))))),'R',R,'force',f.tolist(),'max_force_error',float(np.max(abs(f-d.efc_force))))
print('PASS')
