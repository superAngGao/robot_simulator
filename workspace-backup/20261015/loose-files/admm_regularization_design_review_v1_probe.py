import sys

sys.path.insert(0, '/data2/shared/ang.gao/robot_sim_physics_correctness')
import mujoco
import numpy as np
from physics.robot_tree import Body, RobotTreeNumpy
from physics.joint import PrismaticJoint
from physics.spatial import SpatialInertia, SpatialTransform, skew
from physics.dynamics_utils import body_velocity_jacobians

print('MuJoCo', mujoco.__version__)
tree = RobotTreeNumpy()
tree.add_body(Body('slider', 0, PrismaticJoint('z'),
                   SpatialInertia(mass=2., inertia=np.eye(3)*.008, com=np.zeros(3)),
                   SpatialTransform(np.eye(3), np.array([0., 0., .09]))))
tree.finalize()
q, _ = tree.default_state()
H = tree.crba(q)
B = body_velocity_jacobians(tree, q)
J = B[0, :3] - skew(tree.bodies[0].inertia.com) @ B[0, 3:]
w = np.trace(J @ np.linalg.solve(H, J.T))/3
xml = '''<mujoco><option timestep=".002" cone="elliptic" solver="Newton" tolerance="1e-14"><flag diagexact="disable"/></option><default><geom condim="1" solimp=".9 .9 .001 .5 2" solref=".02 1"/></default><worldbody><geom type="plane" size="2 2 .1"/><body name="slider" pos="0 0 .09"><joint type="slide" axis="0 0 1"/><geom type="sphere" size=".1" mass="2"/></body></worldbody></mujoco>'''
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
R_proposal = (1-.9)/.9*w
A = .5
rhs = d.efc_aref[0] - d.efc_J.reshape(d.nefc, m.nv)[0] @ d.qacc_smooth
print('slider', dict(H=H.tolist(), J=J.tolist(), body_simple=int(m.body_simple[1]),
                     w_proposal=w, w_mujoco=m.body_invweight0[1, 0],
                     condim=int(d.contact[0].dim), R_proposal=R_proposal,
                     R_mujoco=d.efc_R[0], rhs=rhs,
                     force_proposal=rhs/(A+R_proposal), force_mujoco=d.efc_force[0]))
assert int(d.contact[0].dim) == 1
np.testing.assert_allclose(m.body_invweight0[1, 0], 3*w, rtol=1e-14)
np.testing.assert_allclose(d.efc_R[0], 3*R_proposal, rtol=1e-14)
for impratio in [1., 1e-15, 1e-16]:
    mx = mujoco.MjModel.from_xml_string(xml.replace('condim="1"', 'condim="3"').replace(
        '<joint type="slide" axis="0 0 1"/>', '<freejoint/>'))
    mx.opt.impratio = impratio
    dx = mujoco.MjData(mx)
    mujoco.mj_forward(mx, dx)
    print('impratio', impratio, 'actual_R', dx.efc_R.tolist(),
          'proposal_Rt', dx.efc_R[0]/impratio)
    np.testing.assert_allclose(dx.efc_R[1], dx.efc_R[0]/max(1e-15, impratio), rtol=1e-14)
