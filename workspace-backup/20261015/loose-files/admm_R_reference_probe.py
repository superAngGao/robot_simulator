import sys
sys.path.insert(0, '/data2/shared/ang.gao/robot_sim_physics_correctness')
import numpy as np
import mujoco
from physics.dynamics_utils import body_velocity_jacobians
from physics.spatial import skew
from tests.validation.rigid_body.models import build_quadruped, build_double_pendulum
for builder in (build_double_pendulum, build_quadruped):
    model, xml = builder()
    tree=model.tree
    q,_=tree.default_state()
    H=tree.crba(q)
    B=body_velocity_jacobians(tree,q)
    mm=mujoco.MjModel.from_xml_string(xml)
    for body in tree.bodies:
        J=B[body.index,:3]-skew(body.inertia.com)@B[body.index,3:]
        w=np.trace(J@np.linalg.solve(H,J.T))/3
        bid=mujoco.mj_name2id(mm,mujoco.mjtObj.mjOBJ_BODY,body.name)
        ref=mm.body_invweight0[bid,0]
        print(builder.__name__,body.name,w,ref,w-ref)
        np.testing.assert_allclose(w,ref,rtol=1e-11,atol=1e-13)
