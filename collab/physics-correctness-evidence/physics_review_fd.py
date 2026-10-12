import os, sys
os.chdir('/data2/shared/ang.gao/robot_sim_physics_correctness')
sys.path.insert(0, os.getcwd())
import numpy as np
from tests.integration.contact_pipeline.test_articulated_pgs_inertia import composite, contact
from tests.validation.rigid_body.models import build_quadruped
from physics.merged_model import merge_models
from physics.solvers.pgs_solver import PGSContactSolver
from physics.dynamics_utils import body_velocity_jacobians, impulses_to_generalized
from physics.constraint_solvers import VelocityLevelAdapter
from physics.dynamics_cache import DynamicsCache

class Capture(PGSContactSolver):
    def _solve_rows(self, contacts, body_v, xs, dt, offsets, w):
        self.physical_w = w.copy()
        return super()._solve_rows(contacts, body_v, xs, dt, offsets, w)

rng = np.random.default_rng(4821)
model, _ = build_quadruped(contact=False)
tree = merge_models({'quad':model, 'weld':composite(.2,[.2,-.1,.3])}).tree
max_w = max_dual = max_b = 0.
for trial in range(3):
    q,v = tree.default_state()
    q = tree.integrate_q(q,rng.normal(size=tree.nv),.4)
    xs = tree.forward_kinematics(q)
    v[:] = rng.normal(size=tree.nv)
    b = body_velocity_jacobians(tree,q)
    for dim in (1,3,4,6):
        c = contact(3,tree.num_bodies-1,[.22,-.16,.08],np.array([.2,-.4,1])/np.linalg.norm([.2,-.4,1]),dim,mu_spin=.3,mu_roll=.2)
        c.mu=.7
        s=Capture(erp=0,cfm=1e-6,max_iter=1000,tolerance=1e-13)
        cache = DynamicsCache.from_tree(tree,q,v,.001,compute_H=True)
        cache.qacc_smooth=np.zeros(tree.nv)
        result=VelocityLevelAdapter(s).solve(tree,cache,np.zeros(tree.nv),[c])
        dirs=[c.normal,c.tangent1,c.tangent2]
        j=np.zeros((dim,tree.nv))
        eps=1e-6
        for col in range(tree.nv):
            e=np.zeros(tree.nv);e[col]=1
            plus=tree.forward_kinematics(tree.integrate_q(q,e,eps))
            minus=tree.forward_kinematics(tree.integrate_q(q,e,-eps))
            for bi,sign in ((c.body_i,1),(c.body_j,-1)):
                local=xs[bi].R.T@(c.point-xs[bi].r)
                vel=((plus[bi].r+plus[bi].R@local)-(minus[bi].r+minus[bi].R@local))/(2*eps)
                dr=(plus[bi].R-minus[bi].R)/(2*eps)@xs[bi].R.T
                omega=np.array([dr[2,1],dr[0,2],dr[1,0]])
                nlin=1 if dim==1 else 3
                for k in range(nlin):j[k,col]+=sign*dirs[k]@vel
                if dim>=4:j[3,col]+=sign*dirs[0]@omega
                if dim==6:
                    for k in (1,2):j[3+k,col]+=sign*dirs[k]@omega
        w=j@np.linalg.solve(cache.H,j.T)
        err=float(np.max(abs(w-s.physical_w)));max_w=max(max_w,err)
        np.testing.assert_allclose(s.physical_w,w,rtol=2e-8,atol=2e-8)
        cache.qdot = -np.linalg.solve(cache.H,j[0])
        result=VelocityLevelAdapter(s).solve(tree,cache,np.zeros(tree.nv),[c])
        lam=next(iter(s._warm_cache.values()))[0][1]
        assert lam[0] > 0.1
        applied=cache.H@(result*.001)
        err=float(np.max(abs(applied-j.T@lam)));max_dual=max(max_dual,err)
        np.testing.assert_allclose(applied,j.T@lam,rtol=2e-8,atol=2e-8)
print({'finite_difference_cases':12,'max_W_error':max_w,'max_J_transpose_impulse_error':max_dual})

