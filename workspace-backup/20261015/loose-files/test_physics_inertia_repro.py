"""Analytical counterexamples for articulated CPU PGS contact mobility."""
import numpy as np
import pytest

from physics.cpu_engine import CpuEngine
from physics.geometry import BodyCollisionGeometry, ShapeInstance, SphereShape
from physics.implicit_contact_step import ImplicitContactStep
from physics.joint import FixedJoint, FreeJoint, RevoluteJoint
from physics.merged_model import merge_models
from physics.robot_tree import Body, RobotTreeNumpy
from physics.solvers.pgs_solver import ContactConstraint, PGSContactSolver
from physics.solvers.pgs_split_impulse import PGSSplitImpulseSolver
from physics.spatial import SpatialInertia, SpatialTransform
from physics.step_pipeline import StepPipeline
from physics.constraint_solvers import VelocityLevelAdapter
from robot.model import RobotModel

SOLVERS = [PGSContactSolver, PGSSplitImpulseSolver]

def composite(child_mass, offset=None, com=None):
    tree = RobotTreeNumpy(gravity=0.)
    offset = np.zeros(3) if offset is None else np.asarray(offset)
    com = np.zeros(3) if com is None else np.asarray(com)
    for i, mass in enumerate([1-child_mass, child_mass]):
        tree.add_body(Body(
            name=f'b{i}', index=i, parent=i-1,
            joint=FreeJoint('root') if i==0 else FixedJoint('weld'),
            inertia=SpatialInertia(mass, np.eye(3)*.01, com),
            X_tree=SpatialTransform.from_translation(np.zeros(3) if i==0 else offset),
        ))
    tree.finalize()
    return RobotModel(tree=tree, contact_body_names=['b1'],
        geometries=[BodyCollisionGeometry(1,[ShapeInstance(SphereShape(.1))])])

def solver(cls, **kwargs):
    return cls(max_iter=500, tolerance=1e-14, erp=0., cfm=0., **kwargs)

def contact(i, j, point, normal, condim=1, **kwargs):
    return ContactConstraint(body_i=i, body_j=j, point=np.asarray(point,dtype=float),
        normal=np.asarray(normal,dtype=float), tangent1=np.zeros(3), tangent2=np.zeros(3),
        depth=.001, mu=0., condim=condim, **kwargs)

@pytest.mark.parametrize('cls', SOLVERS)
@pytest.mark.parametrize('child_mass', [.05,.5,.95])
def test_fixed_child_mass_partition_does_not_change_impact(cls, child_mass):
    merged=merge_models({'r':composite(child_mass)})
    engine=CpuEngine(merged,solver=solver(cls),dt=.001)
    q,v=merged.tree.default_state(); q[6]=.099; v[2]=-1.
    out=engine.step(q,v,np.zeros(merged.nv))
    assert len(engine._last_contacts)==1
    np.testing.assert_allclose(out.qdot_new, np.zeros(6), atol=1e-10, rtol=0)
    # Total mass 1 kg: delta momentum must be +1 N s in the normal direction.
    assert abs((out.qdot_new[2]-v[2])-1.)<1e-10

@pytest.mark.parametrize('cls', SOLVERS)
def test_legacy_articulated_entrypoint_uses_same_inertia(cls):
    tree=composite(.05).tree
    q,v=tree.default_state(); q[6]=.099; v[2]=-1
    c=contact(1,-1,[0,0,-.001],[0,0,1])
    step=ImplicitContactStep(.001,solver(cls))
    _,new_v=step.step(tree,q,v,np.zeros(tree.nv),contacts=[c])
    np.testing.assert_allclose(new_v,0,atol=1e-10,rtol=0)

@pytest.mark.parametrize('cls', SOLVERS)
def test_revolute_contact_removes_incoming_velocity(cls):
    tree=RobotTreeNumpy(gravity=0.)
    tree.add_body(Body(name='link', index=0,parent=-1,joint=RevoluteJoint('hinge',axis=np.array([0.,1.,0.])),
        inertia=SpatialInertia(1.,np.eye(3)*.1,np.array([0.,0.,-.5])),X_tree=SpatialTransform.identity()))
    tree.finalize()
    q,v=tree.default_state(); v[0]=1.
    c=contact(0,-1,[0,0,-1],[1,0,0])
    _,new_v=StepPipeline(.001,constraint_solver=VelocityLevelAdapter(solver(cls))).step(tree,q,v,np.zeros(1),[c])
    # I_y about hinge=.1+1*.5^2=.35; unit moment arm, impulse=.35 N s.
    assert abs(new_v[0])<1e-10

@pytest.mark.parametrize('cls', SOLVERS)
def test_offset_composite_impact_matches_rigid_body_formula(cls):
    model=composite(.2,offset=[.1,0,0],com=[.02,.01,0])
    merged=merge_models({'r':model})
    merged.terrain.mu=0.
    engine=CpuEngine(merged,solver=solver(cls),dt=.001)
    q,v=merged.tree.default_state()
    angle=.7;q[:4]=[np.cos(angle/2),0,0,np.sin(angle/2)];q[6]=.099;v[2]=-1
    # Analytic composite inertia about CoM; contact relative to CoM in base frame.
    c1=np.array([.02,.01,0]);c2=c1+np.array([.1,0,0]);center=.8*c1+.2*c2
    inertia=np.eye(3)*.02
    for m,c in [(.8,c1),(.2,c2)]:
        r=c-center;inertia+=m*(r@r*np.eye(3)-np.outer(r,r))
    arm=np.array([.1,0,-.1])-center
    normal=np.array([0.,0.,1.]);moment=np.cross(arm,normal)
    impulse=1./(1.+moment@np.linalg.solve(inertia,moment))
    omega=np.linalg.solve(inertia,moment*impulse)
    v_com=np.array([0.,0.,-1.+impulse])
    rot=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    v_origin=rot@(v_com-np.cross(omega,center))
    expected=np.r_[v_origin,omega]
    out=engine.step(q,v,np.zeros(6))
    np.testing.assert_allclose(out.qdot_new,expected,atol=1e-9,rtol=0)

@pytest.mark.parametrize('cls', SOLVERS)
@pytest.mark.parametrize('condim',[1,3,4,6])
def test_two_roots_conserve_momentum_with_angular_rows(cls,condim):
    merged=merge_models({'a':composite(.05),'b':composite(.8)})
    tree=merged.tree;q,v=tree.default_state()
    # Both origins and contact coincide: isolated normal + angular rows.
    v[2]=-1.;v[8]=1.
    if condim>=4:v[5]=2.;v[11]=-2.
    if condim==6:v[3]=.2;v[9]=-.2
    c=contact(1,3,[0,0,0],[0,0,1],condim,mu_spin=1.,mu_roll=1.)
    _,new=StepPipeline(.001,constraint_solver=VelocityLevelAdapter(solver(cls,solimp=(1.,1.,.001,.5,2.)))).step(tree,q,v,np.zeros(12),[c])
    np.testing.assert_allclose(new,0.,atol=1e-10,rtol=0)
