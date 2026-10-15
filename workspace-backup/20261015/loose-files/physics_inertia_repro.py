import numpy as np
from physics.cpu_engine import CpuEngine
from physics.geometry import BodyCollisionGeometry, SphereShape, ShapeInstance
from physics.joint import FreeJoint, FixedJoint
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from physics.merged_model import merge_models
from physics.solvers.pgs_solver import PGSContactSolver
from physics.solvers.pgs_split_impulse import PGSSplitImpulseSolver
from robot.model import RobotModel
for solver_type in [PGSContactSolver, PGSSplitImpulseSolver]:
    for child_mass in [.05,.5,.95]:
        tree=RobotTreeNumpy(gravity=0.)
        for i,m in enumerate([1-child_mass,child_mass]):
            tree.add_body(Body(name=f'b{i}',index=i,joint=FreeJoint('root') if i==0 else FixedJoint('weld'),inertia=SpatialInertia(m,np.eye(3)*.01,np.zeros(3)),X_tree=SpatialTransform.identity(),parent=i-1))
        tree.finalize()
        model=RobotModel(tree=tree,contact_body_names=['b1'],geometries=[BodyCollisionGeometry(1,[ShapeInstance(SphereShape(.1))])])
        merged=merge_models({'r':model})
        merged.tree.gravity=0.
        engine=CpuEngine(merged,solver=solver_type(erp=0,cfm=0,tolerance=1e-12),dt=.001)
        q,v=merged.tree.default_state();q[6]=.099;v[2]=-1
        out=engine.step(q,v,np.zeros(merged.nv))
        print(solver_type.__name__,child_mass,'vz_after',out.qdot_new[2],'contacts',len(engine._last_contacts),flush=True)
