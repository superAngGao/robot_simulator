"""
Multi-step trajectory comparison against PyBullet PGS.

Reference: PyBullet (Bullet btSequentialImpulseConstraintSolver).
Both use Projected Gauss-Seidel on hard constraints with similar parameters.

Why Bullet as trajectory reference (not single-step impulse reference):
  - Bullet's internal Baumgarte/ERP formulation differs from ours, so single-step
    impulse magnitudes don't match exactly.
  - Over multiple steps, both solvers converge to the same physical behavior:
    free fall matches kinematics, contact arrests penetration, body settles
    at the correct rest position.
  - Trajectory L2 error quantifies accumulated solver difference over time.

Why not MuJoCo:
  - MuJoCo uses soft constraints + implicit integration — fundamentally different
    contact model. Trajectory shape differs (softer bounce, slower settling).

Scenario: 1 kg sphere (r=0.1 m) dropped from z=0.3 m onto ground plane.
  - Free fall phase: ~200 steps (0.2 s), governed by kinematics
  - Contact phase: sphere touches ground at z=0.1, decelerates
  - Settling phase: sphere rests at z≈0.1

Expected tolerances:
  - Free fall: L2 < 0.001 m (only integrator difference)
  - First contact timing: ±2 steps
  - Rest position: |Δz| < 0.005 m
  - Rest velocity: |vz| < 0.01 m/s
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import pybullet as p
    import pybullet_data

    HAS_BULLET = True
except ImportError:
    HAS_BULLET = False

from physics.collision import NullSelfCollision
from physics.contact import LCPContactModel
from physics.geometry import SphereShape
from physics.integrator import SemiImplicitEuler
from physics.joint import FreeJoint
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from robot.model import RobotModel
from simulator import Simulator

pytestmark = pytest.mark.skipif(not HAS_BULLET, reason="pybullet not installed")

N_STEPS = 500
DT = 0.001
Z0 = 0.3
RADIUS = 0.1


def _bullet_trajectory():
    """Run Bullet PGS and return (z, vz) arrays."""
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    p.setTimeStep(DT, physicsClientId=cid)
    p.setPhysicsEngineParameter(numSolverIterations=30, physicsClientId=cid)

    p.loadURDF("plane.urdf", physicsClientId=cid)
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=RADIUS, physicsClientId=cid)
    body = p.createMultiBody(
        baseMass=1.0, baseCollisionShapeIndex=col, basePosition=[0, 0, Z0], physicsClientId=cid
    )
    p.changeDynamics(
        body,
        -1,
        localInertiaDiagonal=[0.004, 0.004, 0.004],
        lateralFriction=0.5,
        restitution=0.0,
        physicsClientId=cid,
    )

    z = np.zeros(N_STEPS)
    vz = np.zeros(N_STEPS)
    for i in range(N_STEPS):
        pos, _ = p.getBasePositionAndOrientation(body, physicsClientId=cid)
        vel, _ = p.getBaseVelocity(body, physicsClientId=cid)
        z[i] = pos[2]
        vz[i] = vel[2]
        p.stepSimulation(physicsClientId=cid)
    p.disconnect(cid)
    return z, vz


def _our_lcp_trajectory():
    """Run our LCP Simulator and return (z, vz) arrays."""
    tree = RobotTreeNumpy(gravity=9.81)
    b = Body(
        name="ball",
        index=0,
        joint=FreeJoint("root"),
        inertia=SpatialInertia(mass=1.0, inertia=np.diag([0.004] * 3), com=np.zeros(3)),
        X_tree=SpatialTransform.identity(),
        parent=-1,
    )
    tree.add_body(b)
    tree.finalize()

    lcp = LCPContactModel(mu=0.5, condim=3, max_iter=30, erp=0.2)
    lcp.add_contact_body(0, SphereShape(RADIUS), "ball")
    model = RobotModel(tree=tree, contact_model=lcp, self_collision=NullSelfCollision())
    sim = Simulator(model, SemiImplicitEuler(dt=DT))

    q, qdot = tree.default_state()
    q[3] = 1.0
    q[6] = Z0
    tau = np.zeros(tree.nv)

    z = np.zeros(N_STEPS)
    vz = np.zeros(N_STEPS)
    for i in range(N_STEPS):
        z[i] = q[6]
        vz[i] = qdot[2]
        q, qdot = sim.step(q, qdot, tau)
    return z, vz


@pytest.fixture(scope="module")
def trajectories():
    bz, bvz = _bullet_trajectory()
    oz, ovz = _our_lcp_trajectory()
    return bz, bvz, oz, ovz


class TestFreeFallPhase:
    """Before contact (~first 190 steps), both should match kinematics.

    Reference: Bullet PGS trajectory + analytical kinematics z = z0 - 0.5*g*t^2.
    Free fall is integrator-only (no contact), so differences come solely
    from semi-implicit Euler vs Bullet's integrator.
    """

    def test_free_fall_l2_position(self, trajectories):
        """Position L2 error during free fall should be < 1 mm."""
        bz, _, oz, _ = trajectories
        # Free fall ends around step 190
        l2 = np.sqrt(np.mean((bz[:190] - oz[:190]) ** 2))
        assert l2 < 0.001, f"Free fall L2 position error = {l2:.6f} m"

    def test_free_fall_matches_kinematics(self, trajectories):
        """Both should match z = z0 - 0.5*g*t^2 during free fall."""
        bz, _, oz, _ = trajectories
        t = np.arange(190) * DT
        z_analytical = Z0 - 0.5 * 9.81 * t**2
        assert np.max(np.abs(bz[:190] - z_analytical)) < 0.002
        assert np.max(np.abs(oz[:190] - z_analytical)) < 0.002


class TestContactTiming:
    """First contact should occur at the kinematically predicted time.

    Reference: analytical t = sqrt(2*(z0-r)/g) = sqrt(2*0.2/9.81) ≈ 0.2019 s → step ~202.
    Both Bullet and our solver should detect contact within ±5 steps of this.
    """

    def test_first_contact_timing(self, trajectories):
        bz, _, oz, _ = trajectories
        analytical_step = int(np.sqrt(2 * (Z0 - RADIUS) / 9.81) / DT)

        def first_below(arr, thresh):
            for i, v in enumerate(arr):
                if v < thresh:
                    return i
            return -1

        bc = first_below(bz, RADIUS + 0.005)
        oc = first_below(oz, RADIUS + 0.005)

        assert abs(bc - analytical_step) <= 5, f"Bullet contact at {bc}, expected ~{analytical_step}"
        assert abs(oc - analytical_step) <= 5, f"Ours contact at {oc}, expected ~{analytical_step}"
        assert abs(bc - oc) <= 2, f"Timing mismatch: Bullet={bc}, Ours={oc}"


class TestSettling:
    """After contact, sphere should settle at z ≈ radius (0.1 m) with vz ≈ 0.

    Reference: Bullet PGS final state + physical expectation.
    Both hard-constraint PGS solvers should prevent penetration and
    bring the sphere to rest at the ground surface.
    """

    def test_rest_position(self, trajectories):
        bz, _, oz, _ = trajectories
        # Last 50 steps should be near rest
        assert abs(bz[-1] - RADIUS) < 0.005, f"Bullet rest z={bz[-1]:.4f}"
        assert abs(oz[-1] - RADIUS) < 0.005, f"Ours rest z={oz[-1]:.4f}"

    def test_rest_velocity(self, trajectories):
        _, bvz, _, ovz = trajectories
        assert abs(bvz[-1]) < 0.01, f"Bullet rest vz={bvz[-1]:.6f}"
        assert abs(ovz[-1]) < 0.01, f"Ours rest vz={ovz[-1]:.6f}"

    def test_no_penetration(self, trajectories):
        """Sphere center should never go below radius (ground surface)."""
        bz, _, oz, _ = trajectories
        assert np.min(bz) > RADIUS - 0.01, f"Bullet min z={np.min(bz):.4f}"
        assert np.min(oz) > RADIUS - 0.01, f"Ours min z={np.min(oz):.4f}"


class TestFullTrajectoryAgreement:
    """Full trajectory L2 error between Bullet and our LCP.

    Reference: Bullet PGS as the established hard-constraint solver.
    The L2 error measures accumulated differences in integrator + solver
    over the full simulation. Expected to be small (< 5 mm) since both
    use PGS with similar iteration counts.
    """

    def test_position_l2(self, trajectories):
        bz, _, oz, _ = trajectories
        l2 = np.sqrt(np.mean((bz - oz) ** 2))
        assert l2 < 0.005, f"Full trajectory L2 = {l2:.6f} m"
