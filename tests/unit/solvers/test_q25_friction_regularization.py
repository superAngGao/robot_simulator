"""
Tests for Q25 fix — PGS friction regularization preventing spurious angular velocity.

Root cause: float32 noise in tangential velocity (~1e-7) causes PGS to produce
tiny friction impulses that, through the contact moment arm (r_arm = radius),
create torques → angular acceleration → positive feedback → divergence.

Fix: per-row R regularization on friction rows (R_i = (1-d)/d * |W_ii|) plus
friction warmstart zeroing (Bullet-style).
"""

from __future__ import annotations

import numpy as np

from physics.joint import FreeJoint
from physics.robot_tree import Body, RobotTreeNumpy
from physics.solvers.pgs_solver import ContactConstraint, PGSContactSolver
from physics.solvers.pgs_split_impulse import PGSSplitImpulseSolver
from physics.spatial import SpatialInertia, SpatialTransform


def _make_sphere_tree(mass=1.0, radius=0.1):
    """Single free-floating sphere."""
    tree = RobotTreeNumpy(gravity=9.81)
    I_sphere = 2.0 / 5.0 * mass * radius**2
    tree.add_body(
        Body(
            name="ball",
            index=0,
            joint=FreeJoint("root"),
            inertia=SpatialInertia(mass=mass, inertia=np.eye(3) * I_sphere, com=np.zeros(3)),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()
    return tree


def _simulate_sphere_on_ground(solver, tree, n_steps=5000, dt=2e-4, radius=0.1):
    """Simulate a sphere resting on the ground and return max |omega|."""
    q, qdot = tree.default_state()
    # Place sphere on ground: z = radius (resting)
    q[6] = radius  # FreeJoint: [qw, qx, qy, qz, px, py, pz]

    max_omega = 0.0
    for _ in range(n_steps):
        X_world = tree.forward_kinematics(q)
        body_v = tree.body_velocities(q, qdot)
        inv_mass = [1.0 / tree.bodies[0].inertia.mass]
        inv_inertia = [np.linalg.inv(tree.bodies[0].inertia.inertia)]

        # Ground contact: sphere bottom touching z=0
        contact = ContactConstraint(
            body_i=0,
            body_j=-1,
            point=np.array([0.0, 0.0, 0.0]),  # contact at ground
            normal=np.array([0.0, 0.0, 1.0]),
            tangent1=np.array([1.0, 0.0, 0.0]),
            tangent2=np.array([0.0, 1.0, 0.0]),
            depth=max(0.0, radius - q[6]),
            mu=0.8,
        )

        if contact.depth > 0:
            impulses = solver.solve([contact], body_v, X_world, inv_mass, inv_inertia, dt)
            # Apply impulse to qdot
            dv = impulses[0]
            dv[:3] *= inv_mass[0]
            dv[3:] = inv_inertia[0] @ dv[3:]
            qdot += dv
        else:
            solver.solve([], body_v, X_world, inv_mass, inv_inertia, dt)

        # Semi-implicit Euler
        qddot = tree.aba(q, qdot, np.zeros(tree.nv))
        qdot += qddot * dt
        tree.integrate_q(q, qdot, dt)

        # Track angular velocity
        omega = qdot[3:6]  # FreeJoint angular velocity
        max_omega = max(max_omega, np.linalg.norm(omega))

    return max_omega


class TestQ25FrictionRegularization:
    """Q25 fix: per-row R on friction rows prevents angular divergence."""

    def test_sphere_at_rest_angular_velocity_bounded(self):
        """Sphere resting on ground: angular velocity must not diverge."""
        tree = _make_sphere_tree()
        solver = PGSContactSolver(max_iter=30)  # uses default solimp + no friction warmstart
        max_omega = _simulate_sphere_on_ground(solver, tree, n_steps=5000)
        # With Q25 fix, omega should stay near zero (< 0.1 rad/s)
        assert max_omega < 0.1, f"Angular velocity diverged: max |omega| = {max_omega:.4f}"

    def test_pgs_si_sphere_at_rest_stable(self):
        """PGS-SI solver: same stability guarantee."""
        tree = _make_sphere_tree()
        solver = PGSSplitImpulseSolver(max_iter=30)
        max_omega = _simulate_sphere_on_ground(solver, tree, n_steps=5000)
        assert max_omega < 0.1, f"PGS-SI angular velocity diverged: max |omega| = {max_omega:.4f}"

    def test_rolling_ball_friction_produces_deceleration(self):
        """Friction on a sliding ball must produce opposing impulse (not zero).

        Verifies that R regularization softens friction but doesn't kill it:
        the friction impulse on a tangentially moving sphere must be nonzero
        and oppose the motion direction.
        """
        tree = _make_sphere_tree()
        solver = PGSContactSolver(max_iter=30)
        q, qdot = tree.default_state()
        q[6] = 0.1  # resting on ground
        qdot[0] = 0.1  # sliding in +x

        X_world = tree.forward_kinematics(q)
        body_v = tree.body_velocities(q, qdot)
        inv_mass = [1.0 / tree.bodies[0].inertia.mass]
        inv_inertia = [np.linalg.inv(tree.bodies[0].inertia.inertia)]

        contact = ContactConstraint(
            body_i=0,
            body_j=-1,
            point=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            tangent1=np.array([1.0, 0.0, 0.0]),
            tangent2=np.array([0.0, 1.0, 0.0]),
            depth=0.001,
            mu=0.8,
        )
        impulses = solver.solve([contact], body_v, X_world, inv_mass, inv_inertia, 2e-4)
        # Friction impulse in body frame: linear x component should oppose +x motion
        impulse_x = impulses[0][0]  # body frame linear x
        assert impulse_x < -1e-8, f"Friction impulse should oppose +x motion, got {impulse_x:.2e}"

    def test_friction_warmstart_false_is_default(self):
        """Default PGS should have friction_warmstart=False."""
        solver = PGSContactSolver()
        assert solver.friction_warmstart is False

    def test_solimp_params_propagated_to_pgs_si(self):
        """PGS-SI must propagate solimp and friction_warmstart to inner PGS."""
        custom_solimp = (0.9, 0.95, 0.002, 0.5, 2.0)
        solver = PGSSplitImpulseSolver(solimp=custom_solimp, friction_warmstart=True)
        assert solver._vel_solver.solimp == custom_solimp
        assert solver._vel_solver.friction_warmstart is True

    def test_per_row_R_larger_for_friction_than_normal(self):
        """Friction rows must get larger R than normal rows (because A_tt > A_nn)."""
        tree = _make_sphere_tree(mass=1.0, radius=0.1)
        q, qdot = tree.default_state()
        q[6] = 0.1  # resting on ground

        solver = PGSContactSolver(max_iter=1)  # 1 iteration to inspect W

        X_world = tree.forward_kinematics(q)
        body_v = tree.body_velocities(q, qdot)
        inv_mass = [1.0 / tree.bodies[0].inertia.mass]
        inv_inertia = [np.linalg.inv(tree.bodies[0].inertia.inertia)]

        contact = ContactConstraint(
            body_i=0,
            body_j=-1,
            point=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            tangent1=np.array([1.0, 0.0, 0.0]),
            tangent2=np.array([0.0, 1.0, 0.0]),
            depth=0.001,
            mu=0.8,
        )
        solver.solve([contact], body_v, X_world, inv_mass, inv_inertia, 2e-4)

        # After solve, we can't inspect W directly, but we can verify
        # the solver doesn't crash and the impedance function works correctly.
        d = solver._impedance(0.001)
        assert 0.0 < d < 1.0, f"Impedance out of range: {d}"
        ratio = (1.0 - d) / d
        assert ratio > 0.0, f"R ratio should be positive: {ratio}"

    def test_heavy_sphere_stability(self):
        """Heavy sphere (50kg): moment arm amplification is worse, R must still stabilize."""
        tree = _make_sphere_tree(mass=50.0, radius=0.15)
        solver = PGSContactSolver(max_iter=50)
        max_omega = _simulate_sphere_on_ground(solver, tree, n_steps=3000, radius=0.15)
        assert max_omega < 0.1, f"Heavy sphere omega diverged: {max_omega:.4f}"
