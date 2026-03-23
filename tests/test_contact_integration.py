"""
Cross-module integration tests for the Phase 2f contact system.

Tests the full pipeline: URDF → RobotModel → Simulator/LCPContactModel → step.
Also compares Penalty vs LCP contact models on the same scenario.
"""

from __future__ import annotations

import os
import tempfile
import textwrap

import numpy as np
import pytest

from physics.contact import LCPContactModel, PenaltyContactModel, ContactParams, ContactPoint
from physics.geometry import BoxShape, SphereShape, CapsuleShape
from physics.gjk_epa import ground_contact_query, gjk_epa_query, ContactManifold
from physics.lcp_solver import ContactConstraint, PGSContactSolver
from physics.spatial import SpatialTransform
from robot import load_urdf


# ---------------------------------------------------------------------------
# GJK/EPA → ContactConstraint → PGS pipeline
# ---------------------------------------------------------------------------


class TestGJKToLCPPipeline:
    """Full pipeline: collision detection → constraint building → LCP solve."""

    def test_sphere_drop_pipeline(self):
        """Sphere falling onto ground: GJK detects → PGS resolves → upward impulse."""
        sphere = SphereShape(0.5)
        pose = SpatialTransform.from_translation(np.array([0, 0, 0.3]))
        v_body = np.array([0, 0, -2.0, 0, 0, 0])

        # Step 1: Collision detection
        manifold = ground_contact_query(sphere, pose, ground_z=0.0)
        assert manifold is not None

        # Step 2: Build constraints
        constraints = []
        for pt in manifold.points:
            constraints.append(ContactConstraint(
                body_i=0, body_j=-1,
                point=pt, normal=manifold.normal,
                tangent1=np.zeros(3), tangent2=np.zeros(3),
                depth=manifold.depth, mu=0.5,
            ))

        # Step 3: Solve
        solver = PGSContactSolver(max_iter=30)
        impulses = solver.solve(
            constraints, [v_body], [pose],
            [1.0], [np.eye(3) * 5.0], dt=1e-3,
        )

        # Step 4: Verify
        assert impulses[0][2] > 0, "Pipeline should produce upward impulse"

    def test_capsule_on_ground_pipeline(self):
        """Capsule on ground — full pipeline."""
        cap = CapsuleShape(0.3, 1.0)
        pose = SpatialTransform.from_translation(np.array([0, 0, 0.5]))
        v_body = np.array([0, 0, -1.0, 0, 0, 0])

        manifold = ground_contact_query(cap, pose, ground_z=0.0)
        assert manifold is not None

        constraints = [ContactConstraint(
            body_i=0, body_j=-1,
            point=manifold.points[0], normal=manifold.normal,
            tangent1=np.zeros(3), tangent2=np.zeros(3),
            depth=manifold.depth, mu=0.8,
        )]

        solver = PGSContactSolver(max_iter=30)
        impulses = solver.solve(
            constraints, [v_body], [pose],
            [2.0], [np.eye(3) * 3.0], dt=1e-3,
        )
        assert impulses[0][2] > 0


# ---------------------------------------------------------------------------
# LCPContactModel with multiple bodies
# ---------------------------------------------------------------------------


class TestLCPMultiBody:
    def test_two_contact_bodies(self):
        model = LCPContactModel(mu=0.5, max_iter=30)
        model.add_contact_body(0, SphereShape(0.5), "ball1")
        model.add_contact_body(1, SphereShape(0.3), "ball2")

        X = [
            SpatialTransform.from_translation(np.array([0, 0, 0.3])),
            SpatialTransform.from_translation(np.array([2, 0, 0.2])),
        ]
        v = [
            np.array([0, 0, -1.0, 0, 0, 0]),
            np.array([0, 0, -0.5, 0, 0, 0]),
        ]

        forces = model.compute_forces(X, v, num_bodies=2)
        assert forces[0][2] > 0, "Ball 1 should get upward force"
        assert forces[1][2] > 0, "Ball 2 should get upward force"


# ---------------------------------------------------------------------------
# Penalty vs LCP comparison
# ---------------------------------------------------------------------------


class TestPenaltyVsLCP:
    def test_both_produce_upward_force(self):
        """Both models should produce upward contact force for same scenario."""
        # Penalty model
        penalty = PenaltyContactModel(ContactParams(k_normal=5000, b_normal=500, mu=0.5))
        penalty.add_contact_point(ContactPoint(body_index=0, position=np.zeros(3), name="foot"))

        # LCP model
        lcp = LCPContactModel(mu=0.5, max_iter=30)
        lcp.add_contact_body(0, SphereShape(0.01), "foot")  # tiny sphere ≈ point

        # Same scenario: body at z=0 (touching ground), moving down
        X = [SpatialTransform.from_translation(np.array([0, 0, -0.005]))]
        v = [np.array([0, 0, -1.0, 0, 0, 0])]

        f_penalty = penalty.compute_forces(X, v, 1)
        f_lcp = lcp.compute_forces(X, v, 1)

        # Both should produce upward force (z > 0)
        assert f_penalty[0][2] > 0, f"Penalty z-force: {f_penalty[0][2]}"
        assert f_lcp[0][2] > 0, f"LCP z-force: {f_lcp[0][2]}"


# ---------------------------------------------------------------------------
# URDF friction parameter end-to-end
# ---------------------------------------------------------------------------


class TestURDFFrictionE2E:
    def test_friction_parsed_and_used(self):
        """URDF friction parameter should flow through to passive_torques."""
        urdf = """
        <robot name="test">
          <link name="base">
            <inertial><mass value="5.0"/><origin xyz="0 0 0"/>
              <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.05"/>
            </inertial>
          </link>
          <link name="arm">
            <inertial><mass value="1.0"/><origin xyz="0 0 -0.1"/>
              <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
            </inertial>
          </link>
          <joint name="shoulder" type="revolute">
            <parent link="base"/><child link="arm"/>
            <origin xyz="0.1 0 0"/><axis xyz="0 1 0"/>
            <limit lower="-3.14" upper="3.14" effort="100"/>
            <dynamics damping="0.1" friction="0.5"/>
          </joint>
        </robot>
        """
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
        f.write(textwrap.dedent(urdf))
        f.close()

        model = load_urdf(f.name, floating_base=True, contact_links=[])
        os.unlink(f.name)

        tree = model.tree
        q, qdot = tree.default_state()
        # Give the revolute joint some velocity
        qdot[6] = 1.0  # revolute joint (after 6 DOF free joint)

        tau = tree.passive_torques(q, qdot)

        # Should have friction + damping opposing velocity
        # damping: -0.1 * 1.0 = -0.1
        # friction: -0.5 * tanh(1.0 / 0.01) ≈ -0.5
        # total ≈ -0.6
        assert tau[6] < -0.1, f"Expected significant negative torque from friction+damping, got {tau[6]}"

        # Verify friction is actually contributing (not just damping)
        # Check that the joint has non-zero friction
        from physics.joint import RevoluteJoint
        rev_joint = None
        for b in tree.bodies:
            if isinstance(b.joint, RevoluteJoint) and b.joint.friction > 0:
                rev_joint = b.joint
                break
        assert rev_joint is not None, "Should find a RevoluteJoint with friction > 0"
        assert rev_joint.friction == pytest.approx(0.5)
