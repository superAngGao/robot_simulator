"""Tests for LCPContactModel, CapsuleShape, and restitution."""

from __future__ import annotations

import numpy as np
import pytest

from physics.contact import LCPContactModel
from physics.geometry import BoxShape, CapsuleShape, SphereShape
from physics.lcp_solver import ContactConstraint, PGSContactSolver
from physics.spatial import SpatialTransform


class TestCapsuleShape:
    def test_half_extents(self):
        cap = CapsuleShape(radius=0.5, length=2.0)
        he = cap.half_extents_approx()
        np.testing.assert_allclose(he, [0.5, 0.5, 1.5])  # r, r, half_len + r

    def test_support_up(self):
        cap = CapsuleShape(radius=0.5, length=2.0)
        s = cap.support_point(np.array([0, 0, 1.0]))
        # Top hemisphere center at z=1.0, plus radius 0.5 upward
        assert abs(s[2] - 1.5) < 1e-6

    def test_support_down(self):
        cap = CapsuleShape(radius=0.5, length=2.0)
        s = cap.support_point(np.array([0, 0, -1.0]))
        assert abs(s[2] - (-1.5)) < 1e-6

    def test_support_sideways(self):
        cap = CapsuleShape(radius=0.5, length=2.0)
        s = cap.support_point(np.array([1, 0, 0.0]))
        assert abs(s[0] - 0.5) < 1e-6
        # z = 0 ideally, but sign(0) = 0 so segment contributes 0
        # Actually segment picks half_len if dz >= 0 (dz=0 → half_len)
        # This is fine for GJK — any rim point is valid support

    def test_support_diagonal(self):
        cap = CapsuleShape(radius=1.0, length=2.0)
        d = np.array([1, 0, 1.0])
        s = cap.support_point(d)
        # Segment support: z = +1.0 (half_length)
        # Sphere support: radius * normalize(d) = [0.707, 0, 0.707]
        assert s[2] > 1.0  # above half_length
        assert s[0] > 0.0

    def test_ground_contact(self):
        from physics.gjk_epa import ground_contact_query
        cap = CapsuleShape(radius=0.5, length=1.0)
        # Place capsule so bottom touches ground
        pose = SpatialTransform.from_translation(np.array([0, 0, 0.8]))
        result = ground_contact_query(cap, pose, ground_z=0.0)
        # Lowest point: 0.8 - 0.5 - 0.5 = -0.2 → depth 0.2
        assert result is not None
        assert abs(result.depth - 0.2) < 0.01


class TestRestitution:
    def test_zero_restitution(self):
        solver = PGSContactSolver(max_iter=50)
        c = ContactConstraint(
            body_i=0, body_j=-1,
            point=np.array([0, 0, 0.0]),
            normal=np.array([0, 0, 1.0]),
            tangent1=np.zeros(3), tangent2=np.zeros(3),
            depth=0.01, mu=0.5, restitution=0.0,
        )
        body_v = [np.array([0, 0, -2.0, 0, 0, 0])]
        body_X = [SpatialTransform.from_translation(np.array([0, 0, 0.5]))]
        imp = solver.solve([c], body_v, body_X, [1.0], [np.eye(3)], dt=1e-3)
        imp_z = imp[0][2]
        assert imp_z > 0

    def test_restitution_bounce(self):
        """Restitution should increase total impulse (bounce = stop + push back)."""
        # Use erp=0 to isolate restitution from Baumgarte
        solver = PGSContactSolver(max_iter=50, erp=0.0)
        c = ContactConstraint(
            body_i=0, body_j=-1,
            point=np.array([0, 0, 0.0]),
            normal=np.array([0, 0, 1.0]),
            tangent1=np.zeros(3), tangent2=np.zeros(3),
            depth=0.001, mu=0.5, restitution=0.8,
        )
        body_v = [np.array([0, 0, -2.0, 0, 0, 0])]
        body_X = [SpatialTransform.from_translation(np.array([0, 0, 0.5]))]
        imp_bounce = solver.solve([c], body_v, body_X, [1.0], [np.eye(3)], dt=1e-3)

        solver2 = PGSContactSolver(max_iter=50, erp=0.0)
        c2 = ContactConstraint(
            body_i=0, body_j=-1,
            point=np.array([0, 0, 0.0]),
            normal=np.array([0, 0, 1.0]),
            tangent1=np.zeros(3), tangent2=np.zeros(3),
            depth=0.001, mu=0.5, restitution=0.0,
        )
        imp_no_bounce = solver2.solve([c2], body_v, body_X, [1.0], [np.eye(3)], dt=1e-3)

        assert imp_bounce[0][2] > imp_no_bounce[0][2], \
            f"Restitution should increase impulse: bounce={imp_bounce[0][2]:.4f} vs no_bounce={imp_no_bounce[0][2]:.4f}"


class TestLCPContactModel:
    def test_sphere_on_ground(self):
        model = LCPContactModel(mu=0.5, max_iter=30)
        sphere = SphereShape(0.5)
        model.add_contact_body(0, sphere, "ball")

        # Sphere at z=0.3 → penetrates by 0.2
        X = [SpatialTransform.from_translation(np.array([0, 0, 0.3]))]
        v = [np.array([0, 0, -1.0, 0, 0, 0])]

        forces = model.compute_forces(X, v, num_bodies=1)
        # Should have upward force
        assert forces[0][2] > 0, f"Expected upward force, got z={forces[0][2]}"

    def test_no_penetration_no_force(self):
        model = LCPContactModel(mu=0.5, max_iter=30)
        sphere = SphereShape(0.5)
        model.add_contact_body(0, sphere, "ball")

        X = [SpatialTransform.from_translation(np.array([0, 0, 2.0]))]
        v = [np.zeros(6)]

        forces = model.compute_forces(X, v, num_bodies=1)
        np.testing.assert_allclose(forces[0], 0, atol=1e-10)

    def test_active_contacts(self):
        model = LCPContactModel(mu=0.5)
        sphere = SphereShape(0.5)
        model.add_contact_body(0, sphere, "ball")

        X = [SpatialTransform.from_translation(np.array([0, 0, 0.3]))]
        active = model.active_contacts(X)
        assert len(active) == 1
        assert active[0][0] == "ball"

    def test_box_contact(self):
        model = LCPContactModel(mu=0.8)
        box = BoxShape((0.5, 0.5, 0.5))
        model.add_contact_body(0, box, "foot")

        X = [SpatialTransform.from_translation(np.array([0, 0, 0.2]))]
        v = [np.array([0, 0, -0.5, 0, 0, 0])]
        forces = model.compute_forces(X, v, num_bodies=1)
        assert forces[0][2] > 0
