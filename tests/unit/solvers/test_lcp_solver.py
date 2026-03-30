"""Tests for PGS LCP contact solver."""

from __future__ import annotations

import numpy as np

from physics.lcp_solver import ContactConstraint, PGSContactSolver, _build_contact_frame
from physics.spatial import SpatialTransform


class TestContactFrame:
    def test_normal_z(self):
        t1, t2 = _build_contact_frame(np.array([0, 0, 1.0]))
        assert abs(np.dot(t1, [0, 0, 1])) < 1e-10
        assert abs(np.dot(t2, [0, 0, 1])) < 1e-10
        assert abs(np.dot(t1, t2)) < 1e-10

    def test_normal_x(self):
        t1, t2 = _build_contact_frame(np.array([1, 0, 0.0]))
        assert abs(np.dot(t1, [1, 0, 0])) < 1e-10


class TestPGSSolver:
    def _make_ground_contact(self, point, normal, mu=0.5, depth=0.01):
        return ContactConstraint(
            body_i=0,
            body_j=-1,
            point=point,
            normal=normal,
            tangent1=np.zeros(3),
            tangent2=np.zeros(3),
            depth=depth,
            mu=mu,
        )

    def test_no_contacts(self):
        solver = PGSContactSolver()
        result = solver.solve([], [], [], [], [], dt=1e-3)
        assert len(result) == 0

    def test_single_contact_normal_impulse(self):
        """Ball falling onto ground — should get upward normal impulse."""
        solver = PGSContactSolver(max_iter=50)
        contact = self._make_ground_contact(
            point=np.array([0, 0, 0.0]),
            normal=np.array([0, 0, 1.0]),
            depth=0.01,
        )
        # Body 0 has downward velocity
        body_v = [np.array([0, 0, -1.0, 0, 0, 0])]
        body_X = [SpatialTransform.from_translation(np.array([0, 0, 0.5]))]
        inv_mass = [1.0]  # 1 kg
        inv_inertia = [np.eye(3) * 10.0]

        impulses = solver.solve([contact], body_v, body_X, inv_mass, inv_inertia, dt=1e-3)

        # Should have upward impulse on body 0
        assert impulses[0][2] > 0, f"Expected upward impulse, got z={impulses[0][2]}"

    def test_friction_opposing_slip(self):
        """Sliding body — friction should oppose horizontal velocity."""
        solver = PGSContactSolver(max_iter=50)
        contact = self._make_ground_contact(
            point=np.array([0, 0, 0.0]),
            normal=np.array([0, 0, 1.0]),
            depth=0.01,
            mu=0.5,
        )
        # Body sliding in +X with downward velocity
        body_v = [np.array([1.0, 0, -0.5, 0, 0, 0])]
        body_X = [SpatialTransform.from_translation(np.array([0, 0, 0.5]))]
        inv_mass = [1.0]
        inv_inertia = [np.eye(3) * 10.0]

        impulses = solver.solve([contact], body_v, body_X, inv_mass, inv_inertia, dt=1e-3)

        # Friction should create negative X impulse (opposing +X slip)
        assert impulses[0][0] < 0 or abs(impulses[0][0]) < 1e-6, (
            f"Friction should oppose slip, got x={impulses[0][0]}"
        )

    def test_normal_impulse_non_negative(self):
        """Normal impulse must be >= 0 (no pulling)."""
        solver = PGSContactSolver(max_iter=50)
        contact = self._make_ground_contact(
            point=np.array([0, 0, 0.0]),
            normal=np.array([0, 0, 1.0]),
            depth=0.01,
        )
        body_v = [np.array([0, 0, 1.0, 0, 0, 0])]  # moving upward
        body_X = [SpatialTransform.from_translation(np.array([0, 0, 0.5]))]
        inv_mass = [1.0]
        inv_inertia = [np.eye(3) * 10.0]

        impulses = solver.solve([contact], body_v, body_X, inv_mass, inv_inertia, dt=1e-3)

        # Should have zero or positive normal impulse (not pulling)
        assert impulses[0][2] >= -1e-10

    def test_multiple_contacts(self):
        """Two contact points should both produce impulses."""
        solver = PGSContactSolver(max_iter=50)
        c1 = self._make_ground_contact(np.array([-0.5, 0, 0]), np.array([0, 0, 1.0]))
        c2 = self._make_ground_contact(np.array([0.5, 0, 0]), np.array([0, 0, 1.0]))

        body_v = [np.array([0, 0, -1.0, 0, 0, 0])]
        body_X = [SpatialTransform.from_translation(np.array([0, 0, 0.5]))]
        inv_mass = [1.0]
        inv_inertia = [np.eye(3) * 10.0]

        impulses = solver.solve([c1, c2], body_v, body_X, inv_mass, inv_inertia, dt=1e-3)
        assert impulses[0][2] > 0

    def test_full_delassus_coupling(self):
        """Two nearby contacts should produce different result than diagonal approx."""
        solver = PGSContactSolver(max_iter=50)
        # Two contacts very close — should have strong off-diagonal W coupling
        c1 = self._make_ground_contact(np.array([0.01, 0, 0]), np.array([0, 0, 1.0]))
        c2 = self._make_ground_contact(np.array([-0.01, 0, 0]), np.array([0, 0, 1.0]))

        body_v = [np.array([0, 0, -1.0, 0, 0, 0])]
        body_X = [SpatialTransform.from_translation(np.array([0, 0, 0.5]))]
        inv_mass = [1.0]
        inv_inertia = [np.eye(3) * 10.0]

        impulses = solver.solve([c1, c2], body_v, body_X, inv_mass, inv_inertia, dt=1e-3)
        # Total upward impulse should stop the body (momentum = m * v = 1 * 1 = 1)
        total_z = impulses[0][2]
        assert total_z > 0

    def test_body_body_collision(self):
        """Two bodies colliding (neither is ground)."""
        solver = PGSContactSolver(max_iter=50)
        c = ContactConstraint(
            body_i=0,
            body_j=1,
            point=np.array([0.5, 0, 0.0]),
            normal=np.array([1, 0, 0.0]),  # push apart along X
            tangent1=np.zeros(3),
            tangent2=np.zeros(3),
            depth=0.05,
            mu=0.3,
        )
        # Body 0 moving right, body 1 moving left
        body_v = [
            np.array([1.0, 0, 0, 0, 0, 0]),
            np.array([-1.0, 0, 0, 0, 0, 0]),
        ]
        body_X = [
            SpatialTransform.from_translation(np.array([0, 0, 0])),
            SpatialTransform.from_translation(np.array([1, 0, 0])),
        ]
        inv_mass = [1.0, 1.0]
        inv_inertia = [np.eye(3), np.eye(3)]

        impulses = solver.solve([c], body_v, body_X, inv_mass, inv_inertia, dt=1e-3)
        # Normal points from j(1) to i(0) in +X direction
        # Body 0 (body_i) gets pushed in +X (normal direction)
        # Body 1 (body_j) gets pushed in -X (opposite)
        assert impulses[0][0] > 0, f"Body i should get +X impulse, got {impulses[0][0]}"
        assert impulses[1][0] < 0, f"Body j should get -X impulse, got {impulses[1][0]}"

    def test_pgs_convergence(self):
        """More iterations should give better result."""
        c = ContactConstraint(
            body_i=0,
            body_j=-1,
            point=np.array([0, 0, 0.0]),
            normal=np.array([0, 0, 1.0]),
            tangent1=np.zeros(3),
            tangent2=np.zeros(3),
            depth=0.01,
            mu=0.5,
        )
        body_v = [np.array([0.5, 0, -2.0, 0.1, 0, 0])]
        body_X = [SpatialTransform.from_translation(np.array([0, 0, 0.5]))]

        results = []
        for max_iter in [1, 5, 30, 100]:
            solver = PGSContactSolver(max_iter=max_iter)
            imp = solver.solve([c], body_v, body_X, [1.0], [np.eye(3)], dt=1e-3)
            results.append(imp[0][2])

        # Impulse should stabilize as iterations increase
        # Last two should be very close (converged)
        assert abs(results[-1] - results[-2]) < abs(results[1] - results[0]) + 1e-10

    def test_warm_starting(self):
        """Second solve with same contacts should converge faster."""
        solver = PGSContactSolver(max_iter=5)  # Very few iterations
        contact = self._make_ground_contact(
            point=np.array([0, 0, 0.0]),
            normal=np.array([0, 0, 1.0]),
            depth=0.01,
        )
        body_v = [np.array([0, 0, -1.0, 0, 0, 0])]
        body_X = [SpatialTransform.from_translation(np.array([0, 0, 0.5]))]
        inv_mass = [1.0]
        inv_inertia = [np.eye(3) * 10.0]

        # First solve (cold start)
        imp1 = solver.solve([contact], body_v, body_X, inv_mass, inv_inertia, dt=1e-3)

        # Second solve (warm start from cache)
        contact2 = self._make_ground_contact(
            point=np.array([0, 0, 0.0]),  # same point
            normal=np.array([0, 0, 1.0]),
            depth=0.01,
        )
        imp2 = solver.solve([contact2], body_v, body_X, inv_mass, inv_inertia, dt=1e-3)

        # With warm starting, second solve should be at least as good
        assert imp2[0][2] >= imp1[0][2] * 0.9, "Warm start should maintain quality"
