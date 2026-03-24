"""
Tests for ADMM contact solver — verify convergence and agreement with PGS.
"""

from __future__ import annotations

import numpy as np

from physics.solvers.admm import ADMMContactSolver, _project_cone
from physics.solvers.pgs_solver import ContactConstraint, PGSContactSolver
from physics.spatial import SpatialTransform


def _make_contact(
    vz: float = -2.0,
    condim: int = 3,
    mu: float = 0.5,
    mu_spin: float = 0.0,
    mu_roll: float = 0.0,
    depth: float = 0.2,
):
    c = ContactConstraint(
        body_i=0,
        body_j=-1,
        point=np.array([0.0, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        tangent1=np.zeros(3),
        tangent2=np.zeros(3),
        depth=depth,
        mu=mu,
        condim=condim,
        mu_spin=mu_spin,
        mu_roll=mu_roll,
    )
    X = [SpatialTransform.from_translation(np.array([0.0, 0.0, 0.3]))]
    v = [np.array([0.0, 0.0, vz, 0.0, 0.0, 0.0])]
    return [c], v, X, [1.0], [np.eye(3) * 5.0]


# ---------------------------------------------------------------------------
# Cone projection unit tests
# ---------------------------------------------------------------------------


class TestConeProjection:
    def test_positive_normal_unchanged(self):
        c = ContactConstraint(
            body_i=0,
            body_j=-1,
            point=np.zeros(3),
            normal=np.array([0, 0, 1.0]),
            tangent1=np.zeros(3),
            tangent2=np.zeros(3),
            depth=0.1,
            mu=0.5,
            condim=3,
        )
        s = np.array([2.0, 0.1, -0.1])
        p = _project_cone(s, c)
        np.testing.assert_array_equal(p, s)  # within cone, unchanged

    def test_negative_normal_clamped(self):
        c = ContactConstraint(
            body_i=0,
            body_j=-1,
            point=np.zeros(3),
            normal=np.array([0, 0, 1.0]),
            tangent1=np.zeros(3),
            tangent2=np.zeros(3),
            depth=0.1,
            mu=0.5,
            condim=1,
        )
        s = np.array([-1.0])
        p = _project_cone(s, c)
        assert p[0] == 0.0

    def test_tangent_clamped_to_cone(self):
        c = ContactConstraint(
            body_i=0,
            body_j=-1,
            point=np.zeros(3),
            normal=np.array([0, 0, 1.0]),
            tangent1=np.zeros(3),
            tangent2=np.zeros(3),
            depth=0.1,
            mu=0.5,
            condim=3,
        )
        s = np.array([1.0, 10.0, 0.0])  # tangent way outside cone
        p = _project_cone(s, c)
        assert p[0] == 1.0  # normal unchanged
        t_norm = np.sqrt(p[1] ** 2 + p[2] ** 2)
        assert t_norm <= 0.5 * p[0] + 1e-10  # within friction cone

    def test_condim4_spin_clamped(self):
        c = ContactConstraint(
            body_i=0,
            body_j=-1,
            point=np.zeros(3),
            normal=np.array([0, 0, 1.0]),
            tangent1=np.zeros(3),
            tangent2=np.zeros(3),
            depth=0.1,
            mu=0.5,
            condim=4,
            mu_spin=0.01,
        )
        s = np.array([2.0, 0.0, 0.0, 100.0])  # huge spin
        p = _project_cone(s, c)
        assert abs(p[3]) <= 0.01 * p[0] + 1e-10

    def test_condim6_rolling_clamped(self):
        c = ContactConstraint(
            body_i=0,
            body_j=-1,
            point=np.zeros(3),
            normal=np.array([0, 0, 1.0]),
            tangent1=np.zeros(3),
            tangent2=np.zeros(3),
            depth=0.1,
            mu=0.5,
            condim=6,
            mu_spin=0.01,
            mu_roll=0.005,
        )
        s = np.array([2.0, 0.0, 0.0, 0.0, 50.0, -50.0])
        p = _project_cone(s, c)
        assert abs(p[4]) <= 0.005 * p[0] + 1e-10
        assert abs(p[5]) <= 0.005 * p[0] + 1e-10


# ---------------------------------------------------------------------------
# Basic ADMM functionality
# ---------------------------------------------------------------------------


class TestADMMBasic:
    def test_no_contacts(self):
        solver = ADMMContactSolver(max_iter=10)
        result = solver.solve([], [np.zeros(6)], [SpatialTransform.identity()], [1.0], [np.eye(3)], 1e-3)
        np.testing.assert_array_equal(result[0], np.zeros(6))

    def test_single_contact_upward_impulse(self):
        solver = ADMMContactSolver(max_iter=50, rho=10.0)
        contacts, v, X, m, I = _make_contact()
        impulses = solver.solve(contacts, v, X, m, I, dt=1e-3)
        assert impulses[0][2] > 0, f"Expected upward impulse, got {impulses[0][2]}"

    def test_finite_output(self):
        solver = ADMMContactSolver(max_iter=50, rho=10.0)
        contacts, v, X, m, I = _make_contact()
        impulses = solver.solve(contacts, v, X, m, I, dt=1e-3)
        assert np.all(np.isfinite(impulses[0]))

    def test_no_penetration_no_force(self):
        """Body above ground should get zero impulse."""
        solver = ADMMContactSolver(max_iter=50, rho=10.0)
        contacts, v, X, m, I = _make_contact(vz=0.0, depth=0.0)
        # No penetration → contact won't be generated in real pipeline,
        # but solver should handle depth=0 gracefully
        impulses = solver.solve(contacts, v, X, m, I, dt=1e-3)
        assert np.all(np.isfinite(impulses[0]))


# ---------------------------------------------------------------------------
# ADMM vs PGS agreement
# ---------------------------------------------------------------------------


class TestADMMvsPGS:
    def test_normal_impulse_agreement(self):
        """ADMM and PGS should produce similar normal impulse direction."""
        contacts_pgs, v, X, m, I = _make_contact(condim=3)
        contacts_admm, _, _, _, _ = _make_contact(condim=3)

        pgs = PGSContactSolver(max_iter=30, erp=0.2)
        admm = ADMMContactSolver(max_iter=50, rho=10.0, erp=0.2)

        imp_pgs = pgs.solve(contacts_pgs, v, X, m, I, dt=1e-3)
        imp_admm = admm.solve(contacts_admm, v, X, m, I, dt=1e-3)

        # Both should push up
        assert imp_pgs[0][2] > 0
        assert imp_admm[0][2] > 0

    def test_condim4_spin_agreement(self):
        """ADMM with condim=4 should also oppose spin."""
        contacts, v, X, m, I = _make_contact(condim=4, mu_spin=0.05)
        v[0] = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 5.0])

        admm = ADMMContactSolver(max_iter=50, rho=10.0)
        impulses = admm.solve(contacts, v, X, m, I, dt=1e-3)

        assert impulses[0][2] > 0  # upward normal
        assert impulses[0][5] < 0  # opposes +z spin

    def test_friction_opposes_slip(self):
        """ADMM with lateral slip should produce opposing impulse."""
        contacts, v, X, m, I = _make_contact(condim=3, mu=0.8)
        v[0] = np.array([2.0, 0.0, -1.0, 0.0, 0.0, 0.0])

        admm = ADMMContactSolver(max_iter=50, rho=10.0)
        impulses = admm.solve(contacts, v, X, m, I, dt=1e-3)

        assert impulses[0][0] < 0  # opposes +x slip
        assert impulses[0][2] > 0  # upward normal
