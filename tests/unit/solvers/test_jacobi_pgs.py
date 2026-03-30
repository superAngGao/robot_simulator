"""
Tests for Jacobi PGS solver — verify convergence and agreement with serial PGS.
"""

from __future__ import annotations

import numpy as np

from physics.solvers.jacobi_pgs import JacobiPGSContactSolver
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
    """Single body on ground, returns (contacts, v, X, inv_mass, inv_I)."""
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


def _make_multi_contact():
    """Two contacts on same body at different positions."""
    c1 = ContactConstraint(
        body_i=0,
        body_j=-1,
        point=np.array([-0.1, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        tangent1=np.zeros(3),
        tangent2=np.zeros(3),
        depth=0.15,
        mu=0.5,
        condim=3,
    )
    c2 = ContactConstraint(
        body_i=0,
        body_j=-1,
        point=np.array([0.1, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        tangent1=np.zeros(3),
        tangent2=np.zeros(3),
        depth=0.1,
        mu=0.5,
        condim=3,
    )
    X = [SpatialTransform.from_translation(np.array([0.0, 0.0, 0.3]))]
    v = [np.array([0.0, 0.0, -2.0, 0.0, 0.0, 0.0])]
    return [c1, c2], v, X, [1.0], [np.eye(3) * 5.0]


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestJacobiPGSBasic:
    def test_no_contacts(self):
        solver = JacobiPGSContactSolver(max_iter=10)
        result = solver.solve([], [np.zeros(6)], [SpatialTransform.identity()], [1.0], [np.eye(3)], 1e-3)
        np.testing.assert_array_equal(result[0], np.zeros(6))

    def test_single_contact_upward_impulse(self):
        solver = JacobiPGSContactSolver(max_iter=60, omega=0.7)
        contacts, v, X, m, I = _make_contact()
        impulses = solver.solve(contacts, v, X, m, I, dt=1e-3)
        assert impulses[0][2] > 0, "Should produce upward normal impulse"

    def test_finite_output(self):
        solver = JacobiPGSContactSolver(max_iter=60)
        contacts, v, X, m, I = _make_contact()
        impulses = solver.solve(contacts, v, X, m, I, dt=1e-3)
        assert np.all(np.isfinite(impulses[0]))


# ---------------------------------------------------------------------------
# Agreement with serial PGS
# ---------------------------------------------------------------------------


class TestJacobiVsSerialPGS:
    def test_normal_impulse_agreement_condim3(self):
        """Jacobi PGS should converge to similar normal impulse as serial PGS."""
        contacts_gs, v, X, m, I = _make_contact(condim=3)
        contacts_j, _, _, _, _ = _make_contact(condim=3)

        gs = PGSContactSolver(max_iter=30, erp=0.2, cfm=1e-6)
        jac = JacobiPGSContactSolver(max_iter=100, omega=0.7, erp=0.2, cfm=1e-6)

        imp_gs = gs.solve(contacts_gs, v, X, m, I, dt=1e-3)
        imp_j = jac.solve(contacts_j, v, X, m, I, dt=1e-3)

        # Normal impulse (z) should agree within 20%
        z_gs = imp_gs[0][2]
        z_j = imp_j[0][2]
        assert z_gs > 0 and z_j > 0
        assert abs(z_j - z_gs) / abs(z_gs) < 0.2, (
            f"Jacobi z={z_j:.4f} vs GS z={z_gs:.4f}, diff={abs(z_j - z_gs) / abs(z_gs) * 100:.1f}%"
        )

    def test_multi_contact_agreement(self):
        """Multi-contact: Jacobi PGS agrees with serial PGS."""
        contacts_gs, v, X, m, I = _make_multi_contact()
        contacts_j, _, _, _, _ = _make_multi_contact()

        gs = PGSContactSolver(max_iter=30, erp=0.2, cfm=1e-6)
        jac = JacobiPGSContactSolver(max_iter=100, omega=0.7, erp=0.2, cfm=1e-6)

        imp_gs = gs.solve(contacts_gs, v, X, m, I, dt=1e-3)
        imp_j = jac.solve(contacts_j, v, X, m, I, dt=1e-3)

        # Both should produce upward impulse
        assert imp_gs[0][2] > 0 and imp_j[0][2] > 0
        # Agree within 30% (multi-contact coupling makes Jacobi converge slower)
        assert abs(imp_j[0][2] - imp_gs[0][2]) / abs(imp_gs[0][2]) < 0.3

    def test_condim1_agreement(self):
        """Frictionless: Jacobi == serial (only 1 row, no coupling difference)."""
        contacts_gs, v, X, m, I = _make_contact(condim=1)
        contacts_j, _, _, _, _ = _make_contact(condim=1)

        gs = PGSContactSolver(max_iter=30)
        jac = JacobiPGSContactSolver(max_iter=60, omega=1.0)  # omega=1 for single row

        imp_gs = gs.solve(contacts_gs, v, X, m, I, dt=1e-3)
        imp_j = jac.solve(contacts_j, v, X, m, I, dt=1e-3)

        np.testing.assert_allclose(imp_j[0], imp_gs[0], atol=1e-4)

    def test_condim4_agreement(self):
        """condim=4: Jacobi agrees with serial PGS on spin friction."""
        contacts_gs, v, X, m, I = _make_contact(condim=4, mu_spin=0.05)
        contacts_j, _, _, _, _ = _make_contact(condim=4, mu_spin=0.05)
        # Add spin
        v[0] = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 5.0])

        gs = PGSContactSolver(max_iter=30)
        jac = JacobiPGSContactSolver(max_iter=100, omega=0.7)

        imp_gs = gs.solve(contacts_gs, v, X, m, I, dt=1e-3)
        imp_j = jac.solve(contacts_j, v, X, m, I, dt=1e-3)

        # Both should oppose spin (negative angular z impulse)
        assert imp_gs[0][5] < 0 and imp_j[0][5] < 0


# ---------------------------------------------------------------------------
# Relaxation omega
# ---------------------------------------------------------------------------


class TestRelaxation:
    def test_small_omega_smaller_impulse(self):
        """Small omega should produce smaller impulse than default omega."""
        contacts_lo, v, X, m, I = _make_contact()
        contacts_hi, _, _, _, _ = _make_contact()

        solver_lo = JacobiPGSContactSolver(max_iter=10, omega=0.01)
        solver_hi = JacobiPGSContactSolver(max_iter=10, omega=0.7)

        imp_lo = solver_lo.solve(contacts_lo, v, X, m, I, dt=1e-3)
        imp_hi = solver_hi.solve(contacts_hi, v, X, m, I, dt=1e-3)

        assert abs(imp_lo[0][2]) < abs(imp_hi[0][2])

    def test_omega_1_fastest_but_less_stable(self):
        """omega=1 converges faster but may oscillate on coupled problems."""
        solver = JacobiPGSContactSolver(max_iter=60, omega=1.0)
        contacts, v, X, m, I = _make_contact()
        impulses = solver.solve(contacts, v, X, m, I, dt=1e-3)
        assert np.all(np.isfinite(impulses[0]))
        assert impulses[0][2] > 0


# ---------------------------------------------------------------------------
# Mixed condim
# ---------------------------------------------------------------------------


class TestMixedCondim:
    def test_mixed_condim_1_and_4(self):
        """Two contacts: one frictionless, one with spin friction."""
        c1 = ContactConstraint(
            body_i=0,
            body_j=-1,
            point=np.array([-0.1, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            tangent1=np.zeros(3),
            tangent2=np.zeros(3),
            depth=0.1,
            mu=0.0,
            condim=1,
        )
        c2 = ContactConstraint(
            body_i=0,
            body_j=-1,
            point=np.array([0.1, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            tangent1=np.zeros(3),
            tangent2=np.zeros(3),
            depth=0.1,
            mu=0.5,
            condim=4,
            mu_spin=0.02,
        )
        X = [SpatialTransform.from_translation(np.array([0.0, 0.0, 0.3]))]
        v = [np.array([0.0, 0.0, -1.0, 0.0, 0.0, 3.0])]

        solver = JacobiPGSContactSolver(max_iter=80, omega=0.7)
        impulses = solver.solve([c1, c2], v, X, [1.0], [np.eye(3) * 5.0], dt=1e-3)
        assert np.all(np.isfinite(impulses[0]))
        assert impulses[0][2] > 0  # upward normal
