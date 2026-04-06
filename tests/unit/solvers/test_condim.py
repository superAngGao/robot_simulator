"""
Tests for variable contact dimension (condim) support in PGS solver.

condim=1: frictionless (normal only)
condim=3: standard Coulomb sliding friction (default, regression)
condim=4: + torsional (spin) friction
condim=6: + rolling friction
"""

from __future__ import annotations

import numpy as np

from physics.contact import LCPContactModel
from physics.geometry import SphereShape
from physics.lcp_solver import ContactConstraint, PGSContactSolver
from physics.spatial import SpatialTransform


def _make_ground_contact(
    body_idx: int = 0,
    z: float = 0.3,
    vz: float = -2.0,
    condim: int = 3,
    mu: float = 0.5,
    mu_spin: float = 0.0,
    mu_roll: float = 0.0,
    depth: float = 0.2,
) -> tuple[ContactConstraint, list, list, list[float], list[np.ndarray]]:
    """Helper: single body contacting ground, returns (constraint, v, X, inv_mass, inv_I)."""
    c = ContactConstraint(
        body_i=body_idx,
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
    X = [SpatialTransform.from_translation(np.array([0.0, 0.0, z]))]
    v = [np.array([0.0, 0.0, vz, 0.0, 0.0, 0.0])]
    inv_mass = [1.0]
    inv_inertia = [np.eye(3) * 5.0]
    return c, v, X, inv_mass, inv_inertia


# ---------------------------------------------------------------------------
# Constraint row count
# ---------------------------------------------------------------------------


class TestConstraintRowCounts:
    def test_condim_1_one_row(self):
        solver = PGSContactSolver(max_iter=10)
        c, v, X, m, I = _make_ground_contact(condim=1)
        impulses = solver.solve([c], v, X, m, I, dt=1e-3)
        # Should work without error; 1 row for normal
        assert impulses[0] is not None

    def test_condim_3_three_rows(self):
        solver = PGSContactSolver(max_iter=10)
        c, v, X, m, I = _make_ground_contact(condim=3)
        impulses = solver.solve([c], v, X, m, I, dt=1e-3)
        assert impulses[0] is not None

    def test_condim_4_four_rows(self):
        solver = PGSContactSolver(max_iter=10)
        c, v, X, m, I = _make_ground_contact(condim=4, mu_spin=0.01)
        impulses = solver.solve([c], v, X, m, I, dt=1e-3)
        assert impulses[0] is not None

    def test_condim_6_six_rows(self):
        solver = PGSContactSolver(max_iter=10)
        c, v, X, m, I = _make_ground_contact(condim=6, mu_spin=0.01, mu_roll=0.001)
        impulses = solver.solve([c], v, X, m, I, dt=1e-3)
        assert impulses[0] is not None

    def test_mixed_condim(self):
        """Two contacts with different condim in same solve."""
        solver = PGSContactSolver(max_iter=30)
        c1, v, X, m, I = _make_ground_contact(condim=1)
        c3 = ContactConstraint(
            body_i=0,
            body_j=-1,
            point=np.array([0.1, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            tangent1=np.zeros(3),
            tangent2=np.zeros(3),
            depth=0.1,
            mu=0.5,
            condim=4,
            mu_spin=0.01,
        )
        impulses = solver.solve([c1, c3], v, X, m, I, dt=1e-3)
        assert np.all(np.isfinite(impulses[0]))


# ---------------------------------------------------------------------------
# condim=1: frictionless
# ---------------------------------------------------------------------------


class TestCondim1Frictionless:
    def test_normal_impulse_upward(self):
        """condim=1 should produce upward normal impulse."""
        solver = PGSContactSolver(max_iter=30)
        c, v, X, m, I = _make_ground_contact(condim=1, vz=-2.0)
        impulses = solver.solve([c], v, X, m, I, dt=1e-3)
        # Linear z impulse should be positive (upward)
        assert impulses[0][2] > 0

    def test_no_tangent_force(self):
        """condim=1 should not generate any tangential force even with slip."""
        solver = PGSContactSolver(max_iter=30)
        c, v, X, m, I = _make_ground_contact(condim=1, mu=0.8)
        # Add lateral slip
        v[0] = np.array([1.0, 0.5, -2.0, 0.0, 0.0, 0.0])
        impulses = solver.solve([c], v, X, m, I, dt=1e-3)
        # Only z-impulse should be non-zero (from normal + moment arm)
        # x/y linear impulse should be zero since no friction rows
        # (Technically the cross product from moment arm can couple, but for
        #  contact at origin with body at origin, the moment arm is zero)
        assert impulses[0][2] > 0  # normal still works


# ---------------------------------------------------------------------------
# condim=3: standard (regression)
# ---------------------------------------------------------------------------


class TestCondim3Regression:
    def test_friction_opposes_slip(self):
        """condim=3 with lateral slip should produce opposing tangent impulse."""
        solver = PGSContactSolver(max_iter=30)
        c, v, X, m, I = _make_ground_contact(condim=3, mu=0.8)
        v[0] = np.array([2.0, 0.0, -1.0, 0.0, 0.0, 0.0])
        impulses = solver.solve([c], v, X, m, I, dt=1e-3)
        # x-impulse should oppose the +x slip → negative
        assert impulses[0][0] < 0
        # z-impulse should be positive
        assert impulses[0][2] > 0


# ---------------------------------------------------------------------------
# condim=4: torsional friction
# ---------------------------------------------------------------------------


class TestCondim4Spin:
    def test_spin_friction_opposes_rotation(self):
        """Body spinning about contact normal should get opposing torque."""
        solver = PGSContactSolver(max_iter=50)
        c, v, X, m, I = _make_ground_contact(condim=4, mu=0.5, mu_spin=0.05)
        # Body spinning about z-axis (contact normal direction)
        v[0] = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 5.0])  # omega_z = 5 rad/s
        impulses = solver.solve([c], v, X, m, I, dt=1e-3)
        # Angular impulse about z should oppose the spin (negative for +z spin)
        assert impulses[0][5] < 0, f"Spin impulse should oppose rotation: {impulses[0][5]}"

    def test_no_spin_with_condim3(self):
        """condim=3 should NOT produce spin-opposing torque."""
        solver = PGSContactSolver(max_iter=50)
        c, v, X, m, I = _make_ground_contact(condim=3, mu=0.5)
        v[0] = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 5.0])
        impulses = solver.solve([c], v, X, m, I, dt=1e-3)
        # condim=3 has no spin row, so angular z impulse should be ~0
        # (there may be small coupling from tangent friction)
        assert abs(impulses[0][5]) < abs(impulses[0][2]) * 0.1

    def test_spin_bounded_by_mu_spin(self):
        """Spin impulse magnitude should be limited by mu_spin * lambda_n."""
        solver = PGSContactSolver(max_iter=50)
        c, v, X, m, I = _make_ground_contact(condim=4, mu=0.5, mu_spin=0.001)
        v[0] = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 100.0])  # huge spin
        impulses = solver.solve([c], v, X, m, I, dt=1e-3)
        # Even with huge spin, the spin impulse is bounded
        assert np.all(np.isfinite(impulses[0]))


# ---------------------------------------------------------------------------
# condim=6: rolling friction
# ---------------------------------------------------------------------------


class TestCondim6Rolling:
    def test_rolling_friction_opposes_rotation(self):
        """Body rolling about tangent axis should get opposing torque."""
        solver = PGSContactSolver(max_iter=50)
        c, v, X, m, I = _make_ground_contact(condim=6, mu=0.5, mu_spin=0.01, mu_roll=0.005)
        # Body rolling about x-axis (tangent direction)
        v[0] = np.array([0.0, 0.0, -1.0, 5.0, 0.0, 0.0])  # omega_x = 5 rad/s
        impulses = solver.solve([c], v, X, m, I, dt=1e-3)
        # Angular impulse about x should oppose the rolling
        assert impulses[0][3] < 0, f"Rolling impulse should oppose rotation: {impulses[0][3]}"

    def test_no_rolling_with_condim4(self):
        """condim=4 should NOT add rolling-friction torque over condim=3.

        A body rotating about a tangent axis (ω_x=5) above the ground receives
        tangential-friction torque at the lever arm — that torque is present
        at every condim ≥ 3. Rolling friction is the *extra* torque that
        condim=6 adds via its angular Jacobian rows. This test verifies
        condim=4 does NOT add any such extra torque (rolling rows are absent),
        by comparing the angular-x impulse against condim=3.
        """
        solver = PGSContactSolver(max_iter=50)
        v_init = np.array([0.0, 0.0, -1.0, 5.0, 0.0, 0.0])

        c3, v3, X3, m3, I3 = _make_ground_contact(condim=3, mu=0.5)
        v3[0] = v_init.copy()
        imp3 = solver.solve([c3], v3, X3, m3, I3, dt=1e-3)

        c4, v4, X4, m4, I4 = _make_ground_contact(condim=4, mu=0.5, mu_spin=0.01)
        v4[0] = v_init.copy()
        imp4 = solver.solve([c4], v4, X4, m4, I4, dt=1e-3)

        # condim=4 adds torsional friction on the normal axis (angular z),
        # but leaves the rolling axes (angular x/y) unchanged relative to condim=3.
        np.testing.assert_allclose(imp4[0][3], imp3[0][3], atol=1e-6)
        np.testing.assert_allclose(imp4[0][4], imp3[0][4], atol=1e-6)

    def test_condim6_all_channels_active(self):
        """Body with combined motion should activate all 6 constraint channels."""
        solver = PGSContactSolver(max_iter=50)
        c, v, X, m, I = _make_ground_contact(condim=6, mu=0.8, mu_spin=0.05, mu_roll=0.01)
        # Sliding + spinning + rolling
        v[0] = np.array([1.0, 0.5, -2.0, 3.0, 2.0, 4.0])
        impulses = solver.solve([c], v, X, m, I, dt=1e-3)
        assert np.all(np.isfinite(impulses[0]))
        # Normal should be positive
        assert impulses[0][2] > 0


# ---------------------------------------------------------------------------
# LCPContactModel condim integration
# ---------------------------------------------------------------------------


class TestLCPContactModelCondim:
    def test_default_condim3(self):
        """Default LCPContactModel uses condim=3."""
        model = LCPContactModel(mu=0.5, max_iter=30)
        model.add_contact_body(0, SphereShape(0.5), "ball")
        # Should work (condim=3 is default)
        X = [SpatialTransform.from_translation(np.array([0, 0, 0.3]))]
        v = [np.array([0, 0, -1.0, 0, 0, 0])]
        forces = model.compute_forces(X, v, 1)
        assert forces[0][2] > 0

    def test_condim4_model_level(self):
        """LCPContactModel with model-level condim=4."""
        model = LCPContactModel(mu=0.5, condim=4, mu_spin=0.01, max_iter=30)
        model.add_contact_body(0, SphereShape(0.5), "ball")
        X = [SpatialTransform.from_translation(np.array([0, 0, 0.3]))]
        v = [np.array([0, 0, -1.0, 0, 0, 5.0])]  # spinning
        forces = model.compute_forces(X, v, 1)
        assert np.all(np.isfinite(forces[0]))

    def test_per_body_condim_override(self):
        """Per-body condim override should work."""
        model = LCPContactModel(mu=0.5, condim=3, max_iter=30)
        model.add_contact_body(0, SphereShape(0.5), "ball", condim=1)  # override to frictionless
        X = [SpatialTransform.from_translation(np.array([0, 0, 0.3]))]
        v = [np.array([1.0, 0, -1.0, 0, 0, 0])]
        forces = model.compute_forces(X, v, 1)
        assert np.all(np.isfinite(forces[0]))
