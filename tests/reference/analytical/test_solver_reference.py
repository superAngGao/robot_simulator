"""
Reference tests for contact solvers against analytical LCP solutions.

Each test documents:
  - The physical setup
  - The analytical derivation of the reference solution
  - Why this reference was chosen over alternatives

Reference selection rationale:
  - **Analytical LCP**: Hand-derived from the Delassus matrix W = J M⁻¹ Jᵀ
    and velocity-level complementarity conditions. This is the ground truth
    for hard-constraint LCP solvers (PGS, Jacobi PGS).
  - **ADMM**: Solves a velocity-space QP (different formulation), so impulse
    magnitudes may differ from LCP. We verify direction and physical bounds.
  - **MuJoCo**: NOT used — soft constraint model (solref/solimp) with implicit
    integration produces fundamentally different force magnitudes.
  - **Bullet (PyBullet)**: NOT used for single-step impulse — its ERP
    formulation is coupled with the solver in ways that prevent clean
    erp=0 comparison. Suitable for multi-step trajectory comparison (future).

Shared setup across all tests:
  - 1 kg sphere, radius 0.1 m
  - Contact at world origin [0, 0, 0]
  - Body center at [0, 0, 0.1] (one radius above ground)
  - inv_mass = 1.0, inv_inertia = diag(250, 250, 250) (I=0.004*I₃)
  - dt = 0.001 s, erp = 0 (pure velocity-level LCP, no Baumgarte)

Delassus matrix for this setup (condim=3):
  W = [[1.0, 0.0, 0.0],     (normal-normal)
       [0.0, 3.5, 0.0],     (tangent1-tangent1 = 1 + 0.1²*250)
       [0.0, 0.0, 3.5]]     (tangent2-tangent2)
  Off-diagonal = 0 (r parallel to n → no normal-tangent coupling)
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.solvers.pgs_solver import ContactConstraint, PGSContactSolver
from physics.spatial import SpatialTransform

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

DT = 0.001
INV_MASS = 1.0
INV_INERTIA = np.eye(3) * 250.0  # 1/0.004
X = [SpatialTransform.from_translation(np.array([0.0, 0.0, 0.1]))]


def _solve_and_get_velocity(solver, v_init, condim=3, mu=0.5, mu_spin=0.0, mu_roll=0.0, depth=0.001):
    """Run solver, return velocity after applying impulse to body."""
    cc = ContactConstraint(
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
    v = [v_init.copy()]
    imp = solver.solve([cc], v, X, [INV_MASS], [INV_INERTIA], dt=DT)
    v_after = v_init.copy()
    v_after[:3] += imp[0][:3] * INV_MASS
    v_after[3:] += INV_INERTIA @ imp[0][3:]
    return v_after, imp[0]


def _all_solvers():
    """Return (name, solver) pairs for parametrized testing."""
    return [
        ("PGS", PGSContactSolver(max_iter=50, erp=0.0, cfm=1e-8)),
    ]


# ---------------------------------------------------------------------------
# Test 1: Inelastic normal collision
# ---------------------------------------------------------------------------


class TestNormalCollision:
    """Pure vertical drop: v_z=-2 m/s → v_z=0 after contact.

    Reference: analytical LCP solution.
    Derivation:
      v_free_n = J_n @ v = -2.0
      W_nn = 1.0
      LCP: v_free_n + W_nn * λ_n ≥ 0, λ_n ≥ 0
      Solution: λ_n = 2.0, v_z_after = -2.0 + 2.0*1.0 = 0.0

    Why analytical, not Bullet/MuJoCo:
      Bullet's PGS with erp=0 produces near-zero impulse (solver disabled).
      MuJoCo uses soft constraints, producing v_z_after ≈ -1.67 (not hard LCP).
      The analytical solution is exact for velocity-level hard LCP.
    """

    @pytest.mark.parametrize("name,solver", _all_solvers(), ids=lambda x: x if isinstance(x, str) else "")
    def test_vz_goes_to_zero(self, name, solver):
        v_after, imp = _solve_and_get_velocity(solver, np.array([0, 0, -2.0, 0, 0, 0]))
        assert abs(v_after[2]) < 1e-3, f"{name}: v_z_after={v_after[2]:.6f}, expected 0.0"

    @pytest.mark.parametrize("name,solver", _all_solvers(), ids=lambda x: x if isinstance(x, str) else "")
    def test_normal_impulse_is_2(self, name, solver):
        """λ_n = |v_free_n| / W_nn = 2.0 / 1.0 = 2.0."""
        _, imp = _solve_and_get_velocity(solver, np.array([0, 0, -2.0, 0, 0, 0]))
        # ADMM returns M*Δv (momentum change), not constraint-space λ.
        # For this setup M_lin = 1.0, so impulse_z ≈ λ_n.
        assert abs(imp[2] - 2.0) < 0.1, f"{name}: impulse_z={imp[2]:.4f}, expected ≈2.0"


# ---------------------------------------------------------------------------
# Test 2: Friction in sticking regime
# ---------------------------------------------------------------------------


class TestFrictionSticking:
    """Lateral slip vx=2, vz=-2, mu=0.5 → contact velocity goes to zero.

    Reference: analytical LCP solution.
    Derivation:
      W = diag(1.0, 3.5, 3.5)
      v_free = [-2.0, 2.0, 0.0]  (normal, tangent1, tangent2)
      λ_n = 2.0 (same as Test 1)
      λ_t1 = -v_free_t1 / W_t1t1 = -2.0 / 3.5 = -0.571
      Friction limit = μ*λ_n = 0.5*2.0 = 1.0
      |λ_t1| = 0.571 < 1.0 → sticking (not sliding)

    Body velocity after:
      vx_after = 2.0 + λ_t1 * 1.0 = 1.429  (body center, not contact point)
      vz_after = 0.0
      Contact point velocity = 0 in all directions (verified by J@v_after=0)

    Why W_t1t1 = 3.5:
      J_t1 = [1, 0, 0, 0, -0.1, 0]  (moment arm r=[0,0,-0.1], r×t1=[0,-0.1,0])
      W_t1t1 = J_t1 @ M⁻¹ @ J_t1ᵀ = 1²*1 + (-0.1)²*250 = 1 + 2.5 = 3.5
    """

    @pytest.mark.parametrize("name,solver", _all_solvers(), ids=lambda x: x if isinstance(x, str) else "")
    def test_contact_velocity_zero(self, name, solver):
        """Contact point velocity should go to zero (sticking)."""
        v_after, _ = _solve_and_get_velocity(solver, np.array([2.0, 0, -2.0, 0, 0, 0]))
        # Contact velocity = J @ v_after
        # v_contact_x = vx + (-0.1)*omega_y
        v_contact_x = v_after[0] + (-0.1) * v_after[4]
        assert abs(v_after[2]) < 1e-3, f"{name}: v_z={v_after[2]}"
        # Tolerance relaxed from 1e-2 to 0.03: Q25 fix adds per-row R
        # regularization on friction rows (compliance), causing small residual
        # tangential velocity — physically correct (matches MuJoCo soft contact).
        assert abs(v_contact_x) < 0.03, f"{name}: v_contact_x={v_contact_x:.4f}, expected ~0.0"

    def test_pgs_body_vx_analytical(self):
        """PGS body center vx should be near 1.4286 (analytical W_t1t1=3.5).

        Tolerance relaxed from 1e-3 to 0.01: Q25 fix adds per-row R
        regularization on friction rows, making W_eff = W + R > W.
        This slightly reduces the friction impulse, giving larger vx.
        """
        solver = PGSContactSolver(max_iter=50, erp=0.0, cfm=1e-8)
        v_after, _ = _solve_and_get_velocity(solver, np.array([2.0, 0, -2.0, 0, 0, 0]))
        expected_vx = 2.0 - 2.0 / 3.5  # = 1.4286 (hard LCP)
        assert abs(v_after[0] - expected_vx) < 0.01, f"vx={v_after[0]:.4f}, expected ~{expected_vx:.4f}"


# ---------------------------------------------------------------------------
# Test 3: Frictionless (condim=1) preserves tangent velocity
# ---------------------------------------------------------------------------


class TestFrictionless:
    """condim=1: only normal constraint, lateral velocity unchanged.

    Reference: analytical.
    Derivation:
      Only 1 constraint row (normal). λ_n = 2.0.
      No tangent rows → no tangent impulse → vx unchanged.
      vx_after = vx_before = 2.0
      vz_after = 0.0
    """

    @pytest.mark.parametrize("name,solver", _all_solvers(), ids=lambda x: x if isinstance(x, str) else "")
    def test_vx_unchanged(self, name, solver):
        v_after, _ = _solve_and_get_velocity(solver, np.array([2.0, 0, -2.0, 0, 0, 0]), condim=1)
        assert abs(v_after[0] - 2.0) < 1e-3, f"{name}: vx={v_after[0]}"
        assert abs(v_after[2]) < 1e-3, f"{name}: vz={v_after[2]}"


# ---------------------------------------------------------------------------
# Test 4: Torsional (spin) friction, condim=4
# ---------------------------------------------------------------------------


class TestSpinFriction:
    """Spin omega_z=5, condim=4, mu_spin=0.05 → spin arrested.

    Reference: analytical.
    Derivation:
      J_spin = [0,0,0, 0,0,1]  (angular Jacobian along normal)
      W_ss = J_spin @ M⁻¹ @ J_spinᵀ = 1²*250 = 250.0
      v_free_spin = omega_z = 5.0
      λ_s = -5.0 / 250 = -0.02
      Spin limit = μ_spin * λ_n = 0.05 * 2.0 = 0.1
      |λ_s| = 0.02 < 0.1 → not saturated
      omega_z_after = 5.0 + (-0.02)*250 = 0.0

    Why not MuJoCo: MuJoCo gives omega_z_after=4.55 (soft constraint).
    """

    @pytest.mark.parametrize("name,solver", _all_solvers(), ids=lambda x: x if isinstance(x, str) else "")
    def test_spin_arrested(self, name, solver):
        v_after, _ = _solve_and_get_velocity(
            solver,
            np.array([0, 0, -2.0, 0, 0, 5.0]),
            condim=4,
            mu_spin=0.05,
        )
        assert abs(v_after[2]) < 1e-3, f"{name}: vz={v_after[2]}"
        assert abs(v_after[5]) < 0.1, f"{name}: omega_z={v_after[5]:.4f}, expected ~0.0"


# ---------------------------------------------------------------------------
# Test 5: Spin friction saturated regime
# ---------------------------------------------------------------------------


class TestSpinSaturated:
    """Large spin with tiny mu_spin → spin friction saturates.

    Reference: analytical.
    Derivation:
      omega_z = 50 rad/s, mu_spin = 0.001, λ_n = 2.0
      Unsaturated: λ_s = -50/250 = -0.2
      Limit: mu_spin * λ_n = 0.001 * 2.0 = 0.002
      |0.2| > 0.002 → SATURATED, λ_s = -0.002
      Δomega_z = -0.002 * 250 = -0.5
      omega_z_after = 50.0 - 0.5 = 49.5

    This tests the friction cone projection boundary.
    """

    def test_pgs_spin_saturated(self):
        solver = PGSContactSolver(max_iter=50, erp=0.0, cfm=1e-8)
        v_after, _ = _solve_and_get_velocity(
            solver,
            np.array([0, 0, -2.0, 0, 0, 50.0]),
            condim=4,
            mu_spin=0.001,
        )
        assert abs(v_after[5] - 49.5) < 0.5, f"omega_z={v_after[5]:.2f}, expected ~49.5 (saturated)"
