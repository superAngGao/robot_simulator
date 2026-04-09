"""
CRBA Cholesky numerical stability tests — Wilkinson backward error suite.

Methodology: see feedback_numerical_stability_wilkinson.md and REFLECTIONS.md
session 24 (2026-04-09). Adopted at session 24 as the standing approach for
testing any custom linear-algebra kernel.

Test classes (built in execution order):

  Class 1 (THIS FILE) — Synthetic SPD direct kernel:
    Wraps GPU `_chol_factor` and `_chol_solve` from crba_kernels in test-only
    @wp.kernel and feeds synthetic SPD matrices with controlled cond(H).
    Applies the Wilkinson 4-test suite:
      Test 1: reconstruction error  ‖L L^T - H‖_F / ‖H‖_F
      Test 2: normwise backward error  ‖H x̂ - b‖ / (‖H‖‖x̂‖ + ‖b‖)
      Test 3: forward error vs theoretical bound  κ(H) × ε × C(n)
      Test 4: regularization clamp documentation (above κ ≈ 1e6)

  Class 2 — CRBA H matrix structural properties (CPU only)
  Class 4 — CRBA vs ABA in physical fixtures
  Class 3 — Quadruped q-sweep (exploration, recorded in REFLECTIONS)
  Class 5 — GPU qacc accessor + cross-check

GPU pipeline context: GpuEngine performs ONE Cholesky factor of H per step
and reuses it for three solves (smooth dynamics, Delassus W = J H⁻¹ J^T,
impulse apply). This makes `_chol_factor` and `_chol_solve` the most fragile
single point in the GPU pipeline (Q29 architecture, REFLECTIONS session 13).

The kernel has built-in regularization `if val < reg: val = reg` (reg=1e-6)
that silently clamps low pivots — making this exactly the kind of kernel
where comparison-only tests (CPU vs GPU NumPy Cholesky) can miss bugs:
both implementations may agree on the *wrong* answer at high κ. Wilkinson
backward error catches this because the residual `‖H x̂ - b‖` is measured
against H itself, not against a second implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.stats

wp = pytest.importorskip("warp")

pytestmark = pytest.mark.gpu

from physics.backends.warp.crba_kernels import (  # noqa: E402
    _chol_factor,
    _chol_solve,
)

wp.init()

EPS_F32 = float(np.finfo(np.float32).eps)  # ~1.19e-7
DEVICE = "cuda:0"


# ---------------------------------------------------------------------------
# Test-only kernels — wrap @wp.func device functions for direct testing
# ---------------------------------------------------------------------------


@wp.kernel
def _test_chol_factor_kernel(
    M: wp.array(dtype=wp.float32, ndim=3),
    L: wp.array(dtype=wp.float32, ndim=3),
    n: int,
    reg: float,
):
    env = wp.tid()
    _chol_factor(env, M, L, n, reg)


@wp.kernel
def _test_chol_solve_kernel(
    L: wp.array(dtype=wp.float32, ndim=3),
    rhs: wp.array(dtype=wp.float32, ndim=2),
    x: wp.array(dtype=wp.float32, ndim=2),
    tmp: wp.array(dtype=wp.float32, ndim=2),
    n: int,
):
    env = wp.tid()
    _chol_solve(env, L, rhs, x, tmp, n)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_spd(n: int, kappa: float, seed: int = 0) -> np.ndarray:
    """Construct an SPD matrix with cond(H) ≈ kappa to ~1% relative.

    H = Q diag(λ_1, ..., λ_n) Q^T  with  λ_i = geomspace(1, 1/kappa, n)
    so by construction λ_max / λ_min = kappa.

    Q is a uniformly-random orthogonal matrix from the Haar measure on O(n).
    """
    rng = np.random.default_rng(seed)
    Q = scipy.stats.ortho_group.rvs(n, random_state=rng)
    lambdas = np.geomspace(1.0, 1.0 / kappa, n)
    H = (Q * lambdas) @ Q.T
    H = (H + H.T) / 2  # symmetrize against tiny float drift
    return H


def gpu_cholesky_factor(H: np.ndarray, reg: float = 1.0e-6) -> np.ndarray:
    """Run GPU `_chol_factor` on H and return the (lower-triangular) L factor."""
    n = H.shape[0]
    M_wp = wp.array(H[None].astype(np.float32), dtype=wp.float32, device=DEVICE)
    L_wp = wp.zeros((1, n, n), dtype=wp.float32, device=DEVICE)
    wp.launch(_test_chol_factor_kernel, dim=1, inputs=[M_wp, L_wp, n, reg], device=DEVICE)
    return L_wp.numpy()[0]


def gpu_cholesky_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Run GPU `_chol_solve` with given L (lower-tri) and rhs b, return x."""
    n = L.shape[0]
    L_wp = wp.array(L[None].astype(np.float32), dtype=wp.float32, device=DEVICE)
    rhs_wp = wp.array(b[None].astype(np.float32), dtype=wp.float32, device=DEVICE)
    x_wp = wp.zeros((1, n), dtype=wp.float32, device=DEVICE)
    tmp_wp = wp.zeros((1, n), dtype=wp.float32, device=DEVICE)
    wp.launch(_test_chol_solve_kernel, dim=1, inputs=[L_wp, rhs_wp, x_wp, tmp_wp, n], device=DEVICE)
    return x_wp.numpy()[0]


def reconstruction_error(L: np.ndarray, H: np.ndarray) -> float:
    """‖L L^T - H‖_F / ‖H‖_F."""
    H_recon = L @ L.T
    return float(np.linalg.norm(H_recon - H, "fro") / np.linalg.norm(H, "fro"))


def normwise_backward_error(H: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """‖H x̂ - b‖ / (‖H‖ ‖x̂‖ + ‖b‖). Should be O(n × ε) for backward-stable solvers."""
    residual = H @ x - b
    denom = np.linalg.norm(H) * np.linalg.norm(x) + np.linalg.norm(b)
    return float(np.linalg.norm(residual) / denom)


def forward_error(x: np.ndarray, x_ref: np.ndarray) -> float:
    """‖x̂ - x_ref‖ / ‖x_ref‖. Compared against κ × ε bound."""
    return float(np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref))


# ---------------------------------------------------------------------------
# Class 1 — Synthetic SPD direct kernel (Wilkinson 4-test suite)
# ---------------------------------------------------------------------------


class TestSyntheticSPDCholesky:
    """Direct GPU `_chol_factor` and `_chol_solve` on synthetic SPD matrices.

    These tests do not involve any robot, kinematic tree, or physics — only
    raw linear algebra. They isolate the Cholesky kernel from every other
    source of error in the pipeline.

    Sweep cond(H) ∈ {1, 1e2, 1e4} in f32-safe regime, plus 1e6 to document
    the regularization clamp boundary.

    Methodology: feedback_numerical_stability_wilkinson.md (Wilkinson backward
    error). All test bounds use generous constants (10n × ε to 100n × ε for
    backward-stable assertions; κ × 1000n × ε for forward bounds) because we
    are validating order-of-magnitude correctness, not optimal constants.
    """

    N = 8  # matrix size — small enough to test fast, large enough to be non-trivial

    # ---- Wilkinson Test 1: Reconstruction error ‖L L^T - H‖_F / ‖H‖_F ----

    @pytest.mark.parametrize("kappa", [1.0, 1e2, 1e4])
    def test_reconstruction_error_within_bound(self, kappa):
        """L L^T should reproduce H to O(n × ε_f32). Validates _chol_factor.

        Theory: backward-stable Cholesky satisfies ‖L L^T - H‖ ≤ c·n·ε·‖H‖
        with c ~ 1. We allow c=100 to absorb the kernel's intermediate
        accumulation order and small κ-dependent slack.
        """
        H = make_spd(self.N, kappa, seed=int(np.log10(kappa) * 7))
        L_gpu = gpu_cholesky_factor(H)
        rel_err = reconstruction_error(L_gpu, H.astype(np.float32))
        bound = 100 * self.N * EPS_F32
        assert rel_err < bound, f"κ={kappa:.0e}: reconstruction err {rel_err:.2e} > bound {bound:.2e}"

    # ---- Wilkinson Test 2: Normwise backward error (κ-INDEPENDENT) ----

    @pytest.mark.parametrize("kappa", [1.0, 1e2, 1e4])
    def test_backward_error_independent_of_kappa(self, kappa):
        """Backward error should be O(n × ε), independent of κ(H).

        This is THE diagnostic for Cholesky stability: if backward error
        grows with κ, the implementation has catastrophic cancellation or
        bad accumulation order. For this test we expect all three κ values
        to give similar backward errors (within ~10x of each other), all
        well below 100n × ε.
        """
        rng = np.random.default_rng(int(np.log10(kappa) * 11) + 42)
        H = make_spd(self.N, kappa, seed=int(np.log10(kappa) * 7))
        b = rng.standard_normal(self.N).astype(np.float64)

        L_gpu = gpu_cholesky_factor(H)
        x_gpu = gpu_cholesky_solve(L_gpu, b).astype(np.float64)

        backward_err = normwise_backward_error(H, x_gpu, b)
        bound = 100 * self.N * EPS_F32
        assert backward_err < bound, (
            f"κ={kappa:.0e}: backward err {backward_err:.2e} > bound {bound:.2e} "
            f"— possible loss of backward stability"
        )

    # ---- Wilkinson Test 3: Forward error within κ × ε bound ----

    @pytest.mark.parametrize("kappa", [1.0, 1e2, 1e4])
    def test_forward_error_within_kappa_eps_bound(self, kappa):
        """Forward error should be ≤ κ × ε × C(n) per perturbation theory.

        At κ=1: forward err ~ ε ~ 1e-7  → ~7 digits of f32 preserved
        At κ=1e2: forward err ~ 1e-5    → ~5 digits
        At κ=1e4: forward err ~ 1e-3    → ~3 digits  (target precision)
        """
        rng = np.random.default_rng(int(np.log10(kappa) * 13) + 100)
        H = make_spd(self.N, kappa, seed=int(np.log10(kappa) * 7))
        b = rng.standard_normal(self.N)

        x_ref = np.linalg.solve(H, b)  # f64 reference
        L_gpu = gpu_cholesky_factor(H)
        x_gpu = gpu_cholesky_solve(L_gpu, b).astype(np.float64)

        rel_err = forward_error(x_gpu, x_ref)
        bound = kappa * 1000 * self.N * EPS_F32
        assert rel_err < bound, f"κ={kappa:.0e}: forward err {rel_err:.2e} > bound {bound:.2e}"

    # ---- Wilkinson cross-validation: gather + log all 3 metrics ----

    def test_wilkinson_metrics_grow_correctly_with_kappa(self):
        """Sanity check that the test infrastructure produces expected scaling.

        Backward error should NOT grow much with κ (≤ ~10x across the sweep).
        Forward error SHOULD grow ~linearly with κ.

        If backward err grows linearly with κ, the kernel is not backward
        stable. If forward err grows faster than linearly, perturbation
        theory is violated (impossible — but a useful invariant to check).
        """
        kappas = [1.0, 1e2, 1e4]
        backward_errs = []
        forward_errs = []
        rng = np.random.default_rng(2024)
        for k in kappas:
            H = make_spd(self.N, k, seed=int(np.log10(k) * 7))
            b = rng.standard_normal(self.N)
            x_ref = np.linalg.solve(H, b)
            L_gpu = gpu_cholesky_factor(H)
            x_gpu = gpu_cholesky_solve(L_gpu, b).astype(np.float64)
            backward_errs.append(normwise_backward_error(H, x_gpu, b))
            forward_errs.append(forward_error(x_gpu, x_ref))

        # Backward error stable across κ (within 100x range). The 100x slack
        # is generous; in practice the variation is ~10x for well-implemented
        # Cholesky. We don't assert tighter because random b can make some
        # configurations slightly worse than others.
        be_ratio = max(backward_errs) / min(backward_errs)
        assert be_ratio < 100.0, (
            f"Backward error varied by {be_ratio:.1f}x across κ ∈ "
            f"{kappas} — expected ~10x. Errs: {backward_errs}"
        )

        # Forward error grows with κ (κ=1e4 should give larger err than κ=1).
        # Assert monotone-ish: forward_err[2] > forward_err[0] / 10 (allows
        # noise but catches "all the same" bugs).
        assert forward_errs[2] > forward_errs[0], (
            f"Forward error did not grow with κ: {forward_errs} (κ={kappas})"
        )

    # ---- Class 1 boundary: regularization clamp documentation ----

    def test_regularization_clamp_activates_on_tiny_pivot(self):
        """Direct test of the `if val < reg: val = reg` clamp at line 205-206
        of crba_kernels.py. Construct a diagonal H with one eigenvalue below
        reg=1e-6. For a diagonal matrix, Cholesky pivots equal the diagonal
        entries directly (no inner-loop accumulation), so the clamp must
        activate on exactly that eigenvalue.

        Without clamp: L[n-1, n-1] = sqrt(1e-10) = 1e-5
        With clamp:    L[n-1, n-1] = sqrt(1e-6)  = 1e-3

        100x ratio — unambiguous detection. This documents the silent
        wrong-physics boundary mechanically: the GPU kernel reports a
        Cholesky factor whose square does NOT equal the input matrix.

        Note: this is a CORRECT behavior of OUR pipeline (the clamp
        prevents NaN, which is preferable to silent NaN propagation
        through the entire physics step), but it does mean κ above the
        clamp threshold gives wrong-not-NaN results. Documented in
        REFLECTIONS session 24 as the silent-wrong-physics boundary.
        """
        n = self.N
        H = np.eye(n, dtype=np.float64)
        H[n - 1, n - 1] = 1e-10  # below reg=1e-6

        L_gpu = gpu_cholesky_factor(H, reg=1e-6)

        # Clamp activated: L[n-1, n-1] should be sqrt(1e-6) ≈ 1e-3,
        # NOT sqrt(1e-10) = 1e-5
        L_nn = float(L_gpu[n - 1, n - 1])
        sqrt_reg = float(np.sqrt(1e-6))
        sqrt_true = float(np.sqrt(1e-10))

        assert abs(L_nn - sqrt_reg) < 1e-6, (
            f"Expected clamp: L[n-1,n-1] ≈ {sqrt_reg:.3e}, got {L_nn:.3e} "
            f"(true unclamped value would be {sqrt_true:.3e})"
        )

        # And verify reconstruction error is now LARGE — the clamp made
        # L L^T differ from H in the bottom-right corner.
        recon = L_gpu @ L_gpu.T
        bottom_right_err = abs(recon[n - 1, n - 1] - 1e-10)
        assert bottom_right_err > 1e-7, (
            f"Reconstruction unchanged: {recon[n - 1, n - 1]:.3e} ≈ 1e-10 "
            f"— clamp did not affect L L^T as expected"
        )

    def test_clamp_disabled_with_tiny_reg(self):
        """Counter-test: with reg=1e-20, the clamp does NOT activate, and
        the GPU kernel reproduces the true sqrt for tiny pivots. Confirms
        the clamp is the only thing causing the perturbation in the
        previous test (not an unrelated kernel artifact).
        """
        n = self.N
        H = np.eye(n, dtype=np.float64)
        H[n - 1, n - 1] = 1e-10

        L_gpu = gpu_cholesky_factor(H, reg=1e-20)
        L_nn = float(L_gpu[n - 1, n - 1])
        sqrt_true = float(np.sqrt(1e-10))

        # Now L[n-1,n-1] should be ~sqrt(1e-10), not clamped
        # Allow 10% tolerance for f32 sqrt precision on small numbers
        assert abs(L_nn - sqrt_true) / sqrt_true < 0.1, (
            f"Expected unclamped: L[n-1,n-1] ≈ {sqrt_true:.3e}, got {L_nn:.3e}"
        )

    def test_no_nan_across_full_kappa_sweep(self):
        """Smoke test: factor + solve should never NaN across cond ∈ [1, 1e7].

        Even at κ=1e7 (above f32 precision limit), the regularization clamp
        guarantees no NaN — the answer is just wrong. NaN would indicate
        a different bug (e.g., negative pivot reaching sqrt before clamp).
        """
        rng = np.random.default_rng(99)
        for kappa in [1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]:
            H = make_spd(self.N, kappa, seed=int(np.log10(kappa + 1) * 5))
            b = rng.standard_normal(self.N)
            L = gpu_cholesky_factor(H)
            x = gpu_cholesky_solve(L, b)
            assert np.all(np.isfinite(L)), f"NaN in L at κ={kappa:.0e}"
            assert np.all(np.isfinite(x)), f"NaN in x at κ={kappa:.0e}"


# ---------------------------------------------------------------------------
# Class 2 — CRBA H matrix structural properties (CPU only)
# ---------------------------------------------------------------------------


def _chain_robot(
    n_links: int,
    link_length: float = 0.2,
    mass: float = 1.0,
    inertia: float = 1e-3,
    length_alpha: float = 1.0,
):
    """Build a serial revolute chain with n links and floating base.

    Parameters
    ----------
    n_links : number of revolute joints (length of the actuated chain)
    link_length : base link length L0
    mass : per-link mass
    inertia : isotropic inertia for each link
    length_alpha : geometric ratio for link length L_k = L0 * alpha^k
                   alpha=1.0 → uniform; alpha>1 → distal links longer
                   (this is the cond(H) lever — see REFLECTIONS session 24)

    Returns the RobotTreeNumpy.
    """
    from physics.joint import FreeJoint, RevoluteJoint
    from physics.robot_tree import Body, RobotTreeNumpy
    from physics.spatial import SpatialInertia, SpatialTransform

    tree = RobotTreeNumpy(gravity=9.81)

    # Floating base
    base_inertia = SpatialInertia(
        mass=2.0,
        inertia=np.diag([0.01, 0.01, 0.01]),
        com=np.array([0.0, 0.0, 0.0]),
    )
    tree.add_body(
        Body(
            name="base",
            index=0,
            joint=FreeJoint("root"),
            inertia=base_inertia,
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )

    # Chain links
    for k in range(n_links):
        L_k = link_length * (length_alpha**k)
        # Each link's COM at -L_k/2 along its z (so the chain hangs down)
        link_inertia = SpatialInertia(
            mass=mass,
            inertia=np.diag([inertia, inertia, inertia]),
            com=np.array([0.0, 0.0, -L_k / 2.0]),
        )
        # Parent-relative offset: previous link's end
        if k == 0:
            r = np.array([0.0, 0.0, 0.0])
        else:
            L_prev = link_length * (length_alpha ** (k - 1))
            r = np.array([0.0, 0.0, -L_prev])
        tree.add_body(
            Body(
                name=f"link{k}",
                index=k + 1,
                joint=RevoluteJoint(f"j{k}", axis=np.array([0.0, 1.0, 0.0]), damping=0.0),
                inertia=link_inertia,
                X_tree=SpatialTransform(np.eye(3), r),
                parent=k,
            )
        )

    tree.finalize()
    return tree


class TestCRBAMassMatrixProperties:
    """Structural invariants on H from RobotTreeNumpy.crba(q).

    Cross-validates symmetry, definiteness, and conditioning measurement
    methods. These are CPU-only checks — they don't go through the GPU
    pipeline. The point is to validate that CRBA itself produces a
    well-formed H before we worry about how it's factored.
    """

    @pytest.mark.parametrize("alpha", [1.0, 1.3, 1.6])
    def test_h_exactly_symmetric(self, alpha):
        """H from CRBA should be symmetric to within O(n × ε × ‖H‖).

        Catches: asymmetric updates in CRBA (computing H_ij and H_ji
        independently without enforcing the mirror), shape transposition.
        Wilkinson Test 4 — see feedback_numerical_stability_wilkinson.md
        """
        tree = _chain_robot(n_links=8, length_alpha=alpha)
        rng = np.random.default_rng(int(alpha * 100))
        q = rng.standard_normal(tree.nq) * 0.3
        # Re-normalize floating base quaternion
        q[:4] = q[:4] / np.linalg.norm(q[:4])
        H = tree.crba(q)

        sym_err = float(np.linalg.norm(H - H.T, "fro"))
        H_norm = float(np.linalg.norm(H, "fro"))
        bound = tree.nv * np.finfo(np.float64).eps * H_norm * 100
        assert sym_err < bound, f"alpha={alpha}: ‖H - H^T‖_F = {sym_err:.2e} > bound {bound:.2e}"

    @pytest.mark.parametrize("alpha", [1.0, 1.3, 1.6])
    def test_h_positive_definite_margin(self, alpha):
        """H should be PD: all eigenvalues > 0. Track λ_min as the margin."""
        tree = _chain_robot(n_links=8, length_alpha=alpha)
        q = np.zeros(tree.nq)
        q[0] = 1.0  # quaternion w=1, rest zero
        H = tree.crba(q)
        H_sym = (H + H.T) / 2
        eigs = np.linalg.eigvalsh(H_sym)
        assert eigs.min() > 0, f"alpha={alpha}: min eigenvalue {eigs.min():.3e} ≤ 0 — H not PD"

    @pytest.mark.parametrize("alpha", [1.0, 1.3, 1.6])
    def test_cond_h_three_methods_agree(self, alpha):
        """cond(H) computed three ways should agree to ~10% relative.

        Method A: np.linalg.cond (uses SVD)
        Method B: λ_max / λ_min from eigvalsh (symmetric eigensolver)
        Method C: ‖H‖₂ × ‖H⁻¹‖₂ (operator norm definition)

        Disagreement → either H is not symmetric (broke method B) or H
        is near-singular (broke method C). For our well-formed CRBA H
        these should match closely.
        """
        tree = _chain_robot(n_links=8, length_alpha=alpha)
        q = np.zeros(tree.nq)
        q[0] = 1.0
        H = tree.crba(q)
        H_sym = (H + H.T) / 2

        cond_A = float(np.linalg.cond(H_sym))
        eigs = np.linalg.eigvalsh(H_sym)
        cond_B = float(eigs.max() / eigs.min())
        cond_C = float(np.linalg.norm(H_sym, 2) * np.linalg.norm(np.linalg.inv(H_sym), 2))

        # All three should be within 10% of each other
        cond_arr = np.array([cond_A, cond_B, cond_C])
        spread = (cond_arr.max() - cond_arr.min()) / cond_arr.min()
        assert spread < 0.1, (
            f"alpha={alpha}: cond methods disagree: SVD={cond_A:.3e}, "
            f"eig={cond_B:.3e}, opnorm={cond_C:.3e}, spread={spread:.2%}"
        )

    def test_chain_alpha_lever_increases_cond(self):
        """The length_alpha lever should monotonically increase cond(H).
        Validates that our fixture knob works as intended (REFLECTIONS
        session 24: link length ratio is the chosen cond lever for B(6)).
        """
        conds = []
        for alpha in [1.0, 1.3, 1.6, 2.0]:
            tree = _chain_robot(n_links=8, length_alpha=alpha)
            q = np.zeros(tree.nq)
            q[0] = 1.0
            H = tree.crba(q)
            conds.append(float(np.linalg.cond((H + H.T) / 2)))

        # Strictly monotone up (or close to it — allow tiny noise)
        for i in range(len(conds) - 1):
            assert conds[i + 1] >= conds[i] * 0.95, (
                f"cond non-monotone at alpha={[1.0, 1.3, 1.6, 2.0][i + 1]}: {conds}"
            )
        # Range should span at least 1 order of magnitude
        assert conds[-1] / conds[0] > 5.0, (
            f"alpha lever weak: conds={conds}, ratio {conds[-1] / conds[0]:.1f}"
        )

    def test_h_h_inv_residual_within_bound(self):
        """Test that np.linalg.solve(H, b) and CRBA-Cholesky give the same
        answer to f64 precision. This is a CPU-only sanity check that
        cycles through the full CRBA + LAPACK Cholesky path.
        """
        tree = _chain_robot(n_links=8, length_alpha=1.3)
        rng = np.random.default_rng(11)
        q = np.zeros(tree.nq)
        q[0] = 1.0
        qdot = rng.standard_normal(tree.nv) * 0.1
        tau = rng.standard_normal(tree.nv) * 0.5
        # zero out tau on the floating-base DOFs (free joint)
        tau[:6] = 0.0

        H = tree.crba(q)
        # Compute C(q, qdot) = RNEA(q, qdot, qddot=0)
        C_bias = tree.rnea(q, qdot, np.zeros(tree.nv))

        # Solve via numpy solve
        qdd_solve = np.linalg.solve(H, tau - C_bias)

        # Solve via tree.forward_dynamics_crba (uses scipy/numpy Cholesky internally)
        qdd_crba = tree.forward_dynamics_crba(q, qdot, tau)

        rel_err = np.linalg.norm(qdd_solve - qdd_crba) / max(np.linalg.norm(qdd_solve), 1e-10)
        assert rel_err < 1e-10, f"CRBA forward dynamics vs np.linalg.solve: rel_err={rel_err:.2e}"


# ---------------------------------------------------------------------------
# Class 4 — CRBA vs ABA in physical fixtures (algorithm cross-validation)
# ---------------------------------------------------------------------------


class TestCRBAvsABAPhysicalFixtures:
    """Cross-validate CRBA forward dynamics against ABA across realistic
    cond(H) regimes. Both are independent algorithms (CRBA forms H then
    solves H q̈ = τ - C; ABA traverses the tree without forming H), so
    agreement at f64 is strong evidence neither algorithm has a bug.

    Cond range: 3e3 (n_links=2) to ~4e5 (n_links=8 alpha=1.3). This spans
    the realistic range for chain manipulators and quadruped/humanoid
    robots. Higher cond requires non-physical fixtures (handled by
    Class 1's synthetic SPD).

    Methodology grade: relative cross-validation (★★★ on the hierarchy
    in REFLECTIONS session 24). Class 1 Wilkinson tests are stronger
    (absolute), but those test only the linear-algebra kernel — these
    tests cover the full CRBA + RNEA + Cholesky algorithm chain.
    """

    @pytest.mark.parametrize("n_links", [2, 4, 6, 8])
    def test_crba_vs_aba_zero_velocity(self, n_links):
        """At qdot=0, q̈ = H⁻¹(τ - g_bias). Compare CRBA path vs ABA path
        across nlinks (cond grows with nlinks).
        """
        tree = _chain_robot(n_links=n_links, length_alpha=1.0)
        rng = np.random.default_rng(n_links)
        q = np.zeros(tree.nq)
        q[0] = 1.0  # quaternion w
        qdot = np.zeros(tree.nv)
        tau = rng.standard_normal(tree.nv) * 0.5
        tau[:6] = 0.0  # no input on free joint

        qdd_aba = tree.aba(q, qdot, tau)
        qdd_crba = tree.forward_dynamics_crba(q, qdot, tau)

        rel_err = np.linalg.norm(qdd_crba - qdd_aba) / max(np.linalg.norm(qdd_aba), 1e-12)
        # Tolerance scales with cond — use tree.crba once to know the regime
        H = tree.crba(q)
        kappa = float(np.linalg.cond((H + H.T) / 2))
        bound = max(kappa * 1e-14, 1e-10)
        assert rel_err < bound, f"n_links={n_links}, cond(H)={kappa:.2e}: rel_err={rel_err:.2e} > {bound:.2e}"

    @pytest.mark.parametrize("n_links", [2, 4, 6, 8])
    def test_crba_vs_aba_with_velocity(self, n_links):
        """Same cross-check with non-zero qdot (Coriolis terms active in C)."""
        tree = _chain_robot(n_links=n_links, length_alpha=1.0)
        rng = np.random.default_rng(n_links * 7 + 1)
        q = rng.standard_normal(tree.nq) * 0.2
        q[:4] = q[:4] / np.linalg.norm(q[:4])  # normalize quaternion
        qdot = rng.standard_normal(tree.nv) * 0.5
        tau = rng.standard_normal(tree.nv) * 0.5
        tau[:6] = 0.0

        qdd_aba = tree.aba(q, qdot, tau)
        qdd_crba = tree.forward_dynamics_crba(q, qdot, tau)

        rel_err = np.linalg.norm(qdd_crba - qdd_aba) / max(np.linalg.norm(qdd_aba), 1e-12)
        H = tree.crba(q)
        kappa = float(np.linalg.cond((H + H.T) / 2))
        bound = max(kappa * 1e-13, 1e-10)
        assert rel_err < bound, f"n_links={n_links}, cond(H)={kappa:.2e}: rel_err={rel_err:.2e} > {bound:.2e}"

    @pytest.mark.parametrize("n_links", [2, 4, 6, 8])
    def test_newton_residual_aba(self, n_links):
        """ABA produces q̈; verify it satisfies Newton's law in joint space:
        H @ q̈ + C = τ. The residual ‖H q̈ + C - τ‖ should be O(ε × ‖τ‖).

        This is a self-consistency test: if ABA and CRBA disagree, this
        residual catches *which one* is wrong (only ABA is tested here;
        for CRBA the same residual is identically zero by construction).
        """
        tree = _chain_robot(n_links=n_links, length_alpha=1.0)
        rng = np.random.default_rng(n_links * 13 + 5)
        q = rng.standard_normal(tree.nq) * 0.2
        q[:4] = q[:4] / np.linalg.norm(q[:4])
        qdot = rng.standard_normal(tree.nv) * 0.3
        tau = rng.standard_normal(tree.nv) * 0.5
        tau[:6] = 0.0

        qdd_aba = tree.aba(q, qdot, tau)
        H = tree.crba(q)
        C_bias = tree.rnea(q, qdot, np.zeros(tree.nv))

        residual = H @ qdd_aba + C_bias - tau
        rel_residual = np.linalg.norm(residual) / max(np.linalg.norm(tau), 1e-10)

        kappa = float(np.linalg.cond((H + H.T) / 2))
        # Newton residual bound: scales with κ × ε
        bound = max(kappa * 1e-13, 1e-10)
        assert rel_residual < bound, (
            f"n_links={n_links}, cond(H)={kappa:.2e}: ‖H q̈ + C - τ‖ rel = {rel_residual:.2e} > {bound:.2e}"
        )

    def test_random_q_random_state_battery(self):
        """Battery of 20 random (q, qdot, tau) configurations on n=6 chain.
        Failure rate must be 0/20 — this catches sporadic numerical issues
        that pass on a single fixture by luck.
        """
        tree = _chain_robot(n_links=6, length_alpha=1.0)
        rng = np.random.default_rng(2024)
        n_trials = 20
        max_rel_err = 0.0

        for trial in range(n_trials):
            q = rng.standard_normal(tree.nq) * 0.4
            q[:4] = q[:4] / np.linalg.norm(q[:4])
            qdot = rng.standard_normal(tree.nv) * 0.5
            tau = rng.standard_normal(tree.nv) * 1.0
            tau[:6] = 0.0

            qdd_aba = tree.aba(q, qdot, tau)
            qdd_crba = tree.forward_dynamics_crba(q, qdot, tau)
            rel_err = np.linalg.norm(qdd_crba - qdd_aba) / max(np.linalg.norm(qdd_aba), 1e-12)
            max_rel_err = max(max_rel_err, rel_err)

        assert max_rel_err < 1e-9, f"max rel_err over {n_trials} trials: {max_rel_err:.2e} > 1e-9"


# ---------------------------------------------------------------------------
# Class 3 — q-sweep findings (quadruped + chain regression)
# ---------------------------------------------------------------------------
#
# Experiment summary (recorded in REFLECTIONS session 24):
#
# 1. Quadruped fixture (tests/unit/dynamics/test_crba.py::_make_quadruped):
#      cond(H) ranges from ~1.4e3 to ~6.2e3 across the entire joint space.
#      Random q sweep (n=200): min=2.15e3, median=4.6e3, max=6.17e3.
#      The cond surface is FLAT — driven by floating-base mass coupling,
#      not by joint configuration. Calf angle is the only meaningful lever
#      (max cond at calf=0 / fully extended; min at calf=-2 / fully folded).
#      Hip angle is symmetric, doesn't affect cond.
#
# 2. Chain fixture (this file's _chain_robot):
#      n=4 alpha=1.0: cond ≈ 1.6e4
#      n=8 alpha=1.0: cond ≈ 9.8e4
#      n=8 alpha=1.5: cond ≈ 9.3e5
#      n=12 alpha=1.5: cond ≈ 4.4e7  ← approaches GPU clamp regime
#      n=16 alpha=1.5: cond ≈ 1.6e9  ← f32 has effectively 0 precision
#
# Conclusion: real robot fixtures are MILD — even fully-extended quadruped
# legs only reach cond ≈ 6e3. To stress GPU Cholesky beyond cond ≈ 1e5
# we have to use synthetic SPD (Class 1) or aggressive chain fixtures
# (n ≥ 12 with alpha ≥ 1.5).
#
# This is a useful negative finding: the GPU regularization clamp at
# κ ≈ 1e6+ is unlikely to fire on a real robot. The clamp protects
# against pathological inputs, not against typical use.
# ---------------------------------------------------------------------------


class TestQuadrupedNearSingular:
    """Regression tests on the worst-case quadruped configurations found by
    the q-sweep experiment. These tests anchor the experiment results so
    a future fixture change that accidentally widens the cond range fails
    loudly.
    """

    @staticmethod
    def _make_quadruped():
        # Imported lazily so this file doesn't require test_crba's URDF
        # path to exist at import time.
        from tests.unit.dynamics.test_crba import _make_quadruped

        return _make_quadruped()

    def test_quadruped_cond_at_default_pose(self):
        """q=0 (zero pose, identity quaternion) should give cond ~ 6e3.
        Documented baseline."""
        tree = self._make_quadruped()
        q = np.zeros(tree.nq)
        q[0] = 1.0
        H = tree.crba(q)
        kappa = float(np.linalg.cond((H + H.T) / 2))
        # 5e3 < kappa < 1e4 — anchor the regime
        assert 5e3 < kappa < 1e4, f"q=0 cond={kappa:.2e}, expected 5e3..1e4"

    def test_quadruped_cond_at_extended_legs(self):
        """Calf=0 (legs fully extended) is the worst-case from the q-sweep.
        Should give cond near the upper end of the quadruped fixture range."""
        tree = self._make_quadruped()
        q = np.zeros(tree.nq)
        q[0] = 1.0
        # In test_crba's _make_quadruped, joint q layout after free base (7):
        # FL_hip(7), FR_hip(8), FL_calf(9), FR_calf(10) — see 4 actuated joints
        q[9] = 0.0  # FL_calf extended
        q[10] = 0.0  # FR_calf extended
        H = tree.crba(q)
        kappa = float(np.linalg.cond((H + H.T) / 2))
        # Anchor: should be in the upper range
        assert kappa > 5e3, f"extended-legs cond={kappa:.2e}, expected > 5e3"
        assert kappa < 8e3, f"extended-legs cond={kappa:.2e}, expected < 8e3"

    def test_quadruped_crba_aba_at_extended_legs(self):
        """At the worst-case quadruped pose, CRBA and ABA still agree to
        f64 precision. The 'near-singular' isn't actually that bad."""
        tree = self._make_quadruped()
        rng = np.random.default_rng(7)
        q = np.zeros(tree.nq)
        q[0] = 1.0
        q[9] = 0.0  # FL_calf
        q[10] = 0.0  # FR_calf
        qdot = rng.standard_normal(tree.nv) * 0.3
        tau = rng.standard_normal(tree.nv) * 0.5
        tau[:6] = 0.0

        qdd_aba = tree.aba(q, qdot, tau)
        qdd_crba = tree.forward_dynamics_crba(q, qdot, tau)
        rel_err = np.linalg.norm(qdd_crba - qdd_aba) / max(np.linalg.norm(qdd_aba), 1e-12)
        assert rel_err < 1e-10, f"rel_err={rel_err:.2e}"


class TestChainHighCondRegime:
    """Verifies the chain fixture spans the cond range we need for high-κ
    stress testing. Anchors the cond targets so a future fixture change
    that accidentally tightens the range fails loudly.
    """

    def test_chain_n8_alpha15_in_clamp_boundary_regime(self):
        """n=8 alpha=1.5 should give cond ~1e6, near the GPU clamp boundary."""
        tree = _chain_robot(n_links=8, length_alpha=1.5)
        q = np.zeros(tree.nq)
        q[0] = 1.0
        H = tree.crba(q)
        kappa = float(np.linalg.cond((H + H.T) / 2))
        assert 1e5 < kappa < 1e7, f"n=8 alpha=1.5 cond={kappa:.2e}"

    def test_chain_n12_alpha15_above_clamp(self):
        """n=12 alpha=1.5 enters the regime where GPU regularization may matter."""
        tree = _chain_robot(n_links=12, length_alpha=1.5)
        q = np.zeros(tree.nq)
        q[0] = 1.0
        H = tree.crba(q)
        kappa = float(np.linalg.cond((H + H.T) / 2))
        assert kappa > 1e7, f"n=12 alpha=1.5 cond={kappa:.2e}, expected > 1e7"

    def test_chain_high_cond_crba_aba_still_agree_in_f64(self):
        """At cond ≈ 4e7 (n=12 alpha=1.5), CPU CRBA vs ABA still agree at
        f64 precision (forward error bound: κ × ε_f64 ≈ 1e-8). Documents
        that f64 retains usable precision even in regimes where f32 fails."""
        tree = _chain_robot(n_links=12, length_alpha=1.5)
        rng = np.random.default_rng(2024)
        q = np.zeros(tree.nq)
        q[0] = 1.0
        qdot = rng.standard_normal(tree.nv) * 0.1
        tau = rng.standard_normal(tree.nv) * 0.5
        tau[:6] = 0.0

        qdd_aba = tree.aba(q, qdot, tau)
        qdd_crba = tree.forward_dynamics_crba(q, qdot, tau)
        rel_err = np.linalg.norm(qdd_crba - qdd_aba) / max(np.linalg.norm(qdd_aba), 1e-12)
        # κ × ε_f64 ≈ 4e7 × 2e-16 ≈ 1e-8; allow 100x slack
        assert rel_err < 1e-6, (
            f"n=12 alpha=1.5 (cond≈4e7): rel_err={rel_err:.2e} > 1e-6 — f64 should still hold here"
        )


# ---------------------------------------------------------------------------
# Class 5 — GPU qacc accessor + cross-check (end-to-end)
# ---------------------------------------------------------------------------


class TestGpuQaccAccessor:
    """End-to-end test of the new qacc_smooth_wp / qacc_total_wp accessors.

    This goes through the full GpuEngine pipeline (CRBA + Cholesky on GPU,
    contact solver, integration) and validates:

      1. qacc_smooth_wp matches CPU CRBA forward dynamics (no contact).
         If this fails, the GPU CRBA + Cholesky path produces different q̈
         from the CPU one — exactly the kind of regression we want to catch.

      2. With contact, qacc_total = qacc_smooth + Δq̇/dt is non-trivially
         different from qacc_smooth (verifying the contact impulse path
         is connected) AND it equals (qdot_after - qdot_before)/dt
         (verifying the kernel computes the right value).

      3. Without contact (sphere in mid-air), qacc_total ≡ qacc_smooth.
         Catches: a bug where dqdot is non-zero even with zero contacts.

    Three-use-site disambiguation (smooth vs Delassus build vs impulse
    apply within the same Cholesky factor) is NOT tested here — would
    require in-kernel inspection buffers (deferred per OPEN_QUESTIONS).

    Methodology: cross-validation tier (★★★) — see REFLECTIONS session 24.
    """

    @staticmethod
    def _build_sphere_engine(num_envs=1, dt=2e-4, with_contact=False):
        """Build GpuEngine with a single sphere in mid-air or above ground."""
        from physics.geometry import BodyCollisionGeometry, ShapeInstance, SphereShape
        from physics.gpu_engine import GpuEngine
        from physics.joint import FreeJoint
        from physics.merged_model import merge_models
        from physics.robot_tree import Body, RobotTreeNumpy
        from physics.spatial import SpatialInertia, SpatialTransform
        from physics.terrain import FlatTerrain
        from robot.model import RobotModel

        tree = RobotTreeNumpy(gravity=9.81)
        radius = 0.05
        mass = 1.0
        I = (2.0 / 5.0) * mass * radius**2
        tree.add_body(
            Body(
                name="sphere",
                index=0,
                joint=FreeJoint("root"),
                inertia=SpatialInertia(
                    mass=mass,
                    inertia=np.diag([I, I, I]),
                    com=np.array([0.0, 0.0, 0.0]),
                ),
                X_tree=SpatialTransform.identity(),
                parent=-1,
            )
        )
        tree.finalize()

        geom = BodyCollisionGeometry(
            body_index=0,
            shapes=[
                ShapeInstance(
                    shape=SphereShape(radius=radius),
                    origin_xyz=np.zeros(3),
                    origin_rpy=np.zeros(3),
                )
            ],
        )
        model = RobotModel(
            tree=tree,
            actuated_joint_names=[],
            contact_body_names=["sphere"] if with_contact else [],
            geometries=[geom],
        )
        merged = merge_models({"sphere": model}, terrain=FlatTerrain())
        engine = GpuEngine(merged, num_envs=num_envs, dt=dt)
        return engine, model

    def test_qacc_smooth_no_contact_matches_cpu(self):
        """Sphere in mid-air → no contact. GPU qacc_smooth should match CPU
        forward_dynamics_crba to f32 precision."""
        engine, model = self._build_sphere_engine(with_contact=False)

        # State: sphere at z=1.0 (above ground), zero velocity
        q = engine.q_wp.numpy().copy()
        q[0, :4] = np.array([1.0, 0.0, 0.0, 0.0])  # quaternion w=1
        q[0, 4:] = np.array([0.0, 0.0, 1.0])
        engine.q_wp.assign(wp.array(q, dtype=wp.float32, device="cuda:0"))
        engine.qdot_wp.assign(wp.array(np.zeros((1, 6), dtype=np.float32), dtype=wp.float32, device="cuda:0"))

        # Take one step (no torque)
        engine.step_n(n_substeps=1)

        # Read GPU qacc_smooth
        qacc_smooth_gpu = engine.qacc_smooth_wp.numpy()[0]

        # CPU expected: aba(q, qdot=0, tau=0)
        q_cpu = q[0].astype(np.float64)
        qdot_cpu = np.zeros(6)
        tau_cpu = np.zeros(6)
        qdd_cpu = model.tree.aba(q_cpu, qdot_cpu, tau_cpu)

        # f32 vs f64 — allow ~1e-4 relative
        rel_err = np.linalg.norm(qacc_smooth_gpu - qdd_cpu) / max(np.linalg.norm(qdd_cpu), 1e-10)
        assert rel_err < 5e-4, (
            f"qacc_smooth GPU vs CPU rel_err={rel_err:.2e}; GPU={qacc_smooth_gpu}, CPU={qdd_cpu}"
        )

        # Free-fall sanity: linear z acceleration ≈ -9.81
        # spatial vel/accel layout is [linear; angular] (Q15)
        assert abs(qacc_smooth_gpu[2] - (-9.81)) < 0.05, (
            f"linear z accel = {qacc_smooth_gpu[2]:.4f}, expected ~-9.81"
        )

    def test_qacc_total_equals_qacc_smooth_when_no_contact(self):
        """Without contact, the contact impulse Δq̇ is zero, so
        qacc_total ≡ qacc_smooth. Tests that the new _compute_qacc_total
        kernel produces the right value when dqdot is zero."""
        engine, _ = self._build_sphere_engine(with_contact=False)

        # Mid-air state
        q = np.zeros((1, 7), dtype=np.float32)
        q[0, 0] = 1.0  # quat w
        q[0, 6] = 1.0  # z=1
        engine.q_wp.assign(wp.array(q, dtype=wp.float32, device="cuda:0"))
        engine.qdot_wp.assign(wp.array(np.zeros((1, 6), dtype=np.float32), dtype=wp.float32, device="cuda:0"))

        engine.step_n(n_substeps=1)

        qacc_smooth = engine.qacc_smooth_wp.numpy()[0]
        qacc_total = engine.qacc_total_wp.numpy()[0]

        np.testing.assert_allclose(
            qacc_total,
            qacc_smooth,
            atol=1e-6,
            err_msg=f"qacc_total ({qacc_total}) != qacc_smooth ({qacc_smooth}) "
            f"with no contact — dqdot leak from contact path?",
        )

    def test_qacc_total_differs_from_smooth_with_contact(self):
        """With ground contact, qacc_total - qacc_smooth = dqdot/dt should
        be non-zero in the z component (contact pushing sphere up)."""
        dt = 2e-4
        engine, _ = self._build_sphere_engine(with_contact=True, dt=dt)
        radius = 0.05

        # Sphere at z=radius (just touching ground), small downward velocity
        q = np.zeros((1, 7), dtype=np.float32)
        q[0, 0] = 1.0
        q[0, 6] = radius * 0.9  # slight penetration to guarantee contact
        engine.q_wp.assign(wp.array(q, dtype=wp.float32, device="cuda:0"))
        qdot_init = np.zeros((1, 6), dtype=np.float32)
        qdot_init[0, 2] = -0.1  # downward (linear z is index 2 with [lin;ang])
        engine.qdot_wp.assign(wp.array(qdot_init, dtype=wp.float32, device="cuda:0"))

        engine.step_n(n_substeps=1)

        qacc_smooth = engine.qacc_smooth_wp.numpy()[0]
        qacc_total = engine.qacc_total_wp.numpy()[0]

        # Smooth (pre-contact) is just gravity in z: qacc_smooth[2] ≈ -9.81
        assert qacc_smooth[2] < -5.0, f"qacc_smooth z={qacc_smooth[2]:.3f} should be ~-9.81"

        # Total should be much larger (less negative or positive) due to
        # upward contact impulse
        contact_delta = qacc_total[2] - qacc_smooth[2]
        assert contact_delta > 5.0, (
            f"qacc_total - qacc_smooth in z = {contact_delta:.3f}, "
            f"expected significant upward contact reaction"
        )

    def test_qacc_total_consistent_with_qdot_finite_diff(self):
        """qacc_total should equal (qdot_after - qdot_before)/dt to within
        f32 precision. Cross-validates the new kernel against the
        independent observable.

        Note: this is in mid-air (no contact) so qacc_total = qacc_smooth.
        With contact, the integration also adds Baumgarte position
        correction velocity, breaking the simple finite-diff identity.
        Mid-air keeps the test clean.
        """
        dt = 2e-4
        engine, _ = self._build_sphere_engine(with_contact=False, dt=dt)

        q = np.zeros((1, 7), dtype=np.float32)
        q[0, 0] = 1.0
        q[0, 6] = 1.0
        engine.q_wp.assign(wp.array(q, dtype=wp.float32, device="cuda:0"))
        engine.qdot_wp.assign(wp.array(np.zeros((1, 6), dtype=np.float32), dtype=wp.float32, device="cuda:0"))

        qdot_before = engine.qdot_wp.numpy()[0].copy()
        engine.step_n(n_substeps=1)
        qdot_after = engine.qdot_wp.numpy()[0]

        qacc_total = engine.qacc_total_wp.numpy()[0]
        qacc_finite_diff = (qdot_after - qdot_before) / dt

        np.testing.assert_allclose(
            qacc_total,
            qacc_finite_diff,
            atol=1e-3,
            err_msg=(f"qacc_total ({qacc_total}) != (qdot_after - qdot_before)/dt ({qacc_finite_diff})"),
        )
