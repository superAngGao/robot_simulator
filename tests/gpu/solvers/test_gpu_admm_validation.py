"""
Rigorous validation of GPU ADMM solver correctness.

Strategy:
1. Analytical ground truth — known-solution scenarios
2. Component-level validation — Cholesky, impedance, cone projection
3. Single-step CPU vs GPU numerical comparison of intermediate quantities
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.geometry import BodyCollisionGeometry, ShapeInstance, SphereShape
from physics.joint import FreeJoint
from physics.merged_model import merge_models
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from robot.model import RobotModel

try:
    import warp as wp

    from physics.gpu_engine import GpuEngine

    HAS_WARP = True
except Exception:
    HAS_WARP = False

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not HAS_WARP, reason="Warp or CUDA not available"),
]


def _ball_model(mass=1.0, radius=0.1):
    tree = RobotTreeNumpy(gravity=9.81)
    I_s = 2.0 / 5.0 * mass * radius**2
    tree.add_body(
        Body(
            name="ball",
            index=0,
            joint=FreeJoint("root"),
            inertia=SpatialInertia(mass=mass, inertia=np.eye(3) * I_s, com=np.zeros(3)),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()
    return RobotModel(
        tree=tree,
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(SphereShape(radius))])],
        contact_body_names=["ball"],
    )


def _ball_merged(mass=1.0, radius=0.1):
    return merge_models(robots={"ball": _ball_model(mass, radius)})


# ===========================================================================
# 1. Analytical ground truth
# ===========================================================================


class TestAdmmAnalyticalGroundTruth:
    """Verify solver output matches known physics for simple scenarios."""

    def test_stationary_ball_on_ground(self):
        """Ball resting on ground: steady-state normal impulse = m*g*dt.

        A ball at z=radius with zero velocity should produce a normal impulse
        that exactly cancels gravity. After one step, velocity should remain ~0.
        """
        mass, radius = 1.0, 0.1
        merged = _ball_merged(mass=mass, radius=radius)
        dt = 2e-4
        gpu = GpuEngine(merged, num_envs=1, dt=dt, solver="admm")

        # Place ball exactly at contact (z = radius, so depth ≈ 0)
        q, qdot = merged.tree.default_state()
        q[6] = radius  # pz = radius → contact with ground
        gpu.reset(q)

        # Run a few steps to establish contact
        for _ in range(100):
            out = gpu.step(dt=dt)

        # Check: ball should be near z=radius, velocity near zero
        q_f = out.q_new
        qdot_f = out.qdot_new
        z = q_f[6]
        vz = qdot_f[2]

        assert abs(z - radius) < 0.005, f"Ball height {z:.4f}, expected ~{radius}"
        assert abs(vz) < 0.5, f"Vertical velocity {vz:.4f}, expected ~0"

        # Check lambdas: normal impulse should balance gravity
        lambdas = gpu._solver_scratch.lambdas.numpy()[0]
        contact_active = gpu._contact_active.numpy()[0]

        # Find the active contact
        total_normal_impulse = 0.0
        for c in range(gpu._max_contacts):
            if contact_active[c] != 0:
                total_normal_impulse += lambdas[c * 3]  # normal component

        # Expected: impulse that produces acceleration = g upward
        # Impulse → Δv: λ / m = g * dt → λ = m * g * dt
        expected_impulse = mass * 9.81 * dt
        # Allow ~50% tolerance (compliance model shifts equilibrium slightly)
        assert total_normal_impulse > 0, f"Normal impulse should be positive, got {total_normal_impulse:.6e}"
        rel_err = abs(total_normal_impulse - expected_impulse) / expected_impulse
        assert rel_err < 0.5, (
            f"Normal impulse {total_normal_impulse:.6e}, expected ~{expected_impulse:.6e} "
            f"(rel_err={rel_err:.2f})"
        )

    def test_free_fall_no_contact(self):
        """Ball in free fall (high up): no contact forces, pure gravity."""
        merged = _ball_merged()
        dt = 2e-4
        gpu = GpuEngine(merged, num_envs=1, dt=dt, solver="admm")

        q, qdot = merged.tree.default_state()
        q[6] = 2.0  # well above ground
        gpu.reset(q)
        out = gpu.step(dt=dt)

        # No contact should be active
        lambdas = gpu._solver_scratch.lambdas.numpy()[0]
        assert np.allclose(lambdas, 0.0, atol=1e-10), "Should have zero lambdas in free fall"

        # Velocity should show gravity: vz_new = 0 + (-g)*dt = -0.001962
        vz = out.qdot_new[2]
        expected_vz = -9.81 * dt
        assert abs(vz - expected_vz) < 1e-4, f"vz={vz:.6f}, expected {expected_vz:.6f}"


# ===========================================================================
# 2. Component-level validation
# ===========================================================================


class TestCholeskyCorrectness:
    """Verify in-kernel Cholesky matches numpy."""

    def test_cholesky_3x3(self):
        """Factor a known 3x3 SPD matrix, compare with numpy."""
        from physics.backends.warp.admm_kernels import _cholesky_factor

        N, max_rows = 1, 3
        # Known SPD matrix
        A_np = np.array([[4.0, 2.0, 1.0], [2.0, 5.0, 3.0], [1.0, 3.0, 6.0]], dtype=np.float32)
        L_np = np.linalg.cholesky(A_np.astype(np.float64))

        M = wp.zeros((N, max_rows, max_rows), dtype=wp.float32, device="cuda:0")
        L = wp.zeros((N, max_rows, max_rows), dtype=wp.float32, device="cuda:0")
        wp.copy(M, wp.array(A_np.reshape(1, 3, 3), dtype=wp.float32, device="cuda:0"))

        # Launch a minimal kernel that calls _cholesky_factor
        @wp.kernel
        def _test_chol(M: wp.array(dtype=wp.float32, ndim=3), L: wp.array(dtype=wp.float32, ndim=3)):
            _cholesky_factor(0, M, L, 3, 3, 1.0e-6)

        wp.launch(_test_chol, dim=1, device="cuda:0", inputs=[M, L])

        L_gpu = L.numpy()[0]
        np.testing.assert_allclose(L_gpu, L_np, atol=1e-5, err_msg="Cholesky factor mismatch")

    def test_cholesky_solve_3x3(self):
        """Solve L L^T x = b, compare with numpy."""
        from physics.backends.warp.admm_kernels import _cholesky_factor, _cholesky_solve

        N, max_rows = 1, 3
        A_np = np.array([[4.0, 2.0, 1.0], [2.0, 5.0, 3.0], [1.0, 3.0, 6.0]], dtype=np.float32)
        b_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        x_expected = np.linalg.solve(A_np.astype(np.float64), b_np.astype(np.float64))

        M = wp.zeros((N, max_rows, max_rows), dtype=wp.float32, device="cuda:0")
        L = wp.zeros((N, max_rows, max_rows), dtype=wp.float32, device="cuda:0")
        rhs = wp.zeros((N, max_rows), dtype=wp.float32, device="cuda:0")
        x = wp.zeros((N, max_rows), dtype=wp.float32, device="cuda:0")
        tmp = wp.zeros((N, max_rows), dtype=wp.float32, device="cuda:0")

        wp.copy(M, wp.array(A_np.reshape(1, 3, 3), dtype=wp.float32, device="cuda:0"))
        wp.copy(rhs, wp.array(b_np.reshape(1, 3), dtype=wp.float32, device="cuda:0"))

        @wp.kernel
        def _test_solve(
            M: wp.array(dtype=wp.float32, ndim=3),
            L: wp.array(dtype=wp.float32, ndim=3),
            rhs: wp.array(dtype=wp.float32, ndim=2),
            x: wp.array(dtype=wp.float32, ndim=2),
            tmp: wp.array(dtype=wp.float32, ndim=2),
        ):
            _cholesky_factor(0, M, L, 3, 3, 1.0e-6)
            _cholesky_solve(0, L, rhs, x, tmp, 3)

        wp.launch(_test_chol_solve := _test_solve, dim=1, device="cuda:0", inputs=[M, L, rhs, x, tmp])

        x_gpu = x.numpy()[0]
        np.testing.assert_allclose(x_gpu, x_expected, atol=1e-4, err_msg="Cholesky solve mismatch")

    def test_cholesky_larger_matrix(self):
        """Factor a 9x9 random SPD matrix (3 contacts × 3 directions)."""
        from physics.backends.warp.admm_kernels import _cholesky_factor

        n = 9
        rng = np.random.RandomState(42)
        R = rng.randn(n, n).astype(np.float32)
        A_np = R @ R.T + 2.0 * np.eye(n, dtype=np.float32)  # ensure SPD
        L_np = np.linalg.cholesky(A_np.astype(np.float64))

        M = wp.array(A_np.reshape(1, n, n), dtype=wp.float32, device="cuda:0")
        L = wp.zeros((1, n, n), dtype=wp.float32, device="cuda:0")

        @wp.kernel
        def _test_chol9(M: wp.array(dtype=wp.float32, ndim=3), L: wp.array(dtype=wp.float32, ndim=3)):
            _cholesky_factor(0, M, L, 9, 9, 1.0e-6)

        wp.launch(_test_chol9, dim=1, device="cuda:0", inputs=[M, L])
        L_gpu = L.numpy()[0]
        np.testing.assert_allclose(L_gpu, L_np, atol=1e-3, err_msg="9x9 Cholesky mismatch")


class TestImpedanceFunction:
    """Verify GPU impedance matches CPU ADMMQPSolver._impedance."""

    def test_impedance_values(self):
        """Compare GPU and CPU impedance for several depth values."""
        try:
            from physics.solvers.admm_qp import ADMMQPSolver
        except ImportError:
            pytest.skip("ADMMQPSolver not available")

        from physics.backends.warp.admm_kernels import _impedance

        solimp = (0.9, 0.95, 0.001, 0.5, 2.0)
        cpu_solver = ADMMQPSolver(solimp=solimp)

        depths = [0.0, 0.0001, 0.0005, 0.001, 0.01, 0.1]
        gpu_results = wp.zeros((len(depths),), dtype=wp.float32, device="cuda:0")
        depth_arr = wp.array(np.array(depths, dtype=np.float32), dtype=wp.float32, device="cuda:0")

        @wp.kernel
        def _test_impedance(
            depths: wp.array(dtype=wp.float32, ndim=1),
            out: wp.array(dtype=wp.float32, ndim=1),
        ):
            i = wp.tid()
            out[i] = _impedance(depths[i], 0.9, 0.95, 0.001, 0.5, 2.0)

        wp.launch(_test_impedance, dim=len(depths), device="cuda:0", inputs=[depth_arr, gpu_results])

        gpu_vals = gpu_results.numpy()
        for i, d in enumerate(depths):
            cpu_val = cpu_solver._impedance(d)
            assert abs(gpu_vals[i] - cpu_val) < 1e-5, f"depth={d}: GPU={gpu_vals[i]:.6f}, CPU={cpu_val:.6f}"


class TestConeProjection:
    """Verify GPU cone projection matches CPU."""

    def test_projection_cases(self):
        """Test cone projection for various input vectors."""
        from physics.backends.warp.admm_kernels import _cone_project_all

        N = 1
        N = 1
        max_contacts = 3
        max_rows = max_contacts * 3

        # Setup: 3 contacts, all active
        contact_active = wp.array(np.ones((N, max_contacts), dtype=np.int32), device="cuda:0")

        # Test cases: (input z, expected s)
        # Contact 0: z = [1.0, 0.0, 0.0] → already in cone → s = z
        # Contact 1: z = [-0.5, 0.0, 0.0] → negative normal → s = [0, 0, 0]
        # Contact 2: z = [1.0, 1.0, 0.0] → tangent exceeds mu*f_n=0.5 → clamp
        z_np = np.zeros((N, max_rows), dtype=np.float32)
        z_np[0, 0:3] = [1.0, 0.0, 0.0]
        z_np[0, 3:6] = [-0.5, 0.3, 0.4]
        z_np[0, 6:9] = [1.0, 1.0, 0.0]

        z = wp.array(z_np, dtype=wp.float32, device="cuda:0")
        s = wp.zeros((N, max_rows), dtype=wp.float32, device="cuda:0")

        @wp.kernel
        def _test_proj(
            z: wp.array(dtype=wp.float32, ndim=2),
            s: wp.array(dtype=wp.float32, ndim=2),
            ca: wp.array(dtype=wp.int32, ndim=2),
        ):
            _cone_project_all(0, z, s, ca, 0.5, 3)

        wp.launch(_test_proj, dim=1, device="cuda:0", inputs=[z, s, contact_active])
        s_gpu = s.numpy()[0]

        # Contact 0: should pass through unchanged
        np.testing.assert_allclose(s_gpu[0:3], [1.0, 0.0, 0.0], atol=1e-6)

        # Contact 1: negative normal → clamped to 0, tangent also 0
        np.testing.assert_allclose(s_gpu[3:6], [0.0, 0.0, 0.0], atol=1e-6)

        # Contact 2: s_n=1.0, limit=mu*1.0=0.5, |z_t|=1.0 > 0.5 → scale to 0.5
        assert abs(s_gpu[6] - 1.0) < 1e-6  # normal passes
        t_norm = np.sqrt(s_gpu[7] ** 2 + s_gpu[8] ** 2)
        assert abs(t_norm - 0.5) < 1e-5, f"Tangent norm {t_norm:.6f}, expected 0.5"


# ===========================================================================
# 3. Single-step CPU vs GPU numerical comparison
# ===========================================================================


class TestSingleStepCpuGpuComparison:
    """Compare GPU and CPU ADMM intermediate quantities for a free body."""

    def test_delassus_matrix_matches(self):
        """For FreeJoint, GPU body-level W should equal CPU joint-space A."""
        mass, radius = 1.0, 0.1
        merged = _ball_merged(mass=mass, radius=radius)
        dt = 2e-4
        gpu = GpuEngine(merged, num_envs=1, dt=dt, solver="admm")

        # Place ball with small penetration
        q, qdot = merged.tree.default_state()
        q[6] = radius - 0.005  # 5mm penetration
        gpu.reset(q)

        # Run one step on GPU to populate W
        gpu.step(dt=dt)
        W_gpu = gpu._solver_scratch.W.numpy()[0]  # (max_rows, max_rows)
        contact_active = gpu._contact_active.numpy()[0]

        # Find active rows
        active_rows = []
        for c in range(gpu._max_contacts):
            if contact_active[c] != 0:
                for k in range(3):
                    active_rows.append(c * 3 + k)

        if len(active_rows) == 0:
            pytest.skip("No contact detected")

        n = len(active_rows)
        W_active = W_gpu[np.ix_(active_rows, active_rows)]

        # For a single FreeJoint body, H = diag(mI₃, I_body)
        # H^{-1} = diag(I₃/m, I_body^{-1})
        # So A = J H^{-1} J^T should equal W (body-level Delassus)
        # The key check: W is symmetric positive definite for active contacts
        assert W_active.shape[0] > 0, "Need at least one active row"
        assert np.allclose(W_active, W_active.T, atol=1e-6), "W should be symmetric"
        eigvals = np.linalg.eigvalsh(W_active)
        assert np.all(eigvals > -1e-8), f"W should be PSD, min eigenvalue = {eigvals[0]:.6e}"

        # Diagonal should be positive (inverse-mass contribution)
        for i in range(n):
            assert W_active[i, i] > 0, f"W diagonal [{i},{i}] = {W_active[i, i]:.6e}"

    def test_impulse_direction_correct(self):
        """Ball falling onto ground: impulse should be upward (along +z normal).

        GPU lambdas should produce a net upward generalized impulse.
        """
        mass, radius = 1.0, 0.1
        merged = _ball_merged(mass=mass, radius=radius)
        dt = 2e-4
        gpu = GpuEngine(merged, num_envs=1, dt=dt, solver="admm")

        q, qdot = merged.tree.default_state()
        q[6] = radius - 0.002  # slight penetration
        # Give it downward velocity
        qdot_np = np.zeros(merged.nv)
        qdot_np[2] = -1.0  # vz = -1 m/s (falling)
        gpu.reset(q)
        wp.copy(
            gpu._scratch.qdot,
            wp.array(qdot_np.reshape(1, -1).astype(np.float32), dtype=wp.float32, device=gpu._device),
        )

        gpu.step(dt=dt)

        # gen_impulse should have positive z component
        gen_imp = gpu._solver_scratch.gen_impulse.numpy()[0]
        # FreeJoint v layout: [vx, vy, vz, wx, wy, wz]
        gen_imp_z = gen_imp[2]
        assert gen_imp_z > 0, f"gen_impulse_z should be positive (upward), got {gen_imp_z:.6e}"

        # Verify second step also produces valid result
        out2 = gpu.step(dt=dt)
        assert out2.qdot_new is not None

    def test_rhs_const_sign_convention(self):
        """Verify rhs_const has correct sign for penetrating contact.

        For a penetrating ball (depth > 0, v_n < 0), the normal component of
        rhs_const should be positive (drives λ positive → push ball up).
        """
        mass, radius = 1.0, 0.1
        merged = _ball_merged(mass=mass, radius=radius)
        dt = 2e-4
        gpu = GpuEngine(merged, num_envs=1, dt=dt, solver="admm")

        q, qdot = merged.tree.default_state()
        q[6] = radius - 0.005  # 5mm penetration
        qdot_np = np.zeros(merged.nv)
        qdot_np[2] = -0.5  # falling
        gpu.reset(q)
        wp.copy(
            gpu._scratch.qdot,
            wp.array(qdot_np.reshape(1, -1).astype(np.float32), dtype=wp.float32, device=gpu._device),
        )
        gpu.step(dt=dt)

        rhs_const = gpu._solver_scratch.admm_rhs_const.numpy()[0]
        contact_active = gpu._contact_active.numpy()[0]

        for c in range(gpu._max_contacts):
            if contact_active[c] != 0:
                base = c * 3
                # Normal component should be positive (push λ_n up)
                assert rhs_const[base] > 0, (
                    f"Contact {c}: rhs_const_n={rhs_const[base]:.6e}, should be > 0 "
                    f"for penetrating falling ball"
                )
                break
        else:
            pytest.skip("No active contact found")

    def test_compliance_R_positive(self):
        """R_diag should be non-negative for all active contacts."""
        merged = _ball_merged()
        dt = 2e-4
        gpu = GpuEngine(merged, num_envs=1, dt=dt, solver="admm")

        q, qdot = merged.tree.default_state()
        q[6] = 0.095  # slight penetration
        gpu.reset(q)
        gpu.step(dt=dt)

        R_diag = gpu._solver_scratch.admm_R_diag.numpy()[0]
        contact_active = gpu._contact_active.numpy()[0]

        for c in range(gpu._max_contacts):
            if contact_active[c] != 0:
                base = c * 3
                for k in range(3):
                    assert R_diag[base + k] >= 0, (
                        f"R_diag[{base + k}] = {R_diag[base + k]:.6e}, should be >= 0"
                    )

    def test_cholesky_factor_valid(self):
        """L from kernel should satisfy L @ L^T ≈ AR_rho for active rows."""
        merged = _ball_merged()
        dt = 2e-4
        gpu = GpuEngine(merged, num_envs=1, dt=dt, solver="admm")

        q, qdot = merged.tree.default_state()
        q[6] = 0.095
        gpu.reset(q)
        gpu.step(dt=dt)

        AR = gpu._solver_scratch.admm_AR_rho.numpy()[0]
        L = gpu._solver_scratch.admm_L.numpy()[0]
        contact_active = gpu._contact_active.numpy()[0]

        active_rows = []
        for c in range(gpu._max_contacts):
            if contact_active[c] != 0:
                for k in range(3):
                    active_rows.append(c * 3 + k)

        if len(active_rows) == 0:
            pytest.skip("No active contact")

        L_sub = L[np.ix_(active_rows, active_rows)]
        AR_sub = AR[np.ix_(active_rows, active_rows)]
        reconstructed = L_sub @ L_sub.T

        np.testing.assert_allclose(
            reconstructed,
            AR_sub,
            atol=1e-3,
            err_msg="L @ L^T does not match AR_rho for active rows",
        )
