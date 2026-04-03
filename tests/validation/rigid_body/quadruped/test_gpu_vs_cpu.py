"""
GPU vs CPU validation for articulated quadruped.

Uses the CPU engine (validated against MuJoCo in test_dynamics_vs_mujoco.py
and test_contact_vs_mujoco.py) as the ground truth baseline.

This is the first test of the GPU engine with an articulated robot making
ground contact. The impulse-to-generalized kernel must propagate contact
forces through FreeJoint → hip → calf → foot (4 joints deep). The torque
double-counting bug (Q28, fixed in 61f5755) was in exactly this kernel.

Test tiers:
  1. Free fall (no contact): GPU ABA + integration vs CPU
  2. Ground contact: GPU collision + ADMM solver vs CPU
  3. Long-horizon stability: 10000 steps without NaN
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.merged_model import merge_models

from ..models import DT, build_quadruped

try:
    from physics.cpu_engine import CpuEngine
    from physics.gpu_engine import GpuEngine
    from physics.solvers.admm_qp import ADMMQPSolver

    HAS_ENGINES = True
except Exception:
    HAS_ENGINES = False

pytestmark = [
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.skipif(not HAS_ENGINES, reason="GPU/CPU engine not available"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_cpu(merged, q0, qdot0, n_steps, dt=DT):
    """Run CpuEngine, return trajectory arrays."""
    cpu = CpuEngine(merged, solver=ADMMQPSolver(), dt=dt)
    q, qdot = q0.copy(), qdot0.copy()
    nq, nv = merged.nq, merged.nv

    q_traj = np.zeros((n_steps, nq))
    qdot_traj = np.zeros((n_steps, nv))

    for i in range(n_steps):
        q_traj[i] = q
        qdot_traj[i] = qdot
        out = cpu.step(q, qdot, np.zeros(nv), dt=dt)
        q, qdot = out.q_new, out.qdot_new

    return q_traj, qdot_traj


def _run_gpu(merged, q0, qdot0, n_steps, dt=DT, solver="admm"):
    """Run GpuEngine, return trajectory arrays."""
    import warp as wp

    gpu = GpuEngine(merged, num_envs=1, dt=dt, solver=solver)
    gpu.reset(q0)
    # reset() zeros qdot — upload initial qdot explicitly
    if np.any(qdot0 != 0):
        qdot_np = qdot0.reshape(1, -1).astype(np.float32)
        wp.copy(
            gpu._scratch.qdot,
            wp.array(qdot_np, dtype=wp.float32, device=gpu._device),
        )
    nq, nv = merged.nq, merged.nv

    q_traj = np.zeros((n_steps, nq))
    qdot_traj = np.zeros((n_steps, nv))

    for i in range(n_steps):
        q_cur = gpu._scratch.q.numpy()[0]
        qdot_cur = gpu._scratch.qdot.numpy()[0]
        q_traj[i] = q_cur
        qdot_traj[i] = qdot_cur
        gpu.step(dt=dt)

    return q_traj, qdot_traj


# ---------------------------------------------------------------------------
# Tier 1: Free fall (no contact)
# ---------------------------------------------------------------------------


class TestQuadrupedFreeFallGpuVsCpu:
    """GPU vs CPU for quadruped in free fall — pure ABA validation."""

    def test_base_z_matches(self):
        """Base z trajectory: GPU should match CPU within float32 tolerance."""
        model, _ = build_quadruped(contact=False)
        merged = merge_models(robots={"quad": model})
        q, qdot = merged.tree.default_state()
        q[6] = 2.0  # high up, pure free fall

        q_cpu, _ = _run_cpu(merged, q, qdot, 5000)
        q_gpu, _ = _run_gpu(merged, q, qdot, 5000)

        np.testing.assert_allclose(
            q_gpu[:, 6],
            q_cpu[:, 6],
            atol=5e-3,
            err_msg="Free-fall base z: GPU vs CPU diverged",
        )

    def test_joint_angles_match(self):
        """Joint angles in free fall: GPU should match CPU."""
        model, _ = build_quadruped(contact=False)
        merged = merge_models(robots={"quad": model})
        q, qdot = merged.tree.default_state()
        q[6] = 2.0
        # Add some joint velocities for non-trivial dynamics
        rng = np.random.default_rng(42)
        qdot[6:] = rng.standard_normal(merged.nv - 6) * 0.5

        q_cpu, qdot_cpu = _run_cpu(merged, q, qdot, 5000)
        q_gpu, qdot_gpu = _run_gpu(merged, q, qdot, 5000)

        # Joint angles (index 7: onward)
        np.testing.assert_allclose(
            q_gpu[:, 7:],
            q_cpu[:, 7:],
            atol=0.01,
            err_msg="Free-fall joint angles: GPU vs CPU diverged",
        )

    def test_all_velocities_match(self):
        """Full velocity vector: GPU should match CPU."""
        model, _ = build_quadruped(contact=False)
        merged = merge_models(robots={"quad": model})
        q, qdot = merged.tree.default_state()
        q[6] = 2.0
        rng = np.random.default_rng(42)
        qdot[6:] = rng.standard_normal(merged.nv - 6) * 0.5

        _, qdot_cpu = _run_cpu(merged, q, qdot, 5000)
        _, qdot_gpu = _run_gpu(merged, q, qdot, 5000)

        np.testing.assert_allclose(
            qdot_gpu,
            qdot_cpu,
            atol=0.05,
            err_msg="Free-fall velocities: GPU vs CPU diverged",
        )


# ---------------------------------------------------------------------------
# Tier 2: Ground contact
# ---------------------------------------------------------------------------


class TestQuadrupedContactGpuVsCpu:
    """GPU vs CPU for quadruped landing on ground — full contact pipeline."""

    def test_settling_height_matches_cpu(self):
        """GPU settling height should match CPU within 2mm (Q29 fixed)."""
        model, _ = build_quadruped(contact=True)
        merged = merge_models(robots={"quad": model})
        q, qdot = merged.tree.default_state()
        q[6] = 0.45

        q_cpu, _ = _run_cpu(merged, q, qdot, 5000)
        q_gpu, _ = _run_gpu(merged, q, qdot, 5000)

        z_cpu_final = np.mean(q_cpu[-500:, 6])
        z_gpu_final = np.mean(q_gpu[-500:, 6])
        diff_mm = abs(z_gpu_final - z_cpu_final) * 1000

        assert diff_mm < 2.0, (
            f"Settling height: GPU={z_gpu_final:.4f} CPU={z_cpu_final:.4f} diff={diff_mm:.1f}mm"
        )

    def test_gpu_no_nan(self):
        """GPU must not produce NaN during quadruped landing."""
        model, _ = build_quadruped(contact=True)
        merged = merge_models(robots={"quad": model})
        q, qdot = merged.tree.default_state()
        q[6] = 0.45

        q_traj, qdot_traj = _run_gpu(merged, q, qdot, 5000)

        assert np.all(np.isfinite(q_traj)), "NaN in GPU q"
        assert np.all(np.isfinite(qdot_traj)), "NaN in GPU qdot"

    def test_gpu_base_above_ground(self):
        """GPU quadruped should not penetrate ground."""
        model, _ = build_quadruped(contact=True)
        merged = merge_models(robots={"quad": model})
        q, qdot = merged.tree.default_state()
        q[6] = 0.45

        q_traj, _ = _run_gpu(merged, q, qdot, 5000)

        assert np.all(q_traj[:, 6] > 0.1), f"GPU base penetrated ground: min z={np.min(q_traj[:, 6]):.4f}"

    def test_gpu_angular_velocity_bounded(self):
        """No angular velocity explosion (Q28 regression check)."""
        model, _ = build_quadruped(contact=True)
        merged = merge_models(robots={"quad": model})
        q, qdot = merged.tree.default_state()
        q[6] = 0.45

        _, qdot_traj = _run_gpu(merged, q, qdot, 5000)

        max_omega = np.max(np.abs(qdot_traj[:, 3:6]))
        assert max_omega < 50.0, f"Angular velocity diverged: max |ω| = {max_omega:.1f} rad/s"

    def test_joint_angles_bounded(self):
        """Joint angles should stay within physical limits."""
        model, _ = build_quadruped(contact=True)
        merged = merge_models(robots={"quad": model})
        q, qdot = merged.tree.default_state()
        q[6] = 0.45

        q_traj, _ = _run_gpu(merged, q, qdot, 5000)

        max_joint = np.max(np.abs(q_traj[:, 7:]))
        assert max_joint < 2.5, f"Joint angle exceeded limit: max={max_joint:.3f} rad"

    def test_landing_timing_matches(self):
        """First contact timing should be similar between GPU and CPU."""
        model, _ = build_quadruped(contact=True)
        merged = merge_models(robots={"quad": model})
        q, qdot = merged.tree.default_state()
        q[6] = 0.6  # higher drop for clear contact timing

        q_cpu, qdot_cpu = _run_cpu(merged, q, qdot, 5000)
        q_gpu, qdot_gpu = _run_gpu(merged, q, qdot, 5000)

        def _find_contact(vz_traj):
            for i in range(1, len(vz_traj)):
                if vz_traj[i - 1] < -0.1 and vz_traj[i] > vz_traj[i - 1] + 0.05:
                    return i
            return len(vz_traj)

        t_cpu = _find_contact(qdot_cpu[:, 2])
        t_gpu = _find_contact(qdot_gpu[:, 2])

        assert abs(t_cpu - t_gpu) < 200, f"Landing timing: CPU step {t_cpu} vs GPU step {t_gpu}"


# ---------------------------------------------------------------------------
# Tier 3: Long horizon + PGS solver
# ---------------------------------------------------------------------------


class TestQuadrupedLongHorizonGpu:
    """Long-horizon stability on GPU."""

    def test_10k_steps_admm_stable(self):
        """ADMM solver: 10000 steps without NaN."""
        model, _ = build_quadruped(contact=True)
        merged = merge_models(robots={"quad": model})
        q, qdot = merged.tree.default_state()
        q[6] = 0.6

        q_traj, qdot_traj = _run_gpu(merged, q, qdot, 10000, solver="admm")

        assert np.all(np.isfinite(q_traj)), "NaN in 10k-step ADMM trajectory"
        assert np.all(np.isfinite(qdot_traj)), "NaN velocity in 10k-step ADMM"
        # Should settle
        z_final = np.mean(q_traj[-1000:, 6])
        assert 0.38 < z_final < 0.46, f"Unexpected final z={z_final:.4f}"

    def test_10k_steps_pgs_stable(self):
        """Jacobi-PGS-SI solver: 10000 steps without NaN."""
        model, _ = build_quadruped(contact=True)
        merged = merge_models(robots={"quad": model})
        q, qdot = merged.tree.default_state()
        q[6] = 0.6

        q_traj, qdot_traj = _run_gpu(
            merged,
            q,
            qdot,
            10000,
            solver="jacobi_pgs_si",
        )

        assert np.all(np.isfinite(q_traj)), "NaN in 10k-step PGS trajectory"
        assert np.all(np.isfinite(qdot_traj)), "NaN velocity in 10k-step PGS"
        z_final = np.mean(q_traj[-1000:, 6])
        assert 0.1 < z_final < 0.6, f"PGS unexpected final z={z_final:.4f}"
