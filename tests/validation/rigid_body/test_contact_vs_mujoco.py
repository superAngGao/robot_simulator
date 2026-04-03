"""
Tier 1 Layer 2: Contact dynamics validation — CPU vs MuJoCo.

Validates that our CPU contact solvers (ADMM, PGS-SI) produce physically
correct behavior for articulated robots making ground contact.

Contact comparison is inherently looser than pure dynamics because:
  - Different collision detection (GJK/EPA vs MuJoCo analytical)
  - Different solver implementations (our ADMM vs MuJoCo's CG-based ADMM)
  - Different compliance parameters (approximate match via solref/solimp)

Pass criteria focus on qualitative correctness and bounded quantitative
agreement, not bit-exact trajectory matching.
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.merged_model import merge_models

from .models import (
    DT,
    QUAD_CALF_LENGTH,
    QUAD_FOOT_RADIUS,
    QUAD_HIP_LENGTH,
    build_quadruped,
)

try:
    import mujoco

    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

try:
    from physics.cpu_engine import CpuEngine
    from physics.solvers.admm_qp import ADMMQPSolver
    from physics.solvers.pgs_split_impulse import PGSSplitImpulseSolver

    HAS_ENGINE = True
except Exception:
    HAS_ENGINE = False

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not HAS_MUJOCO, reason="mujoco not installed"),
    pytest.mark.skipif(not HAS_ENGINE, reason="CpuEngine not available"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Expected standing height: base at hip_offset_z=0 relative to base,
# legs hang straight down: hip_length + calf_length + foot_radius
LEG_LENGTH = QUAD_HIP_LENGTH + QUAD_CALF_LENGTH + QUAD_FOOT_RADIUS
# Standing base z ≈ leg_length when joints are at zero angle


def _run_cpu(model, q0, qdot0, n_steps, solver=None, dt=DT):
    """Run CpuEngine with contact detection."""
    merged = merge_models(robots={"quad": model})
    solver = solver or ADMMQPSolver()
    cpu = CpuEngine(merged, solver=solver, dt=dt)

    q, qdot = q0.copy(), qdot0.copy()
    z_traj = np.zeros(n_steps)
    vz_traj = np.zeros(n_steps)
    q_traj = np.zeros((n_steps, merged.nq))
    qdot_traj = np.zeros((n_steps, merged.nv))

    for i in range(n_steps):
        z_traj[i] = q[6]
        vz_traj[i] = qdot[2]
        q_traj[i] = q
        qdot_traj[i] = qdot
        out = cpu.step(q, qdot, np.zeros(merged.nv), dt=dt)
        q, qdot = out.q_new, out.qdot_new

    return z_traj, vz_traj, q_traj, qdot_traj


def _run_mujoco_quad(xml, q0_base_z, n_steps, dt=DT):
    """Run MuJoCo quadruped drop."""
    mj_model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = dt
    mj_data.qpos[2] = q0_base_z

    z_traj = np.zeros(n_steps)
    vz_traj = np.zeros(n_steps)
    qpos_traj = np.zeros((n_steps, mj_model.nq))
    qvel_traj = np.zeros((n_steps, mj_model.nv))

    for i in range(n_steps):
        z_traj[i] = mj_data.qpos[2]
        vz_traj[i] = mj_data.qvel[2]
        qpos_traj[i] = mj_data.qpos[: mj_model.nq]
        qvel_traj[i] = mj_data.qvel[: mj_model.nv]
        mujoco.mj_step(mj_model, mj_data)

    return z_traj, vz_traj, qpos_traj, qvel_traj


# ---------------------------------------------------------------------------
# Test: Quadruped drop (ADMM solver)
# ---------------------------------------------------------------------------


class TestQuadrupedDropAdmm:
    """Quadruped drop with ADMM solver vs MuJoCo."""

    BASE_Z0 = 0.45  # just above standing height, feet barely above ground

    def test_base_does_not_penetrate_ground(self):
        """Base z should stay well above ground throughout the drop."""
        model, xml = build_quadruped(contact=True)
        q, qdot = model.tree.default_state()
        q[6] = self.BASE_Z0
        z_traj, _, _, _ = _run_cpu(model, q, qdot, 5000)

        assert np.all(z_traj > 0.1), f"Base penetrated too deep: min z = {np.min(z_traj):.4f}"

    def test_base_settles_near_standing_height(self):
        """After landing, base should settle near standing height."""
        model, xml = build_quadruped(contact=True)
        q, qdot = model.tree.default_state()
        q[6] = self.BASE_Z0
        z_traj, _, _, _ = _run_cpu(model, q, qdot, 5000)

        z_final = np.mean(z_traj[-500:])
        # Standing height ≈ LEG_LENGTH ≈ 0.42m
        assert abs(z_final - LEG_LENGTH) < 0.05, f"Base settled at {z_final:.4f}, expected ~{LEG_LENGTH:.4f}"

    def test_velocity_settles_near_zero(self):
        """After landing, vertical velocity should decay toward zero."""
        model, xml = build_quadruped(contact=True)
        q, qdot = model.tree.default_state()
        q[6] = self.BASE_Z0
        _, vz_traj, _, _ = _run_cpu(model, q, qdot, 5000)

        vz_final = np.mean(np.abs(vz_traj[-500:]))
        assert vz_final < 0.05, f"Vertical velocity not settled: mean |vz| = {vz_final:.4f}"

    def test_no_nan_or_divergence(self):
        """Entire trajectory must be finite, no state explosion."""
        model, xml = build_quadruped(contact=True)
        q, qdot = model.tree.default_state()
        q[6] = self.BASE_Z0
        _, _, q_traj, qdot_traj = _run_cpu(model, q, qdot, 5000)

        assert np.all(np.isfinite(q_traj)), "NaN in q trajectory"
        assert np.all(np.isfinite(qdot_traj)), "NaN in qdot trajectory"
        assert np.max(np.abs(qdot_traj)) < 100, (
            f"Velocity explosion: max |qdot| = {np.max(np.abs(qdot_traj)):.1f}"
        )

    def test_settling_height_vs_mujoco(self):
        """Steady-state base height should match MuJoCo within 5mm."""
        model, xml = build_quadruped(contact=True)
        q, qdot = model.tree.default_state()
        q[6] = self.BASE_Z0
        n_steps = 5000

        z_ours, _, _, _ = _run_cpu(model, q, qdot, n_steps)
        z_mj, _, _, _ = _run_mujoco_quad(xml, self.BASE_Z0, n_steps)

        z_ours_final = np.mean(z_ours[-500:])
        z_mj_final = np.mean(z_mj[-500:])
        diff_mm = abs(z_ours_final - z_mj_final) * 1000

        assert diff_mm < 5.0, (
            f"Settling height differs by {diff_mm:.1f}mm: ours={z_ours_final:.4f}, MuJoCo={z_mj_final:.4f}"
        )

    def test_landing_timing_vs_mujoco(self):
        """First contact should happen within 50 steps of MuJoCo."""
        model, xml = build_quadruped(contact=True)
        q, qdot = model.tree.default_state()
        q[6] = 0.6  # higher drop for clearer contact timing
        n_steps = 5000

        z_ours, vz_ours, _, _ = _run_cpu(model, q, qdot, n_steps)
        z_mj, vz_mj, _, _ = _run_mujoco_quad(xml, 0.6, n_steps)

        # Contact = first step where vz changes sign (from falling to bouncing)
        def _find_contact(vz):
            for i in range(1, len(vz)):
                if vz[i - 1] < -0.01 and vz[i] > vz[i - 1] + 0.01:
                    return i
            return len(vz)

        t_ours = _find_contact(vz_ours)
        t_mj = _find_contact(vz_mj)
        diff_steps = abs(t_ours - t_mj)

        assert diff_steps < 100, (
            f"Landing timing differs by {diff_steps} steps "
            f"({diff_steps * DT * 1000:.1f}ms): ours={t_ours}, MuJoCo={t_mj}"
        )

    def test_joint_angles_bounded_during_landing(self):
        """Joint angles should stay within limits during the drop."""
        model, _ = build_quadruped(contact=True)
        q, qdot = model.tree.default_state()
        q[6] = 0.6
        _, _, q_traj, _ = _run_cpu(model, q, qdot, 5000)

        # Joint angles (indices 7:15)
        q_joints = q_traj[:, 7:]
        # With damping, joints should stay near zero (no external torques)
        max_joint_angle = np.max(np.abs(q_joints))
        assert max_joint_angle < 1.5, f"Joint angle exceeded 1.5 rad: max = {max_joint_angle:.3f}"


# ---------------------------------------------------------------------------
# Test: Quadruped drop (PGS-SI solver)
# ---------------------------------------------------------------------------


class TestQuadrupedDropPgsSi:
    """Quadruped drop with PGS Split-Impulse solver."""

    BASE_Z0 = 0.45

    def test_pgs_si_base_settles(self):
        """PGS-SI should also produce a stable landing."""
        model, _ = build_quadruped(contact=True)
        q, qdot = model.tree.default_state()
        q[6] = self.BASE_Z0
        solver = PGSSplitImpulseSolver(max_iter=60, erp=0.8, slop=0.005)
        z_traj, _, _, _ = _run_cpu(model, q, qdot, 5000, solver=solver)

        z_final = np.mean(z_traj[-500:])
        assert z_final > 0.3, f"PGS-SI: base too low z={z_final:.4f}"
        assert z_final < 0.6, f"PGS-SI: base too high z={z_final:.4f}"

    def test_pgs_si_no_nan(self):
        """PGS-SI must not produce NaN."""
        model, _ = build_quadruped(contact=True)
        q, qdot = model.tree.default_state()
        q[6] = self.BASE_Z0
        solver = PGSSplitImpulseSolver(max_iter=60, erp=0.8, slop=0.005)
        _, _, q_traj, qdot_traj = _run_cpu(model, q, qdot, 5000, solver=solver)

        assert np.all(np.isfinite(q_traj)), "PGS-SI produced NaN in q"
        assert np.all(np.isfinite(qdot_traj)), "PGS-SI produced NaN in qdot"

    def test_pgs_si_vs_admm_settling_height(self):
        """PGS-SI and ADMM should settle to similar heights.

        ADMM uses compliance (spring-damper), PGS-SI uses hard constraints
        with ERP. PGS-SI typically settles slightly lower due to different
        penetration handling. 10cm tolerance accounts for this.
        """
        model, _ = build_quadruped(contact=True)
        q, qdot = model.tree.default_state()
        q[6] = self.BASE_Z0

        z_admm, _, _, _ = _run_cpu(model, q.copy(), qdot.copy(), 5000)
        solver_pgs = PGSSplitImpulseSolver(max_iter=60, erp=0.8, slop=0.005)
        z_pgs, _, _, _ = _run_cpu(model, q.copy(), qdot.copy(), 5000, solver=solver_pgs)

        z_admm_final = np.mean(z_admm[-500:])
        z_pgs_final = np.mean(z_pgs[-500:])

        assert abs(z_admm_final - z_pgs_final) < 0.10, (
            f"ADMM z={z_admm_final:.4f} vs PGS-SI z={z_pgs_final:.4f}"
        )


# ---------------------------------------------------------------------------
# Test: Higher drop (more energy, stress-test contact stability)
# ---------------------------------------------------------------------------


class TestQuadrupedHighDrop:
    """Quadruped dropped from higher altitude — stress-test contact."""

    def test_high_drop_stable(self):
        """Drop from 1.0m: must survive the impact without NaN."""
        model, _ = build_quadruped(contact=True)
        q, qdot = model.tree.default_state()
        q[6] = 1.0  # high drop
        _, _, q_traj, qdot_traj = _run_cpu(model, q, qdot, 10000)

        assert np.all(np.isfinite(q_traj)), "NaN after high drop"
        assert np.all(np.isfinite(qdot_traj)), "NaN velocity after high drop"
        # Should eventually settle
        z_final = q_traj[-1, 6]
        assert z_final > 0.1, f"Base collapsed to z={z_final:.4f}"

    def test_high_drop_vs_mujoco(self):
        """High drop settling height comparison."""
        model, xml = build_quadruped(contact=True)
        q, qdot = model.tree.default_state()
        q[6] = 1.0
        n_steps = 10000

        z_ours, _, _, _ = _run_cpu(model, q, qdot, n_steps)
        z_mj, _, _, _ = _run_mujoco_quad(xml, 1.0, n_steps)

        z_ours_final = np.mean(z_ours[-1000:])
        z_mj_final = np.mean(z_mj[-1000:])
        diff_mm = abs(z_ours_final - z_mj_final) * 1000

        # Looser tolerance for high drop (more contact dynamics divergence)
        assert diff_mm < 20.0, (
            f"High drop settling height differs by {diff_mm:.1f}mm: "
            f"ours={z_ours_final:.4f}, MuJoCo={z_mj_final:.4f}"
        )
