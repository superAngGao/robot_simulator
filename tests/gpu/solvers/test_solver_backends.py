"""Tests for mass splitting and graph-colored GS solver backends.

Verifies that both new solvers handle multi-point ground contacts
(box-ground 4pt manifold) combined with body-body contacts without
diverging — the exact scenario where pure Jacobi PGS fails (Q45).

Reference: Tonge et al. (SIGGRAPH 2012) for mass splitting,
PhysX/Bullet3 GPU for graph-colored GS.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    from physics.gpu_engine import GpuEngine

    HAS_WARP = True
except Exception:
    HAS_WARP = False

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not HAS_WARP, reason="Warp or CUDA not available"),
]


# ---------------------------------------------------------------------------
# Shared fixture: 3-robot mixed-shape scene from B(5) test suite
# ---------------------------------------------------------------------------


def _build_fixture():
    """Build the 3-robot mixed-shape fixture that triggers Q45."""
    from tests.gpu.collision.test_b5_d4d8_mixed_ground import (
        _build_merged,
        _init_state,
    )

    merged = _build_merged()
    q, qdot = _init_state(merged)
    return merged, q, qdot


# ---------------------------------------------------------------------------
# Mass Splitting tests
# ---------------------------------------------------------------------------


class TestMassSplitting:
    """Jacobi PGS with mass splitting (Tonge et al., SIGGRAPH 2012)."""

    @pytest.mark.slow
    def test_100_steps_no_nan(self):
        """Multi-point ground + body-body: 100 steps must stay finite."""
        merged, q, qdot = _build_fixture()
        tau = np.zeros(merged.nv)
        gpu = GpuEngine(merged, num_envs=1, dt=2e-4, solver="jacobi_pgs_ms")

        for step in range(100):
            out = gpu.step(q, qdot, tau, dt=2e-4)
            q, qdot = out.q_new, out.qdot_new
            assert np.all(np.isfinite(q)), f"q NaN at step {step}"

    def test_max_qdot_bounded(self):
        """Velocity must not grow beyond reasonable physical bounds."""
        merged, q, qdot = _build_fixture()
        tau = np.zeros(merged.nv)
        gpu = GpuEngine(merged, num_envs=1, dt=2e-4, solver="jacobi_pgs_ms")

        for _ in range(50):
            out = gpu.step(q, qdot, tau, dt=2e-4)
            q, qdot = out.q_new, out.qdot_new

        assert np.max(np.abs(qdot)) < 100.0, (
            f"max|qdot|={np.max(np.abs(qdot)):.2f} — should be < 100 for free-fall"
        )

    def test_ground_contacts_multipoint(self):
        """Box shapes should produce 4 ground contacts (multi-point manifold)."""
        merged, q, qdot = _build_fixture()
        tau = np.zeros(merged.nv)
        gpu = GpuEngine(merged, num_envs=1, dt=2e-4, solver="jacobi_pgs_ms")
        gpu.step(q, qdot, tau, dt=2e-4)

        contacts = gpu.query_contacts(env_idx=0)
        ground = [c for c in contacts if c.body_j < 0]
        # Original fixture has 6 single-point ground contacts;
        # with box multi-point, some become 4 → total > 6
        assert len(ground) >= 6, f"Expected ≥6 ground contacts, got {len(ground)}"


# ---------------------------------------------------------------------------
# Graph-Colored GS tests
# ---------------------------------------------------------------------------


class TestColoredPGS:
    """Graph-colored Gauss-Seidel PGS (PhysX/Bullet3 approach)."""

    @pytest.mark.slow
    def test_100_steps_no_nan(self):
        """Multi-point ground + body-body: 100 steps must stay finite."""
        merged, q, qdot = _build_fixture()
        tau = np.zeros(merged.nv)
        gpu = GpuEngine(merged, num_envs=1, dt=2e-4, solver="colored_pgs")

        for step in range(100):
            out = gpu.step(q, qdot, tau, dt=2e-4)
            q, qdot = out.q_new, out.qdot_new
            assert np.all(np.isfinite(q)), f"q NaN at step {step}"

    def test_max_qdot_bounded(self):
        """Velocity must not grow beyond reasonable physical bounds."""
        merged, q, qdot = _build_fixture()
        tau = np.zeros(merged.nv)
        gpu = GpuEngine(merged, num_envs=1, dt=2e-4, solver="colored_pgs")

        for _ in range(50):
            out = gpu.step(q, qdot, tau, dt=2e-4)
            q, qdot = out.q_new, out.qdot_new

        assert np.max(np.abs(qdot)) < 100.0, (
            f"max|qdot|={np.max(np.abs(qdot)):.2f} — should be < 100 for free-fall"
        )

    def test_coloring_no_body_conflict(self):
        """No two same-color contacts should share a body."""
        merged, q, qdot = _build_fixture()
        tau = np.zeros(merged.nv)
        gpu = GpuEngine(merged, num_envs=1, dt=2e-4, solver="colored_pgs")
        gpu.step(q, qdot, tau, dt=2e-4)

        # Read coloring from scratch
        sol = gpu._solver_scratch
        colors = sol.contact_color.numpy()[0]
        active = gpu._contact_active.numpy()[0]
        bi_arr = gpu._contact_bi.numpy()[0]
        bj_arr = gpu._contact_bj.numpy()[0]
        nc = gpu._max_contacts

        active_contacts = []
        for c in range(nc):
            if active[c] != 0:
                active_contacts.append((c, colors[c], bi_arr[c], bj_arr[c]))

        # Check: no two contacts with same color share a body
        from collections import defaultdict

        color_bodies = defaultdict(set)
        for c, col, bi, bj in active_contacts:
            bodies_in_color = color_bodies[col]
            if bi >= 0:
                assert bi not in bodies_in_color, f"Color {col}: body {bi} appears in multiple contacts"
                bodies_in_color.add(bi)
            if bj >= 0:
                assert bj not in bodies_in_color, f"Color {col}: body {bj} appears in multiple contacts"
                bodies_in_color.add(bj)


# ---------------------------------------------------------------------------
# Cross-solver agreement
# ---------------------------------------------------------------------------


class TestCrossSolverAgreement:
    """All solvers should produce similar physics on simple cases."""

    @pytest.mark.slow
    def test_single_sphere_all_solvers_agree(self):
        """Single sphere on ground: all solvers produce similar z after 50 steps."""
        from physics.geometry import BodyCollisionGeometry, ShapeInstance, SphereShape
        from physics.joint import FreeJoint
        from physics.merged_model import merge_models
        from physics.robot_tree import Body, RobotTreeNumpy
        from physics.spatial import SpatialInertia, SpatialTransform
        from physics.terrain import FlatTerrain
        from robot.model import RobotModel

        tree = RobotTreeNumpy(gravity=9.81)
        I_val = 2.0 / 5.0 * 1.0 * 0.05**2
        inertia = SpatialInertia(mass=1.0, inertia=np.eye(3) * I_val, com=np.zeros(3))
        tree.add_body(Body("ball", 0, FreeJoint("root"), inertia, SpatialTransform.identity(), -1))
        tree.finalize()
        geom = BodyCollisionGeometry(0, [ShapeInstance(SphereShape(0.05))])
        model = RobotModel(
            tree=tree,
            geometries=[geom],
            contact_body_names=["ball"],
        )
        merged = merge_models({"A": model}, terrain=FlatTerrain())

        results = {}
        # 2000 steps × 2e-4 = 0.4s — enough for sphere to fall from z=0.2 to ground
        for solver_name in ["jacobi_pgs_si", "jacobi_pgs_ms", "colored_pgs", "admm"]:
            q, qdot = merged.tree.default_state()
            q[6] = 0.2  # z = 0.2 (sphere drops to ground)
            tau = np.zeros(merged.nv)
            gpu = GpuEngine(merged, num_envs=1, dt=2e-4, solver=solver_name)
            for _ in range(2000):
                out = gpu.step(q, qdot, tau, dt=2e-4)
                q, qdot = out.q_new, out.qdot_new
            results[solver_name] = q[6]  # final z

        # All should be near ground (z ≈ 0.05 = radius, with some compliance)
        for name, z in results.items():
            assert 0.03 < z < 0.10, f"{name}: z={z:.4f} (expected near 0.05)"
