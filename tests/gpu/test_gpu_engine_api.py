"""
Tests for GpuEngine API extensions (session 16):
  - State accessor properties (q_wp, qdot_wp, v_bodies_wp, x_world_*_wp, contact_*_wp)
  - Per-env reset (reset_envs)
  - Decimation (step_n)
  - StepOutput now populates X_world and v_bodies
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ball_model(mass: float = 1.0, radius: float = 0.1) -> RobotModel:
    """Single FreeJoint sphere."""
    tree = RobotTreeNumpy(gravity=9.81)
    I_sphere = 2.0 / 5.0 * mass * radius**2
    tree.add_body(
        Body(
            name="ball",
            index=0,
            joint=FreeJoint("root"),
            inertia=SpatialInertia(mass=mass, inertia=np.eye(3) * I_sphere, com=np.zeros(3)),
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


def _make_engine(num_envs=4, solver="jacobi_pgs_si"):
    merged = merge_models(robots={"ball": _ball_model()})
    engine = GpuEngine(merged, num_envs=num_envs, solver=solver)
    q0, _ = merged.tree.default_state()
    q0[6] = 0.5  # pz = 0.5 m
    engine.reset(q0=q0)
    return engine, merged


# ---------------------------------------------------------------------------
# State accessor properties
# ---------------------------------------------------------------------------


class TestStateAccessors:
    def test_num_envs(self):
        engine, _ = _make_engine(num_envs=8)
        assert engine.num_envs == 8

    def test_q_wp_shape(self):
        engine, merged = _make_engine()
        nq = merged.tree.nq
        q = engine.q_wp
        assert q.shape == (4, nq)
        assert q.dtype == wp.float32

    def test_qdot_wp_shape(self):
        engine, merged = _make_engine()
        nv = merged.tree.nv
        qdot = engine.qdot_wp
        assert qdot.shape == (4, nv)

    def test_v_bodies_wp_shape(self):
        engine, merged = _make_engine()
        nb = merged.tree.num_bodies
        v = engine.v_bodies_wp
        assert v.shape == (4, nb, 6)

    def test_x_world_R_shape(self):
        engine, merged = _make_engine()
        nb = merged.tree.num_bodies
        R = engine.x_world_R_wp
        assert R.shape == (4, nb, 3, 3)

    def test_x_world_r_shape(self):
        engine, merged = _make_engine()
        nb = merged.tree.num_bodies
        r = engine.x_world_r_wp
        assert r.shape == (4, nb, 3)

    def test_contact_active_shape(self):
        engine, _ = _make_engine()
        ca = engine.contact_active_wp
        # (N, max_contacts) int32
        assert ca.shape[0] == 4
        assert ca.dtype == wp.int32

    def test_contact_count_shape(self):
        engine, _ = _make_engine()
        cc = engine.contact_count_wp
        assert cc.shape == (4,)
        assert cc.dtype == wp.int32

    def test_q_wp_reflects_state(self):
        """q_wp should match the reset state."""
        engine, merged = _make_engine()
        q_np = engine.q_wp.numpy()
        # All envs should have pz = 0.5
        for i in range(4):
            assert abs(q_np[i, 6] - 0.5) < 1e-5

    def test_accessors_are_zero_copy(self):
        """Calling q_wp twice should return the same warp array object."""
        engine, _ = _make_engine()
        a = engine.q_wp
        b = engine.q_wp
        assert a.ptr == b.ptr


# ---------------------------------------------------------------------------
# StepOutput X_world / v_bodies
# ---------------------------------------------------------------------------


class TestStepOutputFields:
    def test_step_output_x_world_not_none(self):
        engine, _ = _make_engine(num_envs=1)
        out = engine.step()
        assert out.X_world is not None

    def test_step_output_v_bodies_not_none(self):
        engine, _ = _make_engine(num_envs=1)
        out = engine.step()
        assert out.v_bodies is not None

    def test_step_output_x_world_single_env(self):
        """For N=1, X_world should be (R, r) with R=(nb,3,3), r=(nb,3)."""
        engine, merged = _make_engine(num_envs=1)
        out = engine.step()
        R, r = out.X_world
        nb = merged.tree.num_bodies
        assert R.shape == (nb, 3, 3)
        assert r.shape == (nb, 3)

    def test_step_output_x_world_multi_env(self):
        """For N>1, X_world should be (R, r) with R=(N,nb,3,3), r=(N,nb,3)."""
        engine, merged = _make_engine(num_envs=4)
        out = engine.step()
        R, r = out.X_world
        nb = merged.tree.num_bodies
        assert R.shape == (4, nb, 3, 3)
        assert r.shape == (4, nb, 3)

    def test_step_output_v_bodies_single_env(self):
        engine, merged = _make_engine(num_envs=1)
        out = engine.step()
        nb = merged.tree.num_bodies
        assert out.v_bodies.shape == (nb, 6)

    def test_step_output_v_bodies_multi_env(self):
        engine, merged = _make_engine(num_envs=4)
        out = engine.step()
        nb = merged.tree.num_bodies
        assert out.v_bodies.shape == (4, nb, 6)

    def test_x_world_position_matches_q(self):
        """Body world position should match the FreeJoint translation in q."""
        engine, _ = _make_engine(num_envs=1)
        out = engine.step()
        _, r = out.X_world
        q = out.q_new
        # FreeJoint: q[4:7] = px, py, pz
        np.testing.assert_allclose(r[0, :3], q[4:7], atol=1e-4)


# ---------------------------------------------------------------------------
# step_n (decimation)
# ---------------------------------------------------------------------------


class TestStepN:
    def test_step_n_returns_output(self):
        engine, _ = _make_engine()
        out = engine.step_n(n_substeps=5)
        assert out.q_new is not None
        assert out.qdot_new is not None

    def test_step_n_equivalent_to_loop(self):
        """step_n(n=10) should give the same result as 10 calls to step()."""
        engine_a, _ = _make_engine(num_envs=1)
        engine_b, _ = _make_engine(num_envs=1)

        # Engine A: use step_n
        out_a = engine_a.step_n(n_substeps=10)

        # Engine B: loop
        for _ in range(10):
            out_b = engine_b.step()

        np.testing.assert_allclose(out_a.q_new, out_b.q_new, atol=1e-6)
        np.testing.assert_allclose(out_a.qdot_new, out_b.qdot_new, atol=1e-6)

    def test_step_n_free_fall_distance(self):
        """After n substeps of free fall, z should decrease."""
        engine, _ = _make_engine(num_envs=1)
        out = engine.step_n(n_substeps=100)
        # Started at z=0.5, should have fallen
        assert out.q_new[6] < 0.5

    def test_step_n_with_tau(self):
        """step_n should accept tau parameter."""
        engine, merged = _make_engine(num_envs=1)
        nv = merged.tree.nv
        tau = np.zeros((1, nv), dtype=np.float32)
        out = engine.step_n(tau=tau, n_substeps=5)
        assert out.q_new is not None


# ---------------------------------------------------------------------------
# Per-env reset (reset_envs)
# ---------------------------------------------------------------------------


class TestResetEnvs:
    def test_reset_empty_ids(self):
        """reset_envs with empty array should be a no-op."""
        engine, _ = _make_engine()
        engine.step()
        q_before = engine.q_wp.numpy().copy()
        engine.reset_envs(np.array([], dtype=np.int32))
        q_after = engine.q_wp.numpy()
        np.testing.assert_array_equal(q_before, q_after)

    def test_reset_single_env(self):
        """Reset env 2 while others continue."""
        engine, merged = _make_engine(num_envs=4)

        # Step all envs forward (free fall)
        for _ in range(50):
            engine.step()

        q_before = engine.q_wp.numpy().copy()
        # All envs should have fallen
        for i in range(4):
            assert q_before[i, 6] < 0.5

        # Reset only env 2
        engine.reset_envs(np.array([2], dtype=np.int32))
        q_after = engine.q_wp.numpy()

        # Env 2 should be back at z=0.5
        assert abs(q_after[2, 6] - 0.5) < 1e-5
        # Env 0,1,3 should be unchanged
        np.testing.assert_array_equal(q_after[0], q_before[0])
        np.testing.assert_array_equal(q_after[1], q_before[1])
        np.testing.assert_array_equal(q_after[3], q_before[3])

    def test_reset_multiple_envs(self):
        """Reset envs 0 and 3."""
        engine, _ = _make_engine(num_envs=4)

        for _ in range(50):
            engine.step()

        engine.reset_envs(np.array([0, 3], dtype=np.int32))
        q_after = engine.q_wp.numpy()

        # Envs 0,3 reset to z=0.5
        assert abs(q_after[0, 6] - 0.5) < 1e-5
        assert abs(q_after[3, 6] - 0.5) < 1e-5
        # Envs 1,2 still fallen
        assert q_after[1, 6] < 0.5
        assert q_after[2, 6] < 0.5

    def test_reset_envs_clears_velocity(self):
        """Reset envs should zero out qdot for reset envs."""
        engine, _ = _make_engine(num_envs=4)

        for _ in range(50):
            engine.step()

        qdot_before = engine.qdot_wp.numpy().copy()
        # After free fall, all envs should have nonzero vz
        for i in range(4):
            assert abs(qdot_before[i, 2]) > 0.01  # vz (FreeJoint: [vx,vy,vz,wx,wy,wz])

        engine.reset_envs(np.array([1], dtype=np.int32))
        qdot_after = engine.qdot_wp.numpy()

        # Env 1 qdot should be zero
        np.testing.assert_allclose(qdot_after[1], 0.0, atol=1e-6)
        # Env 0 unchanged
        np.testing.assert_allclose(qdot_after[0], qdot_before[0], atol=1e-6)

    def test_reset_envs_custom_q0(self):
        """reset_envs with custom q0."""
        engine, merged = _make_engine(num_envs=4)

        q_custom, _ = merged.tree.default_state()
        q_custom[6] = 1.0  # custom height

        engine.reset_envs(np.array([2], dtype=np.int32), q0=q_custom)
        q_after = engine.q_wp.numpy()

        # Env 2 at custom height
        assert abs(q_after[2, 6] - 1.0) < 1e-5
        # Env 0 still at default 0.5
        assert abs(q_after[0, 6] - 0.5) < 1e-5

    def test_reset_env_then_step(self):
        """After partial reset, stepping should work correctly."""
        engine, _ = _make_engine(num_envs=4)

        # Step, reset env 1, step again
        for _ in range(20):
            engine.step()

        engine.reset_envs(np.array([1], dtype=np.int32))

        # Step 20 more — should not crash, env 1 should start falling again
        for _ in range(20):
            engine.step()

        q = engine.q_wp.numpy()
        # Env 1 started from 0.5, fell for 20 steps — should be < 0.5
        assert q[1, 6] < 0.5
        # No NaN
        assert not np.any(np.isnan(q))


class TestResetEnvsADMM:
    """Per-env reset with ADMM solver (warmstart clearing)."""

    def test_reset_clears_admm_warmstart(self):
        engine, _ = _make_engine(num_envs=4, solver="admm")

        # Step to build up warmstart
        for _ in range(50):
            engine.step()

        # Reset env 2
        engine.reset_envs(np.array([2], dtype=np.int32))

        # Step again — should not crash (if warmstart was properly cleared)
        for _ in range(20):
            engine.step()

        q = engine.q_wp.numpy()
        assert not np.any(np.isnan(q))

    def test_reset_admm_env_then_step_gives_correct_physics(self):
        """After ADMM reset, env should behave like a fresh start."""
        engine_fresh, _ = _make_engine(num_envs=1, solver="admm")
        engine_reset, _ = _make_engine(num_envs=4, solver="admm")

        # Dirty the state
        for _ in range(50):
            engine_reset.step()

        # Reset env 0
        engine_reset.reset_envs(np.array([0], dtype=np.int32))

        # Step both for 50 steps
        for _ in range(50):
            engine_fresh.step()
            engine_reset.step()

        q_fresh = engine_fresh.q_wp.numpy()[0]
        q_reset = engine_reset.q_wp.numpy()[0]

        # Should be close (not identical due to warmstart effects on other envs,
        # but physics trajectory should match within reasonable tolerance)
        np.testing.assert_allclose(q_fresh[6], q_reset[6], atol=0.01)
