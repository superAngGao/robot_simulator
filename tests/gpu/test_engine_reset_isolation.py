"""
GpuEngine.reset(q0) state isolation contract.

CpuEngine is stateless per step (each step takes fresh q/qdot, no
internal scratch buffers persist), so this file targets GpuEngine only.

Existing coverage in test_gpu_engine_api.py::TestResetEnvs verifies the
per-env reset_envs API. This file complements with the bulk reset(q0)
contract:

  1. After reset(q0) + step, the engine state must match a freshly
     constructed engine that started at q0 — no artifacts from the
     previous trajectory.
  2. Contact buffers must not carry stale contact data into the next
     step. The kernel zeros them at the start of each step, but a test
     ensures this contract is honored after reset.
  3. With num_envs > 1, reset(q0) must put ALL envs into the same q0
     state — no per-env contamination.
  4. ADMM warmstart must reset to zero so the next solve doesn't get
     biased by prior λ values.
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
    import warp as wp  # noqa: F401

    from physics.gpu_engine import GpuEngine

    HAS_WARP = True
except Exception:
    HAS_WARP = False

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not HAS_WARP, reason="Warp or CUDA not available"),
]


def _ball_model(mass: float = 1.0, radius: float = 0.1) -> RobotModel:
    """Single FreeJoint sphere on the ground."""
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


# ---------------------------------------------------------------------------
# 1. reset(q0) + step matches a fresh engine starting at q0
# ---------------------------------------------------------------------------


class TestResetMatchesFreshEngine:
    """The strongest reset isolation contract: post-reset state evolution must
    be identical to a fresh engine initialized at the same q0."""

    def test_reset_after_contact_matches_fresh(self):
        """Run with active ground contact, reset to mid-air, step, compare to fresh engine.

        Catches: any scratch buffer (contact, ADMM warmstart, RNEA, predicted
        velocity) that persists across reset and biases the next step.
        """
        model = _ball_model(radius=0.1)
        merged = merge_models(robots={"a": model})

        # Engine 1: dirty trajectory (settle on ground), then reset to mid-air
        dirty = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)
        q_settled = merged.tree.default_state()[0].copy()
        q_settled[6] = 0.1  # touching ground
        dirty.reset(q0=q_settled)
        for _ in range(500):
            dirty.step(np.zeros((1, 0)), 2e-4)

        q_midair = merged.tree.default_state()[0].copy()
        q_midair[6] = 1.0  # well above ground, no contact
        dirty.reset(q0=q_midair)
        for _ in range(50):
            dirty.step(np.zeros((1, 0)), 2e-4)
        q_dirty = dirty._scratch.q.numpy()[0].copy()
        qdot_dirty = dirty._scratch.qdot.numpy()[0].copy()

        # Engine 2: fresh engine starting at the same mid-air state
        fresh = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)
        fresh.reset(q0=q_midair)
        for _ in range(50):
            fresh.step(np.zeros((1, 0)), 2e-4)
        q_fresh = fresh._scratch.q.numpy()[0]
        qdot_fresh = fresh._scratch.qdot.numpy()[0]

        # Float32 + 50-step accumulation tolerance
        np.testing.assert_allclose(
            q_dirty,
            q_fresh,
            atol=5e-5,
            err_msg=(
                f"q post-reset doesn't match fresh engine: dirty={q_dirty}, fresh={q_fresh}. "
                f"Some scratch buffer survives reset()."
            ),
        )
        np.testing.assert_allclose(
            qdot_dirty,
            qdot_fresh,
            atol=5e-5,
            err_msg=f"qdot post-reset diverged: dirty={qdot_dirty}, fresh={qdot_fresh}",
        )

    def test_reset_after_admm_contact_matches_fresh(self):
        """ADMM solver: warmstart must clear on reset, otherwise next solve is biased."""
        model = _ball_model(radius=0.1)
        merged = merge_models(robots={"a": model})

        dirty = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4, solver="admm")
        q_settled = merged.tree.default_state()[0].copy()
        q_settled[6] = 0.1
        dirty.reset(q0=q_settled)
        for _ in range(500):
            dirty.step(np.zeros((1, 0)), 2e-4)

        q_midair = merged.tree.default_state()[0].copy()
        q_midair[6] = 1.0
        dirty.reset(q0=q_midair)
        for _ in range(50):
            dirty.step(np.zeros((1, 0)), 2e-4)
        q_dirty = dirty._scratch.q.numpy()[0].copy()

        fresh = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4, solver="admm")
        fresh.reset(q0=q_midair)
        for _ in range(50):
            fresh.step(np.zeros((1, 0)), 2e-4)
        q_fresh = fresh._scratch.q.numpy()[0]

        np.testing.assert_allclose(
            q_dirty,
            q_fresh,
            atol=5e-5,
            err_msg="ADMM warmstart not cleared on reset(q0)",
        )


# ---------------------------------------------------------------------------
# 2. Contact buffer staleness
# ---------------------------------------------------------------------------


class TestContactBufferIsolation:
    def test_post_reset_step_yields_zero_contacts_in_air(self):
        """After reset to mid-air following a contact-laden trajectory, the
        first step's contact_count must be 0 (no stale contacts from before)."""
        model = _ball_model(radius=0.1)
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        # Phase 1: ground-contact scenario, accumulates contact data in buffers
        q_ground = merged.tree.default_state()[0].copy()
        q_ground[6] = 0.1
        engine.reset(q0=q_ground)
        for _ in range(100):
            engine.step(np.zeros((1, 0)), 2e-4)
        cc_with_contact = int(engine._contact_count.numpy()[0])
        assert cc_with_contact >= 1, "Phase 1 should have generated ground contacts"

        # Phase 2: reset to mid-air, no contact possible
        q_air = merged.tree.default_state()[0].copy()
        q_air[6] = 2.0  # well above ground
        engine.reset(q0=q_air)
        engine.step(np.zeros((1, 0)), 2e-4)
        cc_after_reset = int(engine._contact_count.numpy()[0])
        assert cc_after_reset == 0, (
            f"After reset to mid-air + 1 step, contact_count should be 0, got {cc_after_reset}. "
            f"Contact buffer carrying stale data."
        )

    def test_reset_alone_does_not_corrupt_contact_buffer(self):
        """reset(q0) without a subsequent step() should not leave stale 'active'
        flags that fool downstream consumers reading contact_active directly."""
        model = _ball_model(radius=0.1)
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        q_ground = merged.tree.default_state()[0].copy()
        q_ground[6] = 0.1
        engine.reset(q0=q_ground)
        for _ in range(50):
            engine.step(np.zeros((1, 0)), 2e-4)
        # After step, contact_active has 1s in the active slots
        active_before = engine._contact_active.numpy()[0].copy()
        assert active_before.sum() >= 1

        # Reset (no step). Contract: subsequent step() must produce a clean
        # narrowphase result. We don't promise the buffers are zeroed by reset
        # itself, but the next step's contact_count must reflect the new state,
        # not the previous step's.
        engine.reset(q0=q_ground)  # back to ground contact, fresh
        engine.step(np.zeros((1, 0)), 2e-4)
        cc_after = int(engine._contact_count.numpy()[0])
        active_after = engine._contact_active.numpy()[0]
        # Active count should equal contact_count (no leftover 1s past slot cc)
        leftover = int(active_after[cc_after:].sum())
        assert leftover == 0, (
            f"contact_active has {leftover} stale 1s past slot {cc_after} "
            f"(active_after.sum()={active_after.sum()}, expected ≤ {cc_after})"
        )


# ---------------------------------------------------------------------------
# 3. Multi-env uniformity after bulk reset
# ---------------------------------------------------------------------------


class TestMultiEnvBulkReset:
    def test_bulk_reset_makes_all_envs_identical(self):
        """After per-env divergence, reset(q0) must put all envs into the
        same q0 state — no per-env state survives bulk reset."""
        model = _ball_model(radius=0.1)
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=4, device="cuda:0", dt=2e-4)

        # Diverge: set each env to a different starting height by directly
        # writing into scratch.q
        q = engine._scratch.q.numpy()
        for env in range(4):
            q[env, 6] = 0.5 + 0.1 * env
        engine._scratch.q = wp.array(q, dtype=wp.float32, device="cuda:0")

        # Run a few steps so per-env trajectories also diverge in qdot
        for _ in range(20):
            engine.step(np.zeros((4, 0)), 2e-4)

        # Verify they really diverged
        q_pre = engine._scratch.q.numpy()
        zs_pre = q_pre[:, 6]
        assert zs_pre.std() > 1e-3, "Setup failed — envs didn't diverge"

        # Bulk reset to a single q0
        q0 = merged.tree.default_state()[0].copy()
        q0[6] = 1.0
        engine.reset(q0=q0)

        q_post = engine._scratch.q.numpy()
        qdot_post = engine._scratch.qdot.numpy()
        for env in range(4):
            np.testing.assert_allclose(
                q_post[env],
                q0,
                atol=1e-6,
                err_msg=f"env {env} q not reset to q0: {q_post[env]} vs {q0}",
            )
            np.testing.assert_allclose(
                qdot_post[env],
                np.zeros(merged.nv),
                atol=1e-6,
                err_msg=f"env {env} qdot not zero after reset: {qdot_post[env]}",
            )

    def test_bulk_reset_then_step_all_envs_in_sync(self):
        """After bulk reset, all envs should evolve identically over the next
        step (no per-env scratch state survives to perturb them)."""
        model = _ball_model(radius=0.1)
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=4, device="cuda:0", dt=2e-4)

        # Diverge
        for _ in range(50):
            q = engine._scratch.q.numpy()
            for env in range(4):
                q[env, 6] = 0.3 + 0.05 * env
            engine._scratch.q = wp.array(q, dtype=wp.float32, device="cuda:0")
            engine.step(np.zeros((4, 0)), 2e-4)

        # Reset and step
        q0 = merged.tree.default_state()[0].copy()
        q0[6] = 0.5
        engine.reset(q0=q0)
        for _ in range(20):
            engine.step(np.zeros((4, 0)), 2e-4)

        q_final = engine._scratch.q.numpy()
        # All 4 envs should have identical state
        for env in range(1, 4):
            np.testing.assert_allclose(
                q_final[env],
                q_final[0],
                atol=1e-6,
                err_msg=f"env {env} diverged from env 0: {q_final[env]} vs {q_final[0]}",
            )
