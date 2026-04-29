"""
Tests for GpuEngine API extensions (session 16):
  - State accessor properties (q_wp, qdot_wp, v_bodies_wp, x_world_*_wp, contact_*_wp)
  - Per-env reset (reset_envs)
  - Decimation (step_n)
  - StepOutput now populates X_world and v_bodies
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from physics.geometry import BodyCollisionGeometry, ShapeInstance, SphereShape
from physics.joint import FreeJoint
from physics.merged_model import merge_models
from physics.publish import ConsumerState, DeviceConsumerStalledError, HostSnapshotSpec, PublishPolicy
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


if HAS_WARP:

    @wp.kernel
    def _slow_q_checksum(
        q: wp.array2d(dtype=wp.float32),
        nq: int,
        spin_iters: int,
        out: wp.array(dtype=wp.float32),
    ):
        delay = float(0.0)
        for i in range(spin_iters):
            delay = delay + float(i % 2) * 1.0e-30

        checksum = delay
        for i in range(nq):
            checksum = checksum + q[0, i] * float(i + 1)
        out[0] = checksum


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

    def test_contact_mask_shape(self):
        engine, _ = _make_engine()
        mask = engine.contact_mask_wp
        assert mask.shape == (4, engine.nc_sensor)
        assert mask.dtype == wp.int32

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

    def test_lossless_snapshot_acks_after_staging(self):
        engine, _ = _make_engine(num_envs=1)
        engine.step()
        frame = engine.latest_published_frame()
        assert frame is not None
        consumer = ConsumerState(
            consumer_id="dataset",
            consumer_kind="host_export",
            qos_mode="lossless",
            access_mode="snapshot",
        )
        engine.register_consumer(consumer)

        handle = engine.snapshot_frame_to_host(
            "dataset",
            frame.frame_id,
            HostSnapshotSpec(fields=frozenset({"q", "qdot", "contact_mask"})),
        )
        snapshot = handle.result()

        assert handle.frame_id == frame.frame_id
        assert snapshot["q"].shape[0] == 1
        assert snapshot["contact_mask"].shape == (1, engine.nc_sensor)
        assert consumer.acked_frame_id == frame.frame_id

    def test_published_frame_ready_event_is_warp_event(self):
        engine, _ = _make_engine(num_envs=1)
        engine.step()
        frame = engine.latest_published_frame()

        assert frame is not None
        assert isinstance(frame.ready_event, wp.Event)

    def test_device_consumer_waits_and_records_done_event(self):
        engine, _ = _make_engine(num_envs=1)
        engine.step()
        frame = engine.latest_published_frame()
        assert frame is not None
        consumer = ConsumerState(
            consumer_id="camera",
            consumer_kind="render_backed_sensing",
            qos_mode="lossless",
            access_mode="borrow",
            consumer_location="device",
        )
        engine.register_consumer(consumer)
        stream = wp.Stream(device=engine._device)

        borrowed = engine.borrow_device_frame("camera", frame.frame_id, stream=stream)
        done_event = engine.complete_device_consumer("camera", frame.frame_id, stream=stream)

        assert borrowed.frame_id == frame.frame_id
        assert consumer.latest_seen_frame_id == frame.frame_id
        assert consumer.device_completed_frame_id == frame.frame_id
        assert consumer.device_done_event is done_event
        assert isinstance(done_event, wp.Event)

    def test_borrow_device_frame_defaults_to_latest_frame(self):
        engine, _ = _make_engine(num_envs=1)
        engine.step()
        frame = engine.latest_published_frame()
        assert frame is not None
        consumer = ConsumerState(
            consumer_id="camera",
            consumer_kind="render_backed_sensing",
            qos_mode="lossless",
            access_mode="borrow",
            consumer_location="device",
        )
        engine.register_consumer(consumer)
        stream = wp.Stream(device=engine._device)

        borrowed = engine.borrow_device_frame("camera", stream=stream)
        engine.complete_device_consumer("camera", borrowed.frame_id, stream=stream)

        assert borrowed.frame_id == frame.frame_id
        assert consumer.latest_seen_frame_id == frame.frame_id

    def test_borrow_device_frame_without_latest_frame_raises(self):
        engine, _ = _make_engine(num_envs=1)
        consumer = ConsumerState(
            consumer_id="camera",
            consumer_kind="render_backed_sensing",
            qos_mode="lossless",
            access_mode="borrow",
            consumer_location="device",
        )
        engine.register_consumer(consumer)

        with pytest.raises(KeyError, match="No published frame"):
            engine.borrow_device_frame("camera")

    def test_wait_for_device_consumers_before_reuse_waits_done_events(self):
        engine = GpuEngine.__new__(GpuEngine)
        done_event = object()
        waited = []
        engine._published_ring = SimpleNamespace(
            consumers=[
                ConsumerState(
                    consumer_id="camera",
                    consumer_kind="render_backed_sensing",
                    qos_mode="lossless",
                    access_mode="borrow",
                    consumer_location="device",
                    device_completed_frame_id=2,
                    device_done_event=done_event,
                ),
                ConsumerState(
                    consumer_id="imu",
                    consumer_kind="render_backed_sensing",
                    qos_mode="lossless",
                    access_mode="borrow",
                    consumer_location="device",
                    device_completed_frame_id=2,
                    device_done_event=None,
                ),
            ]
        )
        engine._wait_on_event = lambda event, **_: waited.append(event)

        engine._wait_for_device_consumers_before_reuse(SimpleNamespace(frame_id=2))

        assert waited == [done_event]

    def test_wait_for_device_consumers_before_reuse_rejects_incomplete_consumer(self):
        engine = GpuEngine.__new__(GpuEngine)
        engine._published_ring = SimpleNamespace(
            consumers=[
                ConsumerState(
                    consumer_id="camera",
                    consumer_kind="render_backed_sensing",
                    qos_mode="lossless",
                    access_mode="borrow",
                    consumer_location="device",
                    device_completed_frame_id=1,
                    device_done_event=object(),
                )
            ]
        )
        engine._wait_on_event = lambda event, **_: None

        with pytest.raises(RuntimeError, match="camera"):
            engine._wait_for_device_consumers_before_reuse(SimpleNamespace(frame_id=2))

    def test_device_lossless_consumer_must_complete_before_slot_reuse(self):
        engine, _ = _make_engine(num_envs=1)
        engine.set_publish_policy(PublishPolicy(on_ring_full="block"))
        consumer = ConsumerState(
            consumer_id="camera",
            consumer_kind="render_backed_sensing",
            qos_mode="lossless",
            access_mode="borrow",
            consumer_location="device",
        )
        engine.register_consumer(consumer)

        first = engine.step()
        frame = engine.latest_published_frame()
        assert first is not None
        assert frame is not None

        # Slots 1 and 2 are still free; slot 0 is reused on the fourth publish.
        engine.step()
        engine.step()
        with pytest.raises(NotImplementedError, match="device lossless consumers"):
            engine.step()

        stream = wp.Stream(device=engine._device)
        engine.borrow_device_frame("camera", frame.frame_id, stream=stream)
        engine.complete_device_consumer("camera", frame.frame_id, stream=stream)

        # Now slot 0 can be reused. The physics stream waits the recorded done event.
        engine.step()

    def test_device_lossless_consumer_max_lag_raises_stall_error(self):
        engine, _ = _make_engine(num_envs=1)
        engine.set_publish_policy(PublishPolicy(on_ring_full="block"))
        consumer = ConsumerState(
            consumer_id="camera",
            consumer_kind="render_backed_sensing",
            qos_mode="lossless",
            access_mode="borrow",
            consumer_location="device",
            max_lag_frames=3,
        )
        engine.register_consumer(consumer)

        engine.step()
        engine.step()
        engine.step()

        with pytest.raises(DeviceConsumerStalledError) as exc_info:
            engine.step()

        assert exc_info.value.consumer_id == "camera"
        assert exc_info.value.lag_frames == 4
        stats = engine.publish_stats()
        assert stats.stalled_consumer_ids == ("camera",)
        assert stats.stall_count == 1
        assert stats.blocking_consumer_ids == ("camera",)

    def test_device_consumer_reads_consistent_frame_across_slot_reuse(self):
        engine, merged = _make_engine(num_envs=1)
        engine.set_publish_policy(PublishPolicy(on_ring_full="block"))
        consumer = ConsumerState(
            consumer_id="camera",
            consumer_kind="render_backed_sensing",
            qos_mode="lossless",
            access_mode="borrow",
            consumer_location="device",
        )
        engine.register_consumer(consumer)

        first = engine.step()
        frame = engine.latest_published_frame()
        assert first is not None
        assert frame is not None
        expected = np.sum(first.q_new.astype(np.float32) * np.arange(1, merged.tree.nq + 1, dtype=np.float32))

        stream = wp.Stream(device=engine._device)
        checksum = wp.zeros(1, dtype=wp.float32, device=engine._device)
        borrowed = engine.borrow_device_frame("camera", frame.frame_id, stream=stream)
        wp.launch(
            _slow_q_checksum,
            dim=1,
            device=engine._device,
            inputs=[borrowed.q_wp, merged.tree.nq, 200_000],
            outputs=[checksum],
            stream=stream,
        )
        engine.complete_device_consumer("camera", frame.frame_id, stream=stream)

        # The fourth publish reuses slot 0. It must wait on the consumer's
        # done event before overwriting the slot the checksum kernel reads.
        engine.step()
        engine.step()
        engine.step()
        wp.synchronize()

        assert checksum.numpy()[0] == pytest.approx(expected, rel=1e-6, abs=1e-6)


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
