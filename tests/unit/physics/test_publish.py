import math

import pytest

from physics.publish import (
    AckPolicy,
    BorrowedFrameLease,
    ConsumerState,
    GpuPublishedFrame,
    LeaseExpiredError,
    PublishedRing,
    PublishedSlotMeta,
    PublishPlan,
    PublishPolicy,
    SlotReclaimedError,
    SlotReclaimer,
    SnapshotHandle,
    ViewPolicy,
)


class TestPublishPlan:
    def test_disabled_views_do_not_materialize(self):
        plan = PublishPlan.from_policy(frame_id=0, policy=PublishPolicy())

        assert plan.do_publish_core is True
        assert plan.do_realtime_render is False
        assert plan.do_render_backed_sensing is False
        assert plan.do_debug_export is False
        assert plan.do_rigid_block_write is False
        assert plan.do_telemetry_block_write is True

    def test_view_period_and_detail_flow_into_plan(self):
        policy = PublishPolicy(
            realtime_render=ViewPolicy(
                enabled=True, period_steps=2, detail_level="high", env_selector=(0, 3)
            ),
            render_backed_sensing=ViewPolicy(
                enabled=True, period_steps=3, detail_level="low", env_selector=(1,)
            ),
            publish_rigid_block=True,
        )

        frame0 = PublishPlan.from_policy(frame_id=0, policy=policy)
        frame1 = PublishPlan.from_policy(frame_id=1, policy=policy)
        frame3 = PublishPlan.from_policy(frame_id=3, policy=policy)

        assert frame0.do_realtime_render is True
        assert frame0.realtime_variant == "high"
        assert frame0.realtime_env_ids == (0, 3)
        assert frame0.do_render_backed_sensing is True
        assert frame0.render_backed_sensing_variant == "low"
        assert frame0.render_backed_sensing_env_ids == (1,)
        assert frame0.do_rigid_block_write is True

        assert frame1.do_realtime_render is False
        assert frame1.realtime_variant is None
        assert frame1.realtime_env_ids is None
        assert frame1.do_render_backed_sensing is False
        assert frame1.render_backed_sensing_variant is None
        assert frame1.render_backed_sensing_env_ids is None

        assert frame3.do_render_backed_sensing is True

    def test_invalid_period_is_rejected(self):
        with pytest.raises(ValueError, match="period_steps"):
            ViewPolicy(enabled=True, period_steps=0)

    def test_publish_every_n_steps_can_skip_core_publish(self):
        policy = PublishPolicy(publish_every_n_steps=3)

        frame0 = PublishPlan.from_policy(frame_id=0, policy=policy)
        frame1 = PublishPlan.from_policy(frame_id=1, policy=policy)
        frame3 = PublishPlan.from_policy(frame_id=3, policy=policy)

        assert frame0.do_publish_core is True
        assert frame1.do_publish_core is False
        assert frame3.do_publish_core is True

    def test_invalid_publish_every_n_steps_is_rejected(self):
        with pytest.raises(ValueError, match="publish_every_n_steps"):
            PublishPolicy(publish_every_n_steps=0)

    def test_on_ring_full_policy_is_preserved(self):
        policy = PublishPolicy(on_ring_full="skip")
        plan = PublishPlan.from_policy(frame_id=0, policy=policy)

        assert plan.do_publish_core is True
        assert policy.on_ring_full == "skip"


class TestBorrowedFrameLease:
    def test_context_manager_invalidates_lease(self):
        lease = BorrowedFrameLease({"frame_id": 7})

        with lease as frame:
            assert frame["frame_id"] == 7
            assert lease.get()["frame_id"] == 7

        assert lease.active is False
        with pytest.raises(LeaseExpiredError, match="no longer active"):
            lease.get()
        with pytest.raises(LeaseExpiredError, match="no longer active"):
            _ = frame["frame_id"]

    def test_context_manager_triggers_release_callback_once(self):
        released = []
        lease = BorrowedFrameLease(
            {"frame_id": 9}, on_release=lambda frame: released.append(frame["frame_id"])
        )

        with lease as frame:
            assert frame["frame_id"] == 9

        assert released == [9]


class TestSnapshotHandle:
    def test_result_returns_owned_payload(self):
        handle = SnapshotHandle({"frame_id": 5, "q": [1, 2, 3]}, frame_id=5)

        assert handle.staged is True
        assert handle.is_ready is True
        assert handle.frame_id == 5
        assert handle.result()["frame_id"] == 5


class TestGpuPublishedFrame:
    def test_access_raises_after_slot_reclaim(self):
        meta = PublishedSlotMeta(slot_id=1, frame_id=11, state="ready")
        frame = GpuPublishedFrame(
            slot_id=1,
            frame_id=11,
            sim_time=0.01,
            step_index=11,
            env_mask_wp=None,
            q_wp=object(),
            qdot_wp=object(),
            x_world_R_wp=object(),
            x_world_r_wp=object(),
            v_bodies_wp=object(),
            contact_count_wp=None,
            contact_cache_ref=None,
            telemetry_ref=None,
            slot_meta=meta,
        )

        assert frame.q_wp is not None
        meta.invalidated = True

        with pytest.raises(SlotReclaimedError, match="reclaimed"):
            _ = frame.q_wp

    def test_access_raises_after_slot_reuse_even_when_not_invalidated(self):
        meta = PublishedSlotMeta(slot_id=1, frame_id=11, state="ready", invalidated=False)
        frame = GpuPublishedFrame(
            slot_id=1,
            frame_id=11,
            sim_time=0.01,
            step_index=11,
            env_mask_wp=None,
            q_wp=object(),
            qdot_wp=object(),
            x_world_R_wp=object(),
            x_world_r_wp=object(),
            v_bodies_wp=object(),
            contact_count_wp=None,
            contact_cache_ref=None,
            telemetry_ref=None,
            slot_meta=meta,
        )

        meta.frame_id = 14
        meta.invalidated = False

        with pytest.raises(SlotReclaimedError, match="reclaimed"):
            _ = frame.q_wp


class TestAckPolicy:
    def test_best_effort_defaults_to_no_ack(self):
        consumer = ConsumerState(
            consumer_id="rt",
            consumer_kind="realtime_render",
            qos_mode="best_effort",
            access_mode="borrow",
        )

        policy = AckPolicy.default_for(consumer)

        assert policy.ack_point == "none"

    def test_lossless_snapshot_defaults_to_snapshot_staged(self):
        consumer = ConsumerState(
            consumer_id="recorder",
            consumer_kind="host_export",
            qos_mode="lossless",
            access_mode="snapshot",
        )

        policy = AckPolicy.default_for(consumer)

        assert policy.ack_point == "on_snapshot_staged"


class TestSlotReclaimer:
    def test_no_lossless_consumers_means_slots_are_reclaimable(self):
        consumers = [
            ConsumerState(
                consumer_id="rt",
                consumer_kind="realtime_render",
                qos_mode="best_effort",
                access_mode="borrow",
            )
        ]
        reclaimer = SlotReclaimer(consumers)
        slot = PublishedSlotMeta(slot_id=0, frame_id=42, state="ready")

        assert math.isinf(reclaimer.min_lossless_acked_frame_id())
        assert reclaimer.reclaimable(slot) is True


class TestPublishedRing:
    def test_acquire_marks_target_slot_writing(self):
        ring = PublishedRing(slot_buffers=[{"slot": 0}, {"slot": 1}])

        acquired = ring.acquire(frame_id=1)

        assert acquired is not None
        slot_id, slot, meta = acquired
        assert slot_id == 1
        assert slot == {"slot": 1}
        assert meta.state == "writing"
        assert meta.invalidated is True

    def test_acquire_returns_none_when_pinned_and_policy_is_skip(self):
        consumer = ConsumerState(
            consumer_id="dataset",
            consumer_kind="host_export",
            qos_mode="lossless",
            access_mode="snapshot",
            acked_frame_id=-1,
        )
        ring = PublishedRing(
            slot_buffers=[{"slot": 0}],
            consumers=[consumer],
            policy=PublishPolicy(on_ring_full="skip"),
        )
        ring.mark_ready(slot_id=0, frame_id=0, step_index=0, sim_time=0.0, publish_event=object())

        assert ring.acquire(frame_id=1) is None

    def test_acquire_raises_when_pinned_and_policy_is_raise(self):
        consumer = ConsumerState(
            consumer_id="dataset",
            consumer_kind="host_export",
            qos_mode="lossless",
            access_mode="snapshot",
            acked_frame_id=-1,
        )
        ring = PublishedRing(
            slot_buffers=[{"slot": 0}],
            consumers=[consumer],
            policy=PublishPolicy(on_ring_full="raise"),
        )
        ring.mark_ready(slot_id=0, frame_id=0, step_index=0, sim_time=0.0, publish_event=object())

        with pytest.raises(RuntimeError, match="dataset"):
            ring.acquire(frame_id=1)

    def test_mark_ready_latest_and_find_frame(self):
        ring = PublishedRing(slot_buffers=[{"slot": 0}, {"slot": 1}])
        meta = ring.mark_ready(slot_id=0, frame_id=2, step_index=2, sim_time=0.2, publish_event="event")
        frame = object()

        ring.set_latest_frame(frame)
        found = ring.find_frame(2)

        assert meta.state == "ready"
        assert meta.invalidated is False
        assert ring.latest_frame is frame
        assert found == (meta, {"slot": 0})

    def test_reset_clears_meta_and_latest_frame(self):
        ring = PublishedRing(slot_buffers=[{"slot": 0}])
        ring.mark_ready(slot_id=0, frame_id=3, step_index=3, sim_time=0.3, publish_event=object())
        ring.set_latest_frame(object())

        ring.reset()

        assert ring.latest_frame is None
        assert ring.slot_meta[0].state == "free"
        assert ring.slot_meta[0].frame_id == -1

    def test_reclaimer_uses_slowest_lossless_consumer(self):
        consumers = [
            ConsumerState(
                consumer_id="rt",
                consumer_kind="realtime_render",
                qos_mode="best_effort",
                access_mode="borrow",
                latest_seen_frame_id=20,
            ),
            ConsumerState(
                consumer_id="hf_render",
                consumer_kind="recorder",
                qos_mode="lossless",
                access_mode="snapshot",
                acked_frame_id=10,
            ),
            ConsumerState(
                consumer_id="dataset",
                consumer_kind="host_export",
                qos_mode="lossless",
                access_mode="snapshot",
                acked_frame_id=7,
            ),
        ]
        reclaimer = SlotReclaimer(consumers)
        slot = PublishedSlotMeta(slot_id=1, frame_id=9, state="ready")

        assert reclaimer.min_lossless_acked_frame_id() == 7
        assert reclaimer.reclaimable(slot) is False

        stats = reclaimer.ring_pressure_stats(slot)
        assert stats.enabled_lossless_consumers == ("hf_render", "dataset")
        assert stats.blocking_consumer_ids == ("dataset",)

    def test_reclaimer_allows_slot_once_lossless_ack_catches_up(self):
        consumers = [
            ConsumerState(
                consumer_id="dataset",
                consumer_kind="host_export",
                qos_mode="lossless",
                access_mode="snapshot",
                acked_frame_id=12,
            )
        ]
        reclaimer = SlotReclaimer(consumers)
        slot = PublishedSlotMeta(slot_id=2, frame_id=12, state="ready")

        assert reclaimer.reclaimable(slot) is True
