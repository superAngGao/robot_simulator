from __future__ import annotations

import numpy as np

from physics.cpu_engine import CpuEngine
from physics.geometry import BodyCollisionGeometry, ShapeInstance, SphereShape
from physics.joint import FreeJoint
from physics.merged_model import merge_models
from physics.publish import ConsumerState, HostSnapshotSpec
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from physics.terrain import FlatTerrain
from robot.model import RobotModel


def _single_body_model():
    tree = RobotTreeNumpy(gravity=9.81)
    tree.add_body(
        Body(
            name="body",
            index=0,
            joint=FreeJoint("root"),
            inertia=SpatialInertia(1.0, np.eye(3) * 0.001, np.zeros(3)),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()
    return RobotModel(
        tree=tree,
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(shape=SphereShape(0.05))])],
        contact_body_names=["body"],
    )


class TestCpuPublishRuntime:
    def test_step_and_publish_exposes_latest_frame(self):
        merged = merge_models({"r": _single_body_model()}, terrain=FlatTerrain())
        engine = CpuEngine(merged, dt=1e-3)
        q, qdot = merged.tree.default_state()
        tau = np.zeros(merged.nv)

        frame = engine.step_and_publish(q=q, qdot=qdot, tau=tau, dt=1e-3)

        assert engine.latest_published_frame() is frame
        assert frame.frame_id == 0
        assert frame.contact_count is not None
        assert frame.contact_mask is not None
        assert frame.contact_mask.shape == (1,)

    def test_plain_step_also_updates_latest_published_frame(self):
        merged = merge_models({"r": _single_body_model()}, terrain=FlatTerrain())
        engine = CpuEngine(merged, dt=1e-3)
        q, qdot = merged.tree.default_state()
        tau = np.zeros(merged.nv)

        engine.step(q=q, qdot=qdot, tau=tau, dt=1e-3)

        frame = engine.latest_published_frame()
        assert frame is not None
        assert frame.frame_id == 0

    def test_publish_every_n_steps_can_skip_frame_materialization(self):
        merged = merge_models({"r": _single_body_model()}, terrain=FlatTerrain())
        engine = CpuEngine(merged, dt=1e-3)
        engine.set_publish_policy(engine.publish_policy.__class__(publish_every_n_steps=2))
        q, qdot = merged.tree.default_state()
        tau = np.zeros(merged.nv)

        engine.step(q=q, qdot=qdot, tau=tau, dt=1e-3)
        frame0 = engine.latest_published_frame()
        assert frame0 is not None
        assert frame0.frame_id == 0

        engine.step(q=q, qdot=qdot, tau=tau, dt=1e-3)
        frame1 = engine.latest_published_frame()
        assert frame1 is frame0

        engine.step(q=q, qdot=qdot, tau=tau, dt=1e-3)
        frame2 = engine.latest_published_frame()
        assert frame2 is not None
        assert frame2 is not frame0
        assert frame2.frame_id == 2

    def test_lossless_borrow_acks_on_context_exit(self):
        merged = merge_models({"r": _single_body_model()}, terrain=FlatTerrain())
        engine = CpuEngine(merged, dt=1e-3)
        q, qdot = merged.tree.default_state()
        tau = np.zeros(merged.nv)
        frame = engine.step_and_publish(q=q, qdot=qdot, tau=tau, dt=1e-3)

        consumer = ConsumerState(
            consumer_id="hf",
            consumer_kind="recorder",
            qos_mode="lossless",
            access_mode="borrow",
        )
        engine.register_consumer(consumer)

        with engine.borrow_latest_frame("hf") as borrowed:
            assert borrowed.frame_id == frame.frame_id

        assert consumer.latest_seen_frame_id == frame.frame_id
        assert consumer.acked_frame_id == frame.frame_id

    def test_lossless_snapshot_acks_when_staged(self):
        merged = merge_models({"r": _single_body_model()}, terrain=FlatTerrain())
        engine = CpuEngine(merged, dt=1e-3)
        q, qdot = merged.tree.default_state()
        tau = np.zeros(merged.nv)
        frame = engine.step_and_publish(q=q, qdot=qdot, tau=tau, dt=1e-3)

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
            HostSnapshotSpec(fields=frozenset({"q", "qdot", "contact_count"})),
        )
        snapshot = handle.result()

        assert handle.is_ready is True
        assert handle.frame_id == frame.frame_id
        assert snapshot["frame_id"] == frame.frame_id
        assert snapshot["q"].shape == q.shape
        assert snapshot["qdot"].shape == qdot.shape
        assert consumer.latest_seen_frame_id == frame.frame_id
        assert consumer.acked_frame_id == frame.frame_id

    def test_snapshot_can_include_contact_mask(self):
        merged = merge_models({"r": _single_body_model()}, terrain=FlatTerrain())
        engine = CpuEngine(merged, dt=1e-3)
        q, qdot = merged.tree.default_state()
        tau = np.zeros(merged.nv)
        frame = engine.step_and_publish(q=q, qdot=qdot, tau=tau, dt=1e-3)

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
            HostSnapshotSpec(fields=frozenset({"contact_mask"})),
        )

        np.testing.assert_array_equal(handle.result()["contact_mask"], frame.contact_mask)
