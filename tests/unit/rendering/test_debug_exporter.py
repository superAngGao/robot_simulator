from __future__ import annotations

import csv
import json
from unittest.mock import MagicMock

import numpy as np

from physics.cpu_engine import CpuEngine
from physics.geometry import BodyCollisionGeometry, ShapeInstance, SphereShape
from physics.joint import FreeJoint
from physics.merged_model import merge_models
from physics.publish import SnapshotHandle
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from physics.terrain import FlatTerrain
from rendering import DebugExporter
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


class TestDebugExporter:
    def test_cpu_exporter_writes_json_and_csv(self, tmp_path):
        merged = merge_models({"r": _single_body_model()}, terrain=FlatTerrain())
        engine = CpuEngine(merged, dt=1e-3)
        q, qdot = merged.tree.default_state()
        tau = np.zeros(merged.nv)
        engine.step(q=q, qdot=qdot, tau=tau, dt=1e-3)

        exporter = DebugExporter(engine)
        json_path = tmp_path / "frame.json"
        csv_path = tmp_path / "frames.csv"

        snapshot = exporter.write_latest_json(json_path, fields=("q", "qdot", "contact_count"))
        exporter.append_latest_csv(csv_path, fields=("frame_id", "sim_time", "contact_count"))

        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert snapshot["frame_id"] == 0
        assert loaded["frame_id"] == 0
        assert isinstance(loaded["q"], list)
        assert loaded["contact_count"] >= 0

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["frame_id"] == "0"

        exporter.close()

    def test_exporter_uses_engine_surface_contract_only(self, tmp_path):
        frame = MagicMock(frame_id=7)
        engine = MagicMock()
        engine.latest_published_frame.return_value = frame
        engine.snapshot_frame_to_host.return_value = SnapshotHandle(
            {"frame_id": 7, "sim_time": 0.1, "contact_count": 3},
            frame_id=7,
        )

        exporter = DebugExporter(engine, consumer_id="dbg")
        path = tmp_path / "frame.jsonl"
        snapshot = exporter.append_latest_jsonl(path, fields=("contact_count",))

        assert snapshot["frame_id"] == 7
        engine.register_consumer.assert_called_once()
        engine.snapshot_frame_to_host.assert_called_once()
        call = engine.snapshot_frame_to_host.call_args
        assert call.args[0] == "dbg"
        assert call.args[1] == 7
        assert call.args[2].fields == frozenset({"contact_count"})

        exporter.close()
        engine.unregister_consumer.assert_called_once_with("dbg")
