"""Debug export consumer for published physics frames.

Consumes the engine-level published-frame API and produces host-owned debug
snapshots that can be serialized to JSON / JSONL / CSV. This is intentionally
lightweight: phase-1 focuses on validating the consumer contract, not on
building a full recording pipeline.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from physics.engine import PhysicsEngine
from physics.publish import ConsumerState, HostSnapshotSpec


def _to_jsonable(value):
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


class DebugExporter:
    """Minimal host-side debug/export consumer for published frames."""

    def __init__(
        self,
        engine: PhysicsEngine,
        consumer_id: str = "debug_export",
        qos_mode: str = "best_effort",
    ) -> None:
        self._engine = engine
        self._consumer = ConsumerState(
            consumer_id=consumer_id,
            consumer_kind="host_export",
            qos_mode=qos_mode,
            access_mode="snapshot",
        )
        self._engine.register_consumer(self._consumer)

    @property
    def consumer(self) -> ConsumerState:
        return self._consumer

    def close(self) -> None:
        self._engine.unregister_consumer(self._consumer.consumer_id)

    def capture_latest(self, fields: Iterable[str]) -> dict[str, object]:
        frame = self._engine.latest_published_frame()
        if frame is None:
            raise RuntimeError("No published frame is available yet.")
        handle = self._engine.snapshot_frame_to_host(
            self._consumer.consumer_id,
            frame.frame_id,
            HostSnapshotSpec(fields=frozenset(fields)),
        )
        return handle.result()

    def capture_latest_jsonable(self, fields: Iterable[str]) -> dict[str, object]:
        return _to_jsonable(self.capture_latest(fields))

    def write_latest_json(
        self, path: str | Path, fields: Iterable[str], indent: int = 2
    ) -> dict[str, object]:
        snapshot = self.capture_latest_jsonable(fields)
        out_path = Path(path)
        out_path.write_text(json.dumps(snapshot, indent=indent, sort_keys=True), encoding="utf-8")
        return snapshot

    def append_latest_jsonl(self, path: str | Path, fields: Iterable[str]) -> dict[str, object]:
        snapshot = self.capture_latest_jsonable(fields)
        out_path = Path(path)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot, sort_keys=True))
            f.write("\n")
        return snapshot

    def append_latest_csv(self, path: str | Path, fields: Iterable[str]) -> dict[str, object]:
        snapshot = self.capture_latest_jsonable(fields)
        row = {field: snapshot.get(field) for field in fields}
        serialized_row = {
            key: (json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value)
            for key, value in row.items()
        }

        out_path = Path(path)
        file_exists = out_path.exists()
        with out_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(fields))
            if not file_exists:
                writer.writeheader()
            writer.writerow(serialized_row)
        return snapshot
