"""Shared frame-tick contract for multi-product lab workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SimulationFrameTick:
    """One physics-owned frame fact shared by video/debug/observation products."""

    frame_index: int
    env_idx: int
    frame_id: int
    sim_time: float
    published_frame: object
    metadata: Mapping[str, object] = field(default_factory=dict)


def simulation_frame_tick_from_published_frame(
    *,
    frame_index: int,
    published_frame: object,
    env_idx: int = 0,
    metadata: Mapping[str, object] | None = None,
) -> SimulationFrameTick:
    """Build a tick from a published frame carrying ``frame_id`` and ``sim_time``."""

    try:
        frame_id = int(getattr(published_frame, "frame_id"))
        sim_time = float(getattr(published_frame, "sim_time"))
    except AttributeError as exc:
        raise ValueError("published_frame must expose frame_id and sim_time") from exc
    return SimulationFrameTick(
        frame_index=int(frame_index),
        env_idx=int(env_idx),
        frame_id=frame_id,
        sim_time=sim_time,
        published_frame=published_frame,
        metadata=dict(metadata or {}),
    )
