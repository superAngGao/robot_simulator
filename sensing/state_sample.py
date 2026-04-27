"""State-sampled sensor view derived from published physics frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics.publish import CpuPublishedFrame, GpuPublishedFrame
from physics.spatial import SpatialTransform
from physics.telemetry import TelemetrySnapshot, build_telemetry_snapshot_from_published_frame


@dataclass
class StateSampleView:
    """Host-side numeric/state sensor view for one published frame / env."""

    frame_id: int
    step_index: int
    sim_time: float
    env_idx: int

    q: np.ndarray | None
    qdot: np.ndarray | None

    X_world: object | None
    v_bodies: np.ndarray | None
    contact_count: int | None

    telemetry: TelemetrySnapshot | None


def _build_from_cpu_frame(
    engine,
    frame: CpuPublishedFrame,
    env_idx: int,
) -> StateSampleView:
    return StateSampleView(
        frame_id=frame.frame_id,
        step_index=frame.step_index,
        sim_time=frame.sim_time,
        env_idx=env_idx,
        q=np.asarray(frame.q).copy() if frame.q is not None else None,
        qdot=np.asarray(frame.qdot).copy() if frame.qdot is not None else None,
        X_world=frame.X_world,
        v_bodies=np.asarray(frame.v_bodies).copy() if frame.v_bodies is not None else None,
        contact_count=None if frame.contact_count is None else int(frame.contact_count),
        telemetry=build_telemetry_snapshot_from_published_frame(engine, frame=frame, env_idx=env_idx),
    )


def _rebuild_spatial_transforms(frame: GpuPublishedFrame, env_idx: int) -> list[SpatialTransform]:
    R_all = frame.x_world_R_wp.numpy()
    r_all = frame.x_world_r_wp.numpy()
    if env_idx >= R_all.shape[0]:
        raise IndexError(f"env_idx={env_idx} out of bounds for {R_all.shape[0]} environments")
    return [
        SpatialTransform(
            R_all[env_idx, body_idx].astype(np.float64),
            r_all[env_idx, body_idx].astype(np.float64),
        )
        for body_idx in range(R_all.shape[1])
    ]


def _build_from_gpu_frame(
    engine,
    frame: GpuPublishedFrame,
    env_idx: int,
) -> StateSampleView:
    q_all = frame.q_wp.numpy()
    qdot_all = frame.qdot_wp.numpy()
    v_all = frame.v_bodies_wp.numpy()
    count_all = None if frame.contact_count_wp is None else frame.contact_count_wp.numpy()

    if env_idx >= q_all.shape[0]:
        raise IndexError(f"env_idx={env_idx} out of bounds for {q_all.shape[0]} environments")

    return StateSampleView(
        frame_id=frame.frame_id,
        step_index=frame.step_index,
        sim_time=frame.sim_time,
        env_idx=env_idx,
        q=q_all[env_idx].copy(),
        qdot=qdot_all[env_idx].copy(),
        X_world=_rebuild_spatial_transforms(frame, env_idx=env_idx),
        v_bodies=v_all[env_idx].copy(),
        contact_count=None if count_all is None else int(count_all[env_idx]),
        telemetry=build_telemetry_snapshot_from_published_frame(engine, frame=frame, env_idx=env_idx),
    )


def build_state_sample_view(
    engine,
    frame: CpuPublishedFrame | GpuPublishedFrame | None = None,
    env_idx: int = 0,
) -> StateSampleView:
    """Build a host-side state sample view from a published frame.

    This view is intentionally conservative:
    - it only consumes published-frame fields and public bridges;
    - it does not read engine-private scratch;
    - CPU/GPU asymmetry is handled inside `TelemetrySnapshot`.
    """
    if frame is None:
        frame = engine.latest_published_frame()
    if frame is None:
        raise RuntimeError("No published frame is available yet.")

    if isinstance(frame, CpuPublishedFrame):
        return _build_from_cpu_frame(engine, frame=frame, env_idx=env_idx)

    if isinstance(frame, GpuPublishedFrame):
        return _build_from_gpu_frame(engine, frame=frame, env_idx=env_idx)

    raise TypeError(f"Unsupported published frame type: {type(frame).__name__}")
