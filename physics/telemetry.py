"""Host-side telemetry views derived from published physics frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .publish import CpuPublishedFrame, GpuPublishedFrame


@dataclass
class TelemetrySnapshot:
    """Host-owned telemetry view for one published frame / env.

    Field availability is intentionally asymmetric in phase-1 because the CPU
    and GPU published-frame contracts are not yet identical:

    - CPU frames currently provide `qfrc_*`, `tau_smooth`, `qacc_smooth`,
      `qacc_total`, but no `force_sensor`.
    - GPU frames currently provide `qacc_smooth`, `qacc_total`,
      `force_sensor`, but not the generalized `qfrc_*` / `tau_smooth` terms.

    Callers should treat any individual field as optional unless they have
    already constrained the execution path to CPU-only or GPU-only.
    """

    frame_id: int
    step_index: int
    sim_time: float
    env_idx: int

    qfrc_passive: np.ndarray | None = None
    qfrc_actuator: np.ndarray | None = None
    qfrc_applied: np.ndarray | None = None
    tau_smooth: np.ndarray | None = None
    qacc_smooth: np.ndarray | None = None
    qacc_total: np.ndarray | None = None
    force_sensor: np.ndarray | None = None


def _copy_optional_array(value) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value).copy()


def _build_from_cpu_frame(frame: CpuPublishedFrame, env_idx: int) -> TelemetrySnapshot:
    telemetry = frame.telemetry
    if telemetry is None:
        return TelemetrySnapshot(
            frame_id=frame.frame_id,
            step_index=frame.step_index,
            sim_time=frame.sim_time,
            env_idx=env_idx,
        )
    return TelemetrySnapshot(
        frame_id=frame.frame_id,
        step_index=frame.step_index,
        sim_time=frame.sim_time,
        env_idx=env_idx,
        qfrc_passive=_copy_optional_array(telemetry.qfrc_passive),
        qfrc_actuator=_copy_optional_array(telemetry.qfrc_actuator),
        qfrc_applied=_copy_optional_array(telemetry.qfrc_applied),
        tau_smooth=_copy_optional_array(telemetry.tau_smooth),
        qacc_smooth=_copy_optional_array(telemetry.qacc_smooth),
        # ForceState.qacc is the final constrained (total) acceleration.
        qacc_total=_copy_optional_array(telemetry.qacc),
    )


def _build_from_gpu_frame(frame: GpuPublishedFrame, env_idx: int, engine=None) -> TelemetrySnapshot:
    snapshot = TelemetrySnapshot(
        frame_id=frame.frame_id,
        step_index=frame.step_index,
        sim_time=frame.sim_time,
        env_idx=env_idx,
    )
    telemetry_ref = frame.telemetry_ref
    if telemetry_ref is None:
        return snapshot

    snapshot.qacc_smooth = telemetry_ref["qacc_smooth_wp"].numpy()[env_idx].copy()
    snapshot.qacc_total = telemetry_ref["qacc_total_wp"].numpy()[env_idx].copy()

    force_sensor = telemetry_ref.get("force_sensor_wp")
    if force_sensor is not None:
        flat = force_sensor.numpy()[env_idx].copy()
        nc_sensor = getattr(engine, "nc_sensor", None)
        if isinstance(nc_sensor, (int, np.integer)) and nc_sensor > 0:
            snapshot.force_sensor = flat.reshape(int(nc_sensor), 3)
        else:
            snapshot.force_sensor = flat

    return snapshot


def build_telemetry_snapshot_from_published_frame(
    engine,
    frame: CpuPublishedFrame | GpuPublishedFrame | None = None,
    env_idx: int = 0,
) -> TelemetrySnapshot:
    """Build a host-owned telemetry snapshot from a published frame.

    The snapshot is intentionally conservative: it only exposes telemetry that
    is already part of the published-frame contract. It does not reach into
    engine-private scratch buffers.
    """
    if frame is None:
        frame = engine.latest_published_frame()
    if frame is None:
        raise RuntimeError("No published frame is available yet.")

    if isinstance(frame, CpuPublishedFrame):
        return _build_from_cpu_frame(frame, env_idx=env_idx)

    if isinstance(frame, GpuPublishedFrame):
        return _build_from_gpu_frame(frame, env_idx=env_idx, engine=engine)

    raise TypeError(f"Unsupported published frame type: {type(frame).__name__}")
