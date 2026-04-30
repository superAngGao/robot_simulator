"""Sensor reading dataclasses derived from sensing views."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IMUReading:
    """Idealized phase-1 IMU reading derived from state samples.

    `orientation_world_R` is a rotation matrix. `linear_acceleration_body` stays
    `None` until body-frame acceleration is part of the published contract.
    """

    frame_id: int
    sim_time: float
    env_idx: int
    body_index: int

    orientation_world_R: object | None
    angular_velocity_body: object | None
    linear_acceleration_body: object | None


@dataclass
class JointStateReading:
    """Joint position/velocity reading copied from a state sample."""

    frame_id: int
    sim_time: float
    env_idx: int

    joint_pos: object | None
    joint_vel: object | None


@dataclass
class ForceSensorReading:
    """Force-related phase-1 reading backed by `TelemetrySnapshot`.

    CPU telemetry currently exposes generalized force terms such as
    `qfrc_applied` and `tau_smooth`; GPU telemetry can expose contact force
    sensor values. Missing backend fields remain `None`.
    """

    frame_id: int
    sim_time: float
    env_idx: int

    qfrc_applied: object | None
    tau_smooth: object | None
    body_force: object | None
    contact_force: object | None


@dataclass
class ContactStateReading:
    """Minimal contact-state reading.

    `contact_mask` is optional and, when present, is ordered by the published
    contact-body list. Missing backend fields remain `None`.
    """

    frame_id: int
    sim_time: float
    env_idx: int

    contact_count: int | None
    contact_mask: object | None = None


@dataclass
class RangeSensorReading:
    """Range/ray reading derived from a surface-query result.

    `range_m` is metric distance along normalized query rays. Misses keep
    `hit_mask=False` and `range_m=np.inf`. Hit positions and normals are
    optional payloads for downstream debugging or richer sensor models. When
    hit payloads are omitted, `hit_mask` still indicates which rays hit.
    """

    frame_id: int
    sim_time: float
    env_idx: int

    range_m: object
    hit_mask: object
    hit_position_world: object | None = None
    hit_normal_world: object | None = None
