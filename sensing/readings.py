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

    Phase-1 exposes only `contact_count`; per-body active masks wait for a
    published contact-mask contract.
    """

    frame_id: int
    sim_time: float
    env_idx: int

    contact_count: int | None
