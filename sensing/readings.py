"""Sensor reading dataclasses derived from sensing views."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IMUReading:
    frame_id: int
    sim_time: float
    env_idx: int
    body_index: int

    orientation_world_R: object | None
    angular_velocity_body: object | None
    linear_acceleration_body: object | None


@dataclass
class JointStateReading:
    frame_id: int
    sim_time: float
    env_idx: int

    joint_pos: object | None
    joint_vel: object | None


@dataclass
class ForceSensorReading:
    frame_id: int
    sim_time: float
    env_idx: int

    qfrc_applied: object | None
    tau_smooth: object | None
    body_force: object | None
    contact_force: object | None


@dataclass
class ContactStateReading:
    frame_id: int
    sim_time: float
    env_idx: int

    contact_count: int | None
