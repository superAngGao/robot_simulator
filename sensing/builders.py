"""Builders that derive concrete sensor readings from sensing views."""

from __future__ import annotations

import numpy as np

from .readings import ContactStateReading, ForceSensorReading, IMUReading, JointStateReading
from .state_sample import StateSampleView


def _copy_optional_array(value) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value).copy()


def build_joint_state_reading(
    view: StateSampleView,
    *,
    joint_indices: object | None = None,
) -> JointStateReading:
    """Build an ideal joint state reading from `StateSampleView`.

    If `joint_indices` is provided, it is used to slice both position and
    velocity arrays. Missing state fields remain `None`.
    """
    joint_pos = None if view.q is None else np.asarray(view.q).copy()
    joint_vel = None if view.qdot is None else np.asarray(view.qdot).copy()

    if joint_indices is not None:
        if joint_pos is not None:
            joint_pos = joint_pos[joint_indices].copy()
        if joint_vel is not None:
            joint_vel = joint_vel[joint_indices].copy()

    return JointStateReading(
        frame_id=view.frame_id,
        sim_time=view.sim_time,
        env_idx=view.env_idx,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
    )


def build_imu_reading(
    view: StateSampleView,
    *,
    body_index: int,
) -> IMUReading:
    """Build a conservative IMU reading from a state sample.

    Phase-1 exposes orientation and body-frame angular velocity only. Linear
    acceleration is left as `None` until a published acceleration source exists.
    """
    orientation_world_R = None
    if view.X_world is not None:
        orientation_world_R = np.asarray(view.X_world[body_index].R).copy()

    angular_velocity_body = None
    if view.v_bodies is not None:
        angular_velocity_body = np.asarray(view.v_bodies[body_index])[3:6].copy()

    return IMUReading(
        frame_id=view.frame_id,
        sim_time=view.sim_time,
        env_idx=view.env_idx,
        body_index=body_index,
        orientation_world_R=orientation_world_R,
        angular_velocity_body=angular_velocity_body,
        linear_acceleration_body=None,
    )


def build_force_sensor_reading(
    view: StateSampleView,
    *,
    sensor_indices: object | None = None,
) -> ForceSensorReading:
    """Build a force sensor reading from published telemetry fields.

    `contact_force` comes from the published contact-force sensor bridge when
    available. `body_force` is not yet part of the published contract.
    """
    telemetry = view.telemetry
    qfrc_applied = None
    tau_smooth = None
    contact_force = None

    if telemetry is not None:
        qfrc_applied = _copy_optional_array(telemetry.qfrc_applied)
        tau_smooth = _copy_optional_array(telemetry.tau_smooth)
        contact_force = _copy_optional_array(telemetry.force_sensor)

    if sensor_indices is not None and contact_force is not None:
        contact_force = contact_force[sensor_indices].copy()

    return ForceSensorReading(
        frame_id=view.frame_id,
        sim_time=view.sim_time,
        env_idx=view.env_idx,
        qfrc_applied=qfrc_applied,
        tau_smooth=tau_smooth,
        body_force=None,
        contact_force=contact_force,
    )


def build_contact_state_reading(view: StateSampleView) -> ContactStateReading:
    """Build the minimal phase-1 contact-state reading.

    Phase-1 only guarantees `contact_count`.
    """
    return ContactStateReading(
        frame_id=view.frame_id,
        sim_time=view.sim_time,
        env_idx=view.env_idx,
        contact_count=view.contact_count,
        contact_mask=_copy_optional_array(view.contact_mask),
    )
