"""Builders that derive concrete sensor readings from sensing views."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .optical import OpticalCameraImageResult
from .readings import (
    ContactStateReading,
    ForceSensorReading,
    IMUReading,
    JointStateReading,
    OpticalCameraReading,
    RangeSensorReading,
)
from .state_sample import StateSampleView
from .surface_query import SurfaceQueryResult


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


def build_range_sensor_reading(
    result: SurfaceQueryResult,
    *,
    include_hits: bool = True,
) -> RangeSensorReading:
    """Build a range reading from a surface-query result.

    This is intentionally a thin conversion layer: query execution and ray
    pattern generation stay outside the reading builder. CPU results become
    owned NumPy arrays. Future device-array results should be staged by the
    caller before using this host-side builder.
    """
    return RangeSensorReading(
        frame_id=result.frame_id,
        sim_time=result.sim_time,
        env_idx=result.env_idx,
        range_m=np.asarray(result.distance, dtype=np.float64).copy(),
        hit_mask=np.asarray(result.hit_mask, dtype=bool).copy(),
        hit_position_world=(
            None if not include_hits else np.asarray(result.position_world, dtype=np.float64).copy()
        ),
        hit_normal_world=(
            None if not include_hits else np.asarray(result.normal_world, dtype=np.float64).copy()
        ),
    )


def build_optical_camera_reading(
    result: OpticalCameraImageResult,
    *,
    channels: Iterable[str] | None = None,
) -> OpticalCameraReading:
    """Build a host-owned camera reading from an optical image result.

    Execution and image postprocessing stay outside this builder. This function
    only validates the host-side result packet and copies selected image-shaped
    channels into a sensor-facing reading.
    """
    if result.location != "host":
        raise ValueError("optical camera readings require host result channels")

    image_shape = tuple(int(dim) for dim in result.image_shape)
    if len(image_shape) != 2 or image_shape[0] <= 0 or image_shape[1] <= 0:
        raise ValueError("result.image_shape must be a positive (height, width) tuple")

    channel_names = tuple(result.channels) if channels is None else tuple(channels)
    copied_channels: dict[str, np.ndarray] = {}
    for name in channel_names:
        if name not in result.channels:
            raise KeyError(name)
        array = np.asarray(result.channels[name])
        if array.shape[:2] != image_shape:
            raise ValueError("optical camera channels must start with result.image_shape")
        copied_channels[name] = array.copy()

    return OpticalCameraReading(
        frame_id=result.frame_id,
        sim_time=result.sim_time,
        env_idx=result.env_idx,
        sensor_id=result.sensor_id,
        image_shape=image_shape,
        channels=copied_channels,
    )
