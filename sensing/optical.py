"""Optical sensor specs consumed by the optical execution layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from physics.spatial import SpatialTransform

_FRAME_SCALAR_OPTICAL_CHANNELS = frozenset(
    {
        "bvh_stack_overflow_count",
        "bvh_max_stack_depth",
        "shadow_stack_overflow_count",
        "shadow_max_stack_depth",
    }
)


@dataclass
class OpticalRaySensorSpec:
    """World-frame ray batch for first optical executors.

    This spec lives in `sensing/` because it describes the question a sensor is
    asking. The executable scene and material bindings live in `optics/`.
    """

    frame_id: int
    sim_time: float
    env_idx: int
    sensor_id: str
    origins_world: object
    directions_world: object
    max_distance: float = np.inf
    sensor_role: str = "depth"
    ray_shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        origins = np.asarray(self.origins_world, dtype=np.float64)
        directions = np.asarray(self.directions_world, dtype=np.float64)
        if origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError("origins_world must have shape (num_rays, 3)")
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError("directions_world must have shape (num_rays, 3)")
        if origins.shape[0] != directions.shape[0]:
            raise ValueError("origins_world and directions_world must have the same ray count")

        max_distance = float(self.max_distance)
        if max_distance <= 0.0:
            raise ValueError("max_distance must be > 0")
        sensor_role = str(self.sensor_role)
        if not sensor_role:
            raise ValueError("sensor_role must be non-empty")
        ray_shape = _normalize_ray_shape(self.ray_shape, ray_count=origins.shape[0])

        norms = np.linalg.norm(directions, axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError("directions_world must not contain zero-length directions")

        self.origins_world = origins.copy()
        self.directions_world = (directions / norms[:, None]).copy()
        self.max_distance = max_distance
        self.sensor_role = sensor_role
        self.ray_shape = ray_shape

    @property
    def num_rays(self) -> int:
        return int(self.origins_world.shape[0])


@dataclass
class OpticalPinholeCameraSpec:
    """Pinhole camera query that can be lowered to a world-frame ray batch.

    Camera coordinates follow the OpenCV-style convention used by many robotics
    stacks: +X right, +Y down, and +Z along the optical axis.
    """

    frame_id: int
    sim_time: float
    env_idx: int
    sensor_id: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    X_world_camera: SpatialTransform = field(default_factory=SpatialTransform.identity)
    max_distance: float = np.inf
    sensor_role: str = "depth"

    def __post_init__(self) -> None:
        self.width = int(self.width)
        self.height = int(self.height)
        if self.width <= 0:
            raise ValueError("width must be > 0")
        if self.height <= 0:
            raise ValueError("height must be > 0")
        self.fx = float(self.fx)
        self.fy = float(self.fy)
        if self.fx <= 0.0:
            raise ValueError("fx must be > 0")
        if self.fy <= 0.0:
            raise ValueError("fy must be > 0")
        self.cx = float(self.cx)
        self.cy = float(self.cy)
        self.max_distance = float(self.max_distance)
        if self.max_distance <= 0.0:
            raise ValueError("max_distance must be > 0")
        self.sensor_role = str(self.sensor_role)
        if not self.sensor_role:
            raise ValueError("sensor_role must be non-empty")
        if not isinstance(self.X_world_camera, SpatialTransform):
            raise TypeError("X_world_camera must be a SpatialTransform")

    @property
    def image_shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def optical_axis_world(self) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64) @ self.X_world_camera.R.T


@dataclass(frozen=True)
class OpticalCameraImageResult:
    """Image-shaped optical result produced from a flat ray executor result."""

    frame_id: int
    sim_time: float
    env_idx: int
    sensor_id: str
    image_shape: tuple[int, int]
    location: Literal["host", "device", "external"] = "host"
    channels: dict[str, object] = field(default_factory=dict)
    ready_event: object | None = None

    def channel(self, name: str) -> object:
        return self.channels[name]


def build_pinhole_camera_rays(spec: OpticalPinholeCameraSpec) -> OpticalRaySensorSpec:
    """Lower a pinhole camera query to flat world-frame rays."""

    uu, vv = np.meshgrid(
        np.arange(spec.width, dtype=np.float64),
        np.arange(spec.height, dtype=np.float64),
        indexing="xy",
    )
    directions_camera = np.stack(
        [
            (uu - spec.cx) / spec.fx,
            (vv - spec.cy) / spec.fy,
            np.ones_like(uu),
        ],
        axis=-1,
    )
    directions_world = directions_camera.reshape((-1, 3)) @ spec.X_world_camera.R.T
    origins_world = np.repeat(spec.X_world_camera.r[None, :], spec.width * spec.height, axis=0)
    return OpticalRaySensorSpec(
        frame_id=spec.frame_id,
        sim_time=spec.sim_time,
        env_idx=spec.env_idx,
        sensor_id=spec.sensor_id,
        origins_world=origins_world,
        directions_world=directions_world,
        max_distance=spec.max_distance,
        sensor_role=spec.sensor_role,
        ray_shape=spec.image_shape,
    )


def build_pinhole_camera_image_result(
    result: object,
    spec: OpticalPinholeCameraSpec,
    *,
    rays: OpticalRaySensorSpec | None = None,
) -> OpticalCameraImageResult:
    """Reshape flat ray-executor channels and add projected camera `depth_m`."""

    _validate_camera_result_timeline(result, spec)
    if rays is None:
        rays = build_pinhole_camera_rays(spec)
    _validate_camera_rays(rays, spec)
    _validate_required_camera_result_channels(result, {"range_m"})
    channels = _reshape_camera_channels(result.channels, spec.image_shape)
    range_m = np.asarray(result.channel("range_m"), dtype=np.float64)
    projection = rays.directions_world @ spec.optical_axis_world
    channels["depth_m"] = (range_m * projection).reshape(spec.image_shape).copy()
    return OpticalCameraImageResult(
        frame_id=result.frame_id,
        sim_time=result.sim_time,
        env_idx=result.env_idx,
        sensor_id=result.sensor_id,
        image_shape=spec.image_shape,
        location=result.location,
        channels=channels,
        ready_event=result.ready_event,
    )


def _normalize_ray_shape(ray_shape: tuple[int, ...] | None, *, ray_count: int) -> tuple[int, ...] | None:
    if ray_shape is None:
        return None
    normalized = tuple(int(dim) for dim in ray_shape)
    if any(dim <= 0 for dim in normalized):
        raise ValueError("ray_shape dimensions must be > 0")
    product = int(np.prod(normalized, dtype=np.int64))
    if product != ray_count:
        raise ValueError("ray_shape product must match ray count")
    return normalized


def _validate_camera_result_timeline(result: object, spec: OpticalPinholeCameraSpec) -> None:
    if result.frame_id != spec.frame_id:
        raise ValueError("result.frame_id must match spec.frame_id")
    if result.sim_time != spec.sim_time:
        raise ValueError("result.sim_time must match spec.sim_time")
    if result.env_idx != spec.env_idx:
        raise ValueError("result.env_idx must match spec.env_idx")
    if result.sensor_id != spec.sensor_id:
        raise ValueError("result.sensor_id must match spec.sensor_id")


def _validate_required_camera_result_channels(result: object, required_channels: set[str]) -> None:
    missing = sorted(name for name in required_channels if name not in result.channels)
    if missing:
        output_profile = getattr(result, "output_profile", None)
        profile = getattr(output_profile, "value", output_profile)
        detail = f" for output_profile={profile!r}" if profile is not None else ""
        raise ValueError(f"camera image result requires channels {missing}{detail}")


def _validate_camera_rays(rays: OpticalRaySensorSpec, spec: OpticalPinholeCameraSpec) -> None:
    if rays.frame_id != spec.frame_id:
        raise ValueError("rays.frame_id must match spec.frame_id")
    if rays.sim_time != spec.sim_time:
        raise ValueError("rays.sim_time must match spec.sim_time")
    if rays.env_idx != spec.env_idx:
        raise ValueError("rays.env_idx must match spec.env_idx")
    if rays.sensor_id != spec.sensor_id:
        raise ValueError("rays.sensor_id must match spec.sensor_id")
    if rays.ray_shape != spec.image_shape:
        raise ValueError("rays.ray_shape must match spec.image_shape")


def _reshape_flat_channel(value: object, image_shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value)
    ray_count = image_shape[0] * image_shape[1]
    if array.shape[:1] != (ray_count,):
        raise ValueError("camera result channels must have flat ray count as their first dimension")
    return array.reshape((*image_shape, *array.shape[1:])).copy()


def _reshape_camera_channels(
    channels: dict[str, object],
    image_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    ray_count = image_shape[0] * image_shape[1]
    reshaped = {}
    for name, value in channels.items():
        array = np.asarray(value)
        if array.shape[:1] != (ray_count,):
            if name in _FRAME_SCALAR_OPTICAL_CHANNELS:
                continue
            raise ValueError("camera result channels must have flat ray count as their first dimension")
        reshaped[name] = array.reshape((*image_shape, *array.shape[1:])).copy()
    return reshaped
