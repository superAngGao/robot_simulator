"""Scenario configuration vocabulary for the Optical Pipeline Lab."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

DEFAULT_RENDER_WIDTH = 1920
DEFAULT_RENDER_HEIGHT = 1080


class OpticalLabScenarioFamily(Enum):
    """Developer-facing scenario families used by the lab."""

    RENDER_BENCH = "render_bench"
    VIDEO_ORDERED_EXPORT = "video_ordered_export"
    PARITY_DEBUG = "parity_debug"
    REALTIME_PREVIEW = "realtime_preview"
    SENSOR_ORDERED = "sensor_ordered"


class GeometryMode(Enum):
    STATIC = "static"
    DYNAMIC_RIGID = "dynamic_rigid"
    DEFORMABLE = "deformable"
    FLUID = "fluid"


class AccelBackend(Enum):
    CPU_BVH = "cpu_bvh"
    CUDA_LBVH = "cuda_lbvh"
    OPTIX = "optix"


class AccelPolicy(Enum):
    BUILD_ONCE = "build_once"
    REFIT_EACH_FRAME = "refit_each_frame"
    REBUILD_EACH_FRAME = "rebuild_each_frame"
    DOUBLE_BUFFERED_BUILD = "double_buffered_build"


class RenderBackend(Enum):
    WARP_BVH_DIRECT_LIGHT = "warp_bvh_direct_light"
    CUDA_DIRECT_LIGHT = "cuda_direct_light"
    CUDA_FUSED_RGB = "cuda_fused_rgb"
    OPTIX_FIRST_HIT = "optix_first_hit"
    PATH_TRACER = "path_tracer"


class ReadbackPayload(Enum):
    NONE = "none"
    RGB = "rgb"
    FULL = "full"
    RGB8 = "rgb8"
    DIAGNOSTICS = "diagnostics"
    CUSTOM = "custom"


class DeliveryPolicy(Enum):
    SYNC = "sync"
    DEVICE_ONLY = "device_only"
    ASYNC_ORDERED = "async_ordered"
    ASYNC_LATEST = "async_latest"


class WritePolicy(Enum):
    NONE = "none"
    PNG_SEQUENCE = "png_sequence"
    VIDEO_ENCODER = "video_encoder"
    STREAMING_PREVIEW = "streaming_preview"
    SENSOR_PUBLISH = "sensor_publish"


@dataclass(frozen=True)
class OpticalLabScenarioConfig:
    """Structured lab config for one optical pipeline experiment."""

    scenario_name: str
    scenario_family: OpticalLabScenarioFamily
    device: str = "cuda:0"
    width: int = DEFAULT_RENDER_WIDTH
    height: int = DEFAULT_RENDER_HEIGHT
    scene_preset: str = "go2_menagerie_static"
    geometry_mode: GeometryMode = GeometryMode.STATIC
    camera_mode: str = "camera_orbit"
    accel_backend: AccelBackend = AccelBackend.CUDA_LBVH
    accel_policy: AccelPolicy = AccelPolicy.BUILD_ONCE
    render_backend: RenderBackend = RenderBackend.WARP_BVH_DIRECT_LIGHT
    output_profile: str = "rgb_preview"
    readback_payload: ReadbackPayload = ReadbackPayload.RGB
    delivery_policy: DeliveryPolicy = DeliveryPolicy.SYNC
    write_policy: WritePolicy = WritePolicy.NONE
    diagnostics_policy: str = "required"
    shadows: bool = True

    def validate_implemented(self) -> None:
        """Fail loudly for reserved modes that the lab does not execute yet."""
        if self.geometry_mode is not GeometryMode.STATIC:
            raise NotImplementedError(
                f"geometry_mode={self.geometry_mode.value!r} is reserved; use 'static' for now"
            )
        if self.accel_policy is not AccelPolicy.BUILD_ONCE:
            raise NotImplementedError(
                f"accel_policy={self.accel_policy.value!r} is reserved; use 'build_once' for now"
            )
        if self.render_backend is not RenderBackend.WARP_BVH_DIRECT_LIGHT:
            raise NotImplementedError(
                f"render_backend={self.render_backend.value!r} is reserved; "
                "use 'warp_bvh_direct_light' for now"
            )
        if self.delivery_policy not in (DeliveryPolicy.SYNC, DeliveryPolicy.DEVICE_ONLY):
            raise NotImplementedError(
                f"delivery_policy={self.delivery_policy.value!r} is reserved; use sync/device_only for now"
            )
        if self.readback_payload in (
            ReadbackPayload.DIAGNOSTICS,
            ReadbackPayload.CUSTOM,
        ):
            raise NotImplementedError(
                f"readback_payload={self.readback_payload.value!r} is reserved; "
                "use none/rgb/rgb8/full for now"
            )
        if self.write_policy is not WritePolicy.NONE and self.write_policy is not WritePolicy.PNG_SEQUENCE:
            raise NotImplementedError(
                f"write_policy={self.write_policy.value!r} is reserved; use none/png_sequence for now"
            )
