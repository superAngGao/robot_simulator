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


class FrameSourceKind(Enum):
    STATIC_ASSET_BUILDER = "static_asset_builder"
    SYNTHETIC_FRAME_SEQUENCE = "synthetic_frame_sequence"
    PHYSICS_PUBLISHED_FRAME = "physics_published_frame"
    # Legacy spelling kept while scenario metadata migrates source/clock fields.
    PHYSICS_RUNTIME = "physics_runtime"


class ClockOwnerKind(Enum):
    RUNNER = "runner"
    EXTERNAL_PHYSICS_RUNTIME = "external_physics_runtime"


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


def is_physics_published_frame_source(frame_source: FrameSourceKind) -> bool:
    return frame_source in (
        FrameSourceKind.PHYSICS_PUBLISHED_FRAME,
        FrameSourceKind.PHYSICS_RUNTIME,
    )


@dataclass(frozen=True)
class OpticalLabScenarioConfig:
    """Structured lab config for one optical pipeline experiment."""

    scenario_name: str
    scenario_family: OpticalLabScenarioFamily
    device: str = "cuda:0"
    width: int = DEFAULT_RENDER_WIDTH
    height: int = DEFAULT_RENDER_HEIGHT
    scene_preset: str = "go2_menagerie_static"
    frame_source: FrameSourceKind = FrameSourceKind.STATIC_ASSET_BUILDER
    clock_owner: ClockOwnerKind = ClockOwnerKind.RUNNER
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
        """Validate that this config is supported by at least one lab path.

        This is lab-wide support, not `run_scenario(...)` executability. Some
        implemented configs, such as physics runtime smoke, require explicit
        runtime-owner helpers instead of the value-object scenario runner.
        """
        if self._is_implemented_dynamic_smoke():
            return
        if self._is_implemented_physics_runtime_smoke():
            return
        if self.clock_owner is not ClockOwnerKind.RUNNER:
            raise NotImplementedError(
                f"clock_owner={self.clock_owner.value!r} is reserved outside "
                "the physics_body_triangle_video_smoke path"
            )
        if is_physics_published_frame_source(self.frame_source):
            raise NotImplementedError(
                f"frame_source={self.frame_source.value!r} is reserved outside "
                "the physics_body_triangle_video_smoke path"
            )
        if self.frame_source is FrameSourceKind.SYNTHETIC_FRAME_SEQUENCE:
            raise NotImplementedError(
                "frame_source='synthetic_frame_sequence' is currently implemented only by "
                "the synthetic_body_triangle_dynamic_smoke preset"
            )
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

    def _is_implemented_dynamic_smoke(self) -> bool:
        return (
            self.scene_preset == "synthetic_body_triangle"
            and self.frame_source is FrameSourceKind.SYNTHETIC_FRAME_SEQUENCE
            and self.clock_owner is ClockOwnerKind.RUNNER
            and self.geometry_mode is GeometryMode.DYNAMIC_RIGID
            and self.accel_backend is AccelBackend.CPU_BVH
            and self.accel_policy is AccelPolicy.REFIT_EACH_FRAME
            and self.render_backend is RenderBackend.WARP_BVH_DIRECT_LIGHT
            and self.delivery_policy in (DeliveryPolicy.SYNC, DeliveryPolicy.DEVICE_ONLY)
            and self.readback_payload
            not in (
                ReadbackPayload.DIAGNOSTICS,
                ReadbackPayload.CUSTOM,
            )
            and self.write_policy in (WritePolicy.NONE, WritePolicy.PNG_SEQUENCE)
        )

    def _is_implemented_physics_runtime_smoke(self) -> bool:
        return (
            self.scene_preset == "synthetic_body_triangle"
            and is_physics_published_frame_source(self.frame_source)
            and self.clock_owner is ClockOwnerKind.EXTERNAL_PHYSICS_RUNTIME
            and self.geometry_mode is GeometryMode.DYNAMIC_RIGID
            and self.camera_mode == "fixed_view"
            and self.accel_backend is AccelBackend.CPU_BVH
            and self.accel_policy is AccelPolicy.REFIT_EACH_FRAME
            and self.render_backend is RenderBackend.WARP_BVH_DIRECT_LIGHT
            and self.delivery_policy in (DeliveryPolicy.SYNC, DeliveryPolicy.DEVICE_ONLY)
            and self.readback_payload
            not in (
                ReadbackPayload.DIAGNOSTICS,
                ReadbackPayload.CUSTOM,
            )
            and self.write_policy in (WritePolicy.NONE, WritePolicy.PNG_SEQUENCE)
        )
