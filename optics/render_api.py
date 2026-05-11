"""Internal optical render request and delivery contracts.

This module is intentionally dependency-light and CPU-safe. It defines the
runtime vocabulary shared by optics executors, lab tooling, and future public
API adapters without importing Warp, Torch, or lab-specific code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from sensing.optical import OpticalPinholeCameraSpec, OpticalRaySensorSpec

from .execution import OpticalComputeResult, OpticalOutputProfile, normalize_output_profile


class RenderBackend(Enum):
    """Compute backend requested from the render runtime."""

    DIRECT_LIGHT = "direct_light"
    PATH_TRACING = "path_tracing"


class ReadbackPayload(Enum):
    """Host/device payload requested from delivery."""

    NONE = "none"
    RGB = "rgb"
    RGB8 = "rgb8"
    FULL = "full"


class DeliveryPolicy(Enum):
    """How a rendered result should be delivered to the caller."""

    DEVICE_ONLY = "device_only"
    SYNC_HOST = "sync_host"
    TORCH_ASYNC_ORDERED = "torch_async_ordered"


class WritePolicy(Enum):
    """Optional side-effect writer used by higher-level tools."""

    NONE = "none"
    PNG_SEQUENCE = "png_sequence"
    VIDEO_ENCODER = "video_encoder"


@dataclass(frozen=True)
class RenderDiagnosticsRequest:
    """Optional render diagnostics requested by a caller."""

    profile_timing: bool = False
    traversal_counters: bool = False
    fail_on_overflow: bool = True


@dataclass(frozen=True)
class RenderRequest:
    """One camera or ray render request for an optical render session."""

    frame_id: int
    sim_time: float
    env_idx: int
    camera: OpticalPinholeCameraSpec | None = None
    rays: OpticalRaySensorSpec | None = None
    use_gpu_raygen: bool = True
    backend: RenderBackend | str = RenderBackend.DIRECT_LIGHT
    output_profile: OpticalOutputProfile | str = OpticalOutputProfile.RGB_PREVIEW
    diagnostics: RenderDiagnosticsRequest = field(default_factory=RenderDiagnosticsRequest)
    # Reserved for future RenderBackend.PATH_TRACING accumulation streams.
    # It has no effect for RenderBackend.DIRECT_LIGHT.
    accumulation_id: int | None = None

    def __post_init__(self) -> None:
        backend = (
            self.backend if isinstance(self.backend, RenderBackend) else RenderBackend(str(self.backend))
        )
        output_profile = normalize_output_profile(self.output_profile)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "output_profile", output_profile)
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "sim_time", float(self.sim_time))
        object.__setattr__(self, "env_idx", int(self.env_idx))
        object.__setattr__(self, "use_gpu_raygen", bool(self.use_gpu_raygen))

        if (self.camera is None) == (self.rays is None):
            raise ValueError("RenderRequest requires exactly one of camera or rays")
        if self.rays is not None and self.use_gpu_raygen:
            raise ValueError("use_gpu_raygen=True requires a camera request")
        if self.accumulation_id is not None and int(self.accumulation_id) < 0:
            raise ValueError("accumulation_id must be >= 0 when provided")
        if self.accumulation_id is not None:
            object.__setattr__(self, "accumulation_id", int(self.accumulation_id))

        source = self.camera if self.camera is not None else self.rays
        assert source is not None
        if int(source.frame_id) != self.frame_id:
            raise ValueError("RenderRequest frame_id must match camera/rays frame_id")
        # A1/A2 call sites construct both values from the same source. If this
        # becomes public-facing, consider tolerance or a single source of truth.
        if float(source.sim_time) != self.sim_time:
            raise ValueError("RenderRequest sim_time must match camera/rays sim_time")
        if int(source.env_idx) != self.env_idx:
            raise ValueError("RenderRequest env_idx must match camera/rays env_idx")


@dataclass(frozen=True)
class RenderResult:
    """Rendered compute result plus timing and implementation diagnostics."""

    compute: OpticalComputeResult
    timing: Mapping[str, float] = field(default_factory=dict)
    diagnostics: Mapping[str, int | float] = field(default_factory=dict)


@dataclass(frozen=True)
class DeliveryRequest:
    """Delivery request for a rendered result."""

    payload: ReadbackPayload | str
    policy: DeliveryPolicy | str
    ring_depth: int = 2
    write_policy: WritePolicy | str = WritePolicy.NONE

    def __post_init__(self) -> None:
        payload = (
            self.payload if isinstance(self.payload, ReadbackPayload) else ReadbackPayload(str(self.payload))
        )
        policy = self.policy if isinstance(self.policy, DeliveryPolicy) else DeliveryPolicy(str(self.policy))
        write_policy = (
            self.write_policy
            if isinstance(self.write_policy, WritePolicy)
            else WritePolicy(str(self.write_policy))
        )
        ring_depth = int(self.ring_depth)
        if ring_depth <= 0:
            raise ValueError("ring_depth must be > 0")
        if policy is DeliveryPolicy.DEVICE_ONLY and payload is not ReadbackPayload.NONE:
            raise ValueError("DEVICE_ONLY delivery requires payload=NONE")
        if payload is ReadbackPayload.NONE and policy is not DeliveryPolicy.DEVICE_ONLY:
            raise ValueError("payload=NONE requires DEVICE_ONLY delivery")
        if policy is DeliveryPolicy.TORCH_ASYNC_ORDERED and payload not in (
            ReadbackPayload.RGB,
            ReadbackPayload.RGB8,
        ):
            raise ValueError("TORCH_ASYNC_ORDERED delivery requires payload=RGB or RGB8")
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "write_policy", write_policy)
        object.__setattr__(self, "ring_depth", ring_depth)


@dataclass(frozen=True)
class DeliveryResult:
    """Delivered host/device result metadata."""

    frame_index: int
    host_channels: Mapping[str, object] = field(default_factory=dict)
    device_result: OpticalComputeResult | None = None
    timing: Mapping[str, float] = field(default_factory=dict)
    lag_frames: int = 0
    dropped: bool = False
