"""Internal optical render request and delivery contracts.

This module is intentionally dependency-light and CPU-safe. It defines the
runtime vocabulary shared by optics executors, lab tooling, and future public
API adapters without importing Warp, Torch, or lab-specific code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from sensing.optical import OpticalPinholeCameraSpec, OpticalRaySensorSpec

from .execution import OpticalComputeResult, OpticalOutputProfile, normalize_output_profile

_NAN = float("nan")


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
class FramePrepareTiming:
    """Frame preparation timing owned by begin_frame/frame-context setup."""

    snapshot_ms: float = _NAN
    accel_refit_ms: float = _NAN
    accel_rebuild_ms: float = _NAN

    def to_flat_mapping(self) -> dict[str, float]:
        """Return the current flat CSV-compatible timing keys."""

        return {
            "snapshot_ms": float(self.snapshot_ms),
            "accel_refit_ms": float(self.accel_refit_ms),
            "accel_rebuild_ms": float(self.accel_rebuild_ms),
        }


@dataclass(frozen=True)
class RenderTimingSummary:
    """Render-owned timing summary."""

    execute_ms: float = _NAN
    profile_sum_ms: float = _NAN
    overhead_ms: float = _NAN
    phases: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def from_flat_mapping(cls, timing: Mapping[str, float]) -> "RenderTimingSummary":
        phases = {
            key: float(value)
            for key, value in timing.items()
            if key.startswith("render_")
            and key.endswith("_ms")
            and key
            not in {
                "render_execute_ms",
                "render_overhead_ms",
            }
        }
        profile_values = [value for value in phases.values() if value == value]
        profile_sum_ms = sum(profile_values) if profile_values else _NAN
        return cls(
            execute_ms=float(timing.get("render_execute_ms", _NAN)),
            profile_sum_ms=profile_sum_ms,
            overhead_ms=float(timing.get("render_overhead_ms", _NAN)),
            phases=phases,
        )

    def to_flat_mapping(self) -> dict[str, float]:
        """Return the current flat CSV-compatible render timing keys."""

        return {
            "render_execute_ms": float(self.execute_ms),
            "render_overhead_ms": float(self.overhead_ms),
            **{key: float(value) for key, value in self.phases.items()},
        }


@dataclass(frozen=True)
class RenderResult:
    """Rendered compute result plus render-owned timing and diagnostics."""

    compute: OpticalComputeResult
    timing: Mapping[str, float] = field(default_factory=dict)
    diagnostics: Mapping[str, int | float] = field(default_factory=dict)
    render_timing: RenderTimingSummary | None = None


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
class DeliveryTimingSummary:
    """Delivery-owned timing summary."""

    pack_rgb8_ms: float = _NAN
    readback_submit_ms: float = _NAN
    readback_wait_ms: float = _NAN
    readback_host_ms: float = _NAN
    image_build_ms: float = _NAN
    encode_write_ms: float = _NAN

    def to_flat_mapping(self) -> dict[str, float]:
        """Return the current flat CSV-compatible delivery timing keys."""

        return {
            "pack_rgb8_ms": float(self.pack_rgb8_ms),
            "readback_submit_ms": float(self.readback_submit_ms),
            "readback_wait_ms": float(self.readback_wait_ms),
            "readback_host_ms": float(self.readback_host_ms),
            "image_build_ms": float(self.image_build_ms),
            "encode_write_ms": float(self.encode_write_ms),
        }


@dataclass(frozen=True)
class DeliveryResult:
    """Delivered host/device result metadata for a completed frame."""

    completed_frame_index: int
    host_channels: Mapping[str, object] = field(default_factory=dict)
    device_result: OpticalComputeResult | None = None
    delivery: DeliveryTimingSummary = field(default_factory=DeliveryTimingSummary)
    lag_frames: int = 0
    ring_depth: int = 0
    ring_block_count: int = 0
    dropped: bool = False
    backpressure_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "completed_frame_index", int(self.completed_frame_index))
        object.__setattr__(self, "lag_frames", int(self.lag_frames))
        object.__setattr__(self, "ring_depth", int(self.ring_depth))
        object.__setattr__(self, "ring_block_count", int(self.ring_block_count))
        object.__setattr__(self, "backpressure_count", int(self.backpressure_count))
        object.__setattr__(self, "dropped", bool(self.dropped))

    @property
    def frame_index(self) -> int:
        """Transition alias for completed_frame_index."""

        return self.completed_frame_index


@dataclass(frozen=True)
class FrameTimingSummary:
    """Frame-level timing summary."""

    work_sum_ms: float = _NAN
    observed_frame_ms: float = _NAN
    critical_path_ms: float = _NAN
    instant_fps: float = _NAN

    def to_flat_mapping(self) -> dict[str, float]:
        """Return the current flat CSV-compatible frame summary keys."""

        return {
            "frame_total_ms": float(self.observed_frame_ms),
            "instant_fps": float(self.instant_fps),
        }


@dataclass(frozen=True)
class FrameResult:
    """Lightweight completed-frame observation summary.

    This object is frame-bound, but it does not own heavy GPU resources such as
    published physics frames, device snapshots, or acceleration structures.
    """

    frame_id: int
    sim_time: float
    env_idx: int
    prepare: FramePrepareTiming = field(default_factory=FramePrepareTiming)
    render: RenderTimingSummary = field(default_factory=RenderTimingSummary)
    delivery: DeliveryTimingSummary = field(default_factory=DeliveryTimingSummary)
    summary: FrameTimingSummary = field(default_factory=FrameTimingSummary)
    diagnostics: Mapping[str, int | float] = field(default_factory=dict)
    completed_frame_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "sim_time", float(self.sim_time))
        object.__setattr__(self, "env_idx", int(self.env_idx))
        if self.completed_frame_index is not None:
            object.__setattr__(self, "completed_frame_index", int(self.completed_frame_index))

    def to_csv_row(self) -> dict[str, float | int]:
        """Flatten owned timing blocks to the current lab CSV vocabulary."""

        row: dict[str, float | int] = {"sim_time": self.sim_time}
        if self.completed_frame_index is not None:
            row["frame_index"] = self.completed_frame_index
            row["completed_frame_index"] = self.completed_frame_index
        row.update(self.prepare.to_flat_mapping())
        row.update(self.render.to_flat_mapping())
        row.update(self.delivery.to_flat_mapping())
        row.update(self.summary.to_flat_mapping())
        row.update({key: value for key, value in self.diagnostics.items()})
        return row


@runtime_checkable
class RenderFrameContext(Protocol):
    """Frame-scoped internal render context contract."""

    @property
    def frame_id(self) -> int: ...

    @property
    def sim_time(self) -> float: ...

    @property
    def env_idx(self) -> int: ...

    def render(self, request: RenderRequest) -> RenderResult: ...


@runtime_checkable
class OpticalDeliveryRuntime(Protocol):
    """Explicit delivery runtime contract for sync and async delivery."""

    @property
    def request(self) -> DeliveryRequest: ...

    def submit(
        self,
        rendered: RenderResult,
        *,
        frame_start: float | None = None,
    ) -> DeliveryResult | None: ...

    def complete_available(
        self,
        *,
        latest_rendered_frame_index: int | None = None,
    ) -> Sequence[DeliveryResult]: ...

    def flush(self) -> Sequence[DeliveryResult]: ...


@runtime_checkable
class OpticalRenderPipeline(Protocol):
    """Long-lived internal optical render pipeline contract."""

    def begin_frame(
        self,
        frame_inputs: object | None = None,
        *,
        env_idx: int = 0,
    ) -> RenderFrameContext: ...

    def create_delivery_runtime(
        self,
        request: DeliveryRequest,
    ) -> OpticalDeliveryRuntime: ...

    def deliver(
        self,
        rendered: RenderResult,
        request: DeliveryRequest | None = None,
    ) -> DeliveryResult:
        """Sync-only convenience delivery; async ordered delivery uses a runtime."""

        ...
