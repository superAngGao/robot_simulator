"""Video delivery helpers for Optical Pipeline Lab experiments."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from examples.optical_direct_light_preview import linear_rgb_to_preview_uint8
from optics import stage_optical_channels, stage_optical_compute_result_to_host
from optics.render_api import (
    DeliveryPolicy as RuntimeDeliveryPolicy,
)
from optics.render_api import (
    DeliveryRequest,
    DeliveryTimingSummary,
)
from optics.render_api import (
    DeliveryResult as RuntimeDeliveryResult,
)
from optics.render_api import (
    ReadbackPayload as RuntimeReadbackPayload,
)
from optics.render_api import (
    WritePolicy as RuntimeWritePolicy,
)
from tools.optical_pipeline_lab.async_readback import (
    TorchAsyncReadbackJob,
    TorchAsyncReadbackRing,
)
from tools.optical_pipeline_lab.rgb_pack import (
    rgb_pack_available,
    rgb_pack_import_error,
)
from tools.optical_pipeline_lab.timing import NAN as _NAN
from tools.optical_pipeline_lab.timing import SHADOW_TRAVERSAL_COUNTER_FIELDS

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover - optional image writer dependency.
    Image = None
    _PIL_IMPORT_ERROR = exc
else:
    _PIL_IMPORT_ERROR = None

_VIDEO_DIAGNOSTIC_CHANNELS = (
    "bvh_stack_overflow_count",
    "shadow_stack_overflow_count",
    "bvh_max_stack_depth",
    "shadow_max_stack_depth",
)
_VIDEO_SHADOW_TRAVERSAL_CHANNELS = SHADOW_TRAVERSAL_COUNTER_FIELDS
_VIDEO_RGB_CHANNELS = ("rgb",) + _VIDEO_DIAGNOSTIC_CHANNELS
_VIDEO_RGB8_CHANNELS = ("rgb8",) + _VIDEO_DIAGNOSTIC_CHANNELS


@dataclass
class RenderedVideoFrame:
    """Lab video render envelope handed to delivery.

    This carries RenderResult-derived compute/timing plus video-loop metadata;
    it is not a replacement for the generic runtime RenderResult.
    """

    frame_index: int
    camera: object
    result: object
    camera_rays_ms: float
    render_execute_ms: float
    render_profile_row: dict[str, float]
    include_shadow_traversal_stats: bool
    geometry_mode: str = "static"
    prepare_timing: Mapping[str, float] = field(default_factory=dict)


@dataclass
class DeliveredVideoFrame:
    """Completed delivery result for one consumer-visible video frame."""

    rendered: RenderedVideoFrame
    completed_frame_index: int
    host_channels: Mapping[str, object]
    delivery_timing: DeliveryTimingSummary
    observed_frame_ms: float
    frame_path: str = ""
    readback_lag_frames: int = 0
    readback_ring_depth: int = 0
    readback_ring_block_count: int = 0
    overlap_ratio: float = _NAN

    def to_runtime_delivery_result(self) -> RuntimeDeliveryResult:
        """Return the CPU-safe runtime delivery vocabulary for this frame.

        Lab-only analysis and writer fields stay on DeliveredVideoFrame.
        """

        return RuntimeDeliveryResult(
            completed_frame_index=self.completed_frame_index,
            host_channels=self.host_channels,
            delivery=self.delivery_timing,
            lag_frames=self.readback_lag_frames,
            ring_depth=self.readback_ring_depth,
            ring_block_count=self.readback_ring_block_count,
        )


@dataclass
class VideoDeliveryRunConfig:
    """Loop-level metadata needed for stable lab frame CSV rows."""

    video_fps: float
    video_frames: int
    video_raygen: str
    video_ray_cache: str
    delivery_policy_label: str
    fail_on_overflow: bool = True


@dataclass
class _AsyncDeliveryJob:
    frame_start: float
    rendered: RenderedVideoFrame
    readback_job: TorchAsyncReadbackJob
    delivery_timing: DeliveryTimingSummary


def video_delivery_request(
    *,
    readback_mode: str,
    delivery_mode: str,
    ring_depth: int,
    write_frames: bool,
) -> DeliveryRequest:
    payload = RuntimeReadbackPayload(readback_mode)
    if payload is RuntimeReadbackPayload.NONE:
        policy = RuntimeDeliveryPolicy.DEVICE_ONLY
    elif delivery_mode == "torch_async":
        policy = RuntimeDeliveryPolicy.TORCH_ASYNC_ORDERED
    else:
        policy = RuntimeDeliveryPolicy.SYNC_HOST
    return DeliveryRequest(
        payload=payload,
        policy=policy,
        ring_depth=ring_depth,
        write_policy=(RuntimeWritePolicy.PNG_SEQUENCE if write_frames else RuntimeWritePolicy.NONE),
    )


class VideoDeliveryFacade:
    """Lab-local delivery facade for sync and ordered async video readback."""

    def __init__(
        self,
        *,
        request: DeliveryRequest,
        delivery_policy_label: str,
        frame_dir: Path,
        pack_rgb8: Callable[[object], object],
        synchronize_event: Callable[[object], None],
        readback_ring: TorchAsyncReadbackRing | None = None,
    ) -> None:
        self.request = request
        self.delivery_policy_label = str(delivery_policy_label)
        self.frame_dir = frame_dir
        self._pack_rgb8 = pack_rgb8
        self._synchronize_event = synchronize_event
        self._readback_ring = readback_ring
        self._pending: _AsyncDeliveryJob | None = None
        self._ready_to_complete: list[tuple[_AsyncDeliveryJob, int]] = []
        self._ready: list[DeliveredVideoFrame] = []
        self._first_frame_start: float | None = None
        self._last_completion_time: float | None = None
        self._latest_rendered_frame_index = -1
        if self.request.write_policy is RuntimeWritePolicy.PNG_SEQUENCE:
            if Image is None:
                raise SystemExit("Writing video benchmark frames requires Pillow") from _PIL_IMPORT_ERROR
            self.frame_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(
        cls,
        *,
        request: DeliveryRequest,
        delivery_policy_label: str,
        frame_dir: Path,
        pack_rgb8: Callable[[object], object],
        synchronize_event: Callable[[object], None],
        warmup_result_factory: Callable[[], tuple[object, bool]] | None = None,
    ) -> "VideoDeliveryFacade":
        readback_ring = None
        if request.policy is RuntimeDeliveryPolicy.TORCH_ASYNC_ORDERED:
            if warmup_result_factory is None:
                raise ValueError("torch async delivery requires warmup_result_factory")
            warmup_result, include_shadow_traversal_stats = warmup_result_factory()
            if request.payload is RuntimeReadbackPayload.RGB8:
                warmup_result, _ = _pack_rgb8_result(
                    warmup_result,
                    pack_rgb8=pack_rgb8,
                    synchronize_event=synchronize_event,
                )
            readback_ring = TorchAsyncReadbackRing.from_warmup_result(
                warmup_result,
                channels=video_readback_channels(
                    request.payload.value,
                    include_shadow_traversal_stats=include_shadow_traversal_stats,
                ),
                ring_depth=request.ring_depth,
            )
        return cls(
            request=request,
            delivery_policy_label=delivery_policy_label,
            frame_dir=frame_dir,
            pack_rgb8=pack_rgb8,
            synchronize_event=synchronize_event,
            readback_ring=readback_ring,
        )

    def submit(
        self,
        rendered: RenderedVideoFrame,
        *,
        frame_start: float,
    ) -> DeliveredVideoFrame | None:
        self._latest_rendered_frame_index = int(rendered.frame_index)
        if self._first_frame_start is None:
            self._first_frame_start = float(frame_start)
        if self.request.policy is RuntimeDeliveryPolicy.TORCH_ASYNC_ORDERED:
            previous_pending = self._pending
            if (
                previous_pending is not None
                and self._readback_ring is not None
                and self._readback_ring.ring_depth <= 1
            ):
                raise RuntimeError("complete_available() must run before submitting to a full async ring")
            self._pending = None
            current_pending = self._submit_async(rendered, frame_start=frame_start)
            if previous_pending is not None:
                self._ready_to_complete.append((previous_pending, 0))
            self._pending = current_pending
            return None
        return self._deliver_sync(rendered, frame_start=frame_start)

    def complete_available(
        self,
        *,
        latest_rendered_frame_index: int | None = None,
    ) -> list[DeliveredVideoFrame]:
        if latest_rendered_frame_index is not None:
            self._latest_rendered_frame_index = int(latest_rendered_frame_index)
        if self.request.policy is not RuntimeDeliveryPolicy.TORCH_ASYNC_ORDERED:
            return []
        if self._readback_ring is not None and self._readback_ring.ring_depth <= 1:
            if latest_rendered_frame_index is not None and self._pending is not None:
                self._ready.append(self._complete_pending(ring_block_count=1))
            return self._pop_ready()
        while self._ready_to_complete:
            job, ring_block_count = self._ready_to_complete.pop(0)
            self._ready.append(self._complete_job(job, ring_block_count=ring_block_count))
        return self._pop_ready()

    def flush(self) -> list[DeliveredVideoFrame]:
        while self._ready_to_complete:
            job, ring_block_count = self._ready_to_complete.pop(0)
            self._ready.append(self._complete_job(job, ring_block_count=ring_block_count))
        if self.request.policy is RuntimeDeliveryPolicy.TORCH_ASYNC_ORDERED and self._pending is not None:
            self._ready.append(self._complete_pending(ring_block_count=0))
        return self._pop_ready()

    def _deliver_sync(
        self,
        rendered: RenderedVideoFrame,
        *,
        frame_start: float,
    ) -> DeliveredVideoFrame:
        result, pack_rgb8_ms = self._pack_for_delivery(rendered.result)
        readback_start = time.perf_counter()
        host_channels = readback_video_result(
            result,
            self.request.payload.value,
            include_shadow_traversal_stats=rendered.include_shadow_traversal_stats,
        )
        readback_host_ms = (
            (time.perf_counter() - readback_start) * 1000.0
            if self.request.payload is not RuntimeReadbackPayload.NONE
            else _NAN
        )
        frame_path, image_build_ms, encode_or_write_ms = self._write_frame_if_requested(
            host_channels,
            rendered.frame_index,
        )
        observed_frame_ms = (time.perf_counter() - frame_start) * 1000.0
        return DeliveredVideoFrame(
            rendered=rendered,
            completed_frame_index=int(rendered.frame_index),
            host_channels=host_channels,
            delivery_timing=DeliveryTimingSummary(
                pack_rgb8_ms=pack_rgb8_ms,
                readback_submit_ms=_NAN,
                readback_wait_ms=_NAN,
                readback_host_ms=readback_host_ms,
                image_build_ms=image_build_ms,
                encode_write_ms=encode_or_write_ms,
            ),
            observed_frame_ms=observed_frame_ms,
            frame_path=frame_path,
        )

    def _submit_async(
        self,
        rendered: RenderedVideoFrame,
        *,
        frame_start: float,
    ) -> _AsyncDeliveryJob:
        if self._readback_ring is None:
            raise RuntimeError("torch async delivery ring is not initialized")
        result, pack_rgb8_ms = self._pack_for_delivery(rendered.result)
        readback_job = self._readback_ring.submit(result, frame_index=rendered.frame_index)
        return _AsyncDeliveryJob(
            frame_start=float(frame_start),
            rendered=rendered,
            readback_job=readback_job,
            delivery_timing=DeliveryTimingSummary(
                pack_rgb8_ms=pack_rgb8_ms,
                readback_submit_ms=readback_job.submit_ms,
            ),
        )

    def _complete_pending(self, *, ring_block_count: int) -> DeliveredVideoFrame:
        if self._pending is None:
            raise RuntimeError("no pending async delivery job")
        job = self._pending
        self._pending = None
        return self._complete_job(job, ring_block_count=ring_block_count)

    def _complete_job(self, job: _AsyncDeliveryJob, *, ring_block_count: int) -> DeliveredVideoFrame:
        readback_wait_ms = job.readback_job.synchronize()
        completion_time = time.perf_counter()
        readback_copy_ms = job.readback_job.copy_elapsed_ms()
        host_channels = job.readback_job.host_channels()
        frame_path, image_build_ms, encode_or_write_ms = self._write_frame_if_requested(
            host_channels,
            job.rendered.frame_index,
        )
        first_frame_start = (
            self._first_frame_start if self._first_frame_start is not None else job.frame_start
        )
        previous_completion = (
            self._last_completion_time if self._last_completion_time is not None else first_frame_start
        )
        observed_frame_ms = (completion_time - previous_completion) * 1000.0
        self._last_completion_time = completion_time
        render_delivery_ms = _render_delivery_ms(
            render_execute_ms=job.rendered.render_execute_ms,
            pack_rgb8_ms=job.delivery_timing.pack_rgb8_ms,
        )
        overlap_ratio = _overlap_ratio(
            observed_ms=observed_frame_ms,
            render_ms=render_delivery_ms,
            readback_ms=readback_copy_ms,
        )
        return DeliveredVideoFrame(
            rendered=job.rendered,
            completed_frame_index=int(job.rendered.frame_index),
            host_channels=host_channels,
            delivery_timing=DeliveryTimingSummary(
                pack_rgb8_ms=job.delivery_timing.pack_rgb8_ms,
                readback_submit_ms=job.delivery_timing.readback_submit_ms,
                readback_wait_ms=readback_wait_ms,
                readback_host_ms=readback_copy_ms,
                image_build_ms=image_build_ms,
                encode_write_ms=encode_or_write_ms,
            ),
            observed_frame_ms=observed_frame_ms,
            frame_path=frame_path,
            readback_lag_frames=max(
                int(self._latest_rendered_frame_index) - int(job.rendered.frame_index),
                0,
            ),
            readback_ring_depth=0 if self._readback_ring is None else int(self._readback_ring.ring_depth),
            readback_ring_block_count=int(ring_block_count),
            overlap_ratio=overlap_ratio,
        )

    def _pack_for_delivery(self, result):
        if self.request.payload is not RuntimeReadbackPayload.RGB8:
            return result, _NAN
        return _pack_rgb8_result(
            result,
            pack_rgb8=self._pack_rgb8,
            synchronize_event=self._synchronize_event,
        )

    def _write_frame_if_requested(
        self,
        host_channels: Mapping[str, object],
        frame_index: int,
    ) -> tuple[str, float, float]:
        if self.request.write_policy is not RuntimeWritePolicy.PNG_SEQUENCE:
            return "", _NAN, _NAN
        image_start = time.perf_counter()
        rgb_preview = host_rgb_preview_for_readback(host_channels, self.request.payload.value)
        image_build_ms = (time.perf_counter() - image_start) * 1000.0

        write_start = time.perf_counter()
        path = self.frame_dir / f"rgb_{int(frame_index):06d}.png"
        Image.fromarray(rgb_preview).save(path)
        encode_or_write_ms = (time.perf_counter() - write_start) * 1000.0
        return str(path), image_build_ms, encode_or_write_ms

    def _pop_ready(self) -> list[DeliveredVideoFrame]:
        ready = self._ready
        self._ready = []
        return ready


class VideoFrameTimingRowBuilder:
    """Build stable frame_timing.csv rows from render and delivery outputs."""

    def __init__(self, config: VideoDeliveryRunConfig) -> None:
        self.config = config
        self.rolling_window_ms: list[float] = []

    def build_row(self, delivered: DeliveredVideoFrame) -> dict[str, object]:
        rendered = delivered.rendered
        frame_total_ms = float(delivered.observed_frame_ms)
        self.rolling_window_ms.append(frame_total_ms)
        # Match the existing lab rolling FPS window used before the facade split.
        if len(self.rolling_window_ms) > 30:
            self.rolling_window_ms.pop(0)
        instant_fps = 1000.0 / frame_total_ms if frame_total_ms > 0.0 else float("inf")
        rolling_fps = (
            1000.0 * float(len(self.rolling_window_ms)) / sum(self.rolling_window_ms)
            if sum(self.rolling_window_ms) > 0.0
            else float("inf")
        )
        primary_overflow = staged_scalar(delivered.host_channels, "bvh_stack_overflow_count")
        shadow_overflow = staged_scalar(delivered.host_channels, "shadow_stack_overflow_count")
        primary_max_stack = staged_scalar(delivered.host_channels, "bvh_max_stack_depth")
        shadow_max_stack = staged_scalar(delivered.host_channels, "shadow_max_stack_depth")
        if self.config.fail_on_overflow and (primary_overflow or shadow_overflow):
            raise SystemExit(
                f"BVH stack overflow detected at frame {delivered.completed_frame_index}: "
                f"primary={primary_overflow}, shadow={shadow_overflow}"
            )
        row = {
            "frame_index": int(delivered.completed_frame_index),
            "sim_time": rendered.camera.sim_time,
            "video_time": float(delivered.completed_frame_index) / float(self.config.video_fps),
            "geometry_mode": rendered.geometry_mode,
            "delivery_policy": self.config.delivery_policy_label,
            "raygen_mode": self.config.video_raygen,
            "ray_cache_mode": self.config.video_ray_cache,
            "readback_mode": self._readback_mode(),
            "write_mode": ("rgb_png" if delivered.frame_path else "none"),
            "camera_rays_ms": rendered.camera_rays_ms,
            "snapshot_ms": _prepare_timing_value(rendered.prepare_timing, "snapshot_ms"),
            "accel_refit_ms": _prepare_timing_value(rendered.prepare_timing, "accel_refit_ms"),
            "accel_rebuild_ms": _prepare_timing_value(rendered.prepare_timing, "accel_rebuild_ms"),
            "render_execute_ms": rendered.render_execute_ms,
            "pack_rgb8_ms": delivered.delivery_timing.pack_rgb8_ms,
            **rendered.render_profile_row,
            "readback_submit_ms": delivered.delivery_timing.readback_submit_ms,
            "readback_wait_ms": delivered.delivery_timing.readback_wait_ms,
            "readback_host_ms": delivered.delivery_timing.readback_host_ms,
            "image_build_ms": delivered.delivery_timing.image_build_ms,
            "encode_or_write_ms": delivered.delivery_timing.encode_write_ms,
            "frame_total_ms": frame_total_ms,
            "instant_fps": instant_fps,
            "rolling_fps": rolling_fps,
            "primary_overflow": primary_overflow,
            "shadow_overflow": shadow_overflow,
            "primary_max_stack": primary_max_stack,
            "shadow_max_stack": shadow_max_stack,
            **staged_shadow_traversal_fields(delivered.host_channels),
            "frame_path": delivered.frame_path,
        }
        if self.config.delivery_policy_label == "torch_async":
            row.update(
                {
                    "readback_lag_frames": delivered.readback_lag_frames,
                    "readback_ring_depth": delivered.readback_ring_depth,
                    "readback_ring_block_count": delivered.readback_ring_block_count,
                    "completed_frame_index": delivered.completed_frame_index,
                    "overlap_ratio": delivered.overlap_ratio,
                }
            )
        return row

    def progress_line(self, delivered: DeliveredVideoFrame) -> str:
        rendered = delivered.rendered
        readback_ms = delivered.delivery_timing.readback_host_ms
        parts = [
            "  video_frame ",
            f"{delivered.completed_frame_index + 1}/{self.config.video_frames}: ",
            f"total={delivered.observed_frame_ms:.3f}ms, ",
            f"render={rendered.render_execute_ms:.3f}ms, ",
            format_pack_rgb8(delivered.delivery_timing.pack_rgb8_ms),
            format_render_profile(rendered.render_profile_row),
        ]
        if self.config.delivery_policy_label == "torch_async":
            parts.extend(
                [
                    f"readback={readback_ms:.3f}ms, ",
                    f"submit={delivered.delivery_timing.readback_submit_ms:.3f}ms, ",
                    f"wait={delivered.delivery_timing.readback_wait_ms:.3f}ms, ",
                    f"lag={delivered.readback_lag_frames}, ",
                    f"overlap={format_ratio(delivered.overlap_ratio)}, ",
                ]
            )
        else:
            parts.append(f"readback={format_ms(readback_ms)}, ")
        row_channels = delivered.host_channels
        primary_overflow = staged_scalar(row_channels, "bvh_stack_overflow_count")
        shadow_overflow = staged_scalar(row_channels, "shadow_stack_overflow_count")
        instant_fps = (
            1000.0 / delivered.observed_frame_ms if delivered.observed_frame_ms > 0.0 else float("inf")
        )
        rolling_fps = (
            1000.0 * float(len(self.rolling_window_ms)) / sum(self.rolling_window_ms)
            if sum(self.rolling_window_ms) > 0.0
            else float("inf")
        )
        parts.extend(
            [
                f"fps={instant_fps:.2f}, rolling_fps={rolling_fps:.2f}, ",
                f"overflow=({format_scalar(primary_overflow)},{format_scalar(shadow_overflow)})",
            ]
        )
        return "".join(parts)

    def _readback_mode(self) -> str:
        payload = self._payload_value
        if self.config.delivery_policy_label == "torch_async":
            return f"torch_async_{payload}"
        return payload

    @property
    def _payload_value(self) -> str:
        if not hasattr(self, "_bound_payload_value"):
            raise RuntimeError("VideoFrameTimingRowBuilder.bind_request() must be called before build_row()")
        return self._bound_payload_value

    def bind_request(self, request: DeliveryRequest) -> "VideoFrameTimingRowBuilder":
        self._bound_payload_value = request.payload.value
        return self


def readback_video_result(
    result,
    readback_mode: str,
    *,
    include_shadow_traversal_stats: bool = False,
) -> dict[str, np.ndarray]:
    if readback_mode == "none":
        return {}
    channels = video_readback_channels(
        readback_mode,
        include_shadow_traversal_stats=include_shadow_traversal_stats,
    )
    if readback_mode in ("rgb", "rgb8"):
        return stage_optical_channels(result, channels, canonical_dtypes=False)
    return stage_optical_compute_result_to_host(result).channels


def video_readback_channels(
    readback_mode: str,
    *,
    include_shadow_traversal_stats: bool = False,
) -> tuple[str, ...]:
    if readback_mode == "rgb8":
        channels = _VIDEO_RGB8_CHANNELS
    else:
        channels = _VIDEO_RGB_CHANNELS
    if include_shadow_traversal_stats:
        channels = channels + _VIDEO_SHADOW_TRAVERSAL_CHANNELS
    return channels


def host_rgb_preview_for_readback(host_channels: Mapping[str, object], readback_mode: str) -> np.ndarray:
    if readback_mode == "rgb8":
        return np.asarray(host_channels["rgb8"], dtype=np.uint8)
    return _rgb_array_to_preview_uint8(np.asarray(host_channels["rgb"]))


def staged_scalar(channels: Mapping[str, object], name: str) -> float | int:
    if name not in channels:
        return _NAN
    return int(np.asarray(channels[name]).reshape(-1)[0])


def staged_shadow_traversal_fields(channels: Mapping[str, object]) -> dict[str, float | int]:
    return {name: staged_scalar(channels, name) for name in _VIDEO_SHADOW_TRAVERSAL_CHANNELS}


def format_ms(value: float) -> str:
    return "NaN" if np.isnan(float(value)) else f"{float(value):.3f}ms"


def format_pack_rgb8(value: float) -> str:
    return "" if np.isnan(float(value)) else f"pack_rgb8={float(value):.3f}ms, "


def format_render_profile(row: Mapping[str, float]) -> str:
    first_hit = float(row["render_first_hit_kernel_ms"])
    shade = float(row["render_shade_kernel_ms"])
    raygen = float(row["render_raygen_kernel_ms"])
    if np.isnan(first_hit) and np.isnan(shade) and np.isnan(raygen):
        return ""
    parts = []
    if not np.isnan(raygen):
        parts.append(f"raygen={raygen:.3f}ms")
    if not np.isnan(first_hit):
        parts.append(f"first_hit={first_hit:.3f}ms")
    if not np.isnan(shade):
        parts.append(f"shade={shade:.3f}ms")
    return f"profile=({', '.join(parts)}), "


def format_ratio(value: float) -> str:
    return "NaN" if np.isnan(float(value)) else f"{float(value):.3f}"


def format_scalar(value: float | int) -> str:
    return "NaN" if isinstance(value, float) and np.isnan(value) else str(int(value))


def _pack_rgb8_result(
    result,
    *,
    pack_rgb8: Callable[[object], object],
    synchronize_event: Callable[[object], None],
) -> tuple[object, float]:
    if not rgb_pack_available():
        raise SystemExit("--video-readback=rgb8 requires warp RGB8 packing") from rgb_pack_import_error()
    pack_start = time.perf_counter()
    packed = pack_rgb8(result)
    synchronize_event(packed.ready_event)
    return packed, (time.perf_counter() - pack_start) * 1000.0


def _rgb_array_to_preview_uint8(rgb: np.ndarray) -> np.ndarray:
    return linear_rgb_to_preview_uint8(np.asarray(rgb, dtype=np.float64))


def _prepare_timing_value(timing: Mapping[str, float], key: str) -> float:
    return float(timing.get(key, _NAN))


def _render_delivery_ms(*, render_execute_ms: float, pack_rgb8_ms: float) -> float:
    if np.isnan(float(pack_rgb8_ms)):
        return float(render_execute_ms)
    return float(render_execute_ms) + float(pack_rgb8_ms)


def _overlap_ratio(*, observed_ms: float, render_ms: float, readback_ms: float) -> float:
    denominator = float(render_ms) + float(readback_ms)
    if denominator <= 0.0 or np.isnan(denominator):
        return _NAN
    return 1.0 - float(observed_ms) / denominator
