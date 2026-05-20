"""Narrow frame workflow runner for Optical Pipeline Lab experiments."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from .delivery import DeliveredVideoFrame, RenderedVideoFrame, VideoDeliveryFacade
from .render_session import OpticalLabRenderFrameContext

FrameVideoConsumer = Callable[
    [OpticalLabRenderFrameContext, int],
    RenderedVideoFrame | None,
]
DeliveredVideoRecorder = Callable[[DeliveredVideoFrame], None]


@dataclass(frozen=True)
class FrameWorkflowResult:
    """Typed result for one lab frame workflow step."""

    frame_index: int
    video: RenderedVideoFrame | None = None
    delivered_video: tuple[DeliveredVideoFrame, ...] = field(default_factory=tuple)


@dataclass
class FrameWorkflowRunner:
    """Coordinate provider lifecycle, video consumption, and delivery.

    This is intentionally narrower than the future SimulationFrameRuntime. It is
    a lab-internal runner for one video-focused workflow product.
    """

    frame_provider: object
    video_consumer: FrameVideoConsumer
    delivery: VideoDeliveryFacade | None = None
    delivered_video_recorder: DeliveredVideoRecorder | None = None

    def step(
        self,
        frame_index: int,
        *,
        env_idx: int = 0,
        provider_kwargs: Mapping[str, object] | None = None,
    ) -> FrameWorkflowResult:
        """Run one provider-backed frame workflow step."""

        frame_start = time.perf_counter()
        with self.frame_provider.begin_frame(
            frame_index,
            env_idx=env_idx,
            **dict(provider_kwargs or {}),
        ) as frame_context:
            video = self.video_consumer(frame_context, frame_index)

        delivered_video: tuple[DeliveredVideoFrame, ...] = ()
        if video is not None and self.delivery is not None:
            delivered_video = self._submit_video(video, frame_start=frame_start)
        return FrameWorkflowResult(
            frame_index=frame_index,
            video=video,
            delivered_video=delivered_video,
        )

    def flush(self) -> tuple[DeliveredVideoFrame, ...]:
        """Flush pending delivery products."""

        if self.delivery is None:
            return ()
        delivered = tuple(self.delivery.flush())
        self._record_delivered_video(delivered)
        return delivered

    def _submit_video(
        self,
        video: RenderedVideoFrame,
        *,
        frame_start: float,
    ) -> tuple[DeliveredVideoFrame, ...]:
        if self.delivery is None:
            return ()
        delivered: list[DeliveredVideoFrame] = []
        delivered.extend(self.delivery.complete_available(latest_rendered_frame_index=video.frame_index))
        completed = self.delivery.submit(video, frame_start=frame_start)
        if completed is not None:
            delivered.append(completed)
        delivered.extend(self.delivery.complete_available(latest_rendered_frame_index=video.frame_index))
        delivered_tuple = tuple(delivered)
        self._record_delivered_video(delivered_tuple)
        return delivered_tuple

    def _record_delivered_video(self, delivered: tuple[DeliveredVideoFrame, ...]) -> None:
        if self.delivered_video_recorder is None:
            return
        for frame in delivered:
            self.delivered_video_recorder(frame)
