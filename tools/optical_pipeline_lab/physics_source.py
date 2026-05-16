"""Physics-published frame source helpers for the Optical Pipeline Lab."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from physics.publish import ConsumerState, GpuPublishedFrame, QoSMode

from .render_session import (
    OpticalLabRenderFrameContext,
    OpticalLabRenderOptions,
    OpticalLabRenderPipeline,
    OpticalLabRenderSource,
)
from .timing import TimingRecorder


@dataclass(frozen=True)
class PhysicsLabRenderScene:
    """Small scene view for physics-driven lab render sessions."""

    registry: object
    frame: GpuPublishedFrame
    bounds_min: object | None = None
    bounds_max: object | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass
class PhysicsLabFrameLease:
    """Borrowed physics frame prepared for one lab render frame."""

    engine: object
    consumer_id: str
    pipeline: OpticalLabRenderPipeline
    frame: GpuPublishedFrame
    frame_context: OpticalLabRenderFrameContext
    done_event: object | None = None
    _completed: bool = False

    def __enter__(self) -> "PhysicsLabFrameLease":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.complete()

    @property
    def completed(self) -> bool:
        return self._completed

    def complete(self) -> object | None:
        """Mark the borrowed device frame complete on the physics engine.

        `done_event` is populated only after this method runs, including when
        it is called by the context manager's `__exit__`.
        """

        if self._completed:
            return self.done_event
        self.done_event = self.engine.complete_device_consumer(
            self.consumer_id,
            self.frame.frame_id,
            stream=self.pipeline.session.stream,
        )
        self._completed = True
        return self.done_event


def physics_render_consumer(
    consumer_id: str = "optical_lab_physics_render",
    *,
    qos_mode: QoSMode = "lossless",
    max_lag_frames: int | None = None,
) -> ConsumerState:
    """Build the default device-borrow consumer for physics-backed lab renders."""

    return ConsumerState(
        consumer_id=consumer_id,
        consumer_kind="render_backed_sensing",
        qos_mode=qos_mode,
        access_mode="borrow",
        consumer_location="device",
        max_lag_frames=max_lag_frames,
    )


def register_physics_render_consumer(
    engine: object,
    consumer_id: str = "optical_lab_physics_render",
    *,
    consumer: ConsumerState | None = None,
    qos_mode: QoSMode = "lossless",
    max_lag_frames: int | None = None,
) -> ConsumerState:
    """Register a device-borrow physics render consumer on an engine."""

    registered = consumer
    if registered is None:
        registered = physics_render_consumer(
            consumer_id,
            qos_mode=qos_mode,
            max_lag_frames=max_lag_frames,
        )
    engine.register_consumer(registered)
    return registered


def begin_physics_render_frame(
    engine: object,
    pipeline: OpticalLabRenderPipeline,
    *,
    consumer_id: str,
    published_frame: GpuPublishedFrame | None = None,
    env_idx: int = 0,
) -> PhysicsLabFrameLease:
    """Borrow a physics GPU frame and prepare a lab render frame context."""

    source_frame = published_frame if published_frame is not None else engine.latest_published_frame()
    borrowed_frame = engine.borrow_device_frame(
        consumer_id,
        source_frame.frame_id,
        stream=pipeline.session.stream,
    )
    try:
        frame_context = pipeline.begin_frame(frame_inputs=borrowed_frame, env_idx=env_idx)
    except BaseException:
        engine.complete_device_consumer(
            consumer_id,
            borrowed_frame.frame_id,
            stream=pipeline.session.stream,
        )
        raise
    return PhysicsLabFrameLease(
        engine=engine,
        consumer_id=consumer_id,
        pipeline=pipeline,
        frame=borrowed_frame,
        frame_context=frame_context,
    )


def create_physics_render_pipeline(
    *,
    registry: object,
    base_frame: GpuPublishedFrame,
    options: OpticalLabRenderOptions,
    timings: TimingRecorder,
    bounds_min: object | None = None,
    bounds_max: object | None = None,
    scene: object | None = None,
    metadata: Mapping[str, object] | None = None,
) -> OpticalLabRenderPipeline:
    """Create a lab render pipeline from a physics-published base frame.

    When `scene` is omitted, `build_physics_render_source` creates a
    `PhysicsLabRenderScene` view so the session still has a scene object.
    Supplying `scene` overrides that view while keeping the source bundle
    physics-owned.
    """

    return OpticalLabRenderPipeline.create_from_source_factory(
        lambda _workspace: build_physics_render_source(
            registry=registry,
            base_frame=base_frame,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            scene=scene,
            metadata=metadata,
        ),
        options,
        timings,
        scene_for_source=scene_from_physics_render_source,
    )


def build_physics_render_source(
    *,
    registry: object,
    base_frame: GpuPublishedFrame,
    bounds_min: object | None = None,
    bounds_max: object | None = None,
    scene: object | None = None,
    metadata: Mapping[str, object] | None = None,
) -> OpticalLabRenderSource:
    """Wrap a physics-published GPU frame as a lab render source.

    The caller owns the published-frame lifetime. Real-time/lossless callers
    should borrow the frame from the physics publish ring before passing it in.
    When `scene` is omitted, this helper creates a `PhysicsLabRenderScene`
    view and stores it in source metadata for session.scene reconstruction.
    """

    source_metadata = dict(metadata or {})
    if scene is None:
        scene = PhysicsLabRenderScene(
            registry=registry,
            frame=base_frame,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            metadata=source_metadata,
        )
    source_metadata.setdefault("scene", scene)
    source_metadata.setdefault("source_kind", "physics")
    return OpticalLabRenderSource(
        registry=registry,
        base_frame=base_frame,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        metadata=source_metadata,
    )


def scene_from_physics_render_source(source: OpticalLabRenderSource) -> object:
    """Return the scene view stored by `build_physics_render_source`."""

    return source.metadata.get("scene", source)
