"""Optical Pipeline Lab foundation.

The lab is developer tooling for optical/render pipeline tuning. It owns
scenario configuration, timing schemas, and report helpers; production optical
runtime APIs should stay in ``optics``.
"""

from .matrix import MatrixCase, MatrixRunOptions, MatrixSuite
from .scenarios import (
    DEFAULT_RENDER_HEIGHT,
    DEFAULT_RENDER_WIDTH,
    AccelBackend,
    AccelPolicy,
    ClockOwnerKind,
    DeliveryPolicy,
    FrameSourceKind,
    GeometryMode,
    OpticalLabScenarioConfig,
    OpticalLabScenarioFamily,
    ReadbackPayload,
    RenderBackend,
    WritePolicy,
    is_physics_published_frame_source,
)
from .timing import FrameTimingRecorder, TimingRecorder, percentile

__all__ = [
    "AccelBackend",
    "AccelPolicy",
    "ClockOwnerKind",
    "DEFAULT_RENDER_HEIGHT",
    "DEFAULT_RENDER_WIDTH",
    "DeliveryPolicy",
    "FrameTimingRecorder",
    "FrameSourceKind",
    "GeometryMode",
    "MatrixCase",
    "MatrixRunOptions",
    "MatrixSuite",
    "OpticalLabScenarioConfig",
    "OpticalLabScenarioFamily",
    "ReadbackPayload",
    "RenderBackend",
    "TimingRecorder",
    "WritePolicy",
    "is_physics_published_frame_source",
    "percentile",
]
