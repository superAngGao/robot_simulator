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
    DeliveryPolicy,
    GeometryMode,
    OpticalLabScenarioConfig,
    OpticalLabScenarioFamily,
    ReadbackPayload,
    RenderBackend,
    WritePolicy,
)
from .timing import FrameTimingRecorder, TimingRecorder, percentile

__all__ = [
    "AccelBackend",
    "AccelPolicy",
    "DEFAULT_RENDER_HEIGHT",
    "DEFAULT_RENDER_WIDTH",
    "DeliveryPolicy",
    "FrameTimingRecorder",
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
    "percentile",
]
