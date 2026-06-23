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

_LAZY_EXPORTS = {
    "DebugFrameProduct": (".frame_products", "DebugFrameProduct"),
    "ArtifactOutput": (".runner", "ArtifactOutput"),
    "FrameProduct": (".frame_products", "FrameProduct"),
    "FrameProductResult": (".frame_products", "FrameProductResult"),
    "MultiProductFrameRunner": (".frame_products", "MultiProductFrameRunner"),
    "PhysicsOwnedProductWorkflow": (".product_workflow", "PhysicsOwnedProductWorkflow"),
    "PhysicsProductRunResult": (".product_workflow", "PhysicsProductRunResult"),
    "PublishedStateObservationProduct": (".observation_products", "PublishedStateObservationProduct"),
    "SimulationFrameTick": (".frame_tick", "SimulationFrameTick"),
    "simulation_frame_tick_from_published_frame": (
        ".frame_tick",
        "simulation_frame_tick_from_published_frame",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc

    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    "AccelBackend",
    "AccelPolicy",
    "ArtifactOutput",
    "ClockOwnerKind",
    "DEFAULT_RENDER_HEIGHT",
    "DEFAULT_RENDER_WIDTH",
    "DebugFrameProduct",
    "DeliveryPolicy",
    "FrameProduct",
    "FrameProductResult",
    "FrameTimingRecorder",
    "FrameSourceKind",
    "GeometryMode",
    "MatrixCase",
    "MatrixRunOptions",
    "MatrixSuite",
    "MultiProductFrameRunner",
    "OpticalLabScenarioConfig",
    "OpticalLabScenarioFamily",
    "PhysicsOwnedProductWorkflow",
    "PhysicsProductRunResult",
    "PublishedStateObservationProduct",
    "ReadbackPayload",
    "RenderBackend",
    "SimulationFrameTick",
    "TimingRecorder",
    "WritePolicy",
    "is_physics_published_frame_source",
    "percentile",
    "simulation_frame_tick_from_published_frame",
]
