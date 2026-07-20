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
    "DebugProductSpec": (".product_specs", "DebugProductSpec"),
    "ObservationProductSpec": (".product_specs", "ObservationProductSpec"),
    "VideoProductSpec": (".product_specs", "VideoProductSpec"),
    "PhysicsOwnedProductWorkflow": (".product_workflow", "PhysicsOwnedProductWorkflow"),
    "ProductBuildContext": (".product_specs", "ProductBuildContext"),
    "ProductSpec": (".product_specs", "ProductSpec"),
    "PhysicsProductRunResult": (".product_workflow", "PhysicsProductRunResult"),
    "PublishedStateObservationProduct": (".observation_products", "PublishedStateObservationProduct"),
    "resolve_lab_product_specs": (".preset_products", "resolve_lab_product_specs"),
    "SimulationFrameTick": (".frame_tick", "SimulationFrameTick"),
    "create_runtime_for_lab_preset": (".preset_runtime", "create_runtime_for_lab_preset"),
    "supported_lab_product_strings": (".preset_products", "supported_lab_product_strings"),
    "supported_runtime_presets": (".preset_runtime", "supported_runtime_presets"),
    "run_physics_product_preset": (".product_workflow", "run_physics_product_preset"),
    "run_physics_product_scenario": (
        ".product_workflow",
        "run_physics_product_scenario",
    ),
    "run_physics_products": (".product_workflow", "run_physics_products"),
    "run_optical_lab_preset": (".preset_workflows", "run_optical_lab_preset"),
    "run_optical_lab_products": (".product_workflow", "run_optical_lab_products"),
    "run_optical_lab_workflow": (".product_workflow", "run_optical_lab_workflow"),
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
    "DebugProductSpec",
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
    "ObservationProductSpec",
    "PhysicsOwnedProductWorkflow",
    "PhysicsProductRunResult",
    "ProductBuildContext",
    "ProductSpec",
    "PublishedStateObservationProduct",
    "ReadbackPayload",
    "RenderBackend",
    "resolve_lab_product_specs",
    "SimulationFrameTick",
    "TimingRecorder",
    "VideoProductSpec",
    "WritePolicy",
    "create_runtime_for_lab_preset",
    "is_physics_published_frame_source",
    "percentile",
    "run_optical_lab_preset",
    "run_optical_lab_products",
    "run_optical_lab_workflow",
    "run_physics_product_preset",
    "run_physics_product_scenario",
    "run_physics_products",
    "simulation_frame_tick_from_published_frame",
    "supported_lab_product_strings",
    "supported_runtime_presets",
]
