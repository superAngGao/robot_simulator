"""Optical scene synchronization and execution contracts."""

from .builder import (
    OpticalBindingBuildResult,
    OpticalBindingDiagnostic,
    OpticalSourceKey,
    build_optical_registry_from_robot_model,
)
from .execution import CpuReferenceOpticalExecutor, OpticalComputeResult, OpticalExecutor
from .registry import (
    OpticalInstanceSpec,
    OpticalLightSpec,
    OpticalMaterialSpec,
    OpticalPlaneGeometry,
    OpticalTriangleMeshGeometry,
    OpticalWorldRegistry,
)
from .scene import OpticalFrameInputs, OpticalInstanceSnapshot, OpticalSceneCache, OpticalSceneSnapshot

__all__ = [
    "CpuReferenceOpticalExecutor",
    "OpticalBindingBuildResult",
    "OpticalBindingDiagnostic",
    "OpticalComputeResult",
    "OpticalExecutor",
    "OpticalFrameInputs",
    "OpticalInstanceSpec",
    "OpticalInstanceSnapshot",
    "OpticalLightSpec",
    "OpticalMaterialSpec",
    "OpticalPlaneGeometry",
    "OpticalSceneCache",
    "OpticalSceneSnapshot",
    "OpticalSourceKey",
    "OpticalTriangleMeshGeometry",
    "OpticalWorldRegistry",
    "build_optical_registry_from_robot_model",
]
