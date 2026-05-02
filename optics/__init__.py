"""Optical scene synchronization and execution contracts."""

from .builder import (
    OpticalBindingBuildResult,
    OpticalBindingDiagnostic,
    OpticalSourceKey,
    build_optical_registry_from_robot_model,
)
from .execution import (
    CpuBvhOpticalExecutor,
    CpuDirectLightOpticalExecutor,
    CpuReferenceOpticalExecutor,
    MissingAccelerationError,
    OpticalComputeResult,
    OpticalExecutor,
)
from .registry import (
    OpticalInstanceSpec,
    OpticalLightSpec,
    OpticalMaterialSpec,
    OpticalPlaneGeometry,
    OpticalTriangleMeshGeometry,
    OpticalWorldRegistry,
)
from .scene import (
    CpuBvhNode,
    OpticalFrameInputs,
    OpticalInstanceSnapshot,
    OpticalSceneAcceleration,
    OpticalSceneCache,
    OpticalSceneSnapshot,
)

__all__ = [
    "CpuBvhNode",
    "CpuBvhOpticalExecutor",
    "CpuDirectLightOpticalExecutor",
    "CpuReferenceOpticalExecutor",
    "MissingAccelerationError",
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
    "OpticalSceneAcceleration",
    "OpticalSceneCache",
    "OpticalSceneSnapshot",
    "OpticalSourceKey",
    "OpticalTriangleMeshGeometry",
    "OpticalWorldRegistry",
    "build_optical_registry_from_robot_model",
]
