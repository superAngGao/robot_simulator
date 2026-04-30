"""Optical scene synchronization and execution contracts."""

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
    "OpticalTriangleMeshGeometry",
    "OpticalWorldRegistry",
]
