"""Optical scene synchronization and execution contracts."""

from .builder import (
    OpticalBindingBuildResult,
    OpticalBindingDiagnostic,
    OpticalSourceKey,
    build_optical_registry_from_robot_model,
)
from .device import (
    DEVICE_FLOAT32_RECOMMENDED_SCENE_SCALE_M,
    MAX_PRIMITIVES_PER_INSTANCE,
    HostOpticalPrimitiveWorkload,
    build_host_optical_primitive_workload,
    pack_source_order_key,
    stage_optical_compute_result_to_host,
)
from .device_bvh import (
    DeviceBvhBuildStats,
    DeviceOpticalBvh,
    build_device_bvh_from_snapshot,
)
from .device_scene import (
    DeviceOpticalRoleTable,
    DeviceOpticalScene,
    DeviceOpticalSceneCache,
    DeviceOpticalSceneSnapshot,
    build_device_optical_scene,
    update_device_optical_scene_from_gpu_frame,
)
from .execution import (
    CpuBvhOpticalExecutor,
    CpuDirectLightOpticalExecutor,
    CpuReferenceOpticalExecutor,
    MissingAccelerationError,
    OpticalComputeResult,
    OpticalExecutor,
)
from .gpu_runtime import execute_optical_on_gpu_published_frame
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
from .warp_execution import (
    GpuBruteForceOpticalExecutor,
    GpuDeviceBvhOpticalExecutor,
    GpuDeviceSceneOpticalExecutor,
)

__all__ = [
    "CpuBvhNode",
    "CpuBvhOpticalExecutor",
    "CpuDirectLightOpticalExecutor",
    "CpuReferenceOpticalExecutor",
    "DEVICE_FLOAT32_RECOMMENDED_SCENE_SCALE_M",
    "DeviceOpticalRoleTable",
    "DeviceBvhBuildStats",
    "DeviceOpticalBvh",
    "DeviceOpticalScene",
    "DeviceOpticalSceneCache",
    "DeviceOpticalSceneSnapshot",
    "GpuBruteForceOpticalExecutor",
    "GpuDeviceBvhOpticalExecutor",
    "GpuDeviceSceneOpticalExecutor",
    "HostOpticalPrimitiveWorkload",
    "MAX_PRIMITIVES_PER_INSTANCE",
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
    "build_host_optical_primitive_workload",
    "build_device_bvh_from_snapshot",
    "build_device_optical_scene",
    "build_optical_registry_from_robot_model",
    "execute_optical_on_gpu_published_frame",
    "pack_source_order_key",
    "stage_optical_compute_result_to_host",
    "update_device_optical_scene_from_gpu_frame",
]
