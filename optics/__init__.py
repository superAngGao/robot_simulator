"""Optical contracts with source-enumerated, deferred public exports."""

__all__ = [
    "C4_ACTIVE_LEASE_BINDING_SCHEMA_VERSION",
    "C4_ARENA_LIFECYCLE_SCHEMA_VERSION",
    "C4_LIFECYCLE_OPERATION_KINDS",
    "C4_PREPARED_SAMPLE_PLAN_SCHEMA_VERSION",
    "C4_SHUTDOWN_DISPOSITIONS",
    "C4ActiveLeaseBindingEvidence",
    "C4ActiveOutputLease",
    "C4ArenaLifecycleEvidence",
    "C4ArenaSlotLifecycleSnapshot",
    "C4ArenaLifecycleTransition",
    "C4PreparedSamplePlan",
    "C4PreparedSamplePlanEvidence",
    "CUDA_DEVICE_ONLY_DELIVERY_SCHEMA_VERSION",
    "CUDA_NATIVE_ARENA_PLAN_SCHEMA_VERSION",
    "CUDA_NATIVE_ARENA_RESOURCE_SCHEMA_VERSION",
    "CpuBvhNode",
    "CpuBvhOpticalExecutor",
    "CpuDirectLightOpticalExecutor",
    "CpuReferenceOpticalExecutor",
    "CudaDeviceBvhDirectLightOpticalExecutor",
    "CudaDeviceBvhOpticalExecutor",
    "CudaCameraParameters",
    "CudaActiveArenaSample",
    "CudaArenaC3Provenance",
    "CudaArenaLogicalResourceView",
    "CudaArenaPlanEvidence",
    "CudaArenaPlanTarget",
    "CudaArenaResourceRecord",
    "CudaDeferredProfile",
    "CudaDirectLightBuffers",
    "CudaDirectLightNativeRenderer",
    "CudaDirectLightRenderStage",
    "CudaDirectLightStageEvidence",
    "CudaDeviceOnlyDeliveryEvidence",
    "CudaDeviceOnlyDeliveryStage",
    "CudaDeviceResultView",
    "CudaFirstHitInputs",
    "CudaOpticalCompletion",
    "CudaNativeOpticalArena",
    "CudaOpticalLocalScene",
    "CudaOpticalExecutionContext",
    "CudaOpticalTimelineError",
    "CudaOpticalTransaction",
    "CudaOpticalTransactionPoisonedError",
    "CudaOpticalTransactionTemplate",
    "CudaPreparedArenaSample",
    "CUDA_SCENE_UPDATE_OPERATION_ID",
    "CUDA_LBVH_ACCELERATION_ID",
    "CUDA_LBVH_EXTENSION_ABI_VERSION",
    "CUDA_LBVH_MAX_PRIMITIVES",
    "CUDA_LBVH_OPERATION_ID",
    "CUDA_LBVH_STAGE_SCHEMA_VERSION",
    "CUDA_LBVH_TOPOLOGY_LIFETIME",
    "CUDA_LBVH_TOPOLOGY_ORIGIN_ID",
    "CUDA_LBVH_WORKSPACE_SCHEMA_VERSION",
    "CudaLbvhBuffers",
    "CudaLbvhRebuildStage",
    "CudaLbvhRebuildStageEvidence",
    "CudaLbvhRebuildStageFactory",
    "CudaLbvhWorkspace",
    "CudaLbvhWorkspaceRequirements",
    "CudaWorldAabbBuffers",
    "CudaSceneTransformInputs",
    "CudaSceneUpdateStage",
    "CudaSceneUpdateStageEvidence",
    "CudaSceneUpdateStageFactory",
    "CudaWorldSceneBuffers",
    "CudaShadeInputs",
    "DIRECT_LIGHT_FULL_HOST_PER_RAY_CHANNEL_SPECS",
    "DIRECT_LIGHT_FULL_PRIMARY_HOST_CHANNEL_SPECS",
    "DEVICE_FLOAT32_RECOMMENDED_SCENE_SCALE_M",
    "DeviceOpticalRoleTable",
    "DeviceBvhBuildStats",
    "DeviceOpticalBvh",
    "DeviceOpticalScene",
    "DeviceOpticalSceneCache",
    "DeviceOpticalSceneSnapshot",
    "WARP_BVH_REFIT_ACCELERATION_ID",
    "WARP_BVH_TOPOLOGY_ORIGIN_ID",
    "WarpBvhRefitCommandBundle",
    "WarpBvhRefitCommandEvidence",
    "WarpBvhRefitCommandFactory",
    "WarpSceneTransformInputs",
    "WarpSceneUpdateCommandBundle",
    "WarpSceneUpdateCommandEvidence",
    "WarpSceneUpdateCommandFactory",
    "WarpWorldSceneBuffers",
    "GpuBruteForceOpticalExecutor",
    "GpuDeviceBvhDirectLightOpticalExecutor",
    "GpuDeviceBvhOpticalExecutor",
    "GpuDeviceSceneOpticalExecutor",
    "HostOpticalPrimitiveWorkload",
    "HostPayloadContract",
    "HOST_DIAGNOSTIC_OVERLAY_CONTRACTS",
    "HOST_PAYLOAD_CONTRACTS",
    "MAX_PRIMITIVES_PER_INSTANCE",
    "MissingAccelerationError",
    "OpticalBindingBuildResult",
    "OpticalBindingDiagnostic",
    "OpticalComputeResult",
    "OpticalExecutor",
    "OpticalOutputProfile",
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
    "NATIVE_OPTICAL_ARENA_LAYOUT_SCHEMA_VERSION",
    "NATIVE_OPTICAL_ARENA_SPEC_SCHEMA_VERSION",
    "NATIVE_OPTICAL_RESOURCE_MANIFEST_SCHEMA_VERSION",
    "NATIVE_OPTICAL_SLOT_RESOURCE_SCHEMA_VERSION",
    "NativeOpticalArena",
    "NativeOpticalArenaBackpressure",
    "NativeOpticalArenaError",
    "NativeOpticalArenaNotActivatedError",
    "NativeOpticalArenaPoisonedError",
    "NativeOpticalArenaProtocolError",
    "NativeOpticalArenaResourceManifest",
    "NativeOpticalArenaSpec",
    "NativeOpticalPlannedAcquisition",
    "NativeOpticalSlotResourceManifest",
    "PackedOpticalSceneHost",
    "OUTPUT_PROFILE_REQUIRED_CHANNELS",
    "PRIMARY_HOST_SCALAR_CHANNEL_SPECS",
    "SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES",
    "SHADOW_TRAVERSAL_HOST_SCALAR_CHANNEL_SPECS",
    "SCENE_ARRAY_SEMANTICS",
    "SCENE_CONTRACT_SCHEMA_VERSION",
    "SCENE_MATERIALIZER_ID",
    "CanonicalHostChannelSpec",
    "build_host_optical_primitive_workload",
    "build_cuda_dynamic_optical_arena",
    "build_cuda_fixed_optical_arena",
    "allocate_warp_world_scene_buffers",
    "allocate_cuda_direct_light_buffers",
    "allocate_cuda_lbvh_buffers",
    "allocate_cuda_lbvh_workspace",
    "allocate_cuda_world_scene_buffers",
    "cuda_direct_light_available",
    "cuda_execution_stream_token",
    "cuda_lbvh_workspace_requirements",
    "cuda_replay_transform_inputs",
    "channel_is_device",
    "channel_to_numpy",
    "host_diagnostic_overlay_contract_schema",
    "host_payload_contract_schema",
    "channel_to_torch",
    "build_device_bvh_from_snapshot",
    "build_host_median_bvh",
    "build_cuda_lbvh_from_snapshot",
    "build_cuda_optical_scene",
    "build_device_optical_scene",
    "build_device_optical_scene_from_packed",
    "cuda_optical_scene_from_arrays",
    "device_optical_scene_from_arrays",
    "build_optical_registry_from_robot_model",
    "execute_optical_on_gpu_published_frame",
    "pack_source_order_key",
    "plan_native_optical_acquisition",
    "refit_device_bvh_from_snapshot",
    "stage_optical_channels",
    "stage_channels_to_host",
    "stage_optical_compute_result_to_host",
    "update_device_optical_scene_from_gpu_frame",
    "warp_replay_transform_inputs",
]


def __dir__():
    return sorted(set(globals()) | set(__all__))


def __getattr__(name):
    # Static branches, never caller-selected modules or dynamic discovery.
    if name == "OpticalBindingBuildResult":
        from .builder import OpticalBindingBuildResult as value

        globals()[name] = value
        return value
    if name == "OpticalBindingDiagnostic":
        from .builder import OpticalBindingDiagnostic as value

        globals()[name] = value
        return value
    if name == "OpticalSourceKey":
        from .builder import OpticalSourceKey as value

        globals()[name] = value
        return value
    if name == "build_optical_registry_from_robot_model":
        from .builder import build_optical_registry_from_robot_model as value

        globals()[name] = value
        return value
    if name == "CUDA_DEVICE_ONLY_DELIVERY_SCHEMA_VERSION":
        from .cuda_arena import CUDA_DEVICE_ONLY_DELIVERY_SCHEMA_VERSION as value

        globals()[name] = value
        return value
    if name == "CUDA_NATIVE_ARENA_PLAN_SCHEMA_VERSION":
        from .cuda_arena import CUDA_NATIVE_ARENA_PLAN_SCHEMA_VERSION as value

        globals()[name] = value
        return value
    if name == "CUDA_NATIVE_ARENA_RESOURCE_SCHEMA_VERSION":
        from .cuda_arena import CUDA_NATIVE_ARENA_RESOURCE_SCHEMA_VERSION as value

        globals()[name] = value
        return value
    if name == "CudaActiveArenaSample":
        from .cuda_arena import CudaActiveArenaSample as value

        globals()[name] = value
        return value
    if name == "CudaArenaC3Provenance":
        from .cuda_arena import CudaArenaC3Provenance as value

        globals()[name] = value
        return value
    if name == "CudaArenaLogicalResourceView":
        from .cuda_arena import CudaArenaLogicalResourceView as value

        globals()[name] = value
        return value
    if name == "CudaArenaPlanEvidence":
        from .cuda_arena import CudaArenaPlanEvidence as value

        globals()[name] = value
        return value
    if name == "CudaArenaPlanTarget":
        from .cuda_arena import CudaArenaPlanTarget as value

        globals()[name] = value
        return value
    if name == "CudaArenaResourceRecord":
        from .cuda_arena import CudaArenaResourceRecord as value

        globals()[name] = value
        return value
    if name == "CudaDeviceOnlyDeliveryEvidence":
        from .cuda_arena import CudaDeviceOnlyDeliveryEvidence as value

        globals()[name] = value
        return value
    if name == "CudaDeviceOnlyDeliveryStage":
        from .cuda_arena import CudaDeviceOnlyDeliveryStage as value

        globals()[name] = value
        return value
    if name == "CudaDeviceResultView":
        from .cuda_arena import CudaDeviceResultView as value

        globals()[name] = value
        return value
    if name == "CudaNativeOpticalArena":
        from .cuda_arena import CudaNativeOpticalArena as value

        globals()[name] = value
        return value
    if name == "CudaPreparedArenaSample":
        from .cuda_arena import CudaPreparedArenaSample as value

        globals()[name] = value
        return value
    if name == "build_cuda_dynamic_optical_arena":
        from .cuda_arena import build_cuda_dynamic_optical_arena as value

        globals()[name] = value
        return value
    if name == "build_cuda_fixed_optical_arena":
        from .cuda_arena import build_cuda_fixed_optical_arena as value

        globals()[name] = value
        return value
    if name == "cuda_execution_stream_token":
        from .cuda_arena import cuda_execution_stream_token as value

        globals()[name] = value
        return value
    if name == "CudaCameraParameters":
        from .cuda_direct_light import CudaCameraParameters as value

        globals()[name] = value
        return value
    if name == "CudaDeviceBvhDirectLightOpticalExecutor":
        from .cuda_direct_light import CudaDeviceBvhDirectLightOpticalExecutor as value

        globals()[name] = value
        return value
    if name == "CudaDeviceBvhOpticalExecutor":
        from .cuda_direct_light import CudaDeviceBvhOpticalExecutor as value

        globals()[name] = value
        return value
    if name == "CudaDirectLightBuffers":
        from .cuda_direct_light import CudaDirectLightBuffers as value

        globals()[name] = value
        return value
    if name == "CudaDirectLightNativeRenderer":
        from .cuda_direct_light import CudaDirectLightNativeRenderer as value

        globals()[name] = value
        return value
    if name == "CudaDirectLightRenderStage":
        from .cuda_direct_light import CudaDirectLightRenderStage as value

        globals()[name] = value
        return value
    if name == "CudaDirectLightStageEvidence":
        from .cuda_direct_light import CudaDirectLightStageEvidence as value

        globals()[name] = value
        return value
    if name == "CudaFirstHitInputs":
        from .cuda_direct_light import CudaFirstHitInputs as value

        globals()[name] = value
        return value
    if name == "CudaShadeInputs":
        from .cuda_direct_light import CudaShadeInputs as value

        globals()[name] = value
        return value
    if name == "allocate_cuda_direct_light_buffers":
        from .cuda_direct_light import allocate_cuda_direct_light_buffers as value

        globals()[name] = value
        return value
    if name == "cuda_direct_light_available":
        from .cuda_direct_light import cuda_direct_light_available as value

        globals()[name] = value
        return value
    if name == "CUDA_LBVH_ACCELERATION_ID":
        from .cuda_lbvh import CUDA_LBVH_ACCELERATION_ID as value

        globals()[name] = value
        return value
    if name == "CUDA_LBVH_EXTENSION_ABI_VERSION":
        from .cuda_lbvh import CUDA_LBVH_EXTENSION_ABI_VERSION as value

        globals()[name] = value
        return value
    if name == "CUDA_LBVH_MAX_PRIMITIVES":
        from .cuda_lbvh import CUDA_LBVH_MAX_PRIMITIVES as value

        globals()[name] = value
        return value
    if name == "CUDA_LBVH_OPERATION_ID":
        from .cuda_lbvh import CUDA_LBVH_OPERATION_ID as value

        globals()[name] = value
        return value
    if name == "CUDA_LBVH_STAGE_SCHEMA_VERSION":
        from .cuda_lbvh import CUDA_LBVH_STAGE_SCHEMA_VERSION as value

        globals()[name] = value
        return value
    if name == "CUDA_LBVH_TOPOLOGY_LIFETIME":
        from .cuda_lbvh import CUDA_LBVH_TOPOLOGY_LIFETIME as value

        globals()[name] = value
        return value
    if name == "CUDA_LBVH_TOPOLOGY_ORIGIN_ID":
        from .cuda_lbvh import CUDA_LBVH_TOPOLOGY_ORIGIN_ID as value

        globals()[name] = value
        return value
    if name == "CUDA_LBVH_WORKSPACE_SCHEMA_VERSION":
        from .cuda_lbvh import CUDA_LBVH_WORKSPACE_SCHEMA_VERSION as value

        globals()[name] = value
        return value
    if name == "CudaLbvhBuffers":
        from .cuda_lbvh import CudaLbvhBuffers as value

        globals()[name] = value
        return value
    if name == "CudaLbvhRebuildStage":
        from .cuda_lbvh import CudaLbvhRebuildStage as value

        globals()[name] = value
        return value
    if name == "CudaLbvhRebuildStageEvidence":
        from .cuda_lbvh import CudaLbvhRebuildStageEvidence as value

        globals()[name] = value
        return value
    if name == "CudaLbvhRebuildStageFactory":
        from .cuda_lbvh import CudaLbvhRebuildStageFactory as value

        globals()[name] = value
        return value
    if name == "CudaLbvhWorkspace":
        from .cuda_lbvh import CudaLbvhWorkspace as value

        globals()[name] = value
        return value
    if name == "CudaLbvhWorkspaceRequirements":
        from .cuda_lbvh import CudaLbvhWorkspaceRequirements as value

        globals()[name] = value
        return value
    if name == "CudaWorldAabbBuffers":
        from .cuda_lbvh import CudaWorldAabbBuffers as value

        globals()[name] = value
        return value
    if name == "allocate_cuda_lbvh_buffers":
        from .cuda_lbvh import allocate_cuda_lbvh_buffers as value

        globals()[name] = value
        return value
    if name == "allocate_cuda_lbvh_workspace":
        from .cuda_lbvh import allocate_cuda_lbvh_workspace as value

        globals()[name] = value
        return value
    if name == "build_cuda_lbvh_from_snapshot":
        from .cuda_lbvh import build_cuda_lbvh_from_snapshot as value

        globals()[name] = value
        return value
    if name == "cuda_lbvh_workspace_requirements":
        from .cuda_lbvh import cuda_lbvh_workspace_requirements as value

        globals()[name] = value
        return value
    if name == "CUDA_SCENE_UPDATE_OPERATION_ID":
        from .cuda_scene import CUDA_SCENE_UPDATE_OPERATION_ID as value

        globals()[name] = value
        return value
    if name == "CudaOpticalLocalScene":
        from .cuda_scene import CudaOpticalLocalScene as value

        globals()[name] = value
        return value
    if name == "CudaSceneTransformInputs":
        from .cuda_scene import CudaSceneTransformInputs as value

        globals()[name] = value
        return value
    if name == "CudaSceneUpdateStage":
        from .cuda_scene import CudaSceneUpdateStage as value

        globals()[name] = value
        return value
    if name == "CudaSceneUpdateStageEvidence":
        from .cuda_scene import CudaSceneUpdateStageEvidence as value

        globals()[name] = value
        return value
    if name == "CudaSceneUpdateStageFactory":
        from .cuda_scene import CudaSceneUpdateStageFactory as value

        globals()[name] = value
        return value
    if name == "CudaWorldSceneBuffers":
        from .cuda_scene import CudaWorldSceneBuffers as value

        globals()[name] = value
        return value
    if name == "allocate_cuda_world_scene_buffers":
        from .cuda_scene import allocate_cuda_world_scene_buffers as value

        globals()[name] = value
        return value
    if name == "build_cuda_optical_scene":
        from .cuda_scene import build_cuda_optical_scene as value

        globals()[name] = value
        return value
    if name == "cuda_optical_scene_from_arrays":
        from .cuda_scene import cuda_optical_scene_from_arrays as value

        globals()[name] = value
        return value
    if name == "cuda_replay_transform_inputs":
        from .cuda_scene import cuda_replay_transform_inputs as value

        globals()[name] = value
        return value
    if name == "CudaDeferredProfile":
        from .cuda_timeline import CudaDeferredProfile as value

        globals()[name] = value
        return value
    if name == "CudaOpticalCompletion":
        from .cuda_timeline import CudaOpticalCompletion as value

        globals()[name] = value
        return value
    if name == "CudaOpticalExecutionContext":
        from .cuda_timeline import CudaOpticalExecutionContext as value

        globals()[name] = value
        return value
    if name == "CudaOpticalTimelineError":
        from .cuda_timeline import CudaOpticalTimelineError as value

        globals()[name] = value
        return value
    if name == "CudaOpticalTransaction":
        from .cuda_timeline import CudaOpticalTransaction as value

        globals()[name] = value
        return value
    if name == "CudaOpticalTransactionPoisonedError":
        from .cuda_timeline import CudaOpticalTransactionPoisonedError as value

        globals()[name] = value
        return value
    if name == "CudaOpticalTransactionTemplate":
        from .cuda_timeline import CudaOpticalTransactionTemplate as value

        globals()[name] = value
        return value
    if name == "DEVICE_FLOAT32_RECOMMENDED_SCENE_SCALE_M":
        from .device import DEVICE_FLOAT32_RECOMMENDED_SCENE_SCALE_M as value

        globals()[name] = value
        return value
    if name == "MAX_PRIMITIVES_PER_INSTANCE":
        from .device import MAX_PRIMITIVES_PER_INSTANCE as value

        globals()[name] = value
        return value
    if name == "HostOpticalPrimitiveWorkload":
        from .device import HostOpticalPrimitiveWorkload as value

        globals()[name] = value
        return value
    if name == "build_host_optical_primitive_workload":
        from .device import build_host_optical_primitive_workload as value

        globals()[name] = value
        return value
    if name == "pack_source_order_key":
        from .device import pack_source_order_key as value

        globals()[name] = value
        return value
    if name == "stage_optical_channels":
        from .device import stage_optical_channels as value

        globals()[name] = value
        return value
    if name == "stage_optical_compute_result_to_host":
        from .device import stage_optical_compute_result_to_host as value

        globals()[name] = value
        return value
    if name == "WARP_BVH_REFIT_ACCELERATION_ID":
        from .device_bvh import WARP_BVH_REFIT_ACCELERATION_ID as value

        globals()[name] = value
        return value
    if name == "WARP_BVH_TOPOLOGY_ORIGIN_ID":
        from .device_bvh import WARP_BVH_TOPOLOGY_ORIGIN_ID as value

        globals()[name] = value
        return value
    if name == "DeviceBvhBuildStats":
        from .host_bvh import DeviceBvhBuildStats as value

        globals()[name] = value
        return value
    if name == "DeviceOpticalBvh":
        from .device_bvh import DeviceOpticalBvh as value

        globals()[name] = value
        return value
    if name == "WarpBvhRefitCommandBundle":
        from .device_bvh import WarpBvhRefitCommandBundle as value

        globals()[name] = value
        return value
    if name == "WarpBvhRefitCommandEvidence":
        from .device_bvh import WarpBvhRefitCommandEvidence as value

        globals()[name] = value
        return value
    if name == "WarpBvhRefitCommandFactory":
        from .device_bvh import WarpBvhRefitCommandFactory as value

        globals()[name] = value
        return value
    if name == "build_device_bvh_from_snapshot":
        from .device_bvh import build_device_bvh_from_snapshot as value

        globals()[name] = value
        return value
    if name == "build_host_median_bvh":
        from .host_bvh import build_host_median_bvh as value

        globals()[name] = value
        return value
    if name == "refit_device_bvh_from_snapshot":
        from .device_bvh import refit_device_bvh_from_snapshot as value

        globals()[name] = value
        return value
    if name == "channel_is_device":
        from .device_channel import channel_is_device as value

        globals()[name] = value
        return value
    if name == "channel_to_numpy":
        from .device_channel import channel_to_numpy as value

        globals()[name] = value
        return value
    if name == "channel_to_torch":
        from .device_channel import channel_to_torch as value

        globals()[name] = value
        return value
    if name == "stage_channels_to_host":
        from .device_channel import stage_channels_to_host as value

        globals()[name] = value
        return value
    if name == "DeviceOpticalRoleTable":
        from .device_scene import DeviceOpticalRoleTable as value

        globals()[name] = value
        return value
    if name == "DeviceOpticalScene":
        from .device_scene import DeviceOpticalScene as value

        globals()[name] = value
        return value
    if name == "DeviceOpticalSceneCache":
        from .device_scene import DeviceOpticalSceneCache as value

        globals()[name] = value
        return value
    if name == "DeviceOpticalSceneSnapshot":
        from .device_scene import DeviceOpticalSceneSnapshot as value

        globals()[name] = value
        return value
    if name == "WarpSceneTransformInputs":
        from .device_scene import WarpSceneTransformInputs as value

        globals()[name] = value
        return value
    if name == "WarpSceneUpdateCommandBundle":
        from .device_scene import WarpSceneUpdateCommandBundle as value

        globals()[name] = value
        return value
    if name == "WarpSceneUpdateCommandEvidence":
        from .device_scene import WarpSceneUpdateCommandEvidence as value

        globals()[name] = value
        return value
    if name == "WarpSceneUpdateCommandFactory":
        from .device_scene import WarpSceneUpdateCommandFactory as value

        globals()[name] = value
        return value
    if name == "WarpWorldSceneBuffers":
        from .device_scene import WarpWorldSceneBuffers as value

        globals()[name] = value
        return value
    if name == "allocate_warp_world_scene_buffers":
        from .device_scene import allocate_warp_world_scene_buffers as value

        globals()[name] = value
        return value
    if name == "build_device_optical_scene":
        from .device_scene import build_device_optical_scene as value

        globals()[name] = value
        return value
    if name == "build_device_optical_scene_from_packed":
        from .device_scene import build_device_optical_scene_from_packed as value

        globals()[name] = value
        return value
    if name == "device_optical_scene_from_arrays":
        from .device_scene import device_optical_scene_from_arrays as value

        globals()[name] = value
        return value
    if name == "update_device_optical_scene_from_gpu_frame":
        from .device_scene import update_device_optical_scene_from_gpu_frame as value

        globals()[name] = value
        return value
    if name == "warp_replay_transform_inputs":
        from .device_scene import warp_replay_transform_inputs as value

        globals()[name] = value
        return value
    if name == "CpuBvhOpticalExecutor":
        from .execution import CpuBvhOpticalExecutor as value

        globals()[name] = value
        return value
    if name == "CpuDirectLightOpticalExecutor":
        from .execution import CpuDirectLightOpticalExecutor as value

        globals()[name] = value
        return value
    if name == "CpuReferenceOpticalExecutor":
        from .execution import CpuReferenceOpticalExecutor as value

        globals()[name] = value
        return value
    if name == "MissingAccelerationError":
        from .execution import MissingAccelerationError as value

        globals()[name] = value
        return value
    if name == "OpticalComputeResult":
        from .execution import OpticalComputeResult as value

        globals()[name] = value
        return value
    if name == "OpticalExecutor":
        from .execution import OpticalExecutor as value

        globals()[name] = value
        return value
    if name == "OpticalOutputProfile":
        from .execution import OpticalOutputProfile as value

        globals()[name] = value
        return value
    if name == "execute_optical_on_gpu_published_frame":
        from .gpu_runtime import execute_optical_on_gpu_published_frame as value

        globals()[name] = value
        return value
    if name == "C4_ACTIVE_LEASE_BINDING_SCHEMA_VERSION":
        from .native_arena import C4_ACTIVE_LEASE_BINDING_SCHEMA_VERSION as value

        globals()[name] = value
        return value
    if name == "C4_ARENA_LIFECYCLE_SCHEMA_VERSION":
        from .native_arena import C4_ARENA_LIFECYCLE_SCHEMA_VERSION as value

        globals()[name] = value
        return value
    if name == "C4_LIFECYCLE_OPERATION_KINDS":
        from .native_arena import C4_LIFECYCLE_OPERATION_KINDS as value

        globals()[name] = value
        return value
    if name == "C4_PREPARED_SAMPLE_PLAN_SCHEMA_VERSION":
        from .native_arena import C4_PREPARED_SAMPLE_PLAN_SCHEMA_VERSION as value

        globals()[name] = value
        return value
    if name == "C4_SHUTDOWN_DISPOSITIONS":
        from .native_arena import C4_SHUTDOWN_DISPOSITIONS as value

        globals()[name] = value
        return value
    if name == "NATIVE_OPTICAL_ARENA_LAYOUT_SCHEMA_VERSION":
        from .native_arena import NATIVE_OPTICAL_ARENA_LAYOUT_SCHEMA_VERSION as value

        globals()[name] = value
        return value
    if name == "NATIVE_OPTICAL_ARENA_SPEC_SCHEMA_VERSION":
        from .native_arena import NATIVE_OPTICAL_ARENA_SPEC_SCHEMA_VERSION as value

        globals()[name] = value
        return value
    if name == "NATIVE_OPTICAL_RESOURCE_MANIFEST_SCHEMA_VERSION":
        from .native_arena import NATIVE_OPTICAL_RESOURCE_MANIFEST_SCHEMA_VERSION as value

        globals()[name] = value
        return value
    if name == "NATIVE_OPTICAL_SLOT_RESOURCE_SCHEMA_VERSION":
        from .native_arena import NATIVE_OPTICAL_SLOT_RESOURCE_SCHEMA_VERSION as value

        globals()[name] = value
        return value
    if name == "C4ActiveLeaseBindingEvidence":
        from .native_arena import C4ActiveLeaseBindingEvidence as value

        globals()[name] = value
        return value
    if name == "C4ActiveOutputLease":
        from .native_arena import C4ActiveOutputLease as value

        globals()[name] = value
        return value
    if name == "C4ArenaLifecycleEvidence":
        from .native_arena import C4ArenaLifecycleEvidence as value

        globals()[name] = value
        return value
    if name == "C4ArenaLifecycleTransition":
        from .native_arena import C4ArenaLifecycleTransition as value

        globals()[name] = value
        return value
    if name == "C4ArenaSlotLifecycleSnapshot":
        from .native_arena import C4ArenaSlotLifecycleSnapshot as value

        globals()[name] = value
        return value
    if name == "C4PreparedSamplePlan":
        from .native_arena import C4PreparedSamplePlan as value

        globals()[name] = value
        return value
    if name == "C4PreparedSamplePlanEvidence":
        from .native_arena import C4PreparedSamplePlanEvidence as value

        globals()[name] = value
        return value
    if name == "NativeOpticalArena":
        from .native_arena import NativeOpticalArena as value

        globals()[name] = value
        return value
    if name == "NativeOpticalArenaBackpressure":
        from .native_arena import NativeOpticalArenaBackpressure as value

        globals()[name] = value
        return value
    if name == "NativeOpticalArenaError":
        from .native_arena import NativeOpticalArenaError as value

        globals()[name] = value
        return value
    if name == "NativeOpticalArenaNotActivatedError":
        from .native_arena import NativeOpticalArenaNotActivatedError as value

        globals()[name] = value
        return value
    if name == "NativeOpticalArenaPoisonedError":
        from .native_arena import NativeOpticalArenaPoisonedError as value

        globals()[name] = value
        return value
    if name == "NativeOpticalArenaProtocolError":
        from .native_arena import NativeOpticalArenaProtocolError as value

        globals()[name] = value
        return value
    if name == "NativeOpticalArenaResourceManifest":
        from .native_arena import NativeOpticalArenaResourceManifest as value

        globals()[name] = value
        return value
    if name == "NativeOpticalArenaSpec":
        from .native_arena import NativeOpticalArenaSpec as value

        globals()[name] = value
        return value
    if name == "NativeOpticalPlannedAcquisition":
        from .native_arena import NativeOpticalPlannedAcquisition as value

        globals()[name] = value
        return value
    if name == "NativeOpticalSlotResourceManifest":
        from .native_arena import NativeOpticalSlotResourceManifest as value

        globals()[name] = value
        return value
    if name == "plan_native_optical_acquisition":
        from .native_arena import plan_native_optical_acquisition as value

        globals()[name] = value
        return value
    if name == "DIRECT_LIGHT_FULL_HOST_PER_RAY_CHANNEL_SPECS":
        from .output_contract import DIRECT_LIGHT_FULL_HOST_PER_RAY_CHANNEL_SPECS as value

        globals()[name] = value
        return value
    if name == "DIRECT_LIGHT_FULL_PRIMARY_HOST_CHANNEL_SPECS":
        from .output_contract import DIRECT_LIGHT_FULL_PRIMARY_HOST_CHANNEL_SPECS as value

        globals()[name] = value
        return value
    if name == "HOST_DIAGNOSTIC_OVERLAY_CONTRACTS":
        from .output_contract import HOST_DIAGNOSTIC_OVERLAY_CONTRACTS as value

        globals()[name] = value
        return value
    if name == "HOST_PAYLOAD_CONTRACTS":
        from .output_contract import HOST_PAYLOAD_CONTRACTS as value

        globals()[name] = value
        return value
    if name == "OUTPUT_PROFILE_REQUIRED_CHANNELS":
        from .output_contract import OUTPUT_PROFILE_REQUIRED_CHANNELS as value

        globals()[name] = value
        return value
    if name == "PRIMARY_HOST_SCALAR_CHANNEL_SPECS":
        from .output_contract import PRIMARY_HOST_SCALAR_CHANNEL_SPECS as value

        globals()[name] = value
        return value
    if name == "SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES":
        from .output_contract import SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES as value

        globals()[name] = value
        return value
    if name == "SHADOW_TRAVERSAL_HOST_SCALAR_CHANNEL_SPECS":
        from .output_contract import SHADOW_TRAVERSAL_HOST_SCALAR_CHANNEL_SPECS as value

        globals()[name] = value
        return value
    if name == "CanonicalHostChannelSpec":
        from .output_contract import CanonicalHostChannelSpec as value

        globals()[name] = value
        return value
    if name == "HostPayloadContract":
        from .output_contract import HostPayloadContract as value

        globals()[name] = value
        return value
    if name == "host_diagnostic_overlay_contract_schema":
        from .output_contract import host_diagnostic_overlay_contract_schema as value

        globals()[name] = value
        return value
    if name == "host_payload_contract_schema":
        from .output_contract import host_payload_contract_schema as value

        globals()[name] = value
        return value
    if name == "OpticalInstanceSpec":
        from .registry import OpticalInstanceSpec as value

        globals()[name] = value
        return value
    if name == "OpticalLightSpec":
        from .registry import OpticalLightSpec as value

        globals()[name] = value
        return value
    if name == "OpticalMaterialSpec":
        from .registry import OpticalMaterialSpec as value

        globals()[name] = value
        return value
    if name == "OpticalPlaneGeometry":
        from .registry import OpticalPlaneGeometry as value

        globals()[name] = value
        return value
    if name == "OpticalTriangleMeshGeometry":
        from .registry import OpticalTriangleMeshGeometry as value

        globals()[name] = value
        return value
    if name == "OpticalWorldRegistry":
        from .registry import OpticalWorldRegistry as value

        globals()[name] = value
        return value
    if name == "CpuBvhNode":
        from .scene import CpuBvhNode as value

        globals()[name] = value
        return value
    if name == "OpticalFrameInputs":
        from .scene import OpticalFrameInputs as value

        globals()[name] = value
        return value
    if name == "OpticalInstanceSnapshot":
        from .scene import OpticalInstanceSnapshot as value

        globals()[name] = value
        return value
    if name == "OpticalSceneAcceleration":
        from .scene import OpticalSceneAcceleration as value

        globals()[name] = value
        return value
    if name == "OpticalSceneCache":
        from .scene import OpticalSceneCache as value

        globals()[name] = value
        return value
    if name == "OpticalSceneSnapshot":
        from .scene import OpticalSceneSnapshot as value

        globals()[name] = value
        return value
    if name == "SCENE_ARRAY_SEMANTICS":
        from .scene_contract import SCENE_ARRAY_SEMANTICS as value

        globals()[name] = value
        return value
    if name == "SCENE_CONTRACT_SCHEMA_VERSION":
        from .scene_contract import SCENE_CONTRACT_SCHEMA_VERSION as value

        globals()[name] = value
        return value
    if name == "SCENE_MATERIALIZER_ID":
        from .scene_contract import SCENE_MATERIALIZER_ID as value

        globals()[name] = value
        return value
    if name == "PackedOpticalSceneHost":
        from .scene_contract import PackedOpticalSceneHost as value

        globals()[name] = value
        return value
    if name == "GpuBruteForceOpticalExecutor":
        from .warp_execution import GpuBruteForceOpticalExecutor as value

        globals()[name] = value
        return value
    if name == "GpuDeviceBvhDirectLightOpticalExecutor":
        from .warp_execution import GpuDeviceBvhDirectLightOpticalExecutor as value

        globals()[name] = value
        return value
    if name == "GpuDeviceBvhOpticalExecutor":
        from .warp_execution import GpuDeviceBvhOpticalExecutor as value

        globals()[name] = value
        return value
    if name == "GpuDeviceSceneOpticalExecutor":
        from .warp_execution import GpuDeviceSceneOpticalExecutor as value

        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
