"""CUDA direct-light optical executors."""

from __future__ import annotations

import time
from functools import lru_cache

import numpy as np

from sensing.optical import OpticalPinholeCameraSpec, OpticalRaySensorSpec

from .device_bvh import DeviceOpticalBvh
from .device_channel import channel_to_torch
from .device_scene import DeviceOpticalSceneSnapshot
from .execution import OpticalComputeResult, OpticalOutputProfile, normalize_output_profile

try:  # pragma: no cover - exercised in CUDA extension environments.
    import torch
    import warp as wp
    from torch.utils.cpp_extension import load_inline

    _HAS_CUDA_DIRECT_LIGHT_DEPS = True
except Exception:  # pragma: no cover - keeps CPU-only imports working.
    torch = None
    wp = None
    load_inline = None
    _HAS_CUDA_DIRECT_LIGHT_DEPS = False

_DIR_EPS = 1.0e-8
_T_EPS = 1.0e-5
_MAX_BVH_STACK = 32


class CudaDeviceBvhOpticalExecutor:
    """CUDA first-hit executor over a device scene plus CUDA LBVH."""

    capabilities = frozenset(
        {
            "range_m",
            "hit_mask",
            "position_world",
            "normal_world",
            "numeric_instance_id",
            "material_index",
        }
    )
    supported_profiles = frozenset({OpticalOutputProfile.GEOMETRY_FULL})

    def __init__(self, *, device=None, stream=None) -> None:
        _require_cuda_direct_light_deps()
        wp.init()
        self.device = wp.get_device("cuda:0" if device is None else device)
        self.stream = stream

    def execute(
        self,
        snapshot: DeviceOpticalSceneSnapshot,
        bvh: DeviceOpticalBvh,
        spec: OpticalRaySensorSpec,
        *,
        output_profile: OpticalOutputProfile | str = OpticalOutputProfile.GEOMETRY_FULL,
        render_profile: list[tuple[str, float]] | None = None,
    ) -> OpticalComputeResult:
        """Execute CUDA LBVH first-hit traversal for host-provided rays."""

        output_profile = normalize_output_profile(output_profile)
        if output_profile is not OpticalOutputProfile.GEOMETRY_FULL:
            raise ValueError("CudaDeviceBvhOpticalExecutor supports only output_profile='geometry_full'")
        self._validate(snapshot, bvh, spec)
        _synchronize_ready_event(snapshot.ready_event)
        _synchronize_ready_event(bvh.ready_event)

        start = time.perf_counter()
        module = _load_cuda_direct_light_extension()
        _add_profile_ms(render_profile, "cuda_first_hit_load_extension_ms", start)

        tensor_device = _torch_device_for_warp_device(self.device)
        upload_start = time.perf_counter()
        origins = torch.as_tensor(
            np.asarray(spec.origins_world, dtype=np.float32),
            dtype=torch.float32,
            device=tensor_device,
        ).contiguous()
        directions = torch.as_tensor(
            np.asarray(spec.directions_world, dtype=np.float32),
            dtype=torch.float32,
            device=tensor_device,
        ).contiguous()
        _add_profile_ms(render_profile, "cuda_first_hit_ray_upload_ms", upload_start)

        convert_start = time.perf_counter()
        scene = snapshot.scene
        tensors = _FirstHitTensors(
            plane_normal_world=channel_to_torch(snapshot.plane_normal_world),
            plane_point_world=channel_to_torch(snapshot.plane_point_world),
            plane_numeric_instance_id=channel_to_torch(scene.plane_numeric_instance_id),
            plane_source_order_key=channel_to_torch(scene.plane_source_order_key),
            plane_role_mask=channel_to_torch(scene.plane_role_mask),
            plane_material_index=channel_to_torch(scene.plane_material_index),
            triangle_v0_world=channel_to_torch(snapshot.triangle_v0_world),
            triangle_e1_world=channel_to_torch(snapshot.triangle_e1_world),
            triangle_e2_world=channel_to_torch(snapshot.triangle_e2_world),
            triangle_normal_world=channel_to_torch(snapshot.triangle_normal_world),
            triangle_numeric_instance_id=channel_to_torch(scene.triangle_numeric_instance_id),
            triangle_source_order_key=channel_to_torch(scene.triangle_source_order_key),
            triangle_role_mask=channel_to_torch(scene.triangle_role_mask),
            triangle_material_index=channel_to_torch(scene.triangle_material_index),
            bvh_bounds_min=channel_to_torch(bvh.bounds_min),
            bvh_bounds_max=channel_to_torch(bvh.bounds_max),
            bvh_left=channel_to_torch(bvh.left),
            bvh_right=channel_to_torch(bvh.right),
            bvh_start=channel_to_torch(bvh.start),
            bvh_count=channel_to_torch(bvh.count),
            bvh_prim_ids=channel_to_torch(bvh.prim_ids),
        )
        _add_profile_ms(render_profile, "cuda_first_hit_channel_views_ms", convert_start)

        execute_start = time.perf_counter()
        (
            hit_mask,
            range_m,
            position_world,
            normal_world,
            numeric_instance_id,
            material_index,
            bvh_stack_overflow_count,
            bvh_max_stack_depth,
        ) = module.first_hit_rays(
            origins,
            directions,
            float(spec.max_distance),
            tensors.plane_normal_world,
            tensors.plane_point_world,
            tensors.plane_numeric_instance_id,
            tensors.plane_source_order_key,
            tensors.plane_role_mask,
            tensors.plane_material_index,
            int(scene.num_planes),
            tensors.triangle_v0_world,
            tensors.triangle_e1_world,
            tensors.triangle_e2_world,
            tensors.triangle_normal_world,
            tensors.triangle_numeric_instance_id,
            tensors.triangle_source_order_key,
            tensors.triangle_role_mask,
            tensors.triangle_material_index,
            tensors.bvh_bounds_min,
            tensors.bvh_bounds_max,
            tensors.bvh_left,
            tensors.bvh_right,
            tensors.bvh_start,
            tensors.bvh_count,
            tensors.bvh_prim_ids,
            int(bvh.num_nodes),
            int(scene.role_table.mask_for(spec.sensor_role)),
        )
        torch.cuda.synchronize(tensor_device)
        _add_profile_ms(render_profile, "cuda_first_hit_kernel_sync_ms", execute_start)

        return OpticalComputeResult(
            frame_id=spec.frame_id,
            sim_time=spec.sim_time,
            env_idx=spec.env_idx,
            sensor_id=spec.sensor_id,
            location="device",
            channels={
                "hit_mask": hit_mask,
                "range_m": range_m,
                "position_world": position_world,
                "normal_world": normal_world,
                "numeric_instance_id": numeric_instance_id,
                "material_index": material_index,
                "bvh_stack_overflow_count": bvh_stack_overflow_count,
                "bvh_max_stack_depth": bvh_max_stack_depth,
            },
            output_profile=OpticalOutputProfile.GEOMETRY_FULL,
            ready_event=None,
            resources=(
                origins,
                directions,
                hit_mask,
                range_m,
                position_world,
                normal_world,
                numeric_instance_id,
                material_index,
                bvh_stack_overflow_count,
                bvh_max_stack_depth,
            ),
        )

    def execute_camera(
        self,
        snapshot: DeviceOpticalSceneSnapshot,
        bvh: DeviceOpticalBvh,
        camera: OpticalPinholeCameraSpec,
        *,
        output_profile: OpticalOutputProfile | str = OpticalOutputProfile.GEOMETRY_FULL,
        render_profile: list[tuple[str, float]] | None = None,
    ) -> OpticalComputeResult:
        """Execute CUDA camera raygen once P12.3e lands."""

        raise NotImplementedError("cuda_direct_light camera raygen is pending P12.3e")

    def _validate(
        self,
        snapshot: DeviceOpticalSceneSnapshot,
        bvh: DeviceOpticalBvh,
        spec: OpticalRaySensorSpec,
    ) -> None:
        if snapshot.frame_id != spec.frame_id:
            raise ValueError("snapshot.frame_id must match spec.frame_id")
        if snapshot.env_idx != spec.env_idx:
            raise ValueError("snapshot.env_idx must match spec.env_idx")
        if snapshot.scene.device != bvh.device:
            raise ValueError("DeviceOpticalSceneSnapshot and DeviceOpticalBvh devices must match")
        if snapshot.scene.device != self.device:
            raise ValueError("DeviceOpticalSceneSnapshot device must match executor device")
        if bvh.device != self.device:
            raise ValueError("DeviceOpticalBvh device must match executor device")
        if bvh.frame_id != snapshot.frame_id:
            raise ValueError("DeviceOpticalBvh frame_id must match DeviceOpticalSceneSnapshot frame_id")
        if bvh.env_idx != snapshot.env_idx:
            raise ValueError("DeviceOpticalBvh env_idx must match DeviceOpticalSceneSnapshot env_idx")
        if not np.isfinite(spec.max_distance):
            raise ValueError("CudaDeviceBvhOpticalExecutor requires finite max_distance")


class CudaDeviceBvhDirectLightOpticalExecutor:
    """CUDA direct-light executor over a device scene plus CUDA LBVH."""

    capabilities = CudaDeviceBvhOpticalExecutor.capabilities | frozenset({"rgb", "intensity"})
    supported_profiles = frozenset(
        {
            OpticalOutputProfile.DIRECT_LIGHT_FULL,
            OpticalOutputProfile.RGB_PREVIEW,
            OpticalOutputProfile.RENDER_ONLY,
        }
    )

    def __init__(
        self,
        *,
        device=None,
        stream=None,
        shadows: bool = True,
        ambient_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
        background_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
        shadow_bias: float = 1.0e-6,
    ) -> None:
        _require_cuda_direct_light_deps()
        wp.init()
        self.device = wp.get_device("cuda:0" if device is None else device)
        self.stream = stream
        self.shadows = bool(shadows)
        self.ambient_rgb = tuple(float(component) for component in ambient_rgb)
        self.background_rgb = tuple(float(component) for component in background_rgb)
        self.shadow_bias = float(shadow_bias)
        if len(self.ambient_rgb) != 3:
            raise ValueError("ambient_rgb must contain 3 components")
        if len(self.background_rgb) != 3:
            raise ValueError("background_rgb must contain 3 components")
        if self.shadow_bias < 0.0:
            raise ValueError("shadow_bias must be >= 0")
        self._first_hit = CudaDeviceBvhOpticalExecutor(device=self.device, stream=stream)

    def execute(
        self,
        snapshot: DeviceOpticalSceneSnapshot,
        bvh: DeviceOpticalBvh,
        spec: OpticalRaySensorSpec,
        *,
        output_profile: OpticalOutputProfile | str = OpticalOutputProfile.DIRECT_LIGHT_FULL,
        render_profile: list[tuple[str, float]] | None = None,
    ):
        """Execute host-ray CUDA direct-light rendering."""

        output_profile = self._validate_output_profile(output_profile)
        geometry = self._first_hit.execute(
            snapshot,
            bvh,
            spec,
            output_profile=OpticalOutputProfile.GEOMETRY_FULL,
            render_profile=render_profile,
        )
        if self.shadows:
            return self._shade_geometry_with_shadows(
                snapshot,
                bvh,
                geometry,
                sensor_role=spec.sensor_role,
                output_profile=output_profile,
                render_profile=render_profile,
            )
        return self._shade_geometry_no_shadow(
            snapshot,
            geometry,
            output_profile=output_profile,
            render_profile=render_profile,
        )

    def execute_camera(
        self,
        snapshot: DeviceOpticalSceneSnapshot,
        bvh: DeviceOpticalBvh,
        camera: OpticalPinholeCameraSpec,
        *,
        output_profile: OpticalOutputProfile | str = OpticalOutputProfile.DIRECT_LIGHT_FULL,
        render_profile: list[tuple[str, float]] | None = None,
    ):
        """Execute camera-ray CUDA direct-light rendering once P12.3e lands."""

        self._validate_output_profile(output_profile)
        raise NotImplementedError("cuda_direct_light camera raygen is pending P12.3e")

    def _validate_output_profile(self, output_profile: OpticalOutputProfile | str) -> OpticalOutputProfile:
        output_profile = normalize_output_profile(output_profile)
        if output_profile not in self.supported_profiles:
            raise ValueError(f"cuda_direct_light does not support output_profile={output_profile.value!r}")
        return output_profile

    def _shade_geometry_no_shadow(
        self,
        snapshot: DeviceOpticalSceneSnapshot,
        geometry: OpticalComputeResult,
        *,
        output_profile: OpticalOutputProfile,
        render_profile: list[tuple[str, float]] | None = None,
    ) -> OpticalComputeResult:
        scene = snapshot.scene
        start = time.perf_counter()
        module = _load_cuda_direct_light_extension()
        _add_profile_ms(render_profile, "cuda_shade_load_extension_ms", start)

        convert_start = time.perf_counter()
        hit_mask = channel_to_torch(geometry.channels["hit_mask"])
        position_world = channel_to_torch(geometry.channels["position_world"])
        normal_world = channel_to_torch(geometry.channels["normal_world"])
        material_index = channel_to_torch(geometry.channels["material_index"])
        material_albedo_rgb = channel_to_torch(scene.material_albedo_rgb)
        light_kind = channel_to_torch(scene.light_kind)
        light_position_or_direction_world = channel_to_torch(scene.light_position_or_direction_world)
        light_intensity = channel_to_torch(scene.light_intensity)
        light_color_rgb = channel_to_torch(scene.light_color_rgb)
        _add_profile_ms(render_profile, "cuda_shade_channel_views_ms", convert_start)

        execute_start = time.perf_counter()
        (
            rgb,
            intensity,
            shadow_stack_overflow_count,
            shadow_max_stack_depth,
        ) = module.shade_direct_light_no_shadow(
            hit_mask,
            position_world,
            normal_world,
            material_index,
            material_albedo_rgb,
            light_kind,
            light_position_or_direction_world,
            light_intensity,
            light_color_rgb,
            int(scene.num_lights),
            float(self.ambient_rgb[0]),
            float(self.ambient_rgb[1]),
            float(self.ambient_rgb[2]),
            float(self.background_rgb[0]),
            float(self.background_rgb[1]),
            float(self.background_rgb[2]),
        )
        tensor_device = _torch_device_for_warp_device(self.device)
        torch.cuda.synchronize(tensor_device)
        _add_profile_ms(render_profile, "cuda_shade_kernel_sync_ms", execute_start)

        channels = dict(geometry.channels)
        channels["rgb"] = rgb
        channels["intensity"] = intensity
        channels["shadow_stack_overflow_count"] = shadow_stack_overflow_count
        channels["shadow_max_stack_depth"] = shadow_max_stack_depth
        channels = _filter_channels_for_profile(channels, output_profile)
        return OpticalComputeResult(
            frame_id=geometry.frame_id,
            sim_time=geometry.sim_time,
            env_idx=geometry.env_idx,
            sensor_id=geometry.sensor_id,
            location="device",
            channels=channels,
            output_profile=output_profile,
            ready_event=None,
            resources=geometry.resources
            + (
                material_albedo_rgb,
                light_kind,
                light_position_or_direction_world,
                light_intensity,
                light_color_rgb,
                rgb,
                intensity,
                shadow_stack_overflow_count,
                shadow_max_stack_depth,
            ),
        )

    def _shade_geometry_with_shadows(
        self,
        snapshot: DeviceOpticalSceneSnapshot,
        bvh: DeviceOpticalBvh,
        geometry: OpticalComputeResult,
        *,
        sensor_role: str,
        output_profile: OpticalOutputProfile,
        render_profile: list[tuple[str, float]] | None = None,
    ) -> OpticalComputeResult:
        scene = snapshot.scene
        start = time.perf_counter()
        module = _load_cuda_direct_light_extension()
        _add_profile_ms(render_profile, "cuda_shade_load_extension_ms", start)

        convert_start = time.perf_counter()
        hit_mask = channel_to_torch(geometry.channels["hit_mask"])
        position_world = channel_to_torch(geometry.channels["position_world"])
        normal_world = channel_to_torch(geometry.channels["normal_world"])
        material_index = channel_to_torch(geometry.channels["material_index"])
        material_albedo_rgb = channel_to_torch(scene.material_albedo_rgb)
        light_kind = channel_to_torch(scene.light_kind)
        light_position_or_direction_world = channel_to_torch(scene.light_position_or_direction_world)
        light_intensity = channel_to_torch(scene.light_intensity)
        light_color_rgb = channel_to_torch(scene.light_color_rgb)
        triangle_v0_world = channel_to_torch(snapshot.triangle_v0_world)
        triangle_e1_world = channel_to_torch(snapshot.triangle_e1_world)
        triangle_e2_world = channel_to_torch(snapshot.triangle_e2_world)
        triangle_normal_world = channel_to_torch(snapshot.triangle_normal_world)
        triangle_role_mask = channel_to_torch(scene.triangle_role_mask)
        bvh_bounds_min = channel_to_torch(bvh.bounds_min)
        bvh_bounds_max = channel_to_torch(bvh.bounds_max)
        bvh_left = channel_to_torch(bvh.left)
        bvh_right = channel_to_torch(bvh.right)
        bvh_start = channel_to_torch(bvh.start)
        bvh_count = channel_to_torch(bvh.count)
        bvh_prim_ids = channel_to_torch(bvh.prim_ids)
        _add_profile_ms(render_profile, "cuda_shade_channel_views_ms", convert_start)

        execute_start = time.perf_counter()
        (
            rgb,
            intensity,
            shadow_stack_overflow_count,
            shadow_max_stack_depth,
        ) = module.shade_direct_light_with_shadows(
            hit_mask,
            position_world,
            normal_world,
            material_index,
            material_albedo_rgb,
            light_kind,
            light_position_or_direction_world,
            light_intensity,
            light_color_rgb,
            int(scene.num_lights),
            triangle_v0_world,
            triangle_e1_world,
            triangle_e2_world,
            triangle_normal_world,
            triangle_role_mask,
            bvh_bounds_min,
            bvh_bounds_max,
            bvh_left,
            bvh_right,
            bvh_start,
            bvh_count,
            bvh_prim_ids,
            int(bvh.num_nodes),
            int(scene.role_table.mask_for(sensor_role)),
            float(self.shadow_bias),
            float(self.ambient_rgb[0]),
            float(self.ambient_rgb[1]),
            float(self.ambient_rgb[2]),
            float(self.background_rgb[0]),
            float(self.background_rgb[1]),
            float(self.background_rgb[2]),
        )
        tensor_device = _torch_device_for_warp_device(self.device)
        torch.cuda.synchronize(tensor_device)
        _add_profile_ms(render_profile, "cuda_shade_kernel_sync_ms", execute_start)

        channels = dict(geometry.channels)
        channels["rgb"] = rgb
        channels["intensity"] = intensity
        channels["shadow_stack_overflow_count"] = shadow_stack_overflow_count
        channels["shadow_max_stack_depth"] = shadow_max_stack_depth
        channels = _filter_channels_for_profile(channels, output_profile)
        return OpticalComputeResult(
            frame_id=geometry.frame_id,
            sim_time=geometry.sim_time,
            env_idx=geometry.env_idx,
            sensor_id=geometry.sensor_id,
            location="device",
            channels=channels,
            output_profile=output_profile,
            ready_event=None,
            resources=geometry.resources
            + (
                material_albedo_rgb,
                light_kind,
                light_position_or_direction_world,
                light_intensity,
                light_color_rgb,
                triangle_v0_world,
                triangle_e1_world,
                triangle_e2_world,
                triangle_normal_world,
                triangle_role_mask,
                bvh_bounds_min,
                bvh_bounds_max,
                bvh_left,
                bvh_right,
                bvh_start,
                bvh_count,
                bvh_prim_ids,
                rgb,
                intensity,
                shadow_stack_overflow_count,
                shadow_max_stack_depth,
            ),
        )


class _FirstHitTensors:
    def __init__(self, **values):
        self.__dict__.update(values)


def cuda_direct_light_available() -> bool:
    """Return whether CUDA direct-light dependencies are importable and CUDA is available."""

    return _HAS_CUDA_DIRECT_LIGHT_DEPS and bool(torch.cuda.is_available())


def _require_cuda_direct_light_deps() -> None:
    if not _HAS_CUDA_DIRECT_LIGHT_DEPS:
        raise ImportError("CUDA direct-light executor requires torch, warp, and torch CUDA extension tooling")
    if not torch.cuda.is_available():
        raise ImportError("CUDA direct-light executor requires torch CUDA availability")


def _synchronize_ready_event(event: object | None) -> None:
    if event is not None:
        wp.synchronize_event(event)


def _torch_device_for_warp_device(device) -> "torch.device":
    return torch.device(str(device))


def _add_profile_ms(render_profile: list[tuple[str, float]] | None, name: str, start: float) -> None:
    if render_profile is not None:
        render_profile.append((name, (time.perf_counter() - start) * 1000.0))


def _filter_channels_for_profile(
    channels: dict[str, object],
    output_profile: OpticalOutputProfile,
) -> dict[str, object]:
    if output_profile is OpticalOutputProfile.DIRECT_LIGHT_FULL:
        return channels
    guaranteed = output_profile.guaranteed_channels
    return {name: value for name, value in channels.items() if name in guaranteed}


@lru_cache(maxsize=1)
def _load_cuda_direct_light_extension():
    _require_cuda_direct_light_deps()
    return load_inline(
        name="robot_sim_cuda_direct_light_v3",
        cpp_sources=[_CPP_SOURCE],
        cuda_sources=[_CUDA_SOURCE],
        functions=["first_hit_rays", "shade_direct_light_no_shadow", "shade_direct_light_with_shadows"],
        with_cuda=True,
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        verbose=False,
    )


_CPP_SOURCE = r"""
#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> first_hit_rays(
    torch::Tensor origins,
    torch::Tensor directions,
    double max_distance,
    torch::Tensor plane_normals,
    torch::Tensor plane_points,
    torch::Tensor plane_numeric_ids,
    torch::Tensor plane_source_keys,
    torch::Tensor plane_role_masks,
    torch::Tensor plane_material_indices,
    int64_t num_planes,
    torch::Tensor triangle_v0,
    torch::Tensor triangle_e1,
    torch::Tensor triangle_e2,
    torch::Tensor triangle_normal,
    torch::Tensor triangle_numeric_ids,
    torch::Tensor triangle_source_keys,
    torch::Tensor triangle_role_masks,
    torch::Tensor triangle_material_indices,
    torch::Tensor bvh_bounds_min,
    torch::Tensor bvh_bounds_max,
    torch::Tensor bvh_left,
    torch::Tensor bvh_right,
    torch::Tensor bvh_start,
    torch::Tensor bvh_count,
    torch::Tensor bvh_prim_ids,
    int64_t num_bvh_nodes,
    int64_t sensor_role_mask);

std::vector<torch::Tensor> shade_direct_light_no_shadow(
    torch::Tensor hit_mask,
    torch::Tensor position_world,
    torch::Tensor normal_world,
    torch::Tensor material_index,
    torch::Tensor material_albedo_rgb,
    torch::Tensor light_kind,
    torch::Tensor light_position_or_direction_world,
    torch::Tensor light_intensity,
    torch::Tensor light_color_rgb,
    int64_t num_lights,
    double ambient_r,
    double ambient_g,
    double ambient_b,
    double background_r,
    double background_g,
    double background_b);

std::vector<torch::Tensor> shade_direct_light_with_shadows(
    torch::Tensor hit_mask,
    torch::Tensor position_world,
    torch::Tensor normal_world,
    torch::Tensor material_index,
    torch::Tensor material_albedo_rgb,
    torch::Tensor light_kind,
    torch::Tensor light_position_or_direction_world,
    torch::Tensor light_intensity,
    torch::Tensor light_color_rgb,
    int64_t num_lights,
    torch::Tensor triangle_v0,
    torch::Tensor triangle_e1,
    torch::Tensor triangle_e2,
    torch::Tensor triangle_normal,
    torch::Tensor triangle_role_masks,
    torch::Tensor bvh_bounds_min,
    torch::Tensor bvh_bounds_max,
    torch::Tensor bvh_left,
    torch::Tensor bvh_right,
    torch::Tensor bvh_start,
    torch::Tensor bvh_count,
    torch::Tensor bvh_prim_ids,
    int64_t num_bvh_nodes,
    int64_t sensor_role_mask,
    double shadow_bias,
    double ambient_r,
    double ambient_g,
    double ambient_b,
    double background_r,
    double background_g,
    double background_b);
"""


_CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr float kBuildEps = 1.0e-8f;
constexpr float kDirEps = 1.0e-8f;
constexpr float kTEps = 1.0e-5f;
constexpr int kMaxBvhStack = 32;

__device__ __forceinline__ bool better_hit(float t, int64_t key, float best_t, int64_t best_key) {
  if (t < best_t - kTEps) {
    return true;
  }
  return fabsf(t - best_t) <= kTEps && key < best_key;
}

__device__ __forceinline__ bool intersect_aabb_axis(
    float origin,
    float direction,
    float lower,
    float upper,
    float* t_min,
    float* t_max) {
  if (fabsf(direction) <= kDirEps) {
    return origin >= lower && origin <= upper;
  }
  float inv_d = 1.0f / direction;
  float t0 = (lower - origin) * inv_d;
  float t1 = (upper - origin) * inv_d;
  if (t0 > t1) {
    float tmp = t0;
    t0 = t1;
    t1 = tmp;
  }
  *t_min = fmaxf(*t_min, t0);
  *t_max = fminf(*t_max, t1);
  return *t_min <= *t_max;
}

__device__ __forceinline__ bool intersect_aabb(
    float ox,
    float oy,
    float oz,
    float dx,
    float dy,
    float dz,
    const float* bounds_min,
    const float* bounds_max,
    int node,
    float max_distance,
    float* entry_t) {
  float t_min = 0.0f;
  float t_max = max_distance;
  bool hit = intersect_aabb_axis(ox, dx, bounds_min[node * 3 + 0], bounds_max[node * 3 + 0], &t_min, &t_max);
  if (hit) {
    hit = intersect_aabb_axis(oy, dy, bounds_min[node * 3 + 1], bounds_max[node * 3 + 1], &t_min, &t_max);
  }
  if (hit) {
    hit = intersect_aabb_axis(oz, dz, bounds_min[node * 3 + 2], bounds_max[node * 3 + 2], &t_min, &t_max);
  }
  *entry_t = t_min;
  return hit;
}

__device__ __forceinline__ bool intersect_triangle(
    float ox,
    float oy,
    float oz,
    float dx,
    float dy,
    float dz,
    int tri,
    const float* triangle_v0,
    const float* triangle_e1,
    const float* triangle_e2,
    const float* triangle_normal,
    float max_t,
    float* out_t,
    float* out_nx,
    float* out_ny,
    float* out_nz) {
  float v0x = triangle_v0[tri * 3 + 0];
  float v0y = triangle_v0[tri * 3 + 1];
  float v0z = triangle_v0[tri * 3 + 2];
  float e1x = triangle_e1[tri * 3 + 0];
  float e1y = triangle_e1[tri * 3 + 1];
  float e1z = triangle_e1[tri * 3 + 2];
  float e2x = triangle_e2[tri * 3 + 0];
  float e2y = triangle_e2[tri * 3 + 1];
  float e2z = triangle_e2[tri * 3 + 2];

  float pvec_x = dy * e2z - dz * e2y;
  float pvec_y = dz * e2x - dx * e2z;
  float pvec_z = dx * e2y - dy * e2x;
  float det = pvec_x * e1x + pvec_y * e1y + pvec_z * e1z;
  if (fabsf(det) <= kBuildEps) {
    return false;
  }
  float inv_det = 1.0f / det;
  float tvec_x = ox - v0x;
  float tvec_y = oy - v0y;
  float tvec_z = oz - v0z;
  float u = (pvec_x * tvec_x + pvec_y * tvec_y + pvec_z * tvec_z) * inv_det;
  if (u < 0.0f) {
    return false;
  }
  float qvec_x = tvec_y * e1z - tvec_z * e1y;
  float qvec_y = tvec_z * e1x - tvec_x * e1z;
  float qvec_z = tvec_x * e1y - tvec_y * e1x;
  float v = (qvec_x * dx + qvec_y * dy + qvec_z * dz) * inv_det;
  if (v < 0.0f || u + v > 1.0f) {
    return false;
  }
  float t = (qvec_x * e2x + qvec_y * e2y + qvec_z * e2z) * inv_det;
  if (t < 0.0f || t > max_t) {
    return false;
  }
  float nx = triangle_normal[tri * 3 + 0];
  float ny = triangle_normal[tri * 3 + 1];
  float nz = triangle_normal[tri * 3 + 2];
  float ndotd = nx * dx + ny * dy + nz * dz;
  if (ndotd > 0.0f) {
    nx = -nx;
    ny = -ny;
    nz = -nz;
  }
  *out_t = t;
  *out_nx = nx;
  *out_ny = ny;
  *out_nz = nz;
  return true;
}

__global__ void first_hit_rays_kernel(
    const float* origins,
    const float* directions,
    float max_distance,
    const float* plane_normals,
    const float* plane_points,
    const int32_t* plane_numeric_ids,
    const int64_t* plane_source_keys,
    const int64_t* plane_role_masks,
    const int32_t* plane_material_indices,
    int num_planes,
    const float* triangle_v0,
    const float* triangle_e1,
    const float* triangle_e2,
    const float* triangle_normal,
    const int32_t* triangle_numeric_ids,
    const int64_t* triangle_source_keys,
    const int64_t* triangle_role_masks,
    const int32_t* triangle_material_indices,
    const float* bvh_bounds_min,
    const float* bvh_bounds_max,
    const int32_t* bvh_left,
    const int32_t* bvh_right,
    const int32_t* bvh_start,
    const int32_t* bvh_count,
    const int32_t* bvh_prim_ids,
    int num_bvh_nodes,
    int64_t sensor_role_mask,
    int32_t* hit_mask,
    float* range_m,
    float* position_world,
    float* normal_world,
    int32_t* numeric_instance_id,
    int32_t* material_index,
    int32_t* bvh_stack_overflow_count,
    int32_t* bvh_max_stack_depth,
    int num_rays) {
  int ray = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray >= num_rays) {
    return;
  }

  float ox = origins[ray * 3 + 0];
  float oy = origins[ray * 3 + 1];
  float oz = origins[ray * 3 + 2];
  float dx = directions[ray * 3 + 0];
  float dy = directions[ray * 3 + 1];
  float dz = directions[ray * 3 + 2];

  float best_t = max_distance;
  int64_t best_key = std::numeric_limits<int64_t>::max();
  float best_px = 0.0f;
  float best_py = 0.0f;
  float best_pz = 0.0f;
  float best_nx = 0.0f;
  float best_ny = 0.0f;
  float best_nz = 0.0f;
  int32_t best_id = 0;
  int32_t best_material_index = 0;
  bool found = false;

  for (int plane = 0; plane < num_planes; ++plane) {
    if ((plane_role_masks[plane] & sensor_role_mask) == 0) {
      continue;
    }
    float nx = plane_normals[plane * 3 + 0];
    float ny = plane_normals[plane * 3 + 1];
    float nz = plane_normals[plane * 3 + 2];
    float px = plane_points[plane * 3 + 0];
    float py = plane_points[plane * 3 + 1];
    float pz = plane_points[plane * 3 + 2];
    float denom = dx * nx + dy * ny + dz * nz;
    if (fabsf(denom) <= kDirEps) {
      continue;
    }
    float numer = (px - ox) * nx + (py - oy) * ny + (pz - oz) * nz;
    float t = numer / denom;
    if (t < 0.0f || t > max_distance) {
      continue;
    }
    int64_t key = plane_source_keys[plane];
    if (better_hit(t, key, best_t, best_key)) {
      best_t = t;
      best_key = key;
      best_px = ox + dx * t;
      best_py = oy + dy * t;
      best_pz = oz + dz * t;
      best_nx = nx;
      best_ny = ny;
      best_nz = nz;
      best_id = plane_numeric_ids[plane];
      best_material_index = plane_material_indices[plane];
      found = true;
    }
  }

  int stack[kMaxBvhStack];
  float stack_t[kMaxBvhStack];
  int stack_size = 0;
  int local_max_stack = 0;
  if (num_bvh_nodes > 0) {
    float root_t = 0.0f;
    if (intersect_aabb(
            ox, oy, oz, dx, dy, dz, bvh_bounds_min, bvh_bounds_max, 0, best_t, &root_t)) {
      stack[0] = 0;
      stack_t[0] = root_t;
      stack_size = 1;
      local_max_stack = 1;
    }
  }

  while (stack_size > 0) {
    --stack_size;
    int node = stack[stack_size];
    float node_t = stack_t[stack_size];
    if (node_t > best_t) {
      continue;
    }
    int leaf_count = bvh_count[node];
    if (leaf_count > 0) {
      int leaf_start = bvh_start[node];
      for (int offset = 0; offset < leaf_count; ++offset) {
        int tri = bvh_prim_ids[leaf_start + offset];
        if ((triangle_role_masks[tri] & sensor_role_mask) == 0) {
          continue;
        }
        float t = 0.0f;
        float nx = 0.0f;
        float ny = 0.0f;
        float nz = 0.0f;
        if (!intersect_triangle(
                ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                tri,
                triangle_v0,
                triangle_e1,
                triangle_e2,
                triangle_normal,
                best_t,
                &t,
                &nx,
                &ny,
                &nz)) {
          continue;
        }
        int64_t key = triangle_source_keys[tri];
        if (better_hit(t, key, best_t, best_key)) {
          best_t = t;
          best_key = key;
          best_px = ox + dx * t;
          best_py = oy + dy * t;
          best_pz = oz + dz * t;
          best_nx = nx;
          best_ny = ny;
          best_nz = nz;
          best_id = triangle_numeric_ids[tri];
          best_material_index = triangle_material_indices[tri];
          found = true;
        }
      }
    } else {
      int left = bvh_left[node];
      int right = bvh_right[node];
      bool left_hit = false;
      bool right_hit = false;
      float left_t = 0.0f;
      float right_t = 0.0f;
      if (left >= 0) {
        left_hit = intersect_aabb(
            ox, oy, oz, dx, dy, dz, bvh_bounds_min, bvh_bounds_max, left, best_t, &left_t);
      }
      if (right >= 0) {
        right_hit = intersect_aabb(
            ox, oy, oz, dx, dy, dz, bvh_bounds_min, bvh_bounds_max, right, best_t, &right_t);
      }

      int first = left;
      int second = right;
      bool first_hit = left_hit;
      bool second_hit = right_hit;
      float first_t = left_t;
      float second_t = right_t;
      if (right_hit && (!left_hit || right_t < left_t)) {
        first = right;
        second = left;
        first_hit = right_hit;
        second_hit = left_hit;
        first_t = right_t;
        second_t = left_t;
      }

      if (second_hit) {
        if (stack_size < kMaxBvhStack) {
          stack[stack_size] = second;
          stack_t[stack_size] = second_t;
          ++stack_size;
        } else {
          atomicAdd(bvh_stack_overflow_count, 1);
        }
      }
      if (first_hit) {
        if (stack_size < kMaxBvhStack) {
          stack[stack_size] = first;
          stack_t[stack_size] = first_t;
          ++stack_size;
        } else {
          atomicAdd(bvh_stack_overflow_count, 1);
        }
      }
      if (stack_size > local_max_stack) {
        local_max_stack = stack_size;
      }
    }
  }

  atomicMax(bvh_max_stack_depth, local_max_stack);

  if (found) {
    hit_mask[ray] = 1;
    range_m[ray] = best_t;
    position_world[ray * 3 + 0] = best_px;
    position_world[ray * 3 + 1] = best_py;
    position_world[ray * 3 + 2] = best_pz;
    normal_world[ray * 3 + 0] = best_nx;
    normal_world[ray * 3 + 1] = best_ny;
    normal_world[ray * 3 + 2] = best_nz;
    numeric_instance_id[ray] = best_id;
    material_index[ray] = best_material_index;
  }
}

__global__ void shade_direct_light_no_shadow_kernel(
    const int32_t* hit_mask,
    const float* position_world,
    const float* normal_world,
    const int32_t* material_index,
    const float* material_albedo_rgb,
    const int32_t* light_kind,
    const float* light_position_or_direction_world,
    const float* light_intensity,
    const float* light_color_rgb,
    int num_lights,
    float ambient_r,
    float ambient_g,
    float ambient_b,
    float background_r,
    float background_g,
    float background_b,
    float* rgb,
    float* intensity,
    int num_rays) {
  int ray = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray >= num_rays) {
    return;
  }

  float out_r = background_r;
  float out_g = background_g;
  float out_b = background_b;
  if (hit_mask[ray] != 0) {
    int mat = material_index[ray];
    float albedo_r = material_albedo_rgb[mat * 3 + 0];
    float albedo_g = material_albedo_rgb[mat * 3 + 1];
    float albedo_b = material_albedo_rgb[mat * 3 + 2];
    out_r = ambient_r * albedo_r;
    out_g = ambient_g * albedo_g;
    out_b = ambient_b * albedo_b;

    float px = position_world[ray * 3 + 0];
    float py = position_world[ray * 3 + 1];
    float pz = position_world[ray * 3 + 2];
    float nx = normal_world[ray * 3 + 0];
    float ny = normal_world[ray * 3 + 1];
    float nz = normal_world[ray * 3 + 2];

    for (int light = 0; light < num_lights; ++light) {
      float lx = light_position_or_direction_world[light * 3 + 0];
      float ly = light_position_or_direction_world[light * 3 + 1];
      float lz = light_position_or_direction_world[light * 3 + 2];
      float attenuation = 1.0f;
      if (light_kind[light] == 1) {
        lx = lx - px;
        ly = ly - py;
        lz = lz - pz;
        float distance_sq = lx * lx + ly * ly + lz * lz;
        float distance = sqrtf(distance_sq);
        if (distance > kDirEps) {
          float inv_distance = 1.0f / distance;
          lx = lx * inv_distance;
          ly = ly * inv_distance;
          lz = lz * inv_distance;
          attenuation = 1.0f / fmaxf(distance_sq, 1.0e-12f);
        } else {
          attenuation = 0.0f;
        }
      } else {
        float light_norm = sqrtf(lx * lx + ly * ly + lz * lz);
        if (light_norm > kDirEps) {
          float inv_light_norm = 1.0f / light_norm;
          lx = lx * inv_light_norm;
          ly = ly * inv_light_norm;
          lz = lz * inv_light_norm;
        } else {
          attenuation = 0.0f;
        }
      }

      float n_dot_l = nx * lx + ny * ly + nz * lz;
      if (n_dot_l > 0.0f && attenuation > 0.0f) {
        float scale = light_intensity[light] * attenuation * n_dot_l;
        out_r += albedo_r * light_color_rgb[light * 3 + 0] * scale;
        out_g += albedo_g * light_color_rgb[light * 3 + 1] * scale;
        out_b += albedo_b * light_color_rgb[light * 3 + 2] * scale;
      }
    }
  }

  rgb[ray * 3 + 0] = out_r;
  rgb[ray * 3 + 1] = out_g;
  rgb[ray * 3 + 2] = out_b;
  if (hit_mask[ray] != 0) {
    intensity[ray] = out_r * 0.2126f + out_g * 0.7152f + out_b * 0.0722f;
  } else {
    intensity[ray] = 0.0f;
  }
}

__device__ bool shadow_occluded_by_triangles(
    float ox,
    float oy,
    float oz,
    float dx,
    float dy,
    float dz,
    float max_distance,
    const float* triangle_v0,
    const float* triangle_e1,
    const float* triangle_e2,
    const float* triangle_normal,
    const int64_t* triangle_role_masks,
    const float* bvh_bounds_min,
    const float* bvh_bounds_max,
    const int32_t* bvh_left,
    const int32_t* bvh_right,
    const int32_t* bvh_start,
    const int32_t* bvh_count,
    const int32_t* bvh_prim_ids,
    int num_bvh_nodes,
    int64_t sensor_role_mask,
    int32_t* shadow_stack_overflow_count,
    int32_t* shadow_max_stack_depth) {
  if (max_distance <= 0.0f || num_bvh_nodes <= 0) {
    return false;
  }

  int stack[kMaxBvhStack];
  int stack_size = 0;
  int local_max_stack = 0;
  float root_t = 0.0f;
  if (intersect_aabb(
          ox, oy, oz, dx, dy, dz, bvh_bounds_min, bvh_bounds_max, 0, max_distance, &root_t)) {
    stack[0] = 0;
    stack_size = 1;
    local_max_stack = 1;
  }

  bool occluded = false;
  while (stack_size > 0 && !occluded) {
    --stack_size;
    int node = stack[stack_size];
    int leaf_count = bvh_count[node];
    if (leaf_count > 0) {
      int leaf_start = bvh_start[node];
      for (int offset = 0; offset < leaf_count; ++offset) {
        int tri = bvh_prim_ids[leaf_start + offset];
        if ((triangle_role_masks[tri] & sensor_role_mask) == 0) {
          continue;
        }
        float t = 0.0f;
        float nx = 0.0f;
        float ny = 0.0f;
        float nz = 0.0f;
        if (intersect_triangle(
                ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                tri,
                triangle_v0,
                triangle_e1,
                triangle_e2,
                triangle_normal,
                max_distance,
                &t,
                &nx,
                &ny,
                &nz)) {
          occluded = true;
          break;
        }
      }
    } else {
      int left = bvh_left[node];
      int right = bvh_right[node];
      if (right >= 0) {
        float right_t = 0.0f;
        if (intersect_aabb(
                ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                bvh_bounds_min,
                bvh_bounds_max,
                right,
                max_distance,
                &right_t)) {
          if (stack_size < kMaxBvhStack) {
            stack[stack_size] = right;
            ++stack_size;
          } else {
            atomicAdd(shadow_stack_overflow_count, 1);
          }
        }
      }
      if (left >= 0) {
        float left_t = 0.0f;
        if (intersect_aabb(
                ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                bvh_bounds_min,
                bvh_bounds_max,
                left,
                max_distance,
                &left_t)) {
          if (stack_size < kMaxBvhStack) {
            stack[stack_size] = left;
            ++stack_size;
          } else {
            atomicAdd(shadow_stack_overflow_count, 1);
          }
        }
      }
      if (stack_size > local_max_stack) {
        local_max_stack = stack_size;
      }
    }
  }

  atomicMax(shadow_max_stack_depth, local_max_stack);
  return occluded;
}

__global__ void shade_direct_light_with_shadows_kernel(
    const int32_t* hit_mask,
    const float* position_world,
    const float* normal_world,
    const int32_t* material_index,
    const float* material_albedo_rgb,
    const int32_t* light_kind,
    const float* light_position_or_direction_world,
    const float* light_intensity,
    const float* light_color_rgb,
    int num_lights,
    const float* triangle_v0,
    const float* triangle_e1,
    const float* triangle_e2,
    const float* triangle_normal,
    const int64_t* triangle_role_masks,
    const float* bvh_bounds_min,
    const float* bvh_bounds_max,
    const int32_t* bvh_left,
    const int32_t* bvh_right,
    const int32_t* bvh_start,
    const int32_t* bvh_count,
    const int32_t* bvh_prim_ids,
    int num_bvh_nodes,
    int64_t sensor_role_mask,
    float shadow_bias,
    float ambient_r,
    float ambient_g,
    float ambient_b,
    float background_r,
    float background_g,
    float background_b,
    float* rgb,
    float* intensity,
    int32_t* shadow_stack_overflow_count,
    int32_t* shadow_max_stack_depth,
    int num_rays) {
  int ray = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray >= num_rays) {
    return;
  }

  float out_r = background_r;
  float out_g = background_g;
  float out_b = background_b;
  if (hit_mask[ray] != 0) {
    int mat = material_index[ray];
    float albedo_r = material_albedo_rgb[mat * 3 + 0];
    float albedo_g = material_albedo_rgb[mat * 3 + 1];
    float albedo_b = material_albedo_rgb[mat * 3 + 2];
    out_r = ambient_r * albedo_r;
    out_g = ambient_g * albedo_g;
    out_b = ambient_b * albedo_b;

    float px = position_world[ray * 3 + 0];
    float py = position_world[ray * 3 + 1];
    float pz = position_world[ray * 3 + 2];
    float nx = normal_world[ray * 3 + 0];
    float ny = normal_world[ray * 3 + 1];
    float nz = normal_world[ray * 3 + 2];

    for (int light = 0; light < num_lights; ++light) {
      float lx = light_position_or_direction_world[light * 3 + 0];
      float ly = light_position_or_direction_world[light * 3 + 1];
      float lz = light_position_or_direction_world[light * 3 + 2];
      float attenuation = 1.0f;
      float shadow_max_distance = std::numeric_limits<float>::max();
      if (light_kind[light] == 1) {
        lx = lx - px;
        ly = ly - py;
        lz = lz - pz;
        float distance_sq = lx * lx + ly * ly + lz * lz;
        float distance = sqrtf(distance_sq);
        if (distance > kDirEps) {
          float inv_distance = 1.0f / distance;
          lx = lx * inv_distance;
          ly = ly * inv_distance;
          lz = lz * inv_distance;
          attenuation = 1.0f / fmaxf(distance_sq, 1.0e-12f);
          shadow_max_distance = distance - shadow_bias;
        } else {
          attenuation = 0.0f;
          shadow_max_distance = 0.0f;
        }
      } else {
        float light_norm = sqrtf(lx * lx + ly * ly + lz * lz);
        if (light_norm > kDirEps) {
          float inv_light_norm = 1.0f / light_norm;
          lx = lx * inv_light_norm;
          ly = ly * inv_light_norm;
          lz = lz * inv_light_norm;
        } else {
          attenuation = 0.0f;
          shadow_max_distance = 0.0f;
        }
      }

      float n_dot_l = nx * lx + ny * ly + nz * lz;
      if (n_dot_l > 0.0f && attenuation > 0.0f && shadow_max_distance > 0.0f) {
        float shadow_ox = px + nx * shadow_bias;
        float shadow_oy = py + ny * shadow_bias;
        float shadow_oz = pz + nz * shadow_bias;
        // P12.3d treats triangle meshes as CUDA shadow occluders. Planes can
        // still be primary hits but are intentionally non-occluders here.
        bool visible_to_light = !shadow_occluded_by_triangles(
            shadow_ox,
            shadow_oy,
            shadow_oz,
            lx,
            ly,
            lz,
            shadow_max_distance,
            triangle_v0,
            triangle_e1,
            triangle_e2,
            triangle_normal,
            triangle_role_masks,
            bvh_bounds_min,
            bvh_bounds_max,
            bvh_left,
            bvh_right,
            bvh_start,
            bvh_count,
            bvh_prim_ids,
            num_bvh_nodes,
            sensor_role_mask,
            shadow_stack_overflow_count,
            shadow_max_stack_depth);
        if (visible_to_light) {
          float scale = light_intensity[light] * attenuation * n_dot_l;
          out_r += albedo_r * light_color_rgb[light * 3 + 0] * scale;
          out_g += albedo_g * light_color_rgb[light * 3 + 1] * scale;
          out_b += albedo_b * light_color_rgb[light * 3 + 2] * scale;
        }
      }
    }
  }

  rgb[ray * 3 + 0] = out_r;
  rgb[ray * 3 + 1] = out_g;
  rgb[ray * 3 + 2] = out_b;
  if (hit_mask[ray] != 0) {
    intensity[ray] = out_r * 0.2126f + out_g * 0.7152f + out_b * 0.0722f;
  } else {
    intensity[ray] = 0.0f;
  }
}

void check_float_2d_3(torch::Tensor value, const char* name) {
  TORCH_CHECK(value.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(value.dtype() == torch::kFloat32, name, " must be float32");
  TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(value.dim() == 2 && value.size(1) == 3, name, " must have shape [N, 3]");
}

void check_i32_1d(torch::Tensor value, const char* name) {
  TORCH_CHECK(value.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(value.dtype() == torch::kInt32, name, " must be int32");
  TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(value.dim() == 1, name, " must be rank-1");
}

void check_i64_1d(torch::Tensor value, const char* name) {
  TORCH_CHECK(value.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(value.dtype() == torch::kInt64, name, " must be int64");
  TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(value.dim() == 1, name, " must be rank-1");
}

}  // namespace

std::vector<torch::Tensor> first_hit_rays(
    torch::Tensor origins,
    torch::Tensor directions,
    double max_distance,
    torch::Tensor plane_normals,
    torch::Tensor plane_points,
    torch::Tensor plane_numeric_ids,
    torch::Tensor plane_source_keys,
    torch::Tensor plane_role_masks,
    torch::Tensor plane_material_indices,
    int64_t num_planes,
    torch::Tensor triangle_v0,
    torch::Tensor triangle_e1,
    torch::Tensor triangle_e2,
    torch::Tensor triangle_normal,
    torch::Tensor triangle_numeric_ids,
    torch::Tensor triangle_source_keys,
    torch::Tensor triangle_role_masks,
    torch::Tensor triangle_material_indices,
    torch::Tensor bvh_bounds_min,
    torch::Tensor bvh_bounds_max,
    torch::Tensor bvh_left,
    torch::Tensor bvh_right,
    torch::Tensor bvh_start,
    torch::Tensor bvh_count,
    torch::Tensor bvh_prim_ids,
    int64_t num_bvh_nodes,
    int64_t sensor_role_mask) {
  check_float_2d_3(origins, "origins");
  check_float_2d_3(directions, "directions");
  TORCH_CHECK(directions.sizes() == origins.sizes(), "directions must match origins shape");
  TORCH_CHECK(std::isfinite(max_distance), "max_distance must be finite");
  check_float_2d_3(plane_normals, "plane_normals");
  check_float_2d_3(plane_points, "plane_points");
  check_i32_1d(plane_numeric_ids, "plane_numeric_ids");
  check_i64_1d(plane_source_keys, "plane_source_keys");
  check_i64_1d(plane_role_masks, "plane_role_masks");
  check_i32_1d(plane_material_indices, "plane_material_indices");
  check_float_2d_3(triangle_v0, "triangle_v0");
  check_float_2d_3(triangle_e1, "triangle_e1");
  check_float_2d_3(triangle_e2, "triangle_e2");
  check_float_2d_3(triangle_normal, "triangle_normal");
  check_i32_1d(triangle_numeric_ids, "triangle_numeric_ids");
  check_i64_1d(triangle_source_keys, "triangle_source_keys");
  check_i64_1d(triangle_role_masks, "triangle_role_masks");
  check_i32_1d(triangle_material_indices, "triangle_material_indices");
  check_float_2d_3(bvh_bounds_min, "bvh_bounds_min");
  check_float_2d_3(bvh_bounds_max, "bvh_bounds_max");
  check_i32_1d(bvh_left, "bvh_left");
  check_i32_1d(bvh_right, "bvh_right");
  check_i32_1d(bvh_start, "bvh_start");
  check_i32_1d(bvh_count, "bvh_count");
  check_i32_1d(bvh_prim_ids, "bvh_prim_ids");

  int64_t num_rays = origins.size(0);
  TORCH_CHECK(num_rays <= static_cast<int64_t>(INT32_MAX), "too many rays");
  TORCH_CHECK(num_planes >= 0 && num_planes <= plane_normals.size(0), "num_planes out of range");
  TORCH_CHECK(num_bvh_nodes >= 0 && num_bvh_nodes <= bvh_bounds_min.size(0), "num_bvh_nodes out of range");

  auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(origins.device());
  auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(origins.device());
  auto hit_mask = torch::zeros({num_rays}, int_options);
  auto range_m = torch::full({num_rays}, std::numeric_limits<float>::infinity(), float_options);
  auto position_world = torch::full({num_rays, 3}, std::numeric_limits<float>::quiet_NaN(), float_options);
  auto normal_world = torch::full({num_rays, 3}, std::numeric_limits<float>::quiet_NaN(), float_options);
  auto numeric_instance_id = torch::zeros({num_rays}, int_options);
  auto material_index = torch::zeros({num_rays}, int_options);
  auto bvh_stack_overflow_count = torch::zeros({1}, int_options);
  auto bvh_max_stack_depth = torch::zeros({1}, int_options);

  if (num_rays > 0) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int block = 256;
    int grid = (static_cast<int>(num_rays) + block - 1) / block;
    first_hit_rays_kernel<<<grid, block, 0, stream>>>(
        origins.data_ptr<float>(),
        directions.data_ptr<float>(),
        static_cast<float>(max_distance),
        plane_normals.data_ptr<float>(),
        plane_points.data_ptr<float>(),
        plane_numeric_ids.data_ptr<int32_t>(),
        plane_source_keys.data_ptr<int64_t>(),
        plane_role_masks.data_ptr<int64_t>(),
        plane_material_indices.data_ptr<int32_t>(),
        static_cast<int>(num_planes),
        triangle_v0.data_ptr<float>(),
        triangle_e1.data_ptr<float>(),
        triangle_e2.data_ptr<float>(),
        triangle_normal.data_ptr<float>(),
        triangle_numeric_ids.data_ptr<int32_t>(),
        triangle_source_keys.data_ptr<int64_t>(),
        triangle_role_masks.data_ptr<int64_t>(),
        triangle_material_indices.data_ptr<int32_t>(),
        bvh_bounds_min.data_ptr<float>(),
        bvh_bounds_max.data_ptr<float>(),
        bvh_left.data_ptr<int32_t>(),
        bvh_right.data_ptr<int32_t>(),
        bvh_start.data_ptr<int32_t>(),
        bvh_count.data_ptr<int32_t>(),
        bvh_prim_ids.data_ptr<int32_t>(),
        static_cast<int>(num_bvh_nodes),
        sensor_role_mask,
        hit_mask.data_ptr<int32_t>(),
        range_m.data_ptr<float>(),
        position_world.data_ptr<float>(),
        normal_world.data_ptr<float>(),
        numeric_instance_id.data_ptr<int32_t>(),
        material_index.data_ptr<int32_t>(),
        bvh_stack_overflow_count.data_ptr<int32_t>(),
        bvh_max_stack_depth.data_ptr<int32_t>(),
        static_cast<int>(num_rays));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return {
      hit_mask,
      range_m,
      position_world,
      normal_world,
      numeric_instance_id,
      material_index,
      bvh_stack_overflow_count,
      bvh_max_stack_depth,
  };
}

std::vector<torch::Tensor> shade_direct_light_no_shadow(
    torch::Tensor hit_mask,
    torch::Tensor position_world,
    torch::Tensor normal_world,
    torch::Tensor material_index,
    torch::Tensor material_albedo_rgb,
    torch::Tensor light_kind,
    torch::Tensor light_position_or_direction_world,
    torch::Tensor light_intensity,
    torch::Tensor light_color_rgb,
    int64_t num_lights,
    double ambient_r,
    double ambient_g,
    double ambient_b,
    double background_r,
    double background_g,
    double background_b) {
  check_i32_1d(hit_mask, "hit_mask");
  check_float_2d_3(position_world, "position_world");
  check_float_2d_3(normal_world, "normal_world");
  check_i32_1d(material_index, "material_index");
  check_float_2d_3(material_albedo_rgb, "material_albedo_rgb");
  check_i32_1d(light_kind, "light_kind");
  check_float_2d_3(light_position_or_direction_world, "light_position_or_direction_world");
  check_float_2d_3(light_color_rgb, "light_color_rgb");
  TORCH_CHECK(light_intensity.is_cuda(), "light_intensity must be a CUDA tensor");
  TORCH_CHECK(light_intensity.dtype() == torch::kFloat32, "light_intensity must be float32");
  TORCH_CHECK(light_intensity.is_contiguous(), "light_intensity must be contiguous");
  TORCH_CHECK(light_intensity.dim() == 1, "light_intensity must be rank-1");
  TORCH_CHECK(position_world.sizes() == normal_world.sizes(), "position_world/normal_world shape mismatch");
  TORCH_CHECK(hit_mask.size(0) == position_world.size(0), "hit_mask length must match position_world");
  TORCH_CHECK(
      material_index.size(0) == position_world.size(0),
      "material_index length must match position_world");
  TORCH_CHECK(num_lights >= 0 && num_lights <= light_kind.size(0), "num_lights out of range");
  TORCH_CHECK(num_lights <= light_position_or_direction_world.size(0), "num_lights exceeds light vectors");
  TORCH_CHECK(num_lights <= light_intensity.size(0), "num_lights exceeds light intensities");
  TORCH_CHECK(num_lights <= light_color_rgb.size(0), "num_lights exceeds light colors");

  int64_t num_rays = hit_mask.size(0);
  TORCH_CHECK(num_rays <= static_cast<int64_t>(INT32_MAX), "too many rays");
  auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(hit_mask.device());
  auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(hit_mask.device());
  auto rgb = torch::empty({num_rays, 3}, float_options);
  auto intensity = torch::empty({num_rays}, float_options);
  auto shadow_stack_overflow_count = torch::zeros({1}, int_options);
  auto shadow_max_stack_depth = torch::zeros({1}, int_options);

  if (num_rays > 0) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int block = 256;
    int grid = (static_cast<int>(num_rays) + block - 1) / block;
    shade_direct_light_no_shadow_kernel<<<grid, block, 0, stream>>>(
        hit_mask.data_ptr<int32_t>(),
        position_world.data_ptr<float>(),
        normal_world.data_ptr<float>(),
        material_index.data_ptr<int32_t>(),
        material_albedo_rgb.data_ptr<float>(),
        light_kind.data_ptr<int32_t>(),
        light_position_or_direction_world.data_ptr<float>(),
        light_intensity.data_ptr<float>(),
        light_color_rgb.data_ptr<float>(),
        static_cast<int>(num_lights),
        static_cast<float>(ambient_r),
        static_cast<float>(ambient_g),
        static_cast<float>(ambient_b),
        static_cast<float>(background_r),
        static_cast<float>(background_g),
        static_cast<float>(background_b),
        rgb.data_ptr<float>(),
        intensity.data_ptr<float>(),
        static_cast<int>(num_rays));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return {rgb, intensity, shadow_stack_overflow_count, shadow_max_stack_depth};
}

std::vector<torch::Tensor> shade_direct_light_with_shadows(
    torch::Tensor hit_mask,
    torch::Tensor position_world,
    torch::Tensor normal_world,
    torch::Tensor material_index,
    torch::Tensor material_albedo_rgb,
    torch::Tensor light_kind,
    torch::Tensor light_position_or_direction_world,
    torch::Tensor light_intensity,
    torch::Tensor light_color_rgb,
    int64_t num_lights,
    torch::Tensor triangle_v0,
    torch::Tensor triangle_e1,
    torch::Tensor triangle_e2,
    torch::Tensor triangle_normal,
    torch::Tensor triangle_role_masks,
    torch::Tensor bvh_bounds_min,
    torch::Tensor bvh_bounds_max,
    torch::Tensor bvh_left,
    torch::Tensor bvh_right,
    torch::Tensor bvh_start,
    torch::Tensor bvh_count,
    torch::Tensor bvh_prim_ids,
    int64_t num_bvh_nodes,
    int64_t sensor_role_mask,
    double shadow_bias,
    double ambient_r,
    double ambient_g,
    double ambient_b,
    double background_r,
    double background_g,
    double background_b) {
  check_i32_1d(hit_mask, "hit_mask");
  check_float_2d_3(position_world, "position_world");
  check_float_2d_3(normal_world, "normal_world");
  check_i32_1d(material_index, "material_index");
  check_float_2d_3(material_albedo_rgb, "material_albedo_rgb");
  check_i32_1d(light_kind, "light_kind");
  check_float_2d_3(light_position_or_direction_world, "light_position_or_direction_world");
  check_float_2d_3(light_color_rgb, "light_color_rgb");
  TORCH_CHECK(light_intensity.is_cuda(), "light_intensity must be a CUDA tensor");
  TORCH_CHECK(light_intensity.dtype() == torch::kFloat32, "light_intensity must be float32");
  TORCH_CHECK(light_intensity.is_contiguous(), "light_intensity must be contiguous");
  TORCH_CHECK(light_intensity.dim() == 1, "light_intensity must be rank-1");
  check_float_2d_3(triangle_v0, "triangle_v0");
  check_float_2d_3(triangle_e1, "triangle_e1");
  check_float_2d_3(triangle_e2, "triangle_e2");
  check_float_2d_3(triangle_normal, "triangle_normal");
  check_i64_1d(triangle_role_masks, "triangle_role_masks");
  check_float_2d_3(bvh_bounds_min, "bvh_bounds_min");
  check_float_2d_3(bvh_bounds_max, "bvh_bounds_max");
  check_i32_1d(bvh_left, "bvh_left");
  check_i32_1d(bvh_right, "bvh_right");
  check_i32_1d(bvh_start, "bvh_start");
  check_i32_1d(bvh_count, "bvh_count");
  check_i32_1d(bvh_prim_ids, "bvh_prim_ids");
  TORCH_CHECK(position_world.sizes() == normal_world.sizes(), "position_world/normal_world shape mismatch");
  TORCH_CHECK(hit_mask.size(0) == position_world.size(0), "hit_mask length must match position_world");
  TORCH_CHECK(
      material_index.size(0) == position_world.size(0),
      "material_index length must match position_world");
  TORCH_CHECK(num_lights >= 0 && num_lights <= light_kind.size(0), "num_lights out of range");
  TORCH_CHECK(num_lights <= light_position_or_direction_world.size(0), "num_lights exceeds light vectors");
  TORCH_CHECK(num_lights <= light_intensity.size(0), "num_lights exceeds light intensities");
  TORCH_CHECK(num_lights <= light_color_rgb.size(0), "num_lights exceeds light colors");
  TORCH_CHECK(num_bvh_nodes >= 0 && num_bvh_nodes <= bvh_bounds_min.size(0), "num_bvh_nodes out of range");
  TORCH_CHECK(num_bvh_nodes <= bvh_bounds_max.size(0), "num_bvh_nodes exceeds bvh bounds");
  TORCH_CHECK(num_bvh_nodes <= bvh_left.size(0), "num_bvh_nodes exceeds bvh left");
  TORCH_CHECK(num_bvh_nodes <= bvh_right.size(0), "num_bvh_nodes exceeds bvh right");
  TORCH_CHECK(num_bvh_nodes <= bvh_start.size(0), "num_bvh_nodes exceeds bvh start");
  TORCH_CHECK(num_bvh_nodes <= bvh_count.size(0), "num_bvh_nodes exceeds bvh count");
  TORCH_CHECK(shadow_bias >= 0.0, "shadow_bias must be >= 0");

  int64_t num_rays = hit_mask.size(0);
  TORCH_CHECK(num_rays <= static_cast<int64_t>(INT32_MAX), "too many rays");
  auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(hit_mask.device());
  auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(hit_mask.device());
  auto rgb = torch::empty({num_rays, 3}, float_options);
  auto intensity = torch::empty({num_rays}, float_options);
  auto shadow_stack_overflow_count = torch::zeros({1}, int_options);
  auto shadow_max_stack_depth = torch::zeros({1}, int_options);

  if (num_rays > 0) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int block = 256;
    int grid = (static_cast<int>(num_rays) + block - 1) / block;
    shade_direct_light_with_shadows_kernel<<<grid, block, 0, stream>>>(
        hit_mask.data_ptr<int32_t>(),
        position_world.data_ptr<float>(),
        normal_world.data_ptr<float>(),
        material_index.data_ptr<int32_t>(),
        material_albedo_rgb.data_ptr<float>(),
        light_kind.data_ptr<int32_t>(),
        light_position_or_direction_world.data_ptr<float>(),
        light_intensity.data_ptr<float>(),
        light_color_rgb.data_ptr<float>(),
        static_cast<int>(num_lights),
        triangle_v0.data_ptr<float>(),
        triangle_e1.data_ptr<float>(),
        triangle_e2.data_ptr<float>(),
        triangle_normal.data_ptr<float>(),
        triangle_role_masks.data_ptr<int64_t>(),
        bvh_bounds_min.data_ptr<float>(),
        bvh_bounds_max.data_ptr<float>(),
        bvh_left.data_ptr<int32_t>(),
        bvh_right.data_ptr<int32_t>(),
        bvh_start.data_ptr<int32_t>(),
        bvh_count.data_ptr<int32_t>(),
        bvh_prim_ids.data_ptr<int32_t>(),
        static_cast<int>(num_bvh_nodes),
        sensor_role_mask,
        static_cast<float>(shadow_bias),
        static_cast<float>(ambient_r),
        static_cast<float>(ambient_g),
        static_cast<float>(ambient_b),
        static_cast<float>(background_r),
        static_cast<float>(background_g),
        static_cast<float>(background_b),
        rgb.data_ptr<float>(),
        intensity.data_ptr<float>(),
        shadow_stack_overflow_count.data_ptr<int32_t>(),
        shadow_max_stack_depth.data_ptr<int32_t>(),
        static_cast<int>(num_rays));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return {rgb, intensity, shadow_stack_overflow_count, shadow_max_stack_depth};
}
"""
