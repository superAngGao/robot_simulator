"""CUDA direct-light optical executors."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from sensing.optical import OpticalPinholeCameraSpec, OpticalRaySensorSpec

from .cuda_timeline import CudaOpticalExecutionContext, CudaOpticalTransaction
from .device_bvh import DeviceOpticalBvh
from .device_channel import channel_to_torch
from .device_scene import DeviceOpticalSceneSnapshot
from .execution import OpticalComputeResult, OpticalOutputProfile, normalize_output_profile
from .output_contract import SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES

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
_INT32_MAX = (1 << 31) - 1
_INT64_MAX = (1 << 63) - 1
_SHADOW_TRAVERSAL_COUNTER_CHANNELS = SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES


@dataclass(frozen=True)
class CudaFirstHitInputs:
    """Torch-owned scene and acceleration tensors consumed by first-hit."""

    plane_normal_world: object
    plane_point_world: object
    plane_numeric_instance_id: object
    plane_source_order_key: object
    plane_role_mask: object
    plane_material_index: object
    num_planes: int
    triangle_v0_world: object
    triangle_e1_world: object
    triangle_e2_world: object
    triangle_normal_world: object
    triangle_numeric_instance_id: object
    triangle_source_order_key: object
    triangle_role_mask: object
    triangle_material_index: object
    bvh_bounds_min: object
    bvh_bounds_max: object
    bvh_left: object
    bvh_right: object
    bvh_start: object
    bvh_count: object
    bvh_prim_ids: object
    num_bvh_nodes: int
    sensor_role_mask: int
    max_distance: float

    def tensor_items(self) -> tuple[tuple[str, object], ...]:
        return tuple(
            (name, getattr(self, name))
            for name in (
                "plane_normal_world",
                "plane_point_world",
                "plane_numeric_instance_id",
                "plane_source_order_key",
                "plane_role_mask",
                "plane_material_index",
                "triangle_v0_world",
                "triangle_e1_world",
                "triangle_e2_world",
                "triangle_normal_world",
                "triangle_numeric_instance_id",
                "triangle_source_order_key",
                "triangle_role_mask",
                "triangle_material_index",
                "bvh_bounds_min",
                "bvh_bounds_max",
                "bvh_left",
                "bvh_right",
                "bvh_start",
                "bvh_count",
                "bvh_prim_ids",
            )
        )


@dataclass(frozen=True)
class CudaShadeInputs:
    """Torch-owned material and lighting tensors consumed by direct-light."""

    material_albedo_rgb: object
    light_kind: object
    light_position_or_direction_world: object
    light_intensity: object
    light_color_rgb: object
    num_lights: int

    def tensor_items(self) -> tuple[tuple[str, object], ...]:
        return tuple(
            (name, getattr(self, name))
            for name in (
                "material_albedo_rgb",
                "light_kind",
                "light_position_or_direction_world",
                "light_intensity",
                "light_color_rgb",
            )
        )


@dataclass(frozen=True)
class CudaCameraParameters:
    """Host scalar camera parameters bound before native enqueue."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    origin: tuple[float, float, float]
    rotation: tuple[float, ...]

    @classmethod
    def from_spec(cls, camera: OpticalPinholeCameraSpec) -> "CudaCameraParameters":
        origin = np.asarray(camera.X_world_camera.r, dtype=np.float32)
        rotation = np.asarray(camera.X_world_camera.R, dtype=np.float32)
        return cls(
            width=int(camera.width),
            height=int(camera.height),
            fx=float(camera.fx),
            fy=float(camera.fy),
            cx=float(camera.cx),
            cy=float(camera.cy),
            origin=tuple(float(value) for value in origin),
            rotation=tuple(float(value) for value in rotation.reshape(-1)),
        )


@dataclass(frozen=True)
class CudaDirectLightStageEvidence:
    """Immutable render identity attached to a transaction completion."""

    operation: str
    ray_source: str
    output_profile: str
    shadows: bool
    extension_entry: str | None
    num_rays: int
    traversal_diagnostics: bool


@dataclass(frozen=True)
class CudaDirectLightBuffers:
    """Caller-owned ray, geometry, shade, and diagnostic output tensors."""

    origins: object
    directions: object
    hit_mask: object
    range_m: object
    position_world: object
    normal_world: object
    numeric_instance_id: object
    material_index: object
    bvh_stack_overflow_count: object
    bvh_max_stack_depth: object
    rgb: object | None = None
    intensity: object | None = None
    shadow_stack_overflow_count: object | None = None
    shadow_max_stack_depth: object | None = None
    shadow_traversal_counters: tuple[object, ...] = ()

    def tensor_items(self) -> tuple[tuple[str, object], ...]:
        """Return every allocated output tensor in the frozen ABI order."""

        values = (
            ("origins", self.origins),
            ("directions", self.directions),
            ("hit_mask", self.hit_mask),
            ("range_m", self.range_m),
            ("position_world", self.position_world),
            ("normal_world", self.normal_world),
            ("numeric_instance_id", self.numeric_instance_id),
            ("material_index", self.material_index),
            ("bvh_stack_overflow_count", self.bvh_stack_overflow_count),
            ("bvh_max_stack_depth", self.bvh_max_stack_depth),
            ("rgb", self.rgb),
            ("intensity", self.intensity),
            ("shadow_stack_overflow_count", self.shadow_stack_overflow_count),
            ("shadow_max_stack_depth", self.shadow_max_stack_depth),
        )
        counters = (
            tuple(
                (name, tensor)
                for name, tensor in zip(
                    _SHADOW_TRAVERSAL_COUNTER_CHANNELS,
                    self.shadow_traversal_counters,
                    strict=True,
                )
            )
            if self.shadow_traversal_counters
            else ()
        )
        return tuple((name, value) for name, value in values if value is not None) + counters

    def geometry_channels(self) -> dict[str, object]:
        return {
            "hit_mask": self.hit_mask,
            "range_m": self.range_m,
            "position_world": self.position_world,
            "normal_world": self.normal_world,
            "numeric_instance_id": self.numeric_instance_id,
            "material_index": self.material_index,
            "bvh_stack_overflow_count": self.bvh_stack_overflow_count,
            "bvh_max_stack_depth": self.bvh_max_stack_depth,
        }

    def direct_light_channels(
        self,
        *,
        output_profile: OpticalOutputProfile,
        include_traversal_diagnostics: bool,
    ) -> dict[str, object]:
        if any(
            value is None
            for value in (
                self.rgb,
                self.intensity,
                self.shadow_stack_overflow_count,
                self.shadow_max_stack_depth,
            )
        ):
            raise RuntimeError("direct-light buffers are incomplete")
        channels = self.geometry_channels()
        channels.update(
            {
                "rgb": self.rgb,
                "intensity": self.intensity,
                "shadow_stack_overflow_count": self.shadow_stack_overflow_count,
                "shadow_max_stack_depth": self.shadow_max_stack_depth,
            }
        )
        channels = _filter_channels_for_profile(channels, output_profile)
        if include_traversal_diagnostics:
            channels.update(
                dict(
                    zip(
                        _SHADOW_TRAVERSAL_COUNTER_CHANNELS,
                        self.shadow_traversal_counters,
                        strict=True,
                    )
                )
            )
        return channels

    def resource_tuple(self) -> tuple[object, ...]:
        return tuple(value for _, value in self.tensor_items())


class CudaDirectLightNativeRenderer:
    """Prepared CUDA extension bound to one explicit execution context."""

    def __init__(self, *, context: CudaOpticalExecutionContext, module: object) -> None:
        self.context = context
        self.module = module

    @classmethod
    def prepare(cls, context: CudaOpticalExecutionContext) -> "CudaDirectLightNativeRenderer":
        return cls(context=context, module=_load_cuda_direct_light_extension())

    def bind_first_hit(
        self,
        *,
        first_hit_inputs: CudaFirstHitInputs,
        buffers: CudaDirectLightBuffers,
        camera: CudaCameraParameters | None = None,
    ) -> "CudaDirectLightRenderStage":
        stage = CudaDirectLightRenderStage(
            renderer=self,
            first_hit_inputs=first_hit_inputs,
            shade_inputs=None,
            buffers=buffers,
            camera=camera,
            output_profile=OpticalOutputProfile.GEOMETRY_FULL,
            shadows=False,
            ambient_rgb=(0.0, 0.0, 0.0),
            background_rgb=(0.0, 0.0, 0.0),
            shadow_bias=0.0,
        )
        stage._validate_cuda_transaction_stage(self.context)
        stage._freeze_storage_signature()
        return stage

    def bind_direct_light(
        self,
        *,
        first_hit_inputs: CudaFirstHitInputs,
        shade_inputs: CudaShadeInputs,
        buffers: CudaDirectLightBuffers,
        camera: CudaCameraParameters | None = None,
        output_profile: OpticalOutputProfile | str,
        shadows: bool,
        ambient_rgb: tuple[float, float, float],
        background_rgb: tuple[float, float, float],
        shadow_bias: float,
    ) -> "CudaDirectLightRenderStage":
        resolved_profile = normalize_output_profile(output_profile)
        if resolved_profile not in {
            OpticalOutputProfile.DIRECT_LIGHT_FULL,
            OpticalOutputProfile.RGB_PREVIEW,
            OpticalOutputProfile.RENDER_ONLY,
        }:
            raise ValueError("native CUDA direct-light binding requires a direct-light output profile")
        stage = CudaDirectLightRenderStage(
            renderer=self,
            first_hit_inputs=first_hit_inputs,
            shade_inputs=shade_inputs,
            buffers=buffers,
            camera=camera,
            output_profile=resolved_profile,
            shadows=bool(shadows),
            ambient_rgb=ambient_rgb,
            background_rgb=background_rgb,
            shadow_bias=float(shadow_bias),
        )
        stage._validate_cuda_transaction_stage(self.context)
        stage._freeze_storage_signature()
        return stage


class CudaDirectLightRenderStage:
    """One allocation-free raygen/first-hit/shade transaction stage."""

    cuda_transaction_stage_name = "cuda_direct_light_render"

    def __setattr__(self, name: str, value: object) -> None:
        if name != "_sealed" and getattr(self, "_sealed", False):
            raise AttributeError("CUDA render stage binding is immutable after validation")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        renderer: CudaDirectLightNativeRenderer,
        first_hit_inputs: CudaFirstHitInputs,
        shade_inputs: CudaShadeInputs | None,
        buffers: CudaDirectLightBuffers,
        camera: CudaCameraParameters | None,
        output_profile: OpticalOutputProfile,
        shadows: bool,
        ambient_rgb: tuple[float, float, float],
        background_rgb: tuple[float, float, float],
        shadow_bias: float,
    ) -> None:
        self._sealed = False
        self.renderer = renderer
        self.first_hit_inputs = first_hit_inputs
        self.shade_inputs = shade_inputs
        self.buffers = buffers
        self.camera = camera
        self.output_profile = output_profile
        self.shadows = shadows
        self.ambient_rgb = tuple(float(value) for value in ambient_rgb)
        self.background_rgb = tuple(float(value) for value in background_rgb)
        self.shadow_bias = float(shadow_bias)
        self._raygen_entry: object | None = None
        self._raygen_args: tuple[object, ...] = ()
        if camera is not None:
            self._raygen_entry = getattr(renderer.module, "pinhole_camera_rays_out", None)
            self._raygen_args = (
                buffers.origins,
                buffers.directions,
                camera.width,
                camera.height,
                camera.fx,
                camera.fy,
                camera.cx,
                camera.cy,
                camera.origin[0],
                camera.origin[1],
                camera.origin[2],
                camera.rotation[0],
                camera.rotation[1],
                camera.rotation[2],
                camera.rotation[3],
                camera.rotation[4],
                camera.rotation[5],
                camera.rotation[6],
                camera.rotation[7],
                camera.rotation[8],
            )
        self._shade_entry_name: str | None = None
        self._shade_entry: object | None = None
        if shade_inputs is not None:
            self._shade_entry_name = _shade_extension_entry_name(
                shadows=self.shadows,
                traversal_diagnostics=bool(buffers.shadow_traversal_counters),
            )
            self._shade_entry = getattr(renderer.module, self._shade_entry_name, None)
        self._shade_args = self._bind_shade_arguments()
        self._binding_signature: tuple[object, ...] | None = None

    def _bind_shade_arguments(self) -> tuple[object, ...]:
        shade = self.shade_inputs
        if shade is None:
            return ()
        output = self.buffers
        arguments = (
            output.hit_mask,
            output.position_world,
            output.normal_world,
            output.material_index,
            shade.material_albedo_rgb,
            shade.light_kind,
            shade.light_position_or_direction_world,
            shade.light_intensity,
            shade.light_color_rgb,
            shade.num_lights,
        )
        if self.shadows:
            first = self.first_hit_inputs
            arguments += (
                first.triangle_v0_world,
                first.triangle_e1_world,
                first.triangle_e2_world,
                first.triangle_normal_world,
                first.triangle_role_mask,
                first.bvh_bounds_min,
                first.bvh_bounds_max,
                first.bvh_left,
                first.bvh_right,
                first.bvh_start,
                first.bvh_count,
                first.bvh_prim_ids,
                first.num_bvh_nodes,
                first.sensor_role_mask,
                self.shadow_bias,
            )
        arguments += (
            self.ambient_rgb[0],
            self.ambient_rgb[1],
            self.ambient_rgb[2],
            self.background_rgb[0],
            self.background_rgb[1],
            self.background_rgb[2],
            output.rgb,
            output.intensity,
            output.shadow_stack_overflow_count,
            output.shadow_max_stack_depth,
        )
        if output.shadow_traversal_counters:
            arguments += output.shadow_traversal_counters
        return arguments

    def enqueue(self, transaction: CudaOpticalTransaction) -> None:
        module = self.renderer.module
        first = self.first_hit_inputs
        output = self.buffers
        with transaction.enqueue_stage(self):
            if self.camera is not None:
                transaction.record_diagnostic_start("raygen_kernel")
                raygen_entry = self._raygen_entry
                if raygen_entry is None:  # pragma: no cover - rejected during validation.
                    raise RuntimeError("CUDA raygen extension entry is not prepared")
                raygen_entry(*self._raygen_args)
                transaction.record_diagnostic_end("raygen_kernel")

            transaction.record_diagnostic_start("first_hit_kernel")
            module.first_hit_rays_out(
                output.origins,
                output.directions,
                first.max_distance,
                first.plane_normal_world,
                first.plane_point_world,
                first.plane_numeric_instance_id,
                first.plane_source_order_key,
                first.plane_role_mask,
                first.plane_material_index,
                first.num_planes,
                first.triangle_v0_world,
                first.triangle_e1_world,
                first.triangle_e2_world,
                first.triangle_normal_world,
                first.triangle_numeric_instance_id,
                first.triangle_source_order_key,
                first.triangle_role_mask,
                first.triangle_material_index,
                first.bvh_bounds_min,
                first.bvh_bounds_max,
                first.bvh_left,
                first.bvh_right,
                first.bvh_start,
                first.bvh_count,
                first.bvh_prim_ids,
                first.num_bvh_nodes,
                first.sensor_role_mask,
                output.hit_mask,
                output.range_m,
                output.position_world,
                output.normal_world,
                output.numeric_instance_id,
                output.material_index,
                output.bvh_stack_overflow_count,
                output.bvh_max_stack_depth,
            )
            transaction.record_diagnostic_end("first_hit_kernel")

            if self.shade_inputs is None:
                return
            transaction.record_diagnostic_start("shade_kernel")
            shade_entry = self._shade_entry
            if shade_entry is None:  # pragma: no cover - rejected during bind/transaction validation.
                raise RuntimeError("CUDA shade extension entry is not prepared")
            shade_entry(*self._shade_args)
            transaction.record_diagnostic_end("shade_kernel")

    def output_channels(self, *, include_traversal_diagnostics: bool) -> dict[str, object]:
        if self.shade_inputs is None:
            return self.buffers.geometry_channels()
        return self.buffers.direct_light_channels(
            output_profile=self.output_profile,
            include_traversal_diagnostics=include_traversal_diagnostics,
        )

    def _validate_cuda_transaction_stage(self, context: CudaOpticalExecutionContext) -> None:
        if context is not self.renderer.context:
            raise ValueError("CUDA render stage execution context identity mismatch")
        _validate_renderer_module(
            self.renderer.module,
            direct_light=self.shade_inputs is not None,
            camera=self.camera is not None,
        )
        prepared_raygen = getattr(self.renderer.module, "pinhole_camera_rays_out", None)
        if self.camera is not None:
            if not callable(prepared_raygen) or prepared_raygen is not self._raygen_entry:
                raise ValueError("prepared CUDA renderer is missing stable pinhole_camera_rays_out")
            if len(self._raygen_args) != 20:
                raise ValueError("CUDA raygen extension argument binding changed")
        elif self._raygen_entry is not None or self._raygen_args:
            raise ValueError("device-ray CUDA binding cannot contain a raygen entry")
        if self.shade_inputs is None:
            if self.output_profile is not OpticalOutputProfile.GEOMETRY_FULL or self.shadows:
                raise ValueError("geometry-only CUDA binding has invalid render identity")
        elif self.output_profile not in {
            OpticalOutputProfile.DIRECT_LIGHT_FULL,
            OpticalOutputProfile.RGB_PREVIEW,
            OpticalOutputProfile.RENDER_ONLY,
        }:
            raise ValueError("CUDA direct-light binding has invalid output profile")
        if self.shade_inputs is not None:
            expected_entry_name = _shade_extension_entry_name(
                shadows=self.shadows,
                traversal_diagnostics=bool(self.buffers.shadow_traversal_counters),
            )
            if self._shade_entry_name != expected_entry_name:
                raise ValueError("CUDA shade extension entry identity changed")
            prepared_entry = getattr(self.renderer.module, expected_entry_name, None)
            if not callable(prepared_entry) or prepared_entry is not self._shade_entry:
                raise ValueError(f"prepared CUDA renderer is missing stable {expected_entry_name}")
        if len(self.ambient_rgb) != 3 or len(self.background_rgb) != 3:
            raise ValueError("CUDA direct-light RGB scalars must have three components")
        if not all(math.isfinite(value) for value in (*self.ambient_rgb, *self.background_rgb)):
            raise ValueError("CUDA direct-light RGB scalars must be finite")
        if not math.isfinite(self.shadow_bias) or self.shadow_bias < 0.0:
            raise ValueError("CUDA direct-light shadow_bias must be finite and >= 0")
        _validate_first_hit_inputs(self.first_hit_inputs, context.device)
        if self.shade_inputs is not None:
            _validate_shade_inputs(self.shade_inputs, context.device)
        _validate_render_buffers(
            self.buffers,
            device=context.device,
            camera=self.camera,
            direct_light=self.shade_inputs is not None,
        )
        _validate_render_storage_disjointness(self)

    def _freeze_storage_signature(self) -> None:
        self._binding_signature = _render_binding_signature(self)
        self._sealed = True

    def _assert_cuda_transaction_stage_stable(self, context: CudaOpticalExecutionContext) -> None:
        if context is not self.renderer.context:
            raise ValueError("CUDA render stage execution context changed")
        if self._binding_signature is None or _render_binding_signature(self) != self._binding_signature:
            raise ValueError("CUDA render stage binding changed after validation")

    def _cuda_transaction_stage_evidence(self) -> CudaDirectLightStageEvidence:
        return CudaDirectLightStageEvidence(
            operation="first_hit" if self.shade_inputs is None else "direct_light",
            ray_source="camera" if self.camera is not None else "device_rays",
            output_profile=self.output_profile.value,
            shadows=self.shadows,
            extension_entry=self._shade_entry_name,
            num_rays=int(self.buffers.origins.shape[0]),
            traversal_diagnostics=bool(self.buffers.shadow_traversal_counters),
        )


def allocate_cuda_direct_light_buffers(
    *,
    num_rays: int,
    device: object,
    origins: object | None = None,
    directions: object | None = None,
    direct_light: bool,
    traversal_diagnostics: bool,
) -> CudaDirectLightBuffers:
    """Allocate a one-shot bundle outside native enqueue.

    C4 replaces this helper with reusable arena slots for benchmark lanes.
    """

    _require_cuda_direct_light_deps()
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda" or resolved_device.index is None:
        raise ValueError("CUDA render buffers require an indexed CUDA device")
    if not isinstance(num_rays, int) or isinstance(num_rays, bool) or num_rays < 0:
        raise ValueError("num_rays must be a non-negative integer")
    with torch.cuda.device(resolved_device):
        ray_origins = origins
        ray_directions = directions
        if ray_origins is None:
            ray_origins = torch.empty((num_rays, 3), dtype=torch.float32, device=resolved_device)
        if ray_directions is None:
            ray_directions = torch.empty((num_rays, 3), dtype=torch.float32, device=resolved_device)
        kwargs: dict[str, object] = {}
        if direct_light:
            kwargs = {
                "rgb": torch.empty((num_rays, 3), dtype=torch.float32, device=resolved_device),
                "intensity": torch.empty((num_rays,), dtype=torch.float32, device=resolved_device),
                "shadow_stack_overflow_count": torch.empty((1,), dtype=torch.int32, device=resolved_device),
                "shadow_max_stack_depth": torch.empty((1,), dtype=torch.int32, device=resolved_device),
                "shadow_traversal_counters": tuple(
                    torch.empty((1,), dtype=torch.int32, device=resolved_device)
                    for _ in _SHADOW_TRAVERSAL_COUNTER_CHANNELS
                )
                if traversal_diagnostics
                else (),
            }
        return CudaDirectLightBuffers(
            origins=ray_origins,
            directions=ray_directions,
            hit_mask=torch.empty((num_rays,), dtype=torch.int32, device=resolved_device),
            range_m=torch.empty((num_rays,), dtype=torch.float32, device=resolved_device),
            position_world=torch.empty((num_rays, 3), dtype=torch.float32, device=resolved_device),
            normal_world=torch.empty((num_rays, 3), dtype=torch.float32, device=resolved_device),
            numeric_instance_id=torch.empty((num_rays,), dtype=torch.int32, device=resolved_device),
            material_index=torch.empty((num_rays,), dtype=torch.int32, device=resolved_device),
            bvh_stack_overflow_count=torch.empty((1,), dtype=torch.int32, device=resolved_device),
            bvh_max_stack_depth=torch.empty((1,), dtype=torch.int32, device=resolved_device),
            **kwargs,
        )


def _validate_first_hit_inputs(inputs: CudaFirstHitInputs, device: object) -> None:
    float_2d = (
        "plane_normal_world",
        "plane_point_world",
        "triangle_v0_world",
        "triangle_e1_world",
        "triangle_e2_world",
        "triangle_normal_world",
        "bvh_bounds_min",
        "bvh_bounds_max",
    )
    int32_1d = (
        "plane_numeric_instance_id",
        "plane_material_index",
        "triangle_numeric_instance_id",
        "triangle_material_index",
        "bvh_left",
        "bvh_right",
        "bvh_start",
        "bvh_count",
        "bvh_prim_ids",
    )
    int64_1d = (
        "plane_source_order_key",
        "plane_role_mask",
        "triangle_source_order_key",
        "triangle_role_mask",
    )
    for name in float_2d:
        _validate_tensor(getattr(inputs, name), name=name, device=device, dtype=torch.float32, tail=(3,))
    for name in int32_1d:
        _validate_tensor(getattr(inputs, name), name=name, device=device, dtype=torch.int32, tail=())
    for name in int64_1d:
        _validate_tensor(getattr(inputs, name), name=name, device=device, dtype=torch.int64, tail=())
    if not isinstance(inputs.num_planes, int) or isinstance(inputs.num_planes, bool):
        raise ValueError("num_planes must be an integer")
    if not isinstance(inputs.num_bvh_nodes, int) or isinstance(inputs.num_bvh_nodes, bool):
        raise ValueError("num_bvh_nodes must be an integer")
    if not isinstance(inputs.sensor_role_mask, int) or isinstance(inputs.sensor_role_mask, bool):
        raise ValueError("sensor_role_mask must be an integer")
    if not 0 <= inputs.num_planes <= min(inputs.plane_normal_world.shape[0], _INT32_MAX):
        raise ValueError("num_planes is out of range")
    for name in (
        "plane_point_world",
        "plane_numeric_instance_id",
        "plane_source_order_key",
        "plane_role_mask",
        "plane_material_index",
    ):
        if getattr(inputs, name).shape[0] < inputs.num_planes:
            raise ValueError(f"{name} capacity is smaller than num_planes")
    num_triangles = inputs.triangle_v0_world.shape[0]
    if num_triangles > _INT32_MAX:
        raise ValueError("triangle count exceeds CUDA kernel index range")
    for name in (
        "triangle_e1_world",
        "triangle_e2_world",
        "triangle_normal_world",
        "triangle_numeric_instance_id",
        "triangle_source_order_key",
        "triangle_role_mask",
        "triangle_material_index",
    ):
        if getattr(inputs, name).shape[0] != num_triangles:
            raise ValueError(f"{name} length must match triangle_v0_world")
    if not 0 <= inputs.num_bvh_nodes <= min(inputs.bvh_bounds_min.shape[0], _INT32_MAX):
        raise ValueError("num_bvh_nodes is out of range")
    for name in ("bvh_bounds_max", "bvh_left", "bvh_right", "bvh_start", "bvh_count"):
        if getattr(inputs, name).shape[0] < inputs.num_bvh_nodes:
            raise ValueError(f"{name} capacity is smaller than num_bvh_nodes")
    if inputs.bvh_prim_ids.shape[0] < num_triangles:
        raise ValueError("bvh_prim_ids is shorter than the triangle input")
    if not 0 <= inputs.sensor_role_mask <= _INT64_MAX:
        raise ValueError("sensor_role_mask is outside signed int64 range")
    if not math.isfinite(float(inputs.max_distance)):
        raise ValueError("max_distance must be finite")


def _validate_shade_inputs(inputs: CudaShadeInputs, device: object) -> None:
    for name in (
        "material_albedo_rgb",
        "light_position_or_direction_world",
        "light_color_rgb",
    ):
        _validate_tensor(getattr(inputs, name), name=name, device=device, dtype=torch.float32, tail=(3,))
    _validate_tensor(inputs.light_kind, name="light_kind", device=device, dtype=torch.int32, tail=())
    _validate_tensor(
        inputs.light_intensity,
        name="light_intensity",
        device=device,
        dtype=torch.float32,
        tail=(),
    )
    if not isinstance(inputs.num_lights, int) or isinstance(inputs.num_lights, bool):
        raise ValueError("num_lights must be an integer")
    if not 0 <= inputs.num_lights <= min(inputs.light_kind.shape[0], _INT32_MAX):
        raise ValueError("num_lights is out of range")
    if any(
        inputs.num_lights > tensor.shape[0]
        for tensor in (
            inputs.light_position_or_direction_world,
            inputs.light_intensity,
            inputs.light_color_rgb,
        )
    ):
        raise ValueError("num_lights exceeds a light input tensor")


def _validate_render_buffers(
    buffers: CudaDirectLightBuffers,
    *,
    device: object,
    camera: CudaCameraParameters | None,
    direct_light: bool,
) -> None:
    _validate_tensor(buffers.origins, name="origins", device=device, dtype=torch.float32, tail=(3,))
    num_rays = buffers.origins.shape[0]
    if num_rays > _INT32_MAX:
        raise ValueError("ray count exceeds CUDA kernel index range")
    _validate_tensor(
        buffers.directions,
        name="directions",
        device=device,
        dtype=torch.float32,
        exact=(num_rays, 3),
    )
    for name, dtype, exact in (
        ("hit_mask", torch.int32, (num_rays,)),
        ("range_m", torch.float32, (num_rays,)),
        ("position_world", torch.float32, (num_rays, 3)),
        ("normal_world", torch.float32, (num_rays, 3)),
        ("numeric_instance_id", torch.int32, (num_rays,)),
        ("material_index", torch.int32, (num_rays,)),
        ("bvh_stack_overflow_count", torch.int32, (1,)),
        ("bvh_max_stack_depth", torch.int32, (1,)),
    ):
        _validate_tensor(getattr(buffers, name), name=name, device=device, dtype=dtype, exact=exact)
    if camera is not None:
        if (
            not isinstance(camera.width, int)
            or isinstance(camera.width, bool)
            or not isinstance(camera.height, int)
            or isinstance(camera.height, bool)
        ):
            raise ValueError("camera width and height must be integers")
        if camera.width <= 0 or camera.height <= 0 or camera.width * camera.height != num_rays:
            raise ValueError("camera dimensions must match ray buffer rows")
        if not math.isfinite(camera.fx) or not math.isfinite(camera.fy) or camera.fx <= 0 or camera.fy <= 0:
            raise ValueError("camera focal lengths must be finite and > 0")
        if len(camera.origin) != 3 or len(camera.rotation) != 9:
            raise ValueError("camera origin/rotation shapes are invalid")
        if not all(
            math.isfinite(value) for value in (camera.cx, camera.cy, *camera.origin, *camera.rotation)
        ):
            raise ValueError("camera scalar parameters must be finite")
    if direct_light:
        for name, dtype, exact in (
            ("rgb", torch.float32, (num_rays, 3)),
            ("intensity", torch.float32, (num_rays,)),
            ("shadow_stack_overflow_count", torch.int32, (1,)),
            ("shadow_max_stack_depth", torch.int32, (1,)),
        ):
            value = getattr(buffers, name)
            if value is None:
                raise ValueError(f"{name} is required for direct-light")
            _validate_tensor(value, name=name, device=device, dtype=dtype, exact=exact)
        if len(buffers.shadow_traversal_counters) not in (0, len(_SHADOW_TRAVERSAL_COUNTER_CHANNELS)):
            raise ValueError("shadow traversal diagnostics must be completely absent or present")
        for index, value in enumerate(buffers.shadow_traversal_counters):
            _validate_tensor(
                value,
                name=_SHADOW_TRAVERSAL_COUNTER_CHANNELS[index],
                device=device,
                dtype=torch.int32,
                exact=(1,),
            )
    elif (
        any(
            value is not None
            for value in (
                buffers.rgb,
                buffers.intensity,
                buffers.shadow_stack_overflow_count,
                buffers.shadow_max_stack_depth,
            )
        )
        or buffers.shadow_traversal_counters
    ):
        raise ValueError("geometry-only buffers cannot contain shade outputs")


def _validate_tensor(
    value: object,
    *,
    name: str,
    device: object,
    dtype: object,
    tail: tuple[int, ...] | None = None,
    exact: tuple[int, ...] | None = None,
) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if value.device != torch.device(device):
        raise ValueError(f"{name} must be on {torch.device(device)}")
    if value.dtype != dtype:
        raise ValueError(f"{name} has the wrong dtype")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if exact is not None and tuple(value.shape) != exact:
        raise ValueError(f"{name} has the wrong shape")
    if tail is not None:
        expected_rank = 1 + len(tail)
        if value.dim() != expected_rank or tuple(value.shape[1:]) != tail:
            raise ValueError(f"{name} has the wrong shape")


def _validate_render_storage_disjointness(stage: CudaDirectLightRenderStage) -> None:
    inputs = [value for _, value in stage.first_hit_inputs.tensor_items()]
    if stage.shade_inputs is not None:
        inputs.extend(value for _, value in stage.shade_inputs.tensor_items())
    outputs = list(stage.buffers.resource_tuple())
    if stage.camera is None:
        inputs.extend((stage.buffers.origins, stage.buffers.directions))
        outputs = outputs[2:]
    spans: list[tuple[int, int, str]] = []
    for index, value in enumerate(inputs):
        start, end = _tensor_storage_span(value)
        spans.append((start, end, f"input[{index}]"))
    output_spans: list[tuple[int, int, str]] = []
    for index, value in enumerate(outputs):
        start, end = _tensor_storage_span(value)
        label = f"output[{index}]"
        for other_start, other_end, other_label in (*spans, *output_spans):
            if start < other_end and other_start < end:
                raise ValueError(f"CUDA render storage overlap between {label} and {other_label}")
        output_spans.append((start, end, label))


def _render_binding_signature(stage: CudaDirectLightRenderStage) -> tuple[object, ...]:
    tensors = [value for _, value in stage.first_hit_inputs.tensor_items()]
    if stage.shade_inputs is not None:
        tensors.extend(value for _, value in stage.shade_inputs.tensor_items())
    tensors.extend(stage.buffers.resource_tuple())
    storage = tuple(
        (
            id(value),
            int(value.data_ptr()),
            tuple(value.shape),
            tuple(value.stride()),
            value.dtype,
            value.device,
        )
        for value in tensors
    )
    return (
        id(stage.renderer),
        id(stage.renderer.context),
        id(stage.renderer.module),
        id(stage.first_hit_inputs),
        id(stage.shade_inputs),
        id(stage.buffers),
        stage.camera,
        stage.output_profile,
        stage.shadows,
        stage.ambient_rgb,
        stage.background_rgb,
        stage.shadow_bias,
        id(stage._raygen_entry),
        id(stage._raygen_args),
        stage._shade_entry_name,
        id(stage._shade_entry),
        id(stage._shade_args),
        storage,
    )


def _shade_extension_entry_name(*, shadows: bool, traversal_diagnostics: bool) -> str:
    base = "shade_direct_light_with_shadows" if shadows else "shade_direct_light_no_shadow"
    suffix = "_diagnostics_out" if traversal_diagnostics else "_out"
    return f"{base}{suffix}"


def _validate_renderer_module(module: object, *, direct_light: bool, camera: bool) -> None:
    required = ["first_hit_rays_out"]
    if camera:
        required.append("pinhole_camera_rays_out")
    for name in required:
        if not callable(getattr(module, name, None)):
            raise ValueError(f"prepared CUDA renderer is missing {name}")


def _tensor_storage_span(value: object) -> tuple[int, int]:
    start = int(value.data_ptr())
    return start, start + int(value.numel()) * int(value.element_size())


def _first_hit_inputs_from_snapshot(
    snapshot: DeviceOpticalSceneSnapshot,
    bvh: DeviceOpticalBvh,
    *,
    max_distance: float,
    sensor_role: str,
) -> CudaFirstHitInputs:
    scene = snapshot.scene
    return CudaFirstHitInputs(
        plane_normal_world=channel_to_torch(snapshot.plane_normal_world),
        plane_point_world=channel_to_torch(snapshot.plane_point_world),
        plane_numeric_instance_id=channel_to_torch(scene.plane_numeric_instance_id),
        plane_source_order_key=channel_to_torch(scene.plane_source_order_key),
        plane_role_mask=channel_to_torch(scene.plane_role_mask),
        plane_material_index=channel_to_torch(scene.plane_material_index),
        num_planes=int(scene.num_planes),
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
        num_bvh_nodes=int(bvh.num_nodes),
        sensor_role_mask=int(scene.role_table.mask_for(sensor_role)),
        max_distance=float(max_distance),
    )


def _shade_inputs_from_snapshot(snapshot: DeviceOpticalSceneSnapshot) -> CudaShadeInputs:
    scene = snapshot.scene
    return CudaShadeInputs(
        material_albedo_rgb=channel_to_torch(scene.material_albedo_rgb),
        light_kind=channel_to_torch(scene.light_kind),
        light_position_or_direction_world=channel_to_torch(scene.light_position_or_direction_world),
        light_intensity=channel_to_torch(scene.light_intensity),
        light_color_rgb=channel_to_torch(scene.light_color_rgb),
        num_lights=int(scene.num_lights),
    )


def _new_execution_context(device: object) -> CudaOpticalExecutionContext:
    tensor_device = torch.device(str(device))
    with torch.cuda.device(tensor_device):
        stream = torch.cuda.Stream(device=tensor_device)
    return CudaOpticalExecutionContext(device=tensor_device, stream=stream)


def _prepare_compatibility_events(
    context: CudaOpticalExecutionContext,
    phase_names: tuple[str, ...],
) -> tuple[object, dict[str, tuple[object, object]]]:
    with context.activate():
        completion = torch.cuda.Event()
        diagnostics = {
            name: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for name in phase_names
        }
        events = [completion]
        for start, end in diagnostics.values():
            events.extend((start, end))
        for event in events:
            event.record(context.stream)
    events[-1].synchronize()
    return completion, diagnostics


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

    def __init__(self, *, device=None) -> None:
        _require_cuda_direct_light_deps()
        wp.init()
        self.device = wp.get_device("cuda:0" if device is None else device)
        self._execution_context = _new_execution_context(self.device)

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
        self._validate_request(
            snapshot,
            bvh,
            frame_id=spec.frame_id,
            env_idx=spec.env_idx,
            max_distance=spec.max_distance,
        )
        self._wait_for_inputs(snapshot, bvh)

        tensor_device = _torch_device_for_warp_device(self.device)
        upload_start = time.perf_counter()
        with torch.cuda.device(tensor_device):
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

        return self._execute_device_rays(
            snapshot,
            bvh,
            origins,
            directions,
            frame_id=spec.frame_id,
            sim_time=spec.sim_time,
            env_idx=spec.env_idx,
            sensor_id=spec.sensor_id,
            max_distance=spec.max_distance,
            sensor_role=spec.sensor_role,
            render_profile=render_profile,
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
        """Generate pinhole rays on CUDA and execute CUDA LBVH first-hit traversal."""

        output_profile = normalize_output_profile(output_profile)
        if output_profile is not OpticalOutputProfile.GEOMETRY_FULL:
            raise ValueError("CudaDeviceBvhOpticalExecutor supports only output_profile='geometry_full'")
        self._validate_request(
            snapshot,
            bvh,
            frame_id=camera.frame_id,
            env_idx=camera.env_idx,
            max_distance=camera.max_distance,
        )
        self._wait_for_inputs(snapshot, bvh)
        params_start = time.perf_counter()
        camera_parameters = CudaCameraParameters.from_spec(camera)
        _add_profile_ms(render_profile, "raygen_camera_params_ms", params_start)
        return self._execute_device_rays(
            snapshot,
            bvh,
            None,
            None,
            frame_id=camera.frame_id,
            sim_time=camera.sim_time,
            env_idx=camera.env_idx,
            sensor_id=camera.sensor_id,
            max_distance=camera.max_distance,
            sensor_role=camera.sensor_role,
            render_profile=render_profile,
            camera=camera_parameters,
        )

    def _execute_device_rays(
        self,
        snapshot: DeviceOpticalSceneSnapshot,
        bvh: DeviceOpticalBvh,
        origins,
        directions,
        *,
        frame_id: int,
        sim_time: float,
        env_idx: int,
        sensor_id: str,
        max_distance: float,
        sensor_role: str,
        render_profile: list[tuple[str, float]] | None,
        camera: CudaCameraParameters | None = None,
    ) -> OpticalComputeResult:
        load_start = time.perf_counter()
        renderer = CudaDirectLightNativeRenderer.prepare(self._execution_context)
        _add_profile_ms(render_profile, "cuda_first_hit_load_extension_ms", load_start)

        convert_start = time.perf_counter()
        first_hit_inputs = _first_hit_inputs_from_snapshot(
            snapshot,
            bvh,
            max_distance=max_distance,
            sensor_role=sensor_role,
        )
        _add_profile_ms(render_profile, "cuda_first_hit_channel_views_ms", convert_start)

        alloc_start = time.perf_counter()
        num_rays = camera.width * camera.height if camera is not None else int(origins.shape[0])
        buffers = allocate_cuda_direct_light_buffers(
            num_rays=num_rays,
            device=self._execution_context.device,
            origins=origins,
            directions=directions,
            direct_light=False,
            traversal_diagnostics=False,
        )
        _add_profile_ms(render_profile, "raygen_buffer_alloc_ms", alloc_start)

        stage = renderer.bind_first_hit(
            first_hit_inputs=first_hit_inputs,
            buffers=buffers,
            camera=camera,
        )
        phase_names = ("raygen_kernel",) if render_profile is not None and camera is not None else ()
        completion_event, diagnostic_events = _prepare_compatibility_events(
            self._execution_context,
            phase_names,
        )
        transaction = self._execution_context.begin_transaction(
            stages=(stage,),
            completion_event=completion_event,
            diagnostic_events=diagnostic_events,
        )
        execute_start = time.perf_counter()
        stage.enqueue(transaction)
        completion = transaction.finish()
        completion.event.synchronize()
        _add_profile_ms(render_profile, "cuda_first_hit_kernel_sync_ms", execute_start)
        if render_profile is not None and completion.deferred_profile is not None:
            render_profile.extend(completion.deferred_profile.resolve_ms())

        return OpticalComputeResult(
            frame_id=frame_id,
            sim_time=sim_time,
            env_idx=env_idx,
            sensor_id=sensor_id,
            location="device",
            channels=buffers.geometry_channels(),
            output_profile=OpticalOutputProfile.GEOMETRY_FULL,
            ready_event=None,
            resources=buffers.resource_tuple(),
        )

    def _validate_request(
        self,
        snapshot: DeviceOpticalSceneSnapshot,
        bvh: DeviceOpticalBvh,
        *,
        frame_id: int,
        env_idx: int,
        max_distance: float,
    ) -> None:
        if snapshot.frame_id != frame_id:
            raise ValueError("snapshot.frame_id must match request frame_id")
        if snapshot.env_idx != env_idx:
            raise ValueError("snapshot.env_idx must match request env_idx")
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
        if not np.isfinite(max_distance):
            raise ValueError("CudaDeviceBvhOpticalExecutor requires finite max_distance")

    @staticmethod
    def _wait_for_inputs(snapshot: DeviceOpticalSceneSnapshot, bvh: DeviceOpticalBvh) -> None:
        _synchronize_ready_event(snapshot.ready_event)
        _synchronize_ready_event(bvh.ready_event)


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
        shadows: bool = True,
        ambient_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
        background_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
        shadow_bias: float = 1.0e-6,
    ) -> None:
        _require_cuda_direct_light_deps()
        wp.init()
        self.device = wp.get_device("cuda:0" if device is None else device)
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
        self._execution_context = _new_execution_context(self.device)

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
        self._validate_request(
            snapshot,
            bvh,
            frame_id=spec.frame_id,
            env_idx=spec.env_idx,
            max_distance=spec.max_distance,
        )
        self._wait_for_inputs(snapshot, bvh)
        upload_start = time.perf_counter()
        with torch.cuda.device(self._execution_context.device):
            origins = torch.as_tensor(
                np.asarray(spec.origins_world, dtype=np.float32),
                dtype=torch.float32,
                device=self._execution_context.device,
            ).contiguous()
            directions = torch.as_tensor(
                np.asarray(spec.directions_world, dtype=np.float32),
                dtype=torch.float32,
                device=self._execution_context.device,
            ).contiguous()
        _add_profile_ms(render_profile, "cuda_first_hit_ray_upload_ms", upload_start)
        return self._execute_direct_light(
            snapshot,
            bvh,
            origins,
            directions,
            camera=None,
            frame_id=spec.frame_id,
            sim_time=spec.sim_time,
            env_idx=spec.env_idx,
            sensor_id=spec.sensor_id,
            max_distance=spec.max_distance,
            sensor_role=spec.sensor_role,
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
        """Generate camera rays on CUDA and reuse the direct-light shade path."""

        output_profile = self._validate_output_profile(output_profile)
        self._validate_request(
            snapshot,
            bvh,
            frame_id=camera.frame_id,
            env_idx=camera.env_idx,
            max_distance=camera.max_distance,
        )
        self._wait_for_inputs(snapshot, bvh)
        params_start = time.perf_counter()
        camera_parameters = CudaCameraParameters.from_spec(camera)
        _add_profile_ms(render_profile, "raygen_camera_params_ms", params_start)
        return self._execute_direct_light(
            snapshot,
            bvh,
            None,
            None,
            camera=camera_parameters,
            frame_id=camera.frame_id,
            sim_time=camera.sim_time,
            env_idx=camera.env_idx,
            sensor_id=camera.sensor_id,
            max_distance=camera.max_distance,
            sensor_role=camera.sensor_role,
            output_profile=output_profile,
            render_profile=render_profile,
        )

    def _validate_output_profile(self, output_profile: OpticalOutputProfile | str) -> OpticalOutputProfile:
        output_profile = normalize_output_profile(output_profile)
        if output_profile not in self.supported_profiles:
            raise ValueError(f"cuda_direct_light does not support output_profile={output_profile.value!r}")
        return output_profile

    def _execute_direct_light(
        self,
        snapshot: DeviceOpticalSceneSnapshot,
        bvh: DeviceOpticalBvh,
        origins: object | None,
        directions: object | None,
        *,
        camera: CudaCameraParameters | None,
        frame_id: int,
        sim_time: float,
        env_idx: int,
        sensor_id: str,
        max_distance: float,
        sensor_role: str,
        output_profile: OpticalOutputProfile,
        render_profile: list[tuple[str, float]] | None,
    ) -> OpticalComputeResult:
        load_start = time.perf_counter()
        renderer = CudaDirectLightNativeRenderer.prepare(self._execution_context)
        _add_profile_ms(render_profile, "cuda_direct_light_load_extension_ms", load_start)

        convert_start = time.perf_counter()
        first_hit_inputs = _first_hit_inputs_from_snapshot(
            snapshot,
            bvh,
            max_distance=max_distance,
            sensor_role=sensor_role,
        )
        shade_inputs = _shade_inputs_from_snapshot(snapshot)
        _add_profile_ms(render_profile, "cuda_direct_light_channel_views_ms", convert_start)

        alloc_start = time.perf_counter()
        num_rays = camera.width * camera.height if camera is not None else int(origins.shape[0])
        buffers = allocate_cuda_direct_light_buffers(
            num_rays=num_rays,
            device=self._execution_context.device,
            origins=origins,
            directions=directions,
            direct_light=True,
            traversal_diagnostics=render_profile is not None,
        )
        _add_profile_ms(render_profile, "cuda_direct_light_buffer_alloc_ms", alloc_start)

        stage = renderer.bind_direct_light(
            first_hit_inputs=first_hit_inputs,
            shade_inputs=shade_inputs,
            buffers=buffers,
            camera=camera,
            output_profile=output_profile,
            shadows=self.shadows,
            ambient_rgb=self.ambient_rgb,
            background_rgb=self.background_rgb,
            shadow_bias=self.shadow_bias,
        )
        phase_names = ("raygen_kernel",) if render_profile is not None and camera is not None else ()
        completion_event, diagnostic_events = _prepare_compatibility_events(
            self._execution_context,
            phase_names,
        )
        transaction = self._execution_context.begin_transaction(
            stages=(stage,),
            completion_event=completion_event,
            diagnostic_events=diagnostic_events,
        )
        execute_start = time.perf_counter()
        stage.enqueue(transaction)
        completion = transaction.finish()
        completion.event.synchronize()
        _add_profile_ms(render_profile, "cuda_direct_light_transaction_sync_ms", execute_start)
        if render_profile is not None and completion.deferred_profile is not None:
            render_profile.extend(completion.deferred_profile.resolve_ms())

        channels = stage.output_channels(include_traversal_diagnostics=render_profile is not None)
        return OpticalComputeResult(
            frame_id=frame_id,
            sim_time=sim_time,
            env_idx=env_idx,
            sensor_id=sensor_id,
            location="device",
            channels=channels,
            output_profile=output_profile,
            ready_event=None,
            resources=buffers.resource_tuple(),
        )

    def _validate_request(
        self,
        snapshot: DeviceOpticalSceneSnapshot,
        bvh: DeviceOpticalBvh,
        *,
        frame_id: int,
        env_idx: int,
        max_distance: float,
    ) -> None:
        CudaDeviceBvhOpticalExecutor._validate_request(
            self,
            snapshot,
            bvh,
            frame_id=frame_id,
            env_idx=env_idx,
            max_distance=max_distance,
        )

    @staticmethod
    def _wait_for_inputs(snapshot: DeviceOpticalSceneSnapshot, bvh: DeviceOpticalBvh) -> None:
        CudaDeviceBvhOpticalExecutor._wait_for_inputs(snapshot, bvh)


class _CudaEventTiming:
    __slots__ = ("start", "end")

    def __init__(self, start, end) -> None:
        self.start = start
        self.end = end


def _generate_pinhole_camera_ray_tensors(
    camera: OpticalPinholeCameraSpec,
    *,
    device: "torch.device",
    render_profile: list[tuple[str, float]] | None = None,
):
    """Generate flat row-major pinhole rays on an explicit CUDA device."""

    _require_cuda_direct_light_deps()
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("CUDA camera raygen requires a CUDA torch.device")

    params_start = time.perf_counter()
    origin = np.asarray(camera.X_world_camera.r, dtype=np.float32)
    rotation = np.asarray(camera.X_world_camera.R, dtype=np.float32)
    _add_profile_ms(render_profile, "raygen_camera_params_ms", params_start)

    load_start = time.perf_counter()
    module = _load_cuda_direct_light_extension()
    _add_profile_ms(render_profile, "cuda_raygen_load_extension_ms", load_start)

    alloc_start = time.perf_counter()
    num_rays = int(camera.width) * int(camera.height)
    with torch.cuda.device(device):
        origins = torch.empty((num_rays, 3), dtype=torch.float32, device=device)
        directions = torch.empty((num_rays, 3), dtype=torch.float32, device=device)
    _add_profile_ms(render_profile, "raygen_buffer_alloc_ms", alloc_start)

    timing = None
    stream = torch.cuda.current_stream(device)
    if render_profile is not None:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream)

    module.pinhole_camera_rays_out(
        origins,
        directions,
        int(camera.width),
        int(camera.height),
        float(camera.fx),
        float(camera.fy),
        float(camera.cx),
        float(camera.cy),
        float(origin[0]),
        float(origin[1]),
        float(origin[2]),
        float(rotation[0, 0]),
        float(rotation[0, 1]),
        float(rotation[0, 2]),
        float(rotation[1, 0]),
        float(rotation[1, 1]),
        float(rotation[1, 2]),
        float(rotation[2, 0]),
        float(rotation[2, 1]),
        float(rotation[2, 2]),
    )
    if render_profile is not None:
        end_event.record(stream)
        timing = _CudaEventTiming(start_event, end_event)
    return origins, directions, timing


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
        name="robot_sim_cuda_direct_light_v8",
        cpp_sources=[_CPP_SOURCE],
        cuda_sources=[_CUDA_SOURCE],
        functions=[
            "pinhole_camera_rays_out",
            "first_hit_rays_out",
            "shade_direct_light_no_shadow_out",
            "shade_direct_light_no_shadow_diagnostics_out",
            "shade_direct_light_with_shadows_out",
            "shade_direct_light_with_shadows_diagnostics_out",
        ],
        with_cuda=True,
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        verbose=False,
    )


_CPP_SOURCE = r"""
#include <torch/extension.h>

void pinhole_camera_rays_out(
    torch::Tensor origins,
    torch::Tensor directions,
    int64_t width,
    int64_t height,
    double fx,
    double fy,
    double cx,
    double cy,
    double origin_x,
    double origin_y,
    double origin_z,
    double r00,
    double r01,
    double r02,
    double r10,
    double r11,
    double r12,
    double r20,
    double r21,
    double r22);

void first_hit_rays_out(
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
    int64_t sensor_role_mask,
    torch::Tensor hit_mask,
    torch::Tensor range_m,
    torch::Tensor position_world,
    torch::Tensor normal_world,
    torch::Tensor numeric_instance_id,
    torch::Tensor material_index,
    torch::Tensor bvh_stack_overflow_count,
    torch::Tensor bvh_max_stack_depth);

void shade_direct_light_no_shadow_out(
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
    double background_b,
    torch::Tensor rgb,
    torch::Tensor intensity,
    torch::Tensor shadow_stack_overflow_count,
    torch::Tensor shadow_max_stack_depth);

void shade_direct_light_no_shadow_diagnostics_out(
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
    double background_b,
    torch::Tensor rgb,
    torch::Tensor intensity,
    torch::Tensor shadow_stack_overflow_count,
    torch::Tensor shadow_max_stack_depth,
    torch::Tensor traversal_counter_0,
    torch::Tensor traversal_counter_1,
    torch::Tensor traversal_counter_2,
    torch::Tensor traversal_counter_3,
    torch::Tensor traversal_counter_4,
    torch::Tensor traversal_counter_5,
    torch::Tensor traversal_counter_6,
    torch::Tensor traversal_counter_7,
    torch::Tensor traversal_counter_8);

void shade_direct_light_with_shadows_out(
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
    double background_b,
    torch::Tensor rgb,
    torch::Tensor intensity,
    torch::Tensor shadow_stack_overflow_count,
    torch::Tensor shadow_max_stack_depth);

void shade_direct_light_with_shadows_diagnostics_out(
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
    double background_b,
    torch::Tensor rgb,
    torch::Tensor intensity,
    torch::Tensor shadow_stack_overflow_count,
    torch::Tensor shadow_max_stack_depth,
    torch::Tensor traversal_counter_0,
    torch::Tensor traversal_counter_1,
    torch::Tensor traversal_counter_2,
    torch::Tensor traversal_counter_3,
    torch::Tensor traversal_counter_4,
    torch::Tensor traversal_counter_5,
    torch::Tensor traversal_counter_6,
    torch::Tensor traversal_counter_7,
    torch::Tensor traversal_counter_8);
"""


_CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <cstdint>
#include <limits>
namespace {

constexpr float kBuildEps = 1.0e-8f;
constexpr float kDirEps = 1.0e-8f;
constexpr float kTEps = 1.0e-5f;
constexpr int kMaxBvhStack = 32;

__global__ void pinhole_camera_rays_kernel(
    int width,
    int height,
    float fx,
    float fy,
    float cx,
    float cy,
    float origin_x,
    float origin_y,
    float origin_z,
    float r00,
    float r01,
    float r02,
    float r10,
    float r11,
    float r12,
    float r20,
    float r21,
    float r22,
    float* origins,
    float* directions) {
  int ray = blockIdx.x * blockDim.x + threadIdx.x;
  int num_rays = width * height;
  if (ray >= num_rays) {
    return;
  }

  int px = ray % width;
  int py = ray / width;
  float camera_x = (static_cast<float>(px) - cx) / fx;
  float camera_y = (static_cast<float>(py) - cy) / fy;
  float inv_norm = rsqrtf(camera_x * camera_x + camera_y * camera_y + 1.0f);
  camera_x *= inv_norm;
  camera_y *= inv_norm;
  float camera_z = inv_norm;

  origins[ray * 3 + 0] = origin_x;
  origins[ray * 3 + 1] = origin_y;
  origins[ray * 3 + 2] = origin_z;
  directions[ray * 3 + 0] = r00 * camera_x + r01 * camera_y + r02 * camera_z;
  directions[ray * 3 + 1] = r10 * camera_x + r11 * camera_y + r12 * camera_z;
  directions[ray * 3 + 2] = r20 * camera_x + r21 * camera_y + r22 * camera_z;
}

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

  hit_mask[ray] = 0;
  range_m[ray] = INFINITY;
  position_world[ray * 3 + 0] = NAN;
  position_world[ray * 3 + 1] = NAN;
  position_world[ray * 3 + 2] = NAN;
  normal_world[ray * 3 + 0] = NAN;
  normal_world[ray * 3 + 1] = NAN;
  normal_world[ray * 3 + 2] = NAN;
  numeric_instance_id[ray] = 0;
  material_index[ray] = 0;

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
    int32_t* shadow_max_stack_depth,
    int traversal_stats_enabled,
    int32_t* shadow_traversal_node_visit_count,
    int32_t* shadow_traversal_leaf_visit_count,
    int32_t* shadow_traversal_triangle_test_count) {
  if (max_distance <= 0.0f || num_bvh_nodes <= 0) {
    return false;
  }

  int stack[kMaxBvhStack];
  int stack_size = 0;
  int local_max_stack = 0;
  int local_node_visit_count = 0;
  int local_leaf_visit_count = 0;
  int local_triangle_test_count = 0;
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
    ++local_node_visit_count;
    int leaf_count = bvh_count[node];
    if (leaf_count > 0) {
      ++local_leaf_visit_count;
      int leaf_start = bvh_start[node];
      for (int offset = 0; offset < leaf_count; ++offset) {
        int tri = bvh_prim_ids[leaf_start + offset];
        if ((triangle_role_masks[tri] & sensor_role_mask) == 0) {
          continue;
        }
        ++local_triangle_test_count;
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
  if (traversal_stats_enabled != 0) {
    atomicAdd(shadow_traversal_node_visit_count, local_node_visit_count);
    atomicAdd(shadow_traversal_leaf_visit_count, local_leaf_visit_count);
    atomicAdd(shadow_traversal_triangle_test_count, local_triangle_test_count);
  }
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
    int traversal_stats_enabled,
    int32_t* shadow_traversal_ray_count,
    int32_t* shadow_traversal_directional_ray_count,
    int32_t* shadow_traversal_point_ray_count,
    int32_t* shadow_traversal_occluded_count,
    int32_t* shadow_traversal_unoccluded_count,
    int32_t* shadow_traversal_node_visit_count,
    int32_t* shadow_traversal_leaf_visit_count,
    int32_t* shadow_traversal_triangle_test_count,
    int32_t* shadow_traversal_plane_test_count,
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
        if (traversal_stats_enabled != 0) {
          atomicAdd(shadow_traversal_ray_count, 1);
          if (light_kind[light] == 1) {
            atomicAdd(shadow_traversal_point_ray_count, 1);
          } else {
            atomicAdd(shadow_traversal_directional_ray_count, 1);
          }
        }
        // P12.3d treats triangle meshes as CUDA shadow occluders. Planes can
        // still be primary hits but are intentionally non-occluders here.
        bool occluded = shadow_occluded_by_triangles(
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
            shadow_max_stack_depth,
            traversal_stats_enabled,
            shadow_traversal_node_visit_count,
            shadow_traversal_leaf_visit_count,
            shadow_traversal_triangle_test_count);
        if (traversal_stats_enabled != 0) {
          if (occluded) {
            atomicAdd(shadow_traversal_occluded_count, 1);
          } else {
            atomicAdd(shadow_traversal_unoccluded_count, 1);
          }
        }
        bool visible_to_light = !occluded;
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

void check_float_1d(torch::Tensor value, const char* name) {
  TORCH_CHECK(value.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(value.dtype() == torch::kFloat32, name, " must be float32");
  TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(value.dim() == 1, name, " must be rank-1");
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

void check_same_device(torch::Tensor anchor, torch::Tensor value, const char* name) {
  TORCH_CHECK(value.device() == anchor.device(), name, " must be on ", anchor.device());
}

}  // namespace

void pinhole_camera_rays_out(
    torch::Tensor origins,
    torch::Tensor directions,
    int64_t width,
    int64_t height,
    double fx,
    double fy,
    double cx,
    double cy,
    double origin_x,
    double origin_y,
    double origin_z,
    double r00,
    double r01,
    double r02,
    double r10,
    double r11,
    double r12,
    double r20,
    double r21,
    double r22) {
  check_float_2d_3(origins, "origins");
  check_float_2d_3(directions, "directions");
  TORCH_CHECK(directions.sizes() == origins.sizes(), "directions must match origins shape");
  check_same_device(origins, directions, "directions");
  TORCH_CHECK(width > 0 && height > 0, "width and height must be > 0");
  TORCH_CHECK(width <= INT32_MAX && height <= INT32_MAX, "width or height is too large");
  TORCH_CHECK(width <= INT32_MAX / height, "camera ray count exceeds int32 range");
  int64_t num_rays = width * height;
  TORCH_CHECK(origins.size(0) == num_rays, "ray outputs must have width * height rows");
  TORCH_CHECK(std::isfinite(fx) && fx > 0.0, "fx must be finite and > 0");
  TORCH_CHECK(std::isfinite(fy) && fy > 0.0, "fy must be finite and > 0");
  TORCH_CHECK(std::isfinite(cx) && std::isfinite(cy), "cx and cy must be finite");

  const c10::cuda::CUDAGuard device_guard(origins.device());
  if (num_rays > 0) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(origins.get_device());
    int block = 256;
    int grid = (static_cast<int>(num_rays) + block - 1) / block;
    pinhole_camera_rays_kernel<<<grid, block, 0, stream>>>(
        static_cast<int>(width),
        static_cast<int>(height),
        static_cast<float>(fx),
        static_cast<float>(fy),
        static_cast<float>(cx),
        static_cast<float>(cy),
        static_cast<float>(origin_x),
        static_cast<float>(origin_y),
        static_cast<float>(origin_z),
        static_cast<float>(r00),
        static_cast<float>(r01),
        static_cast<float>(r02),
        static_cast<float>(r10),
        static_cast<float>(r11),
        static_cast<float>(r12),
        static_cast<float>(r20),
        static_cast<float>(r21),
        static_cast<float>(r22),
        origins.data_ptr<float>(),
        directions.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void first_hit_rays_out(
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
    int64_t sensor_role_mask,
    torch::Tensor hit_mask,
    torch::Tensor range_m,
    torch::Tensor position_world,
    torch::Tensor normal_world,
    torch::Tensor numeric_instance_id,
    torch::Tensor material_index,
    torch::Tensor bvh_stack_overflow_count,
    torch::Tensor bvh_max_stack_depth) {
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
  check_i32_1d(hit_mask, "hit_mask");
  check_float_1d(range_m, "range_m");
  check_float_2d_3(position_world, "position_world");
  check_float_2d_3(normal_world, "normal_world");
  check_i32_1d(numeric_instance_id, "numeric_instance_id");
  check_i32_1d(material_index, "material_index");
  check_i32_1d(bvh_stack_overflow_count, "bvh_stack_overflow_count");
  check_i32_1d(bvh_max_stack_depth, "bvh_max_stack_depth");
  check_same_device(origins, directions, "directions");
  check_same_device(origins, plane_normals, "plane_normals");
  check_same_device(origins, plane_points, "plane_points");
  check_same_device(origins, plane_numeric_ids, "plane_numeric_ids");
  check_same_device(origins, plane_source_keys, "plane_source_keys");
  check_same_device(origins, plane_role_masks, "plane_role_masks");
  check_same_device(origins, plane_material_indices, "plane_material_indices");
  check_same_device(origins, triangle_v0, "triangle_v0");
  check_same_device(origins, triangle_e1, "triangle_e1");
  check_same_device(origins, triangle_e2, "triangle_e2");
  check_same_device(origins, triangle_normal, "triangle_normal");
  check_same_device(origins, triangle_numeric_ids, "triangle_numeric_ids");
  check_same_device(origins, triangle_source_keys, "triangle_source_keys");
  check_same_device(origins, triangle_role_masks, "triangle_role_masks");
  check_same_device(origins, triangle_material_indices, "triangle_material_indices");
  check_same_device(origins, bvh_bounds_min, "bvh_bounds_min");
  check_same_device(origins, bvh_bounds_max, "bvh_bounds_max");
  check_same_device(origins, bvh_left, "bvh_left");
  check_same_device(origins, bvh_right, "bvh_right");
  check_same_device(origins, bvh_start, "bvh_start");
  check_same_device(origins, bvh_count, "bvh_count");
  check_same_device(origins, bvh_prim_ids, "bvh_prim_ids");
  check_same_device(origins, hit_mask, "hit_mask");
  check_same_device(origins, range_m, "range_m");
  check_same_device(origins, position_world, "position_world");
  check_same_device(origins, normal_world, "normal_world");
  check_same_device(origins, numeric_instance_id, "numeric_instance_id");
  check_same_device(origins, material_index, "material_index");
  check_same_device(origins, bvh_stack_overflow_count, "bvh_stack_overflow_count");
  check_same_device(origins, bvh_max_stack_depth, "bvh_max_stack_depth");

  int64_t num_rays = origins.size(0);
  int64_t num_triangles = triangle_v0.size(0);
  TORCH_CHECK(num_rays <= static_cast<int64_t>(INT32_MAX), "too many rays");
  TORCH_CHECK(num_planes >= 0 && num_planes <= plane_normals.size(0), "num_planes out of range");
  TORCH_CHECK(num_planes <= plane_points.size(0), "num_planes exceeds plane_points");
  TORCH_CHECK(num_planes <= plane_numeric_ids.size(0), "num_planes exceeds plane_numeric_ids");
  TORCH_CHECK(num_planes <= plane_source_keys.size(0), "num_planes exceeds plane_source_keys");
  TORCH_CHECK(num_planes <= plane_role_masks.size(0), "num_planes exceeds plane_role_masks");
  TORCH_CHECK(
      num_planes <= plane_material_indices.size(0), "num_planes exceeds plane_material_indices");
  TORCH_CHECK(triangle_e1.size(0) == num_triangles, "triangle_e1 length mismatch");
  TORCH_CHECK(triangle_e2.size(0) == num_triangles, "triangle_e2 length mismatch");
  TORCH_CHECK(triangle_normal.size(0) == num_triangles, "triangle_normal length mismatch");
  TORCH_CHECK(triangle_numeric_ids.size(0) == num_triangles, "triangle_numeric_ids length mismatch");
  TORCH_CHECK(triangle_source_keys.size(0) == num_triangles, "triangle_source_keys length mismatch");
  TORCH_CHECK(triangle_role_masks.size(0) == num_triangles, "triangle_role_masks length mismatch");
  TORCH_CHECK(
      triangle_material_indices.size(0) == num_triangles,
      "triangle_material_indices length mismatch");
  TORCH_CHECK(num_bvh_nodes >= 0 && num_bvh_nodes <= bvh_bounds_min.size(0), "num_bvh_nodes out of range");
  TORCH_CHECK(num_bvh_nodes <= bvh_bounds_max.size(0), "num_bvh_nodes exceeds bvh_bounds_max");
  TORCH_CHECK(num_bvh_nodes <= bvh_left.size(0), "num_bvh_nodes exceeds bvh_left");
  TORCH_CHECK(num_bvh_nodes <= bvh_right.size(0), "num_bvh_nodes exceeds bvh_right");
  TORCH_CHECK(num_bvh_nodes <= bvh_start.size(0), "num_bvh_nodes exceeds bvh_start");
  TORCH_CHECK(num_bvh_nodes <= bvh_count.size(0), "num_bvh_nodes exceeds bvh_count");
  TORCH_CHECK(bvh_prim_ids.size(0) >= num_triangles, "bvh_prim_ids is shorter than triangles");
  TORCH_CHECK(hit_mask.size(0) == num_rays, "hit_mask length must match origins");
  TORCH_CHECK(range_m.size(0) == num_rays, "range_m length must match origins");
  TORCH_CHECK(position_world.size(0) == num_rays, "position_world length must match origins");
  TORCH_CHECK(normal_world.size(0) == num_rays, "normal_world length must match origins");
  TORCH_CHECK(numeric_instance_id.size(0) == num_rays, "numeric_instance_id length must match origins");
  TORCH_CHECK(material_index.size(0) == num_rays, "material_index length must match origins");
  TORCH_CHECK(bvh_stack_overflow_count.size(0) == 1, "bvh_stack_overflow_count must have shape [1]");
  TORCH_CHECK(bvh_max_stack_depth.size(0) == 1, "bvh_max_stack_depth must have shape [1]");

  const c10::cuda::CUDAGuard device_guard(hit_mask.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(hit_mask.get_device());
  C10_CUDA_CHECK(cudaMemsetAsync(bvh_stack_overflow_count.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(bvh_max_stack_depth.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));

  if (num_rays > 0) {
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
}

void shade_direct_light_no_shadow_impl(
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
    double background_b,
    torch::Tensor rgb,
    torch::Tensor intensity,
    torch::Tensor shadow_stack_overflow_count,
    torch::Tensor shadow_max_stack_depth,
    torch::Tensor* traversal_counter_0,
    torch::Tensor* traversal_counter_1,
    torch::Tensor* traversal_counter_2,
    torch::Tensor* traversal_counter_3,
    torch::Tensor* traversal_counter_4,
    torch::Tensor* traversal_counter_5,
    torch::Tensor* traversal_counter_6,
    torch::Tensor* traversal_counter_7,
    torch::Tensor* traversal_counter_8) {
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
  check_float_2d_3(rgb, "rgb");
  check_float_1d(intensity, "intensity");
  check_i32_1d(shadow_stack_overflow_count, "shadow_stack_overflow_count");
  check_i32_1d(shadow_max_stack_depth, "shadow_max_stack_depth");
  torch::Tensor* traversal_counters[9] = {
      traversal_counter_0,
      traversal_counter_1,
      traversal_counter_2,
      traversal_counter_3,
      traversal_counter_4,
      traversal_counter_5,
      traversal_counter_6,
      traversal_counter_7,
      traversal_counter_8};
  bool traversal_stats_enabled = traversal_counter_0 != nullptr;
  for (torch::Tensor* counter : traversal_counters) {
    TORCH_CHECK(
        (counter != nullptr) == traversal_stats_enabled,
        "shadow traversal counters must be completely absent or present");
    if (counter != nullptr) {
      check_i32_1d(*counter, "shadow_traversal_counter");
      TORCH_CHECK(counter->size(0) == 1, "shadow traversal counters must have shape [1]");
    }
  }
  TORCH_CHECK(position_world.sizes() == normal_world.sizes(), "position_world/normal_world shape mismatch");
  TORCH_CHECK(hit_mask.size(0) == position_world.size(0), "hit_mask length must match position_world");
  TORCH_CHECK(
      material_index.size(0) == position_world.size(0),
      "material_index length must match position_world");
  TORCH_CHECK(num_lights >= 0 && num_lights <= light_kind.size(0), "num_lights out of range");
  TORCH_CHECK(num_lights <= light_position_or_direction_world.size(0), "num_lights exceeds light vectors");
  TORCH_CHECK(num_lights <= light_intensity.size(0), "num_lights exceeds light intensities");
  TORCH_CHECK(num_lights <= light_color_rgb.size(0), "num_lights exceeds light colors");
  TORCH_CHECK(
      std::isfinite(ambient_r) && std::isfinite(ambient_g) && std::isfinite(ambient_b),
      "ambient RGB must be finite");
  TORCH_CHECK(
      std::isfinite(background_r) && std::isfinite(background_g) && std::isfinite(background_b),
      "background RGB must be finite");
  check_same_device(hit_mask, position_world, "position_world");
  check_same_device(hit_mask, normal_world, "normal_world");
  check_same_device(hit_mask, material_index, "material_index");
  check_same_device(hit_mask, material_albedo_rgb, "material_albedo_rgb");
  check_same_device(hit_mask, light_kind, "light_kind");
  check_same_device(hit_mask, light_position_or_direction_world, "light_position_or_direction_world");
  check_same_device(hit_mask, light_intensity, "light_intensity");
  check_same_device(hit_mask, light_color_rgb, "light_color_rgb");
  check_same_device(hit_mask, rgb, "rgb");
  check_same_device(hit_mask, intensity, "intensity");
  check_same_device(hit_mask, shadow_stack_overflow_count, "shadow_stack_overflow_count");
  check_same_device(hit_mask, shadow_max_stack_depth, "shadow_max_stack_depth");
  for (torch::Tensor* counter : traversal_counters) {
    if (counter != nullptr) {
      check_same_device(hit_mask, *counter, "shadow_traversal_counter");
    }
  }

  int64_t num_rays = hit_mask.size(0);
  TORCH_CHECK(num_rays <= static_cast<int64_t>(INT32_MAX), "too many rays");
  TORCH_CHECK(rgb.size(0) == num_rays, "rgb length must match hit_mask");
  TORCH_CHECK(intensity.size(0) == num_rays, "intensity length must match hit_mask");
  TORCH_CHECK(shadow_stack_overflow_count.size(0) == 1, "shadow_stack_overflow_count must have shape [1]");
  TORCH_CHECK(shadow_max_stack_depth.size(0) == 1, "shadow_max_stack_depth must have shape [1]");

  const c10::cuda::CUDAGuard device_guard(rgb.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(rgb.get_device());
  C10_CUDA_CHECK(cudaMemsetAsync(
      shadow_stack_overflow_count.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(shadow_max_stack_depth.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
  for (torch::Tensor* counter : traversal_counters) {
    if (counter != nullptr) {
      C10_CUDA_CHECK(cudaMemsetAsync(counter->data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
    }
  }

  if (num_rays > 0) {
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
}

void shade_direct_light_no_shadow_out(
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
    double background_b,
    torch::Tensor rgb,
    torch::Tensor intensity,
    torch::Tensor shadow_stack_overflow_count,
    torch::Tensor shadow_max_stack_depth) {
  shade_direct_light_no_shadow_impl(
      hit_mask,
      position_world,
      normal_world,
      material_index,
      material_albedo_rgb,
      light_kind,
      light_position_or_direction_world,
      light_intensity,
      light_color_rgb,
      num_lights,
      ambient_r,
      ambient_g,
      ambient_b,
      background_r,
      background_g,
      background_b,
      rgb,
      intensity,
      shadow_stack_overflow_count,
      shadow_max_stack_depth,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr);
}

void shade_direct_light_no_shadow_diagnostics_out(
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
    double background_b,
    torch::Tensor rgb,
    torch::Tensor intensity,
    torch::Tensor shadow_stack_overflow_count,
    torch::Tensor shadow_max_stack_depth,
    torch::Tensor traversal_counter_0,
    torch::Tensor traversal_counter_1,
    torch::Tensor traversal_counter_2,
    torch::Tensor traversal_counter_3,
    torch::Tensor traversal_counter_4,
    torch::Tensor traversal_counter_5,
    torch::Tensor traversal_counter_6,
    torch::Tensor traversal_counter_7,
    torch::Tensor traversal_counter_8) {
  shade_direct_light_no_shadow_impl(
      hit_mask,
      position_world,
      normal_world,
      material_index,
      material_albedo_rgb,
      light_kind,
      light_position_or_direction_world,
      light_intensity,
      light_color_rgb,
      num_lights,
      ambient_r,
      ambient_g,
      ambient_b,
      background_r,
      background_g,
      background_b,
      rgb,
      intensity,
      shadow_stack_overflow_count,
      shadow_max_stack_depth,
      &traversal_counter_0,
      &traversal_counter_1,
      &traversal_counter_2,
      &traversal_counter_3,
      &traversal_counter_4,
      &traversal_counter_5,
      &traversal_counter_6,
      &traversal_counter_7,
      &traversal_counter_8);
}

void shade_direct_light_with_shadows_impl(
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
    double background_b,
    torch::Tensor rgb,
    torch::Tensor intensity,
    torch::Tensor shadow_stack_overflow_count,
    torch::Tensor shadow_max_stack_depth,
    torch::Tensor* traversal_counter_0,
    torch::Tensor* traversal_counter_1,
    torch::Tensor* traversal_counter_2,
    torch::Tensor* traversal_counter_3,
    torch::Tensor* traversal_counter_4,
    torch::Tensor* traversal_counter_5,
    torch::Tensor* traversal_counter_6,
    torch::Tensor* traversal_counter_7,
    torch::Tensor* traversal_counter_8) {
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
  check_float_2d_3(rgb, "rgb");
  check_float_1d(intensity, "intensity");
  check_i32_1d(shadow_stack_overflow_count, "shadow_stack_overflow_count");
  check_i32_1d(shadow_max_stack_depth, "shadow_max_stack_depth");
  torch::Tensor* traversal_counters[9] = {
      traversal_counter_0,
      traversal_counter_1,
      traversal_counter_2,
      traversal_counter_3,
      traversal_counter_4,
      traversal_counter_5,
      traversal_counter_6,
      traversal_counter_7,
      traversal_counter_8};
  bool traversal_stats_enabled = traversal_counter_0 != nullptr;
  for (torch::Tensor* counter : traversal_counters) {
    TORCH_CHECK(
        (counter != nullptr) == traversal_stats_enabled,
        "shadow traversal counters must be completely absent or present");
    if (counter != nullptr) {
      check_i32_1d(*counter, "shadow_traversal_counter");
      TORCH_CHECK(counter->size(0) == 1, "shadow traversal counters must have shape [1]");
    }
  }
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
  int64_t num_triangles = triangle_v0.size(0);
  TORCH_CHECK(triangle_e1.size(0) == num_triangles, "triangle_e1 length mismatch");
  TORCH_CHECK(triangle_e2.size(0) == num_triangles, "triangle_e2 length mismatch");
  TORCH_CHECK(triangle_normal.size(0) == num_triangles, "triangle_normal length mismatch");
  TORCH_CHECK(triangle_role_masks.size(0) == num_triangles, "triangle_role_masks length mismatch");
  TORCH_CHECK(num_bvh_nodes >= 0 && num_bvh_nodes <= bvh_bounds_min.size(0), "num_bvh_nodes out of range");
  TORCH_CHECK(num_bvh_nodes <= bvh_bounds_max.size(0), "num_bvh_nodes exceeds bvh bounds");
  TORCH_CHECK(num_bvh_nodes <= bvh_left.size(0), "num_bvh_nodes exceeds bvh left");
  TORCH_CHECK(num_bvh_nodes <= bvh_right.size(0), "num_bvh_nodes exceeds bvh right");
  TORCH_CHECK(num_bvh_nodes <= bvh_start.size(0), "num_bvh_nodes exceeds bvh start");
  TORCH_CHECK(num_bvh_nodes <= bvh_count.size(0), "num_bvh_nodes exceeds bvh count");
  TORCH_CHECK(bvh_prim_ids.size(0) >= num_triangles, "bvh_prim_ids is shorter than triangles");
  TORCH_CHECK(std::isfinite(shadow_bias) && shadow_bias >= 0.0, "shadow_bias must be finite and >= 0");
  TORCH_CHECK(
      std::isfinite(ambient_r) && std::isfinite(ambient_g) && std::isfinite(ambient_b),
      "ambient RGB must be finite");
  TORCH_CHECK(
      std::isfinite(background_r) && std::isfinite(background_g) && std::isfinite(background_b),
      "background RGB must be finite");
  check_same_device(hit_mask, position_world, "position_world");
  check_same_device(hit_mask, normal_world, "normal_world");
  check_same_device(hit_mask, material_index, "material_index");
  check_same_device(hit_mask, material_albedo_rgb, "material_albedo_rgb");
  check_same_device(hit_mask, light_kind, "light_kind");
  check_same_device(hit_mask, light_position_or_direction_world, "light_position_or_direction_world");
  check_same_device(hit_mask, light_intensity, "light_intensity");
  check_same_device(hit_mask, light_color_rgb, "light_color_rgb");
  check_same_device(hit_mask, triangle_v0, "triangle_v0");
  check_same_device(hit_mask, triangle_e1, "triangle_e1");
  check_same_device(hit_mask, triangle_e2, "triangle_e2");
  check_same_device(hit_mask, triangle_normal, "triangle_normal");
  check_same_device(hit_mask, triangle_role_masks, "triangle_role_masks");
  check_same_device(hit_mask, bvh_bounds_min, "bvh_bounds_min");
  check_same_device(hit_mask, bvh_bounds_max, "bvh_bounds_max");
  check_same_device(hit_mask, bvh_left, "bvh_left");
  check_same_device(hit_mask, bvh_right, "bvh_right");
  check_same_device(hit_mask, bvh_start, "bvh_start");
  check_same_device(hit_mask, bvh_count, "bvh_count");
  check_same_device(hit_mask, bvh_prim_ids, "bvh_prim_ids");
  check_same_device(hit_mask, rgb, "rgb");
  check_same_device(hit_mask, intensity, "intensity");
  check_same_device(hit_mask, shadow_stack_overflow_count, "shadow_stack_overflow_count");
  check_same_device(hit_mask, shadow_max_stack_depth, "shadow_max_stack_depth");
  for (torch::Tensor* counter : traversal_counters) {
    if (counter != nullptr) {
      check_same_device(hit_mask, *counter, "shadow_traversal_counter");
    }
  }

  int64_t num_rays = hit_mask.size(0);
  TORCH_CHECK(num_rays <= static_cast<int64_t>(INT32_MAX), "too many rays");
  TORCH_CHECK(rgb.size(0) == num_rays, "rgb length must match hit_mask");
  TORCH_CHECK(intensity.size(0) == num_rays, "intensity length must match hit_mask");
  TORCH_CHECK(shadow_stack_overflow_count.size(0) == 1, "shadow_stack_overflow_count must have shape [1]");
  TORCH_CHECK(shadow_max_stack_depth.size(0) == 1, "shadow_max_stack_depth must have shape [1]");

  const c10::cuda::CUDAGuard device_guard(rgb.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(rgb.get_device());
  C10_CUDA_CHECK(cudaMemsetAsync(
      shadow_stack_overflow_count.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(shadow_max_stack_depth.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
  for (torch::Tensor* counter : traversal_counters) {
    if (counter != nullptr) {
      C10_CUDA_CHECK(cudaMemsetAsync(counter->data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
    }
  }
  int32_t* traversal_counter_ptrs[9];
  for (int counter = 0; counter < 9; ++counter) {
    traversal_counter_ptrs[counter] = traversal_stats_enabled
        ? traversal_counters[counter]->data_ptr<int32_t>()
        : nullptr;
  }

  if (num_rays > 0) {
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
        static_cast<int>(traversal_stats_enabled),
        traversal_counter_ptrs[0],
        traversal_counter_ptrs[1],
        traversal_counter_ptrs[2],
        traversal_counter_ptrs[3],
        traversal_counter_ptrs[4],
        traversal_counter_ptrs[5],
        traversal_counter_ptrs[6],
        traversal_counter_ptrs[7],
        traversal_counter_ptrs[8],
        static_cast<int>(num_rays));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void shade_direct_light_with_shadows_out(
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
    double background_b,
    torch::Tensor rgb,
    torch::Tensor intensity,
    torch::Tensor shadow_stack_overflow_count,
    torch::Tensor shadow_max_stack_depth) {
  shade_direct_light_with_shadows_impl(
      hit_mask,
      position_world,
      normal_world,
      material_index,
      material_albedo_rgb,
      light_kind,
      light_position_or_direction_world,
      light_intensity,
      light_color_rgb,
      num_lights,
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
      shadow_bias,
      ambient_r,
      ambient_g,
      ambient_b,
      background_r,
      background_g,
      background_b,
      rgb,
      intensity,
      shadow_stack_overflow_count,
      shadow_max_stack_depth,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr);
}

void shade_direct_light_with_shadows_diagnostics_out(
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
    double background_b,
    torch::Tensor rgb,
    torch::Tensor intensity,
    torch::Tensor shadow_stack_overflow_count,
    torch::Tensor shadow_max_stack_depth,
    torch::Tensor traversal_counter_0,
    torch::Tensor traversal_counter_1,
    torch::Tensor traversal_counter_2,
    torch::Tensor traversal_counter_3,
    torch::Tensor traversal_counter_4,
    torch::Tensor traversal_counter_5,
    torch::Tensor traversal_counter_6,
    torch::Tensor traversal_counter_7,
    torch::Tensor traversal_counter_8) {
  shade_direct_light_with_shadows_impl(
      hit_mask,
      position_world,
      normal_world,
      material_index,
      material_albedo_rgb,
      light_kind,
      light_position_or_direction_world,
      light_intensity,
      light_color_rgb,
      num_lights,
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
      shadow_bias,
      ambient_r,
      ambient_g,
      ambient_b,
      background_r,
      background_g,
      background_b,
      rgb,
      intensity,
      shadow_stack_overflow_count,
      shadow_max_stack_depth,
      &traversal_counter_0,
      &traversal_counter_1,
      &traversal_counter_2,
      &traversal_counter_3,
      &traversal_counter_4,
      &traversal_counter_5,
      &traversal_counter_6,
      &traversal_counter_7,
      &traversal_counter_8);
}
"""
