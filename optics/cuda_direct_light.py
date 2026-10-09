"""CUDA direct-light optical executors."""

from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ._cuda_direct_light_source import CPP_SOURCE as _CPP_SOURCE, CUDA_SOURCE as _CUDA_SOURCE
from .cuda_extension_recipes import cuda_extension_kwargs

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
    camera_dimensions: tuple[int, int] | None
    sensor_role_mask: int
    native_scalars_f32: tuple[str, ...]

    def validate_render_parameters(self, payload: dict[str, object]) -> None:
        """Compare a descriptive record to the actual frozen extension ABI."""
        if (
            self.camera_dimensions != (payload.get("width"), payload.get("height"))
            or self.sensor_role_mask != payload.get("sensor_role_mask")
            or self.output_profile != payload.get("profile")
            or self.shadows is not payload.get("shadows")
            or self.traversal_diagnostics is not payload.get("diagnostics")
            or list(self.native_scalars_f32) != payload.get("native_scalars_f32")
        ):
            raise ValueError("bound render parameters differ from actual CUDA scalar ABI")


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
        if getattr(self, "_sealed", False):
            raise AttributeError("CUDA render stage binding is immutable after validation")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("CUDA render stage binding is immutable after validation")
        object.__delattr__(self, name)

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
        self._prepared_evidence: CudaDirectLightStageEvidence | None = None
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
            expected = (
                self.buffers.origins,
                self.buffers.directions,
                self.camera.width,
                self.camera.height,
                self.camera.fx,
                self.camera.fy,
                self.camera.cx,
                self.camera.cy,
                *self.camera.origin,
                *self.camera.rotation,
            )
            if any(actual is not wanted for actual, wanted in zip(self._raygen_args[:2], expected[:2])):
                raise ValueError("CUDA raygen output ABI identity changed")
            if self._raygen_args[2:] != expected[2:]:
                raise ValueError("CUDA raygen scalar ABI changed")
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
            expected_args = self._bind_shade_arguments()
            if len(expected_args) != len(self._shade_args) or any(
                (actual != wanted if isinstance(wanted, (int, float)) else actual is not wanted)
                for actual, wanted in zip(self._shade_args, expected_args)
            ):
                raise ValueError("CUDA shade fixed ABI changed")
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
        if self._sealed or self._binding_signature is not None:
            raise ValueError("CUDA render binding signature cannot be rebuilt")
        self._prepared_evidence = self._build_stage_evidence()
        self._binding_signature = _render_binding_signature(self)
        self._sealed = True

    def _assert_cuda_transaction_stage_stable(self, context: CudaOpticalExecutionContext) -> None:
        if context is not self.renderer.context:
            raise ValueError("CUDA render stage execution context changed")
        if self._binding_signature is None or _render_binding_signature(self) != self._binding_signature:
            raise ValueError("CUDA render stage binding changed after validation")

    def _cuda_transaction_stage_evidence(self) -> CudaDirectLightStageEvidence:
        if not self._sealed or self._prepared_evidence is None:
            raise ValueError("CUDA render evidence requires a frozen binding")
        return self._prepared_evidence

    def _build_stage_evidence(self) -> CudaDirectLightStageEvidence:
        if self._sealed or self._prepared_evidence is not None:
            raise ValueError("CUDA render evidence cannot be rebuilt")
        # This is untimed construction/preflight, not an enqueue operation.
        # Read scalars from the bound tuples that will actually reach pybind.
        scalars = ()
        dimensions = None
        if self.camera is not None and self.shade_inputs is not None:
            self._validate_cuda_transaction_stage(self.renderer.context)
            dimensions = self._raygen_args[2:4]
            shade_start = 25 if self.shadows else 10
            # Off ABI has no bias argument; retain its inactive configuration.
            bias = self._shade_args[24] if self.shadows else self.shadow_bias
            values = (
                *self._raygen_args[4:],
                self.first_hit_inputs.max_distance,
                *self._shade_args[shade_start : shade_start + 6],
                bias,
            )
            scalars = tuple(struct.pack("<f", value).hex() for value in values)
        return CudaDirectLightStageEvidence(
            operation="first_hit" if self.shade_inputs is None else "direct_light",
            ray_source="camera" if self.camera is not None else "device_rays",
            output_profile=self.output_profile.value,
            shadows=self.shadows,
            extension_entry=self._shade_entry_name,
            num_rays=int(self.buffers.origins.shape[0]),
            traversal_diagnostics=bool(self.buffers.shadow_traversal_counters),
            camera_dimensions=dimensions,
            sensor_role_mask=self.first_hit_inputs.sensor_role_mask,
            native_scalars_f32=scalars,
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
        stage.first_hit_inputs.max_distance,
        stage.first_hit_inputs.sensor_role_mask,
        stage.first_hit_inputs.num_planes,
        stage.first_hit_inputs.num_bvh_nodes,
        id(stage.shade_inputs),
        None if stage.shade_inputs is None else stage.shade_inputs.num_lights,
        id(stage._prepared_evidence),
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
    return load_inline(**cuda_extension_kwargs('robot_sim_cuda_direct_light_v8'))
