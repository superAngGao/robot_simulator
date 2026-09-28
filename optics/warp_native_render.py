"""Single source of Warp optical kernel definitions.

The compatibility executors import these exact kernel objects. C4c1b's
caller-owned recorded-command ABI will also bind them here.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass, field
from math import prod
from pathlib import Path
from types import MappingProxyType

from .output_contract import SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES

try:  # pragma: no cover - exercised in GPU environments.
    import warp as wp
    from warp._src.types import array_t as _WarpArrayDescriptor

    _HAS_WARP = True
except Exception:  # pragma: no cover - keeps CPU-only imports working.
    wp = None
    _HAS_WARP = False

_BUILD_EPS = 1.0e-8
_DIR_EPS = 1.0e-8
_T_EPS = 1.0e-5
_ATTENUATION_EPS = 1.0e-12
_MAX_BVH_STACK = 32

WARP_RENDER_COMMAND_SCHEMA = "q54.p12.4.warp-render-command.v1"
WARP_RENDER_ABI = "warp_bvh_direct_light_recorded_v1"


def _render_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, allow_nan=False, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


def _render_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _render_f32(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("render scalar must be a finite float32 value")
    try:
        normalized = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("render scalar must be a finite float32 value") from exc
    if not math.isfinite(normalized):
        raise ValueError("render scalar must be a finite float32 value")
    return normalized


@dataclass(frozen=True, slots=True)
class WarpRenderQueryParameters:
    """Host-only query identity; this record grants no input or GPU authority."""

    num_rays: int
    frame_id: int
    sim_time: float
    env_idx: int
    sensor_id: str
    max_distance: float
    sensor_role: str
    sensor_role_mask: int

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in ("num_rays", "frame_id", "env_idx"):
            if type(getattr(self, name)) is not int or not 0 <= getattr(self, name) <= 2147483647:
                raise ValueError(f"{name} must be a nonnegative int32")
        for name in ("sensor_id", "sensor_role"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        if type(self.sim_time) is not float or not math.isfinite(self.sim_time) or self.sim_time < 0:
            raise ValueError("sim_time must be a finite nonnegative float")
        if (
            type(self.max_distance) is not float
            or _render_f32(self.max_distance) != self.max_distance
            or self.max_distance <= 0
        ):
            raise ValueError("max_distance must be a positive finite normalized float32")
        if type(self.sensor_role_mask) is not int or not 0 <= self.sensor_role_mask <= 9223372036854775807:
            raise ValueError("sensor_role_mask must be a nonnegative int64")

    @property
    def sha256(self) -> str:
        self.validate()
        return _render_digest(asdict(self))


@dataclass(frozen=True, slots=True)
class WarpCameraParameters:
    """Immutable normalized camera scalars, never a mutable host matrix owner."""

    query: WarpRenderQueryParameters
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    origin: tuple[float, float, float]
    rotation: tuple[float, ...]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if type(self.query) is not WarpRenderQueryParameters:
            raise TypeError("camera requires exact normalized query parameters")
        self.query.validate()
        if (
            type(self.width) is not int
            or type(self.height) is not int
            or self.width <= 0
            or self.height <= 0
            or self.width * self.height != self.query.num_rays
        ):
            raise ValueError("camera dimensions must be positive and match its ray count")
        if type(self.origin) is not tuple or len(self.origin) != 3:
            raise ValueError("camera origin must be an exact three-scalar tuple")
        if type(self.rotation) is not tuple or len(self.rotation) != 9:
            raise ValueError("camera rotation must be an exact nine-scalar tuple")
        for value in (self.fx, self.fy, self.cx, self.cy, *self.origin, *self.rotation):
            if type(value) is not float or _render_f32(value) != value:
                raise ValueError("camera scalars must be normalized finite float32")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("camera focal lengths must be positive")

    @classmethod
    def from_spec(cls, camera: object, *, role_table: object) -> "WarpCameraParameters":
        from sensing.optical import OpticalPinholeCameraSpec

        from .device_scene import DeviceOpticalRoleTable

        if type(camera) is not OpticalPinholeCameraSpec or type(role_table) is not DeviceOpticalRoleTable:
            raise TypeError("camera and role table must have their exact production types")
        if tuple(camera.X_world_camera.r.shape) != (3,) or tuple(camera.X_world_camera.R.shape) != (3, 3):
            raise ValueError("camera transform has invalid shape")
        return cls(
            query=WarpRenderQueryParameters(
                num_rays=camera.width * camera.height,
                frame_id=camera.frame_id,
                sim_time=float(camera.sim_time),
                env_idx=camera.env_idx,
                sensor_id=camera.sensor_id,
                max_distance=_render_f32(camera.max_distance),
                sensor_role=camera.sensor_role,
                sensor_role_mask=role_table.mask_for(camera.sensor_role),
            ),
            width=camera.width,
            height=camera.height,
            fx=_render_f32(camera.fx),
            fy=_render_f32(camera.fy),
            cx=_render_f32(camera.cx),
            cy=_render_f32(camera.cy),
            origin=tuple(_render_f32(v) for v in camera.X_world_camera.r),
            rotation=tuple(_render_f32(v) for row in camera.X_world_camera.R for v in row),
        )

    @property
    def sha256(self) -> str:
        self.validate()
        return _render_digest(asdict(self))


def _render_command_ids(ray_source: str, num_rays: int, shadows: bool, diagnostics: bool) -> tuple[str, ...]:
    reset_shadow = (
        "_reset_shadow_diagnostic_counters_kernel" if diagnostics else "_reset_shadow_primary_counters_kernel"
    )
    ids = ["_reset_bvh_primary_counters_kernel"]
    if num_rays:
        if ray_source == "camera":
            ids.insert(0, "_pinhole_camera_raygen_kernel")
        ids.append("_device_scene_bvh_first_hit_kernel")
    ids.append(reset_shadow)
    if num_rays:
        mode = "with_shadows" if shadows else "no_shadow"
        suffix = "diagnostics" if diagnostics else "primary"
        ids.append(f"warp_direct_light_{mode}_{suffix}_v1")
    return tuple(ids)


@dataclass(frozen=True, slots=True)
class WarpDirectLightRenderCommandEvidence:
    """Independent local facts, not a completion ticket or authenticity token."""

    schema_version: str
    operation_id: str
    kernel_abi_id: str
    kernel_source_sha256: str
    device_ordinal: int
    stream_handle: int
    native_input_manifest_sha256: str
    readiness_sha256: str
    scene_contract_sha256: str
    input_contract: str
    fixed_render_fixture_sha256: str | None
    transform_replay_sha256: str | None
    transform_frame_index: int | None
    query_sha256: str
    camera_contract_sha256: str | None
    ray_input_storage_sha256: str | None
    ray_source: str
    num_rays: int
    output_profile: str
    shadows: bool
    traversal_diagnostics: bool
    input_storage_sha256: str
    output_storage_sha256: str
    command_entry_ids: tuple[str, ...]
    command_plan_sha256: str
    dispatch_count: int

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if (self.schema_version, self.operation_id, self.kernel_abi_id) != (
            WARP_RENDER_COMMAND_SCHEMA,
            "warp_bvh_direct_light",
            WARP_RENDER_ABI,
        ):
            raise ValueError("unsupported render evidence identity")
        for name in (
            "kernel_source_sha256",
            "native_input_manifest_sha256",
            "readiness_sha256",
            "scene_contract_sha256",
            "query_sha256",
            "input_storage_sha256",
            "output_storage_sha256",
            "command_plan_sha256",
        ):
            if not _render_sha256(getattr(self, name)):
                raise ValueError(f"invalid render digest: {name}")
        for name in ("device_ordinal", "num_rays", "dispatch_count"):
            if type(getattr(self, name)) is not int or not 0 <= getattr(self, name) <= 2147483647:
                raise ValueError(f"invalid render integer: {name}")
        if type(self.stream_handle) is not int or self.stream_handle <= 0:
            raise ValueError("native render requires an explicit nondefault stream")
        if type(self.shadows) is not bool or type(self.traversal_diagnostics) is not bool:
            raise ValueError("render modes must be exact bools")
        if self.output_profile not in ("direct_light_full", "rgb_preview", "render_only"):
            raise ValueError("invalid direct-light output profile")
        if self.input_contract == "fixed_native_scene_v1":
            if (
                not _render_sha256(self.fixed_render_fixture_sha256)
                or self.transform_replay_sha256 is not None
                or self.transform_frame_index is not None
            ):
                raise ValueError("fixed render provenance must exclude replay")
        elif self.input_contract == "canonical_native_replay_v1":
            if (
                self.fixed_render_fixture_sha256 is not None
                or not _render_sha256(self.transform_replay_sha256)
                or type(self.transform_frame_index) is not int
                or self.transform_frame_index < 0
            ):
                raise ValueError("replay render provenance must exclude fixed fixture")
        else:
            raise ValueError("render command does not accept shipping or generic input provenance")
        if self.ray_source == "camera":
            if (
                not self.num_rays
                or not _render_sha256(self.camera_contract_sha256)
                or self.ray_input_storage_sha256 is not None
            ):
                raise ValueError("camera render provenance is incomplete")
        elif self.ray_source == "device_rays":
            if self.camera_contract_sha256 is not None or not _render_sha256(self.ray_input_storage_sha256):
                raise ValueError("device-ray render provenance is incomplete")
        else:
            raise ValueError("invalid render ray source")
        expected = _render_command_ids(
            self.ray_source, self.num_rays, self.shadows, self.traversal_diagnostics
        )
        if (
            type(self.command_entry_ids) is not tuple
            or self.command_entry_ids != expected
            or self.dispatch_count != len(expected)
        ):
            raise ValueError("render command order/count does not match the fixed ABI")

    def to_payload(self) -> dict[str, object]:
        self.validate()
        payload = asdict(self)
        payload["command_entry_ids"] = list(self.command_entry_ids)
        return payload

    @classmethod
    def from_payload(cls, payload: object) -> "WarpDirectLightRenderCommandEvidence":
        if type(payload) is not dict or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("invalid render evidence fields")
        if type(payload["command_entry_ids"]) is not list or any(
            type(v) is not str for v in payload["command_entry_ids"]
        ):
            raise ValueError("render command IDs must be a JSON string array")
        return cls(**{**payload, "command_entry_ids": tuple(payload["command_entry_ids"])})

    @property
    def sha256(self) -> str:
        return _render_digest(self.to_payload())


# Ordered physical render ABI. Profiles filter channels, not this storage graph.
# (semantic, scalar dtype name, per-ray flag, trailing or scalar shape)
WARP_DIRECT_LIGHT_BUFFER_SPECS = (
    ("origins", "float32", True, (3,)),
    ("directions", "float32", True, (3,)),
    ("hit_mask", "int32", True, ()),
    ("range_m", "float32", True, ()),
    ("position_world", "float32", True, (3,)),
    ("normal_world", "float32", True, (3,)),
    ("numeric_instance_id", "int32", True, ()),
    ("material_index", "int32", True, ()),
    ("bvh_stack_overflow_count", "int32", False, (1,)),
    ("bvh_max_stack_depth", "int32", False, (1,)),
    ("rgb", "float32", True, (3,)),
    ("intensity", "float32", True, ()),
    ("shadow_stack_overflow_count", "int32", False, (1,)),
    ("shadow_max_stack_depth", "int32", False, (1,)),
)


def warp_direct_light_buffer_layout(
    num_rays: int,
    *,
    traversal_diagnostics: bool,
) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    """CPU-only ABI metadata; no initialization, allocation, or device lookup."""
    if type(num_rays) is not int or num_rays < 0:
        raise ValueError("num_rays must be a nonnegative Python integer")
    if type(traversal_diagnostics) is not bool:
        raise TypeError("traversal_diagnostics must be a bool")
    result = tuple(
        (name, dtype, (num_rays, *shape) if per_ray else shape)
        for name, dtype, per_ray, shape in WARP_DIRECT_LIGHT_BUFFER_SPECS
    )
    if traversal_diagnostics:
        result += tuple((name, "int32", (1,)) for name in SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES)
    return result


def _render_array_layout(value: object, *, name: str, device: object) -> tuple:
    """Inspect the layout actually passed by Warp, for both inputs and outputs."""
    if type(value) is not wp.array:
        raise TypeError(f"{name}: expected an exact Warp array")
    if value.device is not device or not value.device.is_cuda:
        raise ValueError(f"{name}: requires the same indexed CUDA device")
    if not value.is_contiguous:
        raise ValueError(f"{name}: requires contiguous storage")
    shape, strides = tuple(value.shape), tuple(value.strides)
    # Contiguity flags are mutable; independently recompute the scalar layout.
    expected_strides = []
    stride = wp.types.type_size_in_bytes(value.dtype)
    for size in reversed(shape):
        if type(size) is not int or size < 0:
            raise ValueError(f"{name}: invalid shape")
        expected_strides.append(stride)
        stride *= size
    if strides != tuple(reversed(expected_strides)) or value.ndim != len(shape):
        raise ValueError(f"{name}: inconsistent contiguous strides/rank")
    active_bytes = prod(shape) * wp.types.type_size_in_bytes(value.dtype)
    if value.size != prod(shape) or value.capacity < active_bytes:
        raise ValueError(f"{name}: invalid storage size/capacity")
    pointer = int(value.ptr or 0)
    if active_bytes and pointer <= 0:
        raise ValueError(f"{name}: null active storage")
    # Warp 1.12 caches array_t, and packs value.__ctype__() at launch time.
    # Inspect only an existing descriptor; do not create or repair one here.
    if "__ctype__" in vars(value):
        raise ValueError(f"{name}: instance-level launch descriptor override")
    descriptor = value.ctype
    if descriptor is not None:
        if type(descriptor) is not _WarpArrayDescriptor:
            raise ValueError(f"{name}: invalid cached launch descriptor type")
        descriptor_shape = shape + (0,) * (len(descriptor.shape) - len(shape))
        descriptor_strides = strides + (0,) * (len(descriptor.strides) - len(strides))
        grad_pointer = int(value.grad.ptr or 0) if value.grad is not None else 0
        if (
            descriptor.data != pointer
            or descriptor.grad != grad_pointer
            or descriptor.ndim != len(shape)
            or tuple(descriptor.shape) != descriptor_shape
            or tuple(descriptor.strides) != descriptor_strides
        ):
            raise ValueError(f"{name}: cached launch descriptor disagrees with storage")
    return shape, strides, pointer, active_bytes


def _render_array_snapshot(value: object, *, name: str, device: object) -> tuple:
    """Untimed physical inspection. This is not a provenance/lease authority.

    Native outputs must be Warp allocations or Warp slices/reshapes retaining
    them. Foreign interop views remain the responsibility of the input binding,
    not an alternate mutable-output allocator in this slice.
    """
    shape, strides, pointer, active_bytes = _render_array_layout(value, name=name, device=device)
    chain = []
    seen = set()
    current = value
    while True:
        if id(current) in seen:
            raise ValueError(f"{name}: cyclic storage owner chain")
        seen.add(id(current))
        if type(current) is not wp.array or current.device is not device:
            raise ValueError(f"{name}: output must retain a native Warp allocation")
        chain.append((current, int(current.ptr or 0), int(current.capacity), current._ref, current.deleter))
        if current._ref is None:
            break
        current = current._ref
    base_pointer, capacity = int(current.ptr or 0), int(current.capacity)
    if capacity < 0 or (capacity and (base_pointer <= 0 or current.deleter is None)):
        raise ValueError(f"{name}: allocation has no retained owning root")
    if active_bytes and not (base_pointer <= pointer and pointer + active_bytes <= base_pointer + capacity):
        raise ValueError(f"{name}: active interval escapes retained allocation")
    return (
        value,
        device,
        value.dtype,
        shape,
        strides,
        pointer,
        active_bytes,
        int(value.capacity),
        tuple(chain),
    )


def _render_snapshot_key(snapshot: tuple) -> tuple:
    value, device, dtype, shape, strides, pointer, size, capacity, chain = snapshot
    return (
        id(value),
        id(device),
        dtype,
        shape,
        strides,
        pointer,
        size,
        capacity,
        tuple((id(owner), ptr, cap, id(ref), id(deleter)) for owner, ptr, cap, ref, deleter in chain),
    )


@dataclass(frozen=True, eq=False, slots=True)
class WarpDirectLightBuffers:
    """Caller-owned native output graph, frozen during untimed preparation.

    Construction neither initializes values nor submits work. No profile can
    omit physical buffers. This object grants no recorded-execution, arena,
    readiness, result, or resource-release authority. The future factory must
    call validate_bound with its additional scene/BVH inputs before recording.
    Ray storage is already included here, including for device-ray bindings.
    """

    num_rays: int
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
    rgb: object
    intensity: object
    shadow_stack_overflow_count: object
    shadow_max_stack_depth: object
    traversal_diagnostics: bool = False
    traversal_counters: tuple[object, ...] = ()
    _snapshots: tuple = field(init=False, repr=False)
    _signature: tuple = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if hasattr(self, "_signature"):
            raise RuntimeError("render storage baseline cannot be rebuilt")
        if not _HAS_WARP:
            raise RuntimeError("Warp is required for native render storage")
        snapshots = self._inspect()
        object.__setattr__(self, "_snapshots", snapshots)
        object.__setattr__(self, "_signature", self._key(snapshots))

    @property
    def arrays(self) -> tuple[tuple[str, object], ...]:
        primary = tuple((name, getattr(self, name)) for name, *_ in WARP_DIRECT_LIGHT_BUFFER_SPECS)
        return primary + tuple(zip(SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES, self.traversal_counters))

    @property
    def device(self) -> object:
        return self.origins.device

    def _inspect(self) -> tuple:
        layout = warp_direct_light_buffer_layout(
            self.num_rays,
            traversal_diagnostics=self.traversal_diagnostics,
        )
        expected_counters = (
            len(SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES) if self.traversal_diagnostics else 0
        )
        if type(self.traversal_counters) is not tuple or len(self.traversal_counters) != expected_counters:
            raise ValueError("traversal counters must be an exact tuple matching diagnostics mode")
        if type(self.origins) is not wp.array or not self.origins.device.is_cuda:
            raise ValueError("origins must be a native Warp array on an indexed CUDA device")
        snapshots = []
        seen_objects = set()
        for (name, dtype, shape), (actual_name, value) in zip(layout, self.arrays, strict=True):
            if name != actual_name or type(value) is not wp.array:
                raise TypeError(f"{name}: expected an exact Warp array")
            if value.dtype is not getattr(wp, dtype) or tuple(value.shape) != shape:
                raise ValueError(f"{name}: dtype/shape does not match render ABI")
            if id(value) in seen_objects:
                raise ValueError(f"{name}: independent array objects are required")
            seen_objects.add(id(value))
            snapshot = _render_array_snapshot(value, name=name, device=self.device)
            for other_name, other in snapshots:
                if _render_intervals_overlap(snapshot, other):
                    raise ValueError(f"{name}: output overlaps {other_name}")
            snapshots.append((name, snapshot))
        return tuple(snapshots)

    def _key(self, snapshots: tuple) -> tuple:
        return (
            self.num_rays,
            self.traversal_diagnostics,
            id(self.traversal_counters),
            tuple((name, _render_snapshot_key(value)) for name, value in snapshots),
        )

    def validate_bound(self, *, read_only_inputs: tuple[tuple[str, object], ...] = ()) -> None:
        """Full, untimed revalidation; never used by measured acquire."""
        snapshots = self._inspect()
        if self._key(snapshots) != self._signature:
            raise ValueError("frozen render storage binding changed")
        # _snapshots itself retains owners; detect replacing that retention graph.
        if tuple((n, _render_snapshot_key(s)) for n, s in self._snapshots) != self._signature[-1]:
            raise ValueError("frozen render storage retention changed")
        if type(read_only_inputs) is not tuple:
            raise TypeError("read_only_inputs must be a frozen tuple")
        for name, value in read_only_inputs:
            _, _, start, size = _render_array_layout(value, name=name, device=self.device)
            for output_name, snapshot in snapshots:
                end = start + size
                output_start, output_size = snapshot[5:7]
                if size and output_size and start < output_start + output_size and output_start < end:
                    raise ValueError(f"{output_name}: output overlaps input {name}")


def _render_intervals_overlap(first: tuple, second: tuple) -> bool:
    start, size = first[5:7]
    other_start, other_size = second[5:7]
    return bool(size and other_size and start < other_start + other_size and other_start < start + size)


if _HAS_WARP:

    @wp.kernel
    def _pinhole_camera_raygen_kernel(
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        origin_x: float,
        origin_y: float,
        origin_z: float,
        R00: float,
        R01: float,
        R02: float,
        R10: float,
        R11: float,
        R12: float,
        R20: float,
        R21: float,
        R22: float,
        origins: wp.array2d(dtype=wp.float32),
        directions: wp.array2d(dtype=wp.float32),
    ):
        ray = wp.tid()
        px = ray - (ray / width) * width
        py = ray / width
        if py < height:
            camera_x = (wp.float32(px) - wp.float32(cx)) / wp.float32(fx)
            camera_y = (wp.float32(py) - wp.float32(cy)) / wp.float32(fy)
            camera_z = wp.float32(1.0)
            inv_norm = wp.float32(1.0) / wp.sqrt(
                camera_x * camera_x + camera_y * camera_y + camera_z * camera_z
            )
            camera_x = camera_x * inv_norm
            camera_y = camera_y * inv_norm
            camera_z = camera_z * inv_norm

            origins[ray, 0] = wp.float32(origin_x)
            origins[ray, 1] = wp.float32(origin_y)
            origins[ray, 2] = wp.float32(origin_z)
            directions[ray, 0] = (
                wp.float32(R00) * camera_x + wp.float32(R01) * camera_y + wp.float32(R02) * camera_z
            )
            directions[ray, 1] = (
                wp.float32(R10) * camera_x + wp.float32(R11) * camera_y + wp.float32(R12) * camera_z
            )
            directions[ray, 2] = (
                wp.float32(R20) * camera_x + wp.float32(R21) * camera_y + wp.float32(R22) * camera_z
            )

    @wp.kernel
    def _brute_force_first_hit_kernel(
        origins: wp.array2d(dtype=wp.float32),
        directions: wp.array2d(dtype=wp.float32),
        max_distance: float,
        plane_normals: wp.array2d(dtype=wp.float32),
        plane_points: wp.array2d(dtype=wp.float32),
        plane_numeric_ids: wp.array(dtype=wp.int32),
        plane_source_keys: wp.array(dtype=wp.int64),
        num_planes: int,
        triangles: wp.array2d(dtype=wp.float32),
        triangle_numeric_ids: wp.array(dtype=wp.int32),
        triangle_source_keys: wp.array(dtype=wp.int64),
        num_triangles: int,
        hit_mask: wp.array(dtype=wp.int32),
        range_m: wp.array(dtype=wp.float32),
        position_world: wp.array2d(dtype=wp.float32),
        normal_world: wp.array2d(dtype=wp.float32),
        numeric_instance_id: wp.array(dtype=wp.int32),
    ):
        ray = wp.tid()
        ox = origins[ray, 0]
        oy = origins[ray, 1]
        oz = origins[ray, 2]
        dx = directions[ray, 0]
        dy = directions[ray, 1]
        dz = directions[ray, 2]

        best_t = wp.float32(max_distance)
        best_key = wp.int64(9223372036854775807)
        best_px = wp.float32(0.0)
        best_py = wp.float32(0.0)
        best_pz = wp.float32(0.0)
        best_nx = wp.float32(0.0)
        best_ny = wp.float32(0.0)
        best_nz = wp.float32(0.0)
        best_id = wp.int32(0)
        found = wp.int32(0)

        for plane_idx in range(num_planes):
            nx = plane_normals[plane_idx, 0]
            ny = plane_normals[plane_idx, 1]
            nz = plane_normals[plane_idx, 2]
            px = plane_points[plane_idx, 0]
            py = plane_points[plane_idx, 1]
            pz = plane_points[plane_idx, 2]
            denom = dx * nx + dy * ny + dz * nz
            if wp.abs(denom) > _DIR_EPS:
                numer = (px - ox) * nx + (py - oy) * ny + (pz - oz) * nz
                t = numer / denom
                if t >= 0.0 and t <= max_distance:
                    key = plane_source_keys[plane_idx]
                    if _is_better_hit(t, key, best_t, best_key):
                        best_t = t
                        best_key = key
                        best_px = ox + dx * t
                        best_py = oy + dy * t
                        best_pz = oz + dz * t
                        best_nx = nx
                        best_ny = ny
                        best_nz = nz
                        best_id = plane_numeric_ids[plane_idx]
                        found = wp.int32(1)

        for tri_idx in range(num_triangles):
            v0x = triangles[tri_idx, 0]
            v0y = triangles[tri_idx, 1]
            v0z = triangles[tri_idx, 2]
            v1x = triangles[tri_idx, 3]
            v1y = triangles[tri_idx, 4]
            v1z = triangles[tri_idx, 5]
            v2x = triangles[tri_idx, 6]
            v2y = triangles[tri_idx, 7]
            v2z = triangles[tri_idx, 8]

            e1x = v1x - v0x
            e1y = v1y - v0y
            e1z = v1z - v0z
            e2x = v2x - v0x
            e2y = v2y - v0y
            e2z = v2z - v0z

            fnx = e1y * e2z - e1z * e2y
            fny = e1z * e2x - e1x * e2z
            fnz = e1x * e2y - e1y * e2x
            normal_norm = wp.sqrt(fnx * fnx + fny * fny + fnz * fnz)

            if normal_norm > _BUILD_EPS:
                fnx = fnx / normal_norm
                fny = fny / normal_norm
                fnz = fnz / normal_norm

                pvec_x = dy * e2z - dz * e2y
                pvec_y = dz * e2x - dx * e2z
                pvec_z = dx * e2y - dy * e2x
                det = pvec_x * e1x + pvec_y * e1y + pvec_z * e1z
                if wp.abs(det) > _BUILD_EPS:
                    inv_det = 1.0 / det
                    tvec_x = ox - v0x
                    tvec_y = oy - v0y
                    tvec_z = oz - v0z
                    u = (pvec_x * tvec_x + pvec_y * tvec_y + pvec_z * tvec_z) * inv_det
                    if u >= 0.0:
                        qvec_x = tvec_y * e1z - tvec_z * e1y
                        qvec_y = tvec_z * e1x - tvec_x * e1z
                        qvec_z = tvec_x * e1y - tvec_y * e1x
                        v = (qvec_x * dx + qvec_y * dy + qvec_z * dz) * inv_det
                        if v >= 0.0 and u + v <= 1.0:
                            t = (qvec_x * e2x + qvec_y * e2y + qvec_z * e2z) * inv_det
                            if t >= 0.0 and t <= max_distance:
                                key = triangle_source_keys[tri_idx]
                                if _is_better_hit(t, key, best_t, best_key):
                                    ndotd = fnx * dx + fny * dy + fnz * dz
                                    if ndotd > 0.0:
                                        fnx = -fnx
                                        fny = -fny
                                        fnz = -fnz
                                    best_t = t
                                    best_key = key
                                    best_px = ox + dx * t
                                    best_py = oy + dy * t
                                    best_pz = oz + dz * t
                                    best_nx = fnx
                                    best_ny = fny
                                    best_nz = fnz
                                    best_id = triangle_numeric_ids[tri_idx]
                                    found = wp.int32(1)

        if found != 0:
            hit_mask[ray] = wp.int32(1)
            range_m[ray] = best_t
            position_world[ray, 0] = best_px
            position_world[ray, 1] = best_py
            position_world[ray, 2] = best_pz
            normal_world[ray, 0] = best_nx
            normal_world[ray, 1] = best_ny
            normal_world[ray, 2] = best_nz
            numeric_instance_id[ray] = best_id

    @wp.func
    def _is_better_hit(t: float, key: wp.int64, best_t: float, best_key: wp.int64):
        better = bool(False)
        if t < best_t - _T_EPS:
            better = True
        elif wp.abs(t - best_t) <= _T_EPS and key < best_key:
            better = True
        return better

    @wp.func
    def _intersect_aabb_axis(
        origin: float, direction: float, lower: float, upper: float, t_min: float, t_max: float
    ):
        hit = bool(True)
        out_min = t_min
        out_max = t_max
        if wp.abs(direction) <= _DIR_EPS:
            if origin < lower - _T_EPS or origin > upper + _T_EPS:
                hit = False
        else:
            inv_d = 1.0 / direction
            t0 = (lower - origin) * inv_d
            t1 = (upper - origin) * inv_d
            if t0 > t1:
                tmp = t0
                t0 = t1
                t1 = tmp
            if t0 > out_min:
                out_min = t0
            if t1 < out_max:
                out_max = t1
            if out_max < out_min:
                hit = False
        return hit, out_min, out_max

    @wp.func
    def _intersect_aabb_for_ray(
        ox: float,
        oy: float,
        oz: float,
        dx: float,
        dy: float,
        dz: float,
        min_x: float,
        min_y: float,
        min_z: float,
        max_x: float,
        max_y: float,
        max_z: float,
        max_t: float,
    ):
        hit, _out_min, _out_max = _intersect_aabb_for_ray_with_interval(
            ox,
            oy,
            oz,
            dx,
            dy,
            dz,
            min_x,
            min_y,
            min_z,
            max_x,
            max_y,
            max_z,
            max_t,
        )
        return hit

    @wp.func
    def _intersect_aabb_for_ray_with_interval(
        ox: float,
        oy: float,
        oz: float,
        dx: float,
        dy: float,
        dz: float,
        min_x: float,
        min_y: float,
        min_z: float,
        max_x: float,
        max_y: float,
        max_z: float,
        max_t: float,
    ):
        t_min = wp.float32(0.0)
        t_max = wp.float32(max_t)
        hit = bool(True)
        hit, t_min, t_max = _intersect_aabb_axis(ox, dx, min_x, max_x, t_min, t_max)
        if hit:
            hit, t_min, t_max = _intersect_aabb_axis(oy, dy, min_y, max_y, t_min, t_max)
        if hit:
            hit, t_min, t_max = _intersect_aabb_axis(oz, dz, min_z, max_z, t_min, t_max)
        return hit, t_min, t_max

    @wp.func
    def _intersect_device_triangle_for_ray(
        ox: float,
        oy: float,
        oz: float,
        dx: float,
        dy: float,
        dz: float,
        triangle_index: int,
        triangle_v0: wp.array2d(dtype=wp.float32),
        triangle_e1: wp.array2d(dtype=wp.float32),
        triangle_e2: wp.array2d(dtype=wp.float32),
        triangle_normal: wp.array2d(dtype=wp.float32),
        max_t: float,
    ):
        hit = bool(False)
        out_t = wp.float32(0.0)
        out_nx = wp.float32(0.0)
        out_ny = wp.float32(0.0)
        out_nz = wp.float32(0.0)

        v0x = triangle_v0[triangle_index, 0]
        v0y = triangle_v0[triangle_index, 1]
        v0z = triangle_v0[triangle_index, 2]
        e1x = triangle_e1[triangle_index, 0]
        e1y = triangle_e1[triangle_index, 1]
        e1z = triangle_e1[triangle_index, 2]
        e2x = triangle_e2[triangle_index, 0]
        e2y = triangle_e2[triangle_index, 1]
        e2z = triangle_e2[triangle_index, 2]
        pvec_x = dy * e2z - dz * e2y
        pvec_y = dz * e2x - dx * e2z
        pvec_z = dx * e2y - dy * e2x
        det = pvec_x * e1x + pvec_y * e1y + pvec_z * e1z
        if wp.abs(det) > _BUILD_EPS:
            inv_det = 1.0 / det
            tvec_x = ox - v0x
            tvec_y = oy - v0y
            tvec_z = oz - v0z
            u = (pvec_x * tvec_x + pvec_y * tvec_y + pvec_z * tvec_z) * inv_det
            if u >= 0.0:
                qvec_x = tvec_y * e1z - tvec_z * e1y
                qvec_y = tvec_z * e1x - tvec_x * e1z
                qvec_z = tvec_x * e1y - tvec_y * e1x
                v = (qvec_x * dx + qvec_y * dy + qvec_z * dz) * inv_det
                if v >= 0.0 and u + v <= 1.0:
                    t = (qvec_x * e2x + qvec_y * e2y + qvec_z * e2z) * inv_det
                    if t >= 0.0 and t <= max_t:
                        out_t = t
                        out_nx = triangle_normal[triangle_index, 0]
                        out_ny = triangle_normal[triangle_index, 1]
                        out_nz = triangle_normal[triangle_index, 2]
                        ndotd = out_nx * dx + out_ny * dy + out_nz * dz
                        if ndotd > 0.0:
                            out_nx = -out_nx
                            out_ny = -out_ny
                            out_nz = -out_nz
                        hit = True
        return hit, out_t, out_nx, out_ny, out_nz

    @wp.kernel
    def _device_scene_first_hit_kernel(
        origins: wp.array2d(dtype=wp.float32),
        directions: wp.array2d(dtype=wp.float32),
        max_distance: float,
        plane_normals: wp.array2d(dtype=wp.float32),
        plane_points: wp.array2d(dtype=wp.float32),
        plane_numeric_ids: wp.array(dtype=wp.int32),
        plane_source_keys: wp.array(dtype=wp.int64),
        plane_role_masks: wp.array(dtype=wp.int64),
        num_planes: int,
        triangle_v0: wp.array2d(dtype=wp.float32),
        triangle_e1: wp.array2d(dtype=wp.float32),
        triangle_e2: wp.array2d(dtype=wp.float32),
        triangle_normal: wp.array2d(dtype=wp.float32),
        triangle_numeric_ids: wp.array(dtype=wp.int32),
        triangle_source_keys: wp.array(dtype=wp.int64),
        triangle_role_masks: wp.array(dtype=wp.int64),
        num_triangles: int,
        sensor_role_mask: wp.int64,
        hit_mask: wp.array(dtype=wp.int32),
        range_m: wp.array(dtype=wp.float32),
        position_world: wp.array2d(dtype=wp.float32),
        normal_world: wp.array2d(dtype=wp.float32),
        numeric_instance_id: wp.array(dtype=wp.int32),
    ):
        ray = wp.tid()
        ox = origins[ray, 0]
        oy = origins[ray, 1]
        oz = origins[ray, 2]
        dx = directions[ray, 0]
        dy = directions[ray, 1]
        dz = directions[ray, 2]

        best_t = wp.float32(max_distance)
        best_key = wp.int64(9223372036854775807)
        best_px = wp.float32(0.0)
        best_py = wp.float32(0.0)
        best_pz = wp.float32(0.0)
        best_nx = wp.float32(0.0)
        best_ny = wp.float32(0.0)
        best_nz = wp.float32(0.0)
        best_id = wp.int32(0)
        found = wp.int32(0)

        for plane_idx in range(num_planes):
            if (plane_role_masks[plane_idx] & sensor_role_mask) != wp.int64(0):
                nx = plane_normals[plane_idx, 0]
                ny = plane_normals[plane_idx, 1]
                nz = plane_normals[plane_idx, 2]
                px = plane_points[plane_idx, 0]
                py = plane_points[plane_idx, 1]
                pz = plane_points[plane_idx, 2]
                denom = dx * nx + dy * ny + dz * nz
                if wp.abs(denom) > _DIR_EPS:
                    numer = (px - ox) * nx + (py - oy) * ny + (pz - oz) * nz
                    t = numer / denom
                    if t >= 0.0 and t <= max_distance:
                        key = plane_source_keys[plane_idx]
                        if _is_better_hit(t, key, best_t, best_key):
                            best_t = t
                            best_key = key
                            best_px = ox + dx * t
                            best_py = oy + dy * t
                            best_pz = oz + dz * t
                            best_nx = nx
                            best_ny = ny
                            best_nz = nz
                            best_id = plane_numeric_ids[plane_idx]
                            found = wp.int32(1)

        for tri_idx in range(num_triangles):
            if (triangle_role_masks[tri_idx] & sensor_role_mask) != wp.int64(0):
                v0x = triangle_v0[tri_idx, 0]
                v0y = triangle_v0[tri_idx, 1]
                v0z = triangle_v0[tri_idx, 2]
                e1x = triangle_e1[tri_idx, 0]
                e1y = triangle_e1[tri_idx, 1]
                e1z = triangle_e1[tri_idx, 2]
                e2x = triangle_e2[tri_idx, 0]
                e2y = triangle_e2[tri_idx, 1]
                e2z = triangle_e2[tri_idx, 2]
                pvec_x = dy * e2z - dz * e2y
                pvec_y = dz * e2x - dx * e2z
                pvec_z = dx * e2y - dy * e2x
                det = pvec_x * e1x + pvec_y * e1y + pvec_z * e1z
                if wp.abs(det) > _BUILD_EPS:
                    inv_det = 1.0 / det
                    tvec_x = ox - v0x
                    tvec_y = oy - v0y
                    tvec_z = oz - v0z
                    u = (pvec_x * tvec_x + pvec_y * tvec_y + pvec_z * tvec_z) * inv_det
                    if u >= 0.0:
                        qvec_x = tvec_y * e1z - tvec_z * e1y
                        qvec_y = tvec_z * e1x - tvec_x * e1z
                        qvec_z = tvec_x * e1y - tvec_y * e1x
                        v = (qvec_x * dx + qvec_y * dy + qvec_z * dz) * inv_det
                        if v >= 0.0 and u + v <= 1.0:
                            t = (qvec_x * e2x + qvec_y * e2y + qvec_z * e2z) * inv_det
                            if t >= 0.0 and t <= max_distance:
                                key = triangle_source_keys[tri_idx]
                                if _is_better_hit(t, key, best_t, best_key):
                                    fnx = triangle_normal[tri_idx, 0]
                                    fny = triangle_normal[tri_idx, 1]
                                    fnz = triangle_normal[tri_idx, 2]
                                    ndotd = fnx * dx + fny * dy + fnz * dz
                                    if ndotd > 0.0:
                                        fnx = -fnx
                                        fny = -fny
                                        fnz = -fnz
                                    best_t = t
                                    best_key = key
                                    best_px = ox + dx * t
                                    best_py = oy + dy * t
                                    best_pz = oz + dz * t
                                    best_nx = fnx
                                    best_ny = fny
                                    best_nz = fnz
                                    best_id = triangle_numeric_ids[tri_idx]
                                    found = wp.int32(1)

        if found != 0:
            hit_mask[ray] = wp.int32(1)
            range_m[ray] = best_t
            position_world[ray, 0] = best_px
            position_world[ray, 1] = best_py
            position_world[ray, 2] = best_pz
            normal_world[ray, 0] = best_nx
            normal_world[ray, 1] = best_ny
            normal_world[ray, 2] = best_nz
            numeric_instance_id[ray] = best_id

    @wp.kernel
    def _device_scene_first_hit_aabb_kernel(
        origins: wp.array2d(dtype=wp.float32),
        directions: wp.array2d(dtype=wp.float32),
        max_distance: float,
        plane_normals: wp.array2d(dtype=wp.float32),
        plane_points: wp.array2d(dtype=wp.float32),
        plane_numeric_ids: wp.array(dtype=wp.int32),
        plane_source_keys: wp.array(dtype=wp.int64),
        plane_role_masks: wp.array(dtype=wp.int64),
        num_planes: int,
        triangle_v0: wp.array2d(dtype=wp.float32),
        triangle_e1: wp.array2d(dtype=wp.float32),
        triangle_e2: wp.array2d(dtype=wp.float32),
        triangle_normal: wp.array2d(dtype=wp.float32),
        triangle_aabb_min: wp.array2d(dtype=wp.float32),
        triangle_aabb_max: wp.array2d(dtype=wp.float32),
        triangle_numeric_ids: wp.array(dtype=wp.int32),
        triangle_source_keys: wp.array(dtype=wp.int64),
        triangle_role_masks: wp.array(dtype=wp.int64),
        num_triangles: int,
        sensor_role_mask: wp.int64,
        hit_mask: wp.array(dtype=wp.int32),
        range_m: wp.array(dtype=wp.float32),
        position_world: wp.array2d(dtype=wp.float32),
        normal_world: wp.array2d(dtype=wp.float32),
        numeric_instance_id: wp.array(dtype=wp.int32),
    ):
        ray = wp.tid()
        ox = origins[ray, 0]
        oy = origins[ray, 1]
        oz = origins[ray, 2]
        dx = directions[ray, 0]
        dy = directions[ray, 1]
        dz = directions[ray, 2]

        best_t = wp.float32(max_distance)
        best_key = wp.int64(9223372036854775807)
        best_px = wp.float32(0.0)
        best_py = wp.float32(0.0)
        best_pz = wp.float32(0.0)
        best_nx = wp.float32(0.0)
        best_ny = wp.float32(0.0)
        best_nz = wp.float32(0.0)
        best_id = wp.int32(0)
        found = wp.int32(0)

        for plane_idx in range(num_planes):
            if (plane_role_masks[plane_idx] & sensor_role_mask) != wp.int64(0):
                nx = plane_normals[plane_idx, 0]
                ny = plane_normals[plane_idx, 1]
                nz = plane_normals[plane_idx, 2]
                px = plane_points[plane_idx, 0]
                py = plane_points[plane_idx, 1]
                pz = plane_points[plane_idx, 2]
                denom = dx * nx + dy * ny + dz * nz
                if wp.abs(denom) > _DIR_EPS:
                    numer = (px - ox) * nx + (py - oy) * ny + (pz - oz) * nz
                    t = numer / denom
                    if t >= 0.0 and t <= max_distance:
                        key = plane_source_keys[plane_idx]
                        if _is_better_hit(t, key, best_t, best_key):
                            best_t = t
                            best_key = key
                            best_px = ox + dx * t
                            best_py = oy + dy * t
                            best_pz = oz + dz * t
                            best_nx = nx
                            best_ny = ny
                            best_nz = nz
                            best_id = plane_numeric_ids[plane_idx]
                            found = wp.int32(1)

        for tri_idx in range(num_triangles):
            if (triangle_role_masks[tri_idx] & sensor_role_mask) != wp.int64(0):
                v0x = triangle_v0[tri_idx, 0]
                v0y = triangle_v0[tri_idx, 1]
                v0z = triangle_v0[tri_idx, 2]
                e1x = triangle_e1[tri_idx, 0]
                e1y = triangle_e1[tri_idx, 1]
                e1z = triangle_e1[tri_idx, 2]
                e2x = triangle_e2[tri_idx, 0]
                e2y = triangle_e2[tri_idx, 1]
                e2z = triangle_e2[tri_idx, 2]

                should_test_triangle = _intersect_aabb_for_ray(
                    ox,
                    oy,
                    oz,
                    dx,
                    dy,
                    dz,
                    triangle_aabb_min[tri_idx, 0],
                    triangle_aabb_min[tri_idx, 1],
                    triangle_aabb_min[tri_idx, 2],
                    triangle_aabb_max[tri_idx, 0],
                    triangle_aabb_max[tri_idx, 1],
                    triangle_aabb_max[tri_idx, 2],
                    best_t,
                )

                if should_test_triangle:
                    pvec_x = dy * e2z - dz * e2y
                    pvec_y = dz * e2x - dx * e2z
                    pvec_z = dx * e2y - dy * e2x
                    det = pvec_x * e1x + pvec_y * e1y + pvec_z * e1z
                    if wp.abs(det) > _BUILD_EPS:
                        inv_det = 1.0 / det
                        tvec_x = ox - v0x
                        tvec_y = oy - v0y
                        tvec_z = oz - v0z
                        u = (pvec_x * tvec_x + pvec_y * tvec_y + pvec_z * tvec_z) * inv_det
                        if u >= 0.0:
                            qvec_x = tvec_y * e1z - tvec_z * e1y
                            qvec_y = tvec_z * e1x - tvec_x * e1z
                            qvec_z = tvec_x * e1y - tvec_y * e1x
                            v = (qvec_x * dx + qvec_y * dy + qvec_z * dz) * inv_det
                            if v >= 0.0 and u + v <= 1.0:
                                t = (qvec_x * e2x + qvec_y * e2y + qvec_z * e2z) * inv_det
                                if t >= 0.0 and t <= max_distance:
                                    key = triangle_source_keys[tri_idx]
                                    if _is_better_hit(t, key, best_t, best_key):
                                        fnx = triangle_normal[tri_idx, 0]
                                        fny = triangle_normal[tri_idx, 1]
                                        fnz = triangle_normal[tri_idx, 2]
                                        ndotd = fnx * dx + fny * dy + fnz * dz
                                        if ndotd > 0.0:
                                            fnx = -fnx
                                            fny = -fny
                                            fnz = -fnz
                                        best_t = t
                                        best_key = key
                                        best_px = ox + dx * t
                                        best_py = oy + dy * t
                                        best_pz = oz + dz * t
                                        best_nx = fnx
                                        best_ny = fny
                                        best_nz = fnz
                                        best_id = triangle_numeric_ids[tri_idx]
                                        found = wp.int32(1)

        if found != 0:
            hit_mask[ray] = wp.int32(1)
            range_m[ray] = best_t
            position_world[ray, 0] = best_px
            position_world[ray, 1] = best_py
            position_world[ray, 2] = best_pz
            normal_world[ray, 0] = best_nx
            normal_world[ray, 1] = best_ny
            normal_world[ray, 2] = best_nz
            numeric_instance_id[ray] = best_id

    @wp.kernel
    def _init_bvh_hit_mask_kernel(
        hit_mask: wp.array(dtype=wp.int32),
    ):
        ray = wp.tid()
        hit_mask[ray] = wp.int32(0)

    @wp.kernel
    def _device_scene_bvh_first_hit_kernel(
        origins: wp.array2d(dtype=wp.float32),
        directions: wp.array2d(dtype=wp.float32),
        max_distance: float,
        plane_normals: wp.array2d(dtype=wp.float32),
        plane_points: wp.array2d(dtype=wp.float32),
        plane_numeric_ids: wp.array(dtype=wp.int32),
        plane_source_keys: wp.array(dtype=wp.int64),
        plane_role_masks: wp.array(dtype=wp.int64),
        plane_material_indices: wp.array(dtype=wp.int32),
        num_planes: int,
        triangle_v0: wp.array2d(dtype=wp.float32),
        triangle_e1: wp.array2d(dtype=wp.float32),
        triangle_e2: wp.array2d(dtype=wp.float32),
        triangle_normal: wp.array2d(dtype=wp.float32),
        triangle_numeric_ids: wp.array(dtype=wp.int32),
        triangle_source_keys: wp.array(dtype=wp.int64),
        triangle_role_masks: wp.array(dtype=wp.int64),
        triangle_material_indices: wp.array(dtype=wp.int32),
        bvh_bounds_min: wp.array2d(dtype=wp.float32),
        bvh_bounds_max: wp.array2d(dtype=wp.float32),
        bvh_left: wp.array(dtype=wp.int32),
        bvh_right: wp.array(dtype=wp.int32),
        bvh_start: wp.array(dtype=wp.int32),
        bvh_count: wp.array(dtype=wp.int32),
        bvh_prim_ids: wp.array(dtype=wp.int32),
        num_bvh_nodes: int,
        sensor_role_mask: wp.int64,
        hit_mask: wp.array(dtype=wp.int32),
        range_m: wp.array(dtype=wp.float32),
        position_world: wp.array2d(dtype=wp.float32),
        normal_world: wp.array2d(dtype=wp.float32),
        numeric_instance_id: wp.array(dtype=wp.int32),
        material_index: wp.array(dtype=wp.int32),
        bvh_stack_overflow_count: wp.array(dtype=wp.int32),
        bvh_max_stack_depth: wp.array(dtype=wp.int32),
    ):
        ray = wp.tid()
        # Every reuse and profile has identical miss defaults, independent of
        # allocator contents or a compatibility caller's host initialization.
        hit_mask[ray] = wp.int32(0)
        range_m[ray] = wp.float32(wp.inf)
        position_world[ray, 0] = wp.float32(wp.nan)
        position_world[ray, 1] = wp.float32(wp.nan)
        position_world[ray, 2] = wp.float32(wp.nan)
        normal_world[ray, 0] = wp.float32(wp.nan)
        normal_world[ray, 1] = wp.float32(wp.nan)
        normal_world[ray, 2] = wp.float32(wp.nan)
        numeric_instance_id[ray] = wp.int32(0)
        material_index[ray] = wp.int32(0)
        ox = origins[ray, 0]
        oy = origins[ray, 1]
        oz = origins[ray, 2]
        dx = directions[ray, 0]
        dy = directions[ray, 1]
        dz = directions[ray, 2]

        best_t = wp.float32(max_distance)
        best_key = wp.int64(9223372036854775807)
        best_px = wp.float32(0.0)
        best_py = wp.float32(0.0)
        best_pz = wp.float32(0.0)
        best_nx = wp.float32(0.0)
        best_ny = wp.float32(0.0)
        best_nz = wp.float32(0.0)
        best_id = wp.int32(0)
        best_material_index = wp.int32(0)
        found = wp.int32(0)

        for plane_idx in range(num_planes):
            if (plane_role_masks[plane_idx] & sensor_role_mask) != wp.int64(0):
                nx = plane_normals[plane_idx, 0]
                ny = plane_normals[plane_idx, 1]
                nz = plane_normals[plane_idx, 2]
                px = plane_points[plane_idx, 0]
                py = plane_points[plane_idx, 1]
                pz = plane_points[plane_idx, 2]
                denom = dx * nx + dy * ny + dz * nz
                if wp.abs(denom) > _DIR_EPS:
                    numer = (px - ox) * nx + (py - oy) * ny + (pz - oz) * nz
                    t = numer / denom
                    if t >= 0.0 and t <= max_distance:
                        key = plane_source_keys[plane_idx]
                        if _is_better_hit(t, key, best_t, best_key):
                            best_t = t
                            best_key = key
                            best_px = ox + dx * t
                            best_py = oy + dy * t
                            best_pz = oz + dz * t
                            best_nx = nx
                            best_ny = ny
                            best_nz = nz
                            best_id = plane_numeric_ids[plane_idx]
                            best_material_index = plane_material_indices[plane_idx]
                            found = wp.int32(1)

        stack = wp.zeros(shape=_MAX_BVH_STACK, dtype=wp.int32)
        stack_t = wp.zeros(shape=_MAX_BVH_STACK, dtype=wp.float32)
        stack_size = wp.int32(0)
        local_max_stack = wp.int32(0)
        if num_bvh_nodes > 0:
            root_hit, root_t, _root_exit_t = _intersect_aabb_for_ray_with_interval(
                ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                bvh_bounds_min[0, 0],
                bvh_bounds_min[0, 1],
                bvh_bounds_min[0, 2],
                bvh_bounds_max[0, 0],
                bvh_bounds_max[0, 1],
                bvh_bounds_max[0, 2],
                best_t,
            )
            if root_hit:
                stack[0] = wp.int32(0)
                stack_t[0] = root_t
                stack_size = wp.int32(1)
                local_max_stack = wp.int32(1)

        while stack_size > 0:
            stack_size = stack_size - wp.int32(1)
            node_index = stack[stack_size]
            node_t = stack_t[stack_size]
            if node_t <= best_t:
                leaf_count = bvh_count[node_index]
                if leaf_count > 0:
                    leaf_start = bvh_start[node_index]
                    for offset in range(leaf_count):
                        tri_idx = bvh_prim_ids[leaf_start + offset]
                        if (triangle_role_masks[tri_idx] & sensor_role_mask) != wp.int64(0):
                            tri_hit, t, fnx, fny, fnz = _intersect_device_triangle_for_ray(
                                ox,
                                oy,
                                oz,
                                dx,
                                dy,
                                dz,
                                tri_idx,
                                triangle_v0,
                                triangle_e1,
                                triangle_e2,
                                triangle_normal,
                                best_t,
                            )
                            if tri_hit:
                                key = triangle_source_keys[tri_idx]
                                if _is_better_hit(t, key, best_t, best_key):
                                    best_t = t
                                    best_key = key
                                    best_px = ox + dx * t
                                    best_py = oy + dy * t
                                    best_pz = oz + dz * t
                                    best_nx = fnx
                                    best_ny = fny
                                    best_nz = fnz
                                    best_id = triangle_numeric_ids[tri_idx]
                                    best_material_index = triangle_material_indices[tri_idx]
                                    found = wp.int32(1)
                else:
                    left = bvh_left[node_index]
                    right = bvh_right[node_index]
                    left_hit = bool(False)
                    right_hit = bool(False)
                    left_t = wp.float32(0.0)
                    right_t = wp.float32(0.0)
                    _left_exit_t = wp.float32(0.0)
                    _right_exit_t = wp.float32(0.0)
                    if left >= 0:
                        left_hit, left_t, _left_exit_t = _intersect_aabb_for_ray_with_interval(
                            ox,
                            oy,
                            oz,
                            dx,
                            dy,
                            dz,
                            bvh_bounds_min[left, 0],
                            bvh_bounds_min[left, 1],
                            bvh_bounds_min[left, 2],
                            bvh_bounds_max[left, 0],
                            bvh_bounds_max[left, 1],
                            bvh_bounds_max[left, 2],
                            best_t,
                        )
                    if right >= 0:
                        right_hit, right_t, _right_exit_t = _intersect_aabb_for_ray_with_interval(
                            ox,
                            oy,
                            oz,
                            dx,
                            dy,
                            dz,
                            bvh_bounds_min[right, 0],
                            bvh_bounds_min[right, 1],
                            bvh_bounds_min[right, 2],
                            bvh_bounds_max[right, 0],
                            bvh_bounds_max[right, 1],
                            bvh_bounds_max[right, 2],
                            best_t,
                        )

                    first = left
                    second = right
                    first_hit = left_hit
                    second_hit = right_hit
                    first_t = left_t
                    second_t = right_t
                    if right_hit and (not left_hit or right_t < left_t):
                        first = right
                        second = left
                        first_hit = right_hit
                        second_hit = left_hit
                        first_t = right_t
                        second_t = left_t

                    if second_hit:
                        if stack_size < _MAX_BVH_STACK:
                            stack[stack_size] = second
                            stack_t[stack_size] = second_t
                            stack_size = stack_size + wp.int32(1)
                        else:
                            wp.atomic_add(bvh_stack_overflow_count, 0, wp.int32(1))
                    if first_hit:
                        if stack_size < _MAX_BVH_STACK:
                            stack[stack_size] = first
                            stack_t[stack_size] = first_t
                            stack_size = stack_size + wp.int32(1)
                        else:
                            wp.atomic_add(bvh_stack_overflow_count, 0, wp.int32(1))
                    if stack_size > local_max_stack:
                        local_max_stack = stack_size

        wp.atomic_max(bvh_max_stack_depth, 0, local_max_stack)

        if found != 0:
            hit_mask[ray] = wp.int32(1)
            range_m[ray] = best_t
            position_world[ray, 0] = best_px
            position_world[ray, 1] = best_py
            position_world[ray, 2] = best_pz
            normal_world[ray, 0] = best_nx
            normal_world[ray, 1] = best_ny
            normal_world[ray, 2] = best_nz
            numeric_instance_id[ray] = best_id
            material_index[ray] = best_material_index

    @wp.struct
    class _ShadowTrace:
        occluded: wp.int32
        overflow: wp.int32
        max_depth: wp.int32
        nodes: wp.int32
        leaves: wp.int32
        triangles: wp.int32
        planes: wp.int32

    @wp.struct
    class _ShadowTotals:
        overflow: wp.int32
        max_depth: wp.int32
        rays: wp.int32
        directional: wp.int32
        point: wp.int32
        occluded: wp.int32
        unoccluded: wp.int32
        nodes: wp.int32
        leaves: wp.int32
        triangles: wp.int32
        planes: wp.int32

    @wp.struct
    class _DirectLightSurface:
        albedo: wp.vec3
        position: wp.vec3
        normal: wp.vec3
        color: wp.vec3

    @wp.struct
    class _DirectLightSample:
        direction: wp.vec3
        attenuation: wp.float32
        cosine: wp.float32
        shadow_limit: wp.float32

    @wp.func
    def _is_occluded_for_ray(
        ox: float,
        oy: float,
        oz: float,
        dx: float,
        dy: float,
        dz: float,
        max_distance: float,
        plane_normals: wp.array2d(dtype=wp.float32),
        plane_points: wp.array2d(dtype=wp.float32),
        plane_role_masks: wp.array(dtype=wp.int64),
        num_planes: int,
        triangle_v0: wp.array2d(dtype=wp.float32),
        triangle_e1: wp.array2d(dtype=wp.float32),
        triangle_e2: wp.array2d(dtype=wp.float32),
        triangle_normal: wp.array2d(dtype=wp.float32),
        triangle_role_masks: wp.array(dtype=wp.int64),
        bvh_bounds_min: wp.array2d(dtype=wp.float32),
        bvh_bounds_max: wp.array2d(dtype=wp.float32),
        bvh_left: wp.array(dtype=wp.int32),
        bvh_right: wp.array(dtype=wp.int32),
        bvh_start: wp.array(dtype=wp.int32),
        bvh_count: wp.array(dtype=wp.int32),
        bvh_prim_ids: wp.array(dtype=wp.int32),
        num_bvh_nodes: int,
        sensor_role_mask: wp.int64,
    ):
        trace = _ShadowTrace()
        occluded = bool(False)
        if max_distance <= 0.0:
            return trace

        stack = wp.zeros(shape=_MAX_BVH_STACK, dtype=wp.int32)
        stack_size = wp.int32(0)
        local_max_stack = wp.int32(0)
        local_node_visit_count = wp.int32(0)
        local_leaf_visit_count = wp.int32(0)
        local_triangle_test_count = wp.int32(0)
        local_plane_test_count = wp.int32(0)
        if num_bvh_nodes > 0:
            root_hit, _root_t, _root_exit_t = _intersect_aabb_for_ray_with_interval(
                ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                bvh_bounds_min[0, 0],
                bvh_bounds_min[0, 1],
                bvh_bounds_min[0, 2],
                bvh_bounds_max[0, 0],
                bvh_bounds_max[0, 1],
                bvh_bounds_max[0, 2],
                max_distance,
            )
            if root_hit:
                stack[0] = wp.int32(0)
                stack_size = wp.int32(1)
                local_max_stack = wp.int32(1)

        while stack_size > 0 and not occluded:
            stack_size = stack_size - wp.int32(1)
            node_index = stack[stack_size]
            local_node_visit_count = local_node_visit_count + wp.int32(1)
            leaf_count = bvh_count[node_index]
            if leaf_count > 0:
                local_leaf_visit_count = local_leaf_visit_count + wp.int32(1)
                leaf_start = bvh_start[node_index]
                for offset in range(leaf_count):
                    tri_idx = bvh_prim_ids[leaf_start + offset]
                    if (triangle_role_masks[tri_idx] & sensor_role_mask) != wp.int64(0):
                        local_triangle_test_count = local_triangle_test_count + wp.int32(1)
                        tri_hit, _t, _fnx, _fny, _fnz = _intersect_device_triangle_for_ray(
                            ox,
                            oy,
                            oz,
                            dx,
                            dy,
                            dz,
                            tri_idx,
                            triangle_v0,
                            triangle_e1,
                            triangle_e2,
                            triangle_normal,
                            max_distance,
                        )
                        if tri_hit:
                            occluded = True
            else:
                left = bvh_left[node_index]
                right = bvh_right[node_index]
                if right >= 0:
                    right_hit, _right_t, _right_exit_t = _intersect_aabb_for_ray_with_interval(
                        ox,
                        oy,
                        oz,
                        dx,
                        dy,
                        dz,
                        bvh_bounds_min[right, 0],
                        bvh_bounds_min[right, 1],
                        bvh_bounds_min[right, 2],
                        bvh_bounds_max[right, 0],
                        bvh_bounds_max[right, 1],
                        bvh_bounds_max[right, 2],
                        max_distance,
                    )
                    if right_hit:
                        if stack_size < _MAX_BVH_STACK:
                            stack[stack_size] = right
                            stack_size = stack_size + wp.int32(1)
                        else:
                            trace.overflow = trace.overflow + wp.int32(1)
                if left >= 0:
                    left_hit, _left_t, _left_exit_t = _intersect_aabb_for_ray_with_interval(
                        ox,
                        oy,
                        oz,
                        dx,
                        dy,
                        dz,
                        bvh_bounds_min[left, 0],
                        bvh_bounds_min[left, 1],
                        bvh_bounds_min[left, 2],
                        bvh_bounds_max[left, 0],
                        bvh_bounds_max[left, 1],
                        bvh_bounds_max[left, 2],
                        max_distance,
                    )
                    if left_hit:
                        if stack_size < _MAX_BVH_STACK:
                            stack[stack_size] = left
                            stack_size = stack_size + wp.int32(1)
                        else:
                            trace.overflow = trace.overflow + wp.int32(1)
                if stack_size > local_max_stack:
                    local_max_stack = stack_size

        trace.max_depth = local_max_stack

        for plane_idx in range(num_planes):
            if not occluded and (plane_role_masks[plane_idx] & sensor_role_mask) != wp.int64(0):
                local_plane_test_count = local_plane_test_count + wp.int32(1)
                nx = plane_normals[plane_idx, 0]
                ny = plane_normals[plane_idx, 1]
                nz = plane_normals[plane_idx, 2]
                px = plane_points[plane_idx, 0]
                py = plane_points[plane_idx, 1]
                pz = plane_points[plane_idx, 2]
                denom = dx * nx + dy * ny + dz * nz
                if wp.abs(denom) > _DIR_EPS:
                    numer = (px - ox) * nx + (py - oy) * ny + (pz - oz) * nz
                    t = numer / denom
                    if t >= 0.0 and t <= max_distance:
                        occluded = True

        trace.occluded = wp.int32(occluded)
        trace.nodes = local_node_visit_count
        trace.leaves = local_leaf_visit_count
        trace.triangles = local_triangle_test_count
        trace.planes = local_plane_test_count
        return trace

    @wp.func
    def _direct_light_begin(
        ray: int,
        hit_mask: wp.array(dtype=wp.int32),
        position_world: wp.array2d(dtype=wp.float32),
        normal_world: wp.array2d(dtype=wp.float32),
        material_index: wp.array(dtype=wp.int32),
        material_albedo_rgb: wp.array2d(dtype=wp.float32),
        ambient_r: float,
        ambient_g: float,
        ambient_b: float,
        background_r: float,
        background_g: float,
        background_b: float,
    ):
        surface = _DirectLightSurface()
        surface.color = wp.vec3(background_r, background_g, background_b)
        if hit_mask[ray] != wp.int32(0):
            mat = material_index[ray]
            surface.albedo = wp.vec3(
                material_albedo_rgb[mat, 0],
                material_albedo_rgb[mat, 1],
                material_albedo_rgb[mat, 2],
            )
            surface.position = wp.vec3(position_world[ray, 0], position_world[ray, 1], position_world[ray, 2])
            surface.normal = wp.vec3(normal_world[ray, 0], normal_world[ray, 1], normal_world[ray, 2])
            surface.color = wp.vec3(
                wp.float32(ambient_r) * surface.albedo[0],
                wp.float32(ambient_g) * surface.albedo[1],
                wp.float32(ambient_b) * surface.albedo[2],
            )
        return surface

    @wp.func
    def _direct_light_sample(
        surface: _DirectLightSurface,
        light_idx: int,
        light_kind: wp.array(dtype=wp.int32),
        light_position_or_direction_world: wp.array2d(dtype=wp.float32),
        shadow_bias: float,
    ):
        sample = _DirectLightSample()
        lx = light_position_or_direction_world[light_idx, 0]
        ly = light_position_or_direction_world[light_idx, 1]
        lz = light_position_or_direction_world[light_idx, 2]
        attenuation = wp.float32(1.0)
        sample.shadow_limit = wp.float32(3.4028234663852886e38)
        if light_kind[light_idx] == wp.int32(1):
            lx = lx - surface.position[0]
            ly = ly - surface.position[1]
            lz = lz - surface.position[2]
            distance_sq = lx * lx + ly * ly + lz * lz
            distance = wp.sqrt(distance_sq)
            sample.shadow_limit = distance - wp.float32(shadow_bias)
            if distance > _DIR_EPS:
                inv_distance = 1.0 / distance
                lx = lx * inv_distance
                ly = ly * inv_distance
                lz = lz * inv_distance
                attenuation = 1.0 / wp.max(distance_sq, _ATTENUATION_EPS)
            else:
                attenuation = wp.float32(0.0)
        else:
            light_norm = wp.sqrt(lx * lx + ly * ly + lz * lz)
            if light_norm > _DIR_EPS:
                inv_light_norm = 1.0 / light_norm
                lx = lx * inv_light_norm
                ly = ly * inv_light_norm
                lz = lz * inv_light_norm
            else:
                attenuation = wp.float32(0.0)
        sample.direction = wp.vec3(lx, ly, lz)
        sample.attenuation = attenuation
        sample.cosine = surface.normal[0] * lx + surface.normal[1] * ly + surface.normal[2] * lz
        return sample

    @wp.func
    def _direct_light_accumulate(
        color: wp.vec3,
        albedo: wp.vec3,
        sample: _DirectLightSample,
        light_idx: int,
        light_intensity: wp.array(dtype=wp.float32),
        light_color_rgb: wp.array2d(dtype=wp.float32),
    ):
        scale = light_intensity[light_idx] * sample.attenuation * sample.cosine
        return wp.vec3(
            color[0] + albedo[0] * light_color_rgb[light_idx, 0] * scale,
            color[1] + albedo[1] * light_color_rgb[light_idx, 1] * scale,
            color[2] + albedo[2] * light_color_rgb[light_idx, 2] * scale,
        )

    @wp.func
    def _direct_light_write(
        ray: int,
        color: wp.vec3,
        hit_mask: wp.array(dtype=wp.int32),
        rgb: wp.array2d(dtype=wp.float32),
        intensity: wp.array(dtype=wp.float32),
    ):
        rgb[ray, 0] = color[0]
        rgb[ray, 1] = color[1]
        rgb[ray, 2] = color[2]
        if hit_mask[ray] != wp.int32(0):
            intensity[ray] = color[0] * 0.2126 + color[1] * 0.7152 + color[2] * 0.0722
        else:
            intensity[ray] = wp.float32(0.0)

    @wp.func
    def _shade_unshadowed_ray(
        ray: int,
        hit_mask: wp.array(dtype=wp.int32),
        position_world: wp.array2d(dtype=wp.float32),
        normal_world: wp.array2d(dtype=wp.float32),
        material_index: wp.array(dtype=wp.int32),
        material_albedo_rgb: wp.array2d(dtype=wp.float32),
        light_kind: wp.array(dtype=wp.int32),
        light_position_or_direction_world: wp.array2d(dtype=wp.float32),
        light_intensity: wp.array(dtype=wp.float32),
        light_color_rgb: wp.array2d(dtype=wp.float32),
        num_lights: int,
        ambient_r: float,
        ambient_g: float,
        ambient_b: float,
        background_r: float,
        background_g: float,
        background_b: float,
    ):
        surface = _direct_light_begin(
            ray,
            hit_mask,
            position_world,
            normal_world,
            material_index,
            material_albedo_rgb,
            ambient_r,
            ambient_g,
            ambient_b,
            background_r,
            background_g,
            background_b,
        )
        color = surface.color
        if hit_mask[ray] != wp.int32(0):
            for light_idx in range(num_lights):
                sample = _direct_light_sample(
                    surface, light_idx, light_kind, light_position_or_direction_world, 0.0
                )
                if sample.cosine > 0.0 and sample.attenuation > 0.0:
                    color = _direct_light_accumulate(
                        color,
                        surface.albedo,
                        sample,
                        light_idx,
                        light_intensity,
                        light_color_rgb,
                    )
        return color

    @wp.func
    def _shade_shadowed_ray(
        ray: int,
        hit_mask: wp.array(dtype=wp.int32),
        position_world: wp.array2d(dtype=wp.float32),
        normal_world: wp.array2d(dtype=wp.float32),
        material_index: wp.array(dtype=wp.int32),
        material_albedo_rgb: wp.array2d(dtype=wp.float32),
        light_kind: wp.array(dtype=wp.int32),
        light_position_or_direction_world: wp.array2d(dtype=wp.float32),
        light_intensity: wp.array(dtype=wp.float32),
        light_color_rgb: wp.array2d(dtype=wp.float32),
        num_lights: int,
        ambient_r: float,
        ambient_g: float,
        ambient_b: float,
        background_r: float,
        background_g: float,
        background_b: float,
        plane_normals: wp.array2d(dtype=wp.float32),
        plane_points: wp.array2d(dtype=wp.float32),
        plane_role_masks: wp.array(dtype=wp.int64),
        num_planes: int,
        triangle_v0: wp.array2d(dtype=wp.float32),
        triangle_e1: wp.array2d(dtype=wp.float32),
        triangle_e2: wp.array2d(dtype=wp.float32),
        triangle_normal: wp.array2d(dtype=wp.float32),
        triangle_role_masks: wp.array(dtype=wp.int64),
        bvh_bounds_min: wp.array2d(dtype=wp.float32),
        bvh_bounds_max: wp.array2d(dtype=wp.float32),
        bvh_left: wp.array(dtype=wp.int32),
        bvh_right: wp.array(dtype=wp.int32),
        bvh_start: wp.array(dtype=wp.int32),
        bvh_count: wp.array(dtype=wp.int32),
        bvh_prim_ids: wp.array(dtype=wp.int32),
        num_bvh_nodes: int,
        sensor_role_mask: wp.int64,
        shadow_bias: float,
    ):
        surface = _direct_light_begin(
            ray,
            hit_mask,
            position_world,
            normal_world,
            material_index,
            material_albedo_rgb,
            ambient_r,
            ambient_g,
            ambient_b,
            background_r,
            background_g,
            background_b,
        )
        color = surface.color
        totals = _ShadowTotals()
        if hit_mask[ray] != wp.int32(0):
            for light_idx in range(num_lights):
                sample = _direct_light_sample(
                    surface, light_idx, light_kind, light_position_or_direction_world, shadow_bias
                )
                if sample.cosine > 0.0 and sample.attenuation > 0.0:
                    visible = bool(True)
                    if sample.shadow_limit > 0.0:
                        trace = _is_occluded_for_ray(
                            surface.position[0] + surface.normal[0] * wp.float32(shadow_bias),
                            surface.position[1] + surface.normal[1] * wp.float32(shadow_bias),
                            surface.position[2] + surface.normal[2] * wp.float32(shadow_bias),
                            sample.direction[0],
                            sample.direction[1],
                            sample.direction[2],
                            sample.shadow_limit,
                            plane_normals,
                            plane_points,
                            plane_role_masks,
                            num_planes,
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
                        )
                        visible = trace.occluded == wp.int32(0)
                        totals.rays = totals.rays + wp.int32(1)
                        if light_kind[light_idx] == wp.int32(1):
                            totals.point = totals.point + wp.int32(1)
                        else:
                            totals.directional = totals.directional + wp.int32(1)
                        totals.occluded = totals.occluded + trace.occluded
                        totals.unoccluded = totals.unoccluded + wp.int32(visible)
                        totals.overflow = totals.overflow + trace.overflow
                        totals.max_depth = wp.max(totals.max_depth, trace.max_depth)
                        totals.nodes = totals.nodes + trace.nodes
                        totals.leaves = totals.leaves + trace.leaves
                        totals.triangles = totals.triangles + trace.triangles
                        totals.planes = totals.planes + trace.planes
                    if visible:
                        color = _direct_light_accumulate(
                            color,
                            surface.albedo,
                            sample,
                            light_idx,
                            light_intensity,
                            light_color_rgb,
                        )
        return color, totals

    @wp.kernel
    def warp_direct_light_no_shadow_primary_v1(
        hit_mask: wp.array(dtype=wp.int32),
        position_world: wp.array2d(dtype=wp.float32),
        normal_world: wp.array2d(dtype=wp.float32),
        material_index: wp.array(dtype=wp.int32),
        material_albedo_rgb: wp.array2d(dtype=wp.float32),
        light_kind: wp.array(dtype=wp.int32),
        light_position_or_direction_world: wp.array2d(dtype=wp.float32),
        light_intensity: wp.array(dtype=wp.float32),
        light_color_rgb: wp.array2d(dtype=wp.float32),
        num_lights: int,
        ambient_r: float,
        ambient_g: float,
        ambient_b: float,
        background_r: float,
        background_g: float,
        background_b: float,
        rgb: wp.array2d(dtype=wp.float32),
        intensity: wp.array(dtype=wp.float32),
        shadow_stack_overflow_count: wp.array(dtype=wp.int32),
        shadow_max_stack_depth: wp.array(dtype=wp.int32),
    ):
        ray = wp.tid()
        color = _shade_unshadowed_ray(
            ray,
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
        )
        _direct_light_write(ray, color, hit_mask, rgb, intensity)

    @wp.kernel
    def warp_direct_light_no_shadow_diagnostics_v1(
        hit_mask: wp.array(dtype=wp.int32),
        position_world: wp.array2d(dtype=wp.float32),
        normal_world: wp.array2d(dtype=wp.float32),
        material_index: wp.array(dtype=wp.int32),
        material_albedo_rgb: wp.array2d(dtype=wp.float32),
        light_kind: wp.array(dtype=wp.int32),
        light_position_or_direction_world: wp.array2d(dtype=wp.float32),
        light_intensity: wp.array(dtype=wp.float32),
        light_color_rgb: wp.array2d(dtype=wp.float32),
        num_lights: int,
        ambient_r: float,
        ambient_g: float,
        ambient_b: float,
        background_r: float,
        background_g: float,
        background_b: float,
        rgb: wp.array2d(dtype=wp.float32),
        intensity: wp.array(dtype=wp.float32),
        shadow_stack_overflow_count: wp.array(dtype=wp.int32),
        shadow_max_stack_depth: wp.array(dtype=wp.int32),
        shadow_traversal_ray_count: wp.array(dtype=wp.int32),
        shadow_traversal_directional_ray_count: wp.array(dtype=wp.int32),
        shadow_traversal_point_ray_count: wp.array(dtype=wp.int32),
        shadow_traversal_occluded_count: wp.array(dtype=wp.int32),
        shadow_traversal_unoccluded_count: wp.array(dtype=wp.int32),
        shadow_traversal_node_visit_count: wp.array(dtype=wp.int32),
        shadow_traversal_leaf_visit_count: wp.array(dtype=wp.int32),
        shadow_traversal_triangle_test_count: wp.array(dtype=wp.int32),
        shadow_traversal_plane_test_count: wp.array(dtype=wp.int32),
    ):
        ray = wp.tid()
        color = _shade_unshadowed_ray(
            ray,
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
        )
        _direct_light_write(ray, color, hit_mask, rgb, intensity)

    @wp.kernel
    def warp_direct_light_with_shadows_primary_v1(
        hit_mask: wp.array(dtype=wp.int32),
        position_world: wp.array2d(dtype=wp.float32),
        normal_world: wp.array2d(dtype=wp.float32),
        material_index: wp.array(dtype=wp.int32),
        material_albedo_rgb: wp.array2d(dtype=wp.float32),
        plane_normals: wp.array2d(dtype=wp.float32),
        plane_points: wp.array2d(dtype=wp.float32),
        plane_role_masks: wp.array(dtype=wp.int64),
        num_planes: int,
        triangle_v0: wp.array2d(dtype=wp.float32),
        triangle_e1: wp.array2d(dtype=wp.float32),
        triangle_e2: wp.array2d(dtype=wp.float32),
        triangle_normal: wp.array2d(dtype=wp.float32),
        triangle_role_masks: wp.array(dtype=wp.int64),
        bvh_bounds_min: wp.array2d(dtype=wp.float32),
        bvh_bounds_max: wp.array2d(dtype=wp.float32),
        bvh_left: wp.array(dtype=wp.int32),
        bvh_right: wp.array(dtype=wp.int32),
        bvh_start: wp.array(dtype=wp.int32),
        bvh_count: wp.array(dtype=wp.int32),
        bvh_prim_ids: wp.array(dtype=wp.int32),
        num_bvh_nodes: int,
        sensor_role_mask: wp.int64,
        light_kind: wp.array(dtype=wp.int32),
        light_position_or_direction_world: wp.array2d(dtype=wp.float32),
        light_intensity: wp.array(dtype=wp.float32),
        light_color_rgb: wp.array2d(dtype=wp.float32),
        num_lights: int,
        shadow_bias: float,
        ambient_r: float,
        ambient_g: float,
        ambient_b: float,
        background_r: float,
        background_g: float,
        background_b: float,
        rgb: wp.array2d(dtype=wp.float32),
        intensity: wp.array(dtype=wp.float32),
        shadow_stack_overflow_count: wp.array(dtype=wp.int32),
        shadow_max_stack_depth: wp.array(dtype=wp.int32),
    ):
        ray = wp.tid()
        color, totals = _shade_shadowed_ray(
            ray,
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
            plane_normals,
            plane_points,
            plane_role_masks,
            num_planes,
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
        )
        wp.atomic_add(shadow_stack_overflow_count, 0, totals.overflow)
        wp.atomic_max(shadow_max_stack_depth, 0, totals.max_depth)
        _direct_light_write(ray, color, hit_mask, rgb, intensity)

    @wp.kernel
    def warp_direct_light_with_shadows_diagnostics_v1(
        hit_mask: wp.array(dtype=wp.int32),
        position_world: wp.array2d(dtype=wp.float32),
        normal_world: wp.array2d(dtype=wp.float32),
        material_index: wp.array(dtype=wp.int32),
        material_albedo_rgb: wp.array2d(dtype=wp.float32),
        plane_normals: wp.array2d(dtype=wp.float32),
        plane_points: wp.array2d(dtype=wp.float32),
        plane_role_masks: wp.array(dtype=wp.int64),
        num_planes: int,
        triangle_v0: wp.array2d(dtype=wp.float32),
        triangle_e1: wp.array2d(dtype=wp.float32),
        triangle_e2: wp.array2d(dtype=wp.float32),
        triangle_normal: wp.array2d(dtype=wp.float32),
        triangle_role_masks: wp.array(dtype=wp.int64),
        bvh_bounds_min: wp.array2d(dtype=wp.float32),
        bvh_bounds_max: wp.array2d(dtype=wp.float32),
        bvh_left: wp.array(dtype=wp.int32),
        bvh_right: wp.array(dtype=wp.int32),
        bvh_start: wp.array(dtype=wp.int32),
        bvh_count: wp.array(dtype=wp.int32),
        bvh_prim_ids: wp.array(dtype=wp.int32),
        num_bvh_nodes: int,
        sensor_role_mask: wp.int64,
        light_kind: wp.array(dtype=wp.int32),
        light_position_or_direction_world: wp.array2d(dtype=wp.float32),
        light_intensity: wp.array(dtype=wp.float32),
        light_color_rgb: wp.array2d(dtype=wp.float32),
        num_lights: int,
        shadow_bias: float,
        ambient_r: float,
        ambient_g: float,
        ambient_b: float,
        background_r: float,
        background_g: float,
        background_b: float,
        rgb: wp.array2d(dtype=wp.float32),
        intensity: wp.array(dtype=wp.float32),
        shadow_stack_overflow_count: wp.array(dtype=wp.int32),
        shadow_max_stack_depth: wp.array(dtype=wp.int32),
        shadow_traversal_ray_count: wp.array(dtype=wp.int32),
        shadow_traversal_directional_ray_count: wp.array(dtype=wp.int32),
        shadow_traversal_point_ray_count: wp.array(dtype=wp.int32),
        shadow_traversal_occluded_count: wp.array(dtype=wp.int32),
        shadow_traversal_unoccluded_count: wp.array(dtype=wp.int32),
        shadow_traversal_node_visit_count: wp.array(dtype=wp.int32),
        shadow_traversal_leaf_visit_count: wp.array(dtype=wp.int32),
        shadow_traversal_triangle_test_count: wp.array(dtype=wp.int32),
        shadow_traversal_plane_test_count: wp.array(dtype=wp.int32),
    ):
        ray = wp.tid()
        color, totals = _shade_shadowed_ray(
            ray,
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
            plane_normals,
            plane_points,
            plane_role_masks,
            num_planes,
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
        )
        wp.atomic_add(shadow_stack_overflow_count, 0, totals.overflow)
        wp.atomic_max(shadow_max_stack_depth, 0, totals.max_depth)
        wp.atomic_add(shadow_traversal_ray_count, 0, totals.rays)
        wp.atomic_add(shadow_traversal_directional_ray_count, 0, totals.directional)
        wp.atomic_add(shadow_traversal_point_ray_count, 0, totals.point)
        wp.atomic_add(shadow_traversal_occluded_count, 0, totals.occluded)
        wp.atomic_add(shadow_traversal_unoccluded_count, 0, totals.unoccluded)
        wp.atomic_add(shadow_traversal_node_visit_count, 0, totals.nodes)
        wp.atomic_add(shadow_traversal_leaf_visit_count, 0, totals.leaves)
        wp.atomic_add(shadow_traversal_triangle_test_count, 0, totals.triangles)
        wp.atomic_add(shadow_traversal_plane_test_count, 0, totals.planes)
        _direct_light_write(ray, color, hit_mask, rgb, intensity)

    @wp.kernel
    def _reset_bvh_primary_counters_kernel(
        bvh_stack_overflow_count: wp.array(dtype=wp.int32),
        bvh_max_stack_depth: wp.array(dtype=wp.int32),
    ):
        bvh_stack_overflow_count[0] = wp.int32(0)
        bvh_max_stack_depth[0] = wp.int32(0)

    @wp.kernel
    def _reset_shadow_primary_counters_kernel(
        shadow_stack_overflow_count: wp.array(dtype=wp.int32),
        shadow_max_stack_depth: wp.array(dtype=wp.int32),
    ):
        shadow_stack_overflow_count[0] = wp.int32(0)
        shadow_max_stack_depth[0] = wp.int32(0)

    @wp.kernel
    def _reset_shadow_diagnostic_counters_kernel(
        shadow_stack_overflow_count: wp.array(dtype=wp.int32),
        shadow_max_stack_depth: wp.array(dtype=wp.int32),
        shadow_traversal_ray_count: wp.array(dtype=wp.int32),
        shadow_traversal_directional_ray_count: wp.array(dtype=wp.int32),
        shadow_traversal_point_ray_count: wp.array(dtype=wp.int32),
        shadow_traversal_occluded_count: wp.array(dtype=wp.int32),
        shadow_traversal_unoccluded_count: wp.array(dtype=wp.int32),
        shadow_traversal_node_visit_count: wp.array(dtype=wp.int32),
        shadow_traversal_leaf_visit_count: wp.array(dtype=wp.int32),
        shadow_traversal_triangle_test_count: wp.array(dtype=wp.int32),
        shadow_traversal_plane_test_count: wp.array(dtype=wp.int32),
    ):
        shadow_stack_overflow_count[0] = wp.int32(0)
        shadow_max_stack_depth[0] = wp.int32(0)
        shadow_traversal_ray_count[0] = wp.int32(0)
        shadow_traversal_directional_ray_count[0] = wp.int32(0)
        shadow_traversal_point_ray_count[0] = wp.int32(0)
        shadow_traversal_occluded_count[0] = wp.int32(0)
        shadow_traversal_unoccluded_count[0] = wp.int32(0)
        shadow_traversal_node_visit_count[0] = wp.int32(0)
        shadow_traversal_leaf_visit_count[0] = wp.int32(0)
        shadow_traversal_triangle_test_count[0] = wp.int32(0)
        shadow_traversal_plane_test_count[0] = wp.int32(0)

else:

    def _pinhole_camera_raygen_kernel(*args, **kwargs):  # pragma: no cover
        raise ImportError("Warp optical ray generation requires the optional warp package")

    def _init_bvh_hit_mask_kernel(*args, **kwargs):  # pragma: no cover
        raise ImportError("Warp BVH first-hit requires the optional warp package")

    def _brute_force_first_hit_kernel(*args, **kwargs):  # pragma: no cover
        raise ImportError("GpuBruteForceOpticalExecutor requires the optional warp package")

    def _device_scene_first_hit_kernel(*args, **kwargs):  # pragma: no cover
        raise ImportError("GpuDeviceSceneOpticalExecutor requires the optional warp package")

    def _device_scene_first_hit_aabb_kernel(*args, **kwargs):  # pragma: no cover
        raise ImportError("GpuDeviceSceneOpticalExecutor requires the optional warp package")

    def _device_scene_bvh_first_hit_kernel(*args, **kwargs):  # pragma: no cover
        raise ImportError("GpuDeviceBvhOpticalExecutor requires the optional warp package")

    def warp_direct_light_no_shadow_primary_v1(*args, **kwargs):  # pragma: no cover
        raise ImportError("Warp optical shading requires the optional warp package")

    def warp_direct_light_no_shadow_diagnostics_v1(*args, **kwargs):  # pragma: no cover
        raise ImportError("Warp optical shading requires the optional warp package")

    def warp_direct_light_with_shadows_primary_v1(*args, **kwargs):  # pragma: no cover
        raise ImportError("Warp optical shading requires the optional warp package")

    def warp_direct_light_with_shadows_diagnostics_v1(*args, **kwargs):  # pragma: no cover
        raise ImportError("Warp optical shading requires the optional warp package")

    def _reset_bvh_primary_counters_kernel(*args, **kwargs):  # pragma: no cover
        raise ImportError("Warp optical reset requires the optional warp package")

    def _reset_shadow_primary_counters_kernel(*args, **kwargs):  # pragma: no cover
        raise ImportError("Warp optical reset requires the optional warp package")

    def _reset_shadow_diagnostic_counters_kernel(*args, **kwargs):  # pragma: no cover
        raise ImportError("Warp optical reset requires the optional warp package")


# Stable argument semantics used by preparation and compatibility. Tests bind
# these ordered declarations directly to the four actual kernel signatures.
_WARP_SHADE_SURFACE_ARGUMENTS = (
    "hit_mask",
    "position_world",
    "normal_world",
    "material_index",
    "material_albedo_rgb",
)
_WARP_SHADE_LIGHT_ARGUMENTS = (
    "light_kind",
    "light_position_or_direction_world",
    "light_intensity",
    "light_color_rgb",
    "num_lights",
)
_WARP_SHADE_COLOR_ARGUMENTS = (
    "ambient_r",
    "ambient_g",
    "ambient_b",
    "background_r",
    "background_g",
    "background_b",
)
_WARP_SHADE_OUTPUT_ARGUMENTS = (
    "rgb",
    "intensity",
    "shadow_stack_overflow_count",
    "shadow_max_stack_depth",
)
_WARP_SHADE_GEOMETRY_ARGUMENTS = (
    "plane_normals",
    "plane_points",
    "plane_role_masks",
    "num_planes",
    "triangle_v0",
    "triangle_e1",
    "triangle_e2",
    "triangle_normal",
    "triangle_role_masks",
    "bvh_bounds_min",
    "bvh_bounds_max",
    "bvh_left",
    "bvh_right",
    "bvh_start",
    "bvh_count",
    "bvh_prim_ids",
    "num_bvh_nodes",
    "sensor_role_mask",
)

WARP_DIRECT_LIGHT_SHADE_ABIS = MappingProxyType(
    {
        (False, False): (
            warp_direct_light_no_shadow_primary_v1,
            _WARP_SHADE_SURFACE_ARGUMENTS
            + _WARP_SHADE_LIGHT_ARGUMENTS
            + _WARP_SHADE_COLOR_ARGUMENTS
            + _WARP_SHADE_OUTPUT_ARGUMENTS,
        ),
        (False, True): (
            warp_direct_light_no_shadow_diagnostics_v1,
            _WARP_SHADE_SURFACE_ARGUMENTS
            + _WARP_SHADE_LIGHT_ARGUMENTS
            + _WARP_SHADE_COLOR_ARGUMENTS
            + _WARP_SHADE_OUTPUT_ARGUMENTS
            + SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES,
        ),
        (True, False): (
            warp_direct_light_with_shadows_primary_v1,
            _WARP_SHADE_SURFACE_ARGUMENTS
            + _WARP_SHADE_GEOMETRY_ARGUMENTS
            + _WARP_SHADE_LIGHT_ARGUMENTS
            + ("shadow_bias",)
            + _WARP_SHADE_COLOR_ARGUMENTS
            + _WARP_SHADE_OUTPUT_ARGUMENTS,
        ),
        (True, True): (
            warp_direct_light_with_shadows_diagnostics_v1,
            _WARP_SHADE_SURFACE_ARGUMENTS
            + _WARP_SHADE_GEOMETRY_ARGUMENTS
            + _WARP_SHADE_LIGHT_ARGUMENTS
            + ("shadow_bias",)
            + _WARP_SHADE_COLOR_ARGUMENTS
            + _WARP_SHADE_OUTPUT_ARGUMENTS
            + SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES,
        ),
    }
)


def warp_direct_light_shade_entry(*, shadows: bool, traversal_diagnostics: bool) -> tuple:
    """Select a fixed ABI during preparation, never during a native enqueue."""
    if type(shadows) is not bool or type(traversal_diagnostics) is not bool:
        raise TypeError("shade modes must be exact bools")
    if not _HAS_WARP:
        raise RuntimeError("Warp is required for native optical shading")
    return WARP_DIRECT_LIGHT_SHADE_ABIS[shadows, traversal_diagnostics]


_RENDER_BVH_ARRAY_NAMES = (
    "bounds_min",
    "bounds_max",
    "left",
    "right",
    "start",
    "count",
    "prim_ids",
)


def _render_kernel_owners() -> tuple:
    primary = (
        _pinhole_camera_raygen_kernel,
        _reset_bvh_primary_counters_kernel,
        _device_scene_bvh_first_hit_kernel,
        _reset_shadow_primary_counters_kernel,
        _reset_shadow_diagnostic_counters_kernel,
    )
    variants = tuple(
        warp_direct_light_shade_entry(shadows=s, traversal_diagnostics=d)[0]
        for s, d in ((False, False), (False, True), (True, False), (True, True))
    )
    for kernel, names in WARP_DIRECT_LIGHT_SHADE_ABIS.values():
        if globals()[kernel.func.__name__] is not kernel or tuple(a.label for a in kernel.adj.args) != names:
            raise ValueError("render fixed shade ABI registry differs from actual kernel")
    return primary + variants


def _render_kernel_facts(kernels: tuple) -> tuple:
    return tuple(
        (
            id(k),
            id(k.func),
            id(k.func.__code__),
            id(k.module),
            id(k.adj),
            tuple((a.label, a.type) for a in k.adj.args),
        )
        for k in kernels
    )


def _render_input_items(scene, world, bvh) -> tuple:
    from .device_scene import _warp_local_scene_arrays, _warp_world_scene_arrays

    return (
        _warp_local_scene_arrays(scene)
        + _warp_world_scene_arrays(world)
        + tuple((f"bvh_{name}", getattr(bvh, name)) for name in _RENDER_BVH_ARRAY_NAMES)
    )


def _render_storage_facts(items: tuple, device) -> tuple:
    return tuple((name, _render_array_snapshot(value, name=name, device=device)) for name, value in items)


def _render_storage_key(facts: tuple) -> tuple:
    return tuple((name, _render_snapshot_key(snapshot)) for name, snapshot in facts)


def _render_storage_digest(items: tuple, device) -> str:
    facts = []
    for name, value in items:
        shape, strides, pointer, count = _render_array_layout(value, name=name, device=device)
        facts.append(
            (
                name,
                id(value),
                pointer,
                count,
                tuple(shape),
                tuple(strides),
                value.dtype.__name__,
                int(device.ordinal),
            )
        )
    return _render_digest(facts)


def _render_factory_facts(factory) -> tuple:
    from optical_worker.native_contract import assert_warp_prepared_authority

    from .device_scene import DeviceOpticalScene, _warp_local_scene_arrays

    assert_warp_prepared_authority(factory.prepared_input, factory.readiness)
    prepared, scene = factory.prepared_input, factory.local_scene
    if type(scene) is not DeviceOpticalScene:
        raise TypeError("render factory requires the exact production local scene")
    if factory.device is not scene.device or int(factory.device.ordinal) != prepared.device_ordinal:
        raise ValueError("render factory target device differs from prepared input")
    if type(factory.stream) is not wp.Stream or factory.stream.device is not factory.device:
        raise ValueError("render factory requires an exact stream on its indexed device")
    if type(factory.stream.cuda_stream) is not int or factory.stream.cuda_stream <= 0:
        raise ValueError("render factory requires a strongly owned nondefault CUDA stream")
    if factory.stream.owner is not True:
        raise ValueError("render factory requires a Warp-owned stream, not a borrowed handle")
    manifest = prepared.manifest
    if manifest.input_contract not in ("fixed_native_scene_v1", "canonical_native_replay_v1"):
        raise ValueError("render factory accepts only fixed or native replay input")
    packed = prepared.source.packed_scene_owner
    if packed is None or scene.scene_contract_sha256 != manifest.scene_contract_sha256:
        raise ValueError("render local scene is not bound to the prepared scene contract")
    for name in (
        "num_instances",
        "num_materials",
        "num_lights",
        "num_triangles",
        "num_planes",
        "max_body_index",
    ):
        if type(getattr(scene, name)) is not int or getattr(scene, name) != getattr(packed, name):
            raise ValueError(f"local scene count differs from prepared source: {name}")
    if dict(scene.role_table.role_to_mask) != dict(packed.role_to_mask):
        raise ValueError("local scene role mapping differs from prepared source")
    items = _warp_local_scene_arrays(scene)
    if set(name for name, _ in items) != set(manifest.scene_array_names):
        raise ValueError("local scene array semantics differ from prepared source")
    for name, value in items:
        if value is not prepared.arrays[name]:
            raise ValueError(f"local scene {name} is not the exact a2 allocation")
    return (
        id(prepared),
        id(factory.readiness),
        id(scene),
        id(factory.device),
        id(factory.stream),
        int(factory.stream.cuda_stream),
        factory.stream.owner,
        manifest.sha256,
        factory.readiness.sha256,
        tuple(
            (name, getattr(scene, name))
            for name in (
                "num_instances",
                "num_materials",
                "num_lights",
                "num_triangles",
                "num_planes",
                "max_body_index",
            )
        ),
        tuple(sorted(scene.role_table.role_to_mask.items())),
        _render_storage_key(_render_storage_facts(items, factory.device)),
        _render_kernel_facts(_render_kernel_owners()),
    )


@dataclass(frozen=True, slots=True, eq=False)
class WarpDirectLightRenderCommandFactory:
    """Untimed immutable binder. It never launches a recorded GPU command.

    Setup/recording belongs to the future supervised setup owner. This factory
    supplies no driver, drain, lease or completion authority.
    """

    prepared_input: object
    readiness: object
    local_scene: object
    device: object
    stream: object
    _signature: tuple = field(init=False, repr=False)
    _source_sha256: str = field(init=False, repr=False)
    _input_owners: tuple = field(init=False, repr=False)
    _kernel_owners: tuple = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if hasattr(self, "_signature"):
            raise RuntimeError("render factory baseline cannot be rebuilt")
        facts = _render_factory_facts(self)
        from .device_scene import _warp_local_scene_arrays

        object.__setattr__(
            self,
            "_input_owners",
            _render_storage_facts(_warp_local_scene_arrays(self.local_scene), self.device),
        )
        object.__setattr__(self, "_source_sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        object.__setattr__(self, "_kernel_owners", _render_kernel_owners())
        object.__setattr__(self, "_signature", facts)

    @classmethod
    def prepare(cls, *, prepared_input, readiness, local_scene, device, stream):
        factory = cls(prepared_input, readiness, local_scene, device, stream)
        # Compilation is setup work, never a native render command. Execution
        # warmup remains in the supervised driver and is not claimed here.
        with wp.ScopedStream(stream):
            wp.load_module(module=__name__, device=device)
        return factory

    def validate_bound(self) -> None:
        if _render_factory_facts(self) != self._signature:
            raise ValueError("render factory provenance/storage/stream changed")
        if _render_storage_key(self._input_owners) != self._signature[-2]:
            raise ValueError("render factory owner retention changed")
        if _render_kernel_facts(self._kernel_owners) != self._signature[-1]:
            raise ValueError("render factory kernel retention changed")
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != self._source_sha256:
            raise ValueError("render production source changed after factory preparation")

    def seal_camera(
        self,
        *,
        world,
        bvh,
        camera,
        buffers,
        output_profile,
        shadows,
        traversal_diagnostics,
        transform_frame_index=None,
        ambient_rgb=(0.1, 0.1, 0.1),
        background_rgb=(0.0, 0.0, 0.0),
        shadow_bias=1e-4,
    ):
        if type(camera) is not WarpCameraParameters:
            raise TypeError("render camera must be the exact normalized camera type")
        camera.validate()
        return self._seal(
            world=world,
            bvh=bvh,
            camera=camera,
            query=camera.query,
            buffers=buffers,
            output_profile=output_profile,
            shadows=shadows,
            traversal_diagnostics=traversal_diagnostics,
            transform_frame_index=transform_frame_index,
            ambient_rgb=ambient_rgb,
            background_rgb=background_rgb,
            shadow_bias=shadow_bias,
        )

    def seal_device_rays(
        self,
        *,
        world,
        bvh,
        query,
        buffers,
        output_profile,
        shadows,
        traversal_diagnostics,
        transform_frame_index=None,
        ambient_rgb=(0.1, 0.1, 0.1),
        background_rgb=(0.0, 0.0, 0.0),
        shadow_bias=1e-4,
    ):
        """Compatibility-only resident rays; never an eligible camera binding."""
        return self._seal(
            world=world,
            bvh=bvh,
            camera=None,
            query=query,
            buffers=buffers,
            output_profile=output_profile,
            shadows=shadows,
            traversal_diagnostics=traversal_diagnostics,
            transform_frame_index=transform_frame_index,
            ambient_rgb=ambient_rgb,
            background_rgb=background_rgb,
            shadow_bias=shadow_bias,
        )

    def _seal(self, **values):
        self.validate_bound()
        # The binding is validated in full before any record_cmd/JIT entry.
        binding = _WarpRenderBinding(factory=self, **values)
        descriptors = binding.command_descriptors()
        commands = []
        with wp.ScopedStream(self.stream):
            for descriptor in descriptors:
                commands.append(
                    wp.launch(
                        descriptor.kernel,
                        dim=descriptor.dim,
                        inputs=descriptor.arguments,
                        device=self.device,
                        stream=self.stream,
                        record_tape=False,
                        record_cmd=True,
                    )
                )
        return WarpDirectLightRenderCommandBundle(
            binding=binding, descriptors=descriptors, commands=tuple(commands)
        )


@dataclass(frozen=True, slots=True, eq=False)
class _WarpRenderDescriptor:
    entry_id: str
    kernel: object
    dim: int
    argument_names: tuple[str, ...]
    arguments: tuple

    def payload(self) -> dict:
        arguments = []
        for name, value in zip(self.argument_names, self.arguments, strict=True):
            if type(value) is wp.array:
                arguments.append((name, "array", id(value), int(value.ptr or 0)))
            else:
                arguments.append((name, "scalar", type(value).__name__, value))
        return dict(
            entry_id=self.entry_id, kernel_id=id(self.kernel), dimension=self.dim, arguments=arguments
        )


@dataclass(frozen=True, slots=True, eq=False)
class _WarpRenderBinding:
    factory: WarpDirectLightRenderCommandFactory
    world: object
    bvh: object
    camera: WarpCameraParameters | None
    query: WarpRenderQueryParameters
    buffers: WarpDirectLightBuffers
    output_profile: object
    shadows: bool
    traversal_diagnostics: bool
    transform_frame_index: int | None
    ambient_rgb: tuple
    background_rgb: tuple
    shadow_bias: float
    _snapshots: tuple = field(init=False, repr=False)
    _signature: tuple = field(init=False, repr=False)

    def __post_init__(self):
        if hasattr(self, "_signature"):
            raise RuntimeError("render binding baseline cannot be rebuilt")
        for name in ("ambient_rgb", "background_rgb"):
            values = getattr(self, name)
            if type(values) is not tuple or len(values) != 3:
                raise ValueError(f"{name} must be an exact three-scalar tuple")
            object.__setattr__(self, name, tuple(_render_f32(v) for v in values))
        object.__setattr__(self, "shadow_bias", _render_f32(self.shadow_bias))
        self._validate()
        snapshots = _render_storage_facts(self.input_items, self.factory.device)
        object.__setattr__(self, "_snapshots", snapshots)
        object.__setattr__(self, "_signature", self._facts(snapshots))

    @property
    def input_items(self):
        return _render_input_items(self.factory.local_scene, self.world, self.bvh)

    def _validate(self):
        from .device_bvh import DeviceOpticalBvh, _validate_warp_bvh_storage
        from .device_scene import WarpWorldSceneBuffers, _validate_warp_world_scene_buffers
        from .execution import OpticalOutputProfile

        if type(self.factory) is not WarpDirectLightRenderCommandFactory:
            raise TypeError("render binding requires the exact prepared factory")
        self.factory.validate_bound()
        scene, manifest = self.factory.local_scene, self.factory.prepared_input.manifest
        if type(self.query) is not WarpRenderQueryParameters:
            raise TypeError("render query must be exact normalized parameters")
        self.query.validate()
        if self.camera is not None:
            if type(self.camera) is not WarpCameraParameters or self.camera.query is not self.query:
                raise ValueError("render camera/query binding changed")
            self.camera.validate()
        if type(self.output_profile) is not OpticalOutputProfile or self.output_profile.value not in (
            "direct_light_full",
            "rgb_preview",
            "render_only",
        ):
            raise ValueError("render binding requires a direct-light output profile")
        if type(self.shadows) is not bool or type(self.traversal_diagnostics) is not bool:
            raise TypeError("render modes must be exact bools")
        if type(self.buffers) is not WarpDirectLightBuffers:
            raise TypeError("render outputs must be exact caller-owned buffers")
        if (
            self.buffers.num_rays != self.query.num_rays
            or self.buffers.traversal_diagnostics != self.traversal_diagnostics
        ):
            raise ValueError("render buffer shape or diagnostic mode differs from the query")
        if type(self.world) is not WarpWorldSceneBuffers or type(self.bvh) is not DeviceOpticalBvh:
            raise TypeError("render requires exact world scene and BVH objects")
        _validate_warp_world_scene_buffers(scene, self.world)
        _validate_warp_bvh_storage(self.bvh)
        if self.bvh.device is not self.factory.device or self.buffers.device is not self.factory.device:
            raise ValueError("render storage device differs from factory target")
        if (
            self.bvh.num_primitives != scene.num_triangles
            or self.bvh.scene_contract_sha256 != scene.scene_contract_sha256
        ):
            raise ValueError("render BVH differs from the prepared scene")
        if (self.bvh.frame_id, self.bvh.env_idx) != (self.query.frame_id, self.query.env_idx):
            raise ValueError("render BVH frame/environment differs from query")
        if scene.role_table.mask_for(self.query.sensor_role) != self.query.sensor_role_mask:
            raise ValueError("render role mask differs from exact scene role table")
        if self.shadow_bias < 0:
            raise ValueError("render shadow bias must be nonnegative")
        if manifest.input_contract == "fixed_native_scene_v1":
            if self.transform_frame_index is not None:
                raise ValueError("fixed render input cannot have a replay frame")
            arrays = self.factory.prepared_input.arrays
            for name, value in self.input_items:
                if value is not arrays[name]:
                    raise ValueError(f"fixed render {name} is not the exact prepared allocation")
        elif (
            type(self.transform_frame_index) is not int
            or not 0 <= self.transform_frame_index < manifest.transform_frame_count
        ):
            raise ValueError("render replay frame is outside the prepared sequence")
        self.buffers.validate_bound(read_only_inputs=self.input_items)

    def _facts(self, snapshots):
        return (
            id(self.factory),
            id(self.world),
            id(self.bvh),
            id(self.camera),
            id(self.query),
            id(self.buffers),
            self.output_profile.value,
            self.shadows,
            self.traversal_diagnostics,
            self.transform_frame_index,
            self.ambient_rgb,
            self.background_rgb,
            self.shadow_bias,
            self.query.sha256,
            None if self.camera is None else self.camera.sha256,
            self.bvh.frame_id,
            self.bvh.env_idx,
            self.bvh.num_nodes,
            self.bvh.num_primitives,
            self.bvh.scene_contract_sha256,
            id(self.world.resources),
            id(self.bvh.resources),
            _render_storage_key(snapshots),
        )

    def validate_bound(self):
        self._validate()
        if self._facts(_render_storage_facts(self.input_items, self.factory.device)) != self._signature:
            raise ValueError("render binding storage or configuration changed")
        if _render_storage_key(self._snapshots) != self._signature[-1]:
            raise ValueError("render binding owner retention changed")

    def command_descriptors(self) -> tuple[_WarpRenderDescriptor, ...]:
        scene, world, bvh, buffers, query = (
            self.factory.local_scene,
            self.world,
            self.bvh,
            self.buffers,
            self.query,
        )
        values = dict(buffers.arrays)
        values.update(
            {
                "max_distance": query.max_distance,
                "plane_normals": world.plane_normal_world,
                "plane_points": world.plane_point_world,
                "plane_numeric_ids": scene.plane_numeric_instance_id,
                "plane_source_keys": scene.plane_source_order_key,
                "plane_role_masks": scene.plane_role_mask,
                "plane_material_indices": scene.plane_material_index,
                "num_planes": scene.num_planes,
                "triangle_v0": world.triangle_v0_world,
                "triangle_e1": world.triangle_e1_world,
                "triangle_e2": world.triangle_e2_world,
                "triangle_normal": world.triangle_normal_world,
                "triangle_numeric_ids": scene.triangle_numeric_instance_id,
                "triangle_source_keys": scene.triangle_source_order_key,
                "triangle_role_masks": scene.triangle_role_mask,
                "triangle_material_indices": scene.triangle_material_index,
                **{f"bvh_{name}": getattr(bvh, name) for name in _RENDER_BVH_ARRAY_NAMES},
                "num_bvh_nodes": bvh.num_nodes,
                "sensor_role_mask": query.sensor_role_mask,
                **{
                    name: getattr(scene, name)
                    for name in (*_WARP_SHADE_LIGHT_ARGUMENTS, "material_albedo_rgb")
                },
                "shadow_bias": self.shadow_bias,
                **dict(zip(_WARP_SHADE_COLOR_ARGUMENTS, self.ambient_rgb + self.background_rgb, strict=True)),
            }
        )
        if self.camera is not None:
            camera = self.camera
            values.update(
                dict(
                    width=camera.width,
                    height=camera.height,
                    fx=camera.fx,
                    fy=camera.fy,
                    cx=camera.cx,
                    cy=camera.cy,
                )
            )
            values.update(dict(zip(("origin_x", "origin_y", "origin_z"), camera.origin, strict=True)))
            values.update(
                dict(zip((f"R{i}{j}" for i in range(3) for j in range(3)), camera.rotation, strict=True))
            )
        entries = _render_command_ids(
            "camera" if self.camera else "device_rays",
            query.num_rays,
            self.shadows,
            self.traversal_diagnostics,
        )
        descriptors = []
        for entry in entries:
            kernel = globals()[entry]
            names = tuple(arg.label for arg in kernel.adj.args)
            descriptors.append(
                _WarpRenderDescriptor(
                    entry,
                    kernel,
                    1 if "reset" in entry else query.num_rays,
                    names,
                    tuple(values[name] for name in names),
                )
            )
        return tuple(descriptors)

    def evidence(self, descriptors):
        manifest, factory = self.factory.prepared_input.manifest, self.factory
        return WarpDirectLightRenderCommandEvidence(
            schema_version=WARP_RENDER_COMMAND_SCHEMA,
            operation_id="warp_bvh_direct_light",
            kernel_abi_id=WARP_RENDER_ABI,
            kernel_source_sha256=factory._source_sha256,
            device_ordinal=int(factory.device.ordinal),
            stream_handle=int(factory.stream.cuda_stream),
            native_input_manifest_sha256=manifest.sha256,
            readiness_sha256=factory.readiness.sha256,
            scene_contract_sha256=manifest.scene_contract_sha256,
            input_contract=manifest.input_contract,
            fixed_render_fixture_sha256=manifest.fixed_render_fixture_sha256,
            transform_replay_sha256=manifest.transform_replay_sha256,
            transform_frame_index=self.transform_frame_index,
            query_sha256=self.query.sha256,
            camera_contract_sha256=self.camera.sha256 if self.camera else None,
            ray_input_storage_sha256=None
            if self.camera
            else _render_storage_digest(self.buffers.arrays[:2], factory.device),
            ray_source="camera" if self.camera else "device_rays",
            num_rays=self.query.num_rays,
            output_profile=self.output_profile.value,
            shadows=self.shadows,
            traversal_diagnostics=self.traversal_diagnostics,
            input_storage_sha256=_render_storage_digest(self.input_items, factory.device),
            output_storage_sha256=_render_storage_digest(self.buffers.arrays, factory.device),
            command_entry_ids=tuple(d.entry_id for d in descriptors),
            command_plan_sha256=_render_digest([d.payload() for d in descriptors]),
            dispatch_count=len(descriptors),
        )


def _render_launch_snapshot(command, descriptor, device) -> tuple:
    """Inspect the actual packed native leaf, not only the Python argument tuple."""
    if (
        type(command) is not wp.Launch
        or command.kernel is not descriptor.kernel
        or command.device is not device
    ):
        raise ValueError("recorded launch kernel/device identity differs from descriptor")
    if "launch" in vars(command) or command.adjoint or command.max_blocks != 0 or command.block_dim != 256:
        raise ValueError("recorded launch execution settings changed")
    if (
        command.bounds.ndim != 1
        or command.bounds.size != descriptor.dim
        or tuple(command.bounds.shape) != (descriptor.dim, 1, 1, 1)
    ):
        raise ValueError("recorded launch dimensions differ from descriptor")
    if len(command.params) != len(descriptor.arguments) + 1 or len(command.params_addr) != len(
        command.params
    ):
        raise ValueError("recorded launch parameter count differs from descriptor")
    if command.params[0] is not command.bounds:
        raise ValueError("recorded bounds owner differs from actual parameter")
    snapshots = []
    for index, parameter in enumerate(command.params):
        if command.params_addr[index] != ctypes.addressof(parameter):
            raise ValueError("recorded launch native argument pointer changed")
        if index:
            expected = descriptor.arguments[index - 1]
            if type(expected) is wp.array:
                shape, strides, pointer, _ = _render_array_layout(
                    expected, name=descriptor.argument_names[index - 1], device=device
                )
                if (
                    type(parameter) is not _WarpArrayDescriptor
                    or parameter.data != pointer
                    or parameter.ndim != len(shape)
                    or tuple(parameter.shape) != shape + (0,) * (4 - len(shape))
                    or tuple(parameter.strides) != strides + (0,) * (4 - len(strides))
                    or parameter.grad != (int(expected.grad.ptr or 0) if expected.grad is not None else 0)
                ):
                    raise ValueError("recorded array descriptor differs from exact bound storage")
            else:
                scalar_type = (
                    ctypes.c_float
                    if type(expected) is float
                    else (
                        ctypes.c_int64
                        if descriptor.argument_names[index - 1] == "sensor_role_mask"
                        else ctypes.c_int32
                    )
                )
                if type(parameter) is not scalar_type or parameter.value != expected:
                    raise ValueError("recorded scalar differs from normalized binding")
        snapshots.append((id(parameter), ctypes.addressof(parameter), bytes(parameter)))
    return (
        id(command),
        id(command.kernel),
        id(command.module_exec),
        id(command.hooks),
        id(command.device),
        command.hooks.forward,
        command.hooks.backward,
        command.hooks.forward_smem_bytes,
        command.hooks.backward_smem_bytes,
        id(command.params),
        id(command.params_addr),
        tuple(snapshots),
        command.adjoint,
        command.max_blocks,
        command.block_dim,
    )


@dataclass(frozen=True, slots=True, eq=False)
class WarpDirectLightRenderCommandBundle:
    """Sealed recorded owners/evidence, deliberately without an execution API.

    Only the future supervised C4c1c driver may execute these commands; this
    object itself has no event, completion ticket, result capability or lease.
    """

    binding: _WarpRenderBinding
    descriptors: tuple[_WarpRenderDescriptor, ...]
    commands: tuple
    evidence: WarpDirectLightRenderCommandEvidence = field(init=False)
    _signature: tuple = field(init=False, repr=False)
    _command_owners: tuple = field(init=False, repr=False)

    def __post_init__(self):
        if hasattr(self, "_signature"):
            raise RuntimeError("render bundle baseline cannot be rebuilt")
        if type(self.binding) is not _WarpRenderBinding:
            raise TypeError("render bundle requires its exact validated binding")
        self.binding.validate_bound()
        object.__setattr__(self, "evidence", self.binding.evidence(self.descriptors))
        object.__setattr__(self, "_command_owners", self.commands)
        object.__setattr__(self, "_signature", self._facts())

    def _facts(self):
        if (
            type(self.binding) is not _WarpRenderBinding
            or type(self.evidence) is not WarpDirectLightRenderCommandEvidence
        ):
            raise TypeError("render bundle binding/evidence type changed")
        expected = self.binding.command_descriptors()
        if (
            type(self.descriptors) is not tuple
            or type(self.commands) is not tuple
            or len(self.commands) != len(expected)
        ):
            raise ValueError("recorded render command count/order changed")
        if len(self.descriptors) != len(expected):
            raise ValueError("render descriptors do not match fixed plan")
        for observed, actual in zip(self.descriptors, expected, strict=True):
            if (
                type(observed) is not _WarpRenderDescriptor
                or type(observed.arguments) is not tuple
                or type(observed.argument_names) is not tuple
                or observed.payload() != actual.payload()
            ):
                raise ValueError("render descriptor differs from exact fixed ABI")
        if self.commands is not self._command_owners:
            raise ValueError("recorded command retention tuple changed")
        return (
            id(self.binding),
            id(self.evidence),
            id(self.descriptors),
            tuple(id(d) for d in self.descriptors),
            id(self.commands),
            tuple(
                _render_launch_snapshot(c, d, self.binding.factory.device)
                for c, d in zip(self.commands, expected, strict=True)
            ),
            self.binding.evidence(expected).sha256,
        )

    def validate_bound(self):
        self.binding.validate_bound()
        if self._facts() != self._signature or self.evidence.sha256 != self._signature[-1]:
            raise ValueError("recorded render bundle or evidence changed")
        if WarpDirectLightRenderCommandEvidence.from_payload(self.evidence.to_payload()) != self.evidence:
            raise ValueError("render evidence is not canonical")
