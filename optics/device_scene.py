"""Device-resident optical scene cache primitives."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from physics.publish import GpuPublishedFrame

from .registry import OpticalWorldRegistry
from .scene_contract import PackedOpticalSceneHost

try:  # pragma: no cover - exercised in GPU environments.
    import warp as wp

    _HAS_WARP = True
except Exception:  # pragma: no cover - keeps CPU-only imports working.
    wp = None
    _HAS_WARP = False

_BUILD_EPS_F32 = 1.0e-8
_MAX_INT64_ROLE_BITS = 63
WARP_SCENE_UPDATE_OPERATION_ID = "warp_local_scene_transform_and_aabb_out_v1"
WARP_SCENE_COMMAND_SCHEMA_VERSION = "q54.p12.4.warp-scene-command.v1"
_DEVICE_SCENE_ARRAY_NAMES = (
    "material_albedo_rgb",
    "light_kind",
    "light_position_or_direction_world",
    "light_intensity",
    "light_color_rgb",
    "triangle_vertices_local",
    "triangle_instance_index",
    "triangle_primitive_global_id",
    "triangle_primitive_index_within_instance",
    "triangle_geometry_index",
    "triangle_geometry_primitive_index",
    "triangle_source_order_key",
    "triangle_role_mask",
    "triangle_numeric_instance_id",
    "triangle_material_index",
    "plane_normal_local",
    "plane_point_local",
    "plane_instance_index",
    "plane_source_order_key",
    "plane_role_mask",
    "plane_numeric_instance_id",
    "plane_material_index",
    "instance_body_index",
    "instance_X_body_geometry_R",
    "instance_X_body_geometry_r",
)


@dataclass(frozen=True)
class DeviceOpticalRoleTable:
    """Deterministic role-to-bitmask mapping for device visibility checks."""

    role_to_mask: MappingProxyType

    @classmethod
    def from_roles(cls, roles: object) -> "DeviceOpticalRoleTable":
        unique_roles = tuple(sorted({str(role) for role in roles}))
        if len(unique_roles) > _MAX_INT64_ROLE_BITS:
            raise ValueError("L5C role bitmask supports at most 63 roles")
        return cls(MappingProxyType({role: 1 << index for index, role in enumerate(unique_roles)}))

    @classmethod
    def from_registry(cls, registry: OpticalWorldRegistry) -> "DeviceOpticalRoleTable":
        roles: set[str] = set()
        for instance in registry.instances:
            roles.update(instance.roles)
        return cls.from_roles(roles)

    @classmethod
    def from_mapping(cls, role_to_mask: object) -> "DeviceOpticalRoleTable":
        mapping = dict(role_to_mask)
        if any(not isinstance(role, str) or not role for role in mapping):
            raise ValueError("role names must be non-empty strings")
        expected = {role: 1 << index for index, role in enumerate(sorted(mapping))}
        if mapping != expected:
            raise ValueError("role-to-mask mapping is not canonical")
        return cls(MappingProxyType(mapping))

    def mask_for(self, role: str) -> int:
        """Return zero for unknown roles so device queries naturally miss."""
        return int(self.role_to_mask.get(role, 0))

    def mask_for_roles(self, roles: object) -> int:
        mask = 0
        for role in roles:
            mask |= self.mask_for(str(role))
        return mask


@dataclass(frozen=True)
class DeviceOpticalScene:
    """Long-lived device buffers built from an `OpticalWorldRegistry`.

    `instance_X_body_geometry_*` follows the host convention:
    `X_body_geometry` is the pose of the geometry frame expressed in the body
    frame. It maps geometry-local points into the body frame. World-static
    instances use `instance_body_index == -1`, in which case
    `X_body_geometry` is interpreted directly as world-from-geometry.
    """

    device: object
    role_table: DeviceOpticalRoleTable
    num_instances: int
    num_materials: int
    num_lights: int
    num_triangles: int
    num_planes: int
    max_body_index: int
    material_albedo_rgb: object
    light_kind: object
    light_position_or_direction_world: object
    light_intensity: object
    light_color_rgb: object
    triangle_vertices_local: object
    triangle_instance_index: object
    triangle_primitive_global_id: object
    triangle_primitive_index_within_instance: object
    triangle_geometry_index: object
    triangle_geometry_primitive_index: object
    triangle_source_order_key: object
    triangle_role_mask: object
    triangle_numeric_instance_id: object
    triangle_material_index: object
    plane_normal_local: object
    plane_point_local: object
    plane_instance_index: object
    plane_source_order_key: object
    plane_role_mask: object
    plane_numeric_instance_id: object
    plane_material_index: object
    instance_body_index: object
    instance_X_body_geometry_R: object
    instance_X_body_geometry_r: object
    scene_contract_sha256: str | None = None
    resources: tuple[object, ...] = ()


@dataclass(frozen=True)
class DeviceOpticalSceneSnapshot:
    """Per-frame device optical scene buffers.

    L5C.0 intentionally allocates fresh world primitive buffers per snapshot.
    This binds buffer lifetime to the snapshot object and avoids introducing a
    device buffer ring before the event dependency model has been validated.
    """

    frame_id: int
    sim_time: float
    env_idx: int
    scene: DeviceOpticalScene
    triangle_v0_world: object
    triangle_e1_world: object
    triangle_e2_world: object
    triangle_normal_world: object
    plane_normal_world: object
    plane_point_world: object
    triangle_aabb_min: object | None = None
    triangle_aabb_max: object | None = None
    ready_event: object | None = None
    resources: tuple[object, ...] = ()


@dataclass(frozen=True)
class WarpSceneTransformInputs:
    """One already-created Warp transform view bound outside native timing."""

    x_world_R: object
    x_world_r: object
    env_idx: int
    input_contract: str
    transform_replay_sha256: str | None
    transform_frame_index: int | None
    resources: tuple[object, ...] = ()


@dataclass(frozen=True)
class WarpWorldSceneBuffers:
    """Caller-owned mutable world geometry and AABB outputs.

    The allocation helper below exists for compatibility callers and C3
    one-shot tests. C4 must supply the final arena-owned instances.
    """

    device: object
    triangle_v0_world: object
    triangle_e1_world: object
    triangle_e2_world: object
    triangle_normal_world: object
    plane_normal_world: object
    plane_point_world: object
    triangle_aabb_min: object
    triangle_aabb_max: object
    resources: tuple[object, ...] = ()

    def as_snapshot(
        self,
        scene: DeviceOpticalScene,
        *,
        frame_id: int,
        sim_time: float,
        env_idx: int,
    ) -> DeviceOpticalSceneSnapshot:
        """Create an untimed compatibility view without recording an event."""

        if scene.device != self.device:
            raise ValueError("Warp world-scene buffers and local scene devices must match")
        return DeviceOpticalSceneSnapshot(
            frame_id=int(frame_id),
            sim_time=float(sim_time),
            env_idx=int(env_idx),
            scene=scene,
            triangle_v0_world=self.triangle_v0_world,
            triangle_e1_world=self.triangle_e1_world,
            triangle_e2_world=self.triangle_e2_world,
            triangle_normal_world=self.triangle_normal_world,
            triangle_aabb_min=self.triangle_aabb_min,
            triangle_aabb_max=self.triangle_aabb_max,
            plane_normal_world=self.plane_normal_world,
            plane_point_world=self.plane_point_world,
            ready_event=None,
            resources=(self,) + tuple(self.resources),
        )


@dataclass(frozen=True)
class WarpSceneUpdateCommandEvidence:
    """Independent facts captured while a Warp scene bundle is sealed."""

    schema_version: str
    operation_id: str
    device_ordinal: int
    stream_handle: int
    native_input_manifest_sha256: str
    scene_contract_sha256: str
    input_contract: str
    transform_replay_sha256: str | None
    transform_frame_index: int | None
    input_storage_sha256: str
    output_storage_sha256: str
    dispatch_count: int
    dispatch_plan_sha256: str

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.__dict__)


class WarpSceneUpdateCommandBundle:
    """Immutable pre-recorded Warp scene commands for one storage graph."""

    __slots__ = (
        "_commands",
        "_poisoned",
        "_sealed",
        "evidence",
        "local_scene",
        "outputs",
        "stream",
        "transforms",
    )

    def __init__(
        self,
        *,
        commands: tuple[object, ...],
        evidence: WarpSceneUpdateCommandEvidence,
        local_scene: DeviceOpticalScene,
        transforms: WarpSceneTransformInputs,
        outputs: WarpWorldSceneBuffers,
        stream: object,
    ) -> None:
        self._commands = commands
        self._poisoned = False
        self.evidence = evidence
        self.local_scene = local_scene
        self.transforms = transforms
        self.outputs = outputs
        self.stream = stream
        self._sealed = True

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("Warp scene-update command bundle is immutable")
        object.__setattr__(self, name, value)

    @property
    def dispatch_count(self) -> int:
        return len(self._commands)

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def enqueue(self) -> None:
        """Launch only the frozen commands on the frozen stream."""

        if self._poisoned:
            raise RuntimeError("Warp scene-update command bundle is poisoned")
        try:
            for command in self._commands:
                command.launch(stream=self.stream)
        except Exception:
            object.__setattr__(self, "_poisoned", True)
            raise


class WarpSceneUpdateCommandFactory:
    """Seal caller-owned world outputs into allocation-free Warp commands."""

    __slots__ = (
        "input_contract",
        "local_scene",
        "native_input_manifest_sha256",
        "stream",
    )

    def __init__(
        self,
        local_scene: DeviceOpticalScene,
        *,
        stream: object,
        native_input_manifest_sha256: str,
        input_contract: str,
    ) -> None:
        _require_warp()
        if not local_scene.scene_contract_sha256:
            raise ValueError("Warp scene command requires scene-contract provenance")
        if not _sha256_text(native_input_manifest_sha256) or not input_contract:
            raise ValueError("Warp scene command provenance must be non-empty")
        _validate_warp_stream(stream, device=local_scene.device)
        self.local_scene = local_scene
        self.stream = stream
        self.native_input_manifest_sha256 = native_input_manifest_sha256
        self.input_contract = input_contract

    def seal(
        self,
        *,
        transforms: WarpSceneTransformInputs,
        outputs: WarpWorldSceneBuffers,
    ) -> WarpSceneUpdateCommandBundle:
        """Validate and record an immutable command bundle outside timing."""

        scene = self.local_scene
        if transforms.input_contract != self.input_contract:
            raise ValueError("Warp transform input contract does not match the factory")
        _validate_warp_transform_inputs(scene, transforms)
        _validate_warp_world_scene_buffers(scene, outputs)
        input_arrays = _warp_scene_command_inputs(scene, transforms)
        output_arrays = _warp_world_scene_arrays(outputs)
        protected_arrays = _warp_local_scene_arrays(scene) + (
            ("x_world_R", transforms.x_world_R),
            ("x_world_r", transforms.x_world_r),
        )
        _validate_disjoint_warp_storage(protected_arrays, output_arrays)

        commands: list[object] = []
        dispatches: list[dict[str, object]] = []
        if scene.num_triangles > 0:
            commands.append(
                wp.launch(
                    _update_triangles_derived_aabb_world_kernel,
                    dim=scene.num_triangles,
                    inputs=(
                        scene.triangle_vertices_local,
                        scene.triangle_instance_index,
                        scene.instance_body_index,
                        scene.instance_X_body_geometry_R,
                        scene.instance_X_body_geometry_r,
                        transforms.x_world_R,
                        transforms.x_world_r,
                        int(transforms.env_idx),
                        outputs.triangle_v0_world,
                        outputs.triangle_e1_world,
                        outputs.triangle_e2_world,
                        outputs.triangle_normal_world,
                        outputs.triangle_aabb_min,
                        outputs.triangle_aabb_max,
                    ),
                    device=scene.device,
                    stream=self.stream,
                    record_tape=False,
                    record_cmd=True,
                )
            )
            dispatches.append(
                {"kernel": "update_triangles_derived_aabb_world", "dim": scene.num_triangles}
            )
        if scene.num_planes > 0:
            commands.append(
                wp.launch(
                    _update_planes_world_kernel,
                    dim=scene.num_planes,
                    inputs=(
                        scene.plane_normal_local,
                        scene.plane_point_local,
                        scene.plane_instance_index,
                        scene.instance_body_index,
                        scene.instance_X_body_geometry_R,
                        scene.instance_X_body_geometry_r,
                        transforms.x_world_R,
                        transforms.x_world_r,
                        int(transforms.env_idx),
                        outputs.plane_normal_world,
                        outputs.plane_point_world,
                    ),
                    device=scene.device,
                    stream=self.stream,
                    record_tape=False,
                    record_cmd=True,
                )
            )
            dispatches.append({"kernel": "update_planes_world", "dim": scene.num_planes})

        evidence = WarpSceneUpdateCommandEvidence(
            schema_version=WARP_SCENE_COMMAND_SCHEMA_VERSION,
            operation_id=WARP_SCENE_UPDATE_OPERATION_ID,
            device_ordinal=int(scene.device.ordinal),
            stream_handle=int(self.stream.cuda_stream),
            native_input_manifest_sha256=self.native_input_manifest_sha256,
            scene_contract_sha256=scene.scene_contract_sha256,
            input_contract=self.input_contract,
            transform_replay_sha256=transforms.transform_replay_sha256,
            transform_frame_index=transforms.transform_frame_index,
            input_storage_sha256=_warp_storage_manifest_sha256(input_arrays),
            output_storage_sha256=_warp_storage_manifest_sha256(output_arrays),
            dispatch_count=len(commands),
            dispatch_plan_sha256=_canonical_sha256(dispatches),
        )
        return WarpSceneUpdateCommandBundle(
            commands=tuple(commands),
            evidence=evidence,
            local_scene=scene,
            transforms=transforms,
            outputs=outputs,
            stream=self.stream,
        )


def allocate_warp_world_scene_buffers(
    scene: DeviceOpticalScene,
    *,
    stream: object | None = None,
) -> WarpWorldSceneBuffers:
    """Allocate one caller-owned C3 test/compatibility output bundle.

    This helper is intentionally not an arena and must run before native
    timing. The sealed command factory itself never allocates outputs.
    """

    _require_warp()
    if stream is not None:
        _validate_warp_stream(stream, device=scene.device)
    with _scoped_stream(stream):
        arrays = {
            "triangle_v0_world": wp.empty(
                (scene.num_triangles, 3), dtype=wp.float32, device=scene.device
            ),
            "triangle_e1_world": wp.empty(
                (scene.num_triangles, 3), dtype=wp.float32, device=scene.device
            ),
            "triangle_e2_world": wp.empty(
                (scene.num_triangles, 3), dtype=wp.float32, device=scene.device
            ),
            "triangle_normal_world": wp.empty(
                (scene.num_triangles, 3), dtype=wp.float32, device=scene.device
            ),
            "plane_normal_world": wp.empty(
                (scene.num_planes, 3), dtype=wp.float32, device=scene.device
            ),
            "plane_point_world": wp.empty(
                (scene.num_planes, 3), dtype=wp.float32, device=scene.device
            ),
            "triangle_aabb_min": wp.empty(
                (scene.num_triangles, 3), dtype=wp.float32, device=scene.device
            ),
            "triangle_aabb_max": wp.empty(
                (scene.num_triangles, 3), dtype=wp.float32, device=scene.device
            ),
        }
    return WarpWorldSceneBuffers(
        device=scene.device,
        resources=(scene,),
        **arrays,
    )


def warp_replay_transform_inputs(
    x_world_R: object,
    x_world_r: object,
    *,
    frame_index: int,
    input_contract: str,
    transform_replay_sha256: str,
    resources: tuple[object, ...] = (),
) -> WarpSceneTransformInputs:
    """Create leading-env Warp views for one already-uploaded replay frame."""

    _require_warp()
    if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
        raise ValueError("transform replay frame index must be a non-negative integer")
    if not input_contract or not _sha256_text(transform_replay_sha256):
        raise ValueError("transform replay provenance must be non-empty")
    _validate_warp_array(x_world_R, name="replay_x_world_R", dtype=wp.float32, ndim=4)
    _validate_warp_array(x_world_r, name="replay_x_world_r", dtype=wp.float32, ndim=3)
    if tuple(x_world_R.shape[:2]) != tuple(x_world_r.shape[:2]):
        raise ValueError("Warp replay transform shapes do not agree")
    if tuple(x_world_R.shape[2:]) != (3, 3) or tuple(x_world_r.shape[2:]) != (3,):
        raise ValueError("Warp replay transform trailing shapes are invalid")
    if frame_index >= int(x_world_R.shape[0]):
        raise IndexError("transform replay frame index is out of range")
    if int(x_world_R.shape[1]) == 0:
        # Warp 1.12 rejects slicing an array that has a later zero-length
        # dimension. Static zero-body replays never dereference transforms, so
        # retain the exact base allocation and select the frame as the env.
        rotation_view = x_world_R
        translation_view = x_world_r
        env_idx = frame_index
    else:
        rotation_view = x_world_R[frame_index : frame_index + 1]
        translation_view = x_world_r[frame_index : frame_index + 1]
        env_idx = 0
    return WarpSceneTransformInputs(
        x_world_R=rotation_view,
        x_world_r=translation_view,
        env_idx=env_idx,
        input_contract=input_contract,
        transform_replay_sha256=transform_replay_sha256,
        transform_frame_index=frame_index,
        resources=tuple(resources) + (x_world_R, x_world_r),
    )


def _validate_warp_transform_inputs(
    scene: DeviceOpticalScene,
    transforms: WarpSceneTransformInputs,
) -> None:
    if not isinstance(transforms, WarpSceneTransformInputs):
        raise TypeError("transforms must be WarpSceneTransformInputs")
    _validate_warp_array(
        transforms.x_world_R,
        name="x_world_R",
        dtype=wp.float32,
        ndim=4,
        device=scene.device,
    )
    _validate_warp_array(
        transforms.x_world_r,
        name="x_world_r",
        dtype=wp.float32,
        ndim=3,
        device=scene.device,
    )
    rotation_shape = tuple(transforms.x_world_R.shape)
    translation_shape = tuple(transforms.x_world_r.shape)
    if rotation_shape[2:] != (3, 3) or translation_shape[2:] != (3,):
        raise ValueError("Warp transform trailing shapes are invalid")
    if rotation_shape[:2] != translation_shape[:2]:
        raise ValueError("Warp transform env/body dimensions do not agree")
    if (
        not isinstance(transforms.env_idx, int)
        or isinstance(transforms.env_idx, bool)
        or transforms.env_idx < 0
        or transforms.env_idx >= rotation_shape[0]
    ):
        raise ValueError("Warp transform env index is out of range")
    if scene.max_body_index >= rotation_shape[1]:
        raise ValueError("Warp transforms do not cover the scene body indices")
    has_replay_hash = transforms.transform_replay_sha256 is not None
    has_replay_index = transforms.transform_frame_index is not None
    if has_replay_hash != has_replay_index:
        raise ValueError("Warp replay hash and frame index must appear together")
    if has_replay_hash:
        if not _sha256_text(transforms.transform_replay_sha256):
            raise ValueError("Warp transform replay hash is invalid")
        if (
            not isinstance(transforms.transform_frame_index, int)
            or isinstance(transforms.transform_frame_index, bool)
            or transforms.transform_frame_index < 0
        ):
            raise ValueError("Warp transform replay frame index is invalid")


def _validate_warp_world_scene_buffers(
    scene: DeviceOpticalScene,
    outputs: WarpWorldSceneBuffers,
) -> None:
    if not isinstance(outputs, WarpWorldSceneBuffers):
        raise TypeError("outputs must be WarpWorldSceneBuffers")
    if wp.get_device(outputs.device) != scene.device:
        raise ValueError("Warp world-scene output device does not match the local scene")
    expected = {
        "triangle_v0_world": scene.num_triangles,
        "triangle_e1_world": scene.num_triangles,
        "triangle_e2_world": scene.num_triangles,
        "triangle_normal_world": scene.num_triangles,
        "triangle_aabb_min": scene.num_triangles,
        "triangle_aabb_max": scene.num_triangles,
        "plane_normal_world": scene.num_planes,
        "plane_point_world": scene.num_planes,
    }
    for name, count in expected.items():
        _validate_warp_array(
            getattr(outputs, name),
            name=name,
            dtype=wp.float32,
            ndim=2,
            shape=(count, 3),
            device=scene.device,
        )


def _validate_warp_stream(stream: object, *, device: object) -> None:
    if getattr(stream, "device", None) != wp.get_device(device):
        raise ValueError("Warp prepared stream device does not match")
    handle = getattr(stream, "cuda_stream", None)
    if not isinstance(handle, int) or isinstance(handle, bool) or handle <= 0:
        raise ValueError("Warp prepared stream must expose a nonzero CUDA handle")


def _validate_warp_array(
    value: object,
    *,
    name: str,
    dtype: object,
    ndim: int,
    shape: tuple[int, ...] | None = None,
    device: object | None = None,
) -> None:
    if not isinstance(value, wp.array):
        raise TypeError(f"Warp array {name!r} has the wrong type")
    if value.dtype != dtype or int(value.ndim) != ndim:
        raise ValueError(f"Warp array {name!r} dtype/rank does not match")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"Warp array {name!r} shape does not match")
    if not value.is_contiguous:
        raise ValueError(f"Warp array {name!r} must be contiguous")
    if device is not None and wp.get_device(value.device) != wp.get_device(device):
        raise ValueError(f"Warp array {name!r} is on the wrong device")
    if int(value.capacity) > 0 and int(value.ptr or 0) <= 0:
        raise ValueError(f"Warp array {name!r} has a null pointer")


def _warp_scene_command_inputs(
    scene: DeviceOpticalScene,
    transforms: WarpSceneTransformInputs,
) -> tuple[tuple[str, object], ...]:
    return (
        ("triangle_vertices_local", scene.triangle_vertices_local),
        ("triangle_instance_index", scene.triangle_instance_index),
        ("plane_normal_local", scene.plane_normal_local),
        ("plane_point_local", scene.plane_point_local),
        ("plane_instance_index", scene.plane_instance_index),
        ("instance_body_index", scene.instance_body_index),
        ("instance_X_body_geometry_R", scene.instance_X_body_geometry_R),
        ("instance_X_body_geometry_r", scene.instance_X_body_geometry_r),
        ("x_world_R", transforms.x_world_R),
        ("x_world_r", transforms.x_world_r),
    )


def _warp_local_scene_arrays(
    scene: DeviceOpticalScene,
) -> tuple[tuple[str, object], ...]:
    return tuple((name, getattr(scene, name)) for name in _DEVICE_SCENE_ARRAY_NAMES)


def _warp_world_scene_arrays(
    outputs: WarpWorldSceneBuffers,
) -> tuple[tuple[str, object], ...]:
    return tuple(
        (name, getattr(outputs, name))
        for name in (
            "triangle_v0_world",
            "triangle_e1_world",
            "triangle_e2_world",
            "triangle_normal_world",
            "plane_normal_world",
            "plane_point_world",
            "triangle_aabb_min",
            "triangle_aabb_max",
        )
    )


def _validate_disjoint_warp_storage(
    inputs: tuple[tuple[str, object], ...],
    outputs: tuple[tuple[str, object], ...],
) -> None:
    seen_outputs: list[tuple[str, int, int]] = []
    input_ranges = tuple(
        (name, int(value.ptr or 0), int(value.capacity))
        for name, value in inputs
        if int(value.capacity) > 0
    )
    for name, value in outputs:
        start = int(value.ptr or 0)
        size = int(value.capacity)
        if size <= 0:
            continue
        end = start + size
        for other_name, other_start, other_size in input_ranges:
            if start < other_start + other_size and other_start < end:
                raise ValueError(f"Warp output {name!r} overlaps input {other_name!r}")
        for other_name, other_start, other_end in seen_outputs:
            if start < other_end and other_start < end:
                raise ValueError(f"Warp outputs {name!r} and {other_name!r} overlap")
        seen_outputs.append((name, start, end))


def _warp_storage_manifest_sha256(values: tuple[tuple[str, object], ...]) -> str:
    return _canonical_sha256(
        [
            {
                "name": name,
                "device_ordinal": int(value.device.ordinal),
                "ptr": int(value.ptr or 0),
                "capacity": int(value.capacity),
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "strides": list(value.strides),
            }
            for name, value in values
        ]
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class DeviceOpticalSceneCache:
    """Device scene cache that realizes registry primitives for a GPU frame."""

    def __init__(self, registry: OpticalWorldRegistry, *, device=None, stream=None) -> None:
        self.scene = build_device_optical_scene(registry, device=device, stream=stream)

    def snapshot_from_gpu_frame(
        self,
        frame: GpuPublishedFrame,
        *,
        env_idx: int = 0,
        stream=None,
        include_aabb: bool = False,
    ) -> DeviceOpticalSceneSnapshot:
        return update_device_optical_scene_from_gpu_frame(
            self.scene,
            frame,
            env_idx=env_idx,
            stream=stream,
            include_aabb=include_aabb,
        )


def build_device_optical_scene(
    registry: OpticalWorldRegistry,
    *,
    device=None,
    stream=None,
) -> DeviceOpticalScene:
    """Upload registry geometry and metadata into long-lived device buffers."""
    packed_scene = PackedOpticalSceneHost.from_registry(registry)
    return build_device_optical_scene_from_packed(
        packed_scene,
        device=device,
        stream=stream,
    )


def build_device_optical_scene_from_packed(
    packed_scene: PackedOpticalSceneHost,
    *,
    device=None,
    stream=None,
) -> DeviceOpticalScene:
    """Upload one canonical host contract into Warp-owned CUDA arrays."""
    _require_warp()
    wp.init()
    resolved_device = wp.get_device("cuda:0" if device is None else device)
    arrays: dict[str, object] = {}
    with _scoped_stream(stream):
        for name, value in packed_scene.arrays.items():
            arrays[name] = _wp_array(
                value,
                dtype=_warp_dtype_for_numpy(value.dtype),
                device=resolved_device,
            )
        return device_optical_scene_from_arrays(
            packed_scene,
            arrays,
            device=resolved_device,
            resources=(),
        )


def device_optical_scene_from_arrays(
    packed_scene: PackedOpticalSceneHost,
    arrays: dict[str, object],
    *,
    device: object,
    resources: tuple[object, ...],
) -> DeviceOpticalScene:
    """Wrap exact caller-owned Warp arrays without allocating or copying."""
    _require_warp()
    resolved_device = wp.get_device(device)
    if set(arrays) != set(packed_scene.arrays):
        raise ValueError("Warp scene arrays do not match the packed scene contract")
    for name, host in packed_scene.arrays.items():
        value = arrays[name]
        if wp.get_device(value.device).ordinal != resolved_device.ordinal:
            raise ValueError(f"Warp scene array {name!r} is on the wrong device")
        if tuple(value.shape) != tuple(host.shape):
            raise ValueError(f"Warp scene array {name!r} shape does not match")
        if value.dtype != _warp_dtype_for_numpy(host.dtype):
            raise ValueError(f"Warp scene array {name!r} dtype does not match")
    return DeviceOpticalScene(
        device=resolved_device,
        role_table=DeviceOpticalRoleTable.from_mapping(packed_scene.role_to_mask),
        num_instances=packed_scene.num_instances,
        num_materials=packed_scene.num_materials,
        num_lights=packed_scene.num_lights,
        num_triangles=packed_scene.num_triangles,
        num_planes=packed_scene.num_planes,
        max_body_index=packed_scene.max_body_index,
        **{name: arrays[name] for name in packed_scene.arrays},
        scene_contract_sha256=packed_scene.scene_contract_sha256,
        resources=tuple(resources) + (packed_scene,),
    )


def update_device_optical_scene_from_gpu_frame(
    scene: DeviceOpticalScene,
    frame: GpuPublishedFrame,
    *,
    env_idx: int = 0,
    stream=None,
    include_aabb: bool = False,
) -> DeviceOpticalSceneSnapshot:
    """Realize per-frame world primitive buffers from a borrowed GPU frame."""
    _require_warp()
    if env_idx < 0:
        raise ValueError("env_idx must be >= 0")
    _validate_gpu_frame_transform_shape(scene, frame, env_idx=env_idx)
    _wait_on_event(frame.ready_event, stream=stream, device=scene.device)

    with _scoped_stream(stream):
        triangle_v0_world = wp.zeros((scene.num_triangles, 3), dtype=wp.float32, device=scene.device)
        triangle_e1_world = wp.zeros((scene.num_triangles, 3), dtype=wp.float32, device=scene.device)
        triangle_e2_world = wp.zeros((scene.num_triangles, 3), dtype=wp.float32, device=scene.device)
        triangle_normal_world = wp.zeros((scene.num_triangles, 3), dtype=wp.float32, device=scene.device)
        triangle_aabb_min = None
        triangle_aabb_max = None
        if include_aabb:
            triangle_aabb_min = wp.zeros((scene.num_triangles, 3), dtype=wp.float32, device=scene.device)
            triangle_aabb_max = wp.zeros((scene.num_triangles, 3), dtype=wp.float32, device=scene.device)
        plane_normal_world = wp.zeros((scene.num_planes, 3), dtype=wp.float32, device=scene.device)
        plane_point_world = wp.zeros((scene.num_planes, 3), dtype=wp.float32, device=scene.device)

        if scene.num_triangles > 0:
            wp.launch(
                _update_triangles_derived_aabb_world_kernel
                if include_aabb
                else _update_triangles_derived_world_kernel,
                dim=scene.num_triangles,
                inputs=[
                    scene.triangle_vertices_local,
                    scene.triangle_instance_index,
                    scene.instance_body_index,
                    scene.instance_X_body_geometry_R,
                    scene.instance_X_body_geometry_r,
                    frame.x_world_R_wp,
                    frame.x_world_r_wp,
                    int(env_idx),
                    triangle_v0_world,
                    triangle_e1_world,
                    triangle_e2_world,
                    triangle_normal_world,
                ]
                + ([triangle_aabb_min, triangle_aabb_max] if include_aabb else []),
                device=scene.device,
                stream=stream,
            )
        if scene.num_planes > 0:
            wp.launch(
                _update_planes_world_kernel,
                dim=scene.num_planes,
                inputs=[
                    scene.plane_normal_local,
                    scene.plane_point_local,
                    scene.plane_instance_index,
                    scene.instance_body_index,
                    scene.instance_X_body_geometry_R,
                    scene.instance_X_body_geometry_r,
                    frame.x_world_R_wp,
                    frame.x_world_r_wp,
                    int(env_idx),
                    plane_normal_world,
                    plane_point_world,
                ],
                device=scene.device,
                stream=stream,
            )

        ready_event = (stream or wp.get_stream(scene.device)).record_event()

    return DeviceOpticalSceneSnapshot(
        frame_id=frame.frame_id,
        sim_time=frame.sim_time,
        env_idx=env_idx,
        scene=scene,
        triangle_v0_world=triangle_v0_world,
        triangle_e1_world=triangle_e1_world,
        triangle_e2_world=triangle_e2_world,
        triangle_normal_world=triangle_normal_world,
        triangle_aabb_min=triangle_aabb_min,
        triangle_aabb_max=triangle_aabb_max,
        plane_normal_world=plane_normal_world,
        plane_point_world=plane_point_world,
        ready_event=ready_event,
        resources=(frame,),
    )


def _validate_gpu_frame_transform_shape(
    scene: DeviceOpticalScene,
    frame: GpuPublishedFrame,
    *,
    env_idx: int,
) -> None:
    rotation_shape = tuple(frame.x_world_R_wp.shape)
    translation_shape = tuple(frame.x_world_r_wp.shape)
    if len(rotation_shape) != 4 or rotation_shape[2:] != (3, 3):
        raise ValueError("GpuPublishedFrame.x_world_R_wp must have shape (num_envs, num_bodies, 3, 3)")
    if len(translation_shape) != 3 or translation_shape[2] != 3:
        raise ValueError("GpuPublishedFrame.x_world_r_wp must have shape (num_envs, num_bodies, 3)")
    if rotation_shape[:2] != translation_shape[:2]:
        raise ValueError("GpuPublishedFrame transform arrays must agree on env/body dimensions")
    if env_idx >= rotation_shape[0]:
        raise IndexError(f"env_idx {env_idx} is out of range for GpuPublishedFrame transforms")
    if scene.max_body_index >= rotation_shape[1]:
        raise IndexError(
            f"body_index {scene.max_body_index} is out of range for GpuPublishedFrame transforms"
        )


def _wp_array(values: np.ndarray, *, dtype, device):
    return wp.array(values, dtype=dtype, device=device)


def _warp_dtype_for_numpy(dtype):
    mapping = {
        np.dtype(np.bool_): wp.bool,
        np.dtype(np.int8): wp.int8,
        np.dtype(np.int16): wp.int16,
        np.dtype(np.int32): wp.int32,
        np.dtype(np.int64): wp.int64,
        np.dtype(np.uint8): wp.uint8,
        np.dtype(np.uint16): wp.uint16,
        np.dtype(np.uint32): wp.uint32,
        np.dtype(np.uint64): wp.uint64,
        np.dtype(np.float16): wp.float16,
        np.dtype(np.float32): wp.float32,
        np.dtype(np.float64): wp.float64,
    }
    try:
        return mapping[np.dtype(dtype)]
    except KeyError as exc:
        raise ValueError(f"unsupported Warp scene dtype {np.dtype(dtype)}") from exc


def _scoped_stream(stream):
    if stream is None:
        return nullcontext()
    return wp.ScopedStream(stream)


def _wait_on_event(event, *, stream, device) -> None:
    if event is None:
        return
    (stream or wp.get_stream(device)).wait_event(event)


def _require_warp() -> None:
    if not _HAS_WARP:
        raise ImportError("DeviceOpticalSceneCache requires the optional warp package")


if _HAS_WARP:

    @wp.func
    def _compose_R_component(
        Rwb00: float,
        Rwb01: float,
        Rwb02: float,
        Rwb10: float,
        Rwb11: float,
        Rwb12: float,
        Rwb20: float,
        Rwb21: float,
        Rwb22: float,
        Rbg: wp.array2d(dtype=wp.float32),
        instance_index: int,
        row: int,
        col: int,
    ):
        out = wp.float32(0.0)
        if row == 0:
            out = (
                Rwb00 * Rbg[instance_index, col]
                + Rwb01 * Rbg[instance_index, 3 + col]
                + Rwb02 * Rbg[instance_index, 6 + col]
            )
        elif row == 1:
            out = (
                Rwb10 * Rbg[instance_index, col]
                + Rwb11 * Rbg[instance_index, 3 + col]
                + Rwb12 * Rbg[instance_index, 6 + col]
            )
        else:
            out = (
                Rwb20 * Rbg[instance_index, col]
                + Rwb21 * Rbg[instance_index, 3 + col]
                + Rwb22 * Rbg[instance_index, 6 + col]
            )
        return out

    @wp.kernel
    def _update_triangles_derived_aabb_world_kernel(
        triangle_vertices_local: wp.array2d(dtype=wp.float32),
        triangle_instance_index: wp.array(dtype=wp.int32),
        instance_body_index: wp.array(dtype=wp.int32),
        instance_X_body_geometry_R: wp.array2d(dtype=wp.float32),
        instance_X_body_geometry_r: wp.array2d(dtype=wp.float32),
        frame_x_world_R: wp.array4d(dtype=wp.float32),
        frame_x_world_r: wp.array3d(dtype=wp.float32),
        env_idx: int,
        triangle_v0_world: wp.array2d(dtype=wp.float32),
        triangle_e1_world: wp.array2d(dtype=wp.float32),
        triangle_e2_world: wp.array2d(dtype=wp.float32),
        triangle_normal_world: wp.array2d(dtype=wp.float32),
        triangle_aabb_min: wp.array2d(dtype=wp.float32),
        triangle_aabb_max: wp.array2d(dtype=wp.float32),
    ):
        primitive = wp.tid()
        instance_index = triangle_instance_index[primitive]
        body_index = instance_body_index[instance_index]
        Rwb00 = wp.float32(1.0)
        Rwb01 = wp.float32(0.0)
        Rwb02 = wp.float32(0.0)
        Rwb10 = wp.float32(0.0)
        Rwb11 = wp.float32(1.0)
        Rwb12 = wp.float32(0.0)
        Rwb20 = wp.float32(0.0)
        Rwb21 = wp.float32(0.0)
        Rwb22 = wp.float32(1.0)
        rwbx = wp.float32(0.0)
        rwby = wp.float32(0.0)
        rwbz = wp.float32(0.0)
        if body_index >= 0:
            Rwb00 = frame_x_world_R[env_idx, body_index, 0, 0]
            Rwb01 = frame_x_world_R[env_idx, body_index, 0, 1]
            Rwb02 = frame_x_world_R[env_idx, body_index, 0, 2]
            Rwb10 = frame_x_world_R[env_idx, body_index, 1, 0]
            Rwb11 = frame_x_world_R[env_idx, body_index, 1, 1]
            Rwb12 = frame_x_world_R[env_idx, body_index, 1, 2]
            Rwb20 = frame_x_world_R[env_idx, body_index, 2, 0]
            Rwb21 = frame_x_world_R[env_idx, body_index, 2, 1]
            Rwb22 = frame_x_world_R[env_idx, body_index, 2, 2]
            rwbx = frame_x_world_r[env_idx, body_index, 0]
            rwby = frame_x_world_r[env_idx, body_index, 1]
            rwbz = frame_x_world_r[env_idx, body_index, 2]

        rbgx = instance_X_body_geometry_r[instance_index, 0]
        rbgy = instance_X_body_geometry_r[instance_index, 1]
        rbgz = instance_X_body_geometry_r[instance_index, 2]
        rwgx = rwbx + Rwb00 * rbgx + Rwb01 * rbgy + Rwb02 * rbgz
        rwgy = rwby + Rwb10 * rbgx + Rwb11 * rbgy + Rwb12 * rbgz
        rwgz = rwbz + Rwb20 * rbgx + Rwb21 * rbgy + Rwb22 * rbgz

        Rwg00 = _compose_R_component(
            Rwb00,
            Rwb01,
            Rwb02,
            Rwb10,
            Rwb11,
            Rwb12,
            Rwb20,
            Rwb21,
            Rwb22,
            instance_X_body_geometry_R,
            instance_index,
            0,
            0,
        )
        Rwg01 = _compose_R_component(
            Rwb00,
            Rwb01,
            Rwb02,
            Rwb10,
            Rwb11,
            Rwb12,
            Rwb20,
            Rwb21,
            Rwb22,
            instance_X_body_geometry_R,
            instance_index,
            0,
            1,
        )
        Rwg02 = _compose_R_component(
            Rwb00,
            Rwb01,
            Rwb02,
            Rwb10,
            Rwb11,
            Rwb12,
            Rwb20,
            Rwb21,
            Rwb22,
            instance_X_body_geometry_R,
            instance_index,
            0,
            2,
        )
        Rwg10 = _compose_R_component(
            Rwb00,
            Rwb01,
            Rwb02,
            Rwb10,
            Rwb11,
            Rwb12,
            Rwb20,
            Rwb21,
            Rwb22,
            instance_X_body_geometry_R,
            instance_index,
            1,
            0,
        )
        Rwg11 = _compose_R_component(
            Rwb00,
            Rwb01,
            Rwb02,
            Rwb10,
            Rwb11,
            Rwb12,
            Rwb20,
            Rwb21,
            Rwb22,
            instance_X_body_geometry_R,
            instance_index,
            1,
            1,
        )
        Rwg12 = _compose_R_component(
            Rwb00,
            Rwb01,
            Rwb02,
            Rwb10,
            Rwb11,
            Rwb12,
            Rwb20,
            Rwb21,
            Rwb22,
            instance_X_body_geometry_R,
            instance_index,
            1,
            2,
        )
        Rwg20 = _compose_R_component(
            Rwb00,
            Rwb01,
            Rwb02,
            Rwb10,
            Rwb11,
            Rwb12,
            Rwb20,
            Rwb21,
            Rwb22,
            instance_X_body_geometry_R,
            instance_index,
            2,
            0,
        )
        Rwg21 = _compose_R_component(
            Rwb00,
            Rwb01,
            Rwb02,
            Rwb10,
            Rwb11,
            Rwb12,
            Rwb20,
            Rwb21,
            Rwb22,
            instance_X_body_geometry_R,
            instance_index,
            2,
            1,
        )
        Rwg22 = _compose_R_component(
            Rwb00,
            Rwb01,
            Rwb02,
            Rwb10,
            Rwb11,
            Rwb12,
            Rwb20,
            Rwb21,
            Rwb22,
            instance_X_body_geometry_R,
            instance_index,
            2,
            2,
        )

        v0x = triangle_vertices_local[primitive, 0]
        v0y = triangle_vertices_local[primitive, 1]
        v0z = triangle_vertices_local[primitive, 2]
        v1x = triangle_vertices_local[primitive, 3]
        v1y = triangle_vertices_local[primitive, 4]
        v1z = triangle_vertices_local[primitive, 5]
        v2x = triangle_vertices_local[primitive, 6]
        v2y = triangle_vertices_local[primitive, 7]
        v2z = triangle_vertices_local[primitive, 8]

        e1x = v1x - v0x
        e1y = v1y - v0y
        e1z = v1z - v0z
        e2x = v2x - v0x
        e2y = v2y - v0y
        e2z = v2z - v0z

        v0wx = Rwg00 * v0x + Rwg01 * v0y + Rwg02 * v0z + rwgx
        v0wy = Rwg10 * v0x + Rwg11 * v0y + Rwg12 * v0z + rwgy
        v0wz = Rwg20 * v0x + Rwg21 * v0y + Rwg22 * v0z + rwgz

        e1wx = Rwg00 * e1x + Rwg01 * e1y + Rwg02 * e1z
        e1wy = Rwg10 * e1x + Rwg11 * e1y + Rwg12 * e1z
        e1wz = Rwg20 * e1x + Rwg21 * e1y + Rwg22 * e1z

        e2wx = Rwg00 * e2x + Rwg01 * e2y + Rwg02 * e2z
        e2wy = Rwg10 * e2x + Rwg11 * e2y + Rwg12 * e2z
        e2wz = Rwg20 * e2x + Rwg21 * e2y + Rwg22 * e2z

        triangle_v0_world[primitive, 0] = v0wx
        triangle_v0_world[primitive, 1] = v0wy
        triangle_v0_world[primitive, 2] = v0wz
        triangle_e1_world[primitive, 0] = e1wx
        triangle_e1_world[primitive, 1] = e1wy
        triangle_e1_world[primitive, 2] = e1wz
        triangle_e2_world[primitive, 0] = e2wx
        triangle_e2_world[primitive, 1] = e2wy
        triangle_e2_world[primitive, 2] = e2wz

        v1wx = v0wx + e1wx
        v1wy = v0wy + e1wy
        v1wz = v0wz + e1wz
        v2wx = v0wx + e2wx
        v2wy = v0wy + e2wy
        v2wz = v0wz + e2wz
        min_x = v0wx
        min_y = v0wy
        min_z = v0wz
        max_x = v0wx
        max_y = v0wy
        max_z = v0wz
        if v1wx < min_x:
            min_x = v1wx
        if v2wx < min_x:
            min_x = v2wx
        if v1wy < min_y:
            min_y = v1wy
        if v2wy < min_y:
            min_y = v2wy
        if v1wz < min_z:
            min_z = v1wz
        if v2wz < min_z:
            min_z = v2wz
        if v1wx > max_x:
            max_x = v1wx
        if v2wx > max_x:
            max_x = v2wx
        if v1wy > max_y:
            max_y = v1wy
        if v2wy > max_y:
            max_y = v2wy
        if v1wz > max_z:
            max_z = v1wz
        if v2wz > max_z:
            max_z = v2wz
        triangle_aabb_min[primitive, 0] = min_x
        triangle_aabb_min[primitive, 1] = min_y
        triangle_aabb_min[primitive, 2] = min_z
        triangle_aabb_max[primitive, 0] = max_x
        triangle_aabb_max[primitive, 1] = max_y
        triangle_aabb_max[primitive, 2] = max_z

        nx = e1y * e2z - e1z * e2y
        ny = e1z * e2x - e1x * e2z
        nz = e1x * e2y - e1y * e2x
        normal_norm = wp.sqrt(nx * nx + ny * ny + nz * nz)
        if normal_norm > _BUILD_EPS_F32:
            nx = nx / normal_norm
            ny = ny / normal_norm
            nz = nz / normal_norm
            triangle_normal_world[primitive, 0] = Rwg00 * nx + Rwg01 * ny + Rwg02 * nz
            triangle_normal_world[primitive, 1] = Rwg10 * nx + Rwg11 * ny + Rwg12 * nz
            triangle_normal_world[primitive, 2] = Rwg20 * nx + Rwg21 * ny + Rwg22 * nz

    @wp.kernel
    def _update_triangles_derived_world_kernel(
        triangle_vertices_local: wp.array2d(dtype=wp.float32),
        triangle_instance_index: wp.array(dtype=wp.int32),
        instance_body_index: wp.array(dtype=wp.int32),
        instance_X_body_geometry_R: wp.array2d(dtype=wp.float32),
        instance_X_body_geometry_r: wp.array2d(dtype=wp.float32),
        frame_x_world_R: wp.array4d(dtype=wp.float32),
        frame_x_world_r: wp.array3d(dtype=wp.float32),
        env_idx: int,
        triangle_v0_world: wp.array2d(dtype=wp.float32),
        triangle_e1_world: wp.array2d(dtype=wp.float32),
        triangle_e2_world: wp.array2d(dtype=wp.float32),
        triangle_normal_world: wp.array2d(dtype=wp.float32),
    ):
        primitive = wp.tid()
        instance_index = triangle_instance_index[primitive]
        body_index = instance_body_index[instance_index]
        Rwb00 = wp.float32(1.0)
        Rwb01 = wp.float32(0.0)
        Rwb02 = wp.float32(0.0)
        Rwb10 = wp.float32(0.0)
        Rwb11 = wp.float32(1.0)
        Rwb12 = wp.float32(0.0)
        Rwb20 = wp.float32(0.0)
        Rwb21 = wp.float32(0.0)
        Rwb22 = wp.float32(1.0)
        rwbx = wp.float32(0.0)
        rwby = wp.float32(0.0)
        rwbz = wp.float32(0.0)
        if body_index >= 0:
            Rwb00 = frame_x_world_R[env_idx, body_index, 0, 0]
            Rwb01 = frame_x_world_R[env_idx, body_index, 0, 1]
            Rwb02 = frame_x_world_R[env_idx, body_index, 0, 2]
            Rwb10 = frame_x_world_R[env_idx, body_index, 1, 0]
            Rwb11 = frame_x_world_R[env_idx, body_index, 1, 1]
            Rwb12 = frame_x_world_R[env_idx, body_index, 1, 2]
            Rwb20 = frame_x_world_R[env_idx, body_index, 2, 0]
            Rwb21 = frame_x_world_R[env_idx, body_index, 2, 1]
            Rwb22 = frame_x_world_R[env_idx, body_index, 2, 2]
            rwbx = frame_x_world_r[env_idx, body_index, 0]
            rwby = frame_x_world_r[env_idx, body_index, 1]
            rwbz = frame_x_world_r[env_idx, body_index, 2]

        rbgx = instance_X_body_geometry_r[instance_index, 0]
        rbgy = instance_X_body_geometry_r[instance_index, 1]
        rbgz = instance_X_body_geometry_r[instance_index, 2]
        rwgx = rwbx + Rwb00 * rbgx + Rwb01 * rbgy + Rwb02 * rbgz
        rwgy = rwby + Rwb10 * rbgx + Rwb11 * rbgy + Rwb12 * rbgz
        rwgz = rwbz + Rwb20 * rbgx + Rwb21 * rbgy + Rwb22 * rbgz

        Rwg00 = _compose_R_component(
            Rwb00,
            Rwb01,
            Rwb02,
            Rwb10,
            Rwb11,
            Rwb12,
            Rwb20,
            Rwb21,
            Rwb22,
            instance_X_body_geometry_R,
            instance_index,
            0,
            0,
        )
        Rwg01 = _compose_R_component(
            Rwb00,
            Rwb01,
            Rwb02,
            Rwb10,
            Rwb11,
            Rwb12,
            Rwb20,
            Rwb21,
            Rwb22,
            instance_X_body_geometry_R,
            instance_index,
            0,
            1,
        )
        Rwg02 = _compose_R_component(
            Rwb00,
            Rwb01,
            Rwb02,
            Rwb10,
            Rwb11,
            Rwb12,
            Rwb20,
            Rwb21,
            Rwb22,
            instance_X_body_geometry_R,
            instance_index,
            0,
            2,
        )
        Rwg10 = _compose_R_component(
            Rwb00,
            Rwb01,
            Rwb02,
            Rwb10,
            Rwb11,
            Rwb12,
            Rwb20,
            Rwb21,
            Rwb22,
            instance_X_body_geometry_R,
            instance_index,
            1,
            0,
        )
        Rwg11 = _compose_R_component(
            Rwb00,
            Rwb01,
            Rwb02,
            Rwb10,
            Rwb11,
            Rwb12,
            Rwb20,
            Rwb21,
            Rwb22,
            instance_X_body_geometry_R,
            instance_index,
            1,
            1,
        )
        Rwg12 = _compose_R_component(
            Rwb00,
            Rwb01,
            Rwb02,
            Rwb10,
            Rwb11,
            Rwb12,
            Rwb20,
            Rwb21,
            Rwb22,
            instance_X_body_geometry_R,
            instance_index,
            1,
            2,
        )
        Rwg20 = _compose_R_component(
            Rwb00,
            Rwb01,
            Rwb02,
            Rwb10,
            Rwb11,
            Rwb12,
            Rwb20,
            Rwb21,
            Rwb22,
            instance_X_body_geometry_R,
            instance_index,
            2,
            0,
        )
        Rwg21 = _compose_R_component(
            Rwb00,
            Rwb01,
            Rwb02,
            Rwb10,
            Rwb11,
            Rwb12,
            Rwb20,
            Rwb21,
            Rwb22,
            instance_X_body_geometry_R,
            instance_index,
            2,
            1,
        )
        Rwg22 = _compose_R_component(
            Rwb00,
            Rwb01,
            Rwb02,
            Rwb10,
            Rwb11,
            Rwb12,
            Rwb20,
            Rwb21,
            Rwb22,
            instance_X_body_geometry_R,
            instance_index,
            2,
            2,
        )

        v0x = triangle_vertices_local[primitive, 0]
        v0y = triangle_vertices_local[primitive, 1]
        v0z = triangle_vertices_local[primitive, 2]
        e1x = triangle_vertices_local[primitive, 3] - v0x
        e1y = triangle_vertices_local[primitive, 4] - v0y
        e1z = triangle_vertices_local[primitive, 5] - v0z
        e2x = triangle_vertices_local[primitive, 6] - v0x
        e2y = triangle_vertices_local[primitive, 7] - v0y
        e2z = triangle_vertices_local[primitive, 8] - v0z

        triangle_v0_world[primitive, 0] = Rwg00 * v0x + Rwg01 * v0y + Rwg02 * v0z + rwgx
        triangle_v0_world[primitive, 1] = Rwg10 * v0x + Rwg11 * v0y + Rwg12 * v0z + rwgy
        triangle_v0_world[primitive, 2] = Rwg20 * v0x + Rwg21 * v0y + Rwg22 * v0z + rwgz
        triangle_e1_world[primitive, 0] = Rwg00 * e1x + Rwg01 * e1y + Rwg02 * e1z
        triangle_e1_world[primitive, 1] = Rwg10 * e1x + Rwg11 * e1y + Rwg12 * e1z
        triangle_e1_world[primitive, 2] = Rwg20 * e1x + Rwg21 * e1y + Rwg22 * e1z
        triangle_e2_world[primitive, 0] = Rwg00 * e2x + Rwg01 * e2y + Rwg02 * e2z
        triangle_e2_world[primitive, 1] = Rwg10 * e2x + Rwg11 * e2y + Rwg12 * e2z
        triangle_e2_world[primitive, 2] = Rwg20 * e2x + Rwg21 * e2y + Rwg22 * e2z

        nx = e1y * e2z - e1z * e2y
        ny = e1z * e2x - e1x * e2z
        nz = e1x * e2y - e1y * e2x
        normal_norm = wp.sqrt(nx * nx + ny * ny + nz * nz)
        if normal_norm > _BUILD_EPS_F32:
            nx = nx / normal_norm
            ny = ny / normal_norm
            nz = nz / normal_norm
            triangle_normal_world[primitive, 0] = Rwg00 * nx + Rwg01 * ny + Rwg02 * nz
            triangle_normal_world[primitive, 1] = Rwg10 * nx + Rwg11 * ny + Rwg12 * nz
            triangle_normal_world[primitive, 2] = Rwg20 * nx + Rwg21 * ny + Rwg22 * nz

    @wp.kernel
    def _update_planes_world_kernel(
        plane_normal_local: wp.array2d(dtype=wp.float32),
        plane_point_local: wp.array2d(dtype=wp.float32),
        plane_instance_index: wp.array(dtype=wp.int32),
        instance_body_index: wp.array(dtype=wp.int32),
        instance_X_body_geometry_R: wp.array2d(dtype=wp.float32),
        instance_X_body_geometry_r: wp.array2d(dtype=wp.float32),
        frame_x_world_R: wp.array4d(dtype=wp.float32),
        frame_x_world_r: wp.array3d(dtype=wp.float32),
        env_idx: int,
        plane_normal_world: wp.array2d(dtype=wp.float32),
        plane_point_world: wp.array2d(dtype=wp.float32),
    ):
        primitive = wp.tid()
        instance_index = plane_instance_index[primitive]
        body_index = instance_body_index[instance_index]
        Rwb00 = wp.float32(1.0)
        Rwb01 = wp.float32(0.0)
        Rwb02 = wp.float32(0.0)
        Rwb10 = wp.float32(0.0)
        Rwb11 = wp.float32(1.0)
        Rwb12 = wp.float32(0.0)
        Rwb20 = wp.float32(0.0)
        Rwb21 = wp.float32(0.0)
        Rwb22 = wp.float32(1.0)
        rwbx = wp.float32(0.0)
        rwby = wp.float32(0.0)
        rwbz = wp.float32(0.0)
        if body_index >= 0:
            Rwb00 = frame_x_world_R[env_idx, body_index, 0, 0]
            Rwb01 = frame_x_world_R[env_idx, body_index, 0, 1]
            Rwb02 = frame_x_world_R[env_idx, body_index, 0, 2]
            Rwb10 = frame_x_world_R[env_idx, body_index, 1, 0]
            Rwb11 = frame_x_world_R[env_idx, body_index, 1, 1]
            Rwb12 = frame_x_world_R[env_idx, body_index, 1, 2]
            Rwb20 = frame_x_world_R[env_idx, body_index, 2, 0]
            Rwb21 = frame_x_world_R[env_idx, body_index, 2, 1]
            Rwb22 = frame_x_world_R[env_idx, body_index, 2, 2]
            rwbx = frame_x_world_r[env_idx, body_index, 0]
            rwby = frame_x_world_r[env_idx, body_index, 1]
            rwbz = frame_x_world_r[env_idx, body_index, 2]

        rbgx = instance_X_body_geometry_r[instance_index, 0]
        rbgy = instance_X_body_geometry_r[instance_index, 1]
        rbgz = instance_X_body_geometry_r[instance_index, 2]
        rwgx = rwbx + Rwb00 * rbgx + Rwb01 * rbgy + Rwb02 * rbgz
        rwgy = rwby + Rwb10 * rbgx + Rwb11 * rbgy + Rwb12 * rbgz
        rwgz = rwbz + Rwb20 * rbgx + Rwb21 * rbgy + Rwb22 * rbgz

        Rwg00 = _compose_R_component(
            Rwb00,
            Rwb01,
            Rwb02,
            Rwb10,
            Rwb11,
            Rwb12,
            Rwb20,
            Rwb21,
            Rwb22,
            instance_X_body_geometry_R,
            instance_index,
            0,
            0,
        )
        Rwg01 = _compose_R_component(
            Rwb00,
            Rwb01,
            Rwb02,
            Rwb10,
            Rwb11,
            Rwb12,
            Rwb20,
            Rwb21,
            Rwb22,
            instance_X_body_geometry_R,
            instance_index,
            0,
            1,
        )
        Rwg02 = _compose_R_component(
            Rwb00,
            Rwb01,
            Rwb02,
            Rwb10,
            Rwb11,
            Rwb12,
            Rwb20,
            Rwb21,
            Rwb22,
            instance_X_body_geometry_R,
            instance_index,
            0,
            2,
        )
        Rwg10 = _compose_R_component(
            Rwb00,
            Rwb01,
            Rwb02,
            Rwb10,
            Rwb11,
            Rwb12,
            Rwb20,
            Rwb21,
            Rwb22,
            instance_X_body_geometry_R,
            instance_index,
            1,
            0,
        )
        Rwg11 = _compose_R_component(
            Rwb00,
            Rwb01,
            Rwb02,
            Rwb10,
            Rwb11,
            Rwb12,
            Rwb20,
            Rwb21,
            Rwb22,
            instance_X_body_geometry_R,
            instance_index,
            1,
            1,
        )
        Rwg12 = _compose_R_component(
            Rwb00,
            Rwb01,
            Rwb02,
            Rwb10,
            Rwb11,
            Rwb12,
            Rwb20,
            Rwb21,
            Rwb22,
            instance_X_body_geometry_R,
            instance_index,
            1,
            2,
        )
        Rwg20 = _compose_R_component(
            Rwb00,
            Rwb01,
            Rwb02,
            Rwb10,
            Rwb11,
            Rwb12,
            Rwb20,
            Rwb21,
            Rwb22,
            instance_X_body_geometry_R,
            instance_index,
            2,
            0,
        )
        Rwg21 = _compose_R_component(
            Rwb00,
            Rwb01,
            Rwb02,
            Rwb10,
            Rwb11,
            Rwb12,
            Rwb20,
            Rwb21,
            Rwb22,
            instance_X_body_geometry_R,
            instance_index,
            2,
            1,
        )
        Rwg22 = _compose_R_component(
            Rwb00,
            Rwb01,
            Rwb02,
            Rwb10,
            Rwb11,
            Rwb12,
            Rwb20,
            Rwb21,
            Rwb22,
            instance_X_body_geometry_R,
            instance_index,
            2,
            2,
        )

        px = plane_point_local[primitive, 0]
        py = plane_point_local[primitive, 1]
        pz = plane_point_local[primitive, 2]
        plane_point_world[primitive, 0] = Rwg00 * px + Rwg01 * py + Rwg02 * pz + rwgx
        plane_point_world[primitive, 1] = Rwg10 * px + Rwg11 * py + Rwg12 * pz + rwgy
        plane_point_world[primitive, 2] = Rwg20 * px + Rwg21 * py + Rwg22 * pz + rwgz

        nx = plane_normal_local[primitive, 0]
        ny = plane_normal_local[primitive, 1]
        nz = plane_normal_local[primitive, 2]
        outx = Rwg00 * nx + Rwg01 * ny + Rwg02 * nz
        outy = Rwg10 * nx + Rwg11 * ny + Rwg12 * nz
        outz = Rwg20 * nx + Rwg21 * ny + Rwg22 * nz
        norm = wp.sqrt(outx * outx + outy * outy + outz * outz)
        if norm > 0.0:
            outx = outx / norm
            outy = outy / norm
            outz = outz / norm
        plane_normal_world[primitive, 0] = outx
        plane_normal_world[primitive, 1] = outy
        plane_normal_world[primitive, 2] = outz

else:

    def _update_triangles_derived_world_kernel(*args, **kwargs):  # pragma: no cover
        raise ImportError("DeviceOpticalSceneCache requires the optional warp package")

    def _update_planes_world_kernel(*args, **kwargs):  # pragma: no cover
        raise ImportError("DeviceOpticalSceneCache requires the optional warp package")
