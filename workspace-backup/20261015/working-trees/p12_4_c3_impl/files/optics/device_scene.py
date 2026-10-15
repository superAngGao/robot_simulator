"""Device-resident optical scene cache primitives."""

from __future__ import annotations

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
