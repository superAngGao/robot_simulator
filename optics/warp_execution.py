"""Warp-based optical executors."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np

from sensing.optical import OpticalRaySensorSpec

from .device import build_host_optical_primitive_workload
from .device_bvh import DeviceOpticalBvh
from .device_scene import DeviceOpticalSceneSnapshot
from .execution import OpticalComputeResult
from .scene import OpticalSceneSnapshot

try:  # pragma: no cover - exercised in GPU environments.
    import warp as wp

    _HAS_WARP = True
except Exception:  # pragma: no cover - keeps CPU-only imports working.
    wp = None
    _HAS_WARP = False

_BUILD_EPS = 1.0e-8
_DIR_EPS = 1.0e-8
_T_EPS = 1.0e-5
_ATTENUATION_EPS = 1.0e-12
_MAX_BVH_STACK = 32


class GpuBruteForceOpticalExecutor:
    """Warp brute-force first-hit executor for robot-scale optical scenes.

    L5A uses float32 device buffers and is intended for robot-scale scenes
    below roughly 1000 meters. It is a correctness-first device result path:
    no GPU BVH, no direct lighting, no shadows, and no string id channels.
    """

    capabilities = frozenset(
        {
            "range_m",
            "hit_mask",
            "position_world",
            "normal_world",
            "numeric_instance_id",
        }
    )

    def __init__(self, *, device=None, stream=None) -> None:
        if not _HAS_WARP:
            raise ImportError("GpuBruteForceOpticalExecutor requires the optional warp package")
        wp.init()
        self.device = wp.get_device("cuda:0" if device is None else device)
        self.stream = stream

    def execute(self, snapshot: OpticalSceneSnapshot, spec: OpticalRaySensorSpec) -> OpticalComputeResult:
        self._validate(snapshot, spec)
        workload = build_host_optical_primitive_workload(snapshot, sensor_role=spec.sensor_role)
        num_rays = spec.num_rays

        with _scoped_stream(self.stream):
            origins = wp.array(
                np.asarray(spec.origins_world, dtype=np.float32),
                dtype=wp.float32,
                device=self.device,
            )
            directions = wp.array(
                np.asarray(spec.directions_world, dtype=np.float32),
                dtype=wp.float32,
                device=self.device,
            )
            triangles = wp.array(
                workload.triangles_world.reshape((-1, 9)),
                dtype=wp.float32,
                device=self.device,
            )
            triangle_numeric_ids = wp.array(
                workload.triangle_numeric_instance_id,
                dtype=wp.int32,
                device=self.device,
            )
            triangle_source_keys = wp.array(
                workload.triangle_source_order_key,
                dtype=wp.int64,
                device=self.device,
            )
            plane_normals = wp.array(workload.plane_normal_world, dtype=wp.float32, device=self.device)
            plane_points = wp.array(workload.plane_point_world, dtype=wp.float32, device=self.device)
            plane_numeric_ids = wp.array(
                workload.plane_numeric_instance_id,
                dtype=wp.int32,
                device=self.device,
            )
            plane_source_keys = wp.array(workload.plane_source_order_key, dtype=wp.int64, device=self.device)

            hit_mask = wp.array(np.zeros(num_rays, dtype=np.int32), dtype=wp.int32, device=self.device)
            range_m = wp.array(
                np.full(num_rays, np.inf, dtype=np.float32),
                dtype=wp.float32,
                device=self.device,
            )
            position_world = wp.array(
                np.full((num_rays, 3), np.nan, dtype=np.float32),
                dtype=wp.float32,
                device=self.device,
            )
            normal_world = wp.array(
                np.full((num_rays, 3), np.nan, dtype=np.float32),
                dtype=wp.float32,
                device=self.device,
            )
            numeric_instance_id = wp.array(
                np.zeros(num_rays, dtype=np.int32),
                dtype=wp.int32,
                device=self.device,
            )

            wp.launch(
                _brute_force_first_hit_kernel,
                dim=num_rays,
                inputs=[
                    origins,
                    directions,
                    float(spec.max_distance),
                    plane_normals,
                    plane_points,
                    plane_numeric_ids,
                    plane_source_keys,
                    int(workload.plane_normal_world.shape[0]),
                    triangles,
                    triangle_numeric_ids,
                    triangle_source_keys,
                    int(workload.triangles_world.shape[0]),
                    hit_mask,
                    range_m,
                    position_world,
                    normal_world,
                    numeric_instance_id,
                ],
                device=self.device,
                stream=self.stream,
            )

        return OpticalComputeResult(
            frame_id=snapshot.frame_id,
            sim_time=snapshot.sim_time,
            env_idx=snapshot.env_idx,
            sensor_id=spec.sensor_id,
            location="device",
            channels={
                "hit_mask": hit_mask,
                "range_m": range_m,
                "position_world": position_world,
                "normal_world": normal_world,
                "numeric_instance_id": numeric_instance_id,
            },
            resources=(
                origins,
                directions,
                triangles,
                triangle_numeric_ids,
                triangle_source_keys,
                plane_normals,
                plane_points,
                plane_numeric_ids,
                plane_source_keys,
            ),
        )

    def _validate(self, snapshot: OpticalSceneSnapshot, spec: OpticalRaySensorSpec) -> None:
        if snapshot.frame_id != spec.frame_id:
            raise ValueError("snapshot.frame_id must match spec.frame_id")
        if snapshot.env_idx != spec.env_idx:
            raise ValueError("snapshot.env_idx must match spec.env_idx")
        if snapshot.location != "host":
            raise ValueError("L5A GpuBruteForceOpticalExecutor consumes host snapshots")
        if not np.isfinite(spec.max_distance):
            raise ValueError("L5A GpuBruteForceOpticalExecutor requires finite max_distance")


class GpuDeviceSceneOpticalExecutor:
    """Warp brute-force first-hit executor over a device-resident optical scene.

    L5C.1a keeps the same ray-major brute-force traversal as L5A, but consumes
    `DeviceOpticalSceneSnapshot` buffers directly. Geometry and primitive
    metadata stay on device; per-call host work is limited to uploading ray
    origins/directions.
    """

    capabilities = GpuBruteForceOpticalExecutor.capabilities

    def __init__(self, *, device=None, stream=None, use_aabb: bool = False) -> None:
        if not _HAS_WARP:
            raise ImportError("GpuDeviceSceneOpticalExecutor requires the optional warp package")
        wp.init()
        self.device = wp.get_device("cuda:0" if device is None else device)
        self.stream = stream
        self.use_aabb = bool(use_aabb)

    def execute(
        self, snapshot: DeviceOpticalSceneSnapshot, spec: OpticalRaySensorSpec
    ) -> OpticalComputeResult:
        self._validate(snapshot, spec)
        num_rays = spec.num_rays
        scene = snapshot.scene
        sensor_role_mask = np.int64(scene.role_table.mask_for(spec.sensor_role))

        with _scoped_stream(self.stream):
            _wait_on_event(snapshot.ready_event, stream=self.stream, device=self.device)
            origins = wp.array(
                np.asarray(spec.origins_world, dtype=np.float32),
                dtype=wp.float32,
                device=self.device,
            )
            directions = wp.array(
                np.asarray(spec.directions_world, dtype=np.float32),
                dtype=wp.float32,
                device=self.device,
            )
            hit_mask = wp.array(np.zeros(num_rays, dtype=np.int32), dtype=wp.int32, device=self.device)
            range_m = wp.array(
                np.full(num_rays, np.inf, dtype=np.float32),
                dtype=wp.float32,
                device=self.device,
            )
            position_world = wp.array(
                np.full((num_rays, 3), np.nan, dtype=np.float32),
                dtype=wp.float32,
                device=self.device,
            )
            normal_world = wp.array(
                np.full((num_rays, 3), np.nan, dtype=np.float32),
                dtype=wp.float32,
                device=self.device,
            )
            numeric_instance_id = wp.array(
                np.zeros(num_rays, dtype=np.int32),
                dtype=wp.int32,
                device=self.device,
            )

            if self.use_aabb:
                if snapshot.triangle_aabb_min is None or snapshot.triangle_aabb_max is None:
                    raise ValueError(
                        "GpuDeviceSceneOpticalExecutor(use_aabb=True) requires AABB snapshot buffers"
                    )
                wp.launch(
                    _device_scene_first_hit_aabb_kernel,
                    dim=num_rays,
                    inputs=[
                        origins,
                        directions,
                        float(spec.max_distance),
                        snapshot.plane_normal_world,
                        snapshot.plane_point_world,
                        scene.plane_numeric_instance_id,
                        scene.plane_source_order_key,
                        scene.plane_role_mask,
                        int(scene.num_planes),
                        snapshot.triangle_v0_world,
                        snapshot.triangle_e1_world,
                        snapshot.triangle_e2_world,
                        snapshot.triangle_normal_world,
                        snapshot.triangle_aabb_min,
                        snapshot.triangle_aabb_max,
                        scene.triangle_numeric_instance_id,
                        scene.triangle_source_order_key,
                        scene.triangle_role_mask,
                        int(scene.num_triangles),
                        sensor_role_mask,
                        hit_mask,
                        range_m,
                        position_world,
                        normal_world,
                        numeric_instance_id,
                    ],
                    device=self.device,
                    stream=self.stream,
                )
            else:
                wp.launch(
                    _device_scene_first_hit_kernel,
                    dim=num_rays,
                    inputs=[
                        origins,
                        directions,
                        float(spec.max_distance),
                        snapshot.plane_normal_world,
                        snapshot.plane_point_world,
                        scene.plane_numeric_instance_id,
                        scene.plane_source_order_key,
                        scene.plane_role_mask,
                        int(scene.num_planes),
                        snapshot.triangle_v0_world,
                        snapshot.triangle_e1_world,
                        snapshot.triangle_e2_world,
                        snapshot.triangle_normal_world,
                        scene.triangle_numeric_instance_id,
                        scene.triangle_source_order_key,
                        scene.triangle_role_mask,
                        int(scene.num_triangles),
                        sensor_role_mask,
                        hit_mask,
                        range_m,
                        position_world,
                        normal_world,
                        numeric_instance_id,
                    ],
                    device=self.device,
                    stream=self.stream,
                )
            ready_event = (self.stream or wp.get_stream(self.device)).record_event()

        return OpticalComputeResult(
            frame_id=snapshot.frame_id,
            sim_time=snapshot.sim_time,
            env_idx=snapshot.env_idx,
            sensor_id=spec.sensor_id,
            location="device",
            channels={
                "hit_mask": hit_mask,
                "range_m": range_m,
                "position_world": position_world,
                "normal_world": normal_world,
                "numeric_instance_id": numeric_instance_id,
            },
            ready_event=ready_event,
            resources=(
                origins,
                directions,
                snapshot.triangle_v0_world,
                snapshot.triangle_e1_world,
                snapshot.triangle_e2_world,
                snapshot.triangle_normal_world,
                snapshot.plane_normal_world,
                snapshot.plane_point_world,
                scene.triangle_numeric_instance_id,
                scene.triangle_source_order_key,
                scene.triangle_role_mask,
                scene.plane_numeric_instance_id,
                scene.plane_source_order_key,
                scene.plane_role_mask,
            )
            + ((snapshot.triangle_aabb_min, snapshot.triangle_aabb_max) if self.use_aabb else ()),
        )

    def _validate(self, snapshot: DeviceOpticalSceneSnapshot, spec: OpticalRaySensorSpec) -> None:
        if snapshot.frame_id != spec.frame_id:
            raise ValueError("snapshot.frame_id must match spec.frame_id")
        if snapshot.env_idx != spec.env_idx:
            raise ValueError("snapshot.env_idx must match spec.env_idx")
        if snapshot.scene.device != self.device:
            raise ValueError("DeviceOpticalSceneSnapshot device must match executor device")
        if not np.isfinite(spec.max_distance):
            raise ValueError("GpuDeviceSceneOpticalExecutor requires finite max_distance")


class GpuDeviceBvhOpticalExecutor:
    """Warp first-hit executor over a device scene plus flat triangle BVH."""

    capabilities = GpuBruteForceOpticalExecutor.capabilities | frozenset({"material_index"})

    def __init__(self, *, device=None, stream=None) -> None:
        if not _HAS_WARP:
            raise ImportError("GpuDeviceBvhOpticalExecutor requires the optional warp package")
        wp.init()
        self.device = wp.get_device("cuda:0" if device is None else device)
        self.stream = stream

    def execute(
        self,
        snapshot: DeviceOpticalSceneSnapshot,
        bvh: DeviceOpticalBvh,
        spec: OpticalRaySensorSpec,
    ) -> OpticalComputeResult:
        self._validate(snapshot, bvh, spec)
        num_rays = spec.num_rays
        scene = snapshot.scene
        sensor_role_mask = np.int64(scene.role_table.mask_for(spec.sensor_role))

        with _scoped_stream(self.stream):
            _wait_on_event(snapshot.ready_event, stream=self.stream, device=self.device)
            _wait_on_event(bvh.ready_event, stream=self.stream, device=self.device)
            origins = wp.array(
                np.asarray(spec.origins_world, dtype=np.float32),
                dtype=wp.float32,
                device=self.device,
            )
            directions = wp.array(
                np.asarray(spec.directions_world, dtype=np.float32),
                dtype=wp.float32,
                device=self.device,
            )
            hit_mask = wp.array(np.zeros(num_rays, dtype=np.int32), dtype=wp.int32, device=self.device)
            range_m = wp.array(
                np.full(num_rays, np.inf, dtype=np.float32),
                dtype=wp.float32,
                device=self.device,
            )
            position_world = wp.array(
                np.full((num_rays, 3), np.nan, dtype=np.float32),
                dtype=wp.float32,
                device=self.device,
            )
            normal_world = wp.array(
                np.full((num_rays, 3), np.nan, dtype=np.float32),
                dtype=wp.float32,
                device=self.device,
            )
            numeric_instance_id = wp.array(
                np.zeros(num_rays, dtype=np.int32),
                dtype=wp.int32,
                device=self.device,
            )
            material_index = wp.array(
                np.zeros(num_rays, dtype=np.int32),
                dtype=wp.int32,
                device=self.device,
            )
            bvh_stack_overflow_count = wp.array(
                np.zeros(1, dtype=np.int32),
                dtype=wp.int32,
                device=self.device,
            )
            bvh_max_stack_depth = wp.array(
                np.zeros(1, dtype=np.int32),
                dtype=wp.int32,
                device=self.device,
            )

            wp.launch(
                _device_scene_bvh_first_hit_kernel,
                dim=num_rays,
                inputs=[
                    origins,
                    directions,
                    float(spec.max_distance),
                    snapshot.plane_normal_world,
                    snapshot.plane_point_world,
                    scene.plane_numeric_instance_id,
                    scene.plane_source_order_key,
                    scene.plane_role_mask,
                    scene.plane_material_index,
                    int(scene.num_planes),
                    snapshot.triangle_v0_world,
                    snapshot.triangle_e1_world,
                    snapshot.triangle_e2_world,
                    snapshot.triangle_normal_world,
                    scene.triangle_numeric_instance_id,
                    scene.triangle_source_order_key,
                    scene.triangle_role_mask,
                    scene.triangle_material_index,
                    bvh.bounds_min,
                    bvh.bounds_max,
                    bvh.left,
                    bvh.right,
                    bvh.start,
                    bvh.count,
                    bvh.prim_ids,
                    int(bvh.num_nodes),
                    sensor_role_mask,
                    hit_mask,
                    range_m,
                    position_world,
                    normal_world,
                    numeric_instance_id,
                    material_index,
                    bvh_stack_overflow_count,
                    bvh_max_stack_depth,
                ],
                device=self.device,
                stream=self.stream,
            )
            ready_event = (self.stream or wp.get_stream(self.device)).record_event()

        return OpticalComputeResult(
            frame_id=snapshot.frame_id,
            sim_time=snapshot.sim_time,
            env_idx=snapshot.env_idx,
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
            ready_event=ready_event,
            resources=(
                origins,
                directions,
                snapshot.triangle_v0_world,
                snapshot.triangle_e1_world,
                snapshot.triangle_e2_world,
                snapshot.triangle_normal_world,
                snapshot.plane_normal_world,
                snapshot.plane_point_world,
                scene.triangle_numeric_instance_id,
                scene.triangle_source_order_key,
                scene.triangle_role_mask,
                scene.triangle_material_index,
                scene.plane_numeric_instance_id,
                scene.plane_source_order_key,
                scene.plane_role_mask,
                scene.plane_material_index,
                material_index,
                bvh_stack_overflow_count,
                bvh_max_stack_depth,
            )
            + bvh.resources,
        )

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
        if snapshot.scene.device != self.device:
            raise ValueError("DeviceOpticalSceneSnapshot device must match executor device")
        if bvh.device != self.device:
            raise ValueError("DeviceOpticalBvh device must match executor device")
        if bvh.frame_id != snapshot.frame_id:
            raise ValueError("DeviceOpticalBvh frame_id must match DeviceOpticalSceneSnapshot frame_id")
        if bvh.env_idx != snapshot.env_idx:
            raise ValueError("DeviceOpticalBvh env_idx must match DeviceOpticalSceneSnapshot env_idx")
        if not np.isfinite(spec.max_distance):
            raise ValueError("GpuDeviceBvhOpticalExecutor requires finite max_distance")


class GpuDeviceBvhDirectLightOpticalExecutor:
    """Direct-light executor over a device scene plus flat triangle BVH."""

    capabilities = GpuDeviceBvhOpticalExecutor.capabilities | frozenset({"rgb", "intensity"})

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
        if not _HAS_WARP:
            raise ImportError("GpuDeviceBvhDirectLightOpticalExecutor requires the optional warp package")
        wp.init()
        self.device = wp.get_device("cuda:0" if device is None else device)
        self.stream = stream
        self.shadows = bool(shadows)
        self.ambient_rgb = _as_rgb_array(ambient_rgb, name="ambient_rgb")
        self.background_rgb = _as_rgb_array(background_rgb, name="background_rgb")
        self.shadow_bias = float(shadow_bias)
        if self.shadow_bias < 0.0:
            raise ValueError("shadow_bias must be >= 0")
        self._first_hit = GpuDeviceBvhOpticalExecutor(device=self.device, stream=stream)

    def execute(
        self,
        snapshot: DeviceOpticalSceneSnapshot,
        bvh: DeviceOpticalBvh,
        spec: OpticalRaySensorSpec,
    ) -> OpticalComputeResult:
        geometry = self._first_hit.execute(snapshot, bvh, spec)
        scene = snapshot.scene
        num_rays = spec.num_rays

        with _scoped_stream(self.stream):
            _wait_on_event(geometry.ready_event, stream=self.stream, device=self.device)
            rgb = wp.array(
                np.repeat(self.background_rgb[None, :], num_rays, axis=0),
                dtype=wp.float32,
                device=self.device,
            )
            intensity = wp.array(
                np.zeros(num_rays, dtype=np.float32),
                dtype=wp.float32,
                device=self.device,
            )
            wp.launch(
                _device_scene_direct_light_kernel,
                dim=num_rays,
                inputs=[
                    geometry.channels["hit_mask"],
                    geometry.channels["position_world"],
                    geometry.channels["normal_world"],
                    geometry.channels["material_index"],
                    scene.material_albedo_rgb,
                    snapshot.plane_normal_world,
                    snapshot.plane_point_world,
                    scene.plane_role_mask,
                    int(scene.num_planes),
                    snapshot.triangle_v0_world,
                    snapshot.triangle_e1_world,
                    snapshot.triangle_e2_world,
                    snapshot.triangle_normal_world,
                    scene.triangle_role_mask,
                    bvh.bounds_min,
                    bvh.bounds_max,
                    bvh.left,
                    bvh.right,
                    bvh.start,
                    bvh.count,
                    bvh.prim_ids,
                    int(bvh.num_nodes),
                    scene.light_kind,
                    scene.light_position_or_direction_world,
                    scene.light_intensity,
                    scene.light_color_rgb,
                    int(scene.num_lights),
                    np.int64(scene.role_table.mask_for(spec.sensor_role)),
                    int(self.shadows),
                    float(self.shadow_bias),
                    float(self.ambient_rgb[0]),
                    float(self.ambient_rgb[1]),
                    float(self.ambient_rgb[2]),
                    float(self.background_rgb[0]),
                    float(self.background_rgb[1]),
                    float(self.background_rgb[2]),
                    rgb,
                    intensity,
                ],
                device=self.device,
                stream=self.stream,
            )
            ready_event = (self.stream or wp.get_stream(self.device)).record_event()

        channels = dict(geometry.channels)
        channels["rgb"] = rgb
        channels["intensity"] = intensity
        return OpticalComputeResult(
            frame_id=geometry.frame_id,
            sim_time=geometry.sim_time,
            env_idx=geometry.env_idx,
            sensor_id=geometry.sensor_id,
            location="device",
            channels=channels,
            ready_event=ready_event,
            resources=geometry.resources
            + (
                scene.material_albedo_rgb,
                scene.light_kind,
                scene.light_position_or_direction_world,
                scene.light_intensity,
                scene.light_color_rgb,
                rgb,
                intensity,
            ),
        )


def _as_rgb_array(value: tuple[float, float, float], *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values")
    return array.copy()


def _scoped_stream(stream):
    if stream is None:
        return nullcontext()
    return wp.ScopedStream(stream)


def _wait_on_event(event, *, stream, device) -> None:
    if event is None:
        return
    (stream or wp.get_stream(device)).wait_event(event)


if _HAS_WARP:

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
        occluded = bool(False)
        if max_distance <= 0.0:
            return occluded

        stack = wp.zeros(shape=_MAX_BVH_STACK, dtype=wp.int32)
        stack_t = wp.zeros(shape=_MAX_BVH_STACK, dtype=wp.float32)
        stack_size = wp.int32(0)
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
                max_distance,
            )
            if root_hit:
                stack[0] = wp.int32(0)
                stack_t[0] = root_t
                stack_size = wp.int32(1)

        while stack_size > 0 and not occluded:
            stack_size = stack_size - wp.int32(1)
            node_index = stack[stack_size]
            node_t = stack_t[stack_size]
            if node_t <= max_distance:
                leaf_count = bvh_count[node_index]
                if leaf_count > 0:
                    leaf_start = bvh_start[node_index]
                    for offset in range(leaf_count):
                        tri_idx = bvh_prim_ids[leaf_start + offset]
                        if (triangle_role_masks[tri_idx] & sensor_role_mask) != wp.int64(0):
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
                            max_distance,
                        )
                        if right_hit and stack_size < _MAX_BVH_STACK:
                            stack[stack_size] = right
                            stack_t[stack_size] = right_t
                            stack_size = stack_size + wp.int32(1)
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
                            max_distance,
                        )
                        if left_hit and stack_size < _MAX_BVH_STACK:
                            stack[stack_size] = left
                            stack_t[stack_size] = left_t
                            stack_size = stack_size + wp.int32(1)

        for plane_idx in range(num_planes):
            if not occluded and (plane_role_masks[plane_idx] & sensor_role_mask) != wp.int64(0):
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

        return occluded

    @wp.kernel
    def _device_scene_direct_light_kernel(
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
        light_kind: wp.array(dtype=wp.int32),
        light_position_or_direction_world: wp.array2d(dtype=wp.float32),
        light_intensity: wp.array(dtype=wp.float32),
        light_color_rgb: wp.array2d(dtype=wp.float32),
        num_lights: int,
        sensor_role_mask: wp.int64,
        shadows: int,
        shadow_bias: float,
        ambient_r: float,
        ambient_g: float,
        ambient_b: float,
        background_r: float,
        background_g: float,
        background_b: float,
        rgb: wp.array2d(dtype=wp.float32),
        intensity: wp.array(dtype=wp.float32),
    ):
        ray = wp.tid()
        out_r = wp.float32(background_r)
        out_g = wp.float32(background_g)
        out_b = wp.float32(background_b)

        if hit_mask[ray] != wp.int32(0):
            mat = material_index[ray]
            albedo_r = material_albedo_rgb[mat, 0]
            albedo_g = material_albedo_rgb[mat, 1]
            albedo_b = material_albedo_rgb[mat, 2]
            out_r = wp.float32(ambient_r) * albedo_r
            out_g = wp.float32(ambient_g) * albedo_g
            out_b = wp.float32(ambient_b) * albedo_b

            px = position_world[ray, 0]
            py = position_world[ray, 1]
            pz = position_world[ray, 2]
            nx = normal_world[ray, 0]
            ny = normal_world[ray, 1]
            nz = normal_world[ray, 2]

            for light_idx in range(num_lights):
                lx = light_position_or_direction_world[light_idx, 0]
                ly = light_position_or_direction_world[light_idx, 1]
                lz = light_position_or_direction_world[light_idx, 2]
                attenuation = wp.float32(1.0)
                if light_kind[light_idx] == wp.int32(1):
                    lx = lx - px
                    ly = ly - py
                    lz = lz - pz
                    distance_sq = lx * lx + ly * ly + lz * lz
                    distance = wp.sqrt(distance_sq)
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

                n_dot_l = nx * lx + ny * ly + nz * lz
                if n_dot_l > 0.0 and attenuation > 0.0:
                    visible_to_light = bool(True)
                    if shadows != 0:
                        shadow_max_distance = wp.float32(3.4028234663852886e38)
                        if light_kind[light_idx] == wp.int32(1):
                            light_distance = wp.sqrt(
                                (light_position_or_direction_world[light_idx, 0] - px)
                                * (light_position_or_direction_world[light_idx, 0] - px)
                                + (light_position_or_direction_world[light_idx, 1] - py)
                                * (light_position_or_direction_world[light_idx, 1] - py)
                                + (light_position_or_direction_world[light_idx, 2] - pz)
                                * (light_position_or_direction_world[light_idx, 2] - pz)
                            )
                            shadow_max_distance = light_distance - wp.float32(shadow_bias)
                        if shadow_max_distance > 0.0:
                            shadow_ox = px + nx * wp.float32(shadow_bias)
                            shadow_oy = py + ny * wp.float32(shadow_bias)
                            shadow_oz = pz + nz * wp.float32(shadow_bias)
                            visible_to_light = not _is_occluded_for_ray(
                                shadow_ox,
                                shadow_oy,
                                shadow_oz,
                                lx,
                                ly,
                                lz,
                                shadow_max_distance,
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
                    if visible_to_light:
                        scale = light_intensity[light_idx] * attenuation * n_dot_l
                        out_r = out_r + albedo_r * light_color_rgb[light_idx, 0] * scale
                        out_g = out_g + albedo_g * light_color_rgb[light_idx, 1] * scale
                        out_b = out_b + albedo_b * light_color_rgb[light_idx, 2] * scale

        rgb[ray, 0] = out_r
        rgb[ray, 1] = out_g
        rgb[ray, 2] = out_b
        if hit_mask[ray] != wp.int32(0):
            intensity[ray] = out_r * 0.2126 + out_g * 0.7152 + out_b * 0.0722
        else:
            intensity[ray] = wp.float32(0.0)

else:

    def _brute_force_first_hit_kernel(*args, **kwargs):  # pragma: no cover
        raise ImportError("GpuBruteForceOpticalExecutor requires the optional warp package")

    def _device_scene_first_hit_kernel(*args, **kwargs):  # pragma: no cover
        raise ImportError("GpuDeviceSceneOpticalExecutor requires the optional warp package")

    def _device_scene_first_hit_aabb_kernel(*args, **kwargs):  # pragma: no cover
        raise ImportError("GpuDeviceSceneOpticalExecutor requires the optional warp package")

    def _device_scene_bvh_first_hit_kernel(*args, **kwargs):  # pragma: no cover
        raise ImportError("GpuDeviceBvhOpticalExecutor requires the optional warp package")

    def _device_scene_no_shadow_direct_light_kernel(*args, **kwargs):  # pragma: no cover
        raise ImportError("GpuDeviceBvhDirectLightOpticalExecutor requires the optional warp package")
