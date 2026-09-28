"""Warp-based optical executors."""

from __future__ import annotations

import time
from contextlib import nullcontext

import numpy as np

from sensing.optical import OpticalPinholeCameraSpec, OpticalRaySensorSpec

from .device import build_host_optical_primitive_workload
from .device_bvh import DeviceOpticalBvh
from .device_scene import DeviceOpticalSceneSnapshot
from .execution import OpticalComputeResult, OpticalOutputProfile, normalize_output_profile
from .output_contract import SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES
from .scene import OpticalSceneSnapshot
from .warp_native_render import (
    _brute_force_first_hit_kernel,
    _device_scene_bvh_first_hit_kernel,
    _device_scene_direct_light_kernel,
    _device_scene_first_hit_aabb_kernel,
    _device_scene_first_hit_kernel,
    _init_bvh_hit_mask_kernel,
    _pinhole_camera_raygen_kernel,
)

try:  # pragma: no cover - exercised in GPU environments.
    import warp as wp

    _HAS_WARP = True
except Exception:  # pragma: no cover - keeps CPU-only imports working.
    wp = None
    _HAS_WARP = False

_SHADOW_TRAVERSAL_COUNTER_CHANNELS = SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES


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
    origins/directions. This executor is first-hit only. Shaded RGB output is
    intentionally available through `GpuDeviceBvhDirectLightOpticalExecutor`,
    because direct-light shadows require BVH any-hit traversal for production
    robot mesh scenes.
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
    supported_profiles = frozenset({OpticalOutputProfile.GEOMETRY_FULL})

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
        *,
        initialize_full_outputs: bool = True,
        render_profile: list[tuple[str, float]] | None = None,
    ) -> OpticalComputeResult:
        self._validate(snapshot, bvh, spec)
        num_rays = spec.num_rays

        with _scoped_stream(self.stream):
            start = time.perf_counter()
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
            _add_profile_ms(render_profile, "host_ray_upload_ms", start)

        return self._execute_device_ray_arrays(
            snapshot,
            bvh,
            frame_id=spec.frame_id,
            sim_time=spec.sim_time,
            env_idx=spec.env_idx,
            sensor_id=spec.sensor_id,
            origins=origins,
            directions=directions,
            num_rays=num_rays,
            max_distance=spec.max_distance,
            sensor_role=spec.sensor_role,
            initialize_full_outputs=initialize_full_outputs,
            render_profile=render_profile,
        )

    def execute_camera(
        self,
        snapshot: DeviceOpticalSceneSnapshot,
        bvh: DeviceOpticalBvh,
        camera: OpticalPinholeCameraSpec,
        *,
        initialize_full_outputs: bool = True,
        render_profile: list[tuple[str, float]] | None = None,
    ) -> OpticalComputeResult:
        """Execute first-hit BVH traversal for a pinhole camera without host ray materialization."""

        self._validate_camera(snapshot, bvh, camera)
        num_rays = int(camera.width * camera.height)
        start = time.perf_counter()
        R = np.asarray(camera.X_world_camera.R, dtype=np.float32)
        r = np.asarray(camera.X_world_camera.r, dtype=np.float32)
        _add_profile_ms(render_profile, "raygen_camera_params_ms", start)

        with _scoped_stream(self.stream):
            start = time.perf_counter()
            origins = wp.empty((num_rays, 3), dtype=wp.float32, device=self.device)
            directions = wp.empty((num_rays, 3), dtype=wp.float32, device=self.device)
            _add_profile_ms(render_profile, "raygen_buffer_alloc_ms", start)
            start = time.perf_counter()
            wp.launch(
                _pinhole_camera_raygen_kernel,
                dim=num_rays,
                inputs=[
                    int(camera.width),
                    int(camera.height),
                    float(camera.fx),
                    float(camera.fy),
                    float(camera.cx),
                    float(camera.cy),
                    float(r[0]),
                    float(r[1]),
                    float(r[2]),
                    float(R[0, 0]),
                    float(R[0, 1]),
                    float(R[0, 2]),
                    float(R[1, 0]),
                    float(R[1, 1]),
                    float(R[1, 2]),
                    float(R[2, 0]),
                    float(R[2, 1]),
                    float(R[2, 2]),
                    origins,
                    directions,
                ],
                device=self.device,
                stream=self.stream,
            )
            _record_profiled_event(self.stream, self.device, render_profile, "raygen_kernel_ms", start)

        return self._execute_device_ray_arrays(
            snapshot,
            bvh,
            frame_id=camera.frame_id,
            sim_time=camera.sim_time,
            env_idx=camera.env_idx,
            sensor_id=camera.sensor_id,
            origins=origins,
            directions=directions,
            num_rays=num_rays,
            max_distance=camera.max_distance,
            sensor_role=camera.sensor_role,
            initialize_full_outputs=initialize_full_outputs,
            render_profile=render_profile,
        )

    def _execute_device_ray_arrays(
        self,
        snapshot: DeviceOpticalSceneSnapshot,
        bvh: DeviceOpticalBvh,
        *,
        frame_id: int,
        sim_time: float,
        env_idx: int,
        sensor_id: str,
        origins,
        directions,
        num_rays: int,
        max_distance: float,
        sensor_role: str,
        initialize_full_outputs: bool = True,
        render_profile: list[tuple[str, float]] | None = None,
    ) -> OpticalComputeResult:
        scene = snapshot.scene
        sensor_role_mask = np.int64(scene.role_table.mask_for(sensor_role))

        with _scoped_stream(self.stream):
            start = time.perf_counter()
            _wait_on_event(snapshot.ready_event, stream=self.stream, device=self.device)
            _wait_on_event(bvh.ready_event, stream=self.stream, device=self.device)
            _add_profile_ms(render_profile, "first_hit_wait_inputs_ms", start)
            output_alloc_start = time.perf_counter()
            start = time.perf_counter()
            if initialize_full_outputs:
                hit_mask = wp.array(np.zeros(num_rays, dtype=np.int32), dtype=wp.int32, device=self.device)
                _add_profile_ms(render_profile, "first_hit_alloc_hit_mask_ms", start)
                start = time.perf_counter()
                range_m = wp.array(
                    np.full(num_rays, np.inf, dtype=np.float32),
                    dtype=wp.float32,
                    device=self.device,
                )
                _add_profile_ms(render_profile, "first_hit_alloc_range_m_ms", start)
                start = time.perf_counter()
                position_world = wp.array(
                    np.full((num_rays, 3), np.nan, dtype=np.float32),
                    dtype=wp.float32,
                    device=self.device,
                )
                _add_profile_ms(render_profile, "first_hit_alloc_position_world_ms", start)
                start = time.perf_counter()
                normal_world = wp.array(
                    np.full((num_rays, 3), np.nan, dtype=np.float32),
                    dtype=wp.float32,
                    device=self.device,
                )
                _add_profile_ms(render_profile, "first_hit_alloc_normal_world_ms", start)
                start = time.perf_counter()
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
                _add_profile_ms(render_profile, "first_hit_alloc_ids_ms", start)
            else:
                hit_mask = wp.empty(num_rays, dtype=wp.int32, device=self.device)
                _add_profile_ms(render_profile, "first_hit_alloc_hit_mask_ms", start)
                start = time.perf_counter()
                range_m = wp.empty(num_rays, dtype=wp.float32, device=self.device)
                _add_profile_ms(render_profile, "first_hit_alloc_range_m_ms", start)
                start = time.perf_counter()
                position_world = wp.empty((num_rays, 3), dtype=wp.float32, device=self.device)
                _add_profile_ms(render_profile, "first_hit_alloc_position_world_ms", start)
                start = time.perf_counter()
                normal_world = wp.empty((num_rays, 3), dtype=wp.float32, device=self.device)
                _add_profile_ms(render_profile, "first_hit_alloc_normal_world_ms", start)
                start = time.perf_counter()
                numeric_instance_id = wp.empty(num_rays, dtype=wp.int32, device=self.device)
                material_index = wp.empty(num_rays, dtype=wp.int32, device=self.device)
                _add_profile_ms(render_profile, "first_hit_alloc_ids_ms", start)
            start = time.perf_counter()
            bvh_stack_overflow_count = wp.zeros(1, dtype=wp.int32, device=self.device)
            bvh_max_stack_depth = wp.zeros(1, dtype=wp.int32, device=self.device)
            _add_profile_ms(render_profile, "first_hit_alloc_diagnostics_ms", start)
            _add_profile_ms(render_profile, "first_hit_output_alloc_ms", output_alloc_start)

            if not initialize_full_outputs:
                start = time.perf_counter()
                wp.launch(
                    _init_bvh_hit_mask_kernel,
                    dim=num_rays,
                    inputs=[hit_mask],
                    device=self.device,
                    stream=self.stream,
                )
                _record_profiled_event(
                    self.stream,
                    self.device,
                    render_profile,
                    "first_hit_init_outputs_ms",
                    start,
                )

            start = time.perf_counter()
            wp.launch(
                _device_scene_bvh_first_hit_kernel,
                dim=num_rays,
                inputs=[
                    origins,
                    directions,
                    float(max_distance),
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
            _sync_profiled_event(ready_event, render_profile, "first_hit_kernel_ms", start)

        return OpticalComputeResult(
            frame_id=frame_id,
            sim_time=sim_time,
            env_idx=env_idx,
            sensor_id=sensor_id,
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

    def _validate_camera(
        self,
        snapshot: DeviceOpticalSceneSnapshot,
        bvh: DeviceOpticalBvh,
        camera: OpticalPinholeCameraSpec,
    ) -> None:
        if snapshot.frame_id != camera.frame_id:
            raise ValueError("snapshot.frame_id must match camera.frame_id")
        if snapshot.env_idx != camera.env_idx:
            raise ValueError("snapshot.env_idx must match camera.env_idx")
        if snapshot.scene.device != self.device:
            raise ValueError("DeviceOpticalSceneSnapshot device must match executor device")
        if bvh.device != self.device:
            raise ValueError("DeviceOpticalBvh device must match executor device")
        if bvh.frame_id != snapshot.frame_id:
            raise ValueError("DeviceOpticalBvh frame_id must match DeviceOpticalSceneSnapshot frame_id")
        if bvh.env_idx != snapshot.env_idx:
            raise ValueError("DeviceOpticalBvh env_idx must match DeviceOpticalSceneSnapshot env_idx")
        if not np.isfinite(camera.max_distance):
            raise ValueError("GpuDeviceBvhOpticalExecutor requires finite max_distance")


class GpuDeviceBvhDirectLightOpticalExecutor:
    """Direct-light executor over a device scene plus flat triangle BVH.

    `rgb` and `intensity` are deterministic direct-light shading outputs, not
    path-traced radiance/sample accumulation channels. Shadow rays use an
    explicit `shadow_bias` and report aggregate shadow traversal diagnostics in
    the result channels.
    """

    capabilities = GpuDeviceBvhOpticalExecutor.capabilities | frozenset({"rgb", "intensity"})
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
        *,
        output_profile: OpticalOutputProfile | str = OpticalOutputProfile.DIRECT_LIGHT_FULL,
        render_profile: list[tuple[str, float]] | None = None,
    ) -> OpticalComputeResult:
        output_profile = normalize_output_profile(output_profile)
        _validate_supported_output_profile(output_profile, self.supported_profiles, type(self).__name__)
        initialize_full_outputs = output_profile is OpticalOutputProfile.DIRECT_LIGHT_FULL
        geometry = self._first_hit.execute(
            snapshot,
            bvh,
            spec,
            initialize_full_outputs=initialize_full_outputs,
            render_profile=render_profile,
        )
        return self._shade_geometry(
            snapshot,
            bvh,
            geometry,
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
    ) -> OpticalComputeResult:
        """Render a pinhole camera without materializing/uploading host ray buffers."""

        output_profile = normalize_output_profile(output_profile)
        _validate_supported_output_profile(output_profile, self.supported_profiles, type(self).__name__)
        initialize_full_outputs = output_profile is OpticalOutputProfile.DIRECT_LIGHT_FULL
        geometry = self._first_hit.execute_camera(
            snapshot,
            bvh,
            camera,
            initialize_full_outputs=initialize_full_outputs,
            render_profile=render_profile,
        )
        return self._shade_geometry(
            snapshot,
            bvh,
            geometry,
            sensor_role=camera.sensor_role,
            output_profile=output_profile,
            render_profile=render_profile,
        )

    def _shade_geometry(
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
        num_rays = int(geometry.channels["hit_mask"].shape[0])

        with _scoped_stream(self.stream):
            start = time.perf_counter()
            _wait_on_event(geometry.ready_event, stream=self.stream, device=self.device)
            _add_profile_ms(render_profile, "shade_wait_geometry_ms", start)
            start = time.perf_counter()
            output_alloc_start = start
            rgb = wp.empty((num_rays, 3), dtype=wp.float32, device=self.device)
            _add_profile_ms(render_profile, "shade_alloc_rgb_ms", start)
            start = time.perf_counter()
            intensity = wp.empty(num_rays, dtype=wp.float32, device=self.device)
            _add_profile_ms(render_profile, "shade_alloc_intensity_ms", start)
            start = time.perf_counter()
            shadow_stack_overflow_count = wp.zeros(1, dtype=wp.int32, device=self.device)
            shadow_max_stack_depth = wp.zeros(1, dtype=wp.int32, device=self.device)
            shadow_traversal_stats_enabled = int(render_profile is not None)
            shadow_traversal_counters = (
                {
                    name: wp.zeros(1, dtype=wp.int32, device=self.device)
                    for name in _SHADOW_TRAVERSAL_COUNTER_CHANNELS
                }
                if render_profile is not None
                else {name: shadow_stack_overflow_count for name in _SHADOW_TRAVERSAL_COUNTER_CHANNELS}
            )
            _add_profile_ms(render_profile, "shade_alloc_diagnostics_ms", start)
            _add_profile_ms(render_profile, "shade_output_alloc_ms", output_alloc_start)
            start = time.perf_counter()
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
                    np.int64(scene.role_table.mask_for(sensor_role)),
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
                    shadow_stack_overflow_count,
                    shadow_max_stack_depth,
                    shadow_traversal_stats_enabled,
                    shadow_traversal_counters["shadow_traversal_ray_count"],
                    shadow_traversal_counters["shadow_traversal_directional_ray_count"],
                    shadow_traversal_counters["shadow_traversal_point_ray_count"],
                    shadow_traversal_counters["shadow_traversal_occluded_count"],
                    shadow_traversal_counters["shadow_traversal_unoccluded_count"],
                    shadow_traversal_counters["shadow_traversal_node_visit_count"],
                    shadow_traversal_counters["shadow_traversal_leaf_visit_count"],
                    shadow_traversal_counters["shadow_traversal_triangle_test_count"],
                    shadow_traversal_counters["shadow_traversal_plane_test_count"],
                ],
                device=self.device,
                stream=self.stream,
            )
            ready_event = (self.stream or wp.get_stream(self.device)).record_event()
            _sync_profiled_event(ready_event, render_profile, "shade_kernel_ms", start)

        channels = dict(geometry.channels)
        channels["rgb"] = rgb
        channels["intensity"] = intensity
        channels["shadow_stack_overflow_count"] = shadow_stack_overflow_count
        channels["shadow_max_stack_depth"] = shadow_max_stack_depth
        channels = _filter_device_channels_for_profile(channels, output_profile)
        if render_profile is not None:
            channels.update(shadow_traversal_counters)
        return OpticalComputeResult(
            frame_id=geometry.frame_id,
            sim_time=geometry.sim_time,
            env_idx=geometry.env_idx,
            sensor_id=geometry.sensor_id,
            location="device",
            channels=channels,
            output_profile=output_profile,
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
                shadow_stack_overflow_count,
                shadow_max_stack_depth,
                *(tuple(shadow_traversal_counters.values()) if render_profile is not None else ()),
            ),
        )


def _as_rgb_array(value: tuple[float, float, float], *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values")
    return array.copy()


def _validate_supported_output_profile(
    output_profile: OpticalOutputProfile,
    supported_profiles: frozenset[OpticalOutputProfile],
    executor_name: str,
) -> None:
    if output_profile not in supported_profiles:
        choices = ", ".join(sorted(profile.value for profile in supported_profiles))
        raise ValueError(
            f"{executor_name} does not support output_profile={output_profile.value!r}; expected {choices}"
        )


def _filter_device_channels_for_profile(
    channels: dict[str, object],
    output_profile: OpticalOutputProfile,
) -> dict[str, object]:
    if output_profile is OpticalOutputProfile.DIRECT_LIGHT_FULL:
        return channels
    guaranteed = output_profile.guaranteed_channels
    return {name: value for name, value in channels.items() if name in guaranteed}


def _scoped_stream(stream):
    if stream is None:
        return nullcontext()
    return wp.ScopedStream(stream)


def _wait_on_event(event, *, stream, device) -> None:
    if event is None:
        return
    (stream or wp.get_stream(device)).wait_event(event)


def _add_profile_ms(
    render_profile: list[tuple[str, float]] | None,
    name: str,
    start: float,
) -> None:
    if render_profile is not None:
        render_profile.append((name, (time.perf_counter() - start) * 1000.0))


def _record_profiled_event(
    stream,
    device,
    render_profile: list[tuple[str, float]] | None,
    name: str,
    start: float,
) -> None:
    if render_profile is None:
        return
    event = (stream or wp.get_stream(device)).record_event()
    _sync_profiled_event(event, render_profile, name, start)


def _sync_profiled_event(
    event,
    render_profile: list[tuple[str, float]] | None,
    name: str,
    start: float,
) -> None:
    if render_profile is None:
        return
    wp.synchronize_event(event)
    render_profile.append((name, (time.perf_counter() - start) * 1000.0))
