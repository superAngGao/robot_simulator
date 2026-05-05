"""Robot-like optical scenes shared by benchmarks and examples."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from optics import (
    OpticalInstanceSpec,
    OpticalLightSpec,
    OpticalMaterialSpec,
    OpticalWorldRegistry,
)
from physics.publish import CpuPublishedFrame, GpuPublishedFrame
from physics.spatial import SpatialTransform
from sensing import OpticalPinholeCameraSpec, build_pinhole_camera_rays


@dataclass(frozen=True)
class RobotOpticalScene:
    registry: OpticalWorldRegistry
    cpu_frame: CpuPublishedFrame
    num_triangles: int
    num_bodies: int


def build_robot_optical_scene(
    *,
    detail: str = "dense",
    num_robots: int = 1,
    include_floor: bool = True,
) -> RobotOpticalScene:
    """Build a deterministic body-bound quadruped-like optical scene."""

    params = _detail_params(detail)
    registry = OpticalWorldRegistry()
    _add_materials_and_lights(registry)
    roles = frozenset({"rgb", "depth", "lidar", "segmentation"})

    if include_floor:
        registry.add_plane_geometry("floor", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 0.0])
        registry.add_instance(OpticalInstanceSpec("floor", "floor", "mat_floor", roles=roles))

    torso_vertices, torso_triangles = subdivided_box_mesh(
        size=(0.58, 0.28, 0.18), subdivisions=params["box_subdiv"]
    )
    leg_vertices, leg_triangles = cylinder_mesh(
        radius=0.035,
        length=1.0,
        segments=params["cylinder_segments"],
        rings=params["cylinder_rings"],
    )
    foot_vertices, foot_triangles = uv_sphere_mesh(
        radius=0.055,
        segments=params["sphere_segments"],
        rings=params["sphere_rings"],
    )
    registry.add_triangle_mesh_geometry(
        "torso_mesh", vertices_local=torso_vertices, triangles=torso_triangles
    )
    registry.add_triangle_mesh_geometry("leg_mesh", vertices_local=leg_vertices, triangles=leg_triangles)
    registry.add_triangle_mesh_geometry("foot_mesh", vertices_local=foot_vertices, triangles=foot_triangles)

    transforms: list[SpatialTransform] = []
    robot_offsets = _robot_offsets(num_robots)
    for robot_index, offset in enumerate(robot_offsets):
        prefix = f"robot_{robot_index}"
        _add_robot_instances(
            registry,
            transforms,
            prefix=prefix,
            offset=offset,
            roles=roles,
        )

    return RobotOpticalScene(
        registry=registry,
        cpu_frame=make_cpu_frame(transforms),
        num_triangles=_count_registry_triangles(registry),
        num_bodies=len(transforms),
    )


def make_robot_gpu_frame(scene: RobotOpticalScene, *, device) -> GpuPublishedFrame:
    """Create a single-env GPU frame whose body transforms match `scene`."""

    import warp as wp

    rotations = np.asarray([X.R for X in scene.cpu_frame.X_world], dtype=np.float32)[None, :, :, :]
    translations = np.asarray([X.r for X in scene.cpu_frame.X_world], dtype=np.float32)[None, :, :]
    return GpuPublishedFrame(
        slot_id=0,
        frame_id=scene.cpu_frame.frame_id,
        sim_time=scene.cpu_frame.sim_time,
        step_index=scene.cpu_frame.step_index,
        env_mask_wp=None,
        q_wp=None,
        qdot_wp=None,
        x_world_R_wp=wp.array(rotations, dtype=wp.float32, device=device),
        x_world_r_wp=wp.array(translations, dtype=wp.float32, device=device),
        v_bodies_wp=None,
        contact_count_wp=None,
        contact_cache_ref=None,
        telemetry_ref=None,
        ready_event=None,
        slot_meta=None,
    )


def make_robot_camera(
    *,
    width: int,
    height: int,
    frame_id: int,
    sim_time: float,
    num_robots: int = 1,
    view: str = "front",
    sensor_role: str = "depth",
) -> OpticalPinholeCameraSpec:
    if view == "ego":
        eye = np.array([0.05, -0.55, 0.82], dtype=np.float64)
        target = np.array([0.20, 0.35, 0.45], dtype=np.float64)
        focal_scale = 0.78
    elif view == "top":
        span = max(1.0, math.sqrt(max(num_robots, 1)) * 1.1)
        eye = np.array([0.0, -0.1, 4.2 + 0.35 * span], dtype=np.float64)
        target = np.array([0.0, 0.0, 0.45], dtype=np.float64)
        focal_scale = 0.9
    else:
        span = max(1.0, math.sqrt(max(num_robots, 1)) * 1.15)
        eye = np.array([2.4 * span, -3.8 * span, 1.65], dtype=np.float64)
        target = np.array([0.0, 0.0, 0.48], dtype=np.float64)
        focal_scale = 0.72
    focal = focal_scale * float(width)
    return OpticalPinholeCameraSpec(
        frame_id=frame_id,
        sim_time=sim_time,
        env_idx=0,
        sensor_id=f"robot_{view}_camera",
        width=width,
        height=height,
        fx=focal,
        fy=focal,
        cx=(width - 1) / 2.0,
        cy=(height - 1) / 2.0,
        X_world_camera=SpatialTransform(look_at_camera_R(eye, target), eye),
        max_distance=20.0,
        sensor_role=sensor_role,
    )


def make_robot_camera_rays(
    *,
    num_rays: int,
    frame_id: int,
    sim_time: float,
    num_robots: int = 1,
    view: str = "front",
    sensor_role: str = "depth",
):
    side = int(math.ceil(math.sqrt(num_rays)))
    camera = make_robot_camera(
        width=side,
        height=side,
        frame_id=frame_id,
        sim_time=sim_time,
        num_robots=num_robots,
        view=view,
        sensor_role=sensor_role,
    )
    rays = build_pinhole_camera_rays(camera)
    return type(rays)(
        frame_id=rays.frame_id,
        sim_time=rays.sim_time,
        env_idx=rays.env_idx,
        sensor_id=rays.sensor_id,
        origins_world=rays.origins_world[:num_rays],
        directions_world=rays.directions_world[:num_rays],
        max_distance=rays.max_distance,
        sensor_role=rays.sensor_role,
        ray_shape=None,
    )


def make_cpu_frame(transforms: list[SpatialTransform]) -> CpuPublishedFrame:
    return CpuPublishedFrame(
        frame_id=1,
        sim_time=0.016,
        step_index=1,
        env_mask=None,
        q=np.zeros(0, dtype=np.float64),
        qdot=np.zeros(0, dtype=np.float64),
        X_world=list(transforms),
        v_bodies=[],
        contact_count=0,
        contacts=[],
        telemetry=None,
    )


def subdivided_box_mesh(
    *, size: tuple[float, float, float], subdivisions: int
) -> tuple[np.ndarray, np.ndarray]:
    sx, sy, sz = (float(v) for v in size)
    h = np.array([sx, sy, sz], dtype=np.float64) * 0.5
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    faces = (
        (0, 1, 2, +1.0),
        (0, 1, 2, -1.0),
        (1, 0, 2, +1.0),
        (1, 0, 2, -1.0),
        (2, 0, 1, +1.0),
        (2, 0, 1, -1.0),
    )
    n = max(1, int(subdivisions))
    for fixed_axis, u_axis, v_axis, sign in faces:
        base = len(vertices)
        for j in range(n + 1):
            v = -1.0 + 2.0 * float(j) / float(n)
            for i in range(n + 1):
                u = -1.0 + 2.0 * float(i) / float(n)
                p = np.zeros(3, dtype=np.float64)
                p[fixed_axis] = sign * h[fixed_axis]
                p[u_axis] = u * h[u_axis]
                p[v_axis] = v * h[v_axis]
                vertices.append(p.tolist())
        for j in range(n):
            for i in range(n):
                a = base + j * (n + 1) + i
                b = a + 1
                c = a + (n + 1)
                d = c + 1
                if sign > 0.0:
                    triangles.extend([[a, b, d], [a, d, c]])
                else:
                    triangles.extend([[a, d, b], [a, c, d]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int64)


def cylinder_mesh(
    *, radius: float, length: float, segments: int, rings: int
) -> tuple[np.ndarray, np.ndarray]:
    segments = max(3, int(segments))
    rings = max(1, int(rings))
    radius = float(radius)
    half = float(length) * 0.5
    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    zs = np.linspace(-half, half, rings + 1)
    vertices: list[list[float]] = []
    for z in zs:
        for angle in angles:
            vertices.append([radius * math.cos(angle), radius * math.sin(angle), float(z)])
    triangles: list[list[int]] = []
    for ring in range(rings):
        row = ring * segments
        next_row = (ring + 1) * segments
        for i in range(segments):
            j = (i + 1) % segments
            a = row + i
            b = row + j
            c = next_row + i
            d = next_row + j
            triangles.extend([[a, b, d], [a, d, c]])
    bottom_center = len(vertices)
    vertices.append([0.0, 0.0, -half])
    top_center = len(vertices)
    vertices.append([0.0, 0.0, half])
    top_row = rings * segments
    for i in range(segments):
        j = (i + 1) % segments
        triangles.append([bottom_center, j, i])
        triangles.append([top_center, top_row + i, top_row + j])
    return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int64)


def uv_sphere_mesh(*, radius: float, segments: int, rings: int) -> tuple[np.ndarray, np.ndarray]:
    segments = max(3, int(segments))
    rings = max(2, int(rings))
    radius = float(radius)
    vertices = [[0.0, 0.0, radius]]
    for ring in range(1, rings):
        phi = math.pi * float(ring) / float(rings)
        z = radius * math.cos(phi)
        r = radius * math.sin(phi)
        for i in range(segments):
            theta = 2.0 * math.pi * float(i) / float(segments)
            vertices.append([r * math.cos(theta), r * math.sin(theta), z])
    bottom = len(vertices)
    vertices.append([0.0, 0.0, -radius])

    triangles: list[list[int]] = []
    first = 1
    for i in range(segments):
        triangles.append([0, first + i, first + ((i + 1) % segments)])
    for ring in range(rings - 2):
        row = first + ring * segments
        next_row = row + segments
        for i in range(segments):
            j = (i + 1) % segments
            triangles.extend([[row + i, next_row + i, next_row + j], [row + i, next_row + j, row + j]])
    last_row = first + (rings - 2) * segments
    for i in range(segments):
        triangles.append([bottom, last_row + ((i + 1) % segments), last_row + i])
    return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int64)


def look_at_camera_R(eye: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    z_axis = target - eye
    z_axis = z_axis / np.linalg.norm(z_axis)
    up_world = np.asarray(up, dtype=np.float64)
    x_axis = np.cross(z_axis, up_world)
    if np.linalg.norm(x_axis) < 1.0e-9:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def _add_materials_and_lights(registry: OpticalWorldRegistry) -> None:
    registry.add_material(OpticalMaterialSpec("mat_floor", albedo_rgb=(0.55, 0.57, 0.54)))
    registry.add_material(OpticalMaterialSpec("mat_torso", albedo_rgb=(0.82, 0.22, 0.16)))
    registry.add_material(OpticalMaterialSpec("mat_leg", albedo_rgb=(0.18, 0.42, 0.78)))
    registry.add_material(OpticalMaterialSpec("mat_foot", albedo_rgb=(0.08, 0.09, 0.10)))
    registry.add_light(
        OpticalLightSpec(
            "key",
            kind="directional",
            position_or_direction_world=np.array([-0.4, -0.5, 1.0], dtype=np.float64),
            intensity=1.9,
            color_rgb=(1.0, 0.96, 0.9),
        )
    )
    registry.add_light(
        OpticalLightSpec(
            "fill",
            kind="point",
            position_or_direction_world=np.array([1.6, -2.8, 2.1], dtype=np.float64),
            intensity=3.2,
            color_rgb=(0.55, 0.68, 1.0),
        )
    )


def _add_robot_instances(
    registry: OpticalWorldRegistry,
    transforms: list[SpatialTransform],
    *,
    prefix: str,
    offset: np.ndarray,
    roles: frozenset[str],
) -> None:
    torso_index = _append_transform(
        transforms, SpatialTransform(np.eye(3), offset + np.array([0.0, 0.0, 0.62]))
    )
    registry.add_instance(
        OpticalInstanceSpec(
            f"{prefix}/torso",
            "torso_mesh",
            "mat_torso",
            body_index=torso_index,
            roles=roles,
        )
    )

    leg_signs = (("fl", +1.0, +1.0), ("fr", +1.0, -1.0), ("rl", -1.0, +1.0), ("rr", -1.0, -1.0))
    for leg_name, sx, sy in leg_signs:
        hip = offset + np.array([0.23 * sx, 0.13 * sy, 0.57])
        thigh_vec = np.array([0.09 * sx, 0.025 * sy, -0.30])
        calf_vec = np.array([-0.035 * sx, -0.01 * sy, -0.31])
        knee = hip + thigh_vec
        ankle = knee + calf_vec

        thigh_index = _append_transform(transforms, _link_transform(hip, knee))
        calf_index = _append_transform(transforms, _link_transform(knee, ankle))
        foot_index = _append_transform(
            transforms, SpatialTransform(np.eye(3), ankle + np.array([0.0, 0.0, -0.035]))
        )

        registry.add_instance(
            OpticalInstanceSpec(
                f"{prefix}/{leg_name}_thigh",
                "leg_mesh",
                "mat_leg",
                body_index=thigh_index,
                roles=roles,
            )
        )
        registry.add_instance(
            OpticalInstanceSpec(
                f"{prefix}/{leg_name}_calf",
                "leg_mesh",
                "mat_leg",
                body_index=calf_index,
                roles=roles,
            )
        )
        registry.add_instance(
            OpticalInstanceSpec(
                f"{prefix}/{leg_name}_foot",
                "foot_mesh",
                "mat_foot",
                body_index=foot_index,
                roles=roles,
            )
        )


def _link_transform(start: np.ndarray, end: np.ndarray) -> SpatialTransform:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 1.0e-12:
        raise ValueError("link endpoints must be distinct")
    return SpatialTransform(_rotation_from_z(direction / length), (start + end) * 0.5)


def _rotation_from_z(z_axis: np.ndarray) -> np.ndarray:
    z = np.asarray(z_axis, dtype=np.float64)
    z = z / np.linalg.norm(z)
    helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(z, helper))) > 0.95:
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x = np.cross(helper, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def _append_transform(transforms: list[SpatialTransform], transform: SpatialTransform) -> int:
    transforms.append(transform)
    return len(transforms) - 1


def _robot_offsets(num_robots: int) -> list[np.ndarray]:
    count = max(1, int(num_robots))
    side = int(math.ceil(math.sqrt(count)))
    spacing = 1.15
    offsets: list[np.ndarray] = []
    for index in range(count):
        row = index // side
        col = index % side
        offsets.append(
            np.array(
                [
                    (col - (side - 1) * 0.5) * spacing,
                    (row - (side - 1) * 0.5) * spacing,
                    0.0,
                ],
                dtype=np.float64,
            )
        )
    return offsets


def _detail_params(detail: str) -> dict[str, int]:
    if detail == "proxy":
        return {
            "box_subdiv": 2,
            "cylinder_segments": 12,
            "cylinder_rings": 2,
            "sphere_segments": 12,
            "sphere_rings": 6,
        }
    if detail == "preview":
        return {
            "box_subdiv": 10,
            "cylinder_segments": 32,
            "cylinder_rings": 10,
            "sphere_segments": 24,
            "sphere_rings": 12,
        }
    if detail == "xlarge":
        return {
            "box_subdiv": 48,
            "cylinder_segments": 160,
            "cylinder_rings": 96,
            "sphere_segments": 96,
            "sphere_rings": 48,
        }
    if detail != "dense":
        raise ValueError("detail must be one of: proxy, preview, dense, xlarge")
    return {
        "box_subdiv": 32,
        "cylinder_segments": 128,
        "cylinder_rings": 64,
        "sphere_segments": 64,
        "sphere_rings": 32,
    }


def _count_registry_triangles(registry: OpticalWorldRegistry) -> int:
    geometry = registry.geometry
    total = 0
    for instance in registry.instances:
        geom = geometry[instance.geometry_id]
        triangles = getattr(geom, "triangles", None)
        if triangles is not None:
            total += int(np.asarray(triangles).shape[0])
    return total
