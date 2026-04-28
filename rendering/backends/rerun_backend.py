"""
Rerun rendering backend.

Streams RenderScene data to the Rerun SDK (https://rerun.io).
Supports both live viewer (rr.connect) and file output (rr.save).

Shape mapping:
  box          -> rr.Boxes3D
  sphere       -> rr.Points3D  (with radii)
  cylinder     -> rr.Cylinders3D
  capsule      -> rr.Capsules3D
  convex_hull  -> rr.Mesh3D    (pre-triangulated faces from scene_builder)
  mesh         -> rr.Mesh3D when vertices + faces are available

Reference: Rerun Python SDK 0.31 archetype API.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Optional

import numpy as np

from ..render_scene import RenderScene
from .base import RenderBackend

log = logging.getLogger(__name__)

_SENSOR_SCALAR_GROUPS = frozenset({"contact", "joint", "force", "imu"})


class RerunBackend(RenderBackend):
    """Rerun-based rendering backend.

    Args:
        app_id    : Rerun application identifier.
        save_path : If set, ``open()`` calls ``rr.save(save_path)`` instead
                    of ``rr.connect()``.
    """

    def __init__(
        self,
        app_id: str = "robot_simulator",
        save_path: Optional[str] = None,
        terrain_half_size: float = 1.0,
        max_sensor_array_scalars: int = 32,
        log_sensor_data: bool = True,
        sensor_scalar_groups: Iterable[str] | None = None,
    ) -> None:
        self._app_id = app_id
        self._save_path = save_path
        self._terrain_half_size = float(terrain_half_size)
        self._max_sensor_array_scalars = int(max_sensor_array_scalars)
        self._log_sensor_data = bool(log_sensor_data)
        self._sensor_scalar_groups = _normalize_sensor_scalar_groups(sensor_scalar_groups)
        if self._terrain_half_size <= 0.0:
            raise ValueError(f"terrain_half_size must be > 0, got {terrain_half_size}")
        if self._max_sensor_array_scalars < 0:
            raise ValueError(f"max_sensor_array_scalars must be >= 0, got {max_sensor_array_scalars}")

    # ------------------------------------------------------------------
    # RenderBackend interface
    # ------------------------------------------------------------------

    def open(self) -> None:
        import rerun as rr  # lazy import -- optional dependency

        rr.init(self._app_id)
        if self._save_path:
            rr.save(self._save_path)
        else:
            rr.connect_grpc()

    def render_frame(self, scene: RenderScene, timestamp: float, env_index: int = 0) -> None:
        import rerun as rr

        rr.set_time("sim_time", timestamp=timestamp)
        prefix = f"env_{env_index}"

        # --- Shapes ---
        for i, shape in enumerate(scene.shapes):
            entity = f"{prefix}/shape_{i}_{shape.body_name}"
            pos = shape.position
            rot_mat = shape.rotation
            quat = _rot_to_quat(rot_mat)  # (w, x, y, z)
            xyzw = quat[[1, 2, 3, 0]]  # Rerun uses (x, y, z, w)

            stype = shape.shape_type
            if stype == "box":
                size = np.asarray(shape.params["size"], dtype=np.float32)
                rr.log(
                    entity,
                    rr.Boxes3D(
                        half_sizes=(size / 2).reshape(1, 3),
                        centers=pos.reshape(1, 3).astype(np.float32),
                        rotations=[rr.Quaternion(xyzw=xyzw)],
                    ),
                )
            elif stype == "sphere":
                rr.log(
                    entity,
                    rr.Points3D(
                        positions=pos.reshape(1, 3).astype(np.float32),
                        radii=[float(shape.params["radius"])],
                    ),
                )
            elif stype == "cylinder":
                rr.log(
                    entity,
                    rr.Cylinders3D(
                        radii=[float(shape.params["radius"])],
                        lengths=[float(shape.params["length"])],
                        centers=pos.reshape(1, 3).astype(np.float32),
                        rotations=[rr.Quaternion(xyzw=xyzw)],
                    ),
                )
            elif stype == "capsule":
                rr.log(
                    entity,
                    rr.Capsules3D(
                        radii=[float(shape.params["radius"])],
                        lengths=[float(shape.params["length"])],
                        translations=pos.reshape(1, 3).astype(np.float32),
                        rotations=[rr.Quaternion(xyzw=xyzw)],
                    ),
                )
            elif stype == "convex_hull":
                verts = shape.params["vertices"].astype(np.float32)
                faces = shape.params["faces"].astype(np.uint32)
                verts_world = (rot_mat @ verts.T).T + pos
                rr.log(
                    entity,
                    rr.Mesh3D(
                        vertex_positions=verts_world.astype(np.float32),
                        triangle_indices=faces,
                    ),
                )
            elif stype == "mesh":
                verts = shape.params.get("vertices")
                faces = shape.params.get("faces")
                if verts is None or faces is None:
                    log.debug(
                        "RerunBackend: 'mesh' requires vertices + faces, skipping %s",
                        entity,
                    )
                    continue
                verts = np.asarray(verts, dtype=np.float32)
                faces = np.asarray(faces, dtype=np.uint32)
                if verts.ndim != 2 or verts.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
                    log.warning(
                        "RerunBackend: expected mesh vertices (N, 3) and faces (F, 3), "
                        "got %s and %s; skipping %s",
                        verts.shape,
                        faces.shape,
                        entity,
                    )
                    continue
                verts_world = (rot_mat @ verts.T).T + pos
                rr.log(
                    entity,
                    rr.Mesh3D(
                        vertex_positions=verts_world.astype(np.float32),
                        triangle_indices=faces,
                    ),
                )
            # unknown / halfspace -- skip silently

        # --- Terrain ---
        terrain_mesh = _terrain_to_mesh(scene.terrain, half_size=self._terrain_half_size)
        if terrain_mesh is not None:
            vertices, faces = terrain_mesh
            rr.log(
                f"{prefix}/terrain",
                rr.Mesh3D(
                    vertex_positions=vertices.astype(np.float32),
                    triangle_indices=faces.astype(np.uint32),
                ),
            )

        # --- Contacts ---
        if scene.contacts:
            origins = np.array([c.position for c in scene.contacts], dtype=np.float32)
            vectors = np.array([c.normal * 0.05 for c in scene.contacts], dtype=np.float32)
            rr.log(f"{prefix}/contacts", rr.Arrows3D(origins=origins, vectors=vectors))

        # --- Skeleton links ---
        if scene.skeleton_links:
            strips = [np.array([p0, p1], dtype=np.float32) for p0, p1 in scene.skeleton_links]
            rr.log(f"{prefix}/skeleton", rr.LineStrips3D(strips))

        if self._log_sensor_data and scene.sensor_data is not None:
            _log_sensor_data(
                rr,
                prefix,
                scene.sensor_data,
                max_array_scalars=self._max_sensor_array_scalars,
                sensor_groups=self._sensor_scalar_groups,
            )

    def close(self) -> None:
        pass  # Rerun handles flush automatically

    def set_output(self, path: str) -> None:
        self._save_path = path

    @property
    def supports_offscreen(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z) via Shepperd's method."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def _terrain_to_mesh(scene_terrain, half_size: float = 1.0) -> tuple[np.ndarray, np.ndarray] | None:
    """Convert supported TerrainInfo values to a small debug plane mesh."""
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    if scene_terrain.terrain_type == "flat":
        z = float(scene_terrain.params.get("z", 0.0))
        vertices = np.array(
            [
                [-half_size, -half_size, z],
                [-half_size, half_size, z],
                [half_size, half_size, z],
                [half_size, -half_size, z],
            ],
            dtype=np.float32,
        )
        return vertices, faces

    if scene_terrain.terrain_type == "halfspace":
        normal = np.asarray(scene_terrain.params["normal"], dtype=np.float64)
        point = np.asarray(scene_terrain.params["point"], dtype=np.float64)
        nrm = np.linalg.norm(normal)
        if nrm < 1e-12:
            log.warning("RerunBackend: halfspace terrain has near-zero normal, skipping terrain")
            return None
        normal = normal / nrm
        if abs(normal[2]) < 0.99:
            t1 = np.cross(normal, np.array([0.0, 0.0, 1.0]))
        else:
            t1 = np.cross(normal, np.array([1.0, 0.0, 0.0]))
        t1 = t1 / np.linalg.norm(t1)
        t2 = np.cross(normal, t1)
        t2 = t2 / np.linalg.norm(t2)
        vertices = np.array(
            [
                point - half_size * t1 - half_size * t2,
                point - half_size * t1 + half_size * t2,
                point + half_size * t1 + half_size * t2,
                point + half_size * t1 - half_size * t2,
            ],
            dtype=np.float32,
        )
        return vertices, faces

    return None


def _normalize_sensor_scalar_groups(sensor_scalar_groups: Iterable[str] | None) -> frozenset[str]:
    if sensor_scalar_groups is None:
        return _SENSOR_SCALAR_GROUPS

    if isinstance(sensor_scalar_groups, str):
        sensor_scalar_groups = (sensor_scalar_groups,)

    groups = frozenset(str(group).lower() for group in sensor_scalar_groups)
    unknown = groups - _SENSOR_SCALAR_GROUPS
    if unknown:
        allowed = ", ".join(sorted(_SENSOR_SCALAR_GROUPS))
        got = ", ".join(sorted(unknown))
        raise ValueError(f"unknown sensor_scalar_groups: {got}; allowed: {allowed}")
    return groups


def _log_sensor_data(
    rr,
    prefix: str,
    sensor_data,
    max_array_scalars: int,
    sensor_groups: frozenset[str],
) -> None:
    """Log narrow RenderSensorData numeric readings as Rerun scalar timelines."""
    sensor_prefix = f"{prefix}/sensors"

    contact = getattr(sensor_data, "contact", None)
    if "contact" in sensor_groups and contact is not None:
        if getattr(contact, "contact_count", None) is not None:
            _log_scalar(rr, f"{sensor_prefix}/contact/contact_count", contact.contact_count)
        _log_array_scalars(
            rr,
            f"{sensor_prefix}/contact/contact_mask",
            getattr(contact, "contact_mask", None),
            max_array_scalars=max_array_scalars,
        )

    joint_state = getattr(sensor_data, "joint_state", None)
    if "joint" in sensor_groups and joint_state is not None:
        _log_array_scalars(
            rr,
            f"{sensor_prefix}/joint/q",
            getattr(joint_state, "joint_pos", None),
            max_array_scalars=max_array_scalars,
        )
        _log_array_scalars(
            rr,
            f"{sensor_prefix}/joint/qdot",
            getattr(joint_state, "joint_vel", None),
            max_array_scalars=max_array_scalars,
        )

    force = getattr(sensor_data, "force", None)
    if "force" in sensor_groups and force is not None:
        _log_array_scalars(
            rr,
            f"{sensor_prefix}/force/qfrc_applied",
            getattr(force, "qfrc_applied", None),
            max_array_scalars=max_array_scalars,
        )
        _log_array_scalars(
            rr,
            f"{sensor_prefix}/force/tau_smooth",
            getattr(force, "tau_smooth", None),
            max_array_scalars=max_array_scalars,
        )
        _log_array_scalars(
            rr,
            f"{sensor_prefix}/force/contact_force",
            getattr(force, "contact_force", None),
            max_array_scalars=max_array_scalars,
        )

    if "imu" in sensor_groups:
        for imu in getattr(sensor_data, "imu_readings", []) or []:
            body_idx = getattr(imu, "body_index", None)
            if body_idx is None:
                continue
            imu_prefix = f"{sensor_prefix}/imu/body_{int(body_idx)}"
            _log_array_scalars(
                rr,
                f"{imu_prefix}/angular_velocity_body",
                getattr(imu, "angular_velocity_body", None),
                max_array_scalars=max_array_scalars,
            )
            _log_array_scalars(
                rr,
                f"{imu_prefix}/linear_acceleration_body",
                getattr(imu, "linear_acceleration_body", None),
                max_array_scalars=max_array_scalars,
            )


def _log_scalar(rr, entity: str, value) -> None:
    rr.log(entity, rr.Scalars(float(value)))


def _log_array_scalars(rr, entity: str, values, max_array_scalars: int) -> None:
    if values is None:
        return

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return

    flat = arr.reshape(-1)
    if flat.size == 1:
        _log_scalar(rr, entity, flat[0])
        return

    rr.log(f"{entity}/norm", rr.Scalars(float(np.linalg.norm(flat))))
    for idx, value in enumerate(flat[:max_array_scalars]):
        rr.log(f"{entity}/{idx}", rr.Scalars(float(value)))

    if flat.size > max_array_scalars:
        rr.log(f"{entity}/truncated_size", rr.Scalars(float(flat.size)))
