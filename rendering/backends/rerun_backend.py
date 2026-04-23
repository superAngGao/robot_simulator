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
  mesh         -> skipped (warning logged)

Reference: Rerun Python SDK 0.31 archetype API.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..render_scene import RenderScene
from .base import RenderBackend

log = logging.getLogger(__name__)


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
    ) -> None:
        self._app_id = app_id
        self._save_path = save_path

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
                log.warning("RerunBackend: 'mesh' shape type not supported, skipping %s", entity)
            # unknown / halfspace -- skip silently

        # --- Contacts ---
        if scene.contacts:
            origins = np.array([c.position for c in scene.contacts], dtype=np.float32)
            vectors = np.array([c.normal * 0.05 for c in scene.contacts], dtype=np.float32)
            rr.log(f"{prefix}/contacts", rr.Arrows3D(origins=origins, vectors=vectors))

        # --- Skeleton links ---
        if scene.skeleton_links:
            strips = [np.array([p0, p1], dtype=np.float32) for p0, p1 in scene.skeleton_links]
            rr.log(f"{prefix}/skeleton", rr.LineStrips3D(strips))

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
