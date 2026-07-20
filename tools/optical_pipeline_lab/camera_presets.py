"""Camera preset builders for Optical Pipeline Lab video workflows."""

from __future__ import annotations

import argparse
import math

import numpy as np

from physics.spatial import SpatialTransform
from sensing import OpticalPinholeCameraSpec


def build_model_bounds_camera(
    *,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    width: int,
    height: int,
    frame_id: int,
    sim_time: float,
    view: str,
) -> OpticalPinholeCameraSpec:
    """Build a fixed camera framing a model from scene bounds."""

    center = (bounds_min + bounds_max) * 0.5
    extent = float(np.linalg.norm(bounds_max - bounds_min))
    if extent <= 1.0e-9:
        extent = 1.0
    if view == "top":
        eye = center + np.array([0.0, -0.15 * extent, 1.7 * extent], dtype=np.float64)
    elif view == "side":
        eye = center + np.array([1.6 * extent, -0.08 * extent, 0.35 * extent], dtype=np.float64)
    else:
        eye = center + np.array([1.25 * extent, -1.85 * extent, 0.75 * extent], dtype=np.float64)
    focal = 0.72 * float(width)
    return OpticalPinholeCameraSpec(
        frame_id=frame_id,
        sim_time=sim_time,
        env_idx=0,
        sensor_id="optical_lab_preview_camera",
        width=int(width),
        height=int(height),
        fx=focal,
        fy=focal,
        cx=(int(width) - 1) / 2.0,
        cy=(int(height) - 1) / 2.0,
        X_world_camera=SpatialTransform(_look_at_camera_rotation(eye, center), eye),
        max_distance=20.0,
        sensor_role="rgb",
    )


def build_lab_video_camera(scene, args: argparse.Namespace, frame_index: int) -> OpticalPinholeCameraSpec:
    """Build the reviewed lab video camera for a scene-bounds video frame."""

    if args.video_mode == "fixed_view":
        return build_model_bounds_camera(
            bounds_min=scene.bounds_min,
            bounds_max=scene.bounds_max,
            width=args.width,
            height=args.height,
            frame_id=scene.frame.frame_id,
            sim_time=scene.frame.sim_time,
            view=args.view,
        )
    if args.video_mode == "pose_sequence":
        raise SystemExit("video-mode=pose_sequence is reserved for future changing-geometry benchmarks")
    center = (scene.bounds_min + scene.bounds_max) * 0.5
    extent = float(np.linalg.norm(scene.bounds_max - scene.bounds_min))
    if extent <= 1.0e-9:
        extent = 1.0
    angle = 2.0 * math.pi * float(frame_index) / max(float(args.video_frames), 1.0)
    radius = 1.85 * extent
    eye = center + np.array(
        [
            radius * math.cos(angle - math.pi * 0.32),
            radius * math.sin(angle - math.pi * 0.32),
            0.72 * extent,
        ],
        dtype=np.float64,
    )
    focal = 0.72 * float(args.width)
    return OpticalPinholeCameraSpec(
        frame_id=scene.frame.frame_id,
        sim_time=scene.frame.sim_time,
        env_idx=0,
        sensor_id=f"optical_lab_video_camera_{frame_index:06d}",
        width=int(args.width),
        height=int(args.height),
        fx=focal,
        fy=focal,
        cx=(int(args.width) - 1) / 2.0,
        cy=(int(args.height) - 1) / 2.0,
        X_world_camera=SpatialTransform(_look_at_camera_rotation(eye, center), eye),
        max_distance=20.0,
        sensor_role="rgb",
    )


def _look_at_camera_rotation(eye: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
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
