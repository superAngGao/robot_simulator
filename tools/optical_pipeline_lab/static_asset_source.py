"""Static asset render source builders for the Optical Pipeline Lab."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from examples.mujoco_menagerie_robot_preview import import_mjcf_visual_scene
from physics.publish import CpuPublishedFrame, GpuPublishedFrame
from physics.spatial import SpatialTransform
from tools.optical_pipeline_lab import dynamic_frames
from tools.optical_pipeline_lab.render_session import (
    OpticalLabRenderSource,
    OpticalLabRenderWorkspace,
)

try:
    import warp as wp
except Exception as exc:  # pragma: no cover - example/lab-only guard.
    wp = None
    _WARP_IMPORT_ERROR = exc
else:
    _WARP_IMPORT_ERROR = None


def build_static_asset_render_source(
    args: argparse.Namespace,
    *,
    workspace: OpticalLabRenderWorkspace,
) -> OpticalLabRenderSource:
    """Build a lab render source from non-simulated static assets."""

    scene_preset = getattr(args, "scene_preset", "go2_menagerie_static")
    scene = build_static_asset_scene_for_preset(scene_preset, args)
    base_frame = (
        base_cpu_frame_for_static_asset_scene(
            scene_preset,
            frame_id=scene.frame.frame_id,
            sim_time=scene.frame.sim_time,
        )
        if workspace.device == "cpu"
        else base_gpu_frame_for_static_asset_scene(
            scene_preset,
            frame_id=scene.frame.frame_id,
            sim_time=scene.frame.sim_time,
            device=workspace.device,
        )
    )
    return OpticalLabRenderSource(
        registry=scene.registry,
        base_frame=base_frame,
        bounds_min=scene.bounds_min,
        bounds_max=scene.bounds_max,
        metadata={
            "scene": scene,
            "scene_preset": scene_preset,
            "source_kind": "static_asset",
            "cpu_base_frame": base_cpu_frame_for_static_asset_scene(
                scene_preset,
                frame_id=scene.frame.frame_id,
                sim_time=scene.frame.sim_time,
            ),
        },
    )


def scene_from_static_asset_render_source(source: OpticalLabRenderSource):
    """Return the scene object carried by a static asset render source."""

    return source.metadata["scene"]


def build_static_asset_scene_for_preset(scene_preset: str, args: argparse.Namespace):
    """Build a static asset scene for the selected lab preset."""

    if scene_preset == "go2_menagerie_static":
        return import_mjcf_visual_scene(Path(args.model_dir), model_xml=args.model_xml)
    if scene_preset == "synthetic_body_triangle":
        return synthetic_body_triangle_scene()
    raise NotImplementedError(
        f"scene_preset={scene_preset!r} is reserved; use go2_menagerie_static/synthetic_body_triangle for now"
    )


def synthetic_body_triangle_scene():
    """Return the tiny body-bound triangle scene used by dynamic lab smokes."""

    registry = dynamic_frames.make_body_bound_triangle_registry()
    return SimpleNamespace(
        registry=registry,
        frame=SimpleNamespace(frame_id=0, sim_time=0.0),
        bounds_min=np.array([-0.35, -0.35, -0.05], dtype=np.float64),
        bounds_max=np.array([0.45, 0.45, 0.85], dtype=np.float64),
        num_visual_geoms=1,
        num_triangles=1,
    )


def base_gpu_frame_for_static_asset_scene(
    scene_preset: str,
    *,
    frame_id: int,
    sim_time: float,
    device,
) -> GpuPublishedFrame:
    """Build the base GPU frame for a static asset scene."""

    _require_warp()
    if scene_preset == "synthetic_body_triangle":
        return dynamic_frames.make_gpu_pose_frame(
            wp_module=wp,
            translations=np.zeros((1, 1, 3), dtype=np.float32),
            frame_id=frame_id,
            sim_time=sim_time,
            device=device,
        )
    return static_gpu_frame(frame_id=frame_id, sim_time=sim_time, device=device)


def base_cpu_frame_for_static_asset_scene(
    scene_preset: str,
    *,
    frame_id: int,
    sim_time: float,
) -> CpuPublishedFrame:
    """Build the base CPU frame for CPU direct-light static asset rendering."""

    body_count = 1 if scene_preset == "synthetic_body_triangle" else 0
    return CpuPublishedFrame(
        frame_id=frame_id,
        sim_time=sim_time,
        step_index=frame_id,
        env_mask=None,
        q=None,
        qdot=None,
        X_world=tuple(SpatialTransform.identity() for _ in range(body_count)),
        v_bodies=None,
        contact_count=None,
        contacts=None,
        telemetry=None,
    )


def configure_dynamic_video_frame_inputs(args: argparse.Namespace, session) -> None:
    """Attach synthetic dynamic video frame inputs for supported static asset smokes."""

    if getattr(args, "video_frame_inputs", None) is not None:
        return
    if getattr(args, "scene_preset", "go2_menagerie_static") != "synthetic_body_triangle":
        return
    args.video_frame_inputs = synthetic_body_triangle_video_frame_inputs(
        session.gpu_frame,
        frames=int(args.video_frames),
        fps=float(args.video_fps),
    )
    args.video_geometry_mode = "dynamic_rigid"


def synthetic_body_triangle_video_frame_inputs(
    base_frame: GpuPublishedFrame,
    *,
    frames: int,
    fps: float,
) -> list[GpuPublishedFrame]:
    """Return perturbed synthetic triangle pose frames for dynamic video smokes."""

    _require_warp()
    frame_inputs: list[GpuPublishedFrame] = []
    sim_dt = 1.0 / fps if fps > 0.0 else 0.0
    for frame_index in range(max(frames, 0)):
        z_offset = 0.04 * float(frame_index % 4)
        frame_inputs.append(
            dynamic_frames.clone_and_perturb_gpu_published_pose_frame(
                base_frame,
                wp_module=wp,
                translation_offsets={(0, 0): [0.0, 0.0, z_offset]},
                frame_id=base_frame.frame_id + frame_index,
                sim_time=base_frame.sim_time + sim_dt * float(frame_index),
                step_index=base_frame.step_index + frame_index,
                slot_id=frame_index,
            )
        )
    return frame_inputs


def static_gpu_frame(*, frame_id: int, sim_time: float, device) -> GpuPublishedFrame:
    """Return an empty-pose GPU frame for world-static assets."""

    _require_warp()
    x_world_R = wp.zeros((1, 0, 3, 3), dtype=wp.float32, device=device)
    x_world_r = wp.zeros((1, 0, 3), dtype=wp.float32, device=device)
    return GpuPublishedFrame(
        slot_id=0,
        frame_id=frame_id,
        sim_time=sim_time,
        step_index=frame_id,
        env_mask_wp=None,
        q_wp=None,
        qdot_wp=None,
        x_world_R_wp=x_world_R,
        x_world_r_wp=x_world_r,
        v_bodies_wp=None,
        contact_count_wp=None,
        contact_cache_ref=None,
        telemetry_ref=None,
        ready_event=None,
        slot_meta=None,
    )


def _require_warp() -> None:
    if wp is None:
        raise SystemExit("static asset render sources require warp with CUDA support") from _WARP_IMPORT_ERROR
