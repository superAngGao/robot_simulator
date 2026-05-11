"""Helpers for synthetic dynamic GPU published frames in the optical lab."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from physics.publish import GpuPublishedFrame


def gpu_pose_shape(frame: GpuPublishedFrame) -> tuple[int, int]:
    """Return `(num_envs, num_bodies)` for a GPU published pose frame."""

    rotation_shape = tuple(frame.x_world_R_wp.shape)
    translation_shape = tuple(frame.x_world_r_wp.shape)
    if len(rotation_shape) != 4 or rotation_shape[2:] != (3, 3):
        raise ValueError("GpuPublishedFrame.x_world_R_wp must have shape (num_envs, num_bodies, 3, 3)")
    if len(translation_shape) != 3 or translation_shape[2] != 3:
        raise ValueError("GpuPublishedFrame.x_world_r_wp must have shape (num_envs, num_bodies, 3)")
    if rotation_shape[:2] != translation_shape[:2]:
        raise ValueError("GpuPublishedFrame transform arrays must agree on env/body dimensions")
    return int(rotation_shape[0]), int(rotation_shape[1])


def clone_gpu_published_pose_frame(
    frame: GpuPublishedFrame,
    *,
    wp_module: Any,
    frame_id: int | None = None,
    sim_time: float | None = None,
    step_index: int | None = None,
    slot_id: int | None = None,
) -> GpuPublishedFrame:
    """Clone only the pose arrays needed by `DeviceOpticalSceneCache`.

    The returned frame is intentionally pose-only: broad physics buffers and
    slot metadata are not retained, so the clone is independent from the
    producer's publish ring. This helper is for controlled lab smokes, not the
    future real-time physics integration path where the renderer should borrow
    the publisher-owned frame.
    """

    gpu_pose_shape(frame)
    return _pose_only_frame(
        x_world_R_wp=_clone_wp_array(frame.x_world_R_wp, wp_module=wp_module),
        x_world_r_wp=_clone_wp_array(frame.x_world_r_wp, wp_module=wp_module),
        frame_id=frame.frame_id if frame_id is None else int(frame_id),
        sim_time=frame.sim_time if sim_time is None else float(sim_time),
        step_index=frame.step_index if step_index is None else int(step_index),
        slot_id=frame.slot_id if slot_id is None else int(slot_id),
    )


def clone_and_perturb_gpu_published_pose_frame(
    frame: GpuPublishedFrame,
    *,
    wp_module: Any,
    translation_offsets: Mapping[tuple[int, int], object],
    frame_id: int | None = None,
    sim_time: float | None = None,
    step_index: int | None = None,
    slot_id: int | None = None,
) -> GpuPublishedFrame:
    """Clone a pose frame and apply deterministic translation offsets.

    `translation_offsets` is keyed by `(env_idx, body_idx)` and each value is a
    3-vector added to `x_world_r`. The helper stages the translation array
    through host memory, which is acceptable for lab-only synthetic smokes but
    should not be used in the production physics/render hot path.
    """

    num_envs, num_bodies = gpu_pose_shape(frame)
    translated = np.asarray(frame.x_world_r_wp.numpy(), dtype=np.float32).copy()
    for (env_idx, body_idx), offset in translation_offsets.items():
        env_idx = int(env_idx)
        body_idx = int(body_idx)
        if env_idx < 0 or env_idx >= num_envs:
            raise IndexError(f"env_idx {env_idx} is out of range for GpuPublishedFrame transforms")
        if body_idx < 0 or body_idx >= num_bodies:
            raise IndexError(f"body_idx {body_idx} is out of range for GpuPublishedFrame transforms")
        delta = np.asarray(offset, dtype=np.float32)
        if delta.shape != (3,):
            raise ValueError("translation offset must have shape (3,)")
        translated[env_idx, body_idx, :] += delta

    return _pose_only_frame(
        x_world_R_wp=_clone_wp_array(frame.x_world_R_wp, wp_module=wp_module),
        x_world_r_wp=wp_module.array(
            translated,
            dtype=getattr(frame.x_world_r_wp, "dtype", None),
            device=getattr(frame.x_world_r_wp, "device", None),
        ),
        frame_id=frame.frame_id if frame_id is None else int(frame_id),
        sim_time=frame.sim_time if sim_time is None else float(sim_time),
        step_index=frame.step_index if step_index is None else int(step_index),
        slot_id=frame.slot_id if slot_id is None else int(slot_id),
    )


def _clone_wp_array(source, *, wp_module: Any):
    cloned = wp_module.zeros(
        tuple(source.shape),
        dtype=getattr(source, "dtype", None),
        device=getattr(source, "device", None),
    )
    wp_module.copy(cloned, source)
    return cloned


def _pose_only_frame(
    *,
    x_world_R_wp,
    x_world_r_wp,
    frame_id: int,
    sim_time: float,
    step_index: int,
    slot_id: int,
) -> GpuPublishedFrame:
    return GpuPublishedFrame(
        slot_id=slot_id,
        frame_id=frame_id,
        sim_time=sim_time,
        step_index=step_index,
        env_mask_wp=None,
        q_wp=None,
        qdot_wp=None,
        x_world_R_wp=x_world_R_wp,
        x_world_r_wp=x_world_r_wp,
        v_bodies_wp=None,
        contact_count_wp=None,
        contact_cache_ref=None,
        telemetry_ref=None,
        contact_mask_wp=None,
        ready_event=None,
        slot_meta=None,
    )
