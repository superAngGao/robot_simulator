"""Runtime helpers for GPU optical execution over published GPU frames."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from physics.publish import CpuPublishedFrame, GpuPublishedFrame
from physics.spatial import SpatialTransform
from sensing.optical import OpticalRaySensorSpec

from .execution import OpticalComputeResult
from .registry import OpticalWorldRegistry
from .scene import OpticalFrameInputs, OpticalSceneCache
from .warp_execution import GpuBruteForceOpticalExecutor


def execute_optical_on_gpu_published_frame(
    engine,
    consumer_id: str,
    registry: OpticalWorldRegistry,
    spec: OpticalRaySensorSpec,
    *,
    frame: GpuPublishedFrame | None = None,
    stream=None,
    executor: GpuBruteForceOpticalExecutor | None = None,
) -> OpticalComputeResult:
    """Execute L5B GPU optical work against a Q52-borrowed GPU frame.

    The helper keeps the Q52 lifecycle explicit:

    1. borrow the GPU published frame on the optical stream;
    2. launch optical kernels writing executor-owned result buffers;
    3. complete the device consumer and attach the done event to the result.

    L5B.1 supports world-static instances and body-bound rigid instances by
    host-staging the selected env's `GpuPublishedFrame.x_world_*` transforms
    before GPU primitive upload. A fully device-resident scene cache is still
    deferred until kernels read those transforms directly.
    """
    borrowed = engine.borrow_device_frame(
        consumer_id,
        None if frame is None else frame.frame_id,
        stream=stream,
    )
    _validate_spec_matches_frame(spec, borrowed)

    snapshot = _snapshot_from_gpu_frame(registry, borrowed, env_idx=spec.env_idx)
    optical_executor = executor
    if optical_executor is None:
        optical_executor = GpuBruteForceOpticalExecutor(
            device=getattr(engine, "_device", None),
            stream=stream,
        )
    result = optical_executor.execute(snapshot, spec)
    done_event = engine.complete_device_consumer(consumer_id, borrowed.frame_id, stream=stream)
    return replace(result, ready_event=done_event)


def _validate_spec_matches_frame(spec: OpticalRaySensorSpec, frame: GpuPublishedFrame) -> None:
    if spec.frame_id != frame.frame_id:
        raise ValueError("spec.frame_id must match the borrowed GPU frame")
    if spec.sim_time != frame.sim_time:
        raise ValueError("spec.sim_time must match the borrowed GPU frame")


def _snapshot_from_gpu_frame(
    registry: OpticalWorldRegistry,
    frame: GpuPublishedFrame,
    *,
    env_idx: int,
):
    X_world = _host_stage_body_transforms_if_needed(registry, frame, env_idx=env_idx)
    cpu_frame = CpuPublishedFrame(
        frame_id=frame.frame_id,
        sim_time=frame.sim_time,
        step_index=frame.step_index,
        env_mask=None,
        q=np.zeros(0, dtype=np.float64),
        qdot=np.zeros(0, dtype=np.float64),
        X_world=X_world,
        v_bodies=[],
        contact_count=0,
        contacts=[],
        telemetry=None,
    )
    return OpticalSceneCache(registry).snapshot_from_frame_inputs(
        OpticalFrameInputs.from_published_frame(cpu_frame, env_idx=env_idx)
    )


def _host_stage_body_transforms_if_needed(
    registry: OpticalWorldRegistry,
    frame: GpuPublishedFrame,
    *,
    env_idx: int,
) -> list[SpatialTransform]:
    max_body_index = max(
        (instance.body_index for instance in registry.instances if instance.body_index is not None),
        default=None,
    )
    if max_body_index is None:
        return []

    _synchronize_frame_ready_event(frame.ready_event)
    rotations = np.asarray(frame.x_world_R_wp.numpy(), dtype=np.float64)
    translations = np.asarray(frame.x_world_r_wp.numpy(), dtype=np.float64)
    if rotations.ndim != 4 or rotations.shape[2:] != (3, 3):
        raise ValueError("GpuPublishedFrame.x_world_R_wp must have shape (num_envs, num_bodies, 3, 3)")
    if translations.ndim != 3 or translations.shape[2] != 3:
        raise ValueError("GpuPublishedFrame.x_world_r_wp must have shape (num_envs, num_bodies, 3)")
    if rotations.shape[:2] != translations.shape[:2]:
        raise ValueError("GpuPublishedFrame transform arrays must agree on env/body dimensions")
    if env_idx >= rotations.shape[0]:
        raise IndexError(f"env_idx {env_idx} is out of range for GpuPublishedFrame transforms")
    if max_body_index >= rotations.shape[1]:
        raise IndexError(f"body_index {max_body_index} is out of range for GpuPublishedFrame transforms")

    return [
        SpatialTransform(rotations[env_idx, body_index], translations[env_idx, body_index])
        for body_index in range(rotations.shape[1])
    ]


def _synchronize_frame_ready_event(ready_event: object | None) -> None:
    if ready_event is None:
        return
    try:
        import warp as wp
    except Exception:
        return
    wp.synchronize_event(ready_event)
