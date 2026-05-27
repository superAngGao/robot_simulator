"""Runner entry points for the Optical Pipeline Lab."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

from sensing import OpticalPinholeCameraSpec

from .delivery import VideoDeliveryFacade, VideoDeliveryRunConfig, VideoFrameTimingRowBuilder
from .frame_runtime import FrameWorkflowRunner
from .render_session import OpticalLabRenderOptions
from .scenarios import (
    AccelBackend,
    FrameSourceKind,
    OpticalLabScenarioConfig,
    ReadbackPayload,
    WritePolicy,
)
from .timing import FrameTimingRecorder, TimingRecorder
from .video_loop import (
    FrameIdentity,
    build_video_render_plan,
    record_delivered_video_frame,
    render_video_frame_from_context,
    video_delivery_request_from_options,
)

DEFAULT_LAB_WARMUP_RENDERS = 5
PhysicsPublishedFrameForIndex = Callable[[int], object]
# Called today as step_physics_frame(frame_index); kept wide for future action/control inputs.
PhysicsPublishedFrameStepper = Callable[..., object]
PhysicsVideoCameraBuilder = Callable[[object, object, int], OpticalPinholeCameraSpec]


class RunScenarioUnsupportedError(NotImplementedError):
    """Raised when a valid lab config cannot be owned by ``run_scenario(...)``."""


@dataclass(frozen=True)
class LabRunOptions:
    """Execution options that are intentionally outside scenario semantics."""

    out: Path
    model_dir: str = "out/external/mujoco_menagerie/unitree_go2"
    model_xml: str = "go2.xml"
    frames: int = 10
    fps: float = 30.0
    warmup_renders: int = DEFAULT_LAB_WARMUP_RENDERS
    progress_every: int = 5
    video_raygen: str = "gpu"
    video_ray_cache: str = "off"
    video_readback_delivery: str = "sync"
    video_readback_ring_depth: int = 2
    render_profile: bool = False
    fail_on_overflow: bool = True
    verbose_warp: bool = False


def validate_scenario(config: OpticalLabScenarioConfig) -> None:
    """Validate lab-wide config support, independent of runner ownership."""
    config.validate_implemented()


def can_run_scenario(config: OpticalLabScenarioConfig) -> bool:
    """Return whether ``run_scenario(...)`` can execute a valid lab config.

    Invalid configs raise their validation error; valid configs that require a
    different runtime owner return ``False``.
    """
    try:
        validate_run_scenario_supported(config)
    except RunScenarioUnsupportedError:
        return False
    return True


def validate_run_scenario_supported(config: OpticalLabScenarioConfig) -> None:
    """Validate that ``run_scenario(...)`` owns enough runtime state."""
    validate_scenario(config)
    if config.frame_source is FrameSourceKind.PHYSICS_RUNTIME:
        raise RunScenarioUnsupportedError(
            "run_scenario(...) cannot construct a physics engine; use "
            "run_physics_video_scenario(...) or run_physics_stepped_video_scenario(...) "
            "with explicit engine/runtime inputs"
        )


def apply_run_overrides(
    config: OpticalLabScenarioConfig,
    *,
    device: str | None = None,
    width: int | None = None,
    height: int | None = None,
    readback: str | None = None,
    shadows: bool | None = None,
    write_frames: bool | None = None,
) -> OpticalLabScenarioConfig:
    """Return a config with CLI-style run overrides applied."""
    changes: dict[str, object] = {}
    if device is not None:
        changes["device"] = device
    if width is not None:
        changes["width"] = int(width)
    if height is not None:
        changes["height"] = int(height)
    if readback is not None:
        changes["readback_payload"] = ReadbackPayload(readback)
        if readback == "full":
            changes["output_profile"] = "direct_light_full"
        elif readback in ("rgb", "rgb8"):
            changes["output_profile"] = "rgb_preview"
        elif readback == "none":
            changes["output_profile"] = "render_only"
    if shadows is not None:
        changes["shadows"] = bool(shadows)
    if write_frames is not None:
        changes["write_policy"] = WritePolicy.PNG_SEQUENCE if write_frames else WritePolicy.NONE
    return replace(config, **changes)


def run_scenario(config: OpticalLabScenarioConfig, options: LabRunOptions) -> None:
    """Run a currently implemented lab scenario.

    Stage C0 delegates the Go2 static video path to the existing Menagerie GPU
    example. The lab owns config validation and output metadata; the production
    render/session boundary is still future work.
    """
    validate_run_scenario_supported(config)
    _validate_run_options(config, options)
    if config.scene_preset not in ("go2_menagerie_static", "synthetic_body_triangle"):
        raise NotImplementedError(
            f"scene_preset={config.scene_preset!r} is reserved; "
            "use go2_menagerie_static/synthetic_body_triangle for now"
        )
    if config.camera_mode not in ("camera_orbit", "fixed_view"):
        raise NotImplementedError(
            f"camera_mode={config.camera_mode!r} is reserved; use camera_orbit/fixed_view for now"
        )

    options.out.mkdir(parents=True, exist_ok=True)
    write_scenario_config(options.out / "scenario_config.json", config, options)

    from .go2_backend import render_many_views

    render_many_views(build_menagerie_example_args(config, options))


def run_physics_video_scenario(
    config: OpticalLabScenarioConfig,
    options: LabRunOptions,
    *,
    engine: object,
    registry: object,
    base_frame: object,
    published_frame_for_index: PhysicsPublishedFrameForIndex,
    build_video_camera: PhysicsVideoCameraBuilder,
    synchronize_event: Callable[[object], None],
    pack_rgb8: Callable[[object], object],
    bounds_min: object | None = None,
    bounds_max: object | None = None,
    metadata: Mapping[str, object] | None = None,
    consumer_id: str = "optical_lab_physics_video",
) -> FrameTimingRecorder:
    """Run the tiny explicit physics-runtime video scenario.

    Physics time ownership stays with `published_frame_for_index`: this helper
    only assembles render/runtime/video/delivery around each published frame.
    """

    validate_physics_video_run(config, options)
    options.out.mkdir(parents=True, exist_ok=True)
    write_scenario_config(options.out / "scenario_config.json", config, options)

    args = build_physics_video_args(config, options)
    timings = TimingRecorder()
    runtime = create_physics_render_runtime_for_config(
        config,
        options,
        engine=engine,
        registry=registry,
        base_frame=base_frame,
        timings=timings,
        consumer_id=consumer_id,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        metadata=metadata,
    )

    from . import frame_contexts

    frame_provider = frame_contexts.physics_frame_context_provider(
        runtime,
        delivery_mode=options.video_readback_delivery,
    )
    delivery_request = video_delivery_request_from_options(
        readback_mode=args.video_readback,
        delivery_mode=args.video_readback_delivery,
        ring_depth=int(args.video_readback_ring_depth),
        write_frames=bool(args.write_frames),
    )
    rows = FrameTimingRecorder(
        csv_path=Path(args.frame_timing_csv),
        default_fields=args.lab_frame_defaults,
    )
    row_builder = VideoFrameTimingRowBuilder(
        VideoDeliveryRunConfig(
            video_fps=float(args.video_fps),
            video_frames=int(args.video_frames),
            video_raygen=args.video_raygen,
            video_ray_cache=args.video_ray_cache,
            delivery_policy_label=args.video_readback_delivery,
            fail_on_overflow=bool(args.fail_on_overflow),
        )
    ).bind_request(delivery_request)
    delivery = VideoDeliveryFacade.create(
        request=delivery_request,
        delivery_policy_label=args.video_readback_delivery,
        frame_dir=options.out / "frames",
        pack_rgb8=pack_rgb8,
        synchronize_event=synchronize_event,
    )

    def consume_video(frame_context, frame_index: int):
        frame_identity = FrameIdentity(
            frame_id=frame_context.frame_id,
            sim_time=frame_context.sim_time,
            env_idx=frame_context.env_idx,
        )
        plan = build_video_render_plan(
            runtime.pipeline.session.scene,
            args,
            frame_index,
            None,
            build_video_camera=build_video_camera,
            frame_identity=frame_identity,
            geometry_mode=config.geometry_mode.value,
        )
        return render_video_frame_from_context(
            frame_context,
            plan,
            frame_index=frame_index,
        )

    def record_delivered(delivered) -> None:
        record_delivered_video_frame(rows, row_builder, delivered, args)

    workflow = FrameWorkflowRunner(
        frame_provider=frame_provider,
        video_consumer=consume_video,
        delivery=delivery,
        delivered_video_recorder=record_delivered,
    )
    for frame_index in range(int(options.frames)):
        published_frame = published_frame_for_index(frame_index)
        workflow.step(
            frame_index,
            env_idx=0,
            provider_kwargs={"published_frame": published_frame},
        )
    workflow.flush()
    rows.write_csv()
    return rows


def run_physics_stepped_video_scenario(
    config: OpticalLabScenarioConfig,
    options: LabRunOptions,
    *,
    engine: object,
    registry: object,
    base_frame: object,
    step_physics_frame: PhysicsPublishedFrameStepper,
    build_video_camera: PhysicsVideoCameraBuilder,
    synchronize_event: Callable[[object], None],
    pack_rgb8: Callable[[object], object],
    bounds_min: object | None = None,
    bounds_max: object | None = None,
    metadata: Mapping[str, object] | None = None,
    consumer_id: str = "optical_lab_physics_video",
) -> FrameTimingRecorder:
    """Run a physics-owned stepped video scenario through the P6 bridge.

    `step_physics_frame(frame_index)` is expected to advance or select physics
    time for that frame and return the published frame. The lower-level
    `run_physics_video_scenario(...)` remains the replay/selection entry that
    only asks for a frame by index.
    """

    def published_frame_for_index(frame_index: int) -> object:
        return step_physics_frame(frame_index)

    return run_physics_video_scenario(
        config,
        options,
        engine=engine,
        registry=registry,
        base_frame=base_frame,
        published_frame_for_index=published_frame_for_index,
        build_video_camera=build_video_camera,
        synchronize_event=synchronize_event,
        pack_rgb8=pack_rgb8,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        metadata=metadata,
        consumer_id=consumer_id,
    )


def build_physics_video_args(
    config: OpticalLabScenarioConfig,
    options: LabRunOptions,
) -> argparse.Namespace:
    """Translate a physics-runtime lab scenario into video runner args."""

    validate_physics_video_run(config, options)
    render_options = render_options_for_config(config, options)
    return argparse.Namespace(
        scene_preset=config.scene_preset,
        device=render_options.device,
        width=int(config.width),
        height=int(config.height),
        out=str(options.out),
        no_shadows=not render_options.shadows,
        bvh_backend=render_options.bvh_backend,
        bvh_split_strategy=render_options.bvh_split_strategy,
        fail_on_overflow=bool(options.fail_on_overflow),
        video_frames=int(options.frames),
        video_fps=float(options.fps),
        video_mode=config.camera_mode,
        video_ray_cache=options.video_ray_cache,
        video_raygen=options.video_raygen,
        video_readback=config.readback_payload.value,
        video_readback_delivery=options.video_readback_delivery,
        video_readback_ring_depth=int(options.video_readback_ring_depth),
        video_geometry_mode=config.geometry_mode.value,
        frame_timing_csv=str(options.out / "frame_timing.csv"),
        progress_every=int(options.progress_every),
        render_profile=bool(options.render_profile),
        write_frames=config.write_policy is WritePolicy.PNG_SEQUENCE,
        lab_frame_defaults=frame_defaults_for_config(config),
    )


def build_menagerie_example_args(
    config: OpticalLabScenarioConfig,
    options: LabRunOptions,
) -> argparse.Namespace:
    """Translate a lab scenario into the transitional Menagerie example args."""
    validate_run(config, options)
    if config.frame_source is FrameSourceKind.PHYSICS_RUNTIME:
        raise NotImplementedError(
            "build_menagerie_example_args(...) is for static/synthetic transitional paths; "
            "use build_physics_video_args(...) for frame_source='physics_runtime'"
        )
    render_options = render_options_for_config(config, options)
    return argparse.Namespace(
        model_dir=options.model_dir,
        model_xml=options.model_xml,
        scene_preset=config.scene_preset,
        device=render_options.device,
        width=int(config.width),
        height=int(config.height),
        view="front",
        views=["front"],
        out=str(options.out),
        no_shadows=not render_options.shadows,
        bvh_backend=render_options.bvh_backend,
        bvh_split_strategy=render_options.bvh_split_strategy,
        fail_on_overflow=bool(options.fail_on_overflow),
        timing_csv=str(options.out / "timing.csv"),
        render_warmup=0,
        warmup_renders=int(options.warmup_renders),
        render_repeat=0,
        setup_warmup=0,
        setup_repeat=0,
        video_frames=int(options.frames),
        video_fps=float(options.fps),
        video_mode=config.camera_mode,
        video_ray_cache=options.video_ray_cache,
        video_raygen=options.video_raygen,
        video_readback=config.readback_payload.value,
        video_readback_delivery=options.video_readback_delivery,
        video_readback_ring_depth=int(options.video_readback_ring_depth),
        video_geometry_mode=config.geometry_mode.value,
        frame_timing_csv=str(options.out / "frame_timing.csv"),
        progress_every=int(options.progress_every),
        render_profile=bool(options.render_profile),
        write_frames=config.write_policy is WritePolicy.PNG_SEQUENCE,
        verbose_warp=render_options.verbose_warp,
        lab_frame_defaults=frame_defaults_for_config(config),
    )


def render_options_for_config(
    config: OpticalLabScenarioConfig,
    options: LabRunOptions,
) -> OpticalLabRenderOptions:
    """Return render-session options derived from lab scenario/run config.

    This helper intentionally maps render-session intent only. Scenario
    executability stays with validate_run/run_scenario so reserved frame
    sources can still reuse the mapping when their runner path lands.
    """
    return OpticalLabRenderOptions(
        device=config.device,
        bvh_backend=_example_bvh_backend(config.accel_backend),
        bvh_split_strategy="sort",
        shadows=config.shadows,
        verbose_warp=options.verbose_warp,
    )


def create_physics_render_runtime_for_config(
    config: OpticalLabScenarioConfig,
    options: LabRunOptions,
    *,
    engine: object,
    registry: object,
    base_frame: object,
    timings: TimingRecorder,
    consumer_id: str = "optical_lab_physics_render",
    consumer: object | None = None,
    qos_mode: object = "lossless",
    max_lag_frames: int | None = None,
    bounds_min: object | None = None,
    bounds_max: object | None = None,
    scene: object | None = None,
    metadata: Mapping[str, object] | None = None,
) -> object:
    """Create a physics-backed render runtime from lab config.

    This is assembly-only: it derives render-session options from config and
    delegates to physics_source without enabling run_scenario's physics loop.
    """

    if config.frame_source is not FrameSourceKind.PHYSICS_RUNTIME:
        raise ValueError("create_physics_render_runtime_for_config requires frame_source='physics_runtime'")

    from . import physics_source

    return physics_source.create_physics_render_runtime(
        engine=engine,
        registry=registry,
        base_frame=base_frame,
        options=render_options_for_config(config, options),
        timings=timings,
        consumer_id=consumer_id,
        consumer=consumer,
        qos_mode=qos_mode,
        max_lag_frames=max_lag_frames,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        scene=scene,
        metadata=metadata,
    )


def write_scenario_config(path: Path, config: OpticalLabScenarioConfig, options: LabRunOptions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario": scenario_config_dict(config),
        "run_options": run_options_dict(options),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def frame_defaults_for_config(config: OpticalLabScenarioConfig) -> dict[str, str | int]:
    """Return per-frame CSV defaults derived from a scenario config."""
    return {
        "scenario_name": config.scenario_name,
        "device": config.device,
        "width": int(config.width),
        "height": int(config.height),
        "scene_preset": config.scene_preset,
        "frame_source": config.frame_source.value,
        "camera_mode": config.camera_mode,
        "geometry_mode": config.geometry_mode.value,
        "accel_backend": config.accel_backend.value,
        "accel_policy": config.accel_policy.value,
        "render_backend": config.render_backend.value,
        "output_profile": config.output_profile,
        "readback_payload": config.readback_payload.value,
        "delivery_policy": config.delivery_policy.value,
        "write_policy": config.write_policy.value,
    }


def validate_run(config: OpticalLabScenarioConfig, options: LabRunOptions) -> None:
    """Validate a concrete lab run before any GPU work starts."""
    validate_scenario(config)
    _validate_run_options(config, options)


def _validate_run_options(config: OpticalLabScenarioConfig, options: LabRunOptions) -> None:
    """Validate run options after the scenario config has been accepted."""
    if options.frames < 0:
        raise ValueError("frames must be >= 0")
    if options.fps <= 0.0:
        raise ValueError("fps must be > 0")
    if options.progress_every < 0:
        raise ValueError("progress_every must be >= 0")
    if options.video_raygen == "gpu" and options.video_ray_cache != "off":
        raise ValueError("video_raygen='gpu' computes camera rays on device; use video_ray_cache='off'")
    if options.video_readback_delivery not in ("sync", "torch_async"):
        raise ValueError("video_readback_delivery must be 'sync' or 'torch_async'")
    if options.video_readback_ring_depth <= 0:
        raise ValueError("video_readback_ring_depth must be > 0")
    if options.video_readback_delivery == "torch_async" and config.readback_payload not in (
        ReadbackPayload.RGB,
        ReadbackPayload.RGB8,
    ):
        raise ValueError(
            "video_readback_delivery='torch_async' currently requires readback_payload='rgb' or 'rgb8'"
        )
    if config.readback_payload is ReadbackPayload.NONE and config.write_policy is WritePolicy.PNG_SEQUENCE:
        raise ValueError("readback_payload='none' cannot be combined with write_policy='png_sequence'")
    if config.readback_payload is ReadbackPayload.NONE and options.fail_on_overflow:
        raise ValueError("readback_payload='none' cannot honor fail_on_overflow")


def validate_physics_video_run(config: OpticalLabScenarioConfig, options: LabRunOptions) -> None:
    """Validate the explicit physics-runtime video runner path."""

    validate_run(config, options)
    if config.frame_source is not FrameSourceKind.PHYSICS_RUNTIME:
        raise ValueError("run_physics_video_scenario requires frame_source='physics_runtime'")
    if config.scene_preset != "synthetic_body_triangle":
        raise NotImplementedError(
            f"scene_preset={config.scene_preset!r} is reserved for physics runtime; "
            "use synthetic_body_triangle for now"
        )
    if config.camera_mode != "fixed_view":
        raise NotImplementedError(
            f"camera_mode={config.camera_mode!r} is reserved for physics runtime; use fixed_view for now"
        )
    if options.video_readback_delivery == "torch_async":
        raise NotImplementedError(
            "physics runtime video requires provider-backed torch_async warmup before torch_async delivery"
        )


def scenario_config_dict(config: OpticalLabScenarioConfig) -> dict[str, object]:
    return {field: _serialize_value(value) for field, value in config.__dict__.items()}


def run_options_dict(options: LabRunOptions) -> dict[str, object]:
    return {field: _serialize_value(value) for field, value in options.__dict__.items()}


def _serialize_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


def _example_bvh_backend(accel_backend: AccelBackend) -> str:
    if accel_backend is AccelBackend.CUDA_LBVH:
        return "cuda_lbvh"
    if accel_backend is AccelBackend.CPU_BVH:
        return "cpu"
    raise NotImplementedError(
        f"accel_backend={accel_backend.value!r} is reserved; use cpu_bvh/cuda_lbvh for now"
    )
