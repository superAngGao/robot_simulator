"""Runner entry points for the Optical Pipeline Lab."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

from .scenarios import (
    AccelBackend,
    OpticalLabScenarioConfig,
    ReadbackPayload,
    WritePolicy,
)

DEFAULT_LAB_WARMUP_RENDERS = 5


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
    """Validate that a config uses only currently implemented lab modes."""
    config.validate_implemented()


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
    validate_run(config, options)
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


def build_menagerie_example_args(
    config: OpticalLabScenarioConfig,
    options: LabRunOptions,
) -> argparse.Namespace:
    """Translate a lab scenario into the transitional Menagerie example args."""
    validate_run(config, options)
    return argparse.Namespace(
        model_dir=options.model_dir,
        model_xml=options.model_xml,
        scene_preset=config.scene_preset,
        device=config.device,
        width=int(config.width),
        height=int(config.height),
        view="front",
        views=["front"],
        out=str(options.out),
        no_shadows=not config.shadows,
        bvh_backend=_example_bvh_backend(config.accel_backend),
        bvh_split_strategy="sort",
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
        verbose_warp=bool(options.verbose_warp),
        lab_frame_defaults=frame_defaults_for_config(config),
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
