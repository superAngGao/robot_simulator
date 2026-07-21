"""Preset runtime factories for Optical Pipeline Lab workflows."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from .frame_tick import SimulationFrameTick
from .physics_runtime import (
    PhysicsLabScenarioRuntime,
    create_physics_body_triangle_lab_runtime,
)
from .render_session import OpticalLabRenderPipeline
from .runner import ArtifactOutput, render_options_for_config
from .timing import TimingRecorder

PHYSICS_BODY_TRIANGLE_VIDEO_SMOKE_PRESET = "physics_body_triangle_video_smoke"
GO2_VIDEO_ORDERED_STATIC_PRESET = "go2_video_ordered_static"


@dataclass
class StaticAssetLabRuntime:
    """Static asset runtime owner for product workflow ticks."""

    pipeline: object
    scene: object
    base_frame: object
    metadata: Mapping[str, object]
    fps: float
    closed: bool = False

    def step_tick(self, frame_index: int, *, env_idx: int = 0) -> SimulationFrameTick:
        frame_count = int(frame_index)
        sim_dt = 1.0 / self.fps if self.fps > 0.0 else 0.0
        return SimulationFrameTick(
            frame_index=frame_count,
            env_idx=int(env_idx),
            frame_id=int(getattr(self.base_frame, "frame_id")) + frame_count,
            sim_time=float(getattr(self.base_frame, "sim_time")) + sim_dt * float(frame_count),
            published_frame=self.base_frame,
            metadata={
                **dict(self.metadata),
                "frame_source": "static_asset_builder",
            },
        )

    def close(self) -> None:
        self.closed = True


def _create_physics_body_triangle_video_smoke_runtime(
    **runtime_kwargs: object,
) -> PhysicsLabScenarioRuntime:
    return create_physics_body_triangle_lab_runtime(**runtime_kwargs)


_RUNTIME_FACTORIES: dict[str, Callable[..., PhysicsLabScenarioRuntime]] = {
    PHYSICS_BODY_TRIANGLE_VIDEO_SMOKE_PRESET: _create_physics_body_triangle_video_smoke_runtime,
}

_STATIC_ASSET_PRESET_DEFAULTS: dict[str, Mapping[str, str]] = {
    GO2_VIDEO_ORDERED_STATIC_PRESET: {
        "scene_preset": "go2_menagerie_static",
        "model_dir": "out/external/mujoco_menagerie/unitree_go2",
        "model_xml": "go2.xml",
    },
}


def create_runtime_for_lab_preset(
    preset: str,
    *,
    device: str | None = None,
    **runtime_kwargs: object,
) -> PhysicsLabScenarioRuntime:
    """Create a live physics runtime for a reviewed lab preset."""

    try:
        factory = _RUNTIME_FACTORIES[str(preset)]
    except KeyError as exc:
        choices = ", ".join(sorted(_RUNTIME_FACTORIES))
        raise NotImplementedError(
            f"Optical Lab runtime factory is not registered for preset {preset!r}; "
            f"supported presets: {choices}"
        ) from exc

    kwargs = dict(runtime_kwargs)
    if device is not None:
        kwargs["device"] = device
    return factory(**kwargs)


def create_runtime_for_lab_workflow(
    preset: str,
    *,
    output: ArtifactOutput,
    device: str | None = None,
    runtime_kwargs: Mapping[str, object] | None = None,
) -> object:
    """Create a runtime for a reviewed lab workflow preset."""

    preset_name = str(preset)
    if preset_name in _RUNTIME_FACTORIES:
        kwargs = dict(runtime_kwargs or {})
        return create_runtime_for_lab_preset(
            preset_name,
            device=device,
            **kwargs,
        )
    if preset_name in _STATIC_ASSET_PRESET_DEFAULTS:
        return create_static_asset_lab_runtime(
            preset_name,
            output=output,
            device=device,
            runtime_kwargs=runtime_kwargs,
        )
    choices = ", ".join(sorted((*_RUNTIME_FACTORIES, *_STATIC_ASSET_PRESET_DEFAULTS)))
    raise NotImplementedError(
        f"Optical Lab workflow runtime is not registered for preset {preset!r}; supported presets: {choices}"
    )


def create_static_asset_lab_runtime(
    preset: str,
    *,
    output: ArtifactOutput,
    device: str | None = None,
    runtime_kwargs: Mapping[str, object] | None = None,
) -> StaticAssetLabRuntime:
    """Create a static asset runtime for a reviewed lab workflow preset."""

    preset_name = str(preset)
    try:
        defaults = dict(_STATIC_ASSET_PRESET_DEFAULTS[preset_name])
    except KeyError as exc:
        choices = ", ".join(sorted(_STATIC_ASSET_PRESET_DEFAULTS))
        raise NotImplementedError(
            f"Optical Lab static asset runtime is not registered for preset {preset!r}; "
            f"supported static presets: {choices}"
        ) from exc

    options = defaults | dict(runtime_kwargs or {})
    from . import static_asset_source
    from .presets import get_preset

    config = get_preset(preset_name)
    if device is not None:
        config = replace(config, device=device)
    args = argparse.Namespace(
        scene_preset=options["scene_preset"],
        model_dir=options["model_dir"],
        model_xml=options["model_xml"],
    )
    timings = TimingRecorder()
    pipeline = OpticalLabRenderPipeline.create_from_source_factory(
        lambda workspace: static_asset_source.build_static_asset_render_source(
            args,
            workspace=workspace,
        ),
        render_options_for_config(config, output),
        timings,
        scene_for_source=static_asset_source.scene_from_static_asset_render_source,
    )
    scene = pipeline.session.scene
    base_frame = pipeline.session.gpu_frame
    return StaticAssetLabRuntime(
        pipeline=pipeline,
        scene=scene,
        base_frame=base_frame,
        metadata={
            "runtime_owner": "static_asset_lab_runtime",
            "preset": preset_name,
            "scene_preset": str(options["scene_preset"]),
            "model_dir": str(options["model_dir"]),
            "model_xml": str(options["model_xml"]),
        },
        fps=float(output.fps),
    )


def supported_runtime_presets() -> tuple[str, ...]:
    """Return lab presets with registered runtime factories."""

    return tuple(sorted(_RUNTIME_FACTORIES))
