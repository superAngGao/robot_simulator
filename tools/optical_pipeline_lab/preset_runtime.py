"""Preset runtime factories for Optical Pipeline Lab workflows."""

from __future__ import annotations

from collections.abc import Callable

from .physics_runtime import (
    PhysicsLabScenarioRuntime,
    create_physics_body_triangle_lab_runtime,
)

PHYSICS_BODY_TRIANGLE_VIDEO_SMOKE_PRESET = "physics_body_triangle_video_smoke"


def _create_physics_body_triangle_video_smoke_runtime(
    **runtime_kwargs: object,
) -> PhysicsLabScenarioRuntime:
    return create_physics_body_triangle_lab_runtime(**runtime_kwargs)


_RUNTIME_FACTORIES: dict[str, Callable[..., PhysicsLabScenarioRuntime]] = {
    PHYSICS_BODY_TRIANGLE_VIDEO_SMOKE_PRESET: _create_physics_body_triangle_video_smoke_runtime,
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


def supported_runtime_presets() -> tuple[str, ...]:
    """Return lab presets with registered runtime factories."""

    return tuple(sorted(_RUNTIME_FACTORIES))
