"""Physics-published frame source helpers for the Optical Pipeline Lab."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from physics.publish import GpuPublishedFrame

from .render_session import OpticalLabRenderSource


@dataclass(frozen=True)
class PhysicsLabRenderScene:
    """Small scene view for physics-driven lab render sessions."""

    registry: object
    frame: GpuPublishedFrame
    bounds_min: object | None = None
    bounds_max: object | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


def build_physics_render_source(
    *,
    registry: object,
    base_frame: GpuPublishedFrame,
    bounds_min: object | None = None,
    bounds_max: object | None = None,
    scene: object | None = None,
    metadata: Mapping[str, object] | None = None,
) -> OpticalLabRenderSource:
    """Wrap a physics-published GPU frame as a lab render source.

    The caller owns the published-frame lifetime. Real-time/lossless callers
    should borrow the frame from the physics publish ring before passing it in.
    """

    source_metadata = dict(metadata or {})
    if scene is None:
        scene = PhysicsLabRenderScene(
            registry=registry,
            frame=base_frame,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            metadata=source_metadata,
        )
    source_metadata.setdefault("scene", scene)
    source_metadata.setdefault("source_kind", "physics")
    return OpticalLabRenderSource(
        registry=registry,
        base_frame=base_frame,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        metadata=source_metadata,
    )


def scene_from_physics_render_source(source: OpticalLabRenderSource) -> object:
    """Return the scene view stored by `build_physics_render_source`."""

    return source.metadata.get("scene", source)
