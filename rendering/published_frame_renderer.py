"""Thin helpers for rendering directly from published physics frames."""

from __future__ import annotations

from .backends.base import RenderBackend
from .scene_builder import build_render_scene_from_published_frame


def render_published_frame(
    engine,
    backend: RenderBackend,
    frame=None,
    env_idx: int = 0,
    include_contacts: bool = True,
):
    """Build a RenderScene from a published frame and send it to a backend."""
    if frame is None:
        frame = engine.latest_published_frame()
    if frame is None:
        raise RuntimeError("No published frame is available yet.")
    scene = build_render_scene_from_published_frame(
        engine,
        frame=frame,
        env_idx=env_idx,
        include_contacts=include_contacts,
    )
    backend.render_frame(scene, timestamp=frame.sim_time, env_index=env_idx)
    return scene


def render_latest_published_frame(
    engine,
    backend: RenderBackend,
    env_idx: int = 0,
    include_contacts: bool = True,
):
    """Convenience wrapper around `render_published_frame(..., frame=None)`."""
    return render_published_frame(
        engine,
        backend,
        frame=None,
        env_idx=env_idx,
        include_contacts=include_contacts,
    )
