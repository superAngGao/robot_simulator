"""Reviewed video product factories for Optical Pipeline Lab presets."""

from __future__ import annotations

from typing import Any

from .camera_presets import build_lab_video_camera
from .product_specs import VideoProductSpec, build_static_video_product_from_spec
from .video_loop import pack_video_rgb8


def create_physics_body_triangle_video_product_spec() -> VideoProductSpec:
    """Create the reviewed video product spec for the physics body-triangle preset."""

    return VideoProductSpec(
        build_video_camera=build_lab_video_camera,
        synchronize_event=synchronize_ready_event,
        pack_rgb8=pack_video_rgb8,
    )


def create_go2_video_ordered_static_product_spec() -> VideoProductSpec:
    """Create the reviewed video product spec for the Go2 static preset."""

    return VideoProductSpec(
        build_video_camera=build_lab_video_camera,
        synchronize_event=synchronize_ready_event,
        pack_rgb8=pack_video_rgb8,
        product_builder=build_static_video_product_from_spec,
        consumer_id="optical_lab_static_video_product",
    )


def synchronize_ready_event(event: Any) -> None:
    """Synchronize a ready event without making module import depend on Warp."""

    if event is None:
        return
    try:
        import warp as wp
    except Exception:
        return
    synchronize_event = getattr(wp, "synchronize_event", None)
    if synchronize_event is not None:
        synchronize_event(event)
