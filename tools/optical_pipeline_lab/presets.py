"""Preset registry for Optical Pipeline Lab scenarios."""

from __future__ import annotations

from .scenarios import (
    AccelBackend,
    AccelPolicy,
    DeliveryPolicy,
    GeometryMode,
    OpticalLabScenarioConfig,
    OpticalLabScenarioFamily,
    ReadbackPayload,
    RenderBackend,
    WritePolicy,
)


def go2_video_ordered_static_preset() -> OpticalLabScenarioConfig:
    """Return the current Go2 static video ordered export baseline preset."""
    return OpticalLabScenarioConfig(
        scenario_name="go2_video_ordered_static",
        scenario_family=OpticalLabScenarioFamily.VIDEO_ORDERED_EXPORT,
        scene_preset="go2_menagerie_static",
        geometry_mode=GeometryMode.STATIC,
        camera_mode="camera_orbit",
        accel_backend=AccelBackend.CUDA_LBVH,
        accel_policy=AccelPolicy.BUILD_ONCE,
        render_backend=RenderBackend.WARP_BVH_DIRECT_LIGHT,
        output_profile="rgb_preview",
        readback_payload=ReadbackPayload.RGB,
        delivery_policy=DeliveryPolicy.SYNC,
        write_policy=WritePolicy.NONE,
        diagnostics_policy="required",
        shadows=True,
    )


PRESETS = {
    "go2_video_ordered_static": go2_video_ordered_static_preset,
}


def get_preset(name: str) -> OpticalLabScenarioConfig:
    try:
        return PRESETS[name]()
    except KeyError as exc:
        choices = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown optical pipeline lab preset {name!r}; expected one of: {choices}") from exc
