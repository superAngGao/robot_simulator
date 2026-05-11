import numpy as np
import pytest

from optics.execution import OpticalOutputProfile
from optics.render_api import (
    DeliveryPolicy,
    DeliveryRequest,
    OpticalRenderPipeline,
    ReadbackPayload,
    RenderBackend,
    RenderDiagnosticsRequest,
    RenderFrameContext,
    RenderRequest,
    WritePolicy,
)
from sensing.optical import OpticalPinholeCameraSpec, OpticalRaySensorSpec


def _camera() -> OpticalPinholeCameraSpec:
    return OpticalPinholeCameraSpec(
        frame_id=7,
        sim_time=0.25,
        env_idx=2,
        sensor_id="cam",
        width=16,
        height=8,
        fx=10.0,
        fy=10.0,
        cx=7.5,
        cy=3.5,
        max_distance=10.0,
        sensor_role="rgb",
    )


def _rays() -> OpticalRaySensorSpec:
    return OpticalRaySensorSpec(
        frame_id=7,
        sim_time=0.25,
        env_idx=2,
        sensor_id="rays",
        origins_world=np.zeros((1, 3), dtype=np.float64),
        directions_world=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        max_distance=10.0,
        sensor_role="rgb",
    )


def test_render_request_accepts_camera_and_normalizes_enums():
    request = RenderRequest(
        frame_id=7,
        sim_time=0.25,
        env_idx=2,
        camera=_camera(),
        backend="direct_light",
        output_profile="rgb_preview",
        accumulation_id=3,
    )

    assert request.backend is RenderBackend.DIRECT_LIGHT
    assert request.output_profile is OpticalOutputProfile.RGB_PREVIEW
    assert request.use_gpu_raygen is True
    assert request.accumulation_id == 3


def test_render_request_accepts_rays_when_gpu_raygen_disabled():
    request = RenderRequest(
        frame_id=7,
        sim_time=0.25,
        env_idx=2,
        rays=_rays(),
        use_gpu_raygen=False,
    )

    assert request.rays is not None
    assert request.camera is None


def test_render_request_rejects_ambiguous_sources():
    with pytest.raises(ValueError, match="exactly one"):
        RenderRequest(frame_id=7, sim_time=0.25, env_idx=2)

    with pytest.raises(ValueError, match="exactly one"):
        RenderRequest(frame_id=7, sim_time=0.25, env_idx=2, camera=_camera(), rays=_rays())


def test_render_request_rejects_inconsistent_source_metadata():
    with pytest.raises(ValueError, match="frame_id"):
        RenderRequest(frame_id=8, sim_time=0.25, env_idx=2, camera=_camera())

    with pytest.raises(ValueError, match="use_gpu_raygen"):
        RenderRequest(frame_id=7, sim_time=0.25, env_idx=2, rays=_rays())


def test_render_diagnostics_request_defaults_are_stable():
    diagnostics = RenderDiagnosticsRequest()

    assert diagnostics.profile_timing is False
    assert diagnostics.traversal_counters is False
    assert diagnostics.fail_on_overflow is True


def test_delivery_request_validates_payload_policy_combinations():
    async_request = DeliveryRequest(
        payload=ReadbackPayload.RGB8,
        policy=DeliveryPolicy.TORCH_ASYNC_ORDERED,
        ring_depth=2,
        write_policy=WritePolicy.NONE,
    )
    assert async_request.payload is ReadbackPayload.RGB8
    assert async_request.policy is DeliveryPolicy.TORCH_ASYNC_ORDERED
    assert async_request.ring_depth == 2

    with pytest.raises(ValueError, match="ring_depth"):
        DeliveryRequest(payload=ReadbackPayload.RGB8, policy=DeliveryPolicy.TORCH_ASYNC_ORDERED, ring_depth=0)

    with pytest.raises(ValueError, match="DEVICE_ONLY"):
        DeliveryRequest(payload=ReadbackPayload.RGB8, policy=DeliveryPolicy.DEVICE_ONLY)

    with pytest.raises(ValueError, match="payload=NONE"):
        DeliveryRequest(payload=ReadbackPayload.NONE, policy=DeliveryPolicy.SYNC_HOST)

    with pytest.raises(ValueError, match="RGB or RGB8"):
        DeliveryRequest(payload=ReadbackPayload.FULL, policy=DeliveryPolicy.TORCH_ASYNC_ORDERED)


def test_pipeline_protocols_are_internal_and_import_safe():
    import optics

    assert RenderFrameContext.__name__ == "RenderFrameContext"
    assert OpticalRenderPipeline.__name__ == "OpticalRenderPipeline"
    assert not hasattr(optics, "RenderFrameContext")
    assert not hasattr(optics, "OpticalRenderPipeline")
