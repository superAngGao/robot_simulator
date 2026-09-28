"""Host scalar/evidence contracts do not confer recorded execution authority."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from optics.device_scene import DeviceOpticalRoleTable
from optics.warp_native_render import (
    WARP_RENDER_ABI,
    WARP_RENDER_COMMAND_SCHEMA,
    WarpCameraParameters,
    WarpDirectLightRenderCommandEvidence,
    _render_command_ids,
)
from physics.spatial import SpatialTransform
from sensing.optical import OpticalPinholeCameraSpec


def _camera():
    spec = OpticalPinholeCameraSpec(
        frame_id=3,
        sim_time=0.25,
        env_idx=0,
        sensor_id="camera",
        width=7,
        height=3,
        fx=3.3,
        fy=2.7,
        cx=2.1,
        cy=0.7,
        X_world_camera=SpatialTransform(
            R=np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            r=np.array([0.1, 0.2, 1.3]),
        ),
        max_distance=100.0,
        sensor_role="rgb",
    )
    return spec, WarpCameraParameters.from_spec(
        spec,
        role_table=DeviceOpticalRoleTable.from_roles(("rgb", "depth")),
    )


def test_camera_detaches_scalar_content_from_mutable_host_spec():
    spec, camera = _camera()
    original = camera.sha256
    assert camera.query.num_rays == 21
    assert camera.query.sensor_role_mask == 2
    assert camera.fx == float(np.float32(spec.fx))
    assert camera.rotation == (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    spec.X_world_camera.r[:] = 99
    spec.X_world_camera.R[:] = 99
    spec.fx = 42
    assert camera.sha256 == original
    with pytest.raises(FrozenInstanceError):
        camera.fx = 4.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("fx", 0.0),
        ("fy", -1.0),
        ("cx", float("nan")),
        ("cy", float("inf")),
        ("fx", 3.3),
        ("width", True),
        ("height", 0),
        ("width", 8),
        ("origin", [0.0, 0.0, 0.0]),
        ("rotation", (0.0,) * 8),
    ],
)
def test_camera_rejects_noncanonical_or_inconsistent_scalars(field, value):
    _, camera = _camera()
    with pytest.raises((ValueError, TypeError)):
        replace(camera, **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_rays", -1),
        ("num_rays", 2147483648),
        ("frame_id", True),
        ("env_idx", -1),
        ("sim_time", float("nan")),
        ("sensor_id", ""),
        ("max_distance", float("inf")),
        ("max_distance", 1e100),
        ("sensor_role_mask", 1 << 63),
        ("sensor_role_mask", True),
    ],
)
def test_query_boundary_fields_fail_closed(field, value):
    _, camera = _camera()
    with pytest.raises((ValueError, TypeError)):
        replace(camera.query, **{field: value})
    assert replace(camera.query, num_rays=0).num_rays == 0
    assert replace(camera.query, num_rays=2147483647).num_rays == 2147483647


def _evidence(ray_source="camera", num_rays=21, shadows=True, diagnostics=True, replay=False):
    entries = _render_command_ids(ray_source, num_rays, shadows, diagnostics)
    return WarpDirectLightRenderCommandEvidence(
        schema_version=WARP_RENDER_COMMAND_SCHEMA,
        operation_id="warp_bvh_direct_light",
        kernel_abi_id=WARP_RENDER_ABI,
        kernel_source_sha256="1" * 64,
        device_ordinal=0,
        stream_handle=17,
        native_input_manifest_sha256="2" * 64,
        readiness_sha256="3" * 64,
        scene_contract_sha256="4" * 64,
        input_contract="canonical_native_replay_v1" if replay else "fixed_native_scene_v1",
        fixed_render_fixture_sha256=None if replay else "5" * 64,
        transform_replay_sha256="6" * 64 if replay else None,
        transform_frame_index=0 if replay else None,
        query_sha256="7" * 64,
        camera_contract_sha256="8" * 64 if ray_source == "camera" else None,
        ray_input_storage_sha256=None if ray_source == "camera" else "8" * 64,
        ray_source=ray_source,
        num_rays=num_rays,
        output_profile="direct_light_full",
        shadows=shadows,
        traversal_diagnostics=diagnostics,
        input_storage_sha256="9" * 64,
        output_storage_sha256="a" * 64,
        command_entry_ids=entries,
        command_plan_sha256="b" * 64,
        dispatch_count=len(entries),
    )


@pytest.mark.parametrize("ray_source,num_rays", [("camera", 21), ("device_rays", 21), ("device_rays", 0)])
@pytest.mark.parametrize("shadows,diagnostics", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("replay", [False, True])
def test_command_evidence_round_trip_and_total_order(ray_source, num_rays, shadows, diagnostics, replay):
    evidence = _evidence(ray_source, num_rays, shadows, diagnostics, replay)
    assert WarpDirectLightRenderCommandEvidence.from_payload(evidence.to_payload()) == evidence
    assert len(evidence.sha256) == 64
    assert evidence.dispatch_count == (5 if ray_source == "camera" else 4 if num_rays else 2)
    if not num_rays:
        assert all("reset" in name for name in evidence.command_entry_ids)


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_contract", "gpu_engine_published_frame_v1"),
        ("transform_replay_sha256", "0" * 64),
        ("transform_frame_index", 0),
        ("fixed_render_fixture_sha256", None),
        ("camera_contract_sha256", None),
        ("ray_input_storage_sha256", "0" * 64),
        ("command_entry_ids", []),
        ("dispatch_count", 99),
        ("num_rays", 0),
        ("shadows", 1),
        ("output_profile", "geometry_full"),
        ("stream_handle", 0),
        ("device_ordinal", True),
        ("kernel_source_sha256", "A" * 64),
    ],
)
def test_rehashed_evidence_cannot_hide_invalid_local_facts(field, value):
    payload = _evidence().to_payload()
    payload[field] = value
    with pytest.raises((ValueError, TypeError)):
        WarpDirectLightRenderCommandEvidence.from_payload(payload)


def test_evidence_rejects_missing_extra_outer_or_self_references():
    original = _evidence().to_payload()
    for field in original:
        with pytest.raises(ValueError):
            WarpDirectLightRenderCommandEvidence.from_payload(
                {k: v for k, v in original.items() if k != field}
            )
    for field in ("sha256", "binding_sha256", "completion_ticket", "arena_id"):
        with pytest.raises(ValueError):
            WarpDirectLightRenderCommandEvidence.from_payload({**original, field: "0" * 64})
    for fields in (
        {"command_entry_ids": list(reversed(original["command_entry_ids"]))},
        {"command_entry_ids": original["command_entry_ids"][:4], "dispatch_count": 4},
    ):
        with pytest.raises(ValueError):
            WarpDirectLightRenderCommandEvidence.from_payload({**original, **fields})
