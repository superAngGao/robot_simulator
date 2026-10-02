"""CPU sequence/source/ABI data; deliberately no GPU or result authority."""

import json
import math
import struct
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from optical_worker.camera_contract import resolve_frozen_camera_contract
from optical_worker.cases import canonical_json_bytes
from optical_worker.render_parameters import (
    BoundRenderParameters,
    NATIVE_SCALAR_ORDER,
    ResolvedRenderSequence,
    bind_render_parameters,
    materialize_render_source,
)
from optics.camera_presets import build_model_bounds_camera

ASYM = "asymmetric_synthetic_7x3_v1"
DYNAMIC = "synthetic_dynamic_rigid_960x640_v1"


@pytest.fixture(scope="module")
def source():
    return materialize_render_source(ASYM)


def test_asymmetric_reuses_exact_reviewed_camera_and_same_scene(source):
    payload = source.sequence.to_payload()
    frame = payload["frames"][0]
    camera = resolve_frozen_camera_contract(ASYM).to_payload()
    assert frame["intrinsics_f64"] == camera["intrinsics_f64"]
    assert frame["origin_f64"] == camera["origin_world_f64"]
    assert frame["rotation_f64"] == camera["rotation_world_from_camera_f64"]
    assert payload["scene_contract_sha256"] == source.packed_scene.scene_contract_sha256
    assert len(payload["frames"]) == 1
    with pytest.raises(FrozenInstanceError):
        source.sequence.canonical_bytes = b"{}"
    payload["frames"][0]["width"] = 100
    assert source.sequence.to_payload()["frames"][0]["width"] == 7


def test_dynamic_bounds_include_instance_translation_and_never_follow_z_cycle():
    source = materialize_render_source(DYNAMIC)
    payload = source.sequence.to_payload()
    bounds = np.array([[float.fromhex(v) for v in bound] for bound in payload["bounds_f64"]])
    # The actual packed triangle uses float32 .2 and the instance shifts it by .25.
    np.testing.assert_array_equal(
        bounds, [[0.0, 0.0, 0.25], [float(np.float32(0.2)), float(np.float32(0.2)), 0.25]]
    )
    expected = build_model_bounds_camera(
        bounds_min=bounds[0],
        bounds_max=bounds[1],
        width=960,
        height=640,
        frame_id=1,
        sim_time=0.0,
        view="front",
    )
    for index, frame in enumerate(payload["frames"]):
        assert frame["camera_index"] == index
        assert frame["origin_f64"] == [float(v).hex() for v in expected.X_world_camera.r]
        assert frame["rotation_f64"] == [float(v).hex() for v in expected.X_world_camera.R.reshape(-1)]
    assert len(payload["frames"]) == 120


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(unexpected=True),
        lambda p: p.update(scene_contract_sha256="0" * 64),
        lambda p: p.update(case_sha256="0" * 64),
        lambda p: p.update(bounds_f64=[["0x0.0p+0"] * 3]),
        lambda p: p["bounds_f64"].reverse(),
        lambda p: p.update(frames=[]),
        lambda p: p["frames"].append(dict(p["frames"][0])),
        lambda p: p["frames"][0].update(camera_index=True),
        lambda p: p["frames"][0].update(width=7.0),
        lambda p: p["frames"][0].update(sensor_role="depth"),
        lambda p: p["frames"][0].update(rotation_f64=["0x0.0p+0"] * 8),
        lambda p: p["frames"][0]["intrinsics_f64"].__setitem__(0, "nan"),
        lambda p: p["frames"][0]["intrinsics_f64"].__setitem__(0, "0x1p+0"),
        lambda p: p["frames"][0]["intrinsics_f64"].__setitem__(0, 5.25),
        lambda p: p.update(shadow_bias_f64="-0x1.0000000000000p+0"),
    ],
)
def test_malformed_sequence_is_rejected(source, change):
    payload = source.sequence.to_payload()
    change(payload)
    with pytest.raises((ValueError, TypeError)):
        ResolvedRenderSequence.from_payload(payload)


def test_noncanonical_bytes_duplicates_and_mutable_storage_reject(source):
    with pytest.raises(ValueError):
        ResolvedRenderSequence(source.sequence.canonical_bytes + b"\n")
    with pytest.raises(TypeError):
        ResolvedRenderSequence(bytearray(source.sequence.canonical_bytes))
    duplicate = source.sequence.canonical_bytes[:-1] + b',"schema_version":"wrong"}'
    with pytest.raises(ValueError):
        ResolvedRenderSequence(duplicate)


@pytest.mark.parametrize("profile", ["direct_light_full", "rgb_preview", "render_only"])
@pytest.mark.parametrize("shadows", [False, True])
@pytest.mark.parametrize("pass_mode", ["primary_unprofiled", "diagnostic_profiled"])
@pytest.mark.parametrize("sample_index", [0, 9, 119, 120, 241])
def test_bound_exact_binary32_order_and_metadata(source, profile, shadows, pass_mode, sample_index):
    kwargs = dict(
        sample_index=sample_index,
        profile=profile,
        shadows=shadows,
        pass_mode=pass_mode,
        role_to_mask=source.packed_scene.role_to_mask,
    )
    bound = bind_render_parameters(source.sequence, **kwargs)
    payload = bound.validate(source.sequence, **kwargs)
    camera = resolve_frozen_camera_contract(ASYM)
    intrinsics, origin, rotation, distance = camera.scalar_values()
    expected = (*intrinsics, *origin, *rotation, distance, 0.03, 0.05, 0.07, 0.0, 0.0, 0.0, 1e-4)
    assert len(expected) == len(NATIVE_SCALAR_ORDER) == 24
    assert payload["native_scalars_f32"] == [struct.pack("<f", v).hex() for v in expected]
    assert payload["frame_id"] == sample_index + 1
    assert payload["sim_time_f64"] == (sample_index / 30.0).hex()
    assert payload["camera_index"] == 0
    assert payload["sensor_role_mask"] == source.packed_scene.role_to_mask["rgb"]
    assert payload["sensor_id"] == "q54:" + ASYM
    assert payload["diagnostics"] is (pass_mode == "diagnostic_profiled")
    assert payload["env_idx"] == 0
    assert payload["profile"] == profile and payload["shadows"] is shadows


@pytest.mark.parametrize("index,expected", [(0, 0), (119, 119), (120, 0), (241, 1)])
def test_dynamic_camera_modulo_and_no_warmup_offset(index, expected):
    source = materialize_render_source(DYNAMIC)
    bound = bind_render_parameters(
        source.sequence,
        sample_index=index,
        profile="rgb_preview",
        shadows=False,
        pass_mode="primary_unprofiled",
        role_to_mask=source.packed_scene.role_to_mask,
    )
    assert json.loads(bound.canonical_bytes)["camera_index"] == expected


@pytest.mark.parametrize(
    "field,value",
    [
        ("sample_index", -1),
        ("sample_index", True),
        ("sample_index", 1.0),
        ("profile", "full"),
        ("shadows", 1),
        ("pass_mode", True),
        ("role_to_mask", {"rgb": True}),
        ("role_to_mask", {"rgb": 0}),
        ("role_to_mask", {"rgb": 1 << 63}),
    ],
)
def test_bound_invalid_request_rejects(source, field, value):
    kwargs = dict(
        sample_index=0,
        profile="direct_light_full",
        shadows=True,
        pass_mode="primary_unprofiled",
        role_to_mask=source.packed_scene.role_to_mask,
    )
    kwargs[field] = value
    with pytest.raises((ValueError, TypeError)):
        bind_render_parameters(source.sequence, **kwargs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("frame_id", 99),
        ("camera_index", 1),
        ("sequence_sha256", "0" * 64),
        ("diagnostics", True),
        ("sensor_role_mask", 32),
        ("env_idx", 1),
        ("shadows", False),
        ("profile", "rgb_preview"),
        ("width", 999),
        ("sensor_id", "lane_local"),
        ("sim_time_f64", "0x1.0000000000000p+0"),
    ],
)
def test_rehashed_or_reconstructed_bound_claims_do_not_validate(source, field, value):
    kwargs = dict(
        sample_index=0,
        profile="direct_light_full",
        shadows=True,
        pass_mode="primary_unprofiled",
        role_to_mask=source.packed_scene.role_to_mask,
    )
    bound = bind_render_parameters(source.sequence, **kwargs)
    payload = json.loads(bound.canonical_bytes)
    payload[field] = value
    altered = BoundRenderParameters(canonical_json_bytes(payload))
    assert altered.sha256 != bound.sha256
    with pytest.raises(ValueError):
        altered.validate(source.sequence, **kwargs)


def test_binary32_ties_nearest_even_signed_zero_and_no_second_math(source):
    # A descriptive sequence is not frozen-source authority. Exercise encoding
    # with representable inputs; native admission must separately check artifact.
    payload = source.sequence.to_payload()
    scalar = 1.0 + 2.0**-24
    payload["frames"][0]["intrinsics_f64"][0] = scalar.hex()
    payload["frames"][0]["origin_f64"][0] = (-0.0).hex()
    sequence = ResolvedRenderSequence.from_payload(payload)
    record = bind_render_parameters(
        sequence,
        sample_index=0,
        profile="render_only",
        shadows=False,
        pass_mode="primary_unprofiled",
        role_to_mask={"rgb": 8},
    )
    values = json.loads(record.canonical_bytes)["native_scalars_f32"]
    assert values[0] == "0000803f"
    assert values[4] == "00000080"
    assert math.copysign(1.0, struct.unpack("<f", bytes.fromhex(values[4]))[0]) == -1.0


def test_shipping_and_missing_go2_reject_before_materialization(monkeypatch):
    import optical_worker.render_parameters as module

    def forbidden(*args, **kwargs):
        raise AssertionError("shipping materialization attempted")

    with monkeypatch.context() as context:
        context.setattr(module, "_materialize_canonical_scene", forbidden)
        with pytest.raises(ValueError, match="shipping"):
            materialize_render_source("physics_body_triangle_960x640_v1")
    with pytest.raises(ValueError, match="preflighted"):
        materialize_render_source("go2_static_960x640_v1")


def test_source_module_stays_cpu_only_in_fresh_process():
    script = """
import importlib.abc, sys
blocked = {'benchmarks', 'examples', 'tools', 'torch', 'warp'}
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Deny())
from optical_worker.render_parameters import materialize_render_source
for case in ('asymmetric_synthetic_7x3_v1', 'synthetic_dynamic_rigid_960x640_v1'):
    materialize_render_source(case)
assert not blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
