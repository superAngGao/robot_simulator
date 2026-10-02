from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest

import optical_worker.camera_contract as contract_module
from optical_worker.camera_contract import (
    ASYMMETRIC_CAMERA_CASE_ID as CASE,
)
from optical_worker.camera_contract import (
    ResolvedPinholeCameraContract,
    resolve_frozen_camera_contract,
)

EXPECTED_SHA = "04ab7bc2571b8396f06d36b7baea66224868001894c4ce7a5ea12e4faecda1ef"


def _axis(axis, tangent):
    cosine = (1 - tangent * tangent) / (1 + tangent * tangent)
    sine = 2 * tangent / (1 + tangent * tangent)
    result = np.eye(3, dtype=object)
    result[:] = [[F(int(v)) for v in row] for row in result]
    a, b = ((1, 2), (2, 0), (0, 1))[axis]
    result[a, a] = result[b, b] = cosine
    result[a, b], result[b, a] = -sine, sine
    return result


def _rotation():
    x, y, z = (_axis(i, t) for i, t in enumerate((F(-1, 20), F(-1, 8), F(1, 12))))
    return z @ y @ x, x @ y @ z


def _camera(*, quantized=False, rotation=None):
    from physics.spatial import SpatialTransform
    from sensing.optical import OpticalPinholeCameraSpec

    record = resolve_frozen_camera_contract(CASE)
    intrinsic, origin, matrix, distance = record.scalar_values()
    values = np.array((*intrinsic, *origin, *matrix, distance), dtype=np.float64)
    if quantized:
        values = values.astype(np.float32).astype(np.float64)
    actual_matrix = values[7:16].reshape(3, 3) if rotation is None else rotation
    return OpticalPinholeCameraSpec(
        0,
        0.0,
        0,
        "canonical_cpu_camera",
        record.width,
        record.height,
        *values[:4],
        SpatialTransform(actual_matrix, values[4:7]),
        values[-1],
        record.sensor_role,
    )


def _snapshot():
    from benchmarks.optical.workloads import (
        _asymmetric_registry,
        build_canonical_fixed_fixture,
        build_canonical_packed_scene,
    )
    from optics.scene import OpticalInstanceSnapshot, OpticalSceneSnapshot, build_cpu_bvh_acceleration
    from physics.spatial import SpatialTransform

    # Real canonical materializer's external content anchors remain required.
    packed = build_canonical_packed_scene(CASE)
    fixture = build_canonical_fixed_fixture(packed)
    assert packed.scene_contract_sha256 == "ab3b06cd48e0315ce5d33c0f81a880e71fe5df7369c805a2ae85f5c0197b9e35"
    assert (
        fixture.fixed_render_fixture_sha256
        == "4fcece950264cb2ebb78cf1ec06ad52f5a151b3d2cf2970314f1cae5a2aa7907"
    )
    registry = _asymmetric_registry()
    material_ordinals = {name: index for index, name in enumerate(registry.materials)}
    instances = tuple(
        OpticalInstanceSnapshot(
            item.instance_id,
            item.numeric_instance_id,
            item.geometry_id,
            registry.materials[item.material_id],
            material_ordinals[item.material_id],
            registry.geometry[item.geometry_id],
            SpatialTransform.identity(),
            item.roles,
        )
        for item in registry.instances
    )
    return OpticalSceneSnapshot(
        0,
        0.0,
        0,
        instances,
        tuple(registry.lights.values()),
        acceleration=build_cpu_bvh_acceleration(instances),
    )


def test_exact_fraction_matrix_and_canonical_identity():
    record = resolve_frozen_camera_contract(CASE)
    exact, _ = _rotation()
    assert (exact.T @ exact == np.eye(3, dtype=object)).all()
    assert tuple(float(v).hex() for v in exact.flat) == record.rotation_world_from_camera_f64
    matrix = np.array(record.scalar_values()[2]).reshape(3, 3)
    assert np.linalg.det(matrix) > 0
    assert np.max(abs(matrix.T @ matrix - np.eye(3))) < 2e-15
    assert record.sha256 == EXPECTED_SHA
    assert ResolvedPinholeCameraContract.from_payload(record.to_payload()) == record
    payload = record.to_payload()
    payload["rotation_world_from_camera_f64"][0] = "0x0.0p+0"
    assert record.sha256 == resolve_frozen_camera_contract(CASE).sha256 == EXPECTED_SHA
    with pytest.raises(FrozenInstanceError):
        record.width = 8
    with pytest.raises(ValueError):
        replace(record, width=8)
    with pytest.raises(ValueError):
        replace(record, intrinsics_f64=list(record.intrinsics_f64))
    object.__setattr__(
        record, "rotation_world_from_camera_f64", tuple(reversed(record.rotation_world_from_camera_f64))
    )
    with pytest.raises(ValueError):
        record.scalar_values()
    with pytest.raises(ValueError):
        record.to_payload()


@pytest.mark.parametrize("field", tuple(resolve_frozen_camera_contract(CASE).to_payload()))
@pytest.mark.parametrize("mutation", ["delete", "null", "changed"])
def test_every_payload_field_is_required_and_frozen(field, mutation):
    payload = resolve_frozen_camera_contract(CASE).to_payload()
    if mutation == "delete":
        del payload[field]
    elif mutation == "null":
        payload[field] = None
    elif type(payload[field]) is list:
        payload[field] = list(reversed(payload[field]))
    elif type(payload[field]) is int:
        payload[field] += 1
    else:
        payload[field] += "_changed"
    with pytest.raises(ValueError):
        ResolvedPinholeCameraContract.from_payload(payload)


@pytest.mark.parametrize("bad", [True, 7.0, "7", [], None])
def test_dimensions_are_exact_integers(bad):
    payload = resolve_frozen_camera_contract(CASE).to_payload()
    payload["width"] = bad
    with pytest.raises(ValueError):
        ResolvedPinholeCameraContract.from_payload(payload)


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "0X1.5000000000000P+2", "0x1.5p+2", 5.25, True])
def test_scalar_strings_must_be_exact_canonical_hex(bad):
    payload = resolve_frozen_camera_contract(CASE).to_payload()
    payload["intrinsics_f64"][0] = bad
    with pytest.raises(ValueError):
        ResolvedPinholeCameraContract.from_payload(payload)


def test_shapes_extra_fields_subclasses_and_unknown_cases():
    good = resolve_frozen_camera_contract(CASE).to_payload()
    for name in ("intrinsics_f64", "origin_world_f64", "rotation_world_from_camera_f64"):
        for value in (good[name][:-1], good[name] + good[name][:1], tuple(good[name])):
            with pytest.raises(ValueError):
                ResolvedPinholeCameraContract.from_payload({**good, name: value})
    with pytest.raises(ValueError):
        ResolvedPinholeCameraContract.from_payload({**good, "sha256": EXPECTED_SHA})

    class ForgedDict(dict):
        pass

    with pytest.raises(ValueError):
        ResolvedPinholeCameraContract.from_payload(ForgedDict(good))
    for case in ("go2_static_960x640_v1", "missing", True, None, {}):
        with pytest.raises(ValueError):
            resolve_frozen_camera_contract(case)


@pytest.mark.parametrize("changed", ["rotation", "fx", "width"])
def test_source_case_drift_rejected(monkeypatch, changed):
    case = copy.deepcopy(contract_module.get_case(CASE))
    if changed == "width":
        case = replace(case, width=8)
    else:
        case.camera[changed] = "other_rotation" if changed == "rotation" else 6.0
    monkeypatch.setattr(contract_module, "get_case", lambda unused: case)
    with pytest.raises(ValueError, match="source case"):
        resolve_frozen_camera_contract(CASE)


def test_all_host_rays_and_wrong_conventions():
    from sensing.optical import build_pinhole_camera_rays

    camera = _camera()
    rays = build_pinhole_camera_rays(camera)
    exact, reversed_axes = _rotation()
    expected = []
    for py in range(3):
        for px in range(7):
            vector = [(px - 1.3) / 5.25, (py - 0.8) / 7.75, 1.0]
            world = np.array(
                [sum(float(exact[row, col]) * vector[col] for col in range(3)) for row in range(3)]
            )
            expected.append(world / np.linalg.norm(world))
    np.testing.assert_allclose(rays.directions_world, expected, rtol=0, atol=3e-16)
    np.testing.assert_array_equal(rays.origins_world, np.tile([0.31, -0.27, 0.43], (21, 1)))
    assert np.unique(rays.directions_world, axis=0).shape == (21, 3)
    for boundary in (0, 6, 7, 13, 14, 20):
        np.testing.assert_allclose(rays.directions_world[boundary], expected[boundary], rtol=0, atol=3e-16)
    for wrong in (camera.X_world_camera.R.T, np.eye(3), np.array(reversed_axes, dtype=float)):
        assert not np.allclose(build_pinhole_camera_rays(_camera(rotation=wrong)).directions_world, expected)
    permutation = np.arange(21).reshape(3, 7).T.ravel()
    assert not np.allclose(rays.directions_world[permutation], expected)


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("shadows", [False, True])
def test_actual_canonical_cpu_scene_observability(quantized, shadows):
    from optics.execution import CpuDirectLightOpticalExecutor, CpuReferenceOpticalExecutor
    from sensing.optical import build_pinhole_camera_rays

    snapshot = _snapshot()
    rays = build_pinhole_camera_rays(_camera(quantized=quantized))
    result = CpuDirectLightOpticalExecutor(shadows=shadows, ambient_rgb=(0.03, 0.05, 0.07)).execute(
        snapshot, rays
    )
    brute = CpuReferenceOpticalExecutor().execute(snapshot, rays)
    hit = np.asarray(result.channel("hit_mask"), dtype=bool)
    np.testing.assert_array_equal(np.flatnonzero(hit), [4, 11, 16, 17])
    np.testing.assert_array_equal(result.channel("material_index")[hit], [5, 12, 16, 18])
    np.testing.assert_array_equal(brute.channel("hit_mask"), hit)
    np.testing.assert_allclose(brute.channel("range_m"), result.channel("range_m"))
    for name in ("range_m", "position_world", "normal_world", "rgb", "intensity"):
        assert np.isfinite(result.channel(name)[hit]).all()
    assert np.isposinf(result.channel("range_m")[~hit]).all()
    assert np.isnan(result.channel("position_world")[~hit]).all()
    assert np.isnan(result.channel("normal_world")[~hit]).all()
    assert (result.channel("material_index")[~hit] == 0).all()
    assert (result.channel("rgb")[~hit] == 0).all()
    assert (result.channel("intensity")[~hit] == 0).all()


def test_contract_cold_import_attempts_are_stdlib_only():
    root = Path(__file__).resolve().parents[3]
    source = f"""
import builtins, sys
sys.path.insert(0, {str(root)!r})
original = builtins.__import__
attempts=[]
def observe(name, *args, **kwargs):
    forbidden=('numpy','torch','warp','benchmarks','examples','tools','optics','physics','sensing')
    if name.split('.')[0] in forbidden:
        attempts.append(name)
        raise ImportError('forbidden import attempt')
    return original(name,*args,**kwargs)
builtins.__import__=observe
try:
    import numpy
except ImportError:
    pass
assert attempts == ['numpy']
attempts.clear()
from optical_worker.camera_contract import resolve_frozen_camera_contract
assert resolve_frozen_camera_contract({CASE!r}).sha256 == {EXPECTED_SHA!r}
assert not attempts, attempts
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", source], cwd=root, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_existing_cases_and_content_are_unchanged():
    from optical_worker.cases import case_hashes
    from optical_worker.workload_contract import frozen_workload_payloads

    root = Path(__file__).resolve().parents[3]
    gate = json.loads((root / "benchmarks/results/q54_p12_4/gate.json").read_text())
    assert case_hashes() == {name: value["sha256"] for name, value in gate["cases"].items()}
    for name, workload in frozen_workload_payloads().items():
        catalog = gate["cases"][name]
        assert set(catalog) == set(workload) | {
            "sha256",
            "resolved_render_sequence_sha256",
            "resolved_render_scope",
        }
        assert workload == {key: catalog[key] for key in workload}
