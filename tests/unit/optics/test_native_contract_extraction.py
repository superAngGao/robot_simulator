"""C4c2a relocation checks anchored to the approved pre-move implementation."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
BASE = "a1f6c8c"
MOVED = {
    "native": "native_input",
    "cases": "cases",
    "workload_contract": "workload_contract",
    "receipts": "provenance",
    "timing": "timing",
}


def _base_source(path):
    return subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT, text=True)


def _definitions(source):
    return {
        n.name: ast.dump(n, include_attributes=False)
        for n in ast.parse(source).body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef))
    }


@pytest.mark.parametrize("old,new", MOVED.items())
def test_single_definitions_preserve_all_bodies_and_alias_identities(old, new):
    legacy = importlib.import_module(f"benchmarks.optical.{old}")
    production = importlib.import_module(f"optical_worker.{new}")
    old_definitions = _definitions(_base_source(f"benchmarks/optical/{old}.py"))
    current = _definitions(Path(production.__file__).read_text())
    if new == "provenance":
        # C2c4 v13 deliberately extends only these binding/parameter definitions.
        changed = {
            "NativeStageEvidence",
            "C3OneShotNativeExecutionBindingEvidence",
            "build_c3_one_shot_native_execution_binding",
        }
        assert set(current) == set(old_definitions) | {"_validate_render_record"}
        assert {name for name in old_definitions if current[name] != old_definitions[name]} == changed
    elif new == "native_input":
        # Preserve unaffected relocation bodies and exact legacy aliases.
        # Later reviewed factory preparation adds extension ownership; fixed
        # render wrapping retains the original host BVH's actual statistics.
        from optical_worker import native_manifest

        moved = {"NativeArrayManifest", "NativeInputManifest", "_canonical_json_bytes", "_sha256_payload"}
        tree = ast.parse(Path(native_manifest.__file__).read_text())
        definitions = {}
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in moved:
                if isinstance(node, ast.ClassDef):
                    node.body = [item for item in node.body
                                 if not isinstance(item, ast.FunctionDef) or item.name != "from_payload"]
                definitions[node.name] = ast.dump(node, include_attributes=False)
                assert getattr(production, node.name) is getattr(native_manifest, node.name)
        assert not set(current) & moved
        merged = current | definitions
        added = {"_derive_fixed_fixture", "_prepared_fixed_fixture",
                 "build_prepared_warp_fixed_render_inputs", "build_prepared_cuda_fixed_render_inputs"}
        changed = {"CanonicalFixedRenderFixtureHost", "_derive_fixed_fixture_arrays",
                   "build_prepared_cuda_scene_update_factory", "build_prepared_cuda_lbvh_rebuild_factory"}
        assert set(merged) == set(old_definitions) | added
        assert {name for name in old_definitions if merged[name] != old_definitions[name]} == changed
    else:
        assert current == old_definitions
    assert not _definitions(Path(legacy.__file__).read_text())
    for name in old_definitions:
        assert getattr(legacy, name) is getattr(production, name)
    # Include private registries, ContextVars, and imported authority types.
    for node in ast.parse(Path(legacy.__file__).read_text()).body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                assert getattr(legacy, alias.asname or alias.name) is getattr(production, alias.name)


def test_host_bvh_is_an_exact_math_move_not_a_second_algorithm():
    from optics import device_bvh, device_scene, host_bvh

    before = _definitions(_base_source("optics/device_bvh.py"))
    after = _definitions(Path(host_bvh.__file__).read_text())
    for name in (
        "DeviceBvhBuildStats",
        "_HostNode",
        "build_host_median_bvh",
        "host_bvh_topology_sha256",
        "_surface_area",
    ):
        assert after[name] == before[name]
        assert name not in _definitions(Path(device_bvh.__file__).read_text())
        assert getattr(device_bvh, name) is getattr(host_bvh, name)
    assert (
        after["_canonical_sha256"]
        == _definitions(_base_source("optics/device_scene.py"))["_canonical_sha256"]
    )
    assert device_scene._canonical_sha256 is host_bvh._canonical_sha256


@pytest.mark.parametrize("count", [0, 1, 2, 17])
@pytest.mark.parametrize("strategy", ["sort", "partition"])
def test_host_bvh_edge_cases(count, strategy):
    from optics.host_bvh import build_host_median_bvh, host_bvh_topology_sha256

    for degenerate in (False, True):
        lower = np.zeros((count, 3), np.float32)
        if not degenerate:
            lower[:, 0] = np.arange(count)
        upper = lower + 1
        source_keys = np.arange(count, dtype=np.int64)[::-1].copy()
        result = build_host_median_bvh(lower, upper, source_keys, leaf_size=4, split_strategy=strategy)
        again = build_host_median_bvh(lower, upper, source_keys, leaf_size=4, split_strategy=strategy)
        assert host_bvh_topology_sha256(result) == host_bvh_topology_sha256(again)
        np.testing.assert_array_equal(np.sort(result["prim_ids"]), np.arange(count))
        assert result["stats"].split_strategy == strategy
        assert result["stats"].leaf_size == 4
        if count:
            np.testing.assert_array_equal(result["bounds_min"][0], lower.min(axis=0))
            np.testing.assert_array_equal(result["bounds_max"][0], upper.max(axis=0))


_COLD_PROBE = r"""
import builtins, importlib, json, sys
attempts = []
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'warp', 'benchmarks', 'examples', 'tools'}:
        attempts.append(name)
        raise ImportError('forbidden import attempt: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
mode = sys.argv[1]
if mode == 'calibrate':
    try:
        import torch
    except ImportError:
        pass
    try:
        import benchmarks.optical
    except ImportError:
        pass
    assert attempts == ['torch', 'benchmarks.optical']
elif mode == 'host':
    import numpy as np
    from optics.registry import (OpticalWorldRegistry, OpticalMaterialSpec,
                                 OpticalLightSpec, OpticalInstanceSpec)
    from optics.scene_contract import PackedOpticalSceneHost
    from physics.spatial import SpatialTransform
    from optical_worker.native_input import CanonicalFixedRenderFixtureHost, CanonicalNativeInput
    from optical_worker.native_input import CanonicalTransformReplayHost
    from optical_worker.cases import get_case
    from optical_worker.workload_contract import frozen_workload_content
    data = json.load(sys.stdin)
    registry = OpticalWorldRegistry()
    for item in data['materials']:
        registry.add_material(OpticalMaterialSpec(**item))
    for item in data['lights']:
        registry.add_light(OpticalLightSpec(**item))
    for name, item in data['geometry'].items():
        registry.add_triangle_mesh_geometry(name, **item)
    for item in data['instances']:
        item['X_body_geometry'] = SpatialTransform(**item['X_body_geometry'])
        registry.add_instance(OpticalInstanceSpec(**item))
    case = get_case('asymmetric_synthetic_7x3_v1')
    packed = PackedOpticalSceneHost.from_registry(registry, source_contract_id=case.case_id,
        source_contract_sha256=case.sha256, scene_id=case.scene_id)
    fixture = CanonicalFixedRenderFixtureHost.from_scene_contract(packed,
        fixed_transform_id='asymmetric_fixed_world_v1',
        x_world_R=np.empty((0, 3, 3), np.float32), x_world_r=np.empty((0, 3), np.float32))
    native = CanonicalNativeInput.from_scene_contract(packed, input_contract='fixed_native_scene_v1',
        fixed_fixture=fixture)
    expected = frozen_workload_content(case.case_id)
    assert packed.scene_contract_sha256 == expected.scene_contract_sha256
    assert fixture.fixed_render_fixture_sha256 == expected.fixed_render_fixture_sha256
    assert native.manifest.result_capable
    replay = CanonicalTransformReplayHost.from_arrays(case_id=data['replay_case'],
        x_world_R=data['replay_R'], x_world_r=data['replay_r'])
    expected_replay = frozen_workload_content(data['replay_case']).transform_replay_sha256
    assert replay.transform_replay_sha256 == expected_replay
else:
    importlib.import_module(mode)
if mode != 'calibrate':
    assert not attempts, attempts
    assert not any(n.split('.')[0] in {'torch','warp','benchmarks','examples','tools'} for n in sys.modules)
    assert not any(n.startswith(('optics.cuda_', 'optics.device_scene', 'optics.device_bvh',
                                 'optics.warp_', 'optics.gpu_runtime')) for n in sys.modules)
print('cold import verified')
"""


def _probe(mode, data=None):
    result = subprocess.run(
        [sys.executable, "-B", "-c", _COLD_PROBE, mode],
        cwd=ROOT,
        input=data,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "entry",
    [
        "calibrate",
        "optics",
        "optics.host_bvh",
        "optics.scene_contract",
        "optical_worker.native_contract",
        "optical_worker.native_input",
        "optical_worker.provenance",
        "optical_worker.timing",
        "optical_worker.cases",
        "optical_worker.lane_contract",
        "optical_worker.workload_contract",
    ],
)
def test_cold_contract_import_attempts(entry):
    _probe(entry)


def test_cold_real_packer_fixture_and_replay_construction():
    # Serialize logical registry data, not a copied packer or an already-packed
    # object. The child re-packs through the sole production implementation.
    from benchmarks.optical.workloads import _asymmetric_registry, build_canonical_transform_replay
    from physics.spatial import SpatialTransform

    registry = _asymmetric_registry()
    replay = build_canonical_transform_replay("synthetic_dynamic_rigid_960x640_v1")
    data = dict(
        materials=list(registry.materials.values()),
        lights=list(registry.lights.values()),
        geometry=registry.geometry,
        instances=registry.instances,
        replay_case=replay.case_id,
        replay_R=replay.arrays["replay_x_world_R"],
        replay_r=replay.arrays["replay_x_world_r"],
    )

    def encode(value):
        if isinstance(value, SpatialTransform):
            return {"R": value.R, "r": value.r}
        if dataclasses.is_dataclass(value):
            return {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, frozenset):
            return sorted(value)
        raise TypeError(type(value))

    _probe("host", json.dumps(data, default=encode))


@pytest.mark.parametrize("reverse", [False, True])
def test_lazy_export_import_order_and_exact_public_inventory(reverse):
    code = """
import ast, importlib, subprocess, sys
root = importlib.import_module('optics')
before = ast.parse(subprocess.check_output(['git','show',sys.argv[1]+':optics/__init__.py'], text=True))
names = next(ast.literal_eval(n.value) for n in before.body if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id == '__all__' for t in n.targets))
assert root.__all__ == names
assert set(names) <= set(dir(root))
loaded = set(sys.modules)
try:
    root.not_a_known_export
except AttributeError:
    pass
else:
    raise AssertionError('unknown export accepted')
assert set(sys.modules) == loaded
for n in before.body:
    if isinstance(n, ast.ImportFrom):
        for item in n.names:
            if sys.argv[2] == 'True':
                expected = getattr(importlib.import_module('optics.'+n.module), item.name)
                actual = getattr(root, item.name)
            else:
                actual = getattr(root, item.name)
                expected = getattr(importlib.import_module('optics.'+n.module), item.name)
            assert actual is expected is getattr(root, item.name)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, BASE, str(reverse)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("reverse", [False, True])
def test_shared_timing_context_rejects_cross_path_and_restores(reverse):
    from benchmarks.optical import timing as legacy
    from optical_worker import timing as production

    enter, observe = (legacy, production) if reverse else (production, legacy)
    with enter.native_timed_submission(row_id="cross-path", lane="warp", boundary="R0") as current:
        assert observe.active_timed_boundary() is current
        with pytest.raises(observe.TimedBoundaryViolation, match="allocation"):
            observe.assert_operation_allowed("allocation")
        with pytest.raises(observe.TimedBoundaryViolation, match="nested"):
            with observe.native_timed_submission(row_id="nested", lane="warp", boundary="R0"):
                pytest.fail("nested submission accepted")
        assert observe.active_timed_boundary() is current
    assert enter.active_timed_boundary() is observe.active_timed_boundary() is None
    previous = observe._ACTIVE_RUNTIME_HOOK_CAPABILITIES.get()
    with pytest.raises(RuntimeError, match="sentinel"):
        with enter.activate_runtime_hooks(frozenset({"entry"})):
            assert set(observe._ACTIVE_RUNTIME_HOOK_CAPABILITIES.get()) == {"entry"}
            raise RuntimeError("sentinel")
    assert observe._ACTIVE_RUNTIME_HOOK_CAPABILITIES.get() is previous


def test_forged_c3_never_invokes_claimed_authority_callback():
    from optics.cuda_arena import _canonical_c3_binding, _validate_c3_adapter_authority
    from optics.native_arena import NativeOpticalArenaProtocolError

    calls = []
    fake_type = type(
        "C3OneShotNativeExecutionBindingEvidence",
        (),
        {
            "__module__": "optical_worker.provenance",
            "_assert_cuda_arena_adapter_authority": lambda *a, **k: calls.append("forged"),
            "to_payload": lambda *a: calls.append("serialized"),
        },
    )
    fake = fake_type()
    with pytest.raises(TypeError, match="canonical C3"):
        _canonical_c3_binding(fake)
    with pytest.raises(NativeOpticalArenaProtocolError, match="canonical arena adapter"):
        _validate_c3_adapter_authority(object(), fake, object(), object())
    assert calls == []


def test_historical_gate_and_closed_product_source_boundary_are_preserved():
    historical = json.loads((ROOT / "tests/unit/benchmarks/data/render_migration_v12.json").read_text())
    assert historical["gate"] == json.loads(_base_source("benchmarks/results/q54_p12_4/gate.json"))
    # The complete v12 -> v13 field/preimage migration is checked independently
    # in test_optical_render_migration. The approved C0 CPU-owner extraction
    # extends only its literal source list; every other contract field is fixed.
    contract_path = "benchmarks/results/q54_p12_4/closed_worker_runtime_contract.json"
    original = json.loads(_base_source(contract_path))
    current = json.loads((ROOT / contract_path).read_text(encoding="utf-8"))
    original["trusted_source_paths"] = sorted(original["trusted_source_paths"] + [
        "optical_worker/lane_contract.py", "optical_worker/native_manifest.py",
    ])
    assert current == original
    paths = [
        "optical_worker/worker.py",
        "optical_worker/manifest.py",
        "optical_worker/protocol.py",
        "tools/optical_closed_worker_supervisor.py",
    ]
    for path in paths:
        assert (ROOT / path).read_text(encoding="utf-8") == _base_source(path)
