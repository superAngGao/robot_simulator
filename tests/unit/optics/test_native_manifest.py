"""CPU replay of retained real canonical manifests and malformed descriptors."""

import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from optical_worker.native_manifest import NativeArrayManifest, NativeInputManifest, _sha256_payload

DATA = Path(__file__).parent / "data/native_manifests"
HASHES = {
    "asymmetric_synthetic_7x3_v1": "f65056b303775f4d13ad1a48660467cbae06a9d9dbe09fc5596a341ba5af6728",
    "synthetic_dynamic_rigid_960x640_v1": "fd63ec133149ac62ef49e37534783d8c3d4e9b6f21132fa90ac02b606187b023",
    "go2_static_960x640_v1": "ac3d950309917168d0bb5758e9916adcffcbb17bd68f6611e4fe0f3bdfed3a12",
    "go2_static_1920x1080_v1": "4785c42146d8707453654eefbc4c94e38e683ec4b5659536240e6b1048b2279e",
}


def payload(case="asymmetric_synthetic_7x3_v1"):
    return json.loads((DATA / (case + ".json")).read_bytes())


@pytest.mark.parametrize("case,digest", HASHES.items())
def test_retained_real_cpu_manifest_bytes_and_immutable_parse(case, digest):
    raw = (DATA / (case + ".json")).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
    p = json.loads(raw)
    parsed = NativeInputManifest.from_payload(p)
    assert parsed.sha256 == digest
    assert parsed.to_payload() == p
    p["arrays"][0]["shape"].append(9)
    p["semantics"]["scene_id"] = "substitution"
    assert parsed.sha256 == digest
    with pytest.raises(TypeError):
        parsed.semantics["scene_id"] = "substitution"
    with pytest.raises(FrozenInstanceError):
        parsed.case_id = "substitution"


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(extra=0),
    lambda p: p.update(schema_version="old"),
    lambda p: p.update(case_id=""),
    lambda p: p.update(case_sha256="A" * 64),
    lambda p: p["semantics"].update(scene_id="another"),
    lambda p: p["semantics"].update(extra=0),
    lambda p: p["arrays"].reverse(),
    lambda p: p["arrays"].append(copy.deepcopy(p["arrays"][0])),
    lambda p: p["arrays"][0].update(shape=[True]),
    lambda p: p["arrays"][0].update(shape=[-1]),
    lambda p: p["arrays"][0].update(nbytes=True),
    lambda p: p["arrays"][0].update(nbytes=999),
    lambda p: p["arrays"][0].update(c_contiguous=1),
    lambda p: p["arrays"][0].update(dtype=">f4"),
    lambda p: p["arrays"][0].update(dtype="<f8"),
    lambda p: p["arrays"][0].update(content_sha256="0" * 64),
    lambda p: p.update(content_sha256="0" * 64),
    lambda p: p.update(scene_array_names=["missing"]),
    lambda p: p.update(scene_array_names=[]),
    lambda p: p.update(fixed_render_fixture_arrays=p["scene_array_names"]),
    lambda p: p.update(transform_replay_sha256="0" * 64),
    lambda p: p.update(transform_frame_count=True),
    lambda p: p.update(fixed_render_fixture_sha256=None),
])
def test_reject_inconsistent_canonical_manifest(mutate):
    p = payload()
    mutate(p)
    with pytest.raises(ValueError):
        NativeInputManifest.from_payload(p)


def test_replay_layout_must_match_frame_count_even_with_updated_content_hash():
    p = payload("synthetic_dynamic_rigid_960x640_v1")
    p["transform_frame_count"] += 1
    with pytest.raises(ValueError, match="replay array layout"):
        NativeInputManifest.from_payload(p)
    p = payload("synthetic_dynamic_rigid_960x640_v1")
    row = next(row for row in p["arrays"] if row["name"] == "replay_x_world_r")
    row["shape"][-1] = 1
    row["nbytes"] //= 3
    p["content_sha256"] = _sha256_payload(p["arrays"])
    with pytest.raises(ValueError, match="replay array layout"):
        NativeInputManifest.from_payload(p)


def test_scalar_and_zero_extent_and_generic_constructor_compatibility():
    base = dict(name="a", semantic="a", dtype="<f4", c_contiguous=True, content_sha256="a" * 64)
    assert NativeArrayManifest.from_payload(dict(base, shape=[], nbytes=4)).shape == ()
    assert NativeArrayManifest.from_payload(dict(base, shape=[3, 0, 2], nbytes=0)).nbytes == 0
    canonical = NativeInputManifest.from_payload(payload())
    generic = replace(canonical, semantics={"gain": 1.5}, input_contract=None,
                      scene_contract_sha256=None)
    assert generic.to_payload()["semantics"] == {"gain": 1.5}
    assert generic.sha256 == _sha256_payload(generic.to_payload())
    assert generic.result_capable is False
    with pytest.raises(ValueError):
        NativeInputManifest.from_payload(generic.to_payload())


def test_manifest_import_and_replay_are_stdlib_only():
    root = Path(__file__).resolve().parents[3]
    code = '''
import builtins, json, pathlib, sys
sys.path.insert(0, sys.argv[1])
original = builtins.__import__
def guarded(name, *args, **kwargs):
    assert name.split('.')[0] not in {'numpy', 'torch', 'warp', 'optics', 'physics'}
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from optical_worker.native_manifest import NativeInputManifest
for path in pathlib.Path(sys.argv[2]).glob('*.json'):
    NativeInputManifest.from_payload(json.loads(path.read_bytes()))
assert not {'numpy', 'torch', 'warp'} & set(sys.modules)
'''
    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code, str(root), str(DATA)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
