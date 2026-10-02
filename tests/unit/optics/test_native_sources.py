"""Literal release catalog and actual cold-process import ledger regressions."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import site
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from optical_worker.native_sources import CATALOG_PATH, capture_native_source_snapshot, native_source_paths
from optical_worker.protocol import ClosedWorkerProtocolError, canonical_json_bytes

ROOT = Path(__file__).resolve().parents[3]


def _git(root, *args):
    return subprocess.check_output([
        "git", "-c", "core.autocrlf=false", "-c", "commit.gpgsign=false",
        "-c", "user.name=source-fixture", "-c", "user.email=fixture@example.invalid", *args,
    ], cwd=root, timeout=20).decode().strip()


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    # Explicit test packaging. The production verifier never normalizes CRLF.
    root = tmp_path_factory.mktemp("native-source-release")
    catalog = json.loads((ROOT / CATALOG_PATH).read_text())
    digests = {entry["relative_path"]: entry["sha256"] for entry in catalog["source_digests"]}
    paths = sorted(set(native_source_paths("warp") + native_source_paths("cuda")))
    data = [CATALOG_PATH, *(f"optical_worker/data/render_sequences/v1/{case}.jsonl" for case in (
        "asymmetric_synthetic_7x3_v1", "synthetic_dynamic_rigid_960x640_v1",
        "go2_static_960x640_v1", "go2_static_1920x1080_v1",
    ))]
    for relative in paths + data:
        content = (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
        if relative in paths:
            assert hashlib.sha256(content).hexdigest() == digests[relative], relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "LF source release fixture")
    return root, _git(root, "rev-parse", "HEAD")


def _snapshot(release, lane="warp"):
    root, revision = release
    return capture_native_source_snapshot(root, lane, repository_revision=revision)


def _cold(release, body, *, lane="warp", cpu_only=False):
    root, revision = release
    # The local development interpreter keeps NumPy in its known user-site.
    # -I excludes that by default; explicitly declare this installed dependency
    # location for materializer tests only, never copy the parent's sys.path.
    dependency_paths = [] if cpu_only else [site.getusersitepackages()]
    script = f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(root)!r})
sys.path.extend({dependency_paths!r})
from optical_worker.native_sources import capture_native_source_snapshot
from optical_worker.protocol import ClosedWorkerProtocolError
snapshot = capture_native_source_snapshot(Path({str(root)!r}), {lane!r}, repository_revision={revision!r})
{body}
"""
    command = [sys.executable, "-B", "-I", *(["-S"] if cpu_only else []), "-c", script]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def test_literal_lane_catalogs_have_no_losing_optical_modules():
    warp, cuda = native_source_paths("warp"), native_source_paths("cuda")
    assert warp == tuple(sorted(set(warp)))
    assert cuda == tuple(sorted(set(cuda)))
    assert set(warp) - set(cuda) == {"optics/warp_native_render.py"}
    assert set(cuda) - set(warp) == {
        "optics/_cuda_lbvh_source.py", "optics/cuda_arena.py", "optics/cuda_direct_light.py",
        "optics/cuda_lbvh.py", "optics/cuda_scene.py", "optics/cuda_timeline.py",
    }
    for paths in (warp, cuda):
        assert not any(p.startswith(("benchmarks/", "examples/", "tests/")) for p in paths)
        for p in paths:
            parent = Path(p).parent
            if parent.as_posix() != "tools":
                assert f"{parent.as_posix()}/__init__.py" in paths


@pytest.mark.parametrize("lane", ["warp", "cuda"])
def test_complete_catalog_capture_is_cpu_safe_and_byte_exact(release, lane):
    snapshot = _snapshot(release, lane)
    assert tuple(item.relative_path for item in snapshot.sources) == native_source_paths(lane)
    assert all(hashlib.sha256(s.source).hexdigest() == s.sha256 for s in snapshot.sources)
    assert snapshot == _snapshot(release, lane)
    snapshot.verify_unchanged()
    _cold(release, """
assert not set(sys.modules) & {'numpy', 'torch', 'warp', 'optics', 'benchmarks'}
observed = snapshot.check_loaded_modules()
assert 'optical_worker/native_sources.py' in observed
print(snapshot.sha256)
""", lane=lane, cpu_only=True)


@pytest.mark.parametrize("lane", ["warp", "cuda"])
def test_actual_materializer_import_closure_passes_without_runtime_imports(release, lane):
    _cold(release, """
from optical_worker.render_parameters import materialize_render_source
from optical_worker.workloads import build_canonical_transform_replay, build_canonical_fixed_fixture
source = materialize_render_source('asymmetric_synthetic_7x3_v1')
materialize_render_source('synthetic_dynamic_rigid_960x640_v1')
build_canonical_transform_replay('synthetic_dynamic_rigid_960x640_v1')
build_canonical_fixed_fixture(source.packed_scene)
observed = snapshot.check_loaded_modules()
assert 'physics/__init__.py' in observed
assert 'sensing/__init__.py' in observed
assert not set(sys.modules) & {'torch', 'warp', 'benchmarks', 'tests', 'examples'}
print(observed)
""", lane=lane)


@pytest.mark.parametrize("lane", ["warp", "cuda"])
def test_real_go2_cpu_materialization_has_no_unlisted_imports(release, lane):
    asset = os.environ.get("ROBOT_SIM_TEST_GO2_MODEL_DIR")
    if not asset:
        pytest.skip("real Go2 asset path not configured; required remote invocation supplies it")
    _cold(release, f"""
from optical_worker.assets import preflight_case_asset
from optical_worker.render_parameters import materialize_render_source
model = Path({asset!r})
for case in ('go2_static_960x640_v1', 'go2_static_1920x1080_v1'):
    evidence = preflight_case_asset(case, repo_root=model.parent, model_dir=model)
    materialize_render_source(case, go2_model_dir=model, asset_evidence=evidence)
observed = snapshot.check_loaded_modules()
assert 'optics/mjcf_visual.py' in observed
assert not set(sys.modules) & {{'torch', 'warp', 'benchmarks', 'tests', 'examples'}}
print(observed)
""", lane=lane)


@pytest.mark.parametrize("lane", ["warp", "cuda"])
def test_actual_posix_checkout_catalog_matches_reviewed_git_bytes(lane):
    if os.name != "posix":
        pytest.skip("POSIX release-byte check; Windows uses the explicit LF fixture")
    # No test packaging and no normalization substitutes for the real checkout.
    snapshot = capture_native_source_snapshot(ROOT, lane, repository_revision=_git(ROOT, "rev-parse", "HEAD"))
    snapshot.verify_unchanged()


@pytest.mark.parametrize("lane", [None, "CUDA", "", [], 0, True])
def test_unknown_lane_rejects_before_io(lane):
    with pytest.raises(ClosedWorkerProtocolError, match="lane"):
        capture_native_source_snapshot(Path("not-present"), lane, repository_revision="0" * 40)


@pytest.mark.parametrize("revision", ["HEAD", "73b4172", "0" * 40, True])
def test_wrong_or_symbolic_revision_rejects(release, revision):
    with pytest.raises(ClosedWorkerProtocolError):
        capture_native_source_snapshot(release[0], "warp", repository_revision=revision)


@pytest.mark.parametrize("mutation", ["source", "catalog_rehash", "missing", "reorder", "extra"])
def test_release_mutations_reject_before_snapshot(release, tmp_path, mutation):
    root = tmp_path / "release"
    shutil.copytree(release[0], root)
    path = root / "optical_worker/cases.py"
    catalog_path = root / CATALOG_PATH
    catalog = json.loads(catalog_path.read_text())
    if mutation in {"source", "catalog_rehash"}:
        path.write_bytes(path.read_bytes() + b"\n# replaced source\n")
        if mutation == "catalog_rehash":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            next(e for e in catalog["source_digests"] if e["relative_path"] == "optical_worker/cases.py")[
                "sha256"
            ] = digest
    elif mutation == "missing":
        path.unlink()
    elif mutation == "reorder":
        catalog["lanes"]["warp"].reverse()
    else:
        catalog["lanes"]["warp"].append("benchmarks/optical/native.py")
    if mutation in {"catalog_rehash", "reorder", "extra"}:
        catalog_path.write_bytes(canonical_json_bytes(catalog) + b"\n")
    with pytest.raises(ClosedWorkerProtocolError):
        _snapshot((root, release[1]))


def test_snapshot_revalidation_refuses_drift_or_fabricated_files(release, tmp_path):
    root = tmp_path / "release"
    shutil.copytree(release[0], root)
    snapshot = _snapshot((root, release[1]))
    with pytest.raises(ClosedWorkerProtocolError, match="changed after capture"):
        replace(snapshot, sources=snapshot.sources[:-1]).verify_unchanged()
    path = root / "optical_worker/cases.py"
    path.write_bytes(path.read_bytes() + b"\n# drift\n")
    with pytest.raises(ClosedWorkerProtocolError, match="manifest"):
        snapshot.verify_unchanged()


@pytest.mark.parametrize("mutation", [
    "missing", "duplicate", "order", "unknown_field", "bad_digest", "policy",
])
def test_even_committed_catalog_must_match_literal_schema(release, tmp_path, mutation):
    root = tmp_path / "release"
    shutil.copytree(release[0], root)
    path = root / CATALOG_PATH
    catalog = json.loads(path.read_text())
    entries = catalog["source_digests"]
    if mutation == "missing":
        entries.pop()
    elif mutation == "duplicate":
        entries.append(entries[0])
    elif mutation == "order":
        entries.reverse()
    elif mutation == "unknown_field":
        entries[0]["trusted"] = True
    elif mutation == "bad_digest":
        entries[0]["sha256"] = "X" * 64
    else:
        catalog["ast_policy_sha256"] = "0" * 64
    path.write_bytes(canonical_json_bytes(catalog) + b"\n")
    _git(root, "add", CATALOG_PATH)
    _git(root, "commit", "-qm", "invalid catalog test release")
    with pytest.raises(ClosedWorkerProtocolError):
        _snapshot((root, _git(root, "rev-parse", "HEAD")))


@pytest.mark.parametrize("directory", [False, True])
def test_source_snapshot_rejects_symlink_components(release, tmp_path, directory):
    root = tmp_path / "release"
    shutil.copytree(release[0], root)
    target = root / "optical_worker" / ("data" if directory else "cases.py")
    retained = tmp_path / target.name
    target.rename(retained)
    try:
        target.symlink_to(retained, target_is_directory=directory)
    except OSError as exc:
        pytest.skip(f"symlink privilege unavailable: {exc}")
    with pytest.raises(ClosedWorkerProtocolError, match="redirects"):
        _snapshot((root, release[1]))


@pytest.mark.parametrize("mutation", ["name", "file", "spec", "bytecode", "missing_file"])
def test_actual_loaded_module_metadata_drift_rejects(release, mutation):
    code = {
        "name": "module.__name__ = 'optical_worker.fake'",
        "file": "module.__file__ = str(Path(module.__file__).parent / 'cases.py')",
        "spec": "module.__spec__.origin = '/outside/native_contract.py'",
        "bytecode": "module.__file__ = module.__file__ + 'c'",
        "missing_file": "del module.__file__",
    }[mutation]
    _cold(release, f"""
import optical_worker.native_contract as module
{code}
try:
    snapshot.check_loaded_modules()
except ClosedWorkerProtocolError:
    pass
else:
    raise AssertionError('changed loaded-module metadata accepted')
""", cpu_only=True)


def test_actual_import_outside_literal_catalog_rejects(release, tmp_path):
    root = tmp_path / "release"
    shutil.copytree(release[0], root)
    # An actual import, not a caller-filled list of module names.
    (root / "optical_worker" / "unlisted.py").write_text("value = 1\n")
    _cold((root, release[1]), """
import optical_worker.unlisted
try:
    snapshot.check_loaded_modules()
except ClosedWorkerProtocolError as error:
    assert 'outside lane' in str(error)
else:
    raise AssertionError('unlisted real import accepted')
""", cpu_only=True)


def test_other_root_import_with_expected_name_rejects(release, tmp_path):
    other = tmp_path / "other"
    (other / "optics").mkdir(parents=True)
    (other / "optics" / "__init__.py").write_text("value = 1\n")
    _cold(release, f"""
sys.path.insert(0, {str(other)!r})
import optics
try:
    snapshot.check_loaded_modules()
except ClosedWorkerProtocolError as error:
    assert 'different root' in str(error)
else:
    raise AssertionError('other checkout import accepted')
""", cpu_only=True)


def test_no_production_activation_or_c0_source_change():
    path = ROOT / "benchmarks/results/q54_p12_4/closed_worker_runtime_contract.json"
    contract = json.loads(path.read_text())
    assert contract["gpu_preparation_enabled"] is False
    assert "optical_worker/native_sources.py" not in contract["trusted_source_paths"]
    for path in ("optical_worker/worker.py", "tools/optical_closed_worker.py"):
        assert "native_sources" not in (ROOT / path).read_text()
