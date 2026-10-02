"""CPU custody tests; miniature files test mechanics, never canonical Go2 parity."""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from optical_worker import assets
from optics import mjcf_visual


@pytest.fixture
def asset_checkout(tmp_path, monkeypatch):
    checkout = tmp_path / "mujoco_menagerie"
    model = checkout / "unitree_go2"
    (model / "assets").mkdir(parents=True)
    content = {
        "assets/body.obj": b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        "go2.xml": (
            b'<mujoco><compiler meshdir="assets"/><asset><mesh name="body" file="body.obj"/>'
            b'</asset><worldbody><body><geom mesh="body" class="visual"/></body></worldbody></mujoco>'
        ),
        "scene.xml": b"<mujoco/>\n",
    }
    for name, raw in content.items():
        (model / name).write_bytes(raw)

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(checkout), *args], check=True, capture_output=True, text=True, timeout=10
        ).stdout.strip()

    git("init")
    git("config", "core.autocrlf", "false")
    git("config", "user.name", "CPU custody test")
    git("config", "user.email", "custody@example.invalid")
    git("remote", "add", "origin", assets.GO2_SOURCE_REPOSITORY)
    git("add", "unitree_go2")
    git("commit", "-m", "mechanics fixture, not canonical Go2")
    revision = git("rev-parse", "HEAD")
    records = [
        {"relative_path": name, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        for name, raw in sorted(content.items())
    ]
    aggregate = assets._canonical_sha256(records)
    manifest = {
        "schema_version": assets.ASSET_MANIFEST_SCHEMA_VERSION,
        "requirement_id": assets.GO2_ASSET_REQUIREMENT,
        "source_repository": assets.GO2_SOURCE_REPOSITORY,
        "source_revision": revision,
        "entrypoint": "scene.xml",
        "files": records,
        "aggregate_sha256": aggregate,
    }
    (model / assets.DEFAULT_ASSET_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    # Only this mechanics fixture patches frozen constants. Real Go2 acceptance
    # runs separately without monkeypatch, through the same production functions.
    monkeypatch.setattr(assets, "GO2_FROZEN_SOURCE_REVISION", revision)
    monkeypatch.setattr(assets, "GO2_FROZEN_ASSET_AGGREGATE_SHA256", aggregate)
    return tmp_path, model, content


def capture(fixture, **kwargs):
    root, model, _ = fixture
    return assets.capture_case_asset("go2_static_960x640_v1", repo_root=root, model_dir=model, **kwargs)


def test_capture_is_complete_immutable_and_independent_of_future_disk_changes(asset_checkout):
    _, model, content = asset_checkout
    captured = capture(asset_checkout)
    assert dict(captured.files) == content
    with pytest.raises(TypeError):
        captured.files["go2.xml"] = b"other"
    (model / "go2.xml").write_bytes(b"later disk change")
    assert captured.files["go2.xml"] == content["go2.xml"]


def test_expected_evidence_is_only_a_constraint_not_authority(asset_checkout):
    evidence = capture(asset_checkout).evidence
    with pytest.raises(assets.AssetPreflightError, match="supplied evidence"):
        capture(asset_checkout, expected_evidence=replace(evidence, manifest_sha256="0" * 64))
    with pytest.raises(assets.AssetPreflightError, match="exact production type"):
        capture(asset_checkout, expected_evidence=SimpleNamespace(**evidence.__dict__))
    (asset_checkout[1] / "scene.xml").write_bytes(b"changed")
    with pytest.raises(assets.AssetPreflightError, match="modified tracked"):
        capture(asset_checkout, expected_evidence=evidence)


@pytest.mark.parametrize("name", ["go2.xml", "assets/body.obj", "scene.xml"])
def test_changed_bytes_between_preflight_and_capture_fail(asset_checkout, monkeypatch, name):
    original = assets._read_capture_file

    def changed(root, relative):
        raw = original(root, relative)
        return raw + b"x" if relative.as_posix() == name else raw

    monkeypatch.setattr(assets, "_read_capture_file", changed)
    with pytest.raises(assets.AssetPreflightError, match="bytes changed after preflight"):
        capture(asset_checkout)


def test_changed_manifest_between_preflight_and_capture_fails(asset_checkout, monkeypatch):
    original = assets._read_capture_file

    def changed(root, relative):
        raw = original(root, relative)
        return raw + b" " if relative.name == assets.DEFAULT_ASSET_MANIFEST_NAME else raw

    monkeypatch.setattr(assets, "_read_capture_file", changed)
    with pytest.raises(assets.AssetPreflightError, match="manifest changed"):
        capture(asset_checkout)


def test_late_inventory_change_fails_final_preflight(asset_checkout, monkeypatch):
    original = assets._read_capture_file

    def changed(root, relative):
        raw = original(root, relative)
        if relative.name == assets.DEFAULT_ASSET_MANIFEST_NAME:
            (root / "unexpected.dat").write_bytes(b"unexpected")
        return raw

    monkeypatch.setattr(assets, "_read_capture_file", changed)
    with pytest.raises(assets.AssetPreflightError, match="every and only model file"):
        capture(asset_checkout)


@pytest.mark.parametrize("reference", ["../outside.obj", "/outside.obj", "missing.obj"])
def test_captured_mesh_cannot_fall_back_to_disk(reference):
    files = {"assets/body.obj": b"abc"}
    with pytest.raises(ValueError, match="captured"):
        mjcf_visual._captured_file(files, Path("model"), Path("model") / reference)


@pytest.mark.parametrize(
    "key,value",
    [("../x", b"x"), ("/x", b"x"), ("a\\b", b"x"), ("C:x", b"x"), ("a/./b", b"x"), ("a", bytearray(b"x"))],
)
def test_captured_input_names_and_bytes_are_strict(key, value):
    with pytest.raises(ValueError, match="immutable bytes"):
        mjcf_visual._copy_mjcf_bytes({key: value})


def test_loader_receives_captured_stream_and_no_external_resolver():
    calls = []

    def load_mesh(source, **kwargs):
        calls.append((source.read(), kwargs))
        return "loaded"

    result = mjcf_visual._load_visual_mesh(
        SimpleNamespace(load_mesh=load_mesh),
        Path("model/assets/a.obj"),
        Path("model"),
        {"assets/a.obj": b"verified bytes"},
    )
    assert result == "loaded"
    assert calls == [(b"verified bytes", {"file_type": "obj", "process": False, "resolver": {}})]


def test_actual_importer_uses_captured_xml_and_obj_without_reopening(asset_checkout, monkeypatch):
    pytest.importorskip("trimesh")  # Real parser is mandatory in robot_sim acceptance.
    _, model, _ = asset_checkout
    original = mjcf_visual.import_mjcf_visual_scene(model, model_xml="go2.xml")
    captured = capture(asset_checkout)
    for name in captured.files:
        (model / name).unlink()

    def deny(*args, **kwargs):
        raise AssertionError("captured importer must not reopen a file")

    monkeypatch.setattr(Path, "open", deny)
    monkeypatch.setattr("builtins.open", deny)
    result = mjcf_visual.import_mjcf_visual_scene(model, model_xml="go2.xml", file_bytes=captured.files)
    assert result.num_triangles == original.num_triangles == 1
    np.testing.assert_array_equal(result.bounds_min, original.bounds_min)
    np.testing.assert_array_equal(result.bounds_max, original.bounds_max)
    np.testing.assert_array_equal(
        result.registry.geometry["body"].vertices_local, original.registry.geometry["body"].vertices_local
    )


@pytest.mark.skipif(os.name != "posix", reason="actual POSIX symlink/FIFO fault tests run on robot_sim")
def test_capture_rejects_symlink_and_fifo_without_blocking(tmp_path):
    (tmp_path / "real").write_bytes(b"real")
    (tmp_path / "link").symlink_to(tmp_path / "real")
    with pytest.raises(assets.AssetPreflightError, match="symlink"):
        assets._read_capture_file(tmp_path, Path("link"))
    os.mkfifo(tmp_path / "fifo")
    code = (
        "import sys; from pathlib import Path; from optical_worker import assets\n"
        "try: assets._read_capture_file(Path(sys.argv[1]), Path('fifo'))\n"
        "except assets.AssetPreflightError as e: assert 'not a regular file' in str(e)\n"
        "else: raise AssertionError('FIFO accepted')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=10
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
