"""C2c1 single-owner extraction; no worker admission or new render contract."""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
BASELINE = json.loads(
    (Path(__file__).parent / "data/cpu_materialization_extraction_baseline.json").read_text(encoding="utf-8")
)
MOVES = (
    ("benchmarks.optical.workloads", "optical_worker.workloads", None),
    ("benchmarks.optical.assets", "optical_worker.assets", None),
    ("tools.optical_pipeline_lab.camera_presets", "optics.camera_presets", None),
    (
        "examples.mujoco_menagerie_robot_preview",
        "optics.mjcf_visual",
        (
            "ImportedMjcfVisualScene",
            "_MeshAsset",
            "import_mjcf_visual_scene",
            "_read_mesh_assets",
            "_walk_body",
            "_body_transform",
            "_add_visual_geom",
            "_add_materials",
            "_add_lights",
            "_geom_material_id",
            "_home_joint_values",
            "_parse_vec",
            "_quat_to_R",
            "_axis_angle_R",
        ),
    ),
)


def _path(module):
    return module.replace(".", "/") + ".py"


def _names(old, selected):
    return tuple(BASELINE["sources"][_path(old)]["definitions"]) if selected is None else selected


@pytest.mark.parametrize("old,new,selected", MOVES)
def test_every_moved_definition_is_the_same_object_at_legacy_import(old, new, selected):
    legacy, production = importlib.import_module(old), importlib.import_module(new)
    legacy_tree = ast.parse((ROOT / _path(old)).read_text(encoding="utf-8"))
    legacy_definitions = {
        node.name for node in ast.walk(legacy_tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    for name in _names(old, selected):
        assert getattr(legacy, name) is getattr(production, name)
        assert getattr(production, name).__module__ == new
        assert name not in legacy_definitions, f"second implementation at {old}.{name}"


@pytest.mark.parametrize("old,new,selected", MOVES)
def test_moved_ast_matches_frozen_pre_extraction_definition(old, new, selected):
    tree = ast.parse((ROOT / _path(new)).read_text(encoding="utf-8"))
    definitions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    for name in _names(old, selected):
        node = definitions[name]
        # This sole import changes ownership, not the algorithm. Normalize only
        # the exact approved module/name/alias triple; no blanket import removal.
        for child in ast.walk(node):
            if (
                isinstance(child, ast.ImportFrom)
                and child.module == "optics.mjcf_visual"
                and child.level == 0
                and [(item.name, item.asname) for item in child.names] == [("import_mjcf_visual_scene", None)]
            ):
                child.module = "examples.mujoco_menagerie_robot_preview"
        actual = hashlib.sha256(ast.dump(node, include_attributes=False).encode("utf-8")).hexdigest()
        assert actual == BASELINE["sources"][_path(old)]["definitions"][name]["ast_sha256"]


@pytest.mark.parametrize(
    "case_id",
    ["asymmetric_synthetic_7x3_v1", "synthetic_dynamic_rigid_960x640_v1", "physics_body_triangle_960x640_v1"],
)
def test_production_materializers_preserve_frozen_scene_replay_fixture_identities(case_id):
    from optical_worker.workload_contract import frozen_workload_content
    from optical_worker.workloads import (
        build_canonical_fixed_fixture,
        build_canonical_packed_scene,
        build_canonical_transform_replay,
    )

    packed = build_canonical_packed_scene(case_id)
    expected = frozen_workload_content(case_id)
    assert packed.scene_contract_sha256 == expected.scene_contract_sha256
    if case_id.startswith("asymmetric"):
        fixture = build_canonical_fixed_fixture(packed)
        assert fixture.fixed_render_fixture_sha256 == expected.fixed_render_fixture_sha256
    elif case_id.startswith("synthetic_dynamic"):
        replay = build_canonical_transform_replay(case_id)
        assert replay.transform_replay_sha256 == expected.transform_replay_sha256


def test_production_go2_still_requires_real_preflighted_assets():
    from optical_worker.assets import AssetPreflightError, preflight_case_asset
    from optical_worker.workloads import build_canonical_packed_scene

    with pytest.raises(ValueError, match="preflighted"):
        build_canonical_packed_scene("go2_static_960x640_v1")
    with pytest.raises(AssetPreflightError):
        preflight_case_asset("go2_static_960x640_v1", repo_root=ROOT / "absent-c2c1-assets")


def test_production_materialization_does_not_attempt_benchmark_example_tool_or_gpu_imports():
    script = r"""
import importlib.abc
import sys
from types import SimpleNamespace
import numpy as np

blocked = {"benchmarks", "examples", "tools", "torch", "warp"}
attempts = []
class DenyForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            attempts.append(fullname)
            raise AssertionError("forbidden import: " + fullname)
sys.meta_path.insert(0, DenyForbidden())
assert not blocked.intersection(sys.modules)
from optical_worker.workloads import build_canonical_packed_scene, build_canonical_fixed_fixture
from optical_worker.assets import preflight_case_asset
from optics.mjcf_visual import import_mjcf_visual_scene
from optics.camera_presets import build_lab_video_camera

packed = build_canonical_packed_scene("asymmetric_synthetic_7x3_v1")
build_canonical_fixed_fixture(packed)
assert preflight_case_asset("asymmetric_synthetic_7x3_v1", repo_root=".") is None
scene = SimpleNamespace(bounds_min=np.array([-1., -2., 0.]), bounds_max=np.array([2., 1., 1.]),
                        frame=SimpleNamespace(frame_id=1, sim_time=0.))
args = SimpleNamespace(video_mode="camera_orbit", video_frames=120, width=960, height=640)
for i in range(120):
    camera = build_lab_video_camera(scene, args, i)
    assert np.isfinite(camera.X_world_camera.R).all()
    assert np.isfinite(camera.X_world_camera.r).all()
assert attempts == []
assert not blocked.intersection(sys.modules)
for name in sorted(blocked):
    try:
        __import__(name)
    except AssertionError:
        pass
    else:
        raise AssertionError("negative control did not trip")
assert len(attempts) == len(blocked)
print("production CPU closure: passed")
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "production CPU closure: passed" in result.stdout


def test_extraction_does_not_expose_example_cli_as_production_importer():
    from optics import mjcf_visual

    assert not {"main", "render_imported_scene", "make_model_camera", "_parse_args"}.intersection(
        vars(mjcf_visual)
    )
