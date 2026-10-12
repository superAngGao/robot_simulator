"""C2c1 single-owner extraction; no worker admission or new render contract."""

from __future__ import annotations

import ast
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.structural_ast import AST_ENCODING, structural_ast, structural_ast_sha256

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


def _normalize_capture_io(node, module):
    """Exact C2c2 IO substitutions only; retain the historical mathematical hash."""
    replacements = []
    if module == "optics.mjcf_visual" and node.name == "import_mjcf_visual_scene":
        signature = ast.parse("def f(*, file_bytes: Mapping[str, bytes] | None = None): pass").body[0].args
        assert structural_ast(node.args.kwonlyargs[-1]) == structural_ast(signature.kwonlyargs[0])
        assert structural_ast(node.args.kw_defaults[-1]) == structural_ast(signature.kw_defaults[0])
        node.args.kwonlyargs.pop()
        node.args.kw_defaults.pop()
        replacements = [
            ("captured = _copy_mjcf_bytes(file_bytes)", None),
            (
                "root = ET.parse(xml_path).getroot() if captured is None else ET.fromstring("
                "_captured_file(captured, model_dir, xml_path))",
                "root = ET.parse(xml_path).getroot()",
            ),
            (
                "loaded = _load_visual_mesh(trimesh, mesh.path, model_dir, captured)",
                "loaded = trimesh.load_mesh(mesh.path, process=False)",
            ),
        ]
    elif module == "optical_worker.workloads" and node.name == "build_canonical_packed_scene":
        replacements = [
            ("imported_bounds = None", None),
            ("model_dir = Path(go2_model_dir).resolve()", None),
            (
                "captured = capture_case_asset(case_id, repo_root=model_dir.parent, model_dir=model_dir, "
                "expected_evidence=asset_evidence)",
                None,
            ),
            (
                'imported = import_mjcf_visual_scene(model_dir, model_xml="go2.xml", '
                'file_bytes=captured.files)',
                'registry = import_mjcf_visual_scene(Path(go2_model_dir), model_xml="go2.xml").registry',
            ),
            ("registry = imported.registry", None),
            (
                "imported_bounds = tuple(tuple(float(value) for value in bound) "
                "for bound in (imported.bounds_min, imported.bounds_max))",
                None,
            ),
            ("return (packed, imported_bounds, registry)", "return packed"),
        ]
    elif module == "optical_worker.workloads" and node.name == "build_canonical_fixed_fixture":
        # Approved double-query migration 3c1a08f changed these two empty
        # transform inputs. Normalize only those exact keyword expressions;
        # keep every nonempty mathematical statement under the original hash.
        expected = {
            "x_world_R": "np.empty((0, 3, 3), dtype=np.float64)",
            "x_world_r": "np.empty((0, 3), dtype=np.float64)",
        }
        found = set()
        for child in ast.walk(node):
            if isinstance(child, ast.keyword) and child.arg in expected:
                assert child.arg not in found
                current = ast.parse(expected[child.arg], mode="eval").body
                assert structural_ast(child.value) == structural_ast(current)
                previous = expected[child.arg].replace("np.float64", "np.float32")
                child.value = ast.parse(previous, mode="eval").body
                found.add(child.arg)
        assert found == set(expected)
    for source, replacement in replacements:
        target = structural_ast(ast.parse(source).body[0])
        matched = 0
        for parent in ast.walk(node):
            for _, children in ast.iter_fields(parent):
                if not isinstance(children, list):
                    continue
                for index, child in list(enumerate(children)):
                    if isinstance(child, ast.stmt) and structural_ast(child) == target:
                        children[index : index + 1] = (
                            [] if replacement is None else ast.parse(replacement).body
                        )
                        matched += 1
        assert matched == 1, f"unexpected capture IO delta: {source}"


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
    assert BASELINE["ast_encoding"] == AST_ENCODING
    tree = ast.parse((ROOT / _path(new)).read_text(encoding="utf-8"))
    definitions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    for name in _names(old, selected):
        node = definitions[name]
        if new == "optical_worker.workloads" and name == "build_canonical_packed_scene":
            # Only the public wrapper delegates; the sole materializer retains
            # every historical math statement. Validate the delegation exactly.
            assert len(node.body) == 2
            assert structural_ast(node.body[-1]) == structural_ast(ast.parse(
                "return _materialize_canonical_scene(case_id, go2_model_dir=go2_model_dir, "
                "asset_evidence=asset_evidence)[0]"
            ).body[0])
            original = node
            node = definitions["_materialize_canonical_scene"]
            assert structural_ast(node.args) == structural_ast(original.args)
            node.name = original.name
            node.returns = original.returns
            node.body[0] = original.body[0]
        _normalize_capture_io(node, new)
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
        actual = structural_ast_sha256(node)
        assert actual == BASELINE["sources"][_path(old)]["definitions"][name]["ast_sha256"]


def test_structural_ast_retains_empty_fields_and_optional_values():
    tree = ast.parse("def f(): pass")
    encoded = structural_ast(tree.body[0])
    fields = dict(encoded[1])
    assert fields["decorator_list"] == []
    assert fields["returns"] is None
    assert dict(fields["args"][1])["defaults"] == []
    # PEP 695's absent/empty field is the sole semantic no-op normalization.
    without_field = ast.parse("def f(): pass")
    if hasattr(without_field.body[0], "type_params"):
        del without_field.body[0].type_params
    assert structural_ast_sha256(without_field) == structural_ast_sha256(tree)


@pytest.mark.parametrize(
    "changed",
    [
        "def f(a, b=1): return a - b",
        "def f(b, a=1): return a + b",
        "def f(a, b=2): return a + b",
        "def f(a, b=1):\n from another_owner import value\n return a + b",
    ],
)
def test_structural_ast_rejects_real_algorithm_argument_default_and_import_drift(changed):
    original = ast.parse("def f(a, b=1): return a + b")
    assert structural_ast_sha256(ast.parse(changed)) != structural_ast_sha256(original)


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
