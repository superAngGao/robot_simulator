"""CPU checks for the actual shared recipe and compatibility wrappers."""

import ast
import hashlib
import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

from optics.cuda_extension_recipes import CUDA_EXTENSION_RECIPES, cuda_extension_kwargs


def test_original_kernel_bytes_and_entry_abis_are_preserved():
    expected = (
        ("7bec3f1f2ec03fa441924a699a7d276795fa2ec10dd845002ed7a03a0cd7c5b6",
         "b8f33c61b8c68ae5072aa298523eec3557f12c69bf24d307b474a988fd0b5831"),
        ("65783ded7966b45f71c4dd4330530c3f20817622d86710a9ac9a3d72511c354c",
         "3643883860c27fbff71e54817855015890753e454628976e0cf4c05fc518f79e"),
        ("2f9b088f191936092412a908137e767c906f3b0fcfe7505ef0f88709af55b0b0",
         "9c321f37665ae1fbb589f44e6170f588e90835532347523fc757fa315df79f4c"),
    )
    for recipe, hashes in zip(CUDA_EXTENSION_RECIPES, expected, strict=True):
        cpp, cuda = recipe.source_module.CPP_SOURCE, recipe.source_module.CUDA_SOURCE
        assert tuple(hashlib.sha256(s.encode()).hexdigest() for s in (cpp, cuda)) == hashes
        for name in recipe.entries:
            assert name in cpp and name in cuda
        if not recipe.generate_bindings:
            assert tuple(re.findall(r'm\.def\(\s*"([^"]+)"', cpp)) == recipe.entries


def test_cold_recipe_import_has_no_backend_or_compatibility_side_effect():
    probe = subprocess.run(
        [sys.executable, "-B", "-c", "import sys; import optics.cuda_extension_recipes; "
         "assert not any(n == 'torch' or n.startswith('torch.') or n == 'warp' or "
         "n.startswith('warp.') for n in sys.modules); "
         "assert 'optics.cuda_direct_light' not in sys.modules; "
         "assert 'optics.cuda_scene' not in sys.modules; "
         "assert 'optics.cuda_lbvh' not in sys.modules"],
        capture_output=True, text=True, timeout=30,
    )
    assert probe.returncode == 0, probe.stderr


@pytest.mark.parametrize("lane,require_name", (
    ("scene", "_torch_module"), ("lbvh", "_require_torch_cuda_lbvh"),
    ("direct_light", "_require_cuda_direct_light_deps"),
))
def test_actual_legacy_wrapper_delegates_once_and_keeps_source_aliases(monkeypatch, lane, require_name):
    module = importlib.import_module("optics.cuda_" + lane)
    recipe = next(r for r in CUDA_EXTENSION_RECIPES if r.source_module.__name__ == "optics._cuda_" + lane + "_source")
    sentinel = object()
    calls = []

    def load_inline(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(module, require_name, lambda: None)
    if lane == "scene":
        # Scene intentionally resolves Torch only when its legacy loader runs.
        from types import ModuleType
        fake = ModuleType("torch.utils.cpp_extension")
        fake.load_inline = load_inline
        monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", fake)
    else:
        monkeypatch.setattr(module, "load_inline", load_inline)
    loader = getattr(module, "_load_cuda_" + lane + "_extension")
    loader.cache_clear()
    try:
        assert loader() is sentinel and loader() is sentinel
        assert calls == [cuda_extension_kwargs(recipe.name)]
    finally:
        loader.cache_clear()
    prefix = "" if lane == "lbvh" else "_"
    assert getattr(module, prefix + "CPP_SOURCE") is recipe.source_module.CPP_SOURCE
    assert getattr(module, prefix + "CUDA_SOURCE") is recipe.source_module.CUDA_SOURCE
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    assert not any(isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
                   and type(n.value.value) is str and "#include <" in n.value.value for n in tree.body)


def test_recipe_kwargs_cannot_mutate_later_build_or_choose_arbitrary_recipe():
    for recipe in CUDA_EXTENSION_RECIPES:
        original = cuda_extension_kwargs(recipe.name)
        changed = cuda_extension_kwargs(recipe.name)
        for value in changed.values():
            if type(value) is list:
                value.clear()
        assert cuda_extension_kwargs(recipe.name) == original
        assert original["use_pch"] is False and original["is_python_module"] is True
        assert original["extra_cflags"] == original["extra_cuda_cflags"] == ["-O2"]
    for invalid in (None, True, 0, {}, "robot_sim_cuda_scene_v1_v2", "../source"):
        with pytest.raises(ValueError):
            cuda_extension_kwargs(invalid)
