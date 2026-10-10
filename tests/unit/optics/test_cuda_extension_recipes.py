"""CPU checks for the actual shared recipe and compatibility wrappers."""

import ast
import hashlib
import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

from optics.cuda_extension_recipes import CUDA_EXTENSION_RECIPES


def test_fixed_recipe_kernel_bytes_and_entry_abis():
    expected = (
        ("baebdb1d67ea644ccc0b23709a145acbd5c0248dc4b0f660316279d0180ebced",
         "9f54e05e7d1b49af61d6fd0d5ac3fe57afb68b9947cc04b84c6e31855d73da9f"),
        ("65783ded7966b45f71c4dd4330530c3f20817622d86710a9ac9a3d72511c354c",
         "3643883860c27fbff71e54817855015890753e454628976e0cf4c05fc518f79e"),
        ("2f9b088f191936092412a908137e767c906f3b0fcfe7505ef0f88709af55b0b0",
         "2ceea84854157f4b93acbbfce53b500be9d4de6f9a354054a964d20bd003c850"),
    )
    for recipe, hashes in zip(CUDA_EXTENSION_RECIPES, expected, strict=True):
        cpp, cuda = recipe.source_module.CPP_SOURCE, recipe.source_module.CUDA_SOURCE
        # Scene v3 intentionally changes geometry/indices and binary64 arithmetic
        # under input-precision design v4, together with renderer v10.
        # The LBVH source stays unchanged.
        # These pins detect drift; they do not prove numerical correctness.
        original_cpp = cpp
        original_cuda = cuda.replace("#include <torch/types.h>\n", "", 1)
        if recipe.name == "robot_sim_cuda_direct_light_v10":
            original_cpp, binding = cpp.split("\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {", 1)
            original_cuda = original_cuda.replace("#include <cuda.h>\n", "", 1)
            original_cuda = original_cuda.replace("#include <cuda_runtime.h>\n", "", 1)
            wrapped = re.findall(r'm\.def\("([^"]+)", torch::wrap_pybind_function\((\w+)\), "([^"]+)"\);', binding)
            assert wrapped == [(name, name, name) for name in recipe.entries]
        assert tuple(hashlib.sha256(s.encode()).hexdigest() for s in (original_cpp, original_cuda)) == hashes
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
def test_actual_wrapper_rechecks_fixed_resolution_and_keeps_source_aliases(monkeypatch, lane, require_name):
    module = importlib.import_module("optics.cuda_" + lane)
    recipe = next(r for r in CUDA_EXTENSION_RECIPES if r.source_module.__name__ == "optics._cuda_" + lane + "_source")
    sentinel = object()
    calls = []

    def resolve(name, *, extensions, context):
        calls.append((name, extensions, context))
        return sentinel

    monkeypatch.setattr(module, require_name, lambda: None)
    monkeypatch.setattr(module, "load_cuda_extension", resolve)
    loader = getattr(module, "_load_cuda_" + lane + "_extension")
    extensions, context = object(), object()
    assert loader() is sentinel and loader(extensions=extensions, context=context) is sentinel
    assert calls == [(recipe.name, None, None), (recipe.name, extensions, context)]
    prefix = "" if lane == "lbvh" else "_"
    assert getattr(module, prefix + "CPP_SOURCE") is recipe.source_module.CPP_SOURCE
    assert getattr(module, prefix + "CUDA_SOURCE") is recipe.source_module.CUDA_SOURCE
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    assert not any(isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
                   and type(n.value.value) is str and "#include <" in n.value.value for n in tree.body)
