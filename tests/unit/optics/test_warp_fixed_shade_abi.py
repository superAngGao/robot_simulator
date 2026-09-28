"""The fixed Warp dispatch ABI is inspectable without recording or launching."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from optics import warp_native_render as native
from optics.output_contract import SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES


def _kernels():
    tree = ast.parse(Path(native.__file__).read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(isinstance(d, ast.Attribute) and d.attr == "kernel" for d in node.decorator_list)
    }


@pytest.mark.parametrize("shadows", [False, True])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_four_actual_kernel_signatures_match_exact_fixed_abi(shadows, diagnostics):
    entry, names = native.WARP_DIRECT_LIGHT_SHADE_ABIS[shadows, diagnostics]
    suffix = "diagnostics" if diagnostics else "primary"
    mode = "with_shadows" if shadows else "no_shadow"
    name = f"warp_direct_light_{mode}_{suffix}_v1"
    assert getattr(native, name) is entry
    assert tuple(arg.arg for arg in _kernels()[name].args.args) == names
    assert "shadows" not in names
    assert "shadow_traversal_stats_enabled" not in names
    assert "shadow_stack_overflow_count" in names
    assert "shadow_max_stack_depth" in names
    assert ("sensor_role_mask" in names) is shadows
    assert ("shadow_bias" in names) is shadows
    assert any(n.startswith("bvh_") for n in names) is shadows
    assert any(n.startswith(("plane_", "triangle_")) for n in names) is shadows
    overlay = tuple(n for n in names if n.startswith("shadow_traversal_"))
    assert overlay == (SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES if diagnostics else ())
    if diagnostics:
        assert names[-9:] == SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES
    if native._HAS_WARP:
        assert (
            native.warp_direct_light_shade_entry(
                shadows=shadows,
                traversal_diagnostics=diagnostics,
            )
            is native.WARP_DIRECT_LIGHT_SHADE_ABIS[shadows, diagnostics]
        )


@pytest.mark.parametrize("bad", [0, 1, None, "true", ()])
def test_selector_rejects_non_bool_before_any_runtime_work(bad):
    with pytest.raises(TypeError, match="exact bools"):
        native.warp_direct_light_shade_entry(shadows=bad, traversal_diagnostics=False)
    with pytest.raises(TypeError, match="exact bools"):
        native.warp_direct_light_shade_entry(shadows=False, traversal_diagnostics=bad)


def test_reset_entries_have_only_fixed_independent_scalar_parameters():
    kernels = _kernels()
    primary = ("shadow_stack_overflow_count", "shadow_max_stack_depth")
    contracts = {
        "_reset_bvh_primary_counters_kernel": ("bvh_stack_overflow_count", "bvh_max_stack_depth"),
        "_reset_shadow_primary_counters_kernel": primary,
        "_reset_shadow_diagnostic_counters_kernel": primary + SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES,
    }
    for name, parameters in contracts.items():
        kernel = kernels[name]
        assert tuple(arg.arg for arg in kernel.args.args) == parameters
        # Reset writes exactly one [0] zero per input; no loop, allocation or launch.
        assert len(kernel.body) == len(parameters)
        for statement, parameter in zip(kernel.body, parameters, strict=True):
            assert isinstance(statement, ast.Assign)
            assert ast.unparse(statement) == f"{parameter}[0] = wp.int32(0)"


def test_variants_delegate_math_to_one_shadowed_or_unshadowed_function():
    kernels = _kernels()
    for (shadows, _), (entry, _) in native.WARP_DIRECT_LIGHT_SHADE_ABIS.items():
        function = entry.func if native._HAS_WARP else entry
        kernel = kernels[function.__name__]
        calls = [
            n.func.id for n in ast.walk(kernel) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        ]
        expected = "_shade_shadowed_ray" if shadows else "_shade_unshadowed_ray"
        assert calls.count(expected) == 1
        assert calls.count("_direct_light_write") == 1
