"""The compatibility path and native path must share one Warp kernel source."""

from __future__ import annotations

import ast
from pathlib import Path

from optics import warp_execution, warp_native_render

_SHARED_ENTRIES = (
    "_brute_force_first_hit_kernel",
    "_device_scene_first_hit_kernel",
    "_device_scene_first_hit_aabb_kernel",
    "_pinhole_camera_raygen_kernel",
    "_init_bvh_hit_mask_kernel",
    "_device_scene_bvh_first_hit_kernel",
    "warp_direct_light_shade_entry",
)


def _defined_functions(module: object) -> set[str]:
    source = Path(module.__file__).read_text(encoding="utf-8")
    return {
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_compatibility_executor_uses_exact_shared_kernel_objects() -> None:
    for name in _SHARED_ENTRIES:
        assert getattr(warp_execution, name) is getattr(warp_native_render, name)


def test_legacy_executor_does_not_define_a_second_kernel_source() -> None:
    old_definitions = _defined_functions(warp_execution)
    native_definitions = _defined_functions(warp_native_render)
    for name in _SHARED_ENTRIES:
        assert name not in old_definitions
        assert name in native_definitions


def test_runtime_flag_shade_entry_is_removed_not_retained_as_a_second_source():
    assert "_device_scene_direct_light_kernel" not in _defined_functions(warp_native_render)
    assert not hasattr(warp_execution, "_device_scene_direct_light_kernel")
