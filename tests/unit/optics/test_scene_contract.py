from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from optics.device import pack_source_order_key
from optics.registry import OpticalInstanceSpec, OpticalLightSpec, OpticalMaterialSpec, OpticalWorldRegistry
from optics.scene_contract import SCENE_ARRAY_SEMANTICS, PackedOpticalSceneHost


def test_packed_scene_is_stable_read_only_and_semantically_complete():
    first = PackedOpticalSceneHost.from_registry(
        _registry(),
        source_contract_id="asymmetric_synthetic_7x3_v1",
        source_contract_sha256="1" * 64,
        scene_id="asymmetric_triangle_material_grid_v1",
    )
    second = PackedOpticalSceneHost.from_registry(
        _registry(),
        source_contract_id="asymmetric_synthetic_7x3_v1",
        source_contract_sha256="1" * 64,
        scene_id="asymmetric_triangle_material_grid_v1",
    )

    assert first.scene_contract_sha256 == second.scene_contract_sha256
    assert set(first.arrays) == set(SCENE_ARRAY_SEMANTICS)
    assert len(set(first.array_semantics.values())) == len(first.array_semantics)
    assert first.num_instances == 2
    assert first.num_triangles == 2  # retain source faces; world queries reject degeneracy
    assert first.num_planes == 1
    assert first.num_lights == 1  # disabled lights are not materialized
    assert first.max_body_index == 2
    assert first.role_to_mask == {"depth": 1, "rgb": 2}
    np.testing.assert_array_equal(
        first.arrays["triangle_source_order_key"],
        [pack_source_order_key(0, 0), pack_source_order_key(0, 1)],
    )
    np.testing.assert_array_equal(
        first.arrays["plane_source_order_key"],
        [pack_source_order_key(1, 0)],
    )

    for name, value in first.arrays.items():
        assert value.flags.c_contiguous, name
        assert not value.flags.writeable, name
        np.testing.assert_array_equal(value, second.arrays[name])
        if value.size:
            with pytest.raises(ValueError):
                value.flat[0] = value.flat[0]


def test_scene_hash_binds_case_and_scene_provenance():
    common = dict(
        registry=_registry(),
        source_contract_sha256="2" * 64,
        scene_id="scene_a",
    )
    case_a = PackedOpticalSceneHost.from_registry(source_contract_id="case_a", **common)
    case_b = PackedOpticalSceneHost.from_registry(source_contract_id="case_b", **common)

    assert case_a.scene_contract_sha256 != case_b.scene_contract_sha256


def test_scene_contract_module_has_no_torch_warp_or_benchmark_imports():
    source = Path("optics/scene_contract.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert "torch" not in roots
    assert "warp" not in roots
    assert "benchmarks" not in roots


def _registry() -> OpticalWorldRegistry:
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat", albedo_rgb=(0.2, 0.4, 0.6)))
    registry.add_light(
        OpticalLightSpec(
            "sun",
            "directional",
            [1.0, 2.0, 3.0],
            intensity=2.5,
            color_rgb=(0.9, 0.8, 0.7),
        )
    )
    registry.add_light(
        OpticalLightSpec("disabled", "point", [0.0, 0.0, 1.0], enabled=False)
    )
    registry.add_triangle_mesh_geometry(
        "mesh",
        vertices_local=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ),
        triangles=np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64),
    )
    registry.add_plane_geometry(
        "plane",
        normal_local=[0.0, 0.0, 1.0],
        point_local=[0.0, 0.0, -1.0],
    )
    registry.add_instance(
        OpticalInstanceSpec(
            "mesh_instance",
            "mesh",
            "mat",
            body_index=2,
            roles=frozenset({"rgb", "depth"}),
        )
    )
    registry.add_instance(
        OpticalInstanceSpec(
            "plane_instance",
            "plane",
            "mat",
            roles=frozenset({"rgb"}),
        )
    )
    return registry
