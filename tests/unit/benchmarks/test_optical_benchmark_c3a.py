from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from benchmarks.optical.cases import get_case
from benchmarks.optical.eligibility import (
    CANONICAL_NATIVE_REPLAY_INPUT,
    FIXED_NATIVE_SCENE_INPUT,
    GPU_ENGINE_PUBLISHED_FRAME_INPUT,
)
from benchmarks.optical.native import (
    CanonicalFixedRenderFixtureHost,
    CanonicalNativeInput,
    CanonicalTransformReplayHost,
    WarpPreparedNativeInput,
    prepared_transform_frame_signatures,
    wrap_prepared_fixed_render_inputs,
    wrap_prepared_optical_local_scene,
)
from optics.registry import OpticalInstanceSpec, OpticalMaterialSpec, OpticalWorldRegistry
from optics.scene_contract import PackedOpticalSceneHost


def test_r0_fixture_owns_only_derived_arrays_and_native_union_preserves_identity():
    packed = _packed("asymmetric_synthetic_7x3_v1")
    fixture = _fixture(packed)
    native = CanonicalNativeInput.from_scene_contract(
        packed,
        input_contract=FIXED_NATIVE_SCENE_INPUT,
        fixed_fixture=fixture,
    )

    assert native.manifest.result_capable
    assert native.manifest.scene_contract_sha256 == packed.scene_contract_sha256
    assert native.manifest.fixed_render_fixture_sha256 == fixture.fixed_render_fixture_sha256
    assert set(native.manifest.scene_array_names) == set(packed.arrays)
    assert set(native.manifest.fixed_render_fixture_arrays) == set(fixture.arrays)
    assert set(packed.arrays).isdisjoint(fixture.arrays)
    assert set(packed.array_semantics.values()).isdisjoint(fixture.array_semantics.values())
    for name, value in packed.arrays.items():
        assert native.arrays[name] is value
    for name, value in fixture.arrays.items():
        assert native.arrays[name] is value

    np.testing.assert_allclose(fixture.arrays["triangle_v0_world"], [[0.25, -0.5, 2.0]])
    np.testing.assert_allclose(fixture.arrays["triangle_e1_world"], [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(fixture.arrays["triangle_e2_world"], [[0.0, 1.0, 0.0]])
    np.testing.assert_allclose(fixture.arrays["triangle_normal_world"], [[0.0, 0.0, 1.0]])
    np.testing.assert_array_equal(fixture.arrays["bvh_prim_ids"], [0])
    assert "material_albedo_rgb" not in fixture.arrays
    assert "triangle_material_index" not in fixture.arrays


def test_fixed_fixture_requires_the_exact_packed_owner_even_when_content_is_equal():
    packed_a = _packed("asymmetric_synthetic_7x3_v1")
    packed_b = _packed("asymmetric_synthetic_7x3_v1")
    fixture_a = _fixture(packed_a)

    assert packed_a is not packed_b
    assert packed_a.scene_contract_sha256 == packed_b.scene_contract_sha256
    with pytest.raises(ValueError, match="exact packed scene object"):
        CanonicalNativeInput.from_scene_contract(
            packed_b,
            input_contract=FIXED_NATIVE_SCENE_INPUT,
            fixed_fixture=fixture_a,
        )


@pytest.mark.parametrize("collision_kind", ["name", "semantic"])
def test_fixed_fixture_composition_rejects_duplicate_scene_ownership(collision_kind: str):
    packed = _packed("asymmetric_synthetic_7x3_v1")
    fixture = _fixture(packed)
    forged = _forge_fixture_collision(fixture, packed, collision_kind=collision_kind)

    with pytest.raises(ValueError, match="duplicates native"):
        CanonicalNativeInput.from_scene_contract(
            packed,
            input_contract=FIXED_NATIVE_SCENE_INPUT,
            fixed_fixture=forged,
        )


def test_replay_and_shipping_contracts_are_mutually_exclusive():
    case = get_case("synthetic_dynamic_rigid_960x640_v1")
    packed = _packed(case.case_id)
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (120, 1, 3, 3)).copy()
    translations = np.zeros((120, 1, 3), dtype=np.float32)
    replay = CanonicalTransformReplayHost.from_arrays(
        case_id=case.case_id,
        x_world_R=rotations,
        x_world_r=translations,
    )
    native = CanonicalNativeInput.from_scene_contract(
        packed,
        input_contract=CANONICAL_NATIVE_REPLAY_INPUT,
        replay=replay,
    )

    assert native.manifest.transform_replay_sha256 == replay.transform_replay_sha256
    assert native.manifest.transform_frame_count == 120
    assert native.arrays["replay_x_world_R"] is replay.arrays["replay_x_world_R"]
    assert native.manifest.fixed_render_fixture_sha256 is None
    with pytest.raises(ValueError, match="forbids replay"):
        CanonicalNativeInput.from_scene_contract(
            packed,
            input_contract=GPU_ENGINE_PUBLISHED_FRAME_INPUT,
            replay=replay,
        )


def test_prepared_wrappers_select_exact_storage_without_substitution():
    packed = _packed("asymmetric_synthetic_7x3_v1")
    fixture = _fixture(packed)
    native = CanonicalNativeInput.from_scene_contract(
        packed,
        input_contract=FIXED_NATIVE_SCENE_INPUT,
        fixed_fixture=fixture,
    )
    device_arrays = {name: object() for name in native.arrays}
    prepared = WarpPreparedNativeInput(
        manifest=native.manifest,
        device_ordinal=0,
        arrays=MappingProxyType(device_arrays),
        ready_event=object(),
        source=native,
    )

    scene = wrap_prepared_optical_local_scene(prepared)
    render = wrap_prepared_fixed_render_inputs(prepared)
    for name in native.manifest.scene_array_names:
        assert scene.arrays[name] is device_arrays[name]
        assert render.scene_arrays[name] is device_arrays[name]
    for name in native.manifest.fixed_render_fixture_arrays:
        assert render.derived_fixture_arrays[name] is device_arrays[name]


def test_selected_replay_frame_binds_base_allocation_and_exact_active_range():
    case = get_case("synthetic_dynamic_rigid_960x640_v1")
    packed = _packed(case.case_id)
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (120, 1, 3, 3)).copy()
    translations = np.zeros((120, 1, 3), dtype=np.float32)
    replay = CanonicalTransformReplayHost.from_arrays(
        case_id=case.case_id,
        x_world_R=rotations,
        x_world_r=translations,
    )
    native = CanonicalNativeInput.from_scene_contract(
        packed,
        input_contract=CANONICAL_NATIVE_REPLAY_INPUT,
        replay=replay,
    )
    device_arrays = {
        name: _PointerOnly(index + 0x1000) for index, name in enumerate(native.arrays)
    }
    prepared = WarpPreparedNativeInput(
        manifest=native.manifest,
        device_ordinal=0,
        arrays=MappingProxyType(device_arrays),
        ready_event=object(),
        source=native,
    )

    first = prepared_transform_frame_signatures(prepared, 0)
    seventh = prepared_transform_frame_signatures(prepared, 7)
    assert tuple(value.name for value in seventh) == (
        "replay_x_world_R",
        "replay_x_world_r",
    )
    for first_value, selected in zip(first, seventh, strict=True):
        assert selected.storage_token == first_value.storage_token
        assert selected.active_start == selected.active_count * 7
        assert selected.shape == first_value.shape
    with pytest.raises(ValueError, match="out of range"):
        prepared_transform_frame_signatures(prepared, 120)


def test_generic_native_fixture_cannot_receive_result_capable_provenance():
    native = CanonicalNativeInput.from_arrays(
        case_id="asymmetric_synthetic_7x3_v1",
        materializer_id="unit_only_v1",
        arrays={"ids": np.array([0], dtype=np.int32)},
        array_semantics={"ids": "unit.ids"},
        topology_arrays=("ids",),
        semantics={"scope": "unit"},
    )

    assert not native.manifest.result_capable
    with pytest.raises(ValueError, match="result-capable"):
        wrap_prepared_optical_local_scene(
            WarpPreparedNativeInput(
                manifest=native.manifest,
                device_ordinal=0,
                arrays=MappingProxyType({"ids": object()}),
                ready_event=object(),
            )
        )


def _packed(case_id: str) -> PackedOpticalSceneHost:
    case = get_case(case_id)
    return PackedOpticalSceneHost.from_registry(
        _registry(),
        source_contract_id=case.case_id,
        source_contract_sha256=case.sha256,
        scene_id=case.scene_id,
    )


def _fixture(packed: PackedOpticalSceneHost) -> CanonicalFixedRenderFixtureHost:
    return CanonicalFixedRenderFixtureHost.from_scene_contract(
        packed,
        fixed_transform_id="asymmetric_fixed_world_v1",
        x_world_R=np.eye(3, dtype=np.float32)[None, :, :],
        x_world_r=np.array([[0.25, -0.5, 2.0]], dtype=np.float32),
    )


def _registry() -> OpticalWorldRegistry:
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat", albedo_rgb=(0.2, 0.3, 0.4)))
    registry.add_triangle_mesh_geometry(
        "triangle",
        vertices_local=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        triangles=[[0, 1, 2]],
    )
    registry.add_instance(
        OpticalInstanceSpec(
            "triangle_instance",
            "triangle",
            "mat",
            body_index=0,
            roles=frozenset({"rgb"}),
        )
    )
    return registry


def _forge_fixture_collision(
    fixture: CanonicalFixedRenderFixtureHost,
    packed: PackedOpticalSceneHost,
    *,
    collision_kind: str,
) -> CanonicalFixedRenderFixtureHost:
    forged = object.__new__(CanonicalFixedRenderFixtureHost)
    for field in CanonicalFixedRenderFixtureHost.__dataclass_fields__:
        object.__setattr__(forged, field, getattr(fixture, field))
    arrays = dict(fixture.arrays)
    semantics = dict(fixture.array_semantics)
    if collision_kind == "name":
        arrays["material_albedo_rgb"] = packed.arrays["material_albedo_rgb"]
        semantics["material_albedo_rgb"] = "forged.fixture.material"
    else:
        value = np.array(packed.arrays["material_albedo_rgb"], copy=True)
        value.setflags(write=False)
        arrays["fixture_material_alias"] = value
        semantics["fixture_material_alias"] = packed.array_semantics["material_albedo_rgb"]
    object.__setattr__(forged, "arrays", MappingProxyType(arrays))
    object.__setattr__(forged, "array_semantics", MappingProxyType(semantics))
    return forged


class _PointerOnly:
    def __init__(self, pointer: int) -> None:
        self.ptr = pointer
