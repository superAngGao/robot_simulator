"""Source-preserving geometry, original CPU arithmetic, and outward bounds."""
from dataclasses import replace

import numpy as np
import pytest

from optical_worker.native_input import _derive_fixed_fixture
from optics.query_numerics import BUILD_EPS, outward_float32_bounds
from optics.registry import OpticalInstanceSpec, OpticalMaterialSpec, OpticalWorldRegistry
from optics.scene import transform_points
from optics.scene_contract import PackedOpticalSceneHost, validate_indexed_geometry
from physics.spatial import SpatialTransform, rot_y, rot_z


def _registry():
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat"))
    registry.add_plane_geometry("plane", normal_local=[0, 0, 1], point_local=[0, 0, 0.1])
    vertices = np.array([[100., 100., 100.], [101., 100., 100.],
                         [101., 100.0001, 100.00023], [0., 0., 0.],
                         [1., 0., 0.], [1., 1.1e-12, 0.]])
    # A large unused block exposes whole-mesh versus per-face BLAS evaluation.
    vertices = np.concatenate([vertices, np.zeros((8193, 3))])
    faces = [[0, 0, 0], [0, 1, 2], [0, 2, 1], [3, 4, 5]]
    registry.add_triangle_mesh_geometry("mesh", vertices_local=vertices, triangles=faces)
    for i in range(2):
        registry.add_instance(OpticalInstanceSpec(
            f"mesh{i}", "mesh", "mat", body_index=i,
            X_body_geometry=SpatialTransform(rot_y(0.17 + i), np.array([0.13, 0.2, 0.3]))))
    registry.add_instance(OpticalInstanceSpec("floor", "plane", "mat"))
    return registry


def test_shared_mesh_preserves_unused_vertices_all_faces_and_stable_keys():
    registry = _registry()
    scene = PackedOpticalSceneHost.from_registry(registry)
    validate_indexed_geometry(scene)
    mesh = registry.geometry["mesh"]
    np.testing.assert_array_equal(scene.arrays["geometry_vertices_local"], mesh.vertices_local)
    np.testing.assert_array_equal(scene.arrays["geometry_vertex_offsets"], [0, 0, len(mesh.vertices_local)])
    np.testing.assert_array_equal(scene.arrays["triangle_vertex_indices"], np.tile(mesh.triangles, (2, 1)))
    np.testing.assert_array_equal(scene.arrays["triangle_source_order_key"],
                                  [0, 1, 2, 3, 2**32, 2**32+1, 2**32+2, 2**32+3])
    np.testing.assert_array_equal(scene.arrays["triangle_primitive_global_id"], np.arange(8))
    assert "triangle_normal_local" not in scene.arrays
    assert "triangle_vertices_local" not in scene.arrays


def test_fixed_fixture_matches_original_whole_mesh_arithmetic_and_bounds():
    registry = _registry()
    scene = PackedOpticalSceneHost.from_registry(registry)
    rotations = np.array([rot_z(0.3), rot_y(0.43)])
    translations = np.array([[0.11111111111, 0.23, -0.71], [-0.29, 0.2, 0.51]])
    arrays, stats = _derive_fixed_fixture(scene, rotations, translations)
    all_world = []
    for instance in registry.instances[:2]:
        mesh = registry.geometry[instance.geometry_id]
        body = instance.body_index
        transform = SpatialTransform(rotations[body], translations[body]) @ instance.X_body_geometry
        world = transform_points(transform, mesh.vertices_local)[mesh.triangles]
        all_world.extend(world)
    world = np.asarray(all_world)
    np.testing.assert_array_equal(arrays["triangle_v0_world"], world[:, 0])
    np.testing.assert_array_equal(arrays["triangle_e1_world"], world[:, 1] - world[:, 0])
    np.testing.assert_array_equal(arrays["triangle_e2_world"], world[:, 2] - world[:, 0])
    for row, triangle in enumerate(world):
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        norm = np.linalg.norm(normal)
        expected = normal / norm if norm > BUILD_EPS else np.zeros(3)
        np.testing.assert_array_equal(arrays["triangle_normal_world"][row], expected.astype(np.float32))
    assert np.all(arrays["triangle_aabb_min"][:, None, :] <= world)
    assert np.all(arrays["triangle_aabb_max"][:, None, :] >= world)
    assert arrays["triangle_v0_world"].dtype == np.float64
    assert arrays["triangle_normal_world"].dtype == np.float32
    assert arrays["triangle_aabb_min"].dtype == np.float32
    np.testing.assert_array_equal(np.sort(arrays["bvh_prim_ids"]), np.arange(8))


@pytest.mark.parametrize("indices", [[[0.5, 1, 2]], [[True, False, True]],
                                      [[-1, 1, 2]], [[0, 1, 3]],
                                      np.array([[2**64-1, 1, 2]], dtype=np.uint64)])
def test_source_indices_are_rejected_before_conversion(indices):
    registry = OpticalWorldRegistry()
    with pytest.raises(ValueError):
        registry.add_triangle_mesh_geometry("bad", vertices_local=np.eye(3), triangles=indices)


def test_cross_geometry_index_substitution_is_rejected_before_gather():
    registry = _registry()
    registry.add_triangle_mesh_geometry("other", vertices_local=np.eye(3), triangles=[[0, 1, 2]])
    scene = PackedOpticalSceneHost.from_registry(registry)
    arrays = dict(scene.arrays)
    indices = arrays["triangle_vertex_indices"].copy()
    indices[0, 0] = arrays["geometry_vertex_offsets"][-2]
    arrays["triangle_vertex_indices"] = indices
    with pytest.raises(ValueError, match="geometry interval"):
        _derive_fixed_fixture(replace(scene, arrays=arrays), np.zeros((2, 3, 3)), np.zeros((2, 3)))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_unused_source_vertex_is_rejected(value):
    registry = _registry()
    registry.geometry["mesh"].vertices_local[-1, 0] = value
    with pytest.raises(ValueError, match="finite"):
        PackedOpticalSceneHost.from_registry(registry)


def test_empty_source_and_fixture_keep_typed_shapes():
    scene = PackedOpticalSceneHost.from_registry(OpticalWorldRegistry())
    arrays, stats = _derive_fixed_fixture(scene, np.empty((0, 3, 3)), np.empty((0, 3)))
    np.testing.assert_array_equal(scene.arrays["geometry_vertex_offsets"], [0])
    assert scene.arrays["geometry_vertices_local"].shape == (0, 3)
    assert scene.arrays["triangle_vertex_indices"].shape == (0, 3)
    assert arrays["triangle_v0_world"].shape == (0, 3)
    assert stats.num_nodes == 0


def test_outward_bounds_are_minimal_and_do_not_pad_exact_values():
    values = np.array([[1 + 2**-25, -1 - 2**-25, 0],
                       [1 + 2**-24, -1 - 2**-24, 0], [2, -2, 0]], dtype=np.float64)
    lo, hi = outward_float32_bounds(values)
    assert np.all(lo <= values.min(axis=0)) and np.all(hi >= values.max(axis=0))
    assert lo[2] == hi[2] == 0
    assert np.nextafter(lo[0], np.float32(np.inf)) > values[:, 0].min()
    assert np.nextafter(hi[1], np.float32(-np.inf)) < values[:, 1].max()
    with pytest.raises(ValueError, match="representable"):
        outward_float32_bounds(np.full((3, 3), 1e100))
