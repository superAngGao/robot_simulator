"""Execute production query kernels on CPU against observable hit regressions."""
import numpy as np
import pytest

wp = pytest.importorskip("warp")

from optics import warp_native_render as kernels
from optics.host_bvh import build_host_median_bvh
from optics.query_numerics import outward_float32_bounds


@wp.kernel
def _probe_area_boundary(v0: wp.array2d(dtype=wp.float64), e1: wp.array2d(dtype=wp.float64),
                         e2: wp.array2d(dtype=wp.float64), normals: wp.array2d(dtype=wp.float32),
                         output: wp.array(dtype=wp.int32)):
    i = wp.tid()
    hit, t, nx, ny, nz = kernels._intersect_device_triangle_for_ray(
        wp.float64(0.25), e2[i, 1] * wp.float64(0.25), wp.float64(0.0),
        wp.float64(0.0), wp.float64(0.0), wp.float64(1.0), i, v0, e1, e2, normals, wp.float64(2.0))
    output[i] = wp.int32(hit)


def test_world_area_rejection_uses_exact_binary64_policy_constant():
    wp.init()
    areas = np.array([np.nextafter(1e-12, 0.), 1e-12, np.nextafter(1e-12, np.inf)])
    source = [np.array([[0., 0., 1.]]*3), np.array([[1., 0., 0.]]*3),
              np.column_stack([np.zeros(3), areas, np.zeros(3)])]
    inputs = [wp.array(value, dtype=wp.float64, device='cpu') for value in source]
    normals = wp.zeros((3, 3), dtype=wp.float32, device='cpu')
    output = wp.zeros(3, dtype=wp.int32, device='cpu')
    wp.launch(_probe_area_boundary, dim=3, inputs=inputs+[normals, output], device='cpu')
    assert output.numpy().tolist() == [0, 0, 1]


@pytest.mark.parametrize("entry", ["_brute_force_first_hit_kernel", "_device_scene_first_hit_kernel",
                                  "_device_scene_first_hit_aabb_kernel", "_device_scene_bvh_first_hit_kernel"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("gap,limit,expected", [(2e-8, 2., 20), (5e-10, 2., 1), (5e-10, 1., 20)])
def test_unrounded_nearest_order_ties_and_strict_physical_cutoff(entry, reverse, gap, limit, expected):
    wp.init()
    faces = np.array([[[0., 0., z], [2., 0., z], [0., 2., z]] for z in (1., 1.+gap)])
    keys = np.array([20, 1], dtype=np.int64)
    ids = np.array([20, 1], dtype=np.int32)
    if reverse:
        faces, keys, ids = faces[::-1].copy(), keys[::-1].copy(), ids[::-1].copy()
    lower, upper = outward_float32_bounds(faces)
    tree = build_host_median_bvh(lower, upper, keys, leaf_size=1, split_strategy="sort")
    values = {
        "origins": np.array([[0.2, 0.2, 0.]]), "directions": np.array([[0., 0., 1.]]),
        "max_distance": float(limit), "plane_normals": np.empty((0, 3)), "plane_points": np.empty((0, 3)),
        "plane_numeric_ids": np.empty(0, dtype=np.int32), "plane_source_keys": np.empty(0, dtype=np.int64),
        "plane_role_masks": np.empty(0, dtype=np.int64), "plane_material_indices": np.empty(0, dtype=np.int32),
        "num_planes": 0, "triangles": faces.reshape(-1, 9),
        "triangle_v0": faces[:, 0].copy(), "triangle_e1": faces[:, 1]-faces[:, 0],
        "triangle_e2": faces[:, 2]-faces[:, 0],
        "triangle_normal": np.array([[0., 0., 1.]]*2, dtype=np.float32),
        "triangle_numeric_ids": ids, "triangle_source_keys": keys,
        "triangle_role_masks": np.ones(2, dtype=np.int64), "triangle_material_indices": np.ones(2, dtype=np.int32),
        "triangle_aabb_min": lower, "triangle_aabb_max": upper, "num_triangles": 2,
        "sensor_role_mask": wp.int64(1), "num_bvh_nodes": tree["stats"].num_nodes,
        "hit_mask": np.zeros(1, dtype=np.int32), "range_m": np.zeros(1, dtype=np.float32),
        "position_world": np.zeros((1, 3), dtype=np.float32), "normal_world": np.zeros((1, 3), dtype=np.float32),
        "numeric_instance_id": np.zeros(1, dtype=np.int32), "material_index": np.zeros(1, dtype=np.int32),
        "bvh_stack_overflow_count": np.zeros(1, dtype=np.int32), "bvh_max_stack_depth": np.zeros(1, dtype=np.int32),
    }
    for name in ("bounds_min", "bounds_max", "left", "right", "start", "count", "prim_ids"):
        values["bvh_"+name] = tree[name]
    scalar_types = {np.dtype('float64'): wp.float64, np.dtype('float32'): wp.float32,
                    np.dtype('int32'): wp.int32, np.dtype('int64'): wp.int64}
    native = {name: wp.array(value, dtype=scalar_types[value.dtype], device='cpu')
              if isinstance(value, np.ndarray) else value for name, value in values.items()}
    kernel = getattr(kernels, entry)
    wp.launch(kernel, dim=1, inputs=[native[arg.label] for arg in kernel.adj.args], device='cpu')
    assert native['hit_mask'].numpy().tolist() == [1]
    assert native['numeric_instance_id'].numpy().tolist() == [expected]
    assert native['range_m'].numpy().tolist() == [1.0]
    assert native['bvh_stack_overflow_count'].numpy().tolist() == [0]


def test_native_camera_keeps_binary64_origin_and_world_normalization():
    wp.init()
    origin = [0.123456789123, -0.222222222222, 0.333333333333]
    R = np.array([[0.6, -0.8, 0.], [0.8, 0.6, 0.], [0., 0., 1.]])
    origins = wp.zeros((21, 3), dtype=wp.float64, device='cpu')
    directions = wp.zeros((21, 3), dtype=wp.float64, device='cpu')
    wp.launch(kernels._pinhole_camera_raygen_kernel, dim=21,
              inputs=[7, 3, 4.123456789, 5.234567891, 2.345678912, 0.123456789,
                      *origin, *R.reshape(9), origins, directions], device='cpu')
    pixels = np.array([[(u-2.345678912)/4.123456789, (v-0.123456789)/5.234567891, 1.]
                       for v in range(3) for u in range(7)])
    world = pixels @ R.T
    reference = world / np.linalg.norm(world, axis=1)[:, None]
    np.testing.assert_array_equal(origins.numpy(), np.tile(origin, (21, 1)))
    np.testing.assert_allclose(directions.numpy(), reference, rtol=0, atol=3.4e-16)
