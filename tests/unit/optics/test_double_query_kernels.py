"""Execute production query kernels on CPU against observable hit regressions."""
import numpy as np
import pytest

wp = pytest.importorskip("warp")

from optics import warp_native_render as kernels
from optics.host_bvh import build_host_median_bvh
from optics.query_numerics import outward_float32_bounds
from optics.execution import _intersect_triangle_scalar


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
    native = _launch_first_hit(entry, faces, keys, ids, [0.2, 0.2, 0.], [0., 0., 1.], limit)
    assert native['hit_mask'].numpy().tolist() == [1]
    assert native['numeric_instance_id'].numpy().tolist() == [expected]
    assert native['range_m'].numpy().tolist() == [1.0]
    assert native['bvh_stack_overflow_count'].numpy().tolist() == [0]


def _launch_first_hit(entry, faces, keys, ids, origin, direction, limit):
    lower, upper = outward_float32_bounds(faces)
    tree = build_host_median_bvh(lower, upper, keys, leaf_size=1, split_strategy="sort")
    normals = np.cross(faces[:, 1]-faces[:, 0], faces[:, 2]-faces[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    count = len(faces)
    values = {
        "origins": np.array([origin]), "directions": np.array([direction]),
        "max_distance": float(limit), "plane_normals": np.empty((0, 3)), "plane_points": np.empty((0, 3)),
        "plane_numeric_ids": np.empty(0, dtype=np.int32), "plane_source_keys": np.empty(0, dtype=np.int64),
        "plane_role_masks": np.empty(0, dtype=np.int64), "plane_material_indices": np.empty(0, dtype=np.int32),
        "num_planes": 0, "triangles": faces.reshape(-1, 9),
        "triangle_v0": faces[:, 0].copy(), "triangle_e1": faces[:, 1]-faces[:, 0],
        "triangle_e2": faces[:, 2]-faces[:, 0],
        "triangle_normal": normals.astype(np.float32),
        "triangle_numeric_ids": ids, "triangle_source_keys": keys,
        "triangle_role_masks": np.ones(count, dtype=np.int64), "triangle_material_indices": np.zeros(count, dtype=np.int32),
        "triangle_aabb_min": lower, "triangle_aabb_max": upper, "num_triangles": count,
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
    return native


@pytest.mark.parametrize("entry", ["_brute_force_first_hit_kernel", "_device_scene_first_hit_kernel",
                                  "_device_scene_first_hit_aabb_kernel", "_device_scene_bvh_first_hit_kernel"])
@pytest.mark.parametrize("reverse_ray", [False, True])
@pytest.mark.parametrize("reverse_winding", [False, True])
def test_compiled_grazing_hit_keeps_cpu_normal_orientation(entry, reverse_ray, reverse_winding):
    wp.init()
    faces = np.array([[[0., 0., 0.], [-0.123456789, 1., 0.], [-0.3, 0., 1.]]])
    origin = np.array([-0.26367883497322825, 1.3257985665584018, 0.333333333012683])
    direction = np.array([0.12252657197322826, -0.9924652332250684, 3.2065034296667426e-10])
    if reverse_ray:
        origin, direction = origin + 2.*direction, -direction
    if reverse_winding:
        faces = faces[:, [0, 2, 1]]
    truth = _intersect_triangle_scalar(origin, direction, 2., *faces[0])
    assert truth is not None
    native = _launch_first_hit(entry, faces, np.array([17], np.int64), np.array([17], np.int32),
                               origin, direction, 2.)
    assert native['hit_mask'].numpy().tolist() == [1]
    np.testing.assert_array_equal(native['normal_world'].numpy()[0], truth.normal_world.astype(np.float32))
    if entry == '_device_scene_bvh_first_hit_kernel':
        # Lighting along the CPU-facing normal should remain fully visible,
        # including a shadow ray offset toward that side of this same surface.
        native.update({
            'material_albedo_rgb': wp.ones((1, 3), dtype=wp.float32, device='cpu'),
            'light_kind': wp.zeros(1, dtype=wp.int32, device='cpu'),
            'light_position_or_direction_world': wp.array(truth.normal_world[None], dtype=wp.float32, device='cpu'),
            'light_intensity': wp.ones(1, dtype=wp.float32, device='cpu'),
            'light_color_rgb': wp.ones((1, 3), dtype=wp.float32, device='cpu'),
            'num_lights': 1, 'shadow_bias': 1.e-6,
            'rgb': wp.zeros((1, 3), dtype=wp.float32, device='cpu'),
            'intensity': wp.zeros(1, dtype=wp.float32, device='cpu'),
            'shadow_stack_overflow_count': wp.zeros(1, dtype=wp.int32, device='cpu'),
            'shadow_max_stack_depth': wp.zeros(1, dtype=wp.int32, device='cpu'),
        })
        native.update({name: 0. for name in ('ambient_r', 'ambient_g', 'ambient_b',
                                            'background_r', 'background_g', 'background_b')})
        for shader in (kernels.warp_direct_light_no_shadow_primary_v1,
                       kernels.warp_direct_light_with_shadows_primary_v1):
            wp.launch(shader, dim=1, inputs=[native[arg.label] for arg in shader.adj.args], device='cpu')
            np.testing.assert_allclose(native['rgb'].numpy(), 1., rtol=0., atol=3.e-7)
            np.testing.assert_allclose(native['intensity'].numpy(), 1., rtol=0., atol=3.e-7)
            assert native['shadow_stack_overflow_count'].numpy().tolist() == [0]


@wp.kernel
def _probe_shadow(
    origins: wp.array2d(dtype=wp.float64), directions: wp.array2d(dtype=wp.float64),
    max_distance: wp.float64, plane_normals: wp.array2d(dtype=wp.float64),
    plane_points: wp.array2d(dtype=wp.float64), plane_role_masks: wp.array(dtype=wp.int64),
    num_planes: int, triangle_v0: wp.array2d(dtype=wp.float64),
    triangle_e1: wp.array2d(dtype=wp.float64), triangle_e2: wp.array2d(dtype=wp.float64),
    triangle_normal: wp.array2d(dtype=wp.float32), triangle_role_masks: wp.array(dtype=wp.int64),
    bvh_bounds_min: wp.array2d(dtype=wp.float32), bvh_bounds_max: wp.array2d(dtype=wp.float32),
    bvh_left: wp.array(dtype=wp.int32), bvh_right: wp.array(dtype=wp.int32),
    bvh_start: wp.array(dtype=wp.int32), bvh_count: wp.array(dtype=wp.int32),
    bvh_prim_ids: wp.array(dtype=wp.int32), num_bvh_nodes: int, sensor_role_mask: wp.int64,
    hit_mask: wp.array(dtype=wp.int32),
):
    trace = kernels._is_occluded_for_ray(
        origins[0, 0], origins[0, 1], origins[0, 2], directions[0, 0], directions[0, 1], directions[0, 2],
        max_distance, plane_normals, plane_points, plane_role_masks, num_planes,
        triangle_v0, triangle_e1, triangle_e2, triangle_normal, triangle_role_masks,
        bvh_bounds_min, bvh_bounds_max, bvh_left, bvh_right, bvh_start, bvh_count, bvh_prim_ids,
        num_bvh_nodes, sensor_role_mask)
    hit_mask[0] = trace.occluded


@pytest.mark.parametrize("outside", [False, True])
@pytest.mark.parametrize("limit", [np.nextafter(1., 0.), 1., 2.])
def test_compiled_primary_and_shadow_preserve_cpu_parallel_slab_policy(outside, limit):
    wp.init()
    faces = np.array([[[0., -1., -1.], [0., 3., -1.], [0., -1., 3.]]])
    # The outside ray intersects at t=1, but CPU BVH policy rejects the
    # parallel slab. A nonparallel ray whose hit is also at t=1 checks the
    # strict physical limit independently of that early rejection.
    origin = [-5e-13, 0., -1.] if outside else [-1., 0., -1.]
    direction = [5e-13, 0., 1.] if outside else [1., 0., 1.]
    native = _launch_first_hit('_device_scene_bvh_first_hit_kernel', faces,
                               np.array([17], np.int64), np.array([17], np.int32), origin, direction, limit)
    expected = int(not outside and limit >= 1.)
    assert native['hit_mask'].numpy().tolist() == [expected]
    wp.launch(_probe_shadow, dim=1, inputs=[native[arg.label] for arg in _probe_shadow.adj.args], device='cpu')
    assert native['hit_mask'].numpy().tolist() == [expected]


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
