"""Independent hand-derived kernel boundary probes; not a recorded bundle."""
import inspect
import sys

import numpy as np
import warp as wp

sys.path.insert(0, "/data3/ang.gao/robot_sim_c4_shade_review1")
from optics import warp_native_render as n
from optics.output_contract import SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES as overlay

wp.init()
print("Warp", wp.__version__)
device = wp.get_device("cuda:0")
stream = wp.Stream(device=device)


def arr(value, dtype=wp.float32):
    return wp.array(np.asarray(value), dtype=dtype, device=device)


def empty_vec():
    return wp.empty((0, 3), dtype=wp.float32, device=device)


def empty_int(dtype=wp.int32):
    return wp.empty(0, dtype=dtype, device=device)


with wp.ScopedStream(stream):
    values = dict(
        hit_mask=arr([1, 0], wp.int32),
        position_world=arr([[0, 0, 0], [np.nan] * 3]),
        normal_world=arr([[0, 0, 1], [np.nan] * 3]),
        material_index=arr([0, 99999], wp.int32),
        material_albedo_rgb=arr([[0.5, 0.25, 0.125]]),
        light_kind=arr([0, 1], wp.int32),
        light_position_or_direction_world=arr([[0, 0, 1], [0, 0, 2]]),
        light_intensity=arr([2, 4]),
        light_color_rgb=arr([[1, 1, 1], [1, 1, 1]]),
        num_lights=2,
        ambient_r=0.1, ambient_g=0.2, ambient_b=0.4,
        background_r=0.2, background_g=0.3, background_b=0.4,
        rgb=wp.empty((2, 3), dtype=wp.float32, device=device),
        intensity=wp.empty(2, dtype=wp.float32, device=device),
        plane_normals=empty_vec(), plane_points=empty_vec(),
        plane_role_masks=empty_int(wp.int64), num_planes=0,
        triangle_v0=empty_vec(), triangle_e1=empty_vec(), triangle_e2=empty_vec(),
        triangle_normal=empty_vec(), triangle_role_masks=empty_int(wp.int64),
        bvh_bounds_min=empty_vec(), bvh_bounds_max=empty_vec(),
        bvh_left=empty_int(), bvh_right=empty_int(), bvh_start=empty_int(),
        bvh_count=empty_int(), bvh_prim_ids=empty_int(), num_bvh_nodes=0,
        sensor_role_mask=np.int64(1), shadow_bias=1e-6,
    )
    primary = ("shadow_stack_overflow_count", "shadow_max_stack_depth")
    counters = primary + overlay
    values.update({name: wp.empty(1, dtype=wp.int32, device=device) for name in counters})
    original = values.copy()
    launches = 0
    for scene in ("empty", "plane", "triangle", "masked_triangle", "no_lights", "degenerate_lights", "short_point"):
        values.update(original)
        if scene == "plane":
            values.update(plane_normals=arr([[0, 0, 1]]), plane_points=arr([[0, 0, 1]]),
                          plane_role_masks=arr([1], wp.int64), num_planes=1)
        if scene in ("triangle", "masked_triangle"):
            values.update(
                triangle_v0=arr([[-1, -1, 1]]), triangle_e1=arr([[2, 0, 0]]),
                triangle_e2=arr([[0, 2, 0]]), triangle_normal=arr([[0, 0, 1]]),
                triangle_role_masks=arr([1 if scene == "triangle" else 2], wp.int64),
                bvh_bounds_min=arr([[-1, -1, 1]]), bvh_bounds_max=arr([[1, 1, 1]]),
                bvh_left=arr([-1], wp.int32), bvh_right=arr([-1], wp.int32),
                bvh_start=arr([0], wp.int32), bvh_count=arr([1], wp.int32),
                bvh_prim_ids=arr([0], wp.int32), num_bvh_nodes=1,
            )
        if scene == "no_lights":
            values["num_lights"] = 0
        if scene == "degenerate_lights":
            values["light_position_or_direction_world"] = arr([[0, 0, 0], [0, 0, 0]])
        if scene == "short_point":
            values.update(light_position_or_direction_world=arr([[0, 0, 1], [0, 0, 0.5]]), shadow_bias=1.0)
        for shadows in (False, True):
            for diagnostics in (False, True):
                kernel, names = n.warp_direct_light_shade_entry(shadows=shadows, traversal_diagnostics=diagnostics)
                active = counters if diagnostics else primary
                for sentinel in (77, -123):
                    for name in (*active, "rgb", "intensity"):
                        values[name].fill_(sentinel)
                    reset = n._reset_shadow_diagnostic_counters_kernel if diagnostics else n._reset_shadow_primary_counters_kernel
                    wp.launch(reset, dim=1, inputs=[values[name] for name in active], stream=stream)
                    wp.launch(kernel, dim=2, inputs=[values[name] for name in names], stream=stream)
                    wp.synchronize_stream(stream)
                    blocked = shadows and scene in ("plane", "triangle")
                    scale = 0 if blocked or scene in ("no_lights", "degenerate_lights") else (18 if scene == "short_point" else 3)
                    color = np.array([0.05, 0.05, 0.05]) + np.array([0.5, 0.25, 0.125]) * scale
                    np.testing.assert_allclose(values["rgb"].numpy(), [color, [0.2, 0.3, 0.4]], rtol=1e-6, atol=1e-6)
                    np.testing.assert_allclose(values["intensity"].numpy(), [color @ [0.2126, 0.7152, 0.0722], 0], rtol=1e-6, atol=1e-6)
                    trace = shadows and scene not in ("no_lights", "degenerate_lights")
                    depth = int(trace and scene in ("triangle", "masked_triangle"))
                    expected = [0, depth]
                    if diagnostics:
                        rays = (1 if scene == "short_point" else 2) if trace else 0
                        point = int(trace and scene != "short_point")
                        expected += [rays, int(trace), point, rays if blocked else 0, 0 if blocked else rays,
                                     rays * depth, rays * depth, rays if trace and scene == "triangle" else 0,
                                     rays if trace and scene == "plane" else 0]
                    assert [int(values[name].numpy()[0]) for name in active] == expected, (scene, shadows, diagnostics, expected)
                    launches += 1
    print("PASS", launches, "shade runs with exact hand-derived output/counter assertions")

    # Empty first-hit must overwrite every physical output despite poisoned old values.
    first = dict(origins=arr([[0, 0, 0], [1, 2, 3]]), directions=arr([[0, 0, 1], [1, 0, 0]]), max_distance=10.0)
    first.update({name: value for name, value in original.items() if name.startswith(("plane_", "triangle_", "bvh_"))})
    first.update(num_planes=0, num_bvh_nodes=0, sensor_role_mask=np.int64(1))
    for prefix in ("plane", "triangle"):
        for suffix in ("numeric_ids", "material_indices"):
            first[prefix + "_" + suffix] = empty_int()
        first[prefix + "_source_keys"] = empty_int(wp.int64)
    first.update(hit_mask=arr([77, 77], wp.int32), range_m=arr([77, 77]),
                 position_world=arr([[77] * 3] * 2), normal_world=arr([[77] * 3] * 2),
                 numeric_instance_id=arr([77, 77], wp.int32), material_index=arr([77, 77], wp.int32),
                 bvh_stack_overflow_count=arr([77], wp.int32), bvh_max_stack_depth=arr([77], wp.int32))
    kernel = n._device_scene_bvh_first_hit_kernel
    for sentinel in (77, -123):
        for name in ("hit_mask", "range_m", "position_world", "normal_world", "numeric_instance_id", "material_index"):
            first[name].fill_(sentinel)
        wp.launch(n._reset_bvh_primary_counters_kernel, dim=1,
                  inputs=[first["bvh_stack_overflow_count"], first["bvh_max_stack_depth"]], stream=stream)
        wp.launch(kernel, dim=2, inputs=[first[name] for name in inspect.signature(kernel.func).parameters], stream=stream)
        wp.synchronize_stream(stream)
        for name in ("hit_mask", "numeric_instance_id", "material_index", "bvh_stack_overflow_count", "bvh_max_stack_depth"):
            assert np.all(first[name].numpy() == 0)
        assert np.all(np.isposinf(first["range_m"].numpy()))
        assert np.all(np.isnan(first["position_world"].numpy()))
        assert np.all(np.isnan(first["normal_world"].numpy()))
    print("PASS repeated empty-scene first-hit initialization")
