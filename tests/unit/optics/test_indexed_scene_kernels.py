"""Execute the real Warp scene kernels on CPU, without substituting their bodies."""
import numpy as np
import pytest

wp = pytest.importorskip("warp")

from optics import device_scene as kernels
from optics.query_numerics import outward_float32_bounds


@pytest.mark.parametrize("with_bounds", [False, True])
@pytest.mark.parametrize("body_bound", [False, True])
def test_indexed_scene_kernel_preserves_double_edges_and_clears_degenerate_normals(with_bounds, body_bound):
    wp.init()
    vertices = np.array([[100, 100, 100], [101, 100, 100], [101, 100.0001, 100.00023],
                         [0., 0., 0.], [1., 0., 0.], [1., 1.1e-12, 0.]], dtype=np.float64)
    indices = np.array([[0, 1, 2], [0, 0, 0], [3, 4, 5]], dtype=np.int64)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    translation = np.array([0.111111111111, 0.22, 0.333333333333])
    frame_r = np.array([[[0.125, 0.25, 0.5]]], dtype=np.float32)
    source = [vertices, indices, np.zeros(3, dtype=np.int32),
              np.array([0 if body_bound else -1], dtype=np.int32), rotation.reshape(1, 9),
              translation.reshape(1, 3), np.eye(3, dtype=np.float32).reshape(1, 1, 3, 3), frame_r]
    dtypes = [wp.float64, wp.int64, wp.int32, wp.int32, wp.float64, wp.float64, wp.float32, wp.float32]
    inputs = [wp.array(a, dtype=dtype, device="cpu") for a, dtype in zip(source, dtypes)]
    outputs = [wp.zeros((3, 3), dtype=wp.float64, device="cpu") for _ in range(3)]
    outputs.append(wp.array(np.ones((3, 3), dtype=np.float32), dtype=wp.float32, device="cpu"))
    if with_bounds:
        outputs.extend(wp.zeros((3, 3), dtype=wp.float32, device="cpu") for _ in range(2))
    kernel = (kernels._update_triangles_derived_aabb_world_kernel if with_bounds
              else kernels._update_triangles_derived_world_kernel)
    wp.launch(kernel, dim=3, inputs=inputs + [0] + outputs, device="cpu")
    shift = translation + (frame_r[0, 0].astype(np.float64) if body_bound else 0)
    world = (vertices @ rotation.T + shift)[indices]
    expected = [world[:, 0], world[:, 1]-world[:, 0], world[:, 2]-world[:, 0]]
    for array, reference in zip(outputs, expected):
        np.testing.assert_array_equal(array.numpy(), reference)
    normals = []
    for face in world:
        n = np.cross(face[1]-face[0], face[2]-face[0])
        norm = np.linalg.norm(n)
        normals.append(n/norm if norm > 1e-12 else np.zeros(3))
    np.testing.assert_array_equal(outputs[3].numpy(), np.asarray(normals, dtype=np.float32))
    if with_bounds:
        lower, upper = outward_float32_bounds(world)
        np.testing.assert_array_equal(outputs[4].numpy(), lower)
        np.testing.assert_array_equal(outputs[5].numpy(), upper)


def test_plane_kernel_preserves_double_source_and_float32_producer_values():
    wp.init()
    source = [np.array([[0., 0., 1.]]), np.array([[0.1111111111111, 0., 0.]]),
              np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
              np.eye(3).reshape(1, 9), np.array([[0.0123456789, 0.2, 0.3]]),
              np.eye(3, dtype=np.float32).reshape(1, 1, 3, 3),
              np.array([[[0.1, 0.2, 0.3]]], dtype=np.float32)]
    types = [wp.float64, wp.float64, wp.int32, wp.int32, wp.float64, wp.float64, wp.float32, wp.float32]
    inputs = [wp.array(a, dtype=t, device="cpu") for a, t in zip(source, types)]
    outputs = [wp.zeros((1, 3), dtype=wp.float64, device="cpu") for _ in range(2)]
    wp.launch(kernels._update_planes_world_kernel, dim=1, inputs=inputs + [0] + outputs, device="cpu")
    np.testing.assert_array_equal(outputs[0].numpy(), source[0])
    np.testing.assert_array_equal(outputs[1].numpy(), source[1] + (source[5] + source[7][0].astype(np.float64)))
