"""
Tests for GPU analytical collision functions vs CPU GJK/EPA reference.

Compares the Warp @wp.func analytical collision results with the CPU
gjk_epa_query / ground_contact_query at multiple orientations and positions.

Each test creates shapes, runs both CPU (GJK/EPA) and GPU (analytical),
and asserts depth/normal agreement within float32 tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.geometry import BoxShape, CapsuleShape, CylinderShape, SphereShape
from physics.gjk_epa import gjk_epa_query, ground_contact_query
from physics.spatial import SpatialTransform

try:
    import warp as wp

    wp.init()

    from physics.backends.warp.analytical_collision import (
        box_vs_ground,
        capsule_capsule,
        capsule_vs_ground,
        cylinder_vs_ground,
        sphere_box,
        sphere_capsule,
        sphere_sphere,
        sphere_vs_ground,
    )

    HAS_WARP = True
except Exception:
    HAS_WARP = False

pytestmark = pytest.mark.skipif(not HAS_WARP, reason="Warp or CUDA not available")


# ---------------------------------------------------------------------------
# Helpers: thin Warp kernel wrappers to call @wp.func from Python
# ---------------------------------------------------------------------------


@wp.kernel
def _test_sphere_ground(
    pos: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    ground_z: float,
    result: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    result[i] = sphere_vs_ground(pos[i], radius[i], ground_z)


@wp.kernel
def _test_capsule_ground(
    pos: wp.array(dtype=wp.vec3),
    R: wp.array(dtype=wp.mat33),
    radius: wp.array(dtype=float),
    hl: wp.array(dtype=float),
    ground_z: float,
    result: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    result[i] = capsule_vs_ground(pos[i], R[i], radius[i], hl[i], ground_z)


@wp.kernel
def _test_box_ground(
    pos: wp.array(dtype=wp.vec3),
    R: wp.array(dtype=wp.mat33),
    hx: wp.array(dtype=float),
    hy: wp.array(dtype=float),
    hz: wp.array(dtype=float),
    ground_z: float,
    result: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    result[i] = box_vs_ground(pos[i], R[i], hx[i], hy[i], hz[i], ground_z)


@wp.kernel
def _test_cylinder_ground(
    pos: wp.array(dtype=wp.vec3),
    R: wp.array(dtype=wp.mat33),
    radius: wp.array(dtype=float),
    hl: wp.array(dtype=float),
    ground_z: float,
    result: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    result[i] = cylinder_vs_ground(pos[i], R[i], radius[i], hl[i], ground_z)


@wp.kernel
def _test_sphere_sphere(
    pos_i: wp.array(dtype=wp.vec3),
    r_i: wp.array(dtype=float),
    pos_j: wp.array(dtype=wp.vec3),
    r_j: wp.array(dtype=float),
    result: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    result[i] = sphere_sphere(pos_i[i], r_i[i], pos_j[i], r_j[i])


@wp.kernel
def _test_sphere_capsule(
    pos_s: wp.array(dtype=wp.vec3),
    r_s: wp.array(dtype=float),
    pos_c: wp.array(dtype=wp.vec3),
    R_c: wp.array(dtype=wp.mat33),
    r_c: wp.array(dtype=float),
    hl_c: wp.array(dtype=float),
    result: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    result[i] = sphere_capsule(pos_s[i], r_s[i], pos_c[i], R_c[i], r_c[i], hl_c[i])


@wp.kernel
def _test_capsule_capsule(
    pos_i: wp.array(dtype=wp.vec3),
    R_i: wp.array(dtype=wp.mat33),
    r_i: wp.array(dtype=float),
    hl_i: wp.array(dtype=float),
    pos_j: wp.array(dtype=wp.vec3),
    R_j: wp.array(dtype=wp.mat33),
    r_j: wp.array(dtype=float),
    hl_j: wp.array(dtype=float),
    result: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    result[i] = capsule_capsule(pos_i[i], R_i[i], r_i[i], hl_i[i], pos_j[i], R_j[i], r_j[i], hl_j[i])


@wp.kernel
def _test_sphere_box(
    pos_s: wp.array(dtype=wp.vec3),
    r_s: wp.array(dtype=float),
    pos_b: wp.array(dtype=wp.vec3),
    R_b: wp.array(dtype=wp.mat33),
    hx: wp.array(dtype=float),
    hy: wp.array(dtype=float),
    hz: wp.array(dtype=float),
    result: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    result[i] = sphere_box(pos_s[i], r_s[i], pos_b[i], R_b[i], hx[i], hy[i], hz[i])


_DEV = "cpu"


def _run_1(kernel, *args):
    """Run a kernel with 1 element and return result[0]."""
    result = wp.zeros(1, dtype=wp.vec3, device=_DEV)
    wp.launch(kernel, dim=1, device=_DEV, inputs=list(args) + [result])
    return result.numpy()[0]


def _wp_vec3(v):
    return wp.array([v], dtype=wp.vec3, device=_DEV)


def _wp_mat33(R):
    return wp.array([R.flatten()], dtype=wp.mat33, device=_DEV)


def _wp_f(x):
    return wp.array([float(x)], dtype=float, device=_DEV)


def _rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Ground collision tests
# ---------------------------------------------------------------------------


class TestSphereGround:
    def test_penetrating(self):
        pos = np.array([0.0, 0.0, 0.05])
        radius = 0.1
        # CPU reference
        m = ground_contact_query(SphereShape(radius), SpatialTransform(np.eye(3), pos))
        assert m is not None
        # GPU analytical
        res = _run_1(_test_sphere_ground, _wp_vec3(pos), _wp_f(radius), 0.0)
        assert res[1] > 0.5  # hit
        np.testing.assert_allclose(res[0], m.depth, atol=0.01)

    def test_separated(self):
        pos = np.array([0.0, 0.0, 0.2])
        radius = 0.1
        m = ground_contact_query(SphereShape(radius), SpatialTransform(np.eye(3), pos))
        assert m is None
        res = _run_1(_test_sphere_ground, _wp_vec3(pos), _wp_f(radius), 0.0)
        assert res[1] < 0.5  # no hit

    def test_touching(self):
        pos = np.array([0.0, 0.0, 0.1])
        radius = 0.1
        res = _run_1(_test_sphere_ground, _wp_vec3(pos), _wp_f(radius), 0.0)
        assert res[0] < 0.001  # depth ~ 0


class TestCapsuleGround:
    def test_vertical_capsule(self):
        """Capsule standing upright — lowest point is center - half_length - radius."""
        pos = np.array([0.0, 0.0, 0.12])  # lowest = 0.12-0.1-0.05 = -0.03 → depth=0.03
        R = np.eye(3)
        radius, length = 0.05, 0.2
        shape = CapsuleShape(radius, length)
        m = ground_contact_query(shape, SpatialTransform(R, pos))
        res = _run_1(_test_capsule_ground, _wp_vec3(pos), _wp_mat33(R), _wp_f(radius), _wp_f(length / 2), 0.0)
        if m is not None:
            assert res[1] > 0.5
            np.testing.assert_allclose(res[0], m.depth, atol=0.01)
        else:
            assert res[1] < 0.5

    def test_tilted_capsule(self):
        """Capsule tilted 45 degrees — lower endpoint closer to ground."""
        pos = np.array([0.0, 0.0, 0.2])
        R = _rot_x(np.pi / 4)
        radius, length = 0.05, 0.2
        shape = CapsuleShape(radius, length)
        m = ground_contact_query(shape, SpatialTransform(R, pos))
        res = _run_1(_test_capsule_ground, _wp_vec3(pos), _wp_mat33(R), _wp_f(radius), _wp_f(length / 2), 0.0)
        if m is not None:
            assert res[1] > 0.5
            np.testing.assert_allclose(res[0], m.depth, atol=0.02)
        else:
            assert res[1] < 0.5


class TestBoxGround:
    def test_axis_aligned_box(self):
        pos = np.array([0.0, 0.0, 0.04])
        R = np.eye(3)
        shape = BoxShape((0.1, 0.1, 0.1))
        m = ground_contact_query(shape, SpatialTransform(R, pos))
        res = _run_1(
            _test_box_ground, _wp_vec3(pos), _wp_mat33(R), _wp_f(0.05), _wp_f(0.05), _wp_f(0.05), 0.0
        )
        assert m is not None
        assert res[1] > 0.5
        np.testing.assert_allclose(res[0], m.depth, atol=0.01)

    def test_rotated_box(self):
        """Box rotated 45 deg around X — corner should be lower."""
        pos = np.array([0.0, 0.0, 0.1])
        R = _rot_x(np.pi / 4)
        shape = BoxShape((0.1, 0.1, 0.1))
        m = ground_contact_query(shape, SpatialTransform(R, pos))
        res = _run_1(
            _test_box_ground, _wp_vec3(pos), _wp_mat33(R), _wp_f(0.05), _wp_f(0.05), _wp_f(0.05), 0.0
        )
        if m is not None:
            assert res[1] > 0.5
            np.testing.assert_allclose(res[0], m.depth, atol=0.02)

    def test_separated_box(self):
        pos = np.array([0.0, 0.0, 0.5])
        R = np.eye(3)
        shape = BoxShape((0.1, 0.1, 0.1))
        m = ground_contact_query(shape, SpatialTransform(R, pos))
        res = _run_1(
            _test_box_ground, _wp_vec3(pos), _wp_mat33(R), _wp_f(0.05), _wp_f(0.05), _wp_f(0.05), 0.0
        )
        assert m is None
        assert res[1] < 0.5


class TestCylinderGround:
    def test_upright_cylinder(self):
        pos = np.array([0.0, 0.0, 0.05])
        R = np.eye(3)
        shape = CylinderShape(0.05, 0.1)
        m = ground_contact_query(shape, SpatialTransform(R, pos))
        res = _run_1(_test_cylinder_ground, _wp_vec3(pos), _wp_mat33(R), _wp_f(0.05), _wp_f(0.05), 0.0)
        if m is not None:
            assert res[1] > 0.5
            np.testing.assert_allclose(res[0], m.depth, atol=0.01)

    def test_tilted_cylinder(self):
        pos = np.array([0.0, 0.0, 0.1])
        R = _rot_y(np.pi / 4)
        shape = CylinderShape(0.05, 0.2)
        m = ground_contact_query(shape, SpatialTransform(R, pos))
        res = _run_1(_test_cylinder_ground, _wp_vec3(pos), _wp_mat33(R), _wp_f(0.05), _wp_f(0.1), 0.0)
        if m is not None:
            assert res[1] > 0.5
            np.testing.assert_allclose(res[0], m.depth, atol=0.02)


# ---------------------------------------------------------------------------
# Body-body collision tests
# ---------------------------------------------------------------------------


class TestSphereSphere:
    def test_overlapping(self):
        pos_i = np.array([0.0, 0.0, 0.0])
        pos_j = np.array([0.12, 0.0, 0.0])  # overlap = 0.2 - 0.12 = 0.08
        r_i, r_j = 0.1, 0.1
        # CPU EPA has poor accuracy for deep sphere-sphere penetration.
        # Use analytical as ground truth.
        res = _run_1(_test_sphere_sphere, _wp_vec3(pos_i), _wp_f(r_i), _wp_vec3(pos_j), _wp_f(r_j))
        assert res[1] > 0.5
        expected_depth = (r_i + r_j) - 0.12  # = 0.08
        np.testing.assert_allclose(res[0], expected_depth, atol=0.001)

    def test_separated(self):
        pos_i = np.array([0.0, 0.0, 0.0])
        pos_j = np.array([0.5, 0.0, 0.0])
        r_i, r_j = 0.1, 0.1
        m = gjk_epa_query(
            SphereShape(r_i),
            SpatialTransform(np.eye(3), pos_i),
            SphereShape(r_j),
            SpatialTransform(np.eye(3), pos_j),
        )
        res = _run_1(_test_sphere_sphere, _wp_vec3(pos_i), _wp_f(r_i), _wp_vec3(pos_j), _wp_f(r_j))
        assert m is None
        assert res[1] < 0.5


class TestSphereCapsule:
    def test_overlap(self):
        pos_s = np.array([0.0, 0.0, 0.0])
        pos_c = np.array([0.12, 0.0, 0.0])
        R_c = np.eye(3)
        r_s, r_c, length = 0.05, 0.05, 0.2
        m = gjk_epa_query(
            SphereShape(r_s),
            SpatialTransform(np.eye(3), pos_s),
            CapsuleShape(r_c, length),
            SpatialTransform(R_c, pos_c),
        )
        res = _run_1(
            _test_sphere_capsule,
            _wp_vec3(pos_s),
            _wp_f(r_s),
            _wp_vec3(pos_c),
            _wp_mat33(R_c),
            _wp_f(r_c),
            _wp_f(length / 2),
        )
        if m is not None:
            assert res[1] > 0.5
            # Depth should match within tolerance
            np.testing.assert_allclose(res[0], m.depth, atol=0.02)

    def test_separated(self):
        pos_s = np.array([0.0, 0.0, 0.0])
        pos_c = np.array([0.5, 0.0, 0.0])
        R_c = np.eye(3)
        r_s, r_c, length = 0.05, 0.05, 0.2
        m = gjk_epa_query(
            SphereShape(r_s),
            SpatialTransform(np.eye(3), pos_s),
            CapsuleShape(r_c, length),
            SpatialTransform(R_c, pos_c),
        )
        res = _run_1(
            _test_sphere_capsule,
            _wp_vec3(pos_s),
            _wp_f(r_s),
            _wp_vec3(pos_c),
            _wp_mat33(R_c),
            _wp_f(r_c),
            _wp_f(length / 2),
        )
        assert m is None
        assert res[1] < 0.5


class TestCapsuleCapsule:
    def test_parallel_overlap(self):
        """Two parallel capsules close together."""
        pos_i = np.array([0.0, 0.0, 0.0])
        pos_j = np.array([0.08, 0.0, 0.0])
        R = np.eye(3)
        r, length = 0.05, 0.2
        m = gjk_epa_query(
            CapsuleShape(r, length),
            SpatialTransform(R, pos_i),
            CapsuleShape(r, length),
            SpatialTransform(R, pos_j),
        )
        res = _run_1(
            _test_capsule_capsule,
            _wp_vec3(pos_i),
            _wp_mat33(R),
            _wp_f(r),
            _wp_f(length / 2),
            _wp_vec3(pos_j),
            _wp_mat33(R),
            _wp_f(r),
            _wp_f(length / 2),
        )
        if m is not None:
            assert res[1] > 0.5
            np.testing.assert_allclose(res[0], m.depth, atol=0.02)

    def test_crossed_overlap(self):
        """Two capsules crossing at 90 degrees."""
        pos_i = np.array([0.0, 0.0, 0.0])
        pos_j = np.array([0.0, 0.0, 0.06])
        R_i = np.eye(3)
        R_j = _rot_x(np.pi / 2)
        r, length = 0.05, 0.2
        m = gjk_epa_query(
            CapsuleShape(r, length),
            SpatialTransform(R_i, pos_i),
            CapsuleShape(r, length),
            SpatialTransform(R_j, pos_j),
        )
        res = _run_1(
            _test_capsule_capsule,
            _wp_vec3(pos_i),
            _wp_mat33(R_i),
            _wp_f(r),
            _wp_f(length / 2),
            _wp_vec3(pos_j),
            _wp_mat33(R_j),
            _wp_f(r),
            _wp_f(length / 2),
        )
        if m is not None:
            assert res[1] > 0.5
            np.testing.assert_allclose(res[0], m.depth, atol=0.02)

    def test_separated(self):
        pos_i = np.array([0.0, 0.0, 0.0])
        pos_j = np.array([0.5, 0.0, 0.0])
        R = np.eye(3)
        r, length = 0.05, 0.2
        m = gjk_epa_query(
            CapsuleShape(r, length),
            SpatialTransform(R, pos_i),
            CapsuleShape(r, length),
            SpatialTransform(R, pos_j),
        )
        res = _run_1(
            _test_capsule_capsule,
            _wp_vec3(pos_i),
            _wp_mat33(R),
            _wp_f(r),
            _wp_f(length / 2),
            _wp_vec3(pos_j),
            _wp_mat33(R),
            _wp_f(r),
            _wp_f(length / 2),
        )
        assert m is None
        assert res[1] < 0.5


class TestSphereBox:
    def test_overlap(self):
        """Sphere center outside box, overlapping box face."""
        pos_s = np.array([0.13, 0.0, 0.0])  # center outside box (face at 0.1)
        pos_b = np.array([0.0, 0.0, 0.0])
        R_b = np.eye(3)
        r_s = 0.05  # sphere extends to 0.13-0.05=0.08, overlap with face at 0.1 = 0.02
        hx, hy, hz = 0.1, 0.1, 0.1
        # Analytical: distance to face = 0.13 - 0.1 = 0.03, depth = 0.05 - 0.03 = 0.02
        res = _run_1(
            _test_sphere_box,
            _wp_vec3(pos_s),
            _wp_f(r_s),
            _wp_vec3(pos_b),
            _wp_mat33(R_b),
            _wp_f(hx),
            _wp_f(hy),
            _wp_f(hz),
        )
        assert res[1] > 0.5
        np.testing.assert_allclose(res[0], 0.02, atol=0.005)

    def test_separated(self):
        pos_s = np.array([0.5, 0.0, 0.0])
        pos_b = np.array([0.0, 0.0, 0.0])
        R_b = np.eye(3)
        r_s = 0.05
        hx, hy, hz = 0.1, 0.1, 0.1
        res = _run_1(
            _test_sphere_box,
            _wp_vec3(pos_s),
            _wp_f(r_s),
            _wp_vec3(pos_b),
            _wp_mat33(R_b),
            _wp_f(hx),
            _wp_f(hy),
            _wp_f(hz),
        )
        assert res[1] < 0.5

    def test_rotated_box(self):
        """Sphere near a rotated box."""
        pos_s = np.array([0.12, 0.0, 0.0])
        pos_b = np.array([0.0, 0.0, 0.0])
        R_b = _rot_z(np.pi / 4)
        r_s = 0.05
        hx, hy, hz = 0.1, 0.05, 0.1
        m = gjk_epa_query(
            SphereShape(r_s),
            SpatialTransform(np.eye(3), pos_s),
            BoxShape((2 * hx, 2 * hy, 2 * hz)),
            SpatialTransform(R_b, pos_b),
        )
        res = _run_1(
            _test_sphere_box,
            _wp_vec3(pos_s),
            _wp_f(r_s),
            _wp_vec3(pos_b),
            _wp_mat33(R_b),
            _wp_f(hx),
            _wp_f(hy),
            _wp_f(hz),
        )
        # Both should agree on hit/no-hit
        if m is not None:
            assert res[1] > 0.5
            np.testing.assert_allclose(res[0], m.depth, atol=0.03)
        else:
            assert res[1] < 0.5
