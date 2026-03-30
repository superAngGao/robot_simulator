"""
Tests for Warp spatial algebra device functions.

Compares Warp (float32) implementations against NumPy (float64) reference
from physics/spatial.py. Tolerance: atol=1e-5 for single operations.
"""

from __future__ import annotations

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from physics.backends.warp.spatial_warp import (  # noqa: E402
    compose_transform_r_wp,
    compose_transform_wp,
    inverse_transform_R,
    inverse_transform_r,
    mat66_mul_vec6,
    mat66f,
    quat_to_rot_wp,
    rodrigues_wp,
    spatial_cross_force_times_f,
    spatial_cross_vel_times_v,
    spatial_transform_matrix,
    transform_force_wp,
    transform_velocity_wp,
    vec6f,
)
from physics.spatial import (  # noqa: E402
    SpatialTransform,
    quat_to_rot,
    rot_z,
    skew,
    spatial_cross_velocity,
)

wp.init()

ATOL = 1e-5
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Test kernels — each wraps one @wp.func to make it testable
# ---------------------------------------------------------------------------


@wp.kernel
def _test_rodrigues(
    axis: wp.array(dtype=wp.vec3),
    angle: wp.array(dtype=float),
    out: wp.array(dtype=wp.mat33),
):
    i = wp.tid()
    out[i] = rodrigues_wp(axis[i], angle[i])


@wp.kernel
def _test_quat_to_rot(
    qw: wp.array(dtype=float),
    qx: wp.array(dtype=float),
    qy: wp.array(dtype=float),
    qz: wp.array(dtype=float),
    out: wp.array(dtype=wp.mat33),
):
    i = wp.tid()
    out[i] = quat_to_rot_wp(qw[i], qx[i], qy[i], qz[i])


@wp.kernel
def _test_transform_velocity(
    R: wp.array(dtype=wp.mat33),
    r: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=vec6f),
    out: wp.array(dtype=vec6f),
):
    i = wp.tid()
    out[i] = transform_velocity_wp(R[i], r[i], v[i])


@wp.kernel
def _test_transform_force(
    R: wp.array(dtype=wp.mat33),
    r: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=vec6f),
    out: wp.array(dtype=vec6f),
):
    i = wp.tid()
    out[i] = transform_force_wp(R[i], r[i], f[i])


@wp.kernel
def _test_compose(
    R1: wp.array(dtype=wp.mat33),
    r1: wp.array(dtype=wp.vec3),
    R2: wp.array(dtype=wp.mat33),
    r2: wp.array(dtype=wp.vec3),
    out_R: wp.array(dtype=wp.mat33),
    out_r: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    out_R[i] = compose_transform_wp(R1[i], r1[i], R2[i], r2[i])
    out_r[i] = compose_transform_r_wp(R1[i], r1[i], r2[i])


@wp.kernel
def _test_inverse(
    R: wp.array(dtype=wp.mat33),
    r: wp.array(dtype=wp.vec3),
    out_R: wp.array(dtype=wp.mat33),
    out_r: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    out_R[i] = inverse_transform_R(R[i])
    out_r[i] = inverse_transform_r(R[i], r[i])


@wp.kernel
def _test_spatial_cross_vel(
    v: wp.array(dtype=vec6f),
    u: wp.array(dtype=vec6f),
    out: wp.array(dtype=vec6f),
):
    i = wp.tid()
    out[i] = spatial_cross_vel_times_v(v[i], u[i])


@wp.kernel
def _test_spatial_cross_force(
    v: wp.array(dtype=vec6f),
    f: wp.array(dtype=vec6f),
    out: wp.array(dtype=vec6f),
):
    i = wp.tid()
    out[i] = spatial_cross_force_times_f(v[i], f[i])


@wp.kernel
def _test_transform_matrix(
    R: wp.array(dtype=wp.mat33),
    r: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=mat66f),
):
    i = wp.tid()
    out[i] = spatial_transform_matrix(R[i], r[i])


@wp.kernel
def _test_mat66_mul_vec6(
    M: wp.array(dtype=mat66f),
    v: wp.array(dtype=vec6f),
    out: wp.array(dtype=vec6f),
):
    i = wp.tid()
    out[i] = mat66_mul_vec6(M[i], v[i])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_rotation():
    """Random rotation matrix via axis-angle."""
    axis = RNG.standard_normal(3)
    axis /= np.linalg.norm(axis)
    angle = RNG.uniform(-np.pi, np.pi)
    K = skew(axis)
    c, s = np.cos(angle), np.sin(angle)
    return c * np.eye(3) + s * K + (1.0 - c) * np.outer(axis, axis)


def _random_transform():
    R = _random_rotation()
    r = RNG.standard_normal(3)
    return SpatialTransform(R, r)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRodrigues:
    def test_identity(self):
        axis = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        angle = np.array([0.0], dtype=np.float32)
        out = wp.zeros(1, dtype=wp.mat33)
        wp.launch(
            _test_rodrigues,
            dim=1,
            inputs=[
                wp.array(axis, dtype=wp.vec3),
                wp.array(angle, dtype=float),
                out,
            ],
        )
        R = out.numpy()[0]
        np.testing.assert_allclose(R, np.eye(3), atol=ATOL)

    def test_90deg_z(self):
        axis = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        angle = np.array([np.pi / 2], dtype=np.float32)
        out = wp.zeros(1, dtype=wp.mat33)
        wp.launch(
            _test_rodrigues,
            dim=1,
            inputs=[
                wp.array(axis, dtype=wp.vec3),
                wp.array(angle, dtype=float),
                out,
            ],
        )
        R_warp = out.numpy()[0]
        R_np = rot_z(np.pi / 2)
        np.testing.assert_allclose(R_warp, R_np, atol=ATOL)

    def test_random_axis(self):
        axis_np = RNG.standard_normal(3).astype(np.float64)
        axis_np /= np.linalg.norm(axis_np)
        angle_val = float(RNG.uniform(-np.pi, np.pi))

        axis_f32 = axis_np.astype(np.float32).reshape(1, 3)
        angle_f32 = np.array([angle_val], dtype=np.float32)
        out = wp.zeros(1, dtype=wp.mat33)
        wp.launch(
            _test_rodrigues,
            dim=1,
            inputs=[
                wp.array(axis_f32, dtype=wp.vec3),
                wp.array(angle_f32, dtype=float),
                out,
            ],
        )
        R_warp = out.numpy()[0]

        # NumPy reference
        K = skew(axis_np)
        c, s = np.cos(angle_val), np.sin(angle_val)
        R_np = c * np.eye(3) + s * K + (1.0 - c) * np.outer(axis_np, axis_np)
        np.testing.assert_allclose(R_warp, R_np, atol=ATOL)


class TestQuatToRot:
    def test_identity_quat(self):
        out = wp.zeros(1, dtype=wp.mat33)
        wp.launch(
            _test_quat_to_rot,
            dim=1,
            inputs=[
                wp.array([1.0], dtype=float),
                wp.array([0.0], dtype=float),
                wp.array([0.0], dtype=float),
                wp.array([0.0], dtype=float),
                out,
            ],
        )
        R = out.numpy()[0]
        np.testing.assert_allclose(R, np.eye(3), atol=ATOL)

    def test_vs_numpy(self):
        # Random quaternion
        q = RNG.standard_normal(4)
        q /= np.linalg.norm(q)

        out = wp.zeros(1, dtype=wp.mat33)
        wp.launch(
            _test_quat_to_rot,
            dim=1,
            inputs=[
                wp.array([q[0]], dtype=float),
                wp.array([q[1]], dtype=float),
                wp.array([q[2]], dtype=float),
                wp.array([q[3]], dtype=float),
                out,
            ],
        )
        R_warp = out.numpy()[0]
        R_np = quat_to_rot(q)
        np.testing.assert_allclose(R_warp, R_np, atol=ATOL)


class TestTransformVelocity:
    def test_vs_numpy(self):
        X = _random_transform()
        v = RNG.standard_normal(6)

        R_f32 = X.R.astype(np.float32).reshape(1, 3, 3)
        r_f32 = X.r.astype(np.float32).reshape(1, 3)
        v_f32 = v.astype(np.float32).reshape(1, 6)
        out = wp.zeros(1, dtype=vec6f)
        wp.launch(
            _test_transform_velocity,
            dim=1,
            inputs=[
                wp.array(R_f32, dtype=wp.mat33),
                wp.array(r_f32, dtype=wp.vec3),
                wp.array(v_f32, dtype=vec6f),
                out,
            ],
        )
        result_warp = out.numpy()[0]
        result_np = X.apply_velocity(v)
        np.testing.assert_allclose(result_warp, result_np, atol=ATOL)


class TestTransformForce:
    def test_vs_numpy(self):
        X = _random_transform()
        f = RNG.standard_normal(6)

        R_f32 = X.R.astype(np.float32).reshape(1, 3, 3)
        r_f32 = X.r.astype(np.float32).reshape(1, 3)
        f_f32 = f.astype(np.float32).reshape(1, 6)
        out = wp.zeros(1, dtype=vec6f)
        wp.launch(
            _test_transform_force,
            dim=1,
            inputs=[
                wp.array(R_f32, dtype=wp.mat33),
                wp.array(r_f32, dtype=wp.vec3),
                wp.array(f_f32, dtype=vec6f),
                out,
            ],
        )
        result_warp = out.numpy()[0]
        result_np = X.apply_force(f)
        np.testing.assert_allclose(result_warp, result_np, atol=ATOL)


class TestCompose:
    def test_vs_numpy(self):
        X1 = _random_transform()
        X2 = _random_transform()
        X12_np = X1.compose(X2)

        R1 = X1.R.astype(np.float32).reshape(1, 3, 3)
        r1 = X1.r.astype(np.float32).reshape(1, 3)
        R2 = X2.R.astype(np.float32).reshape(1, 3, 3)
        r2 = X2.r.astype(np.float32).reshape(1, 3)
        out_R = wp.zeros(1, dtype=wp.mat33)
        out_r = wp.zeros(1, dtype=wp.vec3)
        wp.launch(
            _test_compose,
            dim=1,
            inputs=[
                wp.array(R1, dtype=wp.mat33),
                wp.array(r1, dtype=wp.vec3),
                wp.array(R2, dtype=wp.mat33),
                wp.array(r2, dtype=wp.vec3),
                out_R,
                out_r,
            ],
        )
        np.testing.assert_allclose(out_R.numpy()[0], X12_np.R, atol=ATOL)
        np.testing.assert_allclose(out_r.numpy()[0], X12_np.r, atol=ATOL)


class TestInverse:
    def test_vs_numpy(self):
        X = _random_transform()
        Xinv_np = X.inverse()

        R_f32 = X.R.astype(np.float32).reshape(1, 3, 3)
        r_f32 = X.r.astype(np.float32).reshape(1, 3)
        out_R = wp.zeros(1, dtype=wp.mat33)
        out_r = wp.zeros(1, dtype=wp.vec3)
        wp.launch(
            _test_inverse,
            dim=1,
            inputs=[
                wp.array(R_f32, dtype=wp.mat33),
                wp.array(r_f32, dtype=wp.vec3),
                out_R,
                out_r,
            ],
        )
        np.testing.assert_allclose(out_R.numpy()[0], Xinv_np.R, atol=ATOL)
        np.testing.assert_allclose(out_r.numpy()[0], Xinv_np.r, atol=ATOL)


class TestSpatialCross:
    def test_velocity_cross_vs_numpy(self):
        v = RNG.standard_normal(6)
        u = RNG.standard_normal(6)

        v_f32 = v.astype(np.float32).reshape(1, 6)
        u_f32 = u.astype(np.float32).reshape(1, 6)
        out = wp.zeros(1, dtype=vec6f)
        wp.launch(
            _test_spatial_cross_vel,
            dim=1,
            inputs=[
                wp.array(v_f32, dtype=vec6f),
                wp.array(u_f32, dtype=vec6f),
                out,
            ],
        )
        result_warp = out.numpy()[0]
        result_np = spatial_cross_velocity(v) @ u
        np.testing.assert_allclose(result_warp, result_np, atol=ATOL)

    def test_force_cross_vs_numpy(self):
        v = RNG.standard_normal(6)
        f = RNG.standard_normal(6)

        v_f32 = v.astype(np.float32).reshape(1, 6)
        f_f32 = f.astype(np.float32).reshape(1, 6)
        out = wp.zeros(1, dtype=vec6f)
        wp.launch(
            _test_spatial_cross_force,
            dim=1,
            inputs=[
                wp.array(v_f32, dtype=vec6f),
                wp.array(f_f32, dtype=vec6f),
                out,
            ],
        )
        result_warp = out.numpy()[0]

        # Force cross = -vel_cross^T @ f
        from physics.spatial import spatial_cross_velocity

        result_np = -spatial_cross_velocity(v).T @ f
        np.testing.assert_allclose(result_warp, result_np, atol=ATOL)


class TestTransformMatrix:
    def test_vs_numpy(self):
        X = _random_transform()

        R_f32 = X.R.astype(np.float32).reshape(1, 3, 3)
        r_f32 = X.r.astype(np.float32).reshape(1, 3)
        out = wp.zeros(1, dtype=mat66f)
        wp.launch(
            _test_transform_matrix,
            dim=1,
            inputs=[
                wp.array(R_f32, dtype=wp.mat33),
                wp.array(r_f32, dtype=wp.vec3),
                out,
            ],
        )
        M_warp = out.numpy()[0]
        M_np = X.matrix()
        np.testing.assert_allclose(M_warp, M_np, atol=ATOL)

    def test_matrix_velocity_consistency(self):
        """X.matrix() @ v should equal transform_velocity(v)."""
        X = _random_transform()
        v = RNG.standard_normal(6)

        R_f32 = X.R.astype(np.float32).reshape(1, 3, 3)
        r_f32 = X.r.astype(np.float32).reshape(1, 3)
        v_f32 = v.astype(np.float32).reshape(1, 6)

        out_M = wp.zeros(1, dtype=mat66f)
        out_v = wp.zeros(1, dtype=vec6f)
        wp.launch(
            _test_transform_matrix,
            dim=1,
            inputs=[
                wp.array(R_f32, dtype=wp.mat33),
                wp.array(r_f32, dtype=wp.vec3),
                out_M,
            ],
        )
        wp.launch(
            _test_mat66_mul_vec6,
            dim=1,
            inputs=[
                out_M,
                wp.array(v_f32, dtype=vec6f),
                out_v,
            ],
        )
        Mv = out_v.numpy()[0]

        out_direct = wp.zeros(1, dtype=vec6f)
        wp.launch(
            _test_transform_velocity,
            dim=1,
            inputs=[
                wp.array(R_f32, dtype=wp.mat33),
                wp.array(r_f32, dtype=wp.vec3),
                wp.array(v_f32, dtype=vec6f),
                out_direct,
            ],
        )
        direct = out_direct.numpy()[0]

        np.testing.assert_allclose(Mv, direct, atol=ATOL)
