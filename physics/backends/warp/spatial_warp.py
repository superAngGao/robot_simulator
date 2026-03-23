"""
Warp device functions for spatial algebra.

All functions are @wp.func — callable from @wp.kernel only.
They replicate the NumPy spatial.py operations in float32.

Convention (matches physics/spatial.py):
  - Spatial vectors: [linear(3); angular(3)]
  - SpatialTransform: R (child→parent rotation), r (child origin in parent)
  - Plücker velocity transform: v_child = apply_velocity(R, r, v_parent)
  - Plücker force dual transform: f_parent = apply_force(R, r, f_child)
"""

import warp as wp

# Custom types for 6D spatial algebra
mat66f = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
vec6f = wp.types.vector(length=6, dtype=wp.float32)


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------


@wp.func
def skew_wp(v: wp.vec3) -> wp.mat33:
    """Skew-symmetric matrix [v]× such that [v]× @ u == cross(v, u)."""
    return wp.mat33(
        0.0, -v[2], v[1],
        v[2], 0.0, -v[0],
        -v[1], v[0], 0.0,
    )


@wp.func
def vec6_from_two_vec3(lin: wp.vec3, ang: wp.vec3) -> vec6f:
    """Construct a 6D spatial vector from linear and angular parts."""
    return vec6f(lin[0], lin[1], lin[2], ang[0], ang[1], ang[2])


@wp.func
def vec6_linear(v: vec6f) -> wp.vec3:
    """Extract linear (first 3) components."""
    return wp.vec3(v[0], v[1], v[2])


@wp.func
def vec6_angular(v: vec6f) -> wp.vec3:
    """Extract angular (last 3) components."""
    return wp.vec3(v[3], v[4], v[5])


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------


@wp.func
def rodrigues_wp(axis: wp.vec3, angle: float) -> wp.mat33:
    """Rodrigues rotation: R = I cosθ + sinθ [k]× + (1-cosθ) k kᵀ."""
    c = wp.cos(angle)
    s = wp.sin(angle)
    K = skew_wp(axis)
    R = c * wp.identity(n=3, dtype=float) + s * K + (1.0 - c) * wp.outer(axis, axis)
    return R


@wp.func
def quat_to_rot_wp(qw: float, qx: float, qy: float, qz: float) -> wp.mat33:
    """Quaternion (w,x,y,z) to 3x3 rotation matrix."""
    # Normalize
    n = wp.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    w = qw / n
    x = qx / n
    y = qy / n
    z = qz / n

    return wp.mat33(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
        2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
        2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y),
    )


# ---------------------------------------------------------------------------
# Spatial transform operations
# ---------------------------------------------------------------------------


@wp.func
def transform_velocity_wp(R: wp.mat33, r: wp.vec3, v: vec6f) -> vec6f:
    """Apply Plücker velocity transform: parent frame → child frame.

    SE3 convention (matching spatial.py):
        v_lin_child = R^T @ (v_lin_parent - skew(v_ang_parent) @ r)
        v_ang_child = R^T @ v_ang_parent

    Equivalently:
        v_lin_child = E @ (v_lin + ω × r)   where E = R^T
        but with [lin; ang] convention v[:3] = linear, v[3:] = angular
    """
    E = wp.transpose(R)
    v_lin = wp.vec3(v[0], v[1], v[2])
    v_ang = wp.vec3(v[3], v[4], v[5])
    lin_new = E * (v_lin + wp.cross(v_ang, r))
    ang_new = E * v_ang
    return vec6_from_two_vec3(lin_new, ang_new)


@wp.func
def transform_force_wp(R: wp.mat33, r: wp.vec3, f: vec6f) -> vec6f:
    """Apply Plücker force dual transform: child frame → parent frame.

    SE3 convention (matching spatial.py):
        f_lin_parent = R @ f_lin_child
        f_ang_parent = R @ f_ang_child + r × (R @ f_lin_child)
    """
    f_lin = wp.vec3(f[0], f[1], f[2])
    f_ang = wp.vec3(f[3], f[4], f[5])
    Rf = R * f_lin
    lin_new = Rf
    ang_new = R * f_ang + wp.cross(r, Rf)
    return vec6_from_two_vec3(lin_new, ang_new)


@wp.func
def compose_transform_wp(
    R1: wp.mat33, r1: wp.vec3, R2: wp.mat33, r2: wp.vec3
) -> wp.mat33:
    """Compose two transforms: (R1,r1) @ (R2,r2).

    Returns R_new. Caller must also compute r_new = r1 + R1 @ r2.
    (Split to avoid tuple return.)
    """
    return R1 * R2


@wp.func
def compose_transform_r_wp(
    R1: wp.mat33, r1: wp.vec3, r2: wp.vec3
) -> wp.vec3:
    """Compute the translation part of composed transform: r_new = r1 + R1 @ r2."""
    return r1 + R1 * r2


@wp.func
def inverse_transform_R(R: wp.mat33) -> wp.mat33:
    """Rotation part of inverse transform: R_inv = R^T."""
    return wp.transpose(R)


@wp.func
def inverse_transform_r(R: wp.mat33, r: wp.vec3) -> wp.vec3:
    """Translation part of inverse transform: r_inv = -R^T @ r."""
    return -(wp.transpose(R) * r)


# ---------------------------------------------------------------------------
# Spatial cross products
# ---------------------------------------------------------------------------


@wp.func
def spatial_cross_vel_times_v(v: vec6f, u: vec6f) -> vec6f:
    """Compute v ×_velocity u (spatial cross product for velocities).

    Matrix layout [lin; ang]:
        [ skew(ω),  skew(v_l) ]  @ [u_lin]
        [   0,      skew(ω)   ]    [u_ang]
    """
    v_lin = wp.vec3(v[0], v[1], v[2])
    v_ang = wp.vec3(v[3], v[4], v[5])
    u_lin = wp.vec3(u[0], u[1], u[2])
    u_ang = wp.vec3(u[3], u[4], u[5])

    # Matrix layout [lin;ang]:
    #   [ skew(ω),  skew(v_l) ]
    #   [   0,      skew(ω)   ]
    res_lin = wp.cross(v_ang, u_lin) + wp.cross(v_lin, u_ang)
    res_ang = wp.cross(v_ang, u_ang)
    return vec6_from_two_vec3(res_lin, res_ang)


@wp.func
def spatial_cross_force_times_f(v: vec6f, f: vec6f) -> vec6f:
    """Compute v ×* f (spatial cross product for forces, dual).

    Matrix layout [lin; ang] (= -velocity_cross^T):
        [ skew(ω),     0       ]  @ [f_lin]
        [ skew(v_l),  skew(ω)  ]    [f_ang]
    """
    v_lin = wp.vec3(v[0], v[1], v[2])
    v_ang = wp.vec3(v[3], v[4], v[5])
    f_lin = wp.vec3(f[0], f[1], f[2])
    f_ang = wp.vec3(f[3], f[4], f[5])

    # Matrix layout [lin;ang]:
    #   [ skew(ω),     0       ]
    #   [ skew(v_l),  skew(ω)  ]
    res_lin = wp.cross(v_ang, f_lin)
    res_ang = wp.cross(v_lin, f_lin) + wp.cross(v_ang, f_ang)
    return vec6_from_two_vec3(res_lin, res_ang)


# ---------------------------------------------------------------------------
# 6x6 matrix operations
# ---------------------------------------------------------------------------


@wp.func
def mat66_mul_vec6(M: mat66f, v: vec6f) -> vec6f:
    """Matrix-vector product M @ v for 6x6."""
    result = vec6f()
    for i in range(6):
        s = float(0.0)
        for j in range(6):
            s = s + M[i, j] * v[j]
        result[i] = s
    return result


@wp.func
def mat66_mul_mat66(A: mat66f, B: mat66f) -> mat66f:
    """Matrix-matrix product A @ B for 6x6."""
    C = mat66f()
    for i in range(6):
        for j in range(6):
            s = float(0.0)
            for k in range(6):
                s = s + A[i, k] * B[k, j]
            C[i, j] = s
    return C


@wp.func
def mat66_add(A: mat66f, B: mat66f) -> mat66f:
    """Element-wise addition A + B for 6x6."""
    return A + B


@wp.func
def mat66_sub(A: mat66f, B: mat66f) -> mat66f:
    """Element-wise subtraction A - B for 6x6."""
    return A - B


@wp.func
def mat66_transpose(M: mat66f) -> mat66f:
    """Transpose of 6x6 matrix."""
    return wp.transpose(M)


@wp.func
def mat66_outer_vec6(u: vec6f, v: vec6f) -> mat66f:
    """Outer product u @ v^T for 6D vectors."""
    return wp.outer(u, v)


@wp.func
def vec6_dot(a: vec6f, b: vec6f) -> float:
    """Dot product of two 6D vectors."""
    return wp.dot(a, b)


@wp.func
def mat66_from_spatial_inertia_array(
    I_flat: wp.array(dtype=wp.float32, ndim=1),
) -> mat66f:
    """Load a 6x6 spatial inertia matrix from a flat (36,) array."""
    M = mat66f()
    for i in range(6):
        for j in range(6):
            M[i, j] = I_flat[i * 6 + j]
    return M


# ---------------------------------------------------------------------------
# Spatial transform matrix construction (6x6 Plücker form)
# ---------------------------------------------------------------------------


@wp.func
def spatial_transform_matrix(R: wp.mat33, r: wp.vec3) -> mat66f:
    """Build the 6x6 Plücker velocity transform matrix X.

    [lin; ang] convention:
        X = [ E,  -E @ skew(r) ]
            [ 0,   E           ]
    where E = R^T
    """
    E = wp.transpose(R)
    Er = E * skew_wp(r)  # E @ skew(r)

    M = mat66f()
    # Top-left 3x3: E
    for i in range(3):
        for j in range(3):
            M[i, j] = E[i, j]
    # Top-right 3x3: -E @ skew(r)
    for i in range(3):
        for j in range(3):
            M[i, j + 3] = -Er[i, j]
    # Bottom-left 3x3: 0
    for i in range(3):
        for j in range(3):
            M[i + 3, j] = 0.0
    # Bottom-right 3x3: E
    for i in range(3):
        for j in range(3):
            M[i + 3, j + 3] = E[i, j]
    return M
