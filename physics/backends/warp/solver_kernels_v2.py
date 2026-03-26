"""
Warp GPU solver kernels v2 — supports body-body contacts (not just ground).

Key differences from solver_kernels.py:
  - Reads contact normal/point/body_i/body_j from collision detection output
  - Supports body_j >= 0 (body-body) in addition to body_j = -1 (ground)
  - Jacobian and W assembly account for both bodies' mass contributions
"""

import warp as wp

from .spatial_warp import (
    inverse_transform_R,
    inverse_transform_r,
    transform_force_wp,
    vec6_angular,
    vec6_from_two_vec3,
    vec6_linear,
    vec6f,
)

JOINT_FREE = wp.constant(0)
JOINT_REVOLUTE = wp.constant(1)
JOINT_PRISMATIC = wp.constant(2)
JOINT_FIXED = wp.constant(3)
CONDIM = wp.constant(3)


@wp.func
def _build_tangent_frame(normal: wp.vec3) -> wp.mat33:
    """Build orthonormal tangent frame from contact normal.

    Returns mat33 where row 0 = t1, row 1 = t2 (stored as rows for convenience).
    """
    # Choose axis least parallel to normal
    abs_nx = wp.abs(normal[0])
    abs_ny = wp.abs(normal[1])
    abs_nz = wp.abs(normal[2])

    if abs_nx <= abs_ny and abs_nx <= abs_nz:
        ref = wp.vec3(1.0, 0.0, 0.0)
    elif abs_ny <= abs_nz:
        ref = wp.vec3(0.0, 1.0, 0.0)
    else:
        ref = wp.vec3(0.0, 0.0, 1.0)

    t1 = wp.normalize(wp.cross(normal, ref))
    t2 = wp.cross(normal, t1)
    return wp.mat33(t1[0], t1[1], t1[2], t2[0], t2[1], t2[2], 0.0, 0.0, 0.0)


@wp.kernel
def batched_build_W_vfree_v2(
    X_world_R: wp.array(dtype=wp.float32, ndim=4),
    X_world_r: wp.array3d(dtype=wp.float32),
    v_bodies_pred: wp.array3d(dtype=wp.float32),
    # Contact data from collision detection
    contact_active: wp.array2d(dtype=wp.int32),
    contact_normal: wp.array3d(dtype=wp.float32),  # (N, max_contacts, 3)
    contact_point: wp.array3d(dtype=wp.float32),  # (N, max_contacts, 3)
    contact_bi: wp.array2d(dtype=wp.int32),
    contact_bj: wp.array2d(dtype=wp.int32),
    # Mass properties
    inv_mass: wp.array(dtype=wp.float32),
    inv_inertia: wp.array3d(dtype=wp.float32),
    mu: float,
    cfm: float,
    max_contacts: int,
    max_rows: int,
    # Outputs
    W: wp.array(dtype=wp.float32, ndim=3),
    W_diag: wp.array2d(dtype=wp.float32),
    v_free: wp.array2d(dtype=wp.float32),
    J_body_i: wp.array3d(dtype=wp.float32),  # (N, max_rows, 6)
    J_body_j: wp.array3d(dtype=wp.float32),  # (N, max_rows, 6)
    row_bi: wp.array2d(dtype=wp.int32),  # body_i per row
    row_bj: wp.array2d(dtype=wp.int32),  # body_j per row
):
    """Build Jacobian + Delassus W + v_free for general contacts (ground + body-body)."""
    env_id = wp.tid()

    # Zero outputs
    for r in range(max_rows):
        W_diag[env_id, r] = 0.0
        v_free[env_id, r] = 0.0
        row_bi[env_id, r] = -1
        row_bj[env_id, r] = -1
        for k in range(6):
            J_body_i[env_id, r, k] = 0.0
            J_body_j[env_id, r, k] = 0.0
        for r2 in range(max_rows):
            W[env_id, r, r2] = 0.0

    for c in range(max_contacts):
        if contact_active[env_id, c] == 0:
            continue

        bi = contact_bi[env_id, c]
        bj = contact_bj[env_id, c]
        base = c * CONDIM

        # Contact frame
        normal = wp.vec3(
            contact_normal[env_id, c, 0],
            contact_normal[env_id, c, 1],
            contact_normal[env_id, c, 2],
        )
        frame = _build_tangent_frame(normal)
        t1 = wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2])
        t2 = wp.vec3(frame[1, 0], frame[1, 1], frame[1, 2])
        cp = wp.vec3(
            contact_point[env_id, c, 0],
            contact_point[env_id, c, 1],
            contact_point[env_id, c, 2],
        )

        # ── Body i Jacobian ──
        if bi >= 0:
            R_i = wp.mat33(
                X_world_R[env_id, bi, 0, 0],
                X_world_R[env_id, bi, 0, 1],
                X_world_R[env_id, bi, 0, 2],
                X_world_R[env_id, bi, 1, 0],
                X_world_R[env_id, bi, 1, 1],
                X_world_R[env_id, bi, 1, 2],
                X_world_R[env_id, bi, 2, 0],
                X_world_R[env_id, bi, 2, 1],
                X_world_R[env_id, bi, 2, 2],
            )
            r_i = wp.vec3(X_world_r[env_id, bi, 0], X_world_r[env_id, bi, 1], X_world_r[env_id, bi, 2])
            Rt_i = wp.transpose(R_i)
            r_arm_i = cp - r_i

            v_body_i = vec6f(
                v_bodies_pred[env_id, bi, 0],
                v_bodies_pred[env_id, bi, 1],
                v_bodies_pred[env_id, bi, 2],
                v_bodies_pred[env_id, bi, 3],
                v_bodies_pred[env_id, bi, 4],
                v_bodies_pred[env_id, bi, 5],
            )
            v_lin_i = R_i * vec6_linear(v_body_i)
            omega_i = R_i * vec6_angular(v_body_i)
            v_contact_i = v_lin_i + wp.cross(omega_i, r_arm_i)
        else:
            Rt_i = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            r_arm_i = wp.vec3(0.0, 0.0, 0.0)
            v_contact_i = wp.vec3(0.0, 0.0, 0.0)

        # ── Body j Jacobian ──
        if bj >= 0:
            R_j = wp.mat33(
                X_world_R[env_id, bj, 0, 0],
                X_world_R[env_id, bj, 0, 1],
                X_world_R[env_id, bj, 0, 2],
                X_world_R[env_id, bj, 1, 0],
                X_world_R[env_id, bj, 1, 1],
                X_world_R[env_id, bj, 1, 2],
                X_world_R[env_id, bj, 2, 0],
                X_world_R[env_id, bj, 2, 1],
                X_world_R[env_id, bj, 2, 2],
            )
            r_j = wp.vec3(X_world_r[env_id, bj, 0], X_world_r[env_id, bj, 1], X_world_r[env_id, bj, 2])
            Rt_j = wp.transpose(R_j)
            r_arm_j = cp - r_j

            v_body_j = vec6f(
                v_bodies_pred[env_id, bj, 0],
                v_bodies_pred[env_id, bj, 1],
                v_bodies_pred[env_id, bj, 2],
                v_bodies_pred[env_id, bj, 3],
                v_bodies_pred[env_id, bj, 4],
                v_bodies_pred[env_id, bj, 5],
            )
            v_lin_j = R_j * vec6_linear(v_body_j)
            omega_j = R_j * vec6_angular(v_body_j)
            v_contact_j = v_lin_j + wp.cross(omega_j, r_arm_j)
        else:
            Rt_j = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            r_arm_j = wp.vec3(0.0, 0.0, 0.0)
            v_contact_j = wp.vec3(0.0, 0.0, 0.0)

        # Relative contact velocity (i minus j)
        v_rel = v_contact_i - v_contact_j

        # Build rows for normal + tangent1 + tangent2
        for d_idx in range(CONDIM):
            row = base + d_idx
            if d_idx == 0:
                direction = normal
            elif d_idx == 1:
                direction = t1
            else:
                direction = t2

            # J_i: body i contribution
            if bi >= 0:
                J_lin_i = Rt_i * direction
                rxd_i = wp.cross(r_arm_i, direction)
                J_ang_i = Rt_i * rxd_i
                J_body_i[env_id, row, 0] = J_lin_i[0]
                J_body_i[env_id, row, 1] = J_lin_i[1]
                J_body_i[env_id, row, 2] = J_lin_i[2]
                J_body_i[env_id, row, 3] = J_ang_i[0]
                J_body_i[env_id, row, 4] = J_ang_i[1]
                J_body_i[env_id, row, 5] = J_ang_i[2]

            # J_j: body j contribution (negative — opposing direction)
            if bj >= 0:
                J_lin_j = Rt_j * direction
                rxd_j = wp.cross(r_arm_j, direction)
                J_ang_j = Rt_j * rxd_j
                J_body_j[env_id, row, 0] = J_lin_j[0]
                J_body_j[env_id, row, 1] = J_lin_j[1]
                J_body_j[env_id, row, 2] = J_lin_j[2]
                J_body_j[env_id, row, 3] = J_ang_j[0]
                J_body_j[env_id, row, 4] = J_ang_j[1]
                J_body_j[env_id, row, 5] = J_ang_j[2]

            row_bi[env_id, row] = bi
            row_bj[env_id, row] = bj

            # v_free = relative contact velocity projected onto direction
            v_free[env_id, row] = wp.dot(direction, v_rel)

    # ── Build W = J_i M_i^{-1} J_i^T + J_j M_j^{-1} J_j^T ──
    n_rows = max_contacts * CONDIM
    for r1 in range(n_rows):
        bi1 = row_bi[env_id, r1]
        bj1 = row_bj[env_id, r1]
        if bi1 < 0 and bj1 < 0:
            continue

        for r2 in range(n_rows):
            bi2 = row_bi[env_id, r2]
            bj2 = row_bj[env_id, r2]

            val = float(0.0)

            # Body i contribution (if shared between r1 and r2)
            if bi1 >= 0 and bi1 == bi2:
                m_inv = inv_mass[bi1]
                I_inv = wp.mat33(
                    inv_inertia[bi1, 0, 0],
                    inv_inertia[bi1, 0, 1],
                    inv_inertia[bi1, 0, 2],
                    inv_inertia[bi1, 1, 0],
                    inv_inertia[bi1, 1, 1],
                    inv_inertia[bi1, 1, 2],
                    inv_inertia[bi1, 2, 0],
                    inv_inertia[bi1, 2, 1],
                    inv_inertia[bi1, 2, 2],
                )
                j1 = wp.vec3(J_body_i[env_id, r1, 0], J_body_i[env_id, r1, 1], J_body_i[env_id, r1, 2])
                j1a = wp.vec3(J_body_i[env_id, r1, 3], J_body_i[env_id, r1, 4], J_body_i[env_id, r1, 5])
                j2 = wp.vec3(J_body_i[env_id, r2, 0], J_body_i[env_id, r2, 1], J_body_i[env_id, r2, 2])
                j2a = wp.vec3(J_body_i[env_id, r2, 3], J_body_i[env_id, r2, 4], J_body_i[env_id, r2, 5])
                val = val + wp.dot(j2, m_inv * j1) + wp.dot(j2a, I_inv * j1a)

            # Body j contribution (if shared between r1 and r2)
            if bj1 >= 0 and bj1 == bj2:
                m_inv_j = inv_mass[bj1]
                I_inv_j = wp.mat33(
                    inv_inertia[bj1, 0, 0],
                    inv_inertia[bj1, 0, 1],
                    inv_inertia[bj1, 0, 2],
                    inv_inertia[bj1, 1, 0],
                    inv_inertia[bj1, 1, 1],
                    inv_inertia[bj1, 1, 2],
                    inv_inertia[bj1, 2, 0],
                    inv_inertia[bj1, 2, 1],
                    inv_inertia[bj1, 2, 2],
                )
                j1 = wp.vec3(J_body_j[env_id, r1, 0], J_body_j[env_id, r1, 1], J_body_j[env_id, r1, 2])
                j1a = wp.vec3(J_body_j[env_id, r1, 3], J_body_j[env_id, r1, 4], J_body_j[env_id, r1, 5])
                j2 = wp.vec3(J_body_j[env_id, r2, 0], J_body_j[env_id, r2, 1], J_body_j[env_id, r2, 2])
                j2a = wp.vec3(J_body_j[env_id, r2, 3], J_body_j[env_id, r2, 4], J_body_j[env_id, r2, 5])
                val = val + wp.dot(j2, m_inv_j * j1) + wp.dot(j2a, I_inv_j * j1a)

            W[env_id, r1, r2] = W[env_id, r1, r2] + val

    # CFM on diagonal
    for r in range(n_rows):
        if row_bi[env_id, r] >= 0 or row_bj[env_id, r] >= 0:
            W_diag[env_id, r] = W[env_id, r, r] + cfm


@wp.kernel
def batched_impulse_to_gen_v2(
    lambdas: wp.array2d(dtype=wp.float32),
    contact_active: wp.array2d(dtype=wp.int32),
    contact_normal: wp.array3d(dtype=wp.float32),
    contact_point: wp.array3d(dtype=wp.float32),
    contact_bi: wp.array2d(dtype=wp.int32),
    contact_bj: wp.array2d(dtype=wp.int32),
    X_world_R: wp.array(dtype=wp.float32, ndim=4),
    X_world_r: wp.array3d(dtype=wp.float32),
    X_up_R: wp.array(dtype=wp.float32, ndim=4),
    X_up_r: wp.array3d(dtype=wp.float32),
    joint_type: wp.array(dtype=wp.int32),
    joint_axis: wp.array2d(dtype=wp.float32),
    parent_idx: wp.array(dtype=wp.int32),
    v_idx_start: wp.array(dtype=wp.int32),
    max_contacts: int,
    nb: int,
    nv: int,
    # Outputs
    body_impulses: wp.array3d(dtype=wp.float32),
    gen_impulse: wp.array2d(dtype=wp.float32),
):
    """Convert lambdas → body impulses → generalized impulse (body-body aware)."""
    env_id = wp.tid()

    for i in range(nb):
        for d in range(6):
            body_impulses[env_id, i, d] = 0.0
    for i in range(nv):
        gen_impulse[env_id, i] = 0.0

    # Step 1: lambdas → body-frame spatial impulses
    for c in range(max_contacts):
        if contact_active[env_id, c] == 0:
            continue

        bi = contact_bi[env_id, c]
        bj = contact_bj[env_id, c]
        base = c * CONDIM

        l_n = lambdas[env_id, base]
        l_t1 = lambdas[env_id, base + 1]
        l_t2 = lambdas[env_id, base + 2]

        normal = wp.vec3(
            contact_normal[env_id, c, 0],
            contact_normal[env_id, c, 1],
            contact_normal[env_id, c, 2],
        )
        frame = _build_tangent_frame(normal)
        t1 = wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2])
        t2 = wp.vec3(frame[1, 0], frame[1, 1], frame[1, 2])

        F_world = l_n * normal + l_t1 * t1 + l_t2 * t2
        cp = wp.vec3(
            contact_point[env_id, c, 0],
            contact_point[env_id, c, 1],
            contact_point[env_id, c, 2],
        )

        # Apply to body i (positive force)
        if bi >= 0:
            R_i = wp.mat33(
                X_world_R[env_id, bi, 0, 0],
                X_world_R[env_id, bi, 0, 1],
                X_world_R[env_id, bi, 0, 2],
                X_world_R[env_id, bi, 1, 0],
                X_world_R[env_id, bi, 1, 1],
                X_world_R[env_id, bi, 1, 2],
                X_world_R[env_id, bi, 2, 0],
                X_world_R[env_id, bi, 2, 1],
                X_world_R[env_id, bi, 2, 2],
            )
            r_i = wp.vec3(X_world_r[env_id, bi, 0], X_world_r[env_id, bi, 1], X_world_r[env_id, bi, 2])
            r_arm_i = cp - r_i
            torque_i = wp.cross(r_arm_i, F_world)
            Rinv_i = inverse_transform_R(R_i)
            rinv_i = inverse_transform_r(R_i, r_i)
            f_w_i = vec6_from_two_vec3(F_world, torque_i)
            f_b_i = transform_force_wp(Rinv_i, rinv_i, f_w_i)
            for d in range(6):
                body_impulses[env_id, bi, d] = body_impulses[env_id, bi, d] + f_b_i[d]

        # Apply to body j (negative force — Newton's 3rd law)
        if bj >= 0:
            R_j = wp.mat33(
                X_world_R[env_id, bj, 0, 0],
                X_world_R[env_id, bj, 0, 1],
                X_world_R[env_id, bj, 0, 2],
                X_world_R[env_id, bj, 1, 0],
                X_world_R[env_id, bj, 1, 1],
                X_world_R[env_id, bj, 1, 2],
                X_world_R[env_id, bj, 2, 0],
                X_world_R[env_id, bj, 2, 1],
                X_world_R[env_id, bj, 2, 2],
            )
            r_j = wp.vec3(X_world_r[env_id, bj, 0], X_world_r[env_id, bj, 1], X_world_r[env_id, bj, 2])
            r_arm_j = cp - r_j
            neg_F = wp.vec3(-F_world[0], -F_world[1], -F_world[2])
            torque_j = wp.cross(r_arm_j, neg_F)
            Rinv_j = inverse_transform_R(R_j)
            rinv_j = inverse_transform_r(R_j, r_j)
            f_w_j = vec6_from_two_vec3(neg_F, torque_j)
            f_b_j = transform_force_wp(Rinv_j, rinv_j, f_w_j)
            for d in range(6):
                body_impulses[env_id, bj, d] = body_impulses[env_id, bj, d] + f_b_j[d]

    # Step 2: RNEA backward pass
    for idx in range(nb):
        i = nb - 1 - idx
        f_i = vec6f(
            body_impulses[env_id, i, 0],
            body_impulses[env_id, i, 1],
            body_impulses[env_id, i, 2],
            body_impulses[env_id, i, 3],
            body_impulses[env_id, i, 4],
            body_impulses[env_id, i, 5],
        )

        jtype = joint_type[i]
        vs = v_idx_start[i]

        if jtype == JOINT_REVOLUTE or jtype == JOINT_PRISMATIC:
            axis = wp.vec3(joint_axis[i, 0], joint_axis[i, 1], joint_axis[i, 2])
            if jtype == JOINT_REVOLUTE:
                gen_impulse[env_id, vs] = axis[0] * f_i[3] + axis[1] * f_i[4] + axis[2] * f_i[5]
            else:
                gen_impulse[env_id, vs] = axis[0] * f_i[0] + axis[1] * f_i[1] + axis[2] * f_i[2]
        elif jtype == JOINT_FREE:
            for d in range(6):
                gen_impulse[env_id, vs + d] = f_i[d]

        pi = parent_idx[i]
        if pi >= 0:
            R_up = wp.mat33(
                X_up_R[env_id, i, 0, 0],
                X_up_R[env_id, i, 0, 1],
                X_up_R[env_id, i, 0, 2],
                X_up_R[env_id, i, 1, 0],
                X_up_R[env_id, i, 1, 1],
                X_up_R[env_id, i, 1, 2],
                X_up_R[env_id, i, 2, 0],
                X_up_R[env_id, i, 2, 1],
                X_up_R[env_id, i, 2, 2],
            )
            r_up = wp.vec3(X_up_r[env_id, i, 0], X_up_r[env_id, i, 1], X_up_r[env_id, i, 2])
            f_parent = transform_force_wp(R_up, r_up, f_i)
            for d in range(6):
                body_impulses[env_id, pi, d] = body_impulses[env_id, pi, d] + f_parent[d]
