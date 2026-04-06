"""
Warp GPU kernels for the joint-space Delassus pipeline (Q29).

Replaces the body-level Delassus + ABA-based pipeline with:
  1. CRBA → H (mass matrix)
  2. RNEA → C (bias forces)
  3. Cholesky(H) → L (shared factorization)
  4. Contact Jacobian → J (joint-space, n_rows × nv)
  5. W = J L⁻ᵀ L⁻¹ Jᵀ (joint-space Delassus)
  6. Impulse: gen_impulse = Jᵀ λ, dqdot = L⁻ᵀ L⁻¹ gen_impulse

All 6×6 matrices use (N, nb, 6, 6) arrays — Warp has no mat66 type.

Reference:
  - Featherstone (2008) §6.2 — CRBA
  - robot_tree.py:crba(), contact_jacobian() — CPU reference
"""

from __future__ import annotations

import warp as wp

from .kernels import JOINT_FREE, JOINT_PRISMATIC, JOINT_REVOLUTE
from .solver_kernels import CONDIM, _impedance_wp
from .spatial_warp import (
    quat_to_rot_wp,
    spatial_cross_force_times_f,
    spatial_cross_vel_times_v,
    transform_force_wp,
    transform_velocity_wp,
    vec6f,
)

# ---------------------------------------------------------------------------
# 6×6 array helpers — operate directly on (N, nb, 6, 6) arrays
# ---------------------------------------------------------------------------


@wp.func
def _ic_transform_add(
    IC: wp.array(dtype=wp.float32, ndim=4),
    env: int,
    parent: int,
    child: int,
    R_up: wp.mat33,
    r_up: wp.vec3,
):
    """IC[parent] += X_up^T @ IC[child] @ X_up (inertia transform)."""
    # Compute column-by-column: for each j, col_j = X^T @ (IC @ (X @ e_j))
    for j in range(6):
        # e_j
        ej = vec6f()
        ej[j] = 1.0
        # X @ e_j (velocity transform child→parent)
        x_ej = transform_velocity_wp(R_up, r_up, ej)
        # IC @ (X @ e_j)
        ic_x_ej = vec6f()
        for i in range(6):
            s = float(0.0)
            for k in range(6):
                s += IC[env, child, i, k] * x_ej[k]
            ic_x_ej[i] = s
        # X^T @ result (force transform)
        xt_ic_x_ej = transform_force_wp(R_up, r_up, ic_x_ej)
        # Accumulate into parent
        for i in range(6):
            IC[env, parent, i, j] = IC[env, parent, i, j] + xt_ic_x_ej[i]


@wp.func
def _ic_mul_vec(
    IC: wp.array(dtype=wp.float32, ndim=4),
    env: int,
    body: int,
    v: vec6f,
) -> vec6f:
    """Return IC[body] @ v."""
    result = vec6f()
    for i in range(6):
        s = float(0.0)
        for k in range(6):
            s += IC[env, body, i, k] * v[k]
        result[i] = s
    return result


@wp.func
def _inertia_mul_vec(
    inertia_mat: wp.array(dtype=wp.float32, ndim=3),
    body: int,
    v: vec6f,
) -> vec6f:
    """Return inertia_mat[body] @ v (body inertia, not composite)."""
    result = vec6f()
    for i in range(6):
        s = float(0.0)
        for k in range(6):
            s += inertia_mat[body, i, k] * v[k]
        result[i] = s
    return result


@wp.func
def _motion_subspace_col(jtype: int, axis: wp.vec3, col: int) -> vec6f:
    """Column `col` of motion subspace S. Convention: [linear(3); angular(3)].

    For FreeJoint with MuJoCo mixed-frame convention (world linear, body angular),
    S = [[R_J^T, 0], [0, I_3]].  When R_J is not supplied (identity default),
    this returns the identity column — use _motion_subspace_col_free() for the
    full q-dependent version.
    """
    s = vec6f()
    if jtype == JOINT_REVOLUTE:
        s[3] = axis[0]
        s[4] = axis[1]
        s[5] = axis[2]
    elif jtype == JOINT_PRISMATIC:
        s[0] = axis[0]
        s[1] = axis[1]
        s[2] = axis[2]
    elif jtype == JOINT_FREE:
        s[col] = 1.0
    return s


@wp.func
def _motion_subspace_col_free(col: int, R_J: wp.mat33) -> vec6f:
    """Column `col` of FreeJoint S = [[R_J^T, 0], [0, I]].

    MuJoCo convention: qdot[:3] world linear, qdot[3:] body angular.
    """
    s = vec6f()
    if col < 3:
        # Column col of R_J^T: element [i] = R_J^T[i, col] = R_J[col, i]
        s[0] = R_J[col, 0]
        s[1] = R_J[col, 1]
        s[2] = R_J[col, 2]
    else:
        s[col] = 1.0
    return s


@wp.func
def _load_Rup(
    X_up_R: wp.array(dtype=wp.float32, ndim=4),
    X_up_r: wp.array(dtype=wp.float32, ndim=3),
    env: int,
    body: int,
) -> wp.mat33:
    return wp.mat33(
        X_up_R[env, body, 0, 0],
        X_up_R[env, body, 0, 1],
        X_up_R[env, body, 0, 2],
        X_up_R[env, body, 1, 0],
        X_up_R[env, body, 1, 1],
        X_up_R[env, body, 1, 2],
        X_up_R[env, body, 2, 0],
        X_up_R[env, body, 2, 1],
        X_up_R[env, body, 2, 2],
    )


@wp.func
def _load_rup(
    X_up_r: wp.array(dtype=wp.float32, ndim=3),
    env: int,
    body: int,
) -> wp.vec3:
    return wp.vec3(X_up_r[env, body, 0], X_up_r[env, body, 1], X_up_r[env, body, 2])


@wp.func
def _load_RJ_free(
    q: wp.array(dtype=wp.float32, ndim=2),
    env: int,
    qs: int,
) -> wp.mat33:
    """Compute R_J from FreeJoint quaternion q[env, qs:qs+4]."""
    return quat_to_rot_wp(q[env, qs], q[env, qs + 1], q[env, qs + 2], q[env, qs + 3])


# ---------------------------------------------------------------------------
# Cholesky helpers
# ---------------------------------------------------------------------------


@wp.func
def _chol_factor(
    env: int,
    M: wp.array(dtype=wp.float32, ndim=3),
    L: wp.array(dtype=wp.float32, ndim=3),
    n: int,
    reg: float,
):
    for i in range(n):
        for j in range(n):
            L[env, i, j] = 0.0
    for i in range(n):
        for j in range(i + 1):
            s = float(0.0)
            for k in range(j):
                s += L[env, i, k] * L[env, j, k]
            if i == j:
                val = M[env, i, i] - s
                if val < reg:
                    val = reg
                L[env, i, j] = wp.sqrt(val)
            else:
                L_jj = L[env, j, j]
                if L_jj > 1.0e-12:
                    L[env, i, j] = (M[env, i, j] - s) / L_jj


@wp.func
def _chol_solve(
    env: int,
    L: wp.array(dtype=wp.float32, ndim=3),
    rhs: wp.array(dtype=wp.float32, ndim=2),
    x: wp.array(dtype=wp.float32, ndim=2),
    tmp: wp.array(dtype=wp.float32, ndim=2),
    n: int,
):
    for i in range(n):
        s = rhs[env, i]
        for k in range(i):
            s -= L[env, i, k] * tmp[env, k]
        L_ii = L[env, i, i]
        if L_ii > 1.0e-12:
            tmp[env, i] = s / L_ii
        else:
            tmp[env, i] = 0.0
    for idx in range(n):
        i = n - 1 - idx
        s = tmp[env, i]
        for k in range(i + 1, n):
            s -= L[env, k, i] * x[env, k]
        L_ii = L[env, i, i]
        if L_ii > 1.0e-12:
            x[env, i] = s / L_ii
        else:
            x[env, i] = 0.0


# ---------------------------------------------------------------------------
# Kernel 1: batched_crba_rnea_cholesky
# ---------------------------------------------------------------------------


@wp.kernel
def batched_crba_rnea_cholesky(
    q: wp.array(dtype=wp.float32, ndim=2),
    qdot: wp.array(dtype=wp.float32, ndim=2),
    tau_total: wp.array(dtype=wp.float32, ndim=2),
    joint_type: wp.array(dtype=wp.int32),
    joint_axis: wp.array(dtype=wp.float32, ndim=2),
    parent_idx: wp.array(dtype=wp.int32),
    q_idx_start: wp.array(dtype=wp.int32),
    v_idx_start: wp.array(dtype=wp.int32),
    v_idx_len: wp.array(dtype=wp.int32),
    inertia_mat: wp.array(dtype=wp.float32, ndim=3),  # (nb, 6, 6)
    X_up_R: wp.array(dtype=wp.float32, ndim=4),
    X_up_r: wp.array(dtype=wp.float32, ndim=3),
    gravity: float,
    nb: int,
    nv: int,
    # Scratch
    IC: wp.array(dtype=wp.float32, ndim=4),  # (N, nb, 6, 6)
    rnea_v: wp.array(dtype=wp.float32, ndim=3),
    rnea_a: wp.array(dtype=wp.float32, ndim=3),
    rnea_f: wp.array(dtype=wp.float32, ndim=3),
    chol_tmp: wp.array(dtype=wp.float32, ndim=2),
    # Outputs
    H: wp.array(dtype=wp.float32, ndim=3),  # (N, nv, nv)
    L_H: wp.array(dtype=wp.float32, ndim=3),
    C_bias: wp.array(dtype=wp.float32, ndim=2),  # (N, nv) — overwritten to tau-C
    qacc_smooth: wp.array(dtype=wp.float32, ndim=2),
):
    env = wp.tid()

    # ── 1. Init IC = body inertia ──
    for i in range(nb):
        for r in range(6):
            for c in range(6):
                IC[env, i, r, c] = inertia_mat[i, r, c]

    # ── 2. CRBA backward: IC[parent] += X^T IC[child] X ──
    for idx in range(nb):
        i = nb - 1 - idx
        pi = parent_idx[i]
        if pi >= 0:
            R_up = _load_Rup(X_up_R, X_up_r, env, i)
            r_up = _load_rup(X_up_r, env, i)
            _ic_transform_add(IC, env, pi, i, R_up, r_up)

    # ── 3. Build H ──
    for r in range(nv):
        for c in range(nv):
            H[env, r, c] = 0.0

    for i in range(nb):
        jtype = joint_type[i]
        vl = v_idx_len[i]
        if vl == 0:
            continue
        vs = v_idx_start[i]
        axis = wp.vec3(joint_axis[i, 0], joint_axis[i, 1], joint_axis[i, 2])

        # For FreeJoint, S depends on R_J (q-dependent)
        RJ_i = wp.identity(n=3, dtype=float)
        if jtype == JOINT_FREE:
            qs_i = q_idx_start[i]
            RJ_i = _load_RJ_free(q, env, qs_i)

        for col in range(vl):
            if jtype == JOINT_FREE:
                S_col = _motion_subspace_col_free(col, RJ_i)
            else:
                S_col = _motion_subspace_col(jtype, axis, col)
            F_col = _ic_mul_vec(IC, env, i, S_col)

            # Diagonal block
            for k in range(vl):
                if jtype == JOINT_FREE:
                    S_k = _motion_subspace_col_free(k, RJ_i)
                else:
                    S_k = _motion_subspace_col(jtype, axis, k)
                val = float(0.0)
                for d in range(6):
                    val += S_k[d] * F_col[d]
                if k >= col:
                    H[env, vs + col, vs + k] = val
                    H[env, vs + k, vs + col] = val

            # Off-diagonal: propagate F up tree
            j = i
            while parent_idx[j] >= 0:
                R_j = _load_Rup(X_up_R, X_up_r, env, j)
                r_j = _load_rup(X_up_r, env, j)
                F_col = transform_force_wp(R_j, r_j, F_col)
                j = parent_idx[j]

                vl_j = v_idx_len[j]
                if vl_j > 0:
                    vs_j = v_idx_start[j]
                    ax_j = wp.vec3(joint_axis[j, 0], joint_axis[j, 1], joint_axis[j, 2])
                    jt_j = joint_type[j]

                    # For FreeJoint ancestor, load R_J
                    RJ_j = wp.identity(n=3, dtype=float)
                    if jt_j == JOINT_FREE:
                        qs_j = q_idx_start[j]
                        RJ_j = _load_RJ_free(q, env, qs_j)

                    for k in range(vl_j):
                        if jt_j == JOINT_FREE:
                            S_j = _motion_subspace_col_free(k, RJ_j)
                        else:
                            S_j = _motion_subspace_col(jt_j, ax_j, k)
                        val = float(0.0)
                        for d in range(6):
                            val += S_j[d] * F_col[d]
                        H[env, vs_j + k, vs + col] = val
                        H[env, vs + col, vs_j + k] = val

    # ── 4. RNEA forward: v, a, f ──
    # Featherstone convention: root "accelerates" at -gravity (as if in
    # non-inertial frame). RNEA uses a_root = X_up @ (-a_gravity_spatial).
    # With gravity pointing down: a_gravity_spatial = (0,0,-g,0,0,0),
    # so -a_gravity_spatial = (0,0,+g,0,0,0).
    neg_a_gravity = vec6f(0.0, 0.0, gravity, 0.0, 0.0, 0.0)
    for i in range(nb):
        jtype = joint_type[i]
        vl = v_idx_len[i]
        axis = wp.vec3(joint_axis[i, 0], joint_axis[i, 1], joint_axis[i, 2])
        R_up = _load_Rup(X_up_R, X_up_r, env, i)
        r_up = _load_rup(X_up_r, env, i)

        vJ = vec6f()
        if vl > 0:
            vs = v_idx_start[i]
            if jtype == JOINT_REVOLUTE:
                qd = qdot[env, vs]
                vJ = vec6f(0.0, 0.0, 0.0, axis[0] * qd, axis[1] * qd, axis[2] * qd)
            elif jtype == JOINT_PRISMATIC:
                qd = qdot[env, vs]
                vJ = vec6f(axis[0] * qd, axis[1] * qd, axis[2] * qd, 0.0, 0.0, 0.0)
            elif jtype == JOINT_FREE:
                # MuJoCo convention: qdot[:3] world-frame linear, qdot[3:] body angular.
                # vJ = S @ qdot = [R_J^T @ v_world; omega_body] (body-frame spatial).
                qs_free = q_idx_start[i]
                RJ_free = _load_RJ_free(q, env, qs_free)
                v_world = wp.vec3(qdot[env, vs], qdot[env, vs + 1], qdot[env, vs + 2])
                v_body_lin = wp.transpose(RJ_free) * v_world
                omega_body = wp.vec3(qdot[env, vs + 3], qdot[env, vs + 4], qdot[env, vs + 5])
                vJ = vec6f(
                    v_body_lin[0], v_body_lin[1], v_body_lin[2], omega_body[0], omega_body[1], omega_body[2]
                )

        # Coriolis bias for FreeJoint: c_J = [-omega x v_body_lin, 0]
        cJ_free = vec6f()
        if jtype == JOINT_FREE and vl > 0:
            omega_cJ = wp.vec3(vJ[3], vJ[4], vJ[5])
            vlin_cJ = wp.vec3(vJ[0], vJ[1], vJ[2])
            wxv = wp.cross(omega_cJ, vlin_cJ)
            cJ_free = vec6f(-wxv[0], -wxv[1], -wxv[2], 0.0, 0.0, 0.0)

        pi = parent_idx[i]
        if pi < 0:
            v_i = vJ
            a_i = transform_velocity_wp(R_up, r_up, neg_a_gravity)
            # Add Coriolis bias for FreeJoint root
            if jtype == JOINT_FREE:
                for d in range(6):
                    a_i[d] = a_i[d] + cJ_free[d]
        else:
            v_parent = vec6f(
                rnea_v[env, pi, 0],
                rnea_v[env, pi, 1],
                rnea_v[env, pi, 2],
                rnea_v[env, pi, 3],
                rnea_v[env, pi, 4],
                rnea_v[env, pi, 5],
            )
            v_xf = transform_velocity_wp(R_up, r_up, v_parent)
            v_i = vec6f()
            for d in range(6):
                v_i[d] = v_xf[d] + vJ[d]
            a_parent = vec6f(
                rnea_a[env, pi, 0],
                rnea_a[env, pi, 1],
                rnea_a[env, pi, 2],
                rnea_a[env, pi, 3],
                rnea_a[env, pi, 4],
                rnea_a[env, pi, 5],
            )
            a_xf = transform_velocity_wp(R_up, r_up, a_parent)
            cJ = spatial_cross_vel_times_v(v_i, vJ)
            a_i = vec6f()
            for d in range(6):
                a_i[d] = a_xf[d] + cJ[d] + cJ_free[d]

        for d in range(6):
            rnea_v[env, i, d] = v_i[d]
            rnea_a[env, i, d] = a_i[d]

        # f[i] = I_body @ a + v x* (I_body @ v)
        Ia = _inertia_mul_vec(inertia_mat, i, a_i)
        Iv = _inertia_mul_vec(inertia_mat, i, v_i)
        vxIv = spatial_cross_force_times_f(v_i, Iv)
        for d in range(6):
            rnea_f[env, i, d] = Ia[d] + vxIv[d]

    # RNEA backward: project onto S, propagate up
    for idx in range(nb):
        i = nb - 1 - idx
        jtype = joint_type[i]
        vl = v_idx_len[i]
        f_i = vec6f(
            rnea_f[env, i, 0],
            rnea_f[env, i, 1],
            rnea_f[env, i, 2],
            rnea_f[env, i, 3],
            rnea_f[env, i, 4],
            rnea_f[env, i, 5],
        )

        if vl > 0:
            vs = v_idx_start[i]
            axis = wp.vec3(joint_axis[i, 0], joint_axis[i, 1], joint_axis[i, 2])
            if jtype == JOINT_REVOLUTE:
                C_bias[env, vs] = axis[0] * f_i[3] + axis[1] * f_i[4] + axis[2] * f_i[5]
            elif jtype == JOINT_PRISMATIC:
                C_bias[env, vs] = axis[0] * f_i[0] + axis[1] * f_i[1] + axis[2] * f_i[2]
            elif jtype == JOINT_FREE:
                # S^T = [[R_J, 0], [0, I]]:  tau[:3] = R_J @ f[:3], tau[3:6] = f[3:6]
                qs_bk = q_idx_start[i]
                RJ_bk = _load_RJ_free(q, env, qs_bk)
                f_lin = wp.vec3(f_i[0], f_i[1], f_i[2])
                tau_lin = RJ_bk * f_lin
                C_bias[env, vs + 0] = tau_lin[0]
                C_bias[env, vs + 1] = tau_lin[1]
                C_bias[env, vs + 2] = tau_lin[2]
                C_bias[env, vs + 3] = f_i[3]
                C_bias[env, vs + 4] = f_i[4]
                C_bias[env, vs + 5] = f_i[5]

        pi = parent_idx[i]
        if pi >= 0:
            R_up = _load_Rup(X_up_R, X_up_r, env, i)
            r_up = _load_rup(X_up_r, env, i)
            f_parent_contrib = transform_force_wp(R_up, r_up, f_i)
            for d in range(6):
                rnea_f[env, pi, d] = rnea_f[env, pi, d] + f_parent_contrib[d]

    # ── 5. Cholesky(H) → L ──
    _chol_factor(env, H, L_H, nv, 1.0e-6)

    # ── 6. qacc = L⁻ᵀ L⁻¹ (tau - C) ──
    for k in range(nv):
        C_bias[env, k] = tau_total[env, k] - C_bias[env, k]
    _chol_solve(env, L_H, C_bias, qacc_smooth, chol_tmp, nv)


# ---------------------------------------------------------------------------
# Kernel 2: batched_contact_jacobian
# ---------------------------------------------------------------------------


@wp.kernel
def batched_contact_jacobian(
    q: wp.array(dtype=wp.float32, ndim=2),
    X_world_R: wp.array(dtype=wp.float32, ndim=4),
    X_world_r: wp.array(dtype=wp.float32, ndim=3),
    contact_active: wp.array(dtype=wp.int32, ndim=2),
    contact_normal: wp.array(dtype=wp.float32, ndim=3),
    contact_point: wp.array(dtype=wp.float32, ndim=3),
    contact_bi: wp.array(dtype=wp.int32, ndim=2),
    contact_bj: wp.array(dtype=wp.int32, ndim=2),
    joint_type: wp.array(dtype=wp.int32),
    joint_axis: wp.array(dtype=wp.float32, ndim=2),
    parent_idx: wp.array(dtype=wp.int32),
    q_idx_start: wp.array(dtype=wp.int32),
    v_idx_start: wp.array(dtype=wp.int32),
    v_idx_len: wp.array(dtype=wp.int32),
    max_contacts: int,
    max_rows: int,
    nv: int,
    J_joint: wp.array(dtype=wp.float32, ndim=3),  # (N, max_rows, nv)
):
    """Joint-space contact Jacobian: walk chain from body to root."""
    env = wp.tid()

    for r in range(max_rows):
        for k in range(nv):
            J_joint[env, r, k] = 0.0

    for c in range(max_contacts):
        if contact_active[env, c] == 0:
            continue
        base = c * 3
        bi = contact_bi[env, c]
        bj = contact_bj[env, c]

        normal = wp.vec3(contact_normal[env, c, 0], contact_normal[env, c, 1], contact_normal[env, c, 2])
        cp = wp.vec3(contact_point[env, c, 0], contact_point[env, c, 1], contact_point[env, c, 2])

        # Tangent frame
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

        # Body i chain walk
        if bi >= 0:
            i = bi
            while i >= 0:
                vl = v_idx_len[i]
                if vl > 0:
                    vs = v_idx_start[i]
                    jt = joint_type[i]
                    ax = wp.vec3(joint_axis[i, 0], joint_axis[i, 1], joint_axis[i, 2])
                    R_w = wp.mat33(
                        X_world_R[env, i, 0, 0],
                        X_world_R[env, i, 0, 1],
                        X_world_R[env, i, 0, 2],
                        X_world_R[env, i, 1, 0],
                        X_world_R[env, i, 1, 1],
                        X_world_R[env, i, 1, 2],
                        X_world_R[env, i, 2, 0],
                        X_world_R[env, i, 2, 1],
                        X_world_R[env, i, 2, 2],
                    )
                    origin = wp.vec3(X_world_r[env, i, 0], X_world_r[env, i, 1], X_world_r[env, i, 2])
                    r_arm = cp - origin

                    # Load R_J for FreeJoint
                    RJ_ci = wp.identity(n=3, dtype=float)
                    if jt == JOINT_FREE:
                        qs_ci = q_idx_start[i]
                        RJ_ci = _load_RJ_free(q, env, qs_ci)

                    for k in range(vl):
                        if jt == JOINT_FREE:
                            S_col = _motion_subspace_col_free(k, RJ_ci)
                        else:
                            S_col = _motion_subspace_col(jt, ax, k)
                        s_lin_w = R_w * wp.vec3(S_col[0], S_col[1], S_col[2])
                        s_ang_w = R_w * wp.vec3(S_col[3], S_col[4], S_col[5])
                        v_point = s_lin_w + wp.cross(s_ang_w, r_arm)

                        J_joint[env, base + 0, vs + k] += wp.dot(normal, v_point)
                        J_joint[env, base + 1, vs + k] += wp.dot(t1, v_point)
                        J_joint[env, base + 2, vs + k] += wp.dot(t2, v_point)

                i = parent_idx[i]

        # Body j chain walk (subtract for relative velocity)
        if bj >= 0:
            i = bj
            while i >= 0:
                vl = v_idx_len[i]
                if vl > 0:
                    vs = v_idx_start[i]
                    jt = joint_type[i]
                    ax = wp.vec3(joint_axis[i, 0], joint_axis[i, 1], joint_axis[i, 2])
                    R_w = wp.mat33(
                        X_world_R[env, i, 0, 0],
                        X_world_R[env, i, 0, 1],
                        X_world_R[env, i, 0, 2],
                        X_world_R[env, i, 1, 0],
                        X_world_R[env, i, 1, 1],
                        X_world_R[env, i, 1, 2],
                        X_world_R[env, i, 2, 0],
                        X_world_R[env, i, 2, 1],
                        X_world_R[env, i, 2, 2],
                    )
                    origin = wp.vec3(X_world_r[env, i, 0], X_world_r[env, i, 1], X_world_r[env, i, 2])
                    r_arm = cp - origin

                    # Load R_J for FreeJoint
                    RJ_cj = wp.identity(n=3, dtype=float)
                    if jt == JOINT_FREE:
                        qs_cj = q_idx_start[i]
                        RJ_cj = _load_RJ_free(q, env, qs_cj)

                    for k in range(vl):
                        if jt == JOINT_FREE:
                            S_col = _motion_subspace_col_free(k, RJ_cj)
                        else:
                            S_col = _motion_subspace_col(jt, ax, k)
                        s_lin_w = R_w * wp.vec3(S_col[0], S_col[1], S_col[2])
                        s_ang_w = R_w * wp.vec3(S_col[3], S_col[4], S_col[5])
                        v_point = s_lin_w + wp.cross(s_ang_w, r_arm)

                        J_joint[env, base + 0, vs + k] -= wp.dot(normal, v_point)
                        J_joint[env, base + 1, vs + k] -= wp.dot(t1, v_point)
                        J_joint[env, base + 2, vs + k] -= wp.dot(t2, v_point)

                i = parent_idx[i]


# ---------------------------------------------------------------------------
# Kernel 3: batched_build_W_joint_space
# ---------------------------------------------------------------------------


@wp.kernel
def batched_build_W_joint_space(
    J_joint: wp.array(dtype=wp.float32, ndim=3),  # (N, max_rows, nv)
    L_H: wp.array(dtype=wp.float32, ndim=3),
    v_predicted: wp.array(dtype=wp.float32, ndim=2),
    qdot: wp.array(dtype=wp.float32, ndim=2),
    contact_active: wp.array(dtype=wp.int32, ndim=2),
    contact_depth: wp.array(dtype=wp.float32, ndim=2),
    cfm: float,
    solimp_d0: float,
    solimp_dw: float,
    solimp_width: float,
    solimp_mid: float,
    solimp_power: float,
    erp_pos: float,  # reinterpreted as 1/τ (inverse solref time constant)
    slop: float,
    max_depen_vel: float,  # upper bound on |v_ref| — prevents deep-penetration ejection
    dt: float,  # unused (kept for API compat), position correction is dt-independent
    max_contacts: int,
    max_rows: int,
    nv: int,
    # Scratch
    HinvJt: wp.array(dtype=wp.float32, ndim=3),  # (N, nv, max_rows)
    chol_tmp: wp.array(dtype=wp.float32, ndim=2),
    rhs_col: wp.array(dtype=wp.float32, ndim=2),  # (N, nv) temp
    sol_col: wp.array(dtype=wp.float32, ndim=2),  # (N, nv) temp
    # Outputs
    W: wp.array(dtype=wp.float32, ndim=3),
    W_diag: wp.array(dtype=wp.float32, ndim=2),
    v_free: wp.array(dtype=wp.float32, ndim=2),
    v_current: wp.array(dtype=wp.float32, ndim=2),
):
    """W = J H⁻¹ Jᵀ, v_free = J v_pred + Baumgarte bias, v_current = J qdot."""
    env = wp.tid()

    # Zero
    for r in range(max_rows):
        v_free[env, r] = 0.0
        v_current[env, r] = 0.0
        W_diag[env, r] = 0.0
        for r2 in range(max_rows):
            W[env, r, r2] = 0.0

    n_active_rows = int(0)
    highest = int(-1)
    for c in range(max_contacts):
        if contact_active[env, c] != 0:
            highest = c
    if highest >= 0:
        n_active_rows = (highest + 1) * 3

    if n_active_rows == 0:
        return

    # H⁻¹ Jᵀ: solve L Lᵀ x = J[r,:] for each row r
    for r in range(n_active_rows):
        for k in range(nv):
            rhs_col[env, k] = J_joint[env, r, k]
        _chol_solve(env, L_H, rhs_col, sol_col, chol_tmp, nv)
        for k in range(nv):
            HinvJt[env, k, r] = sol_col[env, k]

    # W = J × HinvJt
    for r1 in range(n_active_rows):
        for r2 in range(n_active_rows):
            val = float(0.0)
            for k in range(nv):
                val += J_joint[env, r1, k] * HinvJt[env, k, r2]
            W[env, r1, r2] = val

    # Per-row regularization (QP dual style).
    # Normal row: small cfm (hard contact — penetration correction via v_ref below).
    # Friction rows: per-row R = (1-d)/d × |W_ii| (Q25 fix, Todorov 2014).
    for ci in range(max_contacts):
        if contact_active[env, ci] == 0:
            continue
        base = ci * CONDIM
        depth_c = contact_depth[env, ci]
        d_imp = _impedance_wp(depth_c, solimp_d0, solimp_dw, solimp_width, solimp_mid, solimp_power)
        ratio = (1.0 - d_imp) / wp.max(d_imp, 1.0e-10)
        W_diag[env, base] = W[env, base, base] + cfm  # normal: hard
        for off in range(1, CONDIM):
            row = base + off
            W_diag[env, row] = W[env, row, row] + ratio * wp.abs(W[env, row, row])  # friction: soft

    # v_free, v_current from J × velocity
    for r in range(n_active_rows):
        vf = float(0.0)
        vc = float(0.0)
        for k in range(nv):
            vf += J_joint[env, r, k] * v_predicted[env, k]
            vc += J_joint[env, r, k] * qdot[env, k]
        v_free[env, r] = vf
        v_current[env, r] = vc

    # Position correction via solref time constant (MuJoCo QP dual).
    # Target velocity on normal row: v_ref = max(depth - slop, 0) / τ
    # where τ = solref_timeconst (default 0.02s = 20ms).
    # RHS becomes: v_free - v_ref (subtract, so PGS produces more force).
    # dt-independent: depth=1mm → 0.05 m/s, depth=20mm → 1.0 m/s.
    # v_ref is clamped to max_depen_vel (Bullet m_splitImpulsePenetrationThreshold
    # / PhysX maxDepenetrationVelocity) to prevent deep-penetration ejection.
    if erp_pos > 0.0:
        inv_tau = erp_pos  # reinterpret erp_pos as 1/τ (see static_data)
        for ci in range(max_contacts):
            if contact_active[env, ci] == 0:
                continue
            depth_c = contact_depth[env, ci]
            effective = depth_c - slop
            if effective > 0.0:
                base = ci * CONDIM
                v_ref = inv_tau * effective
                if v_ref > max_depen_vel:
                    v_ref = max_depen_vel
                v_free[env, base] -= v_ref


# ---------------------------------------------------------------------------
# Kernel 4: batched_apply_contact_impulse
# ---------------------------------------------------------------------------


@wp.kernel
def batched_apply_contact_impulse(
    lambdas: wp.array(dtype=wp.float32, ndim=2),
    J_joint: wp.array(dtype=wp.float32, ndim=3),
    L_H: wp.array(dtype=wp.float32, ndim=3),
    contact_active: wp.array(dtype=wp.int32, ndim=2),
    max_contacts: int,
    max_rows: int,
    nv: int,
    gen_impulse: wp.array(dtype=wp.float32, ndim=2),
    chol_tmp: wp.array(dtype=wp.float32, ndim=2),
    dqdot: wp.array(dtype=wp.float32, ndim=2),
):
    """dqdot = H⁻¹ Jᵀ λ."""
    env = wp.tid()

    n_active_rows = int(0)
    highest = int(-1)
    for c in range(max_contacts):
        if contact_active[env, c] != 0:
            highest = c
    if highest >= 0:
        n_active_rows = (highest + 1) * 3

    # Jᵀ λ
    for k in range(nv):
        val = float(0.0)
        for r in range(n_active_rows):
            val += J_joint[env, r, k] * lambdas[env, r]
        gen_impulse[env, k] = val

    # H⁻¹ gen_impulse
    _chol_solve(env, L_H, gen_impulse, dqdot, chol_tmp, nv)
