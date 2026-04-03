"""
TileLang GPU kernels for batched robot simulation.

Uses T.macro for composable spatial algebra operations.
Kernel factories use module-level globals to inject shape constants
(TileLang's eager builder resolves annotations in global scope).
"""

from __future__ import annotations

import sys

import tilelang
import tilelang.language as T
import torch

# Module globals used by TileLang annotations (set by factory before JIT)
_N: int = 1
_nb: int = 1
_nq: int = 1
_nv: int = 1

_this_module = sys.modules[__name__]


def _build_fk_kernel_impl():
    """Build the FK kernel using current module globals _N, _nb, _nq, _nv."""
    N = _this_module._N
    nb = _this_module._nb

    @tilelang.jit(out_idx=[2, 3, 4, 5, 6])
    def kernel():
        @T.macro
        def mat33_mul(A, B, C):
            for i, j in T.Parallel(3, 3):
                C[i, j] = 0.0
            for i, j in T.Parallel(3, 3):
                for k in T.Serial(3):
                    C[i, j] += A[i, k] * B[k, j]

        @T.macro
        def mat33_vec3_mul(A, v, out):
            for i in T.Parallel(3):
                out[i] = 0.0
            for i in T.Parallel(3):
                for k in T.Serial(3):
                    out[i] += A[i, k] * v[k]

        @T.macro
        def cross3(a, b, out):
            out[0] = a[1] * b[2] - a[2] * b[1]
            out[1] = a[2] * b[0] - a[0] * b[2]
            out[2] = a[0] * b[1] - a[1] * b[0]

        @T.macro
        def transform_vel(R, r, v_in, v_out):
            tmp = T.alloc_local([3], "float32")
            wxr = T.alloc_local([3], "float32")
            w_in = T.alloc_local([3], "float32")
            l_in = T.alloc_local([3], "float32")
            for d in T.Parallel(3):
                l_in[d] = v_in[d]
                w_in[d] = v_in[d + 3]
            cross3(w_in, r, wxr)
            for d in T.Parallel(3):
                tmp[d] = l_in[d] + wxr[d]
            # E = R^T, so result[i] = sum_k R[k,i] * src[k]
            for i in T.Parallel(3):
                v_out[i] = 0.0
                v_out[i + 3] = 0.0
            for i in T.Parallel(3):
                for k in T.Serial(3):
                    v_out[i] += R[k, i] * tmp[k]
                    v_out[i + 3] += R[k, i] * w_in[k]

        @T.prim_func
        def fk_main(
            q: T.Tensor([_N, _nq], "float32"),
            qdot: T.Tensor([_N, _nv], "float32"),
            X_world_R: T.Tensor([_N, _nb, 3, 3], "float32"),
            X_world_r: T.Tensor([_N, _nb, 3], "float32"),
            X_up_R: T.Tensor([_N, _nb, 3, 3], "float32"),
            X_up_r: T.Tensor([_N, _nb, 3], "float32"),
            v_bodies: T.Tensor([_N, _nb, 6], "float32"),
            jtype: T.Tensor([_nb], "int32"),
            jaxis: T.Tensor([_nb, 3], "float32"),
            pidx: T.Tensor([_nb], "int32"),
            qstart: T.Tensor([_nb], "int32"),
            qlen: T.Tensor([_nb], "int32"),
            vstart: T.Tensor([_nb], "int32"),
            vlen: T.Tensor([_nb], "int32"),
            tree_R: T.Tensor([_nb, 3, 3], "float32"),
            tree_r: T.Tensor([_nb, 3], "float32"),
        ):
            with T.Kernel(N, threads=1) as env_id:
                R_J = T.alloc_local([3, 3], "float32")
                r_J = T.alloc_local([3], "float32")
                R_tr = T.alloc_local([3, 3], "float32")
                r_tr = T.alloc_local([3], "float32")
                R_up = T.alloc_local([3, 3], "float32")
                r_up = T.alloc_local([3], "float32")
                R_p = T.alloc_local([3, 3], "float32")
                r_p = T.alloc_local([3], "float32")
                R_w = T.alloc_local([3, 3], "float32")
                r_w = T.alloc_local([3], "float32")
                tmp3 = T.alloc_local([3], "float32")
                vJ = T.alloc_local([6], "float32")
                vpar = T.alloc_local([6], "float32")
                vxf = T.alloc_local([6], "float32")
                ax = T.alloc_local([3], "float32")

                for body in T.Serial(nb):
                    jt = jtype[body]
                    qs = qstart[body]
                    vs = vstart[body]
                    _vl = vlen[body]  # noqa: F841 — read for alignment
                    pid = pidx[body]

                    for d in T.Serial(3):
                        ax[d] = jaxis[body, d]

                    # Identity default
                    for a, b in T.Parallel(3, 3):
                        R_J[a, b] = T.if_then_else(a == b, 1.0, 0.0)
                    for d in T.Parallel(3):
                        r_J[d] = 0.0

                    if jt == 1:  # REVOLUTE
                        angle = q[env_id, qs]
                        c = T.cos(angle)
                        s = T.sin(angle)
                        omc = 1.0 - c
                        # R = c*I + s*K + (1-c)*k*kT
                        # Compute K (skew) into a fragment first
                        K_buf = T.alloc_local([3, 3], "float32")
                        for a, b in T.Parallel(3, 3):
                            K_buf[a, b] = 0.0
                        K_buf[0, 1] = -ax[2]
                        K_buf[0, 2] = ax[1]
                        K_buf[1, 0] = ax[2]
                        K_buf[1, 2] = -ax[0]
                        K_buf[2, 0] = -ax[1]
                        K_buf[2, 1] = ax[0]
                        for a, b in T.Parallel(3, 3):
                            R_J[a, b] = (
                                c * T.if_then_else(a == b, 1.0, 0.0) + s * K_buf[a, b] + omc * ax[a] * ax[b]
                            )

                    if jt == 2:  # PRISMATIC
                        d_val = q[env_id, qs]
                        for d in T.Serial(3):
                            r_J[d] = ax[d] * d_val

                    if jt == 0:  # FREE
                        qw = q[env_id, qs]
                        qx = q[env_id, qs + 1]
                        qy = q[env_id, qs + 2]
                        qz = q[env_id, qs + 3]
                        n = T.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
                        w = qw / n
                        x = qx / n
                        y = qy / n
                        z = qz / n
                        R_J[0, 0] = 1.0 - 2.0 * (y * y + z * z)
                        R_J[0, 1] = 2.0 * (x * y - w * z)
                        R_J[0, 2] = 2.0 * (x * z + w * y)
                        R_J[1, 0] = 2.0 * (x * y + w * z)
                        R_J[1, 1] = 1.0 - 2.0 * (x * x + z * z)
                        R_J[1, 2] = 2.0 * (y * z - w * x)
                        R_J[2, 0] = 2.0 * (x * z - w * y)
                        R_J[2, 1] = 2.0 * (y * z + w * x)
                        R_J[2, 2] = 1.0 - 2.0 * (x * x + y * y)
                        for d in T.Serial(3):
                            r_J[d] = q[env_id, qs + 4 + d]

                    # X_tree
                    for a, b in T.Parallel(3, 3):
                        R_tr[a, b] = tree_R[body, a, b]
                    for d in T.Parallel(3):
                        r_tr[d] = tree_r[body, d]

                    # X_up = X_tree @ X_J
                    mat33_mul(R_tr, R_J, R_up)
                    mat33_vec3_mul(R_tr, r_J, tmp3)
                    for d in T.Parallel(3):
                        r_up[d] = r_tr[d] + tmp3[d]

                    for a, b in T.Parallel(3, 3):
                        X_up_R[env_id, body, a, b] = R_up[a, b]
                    for d in T.Parallel(3):
                        X_up_r[env_id, body, d] = r_up[d]

                    if pid < 0:
                        for a, b in T.Parallel(3, 3):
                            X_world_R[env_id, body, a, b] = R_up[a, b]
                        for d in T.Parallel(3):
                            X_world_r[env_id, body, d] = r_up[d]
                    else:
                        for a, b in T.Parallel(3, 3):
                            R_p[a, b] = X_world_R[env_id, pid, a, b]
                        for d in T.Parallel(3):
                            r_p[d] = X_world_r[env_id, pid, d]
                        mat33_mul(R_p, R_up, R_w)
                        mat33_vec3_mul(R_p, r_up, tmp3)
                        for d in T.Parallel(3):
                            r_w[d] = r_p[d] + tmp3[d]
                        for a, b in T.Parallel(3, 3):
                            X_world_R[env_id, body, a, b] = R_w[a, b]
                        for d in T.Parallel(3):
                            X_world_r[env_id, body, d] = r_w[d]

                    # vJ
                    for d in T.Parallel(6):
                        vJ[d] = 0.0
                    if jt == 1:
                        for d in T.Serial(3):
                            vJ[d + 3] = ax[d] * qdot[env_id, vs]
                    if jt == 2:
                        for d in T.Serial(3):
                            vJ[d] = ax[d] * qdot[env_id, vs]
                    if jt == 0:
                        # MuJoCo convention: qdot[:3] world linear, qdot[3:] body angular
                        # vJ = S @ qdot = [R_J^T @ v_world; omega_body]
                        v_world = T.alloc_local([3], "float32")
                        v_body_lin = T.alloc_local([3], "float32")
                        for d in T.Serial(3):
                            v_world[d] = qdot[env_id, vs + d]
                        # v_body_lin = R_J^T @ v_world
                        for i_r in T.Serial(3):
                            v_body_lin[i_r] = 0.0
                            for k_r in T.Serial(3):
                                v_body_lin[i_r] += R_J[k_r, i_r] * v_world[k_r]
                        for d in T.Serial(3):
                            vJ[d] = v_body_lin[d]
                            vJ[d + 3] = qdot[env_id, vs + 3 + d]

                    if pid < 0:
                        for d in T.Parallel(6):
                            v_bodies[env_id, body, d] = vJ[d]
                    else:
                        for d in T.Serial(6):
                            vpar[d] = v_bodies[env_id, pid, d]
                        transform_vel(R_up, r_up, vpar, vxf)
                        for d in T.Parallel(6):
                            v_bodies[env_id, body, d] = vxf[d] + vJ[d]

        return fk_main

    return kernel


# Cache compiled kernels
_fk_cache: dict[tuple, object] = {}


def get_fk_kernel(N, nb, nq, nv):
    """Get or compile a FK kernel for the given dimensions."""
    key = (N, nb, nq, nv)
    if key not in _fk_cache:
        # Set module globals before building
        _this_module._N = N
        _this_module._nb = nb
        _this_module._nq = nq
        _this_module._nv = nv
        jit_impl = _build_fk_kernel_impl()
        _fk_cache[key] = jit_impl()  # compile
    return _fk_cache[key]


# ---------------------------------------------------------------------------
# ABA kernel
# ---------------------------------------------------------------------------


def _build_aba_kernel_impl():
    """Build the ABA kernel using current module globals."""
    N = _this_module._N
    nb = _this_module._nb
    nv = _this_module._nv

    @tilelang.jit(out_idx=[4])
    def kernel():
        @T.macro
        def mat66_mul_vec6(M, v, out):
            for i in T.Parallel(6):
                out[i] = 0.0
            for i in T.Parallel(6):
                for k in T.Serial(6):
                    out[i] += M[i, k] * v[k]

        @T.macro
        def cross3(a, b, out):
            out[0] = a[1] * b[2] - a[2] * b[1]
            out[1] = a[2] * b[0] - a[0] * b[2]
            out[2] = a[0] * b[1] - a[1] * b[0]

        @T.macro
        def transform_vel(R, r, v_in, v_out):
            _tmp = T.alloc_local([3], "float32")
            _wxr = T.alloc_local([3], "float32")
            _w = T.alloc_local([3], "float32")
            _l = T.alloc_local([3], "float32")
            for d in T.Parallel(3):
                _l[d] = v_in[d]
                _w[d] = v_in[d + 3]
            cross3(_w, r, _wxr)
            for d in T.Parallel(3):
                _tmp[d] = _l[d] + _wxr[d]
            for i in T.Parallel(3):
                v_out[i] = 0.0
                v_out[i + 3] = 0.0
            for i in T.Parallel(3):
                for k in T.Serial(3):
                    v_out[i] += R[k, i] * _tmp[k]
                    v_out[i + 3] += R[k, i] * _w[k]

        @T.macro
        def transform_force(R, r, f_in, f_out):
            _Rf = T.alloc_local([3], "float32")
            _Rt = T.alloc_local([3], "float32")
            _rxRf = T.alloc_local([3], "float32")
            for i in T.Parallel(3):
                _Rf[i] = 0.0
                _Rt[i] = 0.0
            for i in T.Parallel(3):
                for k in T.Serial(3):
                    _Rf[i] += R[i, k] * f_in[k]
                    _Rt[i] += R[i, k] * f_in[k + 3]
            cross3(r, _Rf, _rxRf)
            for d in T.Parallel(3):
                f_out[d] = _Rf[d]
                f_out[d + 3] = _Rt[d] + _rxRf[d]

        @T.macro
        def spatial_cross_force_mv(v, Iv, out):
            """v ×* Iv: [[skew(w),0],[skew(vl),skew(w)]] @ Iv"""
            _vl = T.alloc_local([3], "float32")
            _va = T.alloc_local([3], "float32")
            _fl = T.alloc_local([3], "float32")
            _fa = T.alloc_local([3], "float32")
            _t1 = T.alloc_local([3], "float32")
            _t2 = T.alloc_local([3], "float32")
            _t3 = T.alloc_local([3], "float32")
            for d in T.Parallel(3):
                _vl[d] = v[d]
                _va[d] = v[d + 3]
                _fl[d] = Iv[d]
                _fa[d] = Iv[d + 3]
            cross3(_va, _fl, _t1)
            cross3(_vl, _fl, _t2)
            cross3(_va, _fa, _t3)
            for d in T.Parallel(3):
                out[d] = _t1[d]
                out[d + 3] = _t2[d] + _t3[d]

        @T.macro
        def spatial_cross_vel_mv(v, u, out):
            """v ×_vel u: [[skew(w),skew(vl)],[0,skew(w)]] @ u"""
            _vl = T.alloc_local([3], "float32")
            _va = T.alloc_local([3], "float32")
            _ul = T.alloc_local([3], "float32")
            _ua = T.alloc_local([3], "float32")
            _t1 = T.alloc_local([3], "float32")
            _t2 = T.alloc_local([3], "float32")
            _t3 = T.alloc_local([3], "float32")
            for d in T.Parallel(3):
                _vl[d] = v[d]
                _va[d] = v[d + 3]
                _ul[d] = u[d]
                _ua[d] = u[d + 3]
            cross3(_va, _ul, _t1)
            cross3(_vl, _ua, _t2)
            cross3(_va, _ua, _t3)
            for d in T.Parallel(3):
                out[d] = _t1[d] + _t2[d]
                out[d + 3] = _t3[d]

        @T.macro
        def build_X6(R, r, X6):
            """Build 6x6 Plücker velocity transform matrix. E=R^T."""
            for i, j in T.Parallel(6, 6):
                X6[i, j] = 0.0
            for i, j in T.Parallel(3, 3):
                X6[i, j] = R[j, i]  # top-left: E
                X6[i + 3, j + 3] = R[j, i]  # bottom-right: E
            # top-right: -E @ skew(r)
            _K = T.alloc_local([3, 3], "float32")
            for a, b in T.Parallel(3, 3):
                _K[a, b] = 0.0
            _K[0, 1] = -r[2]
            _K[0, 2] = r[1]
            _K[1, 0] = r[2]
            _K[1, 2] = -r[0]
            _K[2, 0] = -r[1]
            _K[2, 1] = r[0]
            for i in T.Parallel(3):
                for j in T.Serial(3):
                    X6[i, j + 3] = 0.0
                    for k in T.Serial(3):
                        X6[i, j + 3] += -R[k, i] * _K[k, j]

        @T.prim_func
        def aba_main(
            q: T.Tensor([_N, _nq], "float32"),
            qdot: T.Tensor([_N, _nv], "float32"),
            tau_total: T.Tensor([_N, _nv], "float32"),
            ext_forces: T.Tensor([_N, _nb, 6], "float32"),
            # output
            qddot: T.Tensor([_N, _nv], "float32"),
            # static
            jtype: T.Tensor([_nb], "int32"),
            jaxis: T.Tensor([_nb, 3], "float32"),
            pidx: T.Tensor([_nb], "int32"),
            qstart: T.Tensor([_nb], "int32"),
            vstart: T.Tensor([_nb], "int32"),
            vlen: T.Tensor([_nb], "int32"),
            I_mat: T.Tensor([_nb, 6, 6], "float32"),
            X_up_R: T.Tensor([_N, _nb, 3, 3], "float32"),
            X_up_r: T.Tensor([_N, _nb, 3], "float32"),
            gravity_val: T.float32,
        ):
            with T.Kernel(N, threads=1) as env_id:
                # Per-body scratch
                v_arr = T.alloc_local([nb, 6], "float32")
                c_arr = T.alloc_local([nb, 6], "float32")
                IA = T.alloc_local([nb, 6, 6], "float32")
                pA = T.alloc_local([nb, 6], "float32")
                a_arr = T.alloc_local([nb, 6], "float32")
                # Per-body ABA pass 2 storage
                U_arr = T.alloc_local([nb, 6], "float32")  # for 1-DOF
                Dinv_arr = T.alloc_local([nb], "float32")  # scalar for 1-DOF
                u_arr = T.alloc_local([nb], "float32")  # scalar for 1-DOF
                # FreeJoint storage (body 0 only typically)
                u6_arr = T.alloc_local([6], "float32")
                IA6_for_solve = T.alloc_local([6, 6], "float32")

                # Temporaries
                vi = T.alloc_local([6], "float32")
                ci = T.alloc_local([6], "float32")
                vJ = T.alloc_local([6], "float32")
                Iv = T.alloc_local([6], "float32")
                pA_i = T.alloc_local([6], "float32")
                ax = T.alloc_local([3], "float32")
                S_col = T.alloc_local([6], "float32")
                U_i = T.alloc_local([6], "float32")
                tmp6 = T.alloc_local([6], "float32")
                tmp6b = T.alloc_local([6], "float32")
                R_up = T.alloc_local([3, 3], "float32")
                r_up = T.alloc_local([3], "float32")
                X6 = T.alloc_local([6, 6], "float32")
                X6T = T.alloc_local([6, 6], "float32")
                IA_A = T.alloc_local([6, 6], "float32")
                pA_A = T.alloc_local([6], "float32")
                IA_tmp = T.alloc_local([6, 6], "float32")
                IA_parent = T.alloc_local([6, 6], "float32")
                # FreeJoint R_J (computed from quaternion)
                R_J_free = T.alloc_local([3, 3], "float32")
                v_world_aba = T.alloc_local([3], "float32")
                v_body_lin_aba = T.alloc_local([3], "float32")

                # Zero init
                for i in T.Serial(nb):
                    for d in T.Serial(6):
                        v_arr[i, d] = 0.0
                        c_arr[i, d] = 0.0
                        pA[i, d] = 0.0
                        a_arr[i, d] = 0.0
                    for a in T.Serial(6):
                        for b in T.Serial(6):
                            IA[i, a, b] = 0.0
                    U_arr[i, 0] = 0.0  # only first element used for 1-DOF
                    Dinv_arr[i] = 0.0
                    u_arr[i] = 0.0
                for d in T.Serial(nv):
                    qddot[env_id, d] = 0.0

                # =============== Pass 1: forward — velocities & bias forces ===============
                for i in T.Serial(nb):
                    jt = jtype[i]
                    vs = vstart[i]
                    vl = vlen[i]
                    pid = pidx[i]

                    for d in T.Serial(3):
                        ax[d] = jaxis[i, d]
                    for a, b in T.Parallel(3, 3):
                        R_up[a, b] = X_up_R[env_id, i, a, b]
                    for d in T.Parallel(3):
                        r_up[d] = X_up_r[env_id, i, d]

                    # vJ = S @ qdot
                    for d in T.Parallel(6):
                        vJ[d] = 0.0
                    if jt == 1:  # REVOLUTE
                        for d in T.Serial(3):
                            vJ[d + 3] = ax[d] * qdot[env_id, vs]
                    if jt == 2:  # PRISMATIC
                        for d in T.Serial(3):
                            vJ[d] = ax[d] * qdot[env_id, vs]
                    if jt == 0:  # FREE
                        # MuJoCo convention: qdot[:3] world linear, qdot[3:] body angular
                        # Compute R_J from quaternion
                        qs_aba = qstart[i]
                        qw_f = q[env_id, qs_aba]
                        qx_f = q[env_id, qs_aba + 1]
                        qy_f = q[env_id, qs_aba + 2]
                        qz_f = q[env_id, qs_aba + 3]
                        n_f = T.sqrt(qw_f * qw_f + qx_f * qx_f + qy_f * qy_f + qz_f * qz_f)
                        w_f = qw_f / n_f
                        x_f = qx_f / n_f
                        y_f = qy_f / n_f
                        z_f = qz_f / n_f
                        R_J_free[0, 0] = 1.0 - 2.0 * (y_f * y_f + z_f * z_f)
                        R_J_free[0, 1] = 2.0 * (x_f * y_f - w_f * z_f)
                        R_J_free[0, 2] = 2.0 * (x_f * z_f + w_f * y_f)
                        R_J_free[1, 0] = 2.0 * (x_f * y_f + w_f * z_f)
                        R_J_free[1, 1] = 1.0 - 2.0 * (x_f * x_f + z_f * z_f)
                        R_J_free[1, 2] = 2.0 * (y_f * z_f - w_f * x_f)
                        R_J_free[2, 0] = 2.0 * (x_f * z_f - w_f * y_f)
                        R_J_free[2, 1] = 2.0 * (y_f * z_f + w_f * x_f)
                        R_J_free[2, 2] = 1.0 - 2.0 * (x_f * x_f + y_f * y_f)
                        # vJ[:3] = R_J^T @ v_world
                        for d_r in T.Serial(3):
                            v_world_aba[d_r] = qdot[env_id, vs + d_r]
                        for i_r in T.Serial(3):
                            v_body_lin_aba[i_r] = 0.0
                            for k_r in T.Serial(3):
                                v_body_lin_aba[i_r] += R_J_free[k_r, i_r] * v_world_aba[k_r]
                        for d in T.Serial(3):
                            vJ[d] = v_body_lin_aba[d]
                            vJ[d + 3] = qdot[env_id, vs + 3 + d]

                    # Coriolis bias for FreeJoint: c_J_free = [-omega x v_body_lin, 0]
                    cJ_free = T.alloc_local([6], "float32")
                    for d in T.Parallel(6):
                        cJ_free[d] = 0.0
                    if jt == 0:  # FREE
                        # omega x v_body_lin
                        _w_cJ = T.alloc_local([3], "float32")
                        _v_cJ = T.alloc_local([3], "float32")
                        _wxv_cJ = T.alloc_local([3], "float32")
                        for d in T.Serial(3):
                            _w_cJ[d] = vJ[d + 3]
                            _v_cJ[d] = vJ[d]
                        cross3(_w_cJ, _v_cJ, _wxv_cJ)
                        for d in T.Serial(3):
                            cJ_free[d] = -_wxv_cJ[d]

                    if pid < 0:
                        for d in T.Parallel(6):
                            v_arr[i, d] = vJ[d]
                            c_arr[i, d] = cJ_free[d]
                    else:
                        # Load parent velocity
                        for d in T.Serial(6):
                            tmp6[d] = v_arr[pid, d]
                        transform_vel(R_up, r_up, tmp6, vi)
                        for d in T.Parallel(6):
                            v_arr[i, d] = vi[d] + vJ[d]
                        # c = v ×_vel vJ + cJ_free
                        for d in T.Serial(6):
                            vi[d] = v_arr[i, d]
                        spatial_cross_vel_mv(vi, vJ, ci)
                        for d in T.Parallel(6):
                            c_arr[i, d] = ci[d] + cJ_free[d]

                    # IA = I_body
                    for a, b in T.Parallel(6, 6):
                        IA[i, a, b] = I_mat[i, a, b]

                    # pA = v ×* (I @ v) - ext_forces
                    for d in T.Serial(6):
                        vi[d] = v_arr[i, d]
                    for a, b in T.Parallel(6, 6):
                        IA_tmp[a, b] = I_mat[i, a, b]
                    mat66_mul_vec6(IA_tmp, vi, Iv)
                    spatial_cross_force_mv(vi, Iv, pA_i)
                    for d in T.Parallel(6):
                        pA[i, d] = pA_i[d] - ext_forces[env_id, i, d]

                # =============== Pass 2: backward — articulated inertias ===============
                for idx in T.Serial(nb):
                    i = nb - 1 - idx  # reverse
                    jt = jtype[i]
                    vs = vstart[i]
                    vl = vlen[i]
                    pid = pidx[i]

                    for d in T.Serial(3):
                        ax[d] = jaxis[i, d]

                    # Load IA[i], pA[i], c[i]
                    for a, b in T.Parallel(6, 6):
                        IA_A[a, b] = IA[i, a, b]
                    for d in T.Parallel(6):
                        pA_i[d] = pA[i, d]
                        ci[d] = c_arr[i, d]

                    if vl > 0:
                        if jt == 1 or jt == 2:  # 1-DOF joints
                            # Build S column
                            for d in T.Parallel(6):
                                S_col[d] = 0.0
                            if jt == 1:
                                for d in T.Serial(3):
                                    S_col[d + 3] = ax[d]
                            if jt == 2:
                                for d in T.Serial(3):
                                    S_col[d] = ax[d]

                            # U = IA @ S
                            mat66_mul_vec6(IA_A, S_col, U_i)
                            # D = S^T @ U (scalar dot)
                            D_buf = T.alloc_local([1], "float32")
                            D_buf[0] = 0.0
                            for d in T.Serial(6):
                                D_buf[0] += S_col[d] * U_i[d]
                            D_inv_buf = T.alloc_local([1], "float32")
                            D_inv_buf[0] = 1.0 / D_buf[0]
                            # u = tau - S^T @ pA
                            u_buf = T.alloc_local([1], "float32")
                            u_buf[0] = tau_total[env_id, vs]
                            for d in T.Serial(6):
                                u_buf[0] -= S_col[d] * pA_i[d]

                            # Store for pass 3
                            for d in T.Parallel(6):
                                U_arr[i, d] = U_i[d]
                            Dinv_arr[i] = D_inv_buf[0]
                            u_arr[i] = u_buf[0]

                            # IA_A = IA - U @ U^T * D_inv
                            for a, b in T.Parallel(6, 6):
                                IA_A[a, b] = IA[i, a, b] - U_i[a] * U_i[b] * D_inv_buf[0]

                            # pA_A = pA + IA_A @ c + U * D_inv * u
                            mat66_mul_vec6(IA_A, ci, tmp6)
                            for d in T.Parallel(6):
                                pA_A[d] = pA_i[d] + tmp6[d] + U_i[d] * D_inv_buf[0] * u_buf[0]

                        if jt == 0:  # FREE (6-DOF)
                            # With S = [[R_J^T,0],[0,I]], work in body frame:
                            # tau_body[:3] = R_J^T @ tau_gen[:3], tau_body[3:] = tau_gen[3:]
                            # Then standard ABA with S=I on tau_body, and convert qddot back.
                            # u = tau_body - pA
                            tau_body = T.alloc_local([6], "float32")
                            tau_gen_lin = T.alloc_local([3], "float32")
                            for d in T.Serial(3):
                                tau_gen_lin[d] = tau_total[env_id, vs + d]
                            for i_r in T.Serial(3):
                                tau_body[i_r] = 0.0
                                for k_r in T.Serial(3):
                                    tau_body[i_r] += R_J_free[k_r, i_r] * tau_gen_lin[k_r]
                            for d in T.Serial(3):
                                tau_body[d + 3] = tau_total[env_id, vs + 3 + d]
                            for d in T.Serial(6):
                                u6_arr[d] = tau_body[d] - pA_i[d]
                            # Store IA for pass 3
                            for a, b in T.Parallel(6, 6):
                                IA6_for_solve[a, b] = IA[i, a, b]
                            # IA_A = 0
                            for a, b in T.Parallel(6, 6):
                                IA_A[a, b] = 0.0
                            # pA_A = pA + u
                            for d in T.Parallel(6):
                                pA_A[d] = pA_i[d] + u6_arr[d]
                    else:
                        # Fixed joint: IA_A = IA, pA_A = pA + IA @ c
                        mat66_mul_vec6(IA_A, ci, tmp6)
                        for d in T.Parallel(6):
                            pA_A[d] = pA_i[d] + tmp6[d]

                    # Propagate to parent
                    if pid >= 0:
                        for a, b in T.Parallel(3, 3):
                            R_up[a, b] = X_up_R[env_id, i, a, b]
                        for d in T.Parallel(3):
                            r_up[d] = X_up_r[env_id, i, d]

                        # IA[parent] += X^T @ IA_A @ X
                        build_X6(R_up, r_up, X6)
                        for a, b in T.Parallel(6, 6):
                            X6T[a, b] = X6[b, a]
                        # tmp = IA_A @ X6
                        for a, b in T.Parallel(6, 6):
                            IA_tmp[a, b] = 0.0
                        for a, b in T.Parallel(6, 6):
                            for k in T.Serial(6):
                                IA_tmp[a, b] += IA_A[a, k] * X6[k, b]
                        # contrib = X6T @ tmp
                        for a, b in T.Parallel(6, 6):
                            IA_parent[a, b] = 0.0
                        for a, b in T.Parallel(6, 6):
                            for k in T.Serial(6):
                                IA_parent[a, b] += X6T[a, k] * IA_tmp[k, b]
                        for a, b in T.Parallel(6, 6):
                            IA[pid, a, b] += IA_parent[a, b]

                        # pA[parent] += X_up.apply_force(pA_A)
                        transform_force(R_up, r_up, pA_A, tmp6)
                        for d in T.Parallel(6):
                            pA[pid, d] += tmp6[d]

                # =============== Pass 3: forward — accelerations ===============
                for i in T.Serial(nb):
                    jt = jtype[i]
                    vs = vstart[i]
                    vl = vlen[i]
                    pid = pidx[i]

                    for d in T.Serial(3):
                        ax[d] = jaxis[i, d]
                    for a, b in T.Parallel(3, 3):
                        R_up[a, b] = X_up_R[env_id, i, a, b]
                    for d in T.Parallel(3):
                        r_up[d] = X_up_r[env_id, i, d]

                    # a_p
                    if pid < 0:
                        # -a_gravity in body frame
                        neg_grav = T.alloc_local([6], "float32")
                        neg_grav[0] = 0.0
                        neg_grav[1] = 0.0
                        neg_grav[2] = gravity_val  # -(-g) = g
                        neg_grav[3] = 0.0
                        neg_grav[4] = 0.0
                        neg_grav[5] = 0.0
                        transform_vel(R_up, r_up, neg_grav, tmp6)
                    else:
                        for d in T.Serial(6):
                            vi[d] = a_arr[pid, d]
                        transform_vel(R_up, r_up, vi, tmp6)

                    # apc = a_p + c
                    for d in T.Parallel(6):
                        ci[d] = c_arr[i, d]
                        tmp6b[d] = tmp6[d] + ci[d]

                    if vl > 0:
                        if jt == 1 or jt == 2:
                            for d in T.Parallel(6):
                                S_col[d] = 0.0
                            if jt == 1:
                                for d in T.Serial(3):
                                    S_col[d + 3] = ax[d]
                            if jt == 2:
                                for d in T.Serial(3):
                                    S_col[d] = ax[d]

                            for d in T.Serial(6):
                                U_i[d] = U_arr[i, d]
                            D_inv_val = Dinv_arr[i]
                            u_val = u_arr[i]

                            # qddot = Dinv * (u - U^T @ apc)
                            UT_apc_buf = T.alloc_local([1], "float32")
                            UT_apc_buf[0] = 0.0
                            for d in T.Serial(6):
                                UT_apc_buf[0] += U_i[d] * tmp6b[d]
                            D_inv_val = Dinv_arr[i]
                            u_val = u_arr[i]
                            qddot_buf = T.alloc_local([1], "float32")
                            qddot_buf[0] = D_inv_val * (u_val - UT_apc_buf[0])
                            qddot[env_id, vs] = qddot_buf[0]

                            # a = apc + S * qddot
                            for d in T.Parallel(6):
                                a_arr[i, d] = tmp6b[d] + S_col[d] * qddot_buf[0]

                        if jt == 0:  # FREE
                            # Solve IA @ qdd = u - IA @ apc
                            mat66_mul_vec6(IA6_for_solve, tmp6b, tmp6)
                            rhs = T.alloc_local([6], "float32")
                            for d in T.Parallel(6):
                                rhs[d] = u6_arr[d] - tmp6[d]

                            # 6x6 Gaussian elimination (no pivoting, IA is SPD)
                            aug = T.alloc_local([6, 7], "float32")
                            for a in T.Serial(6):
                                for b in T.Serial(6):
                                    aug[a, b] = IA6_for_solve[a, b]
                                aug[a, 6] = rhs[a]

                            factor_buf = T.alloc_local([1], "float32")
                            for col in T.Serial(6):
                                for row in T.Serial(6):
                                    if row > col:
                                        factor_buf[0] = aug[row, col] / aug[col, col]
                                        for k in T.Serial(7):
                                            aug[row, k] -= factor_buf[0] * aug[col, k]

                            # Back substitution
                            qdd = T.alloc_local([6], "float32")
                            for d in T.Parallel(6):
                                qdd[d] = 0.0
                            s_buf = T.alloc_local([1], "float32")
                            for ri in T.Serial(6):
                                row = 5 - ri
                                s_buf[0] = aug[row, 6]
                                for k in T.Serial(6):
                                    if k > row:
                                        s_buf[0] -= aug[row, k] * qdd[k]
                                qdd[row] = s_buf[0] / aug[row, row]

                            # qdd is body-frame. Convert to generalized coords:
                            # qddot_gen[:3] = R_J @ qdd[:3], qddot_gen[3:6] = qdd[3:6]
                            # Recompute R_J from quaternion
                            qs_p3 = qstart[i]
                            qw_p3 = q[env_id, qs_p3]
                            qx_p3 = q[env_id, qs_p3 + 1]
                            qy_p3 = q[env_id, qs_p3 + 2]
                            qz_p3 = q[env_id, qs_p3 + 3]
                            n_p3 = T.sqrt(qw_p3 * qw_p3 + qx_p3 * qx_p3 + qy_p3 * qy_p3 + qz_p3 * qz_p3)
                            w_p3 = qw_p3 / n_p3
                            x_p3 = qx_p3 / n_p3
                            y_p3 = qy_p3 / n_p3
                            z_p3 = qz_p3 / n_p3
                            RJ_p3 = T.alloc_local([3, 3], "float32")
                            RJ_p3[0, 0] = 1.0 - 2.0 * (y_p3 * y_p3 + z_p3 * z_p3)
                            RJ_p3[0, 1] = 2.0 * (x_p3 * y_p3 - w_p3 * z_p3)
                            RJ_p3[0, 2] = 2.0 * (x_p3 * z_p3 + w_p3 * y_p3)
                            RJ_p3[1, 0] = 2.0 * (x_p3 * y_p3 + w_p3 * z_p3)
                            RJ_p3[1, 1] = 1.0 - 2.0 * (x_p3 * x_p3 + z_p3 * z_p3)
                            RJ_p3[1, 2] = 2.0 * (y_p3 * z_p3 - w_p3 * x_p3)
                            RJ_p3[2, 0] = 2.0 * (x_p3 * z_p3 - w_p3 * y_p3)
                            RJ_p3[2, 1] = 2.0 * (y_p3 * z_p3 + w_p3 * x_p3)
                            RJ_p3[2, 2] = 1.0 - 2.0 * (x_p3 * x_p3 + y_p3 * y_p3)
                            qdd_gen_lin = T.alloc_local([3], "float32")
                            for i_r in T.Serial(3):
                                qdd_gen_lin[i_r] = 0.0
                                for k_r in T.Serial(3):
                                    qdd_gen_lin[i_r] += RJ_p3[i_r, k_r] * qdd[k_r]
                            for d in T.Serial(3):
                                qddot[env_id, vs + d] = qdd_gen_lin[d]
                            for d in T.Serial(3):
                                qddot[env_id, vs + 3 + d] = qdd[3 + d]
                            for d in T.Parallel(6):
                                a_arr[i, d] = tmp6b[d] + qdd[d]
                    else:
                        for d in T.Parallel(6):
                            a_arr[i, d] = tmp6b[d]

        return aba_main

    return kernel


_aba_cache: dict[tuple, object] = {}


def get_aba_kernel(N, nb, nq, nv):
    """Get or compile an ABA kernel for the given dimensions."""
    key = ("aba", N, nb, nq, nv)
    if key not in _aba_cache:
        _this_module._N = N
        _this_module._nb = nb
        _this_module._nq = nq
        _this_module._nv = nv
        jit_impl = _build_aba_kernel_impl()
        _aba_cache[key] = jit_impl()
    return _aba_cache[key]


# ---------------------------------------------------------------------------
# PyTorch spatial math helpers (used by contact/collision in tilelang_backend.py)
# ---------------------------------------------------------------------------


def rodrigues_torch(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    N = angle.shape[0]
    k = axis.unsqueeze(0).expand(N, 3)
    c = torch.cos(angle).unsqueeze(-1).unsqueeze(-1)
    s = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)
    K = torch.zeros(N, 3, 3, device=angle.device, dtype=angle.dtype)
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]
    kkT = torch.einsum("ni,nj->nij", k, k)
    I = torch.eye(3, device=angle.device, dtype=angle.dtype).unsqueeze(0)
    return c * I + s * K + (1.0 - c) * kkT


def quat_to_rot_torch(quat: torch.Tensor) -> torch.Tensor:
    q = quat / quat.norm(dim=-1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


def transform_velocity_torch(R, r, v):
    E = R.transpose(-1, -2)
    v_lin, v_ang = v[:, :3], v[:, 3:]
    lin_new = torch.bmm(E, (v_lin + torch.cross(v_ang, r, dim=-1)).unsqueeze(-1)).squeeze(-1)
    ang_new = torch.bmm(E, v_ang.unsqueeze(-1)).squeeze(-1)
    return torch.cat([lin_new, ang_new], dim=-1)


def transform_force_torch(R, r, f):
    f_lin, f_ang = f[:, :3], f[:, 3:]
    Rf = torch.bmm(R, f_lin.unsqueeze(-1)).squeeze(-1)
    ang_new = torch.bmm(R, f_ang.unsqueeze(-1)).squeeze(-1) + torch.cross(r, Rf, dim=-1)
    return torch.cat([Rf, ang_new], dim=-1)


def spatial_cross_force_torch(v, f):
    v_lin, v_ang = v[:, :3], v[:, 3:]
    f_lin, f_ang = f[:, :3], f[:, 3:]
    res_lin = torch.cross(v_ang, f_lin, dim=-1)
    res_ang = torch.cross(v_lin, f_lin, dim=-1) + torch.cross(v_ang, f_ang, dim=-1)
    return torch.cat([res_lin, res_ang], dim=-1)


def spatial_cross_vel_torch(v, u):
    v_lin, v_ang = v[:, :3], v[:, 3:]
    u_lin, u_ang = u[:, :3], u[:, 3:]
    res_lin = torch.cross(v_ang, u_lin, dim=-1) + torch.cross(v_lin, u_ang, dim=-1)
    res_ang = torch.cross(v_ang, u_ang, dim=-1)
    return torch.cat([res_lin, res_ang], dim=-1)


def spatial_transform_matrix_torch(R, r):
    N = R.shape[0]
    E = R.transpose(-1, -2)
    zero = torch.zeros(N, device=R.device, dtype=R.dtype)
    skew_r = torch.stack(
        [
            zero,
            -r[:, 2],
            r[:, 1],
            r[:, 2],
            zero,
            -r[:, 0],
            -r[:, 1],
            r[:, 0],
            zero,
        ],
        dim=-1,
    ).reshape(N, 3, 3)
    M = torch.zeros(N, 6, 6, device=R.device, dtype=R.dtype)
    M[:, :3, :3] = E
    M[:, :3, 3:] = -torch.bmm(E, skew_r)
    M[:, 3:, 3:] = E
    return M
