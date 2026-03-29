"""
Warp GPU kernels for the ADMM constraint solver (velocity-level).

Replaces the Jacobi-PGS iteration loop (step 8 in GpuEngine) with a single
ADMM kernel that performs:
  1. Compliance R computation (MuJoCo solimp impedance model)
  2. Spring-damper bias (MuJoCo solref model)
  3. Cholesky factorization of (W + R + rho*I)
  4. Fixed-iteration ADMM loop (f-update, cone projection, dual update)
  5. Warmstart persistence

The kernel reuses the Delassus matrix W and free velocity v_free from
batched_build_W_vfree_v2 (step 7), and outputs lambdas consumed by
batched_impulse_to_gen_v2 (step 9).

Reference: physics/solvers/mujoco_qp.py (CPU ADMMQPSolver)
"""

import warp as wp

CONDIM = wp.constant(3)


# ---------------------------------------------------------------------------
# Helper: MuJoCo piecewise power-law sigmoid impedance
# Ref: mujoco_qp.py:274-295, MuJoCo engine_core_constraint.c
# ---------------------------------------------------------------------------
@wp.func
def _impedance(
    depth: float,
    d_0: float,
    d_width: float,
    width: float,
    midpoint: float,
    power: float,
) -> float:
    if width < 1.0e-10:
        return d_0
    x = wp.min(1.0, wp.max(0.0, wp.abs(depth) / width))
    if x <= 0.0:
        return d_0
    if x >= 1.0:
        return d_width
    # Piecewise power-law
    y = x
    if power > 1.0:
        if x <= midpoint:
            denom = wp.max(wp.pow(midpoint, power - 1.0), 1.0e-10)
            a = 1.0 / denom
            y = a * wp.pow(x, power)
        else:
            denom = wp.max(wp.pow(1.0 - midpoint, power - 1.0), 1.0e-10)
            b_coeff = 1.0 / denom
            y = 1.0 - b_coeff * wp.pow(1.0 - x, power)
    return d_0 + y * (d_width - d_0)


# ---------------------------------------------------------------------------
# Helper: In-kernel scalar Cholesky factorization
# Factors M[env, :n, :n] -> L[env, :n, :n] (lower triangular)
# Ref: Phase 2g CUDA small_cholesky_solve (kernels.cu:1031-1066)
# ---------------------------------------------------------------------------
@wp.func
def _cholesky_factor(
    env_id: int,
    M: wp.array(dtype=wp.float32, ndim=3),
    L: wp.array(dtype=wp.float32, ndim=3),
    n: int,
    max_rows: int,
    reg: float,
):
    # Zero out L
    for i in range(max_rows):
        for j in range(max_rows):
            L[env_id, i, j] = 0.0

    for i in range(n):
        for j in range(i + 1):
            s = float(0.0)
            for k in range(j):
                s += L[env_id, i, k] * L[env_id, j, k]
            if i == j:
                val = M[env_id, i, i] - s
                if val < reg:
                    val = reg
                L[env_id, i, j] = wp.sqrt(val)
            else:
                L_jj = L[env_id, j, j]
                if L_jj > 1.0e-12:
                    L[env_id, i, j] = (M[env_id, i, j] - s) / L_jj
                else:
                    L[env_id, i, j] = 0.0


# ---------------------------------------------------------------------------
# Helper: Triangular solve using Cholesky factor L
# Solves L L^T x = rhs  via forward-sub (L y = rhs) + back-sub (L^T x = y)
# ---------------------------------------------------------------------------
@wp.func
def _cholesky_solve(
    env_id: int,
    L: wp.array(dtype=wp.float32, ndim=3),
    rhs: wp.array(dtype=wp.float32, ndim=2),
    x: wp.array(dtype=wp.float32, ndim=2),
    tmp: wp.array(dtype=wp.float32, ndim=2),
    n: int,
):
    # Forward substitution: L y = rhs
    for i in range(n):
        s = rhs[env_id, i]
        for k in range(i):
            s -= L[env_id, i, k] * tmp[env_id, k]
        L_ii = L[env_id, i, i]
        if L_ii > 1.0e-12:
            tmp[env_id, i] = s / L_ii
        else:
            tmp[env_id, i] = 0.0

    # Backward substitution: L^T x = y
    for idx in range(n):
        i = n - 1 - idx
        s = tmp[env_id, i]
        for k in range(i + 1, n):
            s -= L[env_id, k, i] * x[env_id, k]  # L^T[i,k] = L[k,i]
        L_ii = L[env_id, i, i]
        if L_ii > 1.0e-12:
            x[env_id, i] = s / L_ii
        else:
            x[env_id, i] = 0.0


# ---------------------------------------------------------------------------
# Helper: Friction cone projection for all contacts
# Projects z -> s: f_n >= 0, ||f_t|| <= mu * f_n
# ---------------------------------------------------------------------------
@wp.func
def _cone_project_all(
    env_id: int,
    z: wp.array(dtype=wp.float32, ndim=2),
    s: wp.array(dtype=wp.float32, ndim=2),
    contact_active: wp.array(dtype=wp.int32, ndim=2),
    mu: float,
    max_contacts: int,
):
    for c in range(max_contacts):
        base = c * CONDIM
        if contact_active[env_id, c] == 0:
            s[env_id, base] = 0.0
            s[env_id, base + 1] = 0.0
            s[env_id, base + 2] = 0.0
            continue
        # Normal: clamp >= 0
        z_n = z[env_id, base]
        s_n = wp.max(0.0, z_n)
        s[env_id, base] = s_n
        # Tangent: project onto friction disk  ||f_t|| <= mu * f_n
        limit = mu * s_n
        z_t1 = z[env_id, base + 1]
        z_t2 = z[env_id, base + 2]
        t_norm = wp.sqrt(z_t1 * z_t1 + z_t2 * z_t2)
        if t_norm > limit and t_norm > 1.0e-12:
            scale = limit / t_norm
            s[env_id, base + 1] = z_t1 * scale
            s[env_id, base + 2] = z_t2 * scale
        else:
            s[env_id, base + 1] = z_t1
            s[env_id, base + 2] = z_t2


# ---------------------------------------------------------------------------
# Main kernel: batched_admm_solve
# ---------------------------------------------------------------------------
@wp.kernel
def batched_admm_solve(
    # -- From step 7 (W build) + step 7b (v_current) --
    W: wp.array(dtype=wp.float32, ndim=3),  # (N, max_rows, max_rows)
    W_diag: wp.array(dtype=wp.float32, ndim=2),  # (N, max_rows)
    v_free: wp.array(dtype=wp.float32, ndim=2),  # (N, max_rows) = J @ v_predicted
    v_current: wp.array(dtype=wp.float32, ndim=2),  # (N, max_rows) = J @ qdot
    # -- From step 6 (collision) --
    contact_active: wp.array(dtype=wp.int32, ndim=2),  # (N, max_contacts)
    contact_depth: wp.array(dtype=wp.float32, ndim=2),  # (N, max_contacts)
    # -- Scalar parameters --
    mu: float,
    rho: float,
    dt: float,
    solref_timeconst: float,
    solref_dampratio: float,
    solimp_d0: float,
    solimp_dwidth: float,
    solimp_width: float,
    solimp_midpoint: float,
    solimp_power: float,
    max_contacts: int,
    max_rows: int,
    admm_iters: int,
    warmstart_enabled: int,
    # -- Warmstart I/O --
    prev_n_active: wp.array(dtype=wp.int32, ndim=1),  # (N,)
    f_prev: wp.array(dtype=wp.float32, ndim=2),
    s_prev: wp.array(dtype=wp.float32, ndim=2),
    u_prev: wp.array(dtype=wp.float32, ndim=2),
    # -- Scratch arrays --
    AR_rho: wp.array(dtype=wp.float32, ndim=3),  # (N, max_rows, max_rows)
    L: wp.array(dtype=wp.float32, ndim=3),  # (N, max_rows, max_rows)
    R_diag: wp.array(dtype=wp.float32, ndim=2),  # (N, max_rows)
    f: wp.array(dtype=wp.float32, ndim=2),  # (N, max_rows)
    s: wp.array(dtype=wp.float32, ndim=2),  # (N, max_rows)
    u: wp.array(dtype=wp.float32, ndim=2),  # (N, max_rows)
    rhs_scratch: wp.array(dtype=wp.float32, ndim=2),  # (N, max_rows)
    tmp_scratch: wp.array(dtype=wp.float32, ndim=2),  # (N, max_rows)
    rhs_const: wp.array(dtype=wp.float32, ndim=2),  # (N, max_rows)
    # -- Output --
    lambdas: wp.array(dtype=wp.float32, ndim=2),  # (N, max_rows)
):
    env = wp.tid()

    # ── 1. Count active contacts, find n_active_rows ──
    n_active_contacts = int(0)
    highest_active = int(-1)
    for c in range(max_contacts):
        if contact_active[env, c] != 0:
            n_active_contacts = n_active_contacts + 1
            highest_active = c
    n_active_rows = int(0)
    if highest_active >= 0:
        n_active_rows = (highest_active + 1) * CONDIM

    # ── 2. Compute compliance R_diag + rhs_const ──
    # Spring-damper parameters from solref
    tc_clamped = wp.max(solref_timeconst, 2.0 * dt)
    d_w = solimp_dwidth  # impedance at full penetration
    b_coeff = 2.0 / (wp.max(d_w, 1.0e-10) * tc_clamped)
    k_coeff = 1.0 / (
        wp.max(d_w, 1.0e-10)
        * wp.max(d_w, 1.0e-10)
        * tc_clamped
        * tc_clamped
        * wp.max(solref_dampratio * solref_dampratio, 1.0e-10)
    )

    # Zero out scratch
    for r in range(max_rows):
        R_diag[env, r] = 0.0
        rhs_const[env, r] = 0.0

    for c in range(max_contacts):
        base = c * CONDIM
        if contact_active[env, c] == 0:
            continue

        depth_c = contact_depth[env, c]
        d_imp = _impedance(depth_c, solimp_d0, solimp_dwidth, solimp_width, solimp_midpoint, solimp_power)

        # Compliance R_diag: (1-d)/d * |W_diag|
        ratio = (1.0 - d_imp) / wp.max(d_imp, 1.0e-10)
        for k in range(CONDIM):
            R_diag[env, base + k] = ratio * wp.abs(W_diag[env, base + k])

        # Exact velocity-space rhs_const using separated v_c and v_free:
        #
        #   rhs = dt * a_ref(v_c) - (v_free - v_c)
        #       = dt * (-b*v_c + k*d*depth) - v_free + v_c   [normal]
        #       = dt * (-b*v_c)             - v_free + v_c   [tangent]
        #
        # This is exact — no v_c ≈ v_free approximation. At equilibrium
        # (v_c=0): rhs = dt*k*d*depth + dt*g (gravity from v_free = -dt*g).
        # During impact (v_c=-2): rhs = dt*(-b*(-2)+k*d*depth) - (-2.002)+(-2)
        #                             ≈ dt*(a_ref - a_uc), matching CPU ADMM.
        vc_n = v_current[env, base]
        vf_n = v_free[env, base]
        rhs_const[env, base] = dt * (-b_coeff * vc_n + k_coeff * d_imp * depth_c) - vf_n + vc_n
        for k in range(1, CONDIM):
            vc_t = v_current[env, base + k]
            vf_t = v_free[env, base + k]
            rhs_const[env, base + k] = dt * (-b_coeff * vc_t) - vf_t + vc_t

    # ── 3. Build AR_rho = W + diag(R_diag) + rho*I ──
    reg = float(1.0e-4)
    for i in range(n_active_rows):
        for j in range(n_active_rows):
            AR_rho[env, i, j] = W[env, i, j]
        AR_rho[env, i, i] = W[env, i, i] + R_diag[env, i] + rho
    # Zero the rest (for clean Cholesky)
    for i in range(n_active_rows, max_rows):
        for j in range(max_rows):
            AR_rho[env, i, j] = 0.0
        for j in range(n_active_rows):
            AR_rho[env, j, i] = 0.0

    # ── 4. Cholesky factorization ──
    if n_active_rows > 0:
        _cholesky_factor(env, AR_rho, L, n_active_rows, max_rows, reg)

    # ── 5. Warmstart ──
    if warmstart_enabled != 0 and prev_n_active[env] == n_active_contacts and n_active_contacts > 0:
        for r in range(n_active_rows):
            f[env, r] = f_prev[env, r]
            s[env, r] = s_prev[env, r]
            u[env, r] = u_prev[env, r]
    else:
        for r in range(max_rows):
            f[env, r] = 0.0
            s[env, r] = 0.0
            u[env, r] = 0.0

    # ── 6. ADMM iterations ──
    if n_active_rows > 0:
        for _iter in range(admm_iters):
            # f-update: rhs = rhs_const + rho*(s - u)
            for r in range(n_active_rows):
                rhs_scratch[env, r] = rhs_const[env, r] + rho * (s[env, r] - u[env, r])
            _cholesky_solve(env, L, rhs_scratch, f, tmp_scratch, n_active_rows)

            # s-update: s = proj_cone(f + u)
            # Write f + u into rhs_scratch (reuse), then project
            for r in range(max_rows):
                rhs_scratch[env, r] = f[env, r] + u[env, r]
            _cone_project_all(env, rhs_scratch, s, contact_active, mu, max_contacts)

            # u-update: u += f - s
            for r in range(n_active_rows):
                u[env, r] = u[env, r] + f[env, r] - s[env, r]

    # ── 7. Output: lambdas = s (already in the cone) ──
    for r in range(max_rows):
        lambdas[env, r] = s[env, r]

    # ── 8. Save warmstart state ──
    prev_n_active[env] = n_active_contacts
    for r in range(max_rows):
        f_prev[env, r] = f[env, r]
        s_prev[env, r] = s[env, r]
        u_prev[env, r] = u[env, r]
