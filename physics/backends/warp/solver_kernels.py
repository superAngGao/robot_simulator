"""
Warp GPU kernels for the Jacobi-PGS-SI constraint solver.

Replaces the penalty contact pipeline (batched_contact + batched_collision)
with a constraint-based solver operating on predicted velocities.

Kernel launch order (from warp_backend.py):
  K1. batched_ground_detect       — contact point vs ground plane
  K2. batched_predicted_velocity  — v_predicted = qdot + dt * a_u
  K3. batched_build_W_vfree       — Jacobian + Delassus matrix + free velocity
  K4. batched_jacobi_pgs_step     — one Jacobi PGS iteration (launch ×60)
  K5. batched_impulse_to_gen      — lambda → body impulses → generalized impulse
  K6. batched_position_correction — split impulse position pushout
  K7. batched_constraint_integrate — qdot_new = v_pred + dqdot; q integration

All kernels launch with dim=N (one thread per environment).
Each thread loops over contacts/bodies/rows sequentially within its env.
"""

import warp as wp

from .spatial_warp import (
    inverse_transform_R,
    inverse_transform_r,
    quat_to_rot_wp,
    transform_force_wp,
    vec6_angular,
    vec6_from_two_vec3,
    vec6_linear,
    vec6f,
)

# Joint type constants (must match static_data.py)
JOINT_FREE = wp.constant(0)
JOINT_REVOLUTE = wp.constant(1)
JOINT_PRISMATIC = wp.constant(2)
JOINT_FIXED = wp.constant(3)

CONDIM = wp.constant(3)  # Fixed condim=3 for ground contact (normal + 2 tangent)


@wp.func
def _impedance_wp(
    depth: float,
    d_0: float,
    d_width: float,
    imp_width: float,
    midpoint: float,
    power: float,
) -> float:
    """Compute impedance d(r) from solimp parameters (MuJoCo sigmoid)."""
    if imp_width < 1.0e-10:
        return d_0
    x = wp.min(1.0, wp.max(0.0, wp.abs(depth) / imp_width))
    if x <= 0.0:
        return d_0
    if x >= 1.0:
        return d_width
    y = x
    if power > 1.0:
        if x <= midpoint:
            a = 1.0 / wp.max(wp.pow(midpoint, power - 1.0), 1.0e-10)
            y = a * wp.pow(x, power)
        else:
            b_c = 1.0 / wp.max(wp.pow(1.0 - midpoint, power - 1.0), 1.0e-10)
            y = 1.0 - b_c * wp.pow(1.0 - x, power)
    return d_0 + y * (d_width - d_0)


# ---------------------------------------------------------------------------
# K1: Ground contact detection
# ---------------------------------------------------------------------------


@wp.kernel
def batched_ground_detect(
    X_world_R: wp.array(dtype=wp.float32, ndim=4),
    X_world_r: wp.array3d(dtype=wp.float32),
    contact_body_idx: wp.array(dtype=wp.int32),
    contact_local_pos: wp.array2d(dtype=wp.float32),
    ground_z: float,
    nc: int,
    # Outputs
    contact_depth: wp.array2d(dtype=wp.float32),
    contact_point_world: wp.array3d(dtype=wp.float32),
    contact_active: wp.array2d(dtype=wp.int32),
):
    """Detect ground contacts for each contact point."""
    env_id = wp.tid()

    for c in range(nc):
        bi = contact_body_idx[c]
        local_pos = wp.vec3(
            contact_local_pos[c, 0],
            contact_local_pos[c, 1],
            contact_local_pos[c, 2],
        )

        R = wp.mat33(
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
        r = wp.vec3(
            X_world_r[env_id, bi, 0],
            X_world_r[env_id, bi, 1],
            X_world_r[env_id, bi, 2],
        )
        pos_world = R * local_pos + r
        depth = ground_z - pos_world[2]

        contact_depth[env_id, c] = depth
        contact_point_world[env_id, c, 0] = pos_world[0]
        contact_point_world[env_id, c, 1] = pos_world[1]
        contact_point_world[env_id, c, 2] = pos_world[2]
        if depth > 0.0:
            contact_active[env_id, c] = 1
        else:
            contact_active[env_id, c] = 0


# ---------------------------------------------------------------------------
# K2: Predicted velocity
# ---------------------------------------------------------------------------


@wp.kernel
def batched_predicted_velocity(
    qdot: wp.array2d(dtype=wp.float32),
    qddot: wp.array2d(dtype=wp.float32),
    dt: float,
    nv: int,
    # Output
    v_predicted: wp.array2d(dtype=wp.float32),
):
    """v_predicted = qdot + dt * qddot (unconstrained)."""
    env_id = wp.tid()
    for i in range(nv):
        v_predicted[env_id, i] = qdot[env_id, i] + dt * qddot[env_id, i]


# ---------------------------------------------------------------------------
# K3: Build Jacobian, Delassus matrix W, and free velocity v_free
# ---------------------------------------------------------------------------


@wp.kernel
def batched_build_W_vfree(
    X_world_R: wp.array(dtype=wp.float32, ndim=4),
    X_world_r: wp.array3d(dtype=wp.float32),
    v_bodies_pred: wp.array3d(dtype=wp.float32),
    contact_body_idx: wp.array(dtype=wp.int32),
    contact_local_pos: wp.array2d(dtype=wp.float32),
    contact_active: wp.array2d(dtype=wp.int32),
    contact_depth: wp.array2d(dtype=wp.float32),
    inv_mass: wp.array(dtype=wp.float32),
    inv_inertia: wp.array3d(dtype=wp.float32),  # (nb, 3, 3)
    mu: float,
    cfm: float,
    solimp_d0: float,
    solimp_dw: float,
    solimp_width: float,
    solimp_mid: float,
    solimp_power: float,
    nc: int,
    max_rows: int,
    # Outputs
    W: wp.array(dtype=wp.float32, ndim=3),  # (N, max_rows, max_rows)
    W_diag: wp.array2d(dtype=wp.float32),
    v_free: wp.array2d(dtype=wp.float32),
    J_body: wp.array3d(dtype=wp.float32),  # (N, max_rows, 6)
    row_body_idx: wp.array2d(dtype=wp.int32),
):
    """Build contact Jacobian, Delassus matrix, and free velocity.

    Ground contact: body_j = ground (infinite mass), normal = [0,0,1],
    tangent1 = [1,0,0], tangent2 = [0,1,0].
    """
    env_id = wp.tid()

    # Zero outputs
    for r in range(max_rows):
        W_diag[env_id, r] = 0.0
        v_free[env_id, r] = 0.0
        for k in range(6):
            J_body[env_id, r, k] = 0.0
        row_body_idx[env_id, r] = -1
        for r2 in range(max_rows):
            W[env_id, r, r2] = 0.0

    # Directions: normal=[0,0,1], t1=[1,0,0], t2=[0,1,0]
    normal = wp.vec3(0.0, 0.0, 1.0)
    t1 = wp.vec3(1.0, 0.0, 0.0)
    t2 = wp.vec3(0.0, 1.0, 0.0)

    for c in range(nc):
        if contact_active[env_id, c] == 0:
            continue

        bi = contact_body_idx[c]
        base = c * CONDIM

        # Load body transform
        R = wp.mat33(
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
        r_body = wp.vec3(
            X_world_r[env_id, bi, 0],
            X_world_r[env_id, bi, 1],
            X_world_r[env_id, bi, 2],
        )
        Rt = wp.transpose(R)

        # Contact point world position and moment arm
        local_pos = wp.vec3(
            contact_local_pos[c, 0],
            contact_local_pos[c, 1],
            contact_local_pos[c, 2],
        )
        pos_world = R * local_pos + r_body
        r_arm = pos_world - r_body  # = R * local_pos

        # Predicted body velocity
        v_body = vec6f(
            v_bodies_pred[env_id, bi, 0],
            v_bodies_pred[env_id, bi, 1],
            v_bodies_pred[env_id, bi, 2],
            v_bodies_pred[env_id, bi, 3],
            v_bodies_pred[env_id, bi, 4],
            v_bodies_pred[env_id, bi, 5],
        )
        v_lin_w = R * vec6_linear(v_body)
        omega_w = R * vec6_angular(v_body)
        v_contact = v_lin_w + wp.cross(omega_w, r_arm)

        # Build Jacobian rows and v_free for each direction
        direction_list_0 = normal
        direction_list_1 = t1
        direction_list_2 = t2

        for d_idx in range(CONDIM):
            row = base + d_idx
            if d_idx == 0:
                direction = direction_list_0
            elif d_idx == 1:
                direction = direction_list_1
            else:
                direction = direction_list_2

            # J_lin = Rᵀ @ direction (body frame)
            J_lin = Rt * direction
            # J_ang = Rᵀ @ cross(r_arm, direction) (body frame)
            rxd = wp.cross(r_arm, direction)
            J_ang = Rt * rxd

            # Store Jacobian row
            J_body[env_id, row, 0] = J_lin[0]
            J_body[env_id, row, 1] = J_lin[1]
            J_body[env_id, row, 2] = J_lin[2]
            J_body[env_id, row, 3] = J_ang[0]
            J_body[env_id, row, 4] = J_ang[1]
            J_body[env_id, row, 5] = J_ang[2]
            row_body_idx[env_id, row] = bi

            # v_free = J @ v_body (contact velocity in constraint space)
            v_free[env_id, row] = wp.dot(direction, v_contact)

    # Build Delassus matrix W = J M⁻¹ Jᵀ
    n_rows = nc * CONDIM
    for r1 in range(n_rows):
        bi1 = row_body_idx[env_id, r1]
        if bi1 < 0:
            continue
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

        # M⁻¹ @ J[r1]ᵀ
        j1_lin = wp.vec3(J_body[env_id, r1, 0], J_body[env_id, r1, 1], J_body[env_id, r1, 2])
        j1_ang = wp.vec3(J_body[env_id, r1, 3], J_body[env_id, r1, 4], J_body[env_id, r1, 5])
        Minv_j1_lin = m_inv * j1_lin
        Minv_j1_ang = I_inv * j1_ang

        for r2 in range(n_rows):
            bi2 = row_body_idx[env_id, r2]
            if bi2 != bi1:
                continue  # Only rows sharing a body contribute

            j2_lin = wp.vec3(J_body[env_id, r2, 0], J_body[env_id, r2, 1], J_body[env_id, r2, 2])
            j2_ang = wp.vec3(J_body[env_id, r2, 3], J_body[env_id, r2, 4], J_body[env_id, r2, 5])

            # W[r1,r2] = J[r2] · M⁻¹J[r1]
            val = wp.dot(j2_lin, Minv_j1_lin) + wp.dot(j2_ang, Minv_j1_ang)
            W[env_id, r1, r2] = W[env_id, r1, r2] + val

    # Per-row regularization: normal rows get uniform cfm,
    # friction rows get R = (1-d)/d * |W_ii| (Q25 fix).
    for c in range(nc):
        if contact_active[env_id, c] == 0:
            continue
        base = c * CONDIM
        depth_c = contact_depth[env_id, c]
        d_imp = _impedance_wp(depth_c, solimp_d0, solimp_dw, solimp_width, solimp_mid, solimp_power)
        ratio = (1.0 - d_imp) / wp.max(d_imp, 1.0e-10)

        # Normal row: uniform cfm
        W_diag[env_id, base] = W[env_id, base, base] + cfm
        # Friction rows: per-row R
        for off in range(1, CONDIM):
            row = base + off
            W_diag[env_id, row] = W[env_id, row, row] + ratio * wp.abs(W[env_id, row, row])


# ---------------------------------------------------------------------------
# K4: One Jacobi PGS iteration
# ---------------------------------------------------------------------------


@wp.kernel
def batched_jacobi_pgs_step(
    W: wp.array(dtype=wp.float32, ndim=3),
    W_diag: wp.array2d(dtype=wp.float32),
    v_free: wp.array2d(dtype=wp.float32),
    lambdas_old: wp.array2d(dtype=wp.float32),
    contact_active: wp.array2d(dtype=wp.int32),
    mu: float,
    omega: float,
    nc: int,
    max_rows: int,
    # Output
    lambdas: wp.array2d(dtype=wp.float32),
):
    """One iteration of Jacobi PGS with friction cone projection.

    All rows read from lambdas_old (previous iteration snapshot).
    erp=0 (no Baumgarte bias) for split impulse stability.
    """
    env_id = wp.tid()
    n_rows = nc * CONDIM

    for c in range(nc):
        if contact_active[env_id, c] == 0:
            continue

        base = c * CONDIM

        # -- Normal row (base + 0) --
        row_n = base
        Wl_n = float(0.0)
        for j in range(n_rows):
            Wl_n = Wl_n + W[env_id, row_n, j] * lambdas_old[env_id, j]
        residual_n = v_free[env_id, row_n] + Wl_n
        diag_n = W_diag[env_id, row_n]
        if diag_n > 1.0e-12:
            delta_n = -residual_n / diag_n
        else:
            delta_n = 0.0
        raw_n = lambdas_old[env_id, row_n] + omega * delta_n
        lambda_n = wp.max(0.0, raw_n)
        lambdas[env_id, row_n] = lambda_n

        # Friction limit
        limit = mu * lambda_n

        # -- Tangent rows (base + 1, base + 2) --
        for off in range(1, CONDIM):
            row_t = base + off
            Wl_t = float(0.0)
            for j in range(n_rows):
                Wl_t = Wl_t + W[env_id, row_t, j] * lambdas_old[env_id, j]
            residual_t = v_free[env_id, row_t] + Wl_t
            diag_t = W_diag[env_id, row_t]
            if diag_t > 1.0e-12:
                delta_t = -residual_t / diag_t
            else:
                delta_t = 0.0
            raw_t = lambdas_old[env_id, row_t] + omega * delta_t
            lambdas[env_id, row_t] = wp.clamp(raw_t, -limit, limit)


# ---------------------------------------------------------------------------
# K5: Convert lambdas → body impulses → generalized impulse
# ---------------------------------------------------------------------------


@wp.kernel
def batched_impulse_to_gen(
    lambdas: wp.array2d(dtype=wp.float32),
    contact_active: wp.array2d(dtype=wp.int32),
    contact_body_idx: wp.array(dtype=wp.int32),
    contact_local_pos: wp.array2d(dtype=wp.float32),
    X_world_R: wp.array(dtype=wp.float32, ndim=4),
    X_world_r: wp.array3d(dtype=wp.float32),
    X_up_R: wp.array(dtype=wp.float32, ndim=4),
    X_up_r: wp.array3d(dtype=wp.float32),
    joint_type: wp.array(dtype=wp.int32),
    joint_axis: wp.array2d(dtype=wp.float32),
    q: wp.array2d(dtype=wp.float32),
    q_idx_start: wp.array(dtype=wp.int32),
    v_idx_start: wp.array(dtype=wp.int32),
    v_idx_len: wp.array(dtype=wp.int32),
    parent_idx: wp.array(dtype=wp.int32),
    nc: int,
    nb: int,
    nv: int,
    # Outputs
    body_impulses: wp.array3d(dtype=wp.float32),  # (N, nb, 6)
    gen_impulse: wp.array2d(dtype=wp.float32),  # (N, nv)
):
    """Convert constraint-space lambdas to generalized impulse via RNEA backward pass."""
    env_id = wp.tid()

    # Zero outputs
    for i in range(nb):
        for d in range(6):
            body_impulses[env_id, i, d] = 0.0
    for i in range(nv):
        gen_impulse[env_id, i] = 0.0

    # Ground contact directions (fixed for flat ground)
    normal = wp.vec3(0.0, 0.0, 1.0)
    t1_dir = wp.vec3(1.0, 0.0, 0.0)
    t2_dir = wp.vec3(0.0, 1.0, 0.0)

    # Step 1: lambdas → body-frame spatial impulses
    for c in range(nc):
        if contact_active[env_id, c] == 0:
            continue
        bi = contact_body_idx[c]
        base = c * CONDIM

        l_n = lambdas[env_id, base]
        l_t1 = lambdas[env_id, base + 1]
        l_t2 = lambdas[env_id, base + 2]

        # Force in world frame
        F_world = l_n * normal + l_t1 * t1_dir + l_t2 * t2_dir

        # Moment arm
        local_pos = wp.vec3(
            contact_local_pos[c, 0],
            contact_local_pos[c, 1],
            contact_local_pos[c, 2],
        )
        R = wp.mat33(
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
        r_body = wp.vec3(
            X_world_r[env_id, bi, 0],
            X_world_r[env_id, bi, 1],
            X_world_r[env_id, bi, 2],
        )
        r_arm = R * local_pos  # contact offset in world frame
        torque_world = wp.cross(r_arm, F_world)

        # Transform to body frame
        Rinv = inverse_transform_R(R)
        rinv = inverse_transform_r(R, r_body)
        f_world_6 = vec6_from_two_vec3(F_world, torque_world)
        f_body = transform_force_wp(Rinv, rinv, f_world_6)

        # Accumulate
        for d in range(6):
            body_impulses[env_id, bi, d] = body_impulses[env_id, bi, d] + f_body[d]

    # Step 2: RNEA backward pass — body impulses → generalized impulse
    # Process bodies in reverse order (leaves → root)
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

        # gen_impulse[v_idx] = Sᵀ @ f_i
        jtype = joint_type[i]
        vs = v_idx_start[i]

        if jtype == JOINT_REVOLUTE or jtype == JOINT_PRISMATIC:
            # S is (6,1) axis vector
            axis = wp.vec3(joint_axis[i, 0], joint_axis[i, 1], joint_axis[i, 2])
            if jtype == JOINT_REVOLUTE:
                # S = [0,0,0, ax,ay,az] → Sᵀf = dot(axis, f_angular)
                gen_impulse[env_id, vs] = axis[0] * f_i[3] + axis[1] * f_i[4] + axis[2] * f_i[5]
            else:
                # S = [ax,ay,az, 0,0,0] → Sᵀf = dot(axis, f_linear)
                gen_impulse[env_id, vs] = axis[0] * f_i[0] + axis[1] * f_i[1] + axis[2] * f_i[2]
        elif jtype == JOINT_FREE:
            # S = [[R_J^T, 0], [0, I]] → S^T = [[R_J, 0], [0, I]]
            # gen_impulse[:3] = R_J @ f[:3],  gen_impulse[3:6] = f[3:6]
            qs_free = q_idx_start[i]
            RJ_free = quat_to_rot_wp(
                q[env_id, qs_free],
                q[env_id, qs_free + 1],
                q[env_id, qs_free + 2],
                q[env_id, qs_free + 3],
            )
            f_lin = wp.vec3(f_i[0], f_i[1], f_i[2])
            tau_lin = RJ_free * f_lin
            gen_impulse[env_id, vs + 0] = tau_lin[0]
            gen_impulse[env_id, vs + 1] = tau_lin[1]
            gen_impulse[env_id, vs + 2] = tau_lin[2]
            gen_impulse[env_id, vs + 3] = f_i[3]
            gen_impulse[env_id, vs + 4] = f_i[4]
            gen_impulse[env_id, vs + 5] = f_i[5]

        # Propagate to parent: f_parent += X_up.apply_force(f_child)
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
            r_up = wp.vec3(
                X_up_r[env_id, i, 0],
                X_up_r[env_id, i, 1],
                X_up_r[env_id, i, 2],
            )
            f_parent = transform_force_wp(R_up, r_up, f_i)
            for d in range(6):
                body_impulses[env_id, pi, d] = body_impulses[env_id, pi, d] + f_parent[d]


# ---------------------------------------------------------------------------
# K6: Position correction (split impulse)
# ---------------------------------------------------------------------------


@wp.kernel
def batched_position_correction(
    contact_active: wp.array2d(dtype=wp.int32),
    contact_depth: wp.array2d(dtype=wp.float32),
    contact_body_idx: wp.array(dtype=wp.int32),
    inv_mass: wp.array(dtype=wp.float32),
    erp: float,
    slop: float,
    nc: int,
    nb: int,
    # Output
    pos_corrections: wp.array3d(dtype=wp.float32),  # (N, nb, 3)
):
    """Compute per-body position corrections for split impulse."""
    env_id = wp.tid()

    # Zero
    for i in range(nb):
        for d in range(3):
            pos_corrections[env_id, i, d] = 0.0

    normal = wp.vec3(0.0, 0.0, 1.0)

    for c in range(nc):
        if contact_active[env_id, c] == 0:
            continue

        depth = contact_depth[env_id, c]
        effective_depth = depth - slop
        if effective_depth <= 0.0:
            continue

        bi = contact_body_idx[c]
        correction = erp * effective_depth

        # For ground contact: body_j = ground (inf mass), so body_i gets full correction
        m_inv = inv_mass[bi]
        if m_inv < 1.0e-10:
            continue

        # body_i gets correction along normal
        pos_corrections[env_id, bi, 0] = pos_corrections[env_id, bi, 0] + correction * normal[0]
        pos_corrections[env_id, bi, 1] = pos_corrections[env_id, bi, 1] + correction * normal[1]
        pos_corrections[env_id, bi, 2] = pos_corrections[env_id, bi, 2] + correction * normal[2]


# ---------------------------------------------------------------------------
# K7: Constraint integration
# ---------------------------------------------------------------------------


@wp.kernel
def batched_constraint_integrate(
    q: wp.array2d(dtype=wp.float32),
    v_predicted: wp.array2d(dtype=wp.float32),
    dqdot: wp.array2d(dtype=wp.float32),
    pos_corrections: wp.array3d(dtype=wp.float32),
    joint_type: wp.array(dtype=wp.int32),
    q_idx_start: wp.array(dtype=wp.int32),
    q_idx_len: wp.array(dtype=wp.int32),
    v_idx_start: wp.array(dtype=wp.int32),
    v_idx_len: wp.array(dtype=wp.int32),
    dt: float,
    nb: int,
    nq: int,
    nv: int,
    # Outputs
    q_new: wp.array2d(dtype=wp.float32),
    qdot_new: wp.array2d(dtype=wp.float32),
):
    """Integrate with constraint correction: qdot_new = v_pred + dqdot, then q integration."""
    env_id = wp.tid()

    # qdot_new = v_predicted + dqdot
    for i in range(nv):
        qdot_new[env_id, i] = v_predicted[env_id, i] + dqdot[env_id, i]

    # q integration (same logic as batched_integrate but reads from new qdot)
    for i in range(nb):
        jtype = joint_type[i]
        qs = q_idx_start[i]
        vs = v_idx_start[i]

        if jtype == JOINT_FREE:
            # Quaternion integration: q_new = normalize(q + 0.5 * dt * omega_quat * q)
            qw = q[env_id, qs + 0]
            qx = q[env_id, qs + 1]
            qy = q[env_id, qs + 2]
            qz = q[env_id, qs + 3]

            # Angular velocity (body frame) from qdot
            wx = qdot_new[env_id, vs + 3]
            wy = qdot_new[env_id, vs + 4]
            wz = qdot_new[env_id, vs + 5]

            # Quaternion derivative: dq/dt = 0.5 * [0,wx,wy,wz] * q
            dqw = 0.5 * (-wx * qx - wy * qy - wz * qz)
            dqx = 0.5 * (wx * qw + wz * qy - wy * qz)
            dqy = 0.5 * (wy * qw - wz * qx + wx * qz)
            dqz = 0.5 * (wz * qw + wy * qx - wx * qy)

            nqw = qw + dt * dqw
            nqx = qx + dt * dqx
            nqy = qy + dt * dqy
            nqz = qz + dt * dqz

            # Normalize
            norm = wp.sqrt(nqw * nqw + nqx * nqx + nqy * nqy + nqz * nqz)
            if norm > 1.0e-10:
                nqw = nqw / norm
                nqx = nqx / norm
                nqy = nqy / norm
                nqz = nqz / norm

            q_new[env_id, qs + 0] = nqw
            q_new[env_id, qs + 1] = nqx
            q_new[env_id, qs + 2] = nqy
            q_new[env_id, qs + 3] = nqz

            # Position: p_new = p + dt * v_linear
            vx = qdot_new[env_id, vs + 0]
            vy = qdot_new[env_id, vs + 1]
            vz = qdot_new[env_id, vs + 2]
            q_new[env_id, qs + 4] = q[env_id, qs + 4] + dt * vx
            q_new[env_id, qs + 5] = q[env_id, qs + 5] + dt * vy
            q_new[env_id, qs + 6] = q[env_id, qs + 6] + dt * vz

            # Apply position corrections to translation
            q_new[env_id, qs + 4] = q_new[env_id, qs + 4] + pos_corrections[env_id, i, 0]
            q_new[env_id, qs + 5] = q_new[env_id, qs + 5] + pos_corrections[env_id, i, 1]
            q_new[env_id, qs + 6] = q_new[env_id, qs + 6] + pos_corrections[env_id, i, 2]

        elif jtype == JOINT_REVOLUTE or jtype == JOINT_PRISMATIC:
            q_new[env_id, qs] = q[env_id, qs] + dt * qdot_new[env_id, vs]

        elif jtype == JOINT_FIXED:
            pass  # no DOFs


# ---------------------------------------------------------------------------
# Helper: scale gen_impulse by 1/dt for ABA trick
# ---------------------------------------------------------------------------


@wp.kernel
def batched_scale_array(
    src: wp.array2d(dtype=wp.float32),
    scale: float,
    n: int,
    # Output
    dst: wp.array2d(dtype=wp.float32),
):
    """dst[env, :] = src[env, :] * scale."""
    env_id = wp.tid()
    for i in range(n):
        dst[env_id, i] = src[env_id, i] * scale


@wp.kernel
def batched_scale_array_inplace(
    arr: wp.array2d(dtype=wp.float32),
    scale: float,
    n: int,
):
    """arr[env, :] *= scale."""
    env_id = wp.tid()
    for i in range(n):
        arr[env_id, i] = arr[env_id, i] * scale
