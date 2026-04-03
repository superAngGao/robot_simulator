"""
Warp GPU kernels for batched robot simulation.

Each kernel launches with dim=N (one thread per environment).
Within each thread, the kernel loops over the robot's body tree sequentially.

Kernel launch order per step:
  1. batched_passive_torques
  2. batched_pd_controller
  3. tau_total = tau_action + tau_passive  (element-wise, done outside)
  4. batched_fk_body_vel
  5. batched_contact
  6. batched_collision
  7. batched_aba
  8. batched_integrate
"""

import warp as wp

from .spatial_warp import (  # noqa: E402
    compose_transform_r_wp,
    compose_transform_wp,
    inverse_transform_R,
    inverse_transform_r,
    mat66_add,
    mat66_mul_mat66,
    mat66_mul_vec6,
    mat66_sub,
    mat66_transpose,
    mat66f,
    quat_to_rot_wp,
    rodrigues_wp,
    spatial_cross_force_times_f,
    spatial_cross_vel_times_v,
    spatial_transform_matrix,
    transform_force_wp,
    transform_velocity_wp,
    vec6_angular,
    vec6_dot,
    vec6_from_two_vec3,
    vec6_linear,
    vec6f,
)

# Augmented matrix type for 6x6 linear system solve (6x7)
mat67f = wp.types.matrix(shape=(6, 7), dtype=wp.float32)

# Joint type constants (must match static_data.py)
JOINT_FREE = wp.constant(0)
JOINT_REVOLUTE = wp.constant(1)
JOINT_PRISMATIC = wp.constant(2)
JOINT_FIXED = wp.constant(3)


# ---------------------------------------------------------------------------
# Helper: compute joint transform X_J and motion subspace product S @ qdot
# ---------------------------------------------------------------------------


@wp.func
def joint_transform_R(
    jtype: int,
    axis: wp.vec3,
    q_start: int,
    q_len: int,
    q: wp.array2d(dtype=wp.float32),
    env_id: int,
) -> wp.mat33:
    """Compute the rotation part of X_J for a given joint."""
    R = wp.identity(n=3, dtype=float)
    if jtype == JOINT_REVOLUTE:
        angle = q[env_id, q_start]
        R = rodrigues_wp(axis, angle)
    elif jtype == JOINT_FREE:
        qw = q[env_id, q_start]
        qx = q[env_id, q_start + 1]
        qy = q[env_id, q_start + 2]
        qz = q[env_id, q_start + 3]
        R = quat_to_rot_wp(qw, qx, qy, qz)
    return R


@wp.func
def joint_transform_r(
    jtype: int,
    axis: wp.vec3,
    q_start: int,
    q_len: int,
    q: wp.array2d(dtype=wp.float32),
    env_id: int,
) -> wp.vec3:
    """Compute the translation part of X_J for a given joint."""
    r = wp.vec3(0.0, 0.0, 0.0)
    if jtype == JOINT_PRISMATIC:
        d = q[env_id, q_start]
        r = axis * d
    elif jtype == JOINT_FREE:
        r = wp.vec3(
            q[env_id, q_start + 4],
            q[env_id, q_start + 5],
            q[env_id, q_start + 6],
        )
    return r


@wp.func
def joint_vJ(
    jtype: int,
    axis: wp.vec3,
    v_start: int,
    v_len: int,
    qdot: wp.array2d(dtype=wp.float32),
    env_id: int,
) -> vec6f:
    """Compute vJ = S @ qdot for a given joint."""
    vJ = vec6f()
    if jtype == JOINT_REVOLUTE:
        # S = [0,0,0, ax,ay,az]^T, vJ = S * qdot_scalar
        qd = qdot[env_id, v_start]
        vJ = vec6f(0.0, 0.0, 0.0, axis[0] * qd, axis[1] * qd, axis[2] * qd)
    elif jtype == JOINT_PRISMATIC:
        qd = qdot[env_id, v_start]
        vJ = vec6f(axis[0] * qd, axis[1] * qd, axis[2] * qd, 0.0, 0.0, 0.0)
    elif jtype == JOINT_FREE:
        # MuJoCo convention: qdot[:3] = world-frame linear, qdot[3:] = body-frame angular.
        # vJ must be body-frame spatial velocity, so apply S = [[R^T, 0], [0, I]]:
        # vJ[:3] = R^T @ qdot[:3],  vJ[3:] = qdot[3:]
        # R_J is not available here; caller must post-multiply.
        # We return raw qdot; the FK kernel applies R^T after computing R_J.
        vJ = vec6f(
            qdot[env_id, v_start],
            qdot[env_id, v_start + 1],
            qdot[env_id, v_start + 2],
            qdot[env_id, v_start + 3],
            qdot[env_id, v_start + 4],
            qdot[env_id, v_start + 5],
        )
    return vJ


# ---------------------------------------------------------------------------
# Kernel 1: FK + body velocities
# ---------------------------------------------------------------------------


@wp.kernel
def batched_fk_body_vel(
    q: wp.array2d(dtype=wp.float32),
    qdot: wp.array2d(dtype=wp.float32),
    # Static data
    joint_type: wp.array(dtype=wp.int32),
    joint_axis: wp.array2d(dtype=wp.float32),
    parent_idx: wp.array(dtype=wp.int32),
    q_idx_start: wp.array(dtype=wp.int32),
    q_idx_len: wp.array(dtype=wp.int32),
    v_idx_start: wp.array(dtype=wp.int32),
    v_idx_len: wp.array(dtype=wp.int32),
    X_tree_R: wp.array3d(dtype=wp.float32),
    X_tree_r: wp.array2d(dtype=wp.float32),
    nb: int,
    # Outputs
    X_world_R: wp.array(dtype=wp.float32, ndim=4),
    X_world_r: wp.array3d(dtype=wp.float32),
    X_up_R: wp.array(dtype=wp.float32, ndim=4),
    X_up_r: wp.array3d(dtype=wp.float32),
    v_bodies: wp.array3d(dtype=wp.float32),
):
    env_id = wp.tid()

    for i in range(nb):
        jtype = joint_type[i]
        axis = wp.vec3(joint_axis[i, 0], joint_axis[i, 1], joint_axis[i, 2])
        qs = q_idx_start[i]
        ql = q_idx_len[i]
        vs = v_idx_start[i]
        vl = v_idx_len[i]
        pid = parent_idx[i]

        # Joint transform
        R_J = joint_transform_R(jtype, axis, qs, ql, q, env_id)
        r_J = joint_transform_r(jtype, axis, qs, ql, q, env_id)

        # X_tree for this body
        R_tree = wp.mat33(
            X_tree_R[i, 0, 0],
            X_tree_R[i, 0, 1],
            X_tree_R[i, 0, 2],
            X_tree_R[i, 1, 0],
            X_tree_R[i, 1, 1],
            X_tree_R[i, 1, 2],
            X_tree_R[i, 2, 0],
            X_tree_R[i, 2, 1],
            X_tree_R[i, 2, 2],
        )
        r_tree = wp.vec3(X_tree_r[i, 0], X_tree_r[i, 1], X_tree_r[i, 2])

        # X_up = X_tree @ X_J
        R_up = compose_transform_wp(R_tree, r_tree, R_J, r_J)
        r_up = compose_transform_r_wp(R_tree, r_tree, r_J)

        # Store X_up (needed by ABA later)
        for a in range(3):
            for b in range(3):
                X_up_R[env_id, i, a, b] = R_up[a, b]
            X_up_r[env_id, i, a] = r_up[a]

        # X_world
        if pid < 0:
            # Root body: X_world = X_up
            for a in range(3):
                for b in range(3):
                    X_world_R[env_id, i, a, b] = R_up[a, b]
                X_world_r[env_id, i, a] = r_up[a]
        else:
            # X_world = X_world[parent] @ X_up
            R_parent = wp.mat33(
                X_world_R[env_id, pid, 0, 0],
                X_world_R[env_id, pid, 0, 1],
                X_world_R[env_id, pid, 0, 2],
                X_world_R[env_id, pid, 1, 0],
                X_world_R[env_id, pid, 1, 1],
                X_world_R[env_id, pid, 1, 2],
                X_world_R[env_id, pid, 2, 0],
                X_world_R[env_id, pid, 2, 1],
                X_world_R[env_id, pid, 2, 2],
            )
            r_parent = wp.vec3(
                X_world_r[env_id, pid, 0],
                X_world_r[env_id, pid, 1],
                X_world_r[env_id, pid, 2],
            )
            R_w = compose_transform_wp(R_parent, r_parent, R_up, r_up)
            r_w = compose_transform_r_wp(R_parent, r_parent, r_up)
            for a in range(3):
                for b in range(3):
                    X_world_R[env_id, i, a, b] = R_w[a, b]
                X_world_r[env_id, i, a] = r_w[a]

        # Body velocity
        vJ = joint_vJ(jtype, axis, vs, vl, qdot, env_id)

        # FreeJoint: apply S = [[R_J^T, 0], [0, I]] to get body-frame velocity,
        # and compute Coriolis bias c_J = [-omega × v_body_lin, 0].
        if jtype == JOINT_FREE:
            v_world = wp.vec3(vJ[0], vJ[1], vJ[2])
            v_body_lin = wp.transpose(R_J) * v_world
            omega_body = wp.vec3(vJ[3], vJ[4], vJ[5])
            vJ = vec6f(
                v_body_lin[0], v_body_lin[1], v_body_lin[2], omega_body[0], omega_body[1], omega_body[2]
            )

        if pid < 0:
            for d in range(6):
                v_bodies[env_id, i, d] = vJ[d]
        else:
            # v[parent] in body frame
            v_parent = vec6f(
                v_bodies[env_id, pid, 0],
                v_bodies[env_id, pid, 1],
                v_bodies[env_id, pid, 2],
                v_bodies[env_id, pid, 3],
                v_bodies[env_id, pid, 4],
                v_bodies[env_id, pid, 5],
            )
            v_xformed = transform_velocity_wp(R_up, r_up, v_parent)
            for d in range(6):
                v_bodies[env_id, i, d] = v_xformed[d] + vJ[d]


# ---------------------------------------------------------------------------
# Kernel 2: Passive torques (joint limits + damping)
# ---------------------------------------------------------------------------


@wp.kernel
def batched_passive_torques(
    q: wp.array2d(dtype=wp.float32),
    qdot: wp.array2d(dtype=wp.float32),
    joint_type: wp.array(dtype=wp.int32),
    q_idx_start: wp.array(dtype=wp.int32),
    v_idx_start: wp.array(dtype=wp.int32),
    q_min: wp.array(dtype=wp.float32),
    q_max: wp.array(dtype=wp.float32),
    k_limit: wp.array(dtype=wp.float32),
    b_limit: wp.array(dtype=wp.float32),
    damping_coeff: wp.array(dtype=wp.float32),
    nb: int,
    # Output
    tau_passive: wp.array2d(dtype=wp.float32),
):
    env_id = wp.tid()

    for i in range(nb):
        jtype = joint_type[i]
        if jtype == JOINT_REVOLUTE:
            qs = q_idx_start[i]
            vs = v_idx_start[i]
            angle = q[env_id, qs]
            omega = qdot[env_id, vs]

            tau_val = float(0.0)

            # Joint limits
            qmin = q_min[i]
            qmax = q_max[i]
            if angle < qmin:
                pen = qmin - angle
                damp = wp.min(omega, 0.0)
                tau_val = k_limit[i] * pen - b_limit[i] * damp
            elif angle > qmax:
                pen = angle - qmax
                damp = wp.max(omega, 0.0)
                tau_val = -(k_limit[i] * pen + b_limit[i] * damp)

            # Viscous damping
            tau_val = tau_val - damping_coeff[i] * omega

            tau_passive[env_id, vs] = tau_val

        elif jtype == JOINT_PRISMATIC:
            vs = v_idx_start[i]
            omega = qdot[env_id, vs]
            tau_passive[env_id, vs] = -damping_coeff[i] * omega


# ---------------------------------------------------------------------------
# Kernel 3: PD controller
# ---------------------------------------------------------------------------


@wp.kernel
def batched_pd_controller(
    actions: wp.array2d(dtype=wp.float32),
    q: wp.array2d(dtype=wp.float32),
    qdot: wp.array2d(dtype=wp.float32),
    actuated_q_idx: wp.array(dtype=wp.int32),
    actuated_v_idx: wp.array(dtype=wp.int32),
    effort_limits: wp.array(dtype=wp.float32),
    has_effort_limits: int,
    kp: float,
    kd: float,
    action_scale: float,
    action_clip: float,  # negative means no clip
    nu: int,
    nv: int,
    # Output
    tau_action: wp.array2d(dtype=wp.float32),
):
    env_id = wp.tid()

    # Zero out tau first
    for j in range(nv):
        tau_action[env_id, j] = 0.0

    for j in range(nu):
        act = actions[env_id, j]
        # Action clip
        if action_clip > 0.0:
            act = wp.clamp(act, -action_clip, action_clip)

        qi = actuated_q_idx[j]
        vi = actuated_v_idx[j]

        target = q[env_id, qi] + act * action_scale
        tau_val = kp * (target - q[env_id, qi]) - kd * qdot[env_id, vi]

        # Effort limits
        if has_effort_limits > 0:
            lim = effort_limits[j]
            tau_val = wp.clamp(tau_val, -lim, lim)

        tau_action[env_id, vi] = tau_val


# ---------------------------------------------------------------------------
# Kernel 4: Penalty contact forces
# ---------------------------------------------------------------------------


@wp.kernel
def batched_contact(
    X_world_R: wp.array(dtype=wp.float32, ndim=4),
    X_world_r: wp.array3d(dtype=wp.float32),
    v_bodies: wp.array3d(dtype=wp.float32),
    contact_body_idx: wp.array(dtype=wp.int32),
    contact_local_pos: wp.array2d(dtype=wp.float32),
    k_normal: float,
    b_normal: float,
    mu: float,
    slip_eps: float,
    ground_z: float,
    nc: int,
    nb: int,
    # Outputs
    ext_forces: wp.array3d(dtype=wp.float32),
    contact_mask: wp.array2d(dtype=wp.int32),
):
    env_id = wp.tid()

    # Zero out ext_forces and contact_mask
    for i in range(nb):
        for d in range(6):
            ext_forces[env_id, i, d] = 0.0
    for c in range(nc):
        contact_mask[env_id, c] = 0

    for c in range(nc):
        bi = contact_body_idx[c]
        local_pos = wp.vec3(
            contact_local_pos[c, 0],
            contact_local_pos[c, 1],
            contact_local_pos[c, 2],
        )

        # World position = R @ local_pos + r
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
        if depth > 0.0:
            contact_mask[env_id, c] = 1

            # Contact velocity in world frame
            v_body = vec6f(
                v_bodies[env_id, bi, 0],
                v_bodies[env_id, bi, 1],
                v_bodies[env_id, bi, 2],
                v_bodies[env_id, bi, 3],
                v_bodies[env_id, bi, 4],
                v_bodies[env_id, bi, 5],
            )
            v_lin_w = R * vec6_linear(v_body)
            omega_w = R * vec6_angular(v_body)
            r_local_w = R * local_pos
            vel_world = v_lin_w + wp.cross(omega_w, r_local_w)

            # Normal force
            v_n = vel_world[2]
            F_n = k_normal * depth - b_normal * v_n
            F_n = wp.max(F_n, 0.0)

            # Tangential friction (regularised Coulomb)
            vx = vel_world[0]
            vy = vel_world[1]
            slip_norm = wp.sqrt(vx * vx + vy * vy + slip_eps * slip_eps)
            Ftx = -mu * F_n * vx / slip_norm
            Fty = -mu * F_n * vy / slip_norm

            # Force in world frame
            F_world = wp.vec3(Ftx, Fty, F_n)

            # Torque from moment arm
            r_arm = pos_world - r  # = R * local_pos
            torque_world = wp.cross(r_arm, F_world)

            # Spatial force in world frame [force; torque]
            f_world = vec6_from_two_vec3(F_world, torque_world)

            # Transform to body frame: X_world.inverse().apply_force()
            Rinv = inverse_transform_R(R)
            rinv = inverse_transform_r(R, r)
            f_body = transform_force_wp(Rinv, rinv, f_world)

            # Accumulate
            for d in range(6):
                ext_forces[env_id, bi, d] = ext_forces[env_id, bi, d] + f_body[d]


# ---------------------------------------------------------------------------
# Kernel 5: AABB self-collision forces
# ---------------------------------------------------------------------------


@wp.kernel
def batched_collision(
    X_world_R: wp.array(dtype=wp.float32, ndim=4),
    X_world_r: wp.array3d(dtype=wp.float32),
    v_bodies: wp.array3d(dtype=wp.float32),
    coll_body_idx: wp.array(dtype=wp.int32),
    coll_half_ext: wp.array2d(dtype=wp.float32),
    pair_i: wp.array(dtype=wp.int32),
    pair_j: wp.array(dtype=wp.int32),
    k_contact: float,
    b_contact: float,
    n_coll: int,
    n_pairs: int,
    nb: int,
    # Output (accumulated onto ext_forces)
    ext_forces: wp.array3d(dtype=wp.float32),
):
    env_id = wp.tid()

    for p in range(n_pairs):
        ii = pair_i[p]
        jj = pair_j[p]
        bi = coll_body_idx[ii]
        bj = coll_body_idx[jj]

        # Load transforms
        Ri = wp.mat33(
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
        ri = wp.vec3(X_world_r[env_id, bi, 0], X_world_r[env_id, bi, 1], X_world_r[env_id, bi, 2])
        Rj = wp.mat33(
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
        rj = wp.vec3(X_world_r[env_id, bj, 0], X_world_r[env_id, bj, 1], X_world_r[env_id, bj, 2])

        # Half extents
        he_i = wp.vec3(coll_half_ext[ii, 0], coll_half_ext[ii, 1], coll_half_ext[ii, 2])
        he_j = wp.vec3(coll_half_ext[jj, 0], coll_half_ext[jj, 1], coll_half_ext[jj, 2])

        # World AABB: world_half[k] = sum_m |R[k,m]| * local_half[m]
        separated = False
        min_overlap = float(1e10)
        min_axis = int(0)

        for k in range(3):
            wh_i = float(0.0)
            wh_j = float(0.0)
            for m in range(3):
                wh_i = wh_i + wp.abs(Ri[k, m]) * he_i[m]
                wh_j = wh_j + wp.abs(Rj[k, m]) * he_j[m]
            min_k = wp.min(ri[k] + wh_i, rj[k] + wh_j)
            max_k = wp.max(ri[k] - wh_i, rj[k] - wh_j)
            overlap = min_k - max_k
            if overlap <= 0.0:
                separated = True
            if overlap < min_overlap:
                min_overlap = overlap
                min_axis = k

        if not separated:
            depth = min_overlap
            # Direction
            sep = ri - rj
            sign = float(1.0)
            if sep[min_axis] < 0.0:
                sign = -1.0
            direction = wp.vec3(0.0, 0.0, 0.0)
            if min_axis == 0:
                direction = wp.vec3(sign, 0.0, 0.0)
            elif min_axis == 1:
                direction = wp.vec3(0.0, sign, 0.0)
            else:
                direction = wp.vec3(0.0, 0.0, sign)

            F_mag = k_contact * depth

            # Velocity damping
            vi_lin = wp.vec3(v_bodies[env_id, bi, 0], v_bodies[env_id, bi, 1], v_bodies[env_id, bi, 2])
            vj_lin = wp.vec3(v_bodies[env_id, bj, 0], v_bodies[env_id, bj, 1], v_bodies[env_id, bj, 2])
            vi_world = Ri * vi_lin
            vj_world = Rj * vj_lin
            v_rel = wp.dot(vi_world - vj_world, direction)
            if v_rel < 0.0:
                F_mag = F_mag - b_contact * v_rel

            F_world = direction * F_mag
            # Spatial force: zero torque (applied at body origin)
            f_sw_i = vec6_from_two_vec3(F_world, wp.vec3(0.0, 0.0, 0.0))
            f_sw_j = vec6_from_two_vec3(-F_world, wp.vec3(0.0, 0.0, 0.0))

            # Transform to body frames
            Rinv_i = inverse_transform_R(Ri)
            rinv_i = inverse_transform_r(Ri, ri)
            f_body_i = transform_force_wp(Rinv_i, rinv_i, f_sw_i)

            Rinv_j = inverse_transform_R(Rj)
            rinv_j = inverse_transform_r(Rj, rj)
            f_body_j = transform_force_wp(Rinv_j, rinv_j, f_sw_j)

            for d in range(6):
                ext_forces[env_id, bi, d] = ext_forces[env_id, bi, d] + f_body_i[d]
                ext_forces[env_id, bj, d] = ext_forces[env_id, bj, d] + f_body_j[d]


# ---------------------------------------------------------------------------
# Kernel 6: ABA (Articulated Body Algorithm) — forward dynamics
# ---------------------------------------------------------------------------


@wp.func
def load_mat66(arr: wp.array(dtype=wp.float32, ndim=4), env_id: int, body: int) -> mat66f:
    """Load a 6x6 matrix from a (N, nb, 6, 6) array."""
    M = mat66f()
    for i in range(6):
        for j in range(6):
            M[i, j] = arr[env_id, body, i, j]
    return M


@wp.func
def store_mat66(arr: wp.array(dtype=wp.float32, ndim=4), env_id: int, body: int, M: mat66f):
    """Store a 6x6 matrix into a (N, nb, 6, 6) array."""
    for i in range(6):
        for j in range(6):
            arr[env_id, body, i, j] = M[i, j]


@wp.func
def load_vec6(arr: wp.array3d(dtype=wp.float32), env_id: int, body: int) -> vec6f:
    return vec6f(
        arr[env_id, body, 0],
        arr[env_id, body, 1],
        arr[env_id, body, 2],
        arr[env_id, body, 3],
        arr[env_id, body, 4],
        arr[env_id, body, 5],
    )


@wp.func
def store_vec6(arr: wp.array3d(dtype=wp.float32), env_id: int, body: int, v: vec6f):
    for d in range(6):
        arr[env_id, body, d] = v[d]


@wp.kernel
def batched_aba(
    q: wp.array2d(dtype=wp.float32),
    qdot: wp.array2d(dtype=wp.float32),
    tau_total: wp.array2d(dtype=wp.float32),
    ext_forces: wp.array3d(dtype=wp.float32),
    # Static data
    joint_type: wp.array(dtype=wp.int32),
    joint_axis: wp.array2d(dtype=wp.float32),
    parent_idx: wp.array(dtype=wp.int32),
    q_idx_start: wp.array(dtype=wp.int32),
    q_idx_len: wp.array(dtype=wp.int32),
    v_idx_start: wp.array(dtype=wp.int32),
    v_idx_len: wp.array(dtype=wp.int32),
    inertia_mat: wp.array(dtype=wp.float32, ndim=3),
    gravity: float,
    nb: int,
    # Scratch (pre-allocated)
    X_up_R: wp.array(dtype=wp.float32, ndim=4),
    X_up_r: wp.array3d(dtype=wp.float32),
    aba_v: wp.array3d(dtype=wp.float32),
    aba_c: wp.array3d(dtype=wp.float32),
    aba_IA: wp.array(dtype=wp.float32, ndim=4),
    aba_pA: wp.array3d(dtype=wp.float32),
    aba_a: wp.array3d(dtype=wp.float32),
    aba_U: wp.array(dtype=wp.float32, ndim=4),
    aba_Dinv: wp.array(dtype=wp.float32, ndim=4),
    aba_u: wp.array3d(dtype=wp.float32),
    # Output
    qddot: wp.array2d(dtype=wp.float32),
):
    env_id = wp.tid()

    # ---- Pass 1: forward — velocities and bias forces ----
    for i in range(nb):
        jtype = joint_type[i]
        axis = wp.vec3(joint_axis[i, 0], joint_axis[i, 1], joint_axis[i, 2])
        qs = q_idx_start[i]
        ql = q_idx_len[i]
        vs = v_idx_start[i]
        vl = v_idx_len[i]
        pid = parent_idx[i]

        # X_up already computed by FK kernel
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

        vJ = joint_vJ(jtype, axis, vs, vl, qdot, env_id)

        # FreeJoint: apply R_J^T to linear part (joint_vJ returns raw qdot)
        # and compute Coriolis bias c_J_free = [-omega x v_body_lin, 0]
        cJ_free = vec6f()
        if jtype == JOINT_FREE:
            R_J = joint_transform_R(jtype, axis, qs, ql, q, env_id)
            v_world = wp.vec3(vJ[0], vJ[1], vJ[2])
            v_body_lin = wp.transpose(R_J) * v_world
            omega_body = wp.vec3(vJ[3], vJ[4], vJ[5])
            vJ = vec6f(
                v_body_lin[0], v_body_lin[1], v_body_lin[2], omega_body[0], omega_body[1], omega_body[2]
            )
            wxv = wp.cross(omega_body, v_body_lin)
            cJ_free = vec6f(-wxv[0], -wxv[1], -wxv[2], 0.0, 0.0, 0.0)

        vi = vec6f()
        ci = vec6f()
        if pid < 0:
            vi = vJ
            # Coriolis bias for FreeJoint
            ci = cJ_free
        else:
            v_parent = load_vec6(aba_v, env_id, pid)
            v_xformed = transform_velocity_wp(R_up, r_up, v_parent)
            for d in range(6):
                vi[d] = v_xformed[d] + vJ[d]
            cJ_standard = spatial_cross_vel_times_v(vi, vJ)
            for d in range(6):
                ci[d] = cJ_standard[d] + cJ_free[d]

        store_vec6(aba_v, env_id, i, vi)
        store_vec6(aba_c, env_id, i, ci)

        # IA = I_body (copy from static inertia)
        I_mat = mat66f()
        for a in range(6):
            for b in range(6):
                I_mat[a, b] = inertia_mat[i, a, b]
        store_mat66(aba_IA, env_id, i, I_mat)

        # pA = v ×* (I @ v) - ext_forces
        Iv = mat66_mul_vec6(I_mat, vi)
        pA_i = spatial_cross_force_times_f(vi, Iv)
        ext_f = load_vec6(ext_forces, env_id, i)
        for d in range(6):
            pA_i[d] = pA_i[d] - ext_f[d]
        store_vec6(aba_pA, env_id, i, pA_i)

    # ---- Pass 2: backward — articulated inertias ----
    for idx in range(nb):
        i = nb - 1 - idx  # reverse order
        jtype = joint_type[i]
        axis = wp.vec3(joint_axis[i, 0], joint_axis[i, 1], joint_axis[i, 2])
        vs = v_idx_start[i]
        vl = v_idx_len[i]
        pid = parent_idx[i]

        IA_i = load_mat66(aba_IA, env_id, i)
        pA_i = load_vec6(aba_pA, env_id, i)
        c_i = load_vec6(aba_c, env_id, i)

        IA_A = IA_i
        pA_A = vec6f()

        if vl > 0:
            if jtype == JOINT_REVOLUTE or jtype == JOINT_PRISMATIC:
                # S is a single column: [axis components]
                S_col = vec6f()
                if jtype == JOINT_REVOLUTE:
                    S_col = vec6f(0.0, 0.0, 0.0, axis[0], axis[1], axis[2])
                else:
                    S_col = vec6f(axis[0], axis[1], axis[2], 0.0, 0.0, 0.0)

                # U = IA @ S (a vec6)
                U_i = mat66_mul_vec6(IA_i, S_col)
                # D = S^T @ U (scalar)
                D_val = vec6_dot(S_col, U_i)
                D_inv_val = 1.0 / D_val
                # u = tau - S^T @ pA (scalar)
                tau_i = tau_total[env_id, vs]
                u_val = tau_i - vec6_dot(S_col, pA_i)

                # Store U, Dinv, u (in first column/element of scratch)
                store_vec6(aba_u, env_id, i, vec6f(u_val, 0.0, 0.0, 0.0, 0.0, 0.0))

                # IA_A = IA - (U @ U^T) * D_inv
                UUT = wp.outer(U_i, U_i)
                IA_A = mat66_sub(IA_i, UUT * D_inv_val)

                # pA_A = pA + IA_A @ c + U * D_inv * u
                IAc = mat66_mul_vec6(IA_A, c_i)
                for d in range(6):
                    pA_A[d] = pA_i[d] + IAc[d] + U_i[d] * D_inv_val * u_val

                # Store U and Dinv for Pass 3
                # Store U as first column of aba_U
                for d in range(6):
                    aba_U[env_id, i, d, 0] = U_i[d]
                aba_Dinv[env_id, i, 0, 0] = D_inv_val

            elif jtype == JOINT_FREE:
                # With S = [[R_J^T,0],[0,I]], work in body frame:
                # tau_body[:3] = R_J^T @ tau_gen[:3], tau_body[3:] = tau_gen[3:]
                # Then standard ABA with S=I on tau_body (IA_A = 0 still holds).
                R_J_free = joint_transform_R(jtype, axis, qs, ql, q, env_id)
                tau_gen_lin = wp.vec3(
                    tau_total[env_id, vs], tau_total[env_id, vs + 1], tau_total[env_id, vs + 2]
                )
                tau_body_lin = wp.transpose(R_J_free) * tau_gen_lin
                u_i = vec6f(
                    tau_body_lin[0] - pA_i[0],
                    tau_body_lin[1] - pA_i[1],
                    tau_body_lin[2] - pA_i[2],
                    tau_total[env_id, vs + 3] - pA_i[3],
                    tau_total[env_id, vs + 4] - pA_i[4],
                    tau_total[env_id, vs + 5] - pA_i[5],
                )
                store_vec6(aba_u, env_id, i, u_i)

                # For free joint: IA_A = 0, pA_A = pA + IA @ IA_inv @ u = pA + u
                # Actually: IA_A = IA - U @ D_inv @ U^T = IA - IA @ IA^{-1} @ IA = 0
                IA_A = mat66f()  # zero
                for d in range(6):
                    pA_A[d] = pA_i[d] + u_i[d]

                # Store IA as U and IA for Pass 3 (we need D_inv = IA^{-1})
                store_mat66(aba_U, env_id, i, IA_i)
                # We'll compute qddot differently for free joint in Pass 3
                # Store IA for D_inv computation
                store_mat66(aba_Dinv, env_id, i, IA_i)  # placeholder

        else:
            # Fixed joint (vl == 0): IA_A = IA, pA_A = pA + IA @ c
            IAc = mat66_mul_vec6(IA_i, c_i)
            for d in range(6):
                pA_A[d] = pA_i[d] + IAc[d]

        # Propagate to parent
        if pid >= 0:
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

            # IA[parent] += X^T @ IA_A @ X
            X6 = spatial_transform_matrix(R_up, r_up)
            X6T = mat66_transpose(X6)
            IA_parent = load_mat66(aba_IA, env_id, pid)
            contrib = mat66_mul_mat66(X6T, mat66_mul_mat66(IA_A, X6))
            IA_parent = mat66_add(IA_parent, contrib)
            store_mat66(aba_IA, env_id, pid, IA_parent)

            # pA[parent] += X_up.apply_force(pA_A)
            pA_parent = load_vec6(aba_pA, env_id, pid)
            pA_contrib = transform_force_wp(R_up, r_up, pA_A)
            for d in range(6):
                pA_parent[d] = pA_parent[d] + pA_contrib[d]
            store_vec6(aba_pA, env_id, pid, pA_parent)

    # ---- Pass 3: forward — accelerations ----
    for i in range(nb):
        jtype = joint_type[i]
        axis = wp.vec3(joint_axis[i, 0], joint_axis[i, 1], joint_axis[i, 2])
        vs = v_idx_start[i]
        vl = v_idx_len[i]
        pid = parent_idx[i]

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

        a_p = vec6f()
        if pid < 0:
            neg_grav = vec6f(0.0, 0.0, gravity, 0.0, 0.0, 0.0)  # -a_gravity
            a_p = transform_velocity_wp(R_up, r_up, neg_grav)
        else:
            a_parent = load_vec6(aba_a, env_id, pid)
            a_p = transform_velocity_wp(R_up, r_up, a_parent)

        c_i = load_vec6(aba_c, env_id, i)
        apc = vec6f()
        for d in range(6):
            apc[d] = a_p[d] + c_i[d]

        if vl > 0:
            if jtype == JOINT_REVOLUTE or jtype == JOINT_PRISMATIC:
                S_col = vec6f()
                if jtype == JOINT_REVOLUTE:
                    S_col = vec6f(0.0, 0.0, 0.0, axis[0], axis[1], axis[2])
                else:
                    S_col = vec6f(axis[0], axis[1], axis[2], 0.0, 0.0, 0.0)

                # U stored in first column of aba_U
                U_i = vec6f()
                for d in range(6):
                    U_i[d] = aba_U[env_id, i, d, 0]
                D_inv_val = aba_Dinv[env_id, i, 0, 0]
                u_val = aba_u[env_id, i, 0]

                # qddot = D_inv * (u - U^T @ (a_p + c))
                UT_apc = vec6_dot(U_i, apc)
                qddot_i = D_inv_val * (u_val - UT_apc)
                qddot[env_id, vs] = qddot_i

                # a = a_p + c + S * qddot
                ai = vec6f()
                for d in range(6):
                    ai[d] = apc[d] + S_col[d] * qddot_i
                store_vec6(aba_a, env_id, i, ai)

            elif jtype == JOINT_FREE:
                # For free joint: qddot = D_inv @ (u - U^T @ (a_p + c))
                # D_inv = IA^{-1}, U = IA, so qddot = IA^{-1} @ (u - IA @ apc) = IA^{-1} @ u - apc
                # Actually: u = tau - pA, and qddot = D_inv @ (u - U^T @ apc)
                # For FreeJoint S=I: qddot = IA^{-1} @ (u - IA @ apc) = IA^{-1} @ u - apc

                u_i = load_vec6(aba_u, env_id, i)
                IA_i = load_mat66(aba_Dinv, env_id, i)  # stored IA here

                # We need IA^{-1} @ (u - IA @ apc)
                IA_apc = mat66_mul_vec6(IA_i, apc)
                rhs = vec6f()
                for d in range(6):
                    rhs[d] = u_i[d] - IA_apc[d]

                # Solve IA @ qddot = rhs via Cholesky (IA is SPD)
                # For simplicity, use direct formula since IA is 6x6
                # We use Gaussian elimination (IA is well-conditioned)
                # Build augmented matrix [IA | rhs]
                aug = mat67f()
                for a in range(6):
                    for b in range(6):
                        aug[a, b] = IA_i[a, b]
                    aug[a, 6] = rhs[a]

                # Forward elimination with partial pivoting
                for col in range(6):
                    # Find pivot
                    max_val = wp.abs(aug[col, col])
                    max_row = col
                    for row in range(col + 1, 6):
                        val = wp.abs(aug[row, col])
                        if val > max_val:
                            max_val = val
                            max_row = row
                    # Swap rows
                    if max_row != col:
                        for k in range(7):
                            tmp = aug[col, k]
                            aug[col, k] = aug[max_row, k]
                            aug[max_row, k] = tmp
                    # Eliminate
                    pivot = aug[col, col]
                    for row in range(col + 1, 6):
                        factor = aug[row, col] / pivot
                        for k in range(col, 7):
                            aug[row, k] = aug[row, k] - factor * aug[col, k]

                # Back substitution
                qdd = vec6f()
                for row_idx in range(6):
                    row = 5 - row_idx
                    s = aug[row, 6]
                    for k in range(row + 1, 6):
                        s = s - aug[row, k] * qdd[k]
                    qdd[row] = s / aug[row, row]

                # qdd is body-frame. Convert to generalized coords:
                # qddot_gen[:3] = R_J @ qdd[:3], qddot_gen[3:6] = qdd[3:6]
                R_J_p3 = joint_transform_R(jtype, axis, qs, ql, q, env_id)
                qdd_body_lin = wp.vec3(qdd[0], qdd[1], qdd[2])
                qdd_gen_lin = R_J_p3 * qdd_body_lin
                qddot[env_id, vs + 0] = qdd_gen_lin[0]
                qddot[env_id, vs + 1] = qdd_gen_lin[1]
                qddot[env_id, vs + 2] = qdd_gen_lin[2]
                qddot[env_id, vs + 3] = qdd[3]
                qddot[env_id, vs + 4] = qdd[4]
                qddot[env_id, vs + 5] = qdd[5]

                # a = apc + qdd_body (body-frame acceleration, S=I in body frame)
                ai = vec6f()
                for d in range(6):
                    ai[d] = apc[d] + qdd[d]
                store_vec6(aba_a, env_id, i, ai)
        else:
            # Fixed joint: a = a_p + c
            store_vec6(aba_a, env_id, i, apc)


# ---------------------------------------------------------------------------
# Kernel 7: Semi-implicit Euler integration
# ---------------------------------------------------------------------------


@wp.kernel
def batched_integrate(
    q: wp.array2d(dtype=wp.float32),
    qdot: wp.array2d(dtype=wp.float32),
    qddot: wp.array2d(dtype=wp.float32),
    joint_type: wp.array(dtype=wp.int32),
    q_idx_start: wp.array(dtype=wp.int32),
    q_idx_len: wp.array(dtype=wp.int32),
    v_idx_start: wp.array(dtype=wp.int32),
    v_idx_len: wp.array(dtype=wp.int32),
    dt: float,
    nb: int,
    nq: int,
    nv: int,
    # Output
    q_new: wp.array2d(dtype=wp.float32),
    qdot_new: wp.array2d(dtype=wp.float32),
):
    env_id = wp.tid()

    # Step 1: qdot_new = qdot + dt * qddot
    for j in range(nv):
        qdot_new[env_id, j] = qdot[env_id, j] + dt * qddot[env_id, j]

    # Step 2: q_new = q + dt * qdot_new (with special quaternion handling)
    for i in range(nb):
        jtype = joint_type[i]
        qs = q_idx_start[i]
        ql = q_idx_len[i]
        vs = v_idx_start[i]
        vl = v_idx_len[i]

        if jtype == JOINT_FREE:
            # Quaternion integration
            qw = q[env_id, qs]
            qx = q[env_id, qs + 1]
            qy = q[env_id, qs + 2]
            qz = q[env_id, qs + 3]

            # qdot for free joint: [vx, vy, vz, wx, wy, wz]
            wx = qdot_new[env_id, vs + 3]
            wy = qdot_new[env_id, vs + 4]
            wz = qdot_new[env_id, vs + 5]

            # dq = 0.5 * Omega(w) @ q
            dqw = 0.5 * (-qx * wx - qy * wy - qz * wz)
            dqx = 0.5 * (qw * wx + qy * wz - qz * wy)
            dqy = 0.5 * (qw * wy - qx * wz + qz * wx)
            dqz = 0.5 * (qw * wz + qx * wy - qy * wx)

            nqw = qw + dqw * dt
            nqx = qx + dqx * dt
            nqy = qy + dqy * dt
            nqz = qz + dqz * dt

            # Normalize
            norm = wp.sqrt(nqw * nqw + nqx * nqx + nqy * nqy + nqz * nqz)
            q_new[env_id, qs] = nqw / norm
            q_new[env_id, qs + 1] = nqx / norm
            q_new[env_id, qs + 2] = nqy / norm
            q_new[env_id, qs + 3] = nqz / norm

            # Position: p_new = p + dt * v_lin
            q_new[env_id, qs + 4] = q[env_id, qs + 4] + dt * qdot_new[env_id, vs]
            q_new[env_id, qs + 5] = q[env_id, qs + 5] + dt * qdot_new[env_id, vs + 1]
            q_new[env_id, qs + 6] = q[env_id, qs + 6] + dt * qdot_new[env_id, vs + 2]

        elif vl > 0:
            # Standard Euler for revolute/prismatic
            for k in range(ql):
                q_new[env_id, qs + k] = q[env_id, qs + k] + dt * qdot_new[env_id, vs + k]
