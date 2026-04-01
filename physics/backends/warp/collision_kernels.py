"""
Warp GPU kernels for collision detection (ground + body-body).

Two kernels:
  - batched_detect_all_contacts: original sphere-only (kept for backward compat)
  - batched_detect_analytical:   shape-type dispatch with analytical functions

All contacts produce: depth, normal, point, body_i, body_j, active flag.
The solver kernels then process these uniformly.
"""

import warp as wp

from .analytical_collision import (
    SHAPE_BOX,
    SHAPE_CAPSULE,
    SHAPE_CYLINDER,
    SHAPE_SPHERE,
    _box_ground_contact_point,
    _capsule_ground_contact_point,
    box_vs_ground,
    capsule_capsule,
    capsule_vs_ground,
    closest_points_seg_seg_both,
    closest_points_segment_segment,
    cylinder_vs_ground,
    sphere_box,
    sphere_box_normal,
    sphere_capsule,
    sphere_capsule_normal_point,
    sphere_sphere,
    sphere_sphere_normal,
    sphere_vs_ground,
)


@wp.kernel
def batched_detect_all_contacts(
    X_world_R: wp.array(dtype=wp.float32, ndim=4),
    X_world_r: wp.array3d(dtype=wp.float32),
    # Ground contacts
    contact_body_idx: wp.array(dtype=wp.int32),
    contact_local_pos: wp.array2d(dtype=wp.float32),
    ground_z: float,
    nc_ground: int,
    # Body-body pairs
    pair_body_i: wp.array(dtype=wp.int32),
    pair_body_j: wp.array(dtype=wp.int32),
    body_radius: wp.array(dtype=wp.float32),
    n_pairs: int,
    # Total max contacts
    max_contacts: int,
    # Outputs (max_contacts = nc_ground + n_pairs)
    contact_depth: wp.array2d(dtype=wp.float32),
    contact_normal: wp.array3d(dtype=wp.float32),  # (N, max_contacts, 3)
    contact_point: wp.array3d(dtype=wp.float32),  # (N, max_contacts, 3)
    contact_bi: wp.array2d(dtype=wp.int32),  # (N, max_contacts)
    contact_bj: wp.array2d(dtype=wp.int32),  # (N, max_contacts)
    contact_active: wp.array2d(dtype=wp.int32),  # (N, max_contacts)
):
    """Detect ground contacts + body-body sphere collisions for all envs."""
    env_id = wp.tid()

    # Zero all outputs
    for c in range(max_contacts):
        contact_depth[env_id, c] = 0.0
        contact_active[env_id, c] = 0
        contact_bi[env_id, c] = -1
        contact_bj[env_id, c] = -1
        for d in range(3):
            contact_normal[env_id, c, d] = 0.0
            contact_point[env_id, c, d] = 0.0

    # ── Ground contacts ──
    # For spherical bodies, contact_local_pos encodes the offset from body
    # origin to the contact surface. The lowest point is computed using
    # the offset MAGNITUDE (radius) along the ground normal, independent
    # of body rotation. This avoids the problem where a rotating body's
    # fixed contact point swings above the ground.
    for c in range(nc_ground):
        bi = contact_body_idx[c]
        local_pos = wp.vec3(
            contact_local_pos[c, 0],
            contact_local_pos[c, 1],
            contact_local_pos[c, 2],
        )
        r = wp.vec3(
            X_world_r[env_id, bi, 0],
            X_world_r[env_id, bi, 1],
            X_world_r[env_id, bi, 2],
        )

        # Use offset magnitude as radius, project along ground normal (z-axis)
        radius = wp.length(local_pos)
        lowest_z = r[2] - radius  # lowest point of sphere
        depth = ground_z - lowest_z

        if depth > 0.0:
            contact_depth[env_id, c] = depth
            contact_normal[env_id, c, 0] = 0.0
            contact_normal[env_id, c, 1] = 0.0
            contact_normal[env_id, c, 2] = 1.0
            # Contact point at lowest point of sphere
            contact_point[env_id, c, 0] = r[0]
            contact_point[env_id, c, 1] = r[1]
            contact_point[env_id, c, 2] = ground_z
            contact_bi[env_id, c] = bi
            contact_bj[env_id, c] = -1  # ground
            contact_active[env_id, c] = 1

    # ── Body-body sphere collisions ──
    for p in range(n_pairs):
        ci = nc_ground + p  # contact slot index
        bi = pair_body_i[p]
        bj = pair_body_j[p]

        pos_i = wp.vec3(
            X_world_r[env_id, bi, 0],
            X_world_r[env_id, bi, 1],
            X_world_r[env_id, bi, 2],
        )
        pos_j = wp.vec3(
            X_world_r[env_id, bj, 0],
            X_world_r[env_id, bj, 1],
            X_world_r[env_id, bj, 2],
        )

        diff = pos_i - pos_j
        dist = wp.length(diff)
        ri = body_radius[bi]
        rj = body_radius[bj]
        overlap = (ri + rj) - dist

        if overlap > 0.0 and dist > 1.0e-10:
            normal = diff / dist  # from j to i
            contact_depth[env_id, ci] = overlap
            contact_normal[env_id, ci, 0] = normal[0]
            contact_normal[env_id, ci, 1] = normal[1]
            contact_normal[env_id, ci, 2] = normal[2]
            # Contact point on body j's surface
            cp = pos_j + normal * rj
            contact_point[env_id, ci, 0] = cp[0]
            contact_point[env_id, ci, 1] = cp[1]
            contact_point[env_id, ci, 2] = cp[2]
            contact_bi[env_id, ci] = bi
            contact_bj[env_id, ci] = bj
            contact_active[env_id, ci] = 1


# ---------------------------------------------------------------------------
# Helper: load body rotation matrix from flat array
# ---------------------------------------------------------------------------


@wp.func
def _load_R(X_world_R: wp.array(dtype=wp.float32, ndim=4), env_id: int, bi: int) -> wp.mat33:
    return wp.mat33(
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


@wp.func
def _load_pos(X_world_r: wp.array3d(dtype=wp.float32), env_id: int, bi: int) -> wp.vec3:
    return wp.vec3(X_world_r[env_id, bi, 0], X_world_r[env_id, bi, 1], X_world_r[env_id, bi, 2])


# ---------------------------------------------------------------------------
# Helper: write contact to output buffers
# ---------------------------------------------------------------------------


@wp.func
def _write_contact(
    env_id: int,
    ci: int,
    depth: float,
    normal: wp.vec3,
    point: wp.vec3,
    bi: int,
    bj: int,
    contact_depth: wp.array2d(dtype=wp.float32),
    contact_normal: wp.array3d(dtype=wp.float32),
    contact_point: wp.array3d(dtype=wp.float32),
    contact_bi: wp.array2d(dtype=wp.int32),
    contact_bj: wp.array2d(dtype=wp.int32),
    contact_active: wp.array2d(dtype=wp.int32),
):
    contact_depth[env_id, ci] = depth
    contact_normal[env_id, ci, 0] = normal[0]
    contact_normal[env_id, ci, 1] = normal[1]
    contact_normal[env_id, ci, 2] = normal[2]
    contact_point[env_id, ci, 0] = point[0]
    contact_point[env_id, ci, 1] = point[1]
    contact_point[env_id, ci, 2] = point[2]
    contact_bi[env_id, ci] = bi
    contact_bj[env_id, ci] = bj
    contact_active[env_id, ci] = 1


# ---------------------------------------------------------------------------
# Analytical collision kernel with shape-type dispatch
# ---------------------------------------------------------------------------


@wp.kernel
def batched_detect_analytical(
    X_world_R: wp.array(dtype=wp.float32, ndim=4),
    X_world_r: wp.array3d(dtype=wp.float32),
    # Shape data
    shape_type: wp.array(dtype=wp.int32),  # (nb,)
    shape_params: wp.array2d(dtype=wp.float32),  # (nb, 4)
    # Ground contacts
    contact_body_idx: wp.array(dtype=wp.int32),
    ground_z: float,
    nc_ground: int,
    # Body-body pairs
    pair_body_i: wp.array(dtype=wp.int32),
    pair_body_j: wp.array(dtype=wp.int32),
    body_radius: wp.array(dtype=wp.float32),  # fallback radius
    n_pairs: int,
    max_contacts: int,
    # Outputs
    contact_depth: wp.array2d(dtype=wp.float32),
    contact_normal: wp.array3d(dtype=wp.float32),
    contact_point: wp.array3d(dtype=wp.float32),
    contact_bi: wp.array2d(dtype=wp.int32),
    contact_bj: wp.array2d(dtype=wp.int32),
    contact_active: wp.array2d(dtype=wp.int32),
):
    """Analytical collision detection with per-shape-type dispatch."""
    env_id = wp.tid()

    # Zero outputs
    for c in range(max_contacts):
        contact_depth[env_id, c] = 0.0
        contact_active[env_id, c] = 0
        contact_bi[env_id, c] = -1
        contact_bj[env_id, c] = -1
        for d in range(3):
            contact_normal[env_id, c, d] = 0.0
            contact_point[env_id, c, d] = 0.0

    # ── Ground contacts (shape-type dispatch) ──
    for c in range(nc_ground):
        bi = contact_body_idx[c]
        pos = _load_pos(X_world_r, env_id, bi)
        R = _load_R(X_world_R, env_id, bi)
        st = shape_type[bi]

        depth = float(0.0)
        hit = float(0.0)
        cp = wp.vec3(pos[0], pos[1], ground_z)

        if st == SHAPE_SPHERE:
            radius = shape_params[bi, 0]
            res = sphere_vs_ground(pos, radius, ground_z)
            depth = res[0]
            hit = res[1]
            cp = wp.vec3(pos[0], pos[1], ground_z)

        elif st == SHAPE_CAPSULE:
            radius = shape_params[bi, 0]
            hl = shape_params[bi, 1]
            res = capsule_vs_ground(pos, R, radius, hl, ground_z)
            depth = res[0]
            hit = res[1]
            low = _capsule_ground_contact_point(pos, R, hl)
            cp = wp.vec3(low[0], low[1], ground_z)

        elif st == SHAPE_BOX:
            hx = shape_params[bi, 0]
            hy = shape_params[bi, 1]
            hz = shape_params[bi, 2]
            res = box_vs_ground(pos, R, hx, hy, hz, ground_z)
            depth = res[0]
            hit = res[1]
            low = _box_ground_contact_point(pos, R, hx, hy, hz)
            cp = wp.vec3(low[0], low[1], ground_z)

        elif st == SHAPE_CYLINDER:
            radius = shape_params[bi, 0]
            hl = shape_params[bi, 1]
            res = cylinder_vs_ground(pos, R, radius, hl, ground_z)
            depth = res[0]
            hit = res[1]
            # Approximate contact point at body position projected to ground
            cp = wp.vec3(pos[0], pos[1], ground_z)

        else:
            # SHAPE_NONE or unknown: fallback to sphere with body_radius
            radius = body_radius[bi]
            if radius > 0.0:
                res = sphere_vs_ground(pos, radius, ground_z)
                depth = res[0]
                hit = res[1]
                cp = wp.vec3(pos[0], pos[1], ground_z)

        if hit > 0.5:
            normal = wp.vec3(0.0, 0.0, 1.0)
            _write_contact(
                env_id,
                c,
                depth,
                normal,
                cp,
                bi,
                -1,
                contact_depth,
                contact_normal,
                contact_point,
                contact_bi,
                contact_bj,
                contact_active,
            )

    # ── Body-body contacts (shape-type dispatch) ──
    for p in range(n_pairs):
        ci = nc_ground + p
        bi = pair_body_i[p]
        bj = pair_body_j[p]
        pos_i = _load_pos(X_world_r, env_id, bi)
        pos_j = _load_pos(X_world_r, env_id, bj)
        R_i = _load_R(X_world_R, env_id, bi)
        R_j = _load_R(X_world_R, env_id, bj)
        ti = shape_type[bi]
        tj = shape_type[bj]

        depth = float(0.0)
        hit = float(0.0)
        normal = wp.vec3(0.0, 0.0, 1.0)
        cp = wp.vec3(0.0, 0.0, 0.0)

        # Canonicalize: ensure ti <= tj to reduce dispatch cases
        # If swapped, negate normal at the end
        swap = int(0)
        a_pos = pos_i
        b_pos = pos_j
        a_R = R_i
        b_R = R_j
        a_t = ti
        b_t = tj
        a_bi = bi
        b_bi = bj
        if ti > tj:
            swap = 1
            a_pos = pos_j
            b_pos = pos_i
            a_R = R_j
            b_R = R_i
            a_t = tj
            b_t = ti
            a_bi = bj
            b_bi = bi

        # Now a_t <= b_t
        if a_t == SHAPE_SPHERE and b_t == SHAPE_SPHERE:
            r_a = shape_params[a_bi, 0]
            r_b = shape_params[b_bi, 0]
            res = sphere_sphere(a_pos, r_a, b_pos, r_b)
            depth = res[0]
            hit = res[1]
            normal = sphere_sphere_normal(a_pos, b_pos)
            cp = b_pos + normal * r_b  # contact on b's surface

        elif a_t == SHAPE_SPHERE and b_t == SHAPE_BOX:
            r_a = shape_params[a_bi, 0]
            hx = shape_params[b_bi, 0]
            hy = shape_params[b_bi, 1]
            hz = shape_params[b_bi, 2]
            res = sphere_box(a_pos, r_a, b_pos, b_R, hx, hy, hz)
            depth = res[0]
            hit = res[1]
            normal = sphere_box_normal(a_pos, r_a, b_pos, b_R, hx, hy, hz)
            cp = a_pos - normal * r_a  # contact on sphere surface toward box

        elif a_t == SHAPE_SPHERE and b_t == SHAPE_CAPSULE:
            r_a = shape_params[a_bi, 0]
            r_b = shape_params[b_bi, 0]
            hl_b = shape_params[b_bi, 1]
            res = sphere_capsule(a_pos, r_a, b_pos, b_R, r_b, hl_b)
            depth = res[0]
            hit = res[1]
            normal = sphere_capsule_normal_point(a_pos, r_a, b_pos, b_R, r_b, hl_b)
            cp = a_pos - normal * r_a

        elif a_t == SHAPE_CAPSULE and b_t == SHAPE_CAPSULE:
            r_a = shape_params[a_bi, 0]
            hl_a = shape_params[a_bi, 1]
            r_b = shape_params[b_bi, 0]
            hl_b = shape_params[b_bi, 1]
            res = capsule_capsule(a_pos, a_R, r_a, hl_a, b_pos, b_R, r_b, hl_b)
            depth = res[0]
            hit = res[1]
            # Normal from closest points
            axis_a = a_R * wp.vec3(0.0, 0.0, 1.0)
            axis_b = b_R * wp.vec3(0.0, 0.0, 1.0)
            seg_a_start = a_pos - hl_a * axis_a
            seg_b_start = b_pos - hl_b * axis_b
            pt_a = closest_points_segment_segment(
                seg_a_start,
                axis_a,
                2.0 * hl_a,
                seg_b_start,
                axis_b,
                2.0 * hl_b,
            )
            pt_b = closest_points_seg_seg_both(
                seg_a_start,
                axis_a,
                2.0 * hl_a,
                seg_b_start,
                axis_b,
                2.0 * hl_b,
            )
            diff = pt_a - pt_b
            dist = wp.length(diff)
            if dist > 1.0e-10:
                normal = diff / dist
            cp = pt_b + normal * r_b

        else:
            # Unsupported pair: fallback to sphere-sphere with body_radius
            r_a = body_radius[a_bi]
            r_b = body_radius[b_bi]
            res = sphere_sphere(a_pos, r_a, b_pos, r_b)
            depth = res[0]
            hit = res[1]
            normal = sphere_sphere_normal(a_pos, b_pos)
            cp = b_pos + normal * r_b

        # If we swapped bodies, negate normal (normal should point from bj→bi original)
        if swap == 1:
            normal = wp.vec3(-normal[0], -normal[1], -normal[2])

        if hit > 0.5:
            _write_contact(
                env_id,
                ci,
                depth,
                normal,
                cp,
                bi,
                bj,
                contact_depth,
                contact_normal,
                contact_point,
                contact_bi,
                contact_bj,
                contact_active,
            )


# ---------------------------------------------------------------------------
# Helper: load shape local rotation from flat (nshape, 9) array
# ---------------------------------------------------------------------------


@wp.func
def _load_shape_R(shape_rotation: wp.array2d(dtype=wp.float32), s_idx: int) -> wp.mat33:
    return wp.mat33(
        shape_rotation[s_idx, 0],
        shape_rotation[s_idx, 1],
        shape_rotation[s_idx, 2],
        shape_rotation[s_idx, 3],
        shape_rotation[s_idx, 4],
        shape_rotation[s_idx, 5],
        shape_rotation[s_idx, 6],
        shape_rotation[s_idx, 7],
        shape_rotation[s_idx, 8],
    )


@wp.func
def _load_shape_offset(shape_offset: wp.array2d(dtype=wp.float32), s_idx: int) -> wp.vec3:
    return wp.vec3(shape_offset[s_idx, 0], shape_offset[s_idx, 1], shape_offset[s_idx, 2])


@wp.func
def _compose_shape_world(
    X_world_R: wp.array(dtype=wp.float32, ndim=4),
    X_world_r: wp.array3d(dtype=wp.float32),
    shape_offset: wp.array2d(dtype=wp.float32),
    shape_rotation: wp.array2d(dtype=wp.float32),
    env_id: int,
    bi: int,
    s_idx: int,
) -> wp.mat33:
    """Compose shape world rotation: R_world = R_body @ R_local.

    Returns R_shape_world.  Use _compose_shape_world_pos for position.
    """
    R_body = _load_R(X_world_R, env_id, bi)
    R_local = _load_shape_R(shape_rotation, s_idx)
    return R_body * R_local


@wp.func
def _compose_shape_world_pos(
    X_world_R: wp.array(dtype=wp.float32, ndim=4),
    X_world_r: wp.array3d(dtype=wp.float32),
    shape_offset: wp.array2d(dtype=wp.float32),
    env_id: int,
    bi: int,
    s_idx: int,
) -> wp.vec3:
    """Compose shape world position: p_world = R_body @ offset + p_body."""
    R_body = _load_R(X_world_R, env_id, bi)
    p_body = _load_pos(X_world_r, env_id, bi)
    offset = _load_shape_offset(shape_offset, s_idx)
    return p_body + R_body * offset


# ---------------------------------------------------------------------------
# Multi-shape collision kernel with dynamic N² broadphase (Q26-gpu)
# ---------------------------------------------------------------------------


@wp.kernel
def batched_detect_multishape(
    X_world_R: wp.array(dtype=wp.float32, ndim=4),
    X_world_r: wp.array3d(dtype=wp.float32),
    # Flat shape arrays (MuJoCo-style)
    flat_shape_type: wp.array(dtype=wp.int32),
    flat_shape_params: wp.array2d(dtype=wp.float32),
    flat_shape_offset: wp.array2d(dtype=wp.float32),
    flat_shape_rotation: wp.array2d(dtype=wp.float32),
    body_shape_adr: wp.array(dtype=wp.int32),
    body_shape_num: wp.array(dtype=wp.int32),
    body_collision_radius: wp.array(dtype=wp.float32),
    # Ground contact bodies
    contact_body_idx: wp.array(dtype=wp.int32),
    ground_z: float,
    nc_ground_bodies: int,
    # Dynamic broadphase
    collision_excluded: wp.array2d(dtype=wp.int32),
    nb: int,
    broadphase_margin: float,
    max_contacts: int,
    # Outputs
    contact_count: wp.array(dtype=wp.int32),
    contact_depth: wp.array2d(dtype=wp.float32),
    contact_normal: wp.array3d(dtype=wp.float32),
    contact_point: wp.array3d(dtype=wp.float32),
    contact_bi: wp.array2d(dtype=wp.int32),
    contact_bj: wp.array2d(dtype=wp.int32),
    contact_active: wp.array2d(dtype=wp.int32),
):
    """Multi-shape collision with dynamic N² broadphase.

    Three stages per env:
      A. Zero outputs + reset contact counter
      B. Ground contacts: iterate all shapes per contact body
      C. N² broadphase (bounding sphere) → multi-shape narrowphase
    """
    env_id = wp.tid()

    # ── Stage A: Zero outputs ──
    contact_count[env_id] = 0
    for c in range(max_contacts):
        contact_active[env_id, c] = 0

    # ── Stage B: Ground contacts (multi-shape) ──
    for cb in range(nc_ground_bodies):
        bi = contact_body_idx[cb]
        ns = body_shape_num[bi]
        adr = body_shape_adr[bi]

        for si in range(ns):
            s_idx = adr + si
            st = flat_shape_type[s_idx]

            # Compose shape world pose
            pos_s = _compose_shape_world_pos(X_world_R, X_world_r, flat_shape_offset, env_id, bi, s_idx)
            R_s = _compose_shape_world(
                X_world_R,
                X_world_r,
                flat_shape_offset,
                flat_shape_rotation,
                env_id,
                bi,
                s_idx,
            )

            # Inline ground dispatch (Warp @wp.func can't return tuples)
            g_depth = float(0.0)
            g_hit = float(0.0)
            g_cp = wp.vec3(pos_s[0], pos_s[1], ground_z)

            if st == SHAPE_SPHERE:
                g_res = sphere_vs_ground(pos_s, flat_shape_params[s_idx, 0], ground_z)
                g_depth = g_res[0]
                g_hit = g_res[1]
                g_cp = wp.vec3(pos_s[0], pos_s[1], ground_z)
            elif st == SHAPE_CAPSULE:
                g_r = flat_shape_params[s_idx, 0]
                g_hl = flat_shape_params[s_idx, 1]
                g_res = capsule_vs_ground(pos_s, R_s, g_r, g_hl, ground_z)
                g_depth = g_res[0]
                g_hit = g_res[1]
                g_low = _capsule_ground_contact_point(pos_s, R_s, g_hl)
                g_cp = wp.vec3(g_low[0], g_low[1], ground_z)
            elif st == SHAPE_BOX:
                g_hx = flat_shape_params[s_idx, 0]
                g_hy = flat_shape_params[s_idx, 1]
                g_hz = flat_shape_params[s_idx, 2]
                g_res = box_vs_ground(pos_s, R_s, g_hx, g_hy, g_hz, ground_z)
                g_depth = g_res[0]
                g_hit = g_res[1]
                g_low = _box_ground_contact_point(pos_s, R_s, g_hx, g_hy, g_hz)
                g_cp = wp.vec3(g_low[0], g_low[1], ground_z)
            elif st == SHAPE_CYLINDER:
                g_r = flat_shape_params[s_idx, 0]
                g_hl = flat_shape_params[s_idx, 1]
                g_res = cylinder_vs_ground(pos_s, R_s, g_r, g_hl, ground_z)
                g_depth = g_res[0]
                g_hit = g_res[1]
            else:
                g_br = body_collision_radius[bi]
                if g_br > 0.0:
                    g_res = sphere_vs_ground(pos_s, g_br, ground_z)
                    g_depth = g_res[0]
                    g_hit = g_res[1]

            if g_hit > 0.5:
                slot = wp.atomic_add(contact_count, env_id, 1)
                if slot < max_contacts:
                    _write_contact(
                        env_id,
                        slot,
                        g_depth,
                        wp.vec3(0.0, 0.0, 1.0),
                        g_cp,
                        bi,
                        -1,
                        contact_depth,
                        contact_normal,
                        contact_point,
                        contact_bi,
                        contact_bj,
                        contact_active,
                    )

    # ── Stage C: Dynamic N² broadphase + multi-shape narrowphase ──
    for bi in range(nb):
        ns_i = body_shape_num[bi]
        if ns_i == 0:
            continue
        pos_bi = _load_pos(X_world_r, env_id, bi)
        ri = body_collision_radius[bi]

        for bj in range(bi + 1, nb):
            # Collision filter
            if collision_excluded[bi, bj] == 1:
                continue

            ns_j = body_shape_num[bj]
            if ns_j == 0:
                continue

            # Bounding sphere separation (cheapest test)
            pos_bj = _load_pos(X_world_r, env_id, bj)
            rj = body_collision_radius[bj]
            diff = pos_bi - pos_bj
            dist = wp.length(diff)
            if dist > ri + rj + broadphase_margin:
                continue

            # Narrowphase: all shape pairs
            adr_i = body_shape_adr[bi]
            adr_j = body_shape_adr[bj]

            for si in range(ns_i):
                s_idx_i = adr_i + si
                st_i = flat_shape_type[s_idx_i]
                pos_si = _compose_shape_world_pos(
                    X_world_R, X_world_r, flat_shape_offset, env_id, bi, s_idx_i
                )
                R_si = _compose_shape_world(
                    X_world_R,
                    X_world_r,
                    flat_shape_offset,
                    flat_shape_rotation,
                    env_id,
                    bi,
                    s_idx_i,
                )

                for sj in range(ns_j):
                    s_idx_j = adr_j + sj
                    st_j = flat_shape_type[s_idx_j]
                    pos_sj = _compose_shape_world_pos(
                        X_world_R, X_world_r, flat_shape_offset, env_id, bj, s_idx_j
                    )
                    R_sj = _compose_shape_world(
                        X_world_R,
                        X_world_r,
                        flat_shape_offset,
                        flat_shape_rotation,
                        env_id,
                        bj,
                        s_idx_j,
                    )

                    # Canonicalize: a_t <= b_t
                    swap = int(0)
                    a_t = st_i
                    b_t = st_j
                    a_pos = pos_si
                    b_pos = pos_sj
                    a_R = R_si
                    b_R = R_sj
                    a_idx = s_idx_i
                    b_idx = s_idx_j
                    a_br = ri
                    b_br = rj
                    if st_i > st_j:
                        swap = 1
                        a_t = st_j
                        b_t = st_i
                        a_pos = pos_sj
                        b_pos = pos_si
                        a_R = R_sj
                        b_R = R_si
                        a_idx = s_idx_j
                        b_idx = s_idx_i
                        a_br = rj
                        b_br = ri

                    # Inline narrowphase dispatch (a_t <= b_t)
                    n_depth = float(0.0)
                    n_hit = float(0.0)
                    n_normal = wp.vec3(0.0, 0.0, 1.0)
                    n_cp = wp.vec3(0.0, 0.0, 0.0)

                    if a_t == SHAPE_SPHERE and b_t == SHAPE_SPHERE:
                        r_a = flat_shape_params[a_idx, 0]
                        r_b = flat_shape_params[b_idx, 0]
                        n_res = sphere_sphere(a_pos, r_a, b_pos, r_b)
                        n_depth = n_res[0]
                        n_hit = n_res[1]
                        n_normal = sphere_sphere_normal(a_pos, b_pos)
                        n_cp = b_pos + n_normal * r_b
                    elif a_t == SHAPE_SPHERE and b_t == SHAPE_BOX:
                        r_a = flat_shape_params[a_idx, 0]
                        n_hx = flat_shape_params[b_idx, 0]
                        n_hy = flat_shape_params[b_idx, 1]
                        n_hz = flat_shape_params[b_idx, 2]
                        n_res = sphere_box(a_pos, r_a, b_pos, b_R, n_hx, n_hy, n_hz)
                        n_depth = n_res[0]
                        n_hit = n_res[1]
                        n_normal = sphere_box_normal(a_pos, r_a, b_pos, b_R, n_hx, n_hy, n_hz)
                        n_cp = a_pos - n_normal * r_a
                    elif a_t == SHAPE_SPHERE and b_t == SHAPE_CAPSULE:
                        r_a = flat_shape_params[a_idx, 0]
                        r_b = flat_shape_params[b_idx, 0]
                        n_hl_b = flat_shape_params[b_idx, 1]
                        n_res = sphere_capsule(a_pos, r_a, b_pos, b_R, r_b, n_hl_b)
                        n_depth = n_res[0]
                        n_hit = n_res[1]
                        n_normal = sphere_capsule_normal_point(a_pos, r_a, b_pos, b_R, r_b, n_hl_b)
                        n_cp = a_pos - n_normal * r_a
                    elif a_t == SHAPE_CAPSULE and b_t == SHAPE_CAPSULE:
                        r_a = flat_shape_params[a_idx, 0]
                        hl_a = flat_shape_params[a_idx, 1]
                        r_b = flat_shape_params[b_idx, 0]
                        hl_b = flat_shape_params[b_idx, 1]
                        n_res = capsule_capsule(a_pos, a_R, r_a, hl_a, b_pos, b_R, r_b, hl_b)
                        n_depth = n_res[0]
                        n_hit = n_res[1]
                        n_axis_a = a_R * wp.vec3(0.0, 0.0, 1.0)
                        n_axis_b = b_R * wp.vec3(0.0, 0.0, 1.0)
                        n_seg_a = a_pos - hl_a * n_axis_a
                        n_seg_b = b_pos - hl_b * n_axis_b
                        n_pt_a = closest_points_segment_segment(
                            n_seg_a, n_axis_a, 2.0 * hl_a, n_seg_b, n_axis_b, 2.0 * hl_b
                        )
                        n_pt_b = closest_points_seg_seg_both(
                            n_seg_a, n_axis_a, 2.0 * hl_a, n_seg_b, n_axis_b, 2.0 * hl_b
                        )
                        n_diff = n_pt_a - n_pt_b
                        n_dist = wp.length(n_diff)
                        if n_dist > 1.0e-10:
                            n_normal = n_diff / n_dist
                        n_cp = n_pt_b + n_normal * r_b
                    else:
                        # Fallback: sphere-sphere with body radius
                        r_a = a_br
                        r_b = b_br
                        if r_a > 0.0 and r_b > 0.0:
                            n_res = sphere_sphere(a_pos, r_a, b_pos, r_b)
                            n_depth = n_res[0]
                            n_hit = n_res[1]
                            n_normal = sphere_sphere_normal(a_pos, b_pos)
                            n_cp = b_pos + n_normal * r_b

                    if swap == 1:
                        n_normal = wp.vec3(-n_normal[0], -n_normal[1], -n_normal[2])

                    if n_hit > 0.5:
                        slot = wp.atomic_add(contact_count, env_id, 1)
                        if slot < max_contacts:
                            _write_contact(
                                env_id,
                                slot,
                                n_depth,
                                n_normal,
                                n_cp,
                                bi,
                                bj,
                                contact_depth,
                                contact_normal,
                                contact_point,
                                contact_bi,
                                contact_bj,
                                contact_active,
                            )
