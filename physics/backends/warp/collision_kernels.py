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
