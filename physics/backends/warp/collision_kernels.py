"""
Warp GPU kernels for collision detection (ground + body-body).

Replaces the separate ground-only detection with a unified collision
pipeline that handles both ground contacts and body-body sphere collisions.

All contacts produce: depth, normal, point, body_i, body_j, active flag.
The solver kernels then process these uniformly.
"""

import warp as wp


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
