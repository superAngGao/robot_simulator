"""Graph-colored Gauss-Seidel PGS kernels (PhysX / Bullet3 GPU approach).

Contacts are colored so that no two same-color contacts share a body.
Within each color, contacts are solved in parallel (safe — no shared bodies).
Between colors, results are applied sequentially (GS ordering).

This gives GS convergence properties with partial GPU parallelism.

Two kernels:
  1. batched_greedy_coloring — assigns colors after collision detection
  2. batched_colored_pgs_step — one PGS pass for a single color
"""

import warp as wp

CONDIM = wp.constant(3)
MAX_COLORS = wp.constant(16)


@wp.kernel
def batched_greedy_coloring(
    contact_active: wp.array2d(dtype=wp.int32),
    contact_bi: wp.array2d(dtype=wp.int32),
    contact_bj: wp.array2d(dtype=wp.int32),
    max_contacts: int,
    nb: int,
    # Output
    contact_color: wp.array2d(dtype=wp.int32),  # (N, max_contacts) — 0..15, -1 if inactive
    n_colors_out: wp.array(dtype=wp.int32),  # (N,) — max color + 1
):
    """Greedy graph coloring: contacts sharing a body get different colors.

    Algorithm (sequential per env, parallel across envs):
      For each active contact c in slot order:
        - Collect colors already used by contacts on body_i and body_j
          via a per-body bitmask array
        - Assign the smallest unused color (first zero bit)
        - Update bitmasks for body_i and body_j

    body_color_mask[b] is a 16-bit mask: bit k set means color k is used
    by some contact on body b.
    """
    env = wp.tid()

    max_color = int(0)

    # Zero outputs and per-body bitmask (reuse a local approach)
    # Since Warp has no local arrays, we use a flat scan approach:
    # For each contact, scan all previous contacts to find conflicts.
    # This is O(nc²) but nc is small (< 100 typically).

    for c in range(max_contacts):
        contact_color[env, c] = -1

    for c in range(max_contacts):
        if contact_active[env, c] == 0:
            continue

        bi_c = contact_bi[env, c]
        bj_c = contact_bj[env, c]

        # Collect colors used by earlier contacts on same bodies
        used_mask = int(0)
        for p in range(c):
            if contact_active[env, p] == 0:
                continue
            color_p = contact_color[env, p]
            if color_p < 0:
                continue
            bi_p = contact_bi[env, p]
            bj_p = contact_bj[env, p]

            # Check if contact p shares a body with contact c
            conflict = int(0)
            if bi_c >= 0 and (bi_c == bi_p or bi_c == bj_p):
                conflict = 1
            if bj_c >= 0 and (bj_c == bi_p or bj_c == bj_p):
                conflict = 1

            if conflict == 1:
                # Mark color_p as used
                if color_p < 16:
                    used_mask = used_mask | (1 << color_p)

        # Find first unused color (first zero bit in used_mask)
        chosen = int(0)
        for k in range(16):
            if (used_mask & (1 << k)) == 0:
                chosen = k
                break

        contact_color[env, c] = chosen
        if chosen + 1 > max_color:
            max_color = chosen + 1

    n_colors_out[env] = max_color


@wp.kernel
def batched_colored_pgs_step(
    W: wp.array(dtype=wp.float32, ndim=3),
    W_diag: wp.array2d(dtype=wp.float32),
    v_free: wp.array2d(dtype=wp.float32),
    lambdas: wp.array2d(dtype=wp.float32),  # read AND write (no double buffer)
    contact_active: wp.array2d(dtype=wp.int32),
    contact_color: wp.array2d(dtype=wp.int32),
    mu: float,
    current_color: int,
    nc: int,
    max_rows: int,
):
    """One GS pass for a single color.

    Only contacts with contact_color == current_color are updated.
    Same-color contacts don't share bodies → safe to read/write lambdas
    in parallel (no data race on the velocity correction).

    Reads latest lambdas (not a snapshot) — this is the GS property.
    """
    env_id = wp.tid()
    n_rows = nc * CONDIM

    for c in range(nc):
        if contact_active[env_id, c] == 0:
            continue
        if contact_color[env_id, c] != current_color:
            continue

        base = c * CONDIM

        # -- Normal row --
        row_n = base
        Wl_n = float(0.0)
        for j in range(n_rows):
            Wl_n = Wl_n + W[env_id, row_n, j] * lambdas[env_id, j]
        residual_n = v_free[env_id, row_n] + Wl_n
        diag_n = W_diag[env_id, row_n]
        if diag_n > 1.0e-12:
            delta_n = -residual_n / diag_n
        else:
            delta_n = 0.0
        raw_n = lambdas[env_id, row_n] + delta_n  # omega=1 for GS
        lambda_n = wp.max(0.0, raw_n)
        lambdas[env_id, row_n] = lambda_n

        # Friction limit
        limit = mu * lambda_n

        # -- Tangent rows --
        for off in range(1, CONDIM):
            row_t = base + off
            Wl_t = float(0.0)
            for j in range(n_rows):
                Wl_t = Wl_t + W[env_id, row_t, j] * lambdas[env_id, j]
            residual_t = v_free[env_id, row_t] + Wl_t
            diag_t = W_diag[env_id, row_t]
            if diag_t > 1.0e-12:
                delta_t = -residual_t / diag_t
            else:
                delta_t = 0.0
            raw_t = lambdas[env_id, row_t] + delta_t
            lambdas[env_id, row_t] = wp.clamp(raw_t, -limit, limit)
