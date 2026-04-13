"""Mass splitting kernels for Jacobi PGS stability (Tonge et al., SIGGRAPH 2012).

When N contacts act on the same body, pure Jacobi applies N× the correct
impulse (each contact independently computes the full correction).  Mass
splitting divides the effective mass by N so each contact contributes 1/N.

In the Delassus framework this is equivalent to scaling W_diag:
    W_diag[contact_i] *= max(N_body_i, N_body_j)

Two kernels launched once per step, after W is built and before PGS iterations.
"""

import warp as wp

CONDIM = wp.constant(3)


@wp.kernel
def batched_count_contacts_per_body(
    contact_active: wp.array2d(dtype=wp.int32),
    contact_bi: wp.array2d(dtype=wp.int32),
    contact_bj: wp.array2d(dtype=wp.int32),
    max_contacts: int,
    nb: int,
    # Output
    n_contacts_per_body: wp.array2d(dtype=wp.int32),  # (N, nb)
):
    """Count active contacts per body (ground contacts count on body_i only)."""
    env = wp.tid()

    for b in range(nb):
        n_contacts_per_body[env, b] = 0

    for c in range(max_contacts):
        if contact_active[env, c] == 0:
            continue
        bi = contact_bi[env, c]
        bj = contact_bj[env, c]
        if bi >= 0:
            n_contacts_per_body[env, bi] = n_contacts_per_body[env, bi] + 1
        if bj >= 0:
            n_contacts_per_body[env, bj] = n_contacts_per_body[env, bj] + 1


@wp.kernel
def batched_apply_mass_splitting(
    contact_active: wp.array2d(dtype=wp.int32),
    contact_bi: wp.array2d(dtype=wp.int32),
    contact_bj: wp.array2d(dtype=wp.int32),
    n_contacts_per_body: wp.array2d(dtype=wp.int32),
    max_contacts: int,
    # In/out
    W_diag: wp.array2d(dtype=wp.float32),
):
    """Scale W_diag by contact count to prevent Jacobi overshoot.

    For contact between body_i and body_j (or ground if bj=-1),
    the scale factor is max(N_i, N_j) where N is the contact count
    on each body.  This makes the Jacobi iteration conservative:
    each contact applies 1/N of the impulse it would have applied alone.
    """
    env = wp.tid()

    for c in range(max_contacts):
        if contact_active[env, c] == 0:
            continue
        bi = contact_bi[env, c]
        bj = contact_bj[env, c]

        n_bi = 1
        if bi >= 0:
            n_bi = n_contacts_per_body[env, bi]
        n_bj = 1
        if bj >= 0:
            n_bj = n_contacts_per_body[env, bj]

        scale = float(n_bi)
        if float(n_bj) > scale:
            scale = float(n_bj)

        if scale > 1.0:
            base = c * CONDIM
            for off in range(CONDIM):
                W_diag[env, base + off] = W_diag[env, base + off] * scale
