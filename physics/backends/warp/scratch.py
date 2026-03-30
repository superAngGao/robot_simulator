"""
Pre-allocated GPU scratch buffers for batched ABA computation.

Warp kernels cannot allocate per-thread dynamic arrays, so all per-body
per-environment working memory is pre-allocated here as 2D/3D wp.arrays.
Each thread indexes its slice via env_id = wp.tid().
"""

from __future__ import annotations

import warp as wp


class ABABatchScratch:
    """Pre-allocated GPU buffers for batched ABA + FK.

    Args:
        N  : Number of environments.
        nb : Number of bodies.
        nq : Total generalised position dimension.
        nv : Total generalised velocity dimension.
        nc : Number of contact points.
        device : Warp device string.
    """

    def __init__(self, N: int, nb: int, nq: int, nv: int, nc: int, device: str = "cuda:0") -> None:
        self.N = N
        self.nb = nb
        self.nq = nq
        self.nv = nv
        self.nc = nc
        self.device = device

        # -- Dynamic state --
        self.q = wp.zeros((N, nq), dtype=wp.float32, device=device)
        self.qdot = wp.zeros((N, nv), dtype=wp.float32, device=device)

        # -- FK outputs --
        self.X_world_R = wp.zeros((N, nb, 3, 3), dtype=wp.float32, device=device)
        self.X_world_r = wp.zeros((N, nb, 3), dtype=wp.float32, device=device)
        self.v_bodies = wp.zeros((N, nb, 6), dtype=wp.float32, device=device)

        # -- X_up per body (from FK, reused in ABA) --
        self.X_up_R = wp.zeros((N, nb, 3, 3), dtype=wp.float32, device=device)
        self.X_up_r = wp.zeros((N, nb, 3), dtype=wp.float32, device=device)

        # -- ABA scratch --
        self.aba_v = wp.zeros((N, nb, 6), dtype=wp.float32, device=device)
        self.aba_c = wp.zeros((N, nb, 6), dtype=wp.float32, device=device)
        self.aba_IA = wp.zeros((N, nb, 6, 6), dtype=wp.float32, device=device)
        self.aba_pA = wp.zeros((N, nb, 6), dtype=wp.float32, device=device)
        self.aba_a = wp.zeros((N, nb, 6), dtype=wp.float32, device=device)
        self.aba_U = wp.zeros((N, nb, 6, 6), dtype=wp.float32, device=device)
        self.aba_Dinv = wp.zeros((N, nb, 6, 6), dtype=wp.float32, device=device)
        self.aba_u = wp.zeros((N, nb, 6), dtype=wp.float32, device=device)

        # -- Torques --
        self.tau_passive = wp.zeros((N, nv), dtype=wp.float32, device=device)
        self.tau_action = wp.zeros((N, nv), dtype=wp.float32, device=device)
        self.tau_total = wp.zeros((N, nv), dtype=wp.float32, device=device)
        self.qddot = wp.zeros((N, nv), dtype=wp.float32, device=device)

        # -- External forces --
        self.ext_forces = wp.zeros((N, nb, 6), dtype=wp.float32, device=device)

        # -- Contact mask --
        self.contact_mask = wp.zeros((N, nc), dtype=wp.int32, device=device)

        # -- New state (after integration) --
        self.q_new = wp.zeros((N, nq), dtype=wp.float32, device=device)
        self.qdot_new = wp.zeros((N, nv), dtype=wp.float32, device=device)

        # -- CRBA scratch (Q29 joint-space Delassus pipeline) --
        self.IC = wp.zeros((N, nb, 6, 6), dtype=wp.float32, device=device)
        self.rnea_v = wp.zeros((N, nb, 6), dtype=wp.float32, device=device)
        self.rnea_a = wp.zeros((N, nb, 6), dtype=wp.float32, device=device)
        self.rnea_f = wp.zeros((N, nb, 6), dtype=wp.float32, device=device)
