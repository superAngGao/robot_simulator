"""
Pre-allocated GPU scratch buffers for the Jacobi-PGS-SI constraint solver.

Extends ABABatchScratch with solver-specific arrays. Only allocated when
the Warp backend is configured with contact_solver="jacobi_pgs_si".
"""

from __future__ import annotations

import warp as wp


class SolverScratch:
    """Pre-allocated GPU buffers for batched Jacobi-PGS-SI solver.

    Args:
        N        : Number of environments.
        nb       : Number of bodies.
        nq       : Total generalized position dimension.
        nv       : Total generalized velocity dimension.
        nc       : Number of contact points.
        max_rows : Maximum constraint rows (nc * condim, typically nc * 3).
        device   : Warp device string.
    """

    def __init__(
        self,
        N: int,
        nb: int,
        nq: int,
        nv: int,
        nc: int,
        max_rows: int,
        device: str = "cuda:0",
    ) -> None:
        self.N = N
        self.nb = nb
        self.nq = nq
        self.nv = nv
        self.nc = nc
        self.max_rows = max_rows
        self.device = device

        # -- Contact detection outputs --
        self.contact_depth = wp.zeros((N, nc), dtype=wp.float32, device=device)
        self.contact_point_world = wp.zeros((N, nc, 3), dtype=wp.float32, device=device)
        self.contact_active = wp.zeros((N, nc), dtype=wp.int32, device=device)

        # -- Delassus matrix and solver state --
        self.W = wp.zeros((N, max_rows, max_rows), dtype=wp.float32, device=device)
        self.W_diag = wp.zeros((N, max_rows), dtype=wp.float32, device=device)
        self.v_free = wp.zeros((N, max_rows), dtype=wp.float32, device=device)
        self.lambdas = wp.zeros((N, max_rows), dtype=wp.float32, device=device)
        self.lambdas_old = wp.zeros((N, max_rows), dtype=wp.float32, device=device)

        # -- Predicted velocity --
        self.v_predicted = wp.zeros((N, nv), dtype=wp.float32, device=device)
        self.v_bodies_pred = wp.zeros((N, nb, 6), dtype=wp.float32, device=device)

        # -- Body impulses and generalized impulse --
        self.body_impulses = wp.zeros((N, nb, 6), dtype=wp.float32, device=device)
        self.gen_impulse = wp.zeros((N, nv), dtype=wp.float32, device=device)
        self.dqdot = wp.zeros((N, nv), dtype=wp.float32, device=device)

        # -- Position corrections --
        self.pos_corrections = wp.zeros((N, nb, 3), dtype=wp.float32, device=device)

        # -- ABA trick temporaries (always zero) --
        self.qdot_zero = wp.zeros((N, nv), dtype=wp.float32, device=device)
        self.ext_forces_zero = wp.zeros((N, nb, 6), dtype=wp.float32, device=device)

        # -- Jacobian rows (per contact, 6D body-frame) --
        # J_body[env, row, :] = 6D Jacobian row in body frame
        self.J_body = wp.zeros((N, max_rows, 6), dtype=wp.float32, device=device)
        # Which body each row belongs to
        self.row_body_idx = wp.zeros((N, max_rows), dtype=wp.int32, device=device)
