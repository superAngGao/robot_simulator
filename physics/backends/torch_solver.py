"""
Shared PyTorch Jacobi-PGS-SI solver for GPU backends (TileLang + CUDA).

All operations are batched PyTorch tensor ops on CUDA. The solver receives
backend state as arguments — no dependency on specific backend class.

Used by: TileLangBatchBackend._step_jacobi_pgs_si()
         CudaBatchBackend._step_jacobi_pgs_si()
"""

from __future__ import annotations

import torch

from .static_data import JOINT_FREE, JOINT_PRISMATIC, JOINT_REVOLUTE


def jacobi_pgs_si_step(
    q: torch.Tensor,
    qdot: torch.Tensor,
    tau_smooth: torch.Tensor,
    X_world_R: torch.Tensor,
    X_world_r: torch.Tensor,
    X_up_R: torch.Tensor,
    X_up_r: torch.Tensor,
    v_bodies_pred: torch.Tensor,
    contact_depth: torch.Tensor,
    contact_active: torch.Tensor,
    contact_point_world: torch.Tensor,
    *,
    contact_body_idx: torch.Tensor,
    contact_local_pos: torch.Tensor,
    inv_mass: torch.Tensor,
    inv_inertia: torch.Tensor,
    joint_type: torch.Tensor,
    joint_axis: torch.Tensor,
    parent_idx: torch.Tensor,
    v_idx_start: torch.Tensor,
    q_idx_start: torch.Tensor,
    mu: float,
    cfm: float,
    erp: float,
    slop: float,
    omega: float,
    max_iter: int,
    nc: int,
    nb: int,
    nv: int,
    dt: float,
    device: str,
    transform_force_fn,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run Jacobi-PGS-SI solver on predicted velocities.

    Returns:
        gen_impulse : (N, nv) generalized impulse from contact
        pos_corr    : (N, nb, 3) position corrections (split impulse)
        contact_mask: (N, nc) bool active contacts
    """
    N = q.shape[0]
    max_rows = nc * 3

    normal = torch.tensor([0.0, 0.0, 1.0], device=device)
    t1_dir = torch.tensor([1.0, 0.0, 0.0], device=device)
    t2_dir = torch.tensor([0.0, 1.0, 0.0], device=device)
    directions = torch.stack([normal, t1_dir, t2_dir])

    # ── Build Jacobian + W + v_free ──
    J_body = torch.zeros(N, max_rows, 6, device=device)
    row_body = torch.full((nc,), -1, dtype=torch.long, device=device)
    v_free = torch.zeros(N, max_rows, device=device)

    for c in range(nc):
        bi = int(contact_body_idx[c])
        row_body[c] = bi
        base = c * 3
        R = X_world_R[:, bi]
        Rt = R.transpose(-1, -2)
        r_body = X_world_r[:, bi]
        r_arm = contact_point_world[:, c] - r_body

        v_body = v_bodies_pred[:, bi]
        v_lin_w = torch.bmm(R, v_body[:, :3].unsqueeze(-1)).squeeze(-1)
        omega_w = torch.bmm(R, v_body[:, 3:].unsqueeze(-1)).squeeze(-1)
        v_contact = v_lin_w + torch.linalg.cross(omega_w, r_arm)

        for d in range(3):
            row = base + d
            direction = directions[d]
            rxd = torch.linalg.cross(r_arm, direction.unsqueeze(0).expand(N, -1))
            J_lin = torch.bmm(Rt, direction.unsqueeze(0).expand(N, -1).unsqueeze(-1)).squeeze(-1)
            J_ang = torch.bmm(Rt, rxd.unsqueeze(-1)).squeeze(-1)
            J_body[:, row, :3] = J_lin
            J_body[:, row, 3:] = J_ang
            v_free[:, row] = (v_contact * direction.unsqueeze(0)).sum(-1)

    # ── Build W ──
    W = torch.zeros(N, max_rows, max_rows, device=device)
    for c in range(nc):
        bi = int(row_body[c])
        m_inv = inv_mass[bi]
        I_inv = inv_inertia[bi]
        base = c * 3
        rows = slice(base, base + 3)
        J_lin_c = J_body[:, rows, :3]
        J_ang_c = J_body[:, rows, 3:]
        Minv_lin = m_inv * J_lin_c
        Minv_ang = torch.matmul(J_ang_c, I_inv.unsqueeze(0).expand(N, -1, -1).transpose(-1, -2))
        for c2 in range(nc):
            if int(row_body[c2]) != bi:
                continue
            base2 = c2 * 3
            rows2 = slice(base2, base2 + 3)
            J2_lin = J_body[:, rows2, :3]
            J2_ang = J_body[:, rows2, 3:]
            block = torch.bmm(J2_lin, Minv_lin.transpose(-1, -2)) + torch.bmm(
                J2_ang, Minv_ang.transpose(-1, -2)
            )
            W[:, rows, rows2] += block

    W_diag = W.diagonal(dim1=-2, dim2=-1).clone() + cfm

    # ── Jacobi PGS iterations (erp=0) ──
    lambdas = torch.zeros(N, max_rows, device=device)
    for _ in range(max_iter):
        lambdas_old = lambdas.clone()
        Wl = torch.bmm(W, lambdas_old.unsqueeze(-1)).squeeze(-1)
        for c in range(nc):
            active = contact_active[:, c]
            base = c * 3
            residual_n = v_free[:, base] + Wl[:, base]
            delta_n = torch.where(
                W_diag[:, base] > 1e-12,
                -residual_n / W_diag[:, base],
                torch.zeros_like(residual_n),
            )
            raw_n = lambdas_old[:, base] + omega * delta_n
            lambda_n = torch.where(active, torch.clamp(raw_n, min=0.0), lambdas[:, base])
            lambdas[:, base] = lambda_n
            limit = mu * lambda_n
            for off in range(1, 3):
                row = base + off
                residual_t = v_free[:, row] + Wl[:, row]
                delta_t = torch.where(
                    W_diag[:, row] > 1e-12,
                    -residual_t / W_diag[:, row],
                    torch.zeros_like(residual_t),
                )
                raw_t = lambdas_old[:, row] + omega * delta_t
                lambda_t = torch.where(active, torch.clamp(raw_t, -limit, limit), lambdas[:, row])
                lambdas[:, row] = lambda_t

    # ── Convert lambdas → body impulses → generalized impulse ──
    body_impulses = torch.zeros(N, nb, 6, device=device)
    for c in range(nc):
        bi = int(contact_body_idx[c])
        base = c * 3
        l_n = lambdas[:, base]
        l_t1 = lambdas[:, base + 1]
        l_t2 = lambdas[:, base + 2]
        F_world = l_n.unsqueeze(-1) * normal + l_t1.unsqueeze(-1) * t1_dir + l_t2.unsqueeze(-1) * t2_dir
        R = X_world_R[:, bi]
        r_body = X_world_r[:, bi]
        r_arm = contact_point_world[:, c] - r_body
        torque_world = torch.linalg.cross(r_arm, F_world)
        Rinv = R.transpose(-1, -2)
        f_lin_body = torch.bmm(Rinv, F_world.unsqueeze(-1)).squeeze(-1)
        f_ang_body = torch.bmm(Rinv, torque_world.unsqueeze(-1)).squeeze(-1)
        body_impulses[:, bi] += torch.cat([f_lin_body, f_ang_body], dim=-1)

    # RNEA backward → generalized impulse
    gen_impulse = torch.zeros(N, nv, device=device)
    f_prop = body_impulses.clone()
    for idx in range(nb):
        i = nb - 1 - idx
        f_i = f_prop[:, i]
        jtype = int(joint_type[i])
        vs = int(v_idx_start[i])
        if jtype == JOINT_REVOLUTE or jtype == JOINT_PRISMATIC:
            axis = joint_axis[i]
            if jtype == JOINT_REVOLUTE:
                gen_impulse[:, vs] = (f_i[:, 3:] * axis.unsqueeze(0)).sum(-1)
            else:
                gen_impulse[:, vs] = (f_i[:, :3] * axis.unsqueeze(0)).sum(-1)
        elif jtype == JOINT_FREE:
            gen_impulse[:, vs : vs + 6] = f_i
        pi = int(parent_idx[i])
        if pi >= 0:
            f_parent = transform_force_fn(X_up_R[:, i], X_up_r[:, i], f_i)
            f_prop[:, pi] += f_parent

    # ── Position correction ──
    pos_corr = torch.zeros(N, nb, 3, device=device)
    for c in range(nc):
        bi = int(contact_body_idx[c])
        eff_depth = contact_depth[:, c] - slop
        correction = erp * torch.clamp(eff_depth, min=0.0)
        pos_corr[:, bi] += correction.unsqueeze(-1) * normal.unsqueeze(0)

    return gen_impulse, pos_corr, contact_active
