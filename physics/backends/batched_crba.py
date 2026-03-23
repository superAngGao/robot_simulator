"""
Batched CRBA — GPU-accelerated mass matrix computation + Cholesky solve.

Computes forward dynamics for N parallel environments using:
  1. Batched composite inertia propagation (sequential over bodies, parallel over N)
  2. Batched mass matrix H assembly (nv x nv per env)
  3. Batched bias force C via batched RNEA (sequential over bodies)
  4. torch.linalg.cholesky_solve for qddot = H^{-1}(tau - C)

Step 4 uses cuSOLVER which leverages tensor cores on Hopper (H100/H200).

Usage from any backend:
    crba = BatchedCRBA(static_data, device="cuda:0")
    qddot = crba.forward_dynamics(q, qdot, tau_total, ext_forces)
"""

from __future__ import annotations

import torch
import numpy as np

from .static_data import StaticRobotData, JOINT_FREE, JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_FIXED


class BatchedCRBA:
    """Batched CRBA forward dynamics on GPU.

    Args:
        static : StaticRobotData (robot constants).
        device : torch device.
    """

    def __init__(self, static: StaticRobotData, device: str = "cuda:0") -> None:
        self._s = static
        self._device = device
        nb = static.nb

        # Pre-upload static inertia matrices (nb, 6, 6)
        self._I_body = torch.from_numpy(static.inertia_mat).to(device)  # (nb, 6, 6)

        # Static tree data
        self._joint_type = torch.from_numpy(static.joint_type).int().to(device)
        self._joint_axis = torch.from_numpy(static.joint_axis).to(device)
        self._parent_idx = torch.from_numpy(static.parent_idx).int().to(device)
        self._q_idx_start = static.q_idx_start  # keep as numpy for Python indexing
        self._q_idx_len = static.q_idx_len
        self._v_idx_start = static.v_idx_start
        self._v_idx_len = static.v_idx_len
        self._X_tree_R = torch.from_numpy(static.X_tree_R).to(device)
        self._X_tree_r = torch.from_numpy(static.X_tree_r).to(device)
        self._gravity = static.gravity

    def forward_dynamics(
        self,
        q: torch.Tensor,        # (N, nq)
        qdot: torch.Tensor,     # (N, nv)
        tau: torch.Tensor,      # (N, nv)
        ext_forces: torch.Tensor,  # (N, nb, 6)
        X_up_R: torch.Tensor,   # (N, nb, 3, 3) — from FK
        X_up_r: torch.Tensor,   # (N, nb, 3) — from FK
    ) -> torch.Tensor:
        """Compute qddot = H^{-1}(tau - C) for N environments.

        Returns: qddot (N, nv).
        """
        s = self._s
        N = q.shape[0]
        nb, nv = s.nb, s.nv
        dev = self._device

        # --- 1. Batched composite inertias (backward pass) ---
        IC = self._I_body.unsqueeze(0).expand(N, -1, -1, -1).clone()  # (N, nb, 6, 6)

        for idx in range(nb - 1, -1, -1):
            pid = int(self._parent_idx[idx])
            if pid >= 0:
                # IC[parent] += X^T @ IC[child] @ X
                X6 = self._build_X6_batched(X_up_R[:, idx], X_up_r[:, idx])  # (N, 6, 6)
                X6T = X6.transpose(-1, -2)
                contrib = X6T @ IC[:, idx] @ X6  # (N, 6, 6)
                IC[:, pid] = IC[:, pid] + contrib

        # --- 2. Build H (nv x nv) ---
        H = torch.zeros(N, nv, nv, device=dev)

        # Pre-compute motion subspace columns for each body
        S_cols = []  # list of (nv_i, 6) per body, or None
        for i in range(nb):
            jt = int(self._joint_type[i])
            vl = int(self._v_idx_len[i])
            if vl == 0:
                S_cols.append(None)
            elif jt == JOINT_REVOLUTE:
                S = torch.zeros(6, device=dev)
                axis = self._joint_axis[i]
                S[3:] = axis
                S_cols.append(S.unsqueeze(1))  # (6, 1)
            elif jt == JOINT_PRISMATIC:
                S = torch.zeros(6, device=dev)
                axis = self._joint_axis[i]
                S[:3] = axis
                S_cols.append(S.unsqueeze(1))  # (6, 1)
            elif jt == JOINT_FREE:
                S_cols.append(torch.eye(6, device=dev))  # (6, 6)
            else:
                S_cols.append(None)

        for i in range(nb):
            if S_cols[i] is None:
                continue
            S_i = S_cols[i]  # (6, nv_i)
            vs_i = int(self._v_idx_start[i])
            vl_i = int(self._v_idx_len[i])

            # F = IC[i] @ S_i  — (N, 6, nv_i)
            F = IC[:, i] @ S_i.unsqueeze(0).expand(N, -1, -1)  # (N, 6, nv_i)

            # Diagonal: H[vi, vi] = S_i^T @ F
            diag_block = (S_i.unsqueeze(0).expand(N, -1, -1).transpose(-1, -2) @ F)  # (N, nv_i, nv_i)
            H[:, vs_i:vs_i+vl_i, vs_i:vs_i+vl_i] = diag_block

            # Off-diagonal: propagate F up tree
            j = i
            while int(self._parent_idx[j]) >= 0:
                # F = X_up[j].apply_force(F)  — transform each column
                R_up = X_up_R[:, j]  # (N, 3, 3)
                r_up = X_up_r[:, j]  # (N, 3)
                F = self._apply_force_batched_multi(R_up, r_up, F)
                j = int(self._parent_idx[j])

                if S_cols[j] is not None:
                    S_j = S_cols[j]
                    vs_j = int(self._v_idx_start[j])
                    vl_j = int(self._v_idx_len[j])
                    # block = S_j^T @ F  — (N, nv_j, nv_i)
                    block = S_j.unsqueeze(0).expand(N, -1, -1).transpose(-1, -2) @ F
                    H[:, vs_j:vs_j+vl_j, vs_i:vs_i+vl_i] = block
                    H[:, vs_i:vs_i+vl_i, vs_j:vs_j+vl_j] = block.transpose(-1, -2)

        # --- 3. Batched bias forces C = RNEA(q, qdot, 0) ---
        C = self._batched_rnea(q, qdot, X_up_R, X_up_r, ext_forces)  # (N, nv)

        # --- 4. Cholesky solve: qddot = H^{-1}(tau - C) ---
        rhs = (tau - C).unsqueeze(-1)  # (N, nv, 1)
        L = torch.linalg.cholesky(H)
        qddot = torch.cholesky_solve(rhs, L).squeeze(-1)  # (N, nv)

        return qddot

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_X6_batched(self, R: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Build batched 6x6 Plücker velocity transform. (N,3,3), (N,3) -> (N,6,6)."""
        N = R.shape[0]
        E = R.transpose(-1, -2)
        zero = torch.zeros(N, device=R.device, dtype=R.dtype)
        skew_r = torch.stack([
            zero, -r[:, 2], r[:, 1],
            r[:, 2], zero, -r[:, 0],
            -r[:, 1], r[:, 0], zero,
        ], dim=-1).reshape(N, 3, 3)
        M = torch.zeros(N, 6, 6, device=R.device, dtype=R.dtype)
        M[:, :3, :3] = E
        M[:, :3, 3:] = -(E @ skew_r)
        M[:, 3:, 3:] = E
        return M

    def _apply_force_batched_multi(
        self, R: torch.Tensor, r: torch.Tensor, F: torch.Tensor
    ) -> torch.Tensor:
        """Apply Plücker force transform to multiple columns. F: (N, 6, k) -> (N, 6, k)."""
        # f_lin = F[:3], f_ang = F[3:]
        f_lin = F[:, :3, :]  # (N, 3, k)
        f_ang = F[:, 3:, :]  # (N, 3, k)
        Rf = R @ f_lin  # (N, 3, k)
        Rt = R @ f_ang  # (N, 3, k)
        # r × Rf for each column
        # cross product: r (N,3) x Rf (N,3,k)
        r_exp = r.unsqueeze(-1).expand_as(Rf)  # (N, 3, k)
        rxRf = torch.cross(r_exp, Rf, dim=1)  # (N, 3, k)
        return torch.cat([Rf, Rt + rxRf], dim=1)  # (N, 6, k)

    def _batched_rnea(
        self,
        q: torch.Tensor,
        qdot: torch.Tensor,
        X_up_R: torch.Tensor,
        X_up_r: torch.Tensor,
        ext_forces: torch.Tensor,
    ) -> torch.Tensor:
        """Batched RNEA with qddot=0 to compute bias forces C. Returns (N, nv)."""
        s = self._s
        N = q.shape[0]
        nb, nv = s.nb, s.nv
        dev = self._device

        a_gravity = torch.tensor([0, 0, -s.gravity, 0, 0, 0], device=dev, dtype=torch.float32)
        neg_grav = -a_gravity  # [0, 0, g, 0, 0, 0]

        v = torch.zeros(N, nb, 6, device=dev)
        a = torch.zeros(N, nb, 6, device=dev)
        f = torch.zeros(N, nb, 6, device=dev)

        # Forward pass
        for i in range(nb):
            jt = int(self._joint_type[i])
            vs = int(self._v_idx_start[i])
            vl = int(self._v_idx_len[i])
            pid = int(self._parent_idx[i])

            R_up = X_up_R[:, i]  # (N, 3, 3)
            r_up = X_up_r[:, i]  # (N, 3)

            # vJ
            vJ = torch.zeros(N, 6, device=dev)
            if jt == JOINT_REVOLUTE:
                axis = self._joint_axis[i]
                qd = qdot[:, vs:vs+1]
                vJ[:, 3:] = axis.unsqueeze(0) * qd
            elif jt == JOINT_PRISMATIC:
                axis = self._joint_axis[i]
                qd = qdot[:, vs:vs+1]
                vJ[:, :3] = axis.unsqueeze(0) * qd
            elif jt == JOINT_FREE:
                vJ = qdot[:, vs:vs+6]

            if pid < 0:
                v[:, i] = vJ
                a[:, i] = self._transform_vel_batched(R_up, r_up, neg_grav.unsqueeze(0).expand(N, -1))
            else:
                v_xf = self._transform_vel_batched(R_up, r_up, v[:, pid])
                v[:, i] = v_xf + vJ
                a_xf = self._transform_vel_batched(R_up, r_up, a[:, pid])
                # c = v ×_vel vJ
                c = self._spatial_cross_vel(v[:, i], vJ)
                a[:, i] = a_xf + c

            # f = I*a + v ×* (I*v) - ext
            I_i = self._I_body[i].unsqueeze(0).expand(N, -1, -1)
            Iv = (I_i @ v[:, i].unsqueeze(-1)).squeeze(-1)
            Ia = (I_i @ a[:, i].unsqueeze(-1)).squeeze(-1)
            vxIv = self._spatial_cross_force(v[:, i], Iv)
            f[:, i] = Ia + vxIv - ext_forces[:, i]

        # Backward pass
        tau_out = torch.zeros(N, nv, device=dev)
        for i in range(nb - 1, -1, -1):
            jt = int(self._joint_type[i])
            vs = int(self._v_idx_start[i])
            vl = int(self._v_idx_len[i])
            pid = int(self._parent_idx[i])

            if vl > 0:
                if jt == JOINT_REVOLUTE:
                    axis = self._joint_axis[i]
                    S = torch.zeros(6, device=dev)
                    S[3:] = axis
                    tau_out[:, vs] = (f[:, i] * S.unsqueeze(0)).sum(dim=-1)
                elif jt == JOINT_PRISMATIC:
                    axis = self._joint_axis[i]
                    S = torch.zeros(6, device=dev)
                    S[:3] = axis
                    tau_out[:, vs] = (f[:, i] * S.unsqueeze(0)).sum(dim=-1)
                elif jt == JOINT_FREE:
                    tau_out[:, vs:vs+6] = f[:, i]

            if pid >= 0:
                R_up = X_up_R[:, i]
                r_up = X_up_r[:, i]
                f_prop = self._transform_force_batched(R_up, r_up, f[:, i])
                f[:, pid] = f[:, pid] + f_prop

        return tau_out

    def _transform_vel_batched(self, R, r, v):
        """Plücker velocity transform. (N,3,3), (N,3), (N,6) -> (N,6)."""
        E = R.transpose(-1, -2)
        v_lin, v_ang = v[:, :3], v[:, 3:]
        wxr = torch.cross(v_ang, r, dim=-1)
        lin_new = (E @ (v_lin + wxr).unsqueeze(-1)).squeeze(-1)
        ang_new = (E @ v_ang.unsqueeze(-1)).squeeze(-1)
        return torch.cat([lin_new, ang_new], dim=-1)

    def _transform_force_batched(self, R, r, f):
        """Plücker force transform. (N,3,3), (N,3), (N,6) -> (N,6)."""
        f_lin, f_ang = f[:, :3], f[:, 3:]
        Rf = (R @ f_lin.unsqueeze(-1)).squeeze(-1)
        Rt = (R @ f_ang.unsqueeze(-1)).squeeze(-1)
        rxRf = torch.cross(r, Rf, dim=-1)
        return torch.cat([Rf, Rt + rxRf], dim=-1)

    def _spatial_cross_vel(self, v, u):
        vl, va = v[:, :3], v[:, 3:]
        ul, ua = u[:, :3], u[:, 3:]
        res_lin = torch.cross(va, ul, dim=-1) + torch.cross(vl, ua, dim=-1)
        res_ang = torch.cross(va, ua, dim=-1)
        return torch.cat([res_lin, res_ang], dim=-1)

    def _spatial_cross_force(self, v, f):
        vl, va = v[:, :3], v[:, 3:]
        fl, fa = f[:, :3], f[:, 3:]
        res_lin = torch.cross(va, fl, dim=-1)
        res_ang = torch.cross(vl, fl, dim=-1) + torch.cross(va, fa, dim=-1)
        return torch.cat([res_lin, res_ang], dim=-1)
