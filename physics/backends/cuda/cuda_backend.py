"""
CudaBatchBackend — raw CUDA C++ kernel backend via torch.utils.cpp_extension.

All physics (FK, ABA, contact, passive torques, PD, integration) fused into
a single CUDA kernel launch per step. JIT-compiled on first use.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..batch_backend import BatchBackend, StepResult
from ..static_data import StaticRobotData

if TYPE_CHECKING:
    from rl_env.cfg import EnvCfg
    from robot.model import RobotModel

# JIT-compiled CUDA module (loaded once)
_cuda_module = None


def _get_cuda_module():
    global _cuda_module
    if _cuda_module is None:
        from torch.utils.cpp_extension import load

        cuda_src = os.path.join(os.path.dirname(__file__), "kernels.cu")
        _cuda_module = load(
            name="robot_sim_cuda",
            sources=[cuda_src],
            verbose=False,
        )
    return _cuda_module


class CudaBatchBackend(BatchBackend):
    """GPU backend using raw CUDA C++ kernels.

    Fuses all physics into a single kernel launch per step for minimal
    launch overhead. JIT-compiled via torch.utils.cpp_extension.
    """

    def __init__(
        self,
        model: "RobotModel",
        cfg: "EnvCfg",
        num_envs: int,
        device: str = "cuda:0",
        dynamics: str = "aba",  # "aba" or "crba"
    ) -> None:
        self._device = device
        self._cfg = cfg
        self._num_envs = num_envs
        self._dynamics = dynamics
        self._cuda = _get_cuda_module()

        # Group metadata for grouped Schur (computed lazily)
        self._group_data = None

        s = StaticRobotData.from_model(model)
        self._static = s

        # Upload static data
        self._joint_type = torch.from_numpy(s.joint_type).int().to(device)
        self._joint_axis = torch.from_numpy(s.joint_axis).to(device)
        self._parent_idx = torch.from_numpy(s.parent_idx).int().to(device)
        self._q_idx_start = torch.from_numpy(s.q_idx_start).int().to(device)
        self._q_idx_len = torch.from_numpy(s.q_idx_len).int().to(device)
        self._v_idx_start = torch.from_numpy(s.v_idx_start).int().to(device)
        self._v_idx_len = torch.from_numpy(s.v_idx_len).int().to(device)
        self._X_tree_R = torch.from_numpy(s.X_tree_R.reshape(s.nb, 9)).to(device)
        self._X_tree_r = torch.from_numpy(s.X_tree_r).to(device)
        self._inertia_mat = torch.from_numpy(s.inertia_mat.reshape(s.nb, 36)).to(device)
        self._q_min = torch.from_numpy(s.q_min).to(device)
        self._q_max = torch.from_numpy(s.q_max).to(device)
        self._k_limit = torch.from_numpy(s.k_limit).to(device)
        self._b_limit = torch.from_numpy(s.b_limit).to(device)
        self._damping = torch.from_numpy(s.damping).to(device)
        self._actuated_q_idx = torch.from_numpy(s.actuated_q_indices).int().to(device)
        self._actuated_v_idx = torch.from_numpy(s.actuated_v_indices).int().to(device)
        self._has_effort_limits = 0
        if s.effort_limits is not None:
            self._effort_limits = torch.from_numpy(s.effort_limits).to(device)
            self._has_effort_limits = 1
        else:
            self._effort_limits = torch.zeros(max(s.nu, 1), device=device)

        # Contact (body_idx as float for kernel simplicity)
        self._contact_body_idx = torch.from_numpy(s.contact_body_idx.astype(np.float32)).to(device)
        self._contact_local_pos = torch.from_numpy(s.contact_local_pos).to(device)

        # Constraint solver
        self._contact_solver = getattr(cfg, "contact_solver", "penalty")
        if self._contact_solver == "jacobi_pgs_si":
            self._contact_body_idx_long = torch.from_numpy(s.contact_body_idx).long().to(device)
            self._inv_mass = torch.from_numpy(s.inv_mass_per_body).to(device)
            self._inv_inertia = torch.from_numpy(s.inv_inertia_per_body).to(device)

        # State
        N = num_envs
        self._q = torch.zeros(N, s.nq, device=device)
        self._qdot = torch.zeros(N, s.nv, device=device)
        self._default_q = torch.from_numpy(s.default_q).to(device)
        self._default_qdot = torch.from_numpy(s.default_qdot).to(device)

    # ------------------------------------------------------------------
    # BatchBackend interface
    # ------------------------------------------------------------------

    def reset_all(self, init_noise_scale: float = 0.0) -> StepResult:
        N = self._num_envs
        self._q[:] = self._default_q.unsqueeze(0).expand(N, -1)
        self._qdot[:] = self._default_qdot.unsqueeze(0).expand(N, -1)
        if init_noise_scale > 0:
            self._q += torch.empty_like(self._q).uniform_(-init_noise_scale, init_noise_scale)
        return self._run_fk_and_build()

    def reset_envs(self, env_ids: torch.Tensor, init_noise_scale: float = 0.0) -> None:
        self._q[env_ids] = self._default_q.unsqueeze(0)
        self._qdot[env_ids] = self._default_qdot.unsqueeze(0)
        if init_noise_scale > 0:
            noise = torch.empty(len(env_ids), self._static.nq, device=self._device)
            noise.uniform_(-init_noise_scale, init_noise_scale)
            self._q[env_ids] += noise

    def step_batch(self, actions: torch.Tensor) -> StepResult:
        s = self._static
        N = self._num_envs
        actions = actions.to(device=self._device, dtype=torch.float32)
        action_clip = self._cfg.action_clip if self._cfg.action_clip is not None else -1.0

        if self._contact_solver == "jacobi_pgs_si":
            return self._step_jacobi_pgs_si(actions, action_clip)

        if self._dynamics == "grouped_schur":
            return self._step_grouped_schur(actions, action_clip)
        if self._dynamics == "crba_tc":
            return self._step_crba_tc(actions, action_clip)

        step_fn = self._cuda.physics_step_crba if self._dynamics == "crba" else self._cuda.physics_step
        results = step_fn(
            self._q,
            self._qdot,
            actions,
            self._joint_type,
            self._joint_axis,
            self._parent_idx,
            self._q_idx_start,
            self._q_idx_len,
            self._v_idx_start,
            self._v_idx_len,
            self._X_tree_R,
            self._X_tree_r,
            self._inertia_mat,
            self._q_min,
            self._q_max,
            self._k_limit,
            self._b_limit,
            self._damping,
            self._actuated_q_idx,
            self._actuated_v_idx,
            self._effort_limits,
            self._contact_body_idx,
            self._contact_local_pos,
            N,
            s.nb,
            s.nq,
            s.nv,
            s.nu,
            s.nc,
            self._has_effort_limits,
            self._cfg.dt,
            s.gravity,
            self._cfg.kp,
            self._cfg.kd,
            self._cfg.action_scale,
            action_clip,
            s.contact_k_normal,
            s.contact_b_normal,
            s.contact_mu,
            s.contact_slip_eps,
            s.contact_ground_z,
        )
        q_new, qdot_new, X_world_R, X_world_r, v_bodies, contact_mask = results

        self._q = q_new
        self._qdot = qdot_new

        # Re-run FK on new state for observation cache
        fk_results = self._cuda.fk_only(
            self._q,
            self._qdot,
            self._joint_type,
            self._joint_axis,
            self._parent_idx,
            self._q_idx_start,
            self._v_idx_start,
            self._v_idx_len,
            self._X_tree_R,
            self._X_tree_r,
            N,
            s.nb,
            s.nq,
            s.nv,
        )
        X_world_R, X_world_r, v_bodies = fk_results

        X_world_flat = torch.cat(
            [
                X_world_R.reshape(N, s.nb, 9),
                X_world_r,
            ],
            dim=-1,
        )

        return StepResult(
            q=self._q.cpu(),
            qdot=self._qdot.cpu(),
            X_world=X_world_flat.cpu(),
            v_bodies=v_bodies.cpu(),
            contact_mask=contact_mask.bool().cpu(),
        )

    # ------------------------------------------------------------------
    # Jacobi-PGS-SI constraint solver (PyTorch ops + CUDA FK)
    # ------------------------------------------------------------------

    def _step_jacobi_pgs_si(self, actions, action_clip):
        """Constraint solver path using PyTorch ops for solver, CUDA FK for transforms."""
        from ..tilelang.kernels_tl import transform_force_torch
        from ..torch_solver import jacobi_pgs_si_step

        s = self._static
        N = self._num_envs
        dt = self._cfg.dt
        nc, nb, nv = s.nc, s.nb, s.nv

        # 1-3. Passive torques + PD controller + tau_total
        #    Use the fused CUDA kernel for one step with penalty, capture tau_total
        #    Actually, we need tau_total only. Build it manually:
        # Passive torques (PyTorch, same as TileLang)
        tau_passive = torch.zeros(N, nv, device=self._device)
        for i in range(nb):
            jtype = int(self._joint_type[i])
            if jtype == 1:  # REVOLUTE
                vs = int(self._v_idx_start[i])
                qs = int(self._q_idx_start[i])
                q_val = self._q[:, qs]
                qdot_val = self._qdot[:, vs]
                qmin = float(self._q_min[i])
                qmax = float(self._q_max[i])
                k_lim = float(self._k_limit[i])
                b_lim = float(self._b_limit[i])
                damp = float(self._damping[i])
                # Joint limit penalty
                pen_lo = torch.clamp(qmin - q_val, min=0.0)
                pen_hi = torch.clamp(q_val - qmax, min=0.0)
                tau_lim = k_lim * pen_lo - b_lim * torch.clamp(qdot_val, max=0.0) * (pen_lo > 0).float()
                tau_lim = (
                    tau_lim - k_lim * pen_hi - b_lim * torch.clamp(qdot_val, min=0.0) * (pen_hi > 0).float()
                )
                tau_damp = -damp * qdot_val
                tau_passive[:, vs] = tau_lim + tau_damp

        # PD controller
        tau_action = torch.zeros(N, nv, device=self._device)
        for j in range(s.nu):
            qi = int(self._actuated_q_idx[j])
            vi = int(self._actuated_v_idx[j])
            target = self._q[:, qi] + self._cfg.action_scale * actions[:, j]
            tau_action[:, vi] = self._cfg.kp * (target - self._q[:, qi]) - self._cfg.kd * self._qdot[:, vi]

        tau_smooth = tau_action + tau_passive

        # 4. FK (using CUDA kernel)
        fk_results = self._cuda.fk_only(
            self._q,
            self._qdot,
            self._joint_type,
            self._joint_axis,
            self._parent_idx,
            self._q_idx_start,
            self._v_idx_start,
            self._v_idx_len,
            torch.from_numpy(s.X_tree_R.reshape(s.nb, 9).astype(np.float32)).to(self._device),
            torch.from_numpy(s.X_tree_r.astype(np.float32)).to(self._device),
            N,
            nb,
            s.nq,
            nv,
        )
        fk_Rw, fk_rw, fk_vb = fk_results[0], fk_results[1], fk_results[2]
        X_world_R = fk_Rw.reshape(N, nb, 3, 3)
        X_world_r = fk_rw.reshape(N, nb, 3)
        v_bodies = fk_vb.reshape(N, nb, 6)
        # X_up not returned by fk_only — use identity (valid for single free body)
        X_up_R = (
            torch.eye(3, device=self._device).unsqueeze(0).unsqueeze(0).expand(N, nb, -1, -1).contiguous()
        )
        X_up_r = torch.zeros(N, nb, 3, device=self._device)

        # 5. ABA (unconstrained) — PyTorch shortcut
        ext_zero = torch.zeros(N, nb, 6, device=self._device)
        a_u = self._compute_aba_pytorch(tau_smooth, ext_zero, X_up_R, X_up_r, v_bodies)

        # 6. Predicted velocity
        v_predicted = self._qdot + dt * a_u

        # 7. FK on predicted velocity → body_v_pred
        _, _, fk_vp = self._cuda.fk_only(
            self._q,
            v_predicted,
            self._joint_type,
            self._joint_axis,
            self._parent_idx,
            self._q_idx_start,
            self._v_idx_start,
            self._v_idx_len,
            torch.from_numpy(s.X_tree_R.reshape(s.nb, 9).astype(np.float32)).to(self._device),
            torch.from_numpy(s.X_tree_r.astype(np.float32)).to(self._device),
            N,
            nb,
            s.nq,
            nv,
        )
        v_bodies_pred = fk_vp.reshape(N, nb, 6)

        # 8. Ground contact detection
        contact_depth = torch.zeros(N, nc, device=self._device)
        contact_active = torch.zeros(N, nc, dtype=torch.bool, device=self._device)
        contact_point_world = torch.zeros(N, nc, 3, device=self._device)
        for c in range(nc):
            bi = int(self._contact_body_idx_long[c])
            R = X_world_R[:, bi]
            r = X_world_r[:, bi]
            local_pos = self._contact_local_pos[c]  # (3,)
            pos_world = (R @ local_pos.unsqueeze(-1)).squeeze(-1) + r  # (N, 3)
            depth = s.contact_ground_z - pos_world[:, 2]
            contact_depth[:, c] = depth
            contact_point_world[:, c] = pos_world
            contact_active[:, c] = depth > 0.0

        # 9-12. Solver + impulse conversion
        gen_impulse, pos_corr, _ = jacobi_pgs_si_step(
            self._q,
            self._qdot,
            tau_smooth,
            X_world_R,
            X_world_r,
            X_up_R,
            X_up_r,
            v_bodies_pred,
            contact_depth,
            contact_active,
            contact_point_world,
            contact_body_idx=self._contact_body_idx_long,
            contact_local_pos=self._contact_local_pos,
            inv_mass=self._inv_mass,
            inv_inertia=self._inv_inertia,
            joint_type=self._joint_type,
            joint_axis=self._joint_axis,
            parent_idx=self._parent_idx,
            v_idx_start=self._v_idx_start,
            q_idx_start=self._q_idx_start,
            mu=s.contact_mu,
            cfm=s.contact_cfm,
            erp=s.contact_erp_pos,
            slop=s.contact_slop,
            omega=s.solver_omega,
            max_iter=s.solver_max_iter,
            nc=nc,
            nb=nb,
            nv=nv,
            dt=dt,
            device=self._device,
            transform_force_fn=transform_force_torch,
        )

        # 13. ABA trick: dqdot = H⁻¹ @ gen_impulse
        #     ABA includes gravity, so subtract: dqdot = (ABA(tau=impulse/dt) - ABA(tau=0)) * dt
        v_zero = torch.zeros_like(v_bodies)
        tau_zero = torch.zeros(N, nv, device=self._device)
        a_with_impulse = self._compute_aba_pytorch(gen_impulse / dt, ext_zero, X_up_R, X_up_r, v_zero)
        a_gravity_only = self._compute_aba_pytorch(tau_zero, ext_zero, X_up_R, X_up_r, v_zero)
        dqdot = (a_with_impulse - a_gravity_only) * dt

        # 14. Integration
        self._qdot = v_predicted + dqdot
        q_new = self._q.clone()
        for i in range(nb):
            jtype = int(self._joint_type[i])
            qs = int(self._q_idx_start[i])
            vs = int(self._v_idx_start[i])
            if jtype == 0:  # FREE
                qw, qx, qy, qz = self._q[:, qs], self._q[:, qs + 1], self._q[:, qs + 2], self._q[:, qs + 3]
                wx, wy, wz = self._qdot[:, vs + 3], self._qdot[:, vs + 4], self._qdot[:, vs + 5]
                dqw = 0.5 * (-wx * qx - wy * qy - wz * qz)
                dqx = 0.5 * (wx * qw + wz * qy - wy * qz)
                dqy = 0.5 * (wy * qw - wz * qx + wx * qz)
                dqz = 0.5 * (wz * qw + wy * qx - wx * qy)
                nqw, nqx, nqy, nqz = qw + dt * dqw, qx + dt * dqx, qy + dt * dqy, qz + dt * dqz
                norm = torch.sqrt(nqw**2 + nqx**2 + nqy**2 + nqz**2).clamp(min=1e-10)
                q_new[:, qs] = nqw / norm
                q_new[:, qs + 1] = nqx / norm
                q_new[:, qs + 2] = nqy / norm
                q_new[:, qs + 3] = nqz / norm
                q_new[:, qs + 4] = self._q[:, qs + 4] + dt * self._qdot[:, vs] + pos_corr[:, i, 0]
                q_new[:, qs + 5] = self._q[:, qs + 5] + dt * self._qdot[:, vs + 1] + pos_corr[:, i, 1]
                q_new[:, qs + 6] = self._q[:, qs + 6] + dt * self._qdot[:, vs + 2] + pos_corr[:, i, 2]
            elif jtype == 1 or jtype == 2:  # REVOLUTE or PRISMATIC
                q_new[:, qs] = self._q[:, qs] + dt * self._qdot[:, vs]
        self._q = q_new

        return self._run_fk_and_build()

    def _compute_aba_pytorch(self, tau, ext_forces, X_up_R, X_up_r, v_bodies):
        """PyTorch ABA forward dynamics (for solver path)."""
        from ..tilelang.kernels_tl import (
            spatial_cross_force_torch,
        )

        s = self._static
        N = self._num_envs
        nb, nv = s.nb, s.nv

        # Pass 1: velocities and bias forces
        v = v_bodies.clone()
        pA = torch.zeros(N, nb, 6, device=self._device)

        # Gravity spatial vector
        g = torch.zeros(6, device=self._device)
        g[2] = -s.gravity  # linear z = -g

        for i in range(nb):
            # For ABA Pass 1 we just need pA = v×*(Iv) - f_ext
            I_i = self._inertia_mat[i].reshape(6, 6)
            Iv = torch.matmul(v[:, i].unsqueeze(1), I_i.T).squeeze(1)
            pA[:, i] = spatial_cross_force_torch(v[:, i], Iv) - ext_forces[:, i]

        # Pass 2: articulated inertias (simplified for single-body case)
        IA = torch.zeros(N, nb, 6, 6, device=self._device)
        for i in range(nb):
            IA[:, i] = self._inertia_mat[i].reshape(6, 6).unsqueeze(0).expand(N, -1, -1)

        # For multi-body, full ABA is needed. For single free body, shortcut:
        if nb == 1 and int(self._joint_type[0]) == 0:  # Single FreeJoint
            # qddot = I⁻¹ @ (tau - pA + g_force)
            I_inv = torch.linalg.inv(IA[:, 0])
            rhs = tau[:, :6] - pA[:, 0]
            # Add gravity contribution
            I_0 = IA[:, 0]
            g_force = torch.matmul(I_0, g.unsqueeze(0).expand(N, -1).unsqueeze(-1)).squeeze(-1)
            qddot = torch.matmul(I_inv, (rhs + g_force).unsqueeze(-1)).squeeze(-1)
            result = torch.zeros(N, nv, device=self._device)
            result[:, :6] = qddot
            return result

        # General case: fall back to full ABA (would need full implementation)
        # For now, use simple H⁻¹ approach
        # This is a simplified placeholder — full ABA PyTorch would be needed for multi-body
        raise NotImplementedError("CUDA solver path requires single free body for now")

    def get_obs_data(self, result: StepResult) -> dict[str, torch.Tensor]:
        s = self._static
        root = s.root_body_idx
        aq = torch.from_numpy(s.actuated_q_indices.astype(np.int64))
        av = torch.from_numpy(s.actuated_v_indices.astype(np.int64))
        return {
            "base_lin_vel": result.v_bodies[:, root, :3],
            "base_ang_vel": result.v_bodies[:, root, 3:6],
            "base_orientation": result.q[:, s.root_q_start : s.root_q_start + 4],
            "joint_pos": result.q[:, aq],
            "joint_vel": result.qdot[:, av],
            "contact_mask": result.contact_mask,
        }

    @property
    def device(self) -> str:
        return self._device

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def nq(self) -> int:
        return self._static.nq

    @property
    def nv(self) -> int:
        return self._static.nv

    @property
    def num_bodies(self) -> int:
        return self._static.nb

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_group_data(self):
        """Compute and cache group metadata for grouped Schur."""
        if self._group_data is not None:
            return self._group_data

        # We need the tree to call auto_detect_groups
        # Reconstruct minimal tree from static data to get grouping
        s = self._static
        # Use a simpler approach: detect groups from parent_idx + children
        parent_np = s.parent_idx
        nb = s.nb
        children = [[] for _ in range(nb)]
        for i in range(nb):
            p = int(parent_np[i])
            if p >= 0:
                children[p].append(i)

        # Find branch points and subtrees
        root_bodies = []
        limb_groups = []
        in_limb = set()

        def get_desc(bi):
            sub = [bi]
            stack = list(children[bi])
            while stack:
                j = stack.pop()
                sub.append(j)
                stack.extend(children[j])
            return sub

        for i in range(nb):
            if i in in_limb:
                continue
            if len(children[i]) > 1:
                root_bodies.append(i)
                for ci in children[i]:
                    sub = get_desc(ci)
                    limb_groups.append(sub)
                    in_limb.update(sub)
            elif i not in in_limb:
                root_bodies.append(i)

        # Build v-index arrays
        v_start = s.v_idx_start
        v_len = s.v_idx_len

        root_v = []
        for bi in root_bodies:
            for j in range(int(v_len[bi])):
                root_v.append(int(v_start[bi]) + j)
        root_v = np.array(root_v, dtype=np.int32)

        limb_v_all = []
        limb_offsets = [0]
        for group in limb_groups:
            for bi in group:
                for j in range(int(v_len[bi])):
                    limb_v_all.append(int(v_start[bi]) + j)
            limb_offsets.append(len(limb_v_all))
        limb_v_all = np.array(limb_v_all, dtype=np.int32)
        limb_offsets = np.array(limb_offsets, dtype=np.int32)

        self._group_data = {
            "root_v": torch.from_numpy(root_v).int().to(self._device),
            "nv_root": len(root_v),
            "limb_v_offsets": torch.from_numpy(limb_offsets).int().to(self._device),
            "limb_v_indices": torch.from_numpy(limb_v_all).int().to(self._device)
            if len(limb_v_all) > 0
            else torch.zeros(1, dtype=torch.int32, device=self._device),
            "ngroups": len(limb_groups),
        }
        return self._group_data

    def _step_grouped_schur(self, actions, action_clip):
        """CRBA with fused grouped Schur complement kernel."""
        s = self._static
        N = self._num_envs
        gd = self._get_group_data()

        results = self._cuda.physics_step_grouped_schur(
            self._q,
            self._qdot,
            actions,
            self._joint_type,
            self._joint_axis,
            self._parent_idx,
            self._q_idx_start,
            self._q_idx_len,
            self._v_idx_start,
            self._v_idx_len,
            self._X_tree_R,
            self._X_tree_r,
            self._inertia_mat,
            self._q_min,
            self._q_max,
            self._k_limit,
            self._b_limit,
            self._damping,
            self._actuated_q_idx,
            self._actuated_v_idx,
            self._effort_limits,
            self._contact_body_idx,
            self._contact_local_pos,
            gd["root_v"],
            gd["nv_root"],
            gd["limb_v_offsets"],
            gd["limb_v_indices"],
            gd["ngroups"],
            N,
            s.nb,
            s.nq,
            s.nv,
            s.nu,
            s.nc,
            self._has_effort_limits,
            self._cfg.dt,
            s.gravity,
            self._cfg.kp,
            self._cfg.kd,
            self._cfg.action_scale,
            action_clip,
            s.contact_k_normal,
            s.contact_b_normal,
            s.contact_mu,
            s.contact_slip_eps,
            s.contact_ground_z,
        )
        q_new, qdot_new, X_world_R, X_world_r, v_bodies, contact_mask = results
        self._q = q_new
        self._qdot = qdot_new

        fk = self._cuda.fk_only(
            self._q,
            self._qdot,
            self._joint_type,
            self._joint_axis,
            self._parent_idx,
            self._q_idx_start,
            self._v_idx_start,
            self._v_idx_len,
            self._X_tree_R,
            self._X_tree_r,
            N,
            s.nb,
            s.nq,
            s.nv,
        )
        X_world_R, X_world_r, v_bodies = fk
        X_flat = torch.cat([X_world_R.reshape(N, s.nb, 9), X_world_r], dim=-1)
        return StepResult(
            q=self._q.cpu(),
            qdot=self._qdot.cpu(),
            X_world=X_flat.cpu(),
            v_bodies=v_bodies.cpu(),
            contact_mask=contact_mask.bool().cpu(),
        )

    def _step_crba_tc(self, actions, action_clip):
        """CRBA with tensor-core Cholesky: split kernel + torch.linalg.cholesky_solve."""
        s = self._static
        N = self._num_envs

        # Kernel 1: fused FK + CRBA(H) + RNEA(C) → H, rhs, FK cache
        results = self._cuda.crba_build(
            self._q,
            self._qdot,
            actions,
            self._joint_type,
            self._joint_axis,
            self._parent_idx,
            self._q_idx_start,
            self._q_idx_len,
            self._v_idx_start,
            self._v_idx_len,
            self._X_tree_R,
            self._X_tree_r,
            self._inertia_mat,
            self._q_min,
            self._q_max,
            self._k_limit,
            self._b_limit,
            self._damping,
            self._actuated_q_idx,
            self._actuated_v_idx,
            self._effort_limits,
            self._contact_body_idx,
            self._contact_local_pos,
            N,
            s.nb,
            s.nq,
            s.nv,
            s.nu,
            s.nc,
            self._has_effort_limits,
            s.gravity,
            self._cfg.kp,
            self._cfg.kd,
            self._cfg.action_scale,
            action_clip,
            s.contact_k_normal,
            s.contact_b_normal,
            s.contact_mu,
            s.contact_slip_eps,
            s.contact_ground_z,
        )
        H, rhs, X_world_R, X_world_r, v_bodies, contact_mask = results

        # Cholesky solve via cuSOLVER (uses wgmma on Hopper)
        L = torch.linalg.cholesky(H)
        qddot = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)

        # Kernel 2: integration
        int_results = self._cuda.integrate_step(
            self._q,
            self._qdot,
            qddot,
            self._joint_type,
            self._q_idx_start,
            self._q_idx_len,
            self._v_idx_start,
            self._v_idx_len,
            self._cfg.dt,
            N,
            s.nb,
            s.nq,
            s.nv,
        )
        self._q, self._qdot = int_results[0], int_results[1]

        # Re-run FK
        fk = self._cuda.fk_only(
            self._q,
            self._qdot,
            self._joint_type,
            self._joint_axis,
            self._parent_idx,
            self._q_idx_start,
            self._v_idx_start,
            self._v_idx_len,
            self._X_tree_R,
            self._X_tree_r,
            N,
            s.nb,
            s.nq,
            s.nv,
        )
        X_world_R, X_world_r, v_bodies = fk

        X_flat = torch.cat([X_world_R.reshape(N, s.nb, 9), X_world_r], dim=-1)
        return StepResult(
            q=self._q.cpu(),
            qdot=self._qdot.cpu(),
            X_world=X_flat.cpu(),
            v_bodies=v_bodies.cpu(),
            contact_mask=contact_mask.bool().cpu(),
        )

    def _run_fk_and_build(self) -> StepResult:
        s = self._static
        N = self._num_envs
        fk = self._cuda.fk_only(
            self._q,
            self._qdot,
            self._joint_type,
            self._joint_axis,
            self._parent_idx,
            self._q_idx_start,
            self._v_idx_start,
            self._v_idx_len,
            self._X_tree_R,
            self._X_tree_r,
            N,
            s.nb,
            s.nq,
            s.nv,
        )
        X_world_R, X_world_r, v_bodies = fk
        X_world_flat = torch.cat([X_world_R.reshape(N, s.nb, 9), X_world_r], dim=-1)
        return StepResult(
            q=self._q.cpu(),
            qdot=self._qdot.cpu(),
            X_world=X_world_flat.cpu(),
            v_bodies=v_bodies.cpu(),
            contact_mask=torch.zeros(N, s.nc, dtype=torch.bool),
        )
