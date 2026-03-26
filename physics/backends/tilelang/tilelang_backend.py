"""
TileLangBatchBackend — GPU-accelerated batched physics using TileLang + PyTorch.

Uses PyTorch CUDA tensor operations for tree-traversal algorithms (FK, ABA)
and TileLang fused kernels for embarrassingly-parallel operations.

All state is stored as PyTorch tensors on CUDA. The tree-traversal is
vectorized across N environments using batched matrix operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..batch_backend import BatchBackend, StepResult
from ..static_data import (
    JOINT_FREE,
    JOINT_PRISMATIC,
    JOINT_REVOLUTE,
    StaticRobotData,
)
from .kernels_tl import (
    get_aba_kernel,
    get_fk_kernel,
    spatial_cross_force_torch,
    spatial_cross_vel_torch,
    spatial_transform_matrix_torch,
    transform_force_torch,
    transform_velocity_torch,
)

if TYPE_CHECKING:
    from rl_env.cfg import EnvCfg
    from robot.model import RobotModel


class TileLangBatchBackend(BatchBackend):
    """GPU batched physics backend using TileLang + PyTorch CUDA tensors.

    Tree-traversal algorithms (FK, ABA) run as batched PyTorch operations.
    Element-wise operations use TileLang fused kernels where beneficial.

    Args:
        model    : Shared RobotModel.
        cfg      : Environment configuration.
        num_envs : Number of parallel environments.
        device   : PyTorch device (default "cuda:0").
    """

    def __init__(
        self,
        model: "RobotModel",
        cfg: "EnvCfg",
        num_envs: int,
        device: str = "cuda:0",
    ) -> None:
        self._device = device
        self._cfg = cfg
        self._num_envs = num_envs

        s = StaticRobotData.from_model(model)
        self._static = s

        # Upload static data to GPU as torch tensors
        self._joint_type = torch.from_numpy(s.joint_type).int().to(device)
        self._joint_axis = torch.from_numpy(s.joint_axis).to(device)
        self._parent_idx = torch.from_numpy(s.parent_idx).int().to(device)
        self._q_idx_start = torch.from_numpy(s.q_idx_start).int().to(device)
        self._q_idx_len = torch.from_numpy(s.q_idx_len).int().to(device)
        self._v_idx_start = torch.from_numpy(s.v_idx_start).int().to(device)
        self._v_idx_len = torch.from_numpy(s.v_idx_len).int().to(device)
        self._X_tree_R = torch.from_numpy(s.X_tree_R).to(device)
        self._X_tree_r = torch.from_numpy(s.X_tree_r).to(device)
        self._inertia_mat = torch.from_numpy(s.inertia_mat).to(device)

        self._q_min = torch.from_numpy(s.q_min).to(device)
        self._q_max = torch.from_numpy(s.q_max).to(device)
        self._k_limit = torch.from_numpy(s.k_limit).to(device)
        self._b_limit = torch.from_numpy(s.b_limit).to(device)
        self._damping = torch.from_numpy(s.damping).to(device)

        self._actuated_q_idx = torch.from_numpy(s.actuated_q_indices).long().to(device)
        self._actuated_v_idx = torch.from_numpy(s.actuated_v_indices).long().to(device)
        self._effort_limits = None
        if s.effort_limits is not None:
            self._effort_limits = torch.from_numpy(s.effort_limits).to(device)

        # Contact
        self._contact_body_idx = torch.from_numpy(s.contact_body_idx).long().to(device)
        self._contact_local_pos = torch.from_numpy(s.contact_local_pos).to(device)

        # Constraint solver
        self._contact_solver = getattr(cfg, "contact_solver", "penalty")
        if self._contact_solver == "jacobi_pgs_si":
            self._inv_mass = torch.from_numpy(s.inv_mass_per_body).to(device)  # (nb,)
            self._inv_inertia = torch.from_numpy(s.inv_inertia_per_body).to(device)  # (nb,3,3)

        # Collision
        self._coll_body_idx = torch.from_numpy(s.collision_body_idx).long().to(device)
        self._coll_half_ext = torch.from_numpy(s.collision_half_ext).to(device)
        self._coll_pair_i = torch.from_numpy(s.collision_pair_i).long().to(device)
        self._coll_pair_j = torch.from_numpy(s.collision_pair_j).long().to(device)

        # State tensors
        N = num_envs
        self._q = torch.zeros(N, s.nq, device=device)
        self._qdot = torch.zeros(N, s.nv, device=device)

        # Default state
        self._default_q = torch.from_numpy(s.default_q).to(device)
        self._default_qdot = torch.from_numpy(s.default_qdot).to(device)

        # Cached FK results
        self._X_world_R = torch.zeros(N, s.nb, 3, 3, device=device)
        self._X_world_r = torch.zeros(N, s.nb, 3, device=device)
        self._X_up_R = torch.zeros(N, s.nb, 3, 3, device=device)
        self._X_up_r = torch.zeros(N, s.nb, 3, device=device)
        self._v_bodies = torch.zeros(N, s.nb, 6, device=device)
        self._contact_mask = torch.zeros(N, s.nc, dtype=torch.bool, device=device)

    # ------------------------------------------------------------------
    # BatchBackend interface
    # ------------------------------------------------------------------

    def reset_all(self, init_noise_scale: float = 0.0) -> StepResult:
        N = self._num_envs
        self._q[:] = self._default_q.unsqueeze(0).expand(N, -1)
        self._qdot[:] = self._default_qdot.unsqueeze(0).expand(N, -1)
        if init_noise_scale > 0:
            self._q += torch.empty_like(self._q).uniform_(-init_noise_scale, init_noise_scale)
        self._run_fk()
        return self._build_result()

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

        # 1. Passive torques
        tau_passive = self._compute_passive_torques()

        # 2. PD controller
        tau_action = self._compute_pd_controller(actions)

        # 3. Total torques
        tau_total = tau_action + tau_passive

        # 4. FK + body velocities
        self._run_fk()

        # ── Branch: penalty vs constraint solver ──
        if self._contact_solver == "jacobi_pgs_si":
            self._step_jacobi_pgs_si(tau_total)
            self._run_fk()
            return self._build_result()

        # 5. Contact forces (penalty path — default)
        ext_forces = torch.zeros(N, s.nb, 6, device=self._device)
        self._contact_mask.zero_()
        if s.nc > 0:
            ext_forces, self._contact_mask = self._compute_contact(ext_forces)

        # 6. Self-collision
        if len(s.collision_pair_i) > 0:
            ext_forces = self._compute_collision(ext_forces)

        # 7. ABA
        qddot = self._compute_aba(tau_total, ext_forces)

        # 8. Integration
        self._integrate(qddot)

        # Re-run FK for observation cache
        self._run_fk()

        return self._build_result()

    # ------------------------------------------------------------------
    # Jacobi-PGS-SI constraint solver (pure PyTorch batched ops)
    # ------------------------------------------------------------------

    def _step_jacobi_pgs_si(self, tau_smooth):
        """Predicted-velocity flow + Jacobi PGS + split impulse (PyTorch)."""
        s = self._static
        N = self._num_envs
        dt = self._cfg.dt
        nc = s.nc
        nb = s.nb
        nv = s.nv

        # 5. ABA with zero ext_forces → unconstrained acceleration
        ext_zero = torch.zeros(N, nb, 6, device=self._device)
        a_u = self._compute_aba(tau_smooth, ext_zero)  # (N, nv)

        # 6. Predicted velocity
        v_predicted = self._qdot + dt * a_u  # (N, nv)

        # 7. FK on predicted velocity → body_v_pred
        old_qdot = self._qdot.clone()
        self._qdot = v_predicted
        self._run_fk()
        v_bodies_pred = self._v_bodies.clone()  # (N, nb, 6)
        self._qdot = old_qdot  # restore

        # 8. Ground contact detection
        contact_depth = torch.zeros(N, nc, device=self._device)
        contact_active = torch.zeros(N, nc, dtype=torch.bool, device=self._device)
        contact_point_world = torch.zeros(N, nc, 3, device=self._device)

        for c in range(nc):
            bi = int(self._contact_body_idx[c])
            R = self._X_world_R[:, bi]  # (N, 3, 3)
            r = self._X_world_r[:, bi]  # (N, 3)
            local_pos = self._contact_local_pos[c]  # (3,)
            pos_world = (R @ local_pos.unsqueeze(-1)).squeeze(-1) + r  # (N, 3)
            depth = s.contact_ground_z - pos_world[:, 2]  # (N,)
            contact_depth[:, c] = depth
            contact_point_world[:, c] = pos_world
            contact_active[:, c] = depth > 0.0

        self._contact_mask = contact_active

        # 9. Build Jacobian + Delassus matrix + v_free (condim=3 fixed, ground contact)
        max_rows = nc * 3
        normal = torch.tensor([0.0, 0.0, 1.0], device=self._device)
        t1_dir = torch.tensor([1.0, 0.0, 0.0], device=self._device)
        t2_dir = torch.tensor([0.0, 1.0, 0.0], device=self._device)
        directions = torch.stack([normal, t1_dir, t2_dir])  # (3, 3)

        J_body = torch.zeros(N, max_rows, 6, device=self._device)
        row_body = torch.full((nc,), -1, dtype=torch.long, device=self._device)
        v_free = torch.zeros(N, max_rows, device=self._device)

        for c in range(nc):
            bi = int(self._contact_body_idx[c])
            row_body[c] = bi
            base = c * 3
            R = self._X_world_R[:, bi]  # (N, 3, 3)
            Rt = R.transpose(-1, -2)  # (N, 3, 3)
            r_body = self._X_world_r[:, bi]  # (N, 3)
            r_arm = contact_point_world[:, c] - r_body  # (N, 3)

            # Body velocity at predicted state
            v_body = v_bodies_pred[:, bi]  # (N, 6)
            v_lin_w = torch.bmm(R, v_body[:, :3].unsqueeze(-1)).squeeze(-1)  # (N, 3)
            omega_w = torch.bmm(R, v_body[:, 3:].unsqueeze(-1)).squeeze(-1)  # (N, 3)
            v_contact = v_lin_w + torch.linalg.cross(omega_w, r_arm)  # (N, 3)

            for d in range(3):
                row = base + d
                direction = directions[d]  # (3,)
                # J_lin = Rᵀ @ direction, J_ang = Rᵀ @ cross(r_arm, direction)
                rxd = torch.linalg.cross(r_arm, direction.unsqueeze(0).expand(N, -1))  # (N, 3)
                J_lin = torch.bmm(Rt, direction.unsqueeze(0).expand(N, -1).unsqueeze(-1)).squeeze(-1)
                J_ang = torch.bmm(Rt, rxd.unsqueeze(-1)).squeeze(-1)
                J_body[:, row, :3] = J_lin
                J_body[:, row, 3:] = J_ang
                # v_free
                v_free[:, row] = (v_contact * direction.unsqueeze(0)).sum(-1)

        # Build W = J M⁻¹ Jᵀ (batched)
        W = torch.zeros(N, max_rows, max_rows, device=self._device)
        for c in range(nc):
            bi = int(row_body[c])
            m_inv = self._inv_mass[bi]  # scalar
            I_inv = self._inv_inertia[bi]  # (3, 3)
            base = c * 3
            rows = slice(base, base + 3)

            J_lin_c = J_body[:, rows, :3]  # (N, 3, 3)
            J_ang_c = J_body[:, rows, 3:]  # (N, 3, 3)

            # Minv_J = [m_inv * J_lin; I_inv @ J_ang]
            Minv_lin = m_inv * J_lin_c  # (N, 3, 3)
            Minv_ang = torch.matmul(J_ang_c, I_inv.unsqueeze(0).expand(N, -1, -1).transpose(-1, -2))

            # For all row pairs touching this body
            for c2 in range(nc):
                if int(row_body[c2]) != bi:
                    continue
                base2 = c2 * 3
                rows2 = slice(base2, base2 + 3)
                J2_lin = J_body[:, rows2, :3]
                J2_ang = J_body[:, rows2, 3:]
                # W[rows, rows2] += J2 · Minv_J
                block = torch.bmm(J2_lin, Minv_lin.transpose(-1, -2)) + torch.bmm(
                    J2_ang, Minv_ang.transpose(-1, -2)
                )
                W[:, rows, rows2] += block

        W_diag = W.diagonal(dim1=-2, dim2=-1).clone() + s.contact_cfm  # (N, max_rows)

        # 10. Jacobi PGS iterations (erp=0, no Baumgarte)
        lambdas = torch.zeros(N, max_rows, device=self._device)
        mu = s.contact_mu
        omega = s.solver_omega

        for _ in range(s.solver_max_iter):
            lambdas_old = lambdas.clone()
            # Wl = W @ lambdas_old
            Wl = torch.bmm(W, lambdas_old.unsqueeze(-1)).squeeze(-1)  # (N, max_rows)

            for c in range(nc):
                # Mask inactive contacts
                active = contact_active[:, c]  # (N,) bool
                base = c * 3

                # Normal row
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

        # 11. Convert lambdas → body impulses → generalized impulse
        body_impulses = torch.zeros(N, nb, 6, device=self._device)
        for c in range(nc):
            bi = int(self._contact_body_idx[c])
            base = c * 3
            l_n = lambdas[:, base]
            l_t1 = lambdas[:, base + 1]
            l_t2 = lambdas[:, base + 2]

            F_world = l_n.unsqueeze(-1) * normal + l_t1.unsqueeze(-1) * t1_dir + l_t2.unsqueeze(-1) * t2_dir
            R = self._X_world_R[:, bi]
            r_body = self._X_world_r[:, bi]
            r_arm = contact_point_world[:, c] - r_body
            torque_world = torch.linalg.cross(r_arm, F_world)

            Rinv = R.transpose(-1, -2)
            f_lin_body = torch.bmm(Rinv, F_world.unsqueeze(-1)).squeeze(-1)
            f_ang_body = torch.bmm(Rinv, torque_world.unsqueeze(-1)).squeeze(-1)
            f_body = torch.cat([f_lin_body, f_ang_body], dim=-1)
            body_impulses[:, bi] += f_body

        # RNEA backward pass → generalized impulse
        gen_impulse = torch.zeros(N, nv, device=self._device)
        f_prop = body_impulses.clone()
        for idx in range(nb):
            i = nb - 1 - idx
            f_i = f_prop[:, i]  # (N, 6)
            jtype = int(self._joint_type[i])
            vs = int(self._v_idx_start[i])

            if jtype == JOINT_REVOLUTE or jtype == JOINT_PRISMATIC:
                axis = self._joint_axis[i]  # (3,)
                if jtype == JOINT_REVOLUTE:
                    gen_impulse[:, vs] = (f_i[:, 3:] * axis.unsqueeze(0)).sum(-1)
                else:
                    gen_impulse[:, vs] = (f_i[:, :3] * axis.unsqueeze(0)).sum(-1)
            elif jtype == JOINT_FREE:
                gen_impulse[:, vs : vs + 6] = f_i

            pi = int(self._parent_idx[i])
            if pi >= 0:
                R_up = self._X_up_R[:, i]
                r_up = self._X_up_r[:, i]
                f_parent = transform_force_torch(R_up, r_up, f_i)
                f_prop[:, pi] += f_parent

        # 12. ABA trick: dqdot = H⁻¹ @ gen_impulse
        #     ABA includes gravity, so subtract it: dqdot = (ABA(tau=impulse/dt) - ABA(tau=0)) * dt
        old_qdot2 = self._qdot.clone()
        self._qdot.zero_()
        tau_zero = torch.zeros(N, nv, device=self._device)
        a_with_impulse = self._compute_aba(gen_impulse / dt, ext_zero)
        a_gravity_only = self._compute_aba(tau_zero, ext_zero)
        dqdot = (a_with_impulse - a_gravity_only) * dt
        self._qdot = old_qdot2

        # 13. Position correction (split impulse)
        pos_corr = torch.zeros(N, nb, 3, device=self._device)
        erp = s.contact_erp_pos
        slop = s.contact_slop
        for c in range(nc):
            bi = int(self._contact_body_idx[c])
            eff_depth = contact_depth[:, c] - slop  # (N,)
            correction = erp * torch.clamp(eff_depth, min=0.0)  # (N,)
            pos_corr[:, bi] += correction.unsqueeze(-1) * normal.unsqueeze(0)

        # 14. Integration: qdot_new = v_predicted + dqdot
        self._qdot = v_predicted + dqdot
        q_new = self._q.clone()
        for i in range(nb):
            jtype = int(self._joint_type[i])
            qs = int(self._q_idx_start[i])
            vs = int(self._v_idx_start[i])

            if jtype == JOINT_FREE:
                qw = self._q[:, qs]
                qx = self._q[:, qs + 1]
                qy = self._q[:, qs + 2]
                qz = self._q[:, qs + 3]
                wx = self._qdot[:, vs + 3]
                wy = self._qdot[:, vs + 4]
                wz = self._qdot[:, vs + 5]
                dqw = 0.5 * (-wx * qx - wy * qy - wz * qz)
                dqx = 0.5 * (wx * qw + wz * qy - wy * qz)
                dqy = 0.5 * (wy * qw - wz * qx + wx * qz)
                dqz = 0.5 * (wz * qw + wy * qx - wx * qy)
                nqw = qw + dt * dqw
                nqx = qx + dt * dqx
                nqy = qy + dt * dqy
                nqz = qz + dt * dqz
                norm = torch.sqrt(nqw**2 + nqx**2 + nqy**2 + nqz**2).clamp(min=1e-10)
                q_new[:, qs] = nqw / norm
                q_new[:, qs + 1] = nqx / norm
                q_new[:, qs + 2] = nqy / norm
                q_new[:, qs + 3] = nqz / norm
                q_new[:, qs + 4] = self._q[:, qs + 4] + dt * self._qdot[:, vs] + pos_corr[:, i, 0]
                q_new[:, qs + 5] = self._q[:, qs + 5] + dt * self._qdot[:, vs + 1] + pos_corr[:, i, 1]
                q_new[:, qs + 6] = self._q[:, qs + 6] + dt * self._qdot[:, vs + 2] + pos_corr[:, i, 2]
            elif jtype == JOINT_REVOLUTE or jtype == JOINT_PRISMATIC:
                q_new[:, qs] = self._q[:, qs] + dt * self._qdot[:, vs]

        self._q = q_new

    def get_obs_data(self, result: StepResult) -> dict[str, torch.Tensor]:
        s = self._static
        root = s.root_body_idx
        base_lin_vel = result.v_bodies[:, root, :3]
        base_ang_vel = result.v_bodies[:, root, 3:6]
        qs = s.root_q_start
        base_orientation = result.q[:, qs : qs + 4]
        aq = self._actuated_q_idx.cpu()
        av = self._actuated_v_idx.cpu()
        joint_pos = result.q[:, aq]
        joint_vel = result.qdot[:, av]
        return {
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "base_orientation": base_orientation,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
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
    # FK + body velocities (TileLang kernel)
    # ------------------------------------------------------------------

    def _run_fk(self):
        s = self._static
        N = self._num_envs
        fk_kernel = get_fk_kernel(N, s.nb, s.nq, s.nv)
        result = fk_kernel(
            self._q,
            self._qdot,
            self._joint_type,
            self._joint_axis,
            self._parent_idx,
            self._q_idx_start,
            self._q_idx_len,
            self._v_idx_start,
            self._v_idx_len,
            self._X_tree_R,
            self._X_tree_r,
        )
        self._X_world_R = result[0]
        self._X_world_r = result[1]
        self._X_up_R = result[2]
        self._X_up_r = result[3]
        self._v_bodies = result[4]

    def _joint_vJ(self, jtype, body_idx, vs, vl):
        """Compute vJ = S @ qdot for all envs. Returns (N, 6)."""
        N = self._num_envs
        if jtype == JOINT_REVOLUTE:
            axis = self._joint_axis[body_idx]
            qd = self._qdot[:, vs : vs + 1]
            vJ = torch.zeros(N, 6, device=self._device)
            vJ[:, 3:] = axis.unsqueeze(0) * qd
        elif jtype == JOINT_PRISMATIC:
            axis = self._joint_axis[body_idx]
            qd = self._qdot[:, vs : vs + 1]
            vJ = torch.zeros(N, 6, device=self._device)
            vJ[:, :3] = axis.unsqueeze(0) * qd
        elif jtype == JOINT_FREE:
            vJ = self._qdot[:, vs : vs + 6]
        else:
            vJ = torch.zeros(N, 6, device=self._device)
        return vJ

    # ------------------------------------------------------------------
    # Passive torques (PyTorch vectorized)
    # ------------------------------------------------------------------

    def _compute_passive_torques(self):
        s = self._static
        N = self._num_envs
        tau = torch.zeros(N, s.nv, device=self._device)

        for i in range(s.nb):
            jtype = int(self._joint_type[i])
            if jtype == JOINT_REVOLUTE:
                vs = int(self._v_idx_start[i])
                qs = int(self._q_idx_start[i])
                angle = self._q[:, qs]
                omega = self._qdot[:, vs]
                qmin = float(self._q_min[i])
                qmax = float(self._q_max[i])
                k = float(self._k_limit[i])
                b = float(self._b_limit[i])
                d = float(self._damping[i])

                t = torch.zeros(N, device=self._device)
                below = angle < qmin
                above = angle > qmax
                if below.any():
                    pen = qmin - angle[below]
                    damp = torch.clamp(omega[below], max=0.0)
                    t[below] = k * pen - b * damp
                if above.any():
                    pen = angle[above] - qmax
                    damp = torch.clamp(omega[above], min=0.0)
                    t[above] = -(k * pen + b * damp)
                t = t - d * omega
                tau[:, vs] = t

            elif jtype == JOINT_PRISMATIC:
                vs = int(self._v_idx_start[i])
                d = float(self._damping[i])
                tau[:, vs] = -d * self._qdot[:, vs]

        return tau

    # ------------------------------------------------------------------
    # PD controller (PyTorch vectorized)
    # ------------------------------------------------------------------

    def _compute_pd_controller(self, actions):
        s = self._static
        N = self._num_envs
        cfg = self._cfg

        if cfg.action_clip is not None:
            actions = torch.clamp(actions, -cfg.action_clip, cfg.action_clip)

        aq = self._actuated_q_idx
        av = self._actuated_v_idx

        target = self._q[:, aq] + actions * cfg.action_scale
        tau_act = cfg.kp * (target - self._q[:, aq]) - cfg.kd * self._qdot[:, av]

        if self._effort_limits is not None:
            tau_act = torch.clamp(tau_act, -self._effort_limits, self._effort_limits)

        tau = torch.zeros(N, s.nv, device=self._device)
        tau[:, av] = tau_act
        return tau

    # ------------------------------------------------------------------
    # Contact forces (PyTorch vectorized)
    # ------------------------------------------------------------------

    def _compute_contact(self, ext_forces):
        s = self._static
        N = self._num_envs
        contact_mask = torch.zeros(N, s.nc, dtype=torch.bool, device=self._device)

        for c in range(s.nc):
            bi = int(self._contact_body_idx[c])
            local_pos = self._contact_local_pos[c]  # (3,)

            R = self._X_world_R[:, bi]  # (N, 3, 3)
            r = self._X_world_r[:, bi]  # (N, 3)

            pos_world = torch.bmm(R, local_pos.unsqueeze(0).expand(N, -1).unsqueeze(-1)).squeeze(-1) + r
            depth = s.contact_ground_z - pos_world[:, 2]

            active = depth > 0
            contact_mask[:, c] = active

            if not active.any():
                continue

            # Contact velocity
            v_body = self._v_bodies[:, bi]  # (N, 6)
            v_lin_w = torch.bmm(R, v_body[:, :3].unsqueeze(-1)).squeeze(-1)
            omega_w = torch.bmm(R, v_body[:, 3:].unsqueeze(-1)).squeeze(-1)
            r_local_w = torch.bmm(R, local_pos.unsqueeze(0).expand(N, -1).unsqueeze(-1)).squeeze(-1)
            vel_world = v_lin_w + torch.cross(omega_w, r_local_w, dim=-1)

            # Normal force
            F_n = s.contact_k_normal * depth - s.contact_b_normal * vel_world[:, 2]
            F_n = torch.clamp(F_n, min=0.0)

            # Friction
            vx, vy = vel_world[:, 0], vel_world[:, 1]
            slip_norm = torch.sqrt(vx * vx + vy * vy + s.contact_slip_eps**2)
            Ftx = -s.contact_mu * F_n * vx / slip_norm
            Fty = -s.contact_mu * F_n * vy / slip_norm

            F_world = torch.stack([Ftx, Fty, F_n], dim=-1)  # (N, 3)
            r_arm = pos_world - r
            torque_world = torch.cross(r_arm, F_world, dim=-1)
            f_world = torch.cat([F_world, torque_world], dim=-1)  # (N, 6)

            # Transform to body frame
            Rinv = R.transpose(-1, -2)
            rinv = -torch.bmm(Rinv, r.unsqueeze(-1)).squeeze(-1)
            f_body = transform_force_torch(Rinv, rinv, f_world)

            # Mask and accumulate
            f_body[~active] = 0
            ext_forces[:, bi] += f_body

        return ext_forces, contact_mask

    # ------------------------------------------------------------------
    # Self-collision (PyTorch vectorized)
    # ------------------------------------------------------------------

    def _compute_collision(self, ext_forces):
        s = self._static
        N = self._num_envs

        for p in range(len(s.collision_pair_i)):
            ii = int(self._coll_pair_i[p])
            jj = int(self._coll_pair_j[p])
            bi = int(self._coll_body_idx[ii])
            bj = int(self._coll_body_idx[jj])

            Ri = self._X_world_R[:, bi]
            ri = self._X_world_r[:, bi]
            Rj = self._X_world_R[:, bj]
            rj = self._X_world_r[:, bj]

            he_i = self._coll_half_ext[ii]
            he_j = self._coll_half_ext[jj]

            # World AABB
            wh_i = (Ri.abs() @ he_i.unsqueeze(-1)).squeeze(-1)
            wh_j = (Rj.abs() @ he_j.unsqueeze(-1)).squeeze(-1)

            min_corners = torch.min(ri + wh_i, rj + wh_j)
            max_corners = torch.max(ri - wh_i, rj - wh_j)
            overlap = min_corners - max_corners  # (N, 3)

            separated = (overlap <= 0).any(dim=-1)  # (N,)
            colliding = ~separated

            if not colliding.any():
                continue

            # Min overlap axis
            overlap_clamped = overlap.clone()
            overlap_clamped[separated] = 1e10
            min_axis = overlap_clamped.argmin(dim=-1)  # (N,)
            depth = overlap_clamped.gather(1, min_axis.unsqueeze(-1)).squeeze(-1)

            sep_vec = ri - rj
            sign = torch.ones(N, device=self._device)
            for k in range(3):
                mask_k = (min_axis == k) & colliding
                sign[mask_k & (sep_vec[:, k] < 0)] = -1.0

            direction = torch.zeros(N, 3, device=self._device)
            for k in range(3):
                direction[min_axis == k, k] = sign[min_axis == k]

            F_mag = s.collision_k * depth

            # Velocity damping
            vi_lin = torch.bmm(Ri, self._v_bodies[:, bi, :3].unsqueeze(-1)).squeeze(-1)
            vj_lin = torch.bmm(Rj, self._v_bodies[:, bj, :3].unsqueeze(-1)).squeeze(-1)
            v_rel = (vi_lin - vj_lin) * direction
            v_rel_sum = v_rel.sum(dim=-1)
            approaching = v_rel_sum < 0
            F_mag[approaching & colliding] -= s.collision_b * v_rel_sum[approaching & colliding]

            F_world = direction * F_mag.unsqueeze(-1)
            f_sw_i = torch.cat([F_world, torch.zeros(N, 3, device=self._device)], dim=-1)
            f_sw_j = torch.cat([-F_world, torch.zeros(N, 3, device=self._device)], dim=-1)

            Rinv_i = Ri.transpose(-1, -2)
            rinv_i = -torch.bmm(Rinv_i, ri.unsqueeze(-1)).squeeze(-1)
            f_body_i = transform_force_torch(Rinv_i, rinv_i, f_sw_i)

            Rinv_j = Rj.transpose(-1, -2)
            rinv_j = -torch.bmm(Rinv_j, rj.unsqueeze(-1)).squeeze(-1)
            f_body_j = transform_force_torch(Rinv_j, rinv_j, f_sw_j)

            f_body_i[~colliding] = 0
            f_body_j[~colliding] = 0
            ext_forces[:, bi] += f_body_i
            ext_forces[:, bj] += f_body_j

        return ext_forces

    # ------------------------------------------------------------------
    # ABA (PyTorch batched, sequential over bodies)
    # ------------------------------------------------------------------

    def _compute_aba(self, tau_total, ext_forces):
        s = self._static
        N = self._num_envs
        aba_kernel = get_aba_kernel(N, s.nb, s.nq, s.nv)
        result = aba_kernel(
            self._q,
            self._qdot,
            tau_total,
            ext_forces,
            self._joint_type,
            self._joint_axis,
            self._parent_idx,
            self._v_idx_start,
            self._v_idx_len,
            self._inertia_mat,
            self._X_up_R,
            self._X_up_r,
            float(s.gravity),
        )
        return result

    def _compute_aba_pytorch(self, tau_total, ext_forces):
        """PyTorch reference ABA (kept for debugging)."""
        s = self._static
        N = self._num_envs
        nb = s.nb
        dev = self._device

        a_gravity = torch.tensor([0, 0, -s.gravity, 0, 0, 0], device=dev, dtype=torch.float32)

        v = torch.zeros(N, nb, 6, device=dev)
        c = torch.zeros(N, nb, 6, device=dev)
        IA = torch.zeros(N, nb, 6, 6, device=dev)
        pA = torch.zeros(N, nb, 6, device=dev)
        a = torch.zeros(N, nb, 6, device=dev)

        U_store = [None] * nb
        Dinv_store = [None] * nb
        u_store = [None] * nb

        # Pass 1: forward
        for i in range(nb):
            jtype = int(self._joint_type[i])
            vs = int(self._v_idx_start[i])
            vl = int(self._v_idx_len[i])
            pid = int(self._parent_idx[i])

            R_up = self._X_up_R[:, i]
            r_up = self._X_up_r[:, i]

            vJ = self._joint_vJ(jtype, i, vs, vl) if vl > 0 else torch.zeros(N, 6, device=dev)

            if pid < 0:
                v[:, i] = vJ
            else:
                v_xformed = transform_velocity_torch(R_up, r_up, v[:, pid])
                v[:, i] = v_xformed + vJ
                c[:, i] = spatial_cross_vel_torch(v[:, i], vJ)

            I_mat = self._inertia_mat[i].unsqueeze(0).expand(N, -1, -1)  # (N, 6, 6)
            IA[:, i] = I_mat
            Iv = torch.bmm(I_mat, v[:, i].unsqueeze(-1)).squeeze(-1)
            pA[:, i] = spatial_cross_force_torch(v[:, i], Iv) - ext_forces[:, i]

        # Pass 2: backward
        for i in reversed(range(nb)):
            jtype = int(self._joint_type[i])
            vs = int(self._v_idx_start[i])
            vl = int(self._v_idx_len[i])
            pid = int(self._parent_idx[i])

            IA_i = IA[:, i]
            pA_i = pA[:, i]
            c_i = c[:, i]

            if vl > 0:
                if jtype == JOINT_REVOLUTE or jtype == JOINT_PRISMATIC:
                    axis = self._joint_axis[i]
                    S_col = torch.zeros(6, device=dev)
                    if jtype == JOINT_REVOLUTE:
                        S_col[3:] = axis
                    else:
                        S_col[:3] = axis
                    S_col = S_col.unsqueeze(0).expand(N, -1)  # (N, 6)

                    U_i = torch.bmm(IA_i, S_col.unsqueeze(-1)).squeeze(-1)  # (N, 6)
                    D_val = (S_col * U_i).sum(dim=-1)  # (N,)
                    D_inv_val = 1.0 / D_val
                    tau_i = tau_total[:, vs]
                    u_val = tau_i - (S_col * pA_i).sum(dim=-1)

                    U_store[i] = U_i
                    Dinv_store[i] = D_inv_val
                    u_store[i] = u_val

                    UUT = torch.bmm(U_i.unsqueeze(-1), U_i.unsqueeze(-2))
                    IA_A = IA_i - UUT * D_inv_val.unsqueeze(-1).unsqueeze(-1)
                    IAc = torch.bmm(IA_A, c_i.unsqueeze(-1)).squeeze(-1)
                    pA_A = pA_i + IAc + U_i * (D_inv_val * u_val).unsqueeze(-1)

                elif jtype == JOINT_FREE:
                    # S = I, U = IA, D = IA, D_inv = IA^{-1}
                    u_i = tau_total[:, vs : vs + 6] - pA_i
                    u_store[i] = u_i
                    U_store[i] = IA_i
                    Dinv_store[i] = torch.linalg.inv(IA_i)

                    # IA_A = 0
                    IA_A = torch.zeros_like(IA_i)
                    pA_A = pA_i + u_i
            else:
                IA_A = IA_i
                IAc = torch.bmm(IA_i, c_i.unsqueeze(-1)).squeeze(-1)
                pA_A = pA_i + IAc

            if pid >= 0:
                R_up = self._X_up_R[:, i]
                r_up = self._X_up_r[:, i]
                X6 = spatial_transform_matrix_torch(R_up, r_up)
                X6T = X6.transpose(-1, -2)
                contrib = torch.bmm(X6T, torch.bmm(IA_A, X6))
                IA[:, pid] = IA[:, pid] + contrib
                pA_force = transform_force_torch(R_up, r_up, pA_A)
                pA[:, pid] = pA[:, pid] + pA_force

        # Pass 3: forward
        qddot = torch.zeros(N, s.nv, device=dev)

        for i in range(nb):
            jtype = int(self._joint_type[i])
            vs = int(self._v_idx_start[i])
            vl = int(self._v_idx_len[i])
            pid = int(self._parent_idx[i])

            R_up = self._X_up_R[:, i]
            r_up = self._X_up_r[:, i]

            if pid < 0:
                neg_grav = -a_gravity.unsqueeze(0).expand(N, -1)
                a_p = transform_velocity_torch(R_up, r_up, neg_grav)
            else:
                a_p = transform_velocity_torch(R_up, r_up, a[:, pid])

            apc = a_p + c[:, i]

            if vl > 0:
                if jtype == JOINT_REVOLUTE or jtype == JOINT_PRISMATIC:
                    axis = self._joint_axis[i]
                    S_col = torch.zeros(6, device=dev)
                    if jtype == JOINT_REVOLUTE:
                        S_col[3:] = axis
                    else:
                        S_col[:3] = axis
                    S_col = S_col.unsqueeze(0).expand(N, -1)

                    U_i = U_store[i]
                    D_inv_val = Dinv_store[i]
                    u_val = u_store[i]

                    UT_apc = (U_i * apc).sum(dim=-1)
                    qddot_i = D_inv_val * (u_val - UT_apc)
                    qddot[:, vs] = qddot_i
                    a[:, i] = apc + S_col * qddot_i.unsqueeze(-1)

                elif jtype == JOINT_FREE:
                    u_i = u_store[i]
                    IA_i = U_store[i]
                    IA_apc = torch.bmm(IA_i, apc.unsqueeze(-1)).squeeze(-1)
                    rhs = u_i - IA_apc
                    qdd = torch.linalg.solve(IA_i, rhs)
                    qddot[:, vs : vs + 6] = qdd
                    a[:, i] = apc + qdd
            else:
                a[:, i] = apc

        return qddot

    # ------------------------------------------------------------------
    # Integration (PyTorch)
    # ------------------------------------------------------------------

    def _integrate(self, qddot):
        s = self._static
        dt = self._cfg.dt

        qdot_new = self._qdot + dt * qddot

        q_new = self._q.clone()
        for i in range(s.nb):
            jtype = int(self._joint_type[i])
            qs = int(self._q_idx_start[i])
            ql = int(self._q_idx_len[i])
            vs = int(self._v_idx_start[i])
            vl = int(self._v_idx_len[i])

            if jtype == JOINT_FREE:
                # Quaternion integration
                quat = self._q[:, qs : qs + 4]
                omega = qdot_new[:, vs + 3 : vs + 6]
                qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
                wx, wy, wz = omega[:, 0], omega[:, 1], omega[:, 2]

                dqw = 0.5 * (-qx * wx - qy * wy - qz * wz)
                dqx = 0.5 * (qw * wx + qy * wz - qz * wy)
                dqy = 0.5 * (qw * wy - qx * wz + qz * wx)
                dqz = 0.5 * (qw * wz + qx * wy - qy * wx)

                q_quat = quat + dt * torch.stack([dqw, dqx, dqy, dqz], dim=-1)
                q_quat = q_quat / q_quat.norm(dim=-1, keepdim=True)
                q_new[:, qs : qs + 4] = q_quat
                q_new[:, qs + 4 : qs + 7] = self._q[:, qs + 4 : qs + 7] + dt * qdot_new[:, vs : vs + 3]

            elif vl > 0:
                q_new[:, qs : qs + ql] = self._q[:, qs : qs + ql] + dt * qdot_new[:, vs : vs + vl]

        self._q = q_new
        self._qdot = qdot_new

    # ------------------------------------------------------------------
    # Result packing
    # ------------------------------------------------------------------

    def _build_result(self) -> StepResult:
        s = self._static
        N = self._num_envs
        X_world_flat = torch.cat(
            [
                self._X_world_R.reshape(N, s.nb, 9),
                self._X_world_r,
            ],
            dim=-1,
        )

        return StepResult(
            q=self._q.detach().cpu(),
            qdot=self._qdot.detach().cpu(),
            X_world=X_world_flat.detach().cpu(),
            v_bodies=self._v_bodies.detach().cpu(),
            contact_mask=self._contact_mask.detach().cpu(),
        )
