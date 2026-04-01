"""
WarpBatchBackend — GPU-accelerated batched physics using NVIDIA Warp.

Holds all state as Warp arrays on CUDA. Each step launches a sequence
of kernels with dim=N (one thread per environment).

NOTE (Q26-gpu): This backend uses the legacy single-shape-per-body collision
path (batched_detect_analytical) and body-level Delassus matrix. Multi-shape
collision and dynamic broadphase are only available through GpuEngine, which
uses the joint-space CRBA pipeline (Q29) and batched_detect_multishape kernel.
Prefer GpuEngine for new work. WarpBatchBackend is maintained for backward
compatibility with the VecEnv/BatchBackend interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from ..batch_backend import BatchBackend, StepResult
from ..static_data import StaticRobotData
from .kernels import (
    batched_aba,
    batched_collision,
    batched_contact,
    batched_fk_body_vel,
    batched_integrate,
    batched_passive_torques,
    batched_pd_controller,
)
from .scratch import ABABatchScratch
from .solver_kernels import (
    batched_build_W_vfree,
    batched_constraint_integrate,
    batched_ground_detect,
    batched_impulse_to_gen,
    batched_jacobi_pgs_step,
    batched_position_correction,
    batched_predicted_velocity,
    batched_scale_array,
)
from .solver_scratch import SolverScratch

if TYPE_CHECKING:
    from rl_env.cfg import EnvCfg
    from robot.model import RobotModel


class WarpBatchBackend(BatchBackend):
    """GPU batched physics backend using NVIDIA Warp.

    Args:
        model    : Shared RobotModel.
        cfg      : Environment configuration.
        num_envs : Number of parallel environments.
        device   : Warp device (default "cuda:0").
    """

    def __init__(
        self,
        model: "RobotModel",
        cfg: "EnvCfg",
        num_envs: int,
        device: str = "cuda:0",
    ) -> None:
        wp.init()
        self._device = device
        self._cfg = cfg
        self._num_envs = num_envs

        # Extract static data
        self._static = StaticRobotData.from_model(model)
        s = self._static

        # Allocate scratch buffers
        self._scratch = ABABatchScratch(N=num_envs, nb=s.nb, nq=s.nq, nv=s.nv, nc=s.nc, device=device)

        # Upload static data to GPU
        self._gpu_joint_type = wp.array(s.joint_type, dtype=wp.int32, device=device)
        self._gpu_joint_axis = wp.array(s.joint_axis, dtype=wp.float32, device=device)
        self._gpu_parent_idx = wp.array(s.parent_idx, dtype=wp.int32, device=device)
        self._gpu_q_idx_start = wp.array(s.q_idx_start, dtype=wp.int32, device=device)
        self._gpu_q_idx_len = wp.array(s.q_idx_len, dtype=wp.int32, device=device)
        self._gpu_v_idx_start = wp.array(s.v_idx_start, dtype=wp.int32, device=device)
        self._gpu_v_idx_len = wp.array(s.v_idx_len, dtype=wp.int32, device=device)
        self._gpu_X_tree_R = wp.array(s.X_tree_R, dtype=wp.float32, device=device)
        self._gpu_X_tree_r = wp.array(s.X_tree_r, dtype=wp.float32, device=device)
        self._gpu_inertia_mat = wp.array(s.inertia_mat.reshape(s.nb, 6, 6), dtype=wp.float32, device=device)

        # Joint limits
        self._gpu_q_min = wp.array(s.q_min, dtype=wp.float32, device=device)
        self._gpu_q_max = wp.array(s.q_max, dtype=wp.float32, device=device)
        self._gpu_k_limit = wp.array(s.k_limit, dtype=wp.float32, device=device)
        self._gpu_b_limit = wp.array(s.b_limit, dtype=wp.float32, device=device)
        self._gpu_damping = wp.array(s.damping, dtype=wp.float32, device=device)

        # Contact
        self._gpu_contact_body_idx = wp.array(s.contact_body_idx, dtype=wp.int32, device=device)
        self._gpu_contact_local_pos = wp.array(s.contact_local_pos, dtype=wp.float32, device=device)

        # Collision
        self._gpu_coll_body_idx = wp.array(s.collision_body_idx, dtype=wp.int32, device=device)
        self._gpu_coll_half_ext = wp.array(s.collision_half_ext, dtype=wp.float32, device=device)
        self._gpu_pair_i = wp.array(s.collision_pair_i, dtype=wp.int32, device=device)
        self._gpu_pair_j = wp.array(s.collision_pair_j, dtype=wp.int32, device=device)

        # Controller
        self._gpu_actuated_q_idx = wp.array(s.actuated_q_indices, dtype=wp.int32, device=device)
        self._gpu_actuated_v_idx = wp.array(s.actuated_v_indices, dtype=wp.int32, device=device)
        self._has_effort_limits = 0
        if s.effort_limits is not None:
            self._gpu_effort_limits = wp.array(s.effort_limits, dtype=wp.float32, device=device)
            self._has_effort_limits = 1
        else:
            self._gpu_effort_limits = wp.zeros(max(s.nu, 1), dtype=wp.float32, device=device)

        # Constraint solver setup
        self._contact_solver = getattr(cfg, "contact_solver", "penalty")
        if self._contact_solver == "jacobi_pgs_si":
            max_rows = s.nc * 3  # condim=3 fixed
            self._solver_scratch = SolverScratch(
                N=num_envs,
                nb=s.nb,
                nq=s.nq,
                nv=s.nv,
                nc=s.nc,
                max_rows=max_rows,
                device=device,
            )
            self._gpu_inv_mass = wp.array(s.inv_mass_per_body, dtype=wp.float32, device=device)
            self._gpu_inv_inertia = wp.array(s.inv_inertia_per_body, dtype=wp.float32, device=device)

        # Default state (on CPU, for reset)
        self._default_q = s.default_q.copy()
        self._default_qdot = s.default_qdot.copy()

    # ------------------------------------------------------------------
    # BatchBackend interface
    # ------------------------------------------------------------------

    def reset_all(self, init_noise_scale: float = 0.0) -> StepResult:
        N = self._num_envs

        # Build initial state on CPU then upload
        q_np = np.tile(self._default_q, (N, 1))
        qdot_np = np.tile(self._default_qdot, (N, 1))
        if init_noise_scale > 0:
            q_np += np.random.uniform(-init_noise_scale, init_noise_scale, q_np.shape).astype(np.float32)

        wp.copy(self._scratch.q, wp.array(q_np, dtype=wp.float32, device=self._device))
        wp.copy(self._scratch.qdot, wp.array(qdot_np, dtype=wp.float32, device=self._device))

        # Run FK to populate caches
        self._run_fk()
        return self._build_result()

    def reset_envs(self, env_ids: torch.Tensor, init_noise_scale: float = 0.0) -> None:
        # For now, reset on CPU and upload
        for idx in env_ids.tolist():
            q_np = self._default_q.copy()
            qdot_np = self._default_qdot.copy()
            if init_noise_scale > 0:
                q_np += np.random.uniform(-init_noise_scale, init_noise_scale, q_np.shape).astype(np.float32)
            # Copy row by row (not ideal but functional)
            for j in range(self._static.nq):
                self._scratch.q.numpy()[idx, j] = q_np[j]
            for j in range(self._static.nv):
                self._scratch.qdot.numpy()[idx, j] = qdot_np[j]

    def step_batch(self, actions: torch.Tensor) -> StepResult:
        s = self._static
        N = self._num_envs
        sc = self._scratch

        # Upload actions
        actions_np = actions.detach().cpu().numpy().astype(np.float32)
        actions_wp = wp.array(actions_np, dtype=wp.float32, device=self._device)

        # 1. Passive torques
        sc.tau_passive.zero_()
        wp.launch(
            batched_passive_torques,
            dim=N,
            device=self._device,
            inputs=[
                sc.q,
                sc.qdot,
                self._gpu_joint_type,
                self._gpu_q_idx_start,
                self._gpu_v_idx_start,
                self._gpu_q_min,
                self._gpu_q_max,
                self._gpu_k_limit,
                self._gpu_b_limit,
                self._gpu_damping,
                s.nb,
            ],
            outputs=[sc.tau_passive],
        )

        # 2. PD controller
        action_clip = self._cfg.action_clip if self._cfg.action_clip is not None else -1.0
        wp.launch(
            batched_pd_controller,
            dim=N,
            device=self._device,
            inputs=[
                actions_wp,
                sc.q,
                sc.qdot,
                self._gpu_actuated_q_idx,
                self._gpu_actuated_v_idx,
                self._gpu_effort_limits,
                self._has_effort_limits,
                self._cfg.kp,
                self._cfg.kd,
                self._cfg.action_scale,
                action_clip,
                s.nu,
                s.nv,
            ],
            outputs=[sc.tau_action],
        )

        # 3. tau_total = tau_action + tau_passive (via warp)
        # Use numpy for simplicity (small array)
        tau_act = sc.tau_action.numpy()
        tau_pas = sc.tau_passive.numpy()
        tau_tot = tau_act + tau_pas
        wp.copy(sc.tau_total, wp.array(tau_tot, dtype=wp.float32, device=self._device))

        # 4. FK + body velocities
        self._run_fk()

        # ── Branch: penalty vs constraint solver ──
        if self._contact_solver == "jacobi_pgs_si":
            self._step_jacobi_pgs_si(N, s, sc)
            # Update state
            wp.copy(sc.q, sc.q_new)
            wp.copy(sc.qdot, sc.qdot_new)
            self._run_fk()
            return self._build_result()

        # 5. Contact forces (penalty path — default)
        sc.ext_forces.zero_()
        sc.contact_mask.zero_()
        if s.nc > 0:
            wp.launch(
                batched_contact,
                dim=N,
                device=self._device,
                inputs=[
                    sc.X_world_R,
                    sc.X_world_r,
                    sc.v_bodies,
                    self._gpu_contact_body_idx,
                    self._gpu_contact_local_pos,
                    s.contact_k_normal,
                    s.contact_b_normal,
                    s.contact_mu,
                    s.contact_slip_eps,
                    s.contact_ground_z,
                    s.nc,
                    s.nb,
                ],
                outputs=[sc.ext_forces, sc.contact_mask],
            )

        # 6. Self-collision forces (accumulated onto ext_forces)
        n_coll = len(s.collision_body_idx)
        n_pairs = len(s.collision_pair_i)
        if n_pairs > 0:
            wp.launch(
                batched_collision,
                dim=N,
                device=self._device,
                inputs=[
                    sc.X_world_R,
                    sc.X_world_r,
                    sc.v_bodies,
                    self._gpu_coll_body_idx,
                    self._gpu_coll_half_ext,
                    self._gpu_pair_i,
                    self._gpu_pair_j,
                    s.collision_k,
                    s.collision_b,
                    n_coll,
                    n_pairs,
                    s.nb,
                ],
                outputs=[sc.ext_forces],
            )

        # 7. ABA
        sc.qddot.zero_()
        wp.launch(
            batched_aba,
            dim=N,
            device=self._device,
            inputs=[
                sc.q,
                sc.qdot,
                sc.tau_total,
                sc.ext_forces,
                self._gpu_joint_type,
                self._gpu_joint_axis,
                self._gpu_parent_idx,
                self._gpu_q_idx_start,
                self._gpu_q_idx_len,
                self._gpu_v_idx_start,
                self._gpu_v_idx_len,
                self._gpu_inertia_mat,
                s.gravity,
                s.nb,
                sc.X_up_R,
                sc.X_up_r,
                sc.aba_v,
                sc.aba_c,
                sc.aba_IA,
                sc.aba_pA,
                sc.aba_a,
                sc.aba_U,
                sc.aba_Dinv,
                sc.aba_u,
            ],
            outputs=[sc.qddot],
        )

        # 8. Integration
        wp.launch(
            batched_integrate,
            dim=N,
            device=self._device,
            inputs=[
                sc.q,
                sc.qdot,
                sc.qddot,
                self._gpu_joint_type,
                self._gpu_q_idx_start,
                self._gpu_q_idx_len,
                self._gpu_v_idx_start,
                self._gpu_v_idx_len,
                self._cfg.dt,
                s.nb,
                s.nq,
                s.nv,
            ],
            outputs=[sc.q_new, sc.qdot_new],
        )

        # Update state
        wp.copy(sc.q, sc.q_new)
        wp.copy(sc.qdot, sc.qdot_new)

        # Re-run FK for observation cache
        self._run_fk()

        return self._build_result()

    # ------------------------------------------------------------------
    # Jacobi-PGS-SI constraint solver path
    # ------------------------------------------------------------------

    def _step_jacobi_pgs_si(self, N, s, sc):
        """Constraint solver path: predicted-velocity flow + Jacobi PGS + split impulse."""
        sol = self._solver_scratch
        dt = self._cfg.dt
        max_rows = s.nc * 3

        # 5. ABA with zero ext_forces → unconstrained acceleration a_u
        sol.ext_forces_zero.zero_()
        sc.qddot.zero_()
        wp.launch(
            batched_aba,
            dim=N,
            device=self._device,
            inputs=[
                sc.q,
                sc.qdot,
                sc.tau_total,
                sol.ext_forces_zero,
                self._gpu_joint_type,
                self._gpu_joint_axis,
                self._gpu_parent_idx,
                self._gpu_q_idx_start,
                self._gpu_q_idx_len,
                self._gpu_v_idx_start,
                self._gpu_v_idx_len,
                self._gpu_inertia_mat,
                s.gravity,
                s.nb,
                sc.X_up_R,
                sc.X_up_r,
                sc.aba_v,
                sc.aba_c,
                sc.aba_IA,
                sc.aba_pA,
                sc.aba_a,
                sc.aba_U,
                sc.aba_Dinv,
                sc.aba_u,
            ],
            outputs=[sc.qddot],
        )

        # 6. v_predicted = qdot + dt * a_u
        wp.launch(
            batched_predicted_velocity,
            dim=N,
            device=self._device,
            inputs=[sc.qdot, sc.qddot, dt, s.nv],
            outputs=[sol.v_predicted],
        )

        # 7. FK on predicted velocity → body_v_pred
        #    Reuse FK kernel with v_predicted instead of qdot
        #    We need body velocities at current q but predicted qdot
        wp.launch(
            batched_fk_body_vel,
            dim=N,
            device=self._device,
            inputs=[
                sc.q,
                sol.v_predicted,
                self._gpu_joint_type,
                self._gpu_joint_axis,
                self._gpu_parent_idx,
                self._gpu_q_idx_start,
                self._gpu_q_idx_len,
                self._gpu_v_idx_start,
                self._gpu_v_idx_len,
                self._gpu_X_tree_R,
                self._gpu_X_tree_r,
                s.nb,
            ],
            outputs=[
                sc.X_world_R,
                sc.X_world_r,
                sc.X_up_R,
                sc.X_up_r,
                sol.v_bodies_pred,
            ],
        )

        # 8. Ground contact detection
        wp.launch(
            batched_ground_detect,
            dim=N,
            device=self._device,
            inputs=[
                sc.X_world_R,
                sc.X_world_r,
                self._gpu_contact_body_idx,
                self._gpu_contact_local_pos,
                s.contact_ground_z,
                s.nc,
            ],
            outputs=[sol.contact_depth, sol.contact_point_world, sol.contact_active],
        )

        # 9. Build W, v_free, Jacobian
        wp.launch(
            batched_build_W_vfree,
            dim=N,
            device=self._device,
            inputs=[
                sc.X_world_R,
                sc.X_world_r,
                sol.v_bodies_pred,
                self._gpu_contact_body_idx,
                self._gpu_contact_local_pos,
                sol.contact_active,
                sol.contact_depth,
                self._gpu_inv_mass,
                self._gpu_inv_inertia,
                s.contact_mu,
                s.contact_cfm,
                s.solimp_d0,
                s.solimp_dw,
                s.solimp_width,
                s.solimp_mid,
                s.solimp_power,
                s.nc,
                max_rows,
            ],
            outputs=[sol.W, sol.W_diag, sol.v_free, sol.J_body, sol.row_body_idx],
        )

        # 10. Jacobi PGS iterations (×max_iter)
        sol.lambdas.zero_()
        sol.lambdas_old.zero_()
        for _ in range(s.solver_max_iter):
            wp.copy(sol.lambdas_old, sol.lambdas)
            wp.launch(
                batched_jacobi_pgs_step,
                dim=N,
                device=self._device,
                inputs=[
                    sol.W,
                    sol.W_diag,
                    sol.v_free,
                    sol.lambdas_old,
                    sol.contact_active,
                    s.contact_mu,
                    s.solver_omega,
                    s.nc,
                    max_rows,
                ],
                outputs=[sol.lambdas],
            )

        # 11. Convert lambdas → generalized impulse (body impulses + RNEA backward)
        wp.launch(
            batched_impulse_to_gen,
            dim=N,
            device=self._device,
            inputs=[
                sol.lambdas,
                sol.contact_active,
                self._gpu_contact_body_idx,
                self._gpu_contact_local_pos,
                sc.X_world_R,
                sc.X_world_r,
                sc.X_up_R,
                sc.X_up_r,
                self._gpu_joint_type,
                self._gpu_joint_axis,
                sc.q,
                self._gpu_q_idx_start,
                self._gpu_v_idx_start,
                self._gpu_v_idx_len,
                self._gpu_parent_idx,
                s.nc,
                s.nb,
                s.nv,
            ],
            outputs=[sol.body_impulses, sol.gen_impulse],
        )

        # 12. ABA trick: dqdot = H⁻¹ @ gen_impulse
        #     = dt * ABA(q, qdot=0, tau=gen_impulse/dt, ext=0)
        #     First: scale gen_impulse by 1/dt → use as tau
        wp.launch(
            batched_scale_array,
            dim=N,
            device=self._device,
            inputs=[sol.gen_impulse, 1.0 / dt, s.nv],
            outputs=[sol.dqdot],  # temporarily stores gen_impulse/dt
        )
        # Run ABA with qdot=0, tau=gen_impulse/dt
        sol.qdot_zero.zero_()
        sol.ext_forces_zero.zero_()
        sc.qddot.zero_()
        wp.launch(
            batched_aba,
            dim=N,
            device=self._device,
            inputs=[
                sc.q,
                sol.qdot_zero,
                sol.dqdot,
                sol.ext_forces_zero,
                self._gpu_joint_type,
                self._gpu_joint_axis,
                self._gpu_parent_idx,
                self._gpu_q_idx_start,
                self._gpu_q_idx_len,
                self._gpu_v_idx_start,
                self._gpu_v_idx_len,
                self._gpu_inertia_mat,
                0.0,
                s.nb,  # gravity=0 for H⁻¹ trick
                sc.X_up_R,
                sc.X_up_r,
                sc.aba_v,
                sc.aba_c,
                sc.aba_IA,
                sc.aba_pA,
                sc.aba_a,
                sc.aba_U,
                sc.aba_Dinv,
                sc.aba_u,
            ],
            outputs=[sc.qddot],  # = H⁻¹ @ (gen_impulse/dt)
        )
        # dqdot = qddot * dt = H⁻¹ @ gen_impulse
        wp.launch(
            batched_scale_array,
            dim=N,
            device=self._device,
            inputs=[sc.qddot, dt, s.nv],
            outputs=[sol.dqdot],
        )

        # 13. Position correction (split impulse)
        wp.launch(
            batched_position_correction,
            dim=N,
            device=self._device,
            inputs=[
                sol.contact_active,
                sol.contact_depth,
                self._gpu_contact_body_idx,
                self._gpu_inv_mass,
                s.contact_erp_pos,
                s.contact_slop,
                s.nc,
                s.nb,
            ],
            outputs=[sol.pos_corrections],
        )

        # 14. Integration: qdot_new = v_pred + dqdot; q integration + pos corrections
        wp.launch(
            batched_constraint_integrate,
            dim=N,
            device=self._device,
            inputs=[
                sc.q,
                sol.v_predicted,
                sol.dqdot,
                sol.pos_corrections,
                self._gpu_joint_type,
                self._gpu_q_idx_start,
                self._gpu_q_idx_len,
                self._gpu_v_idx_start,
                self._gpu_v_idx_len,
                dt,
                s.nb,
                s.nq,
                s.nv,
            ],
            outputs=[sc.q_new, sc.qdot_new],
        )

        # Update contact mask for observations
        sc.contact_mask.zero_()
        for c_idx in range(s.nc):
            # Copy contact_active to contact_mask via kernel would be cleaner
            # but for now we let observations use the solver's contact_active
            pass
        wp.copy(sc.contact_mask, sol.contact_active)

    def get_obs_data(self, result: StepResult) -> dict[str, torch.Tensor]:
        s = self._static
        root = s.root_body_idx

        base_lin_vel = result.v_bodies[:, root, :3]
        base_ang_vel = result.v_bodies[:, root, 3:6]

        qs = s.root_q_start
        base_orientation = result.q[:, qs : qs + 4]

        aq = torch.from_numpy(s.actuated_q_indices.astype(np.int64))
        av = torch.from_numpy(s.actuated_v_indices.astype(np.int64))
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
    # Internals
    # ------------------------------------------------------------------

    def _run_fk(self):
        """Run FK + body velocity kernel."""
        sc = self._scratch
        s = self._static
        wp.launch(
            batched_fk_body_vel,
            dim=self._num_envs,
            device=self._device,
            inputs=[
                sc.q,
                sc.qdot,
                self._gpu_joint_type,
                self._gpu_joint_axis,
                self._gpu_parent_idx,
                self._gpu_q_idx_start,
                self._gpu_q_idx_len,
                self._gpu_v_idx_start,
                self._gpu_v_idx_len,
                self._gpu_X_tree_R,
                self._gpu_X_tree_r,
                s.nb,
            ],
            outputs=[
                sc.X_world_R,
                sc.X_world_r,
                sc.X_up_R,
                sc.X_up_r,
                sc.v_bodies,
            ],
        )

    def _build_result(self) -> StepResult:
        """Pack current scratch state into a StepResult with torch tensors."""
        sc = self._scratch
        s = self._static
        N = self._num_envs

        q_np = sc.q.numpy()
        qdot_np = sc.qdot.numpy()
        v_np = sc.v_bodies.numpy()
        R_np = sc.X_world_R.numpy()  # (N, nb, 3, 3)
        r_np = sc.X_world_r.numpy()  # (N, nb, 3)

        # Pack X_world as (N, nb, 12) = R(9) + r(3)
        X_world_flat = np.concatenate([R_np.reshape(N, s.nb, 9), r_np], axis=2)

        contact_np = sc.contact_mask.numpy()

        return StepResult(
            q=torch.from_numpy(q_np.copy()),
            qdot=torch.from_numpy(qdot_np.copy()),
            X_world=torch.from_numpy(X_world_flat.copy()),
            v_bodies=torch.from_numpy(v_np.copy()),
            contact_mask=torch.from_numpy(contact_np.copy().astype(bool)),
        )
