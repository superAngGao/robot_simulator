"""
GpuEngine — GPU physics engine using Warp kernels + Jacobi-PGS-SI solver.

Operates on MergedModel's unified tree. Supports ground contacts + body-body
sphere collisions. All N environments processed in parallel.

The full physics step runs as a sequence of GPU kernel launches:
  FK → collision detect → ABA(smooth) → predicted velocity →
  FK(predicted) → build W → PGS iterations → impulse → ABA(H⁻¹) →
  position correction → integration
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .backends.static_data import StaticRobotData
from .backends.warp.collision_kernels import batched_detect_analytical
from .backends.warp.crba_kernels import (
    batched_apply_contact_impulse,
    batched_build_W_joint_space,
    batched_contact_jacobian,
    batched_crba_rnea_cholesky,
)
from .backends.warp.kernels import (
    batched_fk_body_vel,
    batched_passive_torques,
)
from .backends.warp.scratch import ABABatchScratch
from .backends.warp.solver_kernels import (
    batched_constraint_integrate,
    batched_jacobi_pgs_step,
    batched_predicted_velocity,
)
from .backends.warp.solver_scratch import SolverScratch
from .engine import PhysicsEngine, StepOutput

if TYPE_CHECKING:
    from .merged_model import MergedModel


class GpuEngine(PhysicsEngine):
    """GPU physics engine with Warp kernels.

    Args:
        merged   : MergedModel (multi-root tree + collision data).
        num_envs : Number of parallel environments.
        device   : CUDA device string.
        dt       : Default time step [s].
        solver   : Constraint solver ("jacobi_pgs_si" or "admm").
    """

    def __init__(
        self,
        merged: "MergedModel",
        num_envs: int = 1,
        device: str = "cuda:0",
        dt: float = 2e-4,
        solver: str = "jacobi_pgs_si",
    ) -> None:
        super().__init__(merged)
        wp.init()
        self._device = device
        self._num_envs = num_envs
        self._dt = dt
        self._solver = solver

        # Build static data from merged model
        s = StaticRobotData.from_merged(merged)
        self._static = s

        # Allocate main scratch
        self._scratch = ABABatchScratch(
            N=num_envs,
            nb=s.nb,
            nq=s.nq,
            nv=s.nv,
            nc=s.nc,
            device=device,
        )

        # Collision data
        n_pairs = len(merged.collision_pairs)
        max_contacts = s.nc + n_pairs
        max_rows = max_contacts * 3  # condim=3

        # Solver scratch
        self._solver_scratch = SolverScratch(
            N=num_envs,
            nb=s.nb,
            nq=s.nq,
            nv=s.nv,
            nc=max_contacts,
            max_rows=max_rows,
            device=device,
            solver=solver,
        )

        # ADMM solver parameters
        if solver == "admm":
            from .backends.warp.admm_kernels import batched_admm_solve

            self._batched_admm_solve = batched_admm_solve
            self._admm_rho = 1.0
            self._admm_iters = 30
            self._admm_warmstart = True
            self._admm_solref = (0.02, 1.0)  # (timeconst, dampratio)
            self._admm_solimp = (0.9, 0.95, 0.001, 0.5, 2.0)

        # Additional collision buffers
        self._contact_normal = wp.zeros((num_envs, max_contacts, 3), dtype=wp.float32, device=device)
        self._contact_point = wp.zeros((num_envs, max_contacts, 3), dtype=wp.float32, device=device)
        self._contact_bi = wp.zeros((num_envs, max_contacts), dtype=wp.int32, device=device)
        self._contact_bj = wp.zeros((num_envs, max_contacts), dtype=wp.int32, device=device)
        self._contact_active = wp.zeros((num_envs, max_contacts), dtype=wp.int32, device=device)
        self._contact_depth = wp.zeros((num_envs, max_contacts), dtype=wp.float32, device=device)

        # J_body_j for body-body contacts
        self._J_body_j = wp.zeros((num_envs, max_rows, 6), dtype=wp.float32, device=device)
        self._row_bi = wp.zeros((num_envs, max_rows), dtype=wp.int32, device=device)
        self._row_bj = wp.zeros((num_envs, max_rows), dtype=wp.int32, device=device)

        self._max_contacts = max_contacts
        self._max_rows = max_rows
        self._n_pairs = n_pairs
        self._nc_ground = s.nc

        # Upload static data
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
        self._gpu_q_min = wp.array(s.q_min, dtype=wp.float32, device=device)
        self._gpu_q_max = wp.array(s.q_max, dtype=wp.float32, device=device)
        self._gpu_k_limit = wp.array(s.k_limit, dtype=wp.float32, device=device)
        self._gpu_b_limit = wp.array(s.b_limit, dtype=wp.float32, device=device)
        self._gpu_damping = wp.array(s.damping, dtype=wp.float32, device=device)
        self._gpu_contact_body_idx = wp.array(s.contact_body_idx, dtype=wp.int32, device=device)
        self._gpu_contact_local_pos = wp.array(s.contact_local_pos, dtype=wp.float32, device=device)
        self._gpu_inv_mass = wp.array(s.inv_mass_per_body, dtype=wp.float32, device=device)
        self._gpu_inv_inertia = wp.array(s.inv_inertia_per_body, dtype=wp.float32, device=device)
        self._gpu_body_radius = wp.array(s.body_collision_radius, dtype=wp.float32, device=device)
        self._gpu_shape_type = wp.array(s.body_shape_type, dtype=wp.int32, device=device)
        self._gpu_shape_params = wp.array(s.body_shape_params, dtype=wp.float32, device=device)

        # Body-body collision pairs
        if n_pairs > 0:
            self._gpu_pair_bi = wp.array(s.collision_pair_body_i, dtype=wp.int32, device=device)
            self._gpu_pair_bj = wp.array(s.collision_pair_body_j, dtype=wp.int32, device=device)
        else:
            self._gpu_pair_bi = wp.zeros(1, dtype=wp.int32, device=device)
            self._gpu_pair_bj = wp.zeros(1, dtype=wp.int32, device=device)

        # Default state
        q0, qdot0 = merged.tree.default_state()
        self._default_q = q0.astype(np.float32)
        self._default_qdot = qdot0.astype(np.float32)

    def reset(self, q0: np.ndarray | None = None) -> None:
        """Reset all environments to default or given state."""
        N = self._num_envs
        sc = self._scratch
        q_np = np.tile(q0 if q0 is not None else self._default_q, (N, 1)).astype(np.float32)
        qdot_np = np.tile(self._default_qdot, (N, 1)).astype(np.float32)
        wp.copy(sc.q, wp.array(q_np, dtype=wp.float32, device=self._device))
        wp.copy(sc.qdot, wp.array(qdot_np, dtype=wp.float32, device=self._device))
        # Clear ADMM warmstart state
        if self._solver == "admm":
            sol = self._solver_scratch
            sol.admm_f_prev.zero_()
            sol.admm_s_prev.zero_()
            sol.admm_u_prev.zero_()
            sol.admm_prev_n_active.zero_()

    def step(self, q=None, qdot=None, tau=None, dt=None):
        """One physics step. If q/qdot/tau not given, uses internal state."""
        N = self._num_envs
        s = self._static
        sc = self._scratch
        sol = self._solver_scratch
        dt = dt or self._dt

        # If external state provided, upload it
        if q is not None:
            q_np = np.atleast_2d(q).astype(np.float32)
            wp.copy(sc.q, wp.array(q_np, dtype=wp.float32, device=self._device))
        if qdot is not None:
            qdot_np = np.atleast_2d(qdot).astype(np.float32)
            wp.copy(sc.qdot, wp.array(qdot_np, dtype=wp.float32, device=self._device))

        # Tau (default zero)
        if tau is not None:
            tau_np = np.atleast_2d(tau).astype(np.float32)
            wp.copy(sc.tau_total, wp.array(tau_np, dtype=wp.float32, device=self._device))
        else:
            sc.tau_total.zero_()

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
        # tau_total += tau_passive
        tau_pas = sc.tau_passive.numpy()
        tau_tot = sc.tau_total.numpy() + tau_pas
        wp.copy(sc.tau_total, wp.array(tau_tot, dtype=wp.float32, device=self._device))

        # 2. FK + body velocities
        wp.launch(
            batched_fk_body_vel,
            dim=N,
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
            outputs=[sc.X_world_R, sc.X_world_r, sc.X_up_R, sc.X_up_r, sc.v_bodies],
        )

        # 3. CRBA + RNEA + Cholesky + smooth dynamics (replaces ABA)
        wp.launch(
            batched_crba_rnea_cholesky,
            dim=N,
            device=self._device,
            inputs=[
                sc.q,
                sc.qdot,
                sc.tau_total,
                self._gpu_joint_type,
                self._gpu_joint_axis,
                self._gpu_parent_idx,
                self._gpu_q_idx_start,
                self._gpu_v_idx_start,
                self._gpu_v_idx_len,
                self._gpu_inertia_mat,
                sc.X_up_R,
                sc.X_up_r,
                s.gravity,
                s.nb,
                s.nv,
                sc.IC,
                sc.rnea_v,
                sc.rnea_a,
                sc.rnea_f,
                sol.chol_tmp,
            ],
            outputs=[sol.H, sol.L_H, sol.C_bias, sol.qacc_smooth],
        )
        # qacc_smooth → sc.qddot for predicted velocity computation
        wp.copy(sc.qddot, sol.qacc_smooth)

        # 4. Predicted velocity
        wp.launch(
            batched_predicted_velocity,
            dim=N,
            device=self._device,
            inputs=[sc.qdot, sc.qddot, dt, s.nv],
            outputs=[sol.v_predicted],
        )

        # 5. Collision detection (ground + body-body) — analytical shape dispatch
        wp.launch(
            batched_detect_analytical,
            dim=N,
            device=self._device,
            inputs=[
                sc.X_world_R,
                sc.X_world_r,
                self._gpu_shape_type,
                self._gpu_shape_params,
                self._gpu_contact_body_idx,
                s.contact_ground_z,
                self._nc_ground,
                self._gpu_pair_bi,
                self._gpu_pair_bj,
                self._gpu_body_radius,
                self._n_pairs,
                self._max_contacts,
            ],
            outputs=[
                self._contact_depth,
                self._contact_normal,
                self._contact_point,
                self._contact_bi,
                self._contact_bj,
                self._contact_active,
            ],
        )

        # 6. Contact Jacobian (joint-space, Q29)
        wp.launch(
            batched_contact_jacobian,
            dim=N,
            device=self._device,
            inputs=[
                sc.q,
                sc.X_world_R,
                sc.X_world_r,
                self._contact_active,
                self._contact_normal,
                self._contact_point,
                self._contact_bi,
                self._contact_bj,
                self._gpu_joint_type,
                self._gpu_joint_axis,
                self._gpu_parent_idx,
                self._gpu_v_idx_start,
                self._gpu_v_idx_len,
                self._max_contacts,
                self._max_rows,
                s.nv,
            ],
            outputs=[sol.J_joint],
        )

        # 7. Build W (joint-space Delassus) + v_free + v_current
        wp.launch(
            batched_build_W_joint_space,
            dim=N,
            device=self._device,
            inputs=[
                sol.J_joint,
                sol.L_H,
                sol.v_predicted,
                sc.qdot,
                self._contact_active,
                s.contact_cfm,
                self._max_contacts,
                self._max_rows,
                s.nv,
                sol.HinvJt,
                sol.chol_tmp,
                sol.C_bias,  # reuse as rhs_col temp (already consumed)
                sol.qacc_smooth,  # reuse as sol_col temp (already consumed)
            ],
            outputs=[sol.W, sol.W_diag, sol.v_free, sol.v_current],
        )

        # 8. Constraint solver dispatch
        if self._solver == "admm":
            # 8a. ADMM solve (single kernel)
            sol.lambdas.zero_()
            wp.launch(
                self._batched_admm_solve,
                dim=N,
                device=self._device,
                inputs=[
                    sol.W,
                    sol.W_diag,
                    sol.v_free,
                    sol.v_current,
                    self._contact_active,
                    self._contact_depth,
                    s.contact_mu,
                    self._admm_rho,
                    dt,
                    self._admm_solref[0],
                    self._admm_solref[1],
                    self._admm_solimp[0],
                    self._admm_solimp[1],
                    self._admm_solimp[2],
                    self._admm_solimp[3],
                    self._admm_solimp[4],
                    self._max_contacts,
                    self._max_rows,
                    self._admm_iters,
                    1 if self._admm_warmstart else 0,
                    sol.admm_prev_n_active,
                    sol.admm_f_prev,
                    sol.admm_s_prev,
                    sol.admm_u_prev,
                    sol.admm_AR_rho,
                    sol.admm_L,
                    sol.admm_R_diag,
                    sol.admm_f,
                    sol.admm_s,
                    sol.admm_u,
                    sol.admm_rhs,
                    sol.admm_tmp,
                    sol.admm_rhs_const,
                ],
                outputs=[sol.lambdas],
            )
        else:
            # Jacobi PGS iterations (default)
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
                        self._contact_active,
                        s.contact_mu,
                        s.solver_omega,
                        self._max_contacts,
                        self._max_rows,
                    ],
                    outputs=[sol.lambdas],
                )

        # 9. Apply contact impulse: dqdot = H⁻¹ Jᵀ λ (Q29 joint-space)
        wp.launch(
            batched_apply_contact_impulse,
            dim=N,
            device=self._device,
            inputs=[
                sol.lambdas,
                sol.J_joint,
                sol.L_H,
                self._contact_active,
                self._max_contacts,
                self._max_rows,
                s.nv,
                sol.gen_impulse,
                sol.chol_tmp,
            ],
            outputs=[sol.dqdot],
        )

        # 11. Position correction (PGS only — ADMM handles via compliance)
        if self._solver == "admm":
            sol.pos_corrections.zero_()
        else:
            from .backends.warp.solver_kernels import batched_position_correction

            wp.launch(
                batched_position_correction,
                dim=N,
                device=self._device,
                inputs=[
                    self._contact_active,
                    self._contact_depth,
                    self._gpu_contact_body_idx,
                    self._gpu_inv_mass,
                    s.contact_erp_pos,
                    s.contact_slop,
                    self._nc_ground,
                    s.nb,  # only ground contacts get position correction
                ],
                outputs=[sol.pos_corrections],
            )

        # 12. Integration
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

        # Update state
        wp.copy(sc.q, sc.q_new)
        wp.copy(sc.qdot, sc.qdot_new)

        # Build output
        q_out = sc.q.numpy().copy()
        qdot_out = sc.qdot.numpy().copy()

        return StepOutput(
            q_new=q_out[0] if N == 1 else q_out,
            qdot_new=qdot_out[0] if N == 1 else qdot_out,
            X_world=None,  # can be populated if needed
            v_bodies=None,
            contact_active=self._contact_active.numpy().copy(),
            force_state=None,
        )
