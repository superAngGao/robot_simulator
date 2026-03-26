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
from .backends.warp.collision_kernels import batched_detect_all_contacts
from .backends.warp.kernels import (
    batched_aba,
    batched_fk_body_vel,
    batched_passive_torques,
)
from .backends.warp.scratch import ABABatchScratch
from .backends.warp.solver_kernels import (
    batched_constraint_integrate,
    batched_jacobi_pgs_step,
    batched_predicted_velocity,
    batched_scale_array,
)
from .backends.warp.solver_kernels_v2 import (
    batched_build_W_vfree_v2,
    batched_impulse_to_gen_v2,
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
    """

    def __init__(
        self,
        merged: "MergedModel",
        num_envs: int = 1,
        device: str = "cuda:0",
        dt: float = 2e-4,
    ) -> None:
        super().__init__(merged)
        wp.init()
        self._device = device
        self._num_envs = num_envs
        self._dt = dt

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
        )

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

        # 3. ABA (unconstrained, ext_forces=0)
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

        # 4. Predicted velocity
        wp.launch(
            batched_predicted_velocity,
            dim=N,
            device=self._device,
            inputs=[sc.qdot, sc.qddot, dt, s.nv],
            outputs=[sol.v_predicted],
        )

        # 5. FK on predicted velocity → body_v_pred
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
            outputs=[sc.X_world_R, sc.X_world_r, sc.X_up_R, sc.X_up_r, sol.v_bodies_pred],
        )

        # 6. Collision detection (ground + body-body)
        wp.launch(
            batched_detect_all_contacts,
            dim=N,
            device=self._device,
            inputs=[
                sc.X_world_R,
                sc.X_world_r,
                self._gpu_contact_body_idx,
                self._gpu_contact_local_pos,
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

        # 7. Build W + v_free (body-body aware)
        wp.launch(
            batched_build_W_vfree_v2,
            dim=N,
            device=self._device,
            inputs=[
                sc.X_world_R,
                sc.X_world_r,
                sol.v_bodies_pred,
                self._contact_active,
                self._contact_normal,
                self._contact_point,
                self._contact_bi,
                self._contact_bj,
                self._gpu_inv_mass,
                self._gpu_inv_inertia,
                s.contact_mu,
                s.contact_cfm,
                self._max_contacts,
                self._max_rows,
            ],
            outputs=[
                sol.W,
                sol.W_diag,
                sol.v_free,
                sol.J_body,
                self._J_body_j,
                self._row_bi,
                self._row_bj,
            ],
        )

        # 8. Jacobi PGS iterations
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

        # 9. Impulse → generalized (body-body aware)
        wp.launch(
            batched_impulse_to_gen_v2,
            dim=N,
            device=self._device,
            inputs=[
                sol.lambdas,
                self._contact_active,
                self._contact_normal,
                self._contact_point,
                self._contact_bi,
                self._contact_bj,
                sc.X_world_R,
                sc.X_world_r,
                sc.X_up_R,
                sc.X_up_r,
                self._gpu_joint_type,
                self._gpu_joint_axis,
                self._gpu_parent_idx,
                self._gpu_v_idx_start,
                self._max_contacts,
                s.nb,
                s.nv,
            ],
            outputs=[sol.body_impulses, sol.gen_impulse],
        )

        # 10. ABA trick: dqdot = H⁻¹ @ gen_impulse (gravity=0)
        wp.launch(
            batched_scale_array,
            dim=N,
            device=self._device,
            inputs=[sol.gen_impulse, 1.0 / dt, s.nv],
            outputs=[sol.dqdot],
        )
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
            outputs=[sc.qddot],
        )
        wp.launch(
            batched_scale_array,
            dim=N,
            device=self._device,
            inputs=[sc.qddot, dt, s.nv],
            outputs=[sol.dqdot],
        )

        # 11. Position correction
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
