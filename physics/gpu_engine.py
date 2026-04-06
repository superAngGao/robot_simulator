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
from .backends.warp.collision_kernels import batched_detect_multishape
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
    CONDIM,
    batched_constraint_integrate,
    batched_jacobi_pgs_step,
    batched_predicted_velocity,
)
from .backends.warp.solver_kernels_v2 import _build_tangent_frame
from .backends.warp.solver_scratch import SolverScratch
from .engine import PhysicsEngine, StepOutput

if TYPE_CHECKING:
    from .merged_model import MergedModel


# ── Contact force aggregation kernel ──


@wp.kernel
def _aggregate_contact_forces(
    lambdas: wp.array2d(dtype=wp.float32),
    contact_active: wp.array2d(dtype=wp.int32),
    contact_normal: wp.array3d(dtype=wp.float32),
    contact_bi: wp.array2d(dtype=wp.int32),
    sensor_body_idx: wp.array(dtype=wp.int32),
    max_contacts: int,
    nc_sensor: int,
    dt: float,
    force_out: wp.array2d(dtype=wp.float32),
):
    """Aggregate per-contact impulses into per-sensor-body world forces.

    For each active contact whose body_i matches a sensor body, reconstruct
    the 3D world-frame force from (lambda_n, lambda_t1, lambda_t2) and
    atomic-add into force_out[env, sensor_idx*3 + k].

    Args:
        lambdas      : (N, max_rows) constraint impulses from solver.
        contact_active : (N, max_contacts) int, 1 if active.
        contact_normal : (N, max_contacts, 3) contact normals.
        contact_bi   : (N, max_contacts) body index of contact body i.
        sensor_body_idx : (nc_sensor,) body indices to track.
        max_contacts : Max contacts per env.
        nc_sensor    : Number of sensor bodies.
        dt           : Time step (force = impulse / dt).
        force_out    : (N, nc_sensor*3) output, must be zeroed before call.
    """
    env_id = wp.tid()

    for c in range(max_contacts):
        if contact_active[env_id, c] == 0:
            continue

        bi = contact_bi[env_id, c]

        # Find sensor index for this body (no break — Warp limitation)
        si = int(-1)
        for s in range(nc_sensor):
            if sensor_body_idx[s] == bi and si < 0:
                si = s
        if si < 0:
            continue

        # Reconstruct world force from constraint impulses
        normal = wp.vec3(
            contact_normal[env_id, c, 0],
            contact_normal[env_id, c, 1],
            contact_normal[env_id, c, 2],
        )
        frame = _build_tangent_frame(normal)
        t1 = wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2])
        t2 = wp.vec3(frame[1, 0], frame[1, 1], frame[1, 2])

        base = c * CONDIM
        inv_dt = 1.0 / dt
        ln = lambdas[env_id, base + 0] * inv_dt
        lt1 = lambdas[env_id, base + 1] * inv_dt
        lt2 = lambdas[env_id, base + 2] * inv_dt

        f = normal * ln + t1 * lt1 + t2 * lt2

        off = si * 3
        wp.atomic_add(force_out, env_id, off + 0, f[0])
        wp.atomic_add(force_out, env_id, off + 1, f[1])
        wp.atomic_add(force_out, env_id, off + 2, f[2])


# ── Per-env reset kernels ──


@wp.kernel
def _scatter_reset(
    env_ids: wp.array(dtype=wp.int32),
    q_src: wp.array2d(dtype=wp.float32),
    qdot_src: wp.array2d(dtype=wp.float32),
    nq: int,
    nv: int,
    q_dst: wp.array2d(dtype=wp.float32),
    qdot_dst: wp.array2d(dtype=wp.float32),
):
    """Copy q_src[i] → q_dst[env_ids[i]] for each i in range(dim)."""
    i = wp.tid()
    eid = env_ids[i]
    for j in range(nq):
        q_dst[eid, j] = q_src[i, j]
    for j in range(nv):
        qdot_dst[eid, j] = qdot_src[i, j]


@wp.kernel
def _scatter_zero_2d(
    env_ids: wp.array(dtype=wp.int32),
    dim1: int,
    dst: wp.array2d(dtype=wp.float32),
):
    """Zero out dst[env_ids[i], :] for 2D arrays (ADMM warmstart)."""
    i = wp.tid()
    eid = env_ids[i]
    for j in range(dim1):
        dst[eid, j] = 0.0


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

        # Collision data — dynamic broadphase (Q26-gpu)
        # max_contacts: ground shapes + worst-case body-body shape pairs
        max_shapes_per_body = int(s.body_shape_num.max()) if s.nshape > 0 else 1
        max_ground_contacts = s.nc * max_shapes_per_body
        # Count bodies with shapes for body-body upper bound
        n_bodies_with_shapes = int(np.sum(s.body_shape_num > 0))
        max_body_pairs = n_bodies_with_shapes * (n_bodies_with_shapes - 1) // 2
        max_pair_contacts = min(max_body_pairs * max_shapes_per_body**2, 1024)
        max_contacts = max(max_ground_contacts + max_pair_contacts, 16)
        max_contacts = min(max_contacts, 2048)  # cap to prevent excessive memory
        n_pairs = len(merged.collision_pairs)  # kept for legacy warp_backend compat
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
        self._contact_count = wp.zeros(num_envs, dtype=wp.int32, device=device)  # atomic counter

        # J_body_j for body-body contacts
        self._J_body_j = wp.zeros((num_envs, max_rows, 6), dtype=wp.float32, device=device)
        self._row_bi = wp.zeros((num_envs, max_rows), dtype=wp.int32, device=device)
        self._row_bj = wp.zeros((num_envs, max_rows), dtype=wp.int32, device=device)

        self._max_contacts = max_contacts
        self._max_rows = max_rows
        self._n_pairs = n_pairs
        self._nc_ground = s.nc

        # Contact force sensor: per-sensor-body aggregated world forces
        # Layout: (N, nc_sensor * 3) flattened for wp.atomic_add compatibility
        self._nc_sensor = s.nc  # sensor bodies = contact bodies
        self._contact_force_sensor = wp.zeros((num_envs, max(s.nc * 3, 1)), dtype=wp.float32, device=device)

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
        # Legacy per-body shape arrays (for batched_detect_analytical backward compat)
        self._gpu_shape_type = wp.array(s.body_shape_type, dtype=wp.int32, device=device)
        self._gpu_shape_params = wp.array(s.body_shape_params, dtype=wp.float32, device=device)

        # Flat shape arrays (MuJoCo-style, Q26-gpu multi-shape)
        self._gpu_flat_shape_type = wp.array(s.shape_type, dtype=wp.int32, device=device)
        self._gpu_flat_shape_params = wp.array(s.shape_params, dtype=wp.float32, device=device)
        self._gpu_flat_shape_offset = wp.array(s.shape_offset, dtype=wp.float32, device=device)
        self._gpu_flat_shape_rotation = wp.array(s.shape_rotation, dtype=wp.float32, device=device)
        self._gpu_body_shape_adr = wp.array(s.body_shape_adr, dtype=wp.int32, device=device)
        self._gpu_body_shape_num = wp.array(s.body_shape_num, dtype=wp.int32, device=device)
        self._gpu_collision_excluded = wp.array(s.collision_excluded, dtype=wp.int32, device=device)

        # Body-body collision pairs (legacy, kept for backward compat)
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

    # ── Public state accessors (zero-copy warp arrays) ──

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def q_wp(self):
        """Warp array (N, nq) — current generalized positions."""
        return self._scratch.q

    @property
    def qdot_wp(self):
        """Warp array (N, nv) — current generalized velocities."""
        return self._scratch.qdot

    @property
    def v_bodies_wp(self):
        """Warp array (N, nb, 6) — body spatial velocities after last FK."""
        return self._scratch.v_bodies

    @property
    def x_world_R_wp(self):
        """Warp array (N, nb, 3, 3) — body rotation matrices in world frame."""
        return self._scratch.X_world_R

    @property
    def x_world_r_wp(self):
        """Warp array (N, nb, 3) — body positions in world frame."""
        return self._scratch.X_world_r

    @property
    def contact_active_wp(self):
        """Warp array (N, max_contacts) int32 — contact active flags."""
        return self._contact_active

    @property
    def contact_count_wp(self):
        """Warp array (N,) int32 — active contact count per env."""
        return self._contact_count

    @property
    def contact_force_sensor_wp(self):
        """Warp array (N, nc_sensor*3) flat — per-sensor-body net contact force in world frame.

        Reshape to (N, nc_sensor, 3) via numpy: .numpy().reshape(N, nc_sensor, 3).
        """
        return self._contact_force_sensor

    @property
    def nc_sensor(self) -> int:
        """Number of sensor (contact) bodies."""
        return self._nc_sensor

    def reset(self, q0: np.ndarray | None = None) -> None:
        """Reset all environments to default or given state."""
        N = self._num_envs
        sc = self._scratch
        if q0 is not None:
            self._default_q = np.asarray(q0, dtype=np.float32).ravel()
        q_np = np.tile(self._default_q, (N, 1)).astype(np.float32)
        qdot_np = np.tile(self._default_qdot, (N, 1)).astype(np.float32)
        wp.copy(sc.q, wp.array(q_np, dtype=wp.float32, device=self._device))
        wp.copy(sc.qdot, wp.array(qdot_np, dtype=wp.float32, device=self._device))
        self._clear_warmstart()

    def reset_envs(self, env_ids: np.ndarray, q0: np.ndarray | None = None) -> None:
        """Reset specific environments by index.

        Args:
            env_ids: 1D int array of environment indices to reset.
            q0: Optional (nq,) initial state. If None, uses default_q.
        """
        if len(env_ids) == 0:
            return
        sc = self._scratch
        q_row = (q0 if q0 is not None else self._default_q).astype(np.float32)
        qdot_row = self._default_qdot.copy()

        # Build per-env arrays and scatter
        env_ids_i32 = np.asarray(env_ids, dtype=np.int32)
        n_reset = len(env_ids_i32)
        q_tile = np.tile(q_row, (n_reset, 1))
        qdot_tile = np.tile(qdot_row, (n_reset, 1))

        gpu_ids = wp.array(env_ids_i32, dtype=wp.int32, device=self._device)
        gpu_q_src = wp.array(q_tile, dtype=wp.float32, device=self._device)
        gpu_qdot_src = wp.array(qdot_tile, dtype=wp.float32, device=self._device)

        wp.launch(
            _scatter_reset,
            dim=n_reset,
            inputs=[gpu_ids, gpu_q_src, gpu_qdot_src, self._static.nq, self._static.nv],
            outputs=[sc.q, sc.qdot],
            device=self._device,
        )

        # Clear warmstart for reset envs
        if self._solver == "admm":
            sol = self._solver_scratch
            for arr in [sol.admm_f_prev, sol.admm_s_prev, sol.admm_u_prev]:
                wp.launch(
                    _scatter_zero_2d,
                    dim=n_reset,
                    inputs=[gpu_ids, arr.shape[1]],
                    outputs=[arr],
                    device=self._device,
                )

    def _clear_warmstart(self):
        """Clear ADMM warmstart state for all environments."""
        if self._solver == "admm":
            sol = self._solver_scratch
            sol.admm_f_prev.zero_()
            sol.admm_s_prev.zero_()
            sol.admm_u_prev.zero_()
            sol.admm_prev_n_active.zero_()

    def step_n(self, tau=None, dt=None, n_substeps: int = 1) -> StepOutput:
        """Run n_substeps physics steps, return output once at end.

        More efficient than calling step() in a loop because it avoids
        repeated GPU→CPU copies of StepOutput.

        Args:
            tau: (N, nv) generalized forces (held constant across substeps).
            dt: Time step per substep. If None, uses self._dt.
            n_substeps: Number of physics substeps to run.
        """
        for _ in range(n_substeps):
            self._step_physics(tau=tau, dt=dt)
        return self._make_output()

    def step(self, q=None, qdot=None, tau=None, dt=None):
        """One physics step. If q/qdot/tau not given, uses internal state."""
        sc = self._scratch

        # If external state provided, upload it
        if q is not None:
            q_np = np.atleast_2d(q).astype(np.float32)
            wp.copy(sc.q, wp.array(q_np, dtype=wp.float32, device=self._device))
        if qdot is not None:
            qdot_np = np.atleast_2d(qdot).astype(np.float32)
            wp.copy(sc.qdot, wp.array(qdot_np, dtype=wp.float32, device=self._device))

        self._step_physics(tau=tau, dt=dt)
        return self._make_output()

    def _step_physics(self, tau=None, dt=None):
        """Run one physics substep (no output extraction)."""
        N = self._num_envs
        s = self._static
        sc = self._scratch
        sol = self._solver_scratch
        dt = dt or self._dt

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

        # 5. Collision detection — multi-shape + dynamic N² broadphase (Q26-gpu)
        wp.launch(
            batched_detect_multishape,
            dim=N,
            device=self._device,
            inputs=[
                sc.X_world_R,
                sc.X_world_r,
                self._gpu_flat_shape_type,
                self._gpu_flat_shape_params,
                self._gpu_flat_shape_offset,
                self._gpu_flat_shape_rotation,
                self._gpu_body_shape_adr,
                self._gpu_body_shape_num,
                self._gpu_body_radius,
                self._gpu_contact_body_idx,
                s.contact_ground_z,
                self._nc_ground,
                self._gpu_collision_excluded,
                s.nb,
                0.05,  # broadphase_margin
                self._max_contacts,
            ],
            outputs=[
                self._contact_count,
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
                self._gpu_q_idx_start,
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
                self._contact_depth,
                s.contact_cfm,
                s.solimp_d0,
                s.solimp_dw,
                s.solimp_width,
                s.solimp_mid,
                s.solimp_power,
                s.contact_erp_baumgarte if self._solver != "admm" else 0.0,
                s.contact_slop,
                s.max_depenetration_vel,
                dt,
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

        # 11. Position correction — now handled via Baumgarte velocity bias
        # in batched_build_W_joint_space (step 7), propagated through J^T to
        # all generalized coordinates. No separate position correction pass.
        sol.pos_corrections.zero_()

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

        # 13. Aggregate contact forces for sensor bodies
        if self._nc_sensor > 0:
            self._contact_force_sensor.zero_()
            wp.launch(
                _aggregate_contact_forces,
                dim=N,
                device=self._device,
                inputs=[
                    sol.lambdas,
                    self._contact_active,
                    self._contact_normal,
                    self._contact_bi,
                    self._gpu_contact_body_idx,
                    self._max_contacts,
                    self._nc_sensor,
                    dt,
                ],
                outputs=[self._contact_force_sensor],
            )

        # Update state
        wp.copy(sc.q, sc.q_new)
        wp.copy(sc.qdot, sc.qdot_new)

    def _make_output(self) -> StepOutput:
        """Extract current state into a StepOutput (GPU→CPU copy)."""
        N = self._num_envs
        sc = self._scratch
        q_out = sc.q.numpy().copy()
        qdot_out = sc.qdot.numpy().copy()
        x_world_R_out = sc.X_world_R.numpy().copy()
        x_world_r_out = sc.X_world_r.numpy().copy()
        v_bodies_out = sc.v_bodies.numpy().copy()

        return StepOutput(
            q_new=q_out[0] if N == 1 else q_out,
            qdot_new=qdot_out[0] if N == 1 else qdot_out,
            X_world=(x_world_R_out[0], x_world_r_out[0]) if N == 1 else (x_world_R_out, x_world_r_out),
            v_bodies=v_bodies_out[0] if N == 1 else v_bodies_out,
            contact_active=self._contact_active.numpy().copy(),
            force_state=None,
        )
