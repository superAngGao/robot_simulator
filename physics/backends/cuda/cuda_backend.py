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
        self._contact_body_idx = torch.from_numpy(
            s.contact_body_idx.astype(np.float32)
        ).to(device)
        self._contact_local_pos = torch.from_numpy(s.contact_local_pos).to(device)

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
            self._q += torch.empty_like(self._q).uniform_(
                -init_noise_scale, init_noise_scale
            )
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

        if self._dynamics == "grouped_schur":
            return self._step_grouped_schur(actions, action_clip)
        if self._dynamics == "crba_tc":
            return self._step_crba_tc(actions, action_clip)

        step_fn = self._cuda.physics_step_crba if self._dynamics == "crba" else self._cuda.physics_step
        results = step_fn(
            self._q, self._qdot, actions,
            self._joint_type, self._joint_axis,
            self._parent_idx, self._q_idx_start, self._q_idx_len,
            self._v_idx_start, self._v_idx_len,
            self._X_tree_R, self._X_tree_r, self._inertia_mat,
            self._q_min, self._q_max, self._k_limit, self._b_limit, self._damping,
            self._actuated_q_idx, self._actuated_v_idx, self._effort_limits,
            self._contact_body_idx, self._contact_local_pos,
            N, s.nb, s.nq, s.nv, s.nu, s.nc,
            self._has_effort_limits,
            self._cfg.dt, s.gravity,
            self._cfg.kp, self._cfg.kd, self._cfg.action_scale, action_clip,
            s.contact_k_normal, s.contact_b_normal,
            s.contact_mu, s.contact_slip_eps, s.contact_ground_z,
        )
        q_new, qdot_new, X_world_R, X_world_r, v_bodies, contact_mask = results

        self._q = q_new
        self._qdot = qdot_new

        # Re-run FK on new state for observation cache
        fk_results = self._cuda.fk_only(
            self._q, self._qdot,
            self._joint_type, self._joint_axis, self._parent_idx,
            self._q_idx_start, self._v_idx_start, self._v_idx_len,
            self._X_tree_R, self._X_tree_r,
            N, s.nb, s.nq, s.nv,
        )
        X_world_R, X_world_r, v_bodies = fk_results

        X_world_flat = torch.cat([
            X_world_R.reshape(N, s.nb, 9),
            X_world_r,
        ], dim=-1)

        return StepResult(
            q=self._q.cpu(),
            qdot=self._qdot.cpu(),
            X_world=X_world_flat.cpu(),
            v_bodies=v_bodies.cpu(),
            contact_mask=contact_mask.bool().cpu(),
        )

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

        from physics.robot_tree import RobotTreeNumpy
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
            "limb_v_indices": torch.from_numpy(limb_v_all).int().to(self._device) if len(limb_v_all) > 0 else torch.zeros(1, dtype=torch.int32, device=self._device),
            "ngroups": len(limb_groups),
        }
        return self._group_data

    def _step_grouped_schur(self, actions, action_clip):
        """CRBA with fused grouped Schur complement kernel."""
        s = self._static
        N = self._num_envs
        gd = self._get_group_data()

        results = self._cuda.physics_step_grouped_schur(
            self._q, self._qdot, actions,
            self._joint_type, self._joint_axis, self._parent_idx,
            self._q_idx_start, self._q_idx_len, self._v_idx_start, self._v_idx_len,
            self._X_tree_R, self._X_tree_r, self._inertia_mat,
            self._q_min, self._q_max, self._k_limit, self._b_limit, self._damping,
            self._actuated_q_idx, self._actuated_v_idx, self._effort_limits,
            self._contact_body_idx, self._contact_local_pos,
            gd["root_v"], gd["nv_root"],
            gd["limb_v_offsets"], gd["limb_v_indices"], gd["ngroups"],
            N, s.nb, s.nq, s.nv, s.nu, s.nc, self._has_effort_limits,
            self._cfg.dt, s.gravity,
            self._cfg.kp, self._cfg.kd, self._cfg.action_scale, action_clip,
            s.contact_k_normal, s.contact_b_normal, s.contact_mu, s.contact_slip_eps, s.contact_ground_z,
        )
        q_new, qdot_new, X_world_R, X_world_r, v_bodies, contact_mask = results
        self._q = q_new
        self._qdot = qdot_new

        fk = self._cuda.fk_only(
            self._q, self._qdot,
            self._joint_type, self._joint_axis, self._parent_idx,
            self._q_idx_start, self._v_idx_start, self._v_idx_len,
            self._X_tree_R, self._X_tree_r, N, s.nb, s.nq, s.nv,
        )
        X_world_R, X_world_r, v_bodies = fk
        X_flat = torch.cat([X_world_R.reshape(N, s.nb, 9), X_world_r], dim=-1)
        return StepResult(
            q=self._q.cpu(), qdot=self._qdot.cpu(),
            X_world=X_flat.cpu(), v_bodies=v_bodies.cpu(),
            contact_mask=contact_mask.bool().cpu(),
        )

    def _step_crba_tc(self, actions, action_clip):
        """CRBA with tensor-core Cholesky: split kernel + torch.linalg.cholesky_solve."""
        s = self._static
        N = self._num_envs

        # Kernel 1: fused FK + CRBA(H) + RNEA(C) → H, rhs, FK cache
        results = self._cuda.crba_build(
            self._q, self._qdot, actions,
            self._joint_type, self._joint_axis, self._parent_idx,
            self._q_idx_start, self._q_idx_len, self._v_idx_start, self._v_idx_len,
            self._X_tree_R, self._X_tree_r, self._inertia_mat,
            self._q_min, self._q_max, self._k_limit, self._b_limit, self._damping,
            self._actuated_q_idx, self._actuated_v_idx, self._effort_limits,
            self._contact_body_idx, self._contact_local_pos,
            N, s.nb, s.nq, s.nv, s.nu, s.nc, self._has_effort_limits,
            s.gravity, self._cfg.kp, self._cfg.kd, self._cfg.action_scale, action_clip,
            s.contact_k_normal, s.contact_b_normal, s.contact_mu, s.contact_slip_eps, s.contact_ground_z,
        )
        H, rhs, X_world_R, X_world_r, v_bodies, contact_mask = results

        # Cholesky solve via cuSOLVER (uses wgmma on Hopper)
        L = torch.linalg.cholesky(H)
        qddot = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)

        # Kernel 2: integration
        int_results = self._cuda.integrate_step(
            self._q, self._qdot, qddot,
            self._joint_type, self._q_idx_start, self._q_idx_len,
            self._v_idx_start, self._v_idx_len,
            self._cfg.dt, N, s.nb, s.nq, s.nv,
        )
        self._q, self._qdot = int_results[0], int_results[1]

        # Re-run FK
        fk = self._cuda.fk_only(
            self._q, self._qdot,
            self._joint_type, self._joint_axis, self._parent_idx,
            self._q_idx_start, self._v_idx_start, self._v_idx_len,
            self._X_tree_R, self._X_tree_r,
            N, s.nb, s.nq, s.nv,
        )
        X_world_R, X_world_r, v_bodies = fk

        X_flat = torch.cat([X_world_R.reshape(N, s.nb, 9), X_world_r], dim=-1)
        return StepResult(
            q=self._q.cpu(), qdot=self._qdot.cpu(),
            X_world=X_flat.cpu(), v_bodies=v_bodies.cpu(),
            contact_mask=contact_mask.bool().cpu(),
        )

    def _run_fk_and_build(self) -> StepResult:
        s = self._static
        N = self._num_envs
        fk = self._cuda.fk_only(
            self._q, self._qdot,
            self._joint_type, self._joint_axis, self._parent_idx,
            self._q_idx_start, self._v_idx_start, self._v_idx_len,
            self._X_tree_R, self._X_tree_r,
            N, s.nb, s.nq, s.nv,
        )
        X_world_R, X_world_r, v_bodies = fk
        X_world_flat = torch.cat([
            X_world_R.reshape(N, s.nb, 9), X_world_r
        ], dim=-1)
        return StepResult(
            q=self._q.cpu(),
            qdot=self._qdot.cpu(),
            X_world=X_world_flat.cpu(),
            v_bodies=v_bodies.cpu(),
            contact_mask=torch.zeros(N, s.nc, dtype=torch.bool),
        )
