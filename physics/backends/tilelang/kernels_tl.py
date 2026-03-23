"""
TileLang GPU kernels for batched robot simulation.

Uses TileLang (@tilelang.jit) for fused element-wise kernels and
PyTorch CUDA tensor operations for tree-traversal algorithms (FK, ABA).

TileLang kernels require static shapes at compile time, so kernel
factories accept (N, nb, nq, nv, ...) and cache compiled kernels.
"""

from __future__ import annotations

import functools
import math

import tilelang
import tilelang.language as T
import torch


# ---------------------------------------------------------------------------
# Spatial math helpers (pure PyTorch, batched)
# ---------------------------------------------------------------------------


def rodrigues_torch(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation. axis: (3,), angle: (N,) -> R: (N, 3, 3)."""
    N = angle.shape[0]
    k = axis.unsqueeze(0).expand(N, 3)  # (N, 3)
    c = torch.cos(angle).unsqueeze(-1).unsqueeze(-1)  # (N, 1, 1)
    s = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)

    K = torch.zeros(N, 3, 3, device=angle.device, dtype=angle.dtype)
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]

    kkT = torch.einsum("ni,nj->nij", k, k)  # (N, 3, 3)
    I = torch.eye(3, device=angle.device, dtype=angle.dtype).unsqueeze(0)
    R = c * I + s * K + (1.0 - c) * kkT
    return R


def quat_to_rot_torch(quat: torch.Tensor) -> torch.Tensor:
    """Quaternion (N, 4) [w,x,y,z] -> rotation matrix (N, 3, 3)."""
    q = quat / quat.norm(dim=-1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
        2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)
    return R


def transform_velocity_torch(R: torch.Tensor, r: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Plücker velocity transform: parent -> child. R: (N,3,3), r: (N,3), v: (N,6) -> (N,6)."""
    E = R.transpose(-1, -2)  # (N, 3, 3)
    v_lin = v[:, :3]  # (N, 3)
    v_ang = v[:, 3:]  # (N, 3)
    lin_new = torch.bmm(E, (v_lin + torch.cross(v_ang, r, dim=-1)).unsqueeze(-1)).squeeze(-1)
    ang_new = torch.bmm(E, v_ang.unsqueeze(-1)).squeeze(-1)
    return torch.cat([lin_new, ang_new], dim=-1)


def transform_force_torch(R: torch.Tensor, r: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Plücker force dual transform: child -> parent. R: (N,3,3), r: (N,3), f: (N,6) -> (N,6)."""
    f_lin = f[:, :3]
    f_ang = f[:, 3:]
    Rf = torch.bmm(R, f_lin.unsqueeze(-1)).squeeze(-1)
    lin_new = Rf
    ang_new = torch.bmm(R, f_ang.unsqueeze(-1)).squeeze(-1) + torch.cross(r, Rf, dim=-1)
    return torch.cat([lin_new, ang_new], dim=-1)


def spatial_cross_force_torch(v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """v ×* f (force cross product). v: (N,6), f: (N,6) -> (N,6)."""
    v_lin, v_ang = v[:, :3], v[:, 3:]
    f_lin, f_ang = f[:, :3], f[:, 3:]
    res_lin = torch.cross(v_ang, f_lin, dim=-1)
    res_ang = torch.cross(v_lin, f_lin, dim=-1) + torch.cross(v_ang, f_ang, dim=-1)
    return torch.cat([res_lin, res_ang], dim=-1)


def spatial_cross_vel_torch(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """v ×_vel u. v: (N,6), u: (N,6) -> (N,6)."""
    v_lin, v_ang = v[:, :3], v[:, 3:]
    u_lin, u_ang = u[:, :3], u[:, 3:]
    res_lin = torch.cross(v_ang, u_lin, dim=-1) + torch.cross(v_lin, u_ang, dim=-1)
    res_ang = torch.cross(v_ang, u_ang, dim=-1)
    return torch.cat([res_lin, res_ang], dim=-1)


def spatial_transform_matrix_torch(R: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Build 6x6 Plücker velocity transform matrix. R:(N,3,3), r:(N,3) -> (N,6,6)."""
    N = R.shape[0]
    E = R.transpose(-1, -2)  # (N, 3, 3)
    # skew(r)
    zero = torch.zeros(N, device=R.device, dtype=R.dtype)
    skew_r = torch.stack([
        zero, -r[:, 2], r[:, 1],
        r[:, 2], zero, -r[:, 0],
        -r[:, 1], r[:, 0], zero,
    ], dim=-1).reshape(N, 3, 3)
    E_skew_r = torch.bmm(E, skew_r)

    M = torch.zeros(N, 6, 6, device=R.device, dtype=R.dtype)
    M[:, :3, :3] = E
    M[:, :3, 3:] = -E_skew_r
    M[:, 3:, 3:] = E
    return M


# ---------------------------------------------------------------------------
# TileLang fused kernels
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=16)
def _make_passive_torques_kernel(N: int, nb: int, nv: int):
    """TileLang kernel for batched passive torques."""
    # We process N*nb elements, each body contributes 0 or 1 torque value
    total = N * nb

    @tilelang.jit(out_idx=[6])
    def kernel():
        @T.prim_func
        def main(
            q: T.Tensor((N, nv), "float32"),         # (not used for nq, but we pass relevant slices)
            qdot: T.Tensor((N, nv), "float32"),
            q_body: T.Tensor((N, nb), "float32"),     # pre-extracted q per body
            jtype: T.Tensor((nb,), "int32"),
            limits: T.Tensor((nb, 5), "float32"),     # [q_min, q_max, k, b, damping]
            v_idx: T.Tensor((nb,), "int32"),
            tau: T.Tensor((N, nv), "float32"),
        ):
            with T.Kernel(total, threads=1) as tid:
                env_id = tid // nb
                body_id = tid % nb
                jt = jtype[body_id]
                vi = v_idx[body_id]
                # Revolute = 1
                if jt == 1:
                    angle = q_body[env_id, body_id]
                    omega = qdot[env_id, vi]
                    qmin = limits[body_id, 0]
                    qmax = limits[body_id, 1]
                    k_lim = limits[body_id, 2]
                    b_lim = limits[body_id, 3]
                    damp = limits[body_id, 4]

                    tau_val = T.float32(0.0)
                    if angle < qmin:
                        pen = qmin - angle
                        d = T.min(omega, T.float32(0.0))
                        tau_val = k_lim * pen - b_lim * d
                    if angle > qmax:
                        pen = angle - qmax
                        d = T.max(omega, T.float32(0.0))
                        tau_val = -(k_lim * pen + b_lim * d)
                    tau_val = tau_val - damp * omega
                    tau[env_id, vi] = tau_val
                # Prismatic = 2
                if jt == 2:
                    omega = qdot[env_id, vi]
                    damp = limits[body_id, 4]
                    tau[env_id, vi] = -damp * omega

        return main
    return kernel


@functools.lru_cache(maxsize=16)
def _make_pd_controller_kernel(N: int, nu: int, nv: int):
    """TileLang kernel for batched PD controller."""
    total = N * nu

    @tilelang.jit(out_idx=[5])
    def kernel(kp_val: T.float32, kd_val: T.float32, scale_val: T.float32):
        @T.prim_func
        def main(
            actions: T.Tensor((N, nu), "float32"),
            q: T.Tensor((N, nv), "float32"),
            qdot: T.Tensor((N, nv), "float32"),
            aq_idx: T.Tensor((nu,), "int32"),
            av_idx: T.Tensor((nu,), "int32"),
            tau: T.Tensor((N, nv), "float32"),
        ):
            with T.Kernel(total, threads=1) as tid:
                env_id = tid // nu
                j = tid % nu
                qi = aq_idx[j]
                vi = av_idx[j]
                act = actions[env_id, j]
                target = q[env_id, qi] + act * scale_val
                tau_val = kp_val * (target - q[env_id, qi]) - kd_val * qdot[env_id, vi]
                tau[env_id, vi] = tau_val

        return main
    return kernel


@functools.lru_cache(maxsize=16)
def _make_integrate_kernel(N: int, nq: int, nv: int):
    """TileLang kernel for semi-implicit Euler (non-quaternion joints only).

    Quaternion integration is handled separately in PyTorch.
    """
    total = N * nv

    @tilelang.jit(out_idx=[3, 4])
    def kernel(dt_val: T.float32):
        @T.prim_func
        def main(
            q: T.Tensor((N, nq), "float32"),
            qdot: T.Tensor((N, nv), "float32"),
            qddot: T.Tensor((N, nv), "float32"),
            q_new: T.Tensor((N, nq), "float32"),
            qdot_new: T.Tensor((N, nv), "float32"),
        ):
            with T.Kernel(total, threads=1) as tid:
                env_id = tid // nv
                j = tid % nv
                qdot_new[env_id, j] = qdot[env_id, j] + dt_val * qddot[env_id, j]

        return main
    return kernel
