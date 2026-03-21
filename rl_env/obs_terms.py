"""
Standard observation term functions for the RL environment.

Each function has signature:
    fn(env, **params) -> torch.Tensor  shape (dim,)

All reads come from env cached attributes (no FK recomputation).
Phase 2e: swap env.q / env.qdot / env.v_bodies for Warp arrays — signatures unchanged.
"""

from __future__ import annotations

import torch


def base_lin_vel(env, **params) -> torch.Tensor:
    """Linear velocity of the root body in body frame. Shape (3,)."""
    return torch.tensor(env.v_bodies[env.root_body_idx][:3], dtype=torch.float32)


def base_ang_vel(env, **params) -> torch.Tensor:
    """Angular velocity of the root body in body frame. Shape (3,)."""
    return torch.tensor(env.v_bodies[env.root_body_idx][3:6], dtype=torch.float32)


def base_orientation(env, **params) -> torch.Tensor:
    """Root body quaternion [qx, qy, qz, qw]. Shape (4,)."""
    return torch.tensor(env.q[env.root_q_slice][:4], dtype=torch.float32)


def joint_pos(env, **params) -> torch.Tensor:
    """Actuated joint positions. Shape (nu,)."""
    return torch.tensor(env.q[env.actuated_q_indices], dtype=torch.float32)


def joint_vel(env, **params) -> torch.Tensor:
    """Actuated joint velocities. Shape (nu,)."""
    return torch.tensor(env.qdot[env.actuated_v_indices], dtype=torch.float32)


def contact_mask(env, **params) -> torch.Tensor:
    """Binary contact mask for each contact body. Shape (n_feet,).

    Reads env.active_contacts (list of (name, force) tuples from last step).
    """
    active_names = {name for name, _ in env.active_contacts}
    mask = [1.0 if name in active_names else 0.0 for name in env.contact_body_names]
    return torch.tensor(mask, dtype=torch.float32)
