"""
Tests for Phase 2d — RL Environment Layer.

Covers:
  1. ObsManager output shape
  2. Noise applied in train mode
  3. Noise NOT applied in eval mode
  4. Env.step() returns finite obs with correct shape
  5. VecEnv N=4 step obs shape = (4, obs_dim)
  6. PDController effort_limit clips tau
"""

from __future__ import annotations

import numpy as np
import torch

from physics.collision import NullSelfCollision
from physics.contact import NullContactModel
from physics.joint import FreeJoint, RevoluteJoint
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from rl_env import Env, EnvCfg, NoiseCfg, ObsTermCfg, PDController, VecEnv, obs_terms
from robot.model import RobotModel

# ---------------------------------------------------------------------------
# Minimal robot fixture: floating base + 2 revolute joints
# ---------------------------------------------------------------------------


def _make_model() -> RobotModel:
    tree = RobotTreeNumpy(gravity=9.81)

    torso = Body(
        name="torso",
        index=0,
        joint=FreeJoint("root"),
        inertia=SpatialInertia.from_box(5.0, 0.3, 0.2, 0.1),
        X_tree=SpatialTransform.identity(),
        parent=-1,
    )
    torso_idx = tree.add_body(torso)

    link1 = Body(
        name="link1",
        index=0,
        joint=RevoluteJoint("j1", axis=np.array([0.0, 1.0, 0.0])),
        inertia=SpatialInertia.from_cylinder(0.5, 0.02, 0.2),
        X_tree=SpatialTransform(np.eye(3), np.array([0.0, 0.0, -0.15])),
        parent=torso_idx,
    )
    link1_idx = tree.add_body(link1)

    link2 = Body(
        name="link2",
        index=0,
        joint=RevoluteJoint("j2", axis=np.array([0.0, 1.0, 0.0])),
        inertia=SpatialInertia.from_cylinder(0.3, 0.015, 0.2),
        X_tree=SpatialTransform(np.eye(3), np.array([0.0, 0.0, -0.2])),
        parent=link1_idx,
    )
    tree.add_body(link2)
    tree.finalize()

    return RobotModel(
        tree=tree,
        contact_model=NullContactModel(),
        self_collision=NullSelfCollision(),
        actuated_joint_names=["j1", "j2"],
        contact_body_names=[],
    )


def _make_cfg(noise_std: float = 0.0) -> EnvCfg:
    noise = NoiseCfg(noise_type="gaussian", std=noise_std) if noise_std > 0 else None
    return EnvCfg(
        dt=2e-4,
        episode_length=100,
        obs_cfg={
            "joint_pos": ObsTermCfg(func=obs_terms.joint_pos, noise=noise),
            "joint_vel": ObsTermCfg(func=obs_terms.joint_vel),
        },
        kp=20.0,
        kd=0.5,
        action_scale=0.5,
    )


# ---------------------------------------------------------------------------
# 1. ObsManager output shape
# ---------------------------------------------------------------------------


def test_obs_manager_shape():
    model = _make_model()
    cfg = _make_cfg()
    env = Env(model, cfg)
    env.reset()
    obs = env.obs_manager.compute()
    # joint_pos (2,) + joint_vel (2,) = 4
    assert obs.shape == (4,), f"Expected (4,), got {obs.shape}"


# ---------------------------------------------------------------------------
# 2. Noise applied in train mode
# ---------------------------------------------------------------------------


def test_obs_noise_applied_in_train():
    model = _make_model()
    cfg = _make_cfg(noise_std=1.0)
    env = Env(model, cfg)
    env.reset()
    env.obs_manager.train()

    torch.manual_seed(0)
    obs_noisy = env.obs_manager.compute()

    # Compute noiseless reference via eval mode
    env.obs_manager.eval()
    obs_clean = env.obs_manager.compute()

    assert not torch.allclose(obs_noisy, obs_clean), (
        "Train-mode obs should differ from clean obs when noise_std=1.0"
    )


# ---------------------------------------------------------------------------
# 3. Noise NOT applied in eval mode
# ---------------------------------------------------------------------------


def test_obs_noise_not_applied_in_eval():
    model = _make_model()
    cfg = _make_cfg(noise_std=1.0)
    env = Env(model, cfg)
    env.reset()
    env.obs_manager.eval()

    obs1 = env.obs_manager.compute()
    obs2 = env.obs_manager.compute()
    assert torch.allclose(obs1, obs2), "Eval-mode obs should be deterministic (no noise)"


# ---------------------------------------------------------------------------
# 4. Env.step() returns finite obs with correct shape
# ---------------------------------------------------------------------------


def test_env_step_returns_valid():
    model = _make_model()
    cfg = _make_cfg()
    env = Env(model, cfg)
    env.reset()

    nu = len(env.actuated_q_indices)
    action = torch.zeros(nu, dtype=torch.float32)
    obs, rew, term, trunc, info = env.step(action)

    assert obs.shape == (4,), f"Expected (4,), got {obs.shape}"
    assert torch.all(torch.isfinite(obs)), "obs contains non-finite values"


# ---------------------------------------------------------------------------
# 5. VecEnv N=4 step obs shape = (4, obs_dim)
# ---------------------------------------------------------------------------


def test_vec_env_step():
    model = _make_model()
    cfg = _make_cfg()
    vec = VecEnv(model, cfg, num_envs=4)
    vec.reset()

    nu = len(vec.envs[0].actuated_q_indices)
    actions = torch.zeros(4, nu, dtype=torch.float32)
    obs, rew, term, trunc, info = vec.step(actions)

    assert obs.shape == (4, 4), f"Expected (4, 4), got {obs.shape}"
    assert rew.shape == (4,)
    assert term.shape == (4,)
    assert trunc.shape == (4,)


# ---------------------------------------------------------------------------
# 6. PDController effort_limit clips tau
# ---------------------------------------------------------------------------


def test_effort_limit_clips_tau():
    effort_limits = np.array([10.0, 10.0])
    actuated_q = np.array([7, 8], dtype=np.intp)  # after FreeJoint (nq=7)
    actuated_v = np.array([6, 7], dtype=np.intp)  # after FreeJoint (nv=6)
    nv = 8

    ctrl = PDController(
        kp=1000.0,  # large gain → large tau before clipping
        kd=0.0,
        action_scale=1.0,
        actuated_q_indices=actuated_q,
        actuated_v_indices=actuated_v,
        nv=nv,
        effort_limits=effort_limits,
    )

    q = np.zeros(nv + 1)  # nq = nv+1 for FreeJoint
    qdot = np.zeros(nv)
    action = np.array([1.0, 1.0])  # large offset → tau >> 10 before clip

    tau = ctrl.compute(action, q, qdot)

    assert np.all(np.abs(tau[actuated_v]) <= 10.0 + 1e-9), f"tau not clipped: {tau[actuated_v]}"
