"""
Tests for Phase 2d — RL Environment Layer.

Covers:
  1. ObsManager output shape
  2. Noise applied in train mode
  3. Noise NOT applied in eval mode
  4. Env.step() returns finite obs with correct shape
  5. VecEnv N=4 step obs shape = (4, obs_dim)
  6. PDController effort_limit clips tau
  7. TorqueController pass-through and effort clipping
  8. obs_terms individual term functions
  9. Env action_clip
 10. Env episode truncation
 11. Env init_noise_scale
 12. PDController simplifies to kp*action_scale*action for zero state
 13. VecEnv reset returns correct shape
 14. ObsManager with base_lin_vel / base_ang_vel terms
"""

from __future__ import annotations

import numpy as np
import torch

from physics.joint import FreeJoint, RevoluteJoint
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from rl_env import Env, EnvCfg, NoiseCfg, ObsTermCfg, PDController, VecEnv, obs_terms
from rl_env.controllers import TorqueController
from robot.model import RobotModel

# ---------------------------------------------------------------------------
# Minimal robot fixture: floating base + 2 revolute joints
# ---------------------------------------------------------------------------


def _make_model(with_contacts=False) -> RobotModel:
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

    contact_body_names = ["foot"] if with_contacts else []

    return RobotModel(
        tree=tree,
        actuated_joint_names=["j1", "j2"],
        contact_body_names=contact_body_names,
    )


def _make_cfg(noise_std: float = 0.0, **kwargs) -> EnvCfg:
    noise = NoiseCfg(noise_type="gaussian", std=noise_std) if noise_std > 0 else None
    defaults = dict(
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
    defaults.update(kwargs)
    return EnvCfg(**defaults)


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

    nu = len(model.actuated_joint_names)
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


# ===========================================================================
# NEW TESTS
# ===========================================================================


# ---------------------------------------------------------------------------
# 7. TorqueController — pass-through
# ---------------------------------------------------------------------------


def test_torque_controller_passthrough():
    """TorqueController without effort limits passes action directly to tau."""
    actuated_v = np.array([6, 7], dtype=np.intp)
    nv = 8
    ctrl = TorqueController(actuated_v_indices=actuated_v, nv=nv)

    q = np.zeros(9)
    qdot = np.zeros(nv)
    action = np.array([3.5, -2.1])

    tau = ctrl.compute(action, q, qdot)
    assert tau.shape == (nv,)
    np.testing.assert_allclose(tau[actuated_v], action, atol=1e-12)
    # Non-actuated joints should be zero
    mask = np.ones(nv, dtype=bool)
    mask[actuated_v] = False
    np.testing.assert_allclose(tau[mask], 0.0, atol=1e-12)


def test_torque_controller_effort_clip():
    """TorqueController clips to effort limits."""
    effort_limits = np.array([5.0, 5.0])
    actuated_v = np.array([6, 7], dtype=np.intp)
    nv = 8
    ctrl = TorqueController(actuated_v_indices=actuated_v, nv=nv, effort_limits=effort_limits)

    q = np.zeros(9)
    qdot = np.zeros(nv)
    action = np.array([100.0, -100.0])

    tau = ctrl.compute(action, q, qdot)
    np.testing.assert_allclose(tau[6], 5.0, atol=1e-12)
    np.testing.assert_allclose(tau[7], -5.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 8. PDController at zero state → tau = kp * action_scale * action
# ---------------------------------------------------------------------------


def test_pd_controller_zero_state():
    """At q=0, qdot=0: PD output = kp * action_scale * action."""
    kp, kd, action_scale = 20.0, 0.5, 0.5
    actuated_q = np.array([7, 8], dtype=np.intp)
    actuated_v = np.array([6, 7], dtype=np.intp)
    nv = 8

    ctrl = PDController(
        kp=kp,
        kd=kd,
        action_scale=action_scale,
        actuated_q_indices=actuated_q,
        actuated_v_indices=actuated_v,
        nv=nv,
    )

    q = np.zeros(9)
    qdot = np.zeros(nv)
    action = np.array([1.0, -0.5])

    tau = ctrl.compute(action, q, qdot)
    expected = kp * action_scale * action  # kd*0 = 0
    np.testing.assert_allclose(tau[actuated_v], expected, atol=1e-12)


# ---------------------------------------------------------------------------
# 9. obs_terms — individual term functions
# ---------------------------------------------------------------------------


class _FakeEnv:
    """Minimal mock of Env for testing obs_terms independently."""

    def __init__(self, model):
        tree = model.tree
        self.q, self.qdot = tree.default_state()
        self.root_body_idx = 0
        root = tree.bodies[0]
        self.root_q_slice = root.q_idx
        actuated = [b for b in tree.bodies if b.joint.nv > 0 and not isinstance(b.joint, FreeJoint)]
        self.actuated_q_indices = np.array(
            [i for b in actuated for i in range(b.q_idx.start, b.q_idx.stop)], dtype=np.intp
        )
        self.actuated_v_indices = np.array(
            [i for b in actuated for i in range(b.v_idx.start, b.v_idx.stop)], dtype=np.intp
        )
        self.v_bodies = tree.body_velocities(self.q, self.qdot)
        self.active_contacts = []
        self.contact_body_names = list(model.contact_body_names)


def test_obs_base_lin_vel_shape():
    model = _make_model()
    env = _FakeEnv(model)
    result = obs_terms.base_lin_vel(env)
    assert result.shape == (3,)
    assert result.dtype == torch.float32


def test_obs_base_ang_vel_shape():
    model = _make_model()
    env = _FakeEnv(model)
    result = obs_terms.base_ang_vel(env)
    assert result.shape == (3,)
    assert result.dtype == torch.float32


def test_obs_base_orientation_shape():
    model = _make_model()
    env = _FakeEnv(model)
    result = obs_terms.base_orientation(env)
    assert result.shape == (4,)


def test_obs_joint_pos_shape():
    model = _make_model()
    env = _FakeEnv(model)
    result = obs_terms.joint_pos(env)
    assert result.shape == (2,)  # 2 actuated joints


def test_obs_joint_vel_shape():
    model = _make_model()
    env = _FakeEnv(model)
    result = obs_terms.joint_vel(env)
    assert result.shape == (2,)


def test_obs_contact_mask_all_inactive():
    """With NullContactModel, contact mask should be empty."""
    model = _make_model(with_contacts=False)
    env = _FakeEnv(model)
    result = obs_terms.contact_mask(env)
    assert result.shape == (0,)


def test_obs_contact_mask_shape_with_contacts():
    """With one contact body, mask shape is (1,)."""
    model = _make_model(with_contacts=True)
    env = _FakeEnv(model)
    result = obs_terms.contact_mask(env)
    assert result.shape == (1,)
    # Default state (in air) → inactive
    assert result[0] == 0.0


def test_obs_base_lin_vel_zero_at_rest():
    """At rest, base linear velocity should be zero."""
    model = _make_model()
    env = _FakeEnv(model)
    result = obs_terms.base_lin_vel(env)
    np.testing.assert_allclose(result.numpy(), 0.0, atol=1e-10)


def test_obs_base_ang_vel_zero_at_rest():
    """At rest, base angular velocity should be zero."""
    model = _make_model()
    env = _FakeEnv(model)
    result = obs_terms.base_ang_vel(env)
    np.testing.assert_allclose(result.numpy(), 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 10. Env action_clip
# ---------------------------------------------------------------------------


def test_action_clip():
    """Actions are clipped to [-action_clip, +action_clip]."""
    model = _make_model()
    cfg = _make_cfg(action_clip=0.1)
    env = Env(model, cfg)
    env.reset()

    # Large action should be clipped
    large_action = torch.tensor([10.0, -10.0], dtype=torch.float32)
    obs, _, _, _, _ = env.step(large_action)
    # After clip, effective action is [-0.1, 0.1] → tau should be small
    assert torch.all(torch.isfinite(obs))


# ---------------------------------------------------------------------------
# 11. Env episode truncation
# ---------------------------------------------------------------------------


def test_episode_truncation():
    """After episode_length steps, truncated should be True."""
    model = _make_model()
    cfg = _make_cfg(episode_length=5)
    env = Env(model, cfg)
    env.reset()

    nu = len(env.actuated_q_indices)
    action = torch.zeros(nu, dtype=torch.float32)

    for i in range(4):
        _, _, _, trunc, _ = env.step(action)
        assert not trunc, f"Should not be truncated at step {i + 1}"

    _, _, _, trunc, _ = env.step(action)
    assert trunc, "Should be truncated at step 5 (episode_length=5)"


# ---------------------------------------------------------------------------
# 12. Env init_noise_scale
# ---------------------------------------------------------------------------


def test_init_noise_randomizes_state():
    """With init_noise_scale > 0, reset produces different q each time."""
    model = _make_model()
    cfg = _make_cfg(init_noise_scale=0.1)
    env = Env(model, cfg)

    np.random.seed(42)
    env.reset()
    q1 = env.q.copy()

    np.random.seed(99)
    env.reset()
    q2 = env.q.copy()

    assert not np.allclose(q1, q2), "Different seeds should produce different initial states"


# ---------------------------------------------------------------------------
# 13. VecEnv reset shape
# ---------------------------------------------------------------------------


def test_vec_env_reset_shape():
    model = _make_model()
    cfg = _make_cfg()
    vec = VecEnv(model, cfg, num_envs=3)
    obs, infos = vec.reset()
    assert obs.shape == (3, 4), f"Expected (3, 4), got {obs.shape}"
    assert len(infos) == 3


# ---------------------------------------------------------------------------
# 14. ObsManager with base velocity terms
# ---------------------------------------------------------------------------


def test_obs_manager_with_velocity_terms():
    """ObsManager with base_lin_vel and base_ang_vel produces correct total dim."""
    model = _make_model()
    cfg = EnvCfg(
        dt=2e-4,
        episode_length=100,
        obs_cfg={
            "base_lin_vel": ObsTermCfg(func=obs_terms.base_lin_vel),
            "base_ang_vel": ObsTermCfg(func=obs_terms.base_ang_vel),
            "joint_pos": ObsTermCfg(func=obs_terms.joint_pos),
        },
        kp=20.0,
        kd=0.5,
        action_scale=0.5,
    )
    env = Env(model, cfg)
    env.reset()
    obs = env.obs_manager.compute()
    # 3 + 3 + 2 = 8
    assert obs.shape == (8,), f"Expected (8,), got {obs.shape}"


# ---------------------------------------------------------------------------
# 15. Uniform noise
# ---------------------------------------------------------------------------


def test_uniform_noise():
    """NoiseCfg with uniform noise works and differs from clean obs."""
    model = _make_model()
    noise = NoiseCfg(noise_type="uniform", low=-1.0, high=1.0)
    cfg = EnvCfg(
        dt=2e-4,
        episode_length=100,
        obs_cfg={
            "joint_pos": ObsTermCfg(func=obs_terms.joint_pos, noise=noise),
        },
        kp=20.0,
        kd=0.5,
        action_scale=0.5,
    )
    env = Env(model, cfg)
    env.reset()
    env.obs_manager.train()

    obs_noisy = env.obs_manager.compute()
    env.obs_manager.eval()
    obs_clean = env.obs_manager.compute()

    # With range [-1, 1], very likely to differ
    assert not torch.allclose(obs_noisy, obs_clean)


# ---------------------------------------------------------------------------
# 16. PDController with damping
# ---------------------------------------------------------------------------


def test_pd_controller_damping():
    """kd term produces negative torque opposing velocity."""
    kp, kd, action_scale = 0.0, 10.0, 1.0
    actuated_q = np.array([7, 8], dtype=np.intp)
    actuated_v = np.array([6, 7], dtype=np.intp)
    nv = 8

    ctrl = PDController(
        kp=kp,
        kd=kd,
        action_scale=action_scale,
        actuated_q_indices=actuated_q,
        actuated_v_indices=actuated_v,
        nv=nv,
    )

    q = np.zeros(9)
    qdot = np.zeros(nv)
    qdot[6] = 2.0  # joint 1 velocity
    qdot[7] = -1.0  # joint 2 velocity
    action = np.zeros(2)

    tau = ctrl.compute(action, q, qdot)
    # tau = kp*0 - kd*qdot = -10*[2, -1] = [-20, 10]
    np.testing.assert_allclose(tau[6], -20.0, atol=1e-12)
    np.testing.assert_allclose(tau[7], 10.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 17. Env.reset resets step count
# ---------------------------------------------------------------------------


def test_env_reset_resets_step_count():
    """After reset, step count starts at 0 so truncation is fresh."""
    model = _make_model()
    cfg = _make_cfg(episode_length=3)
    env = Env(model, cfg)
    env.reset()

    action = torch.zeros(2, dtype=torch.float32)
    for _ in range(3):
        env.step(action)

    # Reset and step again — should not truncate immediately
    env.reset()
    _, _, _, trunc, _ = env.step(action)
    assert not trunc, "Should not truncate on first step after reset"
