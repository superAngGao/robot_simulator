"""
Tests for WarpBatchBackend — numerical validation against NumpyLoopBackend.

Compares Warp (float32 GPU) output against NumPy (float64 CPU) reference
for identical inputs. Tolerances account for float32 vs float64 precision.
"""

from __future__ import annotations

import os
import tempfile
import textwrap

import numpy as np
import pytest
import torch

wp = pytest.importorskip("warp")

from physics.backends import get_backend  # noqa: E402
from rl_env.cfg import EnvCfg  # noqa: E402
from robot import load_urdf  # noqa: E402

ATOL_SINGLE = 1e-4  # single step tolerance (float32 vs float64)
ATOL_MULTI = 5e-3  # multi-step trajectory tolerance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _quadruped_urdf() -> str:
    return """
    <robot name="quad">
      <link name="base">
        <inertial>
          <mass value="5.0"/><origin xyz="0 0 0" rpy="0 0 0"/>
          <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.05"/>
        </inertial>
        <collision><geometry><box size="0.35 0.20 0.10"/></geometry></collision>
      </link>
      <link name="FL_hip"><inertial><mass value="0.5"/><origin xyz="0 0 -0.1"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
      <joint name="FL_hip_j" type="revolute"><parent link="base"/><child link="FL_hip"/>
        <origin xyz="0.15 0.1 0"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="20"/></joint>
      <link name="FL_calf"><inertial><mass value="0.3"/><origin xyz="0 0 -0.1"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
      <joint name="FL_calf_j" type="revolute"><parent link="FL_hip"/><child link="FL_calf"/>
        <origin xyz="0 0 -0.2"/><axis xyz="0 1 0"/><limit lower="-2" upper="0.5" effort="20"/></joint>
      <link name="FL_foot"><inertial><mass value="0.05"/><origin xyz="0 0 0"/>
        <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial></link>
      <joint name="FL_foot_j" type="fixed"><parent link="FL_calf"/><child link="FL_foot"/>
        <origin xyz="0 0 -0.2"/></joint>
      <link name="FR_hip"><inertial><mass value="0.5"/><origin xyz="0 0 -0.1"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
      <joint name="FR_hip_j" type="revolute"><parent link="base"/><child link="FR_hip"/>
        <origin xyz="0.15 -0.1 0"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="20"/></joint>
      <link name="FR_calf"><inertial><mass value="0.3"/><origin xyz="0 0 -0.1"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
      <joint name="FR_calf_j" type="revolute"><parent link="FR_hip"/><child link="FR_calf"/>
        <origin xyz="0 0 -0.2"/><axis xyz="0 1 0"/><limit lower="-2" upper="0.5" effort="20"/></joint>
      <link name="FR_foot"><inertial><mass value="0.05"/><origin xyz="0 0 0"/>
        <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial></link>
      <joint name="FR_foot_j" type="fixed"><parent link="FR_calf"/><child link="FR_foot"/>
        <origin xyz="0 0 -0.2"/></joint>
    </robot>
    """


@pytest.fixture(scope="module")
def model():
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
    f.write(textwrap.dedent(_quadruped_urdf()))
    f.close()
    m = load_urdf(f.name, floating_base=True, contact_links=["FL_foot", "FR_foot"])
    os.unlink(f.name)
    return m


@pytest.fixture
def cfg():
    return EnvCfg(dt=1e-3, kp=20.0, kd=0.5, action_scale=0.5)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFKAndVelocities:
    """Step 5: FK + body velocity kernel vs NumPy."""

    def test_fk_default_state(self, model, cfg):
        np_backend = get_backend("numpy", model, cfg, num_envs=2)
        wp_backend = get_backend("warp", model, cfg, num_envs=2)

        res_np = np_backend.reset_all()
        res_wp = wp_backend.reset_all()

        # X_world should match
        np.testing.assert_allclose(res_wp.X_world.numpy(), res_np.X_world.numpy(), atol=ATOL_SINGLE)
        # v_bodies (all zero at default state)
        np.testing.assert_allclose(res_wp.v_bodies.numpy(), res_np.v_bodies.numpy(), atol=ATOL_SINGLE)


class TestSingleStep:
    """Compare a single step between backends."""

    def test_zero_action_step(self, model, cfg):
        np_backend = get_backend("numpy", model, cfg, num_envs=2)
        wp_backend = get_backend("warp", model, cfg, num_envs=2)

        np_backend.reset_all()
        wp_backend.reset_all()

        nu = len(model.actuated_joint_names)
        actions = torch.zeros(2, nu)

        res_np = np_backend.step_batch(actions)
        res_wp = wp_backend.step_batch(actions)

        np.testing.assert_allclose(
            res_wp.q.numpy(),
            res_np.q.numpy(),
            atol=ATOL_SINGLE,
            err_msg="q mismatch after single zero-action step",
        )
        np.testing.assert_allclose(
            res_wp.qdot.numpy(),
            res_np.qdot.numpy(),
            atol=ATOL_SINGLE,
            err_msg="qdot mismatch after single zero-action step",
        )

    def test_nonzero_action_step(self, model, cfg):
        np_backend = get_backend("numpy", model, cfg, num_envs=2)
        wp_backend = get_backend("warp", model, cfg, num_envs=2)

        np_backend.reset_all()
        wp_backend.reset_all()

        nu = len(model.actuated_joint_names)
        torch.manual_seed(42)
        actions = torch.randn(2, nu) * 0.5

        res_np = np_backend.step_batch(actions)
        res_wp = wp_backend.step_batch(actions)

        np.testing.assert_allclose(
            res_wp.q.numpy(),
            res_np.q.numpy(),
            atol=ATOL_SINGLE,
            err_msg="q mismatch after nonzero action step",
        )
        np.testing.assert_allclose(
            res_wp.qdot.numpy(),
            res_np.qdot.numpy(),
            atol=ATOL_SINGLE,
            err_msg="qdot mismatch after nonzero action step",
        )


@pytest.fixture(scope="module")
def model_no_contact():
    """Quadruped without contact links (avoids stiff contact divergence at large dt)."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
    f.write(textwrap.dedent(_quadruped_urdf()))
    f.close()
    m = load_urdf(f.name, floating_base=True, contact_links=[])
    os.unlink(f.name)
    return m


class TestMultiStep:
    """Compare multi-step trajectories (no contact, to avoid dt-related divergence)."""

    def test_50_steps_free_fall(self, model_no_contact, cfg):
        """50 steps with zero actions — gravity-only free fall."""
        np_backend = get_backend("numpy", model_no_contact, cfg, num_envs=1)
        wp_backend = get_backend("warp", model_no_contact, cfg, num_envs=1)

        np_backend.reset_all()
        wp_backend.reset_all()

        nu = len(model_no_contact.actuated_joint_names)
        actions = torch.zeros(1, nu)

        for _ in range(50):
            res_np = np_backend.step_batch(actions)
            res_wp = wp_backend.step_batch(actions)

        np.testing.assert_allclose(
            res_wp.q.numpy(), res_np.q.numpy(), atol=ATOL_MULTI, err_msg="q diverged after 50 free-fall steps"
        )
        np.testing.assert_allclose(
            res_wp.qdot.numpy(),
            res_np.qdot.numpy(),
            atol=ATOL_MULTI,
            err_msg="qdot diverged after 50 free-fall steps",
        )


class TestObservations:
    """Observation data extraction."""

    def test_obs_data_keys(self, model, cfg):
        wp_backend = get_backend("warp", model, cfg, num_envs=2)
        result = wp_backend.reset_all()
        obs_data = wp_backend.get_obs_data(result)

        assert "base_lin_vel" in obs_data
        assert "base_ang_vel" in obs_data
        assert "base_orientation" in obs_data
        assert "joint_pos" in obs_data
        assert "joint_vel" in obs_data
        assert "contact_mask" in obs_data

    def test_obs_shapes(self, model, cfg):
        wp_backend = get_backend("warp", model, cfg, num_envs=4)
        result = wp_backend.reset_all()
        obs_data = wp_backend.get_obs_data(result)

        nu = len(model.actuated_joint_names)
        assert obs_data["base_lin_vel"].shape == (4, 3)
        assert obs_data["base_ang_vel"].shape == (4, 3)
        assert obs_data["base_orientation"].shape == (4, 4)
        assert obs_data["joint_pos"].shape == (4, nu)
        assert obs_data["joint_vel"].shape == (4, nu)


class TestEnvIndependence:
    """Verify environments don't contaminate each other."""

    def test_different_initial_states(self, model, cfg):
        wp_backend = get_backend("warp", model, cfg, num_envs=2)
        result = wp_backend.reset_all(init_noise_scale=0.1)

        # With noise, the two envs should have different states
        q = result.q.numpy()
        # They should not be identical
        assert not np.allclose(q[0], q[1]), "Envs should have different initial states with noise"


class TestQuaternionNormalization:
    """Quaternion should stay normalized after integration."""

    def test_quat_norm_after_steps(self, model_no_contact, cfg):
        wp_backend = get_backend("warp", model_no_contact, cfg, num_envs=1)
        wp_backend.reset_all()

        nu = len(model_no_contact.actuated_joint_names)
        actions = torch.zeros(1, nu)

        for _ in range(100):
            result = wp_backend.step_batch(actions)

        # Check quaternion norm (first 4 elements of q for FreeJoint)
        q = result.q.numpy()[0]
        quat = q[:4]
        norm = np.linalg.norm(quat)
        np.testing.assert_allclose(norm, 1.0, atol=1e-5, err_msg="Quaternion not normalized after 100 steps")
