"""Tests for CudaBatchBackend — raw CUDA C++ kernel validation."""

from __future__ import annotations

import os
import tempfile
import textwrap

import numpy as np
import pytest
import torch

from physics.backends import get_backend
from rl_env.cfg import EnvCfg
from robot import load_urdf

ATOL_SINGLE = 1e-4
ATOL_MULTI = 5e-3


def _quadruped_urdf():
    return """
    <robot name="quad">
      <link name="base"><inertial><mass value="5.0"/><origin xyz="0 0 0"/>
        <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.05"/></inertial>
        <collision><geometry><box size="0.35 0.20 0.10"/></geometry></collision></link>
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


@pytest.fixture(scope="module")
def model_no_contact():
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
    f.write(textwrap.dedent(_quadruped_urdf()))
    f.close()
    m = load_urdf(f.name, floating_base=True, contact_links=[])
    os.unlink(f.name)
    return m


@pytest.fixture
def cfg():
    return EnvCfg(dt=1e-3, kp=20.0, kd=0.5, action_scale=0.5)


def test_single_step_zero_action(model, cfg):
    np_b = get_backend("numpy", model, cfg, num_envs=2)
    cu_b = get_backend("cuda", model, cfg, num_envs=2)
    np_b.reset_all(); cu_b.reset_all()
    nu = len(model.actuated_joint_names)
    actions = torch.zeros(2, nu)
    rn = np_b.step_batch(actions)
    rc = cu_b.step_batch(actions)
    np.testing.assert_allclose(rc.q.numpy(), rn.q.numpy(), atol=ATOL_SINGLE)
    np.testing.assert_allclose(rc.qdot.numpy(), rn.qdot.numpy(), atol=ATOL_SINGLE)


def test_single_step_nonzero(model, cfg):
    np_b = get_backend("numpy", model, cfg, num_envs=2)
    cu_b = get_backend("cuda", model, cfg, num_envs=2)
    np_b.reset_all(); cu_b.reset_all()
    nu = len(model.actuated_joint_names)
    torch.manual_seed(42)
    actions = torch.randn(2, nu) * 0.5
    rn = np_b.step_batch(actions)
    rc = cu_b.step_batch(actions)
    np.testing.assert_allclose(rc.q.numpy(), rn.q.numpy(), atol=ATOL_SINGLE)


def test_50_steps_free_fall(model_no_contact, cfg):
    np_b = get_backend("numpy", model_no_contact, cfg, num_envs=1)
    cu_b = get_backend("cuda", model_no_contact, cfg, num_envs=1)
    np_b.reset_all(); cu_b.reset_all()
    nu = len(model_no_contact.actuated_joint_names)
    actions = torch.zeros(1, nu)
    for _ in range(50):
        rn = np_b.step_batch(actions)
        rc = cu_b.step_batch(actions)
    np.testing.assert_allclose(rc.q.numpy(), rn.q.numpy(), atol=ATOL_MULTI)


def test_obs_shapes(model, cfg):
    cu_b = get_backend("cuda", model, cfg, num_envs=4)
    result = cu_b.reset_all()
    obs = cu_b.get_obs_data(result)
    nu = len(model.actuated_joint_names)
    assert obs["base_lin_vel"].shape == (4, 3)
    assert obs["joint_pos"].shape == (4, nu)


def test_quat_norm(model_no_contact, cfg):
    cu_b = get_backend("cuda", model_no_contact, cfg, num_envs=1)
    cu_b.reset_all()
    nu = len(model_no_contact.actuated_joint_names)
    actions = torch.zeros(1, nu)
    for _ in range(100):
        result = cu_b.step_batch(actions)
    quat = result.q.numpy()[0, :4]
    np.testing.assert_allclose(np.linalg.norm(quat), 1.0, atol=1e-5)
