"""
Tests for BatchedCRBA — GPU batched CRBA via Cholesky solve.

Validates batched CRBA forward dynamics against NumPy ABA.
"""

from __future__ import annotations

import os
import tempfile
import textwrap

import numpy as np
import pytest
import torch

from physics.backends.batched_crba import BatchedCRBA
from physics.backends.static_data import StaticRobotData
from robot import load_urdf

ATOL = 1e-4


def _quadruped_urdf():
    return """
    <robot name="quad">
      <link name="base"><inertial><mass value="5.0"/><origin xyz="0 0 0"/>
        <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.05"/></inertial></link>
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
    m = load_urdf(f.name, floating_base=True, contact_links=[])
    os.unlink(f.name)
    return m


@pytest.fixture(scope="module")
def static(model):
    return StaticRobotData.from_model(model)


def _compute_X_up(model, q_np):
    """Compute X_up for a single env using NumPy tree."""
    tree = model.tree
    nb = tree.num_bodies
    X_up_R = np.zeros((nb, 3, 3), dtype=np.float32)
    X_up_r = np.zeros((nb, 3), dtype=np.float32)
    for body in tree.bodies:
        X_J = body.joint.transform(q_np[body.q_idx])
        X_up = body.X_tree @ X_J
        X_up_R[body.index] = X_up.R.astype(np.float32)
        X_up_r[body.index] = X_up.r.astype(np.float32)
    return X_up_R, X_up_r


class TestBatchedCRBAvsABA:
    def test_zero_state(self, model, static):
        tree = model.tree
        q0, qdot0 = tree.default_state()
        tau = np.zeros(tree.nv)
        qddot_aba = tree.aba(q0, qdot0, tau)

        N = 2
        crba = BatchedCRBA(static, device="cuda:0")
        q_t = torch.from_numpy(q0.astype(np.float32)).unsqueeze(0).expand(N, -1).contiguous().cuda()
        qdot_t = torch.from_numpy(qdot0.astype(np.float32)).unsqueeze(0).expand(N, -1).contiguous().cuda()
        tau_t = torch.zeros(N, tree.nv, device="cuda:0")
        ext_t = torch.zeros(N, tree.num_bodies, 6, device="cuda:0")

        Xr, Xt = _compute_X_up(model, q0)
        X_up_R = torch.from_numpy(Xr).unsqueeze(0).expand(N, -1, -1, -1).contiguous().cuda()
        X_up_r = torch.from_numpy(Xt).unsqueeze(0).expand(N, -1, -1).contiguous().cuda()

        qddot_gpu = crba.forward_dynamics(q_t, qdot_t, tau_t, ext_t, X_up_R, X_up_r)
        qddot_gpu_np = qddot_gpu.cpu().numpy()

        for i in range(N):
            np.testing.assert_allclose(
                qddot_gpu_np[i], qddot_aba, atol=ATOL, err_msg=f"Batched CRBA env {i} != ABA"
            )

    def test_nonzero_tau(self, model, static):
        tree = model.tree
        q0, qdot0 = tree.default_state()
        np.random.seed(42)
        tau = np.random.randn(tree.nv).astype(np.float64) * 2.0
        qddot_aba = tree.aba(q0, qdot0, tau)

        N = 4
        crba = BatchedCRBA(static, device="cuda:0")
        q_t = torch.from_numpy(q0.astype(np.float32)).unsqueeze(0).expand(N, -1).contiguous().cuda()
        qdot_t = torch.from_numpy(qdot0.astype(np.float32)).unsqueeze(0).expand(N, -1).contiguous().cuda()
        tau_t = torch.from_numpy(tau.astype(np.float32)).unsqueeze(0).expand(N, -1).contiguous().cuda()
        ext_t = torch.zeros(N, tree.num_bodies, 6, device="cuda:0")

        Xr, Xt = _compute_X_up(model, q0)
        X_up_R = torch.from_numpy(Xr).unsqueeze(0).expand(N, -1, -1, -1).contiguous().cuda()
        X_up_r = torch.from_numpy(Xt).unsqueeze(0).expand(N, -1, -1).contiguous().cuda()

        qddot_gpu = crba.forward_dynamics(q_t, qdot_t, tau_t, ext_t, X_up_R, X_up_r)

        np.testing.assert_allclose(
            qddot_gpu[0].cpu().numpy(),
            qddot_aba,
            atol=ATOL,
            rtol=1e-4,
            err_msg="Batched CRBA != ABA with nonzero tau",
        )

    def test_batch_independence(self, model, static):
        """Different envs with different tau should produce different qddot."""
        tree = model.tree
        q0, qdot0 = tree.default_state()
        N = 4
        crba = BatchedCRBA(static, device="cuda:0")

        q_t = torch.from_numpy(q0.astype(np.float32)).unsqueeze(0).expand(N, -1).contiguous().cuda()
        qdot_t = torch.from_numpy(qdot0.astype(np.float32)).unsqueeze(0).expand(N, -1).contiguous().cuda()
        torch.manual_seed(42)
        tau_t = torch.randn(N, tree.nv, device="cuda:0")
        ext_t = torch.zeros(N, tree.num_bodies, 6, device="cuda:0")

        Xr, Xt = _compute_X_up(model, q0)
        X_up_R = torch.from_numpy(Xr).unsqueeze(0).expand(N, -1, -1, -1).contiguous().cuda()
        X_up_r = torch.from_numpy(Xt).unsqueeze(0).expand(N, -1, -1).contiguous().cuda()

        qddot = crba.forward_dynamics(q_t, qdot_t, tau_t, ext_t, X_up_R, X_up_r)

        # Each env should get different qddot (different tau)
        assert not torch.allclose(qddot[0], qddot[1]), "Envs with different tau should have different qddot"
