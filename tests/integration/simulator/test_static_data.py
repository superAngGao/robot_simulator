"""
Tests for physics/backends/static_data.py — StaticRobotData extraction.

Verifies that StaticRobotData.from_model() correctly flattens a RobotModel
into contiguous arrays suitable for GPU backends.
"""

from __future__ import annotations

import os
import tempfile
import textwrap

import numpy as np
import pytest

from physics.backends.static_data import (
    JOINT_FIXED,
    JOINT_FREE,
    JOINT_REVOLUTE,
    StaticRobotData,
)
from robot import load_urdf

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_urdf(content: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
    f.write(textwrap.dedent(content))
    f.close()
    return f.name


def _quadruped_urdf() -> str:
    """Minimal quadruped with base + 4 legs (hip + calf + foot each)."""
    return """
    <robot name="quad">
      <link name="base">
        <inertial>
          <mass value="5.0"/>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.05"/>
        </inertial>
        <collision><geometry><box size="0.35 0.20 0.10"/></geometry></collision>
      </link>

      <!-- Front-Left leg -->
      <link name="FL_hip"><inertial><mass value="0.5"/><origin xyz="0 0 -0.1"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
      <joint name="FL_hip_joint" type="revolute"><parent link="base"/><child link="FL_hip"/><origin xyz="0.15 0.1 0" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-1.0" upper="1.0" effort="20"/></joint>

      <link name="FL_calf"><inertial><mass value="0.3"/><origin xyz="0 0 -0.1"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
      <joint name="FL_calf_joint" type="revolute"><parent link="FL_hip"/><child link="FL_calf"/><origin xyz="0 0 -0.2" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-2.0" upper="0.5" effort="20"/></joint>

      <link name="FL_foot"><inertial><mass value="0.05"/><origin xyz="0 0 0"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial></link>
      <joint name="FL_foot_joint" type="fixed"><parent link="FL_calf"/><child link="FL_foot"/><origin xyz="0 0 -0.2"/></joint>

      <!-- Front-Right leg -->
      <link name="FR_hip"><inertial><mass value="0.5"/><origin xyz="0 0 -0.1"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
      <joint name="FR_hip_joint" type="revolute"><parent link="base"/><child link="FR_hip"/><origin xyz="0.15 -0.1 0" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-1.0" upper="1.0" effort="20"/></joint>

      <link name="FR_calf"><inertial><mass value="0.3"/><origin xyz="0 0 -0.1"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
      <joint name="FR_calf_joint" type="revolute"><parent link="FR_hip"/><child link="FR_calf"/><origin xyz="0 0 -0.2" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-2.0" upper="0.5" effort="20"/></joint>

      <link name="FR_foot"><inertial><mass value="0.05"/><origin xyz="0 0 0"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial></link>
      <joint name="FR_foot_joint" type="fixed"><parent link="FR_calf"/><child link="FR_foot"/><origin xyz="0 0 -0.2"/></joint>
    </robot>
    """


@pytest.fixture
def quad_model():
    path = _write_urdf(_quadruped_urdf())
    try:
        model = load_urdf(
            path,
            floating_base=True,
            contact_links=["FL_foot", "FR_foot"],
        )
        yield model
    finally:
        os.unlink(path)


@pytest.fixture
def static(quad_model):
    return StaticRobotData.from_model(quad_model)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDimensions:
    def test_nb(self, static, quad_model):
        assert static.nb == quad_model.tree.num_bodies

    def test_nq_nv(self, static, quad_model):
        assert static.nq == quad_model.tree.nq
        assert static.nv == quad_model.tree.nv

    def test_nc(self, static):
        # 2 contact links: FL_foot, FR_foot
        assert static.nc == 2

    def test_nu(self, static, quad_model):
        assert static.nu == len(quad_model.actuated_joint_names)


class TestJointTypes:
    def test_root_is_free(self, static):
        assert static.joint_type[0] == JOINT_FREE

    def test_revolute_joints(self, static):
        # Bodies 1,2 (FL_hip, FL_calf), 4,5 (FR_hip, FR_calf) are revolute
        rev_mask = static.joint_type == JOINT_REVOLUTE
        assert rev_mask.sum() == 4  # 4 revolute joints

    def test_fixed_joints(self, static):
        fix_mask = static.joint_type == JOINT_FIXED
        assert fix_mask.sum() == 2  # FL_foot, FR_foot

    def test_joint_axis_revolute(self, static):
        for i in range(static.nb):
            if static.joint_type[i] == JOINT_REVOLUTE:
                # All hip/calf joints have axis [0, 1, 0]
                np.testing.assert_allclose(static.joint_axis[i], [0, 1, 0], atol=1e-6)


class TestTreeStructure:
    def test_root_parent(self, static):
        assert static.parent_idx[0] == -1

    def test_all_parents_valid(self, static):
        for i in range(1, static.nb):
            assert 0 <= static.parent_idx[i] < i

    def test_q_idx_contiguous(self, static):
        """q index ranges should be contiguous and cover [0, nq)."""
        all_indices = set()
        for i in range(static.nb):
            start = int(static.q_idx_start[i])
            length = int(static.q_idx_len[i])
            for j in range(start, start + length):
                assert j not in all_indices, f"Overlapping q index {j}"
                all_indices.add(j)
        assert all_indices == set(range(static.nq))

    def test_v_idx_contiguous(self, static):
        """v index ranges should be contiguous and cover [0, nv)."""
        all_indices = set()
        for i in range(static.nb):
            start = int(static.v_idx_start[i])
            length = int(static.v_idx_len[i])
            for j in range(start, start + length):
                assert j not in all_indices, f"Overlapping v index {j}"
                all_indices.add(j)
        assert all_indices == set(range(static.nv))


class TestTransformsAndInertia:
    def test_X_tree_shapes(self, static):
        assert static.X_tree_R.shape == (static.nb, 3, 3)
        assert static.X_tree_r.shape == (static.nb, 3)

    def test_inertia_shape(self, static):
        assert static.inertia_mat.shape == (static.nb, 6, 6)

    def test_inertia_matches_tree(self, static, quad_model):
        """6x6 inertia matrix should match the tree's SpatialInertia.matrix()."""
        for i, body in enumerate(quad_model.tree.bodies):
            expected = body.inertia.matrix().astype(np.float32)
            np.testing.assert_allclose(static.inertia_mat[i], expected, atol=1e-6)

    def test_X_tree_matches_tree(self, static, quad_model):
        for i, body in enumerate(quad_model.tree.bodies):
            np.testing.assert_allclose(static.X_tree_R[i], body.X_tree.R.astype(np.float32), atol=1e-6)
            np.testing.assert_allclose(static.X_tree_r[i], body.X_tree.r.astype(np.float32), atol=1e-6)


class TestJointLimits:
    def test_revolute_limits_extracted(self, static, quad_model):
        for i, body in enumerate(quad_model.tree.bodies):
            from physics.joint import RevoluteJoint

            if isinstance(body.joint, RevoluteJoint):
                assert static.q_min[i] == pytest.approx(body.joint.q_min, abs=1e-6)
                assert static.q_max[i] == pytest.approx(body.joint.q_max, abs=1e-6)

    def test_non_revolute_limits_inf(self, static):
        for i in range(static.nb):
            if static.joint_type[i] != JOINT_REVOLUTE:
                assert static.q_min[i] == -np.inf or static.q_min[i] == pytest.approx(-np.inf)


class TestContactData:
    def test_contact_body_idx_shape(self, static):
        assert static.contact_body_idx.shape == (static.nc,)

    def test_contact_local_pos_shape(self, static):
        assert static.contact_local_pos.shape == (static.nc, 3)

    def test_contact_params(self, static):
        assert static.contact_k_normal > 0
        assert static.contact_mu > 0


class TestControllerIndices:
    def test_actuated_indices_shape(self, static):
        assert static.actuated_q_indices.shape == (static.nu,)
        assert static.actuated_v_indices.shape == (static.nu,)

    def test_effort_limits(self, static, quad_model):
        if quad_model.effort_limits is not None:
            assert static.effort_limits is not None
            assert static.effort_limits.shape == (static.nu,)


class TestDefaultState:
    def test_default_q_shape(self, static):
        assert static.default_q.shape == (static.nq,)

    def test_default_qdot_shape(self, static):
        assert static.default_qdot.shape == (static.nv,)

    def test_default_q_matches_tree(self, static, quad_model):
        q0, _ = quad_model.tree.default_state()
        np.testing.assert_allclose(static.default_q, q0.astype(np.float32), atol=1e-6)

    def test_root_info(self, static):
        assert static.root_body_idx == 0
        assert static.root_q_len == 7  # FreeJoint
