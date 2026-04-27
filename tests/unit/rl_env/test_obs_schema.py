from __future__ import annotations

import numpy as np
import pytest
import torch

from rl_env import ObsFieldSpec, locomotion_obs_schema, obs_cfg_from_schema
from rl_env.managers import ObsManager


class _SchemaEnv:
    def __init__(self) -> None:
        self.q = np.array([1.0, 0.0, 0.0, 0.0, 0.2, 0.3, 0.4, 0.5, -0.5])
        self.qdot = np.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 1.5, -1.5])
        self.root_body_idx = 0
        self.root_q_slice = slice(0, 7)
        self.actuated_q_indices = np.array([7, 8], dtype=np.intp)
        self.actuated_v_indices = np.array([6, 7], dtype=np.intp)
        self.v_bodies = [
            np.array([1.0, 2.0, 3.0, 0.4, 0.5, 0.6]),
        ]
        self.active_contacts = [("left_foot", np.array([0.0, 0.0, 1.0]))]
        self.contact_body_names = ["left_foot", "right_foot"]


def test_locomotion_schema_field_order_and_slices():
    schema = locomotion_obs_schema(
        num_actuated_joints=2,
        num_contact_bodies=2,
        include_contact_mask=True,
    )

    assert schema.names == (
        "base_lin_vel_body",
        "base_ang_vel_body",
        "base_orientation_quat_wxyz",
        "joint_pos",
        "joint_vel",
        "contact_mask",
    )
    assert schema.dim == 16
    assert schema.field_slice("base_lin_vel_body") == slice(0, 3)
    assert schema.field_slice("base_ang_vel_body") == slice(3, 6)
    assert schema.field_slice("base_orientation_quat_wxyz") == slice(6, 10)
    assert schema.field_slice("joint_pos") == slice(10, 12)
    assert schema.field_slice("joint_vel") == slice(12, 14)
    assert schema.field_slice("contact_mask") == slice(14, 16)


def test_locomotion_schema_quaternion_convention_is_scalar_first():
    schema = locomotion_obs_schema(num_actuated_joints=2)
    field = schema.fields[2]

    assert field.name == "base_orientation_quat_wxyz"
    assert "[w, x, y, z]" in field.convention

    obs_cfg = obs_cfg_from_schema(schema)
    obs = ObsManager(obs_cfg, _SchemaEnv()).compute()

    np.testing.assert_allclose(
        obs[schema.field_slice("base_orientation_quat_wxyz")].numpy(),
        [1.0, 0.0, 0.0, 0.0],
    )


def test_obs_cfg_from_schema_matches_existing_obs_manager():
    schema = locomotion_obs_schema(
        num_actuated_joints=2,
        num_contact_bodies=2,
        include_contact_mask=True,
    )
    obs_cfg = obs_cfg_from_schema(schema)
    obs = ObsManager(obs_cfg, _SchemaEnv()).compute()

    assert obs.shape == (schema.dim,)
    np.testing.assert_allclose(obs[schema.field_slice("base_lin_vel_body")].numpy(), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(obs[schema.field_slice("base_ang_vel_body")].numpy(), [0.4, 0.5, 0.6])
    np.testing.assert_allclose(obs[schema.field_slice("joint_pos")].numpy(), [0.5, -0.5])
    np.testing.assert_allclose(obs[schema.field_slice("joint_vel")].numpy(), [1.5, -1.5])
    np.testing.assert_allclose(obs[schema.field_slice("contact_mask")].numpy(), [1.0, 0.0])


def test_contact_mask_is_optional_and_records_phase2_requirement():
    without_contacts = locomotion_obs_schema(num_actuated_joints=2)
    with_contacts = locomotion_obs_schema(
        num_actuated_joints=2,
        num_contact_bodies=2,
        include_contact_mask=True,
    )

    assert "contact_mask" not in without_contacts.names
    assert "contact_mask" in with_contacts.names
    assert any("published per-body mask" in item for item in with_contacts.phase2_requirements)


def test_obs_field_scale_supports_scalar_and_vector_scales():
    env = _SchemaEnv()
    scalar_schema = locomotion_obs_schema(num_actuated_joints=2)
    scaled_joint_vel = ObsFieldSpec(
        name="joint_vel_scaled",
        dim=2,
        term=scalar_schema.fields[4].term,
        scale=(0.1, 0.2),
    )

    obs_cfg = obs_cfg_from_schema(type(scalar_schema)(fields=(scaled_joint_vel,)))
    obs = ObsManager(obs_cfg, env).compute()

    assert torch.allclose(obs, torch.tensor([0.15, -0.3]))


def test_obs_field_rejects_bad_scale_length():
    with pytest.raises(ValueError, match="scale length"):
        ObsFieldSpec(name="bad", dim=3, scale=(1.0, 2.0))


def test_obs_field_rejects_negative_dim():
    with pytest.raises(ValueError, match="non-negative"):
        ObsFieldSpec(name="bad", dim=-1)
