"""
Unit tests for robot/urdf_loader.py.

All tests use inline URDF strings written to a temporary file — no external
file dependencies.
"""

from __future__ import annotations

import os
import tempfile
import textwrap

import numpy as np
import pytest

from physics.joint import FixedJoint, FreeJoint, RevoluteJoint
from robot import RobotModel, load_urdf

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_urdf(content: str) -> str:
    """Write URDF string to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
    f.write(textwrap.dedent(content))
    f.close()
    return f.name


def _single_link_urdf() -> str:
    return """
    <robot name="single">
      <link name="base">
        <inertial>
          <mass value="1.0"/>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
        </inertial>
      </link>
    </robot>
    """


def _two_link_urdf(joint_type: str = "revolute", axis: str = "0 0 1") -> str:
    return f"""
    <robot name="two_link">
      <link name="base">
        <inertial>
          <mass value="1.0"/>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
        </inertial>
      </link>
      <link name="child">
        <inertial>
          <mass value="0.5"/>
          <origin xyz="0 0 0.1" rpy="0 0 0"/>
          <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
        </inertial>
        <collision>
          <geometry><box size="0.05 0.05 0.2"/></geometry>
        </collision>
      </link>
      <joint name="j1" type="{joint_type}">
        <parent link="base"/>
        <child link="child"/>
        <origin xyz="0 0 0.5" rpy="0 0 0"/>
        <axis xyz="{axis}"/>
        <limit lower="-1.57" upper="1.57"/>
        <dynamics damping="0.1"/>
      </joint>
    </robot>
    """


def _no_inertial_urdf() -> str:
    return """
    <robot name="no_inertial">
      <link name="base"/>
    </robot>
    """


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_link_floating():
    """Single link with floating base → nq=7, nv=6."""
    path = _write_urdf(_single_link_urdf())
    try:
        model = load_urdf(path, floating_base=True)
        assert isinstance(model, RobotModel)
        assert model.tree.nq == 7
        assert model.tree.nv == 6
        assert model.tree.num_bodies == 1
        root_body = model.tree.bodies[0]
        assert isinstance(root_body.joint, FreeJoint)
    finally:
        os.unlink(path)


def test_fixed_base():
    """floating_base=False → root uses FixedJoint, nq=0 for root."""
    path = _write_urdf(_single_link_urdf())
    try:
        model = load_urdf(path, floating_base=False)
        assert model.tree.nq == 0
        assert model.tree.nv == 0
        root_body = model.tree.bodies[0]
        assert isinstance(root_body.joint, FixedJoint)
    finally:
        os.unlink(path)


def test_revolute_joint():
    """Revolute joint → actuated_joint_names contains the joint name."""
    path = _write_urdf(_two_link_urdf(joint_type="revolute"))
    try:
        model = load_urdf(path, floating_base=False)
        assert "j1" in model.actuated_joint_names
        # child body should have a RevoluteJoint
        child_body = model.tree.body_by_name("child")
        assert isinstance(child_body.joint, RevoluteJoint)
        assert child_body.joint.q_min == pytest.approx(-1.57)
        assert child_body.joint.q_max == pytest.approx(1.57)
        assert child_body.joint.damping == pytest.approx(0.1)
    finally:
        os.unlink(path)


def test_arbitrary_axis():
    """axis=[1,0,0] → RevoluteJoint._axis_vec is [1,0,0]."""
    path = _write_urdf(_two_link_urdf(joint_type="revolute", axis="1 0 0"))
    try:
        model = load_urdf(path, floating_base=False)
        child_body = model.tree.body_by_name("child")
        assert isinstance(child_body.joint, RevoluteJoint)
        np.testing.assert_allclose(child_body.joint._axis_vec, [1.0, 0.0, 0.0])
    finally:
        os.unlink(path)


def test_contact_links():
    """contact_links specified → contact_body_names populated."""
    path = _write_urdf(_two_link_urdf())
    try:
        model = load_urdf(path, floating_base=True, contact_links=["child"])
        assert "child" in model.contact_body_names
    finally:
        os.unlink(path)


def test_missing_inertial():
    """Link with no <inertial> → no error, uses placeholder mass 1e-6."""
    path = _write_urdf(_no_inertial_urdf())
    try:
        model = load_urdf(path, floating_base=True)
        assert model.tree.num_bodies == 1
        root_body = model.tree.bodies[0]
        assert root_body.inertia.mass == pytest.approx(1e-6)
    finally:
        os.unlink(path)
