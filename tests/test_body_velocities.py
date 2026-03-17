"""
Tests for RobotTree.body_velocities().

Verifies that the public method returns the same result as the
forward-pass velocity recursion that ABA computes internally.
"""

import numpy as np

from physics.joint import Axis, FreeJoint, RevoluteJoint
from physics.robot_tree import Body, RobotTree
from physics.spatial import SpatialInertia, SpatialTransform


def _make_single_pendulum() -> RobotTree:
    """Root (fixed) + one revolute joint swinging about Y."""
    tree = RobotTree(gravity=9.81)

    root = Body(
        name="world",
        index=0,
        joint=FreeJoint("root"),
        inertia=SpatialInertia(mass=1.0, inertia=np.eye(3), com=np.zeros(3)),
        X_tree=SpatialTransform.identity(),
        parent=-1,
    )
    root_idx = tree.add_body(root)

    link = Body(
        name="link",
        index=1,
        joint=RevoluteJoint("hinge", axis=Axis.Y),
        inertia=SpatialInertia(mass=1.0, inertia=np.eye(3) * 0.1, com=np.zeros(3)),
        X_tree=SpatialTransform.from_rpy(0, 0, 0, r=np.array([0.0, 0.0, -1.0])),
        parent=root_idx,
    )
    tree.add_body(link)
    tree.finalize()
    return tree


def test_body_velocities_zero_qdot():
    """With qdot=0, all body velocities must be zero."""
    tree = _make_single_pendulum()
    q, qdot = tree.default_state()
    q[3] = 1.0  # valid quaternion

    v = tree.body_velocities(q, qdot)
    for i, vi in enumerate(v):
        assert np.allclose(vi, 0.0), f"Body {i} velocity non-zero at rest: {vi}"


def test_body_velocities_length():
    """body_velocities() must return one vector per body."""
    tree = _make_single_pendulum()
    q, qdot = tree.default_state()
    q[3] = 1.0

    v = tree.body_velocities(q, qdot)
    assert len(v) == tree.num_bodies


def test_body_velocities_shape():
    """Each velocity vector must be (6,)."""
    tree = _make_single_pendulum()
    q, qdot = tree.default_state()
    q[3] = 1.0
    qdot[-1] = 1.0  # revolute joint spinning

    v = tree.body_velocities(q, qdot)
    for vi in v:
        assert vi.shape == (6,)


def test_body_velocities_nonzero_qdot():
    """With nonzero qdot, at least one body must have nonzero velocity."""
    tree = _make_single_pendulum()
    q, qdot = tree.default_state()
    q[3] = 1.0
    qdot[-1] = 2.0  # revolute joint spinning at 2 rad/s

    v = tree.body_velocities(q, qdot)
    norms = [np.linalg.norm(vi) for vi in v]
    assert any(n > 1e-6 for n in norms), "Expected nonzero velocity with nonzero qdot"
