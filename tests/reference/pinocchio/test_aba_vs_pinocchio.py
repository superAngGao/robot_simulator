"""
ABA cross-validation against Pinocchio.

Builds equivalent robot models in both our implementation and Pinocchio,
then compares ABA (forward dynamics) output for identical inputs.

Robots tested:
  1. Single revolute pendulum (Y-axis, CoM offset)
  2. Double pendulum (two revolute joints, different axes)
  3. Fixed-base robot at multiple joint configurations
  4. Non-zero qdot (Coriolis/centrifugal terms)
  5. Non-zero tau (applied torques)

Reference: Featherstone (2008) Algorithm 7.3
"""

import numpy as np
import pytest

try:
    import pinocchio as pin

    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False

from physics.joint import Axis, RevoluteJoint
from physics.robot_tree import Body, RobotTree
from physics.spatial import SpatialInertia, SpatialTransform

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not HAS_PINOCCHIO, reason="pinocchio not installed"),
]

ATOL = 1e-8  # tolerance for ABA comparison


# ---------------------------------------------------------------------------
# Pinocchio model builders
# ---------------------------------------------------------------------------


def _pin_single_pendulum():
    """Single revolute-Y pendulum: mass=1, com=[0,0,-0.5], I=0.1*eye."""
    model = pin.Model()
    model.gravity.linear = np.array([0.0, 0.0, -9.81])
    jid = model.addJoint(0, pin.JointModelRY(), pin.SE3.Identity(), "j0")
    I = pin.Inertia(1.0, np.zeros(3), 0.1 * np.eye(3))
    placement = pin.SE3(np.eye(3), np.array([0.0, 0.0, -0.5]))
    model.appendBodyToJoint(jid, I, placement)
    return model


def _pin_double_pendulum():
    """Two revolute joints: first Y, second X. Different masses and CoM offsets."""
    model = pin.Model()
    model.gravity.linear = np.array([0.0, 0.0, -9.81])

    # Link 1: revolute Y, mass=1, com at [0,0,-0.3]
    jid1 = model.addJoint(0, pin.JointModelRY(), pin.SE3.Identity(), "j0")
    I1 = pin.Inertia(1.0, np.zeros(3), 0.05 * np.eye(3))
    p1 = pin.SE3(np.eye(3), np.array([0.0, 0.0, -0.3]))
    model.appendBodyToJoint(jid1, I1, p1)

    # Link 2: revolute X, attached at [0,0,-0.6] from parent joint, com at [0,0,-0.2]
    X_parent = pin.SE3(np.eye(3), np.array([0.0, 0.0, -0.6]))
    jid2 = model.addJoint(jid1, pin.JointModelRX(), X_parent, "j1")
    I2 = pin.Inertia(0.5, np.zeros(3), 0.02 * np.eye(3))
    p2 = pin.SE3(np.eye(3), np.array([0.0, 0.0, -0.2]))
    model.appendBodyToJoint(jid2, I2, p2)

    return model


# ---------------------------------------------------------------------------
# Our model builders (matching the Pinocchio models above)
# ---------------------------------------------------------------------------


def _our_single_pendulum():
    tree = RobotTree(gravity=9.81)
    body = Body(
        name="link0",
        index=0,
        joint=RevoluteJoint("j0", axis=Axis.Y),
        inertia=SpatialInertia(mass=1.0, inertia=0.1 * np.eye(3), com=np.array([0.0, 0.0, -0.5])),
        X_tree=SpatialTransform.identity(),
        parent=-1,
    )
    tree.add_body(body)
    tree.finalize()
    return tree


def _our_double_pendulum():
    tree = RobotTree(gravity=9.81)

    # Link 1
    b0 = Body(
        name="link0",
        index=0,
        joint=RevoluteJoint("j0", axis=Axis.Y),
        inertia=SpatialInertia(mass=1.0, inertia=0.05 * np.eye(3), com=np.array([0.0, 0.0, -0.3])),
        X_tree=SpatialTransform.identity(),
        parent=-1,
    )
    tree.add_body(b0)

    # Link 2: X_tree = offset [0,0,-0.6] from parent joint
    b1 = Body(
        name="link1",
        index=1,
        joint=RevoluteJoint("j1", axis=Axis.X),
        inertia=SpatialInertia(mass=0.5, inertia=0.02 * np.eye(3), com=np.array([0.0, 0.0, -0.2])),
        X_tree=SpatialTransform(np.eye(3), np.array([0.0, 0.0, -0.6])),
        parent=0,
    )
    tree.add_body(b1)
    tree.finalize()
    return tree


# ---------------------------------------------------------------------------
# Helper: run Pinocchio ABA
# ---------------------------------------------------------------------------


def _pin_aba(model, q, qdot, tau):
    data = model.createData()
    return pin.aba(model, data, q, qdot, tau).copy()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_pendulum_static():
    """Single pendulum at rest: ABA matches Pinocchio at multiple angles."""
    tree = _our_single_pendulum()
    model = _pin_single_pendulum()

    for q_val in [0.0, 0.3, -0.5, np.pi / 4, np.pi / 2]:
        q = np.array([q_val])
        qdot = np.zeros(1)
        tau = np.zeros(1)

        ours = tree.aba(q, qdot, tau)
        pin_ref = _pin_aba(model, q, qdot, tau)

        np.testing.assert_allclose(
            ours, pin_ref, atol=ATOL, err_msg=f"Mismatch at q={q_val:.3f}: ours={ours}, pin={pin_ref}"
        )


def test_single_pendulum_with_velocity():
    """Single pendulum with non-zero qdot: Coriolis/centrifugal terms."""
    tree = _our_single_pendulum()
    model = _pin_single_pendulum()

    for q_val, qdot_val in [(0.3, 1.0), (-0.5, -2.0), (np.pi / 4, 0.5)]:
        q = np.array([q_val])
        qdot = np.array([qdot_val])
        tau = np.zeros(1)

        ours = tree.aba(q, qdot, tau)
        pin_ref = _pin_aba(model, q, qdot, tau)

        np.testing.assert_allclose(ours, pin_ref, atol=ATOL)


def test_single_pendulum_with_torque():
    """Single pendulum with applied torque."""
    tree = _our_single_pendulum()
    model = _pin_single_pendulum()

    for q_val, tau_val in [(0.0, 1.0), (0.3, -2.0), (np.pi / 4, 5.0)]:
        q = np.array([q_val])
        qdot = np.zeros(1)
        tau = np.array([tau_val])

        ours = tree.aba(q, qdot, tau)
        pin_ref = _pin_aba(model, q, qdot, tau)

        np.testing.assert_allclose(ours, pin_ref, atol=ATOL)


def test_double_pendulum_static():
    """Double pendulum at rest: ABA matches Pinocchio."""
    tree = _our_double_pendulum()
    model = _pin_double_pendulum()

    configs = [
        (0.0, 0.0),
        (0.3, 0.2),
        (-0.5, 0.4),
        (np.pi / 4, -np.pi / 6),
    ]
    for q0, q1 in configs:
        q = np.array([q0, q1])
        qdot = np.zeros(2)
        tau = np.zeros(2)

        ours = tree.aba(q, qdot, tau)
        pin_ref = _pin_aba(model, q, qdot, tau)

        np.testing.assert_allclose(
            ours, pin_ref, atol=ATOL, err_msg=f"Mismatch at q={q}: ours={ours}, pin={pin_ref}"
        )


def test_double_pendulum_with_velocity_and_torque():
    """Double pendulum with non-zero qdot and tau."""
    tree = _our_double_pendulum()
    model = _pin_double_pendulum()

    q = np.array([0.3, -0.4])
    qdot = np.array([1.0, -0.5])
    tau = np.array([0.5, -1.0])

    ours = tree.aba(q, qdot, tau)
    pin_ref = _pin_aba(model, q, qdot, tau)

    np.testing.assert_allclose(ours, pin_ref, atol=ATOL)
