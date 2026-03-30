"""
Unit tests for RevoluteJoint joint limits and damping.

Tests cover:
  - No torque when joint is within limits
  - Positive restoring torque when below q_min
  - Negative restoring torque when above q_max
  - Torque magnitude proportional to penetration
  - Damping opposes velocity that deepens violation (unidirectional)
  - Damping does NOT oppose velocity that reduces violation
  - Viscous damping torque (compute_damping_torque)
  - passive_torques() on RobotTree aggregates limit + damping correctly

Reference: joint.py — compute_limit_torque(), compute_damping_torque()
"""

import numpy as np
import pytest

from physics.joint import Axis, RevoluteJoint
from physics.robot_tree import Body, RobotTree
from physics.spatial import SpatialInertia, SpatialTransform

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_joint(q_min=-1.0, q_max=1.0, k=5000.0, b=50.0, damping=0.0):
    return RevoluteJoint(
        "test",
        axis=Axis.Y,
        q_min=q_min,
        q_max=q_max,
        k_limit=k,
        b_limit=b,
        damping=damping,
    )


def _q(angle):
    return np.array([angle], dtype=np.float64)


def _qdot(omega):
    return np.array([omega], dtype=np.float64)


def _make_single_revolute_tree(q_min=-1.0, q_max=1.0, k=5000.0, b=50.0, damping=0.0):
    """Single revolute joint attached to a fixed base."""
    tree = RobotTree(gravity=9.81)
    joint = RevoluteJoint("j0", axis=Axis.Y, q_min=q_min, q_max=q_max, k_limit=k, b_limit=b, damping=damping)
    body = Body(
        name="link",
        index=0,
        joint=joint,
        inertia=SpatialInertia(mass=1.0, inertia=np.eye(3), com=np.zeros(3)),
        X_tree=SpatialTransform.identity(),
        parent=-1,
    )
    tree.add_body(body)
    tree.finalize()
    return tree


# ---------------------------------------------------------------------------
# compute_limit_torque tests
# ---------------------------------------------------------------------------


def test_no_torque_within_limits():
    """Joint inside [q_min, q_max] → zero limit torque."""
    j = _make_joint(q_min=-1.0, q_max=1.0)
    assert j.compute_limit_torque(_q(0.0), _qdot(0.0)) == 0.0
    assert j.compute_limit_torque(_q(0.5), _qdot(1.0)) == 0.0
    assert j.compute_limit_torque(_q(-0.5), _qdot(-1.0)) == 0.0


def test_no_torque_at_limit_boundary():
    """Joint exactly at limit boundary → zero torque (not yet violated)."""
    j = _make_joint(q_min=-1.0, q_max=1.0)
    assert j.compute_limit_torque(_q(-1.0), _qdot(0.0)) == 0.0
    assert j.compute_limit_torque(_q(1.0), _qdot(0.0)) == 0.0


def test_positive_torque_below_q_min():
    """Below q_min → positive (restoring) torque."""
    j = _make_joint(q_min=-1.0, q_max=1.0, k=5000.0, b=0.0)
    tau = j.compute_limit_torque(_q(-1.2), _qdot(0.0))
    assert tau > 0.0, f"Expected positive restoring torque, got {tau}"


def test_negative_torque_above_q_max():
    """Above q_max → negative (restoring) torque."""
    j = _make_joint(q_min=-1.0, q_max=1.0, k=5000.0, b=0.0)
    tau = j.compute_limit_torque(_q(1.2), _qdot(0.0))
    assert tau < 0.0, f"Expected negative restoring torque, got {tau}"


def test_limit_torque_proportional_to_penetration():
    """Limit torque scales linearly with penetration depth (b=0)."""
    k = 5000.0
    j = _make_joint(q_min=-1.0, q_max=1.0, k=k, b=0.0)

    pen1, pen2 = 0.1, 0.3
    tau1 = j.compute_limit_torque(_q(-1.0 - pen1), _qdot(0.0))
    tau2 = j.compute_limit_torque(_q(-1.0 - pen2), _qdot(0.0))

    assert abs(tau1 - k * pen1) < 1e-10
    assert abs(tau2 - k * pen2) < 1e-10
    assert abs(tau2 / tau1 - pen2 / pen1) < 1e-10


def test_damping_deepens_violation_at_lower_limit():
    """Below q_min with ω < 0 (deepening): damping adds to restoring torque."""
    j = _make_joint(q_min=-1.0, q_max=1.0, k=5000.0, b=50.0)
    pen = 0.1
    # ω = -1.0 deepens the violation (going further below q_min)
    tau_static = j.compute_limit_torque(_q(-1.0 - pen), _qdot(0.0))
    tau_deepening = j.compute_limit_torque(_q(-1.0 - pen), _qdot(-1.0))
    assert tau_deepening > tau_static, "Damping should increase restoring torque when deepening violation"


def test_damping_does_not_oppose_recovery_at_lower_limit():
    """Below q_min with ω > 0 (recovering): damping term is zero."""
    j = _make_joint(q_min=-1.0, q_max=1.0, k=5000.0, b=50.0)
    pen = 0.1
    tau_static = j.compute_limit_torque(_q(-1.0 - pen), _qdot(0.0))
    tau_recovering = j.compute_limit_torque(_q(-1.0 - pen), _qdot(1.0))
    # Recovery velocity → damping term = 0, torques should be equal
    assert abs(tau_recovering - tau_static) < 1e-10, "Damping should not oppose recovery from lower limit"


def test_damping_deepens_violation_at_upper_limit():
    """Above q_max with ω > 0 (deepening): damping adds to restoring torque magnitude."""
    j = _make_joint(q_min=-1.0, q_max=1.0, k=5000.0, b=50.0)
    pen = 0.1
    tau_static = j.compute_limit_torque(_q(1.0 + pen), _qdot(0.0))
    tau_deepening = j.compute_limit_torque(_q(1.0 + pen), _qdot(1.0))
    assert tau_deepening < tau_static, (
        "Damping should increase restoring torque magnitude when deepening upper violation"
    )


def test_damping_does_not_oppose_recovery_at_upper_limit():
    """Above q_max with ω < 0 (recovering): damping term is zero."""
    j = _make_joint(q_min=-1.0, q_max=1.0, k=5000.0, b=50.0)
    pen = 0.1
    tau_static = j.compute_limit_torque(_q(1.0 + pen), _qdot(0.0))
    tau_recovering = j.compute_limit_torque(_q(1.0 + pen), _qdot(-1.0))
    assert abs(tau_recovering - tau_static) < 1e-10, "Damping should not oppose recovery from upper limit"


# ---------------------------------------------------------------------------
# compute_damping_torque tests
# ---------------------------------------------------------------------------


def test_viscous_damping_opposes_velocity():
    """Damping torque = -damping * qdot."""
    j = _make_joint(damping=2.0)
    assert j.compute_damping_torque(_qdot(3.0)) == pytest.approx(-6.0)
    assert j.compute_damping_torque(_qdot(-3.0)) == pytest.approx(6.0)
    assert j.compute_damping_torque(_qdot(0.0)) == pytest.approx(0.0)


def test_zero_damping_gives_zero_torque():
    j = _make_joint(damping=0.0)
    assert j.compute_damping_torque(_qdot(10.0)) == 0.0


# ---------------------------------------------------------------------------
# passive_torques() on RobotTree
# ---------------------------------------------------------------------------


def test_passive_torques_zero_within_limits():
    """passive_torques() returns zero when joint is within limits and no damping."""
    tree = _make_single_revolute_tree(q_min=-1.0, q_max=1.0, k=5000.0, b=50.0, damping=0.0)
    q, qdot = tree.default_state()
    tau = tree.passive_torques(q, qdot)
    np.testing.assert_array_equal(tau, np.zeros(tree.nv))


def test_passive_torques_nonzero_outside_limits():
    """passive_torques() returns nonzero when joint exceeds q_max."""
    tree = _make_single_revolute_tree(q_min=-1.0, q_max=1.0, k=5000.0, b=0.0, damping=0.0)
    q, qdot = tree.default_state()
    q[0] = 1.5  # exceed q_max=1.0
    tau = tree.passive_torques(q, qdot)
    assert tau[0] < 0.0, f"Expected negative restoring torque, got {tau[0]}"


def test_passive_torques_includes_damping():
    """passive_torques() includes viscous damping even within limits."""
    damping = 3.0
    tree = _make_single_revolute_tree(q_min=-1.0, q_max=1.0, damping=damping)
    q, qdot = tree.default_state()
    qdot[0] = 2.0
    tau = tree.passive_torques(q, qdot)
    expected = -damping * 2.0  # within limits, only damping
    assert abs(tau[0] - expected) < 1e-10, f"Expected {expected}, got {tau[0]}"
