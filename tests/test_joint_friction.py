"""Tests for Coulomb joint friction."""

from __future__ import annotations

import numpy as np
import pytest

from physics.joint import RevoluteJoint


class TestCoulombFriction:
    def test_zero_friction_param(self):
        j = RevoluteJoint("test", friction=0.0)
        tau = j.compute_friction_torque(np.array([1.0]))
        assert tau == 0.0

    def test_positive_velocity_negative_torque(self):
        j = RevoluteJoint("test", friction=0.5)
        tau = j.compute_friction_torque(np.array([1.0]))
        assert tau < 0, "Friction should oppose positive velocity"

    def test_negative_velocity_positive_torque(self):
        j = RevoluteJoint("test", friction=0.5)
        tau = j.compute_friction_torque(np.array([-1.0]))
        assert tau > 0, "Friction should oppose negative velocity"

    def test_magnitude_bounded(self):
        j = RevoluteJoint("test", friction=0.5)
        tau = j.compute_friction_torque(np.array([100.0]))
        assert abs(tau) <= 0.5 + 1e-10, "Friction magnitude bounded by friction param"

    def test_stiction_zone(self):
        """Near zero velocity, friction should be smooth (not discontinuous)."""
        j = RevoluteJoint("test", friction=1.0)
        tau_small = j.compute_friction_torque(np.array([0.001]))
        tau_large = j.compute_friction_torque(np.array([1.0]))
        assert abs(tau_small) < abs(tau_large), "Small velocity → smaller friction"

    def test_included_in_passive_torques(self):
        """Friction should be part of passive_torques output."""
        from physics.robot_tree import Body, RobotTreeNumpy
        from physics.spatial import SpatialInertia, SpatialTransform
        from physics.joint import FreeJoint

        tree = RobotTreeNumpy()
        tree.add_body(Body(name="base", index=0, joint=FreeJoint("root"),
            inertia=SpatialInertia.from_box(5.0, 0.3, 0.2, 0.1),
            X_tree=SpatialTransform.identity(), parent=-1))
        tree.add_body(Body(name="link", index=1,
            joint=RevoluteJoint("j1", friction=0.5),
            inertia=SpatialInertia(0.5, np.diag([0.001]*3), np.zeros(3)),
            X_tree=SpatialTransform(np.eye(3), np.array([0.1, 0, 0])),
            parent=0))
        tree.finalize()

        q, qdot = tree.default_state()
        qdot[6] = 1.0  # revolute joint velocity

        tau = tree.passive_torques(q, qdot)
        # Friction should contribute a negative torque at index 6
        assert tau[6] < 0, f"Expected negative friction torque, got {tau[6]}"
