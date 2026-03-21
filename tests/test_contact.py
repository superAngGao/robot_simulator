"""
Unit tests for PenaltyContactModel.

Tests cover:
  - No force when contact point is above ground
  - Normal force direction (upward, +z in world)
  - Normal force magnitude (proportional to penetration depth)
  - Friction force opposes slip velocity
  - Zero friction when slip velocity is zero
  - Force expressed in body frame (not world frame)

Reference: contact.py docstring — Azad & Featherstone (2014).
"""

import numpy as np

from physics.contact import ContactParams, ContactPoint, PenaltyContactModel
from physics.spatial import SpatialTransform


def _identity_transform(pos=(0.0, 0.0, 0.0)):
    """Body at given world position, identity orientation."""
    return SpatialTransform(np.eye(3), np.array(pos, dtype=np.float64))


def _make_model(k=5000.0, b=0.0, mu=0.8):
    params = ContactParams(k_normal=k, b_normal=b, mu=mu, slip_eps=1e-6)
    model = PenaltyContactModel(params=params)
    # Single contact point at body origin (position = [0,0,0] in body frame)
    cp = ContactPoint(body_index=0, position=np.zeros(3), name="foot")
    model.add_contact_point(cp)
    return model


def _zero_velocity():
    return np.zeros(6, dtype=np.float64)


# ---------------------------------------------------------------------------


def test_no_force_above_ground():
    """Contact point above ground → zero force."""
    model = _make_model()
    X = _identity_transform(pos=(0.0, 0.0, 0.1))  # z = 0.1 m above ground
    forces = model.compute_forces([X], [_zero_velocity()], num_bodies=1)
    np.testing.assert_array_equal(forces[0], np.zeros(6))


def test_no_force_at_ground_level():
    """Contact point exactly at ground (z=0) → zero force (depth=0)."""
    model = _make_model()
    X = _identity_transform(pos=(0.0, 0.0, 0.0))
    forces = model.compute_forces([X], [_zero_velocity()], num_bodies=1)
    np.testing.assert_array_equal(forces[0], np.zeros(6))


def test_normal_force_upward():
    """Penetrating contact → normal force has positive z component in world frame."""
    model = _make_model(k=5000.0, b=0.0, mu=0.0)
    depth = 0.01  # 1 cm penetration
    X = _identity_transform(pos=(0.0, 0.0, -depth))
    v = _zero_velocity()

    forces = model.compute_forces([X], [v], num_bodies=1)
    f_body = forces[0]

    # Transform back to world to check direction
    f_world = X.apply_force(f_body)
    F_z = f_world[2]  # spatial force: [force(3), torque(3)], force z = index 2
    assert F_z > 0.0, f"Expected upward normal force, got F_z={F_z}"


def test_normal_force_magnitude():
    """Normal force = k * depth when damping=0 and body is stationary."""
    k = 5000.0
    depth = 0.02
    model = _make_model(k=k, b=0.0, mu=0.0)
    X = _identity_transform(pos=(0.0, 0.0, -depth))
    v = _zero_velocity()

    forces = model.compute_forces([X], [v], num_bodies=1)
    f_world = X.apply_force(forces[0])
    F_z = f_world[2]

    expected = k * depth
    assert abs(F_z - expected) < 1e-6, f"Expected {expected}, got {F_z}"


def test_normal_force_proportional_to_depth():
    """Normal force scales linearly with penetration depth."""
    k = 3000.0
    model = _make_model(k=k, b=0.0, mu=0.0)
    v = _zero_velocity()

    depths = [0.005, 0.01, 0.02, 0.05]
    forces_z = []
    for d in depths:
        X = _identity_transform(pos=(0.0, 0.0, -d))
        f_body = model.compute_forces([X], [v], num_bodies=1)[0]
        f_world = X.apply_force(f_body)
        forces_z.append(f_world[2])

    for i in range(1, len(depths)):
        ratio_depth = depths[i] / depths[0]
        ratio_force = forces_z[i] / forces_z[0]
        assert abs(ratio_force - ratio_depth) < 1e-6


def test_friction_opposes_slip():
    """Friction force opposes the slip velocity direction."""
    model = _make_model(k=5000.0, b=0.0, mu=0.8)
    depth = 0.01
    X = _identity_transform(pos=(0.0, 0.0, -depth))

    # Body moving in +x direction → friction should be in -x
    v = np.zeros(6, dtype=np.float64)
    v[0] = 1.0  # linear velocity in body frame x (= world x, identity orientation)

    forces = model.compute_forces([X], [v], num_bodies=1)
    f_world = X.apply_force(forces[0])
    F_x = f_world[0]  # world-frame x force
    assert F_x < 0.0, f"Friction should oppose +x slip, got F_x={F_x}"


def test_zero_friction_at_zero_slip():
    """With zero slip velocity, tangential force should be near zero."""
    model = _make_model(k=5000.0, b=0.0, mu=0.8)
    depth = 0.01
    X = _identity_transform(pos=(0.0, 0.0, -depth))
    v = _zero_velocity()

    forces = model.compute_forces([X], [v], num_bodies=1)
    f_world = X.apply_force(forces[0])
    F_x, F_y = f_world[0], f_world[1]

    # With slip_eps=1e-6 and zero slip, tangential force ≈ 0
    assert abs(F_x) < 1e-3, f"Expected ~0 friction in x, got {F_x}"
    assert abs(F_y) < 1e-3, f"Expected ~0 friction in y, got {F_y}"


def test_damping_reduces_force_on_approach():
    """Downward velocity (approaching ground) reduces normal force."""
    k, b = 5000.0, 500.0
    depth = 0.01
    model = _make_model(k=k, b=b, mu=0.0)
    X = _identity_transform(pos=(0.0, 0.0, -depth))

    # Static contact
    v_static = _zero_velocity()
    f_static = model.compute_forces([X], [v_static], num_bodies=1)[0]
    F_z_static = X.apply_force(f_static)[2]

    # Approaching (downward = negative z velocity in world)
    v_approach = np.zeros(6, dtype=np.float64)
    v_approach[2] = -1.0  # linear z velocity in body frame (identity orientation)
    f_approach = model.compute_forces([X], [v_approach], num_bodies=1)[0]
    F_z_approach = X.apply_force(f_approach)[2]

    assert F_z_approach > F_z_static, (
        f"Approaching contact should increase normal force: {F_z_approach} vs {F_z_static}"
    )


def test_active_contacts_reports_penetrating_only():
    """active_contacts() returns only points below ground."""
    model = _make_model()
    X_above = _identity_transform(pos=(0.0, 0.0, 0.1))
    X_below = _identity_transform(pos=(0.0, 0.0, -0.01))

    # Only below-ground contact
    active = model.active_contacts([X_below])
    assert len(active) == 1

    active = model.active_contacts([X_above])
    assert len(active) == 0
