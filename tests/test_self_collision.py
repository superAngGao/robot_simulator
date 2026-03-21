"""
Unit tests for AABBSelfCollision and NullSelfCollision.

Tests cover:
  - No force when AABBs are separated
  - No force when bodies are adjacent (parent-child excluded)
  - Force direction: bodies pushed apart along MTV axis
  - Force magnitude proportional to penetration depth
  - Newton's third law: equal and opposite forces on both bodies
  - Damping: approaching bodies get extra force, separating do not
  - NullSelfCollision always returns zero
  - build_pairs: correct pair count for chain vs non-adjacent topology
  - _world_aabb: rotated box projects correctly
"""

import numpy as np

from physics.collision import AABBSelfCollision, BodyAABB, NullSelfCollision
from physics.spatial import SpatialTransform

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _transform(pos=(0.0, 0.0, 0.0), R=None):
    R = np.eye(3) if R is None else R
    return SpatialTransform(R, np.array(pos, dtype=np.float64))


def _zero_vel():
    return np.zeros(6, dtype=np.float64)


def _make_two_body_sc(k=1000.0, b=0.0, half_i=(0.1, 0.1, 0.1), half_j=(0.1, 0.1, 0.1), parent_list=(-1, -1)):
    """Two non-adjacent bodies (both roots → no parent-child edge)."""
    sc = AABBSelfCollision(k_contact=k, b_contact=b)
    sc.add_body(BodyAABB(0, np.array(half_i, dtype=np.float64)))
    sc.add_body(BodyAABB(1, np.array(half_j, dtype=np.float64)))
    sc.build_pairs(list(parent_list))
    return sc


# ---------------------------------------------------------------------------
# Separation → no force
# ---------------------------------------------------------------------------


def test_no_force_when_separated_x():
    """Bodies far apart on x-axis → zero force."""
    sc = _make_two_body_sc(k=1000.0)
    X = [_transform((0.0, 0.0, 0.0)), _transform((1.0, 0.0, 0.0))]
    v = [_zero_vel(), _zero_vel()]
    forces = sc.compute_forces(X, v, num_bodies=2)
    np.testing.assert_array_equal(forces[0], np.zeros(6))
    np.testing.assert_array_equal(forces[1], np.zeros(6))


def test_no_force_when_just_touching():
    """AABBs touching exactly (overlap=0) → no force."""
    sc = _make_two_body_sc(k=1000.0, half_i=(0.1, 0.1, 0.1), half_j=(0.1, 0.1, 0.1))
    # centers 0.2 apart → edges just touch
    X = [_transform((0.0, 0.0, 0.0)), _transform((0.2, 0.0, 0.0))]
    v = [_zero_vel(), _zero_vel()]
    forces = sc.compute_forces(X, v, num_bodies=2)
    np.testing.assert_array_equal(forces[0], np.zeros(6))
    np.testing.assert_array_equal(forces[1], np.zeros(6))


# ---------------------------------------------------------------------------
# Adjacent bodies excluded
# ---------------------------------------------------------------------------


def test_adjacent_bodies_excluded():
    """Parent-child pair is excluded from collision even when overlapping."""
    sc = AABBSelfCollision(k_contact=1000.0)
    sc.add_body(BodyAABB(0, np.array([0.5, 0.5, 0.5])))
    sc.add_body(BodyAABB(1, np.array([0.5, 0.5, 0.5])))
    # body 1's parent is body 0 → adjacent, should be excluded
    sc.build_pairs(parent_list=[-1, 0])

    assert sc.num_pairs == 0

    # Even with full overlap, zero force
    X = [_transform((0.0, 0.0, 0.0)), _transform((0.0, 0.0, 0.0))]
    v = [_zero_vel(), _zero_vel()]
    forces = sc.compute_forces(X, v, num_bodies=2)
    np.testing.assert_array_equal(forces[0], np.zeros(6))
    np.testing.assert_array_equal(forces[1], np.zeros(6))


# ---------------------------------------------------------------------------
# Force direction
# ---------------------------------------------------------------------------


def test_force_pushes_bodies_apart_x():
    """Overlapping on x-axis: body 0 (left) pushed left, body 1 (right) pushed right."""
    sc = _make_two_body_sc(k=1000.0, b=0.0)
    # centers 0.15 apart, half_extents 0.1 each → overlap = 0.2 - 0.15 = 0.05
    X = [_transform((0.0, 0.0, 0.0)), _transform((0.15, 0.0, 0.0))]
    v = [_zero_vel(), _zero_vel()]
    forces = sc.compute_forces(X, v, num_bodies=2)

    # Body 0 should be pushed in -x (world), body 1 in +x
    f0_world = X[0].apply_force(forces[0])
    f1_world = X[1].apply_force(forces[1])
    assert f0_world[0] < 0.0, f"Body 0 should be pushed -x, got {f0_world[0]}"
    assert f1_world[0] > 0.0, f"Body 1 should be pushed +x, got {f1_world[0]}"


def test_force_pushes_bodies_apart_z():
    """Overlapping on z-axis: correct separation direction."""
    sc = _make_two_body_sc(k=1000.0, b=0.0)
    X = [_transform((0.0, 0.0, 0.0)), _transform((0.0, 0.0, 0.15))]
    v = [_zero_vel(), _zero_vel()]
    forces = sc.compute_forces(X, v, num_bodies=2)

    f0_world = X[0].apply_force(forces[0])
    f1_world = X[1].apply_force(forces[1])
    assert f0_world[2] < 0.0, f"Body 0 should be pushed -z, got {f0_world[2]}"
    assert f1_world[2] > 0.0, f"Body 1 should be pushed +z, got {f1_world[2]}"


# ---------------------------------------------------------------------------
# Force magnitude
# ---------------------------------------------------------------------------


def test_force_magnitude_proportional_to_depth():
    """Normal force = k * depth (b=0, static)."""
    k = 2000.0
    sc = _make_two_body_sc(k=k, b=0.0)

    for depth in [0.01, 0.03, 0.05]:
        sep = 0.2 - depth  # half_extents sum = 0.2
        X = [_transform((0.0, 0.0, 0.0)), _transform((sep, 0.0, 0.0))]
        v = [_zero_vel(), _zero_vel()]
        forces = sc.compute_forces(X, v, num_bodies=2)
        f0_world = X[0].apply_force(forces[0])
        F_mag = abs(f0_world[0])
        assert abs(F_mag - k * depth) < 1e-6, f"depth={depth}: expected {k * depth}, got {F_mag}"


# ---------------------------------------------------------------------------
# Newton's third law
# ---------------------------------------------------------------------------


def test_newtons_third_law():
    """Forces on body 0 and body 1 are equal and opposite in world frame."""
    sc = _make_two_body_sc(k=1500.0, b=0.0)
    X = [_transform((0.0, 0.0, 0.0)), _transform((0.12, 0.0, 0.0))]
    v = [_zero_vel(), _zero_vel()]
    forces = sc.compute_forces(X, v, num_bodies=2)

    f0_world = X[0].apply_force(forces[0])
    f1_world = X[1].apply_force(forces[1])
    np.testing.assert_allclose(f0_world[:3], -f1_world[:3], atol=1e-10)


# ---------------------------------------------------------------------------
# Damping
# ---------------------------------------------------------------------------


def test_damping_increases_force_when_approaching():
    """Approaching bodies (negative relative velocity along contact) → larger force."""
    k, b = 1000.0, 200.0
    sc = _make_two_body_sc(k=k, b=b)
    X = [_transform((0.0, 0.0, 0.0)), _transform((0.15, 0.0, 0.0))]

    v_static = [_zero_vel(), _zero_vel()]
    f_static = sc.compute_forces(X, v_static, num_bodies=2)
    F_static = abs(X[0].apply_force(f_static[0])[0])

    # body 0 moving +x (toward body 1), body 1 stationary → approaching
    v_approach = [_zero_vel(), _zero_vel()]
    v_approach[0][0] = 1.0  # linear x velocity in body frame
    f_approach = sc.compute_forces(X, v_approach, num_bodies=2)
    F_approach = abs(X[0].apply_force(f_approach[0])[0])

    assert F_approach > F_static, f"Approaching should increase force: {F_approach} vs {F_static}"


def test_damping_not_applied_when_separating():
    """Separating bodies → damping term not applied, force equals static."""
    k, b = 1000.0, 200.0
    sc = _make_two_body_sc(k=k, b=b)
    X = [_transform((0.0, 0.0, 0.0)), _transform((0.15, 0.0, 0.0))]

    v_static = [_zero_vel(), _zero_vel()]
    f_static = sc.compute_forces(X, v_static, num_bodies=2)
    F_static = abs(X[0].apply_force(f_static[0])[0])

    # body 0 moving -x (away from body 1) → separating
    v_sep = [_zero_vel(), _zero_vel()]
    v_sep[0][0] = -1.0
    f_sep = sc.compute_forces(X, v_sep, num_bodies=2)
    F_sep = abs(X[0].apply_force(f_sep[0])[0])

    assert abs(F_sep - F_static) < 1e-10, f"Separating should not change force: {F_sep} vs {F_static}"


# ---------------------------------------------------------------------------
# build_pairs: pair count
# ---------------------------------------------------------------------------


def test_build_pairs_chain_of_three():
    """Chain 0→1→2: pairs (0,2) only — (0,1) and (1,2) are adjacent."""
    sc = AABBSelfCollision()
    for i in range(3):
        sc.add_body(BodyAABB(i, np.ones(3) * 0.1))
    sc.build_pairs(parent_list=[-1, 0, 1])
    assert sc.num_pairs == 1


def test_build_pairs_three_independent_bodies():
    """Three bodies with no parent-child edges → 3 pairs."""
    sc = AABBSelfCollision()
    for i in range(3):
        sc.add_body(BodyAABB(i, np.ones(3) * 0.1))
    sc.build_pairs(parent_list=[-1, -1, -1])
    assert sc.num_pairs == 3


# ---------------------------------------------------------------------------
# Rotated body AABB projection
# ---------------------------------------------------------------------------


def test_rotated_body_aabb_expands():
    """A 45-degree rotated box has a larger world AABB than axis-aligned."""
    from physics.collision import _world_aabb
    from physics.spatial import rot_z

    half = np.array([0.2, 0.1, 0.05])
    babb = BodyAABB(0, half)

    X_identity = _transform((0.0, 0.0, 0.0))
    X_rotated = SpatialTransform(rot_z(np.pi / 4), np.zeros(3))

    _, max_id = _world_aabb(babb, X_identity)
    _, max_rot = _world_aabb(babb, X_rotated)

    # Rotated AABB x/y extents should be larger
    assert max_rot[0] > max_id[0], "Rotated AABB x should be larger"
    assert max_rot[1] > max_id[1], "Rotated AABB y should be larger"


# ---------------------------------------------------------------------------
# NullSelfCollision
# ---------------------------------------------------------------------------


def test_null_self_collision_always_zero():
    """NullSelfCollision returns zero forces regardless of overlap."""
    sc = NullSelfCollision()
    X = [_transform((0.0, 0.0, 0.0)), _transform((0.0, 0.0, 0.0))]
    v = [_zero_vel(), _zero_vel()]
    forces = sc.compute_forces(X, v, num_bodies=2)
    for f in forces:
        np.testing.assert_array_equal(f, np.zeros(6))
