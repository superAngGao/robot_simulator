"""Unit tests for cylinder_cylinder_manifold analytical contact."""

import numpy as np

from physics.geometry import CylinderShape
from physics.narrowphase_analytical import cylinder_cylinder_manifold
from physics.spatial import SpatialTransform


def _pose(pos, rpy=(0, 0, 0)):
    from scipy.spatial.transform import Rotation

    R = Rotation.from_euler("xyz", rpy).as_matrix()
    return SpatialTransform(R=R, r=np.array(pos, dtype=float))


def _identity(pos=(0, 0, 0)):
    return SpatialTransform(R=np.eye(3), r=np.array(pos, dtype=float))


# ---------------------------------------------------------------------------
# coaxial (same axis, side-by-side radially)
# ---------------------------------------------------------------------------


def test_coaxial_penetrating():
    """Two coaxial cylinders (same Z axis) separated radially — should contact."""
    cyl = CylinderShape(radius=0.1, length=0.4)
    pose_a = _identity([0, 0, 0])
    pose_b = _identity([0.15, 0, 0])  # 0.15 < 0.1+0.1=0.2 → penetrating
    m = cylinder_cylinder_manifold(cyl, pose_a, cyl, pose_b)
    assert m is not None
    assert m.depth > 0
    # Normal should point roughly in +X (from b toward a)
    assert abs(m.normal[0]) > 0.9


def test_coaxial_separated():
    """Two coaxial cylinders separated beyond r_sum — no contact."""
    cyl = CylinderShape(radius=0.1, length=0.4)
    pose_a = _identity([0, 0, 0])
    pose_b = _identity([0.25, 0, 0])  # 0.25 > 0.2
    m = cylinder_cylinder_manifold(cyl, pose_a, cyl, pose_b)
    assert m is None


# ---------------------------------------------------------------------------
# side contact (axes parallel, cylinders side by side)
# ---------------------------------------------------------------------------


def test_side_contact_near_parallel_2pts():
    """Two parallel cylinders lying side by side → near-parallel → 2 contact points."""
    cyl = CylinderShape(radius=0.05, length=0.5)
    # Both axes along Z, offset in X by just under r_sum
    pose_a = _identity([0, 0, 0])
    pose_b = _identity([0.09, 0, 0])  # 0.09 < 0.10
    m = cylinder_cylinder_manifold(cyl, pose_a, cyl, pose_b)
    assert m is not None
    assert len(m.points) == 2, f"expected 2 pts, got {len(m.points)}"
    assert m.depth > 0


def test_side_contact_depth():
    """Depth should equal r_sum - distance for parallel side contact."""
    cyl = CylinderShape(radius=0.1, length=0.6)
    gap = 0.15  # < r_sum=0.2
    pose_a = _identity([0, 0, 0])
    pose_b = _identity([gap, 0, 0])
    m = cylinder_cylinder_manifold(cyl, pose_a, cyl, pose_b)
    assert m is not None
    expected_depth = 0.2 - gap
    assert abs(m.depth - expected_depth) < 1e-9


# ---------------------------------------------------------------------------
# tilted (crossing axes — single contact point)
# ---------------------------------------------------------------------------


def test_tilted_single_point():
    """Two cylinders with perpendicular axes crossing → 1 contact point."""
    cyl = CylinderShape(radius=0.05, length=0.4)
    pose_a = _identity([0, 0, 0])  # axis along Z
    # Rotate 90° around X so axis_b is along Y
    pose_b = _pose([0, 0, 0], rpy=(np.pi / 2, 0, 0))
    # Move b slightly so they just touch
    pose_b = SpatialTransform(R=pose_b.R, r=np.array([0, 0, 0.08]))
    m = cylinder_cylinder_manifold(cyl, pose_a, cyl, pose_b)
    assert m is not None
    assert len(m.points) == 1


# ---------------------------------------------------------------------------
# near-parallel overlap interval edge cases
# ---------------------------------------------------------------------------


def test_near_parallel_partial_overlap():
    """Axes parallel but one cylinder shorter — overlap < full length → 2 pts."""
    cyl_long = CylinderShape(radius=0.05, length=1.0)
    cyl_short = CylinderShape(radius=0.05, length=0.3)
    pose_a = _identity([0, 0, 0])
    pose_b = _identity([0.08, 0, 0])  # side contact, partial axial overlap
    m = cylinder_cylinder_manifold(cyl_long, pose_a, cyl_short, pose_b)
    assert m is not None
    assert len(m.points) == 2


def test_near_parallel_no_axial_overlap():
    """Axes parallel but cylinders offset axially beyond overlap → 1 pt (end contact)."""
    cyl = CylinderShape(radius=0.05, length=0.2)
    pose_a = _identity([0, 0, 0])
    # Offset axially so segments don't overlap, but radially close
    pose_b = _identity([0.08, 0, 0.5])
    m = cylinder_cylinder_manifold(cyl, pose_a, cyl, pose_b)
    # Either None (separated) or 1 pt — not 2 pts
    if m is not None:
        assert len(m.points) == 1
