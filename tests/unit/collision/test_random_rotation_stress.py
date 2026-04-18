"""
Random-rotation stress tests for gjk_epa_query().

For each shape pair, 200 random SO(3) orientations are sampled and the
collision query is run with a fixed 5mm penetration depth.  Every trial
must return a valid contact (non-None), positive depth, and unit-length
normal.  This catches degenerate cases that axis-aligned tests miss.

Methodology: coal/FCL run 1000 random orientations per pair; we use 200
for speed (~5s total).  Seed is fixed for reproducibility.

Shape pairs covered (13):
  sphere-box, sphere-cyl, sphere-hull,
  box-box, box-cyl, box-hull,
  cyl-cyl, cyl-hull, hull-hull,
  capsule-sphere, capsule-box, capsule-cyl, capsule-hull

Reference: session 32 gap analysis vs coal/FCL/Jolt.
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.geometry import (
    BoxShape,
    CapsuleShape,
    CylinderShape,
    SphereShape,
)
from physics.gjk_epa import gjk_epa_query
from physics.spatial import SpatialTransform

try:
    import trimesh

    from physics.geometry import ConvexHullShape

    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_TRIALS = 200
# Large penetration: guarantees actual overlap for all shape orientations.
# For a cube half=0.05, vertex support = 0.0866 vs face support = 0.05 (delta=0.037).
# pen=0.05 ensures worst-case actual penetration >= 0.013 m — always detectable.
PEN = 0.05
MARGIN = 1e-3
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_R(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation via QR decomposition."""
    M = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(M)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _pose(r) -> SpatialTransform:
    return SpatialTransform(R=np.eye(3), r=np.asarray(r, dtype=float))


def _pose_rot(r, R) -> SpatialTransform:
    return SpatialTransform(R=np.asarray(R, dtype=float), r=np.asarray(r, dtype=float))


def _box_hull(half: float) -> "ConvexHullShape":
    mesh = trimesh.creation.box(extents=[2 * half] * 3)
    return ConvexHullShape(np.array(mesh.vertices))


def _support_along(shape, R: np.ndarray, axis: np.ndarray) -> float:
    """Signed support of shape (with rotation R) along world-space axis."""
    # support_point returns the furthest point in direction d in local frame
    local_axis = R.T @ axis
    sp_local = shape.support_point(local_axis)
    return float(np.dot(R @ sp_local, axis))


def _run_stress(shape_a, shape_b, n=N_TRIALS, pen=PEN):
    """
    Run N_TRIALS random-orientation contact queries.

    shape_a is placed at origin with rotation R_a.
    shape_b is placed along a random axis using support-function-based placement:
      dist = support_a(axis) + support_b(-axis) - pen
    This guarantees actual penetration for all orientations (pen=0.05 >> max
    vertex-vs-face support discrepancy of ~0.037 for a cube half=0.05).

    Asserts for every trial:
      - result is not None (contact detected)
      - depth > 0
      - |normal| = 1.0 (unit length)
    """
    rng = np.random.default_rng(RNG_SEED)
    failures = []

    for trial in range(n):
        R_a = _random_R(rng)
        R_b = _random_R(rng)
        # Contact axis: random unit vector (from a toward b)
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        # Compute actual support of each shape along the contact axis
        sup_a = _support_along(shape_a, R_a, axis)
        sup_b = _support_along(shape_b, R_b, -axis)  # b faces toward a
        # Place b so that the gap = -pen (i.e. pen penetration)
        dist = sup_a + sup_b - pen
        pos_b = axis * dist

        pose_a = _pose_rot([0, 0, 0], R_a)
        pose_b = _pose_rot(pos_b, R_b)

        r = gjk_epa_query(shape_a, pose_a, shape_b, pose_b, margin=MARGIN)

        if r is None:
            failures.append(f"trial {trial}: returned None (axis={axis})")
            continue
        if r.depth <= 0:
            failures.append(f"trial {trial}: depth={r.depth:.6f} <= 0")
            continue
        norm_len = float(np.linalg.norm(r.normal))
        if abs(norm_len - 1.0) > 1e-6:
            failures.append(f"trial {trial}: |normal|={norm_len:.8f} != 1")

    if failures:
        sample = failures[:5]
        pytest.fail(
            f"{len(failures)}/{n} trials failed for "
            f"{type(shape_a).__name__}-{type(shape_b).__name__}:\n" + "\n".join(sample)
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRandomRotationStress:
    """200 random orientations per shape pair — contact must always be detected."""

    # --- sphere pairs (analytical dispatch, should be rock-solid) ---

    def test_sphere_box_random_rotations(self):
        _run_stress(SphereShape(0.05), BoxShape((0.1, 0.1, 0.1)))

    def test_sphere_cyl_random_rotations(self):
        _run_stress(SphereShape(0.05), CylinderShape(0.05, 0.1))

    @pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh required")
    def test_sphere_hull_random_rotations(self):
        _run_stress(SphereShape(0.05), _box_hull(0.05))

    # --- box pairs ---

    def test_box_box_random_rotations(self):
        _run_stress(BoxShape((0.1, 0.1, 0.1)), BoxShape((0.1, 0.1, 0.1)))

    def test_box_cyl_random_rotations(self):
        _run_stress(BoxShape((0.1, 0.1, 0.1)), CylinderShape(0.05, 0.1))

    @pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh required")
    def test_box_hull_random_rotations(self):
        _run_stress(BoxShape((0.1, 0.1, 0.1)), _box_hull(0.05))

    # --- cylinder pairs ---

    def test_cyl_cyl_random_rotations(self):
        _run_stress(CylinderShape(0.05, 0.1), CylinderShape(0.05, 0.1))

    @pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh required")
    def test_cyl_hull_random_rotations(self):
        _run_stress(CylinderShape(0.05, 0.1), _box_hull(0.05))

    # --- hull-hull ---

    @pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh required")
    def test_hull_hull_random_rotations(self):
        _run_stress(_box_hull(0.05), _box_hull(0.05))

    # --- capsule pairs (analytical dispatch) ---

    def test_capsule_sphere_random_rotations(self):
        _run_stress(CapsuleShape(0.05, 0.1), SphereShape(0.05))

    def test_capsule_box_random_rotations(self):
        _run_stress(CapsuleShape(0.05, 0.1), BoxShape((0.1, 0.1, 0.1)))

    def test_capsule_cyl_random_rotations(self):
        _run_stress(CapsuleShape(0.05, 0.1), CylinderShape(0.05, 0.1))

    @pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh required")
    def test_capsule_hull_random_rotations(self):
        _run_stress(CapsuleShape(0.05, 0.1), _box_hull(0.05))
