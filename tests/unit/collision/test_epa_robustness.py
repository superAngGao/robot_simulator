"""
Tests for EPA degenerate simplex robustness.

Covers the bug where GJK returns a 4-point simplex with a face passing
through the origin, causing EPA to select the wrong expansion direction
and converge to an incorrect depth/normal.

Fix: detect degenerate tetrahedron (min face dist < 1e-8) → rebuild as
hexahedron; skip degenerate faces in main loop.

Reference: session 30 diagnosis, plan floating-drifting-yeti.md Part 1.
"""

import numpy as np
import pytest

from physics.geometry import BoxShape, ConvexHullShape, SphereShape
from physics.gjk_epa import epa, gjk, gjk_epa_query
from physics.spatial import SpatialTransform


def _box(half: float) -> BoxShape:
    """BoxShape with given half-extent (size = 2*half)."""
    return BoxShape((2 * half, 2 * half, 2 * half))


try:
    import trimesh

    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

pytestmark = pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh required")


def _box_hull(half: float) -> ConvexHullShape:
    mesh = trimesh.creation.box(extents=[2 * half] * 3)
    return ConvexHullShape(np.array(mesh.vertices))


def _pose(r):
    return SpatialTransform(R=np.eye(3), r=np.asarray(r, dtype=float))


# ---------------------------------------------------------------------------
# Class 1: The exact bug + systematic degeneracy
# ---------------------------------------------------------------------------


class TestEPADegenerateFace:
    """The exact bug scenario and systematic degeneracy coverage."""

    def test_sphere_vs_hull_face_through_origin(self):
        """THE BUG: sphere(r=0.1)@z=0.25 vs box-hull(half=0.2). EPA used to
        return depth=0.02, normal diagonal. Fixed: depth≈0.05, normal≈(0,0,1)."""
        sphere = SphereShape(radius=0.1)
        hull = _box_hull(0.2)
        r = gjk_epa_query(sphere, _pose([0, 0, 0.25]), hull, _pose([0, 0, 0]), margin=0)
        assert r is not None
        assert abs(r.depth - 0.05) < 5e-3, f"depth={r.depth:.4f}, expected ~0.05"
        assert r.normal[2] > 0.95, f"normal={r.normal}, expected ~(0,0,1)"

    def test_sphere_vs_hull_medium_penetration(self):
        """Medium penetration (20mm) — EPA must be accurate after fix."""
        sphere = SphereShape(radius=0.1)
        hull = _box_hull(0.2)
        # depth=0.02: sphere center at z = 0.2 + 0.1 - 0.02 = 0.28
        r = gjk_epa_query(sphere, _pose([0, 0, 0.28]), hull, _pose([0, 0, 0]), margin=0)
        assert r is not None
        assert abs(r.depth - 0.02) < 3e-3
        assert r.normal[2] > 0.9

    def test_sphere_vs_hull_deep_penetration(self):
        """Deep penetration (depth≈0.15) — EPA must not diverge."""
        sphere = SphereShape(radius=0.1)
        hull = _box_hull(0.2)
        r = gjk_epa_query(sphere, _pose([0, 0, 0.15]), hull, _pose([0, 0, 0]), margin=0)
        assert r is not None
        assert abs(r.depth - 0.15) < 1e-2
        assert r.normal[2] > 0.9

    def test_sphere_on_hull_edge_xz(self):
        """Sphere at +x+z edge of box — non-axis-aligned contact."""
        sphere = SphereShape(radius=0.05)
        hull = _box_hull(0.1)
        r = gjk_epa_query(sphere, _pose([0.13, 0, 0.13]), hull, _pose([0, 0, 0]), margin=0)
        assert r is not None
        assert r.depth > 0
        assert abs(np.linalg.norm(r.normal) - 1.0) < 1e-6

    def test_sphere_on_hull_corner(self):
        """Sphere at (+x,+y,+z) corner — most degenerate GJK simplex.
        Sphere center at (0.12, 0.12, 0.12), corner at (0.1, 0.1, 0.1),
        dist = sqrt(3)*0.02 ≈ 0.035 < r=0.05 → depth ≈ 0.015."""
        sphere = SphereShape(radius=0.05)
        hull = _box_hull(0.1)
        r = gjk_epa_query(sphere, _pose([0.12, 0.12, 0.12]), hull, _pose([0, 0, 0]), margin=0)
        assert r is not None
        assert r.depth > 0
        assert abs(np.linalg.norm(r.normal) - 1.0) < 1e-6

    def test_sphere_centered_inside_hull(self):
        """Sphere at hull center — maximum penetration, any valid normal."""
        sphere = SphereShape(radius=0.05)
        hull = _box_hull(0.2)
        r = gjk_epa_query(sphere, _pose([0, 0, 0]), hull, _pose([0, 0, 0]), margin=0)
        assert r is not None
        assert r.depth > 0.1
        assert abs(np.linalg.norm(r.normal) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Class 2: Systematic depth sweep
# ---------------------------------------------------------------------------


class TestEPAAccuracySweep:
    """Sweep penetration depths to verify EPA accuracy across the range."""

    DEPTHS = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2]

    def _check_depth(self, shape_a, pose_a, shape_b, pose_b, expected_depth, atol=2e-3):
        r = gjk_epa_query(shape_a, pose_a, shape_b, pose_b, margin=0)
        assert r is not None, f"expected contact at depth={expected_depth}"
        assert abs(r.depth - expected_depth) < atol, f"depth={r.depth:.4f}, expected={expected_depth:.4f}"

    def test_depth_sweep_sphere_box(self):
        """Sphere vs box: 8 depths from 1mm to 200mm."""
        sphere = SphereShape(radius=0.1)
        box = _box(0.2)
        for d in self.DEPTHS:
            z = 0.2 + 0.1 - d
            self._check_depth(sphere, _pose([0, 0, z]), box, _pose([0, 0, 0]), d)

    def test_depth_sweep_box_box(self):
        """Box vs box: 8 depths."""
        box_a = _box(0.1)
        box_b = _box(0.1)
        for d in self.DEPTHS:
            z = 0.2 - d
            self._check_depth(box_a, _pose([0, 0, z]), box_b, _pose([0, 0, 0]), d)

    def test_depth_sweep_hull_hull(self):
        """ConvexHull vs ConvexHull: 8 depths."""
        hull_a = _box_hull(0.1)
        hull_b = _box_hull(0.1)
        for d in self.DEPTHS:
            z = 0.2 - d
            self._check_depth(hull_a, _pose([0, 0, z]), hull_b, _pose([0, 0, 0]), d)


# ---------------------------------------------------------------------------
# Class 3: Rotation stress
# ---------------------------------------------------------------------------


class TestEPARotationStress:
    """EPA correctness under various rotation configurations."""

    def _rot_x(self, angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def _rot_z(self, angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def test_sphere_box_6_axis_aligned(self):
        """Sphere penetrating box from ±x, ±y, ±z — normal must align with axis."""
        sphere = SphereShape(radius=0.05)
        box = _box(0.1)
        depth = 0.01
        for axis in range(3):
            for sign in [1, -1]:
                r_vec = np.zeros(3)
                r_vec[axis] = sign * (0.1 + 0.05 - depth)
                r = gjk_epa_query(sphere, _pose(r_vec), box, _pose([0, 0, 0]), margin=0)
                assert r is not None
                assert abs(r.depth - depth) < 2e-3, f"axis={axis} sign={sign} depth={r.depth}"
                assert r.normal[axis] * sign > 0.9, f"axis={axis} sign={sign} normal={r.normal}"

    def test_rotated_box_vs_box(self):
        """Box rotated 30°, 45°, 60° about z — depth must be positive."""
        box_a = _box(0.1)
        box_b = _box(0.1)
        for angle in [30, 45, 60]:
            R = self._rot_z(np.radians(angle))
            r = gjk_epa_query(
                box_a,
                SpatialTransform(R=R, r=np.array([0, 0, 0.195])),
                box_b,
                _pose([0, 0, 0]),
                margin=0,
            )
            assert r is not None, f"angle={angle}"
            assert r.depth > 0, f"angle={angle}"
            assert abs(np.linalg.norm(r.normal) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Class 4: Direct EPA init path unit tests
# ---------------------------------------------------------------------------


class TestEPADirectCall:
    """Unit tests for EPA initialization paths (hexahedron rebuild)."""

    def _sphere_box_simplex(self, depth=0.01):
        """Return a GJK simplex for sphere vs box with given depth."""
        sphere = SphereShape(radius=0.1)
        box = BoxShape(half_extents=np.array([0.1, 0.1, 0.1]))
        z = 0.2 - depth
        intersecting, simplex = gjk(sphere, _pose([0, 0, z]), box, _pose([0, 0, 0]))
        assert intersecting
        return sphere, _pose([0, 0, z]), box, _pose([0, 0, 0]), simplex

    def test_epa_4pt_degenerate_triggers_rebuild(self):
        """The exact bug: 4-pt simplex with face through origin → hexahedron rebuild
        → correct depth/normal."""
        sphere = SphereShape(radius=0.1)
        hull = _box_hull(0.2)
        # This specific configuration triggers the degenerate 4-pt simplex
        intersecting, simplex = gjk(sphere, _pose([0, 0, 0.25]), hull, _pose([0, 0, 0]))
        assert intersecting
        assert len(simplex) == 4, f"expected 4-pt simplex, got {len(simplex)}"
        normal, depth = epa(sphere, _pose([0, 0, 0.25]), hull, _pose([0, 0, 0]), simplex)
        assert abs(depth - 0.05) < 5e-3, f"depth={depth:.4f}"
        assert normal[2] > 0.95, f"normal={normal}"

    def test_epa_normal_sphere_box_depth_sweep(self):
        """EPA gives correct depth for 5 different penetration depths."""
        sphere = SphereShape(radius=0.1)
        box = _box(0.1)
        for depth in [0.005, 0.02, 0.05, 0.1, 0.18]:
            z = 0.2 - depth
            intersecting, simplex = gjk(sphere, _pose([0, 0, z]), box, _pose([0, 0, 0]))
            assert intersecting
            normal, d = epa(sphere, _pose([0, 0, z]), box, _pose([0, 0, 0]), simplex)
            assert abs(d - depth) < 3e-3, f"depth={d:.4f}, expected={depth}"
            assert normal[2] > 0.9, f"normal={normal}"
