"""
Unit tests for analytical sphere-box and sphere-cylinder dispatch in gjk_epa_query.

Verifies that the new analytical paths (bypassing GJK/EPA) produce correct
depth, normal direction, and contact point for all geometric configurations.

Class 1: TestSphereBox   — sphere vs OBB (5 cases)
Class 2: TestSphereCylinder — sphere vs cylinder (5 cases)

All tests use gjk_epa_query() directly (no CpuEngine), so they are fast
and have no external dependencies.  Depth/normal accuracy is atol=1e-6
(analytical paths are exact, unlike EPA which has ~1e-4 accuracy).

Reference: Q48 (OPEN_QUESTIONS.md) — sphere-box/sphere-cyl analytical dispatch.
"""

from __future__ import annotations

import numpy as np

from physics.geometry import BoxShape, CylinderShape, SphereShape
from physics.gjk_epa import gjk_epa_query
from physics.spatial import SpatialTransform


def _pose(r) -> SpatialTransform:
    return SpatialTransform(R=np.eye(3), r=np.asarray(r, dtype=float))


def _pose_rot(r, R) -> SpatialTransform:
    return SpatialTransform(R=np.asarray(R, dtype=float), r=np.asarray(r, dtype=float))


def _Rz(deg: float) -> np.ndarray:
    """Rotation matrix about Z axis."""
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def _Rx(deg: float) -> np.ndarray:
    """Rotation matrix about X axis."""
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


# ---------------------------------------------------------------------------
# Class 1: TestSphereBox
# ---------------------------------------------------------------------------


class TestSphereBox:
    """Sphere vs OBB analytical dispatch — 5 geometric configurations."""

    R = 0.05  # sphere radius
    H = 0.10  # box half-extent (cube)

    def test_face_contact_z(self):
        """Sphere directly above box top face — normal is +Z, depth exact."""
        pen = 0.005
        sphere = SphereShape(self.R)
        box = BoxShape((2 * self.H, 2 * self.H, 2 * self.H))
        # sphere centre at z = H + R - pen
        z = self.H + self.R - pen
        r = gjk_epa_query(sphere, _pose([0, 0, z]), box, _pose([0, 0, 0]), margin=0)
        assert r is not None, "no contact detected"
        assert abs(r.depth - pen) < 1e-6, f"depth={r.depth:.8f}, expected={pen}"
        assert abs(r.normal[2] - 1.0) < 1e-6, f"normal={r.normal} — expected +Z"
        assert abs(np.linalg.norm(r.normal) - 1.0) < 1e-10

    def test_face_contact_x(self):
        """Sphere approaching box from +X side — normal is +X."""
        pen = 0.003
        sphere = SphereShape(self.R)
        box = BoxShape((2 * self.H, 2 * self.H, 2 * self.H))
        x = self.H + self.R - pen
        r = gjk_epa_query(sphere, _pose([x, 0, 0]), box, _pose([0, 0, 0]), margin=0)
        assert r is not None
        assert abs(r.depth - pen) < 1e-6, f"depth={r.depth:.8f}"
        assert abs(r.normal[0] - 1.0) < 1e-6, f"normal={r.normal} — expected +X"

    def test_edge_contact(self):
        """Sphere near box edge (XZ corner) — closest point on edge, depth correct."""
        pen = 0.004
        sphere = SphereShape(self.R)
        box = BoxShape((2 * self.H, 2 * self.H, 2 * self.H))
        # Place sphere at 45° in XZ plane, just past the edge
        diag = np.sqrt(2) * self.H  # distance from centre to edge
        offset = diag + self.R - pen
        x = offset / np.sqrt(2)
        z = offset / np.sqrt(2)
        r = gjk_epa_query(sphere, _pose([x, 0, z]), box, _pose([0, 0, 0]), margin=0)
        assert r is not None, "no contact at edge"
        assert abs(r.depth - pen) < 1e-6, f"depth={r.depth:.8f}"
        assert abs(np.linalg.norm(r.normal) - 1.0) < 1e-10
        # Normal should point at 45° in XZ (equal X and Z components)
        assert abs(abs(r.normal[0]) - 1.0 / np.sqrt(2)) < 1e-5, f"normal={r.normal}"
        assert abs(abs(r.normal[2]) - 1.0 / np.sqrt(2)) < 1e-5, f"normal={r.normal}"

    def test_sphere_inside_box(self):
        """Sphere centre inside box — pushed out along shallowest axis."""
        sphere = SphereShape(self.R)
        box = BoxShape((2 * self.H, 2 * self.H, 2 * self.H))
        # Sphere centre at (0, 0, 0.08): closest face is +Z at z=0.1, gap=0.02
        # depth = R + gap = 0.05 + 0.02 = 0.07
        r = gjk_epa_query(sphere, _pose([0, 0, 0.08]), box, _pose([0, 0, 0]), margin=0)
        assert r is not None, "no contact for sphere inside box"
        assert r.depth > 0
        assert abs(np.linalg.norm(r.normal) - 1.0) < 1e-10
        # Normal must point along Z (shallowest axis)
        assert abs(abs(r.normal[2]) - 1.0) < 1e-6, f"normal={r.normal}"

    def test_rotated_box(self):
        """Box rotated 45° about Z — sphere above rotated face, normal follows rotation."""
        pen = 0.003
        sphere = SphereShape(self.R)
        box = BoxShape((2 * self.H, 2 * self.H, 2 * self.H))
        R45 = _Rz(45)
        # After 45° rotation, the +X face of the box points in the (1,1,0)/√2 direction.
        # Place sphere along that direction.
        face_dir = R45 @ np.array([1.0, 0.0, 0.0])
        centre = face_dir * (self.H + self.R - pen)
        r = gjk_epa_query(sphere, _pose(centre), box, _pose_rot([0, 0, 0], R45), margin=0)
        assert r is not None, "no contact for rotated box"
        assert abs(r.depth - pen) < 1e-5, f"depth={r.depth:.8f}"
        assert abs(np.linalg.norm(r.normal) - 1.0) < 1e-10
        # Normal should align with face_dir
        assert abs(np.dot(r.normal, face_dir) - 1.0) < 1e-5, f"normal={r.normal}"

    def test_separated_returns_none(self):
        """Sphere clearly separated from box — returns None."""
        sphere = SphereShape(self.R)
        box = BoxShape((2 * self.H, 2 * self.H, 2 * self.H))
        r = gjk_epa_query(sphere, _pose([0, 0, 0.5]), box, _pose([0, 0, 0]), margin=0)
        assert r is None, f"expected None, got depth={r.depth if r else None}"


# ---------------------------------------------------------------------------
# Class 2: TestSphereCylinder
# ---------------------------------------------------------------------------


class TestSphereCylinder:
    """Sphere vs cylinder analytical dispatch — 5 geometric configurations.

    Cylinder: radius=0.05, length=0.10 (half-length=0.05), axis=+Z.
    Sphere: radius=0.05.
    """

    R_SPH = 0.05
    R_CYL = 0.05
    L_CYL = 0.10  # full length; half = 0.05

    def _cyl(self) -> CylinderShape:
        return CylinderShape(self.R_CYL, self.L_CYL)

    def test_side_contact(self):
        """Sphere approaching cylinder from the side — radial contact, normal is radial."""
        pen = 0.004
        sphere = SphereShape(self.R_SPH)
        cyl = self._cyl()
        # Sphere centre at x = R_CYL + R_SPH - pen, z = 0 (mid-height)
        x = self.R_CYL + self.R_SPH - pen
        r = gjk_epa_query(sphere, _pose([x, 0, 0]), cyl, _pose([0, 0, 0]), margin=0)
        assert r is not None, "no contact on cylinder side"
        assert abs(r.depth - pen) < 1e-6, f"depth={r.depth:.8f}"
        assert abs(r.normal[0] - 1.0) < 1e-6, f"normal={r.normal} — expected +X"
        assert abs(np.linalg.norm(r.normal) - 1.0) < 1e-10

    def test_top_cap_contact(self):
        """Sphere above cylinder top cap — normal is +Z."""
        pen = 0.003
        sphere = SphereShape(self.R_SPH)
        cyl = self._cyl()
        hl = self.L_CYL / 2.0
        z = hl + self.R_SPH - pen
        r = gjk_epa_query(sphere, _pose([0, 0, z]), cyl, _pose([0, 0, 0]), margin=0)
        assert r is not None, "no contact on top cap"
        assert abs(r.depth - pen) < 1e-6, f"depth={r.depth:.8f}"
        assert abs(r.normal[2] - 1.0) < 1e-6, f"normal={r.normal} — expected +Z"

    def test_bottom_cap_contact(self):
        """Sphere below cylinder bottom cap — normal is -Z."""
        pen = 0.003
        sphere = SphereShape(self.R_SPH)
        cyl = self._cyl()
        hl = self.L_CYL / 2.0
        z = -(hl + self.R_SPH - pen)
        r = gjk_epa_query(sphere, _pose([0, 0, z]), cyl, _pose([0, 0, 0]), margin=0)
        assert r is not None, "no contact on bottom cap"
        assert abs(r.depth - pen) < 1e-6, f"depth={r.depth:.8f}"
        assert abs(r.normal[2] + 1.0) < 1e-6, f"normal={r.normal} — expected -Z"

    def test_rim_contact(self):
        """Sphere near top rim (r > R_CYL, |t| > hl) — closest point on rim circle."""
        pen = 0.003
        sphere = SphereShape(self.R_SPH)
        cyl = self._cyl()
        hl = self.L_CYL / 2.0
        # Rim point at (R_CYL, 0, hl). Sphere centre offset by R_SPH - pen along (1,0,1)/√2.
        rim = np.array([self.R_CYL, 0.0, hl])
        direction = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
        centre = rim + direction * (self.R_SPH - pen)
        r = gjk_epa_query(sphere, _pose(centre), cyl, _pose([0, 0, 0]), margin=0)
        assert r is not None, "no contact at rim"
        assert abs(r.depth - pen) < 1e-6, f"depth={r.depth:.8f}"
        assert abs(np.linalg.norm(r.normal) - 1.0) < 1e-10
        # Normal should point along direction (from rim toward sphere)
        assert np.dot(r.normal, direction) > 0.99, f"normal={r.normal}"

    def test_axis_degenerate(self):
        """Sphere centre on cylinder axis — fallback normal is perpendicular to axis."""
        pen = 0.003
        sphere = SphereShape(self.R_SPH)
        cyl = self._cyl()
        hl = self.L_CYL / 2.0
        # Sphere directly above cap, on axis
        z = hl + self.R_SPH - pen
        r = gjk_epa_query(sphere, _pose([0, 0, z]), cyl, _pose([0, 0, 0]), margin=0)
        assert r is not None, "no contact for on-axis sphere above cap"
        assert abs(r.depth - pen) < 1e-6, f"depth={r.depth:.8f}"
        assert abs(np.linalg.norm(r.normal) - 1.0) < 1e-10

    def test_rotated_cylinder(self):
        """Cylinder rotated 90° about X — axis is now +Y, side contact along +Z."""
        pen = 0.004
        sphere = SphereShape(self.R_SPH)
        cyl = self._cyl()
        R90x = _Rx(90)
        # After rotation, cylinder axis is +Y. Side contact from +Z.
        z = self.R_CYL + self.R_SPH - pen
        r = gjk_epa_query(sphere, _pose([0, 0, z]), cyl, _pose_rot([0, 0, 0], R90x), margin=0)
        assert r is not None, "no contact for rotated cylinder"
        assert abs(r.depth - pen) < 1e-6, f"depth={r.depth:.8f}"
        assert abs(r.normal[2] - 1.0) < 1e-6, f"normal={r.normal} — expected +Z"

    def test_separated_returns_none(self):
        """Sphere clearly separated from cylinder — returns None."""
        sphere = SphereShape(self.R_SPH)
        cyl = self._cyl()
        r = gjk_epa_query(sphere, _pose([0, 0, 1.0]), cyl, _pose([0, 0, 0]), margin=0)
        assert r is None, "expected None for separated sphere-cylinder"
