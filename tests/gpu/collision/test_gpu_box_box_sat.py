"""GPU box-box SAT collision tests.

Verifies that the Warp SAT kernel for OBB-OBB produces depth and normal
consistent with CPU GJK/EPA for various box-box configurations:
  - face-face (axis-aligned)
  - face-face (rotated)
  - edge-edge (crossed bars)
  - separated (no contact)
  - asymmetric half-extents

Reference: Ericson (2004) §4.4, session 27 Q42.3.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import warp as wp

    from physics.backends.warp.analytical_collision import (
        box_box,
        box_box_contact_point,
        box_box_normal,
    )

    HAS_WARP = True
except Exception:
    HAS_WARP = False

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not HAS_WARP, reason="Warp or CUDA not available"),
]

# Warp kernel defined at module level (only when warp is available)
if HAS_WARP:

    @wp.kernel
    def _box_box_test_kernel(
        in_pos_a: wp.array(dtype=wp.vec3),
        in_R_a: wp.array(dtype=wp.mat33),
        in_half_a: wp.array(dtype=wp.vec3),
        in_pos_b: wp.array(dtype=wp.vec3),
        in_R_b: wp.array(dtype=wp.mat33),
        in_half_b: wp.array(dtype=wp.vec3),
        out_depth: wp.array(dtype=wp.float32),
        out_hit: wp.array(dtype=wp.float32),
        out_normal: wp.array(dtype=wp.vec3),
        out_cp: wp.array(dtype=wp.vec3),
    ):
        pa = in_pos_a[0]
        Ra = in_R_a[0]
        ha = in_half_a[0]
        pb = in_pos_b[0]
        Rb = in_R_b[0]
        hb = in_half_b[0]

        res = box_box(pa, Ra, ha[0], ha[1], ha[2], pb, Rb, hb[0], hb[1], hb[2])
        out_depth[0] = res[0]
        out_hit[0] = res[1]

        n = box_box_normal(pa, Ra, ha[0], ha[1], ha[2], pb, Rb, hb[0], hb[1], hb[2])
        out_normal[0] = n

        out_cp[0] = box_box_contact_point(pa, Ra, ha[0], ha[1], ha[2], pb, Rb, hb[0], hb[1], hb[2], n)


def _run_box_box_sat(pos_a, R_a, half_a, pos_b, R_b, half_b):
    """Run GPU box_box SAT and return (depth, hit, normal, contact_point)."""
    in_pos_a = wp.array([pos_a.astype(np.float32)], dtype=wp.vec3)
    in_R_a = wp.array([R_a.astype(np.float32).flatten()], dtype=wp.mat33)
    in_half_a = wp.array([half_a.astype(np.float32)], dtype=wp.vec3)
    in_pos_b = wp.array([pos_b.astype(np.float32)], dtype=wp.vec3)
    in_R_b = wp.array([R_b.astype(np.float32).flatten()], dtype=wp.mat33)
    in_half_b = wp.array([half_b.astype(np.float32)], dtype=wp.vec3)

    out_depth = wp.zeros(1, dtype=wp.float32)
    out_hit = wp.zeros(1, dtype=wp.float32)
    out_normal = wp.zeros(1, dtype=wp.vec3)
    out_cp = wp.zeros(1, dtype=wp.vec3)

    wp.launch(
        _box_box_test_kernel,
        dim=1,
        inputs=[
            in_pos_a,
            in_R_a,
            in_half_a,
            in_pos_b,
            in_R_b,
            in_half_b,
            out_depth,
            out_hit,
            out_normal,
            out_cp,
        ],
    )
    wp.synchronize()

    depth = float(out_depth.numpy()[0])
    hit = float(out_hit.numpy()[0])
    normal = np.array(out_normal.numpy()[0])
    cp = np.array(out_cp.numpy()[0])
    return depth, hit, normal, cp


def _cpu_epa(pos_a, R_a, half_a, pos_b, R_b, half_b):
    """CPU GJK/EPA reference for the same box pair."""
    from physics.geometry import BoxShape
    from physics.gjk_epa import gjk_epa_query
    from physics.spatial import SpatialTransform

    a = BoxShape(tuple(s * 2.0 for s in half_a))
    b = BoxShape(tuple(s * 2.0 for s in half_b))
    pa = SpatialTransform(R_a, pos_a)
    pb = SpatialTransform(R_b, pos_b)
    m = gjk_epa_query(a, pa, b, pb)
    if m is None:
        return 0.0, False, np.zeros(3), np.zeros(3)
    return m.depth, True, m.normal, m.points[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

I3 = np.eye(3, dtype=np.float64)


class TestBoxBoxSATFaceFace:
    """Axis-aligned face-face overlap."""

    def test_depth_z_overlap(self):
        """Two unit boxes, A above B, 0.1 overlap in Z."""
        depth, hit, normal, _ = _run_box_box_sat(
            np.array([0, 0, 0.9]),
            I3,
            np.array([0.5, 0.5, 0.5]),
            np.zeros(3),
            I3,
            np.array([0.5, 0.5, 0.5]),
        )
        assert hit > 0.5
        assert depth == pytest.approx(0.1, abs=1e-3)

    def test_normal_z(self):
        """Normal should point along Z (from B to A)."""
        _, _, normal, _ = _run_box_box_sat(
            np.array([0, 0, 0.9]),
            I3,
            np.array([0.5, 0.5, 0.5]),
            np.zeros(3),
            I3,
            np.array([0.5, 0.5, 0.5]),
        )
        assert abs(normal[2]) > 0.9  # ~Z direction

    def test_x_overlap(self):
        """Overlap along X axis."""
        depth, hit, normal, _ = _run_box_box_sat(
            np.array([0.8, 0, 0]),
            I3,
            np.array([0.5, 0.5, 0.5]),
            np.zeros(3),
            I3,
            np.array([0.5, 0.5, 0.5]),
        )
        assert hit > 0.5
        assert depth == pytest.approx(0.2, abs=1e-3)
        assert abs(normal[0]) > 0.9

    def test_asymmetric_half_extents(self):
        """Different box sizes."""
        depth, hit, _, _ = _run_box_box_sat(
            np.array([0, 0, 0.35]),
            I3,
            np.array([0.5, 0.5, 0.25]),
            np.zeros(3),
            I3,
            np.array([1.0, 1.0, 0.5]),
        )
        assert hit > 0.5
        # Expected: 0.25 + 0.5 - 0.35 = 0.4
        assert depth == pytest.approx(0.4, abs=1e-3)

    def test_cpu_gpu_depth_agreement(self):
        """GPU SAT depth matches CPU EPA depth."""
        pos_a = np.array([0.1, -0.05, 0.85])
        half_a = np.array([0.5, 0.5, 0.5])
        half_b = np.array([0.6, 0.7, 0.5])

        gpu_depth, _, _, _ = _run_box_box_sat(pos_a, I3, half_a, np.zeros(3), I3, half_b)
        cpu_depth, _, _, _ = _cpu_epa(pos_a, I3, half_a, np.zeros(3), I3, half_b)

        assert gpu_depth == pytest.approx(cpu_depth, abs=1e-2)


class TestBoxBoxSATRotated:
    """Rotated box-box contacts."""

    def _rot_y(self, angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

    def _rot_z(self, angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    def test_rotated_45_y(self):
        """Box A rotated 45 deg around Y -- edge-face contact."""
        R = self._rot_y(np.pi / 4)
        depth, hit, _, _ = _run_box_box_sat(
            np.array([0, 0, 1.1]),
            R,
            np.array([0.5, 0.5, 0.5]),
            np.zeros(3),
            I3,
            np.array([0.5, 0.5, 0.5]),
        )
        assert hit > 0.5
        assert depth > 0.0

    def test_rotated_45_y_cpu_agreement(self):
        R = self._rot_y(np.pi / 4)
        pos_a = np.array([0, 0, 1.1])
        half = np.array([0.5, 0.5, 0.5])

        gpu_depth, _, gpu_n, _ = _run_box_box_sat(pos_a, R, half, np.zeros(3), I3, half)
        cpu_depth, _, cpu_n, _ = _cpu_epa(pos_a, R, half, np.zeros(3), I3, half)

        assert gpu_depth == pytest.approx(cpu_depth, abs=2e-2)
        # Normal direction should agree (dot > 0.9)
        assert abs(np.dot(gpu_n, cpu_n)) > 0.9

    def test_crossed_boxes_edge_edge(self):
        """Two thin boxes crossed -- edge-edge contact."""
        Rz90 = self._rot_z(np.pi / 2)
        depth, hit, normal, _ = _run_box_box_sat(
            np.array([0, 0, 0.25]),
            I3,
            np.array([1.0, 0.15, 0.15]),
            np.zeros(3),
            Rz90,
            np.array([1.0, 0.15, 0.15]),
        )
        assert hit > 0.5
        # Expected: 0.15 + 0.15 - 0.25 = 0.05
        assert depth == pytest.approx(0.05, abs=1e-2)

    def test_both_rotated(self):
        """Both boxes rotated, edge-edge axis is minimum."""
        Ry30 = self._rot_y(0.5)
        c, s = np.cos(0.7), np.sin(0.7)
        Rx40 = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
        pos_a = np.array([0, 0, 1.1])
        half = np.array([0.5, 0.5, 0.5])

        gpu_depth, _, gpu_n, _ = _run_box_box_sat(pos_a, Ry30, half, np.zeros(3), Rx40, half)
        cpu_depth, _, cpu_n, _ = _cpu_epa(pos_a, Ry30, half, np.zeros(3), Rx40, half)

        assert gpu_depth == pytest.approx(cpu_depth, abs=2e-2)
        assert abs(np.dot(gpu_n, cpu_n)) > 0.9


class TestBoxBoxSATSeparated:
    """No-contact cases should return hit=0."""

    def test_separated_z(self):
        _, hit, _, _ = _run_box_box_sat(
            np.array([0, 0, 1.5]),
            I3,
            np.array([0.5, 0.5, 0.5]),
            np.zeros(3),
            I3,
            np.array([0.5, 0.5, 0.5]),
        )
        assert hit < 0.5

    def test_separated_diagonal(self):
        _, hit, _, _ = _run_box_box_sat(
            np.array([2.0, 2.0, 2.0]),
            I3,
            np.array([0.5, 0.5, 0.5]),
            np.zeros(3),
            I3,
            np.array([0.5, 0.5, 0.5]),
        )
        assert hit < 0.5

    def test_barely_separated(self):
        """Gap = 0.01 -- should not register contact."""
        _, hit, _, _ = _run_box_box_sat(
            np.array([0, 0, 1.01]),
            I3,
            np.array([0.5, 0.5, 0.5]),
            np.zeros(3),
            I3,
            np.array([0.5, 0.5, 0.5]),
        )
        assert hit < 0.5
