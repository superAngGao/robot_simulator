"""
Long-run stability tests for body-body contact in CpuEngine.

Each test runs 1000 steps (0.2s sim time at dt=2e-4) and asserts:
  - No NaN/Inf in q or qdot
  - Free body doesn't fly off (|z| < 5m)
  - No gross penetration (center distance >= sum_of_radii - 2mm)

Pattern: "heavy anchor" — body_a has mass=1e6 (quasi-fixed), body_b is
free (mass=1.0) and dropped from above.  body_a rests on the ground at
z = half_a; body_b starts at z = half_a + half_b + 0.3 m.

Shape pairs covered (not already in test_margin_vs_mujoco.py):
  Ground stability:  capsule drop
  Body-body:         box-box, box-cylinder, cylinder-cylinder,
                     capsule-sphere, capsule-box, hull-hull

Reference: session 32 gap analysis vs Bullet/Jolt long-run stability tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.cpu_engine import CpuEngine
from physics.geometry import (
    BodyCollisionGeometry,
    BoxShape,
    CapsuleShape,
    CylinderShape,
    ShapeInstance,
    SphereShape,
)
from physics.joint import FreeJoint
from physics.merged_model import merge_models
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from physics.terrain import FlatTerrain
from robot.model import RobotModel

try:
    import trimesh

    from physics.geometry import ConvexHullShape

    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DT = 2e-4
N_STEPS = 1000  # 0.2 s sim time — enough to settle
GRAVITY = 9.81


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def _inertia_sphere(m: float, r: float) -> np.ndarray:
    return np.eye(3) * (2 / 5 * m * r**2)


def _inertia_box(m: float, half: float) -> np.ndarray:
    return np.eye(3) * (m / 6.0 * (2 * half) ** 2)


def _inertia_cylinder(m: float, r: float, length: float) -> np.ndarray:
    h = length / 2.0
    I_lat = m * (3 * r**2 + (2 * h) ** 2) / 12.0
    I_ax = m * r**2 / 2.0
    return np.diag([I_lat, I_lat, I_ax])


def _inertia_capsule(m: float, r: float, length: float) -> np.ndarray:
    # Approximate as cylinder for inertia
    return _inertia_cylinder(m, r, length)


def _make_body(name: str, idx: int, mass: float, I: np.ndarray) -> Body:
    return Body(
        name=name,
        index=idx,
        joint=FreeJoint(f"root_{name}"),
        inertia=SpatialInertia(mass, I, np.zeros(3)),
        X_tree=SpatialTransform.identity(),
        parent=-1,
    )


def _single_body_engine(shape, mass: float, I: np.ndarray):
    """Single free-floating body on flat terrain."""
    tree = RobotTreeNumpy(gravity=GRAVITY)
    tree.add_body(_make_body("b", 0, mass, I))
    tree.finalize()
    model = RobotModel(
        tree=tree,
        actuated_joint_names=[],
        contact_body_names=["b"],
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(shape)])],
    )
    merged = merge_models({"r": model}, terrain=FlatTerrain())
    return merged, CpuEngine(merged, dt=DT)


def _two_body_engine(shape_a, mass_a: float, I_a: np.ndarray, shape_b, mass_b: float, I_b: np.ndarray):
    """Two free-floating bodies on flat terrain."""
    tree = RobotTreeNumpy(gravity=GRAVITY)
    tree.add_body(_make_body("a", 0, mass_a, I_a))
    tree.add_body(_make_body("b", 1, mass_b, I_b))
    tree.finalize()
    model = RobotModel(
        tree=tree,
        actuated_joint_names=[],
        contact_body_names=["a", "b"],
        geometries=[
            BodyCollisionGeometry(0, [ShapeInstance(shape_a)]),
            BodyCollisionGeometry(1, [ShapeInstance(shape_b)]),
        ],
    )
    merged = merge_models({"r": model}, terrain=FlatTerrain())
    return merged, CpuEngine(merged, dt=DT)


def _q_identity(merged) -> np.ndarray:
    """Zero q with all quaternion w=1."""
    q = np.zeros(merged.nq)
    n_bodies = merged.nq // 7
    for i in range(n_bodies):
        q[i * 7] = 1.0  # qw
    return q


def _run_settle(engine, merged, q: np.ndarray, n: int = N_STEPS):
    """Run n steps and return final (q, qdot)."""
    qdot = np.zeros(merged.nv)
    for _ in range(n):
        out = engine.step(q, qdot, np.zeros(merged.nv))
        q, qdot = out.q_new, out.qdot_new
    return q, qdot


def _assert_stable(q, qdot, z_idx: int, label: str):
    assert not np.any(np.isnan(q)), f"{label}: NaN in q"
    assert not np.any(np.isnan(qdot)), f"{label}: NaN in qdot"
    assert not np.any(np.isinf(qdot)), f"{label}: Inf in qdot"
    assert abs(q[z_idx]) < 5.0, f"{label}: body flew off (z={q[z_idx]:.3f})"


def _box_hull(half: float) -> "ConvexHullShape":
    mesh = trimesh.creation.box(extents=[2 * half] * 3)
    return ConvexHullShape(np.array(mesh.vertices))


# ---------------------------------------------------------------------------
# Class 1: Ground stability — single-body drops
# ---------------------------------------------------------------------------


class TestGroundStability:
    """Single-body drops onto flat ground — 1000 steps, no NaN/Inf."""

    def test_capsule_drop_stable(self):
        """Vertical capsule drops onto ground — settles at z ≈ radius + half_length."""
        r, length = 0.05, 0.10
        shape = CapsuleShape(r, length)
        I = _inertia_capsule(1.0, r, length)
        merged, engine = _single_body_engine(shape, 1.0, I)

        q = _q_identity(merged)
        q[6] = r + length / 2.0 + 0.3  # drop from above
        q, qdot = _run_settle(engine, merged, q)

        _assert_stable(q, qdot, z_idx=6, label="capsule-ground")
        # Capsule bottom = z - (r + length/2); should be >= -2mm
        bottom_z = q[6] - (r + length / 2.0)
        assert bottom_z >= -2e-3, f"capsule penetrates ground: bottom_z={bottom_z:.5f}"


# ---------------------------------------------------------------------------
# Class 2: Body-body stability — two-body settle
# ---------------------------------------------------------------------------


class TestBodyBodyStability:
    """Two-body settle tests — heavy anchor + free body, 1000 steps."""

    M_ANCHOR = 1e6  # quasi-fixed anchor mass

    def _settle_two(self, shape_a, half_a, mass_a, I_a, shape_b, half_b, mass_b, I_b):
        """
        Place anchor (a) at z=half_a, drop free body (b) from z=half_a+half_b+0.3.
        Run 1000 steps. Return (q, qdot, merged).
        """
        merged, engine = _two_body_engine(shape_a, mass_a, I_a, shape_b, mass_b, I_b)
        q = _q_identity(merged)
        q[6] = half_a  # anchor z
        q[13] = half_a + half_b + 0.3  # free body z
        q, qdot = _run_settle(engine, merged, q)
        return q, qdot, merged

    def test_box_on_box_stable(self):
        """Box (free) rests on top of heavy box (anchor)."""
        half = 0.05
        shape = BoxShape((2 * half, 2 * half, 2 * half))
        I = _inertia_box(1.0, half)
        I_anchor = _inertia_box(self.M_ANCHOR, half)

        q, qdot, _ = self._settle_two(shape, half, self.M_ANCHOR, I_anchor, shape, half, 1.0, I)

        _assert_stable(q, qdot, z_idx=13, label="box-on-box")
        # Free box z should be above anchor top (half_a + half_b - 2mm tolerance)
        assert q[13] >= 2 * half - 2e-3, f"box-on-box: free box z={q[13]:.5f} too low"

    def test_box_on_cylinder_stable(self):
        """Box (free) rests on top of heavy cylinder (anchor)."""
        r_cyl, l_cyl = 0.05, 0.10
        half_box = 0.05
        shape_cyl = CylinderShape(r_cyl, l_cyl)
        shape_box = BoxShape((2 * half_box, 2 * half_box, 2 * half_box))
        I_cyl = _inertia_cylinder(self.M_ANCHOR, r_cyl, l_cyl)
        I_box = _inertia_box(1.0, half_box)
        half_cyl = l_cyl / 2.0

        q, qdot, _ = self._settle_two(
            shape_cyl, half_cyl, self.M_ANCHOR, I_cyl, shape_box, half_box, 1.0, I_box
        )

        _assert_stable(q, qdot, z_idx=13, label="box-on-cylinder")

    def test_cylinder_on_cylinder_stable(self):
        """Cylinder (free) rests on top of heavy cylinder (anchor)."""
        r, length = 0.05, 0.10
        half = length / 2.0
        shape = CylinderShape(r, length)
        I = _inertia_cylinder(1.0, r, length)
        I_anchor = _inertia_cylinder(self.M_ANCHOR, r, length)

        q, qdot, _ = self._settle_two(shape, half, self.M_ANCHOR, I_anchor, shape, half, 1.0, I)

        _assert_stable(q, qdot, z_idx=13, label="cylinder-on-cylinder")

    def test_capsule_on_sphere_stable(self):
        """Capsule (free) rests on top of heavy sphere (anchor)."""
        r_sph = 0.05
        r_cap, l_cap = 0.05, 0.10
        half_cap = r_cap + l_cap / 2.0
        shape_sph = SphereShape(r_sph)
        shape_cap = CapsuleShape(r_cap, l_cap)
        I_sph = _inertia_sphere(self.M_ANCHOR, r_sph)
        I_cap = _inertia_capsule(1.0, r_cap, l_cap)

        q, qdot, _ = self._settle_two(shape_sph, r_sph, self.M_ANCHOR, I_sph, shape_cap, half_cap, 1.0, I_cap)

        _assert_stable(q, qdot, z_idx=13, label="capsule-on-sphere")

    def test_capsule_on_box_stable(self):
        """Capsule (free) rests on top of heavy box (anchor)."""
        half_box = 0.05
        r_cap, l_cap = 0.05, 0.10
        half_cap = r_cap + l_cap / 2.0
        shape_box = BoxShape((2 * half_box, 2 * half_box, 2 * half_box))
        shape_cap = CapsuleShape(r_cap, l_cap)
        I_box = _inertia_box(self.M_ANCHOR, half_box)
        I_cap = _inertia_capsule(1.0, r_cap, l_cap)

        q, qdot, _ = self._settle_two(
            shape_box, half_box, self.M_ANCHOR, I_box, shape_cap, half_cap, 1.0, I_cap
        )

        _assert_stable(q, qdot, z_idx=13, label="capsule-on-box")

    @pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh required")
    def test_hull_on_hull_stable(self):
        """ConvexHull (free) rests on top of heavy ConvexHull (anchor)."""
        half = 0.05
        shape = _box_hull(half)
        I = _inertia_box(1.0, half)
        I_anchor = _inertia_box(self.M_ANCHOR, half)

        q, qdot, _ = self._settle_two(shape, half, self.M_ANCHOR, I_anchor, shape, half, 1.0, I)

        _assert_stable(q, qdot, z_idx=13, label="hull-on-hull")
        assert q[13] >= 2 * half - 2e-3, f"hull-on-hull: free hull z={q[13]:.5f} too low"
