"""
Integration tests: CpuEngine full pipeline for all shape types.

Verifies that each shape type (Sphere, Box, Capsule, Cylinder, ConvexHull)
can be simulated through CpuEngine.step() without NaN, fall-through, or
divergence.  Also validates multi-point contact counts for polyhedral shapes.

Q48.3 — CpuEngine integration-level coverage (session 33).
Q48.2 — ground_contact_query multi-point for Box/ConvexHull.
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
    from scipy.spatial import ConvexHull as _ScipyConvexHull  # noqa: F401

    from physics.geometry import ConvexHullShape

    HAS_CONVEXHULL = True
except ImportError:
    HAS_CONVEXHULL = False

DT = 2e-4
GRAVITY = 9.81
N_SETTLE = 1500  # steps to settle (box from z=0.35 needs ~1400 steps at DT=2e-4)
N_STABLE = 200  # steps to check stability after settling


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _box_inertia(m, half):
    return np.eye(3) * (m / 6.0 * (2 * half) ** 2)


def _sphere_inertia(m, r):
    return np.eye(3) * (2.0 / 5.0 * m * r**2)


def _capsule_inertia(m, r, length):
    h = length / 2.0
    I_lat = m * (3 * r**2 + (2 * h) ** 2) / 12.0
    I_ax = m * r**2 / 2.0
    return np.diag([I_lat, I_lat, I_ax])


def _make_single_body_engine(shape, mass, I_body):
    """Build a CpuEngine with a single free-floating body."""
    tree = RobotTreeNumpy(gravity=GRAVITY)
    tree.add_body(
        Body(
            name="b",
            index=0,
            joint=FreeJoint("root"),
            inertia=SpatialInertia(mass, I_body, np.zeros(3)),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()
    model = RobotModel(
        tree=tree,
        actuated_joint_names=[],
        contact_body_names=["b"],
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(shape)])],
    )
    merged = merge_models({"r": model}, terrain=FlatTerrain())
    engine = CpuEngine(merged, dt=DT)
    return merged, engine


def _q_at_z(merged, z, R=None):
    """Initial q with body at height z (and optional rotation R)."""
    q = np.zeros(merged.nq)
    if R is not None:
        from scipy.spatial.transform import Rotation as Rot

        quat = Rot.from_matrix(R).as_quat()  # [x,y,z,w]
        q[0] = quat[3]
        q[1] = quat[0]
        q[2] = quat[1]
        q[3] = quat[2]
    else:
        q[0] = 1.0  # qw
    q[6] = z
    return q


def _run(engine, merged, q, n_steps):
    qdot = np.zeros(merged.nv)
    for _ in range(n_steps):
        out = engine.step(q, qdot, np.zeros(merged.nv))
        q, qdot = out.q_new, out.qdot_new
    return q, qdot


def _assert_stable(q, qdot, label, z_min=0.0, omega_max=1.0):
    assert not np.any(np.isnan(q)), f"{label}: NaN in q"
    assert not np.any(np.isnan(qdot)), f"{label}: NaN in qdot"
    assert q[6] > z_min, f"{label}: fell through ground z={q[6]:.4f}"
    omega = float(np.linalg.norm(qdot[3:6]))
    assert omega < omega_max, f"{label}: spinning |omega|={omega:.3f} rad/s"


# ---------------------------------------------------------------------------
# Class 1: Single shape drop onto flat ground
# ---------------------------------------------------------------------------


class TestSingleShapeGroundContact:
    """Each shape type drops from z=0.3 and settles on flat ground."""

    def test_sphere_drop(self):
        r = 0.05
        shape = SphereShape(r)
        I = _sphere_inertia(1.0, r)
        merged, engine = _make_single_body_engine(shape, 1.0, I)
        q = _q_at_z(merged, r + 0.3)
        q, qdot = _run(engine, merged, q, N_SETTLE)
        _assert_stable(q, qdot, "sphere", z_min=r * 0.5)
        # Sphere should rest near z = r
        assert abs(q[6] - r) < 0.02, f"sphere resting z={q[6]:.4f}, expected ~{r}"

    def test_box_drop_flat(self):
        half = 0.05
        shape = BoxShape((2 * half, 2 * half, 2 * half))
        I = _box_inertia(1.0, half)
        merged, engine = _make_single_body_engine(shape, 1.0, I)
        q = _q_at_z(merged, half + 0.3)
        q, qdot = _run(engine, merged, q, N_SETTLE)
        _assert_stable(q, qdot, "box_flat", z_min=half * 0.5)
        # Box should rest near z = half
        assert abs(q[6] - half) < 0.02, f"box resting z={q[6]:.4f}, expected ~{half}"

    def test_capsule_vertical_drop(self):
        r, length = 0.03, 0.15
        shape = CapsuleShape(r, length)
        I = _capsule_inertia(1.0, r, length)
        merged, engine = _make_single_body_engine(shape, 1.0, I)
        # Capsule upright (default orientation, axis along Z)
        z0 = length / 2.0 + r + 0.3
        q = _q_at_z(merged, z0)
        q, qdot = _run(engine, merged, q, N_SETTLE)
        _assert_stable(q, qdot, "capsule_vertical", z_min=r * 0.5)

    def test_cylinder_vertical_drop(self):
        r, length = 0.04, 0.12
        shape = CylinderShape(r, length)
        I = _capsule_inertia(1.0, r, length)
        merged, engine = _make_single_body_engine(shape, 1.0, I)
        z0 = length / 2.0 + 0.3
        q = _q_at_z(merged, z0)
        q, qdot = _run(engine, merged, q, N_SETTLE)
        _assert_stable(q, qdot, "cylinder_vertical", z_min=r * 0.5)

    @pytest.mark.skipif(not HAS_CONVEXHULL, reason="scipy required")
    def test_convexhull_drop(self):
        """ConvexHull box-shaped hull drops and settles."""
        half = 0.05
        verts = np.array(
            [
                [-half, -half, -half],
                [half, -half, -half],
                [half, half, -half],
                [-half, half, -half],
                [-half, -half, half],
                [half, -half, half],
                [half, half, half],
                [-half, half, half],
            ],
            dtype=float,
        )
        shape = ConvexHullShape(verts)
        I = _box_inertia(1.0, half)
        merged, engine = _make_single_body_engine(shape, 1.0, I)
        q = _q_at_z(merged, half + 0.3)
        q, qdot = _run(engine, merged, q, N_SETTLE)
        _assert_stable(q, qdot, "convexhull", z_min=half * 0.5)


# ---------------------------------------------------------------------------
# Class 2: Multi-point contact count validation (Q48.2)
# ---------------------------------------------------------------------------


class TestMultiPointContactCount:
    """Verify that polyhedral shapes produce multiple contact points when flat."""

    def test_box_flat_contact_count(self):
        """Box lying flat on ground should produce >=2 contact points after settling."""
        half = 0.05
        shape = BoxShape((2 * half, 2 * half, 2 * half))
        I = _box_inertia(1.0, half)
        merged, engine = _make_single_body_engine(shape, 1.0, I)
        q = _q_at_z(merged, half + 0.3)
        q, qdot = _run(engine, merged, q, N_SETTLE)
        # One more step to populate _last_contacts
        engine.step(q, qdot, np.zeros(merged.nv))
        contacts = engine.query_contacts()
        n = len(contacts)
        assert n >= 2, f"box flat: expected >=2 contact points, got {n}"

    def test_box_tilted_contact_count(self):
        """Box tilted 45° on edge should produce >=1 contact point after settling."""
        from scipy.spatial.transform import Rotation as Rot

        half = 0.05
        shape = BoxShape((2 * half, 2 * half, 2 * half))
        I = _box_inertia(1.0, half)
        merged, engine = _make_single_body_engine(shape, 1.0, I)
        R = Rot.from_euler("x", 45, degrees=True).as_matrix()
        z0 = half * np.sqrt(2) + 0.3
        q = _q_at_z(merged, z0, R=R)
        q, qdot = _run(engine, merged, q, N_SETTLE)
        engine.step(q, qdot, np.zeros(merged.nv))
        contacts = engine.query_contacts()
        n = len(contacts)
        assert n >= 1, f"box tilted: expected >=1 contact point, got {n}"

    def test_sphere_single_contact_point(self):
        """Sphere always produces exactly 1 contact point after settling."""
        r = 0.05
        shape = SphereShape(r)
        I = _sphere_inertia(1.0, r)
        merged, engine = _make_single_body_engine(shape, 1.0, I)
        q = _q_at_z(merged, r + 0.3)
        q, qdot = _run(engine, merged, q, N_SETTLE)
        engine.step(q, qdot, np.zeros(merged.nv))
        contacts = engine.query_contacts()
        assert len(contacts) == 1, f"sphere: expected 1 contact point, got {len(contacts)}"

    @pytest.mark.skipif(not HAS_CONVEXHULL, reason="scipy required")
    def test_convexhull_flat_contact_count(self):
        """ConvexHull box-shaped hull flat on ground: >=2 contact points after settling."""
        half = 0.05
        verts = np.array(
            [
                [-half, -half, -half],
                [half, -half, -half],
                [half, half, -half],
                [-half, half, -half],
                [-half, -half, half],
                [half, -half, half],
                [half, half, half],
                [-half, half, half],
            ],
            dtype=float,
        )
        shape = ConvexHullShape(verts)
        I = _box_inertia(1.0, half)
        merged, engine = _make_single_body_engine(shape, 1.0, I)
        q = _q_at_z(merged, half + 0.3)
        q, qdot = _run(engine, merged, q, N_SETTLE)
        engine.step(q, qdot, np.zeros(merged.nv))
        contacts = engine.query_contacts()
        n = len(contacts)
        assert n >= 2, f"convexhull flat: expected >=2 contact points, got {n}"


# ---------------------------------------------------------------------------
# Class 3: Body-body contact (two free bodies)
# ---------------------------------------------------------------------------


def _make_two_body_engine(shape_a, mass_a, I_a, shape_b, mass_b, I_b):
    """Two free-floating bodies in a single merged model."""

    def _tree_model(name, shape, mass, I):
        tree = RobotTreeNumpy(gravity=GRAVITY)
        tree.add_body(
            Body(
                name=name,
                index=0,
                joint=FreeJoint(f"root_{name}"),
                inertia=SpatialInertia(mass, I, np.zeros(3)),
                X_tree=SpatialTransform.identity(),
                parent=-1,
            )
        )
        tree.finalize()
        return RobotModel(
            tree=tree,
            actuated_joint_names=[],
            contact_body_names=[name],
            geometries=[BodyCollisionGeometry(0, [ShapeInstance(shape)])],
        )

    model_a = _tree_model("a", shape_a, mass_a, I_a)
    model_b = _tree_model("b", shape_b, mass_b, I_b)
    merged = merge_models({"ra": model_a, "rb": model_b}, terrain=FlatTerrain())
    engine = CpuEngine(merged, dt=DT)
    return merged, engine


class TestBodyBodyContact:
    """Two bodies collide and settle without NaN or divergence."""

    def test_sphere_sphere_collision(self):
        """Sphere dropped onto a resting sphere — upper sphere must bounce off."""
        r = 0.05
        shape = SphereShape(r)
        I = _sphere_inertia(1.0, r)
        merged, engine = _make_two_body_engine(shape, 1.0, I, SphereShape(r), 1.0, I)

        # Body a (lower): resting on ground at z = r
        # Body b (upper): dropped from directly above, z = 3r + 0.1
        q = np.zeros(merged.nq)
        q[0] = 1.0
        q[6] = r  # lower sphere on ground
        q[7] = 1.0
        q[13] = 3 * r + 0.1  # upper sphere above lower
        qdot = np.zeros(merged.nv)

        # Run until upper sphere has had time to fall and interact (~500 steps)
        for _ in range(500):
            out = engine.step(q, qdot, np.zeros(merged.nv))
            q, qdot = out.q_new, out.qdot_new

        assert not np.any(np.isnan(q)), "sphere-sphere: NaN in q"
        assert not np.any(np.isnan(qdot)), "sphere-sphere: NaN in qdot"
        assert q[6] > 0.0, f"lower sphere fell through: z={q[6]:.4f}"
        assert q[13] > 0.0, f"upper sphere fell through: z={q[13]:.4f}"
        # Upper sphere must be above lower sphere (contact prevented pass-through).
        assert q[13] > q[6], f"upper sphere below lower: z_upper={q[13]:.4f} z_lower={q[6]:.4f}"
        # Vertical separation must be >= 2r (no interpenetration).
        sep_z = q[13] - q[6]
        assert sep_z >= 2 * r * 0.9, f"spheres interpenetrating: sep_z={sep_z:.4f} < 2r={2 * r}"

    def test_box_sphere_collision(self):
        """Box on ground, sphere dropped on top — no NaN, sphere stays above box."""
        half = 0.05
        r = 0.04
        box = BoxShape((2 * half, 2 * half, 2 * half))
        sph = SphereShape(r)
        I_box = _box_inertia(2.0, half)
        I_sph = _sphere_inertia(0.5, r)
        merged, engine = _make_two_body_engine(box, 2.0, I_box, sph, 0.5, I_sph)

        q = np.zeros(merged.nq)
        # Box at rest on ground
        q[0] = 1.0
        q[6] = half + 0.005
        # Sphere above box
        q[7] = 1.0
        q[13] = half * 2 + r + 0.1
        qdot = np.zeros(merged.nv)

        for _ in range(N_SETTLE):
            out = engine.step(q, qdot, np.zeros(merged.nv))
            q, qdot = out.q_new, out.qdot_new

        assert not np.any(np.isnan(q)), "box-sphere: NaN in q"
        assert not np.any(np.isnan(qdot)), "box-sphere: NaN in qdot"
        assert q[6] > 0.0, f"box fell through: z={q[6]:.4f}"
        assert q[13] > 0.0, f"sphere fell through: z={q[13]:.4f}"
        # Sphere must rest above the box top face (z > 2*half), not on the ground.
        assert q[13] > half * 2 * 0.9, f"sphere resting below box top: z={q[13]:.4f}"

    def test_box_box_stacking(self):
        """Two boxes stacked — lower box on ground, upper box on top."""
        half = 0.05
        shape = BoxShape((2 * half, 2 * half, 2 * half))
        I = _box_inertia(1.0, half)
        merged, engine = _make_two_body_engine(
            shape, 1.0, I, BoxShape((2 * half, 2 * half, 2 * half)), 1.0, I
        )

        q = np.zeros(merged.nq)
        # Lower box on ground
        q[0] = 1.0
        q[6] = half + 0.005
        # Upper box above lower box
        q[7] = 1.0
        q[13] = half * 3 + 0.05
        qdot = np.zeros(merged.nv)

        for _ in range(N_SETTLE):
            out = engine.step(q, qdot, np.zeros(merged.nv))
            q, qdot = out.q_new, out.qdot_new

        assert not np.any(np.isnan(q)), "box-box: NaN in q"
        assert not np.any(np.isnan(qdot)), "box-box: NaN in qdot"
        assert q[6] > 0.0, f"lower box fell through: z={q[6]:.4f}"
        assert q[13] > 0.0, f"upper box fell through: z={q[13]:.4f}"
        # Upper box must rest above the lower box (z > 2*half), not on the ground.
        assert q[13] > half * 2 * 0.9, f"upper box resting below lower box top: z={q[13]:.4f}"
