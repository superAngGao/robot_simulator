"""
Session 16 — Multi-shape collision & GPU solver test coverage gaps.

Covers the high-risk and medium-risk blind spots identified in OPEN_QUESTIONS Q26:

High-risk:
  1. GPU PGS Q25 球体静止角速度稳定性 (kernel modified but untested on GPU)
  2. Shape offset 接触点精度 (count verified but not world-coordinate accuracy)
  3. Non-Sphere (Box/Capsule) multi-shape dispatch paths
  4. Contact depth physical accuracy (not just count > 0)

Medium-risk:
  5. CPU vs GPU multi-shape consistency
  6. Contact buffer overflow graceful discard
  7. Multi-shape body-body collision
  8. Shape rotation (non-zero origin_rpy)
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.geometry import (
    BodyCollisionGeometry,
    BoxShape,
    CapsuleShape,
    ShapeInstance,
    SphereShape,
)
from physics.joint import FreeJoint
from physics.merged_model import merge_models
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from robot.model import RobotModel

try:
    import warp as wp

    from physics.cpu_engine import CpuEngine
    from physics.gpu_engine import GpuEngine

    HAS_WARP = True
except Exception:
    HAS_WARP = False

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not HAS_WARP, reason="Warp or CUDA not available"),
]


# ---------------------------------------------------------------------------
# Helpers: model builders
# ---------------------------------------------------------------------------


def _free_body(name, mass, inertia_diag, shapes, parent=-1, X_tree=None):
    """Build a Body with FreeJoint and given shapes."""
    return Body(
        name=name,
        index=0,  # re-assigned by tree
        joint=FreeJoint(f"{name}_root"),
        inertia=SpatialInertia(
            mass=mass,
            inertia=np.diag(inertia_diag),
            com=np.zeros(3),
        ),
        X_tree=X_tree or SpatialTransform.identity(),
        parent=parent,
    )


def _sphere_model(mass=1.0, radius=0.1):
    """Single FreeJoint sphere."""
    tree = RobotTreeNumpy(gravity=9.81)
    I = 2.0 / 5.0 * mass * radius**2
    tree.add_body(_free_body("ball", mass, [I, I, I], []))
    tree.finalize()
    return RobotModel(
        tree=tree,
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(SphereShape(radius))])],
        contact_body_names=["ball"],
    )


def _offset_two_sphere_model(radius=0.05, sep=0.2):
    """One body with two spheres at ±sep/2 in x."""
    mass = 1.0
    I = 2.0 / 5.0 * mass * radius**2
    tree = RobotTreeNumpy(gravity=9.81)
    tree.add_body(_free_body("body", mass, [I, I, I], []))
    tree.finalize()
    shapes = [
        ShapeInstance(SphereShape(radius), origin_xyz=np.array([-sep / 2, 0.0, 0.0])),
        ShapeInstance(SphereShape(radius), origin_xyz=np.array([sep / 2, 0.0, 0.0])),
    ]
    return RobotModel(
        tree=tree,
        geometries=[BodyCollisionGeometry(0, shapes)],
        contact_body_names=["body"],
    )


def _box_capsule_model(box_half=(0.05, 0.05, 0.05), cap_radius=0.04, cap_length=0.1):
    """One body with a box at origin and a capsule offset in +x."""
    mass = 2.0
    I = mass * 0.01
    tree = RobotTreeNumpy(gravity=9.81)
    tree.add_body(_free_body("body", mass, [I, I, I], []))
    tree.finalize()
    shapes = [
        ShapeInstance(BoxShape(tuple(2 * h for h in box_half))),
        ShapeInstance(
            CapsuleShape(cap_radius, cap_length),
            origin_xyz=np.array([0.2, 0.0, 0.0]),
        ),
    ]
    return RobotModel(
        tree=tree,
        geometries=[BodyCollisionGeometry(0, shapes)],
        contact_body_names=["body"],
    )


def _rotated_box_model(rpy, box_size=(0.1, 0.06, 0.2)):
    """One body with a box that has non-zero origin_rpy."""
    mass = 1.0
    I = mass * 0.01
    tree = RobotTreeNumpy(gravity=9.81)
    tree.add_body(_free_body("body", mass, [I, I, I], []))
    tree.finalize()
    shapes = [
        ShapeInstance(BoxShape(box_size), origin_rpy=np.array(rpy)),
    ]
    return RobotModel(
        tree=tree,
        geometries=[BodyCollisionGeometry(0, shapes)],
        contact_body_names=["body"],
    )


def _multishape_pair_model(radius=0.05, sep=0.15):
    """One body with two spheres, for body-body tests."""
    mass = 1.0
    I = 2.0 / 5.0 * mass * radius**2
    tree = RobotTreeNumpy(gravity=9.81)
    tree.add_body(_free_body("body", mass, [I, I, I], []))
    tree.finalize()
    shapes = [
        ShapeInstance(SphereShape(radius)),
        ShapeInstance(SphereShape(radius), origin_xyz=np.array([sep, 0.0, 0.0])),
    ]
    return RobotModel(
        tree=tree,
        geometries=[BodyCollisionGeometry(0, shapes)],
        contact_body_names=["body"],
    )


def _set_q(engine, env, q_slice_start, px=None, py=None, pz=None):
    """Modify position in scratch q for a FreeJoint body."""
    q = engine._scratch.q.numpy()
    if px is not None:
        q[env, q_slice_start + 4] = px
    if py is not None:
        q[env, q_slice_start + 5] = py
    if pz is not None:
        q[env, q_slice_start + 6] = pz
    engine._scratch.q = wp.array(q, dtype=wp.float32, device="cuda:0")


def _set_qdot(engine, env, v_slice_start, vx=None, vy=None, vz=None):
    """Modify velocity in scratch qdot."""
    qdot = engine._scratch.qdot.numpy()
    if vx is not None:
        qdot[env, v_slice_start + 0] = vx
    if vy is not None:
        qdot[env, v_slice_start + 1] = vy
    if vz is not None:
        qdot[env, v_slice_start + 2] = vz
    engine._scratch.qdot = wp.array(qdot, dtype=wp.float32, device="cuda:0")


def _get_contacts(engine, env=0):
    """Extract active contacts as list of dicts."""
    count = engine._contact_count.numpy()[env]
    results = []
    for c in range(count):
        if engine._contact_active.numpy()[env, c] == 1:
            results.append(
                {
                    "depth": float(engine._contact_depth.numpy()[env, c]),
                    "normal": engine._contact_normal.numpy()[env, c].copy(),
                    "point": engine._contact_point.numpy()[env, c].copy(),
                    "bi": int(engine._contact_bi.numpy()[env, c]),
                    "bj": int(engine._contact_bj.numpy()[env, c]),
                }
            )
    return results


# ---------------------------------------------------------------------------
# 1. GPU PGS Q25 — sphere at rest angular velocity stability
# ---------------------------------------------------------------------------


class TestGpuQ25FrictionStability:
    """GPU PGS solver: Q25 per-row R must prevent angular velocity divergence."""

    def test_sphere_at_rest_angular_velocity_bounded(self):
        """Sphere resting on ground: max |omega| < 0.1 rad/s over 5000 GPU steps."""
        model = _sphere_model(mass=1.0, radius=0.1)
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        # Place on ground
        q0 = merged.tree.default_state()[0].copy()
        q0[6] = 0.1  # z = radius
        engine.reset(q0=q0)

        max_omega = 0.0
        for step in range(5000):
            engine.step(np.zeros((1, 0)), 2e-4)
            qdot = engine._scratch.qdot.numpy()[0]
            omega = qdot[3:6]  # FreeJoint angular velocity
            max_omega = max(max_omega, float(np.linalg.norm(omega)))

        assert max_omega < 0.1, f"GPU PGS angular velocity diverged: max |omega| = {max_omega:.4f} rad/s"

    def test_heavy_sphere_at_rest_stable(self):
        """Heavy sphere (50 kg) on ground: larger moment arm stress test."""
        model = _sphere_model(mass=50.0, radius=0.15)
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        q0 = merged.tree.default_state()[0].copy()
        q0[6] = 0.15
        engine.reset(q0=q0)

        max_omega = 0.0
        for _ in range(3000):
            engine.step(np.zeros((1, 0)), 2e-4)
            qdot = engine._scratch.qdot.numpy()[0]
            omega = qdot[3:6]
            max_omega = max(max_omega, float(np.linalg.norm(omega)))

        assert max_omega < 0.1, f"GPU heavy sphere omega diverged: max |omega| = {max_omega:.4f}"

    def test_sphere_sliding_decelerates(self):
        """Sliding sphere must decelerate (friction not killed by R)."""
        model = _sphere_model(mass=1.0, radius=0.1)
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        q0 = merged.tree.default_state()[0].copy()
        q0[6] = 0.1
        engine.reset(q0=q0)
        # Give initial tangential velocity
        _set_qdot(engine, 0, 0, vx=2.0)

        for _ in range(500):
            engine.step(np.zeros((1, 0)), 2e-4)

        qdot = engine._scratch.qdot.numpy()[0]
        vx = abs(float(qdot[0]))
        assert vx < 1.5, f"Sphere should have decelerated from 2.0, still at vx={vx:.3f}"


# ---------------------------------------------------------------------------
# 2. Shape offset ��� contact point world-coordinate accuracy
# ---------------------------------------------------------------------------


class TestShapeOffsetContactPrecision:
    """Contact points must reflect shape offsets, not just body origin."""

    def test_two_offset_spheres_contact_x_separation(self):
        """Two spheres at ±sep/2 in x: their ground contact x-coords must differ."""
        sep = 0.2
        radius = 0.05
        model = _offset_two_sphere_model(radius=radius, sep=sep)
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        # Place body center slightly below radius → both spheres penetrate ground
        q0 = merged.tree.default_state()[0].copy()
        q0[6] = radius - 0.005
        engine.reset(q0=q0)

        engine.step(np.zeros((1, 0)), 2e-4)
        contacts = _get_contacts(engine)

        assert len(contacts) >= 2, f"Expected 2 ground contacts, got {len(contacts)}"

        xs = sorted([c["point"][0] for c in contacts])
        x_diff = xs[-1] - xs[0]
        # Contact points should be ~sep apart in x
        assert x_diff > sep * 0.5, f"Contact x separation {x_diff:.4f} too small, expected ~{sep:.4f}"

    def test_offset_sphere_contact_point_near_expected_position(self):
        """Contact point of offset sphere should be near shape center projected to ground."""
        sep = 0.3
        radius = 0.04
        model = _offset_two_sphere_model(radius=radius, sep=sep)
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        q0 = merged.tree.default_state()[0].copy()
        q0[6] = radius - 0.005  # slight penetration
        engine.reset(q0=q0)

        engine.step(np.zeros((1, 0)), 2e-4)
        contacts = _get_contacts(engine)

        assert len(contacts) >= 2, f"Expected 2 contacts, got {len(contacts)}"

        # Expected contact x-coords: -sep/2 and +sep/2
        xs = sorted([c["point"][0] for c in contacts])
        np.testing.assert_allclose(xs[0], -sep / 2, atol=0.02, err_msg="Left sphere contact x")
        np.testing.assert_allclose(xs[1], sep / 2, atol=0.02, err_msg="Right sphere contact x")


# ---------------------------------------------------------------------------
# 3. Non-Sphere shapes: Box and Capsule multi-shape dispatch
# ---------------------------------------------------------------------------


class TestNonSphereMultiShape:
    """Box and Capsule shapes in multi-shape bodies on GPU."""

    def test_box_touches_ground(self):
        """Single box shape body should produce ground contact."""
        mass, I = 1.0, 0.01
        tree = RobotTreeNumpy(gravity=9.81)
        tree.add_body(_free_body("box_body", mass, [I, I, I], []))
        tree.finalize()
        model = RobotModel(
            tree=tree,
            geometries=[BodyCollisionGeometry(0, [ShapeInstance(BoxShape((0.1, 0.1, 0.1)))])],
            contact_body_names=["box_body"],
        )
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        q0 = merged.tree.default_state()[0].copy()
        q0[6] = 0.04  # slightly into ground (half-extent 0.05)
        engine.reset(q0=q0)

        engine.step(np.zeros((1, 0)), 2e-4)
        contacts = _get_contacts(engine)
        assert len(contacts) >= 1, f"Box should produce ground contact, got {len(contacts)}"

    def test_capsule_touches_ground(self):
        """Single capsule shape body should produce ground contact."""
        mass, I = 1.0, 0.01
        tree = RobotTreeNumpy(gravity=9.81)
        tree.add_body(_free_body("cap_body", mass, [I, I, I], []))
        tree.finalize()
        model = RobotModel(
            tree=tree,
            geometries=[BodyCollisionGeometry(0, [ShapeInstance(CapsuleShape(0.05, 0.2))])],
            contact_body_names=["cap_body"],
        )
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        q0 = merged.tree.default_state()[0].copy()
        q0[6] = 0.04  # below capsule radius
        engine.reset(q0=q0)

        engine.step(np.zeros((1, 0)), 2e-4)
        contacts = _get_contacts(engine)
        assert len(contacts) >= 1, f"Capsule should produce ground contact, got {len(contacts)}"

    def test_box_plus_capsule_multishape_both_contact(self):
        """Body with box at origin + capsule offset: both produce ground contacts."""
        model = _box_capsule_model()
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        # Place body low enough that both shapes touch
        q0 = merged.tree.default_state()[0].copy()
        q0[6] = 0.04  # box half-extent=0.05, capsule radius=0.04
        engine.reset(q0=q0)

        engine.step(np.zeros((1, 0)), 2e-4)
        contacts = _get_contacts(engine)
        assert len(contacts) >= 2, f"Box+Capsule multishape should produce >=2 contacts, got {len(contacts)}"

    def test_box_capsule_contact_x_separation(self):
        """Box at origin and capsule at x=0.2: contact x-coords must differ."""
        model = _box_capsule_model()
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        q0 = merged.tree.default_state()[0].copy()
        q0[6] = 0.03
        engine.reset(q0=q0)

        engine.step(np.zeros((1, 0)), 2e-4)
        contacts = _get_contacts(engine)
        assert len(contacts) >= 2, f"Expected >=2 contacts, got {len(contacts)}"

        xs = [c["point"][0] for c in contacts]
        assert max(xs) - min(xs) > 0.1, f"Contacts too close in x: {xs}, expected separation ~0.2"


# ---------------------------------------------------------------------------
# 4. Contact depth physical accuracy
# ---------------------------------------------------------------------------


class TestContactDepthAccuracy:
    """Depth values must be physically reasonable, not just nonzero."""

    def test_sphere_depth_matches_penetration(self):
        """Sphere of radius r at height h < r: depth ≈ r - h."""
        radius = 0.1
        model = _sphere_model(mass=1.0, radius=radius)
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        penetration = 0.02  # 2 cm into ground
        q0 = merged.tree.default_state()[0].copy()
        q0[6] = radius - penetration
        engine.reset(q0=q0)

        engine.step(np.zeros((1, 0)), 2e-4)
        contacts = _get_contacts(engine)
        assert len(contacts) == 1, f"Expected 1 contact, got {len(contacts)}"

        depth = contacts[0]["depth"]
        np.testing.assert_allclose(
            depth,
            penetration,
            atol=0.005,
            err_msg=f"Depth {depth:.4f} doesn't match expected penetration {penetration:.4f}",
        )

    def test_box_depth_matches_half_extent_penetration(self):
        """Box with half-extent hz, center at h < hz: depth ≈ hz - h."""
        hz = 0.05
        model_mass, I = 1.0, 0.01
        tree = RobotTreeNumpy(gravity=9.81)
        tree.add_body(_free_body("box", model_mass, [I, I, I], []))
        tree.finalize()
        model = RobotModel(
            tree=tree,
            geometries=[BodyCollisionGeometry(0, [ShapeInstance(BoxShape((0.1, 0.1, 2 * hz)))])],
            contact_body_names=["box"],
        )
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        penetration = 0.01
        q0 = merged.tree.default_state()[0].copy()
        q0[6] = hz - penetration
        engine.reset(q0=q0)

        engine.step(np.zeros((1, 0)), 2e-4)
        contacts = _get_contacts(engine)
        assert len(contacts) >= 1, f"Expected >=1 contact, got {len(contacts)}"

        depth = contacts[0]["depth"]
        np.testing.assert_allclose(
            depth,
            penetration,
            atol=0.005,
            err_msg=f"Box depth {depth:.4f} vs expected {penetration:.4f}",
        )

    def test_contact_normal_points_up_for_ground(self):
        """Ground contact normal should point upward (+z)."""
        model = _sphere_model(mass=1.0, radius=0.1)
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        q0 = merged.tree.default_state()[0].copy()
        q0[6] = 0.08  # slightly penetrating
        engine.reset(q0=q0)

        engine.step(np.zeros((1, 0)), 2e-4)
        contacts = _get_contacts(engine)
        assert len(contacts) >= 1

        nz = contacts[0]["normal"][2]
        assert nz > 0.9, f"Ground normal z should be ~1.0, got {nz:.4f}"


# ---------------------------------------------------------------------------
# 5. CPU vs GPU multi-shape consistency
# ---------------------------------------------------------------------------


class TestCpuGpuMultiShapeConsistency:
    """CpuEngine and GpuEngine should agree on multi-shape contact count."""

    def test_two_sphere_body_contact_count_agrees(self):
        """CPU and GPU should both detect 2 ground contacts for two-sphere body."""
        model = _offset_two_sphere_model(radius=0.05, sep=0.2)
        merged = merge_models(robots={"a": model})

        cpu = CpuEngine(merged, dt=2e-4)
        gpu = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        q0 = merged.tree.default_state()[0].copy()
        q0[6] = 0.045  # slight penetration (radius=0.05, 0.5mm into ground)
        tau = np.zeros(merged.nv)

        # CPU step
        out_cpu = cpu.step(q0.copy(), np.zeros(merged.nv), tau, dt=2e-4)
        cpu_contacts = int(np.sum(out_cpu.contact_active))

        # GPU step
        gpu.reset(q0=q0)
        gpu.step(np.zeros((1, 0)), 2e-4)
        gpu_contacts = len(_get_contacts(gpu))

        assert cpu_contacts >= 2, f"CPU should detect >=2 contacts, got {cpu_contacts}"
        assert gpu_contacts >= 2, f"GPU should detect >=2 contacts, got {gpu_contacts}"

    def test_box_capsule_body_trajectory_close(self):
        """CPU and GPU trajectories for box+capsule body should be close after 200 steps."""
        model = _box_capsule_model()
        merged = merge_models(robots={"a": model})

        cpu = CpuEngine(merged, dt=2e-4)
        gpu = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        q0 = merged.tree.default_state()[0].copy()
        q0[6] = 0.5  # start above ground
        tau = np.zeros(merged.nv)

        q_cpu, qdot_cpu = q0.copy(), np.zeros(merged.nv)
        gpu.reset(q0=q0)

        for _ in range(200):
            out_cpu = cpu.step(q_cpu, qdot_cpu, tau, dt=2e-4)
            q_cpu, qdot_cpu = out_cpu.q_new, out_cpu.qdot_new
            gpu.step(np.zeros((1, 0)), 2e-4)

        q_gpu = gpu._scratch.q.numpy()[0]
        # Position (indices 4-6) should be within 5 mm
        np.testing.assert_allclose(
            q_gpu[4:7],
            q_cpu[4:7],
            atol=5e-3,
            err_msg="CPU vs GPU position diverged for box+capsule body",
        )


# ---------------------------------------------------------------------------
# 6. Contact buffer overflow — graceful discard
# ---------------------------------------------------------------------------


class TestContactBufferOverflow:
    """When contacts exceed max_contacts, engine must not crash."""

    def test_many_shapes_no_crash(self):
        """Body with many shapes: if contacts > max_contacts, no crash or NaN."""
        mass, I = 1.0, 0.01
        tree = RobotTreeNumpy(gravity=9.81)
        tree.add_body(_free_body("body", mass, [I, I, I], []))
        tree.finalize()

        # 20 small spheres spread in a grid — may exceed max_contacts
        shapes = []
        for ix in range(5):
            for iy in range(4):
                shapes.append(
                    ShapeInstance(
                        SphereShape(0.02),
                        origin_xyz=np.array([ix * 0.05 - 0.1, iy * 0.05 - 0.075, 0.0]),
                    )
                )
        model = RobotModel(
            tree=tree,
            geometries=[BodyCollisionGeometry(0, shapes)],
            contact_body_names=["body"],
        )
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        q0 = merged.tree.default_state()[0].copy()
        q0[6] = 0.02  # all spheres touching ground
        engine.reset(q0=q0)

        # Run 100 steps — must not crash
        for _ in range(100):
            engine.step(np.zeros((1, 0)), 2e-4)

        q = engine._scratch.q.numpy()[0]
        assert np.all(np.isfinite(q)), "NaN after overflow scenario"

    def test_overflow_contacts_capped_at_max(self):
        """Contact count should never exceed max_contacts."""
        mass, I = 1.0, 0.01
        tree = RobotTreeNumpy(gravity=9.81)
        tree.add_body(_free_body("body", mass, [I, I, I], []))
        tree.finalize()

        shapes = [
            ShapeInstance(
                SphereShape(0.02),
                origin_xyz=np.array([i * 0.04 - 0.2, 0.0, 0.0]),
            )
            for i in range(12)
        ]
        model = RobotModel(
            tree=tree,
            geometries=[BodyCollisionGeometry(0, shapes)],
            contact_body_names=["body"],
        )
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        q0 = merged.tree.default_state()[0].copy()
        q0[6] = 0.01
        engine.reset(q0=q0)

        engine.step(np.zeros((1, 0)), 2e-4)
        count = int(engine._contact_count.numpy()[0])
        assert count <= engine._max_contacts, f"Contact count {count} exceeds max {engine._max_contacts}"


# ---------------------------------------------------------------------------
# 7. Multi-shape body-body collision
# ---------------------------------------------------------------------------


class TestMultiShapeBodyBody:
    """Two multi-shape bodies colliding with each other."""

    def test_two_multishape_bodies_overlap_produces_contact(self):
        """Two multi-shape bodies overlapping: broadphase + narrowphase detects contact."""
        m_a = _multishape_pair_model(radius=0.05, sep=0.15)
        m_b = _multishape_pair_model(radius=0.05, sep=0.15)
        merged = merge_models(robots={"a": m_a, "b": m_b})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)
        engine.reset()

        # Place bodies overlapping at z=1 (above ground, no ground contact)
        rs_a = merged.robot_slices["a"]
        rs_b = merged.robot_slices["b"]
        _set_q(engine, 0, rs_a.q_slice.start, px=0.0, pz=1.0)
        _set_q(engine, 0, rs_b.q_slice.start, px=0.08, pz=1.0)  # overlap

        engine.step(np.zeros((1, 0)), 2e-4)
        contacts = _get_contacts(engine)

        body_body = [c for c in contacts if c["bj"] >= 0]
        assert len(body_body) >= 1, (
            f"Expected body-body contacts for overlapping multi-shape bodies, got {len(body_body)}"
        )

    def test_two_multishape_bodies_far_apart_no_contact(self):
        """Two multi-shape bodies far apart: no body-body contact."""
        m_a = _multishape_pair_model(radius=0.05, sep=0.15)
        m_b = _multishape_pair_model(radius=0.05, sep=0.15)
        merged = merge_models(robots={"a": m_a, "b": m_b})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)
        engine.reset()

        rs_a = merged.robot_slices["a"]
        rs_b = merged.robot_slices["b"]
        _set_q(engine, 0, rs_a.q_slice.start, px=-3.0, pz=1.0)
        _set_q(engine, 0, rs_b.q_slice.start, px=3.0, pz=1.0)

        engine.step(np.zeros((1, 0)), 2e-4)
        contacts = _get_contacts(engine)
        body_body = [c for c in contacts if c["bj"] >= 0]
        assert len(body_body) == 0, (
            f"Expected no body-body contact for far-apart bodies, got {len(body_body)}"
        )

    def test_multishape_body_body_500_steps_stable(self):
        """Two multi-shape bodies colliding: 500 steps stable, no NaN."""
        m_a = _multishape_pair_model(radius=0.05, sep=0.15)
        m_b = _multishape_pair_model(radius=0.05, sep=0.15)
        merged = merge_models(robots={"a": m_a, "b": m_b})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)
        engine.reset()

        # Place close together, let them interact
        rs_a = merged.robot_slices["a"]
        rs_b = merged.robot_slices["b"]
        _set_q(engine, 0, rs_a.q_slice.start, px=-0.05, pz=0.5)
        _set_q(engine, 0, rs_b.q_slice.start, px=0.05, pz=0.5)

        for _ in range(500):
            engine.step(np.zeros((1, 0)), 2e-4)

        q = engine._scratch.q.numpy()[0]
        assert np.all(np.isfinite(q)), "NaN after multi-shape body-body collision"


# ---------------------------------------------------------------------------
# 8. Shape rotation (non-zero origin_rpy)
# ---------------------------------------------------------------------------


class TestShapeRotation:
    """Shapes with non-zero origin_rpy must be handled correctly."""

    def test_rotated_box_contacts_ground(self):
        """Box rotated 45° about z-axis: should still contact ground."""
        model = _rotated_box_model(rpy=[0.0, 0.0, np.pi / 4])
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        q0 = merged.tree.default_state()[0].copy()
        q0[6] = 0.04  # close to ground
        engine.reset(q0=q0)

        engine.step(np.zeros((1, 0)), 2e-4)
        contacts = _get_contacts(engine)
        assert len(contacts) >= 1, f"Rotated box should contact ground, got {len(contacts)}"

    def test_rotated_box_depth_reasonable(self):
        """Rotated box depth should reflect actual geometry, not identity-rotation depth."""
        # Box (0.1, 0.06, 0.2) → half-extents (0.05, 0.03, 0.1)
        # Rotated 90° about x: new half-extents become (0.05, 0.1, 0.03)
        # Lowest point at z = center_z - 0.03
        model = _rotated_box_model(rpy=[np.pi / 2, 0.0, 0.0], box_size=(0.1, 0.06, 0.2))
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        # Place center at z = 0.02 → lowest point at 0.02 - 0.03 = -0.01
        # → penetration depth ≈ 0.01
        q0 = merged.tree.default_state()[0].copy()
        q0[6] = 0.02
        engine.reset(q0=q0)

        engine.step(np.zeros((1, 0)), 2e-4)
        contacts = _get_contacts(engine)
        assert len(contacts) >= 1, "Rotated box should contact ground"

        depth = contacts[0]["depth"]
        # After 90° x-rotation, half-extent in z is 0.03 (was hy=0.03)
        expected_depth = 0.03 - 0.02  # 0.01
        np.testing.assert_allclose(
            depth,
            expected_depth,
            atol=0.005,
            err_msg=f"Rotated box depth {depth:.4f} vs expected {expected_depth:.4f}",
        )

    def test_shape_rotation_propagated_to_gpu(self):
        """Non-zero origin_rpy should produce non-identity rotation in flat shape array."""
        model = _rotated_box_model(rpy=[0.0, 0.0, np.pi / 4])
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        # Check flat shape rotation is not identity
        s = engine._static
        R_flat = s.shape_rotation[0].reshape(3, 3)
        assert not np.allclose(R_flat, np.eye(3), atol=1e-6), (
            "Shape rotation should be non-identity for non-zero origin_rpy"
        )

    def test_rotated_box_stable_1000_steps(self):
        """Rotated box: 1000 steps stable, no NaN."""
        model = _rotated_box_model(rpy=[np.pi / 6, np.pi / 4, 0.0])
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        q0 = merged.tree.default_state()[0].copy()
        q0[6] = 0.5
        engine.reset(q0=q0)

        for _ in range(1000):
            engine.step(np.zeros((1, 0)), 2e-4)

        q = engine._scratch.q.numpy()[0]
        assert np.all(np.isfinite(q)), "NaN after 1000 steps with rotated box"
        assert 0.0 < q[6] < 0.5, f"Unexpected z={q[6]:.4f} for rotated box"
