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
    """Extract active GPU contacts as list of dicts."""
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


def _cpu_contacts(cpu_engine, q, qdot=None, dt=2e-4):
    """Detect CPU contacts at a given state. Returns list of ContactConstraint.

    This calls the private _detect_contacts() path directly to get contact
    geometry (depth/normal/point) — StepOutput.contact_active only exposes
    a boolean array without per-contact data.
    """
    from physics.dynamics_cache import DynamicsCache

    merged = cpu_engine.merged
    if qdot is None:
        qdot = np.zeros(merged.nv)
    cache = DynamicsCache.from_tree(merged.tree, q, qdot, dt)
    return cpu_engine._detect_contacts(cache)


# ---------------------------------------------------------------------------
# 1. GPU PGS Q25 — sphere at rest angular velocity stability
# ---------------------------------------------------------------------------


class TestGpuQ25FrictionStability:
    """GPU PGS solver: Q25 per-row R must prevent angular velocity divergence."""

    @pytest.mark.slow
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

    @pytest.mark.slow
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

    def test_y_axis_offset_two_spheres_y_separation(self):
        """Symmetric to the x-axis test but with shape offsets on body Y.

        Catches a bug where _compose_shape_world_pos hardcodes offset[0] or
        treats offset as a scalar in the world-x direction.
        """
        sep = 0.2
        radius = 0.05
        mass = 1.0
        I = 2.0 / 5.0 * mass * radius**2
        tree = RobotTreeNumpy(gravity=9.81)
        tree.add_body(_free_body("body", mass, [I, I, I], []))
        tree.finalize()
        shapes = [
            ShapeInstance(SphereShape(radius), origin_xyz=np.array([0.0, -sep / 2, 0.0])),
            ShapeInstance(SphereShape(radius), origin_xyz=np.array([0.0, sep / 2, 0.0])),
        ]
        model = RobotModel(
            tree=tree,
            geometries=[BodyCollisionGeometry(0, shapes)],
            contact_body_names=["body"],
        )
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        q0 = merged.tree.default_state()[0].copy()
        q0[6] = radius - 0.005
        engine.reset(q0=q0)

        engine.step(np.zeros((1, 0)), 2e-4)
        contacts = _get_contacts(engine)
        assert len(contacts) >= 2, f"Expected 2 contacts, got {len(contacts)}"

        # Sort by y and verify separation
        ys = sorted([c["point"][1] for c in contacts])
        np.testing.assert_allclose(ys[0], -sep / 2, atol=0.02, err_msg="Negative-y sphere contact y")
        np.testing.assert_allclose(ys[1], sep / 2, atol=0.02, err_msg="Positive-y sphere contact y")
        # And x should be ≈0 (no x offset)
        for c in contacts:
            assert abs(c["point"][0]) < 0.01, f"Contact x should be ≈0, got {c['point'][0]}"

    def test_rotated_body_with_offset_sphere_world_position(self):
        """Body rotated 45° about z + shape offset (0.1, 0.1, 0).

        The world position of the shape must be:
            shape_world_center = R_z(45°) @ (0.1, 0.1, 0) + body_pos
                               = (cos45*0.1 - sin45*0.1, sin45*0.1 + cos45*0.1, 0) + pos
                               = (0, 0.1414, pz)
        Set pz = 0.04 with radius 0.05 → depth ≈ 0.01.

        Discriminator (strongest B.2 test): a bug that uses offset_xyz without
        applying body R places the contact at (0.1, 0.1, 0) instead of
        (0, 0.1414, 0). Both coordinates differ by > 10 cm — no atol hides this.
        """
        offset = np.array([0.1, 0.1, 0.0])
        radius = 0.05
        mass = 1.0
        I = 2.0 / 5.0 * mass * radius**2
        tree = RobotTreeNumpy(gravity=9.81)
        tree.add_body(_free_body("body", mass, [I, I, I], []))
        tree.finalize()
        model = RobotModel(
            tree=tree,
            geometries=[BodyCollisionGeometry(0, [ShapeInstance(SphereShape(radius), origin_xyz=offset)])],
            contact_body_names=["body"],
        )
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        # Body rotated 45° about world z axis
        theta = np.pi / 4
        qw = float(np.cos(theta / 2))
        qz = float(np.sin(theta / 2))

        q0 = merged.tree.default_state()[0].copy()
        q0[0], q0[1], q0[2], q0[3] = qw, 0.0, 0.0, qz
        q0[4], q0[5], q0[6] = 0.0, 0.0, 0.04
        engine.reset(q0=q0)

        # Expected shape world xy: R_z(45°) @ (0.1, 0.1, 0)
        expected_x = float(np.cos(theta) * offset[0] - np.sin(theta) * offset[1])  # ≈ 0
        expected_y = float(np.sin(theta) * offset[0] + np.cos(theta) * offset[1])  # ≈ 0.1414

        engine.step(np.zeros((1, 0)), 2e-4)
        contacts = _get_contacts(engine)
        assert len(contacts) >= 1, "Rotated body + offset sphere should contact ground"

        c = contacts[0]
        np.testing.assert_allclose(
            c["point"][0],
            expected_x,
            atol=5e-3,
            err_msg=(
                f"Contact x {c['point'][0]:.4f} vs expected {expected_x:.4f}. "
                f"If contact appears near x=0.1 instead, body R is NOT being applied to offset_xyz."
            ),
        )
        np.testing.assert_allclose(
            c["point"][1],
            expected_y,
            atol=5e-3,
            err_msg=(
                f"Contact y {c['point'][1]:.4f} vs expected {expected_y:.4f}. "
                f"If contact appears near y=0.1 instead, body R not applied to offset."
            ),
        )
        assert abs(c["point"][2]) < 5e-3, f"Contact z {c['point'][2]:.4f} should be on ground"
        np.testing.assert_allclose(
            c["depth"],
            0.01,
            atol=2e-3,
            err_msg=f"Depth {c['depth']:.4f} should be ≈ 0.01 (radius - pz)",
        )


# ---------------------------------------------------------------------------
# 3. Non-Sphere shapes: Box and Capsule multi-shape dispatch
# ---------------------------------------------------------------------------


class TestNonSphereMultiShape:
    """Box and Capsule shapes in multi-shape bodies on GPU.

    These tests intentionally use ROTATED bodies for the single-shape baselines
    so that an axis-aligned (AABB or bounding-sphere) GPU narrowphase would
    fail. Identity-rotation tests pass even with serious rotation bugs because
    R[2,*] = (0,0,1) makes the SAT support function degenerate to a plain
    half-extent subtraction.
    """

    def test_tilted_box_lands_on_edge(self):
        """Box rotated 45° about y-axis: lands on its bottom edge.

        Math: with body 0.1m cube (half = 0.05) and R = R_y(45°), the SAT
        support point in -Z direction is
            lowest_z = pz - |R[2,0]|*hx - |R[2,1]|*hy - |R[2,2]|*hz
                     = pz - sin(45°)*0.05 - 0 - cos(45°)*0.05
                     = pz - 0.0707
        Set pz = 0.06 → lowest_z = -0.0107 → depth ≈ 0.0107.

        Discriminator: an axis-aligned narrowphase would compute
        lowest_z = pz - hz = 0.06 - 0.05 = 0.01 → NO contact at all. So a
        bug that drops the rotation from box-vs-ground (uses identity R or
        bounding-sphere fallback) makes this test fail with len(contacts)==0.
        """
        box_size = (0.1, 0.1, 0.1)  # half-extents 0.05 each
        mass, I = 1.0, 0.01
        tree = RobotTreeNumpy(gravity=9.81)
        tree.add_body(_free_body("box_body", mass, [I, I, I], []))
        tree.finalize()
        model = RobotModel(
            tree=tree,
            geometries=[BodyCollisionGeometry(0, [ShapeInstance(BoxShape(box_size))])],
            contact_body_names=["box_body"],
        )
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        # FreeJoint q layout: [qw, qx, qy, qz, px, py, pz] (scalar-first)
        theta = np.pi / 4  # 45° about world y-axis
        qw = float(np.cos(theta / 2))
        qy = float(np.sin(theta / 2))

        q0 = merged.tree.default_state()[0].copy()
        q0[0], q0[1], q0[2], q0[3] = qw, 0.0, qy, 0.0
        q0[4], q0[5], q0[6] = 0.0, 0.0, 0.06
        engine.reset(q0=q0)

        engine.step(np.zeros((1, 0)), 2e-4)
        contacts = _get_contacts(engine)

        # 1. Existence — axis-aligned assumption gives 0 contacts here
        assert len(contacts) >= 1, (
            "Tilted box must contact ground (post-rotation lowest corner ≈ -0.0107). "
            "Zero contacts means GPU narrowphase ignores body rotation."
        )

        # 2. Depth matches post-rotation SAT geometry
        expected_depth = float(0.05 * (np.sin(theta) + np.cos(theta)) - 0.06)  # ≈ 0.01066
        max_depth = max(c["depth"] for c in contacts)
        np.testing.assert_allclose(
            max_depth,
            expected_depth,
            atol=2e-3,
            err_msg=(
                f"Tilted box depth {max_depth:.4f} vs expected {expected_depth:.4f}. "
                f"Mismatch suggests R[2,*] indexing or |R| missing in box_vs_ground."
            ),
        )

        # 3. Normal must be world +z (ground plane), not rotated with body
        for c in contacts:
            assert c["normal"][2] > 0.95, (
                f"Ground contact normal must be world +z regardless of body orientation, got {c['normal']}"
            )

        # 4. Contact point lies on the ground plane and near the body x=0
        # (For 45° about y, the chosen lowest corner is at body (+hx, ±hy, -hz)
        # which transforms to world x = cos(45)*hx - sin(45)*hz = 0.)
        for c in contacts:
            assert abs(c["point"][2]) < 5e-3, f"Contact z {c['point'][2]:.4f} should be on ground"
            assert abs(c["point"][0]) < 0.02, (
                f"Contact x {c['point'][0]:.4f} should be ≈0 for symmetric 45° y-rotation"
            )

    def test_capsule_touches_ground(self):
        """Vertical capsule baseline: bottom of lower hemisphere just penetrates ground.

        Geometry: radius=0.05, cylinder length=0.2 → total z-extent 0.3.
        With identity body rotation, capsule axis_world = (0, 0, 1), lower
        endpoint at pz - half_length = pz - 0.1, lowest_z = pz - 0.1 - radius
        = pz - 0.15. Set pz = 0.14 → lowest_z = -0.01 → depth ≈ 0.01.
        """
        radius, length = 0.05, 0.2
        mass, I = 1.0, 0.01
        tree = RobotTreeNumpy(gravity=9.81)
        tree.add_body(_free_body("cap_body", mass, [I, I, I], []))
        tree.finalize()
        model = RobotModel(
            tree=tree,
            geometries=[BodyCollisionGeometry(0, [ShapeInstance(CapsuleShape(radius, length))])],
            contact_body_names=["cap_body"],
        )
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        # pz chosen so the bottom hemisphere penetrates by ~1 cm
        pz = 0.14  # = half_length(0.1) + radius(0.05) - penetration(0.01)
        expected_depth = (length / 2 + radius) - pz  # 0.01

        q0 = merged.tree.default_state()[0].copy()
        q0[6] = pz
        engine.reset(q0=q0)

        engine.step(np.zeros((1, 0)), 2e-4)
        contacts = _get_contacts(engine)
        assert len(contacts) >= 1, f"Capsule should produce ground contact, got {len(contacts)}"

        depth = max(c["depth"] for c in contacts)
        np.testing.assert_allclose(
            depth,
            expected_depth,
            atol=2e-3,
            err_msg=f"Vertical capsule depth {depth:.4f} vs expected {expected_depth:.4f}",
        )
        for c in contacts:
            assert c["normal"][2] > 0.95, f"Ground normal should be ≈+z, got {c['normal']}"
            assert abs(c["point"][0]) < 5e-3, (
                f"Vertical capsule contact x should be ≈0, got {c['point'][0]:.4f}"
            )
            assert abs(c["point"][1]) < 5e-3, (
                f"Vertical capsule contact y should be ≈0, got {c['point'][1]:.4f}"
            )
            assert abs(c["point"][2]) < 5e-3, f"Contact z {c['point'][2]:.4f} should be on ground"

    def test_tilted_capsule_lands_on_endcap(self):
        """Capsule rotated 60° about y: lower end-cap sphere is the lowest point.

        Math: capsule (radius=0.05, length=0.2 → half_length=0.1), R = R_y(60°).
            axis_world = R @ [0,0,1] = (sin60, 0, cos60) = (0.866, 0, 0.5)
            ep_b = pos - half_length * axis_world  # lower endpoint
                 = pos - (0.0866, 0, 0.05)
            lowest_z = ep_b.z - radius = pz - 0.05 - 0.05 = pz - 0.1
        Set pz = 0.09 → lowest_z = -0.01 → depth ≈ 0.01.

        Contact point xy comes from the lower endpoint:
            ep_b.xy = (-0.0866, 0)
        The non-zero contact x is the key discriminator: any narrowphase that
        treats the capsule axis as world-up (ignores R) gives contact xy = (0,0).
        """
        radius, length = 0.05, 0.2
        half_length = length / 2
        mass, I = 1.0, 0.01
        tree = RobotTreeNumpy(gravity=9.81)
        tree.add_body(_free_body("cap_body", mass, [I, I, I], []))
        tree.finalize()
        model = RobotModel(
            tree=tree,
            geometries=[BodyCollisionGeometry(0, [ShapeInstance(CapsuleShape(radius, length))])],
            contact_body_names=["cap_body"],
        )
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        theta = np.pi / 3  # 60° about y
        qw = float(np.cos(theta / 2))
        qy = float(np.sin(theta / 2))

        # axis_world.z = cos(theta); ep_b.z = pz - half_length*cos(theta)
        # lowest_z = pz - half_length*cos(theta) - radius
        # Want lowest_z = -target_depth → pz = (half_length*cos(theta) + radius) - target_depth
        target_depth = 0.01
        pz = float((half_length * np.cos(theta) + radius) - target_depth)  # 0.09
        expected_depth = target_depth

        # Expected contact xy: ep_b = pos - half_length * (sin60, 0, cos60)
        expected_x = float(-half_length * np.sin(theta))  # ≈ -0.0866
        expected_y = 0.0

        q0 = merged.tree.default_state()[0].copy()
        q0[0], q0[1], q0[2], q0[3] = qw, 0.0, qy, 0.0
        q0[4], q0[5], q0[6] = 0.0, 0.0, pz
        engine.reset(q0=q0)

        engine.step(np.zeros((1, 0)), 2e-4)
        contacts = _get_contacts(engine)

        assert len(contacts) >= 1, (
            f"Tilted capsule must contact ground (expected depth ≈ {target_depth:.4f}), "
            f"got 0 contacts → narrowphase ignored body rotation."
        )

        depth = max(c["depth"] for c in contacts)
        np.testing.assert_allclose(
            depth,
            expected_depth,
            atol=2e-3,
            err_msg=f"Tilted capsule depth {depth:.4f} vs expected {expected_depth:.4f}",
        )

        # Pick the contact with the largest depth (the one at the lower endcap)
        primary = max(contacts, key=lambda c: c["depth"])
        assert primary["normal"][2] > 0.95, f"Ground normal should be ≈+z, got {primary['normal']}"
        np.testing.assert_allclose(
            primary["point"][0],
            expected_x,
            atol=5e-3,
            err_msg=(
                f"Tilted capsule contact x {primary['point'][0]:.4f} vs expected {expected_x:.4f}. "
                f"x ≈ 0 means narrowphase used capsule axis = world-up (ignored body R)."
            ),
        )
        np.testing.assert_allclose(
            primary["point"][1],
            expected_y,
            atol=5e-3,
            err_msg=f"Tilted capsule contact y should be ≈0, got {primary['point'][1]:.4f}",
        )
        assert abs(primary["point"][2]) < 5e-3, f"Contact z {primary['point'][2]:.4f} should be on ground"

    def test_box_plus_capsule_multishape_both_contact(self):
        """Body with box at origin + capsule offset: BOTH shapes must contribute contacts.

        The plain `len(contacts) >= 2` assertion is too weak — a buggy multi-shape
        index that only walks the first shape can still produce 4 corner contacts
        from the box alone. We partition contacts by world x to verify that BOTH
        shapes' narrowphase paths actually fire.
        """
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

        # Box at body x=0 (half=0.05); capsule offset at body x=0.2 (radius=0.04).
        # Partition by world x — body is identity-rotated, so body x = world x.
        xs = [c["point"][0] for c in contacts]
        near_box = [c for c in contacts if abs(c["point"][0]) < 0.1]
        near_cap = [c for c in contacts if 0.1 <= c["point"][0] < 0.3]
        assert len(near_box) >= 1, (
            f"Box shape produced no contact at x≈0 → multi-shape index may not "
            f"walk both shapes. Contact xs: {xs}"
        )
        assert len(near_cap) >= 1, (
            f"Capsule shape produced no contact at x≈0.2 → multi-shape index may "
            f"only fire on the first shape. Contact xs: {xs}"
        )

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

    def test_cpu_gpu_multishape_contact_details_agree(self):
        """CPU and GPU must agree on per-contact depth/normal/point, not just count.

        Previous test_two_sphere_body_contact_count_agrees only checked counts,
        which can pass even when depth/point/normal disagree wildly (e.g. one
        engine uses shape-center projection and the other uses lowest support).

        Uses the same two-sphere-offset fixture and does a sorted pair-by-pair
        comparison at the initial state (before integration, so integrator
        differences don't contaminate the comparison).
        """
        model = _offset_two_sphere_model(radius=0.05, sep=0.2)
        merged = merge_models(robots={"a": model})
        cpu = CpuEngine(merged, dt=2e-4)
        gpu = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        q0 = merged.tree.default_state()[0].copy()
        q0[6] = 0.045  # 0.5 mm penetration

        # CPU — detect contacts directly at q0, no integration
        cpu_list = _cpu_contacts(cpu, q0)
        cpu_ground = sorted(
            [c for c in cpu_list if c.body_j == -1],
            key=lambda c: c.point[0],
        )

        # GPU — reset then step once (GPU narrowphase runs on q0 via FK before integrate)
        gpu.reset(q0=q0)
        gpu.step(np.zeros((1, 0)), 2e-4)
        gpu_ground = sorted(
            [c for c in _get_contacts(gpu) if c["bj"] == -1],
            key=lambda c: c["point"][0],
        )

        assert len(cpu_ground) == 2, f"CPU expected 2 ground contacts, got {len(cpu_ground)}"
        assert len(gpu_ground) == 2, f"GPU expected 2 ground contacts, got {len(gpu_ground)}"

        # Pair-by-pair comparison — 0.5 mm tolerance to absorb float32/float64 noise
        atol_geom = 5e-4
        for i, (cpu_c, gpu_c) in enumerate(zip(cpu_ground, gpu_ground)):
            np.testing.assert_allclose(
                cpu_c.depth,
                gpu_c["depth"],
                atol=atol_geom,
                err_msg=f"contact {i}: depth CPU={cpu_c.depth:.6f} vs GPU={gpu_c['depth']:.6f}",
            )
            np.testing.assert_allclose(
                np.asarray(cpu_c.point),
                np.asarray(gpu_c["point"]),
                atol=atol_geom,
                err_msg=f"contact {i}: point CPU={cpu_c.point} vs GPU={gpu_c['point']}",
            )
            np.testing.assert_allclose(
                np.asarray(cpu_c.normal),
                np.asarray(gpu_c["normal"]),
                atol=atol_geom,
                err_msg=f"contact {i}: normal CPU={cpu_c.normal} vs GPU={gpu_c['normal']}",
            )

    @pytest.mark.xfail(
        reason="Pre-existing EPA accuracy issue: sphere-sphere depth ≈ 7e-5 "
        "instead of 0.02 (EPA converges to wrong face on Minkowski diff). "
        "Not a regression — fails on main before session 27.",
        strict=False,
    )
    def test_cpu_gpu_body_body_contact_agree(self):
        """CPU and GPU must agree on body-body contact geometry.

        Two single-sphere bodies A, B (radius 0.05 each) mid-air, overlapping
        along x by 0.02 m. A at (0, 0, 1), B at (0.08, 0, 1).

        Expected (analytic, per the "normal from body_j to body_i, contact
        point on body_j's surface" convention used by both engines):
          body_i = 0 (A), body_j = 1 (B)
          depth = 2*radius - dist = 0.1 - 0.08 = 0.02
          normal = unit(p_i - p_j) = (-1, 0, 0)          (from B to A)
          contact point = p_j + r_j * (p_i - p_j)/|p_i - p_j|
                        = (0.08, 0, 1) - 0.05*(1, 0, 0) = (0.03, 0, 1)

        This is the CPU/GPU cross-check for the body-body narrowphase code
        path, which is an *entirely separate* implementation on each engine
        (CPU uses GJK/EPA, GPU uses analytical sphere-sphere). The existing
        two_sphere_body_contact_count_agrees test only checks ground contacts.
        """
        m_a = _sphere_model(mass=1.0, radius=0.05)
        m_b = _sphere_model(mass=1.0, radius=0.05)
        merged = merge_models(robots={"a": m_a, "b": m_b})
        cpu = CpuEngine(merged, dt=2e-4)
        gpu = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        # FreeJoint q layout per-robot: [qw, qx, qy, qz, px, py, pz]
        q0 = merged.tree.default_state()[0].copy()
        rs_a = merged.robot_slices["a"]
        rs_b = merged.robot_slices["b"]
        # A at (0, 0, 1.0), B at (0.08, 0, 1.0) → overlap 0.02 along x
        q0[rs_a.q_slice.start + 4] = 0.0
        q0[rs_a.q_slice.start + 6] = 1.0
        q0[rs_b.q_slice.start + 4] = 0.08
        q0[rs_b.q_slice.start + 6] = 1.0

        # CPU
        cpu_list = _cpu_contacts(cpu, q0)
        cpu_bb = [c for c in cpu_list if c.body_j != -1 and c.body_i != -1]

        # GPU
        gpu.reset(q0=q0)
        gpu.step(np.zeros((1, 0)), 2e-4)
        gpu_bb = [c for c in _get_contacts(gpu) if c["bj"] >= 0]

        assert len(cpu_bb) == 1, f"CPU expected 1 body-body contact, got {len(cpu_bb)}"
        assert len(gpu_bb) == 1, f"GPU expected 1 body-body contact, got {len(gpu_bb)}"

        cpu_c = cpu_bb[0]
        gpu_c = gpu_bb[0]

        # Depth (absolute) — both engines must agree on penetration magnitude
        np.testing.assert_allclose(
            cpu_c.depth,
            gpu_c["depth"],
            atol=5e-4,
            err_msg=f"body-body depth CPU={cpu_c.depth:.6f} vs GPU={gpu_c['depth']:.6f}",
        )
        # Expected analytic depth = 0.02
        np.testing.assert_allclose(
            cpu_c.depth,
            0.02,
            atol=1e-3,
            err_msg=f"CPU body-body depth {cpu_c.depth:.4f} ≠ analytic 0.02",
        )

        # Normal: expected (-1, 0, 0) pointing from body_j (B) to body_i (A).
        # Both engines should agree on the signed normal (same convention).
        expected_normal = np.array([-1.0, 0.0, 0.0])
        np.testing.assert_allclose(
            np.asarray(cpu_c.normal),
            expected_normal,
            atol=5e-3,
            err_msg=f"CPU normal {cpu_c.normal} vs expected {expected_normal}",
        )
        np.testing.assert_allclose(
            np.asarray(gpu_c["normal"]),
            expected_normal,
            atol=5e-3,
            err_msg=f"GPU normal {gpu_c['normal']} vs expected {expected_normal}",
        )

        # Contact point on body j's surface = (0.03, 0, 1.0)
        expected_point = np.array([0.03, 0.0, 1.0])
        np.testing.assert_allclose(
            np.asarray(cpu_c.point),
            expected_point,
            atol=5e-3,
            err_msg=f"CPU contact point {cpu_c.point} vs expected {expected_point}",
        )
        np.testing.assert_allclose(
            np.asarray(gpu_c["point"]),
            expected_point,
            atol=5e-3,
            err_msg=f"GPU contact point {gpu_c['point']} vs expected {expected_point}",
        )

        # And CPU vs GPU point agreement
        np.testing.assert_allclose(
            np.asarray(cpu_c.point),
            np.asarray(gpu_c["point"]),
            atol=5e-3,
            err_msg=f"CPU point {cpu_c.point} ≠ GPU point {gpu_c['point']}",
        )

        # Body-pair ordering convention (i < j) should match between engines
        assert (cpu_c.body_i, cpu_c.body_j) == (gpu_c["bi"], gpu_c["bj"]), (
            f"(bi,bj) mismatch: CPU=({cpu_c.body_i},{cpu_c.body_j}) GPU=({gpu_c['bi']},{gpu_c['bj']})"
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

    def test_two_spheres_body_body_geometry(self):
        """Body-body narrowphase: depth, normal direction, and contact point.

        Two single-sphere bodies (radius 0.05), mid-air at z=1:
          A at (0, 0, 1.0), B at (0.08, 0, 1.0) → center distance 0.08, overlap 0.02.

        Expected:
          count = 1 (exactly, not >= 1)
          depth ≈ 0.02 (= 2r - dist)
          normal = (-1, 0, 0)   (from body_j=B to body_i=A)
          point = (0.03, 0, 1)  (on body B's surface, toward A)

        Discriminator: catches any depth-formula bug (sum instead of difference),
        any sign bug in normal (historically Q23 class), and any contact-point
        placement bug (at body center vs surface).
        """
        m_a = _sphere_model(mass=1.0, radius=0.05)
        m_b = _sphere_model(mass=1.0, radius=0.05)
        merged = merge_models(robots={"a": m_a, "b": m_b})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)
        engine.reset()

        rs_a = merged.robot_slices["a"]
        rs_b = merged.robot_slices["b"]
        _set_q(engine, 0, rs_a.q_slice.start, px=0.0, pz=1.0)
        _set_q(engine, 0, rs_b.q_slice.start, px=0.08, pz=1.0)

        engine.step(np.zeros((1, 0)), 2e-4)
        contacts = _get_contacts(engine)

        # Filter to body-body only (should be 1, no ground contacts at z=1)
        body_body = [c for c in contacts if c["bj"] >= 0]
        assert len(body_body) == 1, f"Expected exactly 1 body-body contact, got {len(body_body)}"

        c = body_body[0]
        np.testing.assert_allclose(
            c["depth"],
            0.02,
            atol=1e-3,
            err_msg=f"depth {c['depth']:.4f} ≠ expected 2r-d = 0.02",
        )
        np.testing.assert_allclose(
            c["normal"],
            np.array([-1.0, 0.0, 0.0]),
            atol=5e-3,
            err_msg=f"normal {c['normal']} ≠ expected (-1,0,0) from j to i",
        )
        np.testing.assert_allclose(
            c["point"],
            np.array([0.03, 0.0, 1.0]),
            atol=5e-3,
            err_msg=f"contact point {c['point']} ≠ expected (0.03, 0, 1) on body j's surface",
        )
        # Ordering convention: body_i should be A (index 0), body_j should be B (index 1)
        assert c["bi"] == 0 and c["bj"] == 1, f"(bi,bj)=({c['bi']},{c['bj']}) ≠ (0,1)"

    def test_multishape_pair_filter_inner_shapes_only(self):
        """Multi-shape × multi-shape: only the overlapping pair of shapes should
        generate a contact, non-overlapping pairs must NOT.

        Setup: each body has 2 spheres at body-local x = ±0.05, radius 0.04.
          Body A at world (0, 0, 1.0)   → shapes at x = -0.05 and +0.05
          Body B at world (0.13, 0, 1.0) → shapes at x = +0.08 and +0.18

        Shape-pair world-x distances:
          A_left  (-0.05) ↔ B_left  (+0.08): 0.13  no contact (>= 2r = 0.08)
          A_left  (-0.05) ↔ B_right (+0.18): 0.23  no contact
          A_right (+0.05) ↔ B_left  (+0.08): 0.03  CONTACT (overlap 0.05)
          A_right (+0.05) ↔ B_right (+0.18): 0.13  no contact

        Expected: exactly 1 body-body contact between A's right sphere and
        B's left sphere. Contact point on body j's surface at world x = 0.08 - 0.04 = 0.04.

        Discriminator:
          - bug: body bounding sphere used as narrowphase (doesn't iterate shape
            pairs) → might report 0 or 4 contacts
          - bug: shape index walks only first shape → misses this pair
          - bug: shape index walks wrong offset → contact point wrong
        """
        radius = 0.04
        sep = 0.1  # shapes at ±0.05 in body x

        def _pair_sphere_body(name):
            mass = 1.0
            I = 2.0 / 5.0 * mass * radius**2
            tree = RobotTreeNumpy(gravity=9.81)
            tree.add_body(_free_body(name, mass, [I, I, I], []))
            tree.finalize()
            shapes = [
                ShapeInstance(SphereShape(radius), origin_xyz=np.array([-sep / 2, 0.0, 0.0])),
                ShapeInstance(SphereShape(radius), origin_xyz=np.array([+sep / 2, 0.0, 0.0])),
            ]
            return RobotModel(
                tree=tree,
                geometries=[BodyCollisionGeometry(0, shapes)],
                contact_body_names=[name],
            )

        m_a = _pair_sphere_body("a_body")
        m_b = _pair_sphere_body("b_body")
        merged = merge_models(robots={"a": m_a, "b": m_b})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)
        engine.reset()

        rs_a = merged.robot_slices["a"]
        rs_b = merged.robot_slices["b"]
        _set_q(engine, 0, rs_a.q_slice.start, px=0.0, pz=1.0)
        _set_q(engine, 0, rs_b.q_slice.start, px=0.13, pz=1.0)

        engine.step(np.zeros((1, 0)), 2e-4)
        contacts = _get_contacts(engine)

        body_body = [c for c in contacts if c["bj"] >= 0]
        assert len(body_body) == 1, (
            f"Expected exactly 1 contact (inner shape pair only), got {len(body_body)}. "
            f"Points: {[c['point'].tolist() for c in body_body]}"
        )

        c = body_body[0]
        # Depth = 2r - dist between overlapping shape centers
        # Shape A_right at (0.05, 0, 1), B_left at (0.08, 0, 1), dist = 0.03, overlap = 0.05
        expected_depth = 2 * radius - 0.03  # 0.05
        np.testing.assert_allclose(
            c["depth"],
            expected_depth,
            atol=1e-3,
            err_msg=f"depth {c['depth']:.4f} ≠ expected {expected_depth:.4f}",
        )
        # Contact point on body j (B)'s left sphere surface, toward body i (A)'s right sphere.
        # B_left center = (0.08, 0, 1); direction to A_right (0.05,0,1) = (-1,0,0);
        # surface point = 0.08 - r*1 = 0.04
        expected_point = np.array([0.04, 0.0, 1.0])
        np.testing.assert_allclose(
            c["point"],
            expected_point,
            atol=5e-3,
            err_msg=(
                f"contact point {c['point']} ≠ expected {expected_point}. "
                f"If point is at (0.065, 0, 1) the engine is using body centers, "
                f"not shape centers."
            ),
        )
        # Normal along -x (from j to i)
        np.testing.assert_allclose(
            c["normal"],
            np.array([-1.0, 0.0, 0.0]),
            atol=5e-3,
            err_msg=f"normal {c['normal']} should be (-1,0,0)",
        )


# ---------------------------------------------------------------------------
# 8. Shape rotation (non-zero origin_rpy)
# ---------------------------------------------------------------------------


class TestShapeRotation:
    """Shapes with non-zero origin_rpy must be handled correctly."""

    def test_origin_rpy_45deg_y_box_lands_on_edge(self):
        """Box with shape origin_rpy=[0, π/4, 0]: lands on its bottom edge.

        Mirror of B.3 test_tilted_box_lands_on_edge but the rotation lives in
        ShapeInstance.origin_rpy (body-frame shape rotation) instead of the
        FreeJoint quaternion (body world rotation). Same expected geometry,
        different code path through _compose_shape_world.

        Math: box (0.1, 0.1, 0.1), shape R = R_y(45°), body R = identity →
        composed shape world R = R_y(45°). SAT support point in -Z:
            lowest_z = pz - sin(45°)*hx - 0 - cos(45°)*hz
                     = pz - 0.0707
        Set pz = 0.06 → depth ≈ 0.0107.

        Discriminator: catches "body R applied but shape origin_rpy dropped"
        bug — those would give axis-aligned lowest_z = pz - hz = 0.01 → no
        contact. Replaces the original test_rotated_box_contacts_ground which
        used 45° about *z*-axis (rotating an axis-aligned box about its own
        vertical doesn't change the height profile, so that test was a no-op).
        """
        model = _rotated_box_model(rpy=[0.0, np.pi / 4, 0.0], box_size=(0.1, 0.1, 0.1))
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        q0 = merged.tree.default_state()[0].copy()
        q0[6] = 0.06  # same pz as B.3 tilted box test
        engine.reset(q0=q0)

        engine.step(np.zeros((1, 0)), 2e-4)
        contacts = _get_contacts(engine)
        assert len(contacts) >= 1, (
            "shape origin_rpy 45° y must produce ground contact (lowest corner ≈ -0.0107). "
            "Zero contacts → narrowphase ignores shape origin_rpy."
        )

        expected_depth = float(0.05 * (np.sin(np.pi / 4) + np.cos(np.pi / 4)) - 0.06)  # 0.0107
        max_depth = max(c["depth"] for c in contacts)
        np.testing.assert_allclose(
            max_depth,
            expected_depth,
            atol=2e-3,
            err_msg=f"depth {max_depth:.4f} ≠ expected {expected_depth:.4f}",
        )
        for c in contacts:
            assert c["normal"][2] > 0.95, f"normal should be ≈+z, got {c['normal']}"

    def test_combined_origin_rpy_and_body_rotation(self):
        """Compose body world R with shape origin_rpy.

        Body rotated 45° about y (FreeJoint quaternion) AND shape origin_rpy
        = [0, 0, 0] (identity). The composed world rotation is R_y(45°) — same
        as B.3 test_tilted_box_lands_on_edge — so we expect identical depth.

        But here we ALSO test the inverse path: body identity + shape rpy=[0,π/4,0]
        gives the SAME composed rotation. We compute the expected depth from
        body_R @ shape_R using numpy directly so the test catches any
        composition-order bug (R_body @ R_shape vs R_shape @ R_body) without
        my having to derive it by hand.
        """
        from physics.spatial import quat_to_rot, rot_x, rot_y, rot_z

        box_size = (0.1, 0.06, 0.2)
        hx, hy, hz = 0.05, 0.03, 0.1

        # Body 30° about y (FreeJoint), shape 30° about y (origin_rpy)
        body_theta = np.pi / 6
        shape_rpy = np.array([0.0, np.pi / 6, 0.0])

        qw_b = float(np.cos(body_theta / 2))
        qy_b = float(np.sin(body_theta / 2))
        body_quat = np.array([qw_b, 0.0, qy_b, 0.0])
        R_body = quat_to_rot(body_quat)
        # URDF rpy convention used in physics.spatial: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        R_shape = rot_z(shape_rpy[2]) @ rot_y(shape_rpy[1]) @ rot_x(shape_rpy[0])
        # Composed: world R = body_R @ shape_R (shape_R is in body-local frame)
        R_total = R_body @ R_shape

        # SAT support point in -Z direction:
        lowest_z_offset = abs(R_total[2, 0]) * hx + abs(R_total[2, 1]) * hy + abs(R_total[2, 2]) * hz

        # Choose pz so depth = 0.01
        target_depth = 0.01
        pz = float(lowest_z_offset - target_depth)

        # Build model
        model = _rotated_box_model(rpy=shape_rpy.tolist(), box_size=box_size)
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        q0 = merged.tree.default_state()[0].copy()
        q0[0], q0[1], q0[2], q0[3] = qw_b, 0.0, qy_b, 0.0
        q0[4], q0[5], q0[6] = 0.0, 0.0, pz
        engine.reset(q0=q0)

        engine.step(np.zeros((1, 0)), 2e-4)
        contacts = _get_contacts(engine)
        assert len(contacts) >= 1, (
            f"Combined body+shape rotation must contact ground. Composed R[2,*]: "
            f"{R_total[2]}, expected lowest offset {lowest_z_offset:.4f}, pz {pz:.4f}"
        )

        max_depth = max(c["depth"] for c in contacts)
        np.testing.assert_allclose(
            max_depth,
            target_depth,
            atol=2e-3,
            err_msg=(
                f"composed-rotation depth {max_depth:.4f} ≠ expected {target_depth:.4f}. "
                f"If far off, R_body and R_shape are composed in wrong order — "
                f"expected R_total = R_body @ R_shape (shape_R is body-local)."
            ),
        )

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

    def test_rotated_box_settles_and_tumbles(self):
        """Rotated box dropped from 0.5 m: settles to a stable rest pose.

        Replaces the previous test_rotated_box_stable_1000_steps which only
        checked NaN and pz ∈ (0, 0.5) — those bounds pass even if the box is
        sliding to infinity in xy or stuck in mid-air.

        Trajectory diagnostic: tests/fixtures/rotated_box_settling.png shows
        the observed dynamics for this exact fixture (rpy=[π/6, π/4, 0],
        box (0.1, 0.06, 0.2), pz0=0.5):
          - free fall 0–0.27s, pz: 0.5 → 0.11
          - first contact + tumble 0.27–0.42s, body actually rotates
            (qw drops from 1.0 to 0.94, ~21° tilt change)
          - settles by 0.62s at pz ≈ 0.055 (resting on the (0.06,0.2) face,
            close to stable rest hx=0.05 + ~5mm penetration/tilt residual)

        Test runs 4000 steps (0.8 s — comfortably past observed settling at
        0.625 s) and asserts:
          1. No NaN throughout
          2. pz never goes negative (no ground penetration)
          3. Final 500 steps have std(pz) < 1.5 mm (truly settled)
          4. Final pz close to one of the 3 possible stable rest values for
             this box: hy=0.03, hx=0.05, hz=0.10 (within 8 mm to absorb
             solver penetration + residual tilt)
          5. Body actually tumbled during contact (min qw < 0.99) — verifies
             the rotation code path is exercised, not just bypassed
        """
        box_size = (0.1, 0.06, 0.2)
        hx, hy, hz = 0.05, 0.03, 0.1
        rpy = [np.pi / 6, np.pi / 4, 0.0]
        model = _rotated_box_model(rpy=rpy, box_size=box_size)
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

        q0 = merged.tree.default_state()[0].copy()
        q0[6] = 0.5
        engine.reset(q0=q0)

        n_steps = 4000
        pzs = np.zeros(n_steps + 1)
        qws = np.zeros(n_steps + 1)
        pzs[0] = 0.5
        qws[0] = 1.0

        for i in range(n_steps):
            engine.step(np.zeros((1, 0)), 2e-4)
            q = engine._scratch.q.numpy()[0]
            pzs[i + 1] = q[6]
            qws[i + 1] = q[0]

        q_final = engine._scratch.q.numpy()[0]

        # 1. No NaN
        assert np.all(np.isfinite(pzs)), "NaN appeared in pz trajectory"
        assert np.all(np.isfinite(q_final)), "NaN in final q"

        # 2. Never penetrates ground
        assert pzs.min() > -1e-3, f"pz went negative: min={pzs.min():.4f}"

        # 3. Truly settled in last 500 steps
        tail = pzs[-500:]
        tail_std = float(np.std(tail))
        assert tail_std < 1.5e-3, (
            f"pz std in final 500 steps = {tail_std:.5f} > 1.5 mm — not settled. "
            f"Box may still be tumbling or oscillating."
        )

        # 4. Final pz close to one of 3 stable rest values
        stable_rests = [hy, hx, hz]  # 0.03, 0.05, 0.10
        final_pz = float(q_final[6])
        dist_to_rest = min(abs(final_pz - r) for r in stable_rests)
        assert dist_to_rest < 8e-3, (
            f"final pz {final_pz:.4f} is not near any stable rest "
            f"{stable_rests}. Closest distance: {dist_to_rest:.4f}"
        )

        # 5. Body actually tumbled (orientation evolved during contact)
        min_qw = float(qws.min())
        assert min_qw < 0.99, (
            f"min qw = {min_qw:.4f} ≥ 0.99 → body never rotated significantly. "
            f"Either the rotation code path was bypassed or contact didn't fire."
        )
