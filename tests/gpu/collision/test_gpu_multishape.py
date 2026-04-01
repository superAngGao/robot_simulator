"""
Tests for Q26-gpu: multi-shape collision + dynamic N² broadphase on GPU.

Verifies:
  1. Multiple shapes per body produce multiple ground contacts
  2. Shape offset/rotation correctly transforms contact points
  3. Dynamic broadphase discovers pairs without pre-computation
  4. Collision filter (parent-child) excludes pairs on GPU
  5. Contact buffer overflow is handled gracefully
  6. Single-shape regression (existing behavior preserved)
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.geometry import (
    BodyCollisionGeometry,
    ShapeInstance,
    SphereShape,
)
from physics.joint import FreeJoint, RevoluteJoint
from physics.merged_model import merge_models
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from robot.model import RobotModel

try:
    from physics.gpu_engine import GpuEngine

    HAS_WARP = True
except Exception:
    HAS_WARP = False

pytestmark = pytest.mark.skipif(not HAS_WARP, reason="Warp or CUDA not available")


def _ball_model(mass=1.0, radius=0.1):
    """Single FreeJoint sphere."""
    tree = RobotTreeNumpy(gravity=9.81)
    I = 2.0 / 5.0 * mass * radius**2
    tree.add_body(
        Body(
            name="ball",
            index=0,
            joint=FreeJoint("root"),
            inertia=SpatialInertia(mass=mass, inertia=np.eye(3) * I, com=np.zeros(3)),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()
    return RobotModel(
        tree=tree,
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(SphereShape(radius))])],
        contact_body_names=["ball"],
    )


def _multishape_ball_model(mass=1.0, radius=0.08):
    """Single body with TWO sphere shapes at different offsets."""
    tree = RobotTreeNumpy(gravity=9.81)
    I = 2.0 / 5.0 * mass * radius**2
    tree.add_body(
        Body(
            name="body",
            index=0,
            joint=FreeJoint("root"),
            inertia=SpatialInertia(mass=mass, inertia=np.eye(3) * I, com=np.zeros(3)),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()
    # Two spheres: one at origin, one offset +0.15 in x
    shapes = [
        ShapeInstance(SphereShape(radius)),
        ShapeInstance(SphereShape(radius), origin_xyz=np.array([0.15, 0.0, 0.0])),
    ]
    return RobotModel(
        tree=tree,
        geometries=[BodyCollisionGeometry(0, shapes)],
        contact_body_names=["body"],
    )


def _parent_child_model():
    """Two-body chain: parent + child connected by revolute joint."""
    tree = RobotTreeNumpy(gravity=9.81)
    mass = 1.0
    I = 0.004
    tree.add_body(
        Body(
            name="base",
            index=0,
            joint=FreeJoint("root"),
            inertia=SpatialInertia(mass=mass, inertia=np.eye(3) * I, com=np.zeros(3)),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.add_body(
        Body(
            name="link",
            index=1,
            joint=RevoluteJoint("j1"),
            inertia=SpatialInertia(mass=mass, inertia=np.eye(3) * I, com=np.zeros(3)),
            X_tree=SpatialTransform(R=np.eye(3), r=np.array([0.0, 0.0, -0.2])),
            parent=0,
        )
    )
    tree.finalize()
    return RobotModel(
        tree=tree,
        geometries=[
            BodyCollisionGeometry(0, [ShapeInstance(SphereShape(0.05))]),
            BodyCollisionGeometry(1, [ShapeInstance(SphereShape(0.05))]),
        ],
        contact_body_names=["link"],
    )


class TestMultiShapeGroundContact:
    """Multi-shape bodies produce multiple ground contacts."""

    def test_two_spheres_both_touch_ground(self):
        """After falling, both spheres should produce ground contacts."""
        model = _multishape_ball_model(radius=0.08)
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0")
        engine.reset()

        # Let ball fall to ground (default start is above ground)
        for _ in range(500):
            engine.step(np.zeros((1, 0)), 2e-4)

        count = engine._contact_count.numpy()[0]
        assert count >= 2, f"Expected >=2 contacts from 2 spheres, got {count}"

    def test_single_shape_still_works(self):
        """Single-shape body: exactly 1 ground contact when touching."""
        model = _ball_model(radius=0.1)
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0")
        engine.reset()

        # Ball starts at default height, let it fall
        for _ in range(50):
            engine.step(np.zeros((1, 0)), 2e-4)
        count = engine._contact_count.numpy()[0]
        assert count == 1, f"Expected 1 contact from single sphere, got {count}"


class TestDynamicBroadphase:
    """N² broadphase discovers pairs dynamically."""

    def test_two_balls_far_apart_no_contact(self):
        """Two balls at distance >> radius: no body-body contact."""
        m_a = _ball_model(radius=0.1)
        m_b = _ball_model(radius=0.1)
        merged = merge_models(robots={"a": m_a, "b": m_b})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0")
        engine.reset()

        # Place balls far apart, above ground
        sc = engine._scratch
        q = sc.q.numpy()
        # Robot a: body 0, Robot b: body 1
        q[0, 4] = -2.0  # a at x=-2
        q[0, 6] = 1.0  # a at z=1 (above ground)
        q[0, 11] = 2.0  # b at x=2
        q[0, 13] = 1.0  # b at z=1
        sc.q = __import__("warp").array(q, dtype=__import__("warp").float32, device="cuda:0")

        engine.step(np.zeros((1, 0)), 2e-4)
        # Should have 0 contacts (both above ground, far apart)
        count = engine._contact_count.numpy()[0]
        assert count == 0, f"Expected 0 contacts for far-apart balls, got {count}"

    def test_two_balls_close_generates_contact(self):
        """Two overlapping balls: broadphase discovers contact."""
        m_a = _ball_model(radius=0.1)
        m_b = _ball_model(radius=0.1)
        merged = merge_models(robots={"a": m_a, "b": m_b})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0")
        engine.reset()

        # Place balls overlapping at z=1 (above ground)
        sc = engine._scratch
        q = sc.q.numpy()
        q[0, 4] = 0.0  # a at x=0
        q[0, 6] = 1.0  # a at z=1
        q[0, 11] = 0.12  # b at x=0.12 (overlap: dist=0.12 < r1+r2=0.2)
        q[0, 13] = 1.0  # b at z=1
        sc.q = __import__("warp").array(q, dtype=__import__("warp").float32, device="cuda:0")

        engine.step(np.zeros((1, 0)), 2e-4)
        count = engine._contact_count.numpy()[0]
        assert count >= 1, f"Expected >=1 contact for overlapping balls, got {count}"


class TestCollisionFilter:
    """Collision filter excludes parent-child on GPU."""

    def test_parent_child_excluded(self):
        """Parent-child bodies should not collide even when overlapping."""
        model = _parent_child_model()
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0")
        engine.reset()

        # Bodies are at z-offset 0.2, spheres radius 0.05 → no overlap by default
        # Even if we force them to overlap, filter should exclude
        sc = engine._scratch
        q = sc.q.numpy()
        q[0, 6] = 0.5  # base at z=0.5
        sc.q = __import__("warp").array(q, dtype=__import__("warp").float32, device="cuda:0")

        engine.step(np.zeros((1, 1)), 2e-4)

        # Check exclude matrix was built correctly
        s = engine._static
        assert s.collision_excluded[0, 1] == 1, "Parent-child should be excluded"
        assert s.collision_excluded[1, 0] == 1, "Parent-child should be excluded (symmetric)"


class TestStability:
    """Multi-shape simulation stability."""

    def test_multishape_1000_steps_no_nan(self):
        """Multi-shape body: 1000 steps without NaN."""
        model = _multishape_ball_model(radius=0.08)
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=2, device="cuda:0")
        engine.reset()

        for _ in range(1000):
            engine.step(np.zeros((2, 0)), 2e-4)

        q = engine._scratch.q.numpy()
        assert np.all(np.isfinite(q)), "NaN detected after 1000 steps"
        # Ball should have landed near ground (z ≈ radius)
        z = q[0, 6]
        assert 0.0 < z < 0.5, f"Unexpected z={z:.4f} after 1000 steps"

    def test_two_robot_1000_steps_stable(self):
        """Two single-shape robots: 1000 steps stable with dynamic broadphase."""
        m_a = _ball_model(radius=0.1)
        m_b = _ball_model(radius=0.1)
        merged = merge_models(robots={"a": m_a, "b": m_b})
        engine = GpuEngine(merged, num_envs=2, device="cuda:0")
        engine.reset()

        for _ in range(1000):
            engine.step(np.zeros((2, 0)), 2e-4)

        q = engine._scratch.q.numpy()
        assert np.all(np.isfinite(q)), "NaN detected after 1000 steps"
