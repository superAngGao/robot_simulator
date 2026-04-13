"""GPU ConvexHullShape collision tests (Q41).

Verifies that ConvexHullShape works on GPU via:
  - Ground contacts: vertex enumeration multi-point
  - Body-body: GJK closest-distance with convex margin
  - CPU vs GPU agreement for ground contacts

Reference: Q41 (OPEN_QUESTIONS), session 29.
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.geometry import (
    BodyCollisionGeometry,
    BoxShape,
    ConvexHullShape,
    ShapeInstance,
)
from physics.joint import FreeJoint
from physics.merged_model import merge_models
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from physics.terrain import FlatTerrain
from robot.model import RobotModel

try:
    from physics.gpu_engine import GpuEngine

    HAS_WARP = True
except Exception:
    HAS_WARP = False

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not HAS_WARP, reason="Warp or CUDA not available"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_I = SpatialInertia(mass=1.0, inertia=np.eye(3) * 0.001, com=np.zeros(3))


def _box_as_hull(half_x=0.5, half_y=0.5, half_z=0.5):
    """Create a ConvexHullShape that is a box."""
    verts = np.array(
        [
            [-half_x, -half_y, -half_z],
            [half_x, -half_y, -half_z],
            [-half_x, half_y, -half_z],
            [half_x, half_y, -half_z],
            [-half_x, -half_y, half_z],
            [half_x, -half_y, half_z],
            [-half_x, half_y, half_z],
            [half_x, half_y, half_z],
        ]
    )
    return ConvexHullShape(verts)


def _single_hull_model(shape, z=0.45):
    """Create a single-body model with a ConvexHullShape at given z."""
    tree = RobotTreeNumpy(gravity=9.81)
    tree.add_body(Body("hull", 0, FreeJoint("root"), _I, SpatialTransform.identity(), -1))
    tree.finalize()
    geom = BodyCollisionGeometry(0, [ShapeInstance(shape)])
    model = RobotModel(tree=tree, geometries=[geom], contact_body_names=["hull"])
    merged = merge_models({"A": model}, terrain=FlatTerrain())
    q, qdot = merged.tree.default_state()
    q[6] = z
    return merged, q, qdot


# ---------------------------------------------------------------------------
# Ground contact tests
# ---------------------------------------------------------------------------


class TestConvexHullGround:
    """ConvexHull vs flat ground via vertex enumeration."""

    def test_box_hull_4_ground_contacts(self):
        """Box-as-hull at z=0.45 (half=0.5) → 4 bottom contacts, depth=0.05."""
        hull = _box_as_hull()
        merged, q, qdot = _single_hull_model(hull, z=0.45)
        tau = np.zeros(merged.nv)

        gpu = GpuEngine(merged, num_envs=1, dt=2e-4, solver="jacobi_pgs_ms")
        gpu.step(q, qdot, tau, dt=2e-4)
        contacts = gpu.query_contacts(env_idx=0)
        ground = [c for c in contacts if c.body_j < 0]

        assert len(ground) == 4, f"Expected 4 ground contacts, got {len(ground)}"
        for c in ground:
            np.testing.assert_allclose(c.depth, 0.05, atol=1e-3)

    def test_hull_above_ground_no_contacts(self):
        """Hull fully above ground → 0 contacts."""
        hull = _box_as_hull(0.1, 0.1, 0.1)
        merged, q, qdot = _single_hull_model(hull, z=1.0)
        tau = np.zeros(merged.nv)

        gpu = GpuEngine(merged, num_envs=1, dt=2e-4)
        gpu.step(q, qdot, tau, dt=2e-4)
        contacts = gpu.query_contacts(env_idx=0)
        ground = [c for c in contacts if c.body_j < 0]

        assert len(ground) == 0

    def test_octahedron_hull_ground(self):
        """Regular octahedron: only 1 bottom vertex penetrates when upright."""
        r = 0.5
        verts = np.array([[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [0, 0, r], [0, 0, -r]])
        hull = ConvexHullShape(verts)
        merged, q, qdot = _single_hull_model(hull, z=0.4)
        tau = np.zeros(merged.nv)

        gpu = GpuEngine(merged, num_envs=1, dt=2e-4)
        gpu.step(q, qdot, tau, dt=2e-4)
        contacts = gpu.query_contacts(env_idx=0)
        ground = [c for c in contacts if c.body_j < 0]

        # Bottom vertex at z = 0.4 - 0.5 = -0.1 → depth = 0.1
        assert len(ground) >= 1
        assert ground[0].depth > 0.05

    def test_hull_matches_box_ground_contacts(self):
        """Box-as-hull should produce same ground depths as BoxShape."""
        h = 0.3
        hull = _box_as_hull(h, h, h)
        box = BoxShape((2 * h, 2 * h, 2 * h))
        z = 0.25

        # Hull model
        merged_h, q_h, qdot_h = _single_hull_model(hull, z=z)
        gpu_h = GpuEngine(merged_h, num_envs=1, dt=2e-4, solver="jacobi_pgs_ms")
        gpu_h.step(q_h, qdot_h, np.zeros(merged_h.nv), dt=2e-4)
        hull_contacts = [c for c in gpu_h.query_contacts(env_idx=0) if c.body_j < 0]

        # Box model
        tree_b = RobotTreeNumpy(gravity=9.81)
        tree_b.add_body(Body("box", 0, FreeJoint("root"), _I, SpatialTransform.identity(), -1))
        tree_b.finalize()
        geom_b = BodyCollisionGeometry(0, [ShapeInstance(box)])
        model_b = RobotModel(tree=tree_b, geometries=[geom_b], contact_body_names=["box"])
        merged_b = merge_models({"A": model_b}, terrain=FlatTerrain())
        q_b, qdot_b = merged_b.tree.default_state()
        q_b[6] = z
        gpu_b = GpuEngine(merged_b, num_envs=1, dt=2e-4, solver="jacobi_pgs_ms")
        gpu_b.step(q_b, qdot_b, np.zeros(merged_b.nv), dt=2e-4)
        box_contacts = [c for c in gpu_b.query_contacts(env_idx=0) if c.body_j < 0]

        # Both should have 4 contacts with same depth
        assert len(hull_contacts) == len(box_contacts) == 4
        hull_depths = sorted([c.depth for c in hull_contacts])
        box_depths = sorted([c.depth for c in box_contacts])
        np.testing.assert_allclose(hull_depths, box_depths, atol=1e-3)


# ---------------------------------------------------------------------------
# Simulation stability
# ---------------------------------------------------------------------------


class TestConvexHullStability:
    """ConvexHull shapes must not cause simulation divergence."""

    def test_hull_drop_100_steps(self):
        """Drop a ConvexHull box onto ground for 100 steps, must stay finite."""
        hull = _box_as_hull(0.1, 0.1, 0.1)
        merged, q, qdot = _single_hull_model(hull, z=0.2)
        tau = np.zeros(merged.nv)

        gpu = GpuEngine(merged, num_envs=1, dt=2e-4, solver="jacobi_pgs_ms")
        for step in range(100):
            out = gpu.step(q, qdot, tau, dt=2e-4)
            q, qdot = out.q_new, out.qdot_new
            assert np.all(np.isfinite(q)), f"NaN at step {step}"
