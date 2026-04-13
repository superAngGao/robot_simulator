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


# ---------------------------------------------------------------------------
# CPU vs GPU agreement
# ---------------------------------------------------------------------------


def _two_body_model(shape_a, shape_b, z_a, z_b, y_sep=0.0):
    """Create a 2-body model with two shapes, both as contact bodies."""
    tree = RobotTreeNumpy(gravity=9.81)
    tree.add_body(Body("a", 0, FreeJoint("ja"), _I, SpatialTransform.identity(), -1))
    tree.add_body(Body("b", 1, FreeJoint("jb"), _I, SpatialTransform.identity(), -1))
    tree.finalize()
    geom_a = BodyCollisionGeometry(0, [ShapeInstance(shape_a)])
    geom_b = BodyCollisionGeometry(1, [ShapeInstance(shape_b)])
    model = RobotModel(
        tree=tree,
        geometries=[geom_a, geom_b],
        contact_body_names=["a", "b"],
    )
    merged = merge_models({"A": model}, terrain=FlatTerrain())
    q, qdot = merged.tree.default_state()
    # Body a: q[0:7], body b: q[7:14]
    q[6] = z_a
    q[5] = 0.0
    q[13] = z_b
    q[12] = y_sep
    return merged, q, qdot


class TestCpuGpuAgreement:
    """CPU (GJK/EPA) vs GPU (GJK + margin / vertex enum) agreement."""

    def test_hull_ground_depth_agrees(self):
        """ConvexHull-ground: CPU and GPU produce same max depth per body."""
        from physics.cpu_engine import CpuEngine

        hull = _box_as_hull(0.3, 0.3, 0.3)
        merged, q, qdot = _single_hull_model(hull, z=0.25)
        tau = np.zeros(merged.nv)

        # CPU
        cpu = CpuEngine(merged, dt=2e-4)
        cpu.step(q.copy(), qdot.copy(), tau, dt=2e-4)
        cpu_contacts = cpu.query_contacts()
        cpu_ground = [c for c in cpu_contacts if c.body_j < 0]

        # GPU
        gpu = GpuEngine(merged, num_envs=1, dt=2e-4)
        gpu.step(q.copy(), qdot.copy(), tau, dt=2e-4)
        gpu_contacts = gpu.query_contacts(env_idx=0)
        gpu_ground = [c for c in gpu_contacts if c.body_j < 0]

        assert len(cpu_ground) > 0, "CPU should detect ground contacts"
        assert len(gpu_ground) > 0, "GPU should detect ground contacts"

        cpu_max_depth = max(c.depth for c in cpu_ground)
        gpu_max_depth = max(c.depth for c in gpu_ground)
        np.testing.assert_allclose(
            gpu_max_depth,
            cpu_max_depth,
            atol=2e-3,
            err_msg="CPU vs GPU ground depth disagree",
        )

    def test_hull_sphere_body_body_detected(self):
        """ConvexHull vs Sphere body-body: both engines detect contact."""
        from physics.cpu_engine import CpuEngine
        from physics.geometry import SphereShape

        hull = _box_as_hull(0.2, 0.2, 0.2)
        sphere = SphereShape(0.15)
        # Place them overlapping: hull center at z=0.5, sphere at z=0.5, y_sep=0.3
        # Hull extends ±0.2 in Y, sphere radius 0.15 → overlap = 0.2+0.15-0.3 = 0.05
        merged, q, qdot = _two_body_model(hull, sphere, z_a=0.5, z_b=0.5, y_sep=0.3)
        tau = np.zeros(merged.nv)

        # CPU
        cpu = CpuEngine(merged, dt=2e-4)
        cpu.step(q.copy(), qdot.copy(), tau, dt=2e-4)
        cpu_bb = [c for c in cpu.query_contacts() if c.body_j >= 0]

        # GPU
        gpu = GpuEngine(merged, num_envs=1, dt=2e-4)
        gpu.step(q.copy(), qdot.copy(), tau, dt=2e-4)
        gpu_bb = [c for c in gpu.query_contacts(env_idx=0) if c.body_j >= 0]

        assert len(cpu_bb) > 0, "CPU should detect hull-sphere body-body"
        assert len(gpu_bb) > 0, "GPU should detect hull-sphere body-body"

        # Depths should be in the same ballpark (GPU uses margin, CPU uses EPA)
        cpu_depth = max(c.depth for c in cpu_bb)
        gpu_depth = max(c.depth for c in gpu_bb)
        # GPU depth is capped at CONVEX_MARGIN (1e-3) for non-deep penetration
        # so we just check both detected the contact
        assert cpu_depth > 0
        assert gpu_depth > 0

    def test_hull_hull_body_body_detected(self):
        """ConvexHull vs ConvexHull body-body: both engines detect contact."""
        from physics.cpu_engine import CpuEngine

        hull_a = _box_as_hull(0.2, 0.2, 0.2)
        hull_b = _box_as_hull(0.15, 0.15, 0.15)
        # Overlapping in Y: hull_a ±0.2, hull_b ±0.15, sep=0.3 → overlap = 0.05
        merged, q, qdot = _two_body_model(hull_a, hull_b, z_a=0.5, z_b=0.5, y_sep=0.3)
        tau = np.zeros(merged.nv)

        cpu = CpuEngine(merged, dt=2e-4)
        cpu.step(q.copy(), qdot.copy(), tau, dt=2e-4)
        cpu_bb = [c for c in cpu.query_contacts() if c.body_j >= 0]

        gpu = GpuEngine(merged, num_envs=1, dt=2e-4)
        gpu.step(q.copy(), qdot.copy(), tau, dt=2e-4)
        gpu_bb = [c for c in gpu.query_contacts(env_idx=0) if c.body_j >= 0]

        assert len(cpu_bb) > 0, "CPU should detect hull-hull body-body"
        assert len(gpu_bb) > 0, "GPU should detect hull-hull body-body"

    def test_separated_hull_no_contact(self):
        """Well-separated ConvexHulls: neither engine detects contact."""
        from physics.cpu_engine import CpuEngine

        hull_a = _box_as_hull(0.1, 0.1, 0.1)
        hull_b = _box_as_hull(0.1, 0.1, 0.1)
        # Far apart: y_sep=1.0 >> 2*0.1
        merged, q, qdot = _two_body_model(hull_a, hull_b, z_a=0.5, z_b=0.5, y_sep=1.0)
        tau = np.zeros(merged.nv)

        cpu = CpuEngine(merged, dt=2e-4)
        cpu.step(q.copy(), qdot.copy(), tau, dt=2e-4)
        cpu_bb = [c for c in cpu.query_contacts() if c.body_j >= 0]

        gpu = GpuEngine(merged, num_envs=1, dt=2e-4)
        gpu.step(q.copy(), qdot.copy(), tau, dt=2e-4)
        gpu_bb = [c for c in gpu.query_contacts(env_idx=0) if c.body_j >= 0]

        assert len(cpu_bb) == 0, f"CPU should not detect contact, got {len(cpu_bb)}"
        assert len(gpu_bb) == 0, f"GPU should not detect contact, got {len(gpu_bb)}"
