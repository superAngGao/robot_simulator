"""
B(5) Step 1 — D2 multi-body cross-robot collision.

Fixture: 2 robots × 3 bodies (FreeJoint root + 2 RevoluteJoint chain) × 1 sphere.
Verifies that merged-model indexing, collision pair construction, and
narrowphase contact detection are correct when multiple multi-body robots
coexist in the same scene.

Layout (Z=0.5 plane, top view):
    Robot A: bodies at (0,0), (0.3,0), (0.6,0)       — Y=0
    Robot B: bodies at (0,0.15), (0.3,0.15), (0.6,0.15) — Y=0.15
    All spheres r=0.1.  Same-X pairs overlap (gap = -0.05).

Expected collision pairs:
    Intra: 2 (body 0–2 per robot; parent-child 0–1, 1–2 excluded)
    Cross: 9 (3×3 all-pairs)
    Total: 11 candidate pairs, 3 actual contacts (A0↔B0, A1↔B1, A2↔B2)

Reference: OPEN_QUESTIONS Q36, session 25.
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.geometry import BodyCollisionGeometry, ShapeInstance, SphereShape
from physics.joint import FreeJoint, RevoluteJoint
from physics.merged_model import merge_models
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from robot.model import RobotModel

try:
    from physics.backends.static_data import StaticRobotData
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
# Fixture constants
# ---------------------------------------------------------------------------

RADIUS = 0.1
LINK_LEN = 0.3  # X_tree translation between bodies
Y_SEP = 0.15  # Y separation between robot A and B roots
Z_HEIGHT = 0.5  # above ground, no ground contact
MASS = 1.0
GRAVITY = 9.81

# Expected values (hand-computed)
N_ROBOTS = 2
N_BODIES_PER_ROBOT = 3
N_BODIES_TOTAL = N_ROBOTS * N_BODIES_PER_ROBOT  # 6
N_INTRA_PAIRS = 2  # body(0,2) per robot, parent-child skipped
N_CROSS_PAIRS = 9  # 3 × 3
N_TOTAL_PAIRS = N_INTRA_PAIRS + N_CROSS_PAIRS  # 11
N_ACTUAL_CONTACTS = 3  # same-X pairs overlap


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def _chain_robot(name_prefix: str = "") -> RobotModel:
    """Build a 3-body chain: FreeJoint root → Revolute → Revolute, 1 sphere each."""
    tree = RobotTreeNumpy(gravity=GRAVITY)
    I_sphere = 2.0 / 5.0 * MASS * RADIUS**2
    inertia = SpatialInertia(mass=MASS, inertia=np.eye(3) * I_sphere, com=np.zeros(3))

    # Body 0: root with FreeJoint
    tree.add_body(
        Body(
            name="link0",
            index=0,
            joint=FreeJoint("root"),
            inertia=inertia,
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    # Body 1: child of 0, RevoluteJoint(Y)
    tree.add_body(
        Body(
            name="link1",
            index=1,
            joint=RevoluteJoint("j1", axis=np.array([0, 1, 0])),
            inertia=inertia,
            X_tree=SpatialTransform(R=np.eye(3), r=np.array([LINK_LEN, 0, 0])),
            parent=0,
        )
    )
    # Body 2: child of 1, RevoluteJoint(Y)
    tree.add_body(
        Body(
            name="link2",
            index=2,
            joint=RevoluteJoint("j2", axis=np.array([0, 1, 0])),
            inertia=inertia,
            X_tree=SpatialTransform(R=np.eye(3), r=np.array([LINK_LEN, 0, 0])),
            parent=1,
        )
    )
    tree.finalize()

    geometries = [BodyCollisionGeometry(i, [ShapeInstance(SphereShape(RADIUS))]) for i in range(3)]
    return RobotModel(
        tree=tree,
        geometries=geometries,
        contact_body_names=["link0", "link1", "link2"],
    )


def _build_merged():
    """Merge two chain robots into a single scene (no terrain contact)."""
    return merge_models(robots={"A": _chain_robot(), "B": _chain_robot()})


def _init_state(merged):
    """Set initial q so that robot A is at Y=0, robot B at Y=0.15, both at Z=0.5.

    FreeJoint q layout: [qw, qx, qy, qz, px, py, pz].
    RevoluteJoint q = 0 (straight chain along +X).
    """
    q, qdot = merged.tree.default_state()
    rs_a = merged.robot_slices["A"]
    rs_b = merged.robot_slices["B"]

    # Robot A root: (0, 0, Z_HEIGHT)
    qa = q[rs_a.q_slice]
    qa[4] = 0.0  # px
    qa[5] = 0.0  # py
    qa[6] = Z_HEIGHT  # pz

    # Robot B root: (0, Y_SEP, Z_HEIGHT)
    qb = q[rs_b.q_slice]
    qb[4] = 0.0  # px
    qb[5] = Y_SEP  # py
    qb[6] = Z_HEIGHT  # pz

    return q, qdot


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStep1MultiBodyCrossRobot:
    """D2 multi-body: structural + geometric assertions for 2R × 3B × 1S."""

    def test_collision_pair_count(self):
        """merged.collision_pairs must have exactly 11 entries."""
        merged = _build_merged()
        assert len(merged.collision_pairs) == N_TOTAL_PAIRS, (
            f"Expected {N_TOTAL_PAIRS} collision pairs, got {len(merged.collision_pairs)}"
        )

    def test_collision_pair_composition(self):
        """Intra-robot pairs skip parent-child; cross-robot includes all 3×3."""
        merged = _build_merged()
        pairs = set(merged.collision_pairs)

        # Body index mapping: A=[0,1,2], B=[3,4,5]
        a0, a1, a2 = 0, 1, 2
        b0, b1, b2 = 3, 4, 5

        # Intra-robot: only (0,2) and (3,5) — parent-child excluded
        assert (a0, a2) in pairs, "Intra-A (0,2) missing"
        assert (b0, b2) in pairs, "Intra-B (3,5) missing"
        # Parent-child must NOT appear
        assert (a0, a1) not in pairs, "Parent-child (0,1) should be excluded"
        assert (a1, a2) not in pairs, "Parent-child (1,2) should be excluded"
        assert (b0, b1) not in pairs, "Parent-child (3,4) should be excluded"
        assert (b1, b2) not in pairs, "Parent-child (4,5) should be excluded"

        # Cross-robot: all 9 pairs (ordered i < j since A bodies < B bodies)
        expected_cross = {
            (a0, b0),
            (a0, b1),
            (a0, b2),
            (a1, b0),
            (a1, b1),
            (a1, b2),
            (a2, b0),
            (a2, b1),
            (a2, b2),
        }
        actual_cross = {p for p in pairs if p[0] < 3 and p[1] >= 3}
        assert actual_cross == expected_cross, (
            f"Cross-robot pairs mismatch.\n"
            f"  Missing: {expected_cross - actual_cross}\n"
            f"  Extra:   {actual_cross - expected_cross}"
        )

    def test_gpu_body_shape_adr(self):
        """GPU static data body_shape_adr/num arrays must be [0,1,2,3,4,5]/[1,1,1,1,1,1]."""
        merged = _build_merged()
        static = StaticRobotData.from_merged(merged)

        expected_adr = np.arange(N_BODIES_TOTAL, dtype=np.int32)
        expected_num = np.ones(N_BODIES_TOTAL, dtype=np.int32)
        np.testing.assert_array_equal(static.body_shape_adr, expected_adr, err_msg="body_shape_adr mismatch")
        np.testing.assert_array_equal(static.body_shape_num, expected_num, err_msg="body_shape_num mismatch")

    def test_gpu_collision_excluded_matrix(self):
        """GPU collision_excluded must match CPU collision_pairs (session 23 Bug #1 guard).

        Both paths independently construct the pair set; they must agree.
        """
        merged = _build_merged()
        static = StaticRobotData.from_merged(merged)
        excl = static.collision_excluded  # (nb, nb)

        # Build effective GPU pair set from exclude matrix
        nb = N_BODIES_TOTAL
        gpu_pairs = set()
        for i in range(nb):
            for j in range(i + 1, nb):
                if excl[i, j] == 0:
                    gpu_pairs.add((i, j))

        cpu_pairs = set(merged.collision_pairs)
        assert gpu_pairs == cpu_pairs, (
            f"CPU/GPU pair set mismatch.\n"
            f"  CPU only: {cpu_pairs - gpu_pairs}\n"
            f"  GPU only: {gpu_pairs - cpu_pairs}"
        )

    def test_narrowphase_contact_count(self):
        """Exactly 3 contacts (same-X overlapping pairs), consistent CPU vs GPU."""
        merged = _build_merged()
        q, qdot = _init_state(merged)
        tau = np.zeros(merged.nv)
        dt = 2e-4

        # --- CPU contact count ---
        cpu = CpuEngine(merged, dt=dt)
        from physics.dynamics_cache import DynamicsCache

        cache = DynamicsCache.from_tree(merged.tree, q, qdot, dt)
        cpu_contacts = cpu._detect_contacts(cache)
        cpu_bb = [c for c in cpu_contacts if c.body_j >= 0]
        n_cpu = len(cpu_bb)

        # --- GPU contact count (via public query_contacts API) ---
        gpu = GpuEngine(merged, num_envs=1, dt=dt)
        gpu.step(q.copy(), qdot.copy(), tau, dt=dt)
        gpu_contacts = gpu.query_contacts(env_idx=0)
        gpu_bb = [c for c in gpu_contacts if c.body_j >= 0]
        gpu_bb_count = len(gpu_bb)

        assert n_cpu == N_ACTUAL_CONTACTS, (
            f"CPU body-body contacts: expected {N_ACTUAL_CONTACTS}, got {n_cpu}"
        )
        assert gpu_bb_count == N_ACTUAL_CONTACTS, (
            f"GPU body-body contacts: expected {N_ACTUAL_CONTACTS}, got {gpu_bb_count}"
        )
