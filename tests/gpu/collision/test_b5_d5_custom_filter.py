"""
B(5) Step 4 — D5 custom CollisionFilter on top of parent-child exclusion.

Fixture: 3R × 3B × 2S sphere (same topology as Step 3, fresh positions).
Adds a CollisionFilter that:
  1. auto_exclude_adjacent (parent-child, same as implicit default)
  2. exclude_pair(A1, B1) — remove one specific cross-robot body-pair
  3. group/mask bits: disable all A↔C collisions via bitmask

Layout (chain along +Z this time, separation in X):
    Robot A at X=0, Robot B at X=0.10, Robot C at X=0.25
    Chain along +Z: bodies at Z=0.4, 0.6, 0.8.
    Each body: 2 spheres r=0.06, offset ±0.05 in Y.

Contacts without filter (Step 3 style):
    A↔B: X_sep=0.10 < 2r=0.12 → 3 body-pairs contact, 2 shapes each = 6
    B↔C: X_sep=0.15 > 2r=0.12 → 0 (too far)
    A↔C: X_sep=0.25 >> 2r → 0

With filter:
    - exclude_pair(A1=1, B1=4): removes 1 body-pair → lose 2 shape contacts
    - A↔C bitmask disabled: removes 9 body-pairs from candidate list
    Contacts: A↔B=4 (was 6, minus 2 from excluded A1↔B1), B↔C=0, A↔C=0

Pair count changes:
    Without filter: 3 intra + 27 cross = 30
    With filter: 3 intra + (9-1) A↔B + 9 B↔C + 0 A↔C = 3+8+9+0 = 20

Reference: OPEN_QUESTIONS Q36, session 25.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from physics.collision_filter import CollisionFilter
from physics.geometry import BodyCollisionGeometry, ShapeInstance, SphereShape
from physics.joint import FreeJoint, RevoluteJoint
from physics.merged_model import merge_models
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from robot.model import RobotModel

try:
    from physics.backends.static_data import StaticRobotData
    from physics.gpu_engine import GpuEngine

    HAS_WARP = True
except Exception:
    HAS_WARP = False

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not HAS_WARP, reason="Warp or CUDA not available"),
]

# ---------------------------------------------------------------------------
# Fixture constants (fresh positions)
# ---------------------------------------------------------------------------

RADIUS = 0.06
SHAPE_Y_OFFSET = 0.05  # ±offset in Y per body
LINK_LEN = 0.20  # X_tree translation along +Z
X_A = 0.00
X_B = 0.10
X_C = 0.25
Z_BASE = 0.40  # lowest body Z
MASS = 1.0
GRAVITY = 9.81

N_ROBOTS = 3
N_BODIES_PER_ROBOT = 3
N_BODIES_TOTAL = 9
N_SHAPES_PER_BODY = 2

# Body indices: A=[0,1,2], B=[3,4,5], C=[6,7,8]
A0, A1, A2 = 0, 1, 2
B0, B1, B2 = 3, 4, 5
C0, C1, C2 = 6, 7, 8

# Pair counts WITH filter:
#   Intra: 3 (body 0↔2 per robot, parent-child excluded by auto_exclude_adjacent)
#   Cross A↔B: 9 - 1 (exclude A1↔B1) = 8
#   Cross B↔C: 9 (filter doesn't touch B↔C)
#   Cross A↔C: 0 (bitmask disables entirely)
N_TOTAL_PAIRS = 3 + 8 + 9 + 0  # = 20

# Contact counts WITH filter:
#   A↔B: 3 same-Z body-pairs (A0↔B0, A1↔B1, A2↔B2) × 2 shapes each = 6
#         minus A1↔B1 excluded → 6 - 2 = 4
#   B↔C: X_sep=0.15 > 0.12 → 0
#   A↔C: filtered out entirely → 0
N_ACTUAL_CONTACTS = 4


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def _chain_robot_z() -> RobotModel:
    """3-body chain along +Z, 2 spheres per body (±Y offset)."""
    tree = RobotTreeNumpy(gravity=GRAVITY)
    I_sphere = 2.0 / 5.0 * MASS * RADIUS**2
    inertia = SpatialInertia(mass=MASS, inertia=np.eye(3) * I_sphere, com=np.zeros(3))

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
    tree.add_body(
        Body(
            name="link1",
            index=1,
            joint=RevoluteJoint("j1", axis=np.array([1, 0, 0])),
            inertia=inertia,
            X_tree=SpatialTransform(R=np.eye(3), r=np.array([0, 0, LINK_LEN])),
            parent=0,
        )
    )
    tree.add_body(
        Body(
            name="link2",
            index=2,
            joint=RevoluteJoint("j2", axis=np.array([1, 0, 0])),
            inertia=inertia,
            X_tree=SpatialTransform(R=np.eye(3), r=np.array([0, 0, LINK_LEN])),
            parent=1,
        )
    )
    tree.finalize()

    geometries = []
    for i in range(3):
        shapes = [
            ShapeInstance(
                SphereShape(RADIUS),
                origin_xyz=np.array([0, +SHAPE_Y_OFFSET, 0]),
            ),
            ShapeInstance(
                SphereShape(RADIUS),
                origin_xyz=np.array([0, -SHAPE_Y_OFFSET, 0]),
            ),
        ]
        geometries.append(BodyCollisionGeometry(i, shapes))

    return RobotModel(
        tree=tree,
        geometries=geometries,
        contact_body_names=["link0", "link1", "link2"],
    )


def _build_filter() -> CollisionFilter:
    """Build a CollisionFilter for 9 bodies with custom rules."""
    cf = CollisionFilter(num_bodies=N_BODIES_TOTAL)

    # 1. Parent-child exclusion (same as default merge_models behavior)
    # Parent list for merged tree: 3 robots each with chain 0→1→2
    parent_list = [
        -1,
        0,
        1,  # Robot A
        -1,
        3,
        4,  # Robot B
        -1,
        6,
        7,  # Robot C
    ]
    cf.auto_exclude_adjacent(parent_list)

    # 2. Explicit pair exclusion: remove A1↔B1
    cf.exclude_pair(A1, B1)

    # 3. Bitmask: disable all A↔C collisions
    # A bodies → group=0x01, mask=0x01 (only see group bit 0)
    # B bodies → group=0x03, mask=0x03 (see both bits → collide with A and C)
    # C bodies → group=0x02, mask=0x02 (only see group bit 1)
    # A↔C: group_A(0x01) & mask_C(0x02) == 0 → blocked
    # A↔B: group_A(0x01) & mask_B(0x03) == 0x01 → allowed
    # B↔C: group_B(0x03) & mask_C(0x02) == 0x02 → allowed
    for i in [A0, A1, A2]:
        cf.set_group_mask(i, group=0x01, mask=0x01)
    for i in [B0, B1, B2]:
        cf.set_group_mask(i, group=0x03, mask=0x03)
    for i in [C0, C1, C2]:
        cf.set_group_mask(i, group=0x02, mask=0x02)

    return cf


def _build_merged():
    """Merge three chain robots with custom CollisionFilter."""
    cf = _build_filter()
    return merge_models(
        robots={"A": _chain_robot_z(), "B": _chain_robot_z(), "C": _chain_robot_z()},
        collision_filter=cf,
    )


def _init_state(merged):
    """A at X=0, B at X=0.10, C at X=0.25. Roots at Z=Z_BASE."""
    q, qdot = merged.tree.default_state()
    for name, x_pos in [("A", X_A), ("B", X_B), ("C", X_C)]:
        rs = merged.robot_slices[name]
        qa = q[rs.q_slice]
        qa[4] = x_pos  # px
        qa[5] = 0.0  # py
        qa[6] = Z_BASE  # pz
    return q, qdot


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStep4CustomFilter:
    """D5 custom filter: explicit exclude + bitmask on 3R × 3B × 2S."""

    def test_collision_pair_count(self):
        """20 pairs after filter (was 30 without): 9 A↔C removed, 1 A1↔B1 removed."""
        merged = _build_merged()
        assert len(merged.collision_pairs) == N_TOTAL_PAIRS, (
            f"Expected {N_TOTAL_PAIRS} pairs, got {len(merged.collision_pairs)}"
        )

    def test_excluded_pair_absent(self):
        """A1↔B1 (1,4) must not appear in collision_pairs."""
        merged = _build_merged()
        pairs = set(merged.collision_pairs)
        assert (A1, B1) not in pairs, "Explicitly excluded pair (1,4) is present"
        assert (B1, A1) not in pairs, "Explicitly excluded pair (4,1) is present"

    def test_ac_pairs_absent(self):
        """All A↔C body-pairs must be absent (bitmask disabled)."""
        merged = _build_merged()
        pairs = set(merged.collision_pairs)
        for ai in [A0, A1, A2]:
            for ci in [C0, C1, C2]:
                pair = (min(ai, ci), max(ai, ci))
                assert pair not in pairs, f"A↔C pair {pair} should be filtered"

    def test_gpu_collision_excluded_agreement(self):
        """GPU exclude matrix must match CPU pair set with filter applied."""
        merged = _build_merged()
        static = StaticRobotData.from_merged(merged)
        excl = static.collision_excluded

        gpu_pairs = set()
        for i in range(N_BODIES_TOTAL):
            for j in range(i + 1, N_BODIES_TOTAL):
                if excl[i, j] == 0:
                    gpu_pairs.add((i, j))

        cpu_pairs = set(merged.collision_pairs)
        assert gpu_pairs == cpu_pairs, (
            f"CPU/GPU pair mismatch with filter.\n"
            f"  CPU only: {cpu_pairs - gpu_pairs}\n"
            f"  GPU only: {gpu_pairs - cpu_pairs}"
        )

    def test_narrowphase_contact_count(self):
        """4 GPU contacts: A↔B minus excluded A1↔B1 = 4."""
        merged = _build_merged()
        q, qdot = _init_state(merged)
        tau = np.zeros(merged.nv)
        dt = 2e-4

        gpu = GpuEngine(merged, num_envs=1, dt=dt)
        gpu.step(q.copy(), qdot.copy(), tau, dt=dt)
        gpu_bb = [c for c in gpu.query_contacts(env_idx=0) if c.body_j >= 0]

        assert len(gpu_bb) == N_ACTUAL_CONTACTS, (
            f"GPU contacts: expected {N_ACTUAL_CONTACTS}, got {len(gpu_bb)}"
        )

    def test_contact_body_pairs_identity(self):
        """Contacts from A0↔B0(2) and A2↔B2(2) only. No A1↔B1, no A↔C."""
        merged = _build_merged()
        q, qdot = _init_state(merged)
        tau = np.zeros(merged.nv)
        dt = 2e-4

        gpu = GpuEngine(merged, num_envs=1, dt=dt)
        gpu.step(q.copy(), qdot.copy(), tau, dt=dt)
        gpu_bb = [c for c in gpu.query_contacts(env_idx=0) if c.body_j >= 0]

        pair_counts = Counter()
        for c in gpu_bb:
            pair = (min(c.body_i, c.body_j), max(c.body_i, c.body_j))
            pair_counts[pair] += 1

        expected = {(A0, B0): 2, (A2, B2): 2}
        assert dict(pair_counts) == expected, (
            f"Contact pair distribution mismatch.\n  Expected: {expected}\n  Got: {dict(pair_counts)}"
        )
