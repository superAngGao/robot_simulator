"""
B(5) Step 3 — D1 third robot: 3R × 3B × 2S sphere.

Fixture: 3 robots × 3 bodies (chain along +X) × 2 spheres (±Z offset).
Robots arranged in Y: A(Y=0), B(Y=0.10), C(Y=0.20).

Layout (top view, XY plane):

    Y=0.20  ── (C0)────(C1)────(C2)      Robot C
                 │        │        │
    Y=0.10  ── (B0)────(B1)────(B2)      Robot B      ← B touches A and C
                 │        │        │
    Y=0.00  ── (A0)────(A1)────(A2)      Robot A
               X=0     X=0.25   X=0.50

    Each body: 2 spheres r=0.06, offset ±0.05 in Z.
    Y_sep = 0.10, 2r = 0.12 → same-level shape pairs overlap.
    A↔C Y_sep = 0.20 > 2r → no contact.

Expected:
    bodies: 9 (3×3), shapes: 18 (9×2)
    collision pairs: 30 (3 intra + 27 cross)
    contacts: 12 (A↔B: 6, B↔C: 6, A↔C: 0)

Discriminator: A↔C has candidate pairs but zero contacts. If pair loop
skips one robot-pair (e.g. off-by-one in triple loop) → contacts ≠ 12.

Reference: OPEN_QUESTIONS Q36, session 25.
"""

from __future__ import annotations

from collections import Counter

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
    from physics.gpu_engine import GpuEngine

    HAS_WARP = True
except Exception:
    HAS_WARP = False

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not HAS_WARP, reason="Warp or CUDA not available"),
]

# ---------------------------------------------------------------------------
# Fixture constants (all fresh)
# ---------------------------------------------------------------------------

RADIUS = 0.06
SHAPE_Z_OFFSET = 0.05  # ±offset in Z per body
LINK_LEN = 0.25  # X_tree translation along +X
Y_A = 0.00
Y_B = 0.10
Y_C = 0.20
Z_HEIGHT = 0.50
MASS = 1.0
GRAVITY = 9.81

N_ROBOTS = 3
N_BODIES_PER_ROBOT = 3
N_BODIES_TOTAL = 9
N_SHAPES_PER_BODY = 2
N_SHAPES_TOTAL = 18
# Intra: 3 robots × 1 pair (body 0↔2, skip parent-child) = 3
N_INTRA_PAIRS = 3
# Cross: C(3,2)=3 robot-pairs × 3×3=9 body-pairs each = 27
N_CROSS_PAIRS = 27
N_TOTAL_PAIRS = N_INTRA_PAIRS + N_CROSS_PAIRS  # 30
# A↔B: 3 same-X body-pairs × 2 same-level shape contacts = 6
# B↔C: same = 6
# A↔C: Y_sep=0.20 > 2r=0.12 → 0
N_CONTACTS_AB = 6
N_CONTACTS_BC = 6
N_CONTACTS_AC = 0
N_ACTUAL_CONTACTS = N_CONTACTS_AB + N_CONTACTS_BC + N_CONTACTS_AC  # 12


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def _chain_robot_x() -> RobotModel:
    """3-body chain along +X, 2 spheres per body (±Z offset)."""
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
            joint=RevoluteJoint("j1", axis=np.array([0, 1, 0])),
            inertia=inertia,
            X_tree=SpatialTransform(R=np.eye(3), r=np.array([LINK_LEN, 0, 0])),
            parent=0,
        )
    )
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

    geometries = []
    for i in range(3):
        shapes = [
            ShapeInstance(
                SphereShape(RADIUS),
                origin_xyz=np.array([0, 0, +SHAPE_Z_OFFSET]),
            ),
            ShapeInstance(
                SphereShape(RADIUS),
                origin_xyz=np.array([0, 0, -SHAPE_Z_OFFSET]),
            ),
        ]
        geometries.append(BodyCollisionGeometry(i, shapes))

    return RobotModel(
        tree=tree,
        geometries=geometries,
        contact_body_names=["link0", "link1", "link2"],
    )


def _build_merged():
    """Merge three chain robots."""
    return merge_models(robots={"A": _chain_robot_x(), "B": _chain_robot_x(), "C": _chain_robot_x()})


def _init_state(merged):
    """A at Y=0, B at Y=0.10, C at Y=0.20. All at Z=0.50. Joints q=0."""
    q, qdot = merged.tree.default_state()
    for name, y_pos in [("A", Y_A), ("B", Y_B), ("C", Y_C)]:
        rs = merged.robot_slices[name]
        qa = q[rs.q_slice]
        # FreeJoint: [qw, qx, qy, qz, px, py, pz]
        qa[4] = 0.0  # px
        qa[5] = y_pos  # py
        qa[6] = Z_HEIGHT  # pz
    return q, qdot


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStep3ThreeRobots:
    """D1 third robot: pair construction + contact count for 3R × 3B × 2S."""

    def test_collision_pair_count(self):
        """30 collision pairs: 3 intra + 27 cross."""
        merged = _build_merged()
        assert len(merged.collision_pairs) == N_TOTAL_PAIRS

    def test_cross_robot_pair_completeness(self):
        """All 3 robot-pairs (A↔B, A↔C, B↔C) must each contribute 9 body-pairs."""
        merged = _build_merged()
        pairs = set(merged.collision_pairs)

        # Body indices: A=[0,1,2], B=[3,4,5], C=[6,7,8]
        robot_ranges = {"A": range(0, 3), "B": range(3, 6), "C": range(6, 9)}
        for r1, r2 in [("A", "B"), ("A", "C"), ("B", "C")]:
            cross = set()
            for bi in robot_ranges[r1]:
                for bj in robot_ranges[r2]:
                    pair = (min(bi, bj), max(bi, bj))
                    if pair in pairs:
                        cross.add(pair)
            assert len(cross) == 9, (
                f"{r1}↔{r2} cross-robot pairs: expected 9, got {len(cross)}. "
                f"Missing: {set((bi, bj) for bi in robot_ranges[r1] for bj in robot_ranges[r2]) - cross}"
            )

    def test_body_shape_adr(self):
        """9 bodies × 2 shapes: adr=[0,2,4,...,16], num=[2,2,...,2]."""
        merged = _build_merged()
        static = StaticRobotData.from_merged(merged)

        expected_adr = np.arange(0, N_SHAPES_TOTAL, N_SHAPES_PER_BODY, dtype=np.int32)
        expected_num = np.full(N_BODIES_TOTAL, N_SHAPES_PER_BODY, dtype=np.int32)
        np.testing.assert_array_equal(static.body_shape_adr, expected_adr)
        np.testing.assert_array_equal(static.body_shape_num, expected_num)

    def test_gpu_collision_excluded_agreement(self):
        """GPU exclude matrix must produce the same 30 pairs as CPU."""
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
            f"CPU/GPU pair mismatch.\n"
            f"  CPU only: {cpu_pairs - gpu_pairs}\n"
            f"  GPU only: {gpu_pairs - cpu_pairs}"
        )

    def test_narrowphase_contact_count(self):
        """12 GPU contacts: A↔B=6, B↔C=6, A↔C=0."""
        merged = _build_merged()
        q, qdot = _init_state(merged)
        tau = np.zeros(merged.nv)
        dt = 2e-4

        gpu = GpuEngine(merged, num_envs=1, dt=dt)
        gpu.step(q.copy(), qdot.copy(), tau, dt=dt)
        gpu_bb = [c for c in gpu.query_contacts(env_idx=0) if c.body_j >= 0]

        assert len(gpu_bb) == N_ACTUAL_CONTACTS, (
            f"GPU body-body contacts: expected {N_ACTUAL_CONTACTS}, got {len(gpu_bb)}"
        )

    def test_contact_robot_pair_distribution(self):
        """Contacts must come from A↔B(6) and B↔C(6), none from A↔C."""
        merged = _build_merged()
        q, qdot = _init_state(merged)
        tau = np.zeros(merged.nv)
        dt = 2e-4

        gpu = GpuEngine(merged, num_envs=1, dt=dt)
        gpu.step(q.copy(), qdot.copy(), tau, dt=dt)
        gpu_bb = [c for c in gpu.query_contacts(env_idx=0) if c.body_j >= 0]

        def robot_of(body_idx):
            if body_idx < 3:
                return "A"
            elif body_idx < 6:
                return "B"
            else:
                return "C"

        robot_pair_counts = Counter()
        for c in gpu_bb:
            r1 = robot_of(c.body_i)
            r2 = robot_of(c.body_j)
            key = tuple(sorted([r1, r2]))
            robot_pair_counts[key] += 1

        assert robot_pair_counts[("A", "B")] == N_CONTACTS_AB, (
            f"A↔B contacts: expected {N_CONTACTS_AB}, got {robot_pair_counts[('A', 'B')]}"
        )
        assert robot_pair_counts[("B", "C")] == N_CONTACTS_BC, (
            f"B↔C contacts: expected {N_CONTACTS_BC}, got {robot_pair_counts[('B', 'C')]}"
        )
        assert robot_pair_counts.get(("A", "C"), 0) == N_CONTACTS_AC, (
            f"A↔C contacts: expected {N_CONTACTS_AC}, got {robot_pair_counts.get(('A', 'C'), 0)}"
        )
