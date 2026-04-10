"""
B(5) Step 2 — D3 multi-shape cross-robot collision.

Fixture: 2 robots × 3 bodies (FreeJoint root + 2 RevoluteJoint chain) × 2 spheres.
Chain extends along +Y (fresh axis vs Step 1's +X). Inter-robot separation along X.

Layout (side view, X-Z plane, one body-pair):
    Each body has 2 spheres r=0.08, offset ±0.08 in Z from body origin.
    Robot A at X=0, Robot B at X=0.12.
    Same-level shape pairs (top↔top, bot↔bot): dist=0.12 < 0.16 → contact.
    Cross-level shape pairs (top↔bot, bot↔top): dist=0.20 > 0.16 → no contact.

Expected:
    collision pairs (body level): 11 (same as Step 1)
    shapes total: 12 (6 bodies × 2 shapes)
    body_shape_adr: [0, 2, 4, 6, 8, 10], body_shape_num: [2, 2, 2, 2, 2, 2]
    contacts: 6 (3 same-Y body-pairs × 2 same-level shape-pairs)

Discriminator vs Step 1: contact count 6 vs 3 (multi-shape doubles it).
If narrowphase shape loop has off-by-one → 3; if cross-level leaks → 12.

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
# Fixture constants (all fresh — no reuse from Step 1)
# ---------------------------------------------------------------------------

RADIUS = 0.08
SHAPE_Z_OFFSET = 0.08  # ±offset in Z per body
LINK_LEN = 0.35  # X_tree translation along +Y
X_SEP = 0.12  # X separation between robot A and B
Z_HEIGHT = 0.50
MASS = 1.0
GRAVITY = 9.81

N_ROBOTS = 2
N_BODIES_PER_ROBOT = 3
N_BODIES_TOTAL = N_ROBOTS * N_BODIES_PER_ROBOT  # 6
N_SHAPES_PER_BODY = 2
N_SHAPES_TOTAL = N_BODIES_TOTAL * N_SHAPES_PER_BODY  # 12
N_INTRA_PAIRS = 2  # body(0,2) per robot
N_CROSS_PAIRS = 9  # 3 × 3
N_TOTAL_PAIRS = N_INTRA_PAIRS + N_CROSS_PAIRS  # 11
N_ACTUAL_CONTACTS = 6  # 3 same-Y body-pairs × 2 same-level shape-pairs


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def _dual_sphere_chain() -> RobotModel:
    """3-body chain along +Y, 2 spheres per body (±Z offset)."""
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
    # Body 1: child of 0, RevoluteJoint(X), chain along +Y
    tree.add_body(
        Body(
            name="link1",
            index=1,
            joint=RevoluteJoint("j1", axis=np.array([1, 0, 0])),
            inertia=inertia,
            X_tree=SpatialTransform(R=np.eye(3), r=np.array([0, LINK_LEN, 0])),
            parent=0,
        )
    )
    # Body 2: child of 1, RevoluteJoint(X)
    tree.add_body(
        Body(
            name="link2",
            index=2,
            joint=RevoluteJoint("j2", axis=np.array([1, 0, 0])),
            inertia=inertia,
            X_tree=SpatialTransform(R=np.eye(3), r=np.array([0, LINK_LEN, 0])),
            parent=1,
        )
    )
    tree.finalize()

    # 2 spheres per body: offset ±SHAPE_Z_OFFSET in Z
    geometries = []
    for i in range(3):
        shapes = [
            ShapeInstance(SphereShape(RADIUS), origin_xyz=np.array([0, 0, +SHAPE_Z_OFFSET])),
            ShapeInstance(SphereShape(RADIUS), origin_xyz=np.array([0, 0, -SHAPE_Z_OFFSET])),
        ]
        geometries.append(BodyCollisionGeometry(i, shapes))

    return RobotModel(
        tree=tree,
        geometries=geometries,
        contact_body_names=["link0", "link1", "link2"],
    )


def _build_merged():
    """Merge two dual-sphere chain robots."""
    return merge_models(robots={"A": _dual_sphere_chain(), "B": _dual_sphere_chain()})


def _init_state(merged):
    """Robot A at X=0, Robot B at X=X_SEP, both at Z=Z_HEIGHT. Joints at q=0."""
    q, qdot = merged.tree.default_state()
    rs_a = merged.robot_slices["A"]
    rs_b = merged.robot_slices["B"]

    # FreeJoint q layout: [qw, qx, qy, qz, px, py, pz]
    qa = q[rs_a.q_slice]
    qa[4] = 0.0  # px
    qa[5] = 0.0  # py
    qa[6] = Z_HEIGHT  # pz

    qb = q[rs_b.q_slice]
    qb[4] = X_SEP  # px
    qb[5] = 0.0  # py
    qb[6] = Z_HEIGHT  # pz

    return q, qdot


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStep2MultiShapeCrossRobot:
    """D3 multi-shape: structural + geometric assertions for 2R × 3B × 2S."""

    def test_collision_pair_count(self):
        """Body-level collision pairs unchanged at 11 (multi-shape doesn't affect pair list)."""
        merged = _build_merged()
        assert len(merged.collision_pairs) == N_TOTAL_PAIRS

    def test_body_shape_adr_stride(self):
        """body_shape_adr must have stride 2; body_shape_num all 2."""
        merged = _build_merged()
        static = StaticRobotData.from_merged(merged)

        expected_adr = np.arange(0, N_SHAPES_TOTAL, N_SHAPES_PER_BODY, dtype=np.int32)
        expected_num = np.full(N_BODIES_TOTAL, N_SHAPES_PER_BODY, dtype=np.int32)
        np.testing.assert_array_equal(
            static.body_shape_adr, expected_adr, err_msg="body_shape_adr stride mismatch"
        )
        np.testing.assert_array_equal(static.body_shape_num, expected_num, err_msg="body_shape_num mismatch")

    def test_flat_shape_body_mapping(self):
        """Flat shape arrays must map each shape back to the correct body index."""
        merged = _build_merged()
        static = StaticRobotData.from_merged(merged)

        # shape_body: each shape's owning body index
        # shapes 0,1 → body 0; shapes 2,3 → body 1; ... shapes 10,11 → body 5
        expected_body = np.repeat(np.arange(N_BODIES_TOTAL, dtype=np.int32), N_SHAPES_PER_BODY)
        np.testing.assert_array_equal(
            static.shape_body, expected_body, err_msg="flat shape→body mapping wrong"
        )

    def test_gpu_collision_excluded_agreement(self):
        """GPU collision_excluded must still agree with CPU collision_pairs."""
        merged = _build_merged()
        static = StaticRobotData.from_merged(merged)
        excl = static.collision_excluded

        nb = N_BODIES_TOTAL
        gpu_pairs = set()
        for i in range(nb):
            for j in range(i + 1, nb):
                if excl[i, j] == 0:
                    gpu_pairs.add((i, j))

        cpu_pairs = set(merged.collision_pairs)
        assert gpu_pairs == cpu_pairs, (
            f"CPU/GPU pair mismatch.\n"
            f"  CPU only: {cpu_pairs - gpu_pairs}\n"
            f"  GPU only: {gpu_pairs - cpu_pairs}"
        )

    def test_narrowphase_contact_count(self):
        """GPU: 6 shape-level contacts. CPU: 3 body-level (known limitation).

        CPU body-body uses body-level sphere approximation (single sphere per body,
        see cpu_engine.py line 132), so it sees 3 contacts (one per body-pair).
        GPU uses per-shape narrowphase, correctly finding 6 (2 per body-pair).
        This asymmetry is a known gap — CPU per-shape port is deferred.
        """
        merged = _build_merged()
        q, qdot = _init_state(merged)
        tau = np.zeros(merged.nv)
        dt = 2e-4

        # CPU — body-level approximation: 1 contact per overlapping body-pair
        cpu = CpuEngine(merged, dt=dt)
        from physics.dynamics_cache import DynamicsCache

        cache = DynamicsCache.from_tree(merged.tree, q, qdot, dt)
        cpu_contacts = cpu._detect_contacts(cache)
        cpu_bb = [c for c in cpu_contacts if c.body_j >= 0]
        n_cpu_body_pairs = 3  # body-level: A0↔B0, A1↔B1, A2↔B2
        assert len(cpu_bb) == n_cpu_body_pairs, (
            f"CPU body-level contacts: expected {n_cpu_body_pairs}, got {len(cpu_bb)}"
        )

        # GPU — per-shape narrowphase: 2 shape contacts per body-pair
        gpu = GpuEngine(merged, num_envs=1, dt=dt)
        gpu.step(q.copy(), qdot.copy(), tau, dt=dt)
        gpu_bb = [c for c in gpu.query_contacts(env_idx=0) if c.body_j >= 0]
        assert len(gpu_bb) == N_ACTUAL_CONTACTS, (
            f"GPU shape-level contacts: expected {N_ACTUAL_CONTACTS}, got {len(gpu_bb)}"
        )

    def test_contact_body_pairs_identity(self):
        """The 6 contacts must come from exactly the 3 expected body-pairs, 2 each."""
        merged = _build_merged()
        q, qdot = _init_state(merged)
        tau = np.zeros(merged.nv)
        dt = 2e-4

        gpu = GpuEngine(merged, num_envs=1, dt=dt)
        gpu.step(q.copy(), qdot.copy(), tau, dt=dt)
        gpu_bb = [c for c in gpu.query_contacts(env_idx=0) if c.body_j >= 0]

        # Expected body-pairs: (A0=0,B0=3), (A1=1,B1=4), (A2=2,B2=5)
        from collections import Counter

        pair_counts = Counter()
        for c in gpu_bb:
            pair = (min(c.body_i, c.body_j), max(c.body_i, c.body_j))
            pair_counts[pair] += 1

        expected = {(0, 3): 2, (1, 4): 2, (2, 5): 2}
        assert dict(pair_counts) == expected, (
            f"Contact body-pair distribution mismatch.\n  Expected: {expected}\n  Got: {dict(pair_counts)}"
        )

    def test_multishape_depth_and_normal(self):
        """Hand-computed depth/normal for multi-shape sphere-sphere contacts.

        Geometry (q=0):
            A bodies at (0, 0, 0.5), (0, 0.35, 0.5), (0, 0.70, 0.5)
            B bodies at (0.12, 0, 0.5), (0.12, 0.35, 0.5), (0.12, 0.70, 0.5)
            Each body: 2 spheres r=0.08, offset ±0.08 in Z
            X separation = 0.12

        Same-level shape pairs (top↔top, bot↔bot):
            dist = 0.12 (pure X separation, same Z)
            depth = 2*0.08 - 0.12 = 0.04
            normal = (p_Ai - p_Bi) / dist = (-1, 0, 0)

        Contact points (on surface of B shape toward A):
            top: p_Bj_top + normal * r = (0.12, y, 0.58) + (-0.08, 0, 0) = (0.04, y, 0.58)
            bot: p_Bj_bot + normal * r = (0.12, y, 0.42) + (-0.08, 0, 0) = (0.04, y, 0.42)
        """
        merged = _build_merged()
        q, qdot = _init_state(merged)
        tau = np.zeros(merged.nv)
        dt = 2e-4

        gpu = GpuEngine(merged, num_envs=1, dt=dt)
        gpu.step(q.copy(), qdot.copy(), tau, dt=dt)
        gpu_bb = [c for c in gpu.query_contacts(env_idx=0) if c.body_j >= 0]
        gpu_bb.sort(key=lambda c: (c.body_i, -c.point[2]))  # sort by body, Z desc

        # Expected: 6 contacts (3 body-pairs × 2 same-level)
        y_vals = [0.0, 0.35, 0.70]
        expected = []
        for k, (bi, bj) in enumerate([(0, 3), (1, 4), (2, 5)]):
            y = y_vals[k]
            expected.append((bi, bj, 0.04, [-1, 0, 0], [0.04, y, 0.58]))  # top
            expected.append((bi, bj, 0.04, [-1, 0, 0], [0.04, y, 0.42]))  # bot

        assert len(gpu_bb) == len(expected), f"Expected {len(expected)} contacts"
        for i, (c, (bi, bj, depth, normal, point)) in enumerate(zip(gpu_bb, expected)):
            assert c.body_i == bi and c.body_j == bj, (
                f"Contact {i}: ({c.body_i},{c.body_j}), expected ({bi},{bj})"
            )
            np.testing.assert_allclose(c.depth, depth, atol=1e-4, err_msg=f"Contact {i} depth")
            np.testing.assert_allclose(c.normal, normal, atol=1e-4, err_msg=f"Contact {i} normal")
            np.testing.assert_allclose(c.point, point, atol=1e-3, err_msg=f"Contact {i} point")
