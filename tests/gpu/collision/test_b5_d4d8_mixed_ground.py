"""
B(5) Step 5 — D4 mixed shapes + D8 ground contact coexistence.

Fixture: 3R × 3B × 2S with mixed shape types + ground (terrain z=0).
Builds on Step 3/4 topology (3 robots, 3-body chains) with fresh positions.

Shape assignment per body (asymmetric to exercise all dispatch paths):
    Robot A: link0=sphere+sphere, link1=sphere+box,     link2=capsule+sphere
    Robot B: link0=box+sphere,    link1=capsule+capsule, link2=sphere+sphere
    Robot C: link0=sphere+capsule, link1=box+box,       link2=sphere+box

This exercises GPU narrowphase dispatch pairs:
    sphere↔sphere, sphere↔box, sphere↔capsule, capsule↔capsule
    (box↔box falls back to sphere-sphere with body radius in current GPU code)

Layout (chain along +X, separation in Y):
    Robot A: Y=0,    root at Z=0.15 (low → ground contact on link0)
    Robot B: Y=0.10, root at Z=0.50 (high → no ground contact)
    Robot C: Y=0.20, root at Z=0.50 (high → no ground contact)
    Chain link_len = 0.30 along +X

Ground contacts: Robot A's link0 shapes touch ground (z=0.15, shapes at ±0.04 in Z,
with r=0.05 → lowest point at 0.15-0.04-0.05=0.06 > 0, but link0 is contact body).
We position A low enough so at least link0 sphere shapes dip into ground.

Actually: A root at Z=0.05, shapes offset ±0.04 in Z:
    top sphere center at z=0.09, bottom at z=0.01.
    With r=0.05: bottom sphere lowest = 0.01-0.05 = -0.04 → ground contact!
    top sphere lowest = 0.09-0.05 = 0.04 > 0 → no ground contact.
    So link0 has 1 ground contact (bottom shape only).

Body-body contacts:
    A↔B: Y_sep=0.10. Need shapes close enough (2r for sphere-sphere = 0.10).
    With r=0.05: overlap = 0.10-0.10=0 → borderline, not reliable.
    Use Y_sep=0.08 for A↔B to guarantee overlap.
    B↔C: Y_sep=0.12 > 2*0.05=0.10 → no body-body contact.
    A↔C: Y_sep=0.20 >> 0.10 → no body-body contact.

Simplified: only verify that ground + body-body contacts coexist and shape
dispatch doesn't corrupt either. Exact count depends on shape params.

Reference: OPEN_QUESTIONS Q36, session 25.
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
from physics.joint import FreeJoint, RevoluteJoint
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
# Constants
# ---------------------------------------------------------------------------

SPHERE_R = 0.05
BOX_SIZE = (0.08, 0.08, 0.08)  # half-extents ~0.04
CAPSULE_R = 0.04
CAPSULE_LEN = 0.06  # half-length 0.03
SHAPE_Z_OFFSET = 0.04  # ±offset in Z
LINK_LEN = 0.30  # chain along +X
MASS = 1.0
GRAVITY = 9.81

# Robot positions
Y_A = 0.00
Y_B = 0.08  # close to A → body-body contact
Y_C = 0.25  # far from B → no body-body contact
Z_A = 0.05  # low → ground contact
Z_B = 0.05  # same Z as A → body-body contact possible
Z_C = 0.50  # high → no ground

N_BODIES_TOTAL = 9


# ---------------------------------------------------------------------------
# Shape specs per robot (asymmetric to test dispatch)
# ---------------------------------------------------------------------------


def _robot_a_shapes():
    """Robot A: sphere+sphere, sphere+box, capsule+sphere."""
    return [
        [
            ShapeInstance(SphereShape(SPHERE_R), origin_xyz=np.array([0, 0, +SHAPE_Z_OFFSET])),
            ShapeInstance(SphereShape(SPHERE_R), origin_xyz=np.array([0, 0, -SHAPE_Z_OFFSET])),
        ],
        [
            ShapeInstance(SphereShape(SPHERE_R), origin_xyz=np.array([0, 0, +SHAPE_Z_OFFSET])),
            ShapeInstance(BoxShape(BOX_SIZE), origin_xyz=np.array([0, 0, -SHAPE_Z_OFFSET])),
        ],
        [
            ShapeInstance(CapsuleShape(CAPSULE_R, CAPSULE_LEN), origin_xyz=np.array([0, 0, +SHAPE_Z_OFFSET])),
            ShapeInstance(SphereShape(SPHERE_R), origin_xyz=np.array([0, 0, -SHAPE_Z_OFFSET])),
        ],
    ]


def _robot_b_shapes():
    """Robot B: box+sphere, capsule+capsule, sphere+sphere."""
    return [
        [
            ShapeInstance(BoxShape(BOX_SIZE), origin_xyz=np.array([0, 0, +SHAPE_Z_OFFSET])),
            ShapeInstance(SphereShape(SPHERE_R), origin_xyz=np.array([0, 0, -SHAPE_Z_OFFSET])),
        ],
        [
            ShapeInstance(CapsuleShape(CAPSULE_R, CAPSULE_LEN), origin_xyz=np.array([0, 0, +SHAPE_Z_OFFSET])),
            ShapeInstance(CapsuleShape(CAPSULE_R, CAPSULE_LEN), origin_xyz=np.array([0, 0, -SHAPE_Z_OFFSET])),
        ],
        [
            ShapeInstance(SphereShape(SPHERE_R), origin_xyz=np.array([0, 0, +SHAPE_Z_OFFSET])),
            ShapeInstance(SphereShape(SPHERE_R), origin_xyz=np.array([0, 0, -SHAPE_Z_OFFSET])),
        ],
    ]


def _robot_c_shapes():
    """Robot C: sphere+capsule, box+box, sphere+box."""
    return [
        [
            ShapeInstance(SphereShape(SPHERE_R), origin_xyz=np.array([0, 0, +SHAPE_Z_OFFSET])),
            ShapeInstance(CapsuleShape(CAPSULE_R, CAPSULE_LEN), origin_xyz=np.array([0, 0, -SHAPE_Z_OFFSET])),
        ],
        [
            ShapeInstance(BoxShape(BOX_SIZE), origin_xyz=np.array([0, 0, +SHAPE_Z_OFFSET])),
            ShapeInstance(BoxShape(BOX_SIZE), origin_xyz=np.array([0, 0, -SHAPE_Z_OFFSET])),
        ],
        [
            ShapeInstance(SphereShape(SPHERE_R), origin_xyz=np.array([0, 0, +SHAPE_Z_OFFSET])),
            ShapeInstance(BoxShape(BOX_SIZE), origin_xyz=np.array([0, 0, -SHAPE_Z_OFFSET])),
        ],
    ]


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def _chain_robot(shapes_per_body) -> RobotModel:
    """3-body chain along +X with given per-body shape lists."""
    tree = RobotTreeNumpy(gravity=GRAVITY)
    I = 2.0 / 5.0 * MASS * SPHERE_R**2
    inertia = SpatialInertia(mass=MASS, inertia=np.eye(3) * I, com=np.zeros(3))

    tree.add_body(Body("link0", 0, FreeJoint("root"), inertia, SpatialTransform.identity(), -1))
    tree.add_body(
        Body(
            "link1",
            1,
            RevoluteJoint("j1", axis=np.array([0, 1, 0])),
            inertia,
            SpatialTransform(R=np.eye(3), r=np.array([LINK_LEN, 0, 0])),
            0,
        )
    )
    tree.add_body(
        Body(
            "link2",
            2,
            RevoluteJoint("j2", axis=np.array([0, 1, 0])),
            inertia,
            SpatialTransform(R=np.eye(3), r=np.array([LINK_LEN, 0, 0])),
            1,
        )
    )
    tree.finalize()

    geometries = [BodyCollisionGeometry(i, shapes_per_body[i]) for i in range(3)]
    return RobotModel(
        tree=tree,
        geometries=geometries,
        contact_body_names=["link0", "link1", "link2"],
    )


def _build_merged():
    """Merge 3 mixed-shape robots with flat terrain."""
    return merge_models(
        robots={
            "A": _chain_robot(_robot_a_shapes()),
            "B": _chain_robot(_robot_b_shapes()),
            "C": _chain_robot(_robot_c_shapes()),
        },
        terrain=FlatTerrain(),
    )


def _init_state(merged):
    """Position robots: A low (ground contact), B and C high."""
    q, qdot = merged.tree.default_state()
    for name, y_pos, z_pos in [("A", Y_A, Z_A), ("B", Y_B, Z_B), ("C", Y_C, Z_C)]:
        rs = merged.robot_slices[name]
        qa = q[rs.q_slice]
        qa[4] = 0.0  # px
        qa[5] = y_pos  # py
        qa[6] = z_pos  # pz
    return q, qdot


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStep5MixedShapesGround:
    """D4 mixed shapes + D8 ground: dispatch correctness + coexistence."""

    def test_shape_type_diversity(self):
        """All 3 shape types must appear in the merged model's geometries."""
        merged = _build_merged()
        shape_types = set()
        for geom in merged.collision_shapes:
            if geom is not None:
                for si in geom.shapes:
                    shape_types.add(type(si.shape).__name__)
        assert shape_types == {"SphereShape", "BoxShape", "CapsuleShape"}, (
            f"Expected all 3 shape types, got {shape_types}"
        )

    def test_ground_and_body_contacts_coexist(self):
        """Both ground (bj=-1) and body-body (bj>=0) contacts must be present."""
        merged = _build_merged()
        q, qdot = _init_state(merged)
        tau = np.zeros(merged.nv)
        dt = 2e-4

        gpu = GpuEngine(merged, num_envs=1, dt=dt)
        gpu.step(q.copy(), qdot.copy(), tau, dt=dt)
        contacts = gpu.query_contacts(env_idx=0)

        ground = [c for c in contacts if c.body_j == -1]
        body_body = [c for c in contacts if c.body_j >= 0]

        assert len(ground) > 0, "No ground contacts detected (Robot A should touch ground)"
        assert len(body_body) > 0, "No body-body contacts detected (A↔B should overlap)"

    def test_ground_contacts_are_low_robots(self):
        """Ground contacts should only come from Robot A and B bodies (low Z).
        Robot C (Z=0.50) should have no ground contacts."""
        merged = _build_merged()
        q, qdot = _init_state(merged)
        tau = np.zeros(merged.nv)
        dt = 2e-4

        gpu = GpuEngine(merged, num_envs=1, dt=dt)
        gpu.step(q.copy(), qdot.copy(), tau, dt=dt)
        ground = [c for c in gpu.query_contacts(env_idx=0) if c.body_j == -1]

        ab_bodies = set(range(0, 6))  # A=[0,1,2], B=[3,4,5]
        c_bodies = set(range(6, 9))
        for c in ground:
            assert c.body_i not in c_bodies, f"Ground contact on Robot C body {c.body_i}, expected none"
            assert c.body_i in ab_bodies, f"Ground contact on unexpected body {c.body_i}"

    def test_body_body_contacts_only_ab(self):
        """Body-body contacts should only come from A↔B (Y_sep=0.08 < 2r).
        B↔C (Y_sep=0.17) and A↔C (Y_sep=0.25) are too far."""
        merged = _build_merged()
        q, qdot = _init_state(merged)
        tau = np.zeros(merged.nv)
        dt = 2e-4

        gpu = GpuEngine(merged, num_envs=1, dt=dt)
        gpu.step(q.copy(), qdot.copy(), tau, dt=dt)
        bb = [c for c in gpu.query_contacts(env_idx=0) if c.body_j >= 0]

        a_bodies = set(range(0, 3))
        b_bodies = set(range(3, 6))
        for c in bb:
            bi, bj = min(c.body_i, c.body_j), max(c.body_i, c.body_j)
            assert bi in a_bodies and bj in b_bodies, (
                f"Body-body contact ({bi},{bj}) not from A↔B. A=[0,1,2], B=[3,4,5]"
            )

    def test_no_nan_in_contact_normals(self):
        """Mixed shape dispatch must not produce NaN normals or points."""
        merged = _build_merged()
        q, qdot = _init_state(merged)
        tau = np.zeros(merged.nv)
        dt = 2e-4

        gpu = GpuEngine(merged, num_envs=1, dt=dt)
        gpu.step(q.copy(), qdot.copy(), tau, dt=dt)
        contacts = gpu.query_contacts(env_idx=0)

        for i, c in enumerate(contacts):
            assert np.all(np.isfinite(c.normal)), f"Contact {i}: NaN normal {c.normal}"
            assert np.all(np.isfinite(c.point)), f"Contact {i}: NaN point {c.point}"
            assert np.isfinite(c.depth), f"Contact {i}: NaN depth {c.depth}"

    def test_simulation_stable_100_steps(self):
        """100 steps with mixed shapes + ground must stay finite (no divergence)."""
        merged = _build_merged()
        q, qdot = _init_state(merged)
        tau = np.zeros(merged.nv)
        dt = 2e-4

        gpu = GpuEngine(merged, num_envs=1, dt=dt)
        for step in range(100):
            out = gpu.step(q, qdot, tau, dt=dt)
            q, qdot = out.q_new, out.qdot_new
            assert np.all(np.isfinite(q)), f"q has NaN/Inf at step {step}"
            assert np.all(np.isfinite(qdot)), f"qdot has NaN/Inf at step {step}"
