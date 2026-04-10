"""
B(5) Step 6 — D6 CPU/GPU agreement + D7 num_envs ≥ 2.

Fixture: Step 5's mixed-shape 3R×3B×2S, but run with num_envs=2 on GPU.
Env 0: same layout as Step 5 (A↔B overlap, ground contact)
Env 1: all robots spread far apart (no contacts at all)

This catches:
    - env axis cross-talk (env 1 contacts leaking into env 0 or vice versa)
    - CPU vs GPU contact count agreement (env 0 only, since CPU is single-env)
    - multi-env state independence

Layout:
    Env 0: A(Y=0, Z=0.05), B(Y=0.08, Z=0.05), C(Y=0.25, Z=0.50) — contacts expected
    Env 1: A(Y=0, Z=2.0),  B(Y=2.0, Z=2.0),  C(Y=4.0, Z=2.0)  — no contacts

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
# Constants (same shape params as Step 5)
# ---------------------------------------------------------------------------

SPHERE_R = 0.05
BOX_SIZE = (0.08, 0.08, 0.08)
CAPSULE_R = 0.04
CAPSULE_LEN = 0.06
SHAPE_Z_OFFSET = 0.04
LINK_LEN = 0.30
MASS = 1.0
GRAVITY = 9.81

NUM_ENVS = 2


# ---------------------------------------------------------------------------
# Shape specs (same as Step 5 for consistency)
# ---------------------------------------------------------------------------


def _robot_a_shapes():
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
    return RobotModel(tree=tree, geometries=geometries, contact_body_names=["link0", "link1", "link2"])


def _build_merged():
    return merge_models(
        robots={
            "A": _chain_robot(_robot_a_shapes()),
            "B": _chain_robot(_robot_b_shapes()),
            "C": _chain_robot(_robot_c_shapes()),
        },
        terrain=FlatTerrain(),
    )


def _set_robot_pos(q, merged, name, px, py, pz):
    """Set root FreeJoint position for one robot in q vector."""
    rs = merged.robot_slices[name]
    qa = q[rs.q_slice]
    qa[4] = px
    qa[5] = py
    qa[6] = pz


def _make_q_2env(merged):
    """Build (2, nq) initial state. Env 0: contacts. Env 1: no contacts."""
    q0, _ = merged.tree.default_state()
    q1 = q0.copy()

    # Env 0: A and B close + low (contacts)
    _set_robot_pos(q0, merged, "A", 0.0, 0.00, 0.05)
    _set_robot_pos(q0, merged, "B", 0.0, 0.08, 0.05)
    _set_robot_pos(q0, merged, "C", 0.0, 0.25, 0.50)

    # Env 1: all far apart (no contacts)
    _set_robot_pos(q1, merged, "A", 0.0, 0.00, 2.00)
    _set_robot_pos(q1, merged, "B", 0.0, 2.00, 2.00)
    _set_robot_pos(q1, merged, "C", 0.0, 4.00, 2.00)

    return np.stack([q0, q1], axis=0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStep6CpuGpuMultiEnv:
    """D6 CPU/GPU + D7 num_envs: contact isolation and backend agreement."""

    def test_env0_has_contacts(self):
        """Env 0 (close layout) must have both ground and body-body contacts."""
        merged = _build_merged()
        q_2env = _make_q_2env(merged)
        qdot_2env = np.zeros((NUM_ENVS, merged.nv), dtype=np.float64)
        tau = np.zeros((NUM_ENVS, merged.nv))
        dt = 2e-4

        gpu = GpuEngine(merged, num_envs=NUM_ENVS, dt=dt)
        gpu.step(q_2env, qdot_2env, tau, dt=dt)
        contacts_0 = gpu.query_contacts(env_idx=0)

        ground = [c for c in contacts_0 if c.body_j == -1]
        bb = [c for c in contacts_0 if c.body_j >= 0]
        assert len(ground) > 0, "Env 0: expected ground contacts"
        assert len(bb) > 0, "Env 0: expected body-body contacts"

    def test_env1_no_contacts(self):
        """Env 1 (spread layout) must have zero contacts."""
        merged = _build_merged()
        q_2env = _make_q_2env(merged)
        qdot_2env = np.zeros((NUM_ENVS, merged.nv), dtype=np.float64)
        tau = np.zeros((NUM_ENVS, merged.nv))
        dt = 2e-4

        gpu = GpuEngine(merged, num_envs=NUM_ENVS, dt=dt)
        gpu.step(q_2env, qdot_2env, tau, dt=dt)
        contacts_1 = gpu.query_contacts(env_idx=1)

        assert len(contacts_1) == 0, (
            f"Env 1: expected 0 contacts, got {len(contacts_1)} (cross-talk from env 0?)"
        )

    def test_env_contact_counts_independent(self):
        """Contact counts per env must be independent (no shared atomic counter)."""
        merged = _build_merged()
        q_2env = _make_q_2env(merged)
        qdot_2env = np.zeros((NUM_ENVS, merged.nv), dtype=np.float64)
        tau = np.zeros((NUM_ENVS, merged.nv))
        dt = 2e-4

        gpu = GpuEngine(merged, num_envs=NUM_ENVS, dt=dt)
        gpu.step(q_2env, qdot_2env, tau, dt=dt)

        count_0 = len(gpu.query_contacts(env_idx=0))
        count_1 = len(gpu.query_contacts(env_idx=1))

        assert count_0 > 0, "Env 0 should have contacts"
        assert count_1 == 0, "Env 1 should have no contacts"
        assert count_0 != count_1, "Env counts should differ (independence check)"

    def test_cpu_gpu_contact_count_agreement(self):
        """CPU and GPU env 0 must agree on contact count (body-body only).

        Note: CPU uses body-level sphere approximation for body-body (Step 2
        finding), so we compare ground contact count which both handle per-shape.
        """
        merged = _build_merged()
        q_2env = _make_q_2env(merged)
        q_env0 = q_2env[0]
        qdot = np.zeros(merged.nv)
        dt = 2e-4

        # CPU single-env
        cpu = CpuEngine(merged, dt=dt)
        from physics.dynamics_cache import DynamicsCache

        cache = DynamicsCache.from_tree(merged.tree, q_env0, qdot, dt)
        cpu_contacts = cpu._detect_contacts(cache)
        cpu_ground = [c for c in cpu_contacts if c.body_j < 0]

        # GPU env 0
        tau_2d = np.zeros((NUM_ENVS, merged.nv))
        qdot_2env = np.zeros((NUM_ENVS, merged.nv), dtype=np.float64)
        gpu = GpuEngine(merged, num_envs=NUM_ENVS, dt=dt)
        gpu.step(q_2env, qdot_2env, tau_2d, dt=dt)
        gpu_ground = [c for c in gpu.query_contacts(env_idx=0) if c.body_j == -1]

        # Ground contacts should match (both do per-shape ground detection)
        assert len(cpu_ground) > 0, "CPU should detect ground contacts"
        assert len(gpu_ground) > 0, "GPU should detect ground contacts"
        # Allow small difference due to float32 vs float64 threshold
        assert abs(len(cpu_ground) - len(gpu_ground)) <= 1, (
            f"CPU/GPU ground contact count mismatch: CPU={len(cpu_ground)}, GPU={len(gpu_ground)}"
        )

    def test_multienv_simulation_stable(self):
        """50 steps with 2 envs must stay finite (no NaN from cross-talk)."""
        merged = _build_merged()
        q_2env = _make_q_2env(merged)
        qdot_2env = np.zeros((NUM_ENVS, merged.nv), dtype=np.float64)
        tau = np.zeros((NUM_ENVS, merged.nv))
        dt = 2e-4

        gpu = GpuEngine(merged, num_envs=NUM_ENVS, dt=dt)
        q, qdot = q_2env.copy(), qdot_2env.copy()
        for step in range(50):
            out = gpu.step(q, qdot, tau, dt=dt)
            q, qdot = out.q_new, out.qdot_new
            assert np.all(np.isfinite(q)), f"q NaN at step {step}"
            assert np.all(np.isfinite(qdot)), f"qdot NaN at step {step}"
