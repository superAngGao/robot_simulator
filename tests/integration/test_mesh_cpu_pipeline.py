"""
End-to-end mesh loading → CPU collision pipeline tests.

Validates the full chain: URDF <mesh> → trimesh load → ConvexHullShape →
CpuEngine GJK/EPA collision detection (ground + body-body).
"""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

from physics.cpu_engine import CpuEngine
from physics.geometry import (
    BodyCollisionGeometry,
    ConvexHullShape,
    ShapeInstance,
)
from physics.joint import FreeJoint, RevoluteJoint
from physics.merged_model import merge_models
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from physics.terrain import FlatTerrain
from robot import load_urdf

_has_trimesh = True
try:
    import trimesh
except ImportError:
    _has_trimesh = False

needs_trimesh = pytest.mark.skipif(not _has_trimesh, reason="trimesh not installed")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CUBE_HALF = 0.05
_CUBE_VERTS = (
    np.array(
        [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]],
        dtype=np.float64,
    )
    * _CUBE_HALF
)


def _one_body_convexhull_model(pos_z: float = 0.5):
    """Single floating body with ConvexHullShape cube."""
    tree = RobotTreeNumpy(gravity=9.81)
    tree.add_body(
        Body(
            name="box",
            index=0,
            joint=FreeJoint("root"),
            inertia=SpatialInertia(1.0, np.eye(3) * 0.001, np.zeros(3)),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()

    from robot.model import RobotModel

    return RobotModel(
        tree=tree,
        actuated_joint_names=[],
        contact_body_names=["box"],
        geometries=[
            BodyCollisionGeometry(
                body_index=0,
                shapes=[ShapeInstance(shape=ConvexHullShape(_CUBE_VERTS))],
            )
        ],
    )


def _two_body_convexhull_model():
    """Two floating bodies with ConvexHullShape cubes, for body-body test."""
    tree = RobotTreeNumpy(gravity=9.81)
    tree.add_body(
        Body(
            name="box_a",
            index=0,
            joint=FreeJoint("root_a"),
            inertia=SpatialInertia(1.0, np.eye(3) * 0.001, np.zeros(3)),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.add_body(
        Body(
            name="box_b",
            index=0,
            joint=FreeJoint("root_b"),
            inertia=SpatialInertia(1.0, np.eye(3) * 0.001, np.zeros(3)),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()

    from robot.model import RobotModel

    return RobotModel(
        tree=tree,
        actuated_joint_names=[],
        contact_body_names=["box_a", "box_b"],
        geometries=[
            BodyCollisionGeometry(
                body_index=0,
                shapes=[ShapeInstance(shape=ConvexHullShape(_CUBE_VERTS))],
            ),
            BodyCollisionGeometry(
                body_index=1,
                shapes=[ShapeInstance(shape=ConvexHullShape(_CUBE_VERTS))],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Ground contact tests
# ---------------------------------------------------------------------------


class TestConvexHullGroundContact:
    def test_ground_contact_detected(self):
        """ConvexHullShape body near ground → contact detected."""
        model = _one_body_convexhull_model()
        terrain = FlatTerrain()
        merged = merge_models({"r": model}, terrain=terrain)
        engine = CpuEngine(merged, dt=1e-4)

        # Set position: center at z=0.03, cube half=0.05 → bottom at z=-0.02
        q = np.zeros(merged.nq)
        q[0] = 1.0  # qw = 1 (identity quaternion)
        q[6] = 0.03  # pz in FreeJoint q layout [qw,qx,qy,qz,px,py,pz]
        qdot = np.zeros(merged.nv)

        engine.step(q, qdot, np.zeros(merged.nv))
        contacts = engine.query_contacts()
        assert len(contacts) >= 1
        # All contacts should be ground (body_j == -1)
        for c in contacts:
            assert c.body_j == -1
            assert c.depth > 0

    def test_no_contact_above_ground(self):
        """ConvexHullShape body well above ground → no contact."""
        model = _one_body_convexhull_model()
        terrain = FlatTerrain()
        merged = merge_models({"r": model}, terrain=terrain)
        engine = CpuEngine(merged, dt=1e-4)

        q = np.zeros(merged.nq)
        q[0] = 1.0  # qw = 1 (identity quaternion)
        q[6] = 1.0  # z = 1.0, well above ground
        qdot = np.zeros(merged.nv)

        engine.step(q, qdot, np.zeros(merged.nv))
        contacts = engine.query_contacts()
        assert len(contacts) == 0


# ---------------------------------------------------------------------------
# Body-body contact tests
# ---------------------------------------------------------------------------


class TestConvexHullBodyBodyContact:
    def test_body_body_contact_detected(self):
        """Two overlapping ConvexHullShape bodies → body-body contact via GJK."""
        model = _two_body_convexhull_model()
        terrain = FlatTerrain()
        merged = merge_models({"r": model}, terrain=terrain)
        engine = CpuEngine(merged, dt=1e-4)

        q = np.zeros(merged.nq)
        q[0] = 1.0  # Body A: qw = 1 (identity quaternion)
        q[7] = 1.0  # Body B: qw = 1
        # FreeJoint q: [qw,qx,qy,qz, px,py,pz]
        # Body A: q[0:7], body B: q[7:14]
        q[6] = 0.5  # A at z=0.5 (above ground)
        q[13] = 0.5  # B at z=0.5
        q[4] = 0.0  # A at x=0
        q[11] = 0.08  # B at x=0.08
        qdot = np.zeros(merged.nv)

        engine.step(q, qdot, np.zeros(merged.nv))
        contacts = engine.query_contacts()

        body_body_contacts = [c for c in contacts if c.body_j >= 0]
        assert len(body_body_contacts) >= 1
        for c in body_body_contacts:
            assert c.depth > 0

    def test_no_body_body_contact_when_separated(self):
        """Two separated ConvexHullShape bodies → no body-body contact."""
        model = _two_body_convexhull_model()
        terrain = FlatTerrain()
        merged = merge_models({"r": model}, terrain=terrain)
        engine = CpuEngine(merged, dt=1e-4)

        q = np.zeros(merged.nq)
        q[0] = 1.0  # Body A: qw = 1
        q[7] = 1.0  # Body B: qw = 1
        q[6] = 0.5
        q[13] = 0.5
        q[4] = 0.0
        q[11] = 0.5  # Far apart: x distance = 0.5 >> 0.1 overlap range
        qdot = np.zeros(merged.nv)

        engine.step(q, qdot, np.zeros(merged.nv))
        contacts = engine.query_contacts()

        body_body_contacts = [c for c in contacts if c.body_j >= 0]
        assert len(body_body_contacts) == 0


# ---------------------------------------------------------------------------
# URDF mesh → CpuEngine end-to-end
# ---------------------------------------------------------------------------


@needs_trimesh
class TestUrdfMeshToCpuEngine:
    def test_urdf_mesh_ground_contact(self, tmp_path):
        """URDF with <mesh> → load → CpuEngine → ground contact detected."""
        # Create STL
        stl_path = tmp_path / "cube.stl"
        trimesh.creation.box(extents=(0.1, 0.1, 0.1)).export(str(stl_path))

        # Create URDF
        urdf_content = textwrap.dedent("""
        <robot name="mesh_test">
          <link name="base">
            <inertial>
              <mass value="1.0"/>
              <origin xyz="0 0 0" rpy="0 0 0"/>
              <inertia ixx="0.001" ixy="0" ixz="0"
                       iyy="0.001" iyz="0" izz="0.001"/>
            </inertial>
            <collision>
              <origin xyz="0 0 0" rpy="0 0 0"/>
              <geometry>
                <mesh filename="cube.stl"/>
              </geometry>
            </collision>
          </link>
        </robot>
        """)
        urdf_path = tmp_path / "robot.urdf"
        urdf_path.write_text(urdf_content)

        # Load
        model = load_urdf(str(urdf_path), floating_base=True, contact_links=["base"])

        # Verify mesh loaded as ConvexHullShape
        assert len(model.geometries) == 1
        shape = model.geometries[0].shapes[0].shape
        assert isinstance(shape, ConvexHullShape)

        # Simulate
        terrain = FlatTerrain()
        merged = merge_models({"r": model}, terrain=terrain)
        engine = CpuEngine(merged, dt=1e-4)

        q = np.zeros(merged.nq)
        q[0] = 1.0  # qw = 1 (identity quaternion)
        q[6] = 0.03  # z=0.03, cube half=0.05 → penetrating ground
        qdot = np.zeros(merged.nv)

        engine.step(q, qdot, np.zeros(merged.nv))
        contacts = engine.query_contacts()
        assert len(contacts) >= 1
        assert contacts[0].depth > 0


# ---------------------------------------------------------------------------
# Two Go2-style legs mid-air collision
# ---------------------------------------------------------------------------

# Approximate Go2 leg link dimensions
_THIGH_LENGTH = 0.20
_CALF_LENGTH = 0.20
_THIGH_RADIUS = 0.02
_CALF_RADIUS = 0.015


def _box_hull(hx: float, hy: float, hz: float) -> ConvexHullShape:
    """Create a box-shaped ConvexHullShape."""
    signs = np.array(
        [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]],
        dtype=np.float64,
    )
    return ConvexHullShape(signs * np.array([hx, hy, hz]))


def _build_go2_leg(name: str):
    """Build a Go2-style single leg: floating hip + thigh + calf.

    All links use ConvexHullShape collision geometry (box hulls
    approximating the link shape).
    """
    from robot.model import RobotModel

    tree = RobotTreeNumpy(gravity=9.81)

    # Hip body (floating base)
    hip_idx = tree.add_body(
        Body(
            name=f"{name}_hip",
            index=0,
            joint=FreeJoint(f"{name}_root"),
            inertia=SpatialInertia(0.5, np.eye(3) * 0.001, np.zeros(3)),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )

    # Thigh body (revolute Y-axis, hangs from hip)
    thigh_idx = tree.add_body(
        Body(
            name=f"{name}_thigh",
            index=0,
            joint=RevoluteJoint(f"{name}_thigh_joint", axis=np.array([0.0, 1.0, 0.0])),
            inertia=SpatialInertia.from_cylinder(0.8, _THIGH_RADIUS, _THIGH_LENGTH),
            X_tree=SpatialTransform.identity(),
            parent=hip_idx,
        )
    )

    # Calf body (revolute Y-axis, hangs from thigh)
    calf_idx = tree.add_body(
        Body(
            name=f"{name}_calf",
            index=0,
            joint=RevoluteJoint(f"{name}_calf_joint", axis=np.array([0.0, 1.0, 0.0])),
            inertia=SpatialInertia.from_cylinder(0.4, _CALF_RADIUS, _CALF_LENGTH),
            X_tree=SpatialTransform(np.eye(3), np.array([0.0, 0.0, -_THIGH_LENGTH])),
            parent=thigh_idx,
        )
    )

    tree.finalize()

    # Collision geometry: box hulls approximating cylindrical links
    geometries = [
        BodyCollisionGeometry(
            hip_idx,
            [ShapeInstance(shape=_box_hull(0.04, 0.03, 0.03))],
        ),
        BodyCollisionGeometry(
            thigh_idx,
            [ShapeInstance(shape=_box_hull(_THIGH_RADIUS, _THIGH_RADIUS, _THIGH_LENGTH / 2))],
        ),
        BodyCollisionGeometry(
            calf_idx,
            [ShapeInstance(shape=_box_hull(_CALF_RADIUS, _CALF_RADIUS, _CALF_LENGTH / 2))],
        ),
    ]

    return RobotModel(
        tree=tree,
        actuated_joint_names=[f"{name}_thigh_joint", f"{name}_calf_joint"],
        contact_body_names=[f"{name}_hip", f"{name}_thigh", f"{name}_calf"],
        geometries=geometries,
    )


class TestTwoLegMidAirCollision:
    """Two Go2-style legs launched at each other in the air.

    Each leg is a 3-link chain (hip + thigh + calf) with ConvexHullShape
    collision geometry. They start separated with opposing initial
    velocities and should produce body-body contacts via GJK/EPA.

    NOTE: Current tests only verify structural properties (contact detected,
    timing reasonable). Post-collision dynamics correctness (rebound direction,
    energy, multi-step trajectory) requires visual verification — deferred to
    Phase 3 rendering (same as B.5-c and B.7-c in PROGRESS.md).
    """

    def test_collision_detected_after_approach(self):
        """Two legs moving toward each other → body-body contacts detected."""
        leg_a = _build_go2_leg("L")
        leg_b = _build_go2_leg("R")

        terrain = FlatTerrain()
        merged = merge_models({"leg_a": leg_a, "leg_b": leg_b}, terrain=terrain)

        dt = 2e-4
        engine = CpuEngine(merged, dt=dt)

        # --- Initial state ---
        q = np.zeros(merged.nq)
        qdot = np.zeros(merged.nv)

        # Leg A: FreeJoint q = [qw,qx,qy,qz, px,py,pz] at indices [0:7]
        # Leg A joint angles: thigh q[7], calf q[8]
        # Leg B: FreeJoint q = [qw,qx,qy,qz, px,py,pz] at indices [9:16]
        # Leg B joint angles: thigh q[16], calf q[17]
        nq_a = leg_a.tree.nq  # 7 (FreeJoint) + 1 (thigh) + 1 (calf) = 9
        nv_a = leg_a.tree.nv  # 6 + 1 + 1 = 8

        # Leg A at x=-0.15, z=0.5 (in the air)
        q[0] = 1.0  # qw
        q[4] = -0.15  # px
        q[6] = 0.5  # pz

        # Leg B at x=+0.15, z=0.5
        q[nq_a + 0] = 1.0  # qw
        q[nq_a + 4] = 0.15  # px
        q[nq_a + 6] = 0.5  # pz

        # Initial velocities: moving toward each other at 2 m/s
        # FreeJoint qdot layout: [vx,vy,vz, wx,wy,wz] (linear-first spatial convention)
        qdot[0] = 2.0  # Leg A: vx = +2 m/s (rightward)
        qdot[nv_a + 0] = -2.0  # Leg B: vx = -2 m/s (leftward)

        # --- Simulate until collision ---
        # Gap = 0.30 m, closing speed = 4 m/s
        # Links have half-extent ~0.04 in x → collision when gap < ~0.08
        # Effective gap to close: 0.30 - 0.08 = 0.22 m
        # Time to collision: ~0.22/4 = 0.055 s = 55 ms
        # Simulate 80ms to be safe (400 steps at dt=2e-4)
        n_steps = 400
        contact_detected = False
        first_contact_step = None
        max_body_body_contacts = 0

        for step_i in range(n_steps):
            out = engine.step(q, qdot, np.zeros(merged.nv), dt)
            q = out.q_new
            qdot = out.qdot_new

            contacts = engine.query_contacts()
            bb_contacts = [c for c in contacts if c.body_j >= 0]
            if bb_contacts and not contact_detected:
                contact_detected = True
                first_contact_step = step_i
            max_body_body_contacts = max(max_body_body_contacts, len(bb_contacts))

        # --- Assertions ---
        assert contact_detected, "Two legs should collide when approaching each other"
        assert first_contact_step is not None
        # Contact should happen roughly when expected (not too early, not too late)
        assert first_contact_step > 100, f"Contact too early at step {first_contact_step}"
        assert first_contact_step < 380, f"Contact too late at step {first_contact_step}"
        assert max_body_body_contacts >= 1

    def test_no_collision_when_parallel(self):
        """Two legs moving in parallel (same direction) → no body-body contact."""
        leg_a = _build_go2_leg("L")
        leg_b = _build_go2_leg("R")

        terrain = FlatTerrain()
        merged = merge_models({"leg_a": leg_a, "leg_b": leg_b}, terrain=terrain)

        dt = 2e-4
        engine = CpuEngine(merged, dt=dt)

        q = np.zeros(merged.nq)
        qdot = np.zeros(merged.nv)

        nq_a = leg_a.tree.nq
        nv_a = leg_a.tree.nv

        # Leg A at x=-0.5, z=0.5
        q[0] = 1.0
        q[4] = -0.5
        q[6] = 0.5

        # Leg B at x=+0.5, z=0.5
        q[nq_a + 0] = 1.0
        q[nq_a + 4] = 0.5
        q[nq_a + 6] = 0.5

        # Both moving rightward at same speed → never collide
        qdot[0] = 1.0
        qdot[nv_a + 0] = 1.0

        for _ in range(200):
            out = engine.step(q, qdot, np.zeros(merged.nv), dt)
            q = out.q_new
            qdot = out.qdot_new

        contacts = engine.query_contacts()
        bb_contacts = [c for c in contacts if c.body_j >= 0]
        assert len(bb_contacts) == 0, "Parallel-moving legs should not collide"
