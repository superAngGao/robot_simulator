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
from physics.joint import FreeJoint
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
