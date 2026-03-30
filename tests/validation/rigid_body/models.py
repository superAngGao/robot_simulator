"""
Shared robot model builders for validation tests.

Each builder produces BOTH our simulator model AND the equivalent MuJoCo XML,
ensuring identical mass/inertia/geometry/joints. This eliminates parameter
mismatch as a source of disagreement in cross-validation.

Conventions:
  - All models use SI units (kg, m, s)
  - Gravity = 9.81 m/s^2 downward (-z)
  - All revolute joints around Y axis unless noted
  - MuJoCo: integrator="Euler", cone="elliptic"
"""

from __future__ import annotations

import numpy as np

from physics.geometry import BodyCollisionGeometry, ShapeInstance, SphereShape
from physics.joint import FixedJoint, FreeJoint, RevoluteJoint
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from robot.model import RobotModel

G = 9.81
DT = 2e-4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(name, index, joint, mass, com, inertia_diag, X_tree, parent):
    """Convenience body builder."""
    return Body(
        name=name,
        index=index,
        joint=joint,
        inertia=SpatialInertia(
            mass=mass,
            inertia=np.diag(inertia_diag),
            com=np.array(com, dtype=float),
        ),
        X_tree=X_tree,
        parent=parent,
    )


def our_q_to_mujoco(q_ours, nq_per_body=7):
    """Convert our FreeJoint qpos [qw,qx,qy,qz,px,py,pz] to MuJoCo [px,py,pz,qw,qx,qy,qz]."""
    q_mj = np.zeros_like(q_ours)
    n_free = len(q_ours) // nq_per_body
    for i in range(n_free):
        base = i * nq_per_body
        q_mj[base : base + 3] = q_ours[base + 4 : base + 7]  # position
        q_mj[base + 3 : base + 7] = q_ours[base : base + 4]  # quaternion
    return q_mj


def mujoco_qpos_to_ours(qpos_mj, nq_per_body=7):
    """Convert MuJoCo qpos to our FreeJoint qpos layout."""
    q = np.zeros_like(qpos_mj)
    n_free = len(qpos_mj) // nq_per_body
    for i in range(n_free):
        base = i * nq_per_body
        q[base : base + 4] = qpos_mj[base + 3 : base + 7]  # quaternion
        q[base + 4 : base + 7] = qpos_mj[base : base + 3]  # position
    return q


# ---------------------------------------------------------------------------
# Model 1: Single Pendulum (1-DOF revolute, fixed base)
# ---------------------------------------------------------------------------


def build_single_pendulum():
    """Single pendulum: fixed base + 1 revolute link.

    Returns: (RobotModel, mujoco_xml_string)
    """
    mass = 1.0
    length = 0.5
    com_z = -length / 2  # CoM at midpoint of link
    # Thin rod: I_xx = I_zz = 1/12 * m * L^2, I_yy ≈ 0 (around rotation axis)
    I_rod = 1.0 / 12.0 * mass * length**2
    inertia = [I_rod, 1e-4, I_rod]

    tree = RobotTreeNumpy(gravity=G)
    tree.add_body(
        _body(
            "link1",
            0,
            RevoluteJoint("j1", axis=np.array([0, 1, 0])),
            mass,
            [0, 0, com_z],
            inertia,
            SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()
    model = RobotModel(tree=tree)

    xml = f"""<mujoco>
      <option timestep="{DT}" gravity="0 0 -{G}" integrator="Euler"/>
      <worldbody>
        <body name="link1" pos="0 0 0">
          <joint name="j1" type="hinge" axis="0 1 0"/>
          <inertial pos="0 0 {com_z}" mass="{mass}"
                    diaginertia="{inertia[0]} {inertia[1]} {inertia[2]}"/>
          <geom type="capsule" fromto="0 0 0 0 0 {-length}" size="0.02"/>
        </body>
      </worldbody>
    </mujoco>"""

    return model, xml


# ---------------------------------------------------------------------------
# Model 2: Double Pendulum (2-DOF revolute chain, fixed base)
# ---------------------------------------------------------------------------


def build_double_pendulum():
    """Double pendulum: fixed base + 2 revolute links.

    Returns: (RobotModel, mujoco_xml_string)
    """
    m1, m2 = 1.0, 0.5
    l1, l2 = 0.4, 0.3
    com1_z, com2_z = -l1 / 2, -l2 / 2
    I1 = 1.0 / 12.0 * m1 * l1**2
    I2 = 1.0 / 12.0 * m2 * l2**2
    inertia1 = [I1, 1e-4, I1]
    inertia2 = [I2, 1e-4, I2]

    tree = RobotTreeNumpy(gravity=G)
    tree.add_body(
        _body(
            "link1",
            0,
            RevoluteJoint("j1", axis=np.array([0, 1, 0])),
            m1,
            [0, 0, com1_z],
            inertia1,
            SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.add_body(
        _body(
            "link2",
            1,
            RevoluteJoint("j2", axis=np.array([0, 1, 0])),
            m2,
            [0, 0, com2_z],
            inertia2,
            SpatialTransform(R=np.eye(3), r=np.array([0, 0, -l1])),
            parent=0,
        )
    )
    tree.finalize()
    model = RobotModel(tree=tree)

    xml = f"""<mujoco>
      <option timestep="{DT}" gravity="0 0 -{G}" integrator="Euler"/>
      <worldbody>
        <body name="link1" pos="0 0 0">
          <joint name="j1" type="hinge" axis="0 1 0"/>
          <inertial pos="0 0 {com1_z}" mass="{m1}"
                    diaginertia="{inertia1[0]} {inertia1[1]} {inertia1[2]}"/>
          <geom type="capsule" fromto="0 0 0 0 0 {-l1}" size="0.02"/>
          <body name="link2" pos="0 0 {-l1}">
            <joint name="j2" type="hinge" axis="0 1 0"/>
            <inertial pos="0 0 {com2_z}" mass="{m2}"
                      diaginertia="{inertia2[0]} {inertia2[1]} {inertia2[2]}"/>
            <geom type="capsule" fromto="0 0 0 0 0 {-l2}" size="0.02"/>
          </body>
        </body>
      </worldbody>
    </mujoco>"""

    return model, xml


# ---------------------------------------------------------------------------
# Model 3: Quadruped (floating base + 4x3 revolute legs)
# ---------------------------------------------------------------------------

# Quadruped physical parameters
QUAD_BASE_MASS = 5.0
QUAD_BASE_SIZE = (0.35, 0.20, 0.10)  # half-extents for collision
QUAD_BASE_INERTIA = [0.05, 0.1, 0.05]

QUAD_HIP_MASS = 0.5
QUAD_HIP_LENGTH = 0.2
QUAD_HIP_INERTIA = [0.001, 0.001, 0.001]

QUAD_CALF_MASS = 0.3
QUAD_CALF_LENGTH = 0.2
QUAD_CALF_INERTIA = [0.001, 0.001, 0.001]

QUAD_FOOT_MASS = 0.05
QUAD_FOOT_INERTIA = [0.0001, 0.0001, 0.0001]
QUAD_FOOT_RADIUS = 0.02

# Hip attachment offsets from base center [x, y, z]
QUAD_HIP_OFFSETS = {
    "FL": [0.15, 0.10, 0.0],
    "FR": [0.15, -0.10, 0.0],
    "RL": [-0.15, 0.10, 0.0],
    "RR": [-0.15, -0.10, 0.0],
}


def build_quadruped(contact=False):
    """Simple quadruped: FreeJoint base + 4 legs × (hip + calf + foot).

    13 bodies, 18 DOF (6 free + 12 revolute).

    Args:
        contact: If True, add collision geometry + contact_body_names for feet.

    Returns: (RobotModel, mujoco_xml_string)
    """
    tree = RobotTreeNumpy(gravity=G)
    geometries = []
    contact_body_names = []
    body_idx = 0

    # Base body (FreeJoint)
    tree.add_body(
        _body(
            "base",
            body_idx,
            FreeJoint("root_joint"),
            QUAD_BASE_MASS,
            [0, 0, 0],
            QUAD_BASE_INERTIA,
            SpatialTransform.identity(),
            parent=-1,
        )
    )
    body_idx += 1

    # 4 legs
    for leg_name, hip_offset in QUAD_HIP_OFFSETS.items():
        hip_name = f"{leg_name}_hip"
        calf_name = f"{leg_name}_calf"
        foot_name = f"{leg_name}_foot"

        # Hip
        tree.add_body(
            _body(
                hip_name,
                body_idx,
                RevoluteJoint(
                    f"{leg_name}_hip_joint",
                    axis=np.array([0, 1, 0]),
                    q_min=-1.0,
                    q_max=1.0,
                    damping=0.1,
                ),
                QUAD_HIP_MASS,
                [0, 0, -QUAD_HIP_LENGTH / 2],
                QUAD_HIP_INERTIA,
                SpatialTransform(R=np.eye(3), r=np.array(hip_offset, dtype=float)),
                parent=0,
            )
        )
        hip_idx = body_idx
        body_idx += 1

        # Calf
        tree.add_body(
            _body(
                calf_name,
                body_idx,
                RevoluteJoint(
                    f"{leg_name}_calf_joint",
                    axis=np.array([0, 1, 0]),
                    q_min=-2.0,
                    q_max=0.5,
                    damping=0.1,
                ),
                QUAD_CALF_MASS,
                [0, 0, -QUAD_CALF_LENGTH / 2],
                QUAD_CALF_INERTIA,
                SpatialTransform(R=np.eye(3), r=np.array([0, 0, -QUAD_HIP_LENGTH])),
                parent=hip_idx,
            )
        )
        calf_idx = body_idx
        body_idx += 1

        # Foot (fixed to calf)
        tree.add_body(
            _body(
                foot_name,
                body_idx,
                FixedJoint(f"{leg_name}_foot_joint"),
                QUAD_FOOT_MASS,
                [0, 0, 0],
                QUAD_FOOT_INERTIA,
                SpatialTransform(R=np.eye(3), r=np.array([0, 0, -QUAD_CALF_LENGTH])),
                parent=calf_idx,
            )
        )
        if contact:
            geometries.append(
                BodyCollisionGeometry(
                    body_idx,
                    [ShapeInstance(SphereShape(QUAD_FOOT_RADIUS))],
                )
            )
            contact_body_names.append(foot_name)
        body_idx += 1

    tree.finalize()
    model = RobotModel(
        tree=tree,
        geometries=geometries if contact else [],
        contact_body_names=contact_body_names if contact else [],
    )

    # MuJoCo XML
    legs_xml = ""
    for leg_name, hip_offset in QUAD_HIP_OFFSETS.items():
        ox, oy, oz = hip_offset
        legs_xml += f"""
        <body name="{leg_name}_hip" pos="{ox} {oy} {oz}">
          <joint name="{leg_name}_hip_joint" type="hinge" axis="0 1 0"
                 range="-57.3 57.3" damping="0.1"/>
          <inertial pos="0 0 {-QUAD_HIP_LENGTH / 2}" mass="{QUAD_HIP_MASS}"
                    diaginertia="{QUAD_HIP_INERTIA[0]} {QUAD_HIP_INERTIA[1]} {QUAD_HIP_INERTIA[2]}"/>
          <geom type="capsule" fromto="0 0 0 0 0 {-QUAD_HIP_LENGTH}" size="0.02"
                contype="0" conaffinity="0"/>
          <body name="{leg_name}_calf" pos="0 0 {-QUAD_HIP_LENGTH}">
            <joint name="{leg_name}_calf_joint" type="hinge" axis="0 1 0"
                   range="-114.6 28.6" damping="0.1"/>
            <inertial pos="0 0 {-QUAD_CALF_LENGTH / 2}" mass="{QUAD_CALF_MASS}"
                      diaginertia="{QUAD_CALF_INERTIA[0]} {QUAD_CALF_INERTIA[1]} {QUAD_CALF_INERTIA[2]}"/>
            <geom type="capsule" fromto="0 0 0 0 0 {-QUAD_CALF_LENGTH}" size="0.02"
                  contype="0" conaffinity="0"/>
            <body name="{leg_name}_foot" pos="0 0 {-QUAD_CALF_LENGTH}">
              <inertial pos="0 0 0" mass="{QUAD_FOOT_MASS}"
                        diaginertia="{QUAD_FOOT_INERTIA[0]} {QUAD_FOOT_INERTIA[1]} {QUAD_FOOT_INERTIA[2]}"/>
              <geom type="sphere" size="{QUAD_FOOT_RADIUS}"
                    friction="0.8 0.005 0.0001"/>
            </body>
          </body>
        </body>"""

    if not contact:
        # Disable all contacts in MuJoCo when testing pure dynamics
        floor_geom = '<geom type="plane" size="10 10 0.1" contype="0" conaffinity="0"/>'
    else:
        floor_geom = '<geom type="plane" size="10 10 0.1" friction="0.8 0.005 0.0001"/>'

    xml = f"""<mujoco>
      <option timestep="{DT}" gravity="0 0 -{G}" integrator="Euler" cone="elliptic"/>
      <worldbody>
        {floor_geom}
        <body name="base" pos="0 0 0.5">
          <freejoint/>
          <inertial pos="0 0 0" mass="{QUAD_BASE_MASS}"
                    diaginertia="{QUAD_BASE_INERTIA[0]} {QUAD_BASE_INERTIA[1]} {QUAD_BASE_INERTIA[2]}"/>
          <geom type="box" size="{QUAD_BASE_SIZE[0]} {QUAD_BASE_SIZE[1]} {QUAD_BASE_SIZE[2]}"
                contype="0" conaffinity="0"/>
          {legs_xml}
        </body>
      </worldbody>
    </mujoco>"""

    return model, xml
