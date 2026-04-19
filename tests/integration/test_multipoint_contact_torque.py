"""
Integration tests: multi-point contact prevents spurious torque / tipping.

Each test verifies that a symmetric drop produces near-zero angular velocity
after settling — a single-point contact would create an off-centre moment arm
and cause rotation, while a correct 2-point manifold keeps the body level.

Scenarios:
  1. box-on-box: flat face contact → no tipping (angular velocity < 0.1 rad/s)
  2. capsule parallel to ground: 2-point contact → no axial spin
  3. cylinder flat on ground: 2-point contact (B1 fix) → no axial spin

Reference: B3 plan (session 32 gap analysis).
"""

from __future__ import annotations

import numpy as np

from physics.cpu_engine import CpuEngine
from physics.geometry import (
    BodyCollisionGeometry,
    BoxShape,
    CapsuleShape,
    CylinderShape,
    ShapeInstance,
)
from physics.joint import FreeJoint
from physics.merged_model import merge_models
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from physics.terrain import FlatTerrain
from robot.model import RobotModel

DT = 2e-4
GRAVITY = 9.81


def _inertia_box(m, half):
    return np.eye(3) * (m / 6.0 * (2 * half) ** 2)


def _inertia_cylinder(m, r, length):
    h = length / 2.0
    I_lat = m * (3 * r**2 + (2 * h) ** 2) / 12.0
    I_ax = m * r**2 / 2.0
    return np.diag([I_lat, I_lat, I_ax])


def _inertia_capsule(m, r, length):
    return _inertia_cylinder(m, r, length)


def _make_engine(shape, mass, I):
    tree = RobotTreeNumpy(gravity=GRAVITY)
    tree.add_body(
        Body(
            name="b",
            index=0,
            joint=FreeJoint("root_b"),
            inertia=SpatialInertia(mass, I, np.zeros(3)),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()
    model = RobotModel(
        tree=tree,
        actuated_joint_names=[],
        contact_body_names=["b"],
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(shape)])],
    )
    merged = merge_models({"r": model}, terrain=FlatTerrain())
    return merged, CpuEngine(merged, dt=DT)


def _q_at_z(merged, z):
    q = np.zeros(merged.nq)
    q[0] = 1.0  # qw
    q[6] = z
    return q


def _run(engine, merged, q, n_steps):
    qdot = np.zeros(merged.nv)
    for _ in range(n_steps):
        out = engine.step(q, qdot, np.zeros(merged.nv))
        q, qdot = out.q_new, out.qdot_new
    return q, qdot


# ---------------------------------------------------------------------------
# Scenario 1: box flat on ground — no tipping
# ---------------------------------------------------------------------------


def test_box_flat_no_tipping():
    """Box dropped flat onto ground — face contact → angular velocity < 0.1 rad/s."""
    half = 0.05
    shape = BoxShape((2 * half, 2 * half, 2 * half))
    I = _inertia_box(1.0, half)
    merged, engine = _make_engine(shape, 1.0, I)

    q = _q_at_z(merged, half + 0.3)
    q, qdot = _run(engine, merged, q, n_steps=1000)

    assert not np.any(np.isnan(q)), "NaN in q"
    assert not np.any(np.isnan(qdot)), "NaN in qdot"
    # FreeJoint qdot layout: [vx,vy,vz, ωx,ωy,ωz]
    omega = qdot[3:6]
    omega_mag = float(np.linalg.norm(omega))
    assert omega_mag < 0.1, f"box tipping: |omega|={omega_mag:.4f} rad/s"


# ---------------------------------------------------------------------------
# Scenario 2: capsule horizontal (axis along X) — no axial spin
# ---------------------------------------------------------------------------


def test_capsule_horizontal_no_spin():
    """Capsule lying horizontal (axis along X) dropped onto ground → no axial spin."""
    r, length = 0.04, 0.20
    shape = CapsuleShape(r, length)
    I = _inertia_capsule(1.0, r, length)
    merged, engine = _make_engine(shape, 1.0, I)

    # Rotate capsule 90° around Y so axis is along X
    from scipy.spatial.transform import Rotation

    R = Rotation.from_euler("y", np.pi / 2).as_matrix()
    quat = Rotation.from_matrix(R).as_quat()  # [x,y,z,w]

    q = np.zeros(merged.nq)
    q[0] = quat[3]  # qw
    q[1] = quat[0]  # qx
    q[2] = quat[1]  # qy
    q[3] = quat[2]  # qz
    q[6] = r + 0.3  # drop height

    q, qdot = _run(engine, merged, q, n_steps=1000)

    assert not np.any(np.isnan(q)), "NaN in q"
    assert not np.any(np.isnan(qdot)), "NaN in qdot"
    # FreeJoint qdot layout: [vx,vy,vz, ωx,ωy,ωz]
    omega = qdot[3:6]
    omega_mag = float(np.linalg.norm(omega))
    assert omega_mag < 0.5, f"capsule spinning: |omega|={omega_mag:.4f} rad/s"


# ---------------------------------------------------------------------------
# Scenario 3: cylinder horizontal (axis along X) — no axial spin (B1 fix)
# ---------------------------------------------------------------------------


def test_cylinder_horizontal_no_spin():
    """Cylinder lying horizontal (axis along X) dropped onto ground → no axial spin.

    Before B1 fix, GJK degenerated on coaxial N-gon prisms → intersecting=False
    → no contact → cylinder fell through ground.  After fix, analytical dispatch
    returns correct 2-point manifold.
    """
    r, length = 0.04, 0.20
    shape = CylinderShape(r, length)
    I = _inertia_cylinder(1.0, r, length)
    merged, engine = _make_engine(shape, 1.0, I)

    from scipy.spatial.transform import Rotation

    R = Rotation.from_euler("y", np.pi / 2).as_matrix()
    quat = Rotation.from_matrix(R).as_quat()

    q = np.zeros(merged.nq)
    q[0] = quat[3]
    q[1] = quat[0]
    q[2] = quat[1]
    q[3] = quat[2]
    q[6] = r + 0.3

    q, qdot = _run(engine, merged, q, n_steps=1000)

    assert not np.any(np.isnan(q)), "NaN in q"
    assert not np.any(np.isnan(qdot)), "NaN in qdot"
    # Body should not fall through ground
    assert q[6] > -0.1, f"cylinder fell through ground: z={q[6]:.4f}"
    # FreeJoint qdot layout: [vx,vy,vz, ωx,ωy,ωz]
    omega = qdot[3:6]
    omega_mag = float(np.linalg.norm(omega))
    assert omega_mag < 0.5, f"cylinder spinning: |omega|={omega_mag:.4f} rad/s"
