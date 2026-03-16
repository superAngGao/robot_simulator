"""
Simple quadruped robot — Phase 1 validation example.

Robot description
-----------------
A minimal symmetric quadruped with 12 revolute joints + 4 fixed foot bodies:

    Torso (floating base)
    ├── FL_hip  →  FL_thigh  →  FL_calf  ──(fixed)──  FL_foot
    ├── FR_hip  →  FR_thigh  →  FR_calf  ──(fixed)──  FR_foot
    ├── RL_hip  →  RL_thigh  →  RL_calf  ──(fixed)──  RL_foot
    └── RR_hip  →  RR_thigh  →  RR_calf  ──(fixed)──  RR_foot

Each leg has 3 revolute joints:
    - Hip   : abduction/adduction (rotation about X)
    - Thigh : flexion/extension   (rotation about Y)
    - Calf  : knee flexion        (rotation about Y)

Each foot is a separate rigid body (small cylinder, mass=0.05 kg) attached to
the calf tip via a FixedJoint with offset [0, 0, -CALF_LENGTH].  The contact
point is placed at the foot body's origin, giving geometrically accurate
ground contact (zero error vs. Phase 1's 0.2 m offset).

Fixes vs. Phase 1
-----------------
1. Foot body at calf tip — contact point is now geometrically correct.
2. Joint limits on all revolute joints (penalty spring-damper).
3. AABB self-collision detection — legs cannot penetrate the torso.

Run
---
    python -m robot_simulator.examples.simple_quadruped
    python -m robot_simulator.examples.simple_quadruped --save out.gif
"""

from __future__ import annotations

import argparse

import numpy as np

from robot_simulator.physics import (
    AABBSelfCollision,
    Axis,
    Body,
    BodyAABB,
    ContactModel,
    ContactParams,
    ContactPoint,
    FixedJoint,
    FreeJoint,
    RevoluteJoint,
    RobotTree,
    SemiImplicitEuler,
    SpatialInertia,
    SpatialTransform,
)
from robot_simulator.rendering import RobotViewer

# ---------------------------------------------------------------------------
# Robot geometry / inertia constants
# ---------------------------------------------------------------------------

TORSO_MASS = 8.0
TORSO_SIZE = (0.35, 0.20, 0.10)  # lx, ly, lz  [m]

THIGH_MASS = 0.8
THIGH_LENGTH = 0.20  # m

CALF_MASS = 0.4
CALF_LENGTH = 0.20  # m

FOOT_MASS = 0.05  # kg — small foot body at calf tip
FOOT_RADIUS = 0.02  # m  — sphere radius (for inertia; contact treated as point)

HIP_OFFSET_X = 0.18  # fore-aft offset of hip from torso centre [m]
HIP_OFFSET_Y = 0.12  # lateral offset [m]
HIP_OFFSET_Z = 0.0

# Initial joint angles: straight legs (vertical stance)
THIGH_ANGLE = 0.0
CALF_ANGLE = 0.0

# ---------------------------------------------------------------------------
# Joint limits [rad]
# ---------------------------------------------------------------------------
# Hip (abduction / adduction, X-axis): ±35°
HIP_Q_MIN, HIP_Q_MAX = -0.61, 0.61
# Thigh (flexion / extension, Y-axis): −90° … +90°
THIGH_Q_MIN, THIGH_Q_MAX = -1.57, 1.57
# Calf / knee (Y-axis, bends forward): −150° … +30°
CALF_Q_MIN, CALF_Q_MAX = -2.62, 0.52

# Limit spring stiffness and damping
K_LIMIT = 5_000.0  # N·m / rad
B_LIMIT = 50.0  # N·m·s / rad


# ---------------------------------------------------------------------------
# Build robot tree
# ---------------------------------------------------------------------------


def build_quadruped() -> tuple[RobotTree, ContactModel, AABBSelfCollision]:
    """Construct the quadruped RobotTree, ContactModel, and self-collision model."""
    tree = RobotTree(gravity=9.81)

    # --- Torso (root, floating base) ---
    torso_inertia = SpatialInertia.from_box(TORSO_MASS, *TORSO_SIZE)
    torso = Body(
        name="torso",
        index=0,
        joint=FreeJoint("root"),
        inertia=torso_inertia,
        X_tree=SpatialTransform.identity(),
        parent=-1,
    )
    torso_idx = tree.add_body(torso)

    contact_model = ContactModel(
        ContactParams(
            k_normal=1_000.0,
            b_normal=300.0,
            mu=0.7,
        )
    )

    self_collision = AABBSelfCollision(k_contact=3_000.0, b_contact=150.0)

    # Torso AABB: half the torso box dimensions
    self_collision.add_body(
        BodyAABB(
            body_index=torso_idx,
            half_extents=np.array([TORSO_SIZE[0] / 2, TORSO_SIZE[1] / 2, TORSO_SIZE[2] / 2]),
        )
    )

    # --- Leg inertias ---
    thigh_inertia = SpatialInertia.from_cylinder(THIGH_MASS, 0.02, THIGH_LENGTH)
    calf_inertia = SpatialInertia.from_cylinder(CALF_MASS, 0.015, CALF_LENGTH)
    # Foot: small sphere approximated as a short cylinder
    foot_inertia = SpatialInertia.from_cylinder(FOOT_MASS, FOOT_RADIUS, FOOT_RADIUS * 2)

    leg_signs = {
        "FL": (+1, +1),  # fore-left
        "FR": (+1, -1),  # fore-right
        "RL": (-1, +1),  # rear-left
        "RR": (-1, -1),  # rear-right
    }

    for leg_name, (sx, sy) in leg_signs.items():
        # --- Hip body (abduction joint, axis=X) ---
        hip_offset = np.array([sx * HIP_OFFSET_X, sy * HIP_OFFSET_Y, HIP_OFFSET_Z])
        hip = Body(
            name=f"{leg_name}_hip",
            index=0,
            joint=RevoluteJoint(
                f"{leg_name}_hip_joint",
                Axis.X,
                q_min=HIP_Q_MIN,
                q_max=HIP_Q_MAX,
                k_limit=K_LIMIT,
                b_limit=B_LIMIT,
            ),
            inertia=SpatialInertia.point_mass(0.3, np.zeros(3)),
            X_tree=SpatialTransform(np.eye(3), hip_offset),
            parent=torso_idx,
        )
        hip_idx = tree.add_body(hip)

        # --- Thigh body (flexion joint, axis=Y) ---
        thigh = Body(
            name=f"{leg_name}_thigh",
            index=0,
            joint=RevoluteJoint(
                f"{leg_name}_thigh_joint",
                Axis.Y,
                q_min=THIGH_Q_MIN,
                q_max=THIGH_Q_MAX,
                k_limit=K_LIMIT,
                b_limit=B_LIMIT,
            ),
            inertia=thigh_inertia,
            X_tree=SpatialTransform(np.eye(3), np.zeros(3)),
            parent=hip_idx,
        )
        thigh_idx = tree.add_body(thigh)

        # --- Calf body (knee joint, axis=Y) ---
        calf = Body(
            name=f"{leg_name}_calf",
            index=0,
            joint=RevoluteJoint(
                f"{leg_name}_calf_joint",
                Axis.Y,
                q_min=CALF_Q_MIN,
                q_max=CALF_Q_MAX,
                k_limit=K_LIMIT,
                b_limit=B_LIMIT,
            ),
            inertia=calf_inertia,
            X_tree=SpatialTransform(np.eye(3), np.array([0.0, 0.0, -THIGH_LENGTH])),
            parent=thigh_idx,
        )
        calf_idx = tree.add_body(calf)

        # --- Foot body: separate rigid body at calf tip via FixedJoint ---
        # X_tree places the foot origin at the calf tip (0, 0, -CALF_LENGTH).
        # The contact point sits at the foot body origin → geometrically exact.
        foot = Body(
            name=f"{leg_name}_foot",
            index=0,
            joint=FixedJoint(f"{leg_name}_foot_joint"),
            inertia=foot_inertia,
            X_tree=SpatialTransform(np.eye(3), np.array([0.0, 0.0, -CALF_LENGTH])),
            parent=calf_idx,
        )
        foot_idx = tree.add_body(foot)

        # Contact point at foot body origin (= geometric foot tip)
        contact_model.add_contact_point(
            ContactPoint(
                body_index=foot_idx,
                position=np.zeros(3),
                name=f"{leg_name}_foot",
            )
        )

        # AABB for calf only (thinner than torso, relevant for body penetration)
        self_collision.add_body(
            BodyAABB(
                body_index=calf_idx,
                half_extents=np.array([0.02, 0.02, CALF_LENGTH / 2]),
            )
        )

    tree.finalize()

    # Build collision pairs after all bodies are registered
    parent_list = [b.parent for b in tree.bodies]
    self_collision.build_pairs(parent_list)

    return tree, contact_model, self_collision


# ---------------------------------------------------------------------------
# Initial state: standing pose
# ---------------------------------------------------------------------------


def standing_state(tree: RobotTree) -> tuple[np.ndarray, np.ndarray]:
    """Return q, qdot for a symmetric standing configuration.

    Uses FK to measure the actual foot-tip position and sets the torso
    height so all feet land exactly at ground level (z = 0) + 1 mm clearance.
    """
    q, qdot = tree.default_state()

    for leg in ("FL", "FR", "RL", "RR"):
        thigh_body = tree.body_by_name(f"{leg}_thigh")
        calf_body = tree.body_by_name(f"{leg}_calf")
        q[thigh_body.q_idx] = THIGH_ANGLE
        q[calf_body.q_idx] = CALF_ANGLE

    # Run FK with torso at z=0 to measure lowest foot position
    root_body = tree.body_by_name("torso")
    q[root_body.q_idx][6] = 0.0
    X_world = tree.forward_kinematics(q)

    min_foot_z = 0.0
    for leg in ("FL", "FR", "RL", "RR"):
        foot_body = tree.body_by_name(f"{leg}_foot")
        foot_pos = X_world[foot_body.index].r  # foot body origin = contact point
        min_foot_z = min(min_foot_z, foot_pos[2])

    clearance = 0.001  # 1 mm — start just touching the ground
    q[root_body.q_idx][6] = -min_foot_z + clearance

    return q, qdot


# ---------------------------------------------------------------------------
# Body-velocity helper (needed for contact + self-collision damping)
# ---------------------------------------------------------------------------


def _compute_body_velocities(
    tree: RobotTree,
    q: np.ndarray,
    qdot: np.ndarray,
) -> list:
    """Return a list of spatial velocities (body frame) for every body."""
    v_bodies = []
    for body in tree.bodies:
        S = body.joint.motion_subspace(q[body.q_idx])
        vJ = S @ qdot[body.v_idx] if S.shape[1] > 0 else np.zeros(6)
        if body.parent < 0:
            v_bodies.append(vJ)
        else:
            Xup = body.X_tree @ body.joint.transform(q[body.q_idx])
            v_bodies.append(Xup.apply_velocity(v_bodies[body.parent]) + vJ)
    return v_bodies


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(save_path: str | None = None) -> None:
    print("Building quadruped...")
    tree, contact_model, self_collision = build_quadruped()
    print(tree)
    print(f"  nq={tree.nq}, nv={tree.nv}")
    print(f"  Contact points : {[cp.name for cp in contact_model.contact_points]}")
    print(f"  Self-collision : {self_collision}")

    q, qdot = standing_state(tree)

    nv = tree.nv
    KP, KD = 50.0, 5.0

    def controller(t: float, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        tau = np.zeros(nv, dtype=np.float64)
        for body in tree.bodies:
            j = body.joint
            if j.nv == 0 or isinstance(j, FreeJoint):
                continue
            tau[body.v_idx] = -KP * q[body.q_idx] - KD * qdot[body.v_idx]
        # Add joint-limit restoring torques
        tau += tree.joint_limit_torques(q, qdot)
        return tau

    dt = 2e-4
    duration = 2.0
    n_steps = int(duration / dt)

    print(f"\nSimulating {duration}s at dt={dt}s ({n_steps} steps)...")
    integrator = SemiImplicitEuler(dt)

    times = np.zeros(n_steps)
    qs = np.zeros((n_steps, tree.nq))

    for i in range(n_steps):
        times[i] = i * dt
        qs[i] = q

        tau = controller(times[i], q, qdot)
        X_world = tree.forward_kinematics(q)
        v_bodies = _compute_body_velocities(tree, q, qdot)

        # Ground contact forces
        contact_forces = contact_model.compute_forces(X_world, v_bodies, tree.num_bodies)
        # AABB self-collision forces
        sc_forces = self_collision.compute_forces(X_world, v_bodies, tree.num_bodies)

        # Merge external forces
        ext_forces = [cf + scf for cf, scf in zip(contact_forces, sc_forces)]

        q, qdot = integrator.step(tree, q, qdot, tau, ext_forces)

        if i % 200 == 0:
            torso = tree.body_by_name("torso")
            z = q[torso.q_idx][6]
            active = contact_model.active_contacts(X_world)
            print(f"  t={times[i]:.2f}s  torso_z={z:.3f}m  contacts={[c[0] for c in active]}")

    print("\nRendering animation...")
    viewer = RobotViewer(
        tree,
        floor_size=0.6,
        contact_names=[f"{l}_calf" for l in ("FL", "FR", "RL", "RR")],
    )

    stride = max(1, int(1 / (50 * dt)))
    viewer.animate(
        times[::stride],
        qs[::stride],
        interval=20,
        title="Quadruped — passive drop test (foot bodies + joint limits + AABB)",
        show=save_path is None,
        save_path=save_path,
    )

    if save_path:
        print(f"Saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quadruped Phase 1 demo")
    parser.add_argument(
        "--save", metavar="PATH", default=None, help="Save animation to .gif or .mp4 (requires pillow/ffmpeg)"
    )
    args = parser.parse_args()
    main(save_path=args.save)
