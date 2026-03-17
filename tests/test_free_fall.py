"""
Free-fall accuracy test: analytic vs ABA.

A single free-floating body (no joints, no contact) dropped from rest.
Analytic solution: z(t) = z0 - 0.5 * g * t²

Reference: Featherstone (2008) §7.3
"""

import numpy as np

from physics.integrator import SemiImplicitEuler
from physics.joint import FreeJoint
from physics.robot_tree import Body, RobotTree
from physics.spatial import SpatialInertia, SpatialTransform


def _make_free_body(gravity: float = 9.81) -> RobotTree:
    """Single free-floating body, 1 kg, unit inertia."""
    tree = RobotTree(gravity=gravity)
    body = Body(
        name="base",
        index=0,
        joint=FreeJoint("root"),
        inertia=SpatialInertia(mass=1.0, inertia=np.eye(3), com=np.zeros(3)),
        X_tree=SpatialTransform.identity(),
        parent=-1,
    )
    tree.add_body(body)
    tree.finalize()
    return tree


def test_free_fall_accuracy():
    """Simulated free-fall z position must match analytic solution within 5 mm at t=1 s."""
    g = 9.81
    z0 = 1.0
    tree = _make_free_body(gravity=g)

    q, qdot = tree.default_state()
    # FreeJoint q layout: [qx, qy, qz, qw, px, py, pz]
    q[3] = 1.0  # qw = 1 (identity quaternion)
    q[6] = z0  # pz = z0

    dt = 2e-4
    duration = 1.0
    n_steps = int(duration / dt)
    integrator = SemiImplicitEuler(dt)
    tau = np.zeros(tree.nv)

    for _ in range(n_steps):
        q, qdot = integrator.step(tree, q, qdot, tau)

    z_sim = q[6]
    z_analytic = z0 - 0.5 * g * duration**2
    error = abs(z_sim - z_analytic)

    assert error < 5e-3, f"Free-fall error {error:.4f} m exceeds 5 mm tolerance"


def test_free_fall_no_horizontal_drift():
    """A body dropped from rest must not drift in x or y."""
    tree = _make_free_body()
    q, qdot = tree.default_state()
    q[3] = 1.0
    q[6] = 1.0

    dt = 2e-4
    integrator = SemiImplicitEuler(dt)
    tau = np.zeros(tree.nv)

    for _ in range(500):
        q, qdot = integrator.step(tree, q, qdot, tau)

    assert abs(q[4]) < 1e-10, f"Unexpected x drift: {q[4]}"
    assert abs(q[5]) < 1e-10, f"Unexpected y drift: {q[5]}"
