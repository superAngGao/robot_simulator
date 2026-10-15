"""Analytic oracles independent of MuJoCo and of the ADMM iteration."""

import numpy as np
import pytest

from physics.joint import FreeJoint
from physics.robot_tree import Body, RobotTreeNumpy
from physics.solvers.admm_qp import ADMMQPSolver
from physics.solvers.pgs_solver import ContactConstraint
from physics.spatial import SpatialInertia, SpatialTransform


def fixture():
    tree = RobotTreeNumpy()
    tree.add_body(
        Body(
            "body",
            0,
            FreeJoint("free"),
            SpatialInertia(2.0, np.eye(3) * 0.04, np.zeros(3)),
            SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()
    q, v = tree.default_state()
    contacts = [
        ContactConstraint(
            0,
            -1,
            np.array([x, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.zeros(3),
            np.zeros(3),
            0.0,
            0.5,
            condim=1,
        )
        for x in (-0.1, 0.1)
    ]
    return tree, q, v, contacts


@pytest.mark.parametrize("policy", ["per_row", "mujoco"])
@pytest.mark.parametrize("rho", [0.1, 1.0, 10.0])
def test_coupled_normals_active_set_oracle(policy, rho):
    tree, q, v, contacts = fixture()
    au = np.array([0.0, 0.0, -1.0, 0.0, 10.0, 0.0])
    s = ADMMQPSolver(
        regularization=policy,
        rho=rho,
        max_iter=3000,
        tolerance=1e-12,
        warmstart=False,
        solimp=(0.9, 0.9, 0.001, 0.5, 2),
    )
    f, J = s.solve(contacts, tree, q, v, au, 0.002)
    # Analytic mobility: inverse mass=.5, lever^2/I=.25. rhs=(0,2).
    r = (0.75 if policy == "per_row" else 0.5) / 9
    H = np.array([[0.75 + r, 0.25], [0.25, 0.75 + r]])
    expected = np.array([0.0, 2 / (0.75 + r)])
    gradient = H @ expected - np.array([0.0, 2.0])
    assert np.all(gradient >= 0)
    np.testing.assert_array_equal(gradient * expected, 0)
    np.testing.assert_allclose(f, expected, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(s.last_R_diag, [r, r], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    "field,value",
    [
        ("solref", (0.02, np.nan)),
        ("solref", (-1, 1)),
        ("solref", (0.02, 1e-300)),
        ("solimp", (0.9, 0.95, np.inf, 0.5, 2)),
    ],
)
def test_reject_before_contact_and_cache_mutation(field, value):
    tree, q, v, contacts = fixture()
    s = ADMMQPSolver(regularization="mujoco")
    s.solve(contacts, tree, q, v, np.zeros(tree.nv), 0.002)
    before = s._prev_f.copy()
    contacts[0].tangent1[:] = 7
    setattr(s, field, value)
    with pytest.raises(ValueError):
        s.solve(contacts, tree, q, v, np.zeros(tree.nv), 0.002)
    np.testing.assert_array_equal(s._prev_f, before)
    np.testing.assert_array_equal(contacts[0].tangent1, 7)
