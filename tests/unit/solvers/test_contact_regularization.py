"""Analytic oracles independent of MuJoCo and of the ADMM iteration."""

import numpy as np
import pytest

from physics.joint import FreeJoint
from physics.robot_tree import Body, RobotTreeNumpy
from physics.solvers.admm_qp import ADMMQPSolver
from physics.solvers.contact_regularization import impedance, reference_coefficients, validate_R_domain
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


@pytest.mark.parametrize("array_parameters", [False, True])
def test_coefficients_use_validated_expression(array_parameters):
    tree, q, v, contacts = fixture()
    ref = (0.02, 1e155)
    if array_parameters:
        ref = np.asarray(ref)
    s = ADMMQPSolver(regularization="mujoco", solref=ref, warmstart=False, tolerance=1e-12)
    b, k = reference_coefficients(ref, s.solimp, 0.002)
    assert np.isfinite(k) and k > 0
    contacts[0].depth = 0.001
    f, _ = s.solve(contacts, tree, q, v, np.zeros(tree.nv), 0.002)
    assert np.all(np.isfinite(f))
    # Mutating the same public field must use the same path as construction.
    s.solref = np.array([0.02, 1e160]) if array_parameters else (0.02, 1e160)
    old = s.last_R_diag
    cached = s._prev_f.copy()
    contacts[0].tangent1[:] = 7
    with pytest.raises(ValueError):
        s.solve(contacts, tree, q, v, np.zeros(tree.nv), 0.002)
    np.testing.assert_array_equal(contacts[0].tangent1, 7)
    np.testing.assert_array_equal(s._prev_f, cached)
    np.testing.assert_array_equal(s.last_R_diag, old)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_R_bounds(dtype):
    imp = (1e-4, 1e-4, 0.001, 0.5, 2)
    maxval = np.finfo(dtype).max
    validate_R_domain(np.array([maxval / 1e5, maxval / 1e5]), imp, 1, 1, dtype)
    with pytest.raises(ValueError):
        validate_R_domain(np.array([maxval / 1e3]), imp, 1, 1, dtype)
    with pytest.raises(ValueError):
        validate_R_domain(np.array([1.0]), (0.9, 0.95, 0.001, 0.5, 2), maxval, 3, dtype)


def test_decreasing_impedance_is_bounded_and_endpoint_exact():
    imp = (0.9999, 0.00013, 0.001, 0.5, 2)
    assert impedance(0.001, imp) == imp[1]
    assert impedance(0.0, imp) == imp[0]
    for depth in np.linspace(0.0, 0.001, 1001):
        d = impedance(depth, imp)
        assert imp[1] <= d <= imp[0]
    # The former CPU edge lies inside the new conservative roundoff exclusion.
    with pytest.raises(ValueError):
        validate_R_domain(np.array([1 / 8.556863841944613e-305] * 2), imp, 1, 1)
    validate_R_domain(np.array([1 / 1e-304] * 2), imp, 1, 1)


@pytest.mark.parametrize("mass,accepted", [(8.556863841944613e-305, False), (1e-304, True)])
def test_cpu_decreasing_native_upper_boundary(mass, accepted):
    tree = RobotTreeNumpy()
    for i in range(2):
        tree.add_body(
            Body(
                str(i),
                i,
                FreeJoint(str(i)),
                SpatialInertia(mass, np.eye(3) * (mass * 0.004), np.zeros(3)),
                SpatialTransform.identity(),
                parent=-1,
            )
        )
    tree.finalize()
    q, v = tree.default_state()
    c = ContactConstraint(
        0, 1, np.zeros(3), np.array([0.0, 0.0, 1.0]), np.full(3, 7.0), np.zeros(3), 0.01, 0.5, condim=1
    )
    s = ADMMQPSolver(regularization="mujoco", solimp=(0.9999, 0.00013, 0.001, 0.5, 2), max_iter=1)
    if accepted:
        s.solve([c], tree, q, v, np.zeros(tree.nv), 0.002)
        np.testing.assert_allclose(s.last_R_diag, [(1 - 0.00013) / 0.00013 * (2 / mass)], rtol=1e-10)
        assert np.all(np.isfinite(s.last_forces))
    else:
        with pytest.raises(ValueError):
            s.solve([c], tree, q, v, np.zeros(tree.nv), 0.002)
        np.testing.assert_array_equal(c.tangent1, 7.0)
        assert s._prev_f is None
