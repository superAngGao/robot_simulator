"""Analytical counterexamples for articulated CPU PGS contact mobility."""

import numpy as np
import pytest

from physics.constraint_solvers import VelocityLevelAdapter
from physics.cpu_engine import CpuEngine
from physics.dynamics_utils import body_velocity_jacobians, impulses_to_generalized
from physics.geometry import BodyCollisionGeometry, ShapeInstance, SphereShape
from physics.implicit_contact_step import ImplicitContactStep
from physics.joint import FixedJoint, FreeJoint, RevoluteJoint
from physics.merged_model import merge_models
from physics.robot_tree import Body, RobotTreeNumpy
from physics.solvers.pgs_solver import ContactConstraint, PGSContactSolver
from physics.solvers.pgs_split_impulse import PGSSplitImpulseSolver
from physics.spatial import SpatialInertia, SpatialTransform
from physics.step_pipeline import StepPipeline
from robot.model import RobotModel

SOLVERS = [PGSContactSolver, PGSSplitImpulseSolver]


def composite(child_mass, offset=None, com=None):
    tree = RobotTreeNumpy(gravity=0.0)
    offset = np.zeros(3) if offset is None else np.asarray(offset)
    com = np.zeros(3) if com is None else np.asarray(com)
    for i, mass in enumerate([1 - child_mass, child_mass]):
        tree.add_body(
            Body(
                name=f"b{i}",
                index=i,
                parent=i - 1,
                joint=FreeJoint("root") if i == 0 else FixedJoint("weld"),
                inertia=SpatialInertia(mass, np.eye(3) * 0.01, com),
                X_tree=SpatialTransform.from_translation(np.zeros(3) if i == 0 else offset),
            )
        )
    tree.finalize()
    return RobotModel(
        tree=tree,
        contact_body_names=["b1"],
        geometries=[BodyCollisionGeometry(1, [ShapeInstance(SphereShape(0.1))])],
    )


def solver(cls, **kwargs):
    return cls(max_iter=500, tolerance=1e-14, erp=0.0, cfm=0.0, **kwargs)


def contact(i, j, point, normal, condim=1, **kwargs):
    return ContactConstraint(
        body_i=i,
        body_j=j,
        point=np.asarray(point, dtype=float),
        normal=np.asarray(normal, dtype=float),
        tangent1=np.zeros(3),
        tangent2=np.zeros(3),
        depth=0.001,
        mu=0.0,
        condim=condim,
        **kwargs,
    )


@pytest.mark.parametrize("cls", SOLVERS)
@pytest.mark.parametrize("child_mass", [0.05, 0.5, 0.95])
def test_fixed_child_mass_partition_does_not_change_impact(cls, child_mass):
    merged = merge_models({"r": composite(child_mass)})
    engine = CpuEngine(merged, solver=solver(cls), dt=0.001)
    q, v = merged.tree.default_state()
    q[6] = 0.099
    v[2] = -1.0
    out = engine.step(q, v, np.zeros(merged.nv))
    assert len(engine._last_contacts) == 1
    np.testing.assert_allclose(out.qdot_new, np.zeros(6), atol=1e-10, rtol=0)
    # Total mass 1 kg: delta momentum must be +1 N s in the normal direction.
    assert abs((out.qdot_new[2] - v[2]) - 1.0) < 1e-10


@pytest.mark.parametrize("cls", SOLVERS)
def test_legacy_articulated_entrypoint_uses_same_inertia(cls):
    tree = composite(0.05).tree
    q, v = tree.default_state()
    q[6] = 0.099
    v[2] = -1
    c = contact(1, -1, [0, 0, -0.001], [0, 0, 1])
    step = ImplicitContactStep(0.001, solver(cls))
    _, new_v = step.step(tree, q, v, np.zeros(tree.nv), contacts=[c])
    np.testing.assert_allclose(new_v, 0, atol=1e-10, rtol=0)


@pytest.mark.parametrize("cls", SOLVERS)
def test_revolute_contact_removes_incoming_velocity(cls):
    tree = RobotTreeNumpy(gravity=0.0)
    tree.add_body(
        Body(
            name="link",
            index=0,
            parent=-1,
            joint=RevoluteJoint("hinge", axis=np.array([0.0, 1.0, 0.0])),
            inertia=SpatialInertia(1.0, np.eye(3) * 0.1, np.array([0.0, 0.0, -0.5])),
            X_tree=SpatialTransform.identity(),
        )
    )
    tree.finalize()
    q, v = tree.default_state()
    v[0] = 1.0
    c = contact(0, -1, [0, 0, -1], [1, 0, 0])
    _, new_v = StepPipeline(0.001, constraint_solver=VelocityLevelAdapter(solver(cls))).step(
        tree, q, v, np.zeros(1), [c]
    )
    # I_y about hinge=.1+1*.5^2=.35; unit moment arm, impulse=.35 N s.
    assert abs(new_v[0]) < 1e-10


@pytest.mark.parametrize("cls", SOLVERS)
def test_offset_composite_impact_matches_rigid_body_formula(cls):
    model = composite(0.2, offset=[0.1, 0, 0], com=[0.02, 0.01, 0])
    merged = merge_models({"r": model})
    merged.terrain.mu = 0.0
    engine = CpuEngine(merged, solver=solver(cls), dt=0.001)
    q, v = merged.tree.default_state()
    angle = 0.7
    q[:4] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
    q[6] = 0.099
    v[2] = -1
    # Analytic composite inertia about CoM; contact relative to CoM in base frame.
    c1 = np.array([0.02, 0.01, 0])
    c2 = c1 + np.array([0.1, 0, 0])
    center = 0.8 * c1 + 0.2 * c2
    inertia = np.eye(3) * 0.02
    for m, c in [(0.8, c1), (0.2, c2)]:
        r = c - center
        inertia += m * (r @ r * np.eye(3) - np.outer(r, r))
    arm = np.array([0.1, 0, -0.1]) - center
    normal = np.array([0.0, 0.0, 1.0])
    moment = np.cross(arm, normal)
    impulse = 1.0 / (1.0 + moment @ np.linalg.solve(inertia, moment))
    omega = np.linalg.solve(inertia, moment * impulse)
    v_com = np.array([0.0, 0.0, -1.0 + impulse])
    rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.0]])
    v_origin = rot @ (v_com - np.cross(omega, center))
    expected = np.r_[v_origin, omega]
    out = engine.step(q, v, np.zeros(6))
    np.testing.assert_allclose(out.qdot_new, expected, atol=1e-9, rtol=0)


@pytest.mark.parametrize("cls", SOLVERS)
@pytest.mark.parametrize("condim", [1, 3, 4, 6])
def test_two_roots_conserve_momentum_with_angular_rows(cls, condim):
    merged = merge_models({"a": composite(0.05), "b": composite(0.8)})
    tree = merged.tree
    q, v = tree.default_state()
    # Both origins and contact coincide: isolated normal + angular rows.
    v[2] = -1.0
    v[8] = 1.0
    if condim >= 4:
        v[5] = 2.0
        v[11] = -2.0
    if condim == 6:
        v[3] = 0.2
        v[9] = -0.2
    c = contact(1, 3, [0, 0, 0], [0, 0, 1], condim, mu_spin=1.0, mu_roll=1.0)
    _, new = StepPipeline(
        0.001, constraint_solver=VelocityLevelAdapter(solver(cls, solimp=(1.0, 1.0, 0.001, 0.5, 2.0)))
    ).step(tree, q, v, np.zeros(12), [c])
    np.testing.assert_allclose(new, 0.0, atol=1e-10, rtol=0)


def test_body_jacobians_match_velocities_and_virtual_work():
    from tests.validation.rigid_body.models import build_quadruped

    model, _ = build_quadruped(contact=False)
    tree = merge_models({"quad": model, "weld": composite(0.2, [0.2, -0.1, 0.3])}).tree
    tree.bodies[-1].X_tree.R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    q, v = tree.default_state()
    q[:4] = np.array([1.0, 0.2, -0.3, 0.1]) / np.linalg.norm([1.0, 0.2, -0.3, 0.1])
    q[7:15] = np.linspace(-0.4, 0.6, 8)
    rng = np.random.default_rng(491)
    v[:] = rng.normal(size=tree.nv)
    jac = body_velocity_jacobians(tree, q)
    np.testing.assert_allclose(jac @ v, tree.body_velocities(q, v), atol=1e-12, rtol=0)
    impulses = rng.normal(size=(tree.num_bodies, 6))
    generalized = sum(b.T @ imp for b, imp in zip(jac, impulses))
    np.testing.assert_allclose(generalized, impulses_to_generalized(tree, q, impulses), atol=1e-12, rtol=0)
    assert np.count_nonzero(jac[-2:, :, :14]) == 0
    assert np.count_nonzero(jac[:-2, :, 14:]) == 0


@pytest.mark.parametrize("cls", SOLVERS)
@pytest.mark.parametrize("bad", ["missing_b", "missing_l", "shape", "nan", "diagonal", "upper"])
def test_bad_mobility_rejected_before_solver_state_changes(cls, bad):
    inner = solver(cls)
    cache_solver = inner._vel_solver if isinstance(inner, PGSSplitImpulseSolver) else inner
    cache_solver._warm_cache[(0, -1)] = [(np.zeros(3), np.ones(3))]
    before = cache_solver._warm_cache[(0, -1)][0][1].copy()
    c = contact(0, -1, [0, 0, 0], [0, 0, 1])
    b = np.eye(6)[None, :, :]
    l = np.eye(6)
    if bad == "missing_b":
        b = None
    if bad == "missing_l":
        l = None
    if bad == "shape":
        b = np.ones((1, 3, 6))
    if bad == "nan":
        b[0, 0, 0] = np.nan
    if bad == "diagonal":
        l[0, 0] = 0
    if bad == "upper":
        l[0, 1] = 0.1
    with pytest.raises(ValueError):
        inner.solve(
            [c],
            [np.zeros(6)],
            [SpatialTransform.identity()],
            [1.0],
            [np.eye(3)],
            0.001,
            body_jacobians=b,
            mass_cholesky=l,
        )
    np.testing.assert_array_equal(c.tangent1, np.zeros(3))
    np.testing.assert_array_equal(cache_solver._warm_cache[(0, -1)][0][1], before)


@pytest.mark.parametrize("cls", SOLVERS)
@pytest.mark.parametrize("erp", [0.8, 10.0])
@pytest.mark.parametrize("dt", [0.01, 0.005])
@pytest.mark.parametrize("nfeet", [1, 4])
def test_foot_equilibrium_obeys_impulse_units(cls, erp, dt, nfeet):
    # Deliberately resolvable CFM: the dt vs dt^2 error cannot hide in tolerance.
    tree = RobotTreeNumpy(gravity=9.81)
    for i, mass in enumerate([2.0 - nfeet * 0.1] + [0.1] * nfeet):
        offset = (
            np.zeros(3)
            if i == 0 or nfeet == 1
            else np.array([0.1 if i % 2 else -0.1, 0.1 if i < 3 else -0.1, 0.0])
        )
        tree.add_body(
            Body(
                name=f"b{i}",
                index=i,
                parent=-1 if i == 0 else 0,
                joint=FreeJoint("root") if i == 0 else FixedJoint(f"w{i}"),
                inertia=SpatialInertia(mass, np.eye(3) * 0.01, np.zeros(3)),
                X_tree=SpatialTransform.from_translation(offset),
            )
        )
    tree.finalize()
    q, v = tree.default_state()
    cfm = 0.001
    slop = 0.005
    force = 2 * 9.81 / nfeet
    excess = cfm * force * (dt**2 if erp <= 1 else dt) / erp
    q[6] = 0.1 - slop - excess
    xs = tree.forward_kinematics(q)
    contacts = [contact(i, -1, xs[i].r - np.array([0.0, 0.0, 0.1]), [0, 0, 1]) for i in range(1, nfeet + 1)]
    for c in contacts:
        c.depth = slop + excess
    # Redundant four-normal load distribution converges slowly at this small
    # CFM. Resolve the equilibrium oracle; default-iteration drops are separate.
    s = cls(max_iter=10000, tolerance=1e-14, erp=erp, slop=slop, cfm=cfm)
    pipeline = StepPipeline(dt, constraint_solver=VelocityLevelAdapter(s))
    new_q, new_v = pipeline.step(tree, q, v, np.zeros(6), contacts)
    np.testing.assert_allclose(new_v, 0, atol=1e-10, rtol=0)
    np.testing.assert_allclose(new_q, q, atol=1e-10, rtol=0)
    cache = s._vel_solver._warm_cache if isinstance(s, PGSSplitImpulseSolver) else s._warm_cache
    impulses = [entries[0][1][0] for entries in cache.values()]
    np.testing.assert_allclose(impulses, force * dt, atol=1e-10, rtol=0)


def test_legacy_preserves_external_force_prediction():
    tree = composite(0.2).tree
    q, v = tree.default_state()
    v[2] = -1
    external = [np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(6)]
    c = contact(1, -1, [0, 0, -0.1], [0, 0, 1])
    _, new = ImplicitContactStep(0.001, solver(PGSContactSolver)).step(
        tree, q, v, np.zeros(6), external, contacts=[c]
    )
    np.testing.assert_allclose(new[:3], [0.002, 0, 0], atol=1e-10, rtol=0)
