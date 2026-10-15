"""Production R assembly versus analytic mechanics and actual MuJoCo 3.15.

Frictional force accuracy is deliberately not asserted: the existing projector
has a separately documented defect. condim3 cases below check assembly only.
"""
import itertools
import xml.etree.ElementTree as ET

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from physics.joint import FixedJoint, FreeJoint, PrismaticJoint
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from physics.solvers.admm_qp import ADMMQPSolver
from physics.solvers.contact_regularization import reference_body_weights
from physics.solvers.pgs_solver import ContactConstraint
from tests.validation.rigid_body.models import build_double_pendulum, build_quadruped


def scalar_pair(*, free=False, mass=2., com=0., axis=(0, 0, 1), parents=0,
                child=False, condim=1, solimp=".9 .9 .001 .5 2", impratio=1.):
    tree = RobotTreeNumpy()
    prefix = ""
    for i in range(parents):
        tree.add_body(Body(f"fixed{i}", i, FixedJoint("fixed"),
                          SpatialInertia(1., np.eye(3)*.01, np.zeros(3)),
                          SpatialTransform.identity(), parent=i-1))
        prefix += f'<body name="fixed{i}"><inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/>'
    joint = FreeJoint("free") if free else PrismaticJoint("slide", axis=np.array(axis, dtype=float))
    idx = tree.add_body(Body("ball", 0, joint,
        SpatialInertia(mass, np.diag([.008, .009, .01]), np.array([com, 0, 0])),
        SpatialTransform.identity() if free else SpatialTransform(np.eye(3), np.array([0, 0, .09])), parent=parents-1))
    inner = ""
    if child:
        tree.add_body(Body("child", 0, FixedJoint("fixed"), SpatialInertia(1., np.eye(3)*.01, np.zeros(3)),
                          SpatialTransform(np.eye(3), np.array([.3, 0, 0])), parent=idx))
        inner = '<body name="child" pos=".3 0 0"><inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/></body>'
    tree.finalize()
    joint_xml = '<freejoint/>' if free else f'<joint type="slide" axis="{" ".join(map(str,axis))}"/>'
    xml = f'''<mujoco><compiler fusestatic="false"/><option timestep=".002" integrator="Euler" cone="elliptic" solver="Newton" tolerance="1e-14" iterations="1000" impratio="{impratio}"><flag diagexact="disable"/></option>
    <default><geom condim="{condim}" solimp="{solimp}" solref=".02 1"/></default><worldbody><geom type="plane" size="2 2 .1"/>{prefix}<body name="ball" pos="0 0 .09">{joint_xml}<inertial pos="{com} 0 0" mass="{mass}" diaginertia=".008 .009 .01"/><geom type="sphere" size=".1"/>{inner}</body>{'</body>'*parents}</worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    q, v = tree.default_state()
    if free:
        q[6] = .09
    return tree, model, data, q, v


def contacts_from_mujoco(tree, model, data):
    names = {b.name: b.index for b in tree.bodies}
    def bid(geom):
        i = model.geom_bodyid[geom]
        return -1 if i == 0 else names[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)]
    return [ContactConstraint(body_i=bid(c.geom2), body_j=bid(c.geom1), point=c.pos.copy(),
                normal=c.frame[:3].copy(), depth=-c.dist, mu=c.friction[0], condim=c.dim)
            for c in data.contact]


def solver(**kw):
    return ADMMQPSolver(regularization="mujoco", max_iter=2000, tolerance=1e-12, warmstart=False, **kw)


@pytest.mark.parametrize("kw", [{}, {"parents": 1}, {"parents": 2}, {"child": True},
    {"com": .01}, {"com": .999e-6}, {"com": 1e-6}, {"axis": (.1, 0, 1)},
    {"axis": (.999e-14, 0, 1)}, {"axis": (1.001e-14, 0, 1)}, {"free": True}, {"mass": 5.}])
def test_reference_weights_and_scalar_force(kw):
    tree, m, d, q, v = scalar_pair(**kw)
    w = reference_body_weights(tree)
    ref = [m.body_invweight0[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b.name), 0] for b in tree.bodies]
    np.testing.assert_allclose(w, ref, rtol=1e-10, atol=1e-12)
    s = solver(solimp=(.9, .9, .001, .5, 2))
    f, J = s.solve(contacts_from_mujoco(tree, m, d), tree, q, v, tree.aba(q, v, np.zeros(tree.nv)), .002)
    np.testing.assert_allclose(s.last_R_diag, d.efc_R, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(f, d.efc_force, rtol=1e-7, atol=1e-8)
    if not kw:
        np.testing.assert_allclose(f, [67.658], atol=1e-8)
    diagnostic = s.last_R_diag
    diagnostic[:] = -1
    assert np.all(s.last_R_diag > 0)
    s.solve([], tree, q, v, np.zeros(tree.nv), .002)
    assert s.last_R_diag.size == 0


@pytest.mark.parametrize("impratio", [1e-16, 1e-15, 2e-15, 1, 10])
@pytest.mark.parametrize("imp", [( .9,.95,.001,.5,2), (.2,.8,0,.5,2), (-1,2,-1,2,.5), (.2,.9,.1,.001,100)])
def test_actual_elliptic_R(impratio, imp):
    tree, m, d, q, v = scalar_pair(free=True, condim=3, impratio=impratio, solimp=" ".join(map(str, imp)))
    # Assembly only: extreme ratios need not converge with this iteration cap.
    s = ADMMQPSolver(regularization="mujoco", impratio=impratio, solimp=imp, max_iter=1)
    s.solve(contacts_from_mujoco(tree, m, d), tree, q, v, np.zeros(tree.nv), .002)
    np.testing.assert_allclose(s.last_R_diag, d.efc_R, rtol=1e-10, atol=1e-12)


def quadruped_pair():
    model, xml = build_quadruped(contact=True)
    root = ET.fromstring(xml)
    opt = root.find("option")
    opt.set("iterations", "1000"); opt.set("tolerance", "1e-13")
    ET.SubElement(opt, "flag", diagexact="disable")
    for geom in root.iter("geom"):
        geom.set("condim", "1")
    for joint in root.iter("joint"):
        joint.set("damping", "0"); joint.set("limited", "false")
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    for b in model.tree.bodies:
        if hasattr(b.joint, "damping"):
            b.joint.damping = 0
            b.joint.q_min = -np.inf; b.joint.q_max = np.inf
    return model, m


@pytest.mark.parametrize("angles,height", [(0., .419), (.2, .411)])
def test_frozen_articulated_multicontact(angles, height):
    model, m = quadruped_pair(); tree = model.tree
    d = mujoco.MjData(m); q, v = tree.default_state()
    q[6] = height; q[7:] = np.tile([angles, -2*angles], 4)
    d.qpos[:] = np.r_[q[4:7], q[:4], q[7:]]
    mujoco.mj_forward(m, d)
    assert d.ncon == 4
    c = contacts_from_mujoco(tree, m, d)
    au = tree.aba(q, v, np.zeros(tree.nv))
    s = solver(); f, J = s.solve(c, tree, q, v, au, m.opt.timestep)
    H = tree.crba(q); A = J @ np.linalg.solve(H, J.T)
    Jref = d.efc_J.reshape(d.nefc, m.nv)
    M = np.zeros_like(H); mujoco.mj_fullM(m, d, M)
    np.testing.assert_allclose(J, Jref, atol=1e-12)
    np.testing.assert_allclose(A, Jref @ np.linalg.solve(M, Jref.T), atol=1e-12)
    assert np.max(np.abs(A-np.diag(np.diag(A)))) > .01
    np.testing.assert_allclose(s.last_R_diag, d.efc_R, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(au, d.qacc_smooth, atol=1e-11)
    expected = np.linalg.solve(A+np.diag(d.efc_R), d.efc_aref-J@au)
    assert np.all(expected > 0)  # Independent unconstrained optimum is feasible.
    np.testing.assert_allclose(f, expected, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(f, d.efc_force, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(au+np.linalg.solve(H,J.T@f), d.qacc, rtol=1e-7, atol=1e-8)


def test_policy_boundaries_and_model_mutation():
    tree, m, d, q, v = scalar_pair(free=True)
    c = contacts_from_mujoco(tree, m, d)
    c[0].point[0] = .2  # Off-centre normal mobility differs from COM average.
    results = []
    for policy, rho in itertools.product(("per_row", "mujoco"), (.1, 10)):
        s = ADMMQPSolver(regularization=policy, rho=rho, max_iter=1)
        s.solve(c, tree, q, v, np.zeros(tree.nv), .002)
        results.append(s.last_R_diag)
    np.testing.assert_array_equal(results[0], results[1])
    np.testing.assert_array_equal(results[2], results[3])
    assert not np.allclose(results[0], results[2])
    s = solver(); s.solve(c, tree, q, v, np.zeros(tree.nv), .002)
    before = s.last_R_diag
    tree.bodies[0].inertia.mass *= 2
    s.solve(c, tree, q, v, np.zeros(tree.nv), .002)
    np.testing.assert_allclose(s.last_R_diag, before/2)
    with pytest.raises(AttributeError): s.regularization = "per_row"
    for kw in ({"regularization": "bad"}, {"impratio": 0}, {"impratio": np.nan}, {"impratio": 2}):
        with pytest.raises(ValueError): ADMMQPSolver(**kw)
    for attr, value in (("body_i", -2), ("condim", 4), ("depth", np.nan)):
        cc = contacts_from_mujoco(tree, m, d); setattr(cc[0], attr, value)
        old = s.last_R_diag.copy()
        with pytest.raises(ValueError): s.solve(cc, tree, q, v, np.zeros(tree.nv), .002)
        np.testing.assert_array_equal(s.last_R_diag, old)


def test_reject_ambiguous_slider_frame():
    tree, _, _, _, _ = scalar_pair()
    tree.bodies[0].inertia.inertia[0, 1] = .001
    tree.bodies[0].inertia.inertia[1, 0] = .001
    with pytest.raises(NotImplementedError): reference_body_weights(tree)


@pytest.mark.parametrize("build", [build_double_pendulum, build_quadruped])
def test_reference_articulated_weights(build):
    model, xml = build(); m = mujoco.MjModel.from_xml_string(xml)
    ref = [m.body_invweight0[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,b.name),0] for b in model.tree.bodies]
    np.testing.assert_allclose(reference_body_weights(model.tree), ref, rtol=1e-10, atol=1e-12)
