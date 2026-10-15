"""Real-device tests of production regularization and condim1 subproblems."""
# ruff: noqa: E402 -- optional dependency must be checked before GPU imports.

import json

import numpy as np
import pytest

wp = pytest.importorskip("warp")
from physics.gpu_engine import GpuEngine
from physics.merged_model import merge_models
from physics.solvers.contact_regularization import impedance, normalize_parameters, reference_body_weights
from tests.gpu.solvers.test_gpu_admm import _ball_merged, _ball_model

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(
    "policy,condim,imp,ratio",
    [
        ("per_row", 1, (0.9, 0.9, 0.001, 0.5, 2), 1),
        ("per_row", 3, (0.9, 0.9, 0.001, 0.5, 2), 1),
        ("mujoco", 1, (0.9, 0.9, 0.001, 0.5, 2), 1),
        ("mujoco", 3, (0.9, 0.95, 0.001, 0.5, 2), 10),
        ("mujoco", 3, (0.2, 0.8, 0, 0.5, 2), 1e-16),
        ("mujoco", 3, (-1, 2, -1, 2, 0.5), 1),
        ("mujoco", 3, (0.2, 0.9, 0.1, 0.001, 100), 1),
    ],
)
def test_real_engine_R_and_normal_impulses(policy, condim, imp, ratio):
    # Independent roots with unequal masses; body-body plus ground contacts.
    merged = merge_models(robots={"a": _ball_model(2.0), "b": _ball_model(5.0)})
    engine = GpuEngine(
        merged, solver="admm", regularization=policy, impratio=ratio, contact_condim=condim, dt=0.002
    )
    engine._admm_solimp = imp
    engine._admm_iters = 1500 if condim == 1 else 1
    engine._admm_warmstart = False
    q, v = merged.tree.default_state()
    q[6] = 0.09
    q[11] = 0.19
    q[13] = 0.09
    engine.reset(q)
    engine.step()
    sol = engine._solver_scratch
    active = engine._contact_active.numpy()[0].astype(bool)
    assert active.sum() >= 3
    bi = engine._contact_bi.numpy()[0]
    bj = engine._contact_bj.numpy()[0]
    assert np.any(active & (bi >= 0) & (bj >= 0))
    W = sol.W.numpy()[0]
    R = sol.admm_R_diag.numpy()[0]
    p = sol.lambdas.numpy()[0]
    depths = engine._contact_depth.numpy()[0]
    weights = reference_body_weights(merged.tree)
    normalized = normalize_parameters((0.02, 1), imp, 0.002)[1] if policy == "mujoco" else imp
    relative_errors = []
    for c in np.flatnonzero(active):
        base = 3 * c
        d = impedance(float(depths[c]), normalized)
        if policy == "per_row":
            expected = (1 - d) / d * np.abs(np.diag(W)[base : base + condim])
        else:
            rn = max(1e-15, (1 - d) / d * sum(weights[i] for i in (bi[c], bj[c]) if i >= 0))
            expected = np.r_[rn, np.full(condim - 1, rn / max(1e-15, ratio))]
        np.testing.assert_allclose(R[base : base + condim], expected, rtol=3e-5, atol=2e-6)
        relative_errors.extend(abs(R[base : base + condim] - expected) / np.maximum(abs(expected), 1e-15))
    unused = np.repeat(~active, 3)
    if condim == 1:
        unused |= np.arange(len(R)) % 3 != 0
    for a in (
        R,
        p,
        sol.admm_rhs_const.numpy()[0],
        sol.admm_f_prev.numpy()[0],
        sol.admm_s_prev.numpy()[0],
        sol.admm_u_prev.numpy()[0],
    ):
        np.testing.assert_array_equal(a[unused], 0)
    if condim == 1:
        rows = 3 * np.flatnonzero(active)
        A = W[np.ix_(rows, rows)] + np.diag(R[rows])
        rhs = sol.admm_rhs_const.numpy()[0, rows]
        # Independent active-set enumeration for a small nonnegative QP.
        best = None
        for bits in range(1 << len(rows)):
            ids = np.array([i for i in range(len(rows)) if bits & (1 << i)], dtype=int)
            x = np.zeros(len(rows))
            if len(ids):
                x[ids] = np.linalg.solve(A[np.ix_(ids, ids)].astype(float), rhs[ids])
            g = A @ x - rhs
            if np.all(x >= -1e-10) and np.all(g >= -1e-7) and np.max(np.abs(g * x)) < 1e-7:
                best = x
                break
        assert best is not None
        np.testing.assert_allclose(p[rows], best, rtol=5e-5, atol=1e-7)
        n = 3 * (np.flatnonzero(active)[-1] + 1)
        mat = sol.admm_AR_rho.numpy()[0, :n, :n].copy()
        mat -= np.eye(n) * engine._admm_rho
        np.testing.assert_array_equal(mat[unused[:n]], 0)
        np.testing.assert_array_equal(mat[:, unused[:n]], 0)
    print(
        "GPU_REGULARIZATION",
        json.dumps(
            dict(
                policy=policy,
                condim=condim,
                impratio=ratio,
                max_R_relative_error=float(max(relative_errors)),
                max_impulse_error=float(np.max(abs(p[rows] - best))) if condim == 1 else None,
            )
        ),
    )
    # Warm cache from active contacts must not survive a now-empty solve.
    engine._admm_warmstart = True
    q[6] = 2.0
    q[13] = 2.0
    q[11] = 1.0
    engine.step(q=q, qdot=v)
    assert not np.any(engine._contact_active.numpy())
    for a in (sol.admm_R_diag, sol.lambdas, sol.admm_s_prev, sol.admm_u_prev):
        np.testing.assert_array_equal(a.numpy(), 0)
    engine.close()


def test_invalid_options_and_native_domain():
    merged = _ball_merged()
    for kw in (
        {"regularization": "bad"},
        {"regularization": "mujoco"},
        {"solver": "admm", "contact_condim": 4},
        {"solver": "admm", "impratio": 2},
    ):
        with pytest.raises(ValueError):
            GpuEngine(merged, **kw)
    engine = GpuEngine(merged, solver="admm", regularization="mujoco", contact_condim=1)
    with pytest.raises(AttributeError):
        engine.impratio = 2
    for ref in ((0.02, 1e-30), (0.02, np.nan), (-1, 1)):
        engine._admm_solref = ref
        with pytest.raises(ValueError):
            engine._step_physics()
    engine.close()


def snapshot(engine):
    sc = engine._scratch
    sol = engine._solver_scratch
    arrays = [
        sc.q,
        sc.qdot,
        sc.tau_total,
        engine._contact_active,
        engine._contact_depth,
        sol.admm_f_prev,
        sol.admm_s_prev,
        sol.admm_u_prev,
        sol.admm_R_diag,
    ]
    return (
        [a.numpy().copy() for a in arrays] + [engine._default_q.copy()],
        (
            engine._physics_run_id,
            engine._publish_frame_id,
            engine._publish_step_index,
            engine._publish_sim_time,
        ),
    )


def assert_unchanged(engine, before):
    after = snapshot(engine)
    for a, b in zip(after[0], before[0], strict=True):
        np.testing.assert_array_equal(a, b)
    assert after[1] == before[1]


def test_public_admission_is_atomic_and_recoverable():
    merged = _ball_merged()
    engine = GpuEngine(merged, num_envs=2, solver="admm", regularization="mujoco", contact_condim=1)
    q, v = merged.tree.default_state()
    q[6] = 0.09
    engine.reset(q)
    engine.step()
    for field in ("q", "qdot", "tau"):
        for bad in (np.nan, np.inf, 1e100):
            value = np.tile(q if field == "q" else v, (2, 1))
            value[1, -1] = bad
            before = snapshot(engine)
            with pytest.raises(ValueError):
                engine.step(**{field: value})
            assert_unchanged(engine, before)
    for dt in (0.0, -1.0, np.nan, np.inf, 1e100, 1e-100):
        for method in (engine.step, engine.step_n):
            before = snapshot(engine)
            with pytest.raises(ValueError):
                method(dt=dt)
            assert_unchanged(engine, before)
    for bad in (np.nan, np.inf, 1e100):
        badq = q.copy()
        badq[6] = bad
        for method in (lambda: engine.reset(badq), lambda: engine.reset_envs(np.array([1]), badq)):
            before = snapshot(engine)
            with pytest.raises(ValueError):
                method()
            assert_unchanged(engine, before)
        tau = np.tile(v, (2, 1))
        tau[1, 2] = bad
        before = snapshot(engine)
        with pytest.raises(ValueError):
            engine.step_n(tau=tau, n_substeps=2)
        assert_unchanged(engine, before)
    # Configuration failure must precede an otherwise valid external-state upload.
    engine._admm_solref = (-1.0, 1.0)
    before = snapshot(engine)
    with pytest.raises(ValueError):
        engine.step(q=np.tile(q, (2, 1)))
    assert_unchanged(engine, before)
    engine._admm_solref = (0.02, 1.0)
    out = engine.step(dt=None)
    assert np.all(np.isfinite(out.q_new))
    engine.reset_envs(np.array([1]), q)
    assert np.all(np.isfinite(engine.step_n(n_substeps=2).q_new))
    engine.close()


@pytest.mark.parametrize("condim", [1, 3])
@pytest.mark.parametrize("two_bodies", [False, True])
def test_native_R_overflow_boundary(condim, two_bodies):
    for mass, accepted in ((1e-34, True), (1e-35, False)):
        robots = {"a": _ball_model(mass)}
        if two_bodies:
            robots["b"] = _ball_model(mass)
        merged = merge_models(robots=robots)
        engine = GpuEngine(merged, solver="admm", regularization="mujoco", contact_condim=condim, dt=0.002)
        q, _ = merged.tree.default_state()
        q[6] = 0.09
        if two_bodies:
            q[11] = 0.19
            q[13] = 0.09
        engine.reset(q)
        engine._admm_solimp = (1e-4, 1e-4, 0.001, 0.5, 2)
        before = snapshot(engine)
        if accepted:
            engine.step()
            active = engine._contact_active.numpy()[0].astype(bool)
            R = engine._solver_scratch.admm_R_diag.numpy()[0]
            assert np.any(active)
            assert np.all(np.isfinite(R))
            assert np.all(R[3 * np.flatnonzero(active)] > 0)
            rows = 3 * np.flatnonzero(active)
            bi = engine._contact_bi.numpy()[0, active]
            bj = engine._contact_bj.numpy()[0, active]
            count = (bi >= 0).astype(int) + (bj >= 0).astype(int)
            expected = (1 - 1e-4) / 1e-4 * count / mass
            np.testing.assert_allclose(R[rows], expected, rtol=3e-5)
            if condim == 3:
                np.testing.assert_allclose(R[rows + 1], expected, rtol=3e-5)
                np.testing.assert_allclose(R[rows + 2], expected, rtol=3e-5)
        else:
            with pytest.raises(ValueError, match="regularization"):
                engine.step()
            assert_unchanged(engine, before)
        engine.close()


@pytest.mark.parametrize("mass,ratio", [(1.0, 1e38), (1e-25, 1e-16)])
def test_reject_native_tangent_underflow_and_overflow(mass, ratio):
    with pytest.raises(ValueError, match="regularization"):
        GpuEngine(_ball_merged(mass=mass), solver="admm", regularization="mujoco", impratio=ratio)


@pytest.mark.parametrize("mass,ratio", [(1.0, 1e35), (1e-23, 1e-16)])
def test_native_tangent_near_boundary(mass, ratio):
    merged = _ball_merged(mass=mass)
    engine = GpuEngine(merged, solver="admm", regularization="mujoco", impratio=ratio, dt=0.002)
    q, _ = merged.tree.default_state()
    q[6] = 0.09
    engine.reset(q)
    engine._admm_iters = 1
    engine.step()
    R = engine._solver_scratch.admm_R_diag.numpy()[0]
    active = engine._contact_active.numpy()[0].astype(bool)
    rows = 3 * np.flatnonzero(active)
    assert len(rows) > 0
    assert np.all(np.isfinite(R))
    assert np.all(R[rows + 1] >= np.finfo(np.float32).tiny)
    rn = (1 - 0.95) / 0.95 / mass
    np.testing.assert_allclose(R[rows], rn, rtol=3e-5)
    np.testing.assert_allclose(R[rows + 1], rn / max(1e-15, ratio), rtol=3e-5)
    engine.close()


@pytest.mark.parametrize(
    "imp",
    [
        (0.9999, 0.00012, 0.001, 0.5, 2),
        (0.9999, 0.00012, 0.010001, 0.5, 2),
        (0.9999, 0.00012, 0.1, 0.001, 100),
        (0.2, 0.8, np.nextafter(1e-15, 0.0), 0.5, 2),
        (0.2, 0.8, 1e-15, 0.5, 2),
        (0.2, 0.8, np.nextafter(1e-15, np.inf), 0.5, 2),
    ],
)
def test_decreasing_and_width_switch_against_actual_mujoco(imp):
    mujoco = pytest.importorskip("mujoco")
    from tests.reference.mujoco.test_admm_regularization import scalar_pair

    merged = _ball_merged(mass=2.0)
    engine = GpuEngine(merged, solver="admm", regularization="mujoco", dt=0.002)
    engine._admm_solimp = imp
    engine._admm_iters = 1
    q, _ = merged.tree.default_state()
    q[6] = 0.09
    engine.reset(q)
    engine.step()
    active = engine._contact_active.numpy()[0].astype(bool)
    slots = np.flatnonzero(active)
    assert len(slots) == 1
    depth = float(engine._contact_depth.numpy()[0, slots[0]])
    _, m, d, _, _ = scalar_pair(free=True, condim=3, solimp=" ".join(map(str, imp)))
    d.qpos[2] = 0.1 - depth  # Compare the actual GPU contact depth, not ideal geometry.
    mujoco.mj_forward(m, d)
    R = engine._solver_scratch.admm_R_diag.numpy()[0, 3 * slots[0] : 3 * slots[0] + 3]
    np.testing.assert_allclose(R, d.efc_R, rtol=3e-5, atol=1e-6)
    engine.close()


@pytest.mark.parametrize("width", [0.001, 0.010001])
@pytest.mark.parametrize("condim", [1, 3])
def test_decreasing_native_upper_boundary(width, condim):
    mass = 4.89730620518322e-35
    merged = merge_models(robots={"a": _ball_model(mass), "b": _ball_model(mass)})
    engine = GpuEngine(merged, solver="admm", regularization="mujoco", contact_condim=condim, dt=0.002)
    engine._admm_solimp = (0.9999, 0.00012, width, 0.5, 2)
    engine._admm_iters = 1
    q, _ = merged.tree.default_state()
    q[6] = q[13] = 0.09
    q[11] = 0.19
    engine.reset(q)
    engine.step()
    active = engine._contact_active.numpy()[0].astype(bool)
    assert active.sum() == 3
    R = engine._solver_scratch.admm_R_diag.numpy()[0]
    assert np.all(np.isfinite(R))
    depth = engine._contact_depth.numpy()[0]
    bi = engine._contact_bi.numpy()[0]
    bj = engine._contact_bj.numpy()[0]
    for c in np.flatnonzero(active):
        d = impedance(float(depth[c]), engine._admm_solimp)
        expected = (1 - d) / d * (int(bi[c] >= 0) + int(bj[c] >= 0)) / mass
        np.testing.assert_allclose(R[3 * c : 3 * c + condim], expected, rtol=3e-5)
    engine.close()
