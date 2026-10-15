"""Frictionless real-engine comparisons; reports full-state error measurements."""
# ruff: noqa: E402 -- reference dependency is optional.

import json

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from physics.cpu_engine import CpuEngine
from physics.geometry import BodyCollisionGeometry, ShapeInstance, SphereShape
from physics.merged_model import merge_models
from robot.model import RobotModel
from tests.reference.mujoco.test_admm_regularization import quadruped_pair, scalar_pair, solver


@pytest.mark.parametrize("kind", ["sphere", "quadruped"])
@pytest.mark.parametrize("dt", [0.001, 0.0005])
def test_matched_frictionless_trajectory(kind, dt):
    if kind == "sphere":
        tree, m, _, _, _ = scalar_pair(free=True)
        model = RobotModel(
            tree=tree,
            geometries=[BodyCollisionGeometry(0, [ShapeInstance(SphereShape(0.1))])],
            contact_body_names=["ball"],
        )
        s = solver(solimp=(0.9, 0.9, 0.001, 0.5, 2))
        duration = 0.5
    else:
        model, m = quadruped_pair()
        s = solver()
        duration = 0.12
    merged = merge_models(robots={"robot": model})
    engine = CpuEngine(merged, solver=s, dt=dt, contact_condim=1)
    q, v = merged.tree.default_state()
    q[6] = 0.25 if kind == "sphere" else 0.411
    if kind == "quadruped":
        q[7:] = np.tile([0.2, -0.4], 4)
    m.opt.timestep = dt
    d = mujoco.MjData(m)
    d.qpos[:] = np.r_[q[4:7], q[:4], q[7:]]
    errors = []
    impulses = []
    depths = []
    counts = []
    for _ in range(round(duration / dt)):
        out = engine.step(q, v, np.zeros(merged.tree.nv))
        # AccelLevelAdapter bypasses the solver on empty frames; last_forces
        # describes the last solve, not necessarily this frame.
        ours_impulse = float(np.sum(s.last_forces)) * dt if engine._last_contacts else 0.0
        our_depth = max([c.depth for c in engine._last_contacts] + [0.0])
        mujoco.mj_step(m, d)
        ref_impulse = float(np.sum(d.efc_force)) * dt
        ref_depth = max([-c.dist for c in d.contact] + [0.0])
        q, v = out.q_new, out.qdot_new
        refq = np.r_[d.qpos[3:7], d.qpos[:3], d.qpos[7:]]
        errors.append([np.max(np.abs(q - refq)), np.max(np.abs(v - d.qvel))])
        impulses.append([ours_impulse, ref_impulse])
        depths.append([our_depth, ref_depth])
        counts.append([len(engine._last_contacts), d.ncon])
    errors = np.array(errors)
    impulses = np.array(impulses)
    depths = np.array(depths)
    result = dict(
        kind=kind,
        dt=dt,
        max_q_error=float(errors[:, 0].max()),
        max_v_error=float(errors[:, 1].max()),
        max_step_impulse_error=float(np.max(abs(impulses[:, 0] - impulses[:, 1]))),
        cumulative_impulse_error=float(abs(np.sum(impulses[:, 0] - impulses[:, 1]))),
        max_penetration_ours=float(depths[:, 0].max()),
        max_penetration_mujoco=float(depths[:, 1].max()),
        max_penetration_error=float(np.max(abs(depths[:, 0] - depths[:, 1]))),
        contact_count_disagreements=int(np.count_nonzero(np.diff(counts, axis=1))),
    )
    print("TRAJECTORY", json.dumps(result, sort_keys=True))
    assert np.all(np.isfinite(errors))
    assert max(c[0] for c in counts) > 0
    # Matched discrete equations on these fixtures agree to rounding plus the
    # 1e-12 solve tolerance accumulated over <=1000 steps. This is a numerical
    # equivalence gate, not a universal geometric/material fidelity tolerance.
    assert result["max_q_error"] < 1e-10
    assert result["max_v_error"] < 1e-9
    assert result["max_penetration_error"] < 1e-10
    assert result["max_step_impulse_error"] < 1e-10
    assert result["cumulative_impulse_error"] < 1e-9
    assert result["contact_count_disagreements"] == 0
