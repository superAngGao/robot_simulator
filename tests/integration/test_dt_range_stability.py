"""
dt range stability — verify the simulator gives consistent results across
a range of integration time steps.

All existing tests run at the default dt=2e-4. Nothing currently checks
that the simulator behaves correctly at smaller or larger dt. The penalty
contact model has a stability bound determined by k_normal/m, but inside
that bound the steady-state geometry should be approximately dt-independent.

Coverage:
  1. Sphere drop settles to the same z across dt ∈ {5e-5, 1e-4, 2e-4, 5e-4}
     within ~2 mm. CPU only — building 4 GpuEngines is slow and adds no
     coverage value because the test is about the integrator path, not
     the GPU dispatch path (B(1) already checks GPU/CPU agreement).
  2. Per-step dt override (passing dt to step() different from constructor
     default) produces the same trajectory as constructor-time dt.
  3. Documentation of the stability bound: dt = 2e-3 should still settle
     (verified, included as a regression check); dt > 5e-3 may oscillate
     but must not produce NaN.
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.cpu_engine import CpuEngine
from physics.geometry import BodyCollisionGeometry, ShapeInstance, SphereShape
from physics.joint import FreeJoint
from physics.merged_model import merge_models
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from robot.model import RobotModel


def _ball_model(mass=1.0, radius=0.1):
    tree = RobotTreeNumpy(gravity=9.81)
    I_sphere = 2.0 / 5.0 * mass * radius**2
    tree.add_body(
        Body(
            name="ball",
            index=0,
            joint=FreeJoint("root"),
            inertia=SpatialInertia(mass=mass, inertia=np.eye(3) * I_sphere, com=np.zeros(3)),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()
    return RobotModel(
        tree=tree,
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(SphereShape(radius))])],
        contact_body_names=["ball"],
    )


def _settle_drop(engine, q0, dt, sim_time):
    """Drop a ball from q0 with the given dt for sim_time seconds; return final q."""
    n_steps = int(round(sim_time / dt))
    q = q0.copy()
    qdot = np.zeros(engine.merged.nv)
    tau = np.zeros(engine.merged.nv)
    for _ in range(n_steps):
        out = engine.step(q, qdot, tau, dt=dt)
        q, qdot = out.q_new, out.qdot_new
    return q, qdot


# ---------------------------------------------------------------------------
# 1. Final-state agreement across dt
# ---------------------------------------------------------------------------


class TestDtRangeFinalState:
    """The settled rest pose of a sphere drop should be insensitive to dt."""

    @pytest.mark.parametrize("dt", [5e-5, 1e-4, 2e-4, 5e-4])
    @pytest.mark.slow
    def test_sphere_drop_settles_consistently(self, dt):
        """Drop a sphere from z=0.5 for 1.5 s of sim time; final z should be near radius."""
        radius = 0.1
        model = _ball_model(radius=radius)
        merged = merge_models(robots={"a": model})
        engine = CpuEngine(merged, dt=dt)

        q0 = merged.tree.default_state()[0].copy()
        q0[6] = 0.5
        q_final, qdot_final = _settle_drop(engine, q0, dt=dt, sim_time=1.5)

        # No NaN
        assert np.all(np.isfinite(q_final)), f"NaN at dt={dt}: q={q_final}"

        # Resting close to radius (penalty equilibrium puts it slightly below)
        # The exact equilibrium depends on k_normal: penetration = m*g/k.
        # For mass=1, g=9.81, k=5000 → penetration ≈ 2 mm.
        z = q_final[6]
        assert 0.090 < z < 0.105, f"dt={dt}: final z={z:.5f} outside settling band [0.090, 0.105]"

        # qdot should be near zero (settled)
        v = np.linalg.norm(qdot_final[:3])
        assert v < 0.05, f"dt={dt}: |v|={v:.4f} not settled (final qdot={qdot_final})"

    @pytest.mark.slow
    def test_final_z_agrees_across_dt(self):
        """Final settled z must agree to within 2 mm across the dt range."""
        radius = 0.1
        dts = [5e-5, 1e-4, 2e-4, 5e-4]
        zs = []
        for dt in dts:
            model = _ball_model(radius=radius)
            merged = merge_models(robots={"a": model})
            engine = CpuEngine(merged, dt=dt)
            q0 = merged.tree.default_state()[0].copy()
            q0[6] = 0.5
            q_final, _ = _settle_drop(engine, q0, dt=dt, sim_time=1.5)
            zs.append(float(q_final[6]))

        z_arr = np.asarray(zs)
        spread = float(z_arr.max() - z_arr.min())
        assert spread < 2e-3, f"Final z varies > 2mm across dt {dts}: zs={zs}, spread={spread:.5f}"


# ---------------------------------------------------------------------------
# 2. Per-step dt override
# ---------------------------------------------------------------------------


class TestPerStepDtOverride:
    """step(dt=...) override must produce same result as constructor-default dt."""

    def test_step_dt_override_matches_constructor_dt(self):
        """Two engines built with different defaults but stepped with the same dt
        must produce identical trajectories."""
        radius = 0.1
        dt_run = 1e-4  # the dt we'll actually run with

        model_a = _ball_model(radius=radius)
        merged_a = merge_models(robots={"a": model_a})
        eng_a = CpuEngine(merged_a, dt=dt_run)  # constructor matches run dt

        model_b = _ball_model(radius=radius)
        merged_b = merge_models(robots={"a": model_b})
        eng_b = CpuEngine(merged_b, dt=2e-4)  # constructor mismatches; we'll override

        q0 = merged_a.tree.default_state()[0].copy()
        q0[6] = 0.5

        q_a, _ = _settle_drop(eng_a, q0, dt=dt_run, sim_time=0.5)
        q_b, _ = _settle_drop(eng_b, q0, dt=dt_run, sim_time=0.5)

        np.testing.assert_allclose(
            q_a,
            q_b,
            atol=1e-6,
            err_msg=(f"Per-step dt override gave different result than constructor dt: q_a={q_a}, q_b={q_b}"),
        )

    def test_step_dt_change_mid_simulation(self):
        """Changing dt mid-simulation should not crash or produce NaN."""
        model = _ball_model(radius=0.1)
        merged = merge_models(robots={"a": model})
        engine = CpuEngine(merged, dt=2e-4)

        q = merged.tree.default_state()[0].copy()
        q[6] = 0.5
        qdot = np.zeros(merged.nv)
        tau = np.zeros(merged.nv)

        # Run 1000 steps at dt=2e-4 (free fall + initial contact)
        for _ in range(1000):
            out = engine.step(q, qdot, tau, dt=2e-4)
            q, qdot = out.q_new, out.qdot_new

        # Switch to dt=1e-4 for the rest of the settling
        for _ in range(5000):
            out = engine.step(q, qdot, tau, dt=1e-4)
            q, qdot = out.q_new, out.qdot_new

        assert np.all(np.isfinite(q))
        assert 0.090 < q[6] < 0.105, f"Mid-simulation dt change broke settling: z={q[6]}"


# ---------------------------------------------------------------------------
# 3. Stability bound documentation
# ---------------------------------------------------------------------------


class TestStabilityBound:
    """Document the dt range over which the default penalty contact is stable.

    Penalty stability bound (semi-implicit Euler):
        dt < 2 / sqrt(k/m)
    For k_normal=5000, m=1: dt_crit ≈ 2 / 70.7 ≈ 28 ms.
    Practical safe bound is ~dt_crit / 4 ≈ 7 ms.
    """

    def test_dt_2ms_settles_no_nan(self):
        """dt=2 ms should settle without NaN (well below 7 ms safe bound)."""
        model = _ball_model(radius=0.1)
        merged = merge_models(robots={"a": model})
        engine = CpuEngine(merged, dt=2e-3)
        q0 = merged.tree.default_state()[0].copy()
        q0[6] = 0.5
        q_final, _ = _settle_drop(engine, q0, dt=2e-3, sim_time=2.0)
        assert np.all(np.isfinite(q_final)), f"dt=2ms produced NaN: {q_final}"
        assert 0.080 < q_final[6] < 0.115, f"dt=2ms final z out of band: {q_final[6]}"

    def test_dt_10ms_does_not_nan(self):
        """dt=10 ms is past the safe bound. We don't require accurate settling,
        only that the simulator doesn't produce NaN — any future divergence
        symptom should be a controlled blow-up, not a silent corruption."""
        model = _ball_model(radius=0.1)
        merged = merge_models(robots={"a": model})
        engine = CpuEngine(merged, dt=1e-2)
        q0 = merged.tree.default_state()[0].copy()
        q0[6] = 0.5
        q_final, _ = _settle_drop(engine, q0, dt=1e-2, sim_time=1.0)
        assert np.all(np.isfinite(q_final)), (
            f"dt=10ms produced NaN: {q_final}. If this fails, the integrator "
            f"silently diverges past stability bound — should add an assertion "
            f"in the engine to bail out instead."
        )
