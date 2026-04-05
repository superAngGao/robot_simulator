"""
Tests for GpuEngine contact force sensor aggregation.

Validates the _aggregate_contact_forces kernel that reconstructs per-sensor-body
world-frame net contact forces from constraint solver impulses (lambdas).
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.geometry import BodyCollisionGeometry, ShapeInstance, SphereShape
from physics.joint import FreeJoint
from physics.merged_model import merge_models
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from robot.model import RobotModel

try:
    import warp as wp

    from physics.gpu_engine import GpuEngine

    HAS_WARP = True
except Exception:
    HAS_WARP = False

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not HAS_WARP, reason="Warp or CUDA not available"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ball_model(mass=1.0, radius=0.1):
    """Single FreeJoint sphere with contact on 'ball' body."""
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


def _make_engine(mass=1.0, num_envs=1, solver="jacobi_pgs_si", z0=0.5):
    merged = merge_models(robots={"ball": _ball_model(mass=mass)})
    engine = GpuEngine(merged, num_envs=num_envs, solver=solver, dt=2e-4)
    q0, _ = merged.tree.default_state()
    q0[6] = z0  # pz
    engine.reset(q0=q0)
    return engine, merged


# ---------------------------------------------------------------------------
# Buffer shape and accessor
# ---------------------------------------------------------------------------


class TestContactForceSensorShape:
    def test_shape(self):
        engine, _ = _make_engine(num_envs=4)
        cf = engine.contact_force_sensor_wp
        # 1 contact body ("ball"), 4 envs, flattened nc*3
        assert cf.shape == (4, 3)
        assert cf.dtype == wp.float32

    def test_nc_sensor(self):
        engine, _ = _make_engine()
        assert engine.nc_sensor == 1

    def test_zero_before_contact(self):
        """Before any contact, force sensor should be zero."""
        engine, _ = _make_engine(z0=1.0)  # high up, no contact
        engine.step()
        cf = engine.contact_force_sensor_wp.numpy().reshape(engine.num_envs, engine.nc_sensor, 3)
        np.testing.assert_allclose(cf, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Force direction and magnitude
# ---------------------------------------------------------------------------


class TestContactForceValues:
    def test_force_points_upward_on_ground(self):
        """When ball is on ground, contact force z > 0 (pointing up)."""
        engine, _ = _make_engine(z0=0.15)  # just above ground
        # Step enough for contact (ball needs ~1000 steps to reach ground)
        for _ in range(2000):
            engine.step()
        cf = engine.contact_force_sensor_wp.numpy().reshape(engine.num_envs, engine.nc_sensor, 3)
        # Force z should be positive (upward reaction)
        assert cf[0, 0, 2] > 0.0, f"Expected upward force, got fz={cf[0, 0, 2]}"

    def test_force_xy_small_at_rest(self):
        """At rest on flat ground, horizontal forces should be near zero."""
        engine, _ = _make_engine(z0=0.15)
        for _ in range(2000):
            engine.step()
        cf = engine.contact_force_sensor_wp.numpy().reshape(engine.num_envs, engine.nc_sensor, 3)
        fz = cf[0, 0, 2]
        fx = abs(cf[0, 0, 0])
        fy = abs(cf[0, 0, 1])
        # Horizontal forces should be much smaller than vertical
        assert fx < 0.1 * fz, f"fx={fx} too large vs fz={fz}"
        assert fy < 0.1 * fz, f"fy={fy} too large vs fz={fz}"

    def test_force_magnitude_matches_gravity(self):
        """At steady state, contact force ≈ mg."""
        mass = 2.0
        engine, _ = _make_engine(mass=mass, z0=0.15)
        # Let it settle
        for _ in range(5000):
            engine.step()
        cf = engine.contact_force_sensor_wp.numpy().reshape(engine.num_envs, engine.nc_sensor, 3)
        fz = cf[0, 0, 2]
        mg = mass * 9.81
        # Should be close to mg (within 10% — solver/penalty tolerance)
        assert abs(fz - mg) / mg < 0.1, f"fz={fz:.2f}, mg={mg:.2f}"

    def test_no_force_in_free_fall(self):
        """During free fall (before contact), force should be zero."""
        engine, _ = _make_engine(z0=1.0)
        # Only a few steps — still in the air
        for _ in range(10):
            engine.step()
        cf = engine.contact_force_sensor_wp.numpy().reshape(engine.num_envs, engine.nc_sensor, 3)
        np.testing.assert_allclose(cf, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Multi-env consistency
# ---------------------------------------------------------------------------


class TestContactForceMultiEnv:
    def test_all_envs_same_force(self):
        """All envs with same state should get same contact force."""
        engine, _ = _make_engine(num_envs=4, z0=0.15)
        for _ in range(2000):
            engine.step()
        cf = engine.contact_force_sensor_wp.numpy().reshape(engine.num_envs, engine.nc_sensor, 3)
        # All 4 envs should have similar forces
        for i in range(1, 4):
            np.testing.assert_allclose(cf[i], cf[0], atol=1.0)

    def test_reset_env_clears_force(self):
        """After reset_envs, the reset env should have zero force (in air)."""
        engine, _ = _make_engine(num_envs=2, z0=0.15)
        # Let both settle on ground
        for _ in range(2000):
            engine.step()
        cf_before = (
            engine.contact_force_sensor_wp.numpy().reshape(engine.num_envs, engine.nc_sensor, 3).copy()
        )
        assert cf_before[0, 0, 2] > 0.0  # env 0 on ground

        # Reset env 0 to high up
        q0_high, _ = merge_models(robots={"ball": _ball_model()}).tree.default_state()
        q0_high[6] = 1.0
        engine.reset_envs(np.array([0], dtype=np.int32), q0=q0_high)
        engine.step()

        cf_after = engine.contact_force_sensor_wp.numpy().reshape(engine.num_envs, engine.nc_sensor, 3)
        # Env 0: in air → zero force
        assert abs(cf_after[0, 0, 2]) < 1.0
        # Env 1: still on ground → nonzero force
        assert cf_after[1, 0, 2] > 0.0


# ---------------------------------------------------------------------------
# ADMM solver
# ---------------------------------------------------------------------------


class TestContactForceADMM:
    def test_admm_force_upward(self):
        """ADMM solver should also produce upward contact force."""
        engine, _ = _make_engine(solver="admm", z0=0.15)
        for _ in range(2000):
            engine.step()
        cf = engine.contact_force_sensor_wp.numpy().reshape(engine.num_envs, engine.nc_sensor, 3)
        assert cf[0, 0, 2] > 0.0

    def test_admm_force_matches_gravity(self):
        """ADMM steady-state force ≈ mg."""
        mass = 1.0
        engine, _ = _make_engine(mass=mass, solver="admm", z0=0.15)
        for _ in range(5000):
            engine.step()
        cf = engine.contact_force_sensor_wp.numpy().reshape(engine.num_envs, engine.nc_sensor, 3)
        fz = cf[0, 0, 2]
        mg = mass * 9.81
        assert abs(fz - mg) / mg < 0.1, f"fz={fz:.2f}, mg={mg:.2f}"
