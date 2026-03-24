"""
Tests for LCPContactModel integrated into the Simulator pipeline.

Verifies:
  1. LCPContactModel receives real mass/inertia from tree via Simulator
  2. Multi-step simulation with LCP contact is stable and produces upward forces
  3. load_urdf with contact_method="lcp" builds a working model
  4. LCP and Penalty produce qualitatively similar results through Simulator
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from physics.collision import NullSelfCollision
from physics.contact import ContactParams, ContactPoint, LCPContactModel, PenaltyContactModel
from physics.geometry import SphereShape
from physics.integrator import SemiImplicitEuler
from physics.joint import FreeJoint, RevoluteJoint
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from robot.model import RobotModel
from simulator import Simulator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lcp_model(
    mass: float = 5.0,
    z0: float = 0.5,
    shape_radius: float = 0.1,
    mu: float = 0.8,
) -> tuple[RobotModel, np.ndarray, np.ndarray]:
    """Single free-floating body with LCP ground contact via a sphere."""
    tree = RobotTreeNumpy(gravity=9.81)
    body = Body(
        name="base",
        index=0,
        joint=FreeJoint("root"),
        inertia=SpatialInertia(
            mass=mass,
            inertia=np.diag([0.1, 0.1, 0.1]),
            com=np.zeros(3),
        ),
        X_tree=SpatialTransform.identity(),
        parent=-1,
    )
    tree.add_body(body)
    tree.finalize()

    lcp = LCPContactModel(mu=mu, max_iter=50)
    lcp.add_contact_body(0, SphereShape(shape_radius), "base")

    model = RobotModel(
        tree=tree,
        contact_model=lcp,
        self_collision=NullSelfCollision(),
    )

    q, qdot = tree.default_state()
    q[3] = 1.0  # qw = 1 (identity quaternion)
    q[6] = z0  # pz
    return model, q, qdot


def _make_two_body_model() -> tuple[RobotModel, np.ndarray, np.ndarray]:
    """Free base + revolute link, both with LCP contact spheres."""
    tree = RobotTreeNumpy(gravity=9.81)

    base = Body(
        name="base",
        index=0,
        joint=FreeJoint("root"),
        inertia=SpatialInertia(mass=5.0, inertia=np.diag([0.1, 0.1, 0.1]), com=np.zeros(3)),
        X_tree=SpatialTransform.identity(),
        parent=-1,
    )
    tree.add_body(base)

    foot = Body(
        name="foot",
        index=1,
        joint=RevoluteJoint("hip", axis=np.array([0, 1, 0])),
        inertia=SpatialInertia(mass=1.0, inertia=np.diag([0.01, 0.01, 0.01]), com=np.array([0, 0, -0.15])),
        X_tree=SpatialTransform(R=np.eye(3), r=np.array([0.0, 0.0, -0.2])),
        parent=0,
    )
    tree.add_body(foot)
    tree.finalize()

    lcp = LCPContactModel(mu=0.8, max_iter=50)
    lcp.add_contact_body(1, SphereShape(0.05), "foot")

    model = RobotModel(
        tree=tree,
        contact_model=lcp,
        self_collision=NullSelfCollision(),
    )

    q, qdot = tree.default_state()
    q[3] = 1.0  # qw
    q[6] = 0.5  # pz (base at 0.5m, foot at ~0.3m)
    return model, q, qdot


# ---------------------------------------------------------------------------
# Tests: LCP through Simulator pipeline
# ---------------------------------------------------------------------------


class TestLCPSimulatorIntegration:
    """LCPContactModel receives correct dt and tree from Simulator."""

    def test_single_step_finite(self):
        """One Simulator step with LCP contact produces finite state."""
        model, q, qdot = _make_lcp_model(z0=0.05)  # near ground
        sim = Simulator(model, SemiImplicitEuler(dt=1e-3))
        tau = np.zeros(model.tree.nv)

        q_new, qdot_new = sim.step(q, qdot, tau)

        assert np.all(np.isfinite(q_new))
        assert np.all(np.isfinite(qdot_new))

    def test_contact_produces_upward_force(self):
        """Body penetrating ground should get pushed up by LCP contact."""
        model, q, qdot = _make_lcp_model(z0=0.05)  # sphere radius 0.1 → penetrating
        sim = Simulator(model, SemiImplicitEuler(dt=1e-3))
        tau = np.zeros(model.tree.nv)
        qdot[2] = -1.0  # falling

        q_new, qdot_new = sim.step(q, qdot, tau)

        # Upward velocity should increase (or downward velocity decrease)
        assert qdot_new[2] > qdot[2], (
            f"LCP contact should reduce downward velocity: {qdot[2]} -> {qdot_new[2]}"
        )

    def test_free_fall_no_contact(self):
        """Body above ground: LCP should produce no forces (free fall)."""
        model, q, qdot = _make_lcp_model(z0=1.0)  # well above ground
        sim = Simulator(model, SemiImplicitEuler(dt=1e-3))
        tau = np.zeros(model.tree.nv)

        q_new, qdot_new = sim.step(q, qdot, tau)

        # Should be pure free fall — vz should decrease by g*dt
        expected_vz = -9.81 * 1e-3
        assert abs(qdot_new[2] - expected_vz) < 1e-6

    def test_multi_step_stable(self):
        """50 steps of LCP contact simulation should remain stable."""
        model, q, qdot = _make_lcp_model(z0=0.2)
        sim = Simulator(model, SemiImplicitEuler(dt=1e-3))
        tau = np.zeros(model.tree.nv)

        for _ in range(50):
            q, qdot = sim.step(q, qdot, tau)

        assert np.all(np.isfinite(q))
        assert np.all(np.isfinite(qdot))
        # Body should not have fallen through the ground
        assert q[6] > -0.5, f"Body z={q[6]} — fell through ground"

    def test_real_mass_affects_dynamics(self):
        """Different masses should produce different contact responses."""
        _, q_light, qdot_light = _make_lcp_model(mass=1.0, z0=0.05)
        _, q_heavy, qdot_heavy = _make_lcp_model(mass=50.0, z0=0.05)
        qdot_light[2] = -1.0
        qdot_heavy[2] = -1.0

        model_light, _, _ = _make_lcp_model(mass=1.0, z0=0.05)
        model_heavy, _, _ = _make_lcp_model(mass=50.0, z0=0.05)

        sim_light = Simulator(model_light, SemiImplicitEuler(dt=1e-3))
        sim_heavy = Simulator(model_heavy, SemiImplicitEuler(dt=1e-3))
        tau = np.zeros(6)

        _, qdot_light_new = sim_light.step(q_light, qdot_light, tau)
        _, qdot_heavy_new = sim_heavy.step(q_heavy, qdot_heavy, tau)

        # Both should get some contact response, but they should differ
        # (if mass were ignored, they'd be identical)
        assert qdot_light_new[2] != pytest.approx(qdot_heavy_new[2], abs=1e-4), (
            "Different masses should produce different velocity changes"
        )

    def test_two_body_lcp(self):
        """Two-body model: foot contacts ground through Simulator."""
        model, q, qdot = _make_two_body_model()
        sim = Simulator(model, SemiImplicitEuler(dt=1e-3))
        tau = np.zeros(model.tree.nv)

        for _ in range(20):
            q, qdot = sim.step(q, qdot, tau)

        assert np.all(np.isfinite(q))
        assert np.all(np.isfinite(qdot))


# ---------------------------------------------------------------------------
# Tests: load_urdf with contact_method="lcp"
# ---------------------------------------------------------------------------

_SIMPLE_URDF = """\
<robot name="test_bot">
  <link name="base">
    <inertial>
      <mass value="5.0"/><origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
    <collision>
      <geometry><box size="0.4 0.3 0.2"/></geometry>
    </collision>
  </link>
  <link name="leg">
    <inertial>
      <mass value="1.0"/><origin xyz="0 0 -0.15"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <collision>
      <geometry><sphere radius="0.05"/></geometry>
    </collision>
  </link>
  <joint name="hip" type="revolute">
    <parent link="base"/><child link="leg"/>
    <origin xyz="0 0 -0.1"/><axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50"/>
  </joint>
</robot>
"""


class TestLoadURDFLCP:
    def _write_urdf(self) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
        f.write(_SIMPLE_URDF)
        f.close()
        return f.name

    def test_load_lcp_creates_lcp_model(self):
        """load_urdf with contact_method='lcp' should create LCPContactModel."""
        from robot import load_urdf

        path = self._write_urdf()
        try:
            model = load_urdf(path, floating_base=True, contact_links=["leg"], contact_method="lcp")
        finally:
            os.unlink(path)

        assert isinstance(model.contact_model, LCPContactModel)

    def test_load_lcp_step_stable(self):
        """URDF-loaded LCP model should work through Simulator."""
        from robot import load_urdf

        path = self._write_urdf()
        try:
            model = load_urdf(path, floating_base=True, contact_links=["leg"], contact_method="lcp")
        finally:
            os.unlink(path)

        sim = Simulator(model, SemiImplicitEuler(dt=1e-3))
        q, qdot = model.tree.default_state()
        q[3] = 1.0  # qw
        q[6] = 0.3  # base height
        tau = np.zeros(model.tree.nv)

        for _ in range(20):
            q, qdot = sim.step(q, qdot, tau)

        assert np.all(np.isfinite(q))
        assert np.all(np.isfinite(qdot))

    def test_load_penalty_still_works(self):
        """Default contact_method='penalty' should still work."""
        from robot import load_urdf

        path = self._write_urdf()
        try:
            model = load_urdf(path, floating_base=True, contact_links=["leg"], contact_method="penalty")
        finally:
            os.unlink(path)

        assert isinstance(model.contact_model, PenaltyContactModel)

    def test_invalid_contact_method_raises(self):
        """Unknown contact_method should raise ValueError."""
        from robot import load_urdf

        path = self._write_urdf()
        try:
            with pytest.raises(ValueError, match="contact_method"):
                load_urdf(path, floating_base=True, contact_links=["leg"], contact_method="invalid")
        finally:
            os.unlink(path)

    def test_lcp_no_geometry_uses_default_sphere(self):
        """Contact link with no collision geometry should get a tiny sphere."""
        urdf_no_geom = """\
<robot name="test">
  <link name="base">
    <inertial><mass value="5.0"/><origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  <link name="foot">
    <inertial><mass value="0.5"/><origin xyz="0 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="hip" type="revolute">
    <parent link="base"/><child link="foot"/>
    <origin xyz="0 0 -0.2"/><axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57"/>
  </joint>
</robot>
"""
        from robot import load_urdf

        f = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
        f.write(urdf_no_geom)
        f.close()
        try:
            model = load_urdf(f.name, floating_base=True, contact_links=["foot"], contact_method="lcp")
        finally:
            os.unlink(f.name)

        assert isinstance(model.contact_model, LCPContactModel)
        # Should have one registered contact body
        assert len(model.contact_model._contact_bodies) == 1


# ---------------------------------------------------------------------------
# Tests: LCP vs Penalty qualitative comparison
# ---------------------------------------------------------------------------


class TestLCPvsPenalty:
    def test_both_prevent_ground_penetration(self):
        """Both models should prevent the body from falling through the ground."""
        dt = 1e-3
        n_steps = 100

        for method in ("penalty", "lcp"):
            if method == "penalty":
                cm = PenaltyContactModel(ContactParams(k_normal=5000, b_normal=500, mu=0.8))
                cm.add_contact_point(ContactPoint(body_index=0, position=np.zeros(3), name="base"))
            else:
                cm = LCPContactModel(mu=0.8, max_iter=50)
                cm.add_contact_body(0, SphereShape(0.1), "base")

            tree = RobotTreeNumpy(gravity=9.81)
            body = Body(
                name="base",
                index=0,
                joint=FreeJoint("root"),
                inertia=SpatialInertia(mass=5.0, inertia=np.diag([0.1, 0.1, 0.1]), com=np.zeros(3)),
                X_tree=SpatialTransform.identity(),
                parent=-1,
            )
            tree.add_body(body)
            tree.finalize()

            model = RobotModel(tree=tree, contact_model=cm, self_collision=NullSelfCollision())
            sim = Simulator(model, SemiImplicitEuler(dt=dt))

            q, qdot = tree.default_state()
            q[3] = 1.0
            q[6] = 0.3  # start above ground
            tau = np.zeros(tree.nv)

            for _ in range(n_steps):
                q, qdot = sim.step(q, qdot, tau)

            assert q[6] > -0.5, f"{method}: body fell through ground (z={q[6]:.3f})"
            assert np.all(np.isfinite(q)), f"{method}: state diverged"
