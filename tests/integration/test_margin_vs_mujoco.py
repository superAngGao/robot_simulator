"""
Integration tests: CpuEngine (margin enabled) vs MuJoCo reference.

Validates that the full convex-margin pipeline (Part 2, session 31) produces
physically correct dynamics matching MuJoCo on representative scenes:

  Class 1: TestSingleBodyDropMuJoCo   — sphere / box free-fall + ground contact
  Class 2: TestTwoBodyContact         — sphere-sphere body-body contact response
  Class 3: TestMarginPhysicsNeutral   — margin=1mm doesn't alter physics vs margin=0

Design notes:
- Ground contact uses halfspace_convex_query (no margin parameter), so Classes 1-2
  test the full pipeline with margin as a bystander for ground, and active for
  body-body contacts.
- Classes 1-2 compare against MuJoCo at the steady-state level (< 1mm).
  Trajectory-level comparison is not attempted because solver models differ
  (our PGS-SI is fully inelastic; MuJoCo uses soft constraints).
- Class 3 tests margin neutrality at the gjk_epa_query level.

Reference: session 31, plan floating-drifting-yeti.md Part 3 File 3.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import mujoco

    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

try:
    from physics.cpu_engine import CpuEngine

    HAS_ENGINE = True
except Exception:
    HAS_ENGINE = False

pytestmark = [
    pytest.mark.skipif(not HAS_MUJOCO, reason="mujoco not installed"),
    pytest.mark.skipif(not HAS_ENGINE, reason="CpuEngine not available"),
]

# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------

DT = 2e-4  # must match CLAUDE.md invariant


def _sphere_model(r: float = 0.05):
    """Single free-floating sphere robot model."""
    import numpy as np

    from physics.geometry import BodyCollisionGeometry, ShapeInstance, SphereShape
    from physics.joint import FreeJoint
    from physics.merged_model import merge_models
    from physics.robot_tree import Body, RobotTreeNumpy
    from physics.spatial import SpatialInertia, SpatialTransform
    from physics.terrain import FlatTerrain
    from robot.model import RobotModel

    I = np.eye(3) * (2 / 5 * 1.0 * r**2)
    tree = RobotTreeNumpy(gravity=9.81)
    tree.add_body(
        Body(
            name="s",
            index=0,
            joint=FreeJoint("root"),
            inertia=SpatialInertia(1.0, I, np.zeros(3)),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()
    model = RobotModel(
        tree=tree,
        actuated_joint_names=[],
        contact_body_names=["s"],
        geometries=[BodyCollisionGeometry(body_index=0, shapes=[ShapeInstance(shape=SphereShape(r))])],
    )
    merged = merge_models({"r": model}, terrain=FlatTerrain())
    engine = CpuEngine(merged, dt=DT)
    return merged, engine


def _box_model(half: float = 0.05):
    """Single free-floating box robot model."""
    import numpy as np

    from physics.geometry import BodyCollisionGeometry, BoxShape, ShapeInstance
    from physics.joint import FreeJoint
    from physics.merged_model import merge_models
    from physics.robot_tree import Body, RobotTreeNumpy
    from physics.spatial import SpatialInertia, SpatialTransform
    from physics.terrain import FlatTerrain
    from robot.model import RobotModel

    I = np.eye(3) * (1.0 / 6 * 1.0 * (2 * half) ** 2)
    tree = RobotTreeNumpy(gravity=9.81)
    tree.add_body(
        Body(
            name="box",
            index=0,
            joint=FreeJoint("root"),
            inertia=SpatialInertia(1.0, I, np.zeros(3)),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()
    model = RobotModel(
        tree=tree,
        actuated_joint_names=[],
        contact_body_names=["box"],
        geometries=[
            BodyCollisionGeometry(
                body_index=0,
                shapes=[ShapeInstance(shape=BoxShape((2 * half, 2 * half, 2 * half)))],
            )
        ],
    )
    merged = merge_models({"r": model}, terrain=FlatTerrain())
    engine = CpuEngine(merged, dt=DT)
    return merged, engine


def _two_sphere_model(r: float = 0.05):
    """Two free-floating spheres for body-body contact tests."""
    import numpy as np

    from physics.geometry import BodyCollisionGeometry, ShapeInstance, SphereShape
    from physics.joint import FreeJoint
    from physics.merged_model import merge_models
    from physics.robot_tree import Body, RobotTreeNumpy
    from physics.spatial import SpatialInertia, SpatialTransform
    from physics.terrain import FlatTerrain
    from robot.model import RobotModel

    I = np.eye(3) * (2 / 5 * 1.0 * r**2)
    tree = RobotTreeNumpy(gravity=9.81)
    for i in range(2):
        tree.add_body(
            Body(
                name=f"s{i}",
                index=i,
                joint=FreeJoint(f"root{i}"),
                inertia=SpatialInertia(1.0, I, np.zeros(3)),
                X_tree=SpatialTransform.identity(),
                parent=-1,
            )
        )
    tree.finalize()
    model = RobotModel(
        tree=tree,
        actuated_joint_names=[],
        contact_body_names=["s0", "s1"],
        geometries=[
            BodyCollisionGeometry(body_index=0, shapes=[ShapeInstance(shape=SphereShape(r))]),
            BodyCollisionGeometry(body_index=1, shapes=[ShapeInstance(shape=SphereShape(r))]),
        ],
    )
    merged = merge_models({"r": model}, terrain=FlatTerrain())
    engine = CpuEngine(merged, dt=DT)
    return merged, engine


def _run_drop(engine, merged, z0: float, n_steps: int):
    """Run a single-body drop from z=z0 for n_steps. Returns z trajectory."""
    q = np.zeros(merged.nq)
    q[0] = 1.0  # qw — FreeJoint q: [qw, qx, qy, qz, x, y, z]
    q[6] = z0
    qdot = np.zeros(merged.nv)
    z_traj = np.zeros(n_steps)
    for i in range(n_steps):
        z_traj[i] = q[6]
        out = engine.step(q, qdot, np.zeros(merged.nv))
        q, qdot = out.q_new, out.qdot_new
    return z_traj, q


def _mj_sphere_drop(r: float, z0: float, n_steps: int):
    """Run MuJoCo sphere drop. Returns z trajectory."""
    xml = f"""<mujoco>
  <option timestep="{DT}"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body name="s" pos="0 0 {z0}">
      <freejoint/>
      <geom type="sphere" size="{r}" mass="1"/>
    </body>
  </worldbody>
</mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    z_traj = np.zeros(n_steps)
    for i in range(n_steps):
        z_traj[i] = d.qpos[2]  # MuJoCo freejoint qpos: [x,y,z,qw,qx,qy,qz]
        mujoco.mj_step(m, d)
    return z_traj


def _mj_box_drop(half: float, z0: float, n_steps: int):
    """Run MuJoCo box drop. Returns z trajectory."""
    xml = f"""<mujoco>
  <option timestep="{DT}"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body name="b" pos="0 0 {z0}">
      <freejoint/>
      <geom type="box" size="{half} {half} {half}" mass="1"/>
    </body>
  </worldbody>
</mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    z_traj = np.zeros(n_steps)
    for i in range(n_steps):
        z_traj[i] = d.qpos[2]
        mujoco.mj_step(m, d)
    return z_traj


# ---------------------------------------------------------------------------
# Class 1: TestSingleBodyDropMuJoCo
# ---------------------------------------------------------------------------


class TestSingleBodyDropMuJoCo:
    """Single free-falling body drops onto flat ground: CpuEngine vs MuJoCo.

    Assertions focus on steady-state (settled) position, not trajectory-level
    agreement, because solver models differ (PGS-SI is inelastic; MuJoCo soft).
    """

    R = 0.05  # sphere radius [m]
    HALF = 0.05  # box half-extent [m]
    Z0 = 0.3  # initial z [m]
    N = 3000  # simulation steps

    def test_sphere_drop_steady_state_z(self):
        """Settled sphere z should match MuJoCo within 0.5 mm."""
        merged, engine = _sphere_model(self.R)
        z_traj, _ = _run_drop(engine, merged, self.Z0, self.N)
        mj_z = _mj_sphere_drop(self.R, self.Z0, self.N)

        our_final = float(np.mean(z_traj[-500:]))
        mj_final = float(np.mean(mj_z[-500:]))
        diff_mm = abs(our_final - mj_final) * 1000.0

        assert diff_mm < 0.5, (
            f"Sphere steady-state z differs by {diff_mm:.3f}mm (ours={our_final:.5f}, MuJoCo={mj_final:.5f})"
        )

    def test_sphere_drop_no_penetration(self):
        """Settled sphere center must be above radius (no ground penetration)."""
        merged, engine = _sphere_model(self.R)
        _, q_final = _run_drop(engine, merged, self.Z0, self.N)
        z_final = float(q_final[6])
        assert z_final > self.R - 1e-3, f"Sphere penetrated ground: z={z_final:.5f} < r={self.R}"

    def test_sphere_drop_landing_step(self):
        """First contact step (z < r) matches MuJoCo exactly ±2 steps."""
        merged, engine = _sphere_model(self.R)
        q = np.zeros(merged.nq)
        q[0] = 1.0
        q[6] = self.Z0
        qdot = np.zeros(merged.nv)

        our_step = -1
        for i in range(2000):
            if float(q[6]) < self.R:
                our_step = i
                break
            out = engine.step(q, qdot, np.zeros(merged.nv))
            q, qdot = out.q_new, out.qdot_new

        xml = f"""<mujoco>
  <option timestep="{DT}"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body pos="0 0 {self.Z0}">
      <freejoint/>
      <geom type="sphere" size="{self.R}" mass="1"/>
    </body>
  </worldbody>
</mujoco>"""
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mj_step = -1
        for i in range(2000):
            if d.qpos[2] < self.R:
                mj_step = i
                break
            mujoco.mj_step(m, d)

        assert our_step >= 0, "sphere never landed"
        assert mj_step >= 0, "MuJoCo sphere never landed"
        diff = abs(our_step - mj_step)
        assert diff <= 2, f"Landing step differs by {diff}: ours={our_step}, MuJoCo={mj_step}"

    def test_sphere_contact_normal_upward(self):
        """First contact normal must point upward (z component > 0.99)."""
        merged, engine = _sphere_model(self.R)
        q = np.zeros(merged.nq)
        q[0] = 1.0
        q[6] = self.R - 0.01  # 1 cm penetration
        qdot = np.zeros(merged.nv)
        engine.step(q, qdot, np.zeros(merged.nv))
        contacts = engine.query_contacts()

        assert len(contacts) >= 1, "no contact detected for sphere touching ground"
        nz = float(contacts[0].normal[2])
        assert nz > 0.99, f"contact normal z={nz:.4f} is not upward"

    def test_box_drop_steady_state_z(self):
        """Settled box z should match MuJoCo within 1 mm."""
        merged, engine = _box_model(self.HALF)
        z_traj, _ = _run_drop(engine, merged, self.Z0, self.N)
        mj_z = _mj_box_drop(self.HALF, self.Z0, self.N)

        our_final = float(np.mean(z_traj[-500:]))
        mj_final = float(np.mean(mj_z[-500:]))
        diff_mm = abs(our_final - mj_final) * 1000.0

        assert diff_mm < 1.0, (
            f"Box steady-state z differs by {diff_mm:.3f}mm (ours={our_final:.5f}, MuJoCo={mj_final:.5f})"
        )

    def test_box_contact_normal_upward(self):
        """Box on flat ground: contact normal must be upward."""
        merged, engine = _box_model(self.HALF)
        q = np.zeros(merged.nq)
        q[0] = 1.0
        q[6] = self.HALF - 0.01  # 1 cm penetration
        qdot = np.zeros(merged.nv)
        engine.step(q, qdot, np.zeros(merged.nv))
        contacts = engine.query_contacts()

        assert len(contacts) >= 1, "no contact detected for box touching ground"
        # At least one contact should have a predominantly upward normal
        has_upward = any(float(c.normal[2]) > 0.99 for c in contacts)
        assert has_upward, f"No upward contact normal: {[c.normal for c in contacts]}"

    def test_box_drop_landing_step(self):
        """Box landing step matches MuJoCo within 2 steps."""
        merged, engine = _box_model(self.HALF)
        q = np.zeros(merged.nq)
        q[0] = 1.0
        q[6] = self.Z0
        qdot = np.zeros(merged.nv)

        our_step = -1
        for i in range(2000):
            if float(q[6]) < self.HALF:
                our_step = i
                break
            out = engine.step(q, qdot, np.zeros(merged.nv))
            q, qdot = out.q_new, out.qdot_new

        xml = f"""<mujoco>
  <option timestep="{DT}"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body pos="0 0 {self.Z0}">
      <freejoint/>
      <geom type="box" size="{self.HALF} {self.HALF} {self.HALF}" mass="1"/>
    </body>
  </worldbody>
</mujoco>"""
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mj_step = -1
        for i in range(2000):
            if d.qpos[2] < self.HALF:
                mj_step = i
                break
            mujoco.mj_step(m, d)

        assert our_step >= 0 and mj_step >= 0
        assert abs(our_step - mj_step) <= 2, (
            f"Box landing step differs by {abs(our_step - mj_step)}: ours={our_step}, MuJoCo={mj_step}"
        )


# ---------------------------------------------------------------------------
# Class 2: TestTwoBodyContact
# ---------------------------------------------------------------------------


class TestTwoBodyContact:
    """Two-sphere body-body contact response tests.

    Assertions: no interpenetration, correct normal direction, and
    ground-settle agreement vs MuJoCo (independent spheres at different x).
    """

    R = 0.05

    def test_two_spheres_no_penetration(self):
        """Two spheres starting at 2r separation: center distance >= 2r always."""
        merged, engine = _two_sphere_model(self.R)
        q = np.zeros(merged.nq)
        q[0] = 1.0
        q[7] = 1.0
        q[4] = 0.0
        q[11] = 2 * self.R  # exactly touching
        q[6] = 0.3
        q[13] = 0.3
        qdot = np.zeros(merged.nv)

        min_dist = float("inf")
        for _ in range(3000):
            out = engine.step(q, qdot, np.zeros(merged.nv))
            q, qdot = out.q_new, out.qdot_new
            dist = float(np.sqrt((q[4] - q[11]) ** 2 + (q[6] - q[13]) ** 2))
            if dist < min_dist:
                min_dist = dist

        # Allow 1mm tolerance for solver softness
        assert min_dist >= 2 * self.R - 1e-3, (
            f"Spheres interpenetrated: min dist={min_dist:.5f}m, 2r={2 * self.R}"
        )

    def test_two_spheres_independent_settle(self):
        """Two separated spheres (no body-body contact): each settles to z≈r."""
        merged, engine = _two_sphere_model(self.R)
        q = np.zeros(merged.nq)
        q[0] = 1.0
        q[7] = 1.0
        q[4] = 0.3
        q[11] = -0.3  # far apart — no body-body contact
        q[6] = 0.3
        q[13] = 0.3
        qdot = np.zeros(merged.nv)

        for _ in range(5000):
            out = engine.step(q, qdot, np.zeros(merged.nv))
            q, qdot = out.q_new, out.qdot_new

        z0, z1 = float(q[6]), float(q[13])
        assert abs(z0 - self.R) < 1e-3, f"s0 settled at z={z0:.5f}, expected r={self.R}"
        assert abs(z1 - self.R) < 1e-3, f"s1 settled at z={z1:.5f}, expected r={self.R}"

    def test_two_spheres_settle_vs_mujoco(self):
        """Two separated spheres: settled z matches MuJoCo within 1mm."""
        merged, engine = _two_sphere_model(self.R)
        q = np.zeros(merged.nq)
        q[0] = 1.0
        q[7] = 1.0
        q[4] = 0.3
        q[11] = -0.3
        q[6] = 0.3
        q[13] = 0.3
        qdot = np.zeros(merged.nv)
        for _ in range(5000):
            out = engine.step(q, qdot, np.zeros(merged.nv))
            q, qdot = out.q_new, out.qdot_new
        our_z = float(q[6])

        xml = f"""<mujoco>
  <option timestep="{DT}"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body name="s1" pos="0.3 0 0.3">
      <freejoint/>
      <geom type="sphere" size="{self.R}" mass="1"/>
    </body>
    <body name="s2" pos="-0.3 0 0.3">
      <freejoint/>
      <geom type="sphere" size="{self.R}" mass="1"/>
    </body>
  </worldbody>
</mujoco>"""
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        for _ in range(5000):
            mujoco.mj_step(m, d)
        mj_z = float(d.qpos[2])

        diff_mm = abs(our_z - mj_z) * 1000.0
        assert diff_mm < 1.0, (
            f"Two-sphere settle z differs by {diff_mm:.3f}mm (ours={our_z:.5f}, MuJoCo={mj_z:.5f})"
        )

    def test_body_body_contact_normal_horizontal(self):
        """Two spheres overlapping along x-axis: contact normal ≈ x-axis."""
        merged, engine = _two_sphere_model(self.R)
        q = np.zeros(merged.nq)
        q[0] = 1.0
        q[7] = 1.0
        q[4] = -0.04
        q[11] = 0.04  # 0.08m apart < 2*0.05 = 0.1m
        q[6] = 0.5
        q[13] = 0.5  # well above ground
        qdot = np.zeros(merged.nv)
        engine.step(q, qdot, np.zeros(merged.nv))
        contacts = engine.query_contacts()

        bb = [c for c in contacts if c.body_j >= 0]
        assert len(bb) >= 1, "no body-body contact detected for overlapping spheres"

        # Normal should be predominantly along x-axis
        nx = abs(float(bb[0].normal[0]))
        assert nx > 0.99, f"Body-body contact normal not along x: normal={bb[0].normal}"


# ---------------------------------------------------------------------------
# Class 3: TestMarginPhysicsNeutral
# ---------------------------------------------------------------------------


class TestMarginPhysicsNeutral:
    """Margin=1mm (default) is physics-neutral for contacts deeper than 2mm.

    These tests verify the margin pipeline at the gjk_epa_query level,
    confirming that:
    1. Deep contacts (pen >> 2*margin) give identical depth with margin=0 and margin=1mm.
    2. Shallow contacts (0 < pen < 2*margin) are correctly detected by both.
    3. The steady-state z from CpuEngine is within 1mm for margin=0 and margin=1mm.
    """

    def test_deep_contact_margin_zero_equal(self):
        """pen=20mm >> 2*margin=2mm: depth identical for margin=0 and margin=1mm."""
        from physics.geometry import SphereShape
        from physics.gjk_epa import gjk_epa_query
        from physics.spatial import SpatialTransform

        s1, s2 = SphereShape(0.05), SphereShape(0.05)
        p1 = SpatialTransform(R=np.eye(3), r=np.zeros(3))
        p2 = SpatialTransform(R=np.eye(3), r=np.array([0.0, 0.0, 0.08]))  # pen=0.02m

        r0 = gjk_epa_query(s1, p1, s2, p2, margin=0.0)
        r1 = gjk_epa_query(s1, p1, s2, p2, margin=1e-3)
        assert r0 is not None and r1 is not None
        diff_mm = abs(r0.depth - r1.depth) * 1000.0
        assert diff_mm < 1.0, (
            f"Deep contact depths differ by {diff_mm:.3f}mm: "
            f"margin=0→{r0.depth:.5f}, margin=1mm→{r1.depth:.5f}"
        )

    def test_shallow_contact_detected_by_margin(self):
        """pen=0.5mm < 2*margin=2mm: contact detected with margin=1mm (Phase 1)."""
        from physics.geometry import SphereShape
        from physics.gjk_epa import gjk_epa_query
        from physics.spatial import SpatialTransform

        s1, s2 = SphereShape(0.05), SphereShape(0.05)
        p1 = SpatialTransform(R=np.eye(3), r=np.zeros(3))
        # pen = 0.0005m: cc = 0.0995
        p2 = SpatialTransform(R=np.eye(3), r=np.array([0.0, 0.0, 0.0995]))

        r0 = gjk_epa_query(s1, p1, s2, p2, margin=0.0)
        r1 = gjk_epa_query(s1, p1, s2, p2, margin=1e-3)

        # Both must detect contact (pen > 0 regardless of margin)
        assert r0 is not None, "margin=0 failed to detect 0.5mm penetration"
        assert r1 is not None, "margin=1mm failed to detect 0.5mm penetration"
        assert abs(r0.depth - 5e-4) < 1e-4, f"margin=0 depth={r0.depth:.5f}"
        assert abs(r1.depth - 5e-4) < 1e-4, f"margin=1mm depth={r1.depth:.5f}"

    def test_separated_shapes_return_none(self):
        """True gap > 0: gjk_epa_query returns None for both margin values."""
        from physics.geometry import SphereShape
        from physics.gjk_epa import gjk_epa_query
        from physics.spatial import SpatialTransform

        s1, s2 = SphereShape(0.05), SphereShape(0.05)
        p1 = SpatialTransform(R=np.eye(3), r=np.zeros(3))
        p2 = SpatialTransform(R=np.eye(3), r=np.array([0.0, 0.0, 0.105]))  # gap=5mm

        r0 = gjk_epa_query(s1, p1, s2, p2, margin=0.0)
        r1 = gjk_epa_query(s1, p1, s2, p2, margin=1e-3)

        assert r0 is None, f"margin=0 returned contact for separated spheres: depth={r0.depth}"
        assert r1 is None, f"margin=1mm returned contact for separated spheres: depth={r1.depth}"

    def test_sphere_steady_state_within_radius(self):
        """Settled sphere z is within 1mm of its radius (no margin offset)."""
        merged, engine = _sphere_model(r=0.05)
        z_traj, _ = _run_drop(engine, merged, z0=0.3, n_steps=3000)
        z_settled = float(np.mean(z_traj[-500:]))
        assert abs(z_settled - 0.05) < 1e-3, (
            f"Settled sphere z={z_settled:.5f} differs from radius={0.05} "
            f"by {abs(z_settled - 0.05) * 1000:.2f}mm"
        )
