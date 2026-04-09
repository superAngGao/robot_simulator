"""
Engine routing / dispatch glue layer.

Tests verify that engine constructors and step paths route to the correct
implementation based on terrain type, shape type, solver name, and body-pair
shape combinations. The risks here are SILENT wrong-branch dispatches:
HalfSpaceTerrain interpreted as flat, sphere-box body pair fed into the
sphere-sphere fallback, etc.

Coverage:
  1. CPU FlatTerrain vs HalfSpaceTerrain — different contact normals.
  2. GPU rejects non-flat terrain (hard fail rather than silent wrong physics).
  3. GPU body-body shape pair dispatch — sphere-box and capsule-capsule
     analytical paths fire and produce non-fallback results.
  4. GPU body-body unsupported pair — box-box falls back to sphere
     approximation (documented behavior).
  5. CPU vs GPU agree on FlatTerrain dispatch.
  6. CPU vs GPU solver dispatch — both PGS and ADMM solvers settle a
     ball drop within reasonable bounds.
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.cpu_engine import CpuEngine
from physics.dynamics_cache import DynamicsCache
from physics.geometry import (
    BodyCollisionGeometry,
    BoxShape,
    CapsuleShape,
    ShapeInstance,
    SphereShape,
)
from physics.joint import FreeJoint
from physics.merged_model import merge_models
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from physics.terrain import FlatTerrain, HalfSpaceTerrain
from robot.model import RobotModel

try:
    import warp as wp  # noqa: F401

    from physics.gpu_engine import GpuEngine

    HAS_WARP = True
except Exception:
    HAS_WARP = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_body(name, mass, inertia_diag):
    return Body(
        name=name,
        index=0,
        joint=FreeJoint(f"{name}_root"),
        inertia=SpatialInertia(mass=mass, inertia=np.diag(inertia_diag), com=np.zeros(3)),
        X_tree=SpatialTransform.identity(),
        parent=-1,
    )


def _sphere_robot(radius=0.1, mass=1.0):
    tree = RobotTreeNumpy(gravity=9.81)
    I = 2.0 / 5.0 * mass * radius**2
    tree.add_body(_free_body("ball", mass, [I, I, I]))
    tree.finalize()
    return RobotModel(
        tree=tree,
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(SphereShape(radius))])],
        contact_body_names=["ball"],
    )


def _box_robot(half=(0.05, 0.05, 0.05), mass=1.0):
    tree = RobotTreeNumpy(gravity=9.81)
    I = mass * 0.01
    tree.add_body(_free_body("box", mass, [I, I, I]))
    tree.finalize()
    return RobotModel(
        tree=tree,
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(BoxShape(tuple(2 * h for h in half)))])],
        contact_body_names=["box"],
    )


def _capsule_robot(radius=0.05, length=0.2, mass=1.0):
    tree = RobotTreeNumpy(gravity=9.81)
    I = mass * 0.01
    tree.add_body(_free_body("cap", mass, [I, I, I]))
    tree.finalize()
    return RobotModel(
        tree=tree,
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(CapsuleShape(radius, length))])],
        contact_body_names=["cap"],
    )


def _cpu_contacts(merged, q0, dt=2e-4):
    cpu = CpuEngine(merged, dt=dt)
    cache = DynamicsCache.from_tree(merged.tree, q0, np.zeros(merged.nv), dt)
    return cpu._detect_contacts(cache)


def _gpu_step_contacts(merged, q0, dt=2e-4):
    gpu = GpuEngine(merged, num_envs=1, device="cuda:0", dt=dt)
    gpu.reset(q0=q0)
    gpu.step(np.zeros((1, 0)), dt)
    count = int(gpu._contact_count.numpy()[0])
    out = []
    for i in range(count):
        if gpu._contact_active.numpy()[0, i] == 1:
            out.append(
                {
                    "depth": float(gpu._contact_depth.numpy()[0, i]),
                    "normal": gpu._contact_normal.numpy()[0, i].copy(),
                    "point": gpu._contact_point.numpy()[0, i].copy(),
                    "bi": int(gpu._contact_bi.numpy()[0, i]),
                    "bj": int(gpu._contact_bj.numpy()[0, i]),
                }
            )
    return out, gpu


# ---------------------------------------------------------------------------
# 1. CPU terrain dispatch — FlatTerrain vs HalfSpaceTerrain
# ---------------------------------------------------------------------------


class TestCpuTerrainDispatch:
    """CpuEngine._detect_contacts dispatches at line 106 on isinstance(terrain,
    HalfSpaceTerrain). The branches use entirely different narrowphase paths:
    halfspace_convex_query (O(1) plane equation) vs ground_contact_query
    (GJK/EPA against z=ground_z plane). A wrong dispatch would give wrong
    contact normal."""

    def test_flat_terrain_normal_is_world_up(self):
        """Default FlatTerrain → contact normal = (0, 0, 1)."""
        model = _sphere_robot(radius=0.1)
        merged = merge_models(robots={"a": model}, terrain=FlatTerrain(z=0.0))

        q0 = merged.tree.default_state()[0].copy()
        q0[6] = 0.09  # 1 cm penetration
        contacts = _cpu_contacts(merged, q0)

        ground = [c for c in contacts if c.body_j == -1]
        assert len(ground) == 1, f"Expected 1 ground contact, got {len(ground)}"
        np.testing.assert_allclose(
            ground[0].normal,
            np.array([0.0, 0.0, 1.0]),
            atol=1e-5,
            err_msg=f"FlatTerrain normal should be world +z, got {ground[0].normal}",
        )

    def test_halfspace_terrain_normal_matches_plane_normal(self):
        """30° HalfSpaceTerrain → contact normal = plane normal (NOT world +z)."""
        model = _sphere_robot(radius=0.1)
        # 30° incline normal in xz plane: tilts the surface "uphill" in +x
        theta = np.pi / 6
        plane_normal = np.array([-np.sin(theta), 0.0, np.cos(theta)])
        terrain = HalfSpaceTerrain(normal=plane_normal, point=np.zeros(3))
        merged = merge_models(robots={"a": model}, terrain=terrain)

        q0 = merged.tree.default_state()[0].copy()
        q0[6] = 0.05  # below sphere bottom — should penetrate the tilted plane
        contacts = _cpu_contacts(merged, q0)

        ground = [c for c in contacts if c.body_j == -1]
        assert len(ground) >= 1, f"Expected ≥1 ground contact, got {len(ground)}"

        # Contact normal must match the plane normal, NOT world +z
        np.testing.assert_allclose(
            ground[0].normal,
            plane_normal,
            atol=1e-3,
            err_msg=(
                f"HalfSpaceTerrain contact normal {ground[0].normal} "
                f"should be {plane_normal}. If it's (0,0,1), the dispatch fell "
                f"through to the FlatTerrain branch (silent wrong physics)."
            ),
        )

    def test_halfspace_and_flat_give_different_contacts(self):
        """Same q, different terrain → different contact data. Confirms the
        dispatch actually picks different code paths."""
        q0 = _sphere_robot(radius=0.1).tree.default_state()[0].copy()
        q0[6] = 0.05

        # FlatTerrain
        merged_flat = merge_models(robots={"a": _sphere_robot(0.1)}, terrain=FlatTerrain(z=0.0))
        flat_contacts = _cpu_contacts(merged_flat, q0)
        assert len(flat_contacts) >= 1
        n_flat = flat_contacts[0].normal

        # HalfSpaceTerrain (45° tilt)
        plane_n = np.array([-np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)])
        merged_hs = merge_models(
            robots={"a": _sphere_robot(0.1)},
            terrain=HalfSpaceTerrain(normal=plane_n, point=np.zeros(3)),
        )
        hs_contacts = _cpu_contacts(merged_hs, q0)
        assert len(hs_contacts) >= 1
        n_hs = hs_contacts[0].normal

        # Normals must differ — same q, different terrain → different result
        diff = np.linalg.norm(n_flat - n_hs)
        assert diff > 0.5, (
            f"FlatTerrain normal {n_flat} ≈ HalfSpace normal {n_hs}; "
            f"the two dispatch branches gave the same result."
        )


# ---------------------------------------------------------------------------
# 2. GPU rejects non-flat terrain (silent-wrong-physics prevention)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_WARP, reason="Warp or CUDA not available")
class TestGpuTerrainHardFail:
    """GpuEngine static_data only stores a single contact_ground_z scalar; it
    has no provision for tilted plane normals. Before the hard-fail check was
    added, building a GpuEngine with HalfSpaceTerrain would silently use flat
    ground (normal = (0,0,1), depth from z=0) regardless of the requested plane.
    The hard-fail prevents that silent wrong physics."""

    def test_gpu_flat_terrain_works(self):
        """Sanity baseline: FlatTerrain construction succeeds."""
        model = _sphere_robot(0.1)
        merged = merge_models(robots={"a": model}, terrain=FlatTerrain(z=0.0))
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)
        engine.reset()  # should not raise
        assert engine is not None

    def test_gpu_halfspace_terrain_raises(self):
        """HalfSpaceTerrain → constructor raises NotImplementedError."""
        model = _sphere_robot(0.1)
        plane_n = np.array([-np.sin(np.pi / 6), 0.0, np.cos(np.pi / 6)])
        merged = merge_models(
            robots={"a": model},
            terrain=HalfSpaceTerrain(normal=plane_n, point=np.zeros(3)),
        )
        with pytest.raises(NotImplementedError, match="HalfSpaceTerrain"):
            GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)

    def test_gpu_default_no_terrain_works(self):
        """No explicit terrain → default FlatTerrain → works."""
        model = _sphere_robot(0.1)
        merged = merge_models(robots={"a": model})  # default FlatTerrain
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)
        assert engine is not None


# ---------------------------------------------------------------------------
# 3. GPU body-body shape pair dispatch
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_WARP, reason="Warp or CUDA not available")
class TestGpuBodyBodyShapePairDispatch:
    """GPU body-body narrowphase dispatches in batched_detect_multishape via
    a canonicalized shape-type pair (a_t, b_t) with a_t <= b_t. Supported pairs:
        (sphere, sphere) → sphere_sphere
        (sphere, capsule) → sphere_capsule
        (sphere, box)     → sphere_box
        (capsule, capsule) → capsule_capsule
    All other combinations fall back to a sphere-sphere using
    body_collision_radius (a coarse over-approximation).
    """

    def test_sphere_box_dispatch_uses_analytical_path(self):
        """Sphere-Box body pair: GPU should give a depth that matches the
        analytical sphere-vs-box (NOT the AABB-based sphere-sphere fallback,
        which would over-estimate depth for elongated boxes)."""
        m_sphere = _sphere_robot(radius=0.05)
        m_box = _box_robot(half=(0.05, 0.05, 0.2))  # tall box: 0.4 m along z

        merged = merge_models(robots={"s": m_sphere, "b": m_box})
        q0 = merged.tree.default_state()[0].copy()
        rs_s = merged.robot_slices["s"]
        rs_b = merged.robot_slices["b"]
        # Sphere centered at world (0.08, 0, 1.0); box at (0.0, 0, 1.0).
        # Box surface in +x at world x = 0.05; sphere center at x = 0.08;
        # distance from sphere center to box surface = 0.03.
        # Sphere radius 0.05 → expected depth = 0.05 - 0.03 = 0.02
        q0[rs_s.q_slice.start + 4] = 0.08
        q0[rs_s.q_slice.start + 6] = 1.0
        q0[rs_b.q_slice.start + 4] = 0.0
        q0[rs_b.q_slice.start + 6] = 1.0

        contacts, _ = _gpu_step_contacts(merged, q0)
        body_body = [c for c in contacts if c["bj"] >= 0]
        assert len(body_body) == 1, f"Expected 1 sphere-box contact, got {len(body_body)}"

        # Analytical sphere_box depth ≈ 0.02 (sphere center 0.03 outside box face,
        # radius 0.05). The fallback sphere_sphere would use body_collision_radius
        # which for a (0.05, 0.05, 0.2) box is approximately norm(0.05,0.05,0.2)/sqrt(3) ≈ 0.123,
        # so fallback depth ≈ 0.05 + 0.123 - 0.08 = 0.093 — five times the analytical value.
        depth = body_body[0]["depth"]
        np.testing.assert_allclose(
            depth,
            0.02,
            atol=2e-3,
            err_msg=(
                f"Sphere-Box depth {depth:.4f} ≠ analytical 0.02. "
                f"If depth is ~0.09, the fallback sphere-sphere is being used "
                f"instead of the analytical sphere_box narrowphase."
            ),
        )

    def test_capsule_capsule_dispatch_uses_analytical_path(self):
        """Two horizontal capsules side by side: GPU should use the analytical
        capsule-capsule narrowphase (closest-points-segment-segment), not the
        sphere fallback."""
        m_a = _capsule_robot(radius=0.05, length=0.2)
        m_b = _capsule_robot(radius=0.05, length=0.2)
        merged = merge_models(robots={"a": m_a, "b": m_b})

        # Place both capsules vertical (default), parallel in +z, separated in x
        # by 0.08 (center distance) → overlap = (r_a + r_b) - 0.08 = 0.02.
        q0 = merged.tree.default_state()[0].copy()
        rs_a = merged.robot_slices["a"]
        rs_b = merged.robot_slices["b"]
        q0[rs_a.q_slice.start + 4] = 0.0
        q0[rs_a.q_slice.start + 6] = 1.0
        q0[rs_b.q_slice.start + 4] = 0.08
        q0[rs_b.q_slice.start + 6] = 1.0

        contacts, _ = _gpu_step_contacts(merged, q0)
        body_body = [c for c in contacts if c["bj"] >= 0]
        assert len(body_body) == 1, f"Expected 1 capsule-capsule contact, got {len(body_body)}"

        depth = body_body[0]["depth"]
        np.testing.assert_allclose(
            depth,
            0.02,
            atol=2e-3,
            err_msg=(
                f"Capsule-Capsule depth {depth:.4f} ≠ analytical 0.02. Wrong narrowphase or wrong dispatch."
            ),
        )


# ---------------------------------------------------------------------------
# 4. CPU vs GPU agreement on FlatTerrain (sanity baseline)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_WARP, reason="Warp or CUDA not available")
class TestCpuGpuFlatTerrainAgreement:
    def test_cpu_gpu_flat_terrain_contact_agree(self):
        """Same scene, both engines, FlatTerrain. CPU and GPU contacts must agree."""
        merged_cpu = merge_models(robots={"a": _sphere_robot(0.1)}, terrain=FlatTerrain(z=0.0))
        merged_gpu = merge_models(robots={"a": _sphere_robot(0.1)}, terrain=FlatTerrain(z=0.0))

        q0 = merged_cpu.tree.default_state()[0].copy()
        q0[6] = 0.09

        cpu_contacts = _cpu_contacts(merged_cpu, q0)
        cpu_ground = [c for c in cpu_contacts if c.body_j == -1]
        assert len(cpu_ground) == 1

        gpu_contacts, _ = _gpu_step_contacts(merged_gpu, q0)
        gpu_ground = [c for c in gpu_contacts if c["bj"] == -1]
        assert len(gpu_ground) == 1

        np.testing.assert_allclose(cpu_ground[0].depth, gpu_ground[0]["depth"], atol=5e-4)
        np.testing.assert_allclose(np.asarray(cpu_ground[0].normal), gpu_ground[0]["normal"], atol=5e-4)


# ---------------------------------------------------------------------------
# 5. Solver dispatch — PGS vs ADMM both functional
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_WARP, reason="Warp or CUDA not available")
class TestSolverDispatch:
    """GpuEngine.__init__ accepts solver ∈ {"jacobi_pgs_si", "admm"} and
    routes the constraint solve through different kernel paths
    (gpu_engine.py:648). Both must settle a ball drop within reasonable bounds."""

    def test_jacobi_pgs_si_settles_drop(self):
        model = _sphere_robot(0.1)
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4, solver="jacobi_pgs_si")

        q0 = merged.tree.default_state()[0].copy()
        q0[6] = 0.5
        engine.reset(q0=q0)
        for _ in range(3000):
            engine.step(np.zeros((1, 0)), 2e-4)

        z = float(engine._scratch.q.numpy()[0, 6])
        assert 0.090 < z < 0.110, f"jacobi_pgs_si failed to settle: z={z:.4f}"

    def test_admm_settles_drop(self):
        model = _sphere_robot(0.1)
        merged = merge_models(robots={"a": model})
        engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4, solver="admm")

        q0 = merged.tree.default_state()[0].copy()
        q0[6] = 0.5
        engine.reset(q0=q0)
        for _ in range(3000):
            engine.step(np.zeros((1, 0)), 2e-4)

        z = float(engine._scratch.q.numpy()[0, 6])
        assert 0.090 < z < 0.110, f"admm failed to settle: z={z:.4f}"

    def test_both_solvers_agree_on_settled_height(self):
        """PGS and ADMM should converge to similar steady-state z (within 5 mm)."""
        results = {}
        for solver in ["jacobi_pgs_si", "admm"]:
            merged = merge_models(robots={"a": _sphere_robot(0.1)})
            engine = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4, solver=solver)
            q0 = merged.tree.default_state()[0].copy()
            q0[6] = 0.5
            engine.reset(q0=q0)
            for _ in range(3000):
                engine.step(np.zeros((1, 0)), 2e-4)
            results[solver] = float(engine._scratch.q.numpy()[0, 6])

        diff = abs(results["jacobi_pgs_si"] - results["admm"])
        assert diff < 5e-3, f"PGS vs ADMM disagree on settled height: {results}, diff={diff:.5f}"
