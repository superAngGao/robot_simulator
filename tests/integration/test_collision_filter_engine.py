"""
End-to-end CollisionFilter integration with CpuEngine and GpuEngine.

`tests/unit/collision/test_collision_filter.py` already covers the
CollisionFilter class itself (group/mask logic, exclude pairs,
auto-exclude) and its integration with `AABBSelfCollision.build_pairs()`
and `load_urdf(collision_exclude_pairs=...)`.

This file targets the missing layer: that the filter actually
prevents physics contact when consumed by the running engines, and
that CPU and GPU agree under non-trivial filter configurations.

Three filter configurations × two engines = 6 contact-presence checks
plus 2 cross-engine consistency checks.

Setup: two single-sphere bodies overlapping in mid-air (no ground contact
to avoid noise from ground narrowphase).
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.collision_filter import CollisionFilter
from physics.cpu_engine import CpuEngine
from physics.dynamics_cache import DynamicsCache
from physics.geometry import BodyCollisionGeometry, ShapeInstance, SphereShape
from physics.joint import FreeJoint
from physics.merged_model import merge_models
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
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


def _ball_model(mass: float = 1.0, radius: float = 0.05) -> RobotModel:
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


def _build_two_ball_scene(filter_setup):
    """Two single-sphere bodies, with the given filter setup function applied
    to a fresh CollisionFilter(2) before merging.

    `filter_setup(f)` mutates the filter in place.
    """
    f = CollisionFilter(2)
    filter_setup(f)
    merged = merge_models(
        robots={"a": _ball_model(), "b": _ball_model()},
        collision_filter=f,
    )
    return merged, f


def _set_overlapping_q(merged):
    """q for two balls overlapping along x by 0.02 (centers 0.08 apart, radius 0.05 each)."""
    q0 = merged.tree.default_state()[0].copy()
    rs_a = merged.robot_slices["a"]
    rs_b = merged.robot_slices["b"]
    # FreeJoint q layout per-robot: [qw, qx, qy, qz, px, py, pz]
    q0[rs_a.q_slice.start + 4] = 0.0
    q0[rs_a.q_slice.start + 6] = 1.0  # mid-air
    q0[rs_b.q_slice.start + 4] = 0.08
    q0[rs_b.q_slice.start + 6] = 1.0
    return q0


def _cpu_body_body_count(merged, q0):
    """Number of body-body contacts CPU narrowphase reports for the given state."""
    cpu = CpuEngine(merged, dt=2e-4)
    cache = DynamicsCache.from_tree(merged.tree, q0, np.zeros(merged.nv), 2e-4)
    contacts = cpu._detect_contacts(cache)
    return sum(1 for c in contacts if c.body_i != -1 and c.body_j != -1)


def _gpu_body_body_count(merged, q0):
    """Number of body-body contacts GPU narrowphase reports for the given state."""
    gpu = GpuEngine(merged, num_envs=1, device="cuda:0", dt=2e-4)
    gpu.reset(q0=q0)
    gpu.step(np.zeros((1, 0)), 2e-4)
    count = int(gpu._contact_count.numpy()[0])
    n_body_body = 0
    for i in range(count):
        if gpu._contact_active.numpy()[0, i] == 1 and int(gpu._contact_bj.numpy()[0, i]) >= 0:
            n_body_body += 1
    return n_body_body


# Filter setup functions (applied to CollisionFilter(2))


def _setup_default(f):
    """Default: both bodies have group=0xFFFFFFFF and mask=0xFFFFFFFF (collide with all)."""
    pass


def _setup_bitmask_isolated(f):
    """Both bodies have mask=0 → collide with nothing."""
    f.set_group_mask(0, group=0b01, mask=0b00)
    f.set_group_mask(1, group=0b01, mask=0b00)


def _setup_separate_groups_no_overlap(f):
    """Body 0 in group 0b01 expects mask 0b01; body 1 in group 0b10 expects mask 0b10.
    Cross-collide check: (group_0=0b01) & (mask_1=0b10) = 0 → no collision.
    Symmetric: (group_1=0b10) & (mask_0=0b01) = 0 → no collision.
    """
    f.set_group_mask(0, group=0b01, mask=0b01)
    f.set_group_mask(1, group=0b10, mask=0b10)


def _setup_separate_groups_with_overlap(f):
    """Body 0 in group 0b01 with mask 0b10; body 1 in group 0b10 with mask 0b01.
    (group_0 & mask_1) = 0b01 & 0b01 = 0b01 ≠ 0 → collide.
    (group_1 & mask_0) = 0b10 & 0b10 = 0b10 ≠ 0 → collide.
    """
    f.set_group_mask(0, group=0b01, mask=0b10)
    f.set_group_mask(1, group=0b10, mask=0b01)


def _setup_explicit_exclude(f):
    """Bitmask permits but explicit exclude_pair blocks."""
    # default group/mask both 0xFFFFFFFF allow collision
    f.exclude_pair(0, 1)


# ---------------------------------------------------------------------------
# 1. CPU end-to-end filter behavior
# ---------------------------------------------------------------------------


class TestCpuFilterBehavior:
    """CpuEngine narrowphase must respect CollisionFilter."""

    def test_default_filter_produces_contact(self):
        """Sanity baseline: default filter (no isolation) → CPU sees the contact."""
        merged, _ = _build_two_ball_scene(_setup_default)
        q0 = _set_overlapping_q(merged)
        n = _cpu_body_body_count(merged, q0)
        assert n == 1, f"default filter should allow contact, got {n}"

    def test_bitmask_isolated_blocks_contact(self):
        """Both bodies mask=0 → CPU must NOT report a contact."""
        merged, _ = _build_two_ball_scene(_setup_bitmask_isolated)
        q0 = _set_overlapping_q(merged)
        n = _cpu_body_body_count(merged, q0)
        assert n == 0, f"bitmask isolation should block contact, got {n}"

    def test_separate_groups_no_overlap_blocks_contact(self):
        """Group 0b01 + mask 0b01 vs group 0b10 + mask 0b10 → no collision."""
        merged, _ = _build_two_ball_scene(_setup_separate_groups_no_overlap)
        q0 = _set_overlapping_q(merged)
        n = _cpu_body_body_count(merged, q0)
        assert n == 0, f"separate groups (no overlap) should block contact, got {n}"

    def test_separate_groups_with_overlap_allows_contact(self):
        """Group 0b01 mask 0b10 vs group 0b10 mask 0b01 → collide."""
        merged, _ = _build_two_ball_scene(_setup_separate_groups_with_overlap)
        q0 = _set_overlapping_q(merged)
        n = _cpu_body_body_count(merged, q0)
        assert n == 1, f"separate groups with overlapping masks should allow contact, got {n}"

    def test_explicit_exclude_overrides_bitmask(self):
        """Default bitmask permits, but exclude_pair(0,1) blocks the contact."""
        merged, f = _build_two_ball_scene(_setup_explicit_exclude)
        # Sanity: filter should report (0,1) as not collide
        assert not f.should_collide(0, 1)
        q0 = _set_overlapping_q(merged)
        n = _cpu_body_body_count(merged, q0)
        assert n == 0, f"explicit exclude_pair should block contact, got {n}"


# ---------------------------------------------------------------------------
# 2. GPU end-to-end filter behavior
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_WARP, reason="Warp or CUDA not available")
class TestGpuFilterBehavior:
    """GpuEngine narrowphase must respect the same filter (via static
    collision_excluded matrix built from the filter at engine construction)."""

    def test_default_filter_produces_contact(self):
        merged, _ = _build_two_ball_scene(_setup_default)
        q0 = _set_overlapping_q(merged)
        assert _gpu_body_body_count(merged, q0) == 1

    def test_bitmask_isolated_blocks_contact(self):
        merged, _ = _build_two_ball_scene(_setup_bitmask_isolated)
        q0 = _set_overlapping_q(merged)
        assert _gpu_body_body_count(merged, q0) == 0

    def test_separate_groups_no_overlap_blocks_contact(self):
        merged, _ = _build_two_ball_scene(_setup_separate_groups_no_overlap)
        q0 = _set_overlapping_q(merged)
        assert _gpu_body_body_count(merged, q0) == 0

    def test_separate_groups_with_overlap_allows_contact(self):
        merged, _ = _build_two_ball_scene(_setup_separate_groups_with_overlap)
        q0 = _set_overlapping_q(merged)
        assert _gpu_body_body_count(merged, q0) == 1

    def test_explicit_exclude_overrides_bitmask(self):
        merged, _ = _build_two_ball_scene(_setup_explicit_exclude)
        q0 = _set_overlapping_q(merged)
        assert _gpu_body_body_count(merged, q0) == 0


# ---------------------------------------------------------------------------
# 3. CPU/GPU agreement under non-trivial filter
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_WARP, reason="Warp or CUDA not available")
class TestCpuGpuFilterConsistency:
    """CPU and GPU must give the same answer under any filter configuration —
    even when the filter blocks contacts that the geometry would otherwise allow."""

    @pytest.mark.parametrize(
        "setup,expected",
        [
            (_setup_default, 1),
            (_setup_bitmask_isolated, 0),
            (_setup_separate_groups_no_overlap, 0),
            (_setup_separate_groups_with_overlap, 1),
            (_setup_explicit_exclude, 0),
        ],
    )
    def test_cpu_gpu_agree_on_filter_outcome(self, setup, expected):
        merged, _ = _build_two_ball_scene(setup)
        q0 = _set_overlapping_q(merged)
        n_cpu = _cpu_body_body_count(merged, q0)
        n_gpu = _gpu_body_body_count(merged, q0)
        assert n_cpu == expected, f"CPU got {n_cpu}, expected {expected}"
        assert n_gpu == expected, f"GPU got {n_gpu}, expected {expected}"
        assert n_cpu == n_gpu, f"CPU/GPU disagree: cpu={n_cpu} gpu={n_gpu}"
