"""Unit tests for `physics.contact_tolerances`."""

from physics.contact_tolerances import (
    CONTACT_CONVEX_MARGIN,
    CONTACT_COPLANAR_DOT,
    CONTACT_FACE_ALIGN_THRESHOLD,
    CONTACT_NEAR_PARALLEL_COS,
    DEFAULT_CONTACT_TOLERANCES,
    ContactTolerances,
)


def test_defaults_match_module_constants():
    t = DEFAULT_CONTACT_TOLERANCES
    assert t.convex_margin == CONTACT_CONVEX_MARGIN
    assert t.face_align_threshold == CONTACT_FACE_ALIGN_THRESHOLD
    assert t.coplanar_dot == CONTACT_COPLANAR_DOT
    assert t.near_parallel_cos == CONTACT_NEAR_PARALLEL_COS


def test_near_parallel_cos_is_twice_as_strict_as_ode():
    # ODE uses 0.03. We picked 2× precision → 0.015 (~0.86°).
    assert CONTACT_NEAR_PARALLEL_COS == 0.015


def test_convex_margin_matches_jolt_bullet_default():
    # Session 29 decision: 1 mm convex margin.
    assert CONTACT_CONVEX_MARGIN == 1.0e-3


def test_contact_tolerances_is_immutable():
    import pytest

    t = ContactTolerances()
    with pytest.raises(Exception):
        t.convex_margin = 0.5  # frozen dataclass


def test_gpu_warp_constants_track_cpu_defaults():
    """The GPU CONVEX_MARGIN wp.constant is initialized from the same scalar,
    so CPU and GPU stay in lockstep."""
    try:
        import warp as wp  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("warp not installed")

    from physics.backends.warp.analytical_collision import CONVEX_MARGIN

    # wp.constant on a scalar float returns the bare value in recent Warp.
    assert float(CONVEX_MARGIN) == CONTACT_CONVEX_MARGIN


def test_gjk_epa_face_align_uses_shared_constant():
    from physics.gjk_epa import _FACE_ALIGN_THRESHOLD

    assert _FACE_ALIGN_THRESHOLD == CONTACT_FACE_ALIGN_THRESHOLD
