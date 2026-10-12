"""Analytic arithmetic and strict contact-boundary checks for audit oracles."""

import numpy as np
import pytest

from .models import G
from .physics_metrics import first_freefall_penetration


@pytest.mark.parametrize(
    "dt, step, expected_z",
    [(0.0004, 196, 0.4196973024), (0.0002, 391, 0.4199280336), (0.0001, 782, 0.4199663907)],
)
def test_first_freefall_penetration(dt, step, expected_z):
    actual_step, actual_z = first_freefall_penetration(0.45, 0.42, dt)
    assert actual_step == step
    assert abs(actual_z - expected_z) < 1e-15
    previous_z = 0.45 - G * dt**2 * (step - 1) * step / 2
    assert 0.42 - previous_z <= 1e-10 < 0.42 - actual_z


def test_tangency_is_not_a_detected_contact():
    dt = 0.01
    z0 = 0.42 + G * dt**2
    step, _ = first_freefall_penetration(z0, 0.42, dt)
    assert step == 2  # step 1 is tangent; detector requires positive depth


@pytest.mark.parametrize("z0, dt", [(0.41, 0.001), (0.45, 0), (0.45, -0.001), (np.nan, 0.001)])
def test_crossing_rejects_invalid_fixture(z0, dt):
    with pytest.raises(ValueError):
        first_freefall_penetration(z0, 0.42, dt)
