"""CPU metadata tests: no new command/arena/result capability in this slice."""

from __future__ import annotations

from dataclasses import fields
from math import prod

import pytest

from optics.output_contract import SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES
from optics.warp_native_render import WarpDirectLightBuffers, warp_direct_light_buffer_layout


@pytest.mark.parametrize("num_rays", [0, 1, 15, 1920 * 1080])
def test_physical_abi_is_complete_and_profile_independent(num_rays):
    primary = warp_direct_light_buffer_layout(num_rays, traversal_diagnostics=False)
    assert tuple(name for name, *_ in primary) == (
        "origins",
        "directions",
        "hit_mask",
        "range_m",
        "position_world",
        "normal_world",
        "numeric_instance_id",
        "material_index",
        "bvh_stack_overflow_count",
        "bvh_max_stack_depth",
        "rgb",
        "intensity",
        "shadow_stack_overflow_count",
        "shadow_max_stack_depth",
    )
    public_fields = {f.name for f in fields(WarpDirectLightBuffers) if f.init}
    assert {name for name, *_ in primary} <= public_fields
    assert "output_profile" not in public_fields
    overlay = warp_direct_light_buffer_layout(num_rays, traversal_diagnostics=True)
    assert overlay[:14] == primary
    assert overlay[14:] == tuple((name, "int32", (1,)) for name in SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES)
    assert sum(4 * prod(shape) for _, _, shape in primary) == 80 * num_rays + 16


@pytest.mark.parametrize("count", [-1, True, False, 1.0, None, "1"])
def test_invalid_ray_count_rejected_without_runtime(count):
    with pytest.raises(ValueError, match="num_rays"):
        warp_direct_light_buffer_layout(count, traversal_diagnostics=False)


@pytest.mark.parametrize("mode", [0, 1, None, "on", ()])
def test_diagnostics_mode_requires_exact_bool(mode):
    with pytest.raises(TypeError, match="bool"):
        warp_direct_light_buffer_layout(0, traversal_diagnostics=mode)


def test_buffer_contract_exposes_no_execution_or_lifecycle_api():
    for name in ("enqueue", "launch", "acquire", "finish", "release", "__del__"):
        assert not hasattr(WarpDirectLightBuffers, name)
