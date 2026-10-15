"""CPU-only import proof: deny Warp imports in a fresh process."""
import builtins
import sys

sys.path.insert(0, "/data3/ang.gao/robot_sim_c4_shade_review1")
original = builtins.__import__


def without_warp(name, *args, **kwargs):
    if name == "warp" or name.startswith("warp."):
        raise ImportError("review probe intentionally denies optional Warp")
    return original(name, *args, **kwargs)


builtins.__import__ = without_warp
from optics import warp_execution, warp_native_render

assert not warp_native_render._HAS_WARP
assert len(warp_native_render.WARP_DIRECT_LIGHT_SHADE_ABIS) == 4
assert warp_execution.warp_direct_light_shade_entry is warp_native_render.warp_direct_light_shade_entry
try:
    warp_native_render.warp_direct_light_shade_entry(shadows=False, traversal_diagnostics=False)
except RuntimeError as exc:
    assert "Warp is required" in str(exc)
else:
    raise AssertionError("selector should defer missing-Warp failure until use")
print("PASS fresh-process import with all Warp imports denied")
