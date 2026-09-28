"""Real-runtime checks for the C4c2a import and shared audit boundary."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmarks.optical.eligibility import verified_runtime_entry_point_audit
from benchmarks.optical.instrumentation import (
    instrument_runtime_entry_points,
    invoke_instrumented_entry_point,
)
from optical_worker.timing import (
    RuntimeOperationTrace,
    TimedBoundaryViolation,
    active_timed_boundary,
    native_timed_submission,
)

pytestmark = pytest.mark.gpu
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "entry", ["optics.host_bvh.build_host_median_bvh", "optics.device_bvh.build_host_median_bvh"]
)
def test_real_moved_and_legacy_bvh_entry_hooks_share_production_timing(entry):
    audit = verified_runtime_entry_point_audit(operation="acceleration", entry_point=entry)
    trace = RuntimeOperationTrace(row_id="moved-bvh", lane="warp", boundary="R1")
    bounds = np.empty((0, 3), np.float32)
    keys = np.empty(0, np.int64)
    with pytest.raises(TimedBoundaryViolation, match="allocation is forbidden"):
        with instrument_runtime_entry_points((audit,)) as observation:
            with native_timed_submission(
                row_id="moved-bvh", lane="warp", boundary="R1", operation_trace=trace
            ):
                invoke_instrumented_entry_point(entry, bounds, bounds, keys, leaf_size=4)
    assert observation.violation_hook_id == "numpy.empty"
    assert [(r.operation, r.capture_source) for r in trace.evidence().records] == [
        ("acceleration", "runtime_hook")
    ]
    assert active_timed_boundary() is None


@pytest.mark.parametrize("entry", ["optics.warp_native_render", "optics.device_bvh", "optics.warp_execution"])
def test_warp_only_import_does_not_attempt_torch_or_cuda_optical(entry):
    code = """
import builtins, importlib, sys
attempts = []
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] == 'torch' or name.startswith('optics.cuda_'):
        attempts.append(name)
        raise ImportError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
importlib.import_module(sys.argv[1])
assert not attempts, attempts
assert not any(n == 'torch' or n.startswith('torch.') or n.startswith('optics.cuda_') for n in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, entry], cwd=ROOT, text=True, capture_output=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
