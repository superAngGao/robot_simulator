# Q54 Optical Pipeline Lab — D2 Review Follow-up

Date: 2026-05-09
Author: Codex

Claude review verdict: D2 RGB8 Pack is ready to merge. Two follow-up items were
called out:

1. Add unit tests for async readback ring depth validation and RGB pack
   import-error path.
2. Add a comment documenting that `TorchAsyncReadbackRing.submit()` expects the
   caller to have waited on `result.ready_event` before submitting the Torch
   copy.

Applied follow-up:

- `tools/optical_pipeline_lab/async_readback.py`
  - Added a comment in `TorchAsyncReadbackRing.submit()` documenting the
    caller-owned ready-event synchronization contract.
- `tests/unit/optics/test_optical_pipeline_lab.py`
  - Added `test_async_readback_ring_rejects_invalid_depth`.
  - Added `test_async_readback_job_uses_start_to_done_event_order`.
  - Added `test_rgb_pack_raises_import_error_when_warp_is_unavailable`.
- `MANIFEST.md`
  - Updated Q54 collected test count from 161 to 164.

Validation:

```text
ruff check tools/optical_pipeline_lab examples/mujoco_menagerie_gpu_preview.py tests/unit/optics/test_optical_pipeline_lab.py
python -m py_compile tools/optical_pipeline_lab/*.py tests/unit/optics/test_optical_pipeline_lab.py examples/mujoco_menagerie_gpu_preview.py
python -m pytest tests/unit/optics/test_optical_pipeline_lab.py -q
python -m pytest --collect-only -q tests/unit/optics tests/unit/sensing tests/gpu/test_optical_warp_executor.py tests/gpu/test_optical_gpu_runtime.py
```

Results:

```text
ruff: all checks passed
py_compile: passed
lab unit tests: 30 passed
Q54 collect-only: 164 tests collected
```
