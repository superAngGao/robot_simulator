Initiative: q54-optical-pipeline-lab-d1-async-readback-helper
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-05-09
Status: implemented
Related Files: tools/optical_pipeline_lab/async_readback.py, tools/optical_pipeline_lab/go2_backend.py, tests/unit/optics/test_optical_pipeline_lab.py, GPU_OPTICAL_PIPELINE_DESIGN.md, MANIFEST.md
Owner Summary: Extracted the D1 Torch async D2H readback ring from the Go2 backend into a reusable Optical Pipeline Lab helper. The helper owns pinned host slots, copy stream, submit timing, CUDA event copy timing, synchronization, and host channel access; the Go2 backend now owns only render/camera metadata and ordered CSV row emission.

# Q54 Optical Pipeline Lab D1 Async Readback Helper

## What Changed

Added:

```text
tools/optical_pipeline_lab/async_readback.py
```

Core types:

```text
TorchAsyncReadbackRing
TorchAsyncReadbackSlot
TorchAsyncReadbackJob
```

The ring helper now owns:

```text
pinned host slot allocation
Torch CUDA copy stream
non_blocking D2H copy submission
submit_ms
CUDA event copy duration
copy completion synchronization
host channel extraction
optional dependency probing
```

The Go2 backend now keeps only:

```text
render execution
camera/frame metadata
ordered drain policy
frame_timing.csv row construction
progress printing
overflow validation
```

## Runtime Check

Ran a short GPU1 smoke after extraction:

```bash
conda run -n env_tilelang_20260119 python -m tools.optical_pipeline_lab run \
  --preset go2_video_ordered_static \
  --out out/optical_pipeline_lab/go2_1080p_shadow_rgb_torch_async_helper_smoke_gpu1 \
  --device cuda:1 \
  --width 1920 \
  --height 1080 \
  --frames 4 \
  --warmup-renders 5 \
  --progress-every 1 \
  --readback rgb \
  --video-readback-delivery torch_async \
  --video-readback-ring-depth 2
```

Result:

```text
frame_p50_ms:              9.405
frame_p90_ms:              14.029
render_execute_mean_ms:    6.412
readback_submit_mean_ms:   0.247
readback_wait_mean_ms:     3.775
readback_host_mean_ms:     9.488
frame_total_mean_ms:       11.005
```

The helper extraction preserved the ring2 async behavior.

## Validation

Ran:

```bash
ruff check tools/optical_pipeline_lab examples/mujoco_menagerie_gpu_preview.py tests/unit/optics/test_optical_pipeline_lab.py
python -m py_compile tools/optical_pipeline_lab/*.py tests/unit/optics/test_optical_pipeline_lab.py examples/mujoco_menagerie_gpu_preview.py
python -m pytest tests/unit/optics/test_optical_pipeline_lab.py -q
python -m pytest --collect-only -q tests/unit/optics tests/unit/sensing tests/gpu/test_optical_warp_executor.py tests/gpu/test_optical_gpu_runtime.py
```

Results:

```text
ruff: passed
py_compile: passed
lab tests: 25 passed
Q54 collect-only: 159 tests collected
```

## Next

The next cleanup step is to make the Go2 ordered drain policy less ad hoc:

```text
explicit pending queue instead of a single pending job
ring depth > 2 behavior and backpressure counters
all-frame vs steady-tail summary helper
event/timing naming that can survive RenderSession extraction
```
