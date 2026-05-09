Initiative: q54-optical-pipeline-lab-d1-async-readback-spike
Stage: implementation-measurement-note
Author: codex
Version: v1
Date: 2026-05-09
Status: spike-proven
Related Files: tools/optical_pipeline_lab/go2_backend.py, tools/optical_pipeline_lab/runner.py, tools/optical_pipeline_lab/__main__.py, tests/unit/optics/test_optical_pipeline_lab.py, GPU_OPTICAL_PIPELINE_DESIGN.md, out/optical_pipeline_lab/go2_1080p_shadow_rgb_torch_async_gpu1/frame_timing.csv
Owner Summary: Added the first D1 async D2H spike path for Go2 video RGB readback using Torch pinned host tensors and non-blocking CUDA copies. The steady 1080p result reaches ~9.20 ms frame cadence with overlap_ratio ~0.41, clearing the D1 go/no-go threshold.

# Q54 Optical Pipeline Lab D1 Async Readback Spike

## What Landed

New lab run option:

```text
video_readback_delivery = sync | torch_async
```

New CLI flag:

```bash
python -m tools.optical_pipeline_lab run \
  --readback rgb \
  --video-readback-delivery torch_async
```

The first spike supports:

```text
readback_payload=rgb
```

and intentionally rejects `none`/`full` for this delivery backend.

## Implementation Shape

The async path keeps the existing render path, then for each rendered frame:

```text
wait render ready event
wrap Warp RGB/diagnostic channels as Torch CUDA tensors
copy into pinned CPU tensors with non_blocking=True on a Torch CUDA stream
render next frame
drain previous readback in order
write one frame_timing.csv row per completed frame
```

Frame CSV additions used by the spike:

```text
readback_submit_ms
readback_wait_ms
readback_host_ms
readback_lag_frames
readback_ring_depth
readback_ring_block_count
completed_frame_index
overlap_ratio
```

For `torch_async`, `readback_host_ms` is the CUDA-event measured copy duration.

## GPU1 Run

Command:

```bash
conda run -n env_tilelang_20260119 python -m tools.optical_pipeline_lab run \
  --preset go2_video_ordered_static \
  --out out/optical_pipeline_lab/go2_1080p_shadow_rgb_torch_async_gpu1 \
  --device cuda:1 \
  --width 1920 \
  --height 1080 \
  --frames 10 \
  --warmup-renders 5 \
  --progress-every 1 \
  --readback rgb \
  --video-readback-delivery torch_async
```

All frames completed without BVH stack overflow.

## Results

Compared sync warmup=5 baseline:

```text
sync all10:
  frame_mean    ~= 24.42 ms
  render_mean   ~= 5.56 ms
  readback_mean ~= 18.74 ms

sync tail3:
  frame_mean    ~= 17.65 ms
  render_mean   ~= 5.27 ms
  readback_mean ~= 12.26 ms
```

Async spike:

```text
all10:
  frame_mean       ~= 13.75 ms
  render_mean      ~= 6.23 ms
  readback_submit  ~= 4.71 ms
  readback_wait    ~= 2.43 ms
  readback_copy    ~= 9.26 ms
  overlap_ratio    ~= 0.11

steady frames 3-8:
  frame_mean       ~= 9.20 ms
  render_mean      ~= 6.31 ms
  readback_submit  ~= 0.47 ms
  readback_wait    ~= 2.18 ms
  readback_copy    ~= 9.19 ms
  overlap_ratio    ~= 0.41

tail3:
  frame_mean       ~= 9.16 ms
  render_mean      ~= 5.87 ms
  readback_submit  ~= 0.52 ms
  readback_wait    ~= 4.66 ms
  readback_copy    ~= 9.15 ms
  overlap_ratio    ~= 0.39
```

The first two frames include allocation/submit overhead from pinned host tensor
setup. Steady frames are the useful go/no-go signal.

## Verdict

The D1 go/no-go threshold was:

```text
overlap_ratio > 0.2
```

The steady spike reaches:

```text
overlap_ratio ~= 0.41
```

So real D2H/render overlap is proven on this path.

## Next

Turn this spike into a cleaner ordered delivery primitive:

```text
preallocate pinned host slots
reuse readback buffers instead of allocating during first frames
separate output cadence from one-frame latency explicitly in reports
support a small ring depth option
decide whether RGB8 CUDA pack should happen before async copy
```

## Ring Slot Follow-Up

Added:

```text
--video-readback-ring-depth
```

and changed the `torch_async` path to preallocate pinned host readback slots
before the timed frame loop. A warmup RGB copy now initializes the copy path and
slot buffers outside the measured frame rows.

Command:

```bash
conda run -n env_tilelang_20260119 python -m tools.optical_pipeline_lab run \
  --preset go2_video_ordered_static \
  --out out/optical_pipeline_lab/go2_1080p_shadow_rgb_torch_async_ring2_gpu1 \
  --device cuda:1 \
  --width 1920 \
  --height 1080 \
  --frames 10 \
  --warmup-renders 5 \
  --progress-every 1 \
  --readback rgb \
  --video-readback-delivery torch_async \
  --video-readback-ring-depth 2
```

Result:

```text
all10:
  frame_mean       ~= 10.78 ms
  render_mean      ~= 6.33 ms
  readback_submit  ~= 0.11 ms
  readback_wait    ~= 4.04 ms
  readback_copy    ~= 10.21 ms
  overlap_ratio    ~= 0.35

steady frames 2-8:
  frame_mean       ~= 10.20 ms
  render_mean      ~= 6.40 ms
  readback_submit  ~= 0.10 ms
  readback_wait    ~= 3.42 ms
  readback_copy    ~= 10.20 ms
  overlap_ratio    ~= 0.39

tail3:
  frame_mean       ~= 10.34 ms
  render_mean      ~= 6.13 ms
  readback_submit  ~= 0.10 ms
  readback_wait    ~= 5.84 ms
  readback_copy    ~= 10.33 ms
  overlap_ratio    ~= 0.37
```

Compared to the first spike, submit cost is no longer polluted by first-frame
pinned host allocation:

```text
first spike all10 submit_mean ~= 4.71 ms
ring2 all10 submit_mean       ~= 0.11 ms
```

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
