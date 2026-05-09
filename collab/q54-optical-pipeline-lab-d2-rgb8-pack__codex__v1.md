# Q54 Optical Pipeline Lab — D2 RGB8 Pack Implementation Note

Date: 2026-05-09
Author: Codex

## Scope

Implemented the first RGB8 delivery-pack path for the Optical Pipeline Lab.
This is a Warp-based GPU pack helper, not a dedicated CUDA extension yet. The
goal for this slice is to quantify payload reduction and validate the lab
delivery contract before deciding whether a CUDA pack extension is worth adding.

## Code Changes

- `tools/optical_pipeline_lab/rgb_pack.py`
  - Adds `pack_linear_rgb_to_preview_uint8()`.
  - Converts linear float32 RGB to preview RGB8 on device.
  - Applies NaN sanitization, clamp to `[0, 1]`, gamma `1/2.2`, round to uint8.
  - Returns a new `OpticalComputeResult` with an added `rgb8` channel and a
    fresh `ready_event`.
- `tools/optical_pipeline_lab/go2_backend.py`
  - Adds `--video-readback=rgb8`.
  - Supports sync RGB8 staging.
  - Supports `--video-readback-delivery=torch_async` with RGB8 ring slots.
  - Synchronizes the RGB8 pack ready event before handing the packed channel to
    the readback path, so timings include real pack completion.
- `tools/optical_pipeline_lab/scenarios.py`
  - Enables `ReadbackPayload.RGB8` for implemented lab runs.
- `tools/optical_pipeline_lab/runner.py`
  - Maps `readback=rgb8` to `output_profile=rgb_preview`.
  - Allows `torch_async` delivery for RGB8 payloads.
- `tools/optical_pipeline_lab/__main__.py`
  - Exposes `--readback rgb8` in the lab CLI.
- `tests/unit/optics/test_optical_pipeline_lab.py`
  - Adds import-safe RGB pack dependency probe.
  - Adds runner validation for RGB8 + torch_async delivery.
  - Adds CPU-only follow-up coverage for async ring depth validation,
    copy event elapsed-time order, and RGB pack ImportError behavior.
- `GPU_OPTICAL_PIPELINE_DESIGN.md`
  - Records D2 RGB8 implementation status and GPU1 measurements.
- `MANIFEST.md`
  - Adds `rgb_pack.py` and updates Q54 collected test count to 161.

## GPU1 Measurement

Command:

```bash
conda run -n env_tilelang_20260119 python -m tools.optical_pipeline_lab run \
  --preset go2_video_ordered_static \
  --out out/optical_pipeline_lab/go2_1080p_shadow_rgb8_torch_async_ring2_gpu1_syncpack \
  --device cuda:1 \
  --width 1920 \
  --height 1080 \
  --frames 10 \
  --warmup-renders 5 \
  --progress-every 1 \
  --readback rgb8 \
  --video-readback-delivery torch_async \
  --video-readback-ring-depth 2
```

Result summary:

```text
all10:
  frame_mean       ~= 6.71 ms
  render+pack_mean ~= 6.04 ms
  readback_copy    ~= 2.33 ms
  readback_submit  ~= 0.11 ms
  readback_wait    ~= 0.27 ms
  overlap_ratio    ~= 0.20

steady frames 1-8:
  frame_mean       ~= 6.44 ms
  render+pack_mean ~= 6.13 ms
  readback_copy    ~= 2.33 ms
  readback_wait    ~= 0.08 ms
  overlap_ratio    ~= 0.24
```

Comparison against previous GPU1 baselines:

- vs float32 RGB async ring2 all10:
  - frame mean: `10.78 ms -> 6.71 ms` (`~1.6x` faster)
  - copy time: `10.21 ms -> 2.33 ms` (`~4.4x` lower)
- vs sync RGB warmup=5 all10:
  - frame mean: `24.42 ms -> 6.71 ms` (`~3.6x` faster)

CSV:

```text
out/optical_pipeline_lab/go2_1080p_shadow_rgb8_torch_async_ring2_gpu1_syncpack/frame_timing.csv
```

The CSV rows report:

```text
readback_mode=torch_async_rgb8
readback_payload=rgb8
delivery_policy=torch_async
```

## Validation

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
GPU1 RGB8 async smoke/run: passed
```

## Review Notes

- The pack helper preserves `OpticalComputeResult` immutability by returning a
  new result instead of mutating the frozen object.
- Pack completion is synchronized before sync/async readback submission. This
  makes `render_execute_ms` include the measured pack completion time and avoids
  depending on implicit cross-framework stream ordering.
- `TorchAsyncReadbackRing.submit()` now documents that callers own the
  `result.ready_event` wait/synchronization contract before submitting to the
  Torch copy stream.
- RGB8 currently uses the same preview transfer function as the existing CPU
  preview path. It is a display/delivery payload optimization, not a change to
  optical compute precision.
- A dedicated CUDA RGB8 pack can still be evaluated later, but current measured
  frame time is now dominated by render plus ordered scheduling rather than
  host RGB materialization.
