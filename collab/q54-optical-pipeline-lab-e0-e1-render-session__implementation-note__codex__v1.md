# Q54 Optical Pipeline Lab — E0.1/E1.1/E1.2 Implementation Note

Date: 2026-05-10
Author: Codex

## Scope

Implemented the first E0/E1 slices after Claude review:

- keep `Go2RenderSession` inside `tools/optical_pipeline_lab/go2_backend.py`
- keep async readback ring ownership outside the session
- split RGB8 pack timing out of `render_execute_ms`
- add `pack_rgb8_ms` to the frame CSV schema and summaries
- fill `render_overhead_ms` when `--render-profile` is enabled

## Changes

- `tools/optical_pipeline_lab/go2_backend.py`
  - Added `Go2RenderSession` to own Go2 render resources:
    scene, device, stream, static GPU frame, scene cache, snapshot, BVH, and
    executor.
  - Added `_RenderedVideoFrame`.
  - Added `_render_video_frame()` shared by sync and `torch_async` video loops.
  - Changed RGB8 pack timing from being folded into `render_execute_ms` to a
    separate `pack_rgb8_ms` value.
  - Kept overlap accounting on the async path based on render + pack + copy so
    D1/D2 overlap semantics remain comparable.
- `tools/optical_pipeline_lab/timing.py`
  - Added `pack_rgb8_ms` to `FRAME_TIMING_FIELDNAMES`.
  - Added `pack_rgb8` to `FRAME_TIMING_SUMMARY_PHASES`.
- `tests/unit/optics/test_optical_pipeline_lab.py`
  - Verifies `pack_rgb8_ms` appears in the CSV schema and summary rows.
  - Verifies `render_overhead_ms` is computed from known render profile phases
    and remains unclamped when timing noise makes it slightly negative.
- `GPU_OPTICAL_PIPELINE_DESIGN.md`
  - Documents the timing semantic change:
    `render_execute_ms` is now executor render only, and RGB8 pack is reported
    separately as `pack_rgb8_ms`.

## Timing Semantics

Before E1.1:

```text
render_execute_ms = executor render + optional RGB8 pack
```

After E1.1:

```text
render_execute_ms = executor render only
pack_rgb8_ms      = RGB8 pack launch + completion wait, or NaN for non-RGB8
```

When comparing against D2 baseline rows, use:

```text
render_delivery_ms = render_execute_ms + pack_rgb8_ms
```

for RGB8 cases.

When `--render-profile` is enabled:

```text
render_overhead_ms = render_execute_ms - sum(known render profile phases)
```

Unknown profile phase names are ignored. The value is intentionally not clamped;
small negative values are useful timing diagnostics.

## Validation

CPU-side validation:

```text
conda run -n env_tilelang_20260119 ruff check tools/optical_pipeline_lab examples/mujoco_menagerie_gpu_preview.py tests/unit/optics/test_optical_pipeline_lab.py
conda run -n env_tilelang_20260119 ruff format --check tools/optical_pipeline_lab examples/mujoco_menagerie_gpu_preview.py tests/unit/optics/test_optical_pipeline_lab.py
conda run -n env_tilelang_20260119 python -m py_compile tools/optical_pipeline_lab/*.py tests/unit/optics/test_optical_pipeline_lab.py examples/mujoco_menagerie_gpu_preview.py
conda run -n env_tilelang_20260119 python -m pytest tests/unit/optics/test_optical_pipeline_lab.py -q
```

Results:

```text
ruff: all checks passed
ruff format --check: 13 files already formatted
py_compile: passed
lab unit tests: 31 passed
```

## GPU Smoke

GPU1 was occupied by another workload during validation, so the comparable
behavioral smoke was run on the idle GPU3:

```bash
conda run -n env_tilelang_20260119 python -m tools.optical_pipeline_lab run \
  --preset go2_video_ordered_static \
  --out out/optical_pipeline_lab/go2_1080p_shadow_rgb8_torch_async_ring2_e0e1_gpu3 \
  --device cuda:3 \
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
  frame_mean       ~= 6.42 ms
  render_mean      ~= 5.66 ms
  pack_rgb8_mean   ~= 0.08 ms
  readback_copy    ~= 2.32 ms
  overlap_ratio    ~= 0.20

steady frames 1-8:
  frame_mean       ~= 6.14 ms
  render_mean      ~= 5.75 ms
  pack_rgb8_mean   ~= 0.07 ms
  readback_copy    ~= 2.33 ms
  overlap_ratio    ~= 0.25
```

CSV:

```text
out/optical_pipeline_lab/go2_1080p_shadow_rgb8_torch_async_ring2_e0e1_gpu3/frame_timing.csv
```

The CSV header includes:

```text
render_execute_ms
pack_rgb8_ms
readback_host_ms
frame_total_ms
```

The split confirms the D2 `render+pack` budget was almost entirely executor
render on an uncontended GPU; RGB8 pack itself is currently around 0.07-0.08 ms.

## Render Profile Smoke

The same case was run with `--render-profile`:

```bash
conda run -n env_tilelang_20260119 python -m tools.optical_pipeline_lab run \
  --preset go2_video_ordered_static \
  --out out/optical_pipeline_lab/go2_1080p_shadow_rgb8_torch_async_ring2_e1_profile_gpu3 \
  --device cuda:3 \
  --width 1920 \
  --height 1080 \
  --frames 10 \
  --warmup-renders 5 \
  --progress-every 1 \
  --readback rgb8 \
  --video-readback-delivery torch_async \
  --video-readback-ring-depth 2 \
  --render-profile
```

Result summary:

```text
all10:
  frame_mean          ~= 6.22 ms
  render_mean         ~= 5.70 ms
  render_overhead     ~= -0.15 ms
  pack_rgb8_mean      ~= 0.12 ms
  raygen_kernel       ~= 0.18 ms
  first_hit_kernel    ~= 1.97 ms
  shade_kernel        ~= 3.16 ms
  readback_copy       ~= 0.15 ms

steady frames 1-8:
  frame_mean          ~= 6.23 ms
  render_mean         ~= 5.73 ms
  render_overhead     ~= -0.15 ms
  pack_rgb8_mean      ~= 0.11 ms
  raygen_kernel       ~= 0.17 ms
  first_hit_kernel    ~= 1.96 ms
  shade_kernel        ~= 3.20 ms
  readback_copy       ~= 0.15 ms
```

CSV:

```text
out/optical_pipeline_lab/go2_1080p_shadow_rgb8_torch_async_ring2_e1_profile_gpu3/frame_timing.csv
```

The first profile split points at shade as the largest render kernel cost,
followed by first-hit traversal. The small negative overhead is expected timing
noise from mixing host-side render wall time with synchronized profile
subphase measurements.
