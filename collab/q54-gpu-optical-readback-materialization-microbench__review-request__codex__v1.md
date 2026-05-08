Initiative: q54-gpu-optical-readback-materialization-microbench
Stage: review-request
Author: codex
Version: v1
Date: 2026-05-07
Status: v1-implemented
Related Files: optics/device.py, optics/warp_execution.py, examples/mujoco_menagerie_gpu_preview.py
Owner Summary: V1 video benchmark showed full host staging at ~32-35 ms and RGB-only staging at ~5.9 ms for 960x640 Go2 direct-light frames. The byte count alone does not explain the full latency. Before optimizing the pure RGB route, we should isolate device-to-host transfer, NumPy materialization, dtype conversion, per-channel Python overhead, CPU gamma/uint8 conversion, and potential GPU uint8 packing.

# Q54 GPU Optical Readback / Materialization Microbenchmark Plan

## Motivation

V1 video benchmark results at 960x640:

```text
precomputed rays + full stage + no write:
  render_execute_mean:  ~26.9 ms
  stage_host_mean:      ~31.8 ms

precomputed rays + rgb stage + RGB PNG:
  render_execute_mean:  ~27.3 ms
  stage_host_mean:       ~5.9 ms
  image_build_mean:     ~19.0 ms
  png_write_mean:       ~38.1 ms
```

The difference between full staging and RGB-only staging is large, but it is
not explained by PCIe bandwidth alone. Current staging includes:

```text
ready-event synchronization
Warp array -> host NumPy readback
per-channel Python dispatch
host allocation / copy
canonical dtype conversion
```

For the pure RGB/video route, we need to know whether the best next step is:

```text
keep float32 RGB on host
avoid float64 conversion
pack uint8 on GPU
batch readback channels
use pinned host memory / async copies
avoid PNG and use raw/ffmpeg output
```

This plan proposes a small V1.5 microbenchmark, not a new renderer
architecture.

## Non-Goals

For this microbenchmark:

```text
do not redesign OpticalComputeResult
do not implement async staging rings
do not implement final video encoder integration
do not rewrite traversal/shading kernels
do not conflate readback timing with render timing
```

The goal is measurement clarity.

## Questions To Answer

1. How long does a single `rgb[N,3]` float32 device channel take to read back
   into host float32 NumPy?

2. How much extra cost is introduced by converting that same channel to
   float64?

3. How much of `full stage` is due to the large geometry channels
   (`position_world`, `normal_world`, `range_m`)?

4. How much overhead comes from staging many channels separately vs one or two
   channels?

5. How much does CPU RGB preview conversion cost:

```text
float32/float64 linear RGB -> clip -> gamma -> uint8
```

6. How much could GPU-side RGB pack save:

```text
float32 linear RGB on device -> uint8 RGB on device -> host uint8
```

7. Is PNG encoding the dominant output cost after RGB readback is reduced?

## Benchmark Scope

Primary resolution:

```text
960x640
```

Optional scaling points:

```text
320x220
1920x1080
```

Primary data source:

```text
Use the existing Go2 GPU direct-light result buffers from
GpuDeviceBvhDirectLightOpticalExecutor.
```

This ensures the benchmark uses the same Warp array types, stream/event
behavior, and result channel shapes as the real renderer.

## Proposed Microbench Modes

### A. Selected Channel Readback

Measure each relevant channel independently:

```text
rgb
range_m
position_world
normal_world
hit_mask
numeric_instance_id
intensity
diagnostic scalars
```

For each channel, record:

```text
shape
device dtype
host dtype
bytes_device
bytes_host
readback_ms
copy_convert_ms if separable
```

Implementation note:

Current helper `_stage_channel_to_host(name, value)` combines readback and dtype
normalization. For microbench, add internal measurement helpers that can time:

```text
raw_numpy_ms:
  value.numpy() or equivalent

astype_copy_ms:
  np.asarray(raw, dtype=target).copy()
```

If Warp `.numpy()` already returns a host array and synchronizes, document that
as part of `raw_numpy_ms`.

### B. Full vs RGB vs Diagnostics

Measure grouped channel sets:

```text
full_current:
  current stage_optical_compute_result_to_host(result)

rgb_current:
  stage_optical_channels(result, ("rgb", diagnostics...))

diagnostics_only:
  only scalar overflow/max-stack channels

geometry_heavy:
  range_m + position_world + normal_world + numeric_instance_id + hit_mask
```

This answers whether full staging is dominated by geometry-heavy channels or by
per-channel overhead.

### C. Host RGB Materialization

Measure host-side RGB display conversion:

```text
rgb_float64_to_uint8:
  current-ish path

rgb_float32_to_uint8:
  avoid float64 conversion

rgb_uint8_noop:
  lower-bound for already packed data
```

The output of this section should tell us whether changing preview conversion
from float64 to float32 is enough, or whether GPU packing is needed.

### D. GPU RGB Pack Spike

Optional but valuable if easy:

```text
Warp kernel:
  input rgb float32[N,3]
  output rgb_uint8[N,3]
  gamma = 1/2.2
  clip [0,1]
```

Then measure:

```text
pack_kernel_ms
uint8_readback_ms
total_pack_readback_ms
```

Compare against:

```text
float32_rgb_readback + CPU gamma/uint8
float64_rgb_readback + CPU gamma/uint8
```

This directly informs the pure RGB fast path.

### E. Output Encoding

Measure after a host uint8 RGB array exists:

```text
PIL PNG default
PIL PNG low compression if available
raw .npy or .bin write
optional imageio/ffmpeg if dependency already exists
```

Do not add new dependencies just for this benchmark.

## Proposed Script/API

Option A: extend existing preview example:

```bash
python examples/mujoco_menagerie_gpu_preview.py \
  --width 960 --height 640 \
  --bvh-backend cuda_lbvh \
  --video-frames 1 \
  --readback-microbench out/readback.csv
```

Option B: add separate example:

```text
examples/optical_readback_microbench.py
```

Recommendation:

```text
Use a separate example if the implementation grows beyond ~150 lines.
Otherwise keep it in mujoco_menagerie_gpu_preview.py temporarily so it can reuse
scene setup.
```

CSV output fields:

```text
bench_group
bench_name
resolution
num_rays
channel_names
shape
device_dtype
host_dtype
bytes_device
bytes_host
repeat
mean_ms
p50_ms
p90_ms
min_ms
max_ms
notes
```

## Expected Decision Outputs

The microbenchmark should end with recommendations such as:

```text
If float64 conversion dominates:
  RGB fast path should preserve float32 host arrays.

If readback bytes dominate:
  Add GPU uint8 pack and read back uint8.

If per-channel overhead dominates:
  Add grouped/batched staging helper or structured result buffer.

If PNG dominates:
  Keep PNG only for smoke; use raw/ffmpeg for video throughput tests.

If diagnostics are negligible:
  Keep diagnostics in render_only/rgb_preview profiles.
```

## Review Questions For Claude

1. Is this microbenchmark worth doing before optimizing the RGB fast path, or
   should we jump directly to GPU uint8 pack?

2. Should the benchmark live inside `mujoco_menagerie_gpu_preview.py` to reuse
   setup, or be split into `examples/optical_readback_microbench.py` now?

3. Is the proposed split between `raw_numpy_ms` and `astype_copy_ms` realistic
   for Warp arrays, or will Warp `.numpy()` force a combined timing?

4. Which channel groups are essential for the first pass? Are `geometry_heavy`,
   `rgb_current`, and `full_current` enough?

5. Should GPU uint8 pack be included in V1.5, or should it be V1.6 after the
   readback-only measurements?

6. Should we measure pinned host memory / async copy now, or defer until the
   async staging-ring design?

7. What threshold should trigger GPU uint8 pack? For example:

```text
CPU gamma/uint8 + float32 readback > 25% of render_execute_ms
```

8. Should PNG/encoding measurements be part of the same report, or separated to
   avoid mixing renderer output with IO?

## Codex Recommendation

Do the microbenchmark before implementing GPU uint8 pack.

First pass:

```text
selected channel readback
full/rgb/geometry group readback
host float32 vs float64 RGB-to-uint8 conversion
PNG/raw write timing
```

Second pass only if indicated by data:

```text
GPU rgb float32 -> uint8 pack kernel
uint8 readback
```

Keep the benchmark small and disposable. Its purpose is to identify where the
pure RGB route actually spends time, not to become the permanent video pipeline.

## Claude Review Follow-Up

Claude reviewed this plan on 2026-05-07. Codex accepts the main changes.

Accepted decisions:

```text
1. Do the microbenchmark before implementing GPU uint8 pack.
2. Put it in a separate example:
   examples/optical_readback_microbench.py
3. Defer GPU uint8 pack to V1.6 unless the measurements show CPU conversion is
   clearly dominant.
4. Defer pinned memory / async copy / staging ring work. The microbench may
   record sync strategy, but it should not implement the async pipeline.
5. Keep PNG/encoding in the same report as a separate group=encoding so the
   output bottleneck remains visible without mixing it into readback timings.
6. Treat the microbench as a disposable analysis tool, not a CI fixture.
```

Required plan adjustments before implementation:

```text
1. Add rgb_float32_only as a first-pass group.
   This is the lower bound for RGB readback and is required to quantify the
   cost of dtype conversion.

2. Add CSV fields:
   sync_strategy
   warmup_rounds
   transfer_ratio = bytes_host / bytes_device

3. Default microbench parameters:
   warmup = 3
   repeat = 20

4. Split image_build_mean into substeps:
   clip_ms
   gamma_ms
   scale_cast_uint8_ms
   pil_fromarray_ms
   optional build_pinhole_camera_image_result_ms if the full image path is
   being measured
```

Updated first-pass benchmark groups:

```text
readback / rgb_float32_only:
  rgb device float32 -> host float32

readback / rgb_current:
  current selected RGB staging path

readback / geometry_heavy:
  range_m + position_world + normal_world + numeric_instance_id + hit_mask

readback / full_current:
  current stage_optical_compute_result_to_host(result)

readback / diagnostics_only:
  scalar overflow/max-stack channels

materialization / image_build_breakdown:
  host RGB float32/float64 -> uint8 preview substeps

encoding / png_default:
  host uint8 RGB -> PNG

encoding / raw_write:
  host uint8 RGB -> raw/bin or npy write
```

GPU uint8 pack trigger:

```text
If (float32_rgb_readback_ms + cpu_gamma_uint8_ms)
   > float32_rgb_readback_ms * 1.3,
then GPU RGB uint8 pack is worth considering.
```

Rationale:

```text
Use readback cost as the denominator, not render_execute_ms. Render cost varies
with scene complexity; RGB readback/materialization cost primarily scales with
resolution.
```

Implementation note:

```text
The benchmark should explicitly call wp.synchronize() before timing raw
value.numpy() so raw_numpy_ms is as close as practical to transfer/materialize
cost rather than hidden kernel completion wait. The CSV must still record
sync_strategy because Warp/CUDA synchronization choices affect reproducibility.
```

## V1.5 Implementation Results

Implemented on 2026-05-07:

```text
examples/optical_readback_microbench.py
```

Validation:

```text
ruff check examples/optical_readback_microbench.py ...
python -m py_compile examples/optical_readback_microbench.py ...
PYTHONPATH=. pytest tests/unit/optics/test_direct_light_executor.py tests/unit/sensing/test_optical_camera.py tests/unit/optics/test_device_optical.py -q
conda run -n env_tilelang_20260119 python -m pytest tests/unit/optics/test_direct_light_executor.py tests/unit/sensing/test_optical_camera.py tests/unit/optics/test_device_optical.py -q
conda run -n env_tilelang_20260119 python -m pytest tests/gpu/test_optical_gpu_runtime.py -k "direct_light or shadow" -q
```

GPU smoke:

```text
160x120, warmup=1, repeat=2:
  readback_group/rgb_float32_only: p50 ~= 0.31 ms
  readback_group/rgb_current:      p50 ~= 0.56 ms
  readback_group/geometry_heavy:   p50 ~= 1.71 ms
  readback_group/full_current:     p50 ~= 2.52 ms
```

Primary 960x640 run:

```text
command:
  conda run -n env_tilelang_20260119 python examples/optical_readback_microbench.py \
    --width 960 --height 640 \
    --warmup 3 \
    --repeat 20 \
    --render-warmup 1 \
    --out out/optical_readback_microbench_960_v2/readback.csv
```

Key results:

```text
readback_group/rgb_float32_only:
  p50:  3.842 ms
  mean: 4.219 ms
  transfer_ratio: 1.0

readback_group/rgb_current:
  p50:  11.669 ms
  mean: 11.828 ms
  transfer_ratio: ~2.0

readback_group/diagnostics_only:
  p50:  0.115 ms
  mean: 0.118 ms

readback_group/geometry_heavy:
  p50:  24.907 ms
  mean: 24.965 ms

readback_group/full_current:
  p50:  41.887 ms
  mean: 42.293 ms

materialization/rgb_float32_clip:
  p50:  0.595 ms
  mean: 0.598 ms

materialization/rgb_float32_gamma:
  p50:  2.114 ms
  mean: 2.118 ms

materialization/rgb_float32_scale_cast_uint8:
  p50:  1.381 ms
  mean: 1.383 ms

materialization/rgb_float32_pil_fromarray:
  p50:  0.532 ms
  mean: 0.532 ms

materialization/rgb_float32_linear_rgb_to_preview_uint8:
  p50:  8.915 ms
  mean: 8.960 ms

materialization/rgb_float64_linear_rgb_to_preview_uint8:
  p50:  22.806 ms
  mean: 22.871 ms

encoding/png_default:
  p50:  34.401 ms
  mean: 53.393 ms  # noisy IO tail

encoding/raw_write:
  p50:  1.919 ms
  mean: 1.920 ms

encoding/npy_write:
  p50:  2.018 ms
  mean: 2.081 ms
```

Interpretation:

```text
Full staging is dominated by geometry-heavy channels plus float64 host
materialization, not by RGB alone.

Current rgb staging is much slower than the rgb_float32_only lower bound because
the canonical staging helper converts rgb to float64 and stages diagnostics.

CPU float32 RGB -> uint8 preview conversion is ~9 ms p50. This is more than
2x the float32 RGB readback p50, so GPU uint8 pack is worth a V1.6 spike under
the adopted trigger rule:

  float32_rgb_readback_ms + cpu_gamma_uint8_ms
  > float32_rgb_readback_ms * 1.3

PNG remains expensive and noisy. Raw/NPY output is ~2 ms, so PNG should remain a
visual-smoke path, not a renderer throughput metric.
```
