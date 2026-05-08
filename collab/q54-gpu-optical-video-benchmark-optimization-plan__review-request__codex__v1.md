Initiative: q54-gpu-optical-video-benchmark-optimization
Stage: review-request
Author: codex
Version: v1
Date: 2026-05-06
Status: v1-implemented-gpu-raygen-spike
Related Files: examples/mujoco_menagerie_gpu_preview.py, optics/cuda_lbvh.py, optics/warp_execution.py, sensing/optical.py, OPEN_QUESTIONS.md
Owner Summary: After CUDA LBVH build became ms-scale inside a warmed session, the next bottleneck for 960x640 video preview is no longer tree build. The first video benchmark shows large per-frame costs in CPU camera ray generation, broad host readback, and image/PNG output. V1 ray caching made the benchmark more truthful, but the correct production direction is GPU pinhole ray generation behind a camera-specific executor path, while the existing ray-batch executor remains the generic query/parity reference path.

# Q54 GPU Optical Video Benchmark Optimization Plan

## Current Measurement

Scene:

```text
asset source: MuJoCo Menagerie Unitree Go2 MJCF
visual geoms: 33
triangles: 398,432
resolution: 960x640
mode: camera_orbit
lighting: direct-light + hard shadows
BVH backend: cuda_lbvh initial build, Warp traversal/shading
geometry: static scene/BVH/snapshot reused for all video frames
```

Current no-write video run:

```text
command:
  TORCH_CUDA_ARCH_LIST=9.0 \
  python examples/mujoco_menagerie_gpu_preview.py \
    --width 960 --height 640 \
    --out out/menagerie_go2_gpu_video_960_nowrite_warmcache \
    --bvh-backend cuda_lbvh \
    --warmup-renders 2 \
    --video-frames 10 \
    --video-fps 30 \
    --video-mode camera_orbit \
    --progress-every 5 \
    --no-write-frames \
    --fail-on-overflow
```

Observed:

```text
fps_mean:                 9.57
frame_p50_ms:             102.99
frame_p90_ms:             113.20

camera_rays_mean:          48.02 ms
render_execute_mean:       26.94 ms
readback_host_mean:        29.51 ms
image_build_mean:           0.00 ms
encode_or_write_mean:       0.00 ms
```

Current write-RGB-PNG video run:

```text
command:
  TORCH_CUDA_ARCH_LIST=9.0 \
  python examples/mujoco_menagerie_gpu_preview.py \
    --width 960 --height 640 \
    --out out/menagerie_go2_gpu_video_960_write \
    --bvh-backend cuda_lbvh \
    --warmup-renders 2 \
    --video-frames 10 \
    --video-fps 30 \
    --video-mode camera_orbit \
    --progress-every 5 \
    --write-frames \
    --fail-on-overflow
```

Observed:

```text
fps_mean:                 5.70
frame_p50_ms:             176.45
frame_p90_ms:             191.78

camera_rays_mean:          49.27 ms
render_execute_mean:       27.59 ms
readback_host_mean:        32.11 ms
image_build_mean:          44.22 ms
encode_or_write_mean:      22.13 ms
```

Interpretation:

```text
render_execute is not the only bottleneck.
camera ray generation + broad host readback dominate the no-write frame.
image build + PNG write dominate the write-frame increment.
```

The current benchmark is useful, but it still mixes several concerns that must
be separated before deeper kernel work:

```text
ray generation
render kernel
host readback
image materialization
encoding / disk IO
```

## Constraints And Non-Goals

For this stage:

```text
keep changes in examples / benchmark harness where possible
do not introduce the formal GpuOpticalRenderSession yet
do not implement async double-buffer build/render yet
do not switch traversal to OptiX/CUDA yet
do not treat static camera-orbit benchmark as changing-geometry evidence
```

The goal is a clearer benchmark surface and low-risk wins, not a final renderer
architecture.

## Optimization Area 1: Camera Rays

Problem:

```text
camera_rays_mean ~= 48-50 ms at 960x640
```

This is expensive enough to hide the true renderer cost.

Why it is expensive today:

```text
build_pinhole_camera_rays(...) creates float64 uu/vv meshgrids
it stacks camera-space directions, rotates them on CPU, and repeats origins
OpticalRaySensorSpec validates, normalizes, and copies the full ray batch
the GPU executor then converts origins/directions to float32 and uploads them
```

Correct architecture:

```text
Do not generate pinhole-camera rays on CPU for the production GPU renderer.

The GPU camera path should pass compact camera parameters:
  width, height, fx, fy, cx, cy
  X_world_camera
  max_distance
  sensor_role / role mask

Then each ray-major GPU thread computes its own pixel ray from thread id:
  x = tid % width
  y = tid / width
  dir_camera = normalize(((x - cx) / fx, (y - cy) / fy, 1))
  dir_world = R_world_camera @ dir_camera
  origin_world = camera_position
```

Compatibility rule:

```text
Keep OpticalRaySensorSpec and execute(..., rays) for LiDAR, arbitrary ray
queries, CPU reference, and parity tests.

Add a camera-specific GPU entry point instead of forcing every camera render
through a materialized ray batch.
```

Recommended bridge step:

```text
Add video ray cache in the example harness.
```

The cache is a benchmark bridge, not the final production shape. It removes
per-frame CPU ray-generation noise so render/readback/encoding measurements can
be trusted before adding the GPU camera path.

Modes:

```text
fixed_view:
  build rays once and reuse for every frame

camera_orbit:
  precompute rays for all requested video frames before the timed frame loop
  record precompute time separately as setup/warmup, not per-frame render cost

pose_sequence:
  reserved for later; ray cache can still work if camera path is known ahead
```

Benchmark switches:

```text
--video-ray-cache off|precompute
```

Expected effect:

```text
fixed_view no-write frame_total should drop by roughly camera_rays_ms.
camera_orbit steady-state per-frame timing should reveal render/readback cost more directly.
```

Open question:

```text
Should ray cache be default for video benchmark, or should the default remain
uncached to expose current end-to-end harness cost?
```

Recommendation:

```text
Default to uncached for now, but make cached mode explicit and report both in
benchmark notes.
```

Next production step:

```text
Add GpuPinholeCameraSpec / DeviceCameraParams and a GPU camera execution path.
The first version may internally call the existing BVH traversal logic after
generating rays on device, but it should not allocate/upload host ray buffers.
The preferred later version fuses raygen + first-hit + direct-light shading for
rgb_preview/render_only profiles.
```

## Optimization Area 2: Host Readback

Problem:

```text
readback_host_mean ~= 29-32 ms at 960x640
```

Current video benchmark calls:

```text
stage_optical_compute_result_to_host(result)
```

This reads back the full result, including channels that a render-only benchmark or
RGB video does not need.

Recommended first steps:

```text
1. Add render-only benchmark mode:
   wait for result.ready_event, collect only GPU-side timing and skip broad host
   readback.

2. Add rgb-only readback mode:
   read back only rgb and scalar diagnostics needed for overflow reporting.

3. Keep full readback mode for parity/debug.
```

Potential CLI:

```text
--video-readback full|rgb|diagnostics|none
```

Semantics:

```text
full:
  current behavior; read back all channels

rgb:
  read back only rgb plus small diagnostics

diagnostics:
  read back only scalar overflow/max-stack diagnostics if possible

none:
  do not read back host data; only wait result.ready_event.
  overflow reporting is skipped or reported as unavailable.
```

Open question:

```text
Can we read back individual Warp arrays cleanly with existing helper APIs, or should
the first version add a new small helper in optics for selected channels?
```

Recommendation:

```text
Add a small selected-channel readback helper rather than duplicating readback
logic in the example.
```

## Optimization Area 3: Image Build / Encoding

Problem when writing RGB PNG frames:

```text
image_build_mean:         ~= 44 ms
encode_or_write_mean:     ~= 22 ms
```

The current write path builds a full `OpticalCameraImageResult` even when only
RGB PNG output is requested.

Recommended first step:

```text
Add an rgb-only frame writer path.
```

For video frames:

```text
result rgb channel -> host rgb ndarray -> gamma preview -> PNG/optional raw
```

Avoid:

```text
build_pinhole_camera_image_result(...)
full depth/segmentation materialization
panel image creation
```

Potential CLI:

```text
--write-frame-kind rgb|none
--frame-format png|npy
```

Open question:

```text
Should video benchmark write PNGs at all, or should it prefer raw `.npy`/`.npz`
or an ffmpeg pipe later?
```

Recommendation:

```text
Keep PNG for visual smoke, but make no-write the default and add rgb-only write
for controlled output-cost measurement.
```

## Optimization Area 4: Render Kernel

Current render kernel cost:

```text
render_execute_mean ~= 27 ms at 960x640 with direct light + shadows
```

This is meaningful, but not yet isolated from ray generation and readback.

Potential future optimizations:

```text
Warp traversal stack/register pressure review
shadow any-hit overhead profiling
ray-major divergence analysis
BVH quality / node ordering metrics
GPU pinhole raygen, then fused raygen + trace + shade
OptiX backend for high-performance comparison
```

Recommendation:

```text
Do not start here until ray-cache and selected readback modes make render-only
benchmark numbers trustworthy.
```

## Proposed Implementation Order

### V1: Benchmark Truthfulness

Files:

```text
examples/mujoco_menagerie_gpu_preview.py
possibly optics readback helper
```

Tasks:

```text
1. Add --video-ray-cache off|precompute.
2. Add --video-readback full|rgb|none.
3. Add selected-channel readback helper if needed.
4. Add RGB-only write path; avoid full image result for video frames.
5. Keep frame_timing.csv schema stable; fill skipped phases as 0 or "unavailable".
```

Expected benchmark matrix:

```text
baseline:
  uncached rays + full readback + no write

ray-cache:
  precomputed rays + full readback + no write

render-only:
  precomputed rays + no readback + no write

rgb-output:
  precomputed rays + rgb readback + rgb PNG write
```

### V2: Changing Geometry Benchmark

Tasks:

```text
1. Add pose_sequence mode with frame-aligned snapshot updates.
2. Measure snapshot_ms + refit_ms for refit-capable BVH.
3. For CUDA LBVH rebuild mode, measure rebuild_ms separately.
4. Compare refit vs rebuild vs static topology quality where applicable.
```

This is the first stage where double-buffer build/render overlap becomes a
data-driven decision.

### V3: Kernel/Backend Work

Only after V1/V2:

```text
evaluate Warp traversal bottlenecks
add GPU pinhole camera raygen behind a camera-specific executor API
consider fused raygen/traversal/shading for rgb_preview/render_only
consider OptiX comparison path
consider async BVH double-buffer publish
```

### V1.6: GPU Camera Raygen Spike

Goal:

```text
Eliminate per-frame CPU pinhole ray materialization for GPU camera preview.
```

Tasks:

```text
1. Define a compact camera parameter object for GPU execution.
2. Add an executor entry point such as execute_camera(snapshot, bvh, camera_spec,
   output_profile=...).
3. Generate origin/direction from pixel id on device.
4. Preserve execute(snapshot, bvh, rays) for generic ray batches and parity.
5. Add parity tests: CPU build_pinhole_camera_rays + existing execute path vs
   GPU camera raygen path for a small deterministic camera/scene.
6. Benchmark 960x640:
   - cached CPU rays + existing execute
   - GPU camera raygen + existing traversal/shading shape
   - later fused rgb_preview path
```

Non-goals:

```text
Do not remove OpticalRaySensorSpec.
Do not make LiDAR/arbitrary ray queries depend on camera-specific code.
Do not conflate GPU raygen speedup with readback/PNG improvements.
```

## V1.6 Implementation Update: GPU Camera Raygen Spike

Date: 2026-05-08

Implemented:

```text
GpuDeviceBvhOpticalExecutor.execute_camera(snapshot, bvh, camera)
GpuDeviceBvhDirectLightOpticalExecutor.execute_camera(snapshot, bvh, camera, output_profile=...)
_pinhole_camera_raygen_kernel
examples/mujoco_menagerie_gpu_preview.py --video-raygen host|gpu
frame_timing.csv raygen_mode field
```

Current design:

```text
execute_camera(...) passes compact pinhole camera parameters to the GPU.
The GPU raygen kernel fills device origins/directions from pixel id.
Existing BVH traversal and direct-light kernels are reused after that.
The old execute(..., rays) path remains the generic ray-batch and parity path.
```

This is not the final fused renderer yet. It still materializes device ray
arrays and still uses the existing first-hit + shade kernel split. However, it
removes CPU `meshgrid` / `OpticalRaySensorSpec` normalization / host-to-device
ray upload from the video frame loop.

Validation:

```text
ruff check optics/warp_execution.py examples/mujoco_menagerie_gpu_preview.py \
  tests/gpu/test_optical_gpu_runtime.py

python -m py_compile optics/warp_execution.py examples/mujoco_menagerie_gpu_preview.py \
  tests/gpu/test_optical_gpu_runtime.py

conda run -n env_tilelang_20260119 python -m pytest \
  tests/gpu/test_optical_gpu_runtime.py -k "camera_raygen" -q
  -> 2 passed

conda run -n env_tilelang_20260119 python -m pytest \
  tests/gpu/test_optical_gpu_runtime.py \
  -k "device_bvh and (direct_light or camera_raygen or preserves_plane_triangle_tie_break)" -q
  -> 7 passed
```

Smoke benchmark:

```text
160x120, Go2, cuda_lbvh, direct-light + shadows, video-raygen=gpu, readback=rgb:
  video_render_execute_mean ~= 1.48 ms
  video_readback_host_mean  ~= 0.36 ms
  video_frame_total_mean    ~= 1.98 ms
```

960x640 benchmark:

```text
video-raygen=gpu, readback=none, no write:
  video_render_execute_mean ~= 16.61 ms
  video_frame_total_mean    ~= 16.76 ms
  fps_mean                  ~= 59.68
  csv:
    out/menagerie_go2_gpu_raygen_960_renderonly/frame_timing.csv

video-raygen=gpu, readback=rgb, no write:
  video_render_execute_mean ~= 18.79 ms
  video_readback_host_mean  ~= 16.09 ms
  video_frame_total_mean    ~= 35.03 ms
  fps_mean                  ~= 28.55
  csv:
    out/menagerie_go2_gpu_raygen_960_rgb_nowrite/frame_timing.csv
```

Interpretation:

```text
GPU raygen removes the previous ~45-50 ms CPU camera_rays phase entirely.
Compared with the previous precomputed-host-ray render-only baseline
(~29.19 ms frame/render), GPU raygen readback=none is now ~16.6 ms.
At 960x640, RGB readback/materialization remains a large next bottleneck.
The next likely step is GPU uint8 pack and/or a fused rgb_preview path.
```

## Naming And Readback Semantics Update

Date: 2026-05-08

The video benchmark API now uses `readback`, not `stage`, for device-to-host
result transfer:

```text
--video-readback full|rgb|none
frame_timing.csv: readback_mode, readback_host_ms
stdout: readback=...
```

Rationale:

```text
The current video benchmark readback path is blocking.
It does not hide latency behind later frames.
It waits for result.ready_event, then performs Warp .numpy() materialization
and host dtype copies/conversions.
```

Internal helpers such as `stage_optical_channels(...)` keep their names because
they describe the lower-level device-result staging contract. The external
benchmark surface and reports should use `readback` so we do not confuse this
blocking phase with a future latency-hiding pipeline stage.

RGB preview readback now requests `output_profile=rgb_preview` and preserves
device-native host dtypes for selected channels:

```text
rgb: float32 host array
diagnostics: int32 host arrays
```

Full readback still uses canonical host dtypes for debug/parity paths.

Validation:

```text
PYTHONPATH=. pytest tests/unit/optics/test_device_optical.py -q
  -> 8 passed

conda run -n env_tilelang_20260119 python -m pytest \
  tests/gpu/test_optical_gpu_runtime.py \
  -k "device_bvh and (direct_light or camera_raygen or preserves_plane_triangle_tie_break)" -q
  -> 7 passed
```

Updated 960x640 result:

```text
video-raygen=gpu, readback=rgb, no write:
  video_render_execute_mean   ~= 16.85 ms
  video_readback_host_mean    ~=  4.91 ms
  video_frame_total_mean      ~= 21.91 ms
  fps_mean                    ~= 45.64
  csv:
    out/menagerie_go2_gpu_readback_960_rgb_nowrite/frame_timing.csv
```

This replaces the earlier `readback=rgb` result where canonical float64 staging
made readback roughly 16 ms. The remaining readback is now close to the
`rgb_float32_only` lower bound measured by the microbench.

## Review Questions For Claude

1. Is the V1 focus on ray cache + selected readback the right next step, or
   should render-kernel profiling start in parallel?

2. Should `--video-ray-cache precompute` be the default for video benchmark, or
   should the default remain uncached to expose end-to-end camera path cost?

3. Should selected-channel readback be implemented as a reusable optics helper
   now, or kept local to the example until the session/runtime API is clearer?

4. Is `--video-readback full|rgb|none` the right surface, or do we need a more
   explicit channel list such as `--video-readback-channels rgb,bvh_stack...`?

5. For RGB frame output, is bypassing `build_pinhole_camera_image_result` safe
   enough for a benchmark example, or should all image materialization continue
   through the camera image helper for semantic consistency?

6. Is it acceptable that `camera_orbit` reuses the same optical
   `snapshot.frame_id/sim_time` while logging separate `video_time`, or should
   the benchmark construct a new frame-aligned snapshot for every video frame
   even when geometry is static?

7. Before implementing double-buffer build/render, what exact benchmark matrix
   should be required for a go/no-go decision?

## Current Position

The current data says:

```text
tree build inside warmed CUDA LBVH kernels is not the main per-frame video cost.
camera rays and readback must be addressed first.
PNG/image output must be separated from renderer performance.
double-buffer build/render remains plausible, but should wait for changing
geometry benchmark data.
```

## Claude Review Follow-Up

Claude reviewed this plan on 2026-05-06. Codex accepts the following changes.

### Accepted Decisions

```text
V1 scope:
  Do ray cache + selected readback + rgb-only write first.
  Do not start render-kernel profiling in parallel.

Ray cache default:
  Keep --video-ray-cache off as the default.
  Treat --video-ray-cache precompute as an explicit benchmark mode.
  Benchmark notes should report uncached and precomputed rows separately.

Selected readback:
  Put selected-channel readback in an optics helper, not duplicated in the
  example, because Warp ownership/stream synchronization details are easy to
  get wrong.

Video readback CLI:
  Use --video-readback full|rgb|none for V1.
  Do not expose a free-form channel list yet.

RGB-only write:
  It is acceptable for the benchmark example to bypass
  build_pinhole_camera_image_result when writing RGB frames, but the path must
  be clearly marked benchmark-only and use the same gamma/clip/channel ordering
  as the normal preview path.

Static camera_orbit timeline:
  Reusing the same snapshot.frame_id/sim_time is acceptable for static-geometry
  camera-orbit benchmarks. CSV/output must label geometry_mode=static so the
  result is not mistaken for dynamic scene data.

Skipped phases:
  Use NaN for not-applicable phases in frame_timing.csv, not 0. A zero means a
  measured phase took zero time; NaN means the phase was not part of that
  benchmark mode.
```

### V1 Implementation Plan After Review

```text
1. Update frame_timing.csv schema:
   add geometry_mode, ray_cache_mode, readback_mode, write_mode.
   emit NaN for snapshot/refit/rebuild/readback/image/write when not applicable.

2. Add --video-ray-cache off|precompute.
   off remains default.
   precompute builds cameras/rays before the timed frame loop and records setup
   time separately.

3. Add a narrow selected-channel readback helper in optics:
   stage_optical_channels(result, channels: Sequence[str]) -> dict[str, np.ndarray]

4. Add --video-readback full|rgb|none.
   full uses current stage_optical_compute_result_to_host.
   rgb reads back rgb plus minimal diagnostics needed for overflow if practical.
   none skips host readback and reports diagnostics as NaN/unavailable.

5. Add benchmark-only RGB write path:
   rgb host array -> existing linear_rgb_to_preview_uint8 -> PNG.
   Do not build full OpticalCameraImageResult for RGB-only video frames.

6. Re-run 960x640 matrix:
   uncached rays + full readback + no write
   precomputed rays + full readback + no write
   precomputed rays + no readback + no write
   precomputed rays + rgb readback + rgb write
```

### Double-Buffer Go/No-Go Gate

Do not implement async double-buffer build/render until V1 and V2 produce at
least:

```text
1. V1 render-only p50/p90:
   precomputed rays + no readback

2. V2 refit_ms p50:
   warm refit for changing-geometry robot poses

3. V2 rebuild_ms p50:
   warm CUDA LBVH full rebuild for large deformation / dynamic topology case
```

Decision rule:

```text
If rebuild_ms is comparable to or greater than render_execute_ms, overlap may
be worth the engineering complexity.

If refit_ms is far below render_execute_ms for the rigid robot path, overlap is
not justified for that path.
```

## V1 Implementation Results

Implemented on 2026-05-07:

```text
stage_optical_channels(result, channels) helper in optics.device
--video-ray-cache off|precompute
--video-readback full|rgb|none
benchmark-only RGB frame writer
frame_timing.csv mode columns:
  geometry_mode, ray_cache_mode, readback_mode, write_mode
NaN for not-applicable phases
```

Validation:

```text
ruff check optics/device.py optics/__init__.py examples/mujoco_menagerie_gpu_preview.py tests/unit/optics/test_device_optical.py
python -m py_compile optics/device.py optics/__init__.py examples/mujoco_menagerie_gpu_preview.py tests/unit/optics/test_device_optical.py
PYTHONPATH=. pytest tests/unit/optics/test_device_optical.py -q
conda run -n env_tilelang_20260119 python -m pytest tests/unit/optics/test_device_optical.py -q
```

GPU smoke:

```text
160x120, precompute + readback=none + no-write:
  render_execute_mean:   2.06 ms
  frame_total_mean:      2.06 ms
  overflow diagnostics:  NaN by design

160x120, precompute + readback=rgb + RGB PNG:
  render_execute_mean:   1.51 ms
  readback_host_mean:    0.34 ms
  image_build_mean:      0.47 ms
  encode_or_write_mean:  2.71 ms
```

960x640 benchmark matrix:

```text
uncached rays + full readback + no write:
  frame_total_mean:      108.10 ms
  fps_mean:                9.25
  camera_rays_mean:       45.89 ms
  render_execute_mean:    27.40 ms
  readback_host_mean:     34.81 ms
  csv:
    out/menagerie_go2_gpu_video_v1_960_uncached_full/frame_timing.csv

precomputed rays + full readback + no write:
  ray_precompute:        444.92 ms total, 44.49 ms/frame setup
  frame_total_mean:       58.63 ms
  fps_mean:               17.06
  render_execute_mean:    26.86 ms
  readback_host_mean:     31.77 ms
  csv:
    out/menagerie_go2_gpu_video_v1_960_precompute_full/frame_timing.csv

precomputed rays + no readback + no write:
  ray_precompute:        485.32 ms total, 48.53 ms/frame setup
  frame_total_mean:       29.19 ms
  fps_mean:               34.26
  render_execute_mean:    29.18 ms
  readback_host_mean:     NaN
  csv:
    out/menagerie_go2_gpu_video_v1_960_precompute_none/frame_timing.csv

precomputed rays + rgb readback + RGB PNG:
  ray_precompute:        449.82 ms total, 44.98 ms/frame setup
  frame_total_mean:       90.19 ms
  fps_mean:               11.09
  render_execute_mean:    27.25 ms
  readback_host_mean:      5.86 ms
  image_build_mean:       18.98 ms
  encode_or_write_mean:   38.09 ms
  csv:
    out/menagerie_go2_gpu_video_v1_960_precompute_rgb_write/frame_timing.csv
```

Interpretation after V1:

```text
Ray cache is a large benchmark-control lever: 9.25 FPS -> 17.06 FPS.
Render-only static Go2 960x640 direct-light+shadow is ~29 ms/frame, ~34 FPS.
Selected RGB readback cuts host readback from ~32-35 ms to ~6 ms.
PNG output remains expensive and should stay separated from renderer metrics.
```
