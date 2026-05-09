Initiative: q54-optical-pipeline-lab-gpu1-baseline-run
Stage: measurement-note
Author: codex
Version: v1
Date: 2026-05-09
Status: measured
Related Files: out/optical_pipeline_lab/go2_video_ordered_legacy_960_gpu1/matrix_summary.csv, out/optical_pipeline_lab/go2_video_ordered_baseline_1080p_gpu1/matrix_summary.csv
Owner Summary: Ran both the historical 960x640 and current 1080p Go2 ordered-video baseline matrix suites on cuda:1 in env_tilelang_20260119. The 960 suite is directly comparable to the old plan records; the 1080p suite establishes the new default-resolution baseline and exposes the D2H readback bottleneck.

# Q54 Optical Pipeline Lab GPU1 Baseline Run

## Commands

Historical comparison:

```bash
conda run -n env_tilelang_20260119 python -m tools.optical_pipeline_lab matrix \
  --suite go2_video_ordered_legacy_960 \
  --out out/optical_pipeline_lab/go2_video_ordered_legacy_960_gpu1 \
  --device cuda:1 \
  --frames 10 \
  --progress-every 1
```

Current default-resolution baseline:

```bash
conda run -n env_tilelang_20260119 python -m tools.optical_pipeline_lab matrix \
  --suite go2_video_ordered_baseline \
  --out out/optical_pipeline_lab/go2_video_ordered_baseline_1080p_gpu1 \
  --device cuda:1 \
  --frames 10 \
  --progress-every 1
```

## 960x640 Historical Suite

All cases passed.

```text
case                                      frame_p50_ms  frame_p90_ms  render_mean_ms  readback_mean_ms  fps_mean
legacy_960x640_shadow_readback_none       2.752         2.843         2.585           NaN               371.858
legacy_960x640_no_shadow_readback_none    1.306         1.447         1.231           NaN               749.558
legacy_960x640_shadow_readback_rgb        6.643         7.042         2.611           4.009             148.221
```

Comparison to the older plan record:

```text
shadow/readback=none: old frame ~2.65 ms, current p50 2.75 ms
no-shadow/readback=none: old frame ~1.29 ms, current p50 1.31 ms
shadow/readback=rgb: old frame ~7.77 ms, current p50 6.64 ms
```

This is close enough to use as the historical anchor, with normal run-to-run and
environment variance.

## 1080p Baseline Suite

All cases passed.

```text
case                                frame_p50_ms  frame_p90_ms  render_mean_ms  readback_mean_ms  fps_mean
smoke_160x120_shadow_readback_none   1.273         1.460         1.215           NaN               759.331
1080p_shadow_readback_none           5.507         5.993         5.535           NaN               177.216
1080p_no_shadow_readback_none        2.471         2.764         2.442           NaN               392.962
1080p_shadow_readback_rgb            52.203        217.090       6.157           111.687           8.458
```

The 1080p RGB readback row includes large first-frame D2H spikes:

```text
frame 0 readback_host_ms: 156.262
frame 1 readback_host_ms: 198.349
frame 2 readback_host_ms: 315.050
frame 3 readback_host_ms: 180.028
```

After those spikes, the last six frames stabilize around:

```text
frame_total_mean_ms: 50.906
readback_host_mean_ms: 44.530
render_execute_mean_ms: 6.082
```

## Interpretation

The 960 suite confirms the new lab path reproduces the previous baseline well
enough to compare future changes.

The 1080p suite shows the next bottleneck clearly: render-only 1080p is already
single-digit milliseconds, while synchronous RGB readback dominates frame time.
This supports moving next to the D1 async D2H overlap spike.
