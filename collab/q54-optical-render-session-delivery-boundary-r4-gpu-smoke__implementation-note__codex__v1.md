# Q54 Optical RenderSession Delivery Boundary R4 GPU Smoke Note

Date: 2026-05-13
Author: Codex
Status: implemented

## Scope

Ran the accepted R4 validation path after R1/R2:

```text
go2_video_delivery_smoke
```

This verifies the current render-video loop, lab delivery facade, runtime
delivery result bridge, and matrix summary path on a real CUDA device.

## Environment

`nvidia-smi` reported H200 GPUs available. The first attempt with the default
Python environment failed at the existing Warp guard:

```text
mujoco_menagerie_gpu_preview.py requires warp with CUDA support
```

The validated run used the project CUDA/Warp conda environment.

## Command

```text
conda run -n env_tilelang_20260119 python -m tools.optical_pipeline_lab matrix \
  --suite go2_video_delivery_smoke \
  --out out/optical_pipeline_lab/go2_video_delivery_smoke_gpu1_r4 \
  --device cuda:1 \
  --frames 5 \
  --warmup-renders 5 \
  --progress-every 1
```

## Result

All three smoke cases passed:

```text
smoke_160x120_shadow_readback_none_sync:              passed
smoke_160x120_shadow_readback_rgb_sync:               passed
smoke_160x120_shadow_readback_rgb8_torch_async_ring2: passed
```

Matrix summary:

```text
none/sync:
  fps_mean ~= 718.02
  frame_p50_ms ~= 1.325
  render_execute_mean_ms ~= 1.226
  readback_host_mean_ms = NaN

rgb/sync:
  fps_mean ~= 510.59
  frame_p50_ms ~= 1.833
  render_execute_mean_ms ~= 1.353
  readback_host_mean_ms ~= 0.419

rgb8/torch_async/ring2:
  fps_mean ~= 576.78
  frame_p50_ms ~= 1.674
  render_execute_mean_ms ~= 1.226
  readback_host_mean_ms ~= 0.047
```

Output:

```text
out/optical_pipeline_lab/go2_video_delivery_smoke_gpu1_r4/matrix_summary.csv
```

## CSV Contract Checks

The async RGB8 case produced:

```text
delivery_policy=torch_async
readback_mode=torch_async_rgb8
readback_ring_depth=2
readback_ring_block_count=0
completed_frame_index=0..4
pack_rgb8_ms populated
readback_submit_ms/readback_wait_ms/readback_host_ms populated
```

The sync cases preserved:

```text
delivery_policy=sync
completed_frame_index=NaN
async ring columns=NaN
```

This confirms the R1/R2 contract work did not disturb the existing flat CSV
schema or the lab-visible delivery labels.

## Note

The async overlap ratio is noisy for this tiny 160x120 / 5-frame smoke and is
not used as a pass/fail criterion here. R4 is validating contract compatibility,
completion identity, and delivery mode coverage, not a throughput benchmark.
