# Q54 P12.3b Implementation Note: CUDA First-Hit Kernel

Author: Codex
Status: ready for review
Related design:
`collab/q54-optical-lab-p12-3-cuda-direct-light-backend__implementation-plan__codex__v1.md`

## Owner Summary

This implements the P12.3b CUDA host-ray first-hit kernel.

The new `CudaDeviceBvhOpticalExecutor` performs CUDA LBVH traversal without
calling Warp render kernels and returns `OpticalComputeResult(location="device")`
with native Torch CUDA tensor channels. `CudaDeviceBvhDirectLightOpticalExecutor`
now owns a CUDA first-hit executor, but direct-light shading still raises
`NotImplementedError` until P12.3c.

This is still not a runnable `cuda_direct_light` Lab video backend. The
`run_scenario(...)` guard from P12.3a remains until RGB shading lands.

## What Changed

### `optics/cuda_direct_light.py`

Added `CudaDeviceBvhOpticalExecutor`:

- accepts `DeviceOpticalSceneSnapshot`, `DeviceOpticalBvh`, and host
  `OpticalRaySensorSpec`;
- uploads host ray origins/directions into Torch CUDA tensors;
- converts existing Warp scene/BVH buffers to Torch views through
  `optics.device_channel.channel_to_torch(...)`;
- launches a hand-written CUDA extension function `first_hit_rays(...)`;
- returns these channels as Torch CUDA tensors:
  - `hit_mask`
  - `range_m`
  - `position_world`
  - `normal_world`
  - `numeric_instance_id`
  - `material_index`
  - `bvh_stack_overflow_count`
  - `bvh_max_stack_depth`

Traversal behavior mirrors the existing Warp BVH first-hit executor:

- tests planes before traversing triangle BVH;
- traverses the flat LBVH with a fixed local stack;
- reports stack overflow and max stack depth;
- uses the existing source-order tie-break rule:
  - closer hit wins;
  - when distances are within `1e-5`, smaller packed source-order key wins;
- flips triangle normals against the ray direction.

The kernel currently uses `torch.cuda.synchronize(...)` before returning. This
is correctness-first and intentionally not the final performance model.
P12.3/P12.5 should later replace this with ordered stream/event readiness.

### `CudaDeviceBvhDirectLightOpticalExecutor`

The direct-light wrapper now constructs a `CudaDeviceBvhOpticalExecutor`.

Its public `execute(...)` still raises:

```text
cuda_direct_light shading is pending P12.3c
```

This keeps P12.3b honest: first-hit exists, RGB direct lighting does not.

### `optics/__init__.py`

Exports:

- `CudaDeviceBvhOpticalExecutor`
- existing `CudaDeviceBvhDirectLightOpticalExecutor`

### `tests/gpu/test_optical_gpu_runtime.py`

Added `test_cuda_direct_light_first_hit_matches_warp_bvh_executor`.

The test builds the same static triangle-grid scene through CUDA LBVH, executes:

- existing `GpuDeviceBvhOpticalExecutor` over CUDA LBVH;
- new `CudaDeviceBvhOpticalExecutor` over CUDA LBVH.

It verifies:

- CUDA result channels are native Torch CUDA tensors;
- hit mask exact parity;
- range/position/normal parity within `1e-6`;
- numeric instance id exact parity;
- material index exact parity;
- stack overflow counter is zero;
- max stack depth is in the expected `[1, 32]` range.

## Validation

Commands run:

```bash
PYTHONPATH=. ruff check \
  optics/__init__.py optics/cuda_direct_light.py \
  tests/unit/optics/test_device_optical.py \
  tests/gpu/test_optical_gpu_runtime.py

PYTHONPATH=. pytest -q tests/unit/optics/

conda run -n env_tilelang_20260119 env PYTHONPATH=. pytest -q \
  tests/gpu/test_optical_gpu_runtime.py \
  -k "cuda_direct_light_first_hit_matches_warp_bvh_executor"
```

Result:

- ruff: clean
- unit optics: 272 passed
- GPU CUDA first-hit parity: 1 passed, 37 deselected

## Known Limits

- No direct-light RGB shading yet.
- No shadow any-hit yet.
- No CUDA camera raygen yet.
- `run_scenario(...)` still rejects `cuda_direct_light`.
- The CUDA first-hit path uses a global Torch CUDA synchronize for correctness.

## Review Questions

1. Is it correct to expose `CudaDeviceBvhOpticalExecutor` as the P12.3b
   first-hit executor, while keeping `CudaDeviceBvhDirectLightOpticalExecutor`
   non-runnable until shading lands?
2. Is the current correctness-first `torch.cuda.synchronize(...)` acceptable for
   P12.3b, given that P12.3/P12.5 still need stream/event cleanup before
   performance claims?
3. Should P12.3c build shading on top of this first-hit result, or should it
   immediately fuse first-hit + no-shadow shading in one CUDA kernel while
   preserving the same output-channel contract?
