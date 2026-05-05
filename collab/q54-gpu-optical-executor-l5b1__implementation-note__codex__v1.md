Initiative: q54-gpu-optical-executor
Stage: l5b1-implementation-note
Author: codex
Version: v1
Date: 2026-05-03
Status: implemented
Related Files: optics/gpu_runtime.py, tests/gpu/test_optical_gpu_runtime.py, MANIFEST.md, OPEN_QUESTIONS.md
Owner Summary: Extended the explicit Q52 GPU optical runtime helper from world-static-only L5B.0 to L5B.1 rigid body-bound optical instances. The transitional implementation host-stages selected-env body transforms from `GpuPublishedFrame.x_world_*`, builds a frame-aligned host `OpticalSceneSnapshot`, then runs the existing Warp first-hit executor.

# Q54 L5B.1 Body-Bound GPU Optical Runtime Implementation Note

## 1. Implemented Scope

L5B.1 extends:

```text
execute_optical_on_gpu_published_frame(...)
```

from world-static optical instances to rigid body-bound instances:

```text
OpticalInstanceSpec(body_index=i)
```

The Q52 lifecycle remains explicit:

```text
borrow_device_frame(...)
  -> build optical snapshot for the borrowed frame
  -> launch GpuBruteForceOpticalExecutor
  -> complete_device_consumer(...)
  -> return OpticalComputeResult(location="device", ready_event=done_event)
```

## 2. Algorithm

The current implementation is a transitional body-bound path:

1. detect whether any registry instance has `body_index is not None`;
2. synchronize the borrowed GPU frame's ready event for host staging;
3. copy selected-env `x_world_R_wp` and `x_world_r_wp` to host NumPy arrays;
4. build a list of `SpatialTransform` objects for that env;
5. wrap those transforms in a minimal `CpuPublishedFrame`;
6. call `OpticalSceneCache.snapshot_from_frame_inputs(...)`;
7. run the existing Warp first-hit executor over the resulting world-space
   primitives.

In other words:

```text
body transform acquisition: host-staged
primitive first-hit query: GPU Warp kernel
```

## 3. Why This Is Not L5C

This implementation intentionally does not claim to be the final GPU optical
scene cache.

It still performs host-side world-space primitive packing every call. That is
correct for validating registry/frame/result semantics, but it is not the
throughput path for many envs or high-resolution cameras.

L5C should move toward:

- long-lived device scene buffers;
- body-index / geometry metadata buffers;
- role visibility bitmasks on device;
- transform application in device kernels;
- GPU BVH or TLAS/BLAS-style acceleration.

## 4. Correctness Rules

The runtime validates:

- `spec.frame_id` matches the borrowed frame;
- `spec.sim_time` matches the borrowed frame;
- staged transform arrays have expected shapes;
- `env_idx` is in range;
- requested `body_index` values are in range.

The result buffer lifecycle remains unchanged from L5B.0: optical result
buffers belong to the result, not to the `GpuPublishedFrame` slot.

## 5. Tests

Updated:

```text
tests/gpu/test_optical_gpu_runtime.py
```

New coverage:

- a body-bound plane follows the body transform published in
  `GpuPublishedFrame.x_world_r_wp`;
- the staged `range_m` and `position_world` match the body height;
- body-bound device results survive published slot reuse after Q52 consumer
  completion.

Verification:

```text
conda run -n env_tilelang_20260119 python -m pytest tests/gpu/test_optical_gpu_runtime.py -q
5 passed
```

## 6. Deferred

L5B.1 still defers:

- fully device-resident body transform reads;
- device scene cache;
- GPU BVH;
- GPU direct-light and shadow rays;
- multi-env batched optical runtime semantics.
