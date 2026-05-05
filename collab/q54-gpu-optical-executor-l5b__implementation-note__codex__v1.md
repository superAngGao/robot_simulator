Initiative: q54-gpu-optical-executor
Stage: l5b-implementation-note
Author: codex
Version: v1
Date: 2026-05-03
Status: implemented-l5b0
Related Files: optics/gpu_runtime.py, optics/device.py, optics/warp_execution.py, optics/execution.py, optics/__init__.py, tests/gpu/test_optical_gpu_runtime.py
Owner Summary: Implemented L5B.0, an explicit Q52 `GpuPublishedFrame` optical runtime helper. It borrows a device frame, launches the Warp first-hit optical executor, completes the device consumer, attaches the Q52 done event to the device `OpticalComputeResult`, and verifies that result buffers survive published slot reuse. L5B.0 intentionally supports world-static optical instances only; body-bound GPU scene packing is deferred.

# Q54 L5B.0 GPU Optical Device Lifecycle Implementation Note

## 1. Implemented Scope

This pass connects the L5A Warp first-hit executor to Q52 device-frame
lifecycle:

```text
GpuEngine published slot
  -> borrow_device_frame(consumer_id, frame_id, stream)
  -> build world-static OpticalSceneSnapshot metadata
  -> GpuBruteForceOpticalExecutor.execute(snapshot, spec)
  -> complete_device_consumer(consumer_id, frame_id, stream)
  -> OpticalComputeResult(location="device", ready_event=done_event)
```

The entry point is:

```text
optics.execute_optical_on_gpu_published_frame(...)
```

## 2. Lifecycle Boundary

The important contract is that Q52 frame-slot lifetime and optical result
lifetime are different:

- `complete_device_consumer(...)` means the optical kernel no longer needs to
  read the borrowed `GpuPublishedFrame` slot.
- It does not mean downstream consumers have already copied or read the
  optical result.

Therefore result buffers are owned by the optical executor/runtime, not by the
published frame slot.

## 3. L5B.0 World-Static Restriction

L5B.0 only accepts registries whose instances are world-static:

```text
instance.body_index is None
```

If any instance is body-bound, the helper raises `NotImplementedError`.

This is deliberate. Current L5A kernels consume a host-packed
`OpticalSceneSnapshot`. Body-bound GPU support should be added when the device
scene packer can read `GpuPublishedFrame.x_world_*` transforms directly, rather
than faking body transforms through an empty CPU frame.

## 4. Result Readiness

`execute_optical_on_gpu_published_frame(...)` attaches the event returned by
`complete_device_consumer(...)`:

```text
result.ready_event = done_event
```

`stage_optical_compute_result_to_host(...)` synchronizes this event before
copying device channels to NumPy. This makes staging safe even when the caller
does not globally synchronize the device first.

## 5. Device Buffer Lifetime

During debugging, the published slot reuse test exposed a subtle asynchronous
lifetime risk: result output buffers were retained by `result.channels`, but
kernel input buffers created inside `execute(...)` could be released by Python
before the stream had finished consuming them.

`OpticalComputeResult` now has:

```text
resources: tuple[object, ...]
```

The Warp executor stores input device arrays there. This keeps all arrays used
by the launched kernel alive until the result object is no longer needed.

## 6. Stream Ordering

All L5A device uploads, result buffer allocations, and kernel launch are scoped
to the optical stream when one is supplied. This avoids accidental default
stream / custom stream races between host-to-device copies and the kernel.

## 7. Tests

New GPU tests:

```text
tests/gpu/test_optical_gpu_runtime.py
```

Coverage:

- helper completes the Q52 device consumer;
- returned result carries the consumer done event;
- staged result matches expected hit/miss values;
- result buffers survive published slot reuse after completion;
- body-bound registries are rejected;
- sensor spec / frame timeline mismatch is rejected.

Verification:

```text
conda run -n env_tilelang_20260119 python -m pytest tests/gpu/test_optical_gpu_runtime.py -q
4 passed

conda run -n env_tilelang_20260119 python -m pytest tests/gpu/test_optical_warp_executor.py tests/gpu/test_optical_gpu_runtime.py -q
9 passed

PYTHONPATH=. pytest tests/unit/optics tests/unit/sensing -q
90 passed
```

## 8. Deferred

L5B.0 does not implement:

- body-bound GPU optical scene packing;
- device scene cache;
- result buffer pooling;
- GPU BVH;
- GPU direct-light or shadow rays;
- multi-env batched optical runtime semantics.
