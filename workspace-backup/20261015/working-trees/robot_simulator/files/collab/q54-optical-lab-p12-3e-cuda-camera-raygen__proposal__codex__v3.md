Initiative: q54-optical-lab-p12-3e-cuda-camera-raygen
Stage: proposal
Author: codex
Version: v3
Date: 2026-08-07
Status: in_review
Related Files: collab/q54-optical-lab-p12-3e-cuda-camera-raygen__proposal__codex__v2.md; optics/cuda_direct_light.py; tests/gpu/test_optical_gpu_runtime.py
Owner Summary: Close the two comments on proposal v2 without expanding P12.3e. The implementation keeps the existing two blocking stages: raygen and first-hit complete at the geometry synchronization, then shade completes at its own synchronization. Raygen CUDA-event timing is resolved after the geometry synchronization. Multi-GPU tests explicitly restore and assert the intentionally wrong ambient device before and after each raygen, geometry-camera, and direct-light-camera call. All other v2 contracts remain unchanged, including CUDA-first ownership and no Warp/Torch stream bridging.

## Purpose

This v3 is the implementation baseline for P12.3e. It incorporates by reference
all of proposal v2 except where the two clarifications below replace v2 text.
It does not add scope beyond the approved-with-comments review.

## Clarification 1: Preserve Two Blocking Stages

P12.3e will preserve the current synchronization ownership in
`optics/cuda_direct_light.py`. It will not refactor geometry and shading into one
internally asynchronous pipeline.

The direct-light camera path is:

```text
Warp-owned scene/snapshot ready event
    -> host-block until ready
    -> target device's current Torch stream
       [raygen -> first-hit]
    -> geometry target-device synchronization
       [resolve raygen CUDA-event timing]
    -> target device's current Torch stream
       [shade, with or without shadows]
    -> shade target-device synchronization
    -> blocking OpticalComputeResult(ready_event=None)
```

The geometry-only camera path ends after the first synchronization. The
direct-light camera path performs the existing second synchronization in its
shade helper.

Consequences:

- `_execute_device_rays(...)` owns the geometry-stage synchronization, not a
  single terminal synchronization for the whole direct-light render;
- raygen and first-hit are ordered on the same target-device Torch current
  stream;
- the raygen start/end CUDA events are read only after the post-first-hit
  geometry synchronization;
- shade begins after that host-visible geometry barrier and retains its own
  current synchronization and timing behavior;
- P12.4 treats this two-barrier implementation as the initial CUDA performance
  baseline;
- unifying the two barriers requires a separate internal async contract and is
  explicitly deferred.

The logical timing phase remains `raygen_kernel`, and the Lab/CSV field remains
`render_raygen_kernel_ms`. No raygen-only synchronization is added.

## Clarification 2: Prove Ambient-Device Mismatch at Call Time

Creating a Warp scene or CUDA LBVH for `cuda:1` may change ambient device state.
The two-GPU test therefore must not assume that an earlier
`torch.cuda.set_device(0)` remains effective.

After all target-`cuda:1` fixtures are constructed, each critical call follows
this pattern independently:

```python
torch.cuda.set_device(0)
assert torch.cuda.current_device() == 0

# The explicit operation target remains cuda:1.
result = target_cuda_1_operation(...)

# The C++/CUDA device guard must restore ambient state.
assert torch.cuda.current_device() == 0
```

Apply this precondition and postcondition separately to:

1. the low-level CUDA pinhole raygen helper;
2. `CudaDeviceBvhOpticalExecutor.execute_camera(...)`;
3. `CudaDeviceBvhDirectLightOpticalExecutor.execute_camera(...)`.

For every call, also assert that all returned tensor channels and retained ray
resources are on `cuda:1`. The test is skipped with an explicit reason when
fewer than two CUDA devices are visible.

The one-GPU suite still verifies output placement, same-device input rejection,
and removal of the CUDA executor `stream` argument, but it is not reported as
evidence for ambient-device restoration.

## Updated Acceptance Checks

The implementation is acceptable only when:

1. geometry camera execution performs one post-first-hit synchronization;
2. direct-light camera execution performs that geometry synchronization plus the
   existing post-shade synchronization;
3. raygen timing is resolved after the geometry synchronization and contains no
   extra raygen-only barrier;
4. the two-GPU test resets and checks ambient device state immediately around
   all three critical calls;
5. the device guard restores the caller's ambient device;
6. all remaining device, parity, profile, Lab-gate, environment-evidence, and
   Warp-retirement requirements from v2 pass unchanged.

## Implementation Decision

Proceed with the smaller two-stage blocking design. Do not introduce an internal
non-blocking geometry API in P12.3e. Optimize or unify synchronization only after
P12.4 quantifies the two barriers in representative end-to-end workloads.
