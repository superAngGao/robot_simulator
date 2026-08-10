Initiative: q54-optical-lab-p12-3e-cuda-camera-raygen
Stage: decision
Author: owner
Version: v1
Date: 2026-08-10
Status: accepted
Related Files: collab/q54-optical-lab-p12-3e-cuda-camera-raygen__proposal__codex__v2.md; collab/q54-optical-lab-p12-3e-cuda-camera-raygen__proposal__codex__v3.md; collab/q54-optical-lab-p12-3e-cuda-camera-raygen__implementation-note__codex__v1.md; optics/cuda_direct_light.py; tests/gpu/test_optical_gpu_runtime.py; deploy/robot_sim_cuda128/README.md
Owner Summary: Approve commit 39a8a13. The implementation satisfies the reviewed CUDA raygen, device/stream, synchronization, parity, profile, Lab, and environment-evidence contracts. Record P12.3e complete and proceed to P12.4. CUDA remains non-default and Warp remains an optical oracle until the P12.4 capability, correctness, stability, and owner-defined performance gates pass.

---

## Chosen Direction

Approve `39a8a13 feat: add CUDA camera ray generation` and close P12.3e.

The accepted execution contract is:

```text
wait for Warp-owned scene/snapshot inputs
-> raygen -> first-hit -> synchronize target device
-> resolve raygen CUDA event
-> shade -> synchronize target device
-> blocking OpticalComputeResult(ready_event=None)
```

The optical architecture direction is CUDA-first. There will be no Warp/Torch
stream bridging work unless the owner explicitly changes this decision.

## Accepted Evidence

The review reported no issue blocking merge or push and confirmed:

- row-major CUDA pinhole ray generation and direct host-ray numerical parity;
- device guards and indexed target-device current streams for raygen,
  first-hit, no-shadow shade, and shadow shade;
- restoration of the caller's ambient CUDA device;
- the reviewed two-stage blocking behavior and raygen event-resolution point;
- all three output profiles and traversal-counter channel contracts;
- the narrow Lab gate change, with existing sync/non-RGB8 restrictions kept;
- removal of the CUDA executor's misleading `stream` argument and its call
  sites;
- tracked environment manifests and verification of all 62 wheel hashes;
- a clean worktree and `git diff --check` for the reviewed commit.

Independent reviewer reruns reported:

```text
related optical unit tests:             212 passed
CUDA raygen/profile/Lab/shadow subset:    4 passed
```

These supplement the exact-commit and GPU evidence already recorded in the
implementation note.

## Accepted Constraints

- CUDA is not made the default by P12.3e.
- The Warp optical executor remains available as a correctness/performance
  oracle and fallback.
- The current CUDA result is blocking and retains two target-device-wide
  synchronizations.
- Scene snapshot and LBVH inputs remain temporarily Warp-owned.
- Async delivery, RGB8, and broader Go2/dynamic acceptance are not implied by
  this approval.

## Deferred to P12.4

- define and record the owner's absolute product workload budget;
- compare CPU/Warp/CUDA capability and numerical parity on common workloads;
- measure warmed end-to-end latency, throughput, memory, and stability;
- quantify the two synchronization barriers and Warp-owned input boundary;
- decide whether CUDA becomes default for a validated workload;
- define the milestone for freezing and later deleting the Warp optical path.

The relative statement “CUDA is faster than Warp” is not sufficient by itself.
CUDA must meet the absolute product budget and all retained capability and
correctness requirements.

## Out of Scope

- Warp/Torch stream interoperation;
- kernel fusion, path tracing, OptiX, accumulation, denoising, or multisampling;
- deleting Warp physics or any non-optical Warp use;
- deleting the Warp optical executor before the P12.4 gate;
- changing the public backend override surface in this decision.

## Acceptance Conditions

1. Commit `39a8a13` remains the approved implementation baseline.
2. P12.3e is marked complete in project-level progress/reflection records.
3. P12.4 begins with a reviewed capability/parity/benchmark proposal.
4. No default switch or Warp optical deletion occurs without a separate,
   evidence-backed owner decision.
