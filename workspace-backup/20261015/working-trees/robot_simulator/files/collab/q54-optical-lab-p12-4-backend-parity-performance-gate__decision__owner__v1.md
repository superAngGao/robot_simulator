Initiative: q54-optical-lab-p12-4-backend-parity-performance-gate
Stage: decision
Author: owner
Version: v1
Date: 2026-08-10
Status: accepted-direction
Related Files: collab/q54-optical-lab-p12-4-backend-parity-performance-gate__proposal__codex__v1.md; collab/q54-optical-lab-p12-3e-cuda-camera-raygen__decision__owner__v1.md; collab/q54-optical-pipeline-lab-gpu1-baseline-run__codex__v1.md; collab/q54-optical-pipeline-lab-e0-e1-render-session__implementation-note__codex__v1.md; collab/q54-optical-pipeline-lab-e2-shadow-optimization-plan__codex__v1.md
Owner Summary: CUDA optical replacement has two mandatory performance gates: it must be no slower than Warp on matched retained workloads and it must meet the absolute product budget. The current Warp optical feature set is retained by default. P12.4 must close CUDA capability gaps and optimize the CUDA-native path before any default switch or Warp retirement; it is not only a measurement/reporting stage.

---

## Correction to P12.4 V1 Direction

The first P12.4 proposal made CUDA-vs-Warp speed informational and used only an
absolute product budget as the performance pass condition. That is rejected.

The accepted performance rule is:

```text
CUDA replacement pass
    = retained functional parity
    + correctness/stability pass
    + CUDA no slower than Warp on matched workloads
    + CUDA meets the absolute product budget
```

Meeting 30 FPS while taking 30 ms is not acceptable when the corresponding Warp
path takes approximately 6 ms. The absolute budget prevents a generally slow
system from passing merely because it beats Warp; the relative gate prevents a
large regression from passing merely because it still meets the product FPS.

## Retained Capability Rule

The current Warp optical feature set is retained by default. A missing CUDA
capability is a replacement blocker, not an optional benchmark row.

This includes, where currently implemented and product-relevant:

- Go2 static rendering;
- dynamic rigid and physics-published-frame workflows;
- GPU camera ray generation;
- shadows against supported geometry, including planes;
- directional/point lights, roles, and multiple lights;
- direct-light full, RGB preview, and render-only profiles;
- none/RGB/full/RGB8 delivery modes currently exposed by the Lab;
- synchronous and Torch async delivery where supported by Warp;
- required diagnostics, overflow counters, and artifact/CSV contracts;
- target-device and readiness/lifetime semantics.

Removing a capability from this list requires a separate owner product-scope
decision. It must not be dropped merely to let CUDA pass.

## Relative Performance Gate

Warp and CUDA must be measured:

- on the same commit and clean worktree;
- on the same physical GPU and environment;
- from the same scene/snapshot/camera sequence;
- with identical output profile, shadows, delivery, and readback;
- after both backends compile and warm up;
- with alternating backend order across paired repeats.

Primary per-row metrics:

- `frame_total_p90_ms` for the real product path;
- `render_execute_p90_ms` for renderer compute non-regression;
- throughput as the reciprocal consistency check.

For each mandatory row:

```text
relative_ratio = median_repeat_p90(CUDA) / median_repeat_p90(Warp)
```

A final replacement pass requires `relative_ratio <= 1.00`. If the result lies
inside the predeclared measurement-noise band and confidence does not resolve
the ordering, the row is `inconclusive`: increase frames/repeats and rerun. An
inconclusive row is not a pass.

## Historical Warp Anchors

Historical numbers are useful only for catching obviously invalid runs. They
are not substituted for the matched rerun:

```text
960x640 Go2, shadow, render-only:   p50 2.752 ms, p90 2.843 ms
960x640 Go2, no shadow:             p50 1.306 ms, p90 1.447 ms
1080p Go2, shadow, render-only:     p50 5.507 ms, p90 5.993 ms
1080p Go2, no shadow:               p50 2.471 ms, p90 2.764 ms
1080p Go2, shadow RGB8 async:       about 6.1-7.7 ms/frame in later runs
```

The measurements came from different dates/devices and therefore define scale,
not the final threshold.

## Current CUDA Replacement Blockers

The code audit found both functional and performance blockers:

Functional:

- Lab explicitly rejects CUDA direct light for Go2;
- dynamic/physics scenario acceptance is restricted to Warp plus CPU BVH;
- RGB8 packing is a Warp kernel and CUDA async delivery is rejected;
- CUDA shadow traversal handles triangle occluders but not plane occluders;
- the CUDA result is blocking and has no native readiness event.

Performance/ownership:

- host-blocking waits on Warp-owned scene and BVH events;
- global target-device synchronization after first-hit and again after shade;
- full geometry buffers are allocated/initialized even for lighter profiles;
- separate per-frame ray, geometry, RGB, intensity, and diagnostics allocations;
- CUDA LBVH tensors are wrapped back into Warp views/events despite being
  produced by Torch/CUDA;
- no CUDA-native RGB8 packing/readback readiness chain;
- fusion and buffer reuse have not been justified or implemented.

## Accepted P12.4 Direction

P12.4 is a CUDA optical replacement program:

1. build a matched same-process Warp/CUDA comparator and capture the current
   common-subset baseline;
2. close retained functional gaps with direct parity tests;
3. remove measured CUDA-native synchronization, ownership, allocation, and
   delivery costs without a Warp/Torch bridge;
4. run the full dual performance gate and retained-capability gate;
5. change the default and retire Warp optical only after an explicit final
   owner decision.

## Non-Negotiable Boundaries

- No Warp/Torch stream bridge.
- No weakening or silently dropping retained Warp capabilities.
- No performance claim from isolated kernel launch timing.
- No default switch on incomplete, incomparable, or inconclusive rows.
- No deletion of non-optical Warp code.
- No package installation during measurement; reuse the verified environment.
