Initiative: q54-optical-lab-p12-4-backend-parity-performance-gate
Stage: decision
Author: owner
Version: v2
Date: 2026-08-10
Status: accepted-correction
Related Files: collab/q54-optical-lab-p12-4-backend-parity-performance-gate__decision__owner__v1.md; collab/q54-optical-lab-p12-4-backend-parity-performance-gate__proposal__codex__v5.md; benchmarks/results/q54_p12_4/gate.json
Owner Summary: Do not invent or freeze a numeric device-memory threshold that is not tied to an explicit product workload and deployment profile. P12.4 retains exact clean single-lane memory measurement and mandatory-workload OOM failure, but no numeric memory acceptance limit. The existing absolute latency and throughput gates remain unchanged.

## Decision

Ambiguous resource limits must not be presented as product requirements merely
because a gate schema has a place for them. P12.4 therefore has no numeric
device-memory acceptance threshold.

This is not permission to ignore memory. Every shipping candidate must be
measured in an independent clean process containing exactly:

- the physics engine;
- one candidate optical backend;
- that candidate's selected acceleration implementation;
- its delivery/readback rings;
- allocator-reserved device memory.

The losing optical lane must not be imported or initialized in that process.
The primary observation is peak NVML process device memory in MiB, sampled from
before physics initialization through final reclaim. Torch allocated/reserved
and Warp allocator/pool values are diagnostics. Baseline subtraction, cache
clearing to manufacture a lower peak, and dual-lane residency are prohibited.

OOM on a mandatory workload is a hard failure. Otherwise memory is reported,
compared, and retained in the selected-backend baseline without a numeric
pass/fail threshold in this initiative.

Any future numeric memory requirement must arrive as a separate owner decision
bound to a named workload, scene/environment scale, resolution/camera count,
delivery policy, ring depth, target deployment device, and measurement
endpoint. It must be frozen before inspecting the measurements it will judge.

## Unchanged Gates

- CUDA must remain no slower than Warp on every matched mandatory row.
- The synchronous shipping latency and ordered asynchronous throughput floors
  remain absolute replacement requirements.
- Mandatory workload correctness, lifecycle, OOM, and final cleanup rules are
  unchanged.
