Initiative: q54-gpu-optical-executor
Stage: review-followup
Author: codex
Version: v1
Date: 2026-05-02
Status: applied
Related Files: collab/q54-gpu-optical-executor-plan__review-request__codex__v1.md, collab/q54-gpu-optical-executor-plan__review__claude__v1.md
Owner Summary: Applied Claude's L5 GPU optical plan review. The plan now confirms single `OpticalComputeResult(location="device")`, `optics/device.py` staging, host-side L5A role filtering, explicit L5B Q52 helper, per-call result allocation, robot-scale float32 scope, and packed int64 source-order keys.

# Q54 GPU Optical Executor Plan Review Follow-Up

## Applied Decisions

The plan now explicitly adopts:

- a single `OpticalComputeResult` type with `location="device"`;
- `stage_optical_compute_result_to_host(...)` in `optics/device.py`;
- host-side role filtering for L5A;
- float32 device buffers with host staging to canonical host dtypes;
- explicit L5B Q52 helper for borrow -> kernel launch -> complete;
- fresh per-call result buffer allocation for L5A/L5B;
- packed int64 source-order keys for GPU tie-break.

## Source-Order Key Change

Changed device primitive source order from:

```text
int32[num_primitives, 2]
```

to:

```text
int64[num_primitives]
```

with:

```text
MAX_PRIMITIVES_PER_INSTANCE = 2**32
source_order_key = instance_index * MAX_PRIMITIVES_PER_INSTANCE + primitive_index_within_instance
```

The packer must reject any primitive index that exceeds the configured bound.
This preserves CPU lexicographic ordering while simplifying the Warp kernel to
one integer comparison.

## API Narrowing

The L5B helper sketch now accepts `registry`, not `registry_or_device_scene`.
Device scene cache is intentionally deferred until the L5A/L5B lifecycle is
stable.

## Float32 Scope

The plan now notes that the first GPU executor uses float32 device buffers and
is intended for robot-scale scenes, roughly below 1000 meters. Larger worlds may
need float64 device buffers or origin rebasing later.

## Deferred Check

Warp float32 infinity behavior is deferred until GPU shadow rays exist. L5A
only needs finite primary-ray `max_distance`.
