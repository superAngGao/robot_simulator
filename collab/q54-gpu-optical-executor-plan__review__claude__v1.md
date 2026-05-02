Initiative: q54-gpu-optical-executor
Stage: review
Author: claude
Version: v1
Date: 2026-05-02
Status: accept-with-edits
Related Files: collab/q54-gpu-optical-executor-plan__review-request__codex__v1.md
Owner Summary: Claude accepts the L5 GPU optical executor plan. Required clarifications: keep one `OpticalComputeResult` type with `location="device"`, put staging in `optics/device.py`, keep host-side role filtering for L5A, use explicit L5B Q52 helper, allow per-call result allocation, and prefer packed int64 source-order keys for simpler GPU tie-break.

# Q54 L5 GPU Optical Executor Plan Review

## Conclusion

The plan can enter L5A implementation.

The L5A -> L5B split is correct:

- L5A validates GPU intersection math and device result schema from a host
  `OpticalSceneSnapshot`.
- L5B integrates with `GpuPublishedFrame` and Q52 device-consumer completion.

This avoids mixing GPU ray-intersection complexity with published-slot lifetime
complexity in the first implementation step.

## Accepted Decisions

1. Keep a single `OpticalComputeResult` type and use `location="device"` for
   device results.
2. Put `stage_optical_compute_result_to_host(...)` in `optics/device.py`, not in
   `sensing/`.
3. Keep L5A role filtering on the host before uploading per-query primitive
   buffers.
4. Use float32 device buffers and normalize staged host channels back to the
   canonical host dtypes.
5. Use an explicit L5B helper for borrow -> kernel launch -> complete rather
   than folding Q52 integration into `OpticalExecutor` immediately.
6. Allocate fresh result buffers per call in L5A/L5B. Defer result pooling until
   correctness and lifecycle are stable.

## Required / Recommended Edits

### Source-Order Tie-Break

The plan originally used `source_order_key: int32[num_primitives, 2]`. That is
valid, but Warp kernels would need explicit lexicographic comparison:

```text
key_a[0] < key_b[0] or (key_a[0] == key_b[0] and key_a[1] < key_b[1])
```

This is easy to write incorrectly.

Recommended change:

```text
packed_source_order_key: int64[num_primitives]
key = instance_index * MAX_PRIMITIVES_PER_INSTANCE + primitive_index
```

The packing phase should enforce:

```text
primitive_index < MAX_PRIMITIVES_PER_INSTANCE
```

A reasonable first constant is:

```text
MAX_PRIMITIVES_PER_INSTANCE = 2**32
```

This keeps the GPU kernel tie-break to a single integer comparison while
preserving CPU `(instance_index, primitive_index_within_instance)` order.

### Float32 Range Scope

Document that L5A uses float32 device buffers and is intended for robot-scale
scenes, roughly below 1000 meters. Very large scenes may need float64 or
origin-rebasing later.

### L5B Helper Input

The proposed `execute_optical_on_gpu_published_frame(...)` helper should accept
`registry` first. Do not use a `registry_or_device_scene` union in L5B. Device
scene cache can be introduced later when that design is stable.

### Infinity In Kernels

L5A does not use shadow-ray infinite distances, but future GPU shadow rays
should explicitly verify Warp behavior for float32 infinity. Do not rely on an
unstated assumption when L5C adds shadows.

## Final Assessment

The plan is accepted with the source-order key packing recommendation and the
small API wording changes above.
